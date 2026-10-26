"""Compare bounded native gate-response compression on CPU.

Uses a previously downloaded original FP8 expert sample and a measured capture
archive. Whole trajectories are selected before fitting. Linear gate responses
and their transitions are proxies; no modified decoder or language is run.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tda import expert_compress as base
from tda import trajectory_compress as trajectory
from tda.block_quant import quantize_reader, save_block_quantization, load_block_quantization
from tda.glm_capture_archive import load_archive_manifest, load_archive_entry
from tda.glm_capture_calibration import gate_calibration_records
from tda.glm_weight_sample import GLMWeightSampleReader
from scripts.expert_compression_test import FactorReader, reconstruction_metrics


MAX_SPLIT_RECORDS = 8192
MAX_SPLIT_PAIRS = 4096


def _position_budget(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_SPLIT_RECORDS:
        raise ValueError(f"{name} must be an integer in 1..{MAX_SPLIT_RECORDS}")


def _working_budget(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("working-byte budget must be a positive integer")


def write_json(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def select_trajectories(records, pairs, *, max_records):
    """Keep complete recorded trajectories in first-occurrence archive order.

    Oversized trajectories are skipped, never truncated. Every stored adjacent
    pair within a selected trajectory is retained, with its original direction.
    """
    _position_budget(max_records, "trajectory position budget")
    groups = OrderedDict()
    for record in records:
        groups.setdefault(record.trajectory_id, []).append(record)
    chosen, selected_ids, skipped = [], set(), []
    for name, group in groups.items():
        if len(chosen) + len(group) > max_records:
            skipped.append(name)
            continue
        chosen.extend(group)
        selected_ids.update(item.record_id for item in group)
    selected_pairs = [pair for pair in pairs if pair.previous_id in selected_ids and pair.next_id in selected_ids]
    return chosen, selected_pairs, {"selection": "whole trajectories in first-occurrence archive order",
        "available_records": len(records), "available_trajectories": len(groups),
        "selected_records": len(chosen), "selected_pairs": len(selected_pairs),
        "selected_trajectory_ids": list(dict.fromkeys(item.trajectory_id for item in chosen)),
        "skipped_trajectory_ids": skipped, "position_budget": max_records}


def router_coverage(archive, manifest, identity, selected, *, max_working_bytes):
    """Actual native router choices for selected positions, separate from fit."""
    names = (f"layer_{identity.layer}_router_ids", f"layer_{identity.layer}_router_weights")
    try:
        numeric_ids = [int(value) for value in identity.expert_ids]
    except ValueError:
        return {"available": False, "reason": "sample expert identifiers are not native numeric router IDs"}
    if len(set(numeric_ids)) != len(numeric_ids):
        raise ValueError("sample expert identifiers alias the same numeric router ID")
    totals = {split: {"positions": 0, "positions_using_sample": 0, "choices": 0,
        "sample_choices": 0, "routed_weight": 0.0, "sample_routed_weight": 0.0,
        "per_expert_choices": {value: 0 for value in identity.expert_ids},
        "per_expert_routed_weight": {value: 0.0 for value in identity.expert_ids}} for split in selected}
    requested = set().union(*selected.values())
    for entry in manifest["entries"]:
        if entry["role"] != "capture":
            continue
        rows = entry["metadata"]["rows"]
        if not any(row["id"] in requested for row in rows):
            continue
        if not all(name in entry["tensors"] for name in names):
            return {"available": False, "reason": "selected position router choices were not captured"}
        if entry["bytes"] * 2 > max_working_bytes:
            raise MemoryError("selected router chunk exceeds working-byte budget")
        tensors = load_archive_entry(archive, entry, identity=manifest["identity"])
        for split, record_ids in selected.items():
            indices = [i for i, row in enumerate(rows) if row["id"] in record_ids]
            if not indices:
                continue
            ids = tensors[names[0]][indices]
            weights = tensors[names[1]][indices].double()
            match = torch.isin(ids, torch.tensor(numeric_ids, dtype=ids.dtype))
            item = totals[split]
            item["positions"] += len(indices)
            item["positions_using_sample"] += int(match.any(dim=1).sum())
            item["choices"] += ids.numel()
            item["sample_choices"] += int(match.sum())
            item["routed_weight"] += float(weights.sum())
            item["sample_routed_weight"] += float(weights[match].sum())
            for expert, numeric in zip(identity.expert_ids, numeric_ids):
                mask = ids == numeric
                item["per_expert_choices"][expert] += int(mask.sum())
                item["per_expert_routed_weight"][expert] += float(weights[mask].sum())
    for item in totals.values():
        item["sample_choice_fraction"] = item["sample_choices"] / item["choices"] if item["choices"] else None
        item["sample_position_fraction"] = item["positions_using_sample"] / item["positions"] if item["positions"] else None
        item["sample_routed_weight_share"] = item["sample_routed_weight"] / item["routed_weight"] if item["routed_weight"] else None
    return {"available": True, "scope": "actual native top-k choices at selected calibration/heldout positions",
        "fit_uses_router_weights": False, "weights_include_native_router_scaling": True, **totals}


def load_samples(archive, identity, *, max_train=256, max_heldout=128, max_working_bytes=1 << 30):
    _position_budget(max_train, "calibration position budget")
    _position_budget(max_heldout, "heldout position budget")
    _working_budget(max_working_bytes)
    manifest = load_archive_manifest(archive)
    precision = {"float8_e4m3fn": "native_fp8_e4m3fn", "float32": "float32",
                 "float16": "float16", "bfloat16": "bfloat16"}.get(identity.source_dtype)
    if precision is None or manifest["identity"]["source_precision"] != precision:
        raise ValueError("capture source precision differs from the original sample")
    candidate_count = sum(row.get("kind") != "probe_framing" for entry in manifest["entries"]
                          if entry["role"] == "capture" for row in entry["metadata"]["rows"])
    largest_chunk = max((entry["bytes"] for entry in manifest["entries"] if entry["role"] == "capture"), default=0)
    # Full split readers are bounded, then whole-trajectory selection discards
    # unused states before fitting. Include two serialized-size chunk buffers
    # and a FP32 upper bound for both splits, even though native states are BF16.
    read_bound = candidate_count * identity.matrix_shape[1] * 4 + 2 * largest_chunk
    if read_bound > max_working_bytes:
        raise MemoryError("archive candidate states and chunk buffers exceed working-byte budget")
    scope, training, train_pairs = gate_calibration_records(archive, identity, max_records=MAX_SPLIT_RECORDS,
        max_pairs=MAX_SPLIT_PAIRS, max_state_bytes=max_working_bytes)
    _, heldout, heldout_pairs = gate_calibration_records(archive, identity, split="heldout", max_records=MAX_SPLIT_RECORDS,
        max_pairs=MAX_SPLIT_PAIRS, max_state_bytes=max_working_bytes)
    training, train_pairs, train_selection = select_trajectories(training, train_pairs, max_records=max_train)
    heldout, heldout_pairs, heldout_selection = select_trajectories(heldout, heldout_pairs, max_records=max_heldout)
    if not training or not heldout or not train_pairs or not heldout_pairs:
        raise ValueError("comparison requires complete calibration and heldout trajectories with adjacent pairs")
    if ({record.record_id for record in training} & {record.record_id for record in heldout} or
            {record.trajectory_id for record in training} & {record.trajectory_id for record in heldout}):
        raise ValueError("calibration and heldout trajectories must be disjoint")
    row_map = {row["id"]: row for entry in manifest["entries"] if entry["role"] == "capture"
               for row in entry["metadata"]["rows"]}
    selected = {"calibration": {r.record_id for r in training}, "heldout": {r.record_id for r in heldout}}
    evidence = {"archive_status": manifest["status"], "origin": manifest["origin"],
        "archive_manifest_sha256": hashlib.sha256((Path(archive) / "manifest.json").read_bytes()).hexdigest(),
        "teacher": manifest["identity"], "input_state_scope": scope.to_dict(),
        "estimated_evidence_read_peak_bytes": read_bound, "heldout_used_for_fit": False,
        "calibration": train_selection, "heldout": heldout_selection,
        "records": [{"split": split, "record_id": r.record_id, "trajectory_id": r.trajectory_id,
                     "environment_id": r.environment_id, "provenance_weight": r.provenance_weight,
                     "stimulation_weight": r.stimulation_weight, "prediction_correct": r.prediction_correct,
                     "token_position": row_map[r.record_id]["token_position"],
                     "conversation_id": row_map[r.record_id].get("conversation_id")}
                    for split, records in (("calibration", training), ("heldout", heldout)) for r in records],
        "pairs": {split: [{"pair_id": p.pair_id, "previous_id": p.previous_id, "next_id": p.next_id,
                           "token_gap": row_map[p.next_id]["token_position"] - row_map[p.previous_id]["token_position"]} for p in pairs]
                  for split, pairs in (("calibration", train_pairs), ("heldout", heldout_pairs))},
        "transition_input": "next state minus previous state; not divided by token gap; not per-token velocity",
        "capture_chunks": [{"file": entry["file"], "bytes": entry["bytes"], "sha256": entry["sha256"]}
                           for entry in manifest["entries"] if entry["role"] == "capture"
                           and any(row["id"] in selected["calibration"] | selected["heldout"] for row in entry["metadata"]["rows"])],
        "router_coverage": router_coverage(archive, manifest, identity, selected, max_working_bytes=max_working_bytes)}
    return scope, training, train_pairs, heldout, heldout_pairs, evidence


def evaluate_reader(source, decoded, records, pairs, *, scope, row_chunk=16, max_working_bytes=1 << 30):
    """The trajectory fitter's balanced response metric, also usable for INT4."""
    identity = source.identity
    if decoded.identity != identity or scope != trajectory.InputStateScope.from_identity(identity, state_site=scope.state_site):
        raise ValueError("linear evaluation requires the exact matrix group and input state scope")
    if not isinstance(records, (list, tuple)) or not isinstance(pairs, (list, tuple)) or not records:
        raise ValueError("linear evaluation requires measured input records")
    if len(records) > MAX_SPLIT_RECORDS or len(pairs) > MAX_SPLIT_PAIRS:
        raise MemoryError("linear evaluation record or pair count exceeds native capture limits")
    _working_budget(max_working_bytes)
    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or not 1 <= row_chunk <= 32:
        raise ValueError("linear evaluation row chunk must be 1..32")
    e, (rows, columns) = identity.expert_count, identity.matrix_shape
    count = len(records) + len(pairs)
    estimate = (2 * len(records) + count) * columns * 4 + 8 * e * row_chunk * columns * 4 + 6 * e * count * row_chunk * 8
    if estimate > max_working_bytes:
        raise MemoryError("linear evaluation exceeds working-byte budget")
    records, pairs = trajectory._snapshot(records, pairs, scope,
        max_records=MAX_SPLIT_RECORDS, max_pairs=MAX_SPLIT_PAIRS)
    inputs, feature_weights = trajectory._features(records, pairs)
    energies, errors = torch.zeros(count, dtype=torch.float64), torch.zeros(count, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            stop = min(rows, start + row_chunk)
            reference = trajectory._response(trajectory._read_rows(source, identity, start, stop), inputs)
            proposed = trajectory._response(trajectory._read_rows(decoded, identity, start, stop), inputs)
            energies += reference.double().square().sum(dim=(0, 2))
            errors += (proposed.double() - reference.double()).square().sum(dim=(0, 2))
    report = {"evaluation_scope": "scoped linear gate responses and within-trajectory state differences",
        "record_origin": records[0].origin, "records": len(records), "pairs": len(pairs),
        "expert_response_weighting": "all sampled experts equally; counterfactual responses, not router-weighted",
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False}
    for name, section in (("point", slice(0, len(records))), ("contrast", slice(len(records), None))):
        source_energy, error, mass = energies[section], errors[section], feature_weights[section]
        valid = source_energy > 0
        denominator = float(mass[valid].sum())
        relative = float((mass[valid] * error[valid] / source_energy[valid]).sum()) / denominator if denominator else None
        report[name + "_relative_rmse"] = math.sqrt(relative) if relative is not None else None
        report[name + "_zero_source_features"] = int((~valid).sum())
        report[name + "_zero_source_absolute_error_sq"] = float(error[~valid].sum())
    return report


def compare(source, archive, out, *, ranks=(0, 1, 2), max_train=256, max_heldout=128,
            max_working_bytes=1 << 30, row_chunk=16, include_int4=True,
            storage_dtype="bfloat16", point_weight=1.0, contrast_weight=1.0,
            include_probe_span=False, include_probe_residual=False,
            include_probe_residual_controls=False, random_control_seed=0,
            include_probe_hybrid=False, hybrid_extra_rank=None):
    if out.exists():
        raise FileExistsError("use a new comparison output directory")
    if include_probe_residual and not include_int4:
        raise ValueError("probe residual comparison requires its identical uncorrected INT4 control")
    if include_probe_residual_controls and not include_probe_residual:
        raise ValueError("equal-rank residual controls require the native probe residual candidate")
    if include_probe_hybrid and not include_probe_residual:
        raise ValueError("probe hybrid requires the native probe residual candidate and INT4 control")
    if hybrid_extra_rank is not None:
        if not include_probe_hybrid:
            raise ValueError("hybrid extra rank requires the probe hybrid candidate")
        if (isinstance(hybrid_extra_rank, bool) or not isinstance(hybrid_extra_rank, int) or
                not 1 <= hybrid_extra_rank < min(source.identity.matrix_shape[1], 2048)):
            raise ValueError("hybrid extra rank must be a positive integer below min(input width,2048)")
    if isinstance(random_control_seed, bool) or not isinstance(random_control_seed, int) or not 0 <= random_control_seed < 2**32:
        raise ValueError("random residual control seed must be an integer in [0, 2**32)")
    maximum_rank = min(source.identity.expert_count - 1, source.identity.column_count)
    if not ranks or len(set(ranks)) != len(ranks) or any(isinstance(r, bool) or not isinstance(r, int) or not 0 <= r <= maximum_rank for r in ranks):
        raise ValueError("comparison ranks must be distinct and within the source geometry")
    scope, train, pairs, test, test_pairs, evidence = load_samples(archive, source.identity,
        max_train=max_train, max_heldout=max_heldout, max_working_bytes=max_working_bytes)
    bank = None
    if include_probe_span or include_probe_residual:
        # Optional implementation is loaded only when requested. Seed contrasts
        # are a separate static objective, never fabricated temporal pairs.
        from tda.glm_capture_calibration import native_probe_bank
        from tda.probe_span_compress import fit_probe_span_group, evaluate_probe_span_group
        bank = native_probe_bank(archive, tensor_name=f"layer_{source.identity.layer}_expert_input")
    if include_probe_residual:
        from tda.probe_residual_compress import (fit_probe_residual_group, save_probe_residual,
                                               load_probe_residual, evaluate_probe_residual_group)
    if include_probe_residual_controls or include_probe_hybrid:
        from tda.probe_residual_compress import (make_calibration_pca_basis, make_random_basis,
                                               fit_scoped_residual_basis)
    if include_probe_hybrid:
        from tda.probe_residual_compress import make_probe_pca_complement_basis
    out.mkdir(parents=True)
    started, reports = time.monotonic(), []
    write_json(out / "evidence.json", evidence)
    if bank is not None:
        torch.save(bank, out / "native_probe_bank.pt")
        summary = {key: value for key, value in bank.items() if not isinstance(value, torch.Tensor)}
        summary["axis_count"] = len(bank["probe_ids"])
        summary["seed_witness_serialized_bytes"] = (out / "native_probe_bank.pt").stat().st_size
        write_json(out / "native_probe_bank.json", summary)
    settings = {"identity": source.identity.to_dict(), "device": "cpu", "ranks": list(ranks),
        "max_working_bytes": max_working_bytes, "factor_storage_dtype": storage_dtype,
        "row_chunk": row_chunk, "point_weight": point_weight, "contrast_weight": contrast_weight,
        "probe_span_enabled": include_probe_span,
        "probe_residual_enabled": include_probe_residual,
        "probe_residual_controls_enabled": include_probe_residual_controls,
        "residual_random_control_seed": random_control_seed if include_probe_residual_controls else None,
        "probe_hybrid_enabled": include_probe_hybrid,
        "hybrid_extra_rank_requested": hybrid_extra_rank,
        "hybrid_extra_rank_default": "native probe span rank" if include_probe_hybrid and hybrid_extra_rank is None else None,
        "hybrid_random_control_seed": random_control_seed if include_probe_hybrid else None,
        "probe_residual_coefficient_dtype": "float32" if include_probe_residual else None,
        "origin": evidence["origin"], "heldout_used_for_fit": False,
        "memory_bound_kind": "conservative numerical tensor arrays excluding allocator and BLAS overhead",
        "full_decoder_evaluated": False, "language_quality_established": False,
        "runnable_checkpoint_exported": False, "cannot_extrapolate_to_entire_model": True}
    write_json(out / "settings.json", settings)
    write_json(out / "status.json", {"status": "running"})
    methods = ("weight_pca", "trajectory_response", "probe_span") if include_probe_span else ("weight_pca", "trajectory_response")
    variants = [(method, rank) for rank in ranks for method in methods]
    if include_int4:
        variants.append(("symmetric_int4", None))
    if include_probe_residual:
        # The correction spans native input directions; its rank is independent
        # of the shared across-expert ranks above. Fit exactly one candidate.
        variants.append(("probe_residual", None))
    if include_probe_residual_controls:
        variants.extend((("residual_calibration_pca", None), ("residual_seeded_random", None)))
    if include_probe_hybrid:
        variants.extend((("probe_pca_hybrid", None), ("residual_calibration_pca_extended", None),
            ("residual_seeded_random_extended", None)))
    extended_methods = {"probe_pca_hybrid", "residual_calibration_pca_extended", "residual_seeded_random_extended"}
    residual_methods = {"probe_residual", "residual_calibration_pca", "residual_seeded_random"} | extended_methods
    native_residual_rank, native_residual_report = None, None
    hybrid_rank, hybrid_report = None, None
    try:
        for method, rank in variants:
            name = method if rank is None else f"{method}_rank_{rank}"
            print(f"[Native calibration] Fitting {name} on CPU.", flush=True)
            control_basis = None
            if method in residual_methods:
                original_int4 = load_block_quantization(out / "symmetric_int4", expected_identity=source.identity,
                    max_artifact_bytes=max_working_bytes)
                residual_options = {"base_quantization": original_int4, "coefficient_dtype": "float32",
                    "chunk_columns": row_chunk * source.identity.matrix_shape[1],
                    "max_working_bytes": max_working_bytes, "expected_origin": evidence["origin"]}
                if method == "probe_residual":
                    encoded = fit_probe_residual_group(source, bank, **residual_options)
                else:
                    if native_residual_rank is None:
                        raise RuntimeError("native residual must establish the matched control rank first")
                    target_rank = hybrid_rank if method in extended_methods else native_residual_rank
                    if method == "probe_pca_hybrid":
                        extra_rank = native_residual_rank if hybrid_extra_rank is None else hybrid_extra_rank
                        if native_residual_rank + extra_rank > min(source.identity.matrix_shape[1], 2048):
                            raise ValueError("protected native span plus extra rank exceeds the bounded input span")
                        control_basis = make_probe_pca_complement_basis(train, bank, expected_scope=scope,
                            extra_rank=extra_rank, heldout_records=test, max_records=max(max_train, max_heldout),
                            max_working_bytes=max_working_bytes, expected_origin=evidence["origin"])
                    elif method in ("residual_calibration_pca", "residual_calibration_pca_extended"):
                        if target_rank is None:
                            raise RuntimeError("hybrid must establish the extended control rank first")
                        control_basis = make_calibration_pca_basis(train, bank, expected_scope=scope,
                            requested_rank=target_rank, heldout_records=test, max_records=max(max_train, max_heldout),
                            max_working_bytes=max_working_bytes, expected_origin=evidence["origin"])
                    else:
                        if target_rank is None:
                            raise RuntimeError("hybrid must establish the extended control rank first")
                        control_basis = make_random_basis(bank, requested_rank=target_rank,
                            seed=random_control_seed, max_working_bytes=max_working_bytes, expected_origin=evidence["origin"])
                    if control_basis.get("rank_matches_request") is not True:
                        raise ValueError("calibration PCA control cannot match the native probe rank or requested hybrid rank; increase calibration positions without adding heldout data")
                    encoded = fit_scoped_residual_basis(source, control_basis, **residual_options)
                del original_int4
                metadata = save_probe_residual(encoded, out / name, max_artifact_bytes=max_working_bytes)
                del encoded
                decoder = load_probe_residual(out / name, expected_identity=source.identity,
                    max_artifact_bytes=max_working_bytes)
            elif method == "symmetric_int4":
                encoded = quantize_reader(source, bits=4, block_size=128, max_working_bytes=max_working_bytes)
                metadata = save_block_quantization(encoded, out / name, max_artifact_bytes=max_working_bytes)
                del encoded
                decoder = load_block_quantization(out / name, expected_identity=source.identity, max_artifact_bytes=max_working_bytes)
            else:
                options = {"storage_dtype": storage_dtype, "chunk_columns": row_chunk * source.identity.matrix_shape[1],
                           "max_working_bytes": max_working_bytes}
                if method == "weight_pca":
                    encoded = base.fit_expert_group(source, rank, **options)
                elif method == "probe_span":
                    encoded = fit_probe_span_group(source, rank, bank, expected_origin=evidence["origin"], **options)
                else:
                    encoded = trajectory.fit_trajectory_group(source, rank, train, pairs, expected_scope=scope,
                        heldout_records=test, heldout_pairs=test_pairs, point_weight=point_weight,
                        contrast_weight=contrast_weight, max_records=max_train + max_heldout,
                        max_pairs=2 * MAX_SPLIT_PAIRS, **options)
                metadata = base.save_expert_compression(encoded, out / name, max_artifact_bytes=max_working_bytes)
                del encoded
                decoder = FactorReader(base.load_expert_compression(out / name, expected_identity=source.identity, max_artifact_bytes=max_working_bytes))
            report = {"variant": name, "method": method, "rank": rank,
                "serialized_total_bytes": metadata["serialized_total_bytes"],
                "factor_tensor_bytes": metadata["factor_tensor_bytes"],
                "source_tensor_bytes_including_scales": source.identity.actual_source_bytes,
                "serialized_storage_ratio": metadata["serialized_storage_ratio"],
                "serialized_is_smaller": metadata["serialized_is_smaller"],
                "weight_error": reconstruction_metrics(source, decoder, chunk_columns=row_chunk * source.identity.matrix_shape[1]),
                "calibration_response": evaluate_reader(source, decoder, train, pairs, scope=scope,
                    row_chunk=row_chunk, max_working_bytes=max_working_bytes),
                "heldout_response": evaluate_reader(source, decoder, test, test_pairs, scope=scope,
                    row_chunk=row_chunk, max_working_bytes=max_working_bytes),
                "heldout_used_for_fit": False, "language_quality_established": False,
                "full_decoder_evaluated": False, "runtime_savings_evaluated": False}
            report["weight_error"]["error_scope"] = "original decoded sample tensors, including checkpoint FP8 block scales when present"
            if method in ("trajectory_response", "probe_span") or method in residual_methods:
                report["fit_evidence"] = metadata["objective"]
            if method in residual_methods:
                report.update({"correction_span_rank": decoder.span_basis.shape[1],
                    "coefficient_storage_dtype": "float32", "base_variant": "symmetric_int4",
                    "base_tensor_bytes": metadata["base_tensor_bytes"],
                    "correction_tensor_bytes": metadata["correction_tensor_bytes"],
                    "shared_expert_rank": None})
                if method == "probe_residual":
                    native_residual_rank = decoder.span_basis.shape[1]
                    report["probe_span_rank"] = native_residual_rank
                    native_residual_report = report
                elif method == "probe_pca_hybrid":
                    report["control_basis"] = {key: value for key, value in control_basis.items() if not isinstance(value, torch.Tensor)}
                    hybrid_rank = decoder.span_basis.shape[1]
                    report["protected_probe_rank"] = control_basis["protected_probe_rank"]
                    report["extra_rank"] = control_basis["extra_rank"]
                    if (report["protected_probe_rank"] != native_residual_rank or
                            hybrid_rank != native_residual_rank + report["extra_rank"] or
                            report["base_tensor_bytes"] != native_residual_report["base_tensor_bytes"]):
                        raise ValueError("hybrid must preserve the full native span on the identical INT4 base")
                    hybrid_report = report
                else:
                    matched_report = hybrid_report if method in extended_methods else native_residual_report
                    if (matched_report is None or
                            decoder.span_basis.shape[1] != matched_report["correction_span_rank"] or
                            report["base_tensor_bytes"] != matched_report["base_tensor_bytes"] or
                            report["factor_tensor_bytes"] != matched_report["factor_tensor_bytes"]):
                        raise ValueError("control correction rank and tensor bytes must match its native or hybrid repair")
                    report["control_basis"] = {key: value for key, value in control_basis.items() if not isinstance(value, torch.Tensor)}
                    report["rank_matched_to_variant"] = matched_report["variant"]
                    report["tensor_bytes_match_matched_repair"] = True
                    report["serialized_total_bytes_difference_from_matched_repair"] = report["serialized_total_bytes"] - matched_report["serialized_total_bytes"]
                    if method not in extended_methods:
                        report["tensor_bytes_match_native_repair"] = True
                        report["serialized_total_bytes_difference_from_native_repair"] = report["serialized_total_bytes_difference_from_matched_repair"]
                if method in extended_methods:
                    report["repair_rank_group"] = "native_plus_calibration_complement"
                    report["tensor_bytes_difference_from_native_repair"] = report["factor_tensor_bytes"] - native_residual_report["factor_tensor_bytes"]
                    report["serialized_total_bytes_difference_from_native_repair"] = report["serialized_total_bytes"] - native_residual_report["serialized_total_bytes"]
                else:
                    report["repair_rank_group"] = "native_probe_rank"
            if include_probe_residual and (method == "symmetric_int4" or method in residual_methods):
                report["static_seed_contrast_response"] = evaluate_probe_residual_group(decoder, source, bank,
                    chunk_columns=row_chunk * source.identity.matrix_shape[1], max_working_bytes=max_working_bytes,
                    expected_origin=evidence["origin"])
            elif bank is not None and method != "symmetric_int4":
                report["static_seed_contrast_response"] = evaluate_probe_span_group(decoder.result, source, bank,
                    chunk_columns=row_chunk * source.identity.matrix_shape[1], max_working_bytes=max_working_bytes,
                    expected_origin=evidence["origin"])
            reports.append(report)
            write_json(out / "results.json", reports)
            transition = report["heldout_response"]["contrast_relative_rmse"]
            error_label = "undefined (zero source energy)" if transition is None else f"{transition:.4f}"
            print(f"[Native calibration] {name}: bytes={report['serialized_total_bytes']}; heldout transition error={error_label}.", flush=True)
            del decoder, control_basis
            gc.collect()
        write_json(out / "status.json", {"status": "finished", "elapsed_seconds": time.monotonic() - started,
            "completed_variants": [report["variant"] for report in reports]})
        return reports
    except BaseException as exc:
        write_json(out / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "completed_variants": [report["variant"] for report in reports]})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", default="0,1,2")
    parser.add_argument("--max-train", type=int, default=256)
    parser.add_argument("--max-heldout", type=int, default=128)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--budget", type=int, default=1 << 30)
    parser.add_argument("--row-chunk", type=int, default=16)
    parser.add_argument("--point-weight", type=float, default=1.0)
    parser.add_argument("--contrast-weight", type=float, default=1.0)
    parser.add_argument("--no-int4", action="store_true")
    parser.add_argument("--probe-span", action="store_true",
        help="Also fit the separate static native probe-span objective and evaluate seed contrast responses.")
    parser.add_argument("--probe-residual", action="store_true",
        help="Also fit one FP32 native-span residual correction on the identical INT4 control.")
    parser.add_argument("--probe-residual-controls", action="store_true",
        help="Also compare equal-rank calibration-PCA and seeded-random corrections on the same INT4 base.")
    parser.add_argument("--probe-hybrid", action="store_true",
        help="Also protect the native span plus calibration-only PCA complement directions, with same-rank PCA/random controls.")
    parser.add_argument("--hybrid-extra-rank", type=int,
        help="Additional calibration-complement directions; defaults to the native probe-span rank.")
    parser.add_argument("--residual-random-seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        ranks = tuple(int(rank) for rank in args.ranks.split(","))
    except ValueError:
        parser.error("ranks must be comma-separated integers")
    if not 1 <= args.threads <= 16 or not 1 << 20 <= args.budget <= 1 << 30:
        parser.error("threads must be 1..16 and budget 1 MiB..1 GiB")
    if not 2 <= args.max_train <= MAX_SPLIT_RECORDS or not 2 <= args.max_heldout <= MAX_SPLIT_RECORDS or not 1 <= args.row_chunk <= 32:
        parser.error(f"split position budgets must be 2..{MAX_SPLIT_RECORDS} and row chunk 1..32")
    if not all(math.isfinite(x) and x >= 0 for x in (args.point_weight, args.contrast_weight)) or not args.point_weight + args.contrast_weight:
        parser.error("response objective weights must be finite nonnegative with positive sum")
    if args.probe_residual and args.no_int4:
        parser.error("--probe-residual requires the uncorrected INT4 control; remove --no-int4")
    if args.probe_residual_controls and not args.probe_residual:
        parser.error("--probe-residual-controls requires --probe-residual")
    if args.probe_hybrid and not args.probe_residual:
        parser.error("--probe-hybrid requires --probe-residual")
    if args.hybrid_extra_rank is not None and (not args.probe_hybrid or not 1 <= args.hybrid_extra_rank < 2048):
        parser.error("--hybrid-extra-rank requires --probe-hybrid and a positive integer below 2048")
    if not 0 <= args.residual_random_seed < 2**32:
        parser.error("residual random seed must be an integer in [0, 2**32)")
    torch.set_num_threads(args.threads)
    with GLMWeightSampleReader(args.sample) as source:
        compare(source, args.archive, args.out, ranks=ranks, max_train=args.max_train, max_heldout=args.max_heldout,
            max_working_bytes=args.budget, row_chunk=args.row_chunk, include_int4=not args.no_int4,
            point_weight=args.point_weight, contrast_weight=args.contrast_weight, include_probe_span=args.probe_span,
            include_probe_residual=args.probe_residual, include_probe_residual_controls=args.probe_residual_controls,
            random_control_seed=args.residual_random_seed, include_probe_hybrid=args.probe_hybrid,
            hybrid_extra_rank=args.hybrid_extra_rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
