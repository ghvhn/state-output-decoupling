"""Test a radial formula for changes in sampled quantized GLM responses.

Original weights, captures and the saved all-INT4 control are read-only. Ordered
levels come from calibration inputs; one level's response targets are excluded
from every response fit. This is a CPU function diagnostic, not a GLM decoder.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tda.block_quant import load_block_quantization
from tda.expert_compress import _cleanup_staging
from tda.glm_weight_sample import GLMWeightSampleReader
from tda import glm_moe_sample_eval as nonlinear
from tda.shell_response_transport import fit_shells, variant_bank, predict_residual, VARIANTS
from scripts.calibrate_glm_compression import load_samples, write_json
from scripts.evaluate_glm_moe_compression import source_receipt, architecture_receipt, read_json, sha_file
from scripts.region_transport_glm import _save_adapter


def shell_metrics(reference, proposed, records, pairs, diagnostic, withheld_shell, excluded_expert):
    """Score actual outputs and independently predicted endpoint differences.

    All subset weights are taken from the existing whole-trajectory evaluator.
    A crossing here means entering/leaving the withheld response level, while
    a touching pair has at least one endpoint in that level.
    """
    if (reference.shape != proposed.shape or reference.ndim != 3 or
            reference.shape[1] != len(records) or not 0 <= excluded_expert < len(reference)):
        raise ValueError("shell metric response shapes or expert scope differ")
    n = len(records)
    labels, regions = diagnostic["shell_ids"], diagnostic["region_ids"]
    supported = diagnostic["supported"]
    if (labels.shape != (n,) or regions.shape != (n,) or supported.shape != (n,) or
            labels.dtype != torch.long or regions.dtype != torch.long or supported.dtype != torch.bool):
        raise ValueError("shell metric diagnostics have invalid shape or dtype")
    positions = {record.record_id: i for i, record in enumerate(records)}
    if len(positions) != n:
        raise ValueError("shell metric record identities are repeated")
    pair_ids, seen_endpoints = set(), set()
    for pair in pairs:
        if pair.previous_id not in positions or pair.next_id not in positions:
            raise ValueError("shell metric pair endpoint is absent")
        endpoints_key = tuple(sorted((pair.previous_id, pair.next_id)))
        if pair.pair_id in pair_ids or endpoints_key in seen_endpoints or pair.previous_id == pair.next_id:
            raise ValueError("shell metric pairs must be distinct and counted once")
        pair_ids.add(pair.pair_id)
        seen_endpoints.add(endpoints_key)
        previous, following = records[positions[pair.previous_id]], records[positions[pair.next_id]]
        if (previous.trajectory_id != following.trajectory_id or
                previous.environment_id != following.environment_id):
            raise ValueError("shell metric pair crosses trajectories")
    endpoints = torch.tensor([[positions[p.previous_id], positions[p.next_id]] for p in pairs],
                             dtype=torch.long).reshape(-1, 2)
    point_weights, pair_weights = nonlinear._metric_weights(records, pairs)
    withheld = labels == withheld_shell
    touching = withheld[endpoints].any(dim=1)
    crossing = withheld[endpoints[:, 0]] != withheld[endpoints[:, 1]]
    both = withheld[endpoints].all(dim=1)
    same_level = labels[endpoints[:, 0]] == labels[endpoints[:, 1]]
    same_region = regions[endpoints[:, 0]] == regions[endpoints[:, 1]]
    selections = {"output_difference": torch.ones(len(pairs), dtype=torch.bool),
        "withheld_shell_touching_difference": touching,
        "withheld_shell_crossing_difference": crossing,
        "withheld_shell_both_endpoints_difference": both,
        "same_level_difference": same_level, "different_level_difference": ~same_level,
        "same_region_difference": same_region, "different_region_difference": ~same_region}
    energy, error = torch.zeros(n, dtype=torch.float64), torch.zeros(n, dtype=torch.float64)
    pair_energy, pair_error = torch.zeros(len(pairs), dtype=torch.float64), torch.zeros(len(pairs), dtype=torch.float64)
    per_expert = []
    for expert in range(len(reference)):
        pe, er = torch.zeros_like(energy), torch.zeros_like(error)
        ce, cr = torch.zeros_like(pair_energy), torch.zeros_like(pair_error)
        nonlinear._point_totals(reference[expert], proposed[expert], pe, er, 128)
        nonlinear._contrast_totals(reference[expert], proposed[expert], endpoints, ce, cr, 128)
        energy += pe
        error += er
        pair_energy += ce
        pair_error += cr
        item = {"expert_index": expert, "excluded_from_operator_fit": expert == excluded_expert,
            "output": nonlinear._metrics(pe, er, point_weights),
            "withheld_shell_output": nonlinear._metrics(pe[withheld], er[withheld], point_weights[withheld])}
        for name, mask in selections.items():
            if name.startswith("withheld_shell") or name == "output_difference":
                item[name] = nonlinear._metrics(ce[mask], cr[mask], pair_weights[mask])
        per_expert.append(item)
    result = {"output": nonlinear._metrics(energy, error, point_weights),
        "withheld_shell_output": nonlinear._metrics(energy[withheld], error[withheld], point_weights[withheld]),
        "supported_withheld_shell_output": nonlinear._metrics(energy[withheld & supported], error[withheld & supported],
                                                               point_weights[withheld & supported]),
        "withheld_shell_records": int(withheld.sum()),
        "supported_withheld_shell_records": int((withheld & supported).sum()),
        "withheld_shell_touching_pairs": int(touching.sum()),
        "withheld_shell_crossing_pairs": int(crossing.sum()),
        "withheld_shell_both_endpoint_pairs": int(both.sum()), "per_expert": per_expert}
    for name, mask in selections.items():
        result[name] = nonlinear._metrics(pair_energy[mask], pair_error[mask], pair_weights[mask])
    return result


def _load_controls(directory, readers, receipts, architecture):
    controls, files = [], []
    for name, reader in zip(("gate", "up", "down"), readers):
        folder = directory / f"int4_{name}"
        codec = load_block_quantization(folder, expected_identity=reader.identity)
        if codec.bits != 4 or codec.block_size != 128:
            raise ValueError("shell control must be the unchanged block-128 INT4 baseline")
        controls.append(codec)
        files.extend({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha_file(path)}
                     for path in sorted(folder.iterdir()) if path.is_file())
    context = {"source_receipts": receipts, "architecture": architecture,
               "base_codec_metadata": [codec.metadata for codec in controls]}
    digest = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    saved_context = read_json(directory / "response_context.json")
    if saved_context != {"sha256": digest, **context}:
        raise ValueError("saved INT4 control context differs from original sources or architecture")
    return controls, context, digest, files


def run_experiment(gate, up, down, archive, metadata, control, out, *, rank=16, regions=4,
                   shells=4, withheld_shell=1, max_train=1024, max_heldout=2432,
                   max_working_bytes=4 << 30):
    for value, name, limit in ((rank, "rank", 64), (regions, "regions", 8), (shells, "shells", 8),
                               (max_train, "calibration count", 2048), (max_heldout, "heldout count", 8192),
                               (max_working_bytes, "numerical budget", 8 << 30)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"{name} is outside its fixed bound")
    if (regions < 2 or shells < 4 or isinstance(withheld_shell, bool) or
            not isinstance(withheld_shell, int) or not 0 <= withheld_shell < shells):
        raise ValueError("at least four shells and one valid predeclared withheld level are required")
    out = Path(out)
    if out.exists():
        raise FileExistsError("choose a new shell experiment directory")
    started = time.monotonic()
    scope, train, train_pairs, test, test_pairs, evidence = load_samples(
        archive, gate.identity, max_train=max_train, max_heldout=max_heldout, max_working_bytes=max_working_bytes)
    nonlinear._identities(gate, up, down, gate, scope, evidence["teacher"], evidence["origin"])
    if evidence["origin"] != "measured_model_state":
        raise ValueError("GLM shell evaluation requires measured native capture inputs")
    conversations = {split: {r["conversation_id"] for r in evidence["records"]
                             if r["split"] == split and r["conversation_id"] is not None}
                     for split in ("calibration", "heldout")}
    if conversations["calibration"] & conversations["heldout"]:
        raise ValueError("calibration and heldout conversations overlap")
    readers = (gate, up, down)
    receipts = {name: source_receipt(reader, evidence["origin"])
                for name, reader in zip(("gate", "up", "down"), readers)}
    architecture = architecture_receipt(metadata, gate, receipts, evidence["teacher"], evidence["origin"])
    experts, hidden, middle = gate.identity.expert_count, gate.identity.matrix_shape[1], gate.identity.matrix_shape[0]
    estimate = ((3 * len(train) + 4 * len(test)) * experts * hidden * 4 +
                4 * (len(train) + len(test)) * hidden * 4 + 6 * middle * hidden * 4 +
                2 * sum(reader.identity.column_count * experts for reader in readers) +
                len(train) ** 2 * 24 + experts * regions * shells * rank * hidden * 16)
    if estimate > max_working_bytes:
        raise MemoryError("shell numerical arrays and temporaries exceed the declared estimate budget")
    controls, context, context_hash, control_files = _load_controls(control, readers, receipts, architecture)
    base_bytes = sum(item["bytes"] for item in control_files)
    excluded = experts - 1
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent.resolve()))
    try:
        write_json(staging / "settings.json", {"format": "shell-glm-response-experiment-v1",
            "device": "cpu", "threads": torch.get_num_threads(), "rank": rank, "regions": regions,
            "shells": shells, "withheld_shell": withheld_shell, "polynomial_degree": 2,
            "ridge": 0.01, "cluster_seed": 0, "operator_excluded_expert": gate.identity.expert_ids[excluded],
            "shell_ordering": "within-region calibration-input radial quantiles; continuous normalized radius in formula",
            "holdout_scope": "withheld-level inputs define geometry; all their response targets are excluded from every response fit",
            "selected_variants": ["all_int4", *VARIANTS], "selection_policy": "fixed before evaluation; no heldout tuning",
            "control_directory": str(control.resolve()), "control_files": control_files,
            "working_byte_budget": max_working_bytes, "estimated_numerical_bytes": estimate,
            "additional_cloud_compute": False, "semantic_shells_validated": False,
            "native_fp8_execution_evaluated": False, "router_or_decoder_evaluated": False,
            "logits_or_generation_evaluated": False})
        write_json(staging / "response_context.json", {"sha256": context_hash, **context})
        write_json(staging / "evidence.json", evidence)
        write_json(staging / "status.json", {"status": "running", "stage": "native expert responses"})
        x = torch.stack([record.state.float() for record in train + test])
        errors = torch.empty(experts, len(train), hidden)
        reference, baseline = torch.empty(experts, len(test), hidden), torch.empty(experts, len(test), hidden)
        for slot, expert in enumerate(gate.identity.expert_ids):
            print(f"[Shells] Measuring original and saved INT4 expert {expert} ({slot + 1}/{experts}).", flush=True)
            responses = []
            for sources in (readers, tuple(controls)):
                matrices = [nonlinear._read_expert_matrix(reader, reader.identity, expert, 16) for reader in sources]
                up_values = nonlinear._project(matrices[1], x, 128)
                responses.append(nonlinear._expert_output(matrices[0], up_values, matrices[2], x, 128))
                del matrices, up_values
            errors[slot] = responses[0][:len(train)] - responses[1][:len(train)]
            reference[slot], baseline[slot] = responses[0][len(train):], responses[1][len(train):]
            del responses
        del controls
        print("[Shells] Fitting formula and controls with one response level excluded.", flush=True)
        bank = fit_shells(train, errors, expected_scope=scope, heldout_records=test,
            rank=rank, regions=regions, shells=shells, withheld_shell=withheld_shell,
            operator_excluded_expert=excluded, response_context_sha256=context_hash,
            max_working_bytes=max_working_bytes)
        del errors
        gc.collect()
        diagnostic_prediction, diagnostic = predict_residual(variant_bank(bank, "identity_transport"),
            x[len(train):], expected_scope=scope, expected_response_context_sha256=context_hash)
        del diagnostic_prediction
        reports = [{"name": "all_int4", "adapter_serialized_bytes": 0,
            "sampled_block_serialized_bytes": base_bytes,
            "metrics": shell_metrics(reference, baseline, test, test_pairs, diagnostic, withheld_shell, excluded)}]
        for name in VARIANTS:
            print(f"[Shells] Evaluating {name} after saving and restricted reload.", flush=True)
            adapter = variant_bank(bank, name)
            receipt, loaded = _save_adapter(adapter, staging / name, scope)
            correction, current = predict_residual(loaded, x[len(train):], expected_scope=scope,
                expected_response_context_sha256=context_hash)
            if name != "global" and any(not torch.equal(current[key], diagnostic[key])
                                        for key in ("region_ids", "shell_ids", "supported")):
                raise ValueError("shell controls changed geometric assignment or support")
            metrics = shell_metrics(reference, baseline + correction, test, test_pairs, diagnostic, withheld_shell, excluded)
            reports.append({"name": name, "adapter": receipt,
                "adapter_serialized_bytes": receipt["serialized_total_bytes"],
                "sampled_block_serialized_bytes": base_bytes + receipt["serialized_total_bytes"],
                "global_fallback_count": current["global_fallback_count"], "metrics": metrics})
            del adapter, loaded, correction
        write_json(staging / "results.json", {"format": "shell-glm-response-results-v1",
            "calibration_records": len(train), "heldout_records": len(test),
            "calibration_trajectories": len({r.trajectory_id for r in train}),
            "heldout_trajectories": len({r.trajectory_id for r in test}), "heldout_pairs": len(test_pairs),
            "shells": shells, "withheld_shell": withheld_shell,
            "shell_ordering": "within-region radial quantiles, using calibration inputs only",
            "evaluation_scope": "eight dequantized CPU expert functions; equal counterfactual expert weighting",
            "original_sampled_three_projection_bytes": sum(r.identity.actual_source_bytes for r in readers),
            "base_int4_serialized_bytes": base_bytes, "control_files": control_files,
            "operator_excluded_expert": gate.identity.expert_ids[excluded],
            "fit": bank["metadata"], "candidates": reports, "elapsed_seconds": time.monotonic() - started,
            "heldout_status": "development comparison; previous experiments inspected this context set",
            "full_model_or_semantic_transfer_established": False, "additional_cloud_compute": False})
        write_json(staging / "status.json", {"status": "complete", "completed_variants": [r["name"] for r in reports]})
        os.rename(staging, out)
        return out
    except BaseException:
        _cleanup_staging(staging, out.parent.resolve(), f".{out.name}-")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=ROOT / "invariants/out/compression_runs/20261002_glm_gate_sample")
    parser.add_argument("--up", type=Path, default=ROOT / "invariants/out/glm_capture_trial_20261002/up_sample")
    parser.add_argument("--down", type=Path, default=ROOT / "invariants/out/glm_capture_trial_20261002/down_sample")
    parser.add_argument("--archive", type=Path, default=ROOT / "invariants/out/glm_capture_trial_20261002/archive")
    parser.add_argument("--metadata", type=Path, default=ROOT / "invariants/out/glm_5_3_preflight")
    parser.add_argument("--control", type=Path, default=ROOT / "invariants/out/glm_capture_trial_20261002/region_transport_20261003")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--shells", type=int, default=4)
    parser.add_argument("--withheld-shell", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=1024)
    parser.add_argument("--max-heldout", type=int, default=2432)
    parser.add_argument("--max-working-mib", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16 or not 1 <= args.max_working_mib <= 8192:
        parser.error("threads or working-byte budget exceed fixed bounds")
    torch.set_num_threads(args.threads)
    with GLMWeightSampleReader(args.gate) as gate, GLMWeightSampleReader(args.up) as up, GLMWeightSampleReader(args.down) as down:
        result = run_experiment(gate, up, down, args.archive, args.metadata, args.control, args.out.resolve(),
            rank=args.rank, regions=args.regions, shells=args.shells, withheld_shell=args.withheld_shell,
            max_train=args.max_train, max_heldout=args.max_heldout, max_working_bytes=args.max_working_mib << 20)
    print(f"[Shells] Complete: {result}", flush=True)


if __name__ == "__main__":
    main()
