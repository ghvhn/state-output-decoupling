"""Evaluate saved gate candidates in sampled nonlinear experts on CPU.

Original up/down weights stay fixed. This is a dequantized FP32 expert proxy,
not native FP8 activation arithmetic, routing, a shared expert, or a decoder.
The default seven variants are declared before reading any evaluation metrics.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scripts.calibrate_glm_compression import load_samples, write_json
from scripts.expert_compression_test import FactorReader
from tda import expert_compress as base
from tda.block_quant import load_block_quantization
from tda.glm_capture_archive import load_archive_manifest, load_archive_entry
from tda.glm_capture_calibration import native_probe_bank
from tda.glm_weight_sample import GLMWeightSampleReader
from tda.probe_residual_compress import load_probe_residual
from tda.trajectory_compress import CalibrationRecord

PRIMARY_VARIANTS = ("symmetric_int4", "probe_residual", "residual_calibration_pca",
    "residual_seeded_random", "probe_pca_hybrid", "residual_calibration_pca_extended",
    "residual_seeded_random_extended")
FACTOR_METHODS = {"weight_pca", "trajectory_response", "probe_span"}
RESIDUAL_METHODS = set(PRIMARY_VARIANTS) - {"symmetric_int4"}
SITE = "post_attention_norm_expert_gate_up_input"
MODEL_IMPLEMENTATION = "https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py"
CONTROL_FIELDS = ("rank", "correction_span_rank", "protected_probe_rank", "extra_rank",
    "coefficient_storage_dtype", "base_variant", "base_tensor_bytes", "correction_tensor_bytes",
    "repair_rank_group", "rank_matched_to_variant", "tensor_bytes_match_matched_repair",
    "tensor_bytes_match_native_repair", "tensor_bytes_difference_from_native_repair",
    "serialized_total_bytes_difference_from_native_repair",
    "serialized_total_bytes_difference_from_matched_repair")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path, limit=16 << 20):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > limit:
        raise ValueError("JSON metadata must be an ordinary bounded file")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate metadata key")
            result[key] = value
        return result
    def finite(value):
        raise ValueError(f"nonfinite JSON constant: {value}")
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=finite)


def source_scopes(gate, up, down):
    reference = gate.identity
    if reference.projection != "gate_proj" or reference.layer != 3:
        raise ValueError("nonlinear CLI requires the original layer 3 gate sample")
    middle, hidden = reference.matrix_shape
    for reader, projection, shape in ((up, "up_proj", (middle, hidden)), (down, "down_proj", (hidden, middle))):
        identity = reader.identity
        if (identity.projection != projection or identity.matrix_shape != shape or
                any(getattr(identity, field) != getattr(reference, field)
                    for field in ("model_id", "revision", "layer", "expert_ids", "source_dtype"))):
            raise ValueError("original gate/up/down source scope or nonlinear matrix geometry differs")


def source_receipt(reader, origin):
    if origin == "synthetic_fixture" and not hasattr(reader, "manifest"):
        return {"kind": "synthetic_fixture_reader", "identity": reader.identity.to_dict(),
            "source_hashes_verified": False}
    if not isinstance(reader, GLMWeightSampleReader):
        raise ValueError("measured evaluation requires checksum-verified original sample readers")
    folder = reader.folder
    manifest = read_json(folder / "sample_manifest.json", 4 << 20)
    digest = sha_file(folder / "sample_manifest.json")
    if manifest != reader.manifest or digest != (folder / "sample_manifest.sha256").read_text().strip():
        raise ValueError("source sample receipt changed after verified loading")
    return {"kind": "verified_original_fp8_weight_and_scale_sample",
        "identity": reader.identity.to_dict(), "sample_manifest_file_sha256": digest,
        "source_metadata_sha256_from_receipt": manifest["metadata_sha256"],
        "source_tensor_sha256_from_verified_receipt": [
            {key: row[key] for key in ("filename", "expert_id", "kind", "payload_bytes", "sha256")}
            for row in manifest["tensors"]], "source_hashes_verified": True}


def architecture_receipt(metadata, gate, receipts, teacher, origin):
    if metadata is None:
        if origin != "synthetic_fixture":
            raise ValueError("measured nonlinear evaluation requires pinned source config metadata")
        return {"hidden_act": "silu", "validation": "explicit synthetic fixture formula",
            "model_implementation_source": MODEL_IMPLEMENTATION}
    path = Path(metadata) / "config.json"
    config = read_json(path, 4 << 20)
    digest = sha_file(path)
    if (config.get("hidden_act") != "silu" or config.get("model_type") != "glm_moe_dsa" or
            config.get("hidden_size") != gate.identity.matrix_shape[1] or
            config.get("moe_intermediate_size") != gate.identity.matrix_shape[0] or
            config.get("num_hidden_layers") != teacher["num_layers"]):
        raise ValueError("pinned config does not establish the sampled SiLU expert geometry")
    for receipt in receipts.values():
        if receipt.get("source_metadata_sha256_from_receipt", {}).get("config.json") != digest:
            raise ValueError("source config hash differs from the sampled weight acquisition receipt")
    return {"config_file_sha256": digest, "hidden_act": "silu", "model_type": config["model_type"],
        "hidden_size": config["hidden_size"], "moe_intermediate_size": config["moe_intermediate_size"],
        "num_hidden_layers": config["num_hidden_layers"], "model_implementation_source": MODEL_IMPLEMENTATION}


def comparison_metadata(folder, identity, origin, variants):
    folder = Path(folder)
    settings = read_json(folder / "settings.json", 1 << 20)
    evidence = read_json(folder / "evidence.json")
    reports = read_json(folder / "results.json")
    status = read_json(folder / "status.json", 128 << 10)
    if (not isinstance(settings, dict) or base.ExpertGroupIdentity.from_dict(settings.get("identity")) != identity or
            settings.get("origin") != origin or settings.get("heldout_used_for_fit") is not False):
        raise ValueError("saved comparison source identity or evidence origin differs")
    if not isinstance(evidence, dict) or evidence.get("origin") != origin:
        raise ValueError("saved comparison evidence origin differs")
    if (not isinstance(status, dict) or not isinstance(reports, list) or len(reports) != 16 or
            any(not isinstance(r, dict) for r in reports) or
            status.get("status") != "finished" or status.get("completed_variants") != [r.get("variant") for r in reports]):
        raise ValueError("nonlinear evaluation requires the finished 16-variant hybrid comparison")
    by_name = {}
    for report in reports:
        name, method, rank = report.get("variant"), report.get("method"), report.get("rank")
        expected = f"{method}_rank_{rank}" if method in FACTOR_METHODS else method
        if (not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name) or
                name != expected or name in by_name or method not in FACTOR_METHODS | set(PRIMARY_VARIANTS) or
                report.get("source_tensor_bytes_including_scales") != identity.actual_source_bytes or
                report.get("heldout_used_for_fit") is not False):
            raise ValueError("saved comparison has an invalid, duplicate or differently scoped variant")
        by_name[name] = report
    selected = tuple(by_name) if variants == "all" else PRIMARY_VARIANTS if variants is None else tuple(variants)
    if not selected or len(set(selected)) != len(selected) or any(name not in by_name for name in selected):
        raise ValueError("chosen variants must be unique saved comparison names")
    hashes = {name: sha_file(folder / name) for name in ("settings.json", "evidence.json", "results.json", "status.json")}
    return settings, evidence, by_name, selected, hashes


def load_seed_records(archive, manifest, scope, bank, *, max_working_bytes):
    """Copy actual recorded pole states, retaining their original storage dtype."""
    name = f"layer_{scope.layer}_expert_input"
    if (bank.get("scope") != {"teacher": manifest["identity"], "tensor_name": name, "state_site": SITE} or
            bank.get("origin") != manifest["origin"] or bank.get("heldout_used_to_fit_axes") is not False):
        raise ValueError("raw seed source bank differs from the original teacher/site/origin")
    ids, poles = bank["probe_ids"], bank["pole_record_ids"]
    if len(ids) != len(poles) or not 1 <= len(ids) <= 128:
        raise ValueError("raw seed endpoint count exceeds its bound")
    wanted = {}
    for probe, endpoints in zip(ids, poles):
        for side, polarity in (("negative", -1), ("positive", 1)):
            record_id = endpoints[side]
            if record_id in wanted:
                raise ValueError("seed endpoints must have distinct observed IDs")
            wanted[record_id] = (probe, side, polarity)
    chosen = []
    for entry in manifest["entries"]:
        if entry["role"] != "capture":
            continue
        rows = [(i, row) for i, row in enumerate(entry["metadata"]["rows"]) if row["id"] in wanted]
        if rows:
            if entry["metadata"]["state_sites"].get(name) != SITE:
                raise ValueError("raw seed poles require the exact original gate/up input site")
            for _, row in rows:
                probe, _, polarity = wanted[row["id"]]
                if (row.get("kind") != "probe_framing" or row["split"] != "calibration" or
                        row.get("probe_id") != probe or row.get("polarity") != polarity):
                    raise ValueError("raw seed pole metadata differs from its bank endpoint")
            chosen.append((entry, rows))
    count = sum(len(rows) for _, rows in chosen)
    estimate = count * scope.matrix_input_width * 4 + 2 * max((entry["bytes"] for entry, _ in chosen), default=0)
    if estimate > max_working_bytes:
        raise MemoryError("raw seed states and chunk buffers exceed the remaining working-byte budget")
    if count != len(wanted):
        raise ValueError("original archive is missing an exact raw seed endpoint")
    records, digest, record_info = {probe: {} for probe in ids}, hashlib.sha256(), []
    for entry, rows in chosen:
        tensors = load_archive_entry(archive, entry, identity=manifest["identity"])
        for index, row in rows:
            probe, side, _ = wanted[row["id"]]
            state = tensors[name][index].detach().clone().contiguous()
            records[probe][side] = CalibrationRecord(row["id"], row["environment_id"], row["trajectory_id"],
                scope, state, row.get("provenance_weight", 1.0), row.get("stimulation_weight", 1.0),
                prediction_correct=None, origin=manifest["origin"])
            item = {"probe_id": probe, "side": side, "record_id": row["id"], "trajectory_id": row["trajectory_id"],
                "token_position": row["token_position"], "dtype": str(state.dtype), "capture_chunk": entry["file"]}
            digest.update(json.dumps(item, sort_keys=True).encode())
            digest.update(state.view(torch.uint8).numpy().tobytes())
            record_info.append(item)
        del tensors
    return records, {"raw_storage_dtype_evidence_sha256": digest.hexdigest(),
        "pole_records": count, "probe_count": len(ids), "records": record_info,
        "source_chunks": [{"file": entry["file"], "bytes": entry["bytes"], "sha256": entry["sha256"]} for entry, _ in chosen],
        "inputs": "exact observed archive states; no normalized-direction reconstruction",
        "estimated_read_peak_bytes": estimate}


def load_gate_candidate(folder, report, identity, *, max_working_bytes):
    path = Path(folder) / report["variant"]
    if path.is_symlink() or not path.is_dir():
        raise ValueError("candidate must be an ordinary saved artifact directory")
    method = report["method"]
    if method in FACTOR_METHODS:
        encoded = base.load_expert_compression(path, expected_identity=identity, max_artifact_bytes=max_working_bytes)
        candidate, metadata = FactorReader(encoded), encoded.metadata
        if encoded.rank != report["rank"]:
            raise ValueError("saved candidate rank differs from comparison report")
    elif method == "symmetric_int4":
        candidate = load_block_quantization(path, expected_identity=identity, max_artifact_bytes=max_working_bytes)
        metadata = candidate.metadata
        if candidate.bits != 4:
            raise ValueError("saved comparison control is not INT4")
    else:
        candidate = load_probe_residual(path, expected_identity=identity, max_artifact_bytes=max_working_bytes)
        metadata = candidate.metadata
        if (candidate.span_basis.shape[1] != report["correction_span_rank"] or
                metadata["coefficient_dtype"] != report["coefficient_storage_dtype"]):
            raise ValueError("saved correction rank or coefficient storage differs")
        baseline = Path(folder) / "symmetric_int4"
        if (metadata["base_metadata_sha256"] != sha_file(baseline / "metadata.json") or
                metadata["base_factors_sha256"] != sha_file(baseline / "factors.pt")):
            raise ValueError("candidate does not contain the identical saved INT4 baseline")
    if (any(metadata.get(field) != report.get(field) for field in
            ("serialized_total_bytes", "factor_tensor_bytes", "serialized_storage_ratio")) or
            report.get("fit_evidence", metadata.get("objective")) != metadata.get("objective")):
        raise ValueError("saved candidate bytes or fitting evidence differ from comparison report")
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("candidate artifacts may not contain linked files")
        if item.is_file():
            files.append({"path": item.relative_to(path).as_posix(), "bytes": item.stat().st_size, "sha256": sha_file(item)})
    if sum(item["bytes"] for item in files) != metadata["serialized_total_bytes"]:
        raise ValueError("recursive candidate serialized bytes differ from the saved report")
    return candidate, {"files": files, "metadata_file_sha256": sha_file(path / "metadata.json"),
        "checksums_and_exact_source_identity_verified": True}


def evaluate_saved_comparison(gate, up, down, archive, comparison, out, *, variants=None,
        metadata=None, row_chunk=16, batch_records=128, expert_chunk=1,
        max_working_bytes=1 << 30, expected_origin="measured_model_state"):
    out = Path(out)
    if out.exists() or out.is_symlink():
        raise FileExistsError("choose a new nonlinear evaluation output directory")
    for value, name, maximum in ((row_chunk, "row chunk", 32), (batch_records, "record batch", 256),
                                (max_working_bytes, "working-byte budget", 1 << 30)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be a positive integer <= {maximum}")
    if isinstance(expert_chunk, bool) or not isinstance(expert_chunk, int) or expert_chunk != 1:
        raise ValueError("sampled nonlinear expert chunk must be exactly 1")
    if expected_origin not in ("measured_model_state", "synthetic_fixture"):
        raise ValueError("explicit measured or synthetic record origin is required")
    source_scopes(gate, up, down)
    settings, stored_evidence, by_name, selected, comparison_hashes = comparison_metadata(
        comparison, gate.identity, expected_origin, variants)
    manifest = load_archive_manifest(archive, expected_identity=stored_evidence["teacher"])
    if manifest["origin"] != expected_origin:
        raise ValueError("archive origin differs from the explicitly requested origin")
    receipts = {name: source_receipt(reader, expected_origin) for name, reader in (("gate", gate), ("up", up), ("down", down))}
    architecture = architecture_receipt(metadata, gate, receipts, manifest["identity"], expected_origin)
    scope, train, pairs, heldout, heldout_pairs, evidence = load_samples(archive, gate.identity,
        max_train=stored_evidence["calibration"]["position_budget"],
        max_heldout=stored_evidence["heldout"]["position_budget"], max_working_bytes=max_working_bytes)
    if evidence != stored_evidence:
        raise ValueError("archive evidence or whole-trajectory selection differs from the finished comparison")
    bank = native_probe_bank(archive, tensor_name=f"layer_{gate.identity.layer}_expert_input")
    old_bank = read_json(Path(comparison) / "native_probe_bank.json", 1 << 20)
    if old_bank.get("evidence_sha256") != bank["evidence_sha256"]:
        raise ValueError("native seed bank differs from the saved comparison evidence")
    resident_states = sum(record.state.numel() * record.state.element_size() for record in train + heldout)
    bank_bytes = sum(value.numel() * value.element_size() for value in bank.values() if isinstance(value, torch.Tensor))
    seeds, seed_evidence = load_seed_records(archive, manifest, scope, bank,
        max_working_bytes=max_working_bytes - resident_states - bank_bytes)
    from tda.glm_moe_sample_eval import evaluate_sampled_moe_candidate
    out.parent.mkdir(parents=True, exist_ok=True)
    parent = out.parent.resolve()
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=parent))
    started, reports = time.monotonic(), []
    try:
        write_json(staging / "settings.json", {"format": "saved-gate-nonlinear-evaluation-v1",
            "device": "cpu", "threads": torch.get_num_threads(), "row_chunk": row_chunk, "batch_records": batch_records,
            "expert_chunk": 1, "working_byte_budget": max_working_bytes, "record_origin": expected_origin,
            "selected_variants": list(selected), "selection_policy": "predeclared candidate names; no selection by heldout metrics",
            "comparison_json_file_sha256": comparison_hashes, "source_receipts": receipts, "architecture": architecture,
            "fitting_performed": False, "original_up_down_unchanged": True,
            "native_fp8_activation_rounding_evaluated": False, "router_mixture_evaluated": False,
            "shared_expert_evaluated": False, "full_decoder_evaluated": False, "logit_quality_evaluated": False})
        write_json(staging / "evidence.json", {"original_comparison_evidence": evidence,
            "source_archive_status": manifest["status"], "raw_seed_poles": seed_evidence,
            "native_seed_bank_evidence_sha256": bank["evidence_sha256"],
            "routing_coverage": evidence["router_coverage"],
            "routing_used_to_weight_responses": False,
            "seed_semantic_meanings_validated": False, "seed_poles_are_heldout": False})
        write_json(staging / "status.json", {"status": "running", "completed_variants": []})
        for name in selected:
            candidate = None
            try:
                print(f"[Sampled nonlinear] Evaluating {name} on CPU.", flush=True)
                candidate, artifact = load_gate_candidate(comparison, by_name[name], gate.identity, max_working_bytes=max_working_bytes)
                result = evaluate_sampled_moe_candidate(gate, up, down, candidate,
                    {"calibration": (train, pairs), "heldout": (heldout, heldout_pairs)}, expected_scope=scope,
                    native_seed_bank=bank, native_seed_records=seeds, expected_teacher=manifest["identity"],
                    row_chunk=row_chunk, batch_records=batch_records, max_working_bytes=max_working_bytes,
                    expected_origin=expected_origin)
                original = by_name[name]
                report = {**result, "variant": name, "compression_method": original["method"],
                    "serialized_total_bytes": original["serialized_total_bytes"],
                    "factor_tensor_bytes": original["factor_tensor_bytes"],
                    "source_gate_tensor_bytes_including_scales": gate.identity.actual_source_bytes,
                    "unchanged_up_down_source_tensor_bytes_including_scales": up.identity.actual_source_bytes + down.identity.actual_source_bytes,
                    "candidate_artifact_verification": artifact, "candidate_refit": False,
                    **{key: original[key] for key in CONTROL_FIELDS if key in original}}
                reports.append(report)
                write_json(staging / "results.json", reports)
                write_json(staging / "status.json", {"status": "running", "completed_variants": [r["variant"] for r in reports]})
                print(f"[Sampled nonlinear] {name}: heldout point={result['splits']['heldout']['point_relative_rmse']}; output difference={result['splits']['heldout']['contrast_relative_rmse']}.", flush=True)
            finally:
                del candidate
                gc.collect()
        write_json(staging / "status.json", {"status": "finished", "elapsed_seconds": time.monotonic() - started,
            "completed_variants": [r["variant"] for r in reports]})
        if staging.parent.resolve() != parent or out.parent.resolve() != parent:
            raise ValueError("atomic output paths escaped the intended parent")
        if out.exists() or out.is_symlink():
            raise FileExistsError("nonlinear output appeared before atomic publication")
        os.rename(staging, out)
        return reports
    except BaseException as exc:
        write_json(staging / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "completed_variants": [r["variant"] for r in reports]})
        setattr(exc, "evaluation_staging_directory", str(staging))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True, help="Original gate sample")
    parser.add_argument("--up-sample", type=Path, required=True)
    parser.add_argument("--down-sample", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "invariants/out/glm_5_3_preflight")
    parser.add_argument("--variants", help="Comma-separated saved names, or all; default is the predeclared seven INT4/repair variants")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--budget", type=int, default=1 << 30)
    parser.add_argument("--row-chunk", type=int, default=16)
    parser.add_argument("--batch-records", type=int, default=128)
    parser.add_argument("--expert-chunk", type=int, choices=(1,), default=1)
    args = parser.parse_args(argv)
    if (not 1 <= args.threads <= 16 or not 1 << 20 <= args.budget <= 1 << 30 or
            not 1 <= args.row_chunk <= 32 or not 1 <= args.batch_records <= 256):
        parser.error("threads 1..16; budget 1 MiB..1 GiB; rows 1..32; record batch 1..256")
    variants = "all" if args.variants == "all" else tuple(name.strip() for name in args.variants.split(",")) if args.variants is not None else None
    torch.set_num_threads(args.threads)
    try:
        with ExitStack() as stack:
            sources = [stack.enter_context(GLMWeightSampleReader(path)) for path in (args.sample, args.up_sample, args.down_sample)]
            evaluate_saved_comparison(*sources, args.archive, args.comparison, args.out, variants=variants,
                metadata=args.metadata, row_chunk=args.row_chunk, batch_records=args.batch_records,
                expert_chunk=args.expert_chunk, max_working_bytes=args.budget)
    except Exception as exc:
        staging = getattr(exc, "evaluation_staging_directory", None)
        suffix = f" Failed output retained at {staging}." if staging else ""
        parser.exit(2, f"[Sampled nonlinear] {type(exc).__name__}: {exc}.{suffix}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
