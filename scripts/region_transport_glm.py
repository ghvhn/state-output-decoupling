"""Bounded local test of shared regional corrections to quantized GLM experts.

Original GLM weights/captures are read-only. All three sampled expert projections
use the same saved INT4 control. New adapters predict nonlinear output errors;
they do not reconstruct original weights or export a full GLM checkpoint.
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

from tda.block_quant import quantize_reader, save_block_quantization, load_block_quantization
from tda.glm_weight_sample import GLMWeightSampleReader
from tda import glm_moe_sample_eval as nonlinear
from tda.region_response_transport import fit_regions, variant_bank, predict_residual, VARIANTS
from tda.expert_compress import _cleanup_staging
from scripts.calibrate_glm_compression import load_samples, write_json
from scripts.evaluate_glm_moe_compression import source_receipt, architecture_receipt


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_adapter(bank, directory, scope):
    directory.mkdir()
    tensors_file = directory / "tensors.pt"
    torch.save(bank["tensors"], tensors_file)
    loaded = torch.load(tensors_file, map_location="cpu", weights_only=True)
    if set(loaded) != set(bank["tensors"]) or any(
            not torch.equal(loaded[name], value) for name, value in bank["tensors"].items()):
        raise ValueError("regional adapter failed its restricted tensor round trip")
    metadata = {**bank["metadata"], "tensors_sha256": _sha(tensors_file)}
    write_json(directory / "metadata.json", metadata)
    size = sum(p.stat().st_size for p in directory.iterdir())
    # Byte totals live outside the files they count, avoiding a stale metadata
    # size after adding its own count. No hypothetical packed size is substituted.
    return {"path": directory.name, "tensor_payload_bytes": metadata["tensor_payload_bytes"],
            "serialized_total_bytes": size, "tensors_sha256": metadata["tensors_sha256"],
            "metadata_sha256": _sha(directory / "metadata.json")}, {"metadata": metadata, "tensors": loaded}


def response_metrics(reference, proposed, records, pairs, labels, excluded_expert):
    """Same trajectory/feature-balanced proxy as the earlier nonlinear test."""
    experts, n, _ = reference.shape
    point_weights, pair_weights = nonlinear._metric_weights(records, pairs)
    indices = {r.record_id: i for i, r in enumerate(records)}
    endpoints = torch.tensor([[indices[p.previous_id], indices[p.next_id]] for p in pairs], dtype=torch.long).reshape(-1, 2)
    point_energy = torch.zeros(n, dtype=torch.float64)
    point_error = torch.zeros(n, dtype=torch.float64)
    pair_energy = torch.zeros(len(pairs), dtype=torch.float64)
    pair_error = torch.zeros(len(pairs), dtype=torch.float64)
    per_expert = []
    for e in range(experts):
        energy = torch.zeros(n, dtype=torch.float64)
        error = torch.zeros(n, dtype=torch.float64)
        contrast_energy = torch.zeros(len(pairs), dtype=torch.float64)
        contrast_error = torch.zeros(len(pairs), dtype=torch.float64)
        nonlinear._point_totals(reference[e], proposed[e], energy, error, 128)
        nonlinear._contrast_totals(reference[e], proposed[e], endpoints, contrast_energy, contrast_error, 128)
        point_energy += energy
        point_error += error
        pair_energy += contrast_energy
        pair_error += contrast_error
        per_expert.append({"expert_index": e,
            "excluded_from_operator_fit": e == excluded_expert,
            "output": nonlinear._metrics(energy, error, point_weights),
            "output_difference": nonlinear._metrics(contrast_energy, contrast_error, pair_weights)})
    same = labels[endpoints[:, 0]] == labels[endpoints[:, 1]]
    return {"output": nonlinear._metrics(point_energy, point_error, point_weights),
            "output_difference": nonlinear._metrics(pair_energy, pair_error, pair_weights),
            "within_region_difference": nonlinear._metrics(pair_energy[same], pair_error[same], pair_weights[same]),
            "cross_region_difference": nonlinear._metrics(pair_energy[~same], pair_error[~same], pair_weights[~same]),
            "within_region_pairs": int(same.sum()), "cross_region_pairs": int((~same).sum()),
            "per_expert": per_expert}


def run_experiment(gate, up, down, archive, metadata, out, *, rank=16, regions=4,
                   max_train=1024, max_heldout=2432, max_working_bytes=4 << 30):
    for value, name, limit in ((rank, "rank", 64), (regions, "regions", 8),
                               (max_train, "calibration budget", 2048),
                               (max_heldout, "heldout budget", 8192),
                               (max_working_bytes, "working-byte budget", 8 << 30)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"{name} must be in1..{limit}")
    if regions < 2:
        raise ValueError("at least two geometric regions are required")
    if out.exists():
        raise FileExistsError("choose a new output directory")
    started = time.monotonic()
    scope, train, pairs, test, test_pairs, evidence = load_samples(
        archive, gate.identity, max_train=max_train, max_heldout=max_heldout,
        max_working_bytes=max_working_bytes)
    nonlinear._identities(gate, up, down, gate, scope, evidence["teacher"], evidence["origin"])
    if evidence["origin"] != "measured_model_state":
        raise ValueError("this GLM experiment requires native measured capture evidence")
    conversations = {split: {r["conversation_id"] for r in evidence["records"]
                            if r["split"] == split and r["conversation_id"] is not None}
                     for split in ("calibration", "heldout")}
    if conversations["calibration"] & conversations["heldout"]:
        raise ValueError("calibration and heldout conversations overlap")
    receipts = {name: source_receipt(reader, evidence["origin"])
                for name, reader in (("gate", gate), ("up", up), ("down", down))}
    architecture = architecture_receipt(metadata, gate, receipts, evidence["teacher"], evidence["origin"])
    e, h, middle = gate.identity.expert_count, gate.identity.matrix_shape[1], gate.identity.matrix_shape[0]
    # Numerical arrays and bounded decode temporaries, not a process RSS promise.
    estimate = ((3 * len(train) + 4 * len(test)) * e * h * 4 +
                4 * (len(train) + len(test)) * h * 4 + 6 * middle * h * 4 +
                2 * sum(reader.identity.column_count * e for reader in (gate, up, down)) +
                len(train)**2 * 24 + e * regions * rank * h * 16)
    if estimate > max_working_bytes:
        raise MemoryError("expert responses and fitting arrays exceed their declared numerical budget")
    excluded = e - 1
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent.resolve()))
    try:
        write_json(staging / "settings.json", {"format": "regional-glm-output-residual-experiment-v1",
            "device": "cpu", "threads": torch.get_num_threads(), "rank": rank, "regions": regions,
            "ridge": 0.01, "cluster_seed": 0, "support_radius_multiple": 1.5,
            "operator_excluded_expert": gate.identity.expert_ids[excluded],
            "operator_transfer_scope": "last expert not used for shared maps; its global slope/offsets are calibrated",
            "selected_variants": list(VARIANTS), "selection_policy": "declared before evaluation; no heldout selection",
            "base_codec": "all three projections symmetric INT4, block_size128, FP32 scales",
            "working_byte_budget": max_working_bytes, "estimated_numerical_bytes": estimate,
            "architecture": architecture, "source_receipts": receipts,
            "semantic_regions_validated": False, "unseen_region_transfer_evaluated": False,
            "native_fp8_activation_rounding_evaluated": False, "router_mixture_evaluated": False,
            "shared_experts_evaluated": False, "full_decoder_evaluated": False,
            "logits_or_generation_evaluated": False, "additional_cloud_compute": False})
        write_json(staging / "evidence.json", evidence)
        write_json(staging / "status.json", {"status": "running", "stage": "quantization"})
        codecs, codec_sizes = [], []
        for name, reader in (("gate", gate), ("up", up), ("down", down)):
            print(f"[Regions] Saving identical INT4 {name} control.", flush=True)
            value = quantize_reader(reader, bits=4, block_size=128, max_working_bytes=1 << 30)
            save_block_quantization(value, staging / f"int4_{name}")
            codec_sizes.append(sum(p.stat().st_size for p in (staging / f"int4_{name}").iterdir()))
            codecs.append(load_block_quantization(staging / f"int4_{name}", expected_identity=reader.identity))
            del value
        x = torch.stack([r.state.float() for r in train + test])
        errors = torch.empty(e, len(train), h)
        reference = torch.empty(e, len(test), h)
        baseline = torch.empty_like(reference)
        for slot, expert in enumerate(gate.identity.expert_ids):
            print(f"[Regions] Measuring original and compressed expert {expert} ({slot+1}/{e}).", flush=True)
            outputs = []
            for readers in ((gate, up, down), tuple(codecs)):
                matrices = [nonlinear._read_expert_matrix(reader, reader.identity, expert, 16) for reader in readers]
                up_values = nonlinear._project(matrices[1], x, 128)
                outputs.append(nonlinear._expert_output(matrices[0], up_values, matrices[2], x, 128))
                del matrices, up_values
            errors[slot] = outputs[0][:len(train)] - outputs[1][:len(train)]
            reference[slot] = outputs[0][len(train):]
            baseline[slot] = outputs[1][len(train):]
            del outputs
        response_context = {"source_receipts": receipts, "architecture": architecture,
                            "base_codec_metadata": [codec.metadata for codec in codecs]}
        context_hash = hashlib.sha256(json.dumps(response_context, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        write_json(staging / "response_context.json", {"sha256": context_hash, **response_context})
        print("[Regions] Fitting calibration-only charts and shared regional maps.", flush=True)
        bank = fit_regions(train, errors, expected_scope=scope, heldout_records=test,
            rank=rank, regions=regions, operator_excluded_expert=excluded,
            response_context_sha256=context_hash,
            max_working_bytes=max_working_bytes)
        del errors, codecs
        gc.collect()
        chart_prediction, chart = predict_residual(variant_bank(bank, "identity_transport"),
            x[len(train):], expected_scope=scope, expected_response_context_sha256=context_hash)
        del chart_prediction
        baseline_metrics = response_metrics(reference, baseline, test, test_pairs, chart["region_ids"], excluded)
        original_bytes = sum(reader.identity.actual_source_bytes for reader in (gate, up, down))
        base_bytes = sum(codec_sizes)
        reports = [{"name": "all_int4", "adapter_serialized_bytes": 0,
                    "sampled_block_serialized_bytes": base_bytes, "metrics": baseline_metrics}]
        for name in VARIANTS:
            print(f"[Regions] Evaluating {name} on untouched heldout trajectories.", flush=True)
            adapter = variant_bank(bank, name)
            receipt, loaded = _save_adapter(adapter, staging / name, scope)
            correction, diagnostic = predict_residual(loaded, x[len(train):], expected_scope=scope,
                expected_response_context_sha256=context_hash)
            proposed = baseline + correction
            metrics = response_metrics(reference, proposed, test, test_pairs, chart["region_ids"], excluded)
            reports.append({"name": name, "adapter": receipt,
                "adapter_serialized_bytes": receipt["serialized_total_bytes"],
                "sampled_block_serialized_bytes": base_bytes + receipt["serialized_total_bytes"],
                "global_fallback_count": diagnostic["global_fallback_count"], "metrics": metrics})
            del correction, proposed, adapter, loaded
        write_json(staging / "results.json", {"format": "regional-glm-output-residual-results-v1",
            "evaluation_scope": "eight dequantized CPU expert functions; counterfactual equal expert weighting",
            "calibration_records": len(train), "heldout_records": len(test),
            "calibration_trajectories": len({r.trajectory_id for r in train}),
            "heldout_trajectories": len({r.trajectory_id for r in test}),
            "heldout_pairs": len(test_pairs), "original_sampled_three_projection_bytes": original_bytes,
            "base_int4_serialized_bytes": base_bytes, "operator_excluded_expert": gate.identity.expert_ids[excluded],
            "fit": bank["metadata"], "candidates": reports,
            "elapsed_seconds": time.monotonic() - started,
            "heldout_status": "development set previously inspected in earlier experiments; fresh confirmation required",
            "comparison_is_language_quality": False, "whole_model_compression_ratio_established": False})
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
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--max-train", type=int, default=1024)
    parser.add_argument("--max-heldout", type=int, default=2432)
    parser.add_argument("--max-working-mib", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16 or not 1 <= args.max_working_mib <= 8192:
        parser.error("threads must be1..16 and numerical budget1..8192MiB")
    torch.set_num_threads(args.threads)
    with GLMWeightSampleReader(args.gate) as gate, GLMWeightSampleReader(args.up) as up, GLMWeightSampleReader(args.down) as down:
        result = run_experiment(gate, up, down, args.archive, args.metadata, args.out.resolve(),
            rank=args.rank, regions=args.regions, max_train=args.max_train, max_heldout=args.max_heldout,
            max_working_bytes=args.max_working_mib << 20)
    print(f"[Regions] Complete: {result}", flush=True)


if __name__ == "__main__":
    main()
