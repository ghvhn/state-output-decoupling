"""Compare angularly related GLM response-error maps and audit sampled winding.

The original samples and saved INT4 control are read-only. One angular sector's
responses are withheld from every fit. Static rotations are periodic; sampled
path travel is measured separately with resets at gaps and invalid geometry.
"""
from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from tda.expert_compress import _cleanup_staging
from tda.glm_weight_sample import GLMWeightSampleReader
from tda import glm_moe_sample_eval as nonlinear
from tda.rotation_geometry import unwrap_ordered_path
from tda.rotation_response_transport import fit_rotations, variant_bank, predict_geometry, predict_residual, VARIANTS
from scripts.calibrate_glm_compression import load_samples, write_json
from scripts.evaluate_glm_moe_compression import source_receipt, architecture_receipt
from scripts.region_transport_glm import _save_adapter
from scripts.shell_transport_glm import _load_controls, shell_metrics


def angular_metrics(reference, proposed, records, pairs, diagnostic, withheld_angle_bin, excluded_expert):
    common = {"region_ids": diagnostic["region_ids"], "shell_ids": diagnostic["angle_bin_ids"],
              "supported": diagnostic["supported"]}
    report = shell_metrics(reference, proposed, records, pairs, common, withheld_angle_bin, excluded_expert)
    def rename(value):
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key.replace("withheld_shell", "withheld_angle").replace("same_level", "same_angle_bin")
                       .replace("different_level", "different_angle_bin"): rename(item) for key, item in value.items()}
        return value
    return rename(report)


def winding_audit(records, evidence, diagnostic, split):
    """Audit the shortest-arc sampled path; never infer missing revolutions.

    Original response-difference metrics retain their recorded pair list.
    This separate audit can chain additional consecutive recorded positions.
    """
    rows = {row["record_id"]: row for row in evidence["records"] if row["split"] == split}
    if set(rows) != {record.record_id for record in records}:
        raise ValueError("winding position evidence does not match the selected split")
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault(record.trajectory_id, []).append(index)
    reports, observations = [], []
    for trajectory, indices in groups.items():
        indices.sort(key=lambda i: rows[records[i].record_id]["token_position"])
        positions = torch.tensor([rows[records[i].record_id]["token_position"] for i in indices], dtype=torch.long)
        phases = diagnostic["phase_rad"][indices].double()
        supported = diagnostic["supported"][indices]
        charts = [f"{trajectory}|{records[i].environment_id}|chart:{int(diagnostic['region_ids'][i])}" for i in indices]
        path = unwrap_ordered_path(phases, positions, supported=supported, trajectory_ids=charts)
        accumulated = path["accumulated_rotation_rad"]
        connections = path["connection_valid"]
        finite = torch.isfinite(accumulated)
        maximum = float(accumulated[finite].abs().amax()) if bool(finite.any()) else 0.0
        longest, current = 0, 0
        for allowed, connected in zip(path["supported"].tolist(), connections.tolist()):
            current = current + 1 if allowed and connected else int(allowed)
            longest = max(longest, current)
        item = {"trajectory_id": trajectory, "records": len(indices), "valid_connections": int(connections.sum()),
                "longest_supported_segment": longest, "max_abs_accumulated_rad": maximum,
                "max_abs_accumulated_degrees": maximum * 180 / torch.pi,
                "observed_segment_exceeds_360_degrees": maximum > 2 * torch.pi + 1e-6,
                "segments": path["segments"]}
        reports.append(item)
        for order, index in enumerate(indices):
            observations.append({"record_id": records[index].record_id,
                "token_position": int(positions[order]), "region_id": int(diagnostic["region_ids"][index]),
                "supported": bool(path["supported"][order]), "phase_rad": float(phases[order]) if torch.isfinite(phases[order]) else None,
                "segment_id": int(path["segment_ids"][order]), "connection_valid": bool(connections[order]),
                "accumulated_rotation_rad": float(accumulated[order]) if finite[order] else None,
                "net_turns": float(path["net_turns"][order]) if finite[order] else None,
                "absolute_angular_travel_rad": float(path["absolute_angular_travel_rad"][order]) if finite[order] else None,
                "branch_excess": int(path["branch_excess"][order]) if path["branch_excess_valid"][order] else None,
                "break_reason": path["break_reasons"][order]})
    return {"split": split, "scope": "signed shortest-arc travel of recorded directions in fixed calibrated planes",
        "sampling_assumption": "resolved increments represent the principal turn; unobserved extra revolutions remain undetermined",
        "closure_scope": "equal projected phase in the calibrated plane; not full latent direction/radius/state closure",
        "resets": "position gap, trajectory/environment/chart change, unsupported phase, ambiguous antipodal jump",
        "absolute_prior_winding_known": False, "winding_used_to_fit_response_maps": False,
        "trajectories": reports, "observations": observations,
        "valid_connections": sum(row["valid_connections"] for row in reports),
        "longest_supported_segment": max((row["longest_supported_segment"] for row in reports), default=0),
        "trajectories_with_observed_segment_over_360": sum(row["observed_segment_exceeds_360_degrees"] for row in reports),
        "max_abs_accumulated_degrees": max((row["max_abs_accumulated_degrees"] for row in reports), default=0)}


def _measure(readers, controls, train, test):
    gate = readers[0]
    experts, hidden = gate.identity.expert_count, gate.identity.matrix_shape[1]
    x = torch.stack([record.state.float() for record in train + test])
    errors = torch.empty(experts, len(train), hidden)
    reference, baseline = torch.empty(experts, len(test), hidden), torch.empty(experts, len(test), hidden)
    for slot, expert in enumerate(gate.identity.expert_ids):
        print(f"[Rotation] Measuring original and saved INT4 expert {expert} ({slot + 1}/{experts}).", flush=True)
        outputs = []
        for sources in (readers, tuple(controls)):
            matrices = [nonlinear._read_expert_matrix(reader, reader.identity, expert, 16) for reader in sources]
            up = nonlinear._project(matrices[1], x, 128)
            outputs.append(nonlinear._expert_output(matrices[0], up, matrices[2], x, 128))
            del matrices, up
        errors[slot] = outputs[0][:len(train)] - outputs[1][:len(train)]
        reference[slot], baseline[slot] = outputs[0][len(train):], outputs[1][len(train):]
        del outputs
    return x, errors, reference, baseline


def run_experiment(gate, up, down, archive, metadata, control, out, *, rank=16, regions=4,
                   angle_bins=4, withheld_angle_bin=1, max_train=1024, max_heldout=2432,
                   max_working_bytes=4 << 30):
    for value, name, limit in ((rank, "rank", 16), (regions, "regions", 8), (angle_bins, "angle bins", 8),
                               (max_train, "calibration count", 2048), (max_heldout, "heldout count", 8192),
                               (max_working_bytes, "numerical budget", 8 << 30)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= limit:
            raise ValueError(f"{name} is outside its fixed bound")
    if (rank < 2 or regions < 2 or angle_bins != 4 or isinstance(withheld_angle_bin, bool) or
            not isinstance(withheld_angle_bin, int) or not 0 <= withheld_angle_bin < angle_bins):
        raise ValueError("rotation requires rank2+, four angle bins and one valid predeclared withheld sector")
    out = Path(out)
    if out.exists():
        raise FileExistsError("choose a new rotation experiment directory")
    started = time.monotonic()
    scope, train, train_pairs, test, test_pairs, evidence = load_samples(
        archive, gate.identity, max_train=max_train, max_heldout=max_heldout, max_working_bytes=max_working_bytes)
    nonlinear._identities(gate, up, down, gate, scope, evidence["teacher"], evidence["origin"])
    if evidence["origin"] != "measured_model_state":
        raise ValueError("GLM angular evaluation requires native measured capture inputs")
    conversations = {split: {row["conversation_id"] for row in evidence["records"]
                            if row["split"] == split and row["conversation_id"] is not None}
                     for split in ("calibration", "heldout")}
    if conversations["calibration"] & conversations["heldout"]:
        raise ValueError("calibration and heldout conversations overlap")
    readers = (gate, up, down)
    receipts = {name: source_receipt(reader, evidence["origin"])
                for name, reader in zip(("gate", "up", "down"), readers)}
    architecture = architecture_receipt(metadata, gate, receipts, evidence["teacher"], evidence["origin"])
    experts, hidden, middle = gate.identity.expert_count, gate.identity.matrix_shape[1], gate.identity.matrix_shape[0]
    p = 3 * (rank + 1)
    estimate = ((3 * len(train) + 4 * len(test)) * experts * hidden * 4 +
                4 * (len(train) + len(test)) * hidden * 4 + 6 * middle * hidden * 4 +
                2 * sum(reader.identity.column_count * experts for reader in readers) + len(train) ** 2 * 24 +
                experts * regions * angle_bins * rank * hidden * 16 + 32 * p * p * rank * rank * 8 +
                4 * (p * rank) ** 2 * 8)
    if estimate > max_working_bytes:
        raise MemoryError("native responses and bounded rotation fitting arrays exceed the declared estimate")
    controls, context, context_hash, control_files = _load_controls(control, readers, receipts, architecture)
    base_bytes = sum(row["bytes"] for row in control_files)
    excluded = experts - 1
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent.resolve()))
    try:
        settings = {"format": "angular-glm-response-experiment-v1", "device": "cpu", "threads": torch.get_num_threads(),
            "rank": rank, "regions": regions, "angle_bins": angle_bins, "withheld_angle_bin": withheld_angle_bin,
            "degree": 2, "ridge": .01, "cluster_seed": 0, "minimum_plane_fraction": .05,
            "operator_excluded_expert": gate.identity.expert_ids[excluded],
            "selected_variants": ["all_int4", *VARIANTS], "selection_policy": "fixed before evaluation; no heldout tuning",
            "holdout_scope": "calibration inputs define PCA/planes; all withheld-sector response targets excluded from every fit",
            "response_model_is_periodic": True, "path_winding_is_separate": True,
            "control_directory": str(control.resolve()), "control_files": control_files,
            "working_byte_budget": max_working_bytes, "estimated_numerical_bytes": estimate,
            "additional_cloud_compute": False, "full_decoder_evaluated": False, "semantic_rotations_validated": False}
        write_json(staging / "settings.json", settings)
        write_json(staging / "evidence.json", evidence)
        write_json(staging / "response_context.json", {"sha256": context_hash, **context})
        write_json(staging / "status.json", {"status": "running", "stage": "native responses"})
        x, errors, reference, baseline = _measure(readers, controls, train, test)
        del controls
        print("[Rotation] Fitting radial, conjugated and periodic angular laws with one sector excluded.", flush=True)
        bank = fit_rotations(train, errors, expected_scope=scope, heldout_records=test,
            rank=rank, regions=regions, angle_bins=angle_bins, withheld_angle_bin=withheld_angle_bin,
            operator_excluded_expert=excluded, response_context_sha256=context_hash,
            max_working_bytes=max_working_bytes)
        del errors
        gc.collect()
        diagnostic = predict_geometry(bank, x[len(train):], expected_scope=scope,
            expected_response_context_sha256=context_hash, max_working_bytes=max_working_bytes)
        train_geometry = predict_geometry(bank, x[:len(train)], expected_scope=scope,
            expected_response_context_sha256=context_hash, max_working_bytes=max_working_bytes)
        winding = {"calibration": winding_audit(train, evidence, train_geometry, "calibration"),
                   "heldout": winding_audit(test, evidence, diagnostic, "heldout")}
        write_json(staging / "winding.json", winding)
        reports = [{"name": "all_int4", "adapter_serialized_bytes": 0, "sampled_block_serialized_bytes": base_bytes,
            "metrics": angular_metrics(reference, baseline, test, test_pairs, diagnostic, withheld_angle_bin, excluded)}]
        for name in VARIANTS:
            print(f"[Rotation] Saving and evaluating {name}.", flush=True)
            value = variant_bank(bank, name)
            receipt, loaded = _save_adapter(value, staging / name, scope)
            correction, current = predict_residual(loaded, x[len(train):], expected_scope=scope,
                expected_response_context_sha256=context_hash, max_working_bytes=max_working_bytes)
            if any(not torch.equal(current[key], diagnostic[key])
                                        for key in ("region_ids", "angle_bin_ids", "supported")):
                raise ValueError("angular controls changed sector assignment or geometric support")
            metrics = angular_metrics(reference, baseline + correction, test, test_pairs, diagnostic, withheld_angle_bin, excluded)
            reports.append({"name": name, "adapter": receipt, "adapter_serialized_bytes": receipt["serialized_total_bytes"],
                "sampled_block_serialized_bytes": base_bytes + receipt["serialized_total_bytes"],
                "global_fallback_count": current["global_fallback_count"], "metrics": metrics})
            del value, loaded, correction
        winding_summary = {split: {key: report[key] for key in ("valid_connections", "longest_supported_segment",
                "trajectories_with_observed_segment_over_360", "max_abs_accumulated_degrees")}
                           for split, report in winding.items()}
        write_json(staging / "results.json", {"format": "angular-glm-response-results-v1",
            "evaluation_scope": "sampled FP32 CPU expert functions; angular covariance imposed on local latent corrections",
            "calibration_records": len(train), "heldout_records": len(test), "heldout_pairs": len(test_pairs),
            "calibration_trajectories": len({r.trajectory_id for r in train}), "heldout_trajectories": len({r.trajectory_id for r in test}),
            "angle_bins": angle_bins, "withheld_angle_bin": withheld_angle_bin,
            "original_sampled_three_projection_bytes": sum(reader.identity.actual_source_bytes for reader in readers),
            "base_int4_serialized_bytes": base_bytes, "control_files": control_files,
            "operator_excluded_expert": gate.identity.expert_ids[excluded], "fit": bank["metadata"],
            "candidates": reports, "winding": winding_summary, "elapsed_seconds": time.monotonic() - started,
            "heldout_status": "development set previously inspected; fresh confirmation required",
            "additional_cloud_compute": False, "model_behavior_or_semantic_transfer_established": False})
        write_json(staging / "status.json", {"status": "complete", "completed_variants": [row["name"] for row in reports]})
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
    parser.add_argument("--angle-bins", type=int, default=4)
    parser.add_argument("--withheld-angle-bin", type=int, default=1)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 16:
        parser.error("threads must be in1..16")
    torch.set_num_threads(args.threads)
    with GLMWeightSampleReader(args.gate) as gate, GLMWeightSampleReader(args.up) as up, GLMWeightSampleReader(args.down) as down:
        output = run_experiment(gate, up, down, args.archive, args.metadata, args.control, args.out.resolve(),
            rank=args.rank, regions=args.regions, angle_bins=args.angle_bins, withheld_angle_bin=args.withheld_angle_bin)
    print(f"[Rotation] Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
