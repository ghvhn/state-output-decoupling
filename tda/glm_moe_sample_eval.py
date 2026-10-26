"""Scoped CPU nonlinear expert responses for a bounded native GLM sample.

The reference is down(SiLU(gate(x)) * up(x)), using dequantized original
FP8/scale matrices and FP32 CPU arithmetic. Only gate is replaced. This is
neither the native dynamic-FP8 kernel nor a router-weighted MoE layer, modified
decoder, logit intervention, generation experiment, or language-quality test.
State differences are f(next) - f(previous), never f(next - previous).
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import torch
import torch.nn.functional as functional

from . import expert_compress as base
from .glm_capture_archive import validate_identity
from .glm_capture_calibration import native_probe_bank_fingerprint
from .probe_span_compress import SITE, _SOURCE_DTYPES
from .trajectory_compress import (InputStateScope, CalibrationRecord, _count,
    _text, _snapshot, _fingerprint)


METHOD = "sampled_dequantized_fp32_cpu_silu_expert_gate_intervention_v1"
MAX_RECORDS = 8192
MAX_PAIRS = 4096
MAX_SEED_PROBES = 2048
_TENSOR_FIELDS = frozenset({"weights", "codes", "scales", "span_basis", "coefficients",
    "mean", "basis", "result", "base_quantization", "directions", "centers", "contrast_magnitudes"})


def _resident_tensor_bytes(*objects):
    """Count reader-owned tensor storage once, including codec base tensors.

    This deliberately ignores arbitrary metadata graphs and read-only NumPy
    source memmaps. The latter are file mappings, not decoded dense matrices;
    their decoded working slices are included separately in the memory plan.
    A custom reader must implement bounded read_columns without hidden dense
    caches. The numerical bound is not an operating-system RSS promise.
    """
    storage, seen = {}, set()

    def visit(value, depth=0):
        if depth > 8:
            raise ValueError("reader tensor ownership exceeds its inspection bound")
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError("CPU proxy reader tensors must be CPU strided storage")
            allocation = value.untyped_storage()
            storage[(allocation.data_ptr(), allocation.nbytes())] = allocation.nbytes()
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, dict):
            fields = value
        else:
            fields = getattr(value, "__dict__", {})
        if not isinstance(fields, dict):
            return
        for name in _TENSOR_FIELDS & fields.keys():
            visit(fields[name], depth + 1)

    for item in objects:
        visit(item)
    return sum(storage.values())


def _layout(identity):
    return identity.layout.replace(f"separate {identity.projection.removesuffix('_proj')} projection;", "separate projection;")


def _identities(gate_source, up_source, down_source, candidate, scope, teacher, origin):
    readers = (gate_source, up_source, down_source, candidate)
    identities = tuple(getattr(reader, "identity", None) for reader in readers)
    if any(not isinstance(value, base.ExpertGroupIdentity) for value in identities):
        raise ValueError("all expert readers require an exact ExpertGroupIdentity")
    gate, up, down, proposed = identities
    if gate.projection != "gate_proj" or up.projection != "up_proj" or down.projection != "down_proj":
        raise ValueError("nonlinear reference requires gate, up and down projection identities")
    if proposed != gate:
        raise ValueError("candidate may replace only the exact gate source group")
    if not re.fullmatch(r"[0-9a-f]{40}", gate.revision):
        raise ValueError("nonlinear source requires an immutable full revision SHA")
    if not 1 <= gate.expert_count <= 8:
        raise ValueError("sampled nonlinear evaluation is bounded to at most eight experts")
    middle, hidden = gate.matrix_shape
    if up.matrix_shape != (middle, hidden) or down.matrix_shape != (hidden, middle):
        raise ValueError("gate/up must be [M,H] and down must be [H,M]")
    for source in (up, down):
        if (source.model_id != gate.model_id or source.revision != gate.revision or
                source.layer != gate.layer or source.expert_ids != gate.expert_ids or
                source.source_dtype != gate.source_dtype or _layout(source) != _layout(gate)):
            raise ValueError("original gate/up/down differ in teacher, layer, expert order, precision or layout")
    if (not isinstance(scope, InputStateScope) or scope.state_site != SITE or
            scope != InputStateScope.from_identity(gate, state_site=SITE)):
        raise ValueError("nonlinear inputs require the exact normalized native gate/up source scope")
    if origin not in ("measured_model_state", "synthetic_fixture"):
        raise ValueError("expected origin must distinguish native evidence and synthetic fixtures")
    if origin == "measured_model_state" and gate.source_dtype != "float8_e4m3fn":
        raise ValueError("measured GLM sample reference requires original native E4M3 FP8/scales")
    if teacher is not None:
        teacher = validate_identity(teacher)
        if (teacher["model_id"] != gate.model_id or teacher["revision"] != gate.revision or
                teacher["hidden_size"] != hidden or gate.layer >= teacher["num_layers"] or
                _SOURCE_DTYPES.get(teacher["source_precision"]) != gate.source_dtype):
            raise ValueError("capture teacher differs from the original sampled expert groups")
    if any(not callable(getattr(reader, "read_columns", None)) for reader in readers):
        raise ValueError("every scoped expert reader needs bounded read_columns")
    return identities, teacher


def _options(splits, max_records, max_pairs, row_chunk, batch_records, max_seed_probes, max_working_bytes):
    for value, name in ((max_records, "max_records"), (max_pairs, "max_pairs"),
                        (max_seed_probes, "max_seed_probes"), (row_chunk, "row_chunk"),
                        (batch_records, "batch_records"), (max_working_bytes, "max_working_bytes")):
        _count(value, name, 1)
    if max_records > MAX_RECORDS or max_pairs > MAX_PAIRS or max_seed_probes > MAX_SEED_PROBES:
        raise ValueError("nonlinear sample count cap exceeds the fixed evidence bounds")
    if row_chunk > 32 or batch_records > 256:
        raise ValueError("nonlinear row chunks must be <=32 and record batches <=256")
    if not isinstance(splits, dict) or not splits or set(splits) - {"calibration", "heldout"}:
        raise ValueError("evaluation requires separately named calibration and/or heldout splits")
    total_records, total_pairs = 0, 0
    for name, value in splits.items():
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("each evaluation split must contain (records, pairs)")
        records, pairs = value
        if not isinstance(records, (list, tuple)) or not isinstance(pairs, (list, tuple)) or not records:
            raise ValueError("each evaluation split requires bounded nonempty records and bounded pairs")
        if len(records) > max_records or len(pairs) > max_pairs:
            raise MemoryError("nonlinear split record or pair count exceeds its sample budget")
        total_records += len(records)
        total_pairs += len(pairs)
    return total_records, total_pairs


def _seed_options(bank, records, gate, scope, teacher, origin, max_seed_probes):
    if bank is None and records is None:
        return None, (), ()
    if bank is None or records is None:
        raise ValueError("seed endpoint evaluation requires both an auditable bank and exact raw pole records")
    if (not isinstance(bank, dict) or bank.get("format") != "glm-native-probe-bank-v1" or
            bank.get("origin") != origin or bank.get("heldout_used_to_fit_axes") is not False or
            bank.get("semantic_meanings_validated") is not False):
        raise ValueError("seed poles require unqualified calibration-only native bank evidence")
    bank_scope = bank.get("scope")
    if not isinstance(bank_scope, dict) or set(bank_scope) != {"teacher", "tensor_name", "state_site"}:
        raise ValueError("seed bank requires the exact source teacher and native input site")
    seed_teacher = validate_identity(bank_scope["teacher"])
    if teacher is not None and teacher != seed_teacher:
        raise ValueError("seed bank teacher differs from the evaluation capture teacher")
    if (seed_teacher["model_id"] != gate.model_id or seed_teacher["revision"] != gate.revision or
            seed_teacher["hidden_size"] != gate.matrix_shape[1] or gate.layer >= seed_teacher["num_layers"] or
            _SOURCE_DTYPES.get(seed_teacher["source_precision"]) != gate.source_dtype or
            bank_scope["tensor_name"] != f"layer_{gate.layer}_expert_input" or bank_scope["state_site"] != SITE):
        raise ValueError("seed bank differs from the sampled teacher, layer, input site or source precision")
    if bank.get("scope_sha256") != hashlib.sha256(json.dumps(bank_scope, sort_keys=True).encode()).hexdigest():
        raise ValueError("seed bank scope checksum differs")
    ids, poles = bank.get("probe_ids"), bank.get("pole_record_ids")
    if not isinstance(ids, list) or not 0 < len(ids) <= max_seed_probes:
        raise ValueError("raw seed records require bounded ordered bank probe IDs")
    for probe in ids:
        _text(probe, "probe ID")
    if (len(set(ids)) != len(ids) or
            not isinstance(poles, list) or len(poles) != len(ids) or
            not isinstance(records, dict) or set(records) != set(ids)):
        raise ValueError("raw seed records must match the bounded ordered bank probe IDs")
    raw = []
    for probe, pole in zip(ids, poles):
        item = records[probe]
        if (not isinstance(pole, dict) or set(pole) != {"negative", "positive"} or
                not isinstance(item, dict) or set(item) != {"negative", "positive"}):
            raise ValueError("every raw seed requires exactly its negative and positive poles")
        for side in ("negative", "positive"):
            _text(pole[side], "seed pole record ID")
            record = item[side]
            if (not isinstance(record, CalibrationRecord) or record.record_id != pole[side] or
                    record.scope != scope or record.origin != origin):
                raise ValueError("raw seed pole ID, scope or origin differs from its bank evidence")
            raw.append(record)
        if pole["negative"] == pole["positive"]:
            raise ValueError("seed polarities must have distinct observed record IDs")
    return seed_teacher, tuple(raw), tuple(ids)


def _validate_seed_values(bank, records, max_seed_probes):
    """Validate recorded pole geometry without replacing those observed inputs."""
    n, hidden = len(bank["probe_ids"]), records[0].scope.matrix_input_width
    for name, shape in (("directions", (n, hidden)), ("centers", (n, hidden)), ("contrast_magnitudes", (n,))):
        value = bank.get(name)
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32 or
                tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
            raise ValueError("seed bank geometry must contain finite exact-shape FP32 CPU tensors")
    for field in ("unpaired_probe_ids", "zero_contrast_probe_ids", "source_capture_sha256"):
        values = bank.get(field)
        if not isinstance(values, list) or len(values) > max_seed_probes:
            raise ValueError("seed evidence identifiers exceed their bound")
        for value in values:
            _text(value, field)
            if field == "source_capture_sha256" and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("seed capture references require SHA256 identities")
    if not bank["source_capture_sha256"]:
        raise ValueError("seed endpoints require their measured source capture references")
    if bank.get("evidence_sha256") != native_probe_bank_fingerprint(bank):
        raise ValueError("seed bank evidence checksum differs")
    for i in range(n):
        negative, positive = records[2*i].state, records[2*i+1].state
        contrast = positive - negative
        magnitude = contrast.norm()
        if not bool(torch.isfinite(contrast).all()) or not torch.isfinite(magnitude) or magnitude <= 1e-6:
            raise ValueError("raw seed pole contrast must be finite and nonzero")
        # The bank stores rounded derived FP32 geometry. Match how the bank
        # was constructed, but feed the exact observed raw records below.
        if (not torch.equal((positive + negative) / 2, bank["centers"][i]) or
                not torch.equal(magnitude, bank["contrast_magnitudes"][i]) or
                not torch.equal(contrast / magnitude, bank["directions"][i])):
            raise ValueError("raw seed states do not reproduce the bank's measured center and contrast")


def _metric_weights(records, pairs):
    """Exactly the trajectory._features normalization, without a dense delta X.

    A trajectory contributes its average provenance*stimulation quality.
    Record and endpoint-degree weighting within it matches the canonical
    linear evaluator. Correct and incorrect predictions both remain evidence.
    """
    quality = {r.record_id: r.provenance_weight * r.stimulation_weight for r in records}
    by_id = {r.record_id: r for r in records}
    groups, pair_groups = {}, {}
    degrees = {r.record_id: 0 for r in records}
    for r in records:
        groups.setdefault(r.trajectory_id, []).append(r.record_id)
    for p in pairs:
        degrees[p.previous_id] += 1
        degrees[p.next_id] += 1
        pair_groups.setdefault(by_id[p.previous_id].trajectory_id, []).append(p)
    point, contrast = {}, {}
    for name, ids in groups.items():
        total = sum(quality[rid] for rid in ids)
        group_quality = total / len(ids)
        for rid in ids:
            point[rid] = group_quality * quality[rid] / total if total else 0.0
        related = pair_groups.get(name, ())
        raw = {p.pair_id: .5 * (quality[p.previous_id] / degrees[p.previous_id] +
            quality[p.next_id] / degrees[p.next_id]) for p in related}
        denominator = sum(raw.values())
        for p in related:
            contrast[p.pair_id] = group_quality * raw[p.pair_id] / denominator if denominator else 0.0
    return (torch.tensor([point[r.record_id] for r in records], dtype=torch.float64),
        torch.tensor([contrast[p.pair_id] for p in pairs], dtype=torch.float64))


def _read_expert_matrix(reader, identity, expert, row_chunk):
    rows, columns = identity.matrix_shape
    result = torch.empty((rows, columns), dtype=torch.float32)
    for start in range(0, rows, row_chunk):
        stop = min(rows, start + row_chunk)
        if reader.identity != identity:
            raise ValueError("expert reader source scope changed before its bounded read")
        value = reader.read_columns(start * columns, stop * columns, expert_ids=(expert,))
        if reader.identity != identity:
            raise ValueError("expert reader source scope changed during its bounded read")
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype not in base._READ_DTYPES or
                tuple(value.shape) != (1, (stop-start)*columns) or not bool(torch.isfinite(value).all())):
            raise ValueError("expert reader returned an invalid or nonfinite bounded source slice")
        result[start:stop] = value.detach().reshape(stop-start, columns)
    return result


def _project(matrix, states, batch_records):
    result = torch.empty((len(states), len(matrix)), dtype=torch.float32)
    for start in range(0, len(states), batch_records):
        stop = min(len(states), start + batch_records)
        value = states[start:stop] @ matrix.T
        if not bool(torch.isfinite(value).all()):
            raise ValueError("native projection overflowed FP32 CPU arithmetic")
        result[start:stop] = value
    return result


def _expert_output(gate, up_values, down, states, batch_records):
    result = torch.empty((len(states), len(down)), dtype=torch.float32)
    for start in range(0, len(states), batch_records):
        stop = min(len(states), start + batch_records)
        gate_values = states[start:stop] @ gate.T
        if not bool(torch.isfinite(gate_values).all()):
            raise ValueError("gate projection overflowed FP32 CPU arithmetic")
        middle = functional.silu(gate_values) * up_values[start:stop]
        if not bool(torch.isfinite(middle).all()):
            raise ValueError("SiLU-times-up expert activation overflowed FP32 CPU arithmetic")
        value = middle @ down.T
        if not bool(torch.isfinite(value).all()):
            raise ValueError("expert down projection overflowed FP32 CPU arithmetic")
        result[start:stop] = value
    return result


def _point_totals(reference, proposed, energies, errors, batch_records):
    for start in range(0, len(reference), batch_records):
        stop = min(len(reference), start + batch_records)
        source = reference[start:stop].double()
        delta = proposed[start:stop].double() - source
        energies[start:stop] += source.square().sum(dim=1)
        errors[start:stop] += delta.square().sum(dim=1)


def _contrast_totals(reference, proposed, endpoints, energies, errors, batch_records):
    for start in range(0, len(endpoints), batch_records):
        stop = min(len(endpoints), start + batch_records)
        negative, positive = endpoints[start:stop].unbind(dim=1)
        # Convert endpoint outputs before subtraction, avoiding artificial
        # FP32 contrast overflow after otherwise finite expert projections.
        source = reference[positive].double() - reference[negative].double()
        candidate = proposed[positive].double() - proposed[negative].double()
        energies[start:stop] += source.square().sum(dim=1)
        errors[start:stop] += (candidate-source).square().sum(dim=1)


def _metrics(energy, error, weights):
    if (not bool(torch.isfinite(energy).all() & torch.isfinite(error).all() & torch.isfinite(weights).all()) or
            bool((energy < 0).any() | (error < 0).any() | (weights < 0).any())):
        raise ValueError("nonlinear aggregate energy, error or weighting is invalid")
    valid = energy > 0
    mass = float(weights[valid].sum())
    squared = float((weights[valid] * error[valid] / energy[valid]).sum()) / mass if mass else None
    return {"relative_rmse": math.sqrt(squared) if squared is not None else None,
        "positive_energy_weight_mass": mass, "zero_source_features": int((~valid).sum()),
        "zero_source_absolute_error_sq": float(error[~valid].sum()),
        "zero_weight_features": int((weights == 0).sum()), "source_energy_sq": float(energy.sum()),
        "absolute_error_sq": float(error.sum())}


def _prefixed_metrics(report, prefix, metrics):
    report.update({f"{prefix}_{name}": value for name, value in metrics.items()})


def evaluate_sampled_moe_candidate(gate_source, up_source, down_source, candidate, splits, *,
        expected_scope, native_seed_bank=None, native_seed_records=None, expected_teacher=None,
        row_chunk=16, batch_records=128, max_records=8192, max_pairs=4096,
        max_seed_probes=128, max_working_bytes=1 << 30, expected_origin="measured_model_state"):
    """Evaluate one already-fitted gate reader without fitting any state.

    ``splits`` maps calibration/heldout to (records, within-trajectory pairs).
    ``native_seed_records`` maps each bank probe ID to its exact observed
    negative/positive CalibrationRecords. No poles are reconstructed from Q.
    Caller loads one candidate at a time; compressed tensor payloads, snapshots,
    one-expert dense matrices, all-state outputs and bounded temporary arrays
    are included in the conservative working-byte check before source reads.
    """
    n_records, n_pairs = _options(splits, max_records, max_pairs, row_chunk, batch_records,
        max_seed_probes, max_working_bytes)
    identities, teacher = _identities(gate_source, up_source, down_source, candidate,
        expected_scope, expected_teacher, expected_origin)
    gate, up, down, candidate_identity = identities
    seed_teacher, raw_seeds, probe_ids = _seed_options(native_seed_bank, native_seed_records,
        gate, expected_scope, teacher, expected_origin, max_seed_probes)
    teacher = teacher if teacher is not None else seed_teacher
    if expected_origin == "measured_model_state" and teacher is None:
        raise ValueError("native nonlinear evidence requires its explicit capture teacher or measured seed bank teacher")
    middle, hidden = gate.matrix_shape
    count = n_records + len(raw_seeds)
    resident = _resident_tensor_bytes(gate_source, up_source, down_source, candidate, native_seed_bank)
    # Original callers' states (FP32 upper bound), frozen states, stacked input;
    # two full one-expert outputs and reusable up response; four decoded expert
    # matrices; bounded decode/matmul/FP64 metric scratch. Not exact process RSS.
    estimate = (resident + 12 * count * hidden + 8 * count * hidden + 4 * count * middle +
        16 * middle * hidden + 48 * row_chunk * max(hidden, middle) +
        48 * min(batch_records, count) * (hidden + middle) +
        64 * (count + n_pairs + len(probe_ids)) + 2 * 1024**2)
    if estimate > max_working_bytes:
        raise MemoryError("nonlinear candidate tensors, exact states and one-expert workspace exceed byte budget")
    frozen, split_info, all_records = {}, {}, []
    record_ids, trajectory_ids = set(), set()
    for name in ("calibration", "heldout"):
        if name not in splits:
            continue
        records, pairs = _snapshot(*splits[name], expected_scope, max_records=max_records, max_pairs=max_pairs)
        if any(record.origin != expected_origin for record in records):
            raise ValueError("evaluation record origin differs from the explicitly expected source origin")
        ids, trajectories = {r.record_id for r in records}, {r.trajectory_id for r in records}
        if ids & record_ids or trajectories & trajectory_ids:
            raise ValueError("calibration and heldout records and trajectories must be disjoint")
        record_ids.update(ids)
        trajectory_ids.update(trajectories)
        offset = len(all_records)
        all_records.extend(records)
        positions = {r.record_id: offset+i for i, r in enumerate(records)}
        endpoints = torch.tensor([(positions[p.previous_id], positions[p.next_id]) for p in pairs], dtype=torch.int64).reshape(-1, 2)
        point_weights, contrast_weights = _metric_weights(records, pairs)
        frozen[name] = (records, pairs)
        split_info[name] = {"offset": offset, "end": len(all_records), "endpoints": endpoints,
            "point_weights": point_weights, "contrast_weights": contrast_weights,
            "contrast_energy": torch.zeros(len(pairs), dtype=torch.float64),
            "contrast_error": torch.zeros(len(pairs), dtype=torch.float64)}
    metadata = getattr(candidate, "metadata", None)
    if metadata is None:
        metadata = getattr(getattr(candidate, "result", None), "metadata", {})
    objective = metadata.get("objective", {}) if isinstance(metadata, dict) else {}
    if isinstance(objective, dict) and "heldout" in frozen:
        held_records, _ = frozen["heldout"]
        if (objective.get("heldout_used_for_fit") is True or
                {r.record_id for r in held_records} & set(objective.get("training_record_ids", ())) or
                {r.trajectory_id for r in held_records} & set(objective.get("training_trajectory_ids", ()))):
            raise ValueError("heldout evaluator cannot score a candidate fitted on heldout records or trajectories")
    seeds, seed_endpoints = (), torch.empty((0, 2), dtype=torch.int64)
    if raw_seeds:
        seeds, _ = _snapshot(raw_seeds, (), expected_scope, max_records=2*max_seed_probes, max_pairs=0)
        if any(r.record_id in record_ids or r.trajectory_id in trajectory_ids for r in seeds):
            raise ValueError("static seed poles must remain separate from conversation calibration and heldout trajectories")
        _validate_seed_values(native_seed_bank, seeds, max_seed_probes)
        seed_offset = len(all_records)
        seed_endpoints = torch.tensor([(seed_offset+2*i, seed_offset+2*i+1) for i in range(len(probe_ids))], dtype=torch.int64)
        all_records.extend(seeds)
    states = torch.stack([record.state for record in all_records])
    energies, errors = torch.zeros(count, dtype=torch.float64), torch.zeros(count, dtype=torch.float64)
    seed_energy, seed_error = torch.zeros(len(probe_ids), dtype=torch.float64), torch.zeros(len(probe_ids), dtype=torch.float64)
    with torch.no_grad():
        for expert in gate.expert_ids:
            original_gate = _read_expert_matrix(gate_source, gate, expert, row_chunk)
            original_up = _read_expert_matrix(up_source, up, expert, row_chunk)
            original_down = _read_expert_matrix(down_source, down, expert, row_chunk)
            proposed_gate = _read_expert_matrix(candidate, candidate_identity, expert, row_chunk)
            up_values = _project(original_up, states, batch_records)
            reference = _expert_output(original_gate, up_values, original_down, states, batch_records)
            proposed = _expert_output(proposed_gate, up_values, original_down, states, batch_records)
            _point_totals(reference, proposed, energies, errors, batch_records)
            for item in split_info.values():
                _contrast_totals(reference, proposed, item["endpoints"], item["contrast_energy"], item["contrast_error"], batch_records)
            _contrast_totals(reference, proposed, seed_endpoints, seed_energy, seed_error, batch_records)
            del original_gate, original_up, original_down, proposed_gate, up_values, reference, proposed
    if any(reader.identity != identity for reader, identity in zip((gate_source, up_source, down_source, candidate), identities)):
        raise ValueError("sampled expert source identity changed during evaluation")
    report = {"method": METHOD, "evaluation_scope": "sampled nonlinear expert outputs and within-trajectory output differences",
        "reference_formula": "down(SiLU(gate(x)) * up(x))", "candidate_intervention": "gate only; original up/down unchanged",
        "arithmetic": "dequantized original FP8/scales where declared; FP32 CPU matrix and SiLU arithmetic; FP64 metrics",
        "source_identities": {"gate": gate.to_dict(), "up": up.to_dict(), "down": down.to_dict()},
        "input_state_scope": expected_scope.to_dict(), "teacher": teacher, "record_origin": expected_origin,
        "expert_response_weighting": "all sampled experts equally; counterfactual outputs, not router-weighted",
        "difference_definition": "f(next state) minus f(previous state); not f(state difference), velocity or causal intervention",
        "fitting_performed": False, "heldout_used_for_fit": False,
        "native_dynamic_fp8_kernel_evaluated": False, "native_router_mixture_evaluated": False,
        "modified_decoder_evaluated": False, "logit_intervention_evaluated": False,
        "generation_evaluated": False, "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
        "estimated_numerical_peak_bytes": estimate, "resident_reader_and_bank_tensor_bytes": resident,
        "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead_and_readonly_file_mappings",
        "working_byte_budget": max_working_bytes, "expert_chunk": 1, "row_chunk": row_chunk,
        "batch_records": batch_records, "splits": {}, "seed_pole_response": None}
    for name, (records, pairs) in frozen.items():
        item = split_info[name]
        section = slice(item["offset"], item["end"])
        result = {"records": len(records), "pairs": len(pairs),
            "record_evidence_sha256": _fingerprint(expected_scope, records, pairs),
            "prediction_correct_counts": {"correct": sum(r.prediction_correct is True for r in records),
                "incorrect": sum(r.prediction_correct is False for r in records),
                "unknown": sum(r.prediction_correct is None for r in records)},
            "all_predictions_retained": True,
            "weighting": "trajectory-balanced provenance times stimulation; endpoint-degree-balanced differences"}
        _prefixed_metrics(result, "point", _metrics(energies[section], errors[section], item["point_weights"]))
        _prefixed_metrics(result, "contrast", _metrics(item["contrast_energy"], item["contrast_error"], item["contrast_weights"]))
        report["splits"][name] = result
    if probe_ids:
        seed_section = slice(n_records, count)
        result = {"evaluation_scope": "static actual observed seed-pole endpoints and positive-minus-negative nonlinear output contrasts",
            "probe_count": len(probe_ids), "pole_records": len(seeds), "origin": expected_origin,
            "inputs": "exact raw measured CalibrationRecord states; no reconstruction from direction/center",
            "seed_bank_evidence_sha256": native_seed_bank["evidence_sha256"],
            "raw_pole_evidence_sha256": _fingerprint(expected_scope, seeds, ()),
            "probe_ids": list(probe_ids), "pole_record_ids": native_seed_bank["pole_record_ids"],
            "seed_bank_used_to_fit_candidate": objective.get("probe_evidence_sha256", objective.get("native_probe_evidence_sha256")) == native_seed_bank["evidence_sha256"],
            "semantic_meanings_validated": False, "heldout_examples": False,
            "weighting": "equal measured seed poles and equal seed contrasts; all sampled experts equally"}
        _prefixed_metrics(result, "point", _metrics(energies[seed_section], errors[seed_section], torch.ones(len(seeds), dtype=torch.float64)))
        _prefixed_metrics(result, "contrast", _metrics(seed_energy, seed_error, torch.ones(len(probe_ids), dtype=torch.float64)))
        result["per_probe"] = [{"probe_id": probe, "source_output_contrast_energy_sq": float(seed_energy[i]),
            "output_contrast_absolute_error_sq": float(seed_error[i]),
            "output_contrast_relative_rmse": math.sqrt(float(seed_error[i] / seed_energy[i])) if seed_energy[i] > 0 else None}
            for i, probe in enumerate(probe_ids)]
        report["seed_pole_response"] = result
    return report
