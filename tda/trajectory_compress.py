"""Scoped response/transition objectives for shared matrix-group compression.

The stored decoder is the same mean + expert/head coefficients + matrix bases
as ``expert_compress``. What changes is the coefficient subspace: it is selected
by responses to recorded input states and within-trajectory state differences,
rather than by weight magnitude alone. Records and correctness labels are
observations, never meanings or semantic truth. This is a CPU reference method;
linear response preservation does not certify nonlinear or language behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import torch

from . import expert_compress as base


METHOD = "trajectory_response_shared_basis_v1"
_READ_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
_ORIGINS = {"measured_model_state", "synthetic_fixture"}


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or "\x00" in value or len(value) > 256:
        raise ValueError(f"{name} must be a nonblank identifier of at most 256 characters")
    return value


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, *, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0 or
            (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be finite and in the permitted range")
    return float(value)


@dataclass(frozen=True)
class InputStateScope:
    model_id: str
    revision: str
    layer: int
    projection: str
    layout: str
    matrix_input_width: int
    state_site: str
    identity_sha256: str

    def __post_init__(self):
        for name in ("model_id", "revision", "projection", "layout", "state_site"):
            _text(getattr(self, name), name)
        _count(self.layer, "layer")
        _count(self.matrix_input_width, "matrix_input_width", 1)
        if (not isinstance(self.identity_sha256, str) or len(self.identity_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in self.identity_sha256)):
            raise ValueError("scope requires a SHA256 source group identity")

    @classmethod
    def from_identity(cls, identity, *, state_site):
        if not isinstance(identity, base.ExpertGroupIdentity):
            raise ValueError("scope requires an ExpertGroupIdentity")
        return cls(identity.model_id, identity.revision, identity.layer,
                   identity.projection, identity.layout, identity.matrix_shape[1],
                   state_site, identity.fingerprint)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class CalibrationRecord:
    record_id: str
    environment_id: str
    trajectory_id: str
    scope: InputStateScope
    state: torch.Tensor
    provenance_weight: float = 1.0
    stimulation_weight: float = 1.0
    prediction_correct: bool | None = None
    origin: str = "measured_model_state"


@dataclass(frozen=True)
class TrajectoryPair:
    pair_id: str
    previous_id: str
    next_id: str


def _snapshot(records, pairs, scope, *, max_records, max_pairs):
    # Bound and validate every scope/sample before asking the weight reader for
    # anything. Freeze states so a reader/caller cannot mutate fitted evidence.
    if not isinstance(records, (list, tuple)) or not isinstance(pairs, (list, tuple)):
        raise ValueError("records and pairs must be bounded lists or tuples")
    if len(records) > max_records or len(pairs) > max_pairs:
        raise MemoryError("calibration record or pair count exceeds its sample budget")
    result, by_id = [], {}
    environments = {}
    for item in records:
        if not isinstance(item, CalibrationRecord):
            raise ValueError("every input state requires a CalibrationRecord")
        for name in ("record_id", "environment_id", "trajectory_id"):
            _text(getattr(item, name), name)
        if item.record_id in by_id:
            raise ValueError("record IDs must be unique")
        if item.scope != scope:
            raise ValueError("calibration input state scope does not match the matrix group/site")
        if item.trajectory_id in environments and environments[item.trajectory_id] != item.environment_id:
            raise ValueError("one trajectory cannot span semantic environments")
        environments[item.trajectory_id] = item.environment_id
        if (not isinstance(item.state, torch.Tensor) or item.state.device.type != "cpu" or
                item.state.dtype not in _READ_DTYPES or
                tuple(item.state.shape) != (scope.matrix_input_width,) or
                not bool(torch.isfinite(item.state).all())):
            raise ValueError("record states must be finite, scoped 1D CPU floating tensors")
        provenance = _number(item.provenance_weight, "provenance_weight", maximum=1)
        stimulation = _number(item.stimulation_weight, "stimulation_weight", maximum=1)
        if item.prediction_correct is not None and not isinstance(item.prediction_correct, bool):
            raise ValueError("prediction_correct is a separate boolean observation or None")
        if item.origin not in _ORIGINS:
            raise ValueError("origin must explicitly declare measured model state or synthetic fixture")
        frozen = CalibrationRecord(item.record_id, item.environment_id, item.trajectory_id,
                                   scope, item.state.detach().float().clone().contiguous(),
                                   provenance, stimulation, item.prediction_correct, item.origin)
        result.append(frozen)
        by_id[item.record_id] = frozen
    if len({item.origin for item in result}) > 1:
        raise ValueError("measured states and synthetic fixtures cannot be pooled")
    pair_ids, endpoints = set(), set()
    for item in pairs:
        if not isinstance(item, TrajectoryPair):
            raise ValueError("every transition requires a TrajectoryPair")
        for name in ("pair_id", "previous_id", "next_id"):
            _text(getattr(item, name), name)
        if item.pair_id in pair_ids or item.previous_id == item.next_id:
            raise ValueError("pair IDs must be unique and endpoints distinct")
        if item.previous_id not in by_id or item.next_id not in by_id:
            raise ValueError("pair endpoint is missing from its own split")
        previous, next_ = by_id[item.previous_id], by_id[item.next_id]
        if (previous.environment_id != next_.environment_id or
                previous.trajectory_id != next_.trajectory_id):
            raise ValueError("transition endpoints must share environment and trajectory")
        undirected = tuple(sorted((item.previous_id, item.next_id)))
        if undirected in endpoints:
            raise ValueError("a transition cannot be duplicated or counted again in reverse")
        pair_ids.add(item.pair_id)
        endpoints.add(undirected)
    return tuple(result), tuple(pairs)


def _fingerprint(scope, records, pairs):
    digest = hashlib.sha256()
    digest.update(json.dumps(scope.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    for item in records:
        fields = {name: getattr(item, name) for name in
                  ("record_id", "environment_id", "trajectory_id", "provenance_weight",
                   "stimulation_weight", "prediction_correct", "origin")}
        digest.update(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
        digest.update(item.state.numpy().tobytes(order="C"))
    for item in pairs:
        digest.update(json.dumps([item.pair_id, item.previous_id, item.next_id],
                                 separators=(",", ":")).encode())
    return digest.hexdigest()


def _features(records, pairs):
    by_id = {item.record_id: item for item in records}
    quality = {item.record_id: item.provenance_weight * item.stimulation_weight for item in records}
    groups = {}
    for item in records:
        groups.setdefault(item.trajectory_id, []).append(item.record_id)
    degrees = {item.record_id: 0 for item in records}
    for pair in pairs:
        degrees[pair.previous_id] += 1
        degrees[pair.next_id] += 1
    pair_groups = {}
    for pair in pairs:
        pair_groups.setdefault(by_id[pair.previous_id].trajectory_id, []).append(pair)
    point_weights, pair_weights = {}, {}
    for trajectory, ids in groups.items():
        total = sum(quality[record_id] for record_id in ids)
        group_quality = total / len(ids)
        for record_id in ids:
            point_weights[record_id] = group_quality * quality[record_id] / total if total else 0.0
        group_pairs = pair_groups.get(trajectory, ())
        raw = {pair.pair_id: 0.5 * (quality[pair.previous_id] / degrees[pair.previous_id] +
                                    quality[pair.next_id] / degrees[pair.next_id]) for pair in group_pairs}
        denominator = sum(raw.values())
        for pair in group_pairs:
            pair_weights[pair.pair_id] = group_quality * raw[pair.pair_id] / denominator if denominator else 0.0
    inputs = [item.state for item in records]
    inputs.extend(by_id[pair.next_id].state - by_id[pair.previous_id].state for pair in pairs)
    weights = [point_weights[item.record_id] for item in records]
    weights.extend(pair_weights[pair.pair_id] for pair in pairs)
    return torch.stack(inputs), torch.tensor(weights, dtype=torch.float64)


def _read_rows(reader, identity, start, stop):
    if reader.identity != identity:
        raise ValueError("weight reader source scope changed during calibration")
    columns = identity.matrix_shape[1]
    result = reader.read_columns(start * columns, stop * columns, expert_ids=identity.expert_ids)
    if (not isinstance(result, torch.Tensor) or result.device.type != "cpu" or
            result.dtype not in _READ_DTYPES or
            tuple(result.shape) != (identity.expert_count, (stop - start) * columns) or
            not bool(torch.isfinite(result).all())):
        raise ValueError("weight reader returned invalid/nonfinite scoped values")
    return result.detach().float().reshape(identity.expert_count, stop - start, columns)


def _response(weights, inputs):
    value = torch.einsum("erc,nc->enr", weights, inputs)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("calibration response overflowed float32")
    return value


def _options(reader, rank, expected_scope, records, pairs, heldout_records, heldout_pairs,
             storage_dtype, chunk_columns, max_working_bytes, max_records, max_pairs):
    identity = reader.identity
    if not isinstance(identity, base.ExpertGroupIdentity):
        raise ValueError("weight reader requires an ExpertGroupIdentity")
    if (not isinstance(expected_scope, InputStateScope) or
            expected_scope != InputStateScope.from_identity(identity, state_site=expected_scope.state_site)):
        raise ValueError("expected input state scope does not match reader identity")
    _count(rank, "rank")
    if rank > min(identity.expert_count - 1, identity.column_count):
        raise ValueError("rank exceeds centered matrix-group geometry")
    for value, name in ((chunk_columns, "chunk_columns"), (max_working_bytes, "max_working_bytes"),
                        (max_records, "max_records"), (max_pairs, "max_pairs")):
        _count(value, name, 1)
    if storage_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError("unsupported factor storage precision")
    if (not isinstance(records, (list, tuple)) or not isinstance(heldout_records, (list, tuple)) or
            not isinstance(pairs, (list, tuple)) or not isinstance(heldout_pairs, (list, tuple))):
        raise ValueError("samples must be bounded lists or tuples")
    if len(records) + len(heldout_records) > max_records or len(pairs) + len(heldout_pairs) > max_pairs:
        raise MemoryError("combined train/heldout sample count exceeds budget")
    rows, columns = identity.matrix_shape
    row_chunk = min(rows, max(1, chunk_columns // columns))
    features = len(records) + len(pairs)
    total_records = len(records) + len(heldout_records)
    element_bytes = 4 if storage_dtype == "float32" else 2
    factors = (rank + 1) * identity.column_count * element_bytes + identity.expert_count * rank * 4
    # Includes frozen states, feature inputs, response and weighted-response
    # arrays, weight rows and E^2 matrices. Backend allocation is not exact RSS.
    functional = (2 * total_records * columns * 4 + features * columns * 4 +
                  6 * identity.expert_count * features * row_chunk * 8 +
                  4 * identity.expert_count * row_chunk * columns * 4 +
                  10 * identity.expert_count ** 2 * 4 + features * 8 * 8)
    projection = base._memory_plan(identity, rank, row_chunk * columns, storage_dtype)["estimated_numerical_peak_bytes"]
    estimate = max(functional + factors, projection + functional)
    if estimate > max_working_bytes:
        raise MemoryError("functional calibration and stored factors exceed working-byte budget")
    training, training_pairs = _snapshot(records, pairs, expected_scope, max_records=max_records, max_pairs=max_pairs)
    holdout, holdout_pairs = _snapshot(heldout_records, heldout_pairs, expected_scope,
                                     max_records=max_records, max_pairs=max_pairs)
    if not training:
        raise ValueError("training input states are required")
    if ({item.record_id for item in training} & {item.record_id for item in holdout} or
            {item.trajectory_id for item in training} & {item.trajectory_id for item in holdout}):
        raise ValueError("training and heldout record/trajectory IDs must be disjoint")
    if holdout and {item.origin for item in training} != {item.origin for item in holdout}:
        raise ValueError("training and heldout origins must agree")
    return identity, training, training_pairs, holdout, holdout_pairs, row_chunk, estimate


def fit_trajectory_group(reader, rank, records, pairs=(), *, expected_scope,
                         heldout_records=(), heldout_pairs=(), point_weight=1.0,
                         contrast_weight=1.0, weight_floor=0.0, storage_dtype="float32",
                         chunk_columns=65536, max_working_bytes=1 << 30,
                         max_records=512, max_pairs=1024):
    """Select a shared coefficient subspace from scoped measured responses.

    Every point and transition contributes a relative-response Gram normalized
    by that feature's source output energy. Record weights depend on provenance
    and stimulation, never prediction correctness. Points receive balanced
    trajectory mass; pair weights additionally balance endpoint graph degrees.
    Differences are taken before normalization and are kept in their actual
    input coordinate system. Reversing a pair changes sign, not this quadratic
    objective. It cannot establish a temporal or causal interpretation.
    """
    point_weight = _number(point_weight, "point_weight")
    contrast_weight = _number(contrast_weight, "contrast_weight")
    weight_floor = _number(weight_floor, "weight_floor", maximum=1)
    if not point_weight + contrast_weight or weight_floor == 1:
        raise ValueError("a functional objective is required; use baseline fitter for weight-only PCA")
    identity, training, training_pairs, holdout, holdout_pairs, row_chunk, estimate = _options(
        reader, rank, expected_scope, records, pairs, heldout_records, heldout_pairs,
        storage_dtype, chunk_columns, max_working_bytes, max_records, max_pairs)
    if contrast_weight and not training_pairs:
        raise ValueError("a contrast objective requires measured transition pairs")
    inputs, feature_weights = _features(training, training_pairs)
    n_points = len(training)
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("state differences overflowed before any weight read")
    if ((not point_weight or not float(feature_weights[:n_points].sum())) and
            (not contrast_weight or not float(feature_weights[n_points:].sum()))):
        raise ValueError("no provenance/stimulation mass supports the selected objective")
    e, rows = identity.expert_count, identity.matrix_shape[0]
    energies = torch.zeros(len(inputs), dtype=torch.float64)
    weight_gram = torch.zeros((e, e), dtype=torch.float32)
    raw_weight_energy = 0.0
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            weights = _read_rows(reader, identity, start, min(rows, start + row_chunk))
            response = _response(weights, inputs)
            energies.add_(response.double().square().sum(dim=(0, 2)))
            flat = weights.flatten(1)
            centered = flat - flat.mean(dim=0)
            weight_gram.add_(centered @ centered.T)
            raw_weight_energy += float(flat.square().sum(dtype=torch.float64))
        if (not bool(torch.isfinite(energies).all()) or not bool(torch.isfinite(weight_gram).all()) or
                not math.isfinite(raw_weight_energy)):
            raise ValueError("functional or weight energy overflowed")
        active = energies > 0
        normalized = torch.zeros_like(feature_weights)
        family_mass = 0.0
        for sl, gain in ((slice(0, n_points), point_weight), (slice(n_points, None), contrast_weight)):
            mass = float(feature_weights[sl][active[sl]].sum())
            if gain and mass:
                normalized[sl] = feature_weights[sl] * active[sl] * gain / mass
                family_mass += gain
        if not family_mass:
            raise ValueError("all selected features have zero source response energy")
        normalized /= family_mass
        scales = torch.zeros_like(energies)
        scales[active] = (normalized[active] / energies[active]).sqrt()
        if not bool(torch.isfinite(scales).all()):
            raise ValueError("relative response normalization overflowed")
        gram = torch.zeros((e, e), dtype=torch.float32)
        for start in range(0, rows, row_chunk):
            weights = _read_rows(reader, identity, start, min(rows, start + row_chunk))
            response = _response(weights, inputs)
            centered = response - response.mean(dim=0)
            scaled = (centered.double() * scales[None, :, None]).float().flatten(1)
            if not bool(torch.isfinite(scaled).all()):
                raise ValueError("normalized response overflowed")
            gram.add_(scaled @ scaled.T)
        gram *= 1 - weight_floor
        if weight_floor and raw_weight_energy:
            gram.add_(weight_gram * (weight_floor / raw_weight_energy))
        gram = (gram + gram.T) * 0.5
        # Project roundoff back onto the centered expert subspace. This is only
        # centering; the baseline fitter separately checks positive semidefiniteness.
        gram = gram - gram.mean(dim=0, keepdim=True) - gram.mean(dim=1, keepdim=True) + gram.mean()
    accuracy = [item.prediction_correct for item in training if item.prediction_correct is not None]
    metadata = {
        "method": METHOD, "identity_sha256": identity.fingerprint,
        "input_state_scope": expected_scope.to_dict(),
        "calibration_sha256": _fingerprint(expected_scope, training, training_pairs),
        "training_record_ids": [item.record_id for item in training],
        "training_trajectory_ids": sorted({item.trajectory_id for item in training}),
        "training_records": len(training), "training_pairs": len(training_pairs),
        "heldout_records": len(holdout), "heldout_pairs": len(holdout_pairs),
        "heldout_sha256": _fingerprint(expected_scope, holdout, holdout_pairs) if holdout else None,
        "heldout_used_for_fit": False,
        "record_origin": training[0].origin,
        "all_records_are_valid_observations": True,
        "prediction_accuracy_is_a_separate_observation": True,
        "prediction_observations": len(accuracy), "correct_predictions": sum(accuracy),
        "controls": {"point_weight": point_weight, "contrast_weight": contrast_weight,
                     "weight_floor": weight_floor},
        "normalization": "per-feature source response energy; balanced trajectory mass; degree-balanced endpoints",
        "pair_difference": "next state minus previous state, before response normalization",
        "pair_reversal_changes_quadratic_gram": False,
        "functional_streaming_passes": 2,
        "zero_response_features": int((~active).sum()),
        "estimated_numerical_peak_bytes": estimate,
        "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead",
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
    }
    # Held-out samples do not affect this Gram or any coefficient selection.
    return base.fit_expert_group(reader, rank, chunk_columns=row_chunk * identity.matrix_shape[1],
                                 storage_dtype=storage_dtype, max_working_bytes=max_working_bytes,
                                 expected_identity=identity, expert_gram=gram,
                                 objective_metadata=metadata)


def evaluate_trajectory_group(result, reader, records, pairs=(), *, expected_scope,
                              row_chunk=16, max_records=512, max_pairs=1024,
                              max_working_bytes=1 << 30):
    """Numerical held-out point/transition errors, with no language-quality claim."""
    identity = result.identity
    base._validate_result(result, expected_identity=identity)
    if reader.identity != identity:
        raise ValueError("evaluation reader does not match factor source scope")
    _count(row_chunk, "row_chunk", 1)
    identity, samples, transitions, _, _, actual_rows, _ = _options(
        reader, result.rank, expected_scope, records, pairs, (), (),
        result.metadata["storage_dtype"], row_chunk * identity.matrix_shape[1],
        max_working_bytes, max_records, max_pairs)
    objective = result.metadata.get("objective", {})
    if objective.get("input_state_scope") is not None and objective["input_state_scope"] != expected_scope.to_dict():
        raise ValueError("evaluation input site differs from fitted state scope")
    if objective.get("record_origin") is not None and objective["record_origin"] != samples[0].origin:
        raise ValueError("evaluation state origin differs from fitted calibration")
    if ({item.record_id for item in samples} & set(objective.get("training_record_ids", ())) or
            {item.trajectory_id for item in samples} & set(objective.get("training_trajectory_ids", ()))):
        raise ValueError("heldout evaluation leaks training records or trajectories")
    inputs, weights = _features(samples, transitions)
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError("evaluation differences overflowed")
    energies = torch.zeros(len(inputs), dtype=torch.float64)
    errors = torch.zeros_like(energies)
    columns, rows = identity.matrix_shape[1], identity.matrix_shape[0]
    with torch.no_grad():
        for start in range(0, rows, actual_rows):
            stop = min(rows, start + actual_rows)
            reference = _response(_read_rows(reader, identity, start, stop), inputs)
            lo, hi = start * columns, stop * columns
            decoded = result.coefficients @ result.basis[:, lo:hi].float() + result.mean[lo:hi].float()
            proposed = _response(decoded.reshape(identity.expert_count, stop - start, columns), inputs)
            energies.add_(reference.double().square().sum(dim=(0, 2)))
            errors.add_((proposed.double() - reference.double()).square().sum(dim=(0, 2)))
    report = {"evaluation_scope": "heldout_scoped_linear_responses_and_state_differences",
              "record_origin": samples[0].origin, "records": len(samples), "pairs": len(transitions),
              "semantic_quality_evaluated": False, "runtime_savings_evaluated": False}
    for name, sl in (("point", slice(0, len(samples))), ("contrast", slice(len(samples), None))):
        source, error, mass = energies[sl], errors[sl], weights[sl]
        valid = source > 0
        denominator = float(mass[valid].sum())
        relative = float((mass[valid] * error[valid] / source[valid]).sum()) / denominator if denominator else None
        report[name + "_relative_rmse"] = math.sqrt(relative) if relative is not None else None
        report[name + "_zero_source_features"] = int((~valid).sum())
        report[name + "_zero_source_absolute_error_sq"] = float(error[~valid].sum())
    return report
