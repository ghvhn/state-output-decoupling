"""CPU calibration correction of a fixed shared matrix-group factor budget.

An external, differentiable full-decoder callback supplies student logits. The
teacher observations remain scoped and frozen. Calibration KL and changes in
within-trajectory token log odds select checkpoints; held-out data are excluded.
This is a reference learned correction, not a language or runtime guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math

import torch
import torch.nn.functional as F

from . import expert_compress as base
from .trajectory_compress import InputStateScope, _count, _number, _text


@dataclass(frozen=True)
class PredictiveTrajectory:
    trajectory_id: str
    environment_id: str
    scope: InputStateScope
    record_ids: tuple[str, ...]
    token_positions: tuple[int, ...]
    teacher_logits: torch.Tensor
    observed_next_token_ids: torch.Tensor
    pairs: tuple[tuple[int, int], ...] = ()
    provenance_weight: float = 1.0
    stimulation_weight: float = 1.0
    origin: str = "measured_model_logits"


def _ids(value, name, *, nonempty=True):
    if not isinstance(value, (tuple, list)) or (nonempty and not value):
        raise ValueError(f"{name} must be an explicit bounded list or tuple")
    for item in value:
        _text(item, name)
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must be unique")
    return tuple(value)


def _validated_batch(batch, scope, *, max_positions, max_vocabulary):
    if not isinstance(batch, PredictiveTrajectory):
        raise ValueError("calibration requires PredictiveTrajectory observations")
    _text(batch.trajectory_id, "trajectory_id")
    _text(batch.environment_id, "environment_id")
    if batch.scope != scope:
        raise ValueError("predictive observation scope differs from matrix group/input site")
    ids = _ids(batch.record_ids, "record_ids")
    if (not isinstance(batch.token_positions, tuple) or len(batch.token_positions) != len(ids) or
            any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in batch.token_positions) or
            tuple(sorted(set(batch.token_positions))) != batch.token_positions):
        raise ValueError("token_positions must uniquely preserve the measured timeline")
    logits = batch.teacher_logits
    targets = batch.observed_next_token_ids
    if (not isinstance(logits, torch.Tensor) or logits.device.type != "cpu" or
            logits.dtype != torch.float32 or logits.ndim != 2 or logits.shape[0] != len(ids) or
            not 2 <= logits.shape[1] <= max_vocabulary or len(ids) > max_positions or
            not bool(torch.isfinite(logits).all())):
        raise ValueError("teacher logits must be finite scoped CPU FP32[position,vocabulary] observations")
    if not bool(torch.isfinite(F.log_softmax(logits.detach(), dim=-1)).all()):
        raise ValueError("teacher log probabilities exceed supported calibration precision")
    if (not isinstance(targets, torch.Tensor) or targets.device.type != "cpu" or
            targets.dtype != torch.int64 or tuple(targets.shape) != (len(ids),) or
            not bool(((targets >= 0) & (targets < logits.shape[1])).all())):
        raise ValueError("observed next token IDs do not match the teacher vocabulary/timeline")
    provenance = _number(batch.provenance_weight, "provenance_weight", maximum=1)
    stimulation = _number(batch.stimulation_weight, "stimulation_weight", maximum=1)
    if batch.origin not in ("measured_model_logits", "synthetic_fixture"):
        raise ValueError("predictive origin must distinguish measured logits from synthetic fixtures")
    if not isinstance(batch.pairs, tuple):
        raise ValueError("pairs must be an explicit bounded tuple")
    seen = set()
    for pair in batch.pairs:
        if (not isinstance(pair, tuple) or len(pair) != 2 or
                any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x < len(ids) for x in pair) or
                pair[0] == pair[1]):
            raise ValueError("pair endpoints must be distinct indices on this trajectory")
        key = tuple(sorted(pair))
        if key in seen:
            raise ValueError("pairs cannot duplicate or recount reversed endpoints")
        seen.add(key)
    return PredictiveTrajectory(batch.trajectory_id, batch.environment_id, scope, ids,
                                batch.token_positions, logits.detach().clone().contiguous(),
                                targets.detach().clone().contiguous(), batch.pairs,
                                provenance, stimulation, batch.origin)


def predictive_objective(student_logits, trajectory, *, top_k=16, token_weight_floor=0.1):
    """Full-vocabulary KL plus signed changes in centered union-token log odds.

    Correctness never gates validity or contribution. A small uniform floor on
    union-token weights gives observed targets weight even when the teacher was
    wrong with high confidence. Point KL still covers the complete vocabulary.
    Both teacher and student pair directions reverse together without changing
    the squared loss. Reversing only the student's timeline changes the loss.
    """
    _count(top_k, "top_k", 1)
    token_weight_floor = _number(token_weight_floor, "token_weight_floor", maximum=1)
    if not isinstance(trajectory, PredictiveTrajectory):
        raise ValueError("predictive objective requires a trajectory observation")
    teacher = trajectory.teacher_logits
    if (not isinstance(teacher, torch.Tensor) or teacher.device.type != "cpu" or
            teacher.dtype != torch.float32 or teacher.ndim != 2 or teacher.shape[0] < 1 or
            teacher.shape[1] < 2 or not bool(torch.isfinite(teacher).all())):
        raise ValueError("invalid finite CPU teacher observations")
    if (not isinstance(student_logits, torch.Tensor) or student_logits.device.type != "cpu" or
            student_logits.dtype not in (torch.float16, torch.bfloat16, torch.float32) or
            student_logits.shape != teacher.shape or not bool(torch.isfinite(student_logits).all())):
        raise ValueError("callback must return finite CPU student logits matching measured positions/vocabulary")
    student_log = F.log_softmax(student_logits.float(), dim=-1)
    teacher_log = F.log_softmax(teacher, dim=-1)
    teacher_prob = teacher_log.exp()
    point = (teacher_prob * (teacher_log - student_log)).sum(dim=-1).mean().clamp_min(0)
    contrast = student_logits.sum() * 0
    if trajectory.pairs:
        degrees = [0] * len(teacher)
        for previous, next_ in trajectory.pairs:
            degrees[previous] += 1
            degrees[next_] += 1
        total_weight = 0.0
        for previous, next_ in trajectory.pairs:
            tokens = torch.unique(torch.cat((teacher[previous].topk(min(top_k, teacher.shape[1])).indices,
                                            teacher[next_].topk(min(top_k, teacher.shape[1])).indices,
                                            trajectory.observed_next_token_ids[[previous, next_]])))
            token_weights = (teacher_prob[previous, tokens] + teacher_prob[next_, tokens]) * 0.5
            token_weights = token_weights / token_weights.sum().clamp_min(torch.finfo(torch.float32).tiny)
            token_weights = (1 - token_weight_floor) * token_weights + token_weight_floor / len(tokens)
            teacher_ends = teacher_log[[previous, next_]][:, tokens]
            student_ends = student_log[[previous, next_]][:, tokens]
            teacher_ends = teacher_ends - teacher_ends.mean(dim=-1, keepdim=True)
            student_ends = student_ends - student_ends.mean(dim=-1, keepdim=True)
            teacher_change = teacher_ends[1] - teacher_ends[0]
            student_change = student_ends[1] - student_ends[0]
            source_energy = (token_weights * teacher_change.square()).sum()
            error = (token_weights * (student_change - teacher_change).square()).sum()
            # A one-nat-squared floor keeps a zero/small observed change a valid
            # target instead of dividing by zero or assigning infinite leverage.
            pair_weight = 0.5 * (1 / degrees[previous] + 1 / degrees[next_])
            contrast = contrast + pair_weight * error / (1 + source_energy)
            total_weight += pair_weight
        contrast = contrast / total_weight
    if not bool(torch.isfinite(point) & torch.isfinite(contrast)):
        raise ValueError("predictive calibration objective overflowed")
    return {"point_kl": point, "contrast_loss": contrast,
            "positions": len(teacher), "pairs": len(trajectory.pairs),
            "teacher_argmax_matches_observed": int((teacher.argmax(-1) == trajectory.observed_next_token_ids).sum())}


def _fingerprint(batches):
    digest = hashlib.sha256()
    for batch in batches:
        fields = {"trajectory": batch.trajectory_id, "environment": batch.environment_id,
                  "scope": batch.scope.to_dict(), "records": batch.record_ids,
                  "positions": batch.token_positions, "pairs": batch.pairs,
                  "provenance": batch.provenance_weight, "stimulation": batch.stimulation_weight,
                  "origin": batch.origin}
        digest.update(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode())
        digest.update(batch.teacher_logits.numpy().tobytes())
        digest.update(batch.observed_next_token_ids.numpy().tobytes())
    return digest.hexdigest()


def _weight_diagnostics(reader, result, chunk):
    identity = result.identity
    e, p = identity.expert_count, identity.column_count
    raw = mean_energy = centered_energy = selected = error = maximum = 0.0
    with torch.no_grad():
        for start in range(0, p, chunk):
            stop = min(p, start + chunk)
            values = base._read_chunk(reader, identity, start, stop)
            average = values.mean(dim=0)
            centered = values - average
            raw += float(values.double().square().sum())
            mean_energy += e * float(average.double().square().sum())
            centered_energy += float(centered.double().square().sum())
            selected += float((result.coefficients.T @ centered).double().square().sum())
            reconstructed = result.coefficients @ result.basis[:, start:stop].float() + result.mean[start:stop].float()
            difference = reconstructed.double() - values.double()
            error += float(difference.square().sum())
            maximum = max(maximum, float(difference.abs().max()))
    if not all(math.isfinite(x) for x in (raw, mean_energy, centered_energy, selected, error, maximum)):
        raise ValueError("corrected weight reconstruction diagnostics overflowed")
    if not raw and error:
        raise ValueError("relative reconstruction error is undefined for altered zero-energy source weights")
    result.metadata["diagnostics"] = {
        "source_energy_frobenius_sq": raw, "centered_energy_frobenius_sq": centered_energy,
        "mean_energy_frobenius_sq": mean_energy, "selected_centered_energy_frobenius_sq": selected,
        "retained_centered_energy_fraction": min(1., selected / centered_energy) if centered_energy else 1.,
        "reconstruction_error_frobenius_sq": error,
        "relative_frobenius_error": math.sqrt(error / raw) if raw else 0.,
        "maximum_absolute_error": maximum, "evaluation_scope": "fitted_weight_reconstruction_only",
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
    }


def refine_predictive_factors(reader, initial_result, trajectory_batches, forward_logits, *, expected_scope,
                              heldout_trajectory_ids=(), heldout_record_ids=(), epochs=4,
                              learning_rate=0.0005, point_weight=1., contrast_weight=0.25,
                              top_k=16, token_weight_floor=0.1, chunk_columns=65536,
                              max_working_bytes=1 << 30, max_trajectories=64,
                              max_positions=2048, max_vocabulary=262144, max_pairs=4096,
                              callback_working_bytes=0):
    """Correct mean/bases at fixed rank, coefficients and exported factor bytes.

    ``forward_logits(batch, mean, basis, coefficients)`` must be a CPU function
    returning differentiable logits at the measured positions. The caller owns
    freezing/restoring the full model and bounding its external callback memory;
    ``callback_working_bytes`` is that declared budget, not a measured RSS bound.
    Only calibration trajectories are passed to it. Best checkpoints are chosen
    by the calibration objective, never a held-out callback or token accuracy.
    """
    identity = initial_result.identity
    base._validate_result(initial_result, expected_identity=identity)
    if reader.identity != identity:
        raise ValueError("correction reader does not match initial factor source")
    if (not isinstance(expected_scope, InputStateScope) or expected_scope !=
            InputStateScope.from_identity(identity, state_site=expected_scope.state_site)):
        raise ValueError("correction input state scope does not match source")
    previous_objective = initial_result.metadata.get("objective", {})
    if previous_objective.get("input_state_scope") is not None and previous_objective["input_state_scope"] != expected_scope.to_dict():
        raise ValueError("correction input site differs from initial calibration")
    if not callable(forward_logits):
        raise ValueError("a differentiable CPU full-decoder callback is required")
    for value, name in ((epochs, "epochs"), (top_k, "top_k"), (chunk_columns, "chunk_columns"),
                        (max_working_bytes, "max_working_bytes"), (max_trajectories, "max_trajectories"),
                        (max_positions, "max_positions"), (max_vocabulary, "max_vocabulary"),
                        (max_pairs, "max_pairs")):
        _count(value, name, 1)
    _count(callback_working_bytes, "callback_working_bytes")
    if epochs > 64:
        raise ValueError("epochs exceeds the bounded reference correction schedule")
    learning_rate = _number(learning_rate, "learning_rate")
    point_weight = _number(point_weight, "point_weight")
    contrast_weight = _number(contrast_weight, "contrast_weight")
    token_weight_floor = _number(token_weight_floor, "token_weight_floor", maximum=1)
    if not learning_rate or not point_weight + contrast_weight:
        raise ValueError("learning rate and total objective weight must be positive")
    if not isinstance(heldout_trajectory_ids, (tuple, list)) or not isinstance(heldout_record_ids, (tuple, list)):
        raise ValueError("heldout exclusions require explicit bounded lists or tuples")
    if len(heldout_trajectory_ids) > max_trajectories or len(heldout_record_ids) > max_positions:
        raise MemoryError("heldout exclusion ID counts exceed sample budget")
    holdout_trajectories = set(_ids(heldout_trajectory_ids, "heldout_trajectory_ids"))
    holdout_records = set(_ids(heldout_record_ids, "heldout_record_ids"))
    if not isinstance(trajectory_batches, (tuple, list)) or not trajectory_batches:
        raise ValueError("calibration batches must be a nonempty bounded list or tuple")
    if len(trajectory_batches) > max_trajectories:
        raise MemoryError("trajectory count exceeds calibration budget")
    # Estimate from shapes/counts before cloning large teacher evidence or calling
    # the decoder/weight reader. Malformed tensor shapes are rejected below.
    positions = pair_count = teacher_elements = maximum_logits = 0
    for batch in trajectory_batches:
        if not isinstance(batch, PredictiveTrajectory) or not isinstance(batch.teacher_logits, torch.Tensor) or batch.teacher_logits.ndim != 2:
            raise ValueError("invalid predictive batch/logit shape")
        positions += batch.teacher_logits.shape[0]
        teacher_elements += batch.teacher_logits.numel()
        maximum_logits = max(maximum_logits, batch.teacher_logits.numel())
        if not isinstance(batch.pairs, tuple):
            raise ValueError("pairs must be bounded tuples")
        pair_count += len(batch.pairs)
    if positions > max_positions or pair_count > max_pairs:
        raise MemoryError("combined positions or transitions exceed calibration sample budget")
    factor_elements = initial_result.mean.numel() + initial_result.basis.numel()
    chunk = min(chunk_columns, identity.column_count)
    estimated = (factor_elements * 4 * 10 + initial_result.coefficients.numel() * 4 +
                 teacher_elements * 4 * 2 + maximum_logits * 4 * 8 + positions * 16 +
                 base._memory_plan(identity, initial_result.rank, chunk,
                                   initial_result.metadata["storage_dtype"])["estimated_numerical_peak_bytes"] +
                 callback_working_bytes)
    if estimated > max_working_bytes:
        raise MemoryError("predictive evidence, factor optimization and declared callback exceed byte budget")
    batches = tuple(_validated_batch(batch, expected_scope, max_positions=max_positions,
                                     max_vocabulary=max_vocabulary) for batch in trajectory_batches)
    trajectories, record_ids = set(), set()
    for batch in batches:
        if batch.trajectory_id in trajectories or record_ids & set(batch.record_ids):
            raise ValueError("calibration trajectory and record IDs must be unique")
        if batch.trajectory_id in holdout_trajectories or set(batch.record_ids) & holdout_records:
            raise ValueError("heldout trajectories/records are excluded from correction")
        trajectories.add(batch.trajectory_id)
        record_ids.update(batch.record_ids)
    if len({batch.origin for batch in batches}) != 1 or len({batch.teacher_logits.shape[1] for batch in batches}) != 1:
        raise ValueError("calibration cannot mix evidence origins or vocabularies")
    if contrast_weight and not pair_count:
        raise ValueError("a contrast objective requires observed within-trajectory pairs")
    coefficients = initial_result.coefficients.detach().clone()
    if initial_result.rank:
        if (not torch.allclose(coefficients.T @ coefficients, torch.eye(initial_result.rank), atol=1e-4, rtol=1e-4) or
                not torch.allclose(coefficients.mean(dim=0), torch.zeros(initial_result.rank), atol=1e-4, rtol=1e-4)):
            raise ValueError("correction requires fixed centered orthonormal coefficient columns")
    quality = [batch.provenance_weight * batch.stimulation_weight for batch in batches]
    quality_mass = sum(quality)
    if not quality_mass:
        raise ValueError("calibration has no provenance/stimulation objective mass")
    mean = torch.nn.Parameter(initial_result.mean.detach().float().clone())
    basis = torch.nn.Parameter(initial_result.basis.detach().float().clone())
    optimizer = torch.optim.Adam((mean, basis), lr=learning_rate)

    def evaluate():
        total = point = contrast = 0.
        with torch.no_grad():
            for batch, weight in zip(batches, quality):
                components = predictive_objective(forward_logits(batch, mean, basis, coefficients), batch,
                                                   top_k=top_k, token_weight_floor=token_weight_floor)
                point += float(components["point_kl"]) * weight / quality_mass
                contrast += float(components["contrast_loss"]) * weight / quality_mass
        total = point_weight * point + contrast_weight * contrast
        if not math.isfinite(total):
            raise ValueError("calibration evaluation is nonfinite")
        return {"total": total, "point_kl": point, "contrast_loss": contrast}

    initial_metrics = evaluate()
    best_metrics, best_epoch = initial_metrics, 0
    best_mean, best_basis = mean.detach().clone(), basis.detach().clone()
    history = [initial_metrics]
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        for batch, weight in zip(batches, quality):
            components = predictive_objective(forward_logits(batch, mean, basis, coefficients), batch,
                                               top_k=top_k, token_weight_floor=token_weight_floor)
            loss = (point_weight * components["point_kl"] + contrast_weight * components["contrast_loss"]) * weight / quality_mass
            if not loss.requires_grad:
                raise ValueError("callback logits must retain a differentiable path to factors")
            loss.backward()
        if not any(parameter.numel() and parameter.grad is not None for parameter in (mean, basis)):
            raise ValueError("callback logits are disconnected from the nonempty factor parameters")
        if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in (mean, basis)):
            raise ValueError("factor correction gradient is nonfinite")
        optimizer.step()
        if not bool(torch.isfinite(mean).all() & torch.isfinite(basis).all()):
            raise ValueError("factor correction parameters overflowed")
        metrics = evaluate()
        history.append(metrics)
        if metrics["total"] < best_metrics["total"]:
            best_metrics, best_epoch = metrics, epoch
            best_mean, best_basis = mean.detach().clone(), basis.detach().clone()
    dtype = initial_result.mean.dtype
    corrected_mean, corrected_basis = best_mean.to(dtype).contiguous(), best_basis.to(dtype).contiguous()
    if not bool(torch.isfinite(corrected_mean).all() & torch.isfinite(corrected_basis).all()):
        raise ValueError("corrected factor storage dtype overflowed")
    with torch.no_grad():
        mean.copy_(corrected_mean.float())
        basis.copy_(corrected_basis.float())
    quantized_metrics = evaluate()
    metadata = copy.deepcopy(initial_result.metadata)
    # A fresh artifact must not inherit another file's checksum/container counts.
    for key in ("factors_sha256", "serialized_tensor_file_bytes", "metadata_json_bytes", "serialized_total_bytes",
                "serialized_storage_ratio", "serialized_is_smaller"):
        metadata.pop(key, None)
    metadata["objective"] = {**copy.deepcopy(previous_objective),
        "method": "trajectory_response_shared_basis_v1", "identity_sha256": identity.fingerprint,
        "input_state_scope": expected_scope.to_dict(), "training_record_ids": sorted(record_ids),
        "training_trajectory_ids": sorted(trajectories), "record_origin": batches[0].origin,
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False}
    metadata["objective"]["predictive_correction"] = {
        "method": "calibration_decoder_kl_and_union_token_logodds_change_v1",
        "initial_selection_method": previous_objective.get("method"),
        "calibration_sha256": _fingerprint(batches), "epochs": epochs,
        "learning_rate": learning_rate, "point_weight": point_weight, "contrast_weight": contrast_weight,
        "top_k": top_k, "token_weight_floor": token_weight_floor,
        "pair_energy_floor_nats_sq": 1., "pair_direction": "next minus previous at both teacher and student",
        "coefficient_columns_fixed": True, "best_epoch": best_epoch,
        "best_checkpoint_selection": "calibration_objective_only",
        "initial_calibration": initial_metrics, "best_float32_calibration": best_metrics,
        "stored_precision_calibration": quantized_metrics, "calibration_history": history,
        "heldout_trajectory_ids": sorted(holdout_trajectories), "heldout_record_ids": sorted(holdout_records),
        "heldout_callback_invoked": False,
        "observed_positions": positions, "observed_pairs": pair_count,
        "teacher_argmax_matches_observed": sum(int((batch.teacher_logits.argmax(-1) == batch.observed_next_token_ids).sum()) for batch in batches),
        "all_observations_valid_independent_of_prediction_accuracy": True,
        "factor_payload_bytes_unchanged": True,
        "estimated_numerical_peak_bytes": estimated, "declared_callback_working_bytes": callback_working_bytes,
        "memory_bound_kind": "conservative_module_tensor_arrays_plus_declared_external_callback_excluding_allocator_overhead",
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
    }
    result = base.ExpertCompression(corrected_mean, corrected_basis, coefficients, metadata)
    _weight_diagnostics(reader, result, chunk)
    base._validate_result(result, expected_identity=identity)
    return result
