"""A bounded static seed-contrast projection proxy for shared expert storage.

Paired native prompt states define a subspace, not a measured transition or a
validated concept. This CPU objective protects dequantized source W times that
subspace. It does not evaluate native FP8 input quantization, a nonlinear MoE
block, changed decoder activations, language quality, or runtime savings.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re

import torch

from . import expert_compress as base
from .glm_capture_archive import validate_identity
from .glm_capture_calibration import native_probe_bank_fingerprint
from .trajectory_compress import InputStateScope, _count, _number, _read_rows


METHOD = "probe_response_shared_basis_v1"
SITE = "post_attention_norm_expert_gate_up_input"
_SOURCE_DTYPES = {"native_fp8_e4m3fn": "float8_e4m3fn", "native_fp8_e5m2": "float8_e5m2",
    **{name: name for name in ("float32", "float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")}}
_ORIGINS = {"measured_model_state", "synthetic_fixture"}


def _identifier(value):
    if not isinstance(value, str) or not value or len(value) > 256 or "\x00" in value:
        raise ValueError("probe evidence requires bounded nonblank identifiers")


def _prepare(reader, bank, rank, storage_dtype, chunk_columns, max_working_bytes,
             max_axes, svd_rtol, expected_origin):
    identity = reader.identity
    if not isinstance(identity, base.ExpertGroupIdentity) or identity.projection not in ("gate_proj", "up_proj"):
        raise ValueError("probe scope requires a gate/up source group, never a down projection")
    _count(rank, "rank")
    if rank > min(identity.expert_count - 1, identity.column_count):
        raise ValueError("rank exceeds centered expert geometry")
    for value, name in ((chunk_columns, "chunk_columns"), (max_working_bytes, "max_working_bytes"),
                        (max_axes, "max_axes")):
        _count(value, name, 1)
    if max_axes > 2048:
        raise ValueError("probe axis budget exceeds 2048")
    svd_rtol = _number(svd_rtol, "svd_rtol", maximum=1)
    if not 0 < svd_rtol < 1:
        raise ValueError("SVD rank tolerance must be strictly between zero and one")
    if storage_dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("unsupported factor storage precision")
    if expected_origin not in _ORIGINS:
        raise ValueError("expected origin must distinguish measured states and synthetic fixtures")
    if not isinstance(bank, dict) or bank.get("format") != "glm-native-probe-bank-v1":
        raise ValueError("an auditable native probe bank is required")
    scope = bank.get("scope")
    if not isinstance(scope, dict) or set(scope) != {"teacher", "tensor_name", "state_site"}:
        raise ValueError("probe scope requires the exact native teacher and input site")
    teacher = validate_identity(scope["teacher"])
    if (teacher["model_id"] != identity.model_id or teacher["revision"] != identity.revision or
            teacher["hidden_size"] != identity.matrix_shape[1] or identity.layer >= teacher["num_layers"] or
            _SOURCE_DTYPES.get(teacher["source_precision"]) != identity.source_dtype or
            scope["tensor_name"] != f"layer_{identity.layer}_expert_input" or scope["state_site"] != SITE):
        raise ValueError("probe scope differs in teacher, revision, layer, width, source dtype or gate/up site")
    if bank.get("origin") != expected_origin:
        raise ValueError("probe evidence origin differs from the explicitly expected origin")
    if bank.get("semantic_meanings_validated") is not False or bank.get("heldout_used_to_fit_axes") is not False:
        raise ValueError("probe bank cannot assert validated semantics or fit axes with heldout examples")
    scope_hash = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    if bank.get("scope_sha256") != scope_hash:
        raise ValueError("probe scope checksum differs")
    directions, centers, magnitudes = (bank.get(name) for name in ("directions", "centers", "contrast_magnitudes"))
    if (not isinstance(directions, torch.Tensor) or directions.ndim != 2 or
            directions.shape[1] != identity.matrix_shape[1] or not 0 < directions.shape[0] <= max_axes):
        raise ValueError("probe axes require a bounded nonempty matrix of the exact native width")
    n, width = directions.shape
    row_chunk = min(identity.matrix_shape[0], max(1, chunk_columns // width))
    factor_plan = base._memory_plan(identity, rank, row_chunk * width, storage_dtype)
    # Reserve source axes/centers, snapshots, finite masks, thin SVD arrays and
    # Q, streamed source/decoded rows, response arrays, E^2 Gram/eigen arrays,
    # and the existing decoder fitter's factors/workspace before any read.
    span_arrays = (14 * n * width * 4 + 8 * n * n * 4 +
        10 * identity.expert_count * row_chunk * width * 4 +
        10 * identity.expert_count * row_chunk * n * 8 + 12 * identity.expert_count**2 * 8)
    estimate = span_arrays + factor_plan["estimated_numerical_peak_bytes"] + 2 * 1024**2
    if estimate > max_working_bytes:
        raise MemoryError("probe axes, SVD, factors and streamed response workspace exceed byte budget")
    for value, shape in ((directions, (n, width)), (centers, (n, width)), (magnitudes, (n,))):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.float32 or
                tuple(value.shape) != shape or not bool(torch.isfinite(value).all())):
            raise ValueError("probe evidence must contain finite FP32 CPU axes, centers and magnitudes")
    if not torch.allclose(directions.norm(dim=1), torch.ones(n), rtol=1e-4, atol=1e-5) or bool((magnitudes <= 0).any()):
        raise ValueError("probe axes must be unit directions with positive raw contrast magnitudes")
    ids, endpoints, hashes = (bank.get(name) for name in ("probe_ids", "pole_record_ids", "source_capture_sha256"))
    if not isinstance(ids, list) or len(ids) != n:
        raise ValueError("probe IDs must uniquely identify every axis")
    for probe in ids:
        _identifier(probe)
    if len(set(ids)) != n:
        raise ValueError("probe IDs must uniquely identify every axis")
    if not isinstance(endpoints, list) or len(endpoints) != n:
        raise ValueError("every probe requires its two source pole record IDs")
    all_endpoints = []
    for probe, pair in zip(ids, endpoints):
        _identifier(probe)
        if not isinstance(pair, dict) or set(pair) != {"negative", "positive"}:
            raise ValueError("probe pole record IDs require explicit negative and positive endpoints")
        for value in pair.values():
            _identifier(value)
            all_endpoints.append(value)
    if len(set(all_endpoints)) != len(all_endpoints):
        raise ValueError("probe source pole record IDs must be unique and distinct")
    if (not isinstance(hashes, list) or not 0 < len(hashes) <= 2048 or
            any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes) or
            len(set(hashes)) != len(hashes)):
        raise ValueError("probe evidence requires bounded source capture checksums")
    for name in ("unpaired_probe_ids", "zero_contrast_probe_ids"):
        values = bank.get(name)
        if not isinstance(values, list) or len(values) > max_axes:
            raise ValueError("probe exclusions require bounded identifier lists")
        for value in values:
            _identifier(value)
    frozen = {**bank, "scope": copy.deepcopy(scope), "probe_ids": ids.copy(),
        "pole_record_ids": copy.deepcopy(endpoints), "source_capture_sha256": hashes.copy(),
        "directions": directions.detach().clone().contiguous(), "centers": centers.detach().clone().contiguous(),
        "contrast_magnitudes": magnitudes.detach().clone().contiguous(),
        "unpaired_probe_ids": bank["unpaired_probe_ids"].copy(), "zero_contrast_probe_ids": bank["zero_contrast_probe_ids"].copy()}
    if frozen.get("evidence_sha256") != native_probe_bank_fingerprint(frozen):
        raise ValueError("probe evidence checksum differs")
    return identity, frozen, row_chunk, estimate, svd_rtol


def _span(bank, svd_rtol):
    directions = bank["directions"]
    pivots = directions.abs().argmax(dim=1)
    signs = directions[torch.arange(len(directions)), pivots].sign()
    # An exact repeated or sign-reversed seed must not increase the leading
    # singular value and thereby change the numerical rank cutoff.
    unique = torch.unique(directions * signs[:, None], dim=0)
    _, values, vh = torch.linalg.svd(unique, full_matrices=False)
    if not bool(torch.isfinite(values).all() & torch.isfinite(vh).all()):
        raise ValueError("probe span SVD overflowed")
    rank = int((values > values[0] * svd_rtol).sum())
    if rank < 1:
        raise ValueError("no nonzero probe span survived its rank tolerance")
    return vh[:rank].T.contiguous(), values


def _responses(weights, q):
    result = torch.matmul(weights, q)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("probe projection response overflowed")
    return result


def fit_probe_span_group(reader, rank, bank, *, probe_gain=1., weight_floor=0.,
                         storage_dtype="float32", chunk_columns=65536,
                         max_working_bytes=1 << 30, max_axes=128, svd_rtol=1e-5,
                         expected_origin="measured_model_state"):
    """Protect a static seed-contrast span; no temporal pairs are fabricated.

    The whole span has one source-response energy normalization. Rotating its
    orthonormal basis, flipping seed poles, or duplicating an axis cannot give
    it more objective mass. A zero gain or weight floor of one honestly returns
    the existing weight-PCA control. No heldout evidence is accepted for fit.
    """
    probe_gain = _number(probe_gain, "probe_gain")
    weight_floor = _number(weight_floor, "weight_floor", maximum=1)
    identity, frozen, row_chunk, estimate, svd_rtol = _prepare(reader, bank, rank, storage_dtype,
        chunk_columns, max_working_bytes, max_axes, svd_rtol, expected_origin)
    if not probe_gain or weight_floor == 1:
        return base.fit_expert_group(reader, rank, storage_dtype=storage_dtype,
            chunk_columns=row_chunk * identity.matrix_shape[1], max_working_bytes=max_working_bytes,
            expected_identity=identity)
    q, values = _span(frozen, svd_rtol)
    e = identity.expert_count
    rows, width = identity.matrix_shape
    gram = torch.zeros((e, e), dtype=torch.float64)
    weight_gram = torch.zeros_like(gram)
    response_energy = weight_energy = 0.
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            weights = _read_rows(reader, identity, start, min(rows, start + row_chunk))
            if reader.identity != identity:
                raise ValueError("weight source scope changed during the probe response read")
            response = _responses(weights, q)
            _, centered = base._center_chunk(response.flatten(1))
            gram.add_(centered.double() @ centered.double().T)
            response_energy += float(response.double().square().sum())
            if weight_floor:
                flat = weights.flatten(1)
                _, centered_weights = base._center_chunk(flat)
                weight_gram.add_(centered_weights.double() @ centered_weights.double().T)
                weight_energy += float(flat.double().square().sum())
    if (not math.isfinite(response_energy + weight_energy) or not bool(torch.isfinite(gram).all()) or
            not bool(torch.isfinite(weight_gram).all())):
        raise ValueError("probe objective energy or Gram overflowed")
    if not response_energy:
        result = base.fit_expert_group(reader, rank, storage_dtype=storage_dtype,
            chunk_columns=row_chunk * width, max_working_bytes=max_working_bytes, expected_identity=identity)
        result.metadata["probe_span_zero_response_fallback"] = {
            "probe_evidence_sha256": frozen["evidence_sha256"], "source_response_energy_sq": 0.,
            "reason": "zero source response gives no probe subspace selection signal"}
        return result
    probe_mass = (1 - weight_floor) * probe_gain
    family_mass = probe_mass + weight_floor
    gram *= (probe_mass / family_mass) / response_energy
    if weight_floor and weight_energy:
        gram.add_(weight_gram * ((weight_floor / family_mass) / weight_energy))
    gram = (gram + gram.T) * .5
    gram = gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + gram.mean()
    metadata = {"method": METHOD, "identity_sha256": identity.fingerprint,
        "input_state_scope": InputStateScope.from_identity(identity, state_site=SITE).to_dict(),
        "probe_bank_scope": frozen["scope"], "probe_scope_sha256": frozen["scope_sha256"],
        "probe_evidence_sha256": frozen["evidence_sha256"], "probe_ids": frozen["probe_ids"],
        "pole_record_ids": frozen["pole_record_ids"], "source_capture_sha256": frozen["source_capture_sha256"],
        "record_origin": frozen["origin"], "probe_axes": len(frozen["probe_ids"]), "probe_span_rank": q.shape[1],
        "svd_rtol": svd_rtol, "span_singular_values": values.tolist(),
        "svd_input_policy": "exact sign-canonical duplicates removed before thin SVD",
        "controls": {"probe_gain": probe_gain, "weight_floor": weight_floor},
        "normalization": "one source-response energy for the complete orthonormal seed-contrast span",
        "source_span_response_energy_sq": response_energy,
        "feature_kind": "derived_static_seed_contrast_span", "measured_temporal_transitions": 0,
        "heldout_used_for_fit": False, "semantic_meanings_validated": False,
        "evaluation_scope": "static_seed_contrast_linear_projection_proxy", "native_fp8_kernel_evaluated": False,
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
        "estimated_numerical_peak_bytes": estimate,
        "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"}
    return base.fit_expert_group(reader, rank, storage_dtype=storage_dtype,
        chunk_columns=row_chunk * width, max_working_bytes=max_working_bytes, expected_identity=identity,
        expert_gram=gram.float(), objective_metadata=metadata)


def evaluate_probe_span_group(result, reader, bank, *, chunk_columns=65536,
                              max_working_bytes=1 << 30, max_axes=128, svd_rtol=1e-5,
                              expected_origin="measured_model_state"):
    """Report static span/axis projection errors, including absolute zero cases.

    If this bank fitted the result, these are calibration diagnostics, never
    heldout accuracy. A different bank must still have the exact native scope.
    """
    base._validate_result(result, expected_identity=reader.identity)
    identity, frozen, row_chunk, estimate, svd_rtol = _prepare(reader, bank, result.rank,
        result.metadata["storage_dtype"], chunk_columns, max_working_bytes, max_axes, svd_rtol, expected_origin)
    objective = result.metadata["objective"]
    if objective.get("record_origin") is not None and objective["record_origin"] != frozen["origin"]:
        raise ValueError("evaluation probe origin differs from fitted evidence")
    expected_scope = InputStateScope.from_identity(identity, state_site=SITE).to_dict()
    if objective.get("input_state_scope") is not None and objective["input_state_scope"] != expected_scope:
        raise ValueError("evaluation probe site differs from fitted scope")
    if objective.get("probe_bank_scope") is not None and objective["probe_bank_scope"] != frozen["scope"]:
        raise ValueError("evaluation probe teacher formatting differs from fitted bank")
    q, _ = _span(frozen, svd_rtol)
    directions = frozen["directions"].T
    energies = torch.zeros(len(frozen["probe_ids"]), dtype=torch.float64)
    errors = torch.zeros_like(energies)
    span_energy = span_error = 0.
    rows, width = identity.matrix_shape
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            stop = min(rows, start + row_chunk)
            original = _read_rows(reader, identity, start, stop)
            if reader.identity != identity:
                raise ValueError("weight source scope changed during the probe evaluation read")
            decoded = (result.coefficients @ result.basis[:, start * width:stop * width].float() +
                       result.mean[start * width:stop * width].float()).reshape(identity.expert_count, stop - start, width)
            reference = _responses(original, q)
            proposed = _responses(decoded, q)
            span_energy += float(reference.double().square().sum())
            span_error += float((proposed.double() - reference.double()).square().sum())
            reference = _responses(original, directions)
            proposed = _responses(decoded, directions)
            energies.add_(reference.double().square().sum((0, 1)))
            errors.add_((proposed.double() - reference.double()).square().sum((0, 1)))
    if not math.isfinite(span_energy + span_error) or not bool(torch.isfinite(energies).all() & torch.isfinite(errors).all()):
        raise ValueError("probe evaluation energy overflowed")
    axes = [{"probe_id": probe, "source_energy_sq": float(energy), "absolute_error_sq": float(error),
        "relative_rmse": math.sqrt(float(error / energy)) if energy > 0 else None,
        "seed_endpoint_difference_error_sq": float(error * magnitude.double().square())}
        for probe, energy, error, magnitude in zip(frozen["probe_ids"], energies, errors, frozen["contrast_magnitudes"])]
    return {"evaluation_scope": "static_seed_contrast_linear_projection_proxy",
        "record_origin": frozen["origin"], "probe_evidence_sha256": frozen["evidence_sha256"],
        "same_probe_bank_used_to_fit": objective.get("probe_evidence_sha256") == frozen["evidence_sha256"],
        "probe_axes": len(axes), "probe_span_rank": q.shape[1],
        "span_response_source_energy_sq": span_energy, "span_response_absolute_error_sq": span_error,
        "span_response_relative_rmse": math.sqrt(span_error / span_energy) if span_energy else None,
        "zero_source_axes": int((energies == 0).sum()), "axis_responses": axes,
        "semantic_meanings_validated": False, "temporal_transition_evaluated": False,
        "native_fp8_kernel_evaluated": False, "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
        "estimated_numerical_peak_bytes": estimate,
        "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"}
