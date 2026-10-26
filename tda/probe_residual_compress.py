"""INT4 plus a per-expert repair on an auditable native seed-contrast span.

The decoded matrix is Wq + C Q.T, where C = (W-Wq) Q. Independent experts
remain independent. These CPU dequantized matrix responses are projection
proxies, not measured temporal transitions, native FP8 kernel interventions,
validated concepts, compressed decoder activations, or language evaluations.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import torch

from . import block_quant as quant
from . import expert_compress as base
from .glm_capture_archive import validate_identity
from .glm_capture_calibration import native_probe_bank_fingerprint
from .probe_span_compress import _prepare, _span, _responses, _SOURCE_DTYPES, SITE
from .trajectory_compress import InputStateScope, CalibrationRecord, _count, _number, _text, _read_rows, _snapshot, _fingerprint


FORMAT = "probe-span-int4-residual-v1"
METHOD = "independent_expert_int4_native_probe_span_residual_v1"
SCOPED_METHOD = "independent_expert_int4_scoped_span_residual_v1"
_COEFFICIENT_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
_BASIS_KINDS = {"calibration_state_second_moment_span": "derived_calibration_states",
    "reproducible_random_span": "synthetic_random_control", "explicit_scoped_basis": "caller_supplied_orthonormal_basis",
    "probe_pca_complement_span": "derived_native_probe_span_and_calibration_complement"}


def residual_basis_fingerprint(value):
    """Fingerprint a declared derived/control basis; no measured-seed claim."""
    fields = {key: item for key, item in value.items() if key not in ("basis", "evidence_sha256")}
    encoded = base._metadata_bytes(fields)
    digest = hashlib.sha256(encoded)
    tensor = value["basis"].detach().contiguous()
    digest.update(json.dumps([list(tensor.shape), str(tensor.dtype)]).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _reference_scope(reference_bank, expected_origin):
    if not isinstance(reference_bank, dict) or reference_bank.get("origin") != expected_origin or expected_origin not in (
            "measured_model_state", "synthetic_fixture"):
        raise ValueError("basis reference origin differs from the explicitly expected origin")
    scope = reference_bank.get("scope")
    if not isinstance(scope, dict) or set(scope) != {"teacher", "tensor_name", "state_site"}:
        raise ValueError("basis requires the exact native reference teacher and input site")
    teacher = validate_identity(scope["teacher"])
    name = scope["tensor_name"]
    if not isinstance(name, str) or not name.startswith("layer_") or not name.endswith("_expert_input"):
        raise ValueError("basis reference must identify a native expert input layer")
    try:
        layer = int(name.removeprefix("layer_").removesuffix("_expert_input"))
    except ValueError as exc:
        raise ValueError("invalid basis reference layer") from exc
    if not 0 <= layer < teacher["num_layers"] or name != f"layer_{layer}_expert_input" or scope["state_site"] != SITE:
        raise ValueError("basis reference requires the exact normalized gate/up site")
    digest = hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()
    if reference_bank.get("scope_sha256") != digest:
        raise ValueError("basis reference scope checksum differs")
    return copy.deepcopy(scope), layer


def _basis_record(scope, origin, kind, q, requested_rank, evidence):
    value = {"format": "glm-native-residual-basis-v1", "scope": scope,
        "scope_sha256": hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest(),
        "origin": origin, "basis_kind": kind, "feature_origin": _BASIS_KINDS[kind],
        "requested_rank": requested_rank, "actual_rank": q.shape[1],
        "rank_matches_request": q.shape[1] == requested_rank, "basis": q.detach().contiguous(),
        "heldout_used_for_fit": False, "semantic_meanings_validated": False, **evidence}
    value["evidence_sha256"] = residual_basis_fingerprint(value)
    return value


def make_random_basis(reference_bank, *, requested_rank, seed=0, max_working_bytes=1 << 30,
                      expected_origin="measured_model_state"):
    """A reproducible Gaussian/QR control, scoped but never measured probe data."""
    _count(requested_rank, "requested_rank", 1)
    _count(seed, "seed")
    _count(max_working_bytes, "max_working_bytes", 1)
    scope, _ = _reference_scope(reference_bank, expected_origin)
    width = scope["teacher"]["hidden_size"]
    if requested_rank > min(width, 2048) or seed > 2**63 - 1:
        raise ValueError("random control rank or seed exceeds its bound")
    estimate = 8 * width * requested_rank * 4 + 8 * requested_rank**2 * 4 + 2 * 1024**2
    if estimate > max_working_bytes:
        raise MemoryError("random basis and QR workspace exceed byte budget")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(width, requested_rank, generator=generator), mode="reduced")
    q *= torch.diagonal(r).sign()[None, :]
    return _basis_record(scope, expected_origin, "reproducible_random_span", q, requested_rank,
        {"seed": seed, "random_algorithm": "CPU torch.Generator Gaussian then reduced QR",
         "torch_version": str(torch.__version__), "training_record_ids": [], "training_trajectory_ids": [],
         "estimated_numerical_peak_bytes": estimate,
         "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"})


def make_calibration_pca_basis(records, reference_bank, *, expected_scope, requested_rank,
                               heldout_records=(), max_records=512, max_working_bytes=1 << 30,
                               svd_rtol=1e-5, expected_origin="measured_model_state"):
    """Thin SVD of calibration X; uncentered, unweighted second-moment control."""
    _count(requested_rank, "requested_rank", 1)
    _count(max_records, "max_records", 1)
    _count(max_working_bytes, "max_working_bytes", 1)
    svd_rtol = _number(svd_rtol, "svd_rtol", maximum=1)
    scope, layer = _reference_scope(reference_bank, expected_origin)
    teacher = scope["teacher"]
    if (not isinstance(expected_scope, InputStateScope) or expected_scope.model_id != teacher["model_id"] or
            expected_scope.revision != teacher["revision"] or expected_scope.layer != layer or
            expected_scope.matrix_input_width != teacher["hidden_size"] or expected_scope.state_site != SITE or
            expected_scope.projection not in ("gate_proj", "up_proj")):
        raise ValueError("PCA state scope differs from the exact native gate/up measurement")
    if not 0 < svd_rtol < 1 or requested_rank > min(teacher["hidden_size"], 2048):
        raise ValueError("invalid PCA rank or numerical tolerance")
    if not isinstance(records, (list, tuple)) or not 0 < len(records) <= max_records:
        raise ValueError("PCA requires a bounded nonempty calibration record sequence")
    if not isinstance(heldout_records, (list, tuple)) or len(heldout_records) > max_records:
        raise ValueError("PCA heldout ID exclusions exceed the declared count bound")
    n, width = len(records), teacher["hidden_size"]
    small = min(n, width)
    estimate = 14 * n * width * 4 + 8 * small**2 * 4 + 2 * 1024**2
    if estimate > max_working_bytes:
        raise MemoryError("calibration states and thin SVD workspace exceed byte budget")
    excluded_records, excluded_trajectories = set(), set()
    for record in heldout_records:
        if not isinstance(record, CalibrationRecord) or record.scope != expected_scope:
            raise ValueError("heldout exclusions require the same scoped record identity")
        _text(record.record_id, "heldout record ID")
        _text(record.trajectory_id, "heldout trajectory ID")
        excluded_records.add(record.record_id)
        excluded_trajectories.add(record.trajectory_id)
    for record in records:
        if not isinstance(record, CalibrationRecord) or record.record_id in excluded_records or record.trajectory_id in excluded_trajectories:
            raise ValueError("heldout records and trajectories cannot enter calibration PCA")
    frozen, _ = _snapshot(records, (), expected_scope, max_records=max_records, max_pairs=0)
    if any(record.origin != expected_origin for record in frozen):
        raise ValueError("PCA calibration origin differs from the expected evidence origin")
    states = torch.stack([record.state for record in frozen])
    _, values, vh = torch.linalg.svd(states, full_matrices=False)
    if not bool(torch.isfinite(values).all() & torch.isfinite(vh).all()):
        raise ValueError("PCA state SVD overflowed")
    available_rank = int((values > values[0] * svd_rtol).sum()) if values[0] > 0 else 0
    rank = min(requested_rank, available_rank)
    if not rank:
        raise ValueError("calibration states have no nonzero numerical second-moment span")
    return _basis_record(scope, expected_origin, "calibration_state_second_moment_span", vh[:rank].T,
        requested_rank, {"available_numerical_rank": available_rank, "svd_rtol": svd_rtol,
            "input_state_scope": expected_scope.to_dict(), "source_evidence_sha256": _fingerprint(expected_scope, frozen, ()),
            "training_record_ids": [record.record_id for record in frozen],
            "training_trajectory_ids": sorted({record.trajectory_id for record in frozen}),
            "excluded_heldout_record_ids": sorted(excluded_records),
            "excluded_heldout_trajectory_ids": sorted(excluded_trajectories),
            "pca_objective": "uncentered unweighted calibration input second moment X.T @ X",
            "estimated_numerical_peak_bytes": estimate,
            "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"})


def make_probe_pca_complement_basis(records, native_bank, *, expected_scope, extra_rank,
                                    heldout_records=(), max_records=512, max_working_bytes=1 << 30,
                                    max_axes=128, svd_rtol=1e-5, expected_origin="measured_model_state"):
    """Keep native probe Q, then fit calibration PCA only in its complement.

    No dense H-by-H projector is formed. Extra directions are obtained from
    X - (X Q) Q.T, with re-projection to suppress FP32 leakage. Rank selection
    includes a source-relative rounding floor, so tiny projection residue of
    states already inside Q cannot be mistaken for real complement evidence.
    Insufficient extra spectrum raises rather than padding with random axes.
    """
    for value, name in ((extra_rank, "extra_rank"), (max_records, "max_records"),
                        (max_working_bytes, "max_working_bytes"), (max_axes, "max_axes")):
        _count(value, name, 1)
    if max_axes > 2048:
        raise ValueError("hybrid native axis budget exceeds 2048")
    svd_rtol = _number(svd_rtol, "svd_rtol", maximum=1)
    if not 0 < svd_rtol < 1:
        raise ValueError("hybrid SVD rank tolerance must lie strictly between zero and one")
    scope, layer = _reference_scope(native_bank, expected_origin)
    teacher = scope["teacher"]
    width = teacher["hidden_size"]
    if (not isinstance(expected_scope, InputStateScope) or expected_scope.model_id != teacher["model_id"] or
            expected_scope.revision != teacher["revision"] or expected_scope.layer != layer or
            expected_scope.matrix_input_width != width or expected_scope.state_site != SITE or
            expected_scope.projection not in ("gate_proj", "up_proj")):
        raise ValueError("hybrid calibration scope differs from the native gate/up probe measurement")
    if native_bank.get("format") != "glm-native-probe-bank-v1" or native_bank.get("semantic_meanings_validated") is not False or native_bank.get("heldout_used_to_fit_axes") is not False:
        raise ValueError("hybrid protected axes require unqualified calibration-only native seed evidence")
    if not isinstance(records, (list, tuple)) or not 0 < len(records) <= max_records:
        raise ValueError("hybrid requires a bounded nonempty calibration record sequence")
    if not isinstance(heldout_records, (list, tuple)) or len(heldout_records) > max_records:
        raise ValueError("hybrid heldout exclusions exceed their count bound")
    directions = native_bank.get("directions")
    if not isinstance(directions, torch.Tensor) or directions.ndim != 2 or directions.shape[1] != width or not 0 < directions.shape[0] <= max_axes:
        raise ValueError("hybrid native probe directions require bounded exact-width axes")
    axes, n = directions.shape[0], len(records)
    small = min(n, width)
    # Account for native snapshots/thin SVD, calibration snapshots/projected
    # states/thin SVD, concatenation, QR and containment checks, before these
    # allocations. Backend/allocator internals are not an exact RSS promise.
    upper_total = min(axes, width) + extra_rank
    estimate = (14 * n * width * 4 + 8 * small**2 * 4 + 14 * axes * width * 4 +
                8 * axes**2 * 4 + 12 * width * upper_total * 4 + 12 * upper_total**2 * 4 + 2 * 1024**2)
    if estimate > max_working_bytes:
        raise MemoryError("protected probes, calibration complement/SVD and combined basis exceed byte budget")
    if extra_rank > min(width, 2048):
        raise ValueError("hybrid extra rank exceeds native input width or codec bound")
    for name, shape in (("directions", (axes, width)), ("centers", (axes, width)), ("contrast_magnitudes", (axes,))):
        tensor = native_bank.get(name)
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != torch.float32 or
                tuple(tensor.shape) != shape or not base._finite_tensor(tensor.contiguous())):
            raise ValueError("hybrid native seed geometry requires finite exact-shape FP32 CPU tensors")
    if not torch.allclose(directions.norm(dim=1), torch.ones(axes), rtol=1e-4, atol=1e-5) or bool((native_bank["contrast_magnitudes"] <= 0).any()):
        raise ValueError("hybrid native seed directions must be unit length with positive pole contrasts")
    _validate_probe_evidence(native_bank, axes)
    for name in ("unpaired_probe_ids", "zero_contrast_probe_ids"):
        values = native_bank.get(name)
        if not isinstance(values, list) or len(values) > max_axes:
            raise ValueError("hybrid native seed exclusion IDs exceed their bound")
        for value in values:
            _text(value, "native seed exclusion ID")
    bank_fields = ("format", "scope", "scope_sha256", "origin", "probe_ids", "pole_record_ids",
        "source_capture_sha256", "unpaired_probe_ids", "zero_contrast_probe_ids",
        "semantic_meanings_validated", "heldout_used_to_fit_axes", "evidence_sha256")
    frozen_bank = {**copy.deepcopy({key: native_bank[key] for key in bank_fields}),
        **{name: native_bank[name].detach().clone().contiguous() for name in ("directions", "centers", "contrast_magnitudes")}}
    if frozen_bank.get("evidence_sha256") != native_probe_bank_fingerprint(frozen_bank):
        raise ValueError("hybrid native seed evidence checksum differs")
    excluded_records, excluded_trajectories = set(), set()
    for record in heldout_records:
        if not isinstance(record, CalibrationRecord) or record.scope != expected_scope:
            raise ValueError("hybrid heldout exclusions require the same scoped record identity")
        excluded_records.add(_text(record.record_id, "heldout record ID"))
        excluded_trajectories.add(_text(record.trajectory_id, "heldout trajectory ID"))
    for record in records:
        if not isinstance(record, CalibrationRecord) or record.record_id in excluded_records or record.trajectory_id in excluded_trajectories:
            raise ValueError("heldout records and trajectories cannot enter hybrid complement PCA")
    frozen, _ = _snapshot(records, (), expected_scope, max_records=max_records, max_pairs=0)
    if any(record.origin != expected_origin for record in frozen):
        raise ValueError("hybrid calibration origin differs from its protected native evidence")
    q, _ = _span(frozen_bank, svd_rtol)
    protected_rank = q.shape[1]
    total = protected_rank + extra_rank
    if total > min(width, 2048):
        raise ValueError("hybrid protected rank plus extra rank exceeds the available native input dimension")
    states = torch.stack([record.state for record in frozen])
    projected = states - (states @ q) @ q.T
    projected -= (projected @ q) @ q.T
    _, values, vh = torch.linalg.svd(projected, full_matrices=False)
    if not bool(torch.isfinite(values).all() & torch.isfinite(vh).all()):
        raise ValueError("hybrid complement SVD overflowed")
    source_norm = float(states.double().square().sum().sqrt())
    rounding_floor = 32 * torch.finfo(torch.float32).eps * source_norm
    cutoff = max(float(values[0]) * svd_rtol, rounding_floor)
    available = int((values > cutoff).sum())
    if available < extra_rank:
        raise ValueError(f"hybrid complement has numerical rank {available}, requires {extra_rank}; no padding was performed")
    extra = vh[:extra_rank].T.contiguous()
    extra -= q @ (q.T @ extra)
    extra -= q @ (q.T @ extra)
    extra, r = torch.linalg.qr(extra, mode="reduced")
    if bool((torch.diagonal(r).abs() < 1e-4).any()):
        raise ValueError("hybrid complement lost independent directions during numerical orthogonalization")
    extra *= torch.diagonal(r).sign()[None, :]
    combined = torch.cat((q, extra), dim=1).contiguous()
    if not torch.allclose(combined.T @ combined, torch.eye(total), atol=2e-4, rtol=2e-4):
        raise ValueError("hybrid protected and complement directions are not numerically orthonormal")
    containment_error = float((combined @ (combined.T @ q) - q).abs().max())
    return _basis_record(scope, expected_origin, "probe_pca_complement_span", combined, total,
        {"input_state_scope": expected_scope.to_dict(), "protected_probe_rank": protected_rank,
            "extra_rank": extra_rank, "available_complement_rank": available,
            "native_probe_evidence_sha256": frozen_bank["evidence_sha256"],
            "native_probe_scope_sha256": frozen_bank["scope_sha256"],
            "contains_protected_native_probe_span": True,
            "protected_basis_sha256": hashlib.sha256(q.numpy().tobytes()).hexdigest(),
            "protected_span_containment_max_error": containment_error,
            "source_evidence_sha256": _fingerprint(expected_scope, frozen, ()),
            "training_record_ids": [record.record_id for record in frozen],
            "training_trajectory_ids": sorted({record.trajectory_id for record in frozen}),
            "excluded_heldout_record_ids": sorted(excluded_records),
            "excluded_heldout_trajectory_ids": sorted(excluded_trajectories),
            "pca_objective": "uncentered unweighted calibration second moment in the orthogonal complement of mandatory native probe span",
            "svd_rtol": svd_rtol, "complement_singular_values": values.tolist(),
            "complement_rank_cutoff": cutoff, "source_relative_rounding_floor": rounding_floor,
            "protected_prefix_policy": "native orthonormal Q columns retained unchanged; only extra columns fitted",
            "estimated_numerical_peak_bytes": estimate,
            "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"})


def _read_plan(identity, span_rank, count, block_size, experts):
    # Includes the INT4 decoder's indexed/unpacked/signed/restored buffers,
    # padded block edges, output and one row of FP32 correction arithmetic.
    return 24 * experts * (count + 2 * block_size) + 8 * experts * span_rank + 8 * experts * min(count, identity.matrix_shape[1])


@dataclass
class ProbeResidualCompression:
    identity: base.ExpertGroupIdentity
    base_quantization: quant.BlockQuantization
    span_basis: torch.Tensor
    coefficients: torch.Tensor
    metadata: dict
    max_read_bytes: int = 128 << 20

    def read_columns(self, start, stop, expert_ids=None):
        """Decode only a bounded requested flattened interval, preserving order."""
        _count(start, "start")
        _count(stop, "stop", 1)
        _count(self.max_read_bytes, "max_read_bytes", 1)
        if not start < stop <= self.identity.column_count:
            raise ValueError("invalid residual reader column interval")
        ids = self.identity.expert_ids if expert_ids is None else tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(value not in self.identity.expert_ids for value in ids):
            raise ValueError("unknown or repeated residual expert ID")
        if self.base_quantization.identity != self.identity:
            raise ValueError("residual base source scope changed")
        rank = self.span_basis.shape[1]
        needed = _read_plan(self.identity, rank, stop - start, self.base_quantization.block_size, len(ids))
        if needed > self.max_read_bytes:
            raise MemoryError("requested residual decode exceeds its working-byte budget")
        result = self.base_quantization.read_columns(start, stop, ids)
        if self.base_quantization.identity != self.identity:
            raise ValueError("residual base source scope changed during decode")
        positions = [self.identity.expert_ids.index(value) for value in ids]
        width = self.identity.matrix_shape[1]
        cursor = start
        while cursor < stop:
            row, column = divmod(cursor, width)
            end = min(stop, (row + 1) * width)
            count = end - cursor
            correction = self.coefficients[positions, row, :].float() @ self.span_basis[column:column + count].T
            result[:, cursor - start:end - start].add_(correction)
            cursor = end
        if not bool(torch.isfinite(result).all()):
            raise ValueError("residual decode overflowed")
        return result


def _options(reader, bank, coefficient_dtype, block_size, chunk_blocks, chunk_columns,
             max_working_bytes, max_axes, svd_rtol, expected_origin, max_read_bytes):
    if not isinstance(coefficient_dtype, str) or coefficient_dtype not in _COEFFICIENT_DTYPES:
        raise ValueError("residual coefficients require explicit bfloat16 or float32 storage")
    for value, name in ((block_size, "block_size"), (chunk_blocks, "chunk_blocks"), (max_read_bytes, "max_read_bytes")):
        _count(value, name, 1)
    if block_size % 2:
        raise ValueError("INT4 block size must be even")
    identity, frozen, row_chunk, span_estimate, svd_rtol = _prepare(reader, bank, 0, "float32",
        chunk_columns, max_working_bytes, max_axes, svd_rtol, expected_origin)
    e, (rows, width) = identity.expert_count, identity.matrix_shape
    upper_rank = min(len(frozen["probe_ids"]), width)
    blocks = (identity.column_count + block_size - 1) // block_size
    base_bytes = e * blocks * (block_size // 2 + 4)
    coefficient_bytes = e * rows * upper_rank * (2 if coefficient_dtype == "bfloat16" else 4)
    q_bytes = width * upper_rank * 4
    quant_workspace = e * min(blocks, chunk_blocks) * block_size * 48
    decode_workspace = _read_plan(identity, upper_rank, row_chunk * width, block_size, e)
    estimate = span_estimate + base_bytes + coefficient_bytes + q_bytes + quant_workspace + decode_workspace
    if estimate > max_working_bytes:
        raise MemoryError("native axes/SVD, INT4 base, residual factors and decode workspace exceed byte budget")
    if decode_workspace > max_read_bytes:
        raise MemoryError("residual row chunk exceeds the declared reader working-byte budget")
    return identity, frozen, row_chunk, estimate, svd_rtol


def _metadata_scope(metadata, expected_identity):
    if not isinstance(metadata, dict):
        raise ValueError("residual metadata must be an object")
    identity = base.ExpertGroupIdentity.from_dict(metadata.get("identity"))
    if (identity != expected_identity or metadata.get("identity_sha256") != identity.fingerprint or
            metadata.get("format") != FORMAT or metadata.get("method") not in (METHOD, SCOPED_METHOD)):
        raise ValueError("residual source scope or method differs")
    if identity.projection not in ("gate_proj", "up_proj"):
        raise ValueError("residual correction requires the native gate/up input site")
    rank = _count(metadata.get("span_rank"), "span_rank", 1)
    if rank > identity.matrix_shape[1] or rank > 2048:
        raise ValueError("invalid residual span rank")
    dtype = metadata.get("coefficient_dtype")
    if not isinstance(dtype, str) or dtype not in _COEFFICIENT_DTYPES or metadata.get("span_basis_dtype") != "float32":
        raise ValueError("invalid residual factor precision")
    block_size = _count(metadata.get("block_size"), "block_size", 1)
    if block_size % 2 or metadata.get("bits") != 4:
        raise ValueError("residual base must use the INT4 control")
    blocks = (identity.column_count + block_size - 1) // block_size
    base_bytes = identity.expert_count * blocks * (block_size // 2 + 4)
    coefficient_bytes = identity.expert_count * identity.matrix_shape[0] * rank * (2 if dtype == "bfloat16" else 4)
    correction_bytes = coefficient_bytes + identity.matrix_shape[1] * rank * 4
    total = base_bytes + correction_bytes
    if (metadata.get("base_tensor_bytes") != base_bytes or metadata.get("correction_tensor_bytes") != correction_bytes or
            metadata.get("factor_tensor_bytes") != total or metadata.get("source_tensor_bytes") != identity.actual_source_bytes or
            isinstance(metadata.get("tensor_storage_ratio"), bool) or metadata.get("tensor_storage_ratio") != total / identity.actual_source_bytes or
            metadata.get("tensor_storage_is_smaller") is not (total < identity.actual_source_bytes)):
        raise ValueError("invalid residual payload byte accounting")
    objective = metadata.get("objective")
    expected_scope = InputStateScope.from_identity(identity, state_site=SITE).to_dict()
    if (not isinstance(objective, dict) or objective.get("input_state_scope") != expected_scope or
            objective.get("identity_sha256") != identity.fingerprint or objective.get("record_origin") not in
            ("measured_model_state", "synthetic_fixture") or objective.get("heldout_used_for_fit") is not False or
            objective.get("measured_temporal_transitions") != 0):
        raise ValueError("invalid residual evidence scope or temporal claim")
    kind = objective.get("basis_kind", "native_probe_seed_span")
    native = kind == "native_probe_seed_span"
    if (native and metadata["method"] != METHOD or not native and
            (metadata["method"] != SCOPED_METHOD or not isinstance(kind, str) or kind not in _BASIS_KINDS or
             objective.get("feature_origin") != _BASIS_KINDS[kind])):
        raise ValueError("residual basis kind, feature origin or method differs")
    names = ("probe_evidence_sha256", "probe_scope_sha256") if native else ("basis_evidence_sha256", "basis_scope_sha256")
    for name in names:
        value = objective.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("residual evidence requires source checksums")
    bank_scope = objective.get("probe_bank_scope") if native else objective.get("basis_scope")
    if not isinstance(bank_scope, dict) or set(bank_scope) != {"teacher", "tensor_name", "state_site"}:
        raise ValueError("residual bank requires the exact native teacher and site")
    teacher = validate_identity(bank_scope["teacher"])
    if (teacher["model_id"] != identity.model_id or teacher["revision"] != identity.revision or
            teacher["hidden_size"] != identity.matrix_shape[1] or identity.layer >= teacher["num_layers"] or
            _SOURCE_DTYPES.get(teacher["source_precision"]) != identity.source_dtype or
            bank_scope["tensor_name"] != f"layer_{identity.layer}_expert_input" or bank_scope["state_site"] != SITE or
            hashlib.sha256(json.dumps(bank_scope, sort_keys=True).encode()).hexdigest() != objective[names[1]]):
        raise ValueError("residual bank teacher, precision, site or scope checksum differs")
    if native:
        _validate_probe_evidence(objective, rank)
    else:
        _validate_control_evidence(objective, rank)
    for name in ("semantic_quality_evaluated", "runtime_savings_evaluated", "native_fp8_kernel_evaluated", "semantic_meanings_validated"):
        if metadata.get(name) is not False:
            raise ValueError("unsupported residual semantics, kernel, quality or runtime claim")
    wanted_scope = "static_seed_contrast_linear_projection_proxy" if native else "static_scoped_input_span_linear_projection_proxy"
    if metadata.get("evaluation_scope") != wanted_scope:
        raise ValueError("invalid residual evaluation scope")
    return identity, rank, dtype, base_bytes, correction_bytes, total


def _validate_control_evidence(objective, rank):
    if any(name in objective for name in ("probe_ids", "pole_record_ids", "probe_evidence_sha256")):
        raise ValueError("random/PCA residual controls cannot claim measured probe seed evidence")
    requested = _count(objective.get("requested_rank"), "requested_rank", 1)
    if requested > 2048 or objective.get("actual_rank") != rank or objective.get("rank_matches_request") is not (requested == rank):
        raise ValueError("invalid residual control rank comparison")
    for name in ("training_record_ids", "training_trajectory_ids", "excluded_heldout_record_ids", "excluded_heldout_trajectory_ids"):
        values = objective.get(name, [])
        if (not isinstance(values, list) or len(values) > 8192 or
                any(not isinstance(value, str) or not value or len(value) > 256 or "\0" in value for value in values) or len(set(values)) != len(values)):
            raise ValueError("residual control calibration/exclusion IDs must be bounded and unique")
    if (set(objective.get("training_record_ids", ())) & set(objective.get("excluded_heldout_record_ids", ())) or
            set(objective.get("training_trajectory_ids", ())) & set(objective.get("excluded_heldout_trajectory_ids", ()))):
        raise ValueError("heldout IDs cannot enter the fitted residual control basis")
    if objective["basis_kind"] == "reproducible_random_span":
        if objective.get("training_record_ids") or objective.get("training_trajectory_ids"):
            raise ValueError("random residual basis cannot claim measured calibration rows")
        seed = _count(objective.get("seed"), "seed")
        if seed > 2**63 - 1:
            raise ValueError("random residual seed exceeds its bound")
    if objective["basis_kind"] in ("calibration_state_second_moment_span", "probe_pca_complement_span"):
        if not objective.get("training_record_ids") or not objective.get("training_trajectory_ids"):
            raise ValueError("PCA residual controls require explicit calibration row evidence")
        wanted = ("uncentered unweighted calibration input second moment X.T @ X" if
            objective["basis_kind"] == "calibration_state_second_moment_span" else
            "uncentered unweighted calibration second moment in the orthogonal complement of mandatory native probe span")
        if objective.get("pca_objective") != wanted:
            raise ValueError("PCA residual objective must describe its actual uncentered construction")
    if objective["basis_kind"] == "probe_pca_complement_span":
        protected = _count(objective.get("protected_probe_rank"), "protected_probe_rank", 1)
        extra = _count(objective.get("extra_rank"), "extra_rank", 1)
        available = _count(objective.get("available_complement_rank"), "available_complement_rank", 1)
        width = objective["input_state_scope"]["matrix_input_width"]
        if (protected + extra != rank or not extra <= available <= min(width - protected, len(objective["training_record_ids"])) or
                objective.get("contains_protected_native_probe_span") is not True):
            raise ValueError("hybrid rank and mandatory native protection declarations differ")
        for name in ("native_probe_evidence_sha256", "native_probe_scope_sha256", "source_evidence_sha256", "protected_basis_sha256"):
            value = objective.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("hybrid evidence requires native-probe and calibration source checksums")
        if objective["native_probe_scope_sha256"] != objective["basis_scope_sha256"]:
            raise ValueError("hybrid protected native and calibration source scopes differ")
        error = _number(objective.get("protected_span_containment_max_error"), "protected_span_containment_max_error")
        if error > 2e-4:
            raise ValueError("hybrid protected span containment exceeds FP32 tolerance")


def _validate_probe_evidence(objective, rank):
    probe_ids, endpoints, capture_hashes = (objective.get(name) for name in ("probe_ids", "pole_record_ids", "source_capture_sha256"))
    if (not isinstance(probe_ids, list) or not rank <= len(probe_ids) <= 2048 or
            any(not isinstance(value, str) or not value or len(value) > 256 or "\0" in value for value in probe_ids) or
            len(set(probe_ids)) != len(probe_ids) or not isinstance(endpoints, list) or len(endpoints) != len(probe_ids)):
        raise ValueError("residual probe IDs and endpoints require a bounded auditable seed bank")
    pole_ids = []
    for pair in endpoints:
        if not isinstance(pair, dict) or set(pair) != {"negative", "positive"}:
            raise ValueError("residual source poles require explicit lower and higher record IDs")
        for value in pair.values():
            if not isinstance(value, str) or not value or len(value) > 256 or "\0" in value:
                raise ValueError("invalid residual source pole record ID")
            pole_ids.append(value)
    if len(set(pole_ids)) != len(pole_ids):
        raise ValueError("residual source pole record IDs must be unique")
    if (not isinstance(capture_hashes, list) or not 0 < len(capture_hashes) <= 2048 or
            any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
                for value in capture_hashes) or len(set(capture_hashes)) != len(capture_hashes)):
        raise ValueError("residual source capture checksums are invalid")


def _validate(value, *, expected_identity):
    if not isinstance(value, ProbeResidualCompression):
        raise ValueError("invalid probe residual object")
    identity, rank, dtype, base_bytes, correction_bytes, total = _metadata_scope(value.metadata, expected_identity)
    if value.identity != identity:
        raise ValueError("residual object and metadata source scopes differ")
    quant._validate(value.base_quantization, expected_identity=identity)
    if value.base_quantization.bits != 4 or value.base_quantization.block_size != value.metadata["block_size"]:
        raise ValueError("residual base precision differs")
    for tensor, shape, wanted_dtype in ((value.span_basis, (identity.matrix_shape[1], rank), torch.float32),
            (value.coefficients, (identity.expert_count, identity.matrix_shape[0], rank), _COEFFICIENT_DTYPES[dtype])):
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != wanted_dtype or
                tuple(tensor.shape) != shape or not tensor.is_contiguous() or not base._finite_tensor(tensor)):
            raise ValueError("invalid finite contiguous residual factors")
    if not torch.allclose(value.span_basis.T @ value.span_basis, torch.eye(rank), atol=2e-4, rtol=2e-4):
        raise ValueError("residual span basis must be orthonormal within FP32 rounding")
    if value.metadata["objective"].get("basis_kind") == "probe_pca_complement_span":
        protected = value.metadata["objective"]["protected_probe_rank"]
        digest = hashlib.sha256(value.span_basis[:, :protected].contiguous().numpy().tobytes()).hexdigest()
        if digest != value.metadata["objective"]["protected_basis_sha256"]:
            raise ValueError("hybrid protected basis prefix differs from its recorded evidence")
    if value.span_basis.numel() * 4 + value.coefficients.numel() * value.coefficients.element_size() != correction_bytes:
        raise ValueError("residual correction tensors differ from declared storage")
    _count(value.max_read_bytes, "max_read_bytes", 1)


def fit_probe_residual_group(reader, bank, *, base_quantization=None, coefficient_dtype="bfloat16",
                             block_size=128, chunk_blocks=256, chunk_columns=65536,
                             max_working_bytes=1 << 30, max_axes=128, svd_rtol=1e-5,
                             expected_origin="measured_model_state", max_read_bytes=128 << 20):
    """Repair exactly the selected INT4 base; never pool independent experts."""
    identity, frozen, row_chunk, estimate, svd_rtol = _options(reader, bank, coefficient_dtype,
        block_size, chunk_blocks, chunk_columns, max_working_bytes, max_axes, svd_rtol, expected_origin, max_read_bytes)
    q, values = _span(frozen, svd_rtol)
    objective = {"identity_sha256": identity.fingerprint,
        "input_state_scope": InputStateScope.from_identity(identity, state_site=SITE).to_dict(),
        "probe_bank_scope": frozen["scope"], "probe_scope_sha256": frozen["scope_sha256"],
        "probe_evidence_sha256": frozen["evidence_sha256"], "probe_ids": frozen["probe_ids"],
        "pole_record_ids": frozen["pole_record_ids"], "source_capture_sha256": frozen["source_capture_sha256"],
        "record_origin": frozen["origin"], "measured_temporal_transitions": 0,
        "heldout_used_for_fit": False, "svd_rtol": svd_rtol, "span_singular_values": values.tolist(),
        "basis_kind": "native_probe_seed_span", "feature_origin": "derived_native_probe_seed_contrasts"}
    return _fit_q_group(reader, q, objective, METHOD, base_quantization, coefficient_dtype,
        block_size, chunk_blocks, row_chunk, estimate, max_working_bytes, max_read_bytes)


def _fit_q_group(reader, q, objective, method, base_quantization, coefficient_dtype,
                 block_size, chunk_blocks, row_chunk, estimate, max_working_bytes, max_read_bytes):
    identity = reader.identity
    if base_quantization is not None:
        quant._validate(base_quantization, expected_identity=identity)
        if base_quantization.bits != 4 or base_quantization.block_size != block_size:
            raise ValueError("provided residual base must be the same scoped INT4 block control")
    if base_quantization is None:
        base_quantization = quant.quantize_reader(reader, bits=4, block_size=block_size,
            chunk_blocks=chunk_blocks, max_working_bytes=max_working_bytes)
    e, (rows, width) = identity.expert_count, identity.matrix_shape
    coefficients = torch.empty((e, rows, q.shape[1]), dtype=_COEFFICIENT_DTYPES[coefficient_dtype])
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            stop = min(rows, start + row_chunk)
            original = _read_rows(reader, identity, start, stop)
            if reader.identity != identity:
                raise ValueError("source scope changed during residual fitting")
            decoded = base_quantization.read_columns(start * width, stop * width).reshape(e, stop - start, width)
            if base_quantization.identity != identity:
                raise ValueError("INT4 source scope changed during residual fitting")
            coefficients[:, start:stop] = _responses(original - decoded, q).to(coefficients.dtype)
    if not base._finite_tensor(coefficients):
        raise ValueError("residual coefficient storage overflowed")
    correction_bytes = q.numel() * 4 + coefficients.numel() * coefficients.element_size()
    base_bytes = base_quantization.metadata["factor_tensor_bytes"]
    total = correction_bytes + base_bytes
    metadata = {"format": FORMAT, "method": method, "identity": identity.to_dict(),
        "identity_sha256": identity.fingerprint, "bits": 4, "block_size": block_size,
        "span_rank": q.shape[1], "span_basis_dtype": "float32", "coefficient_dtype": coefficient_dtype,
        "base_tensor_bytes": base_bytes, "correction_tensor_bytes": correction_bytes,
        "factor_tensor_bytes": total, "source_tensor_bytes": identity.actual_source_bytes,
        "tensor_storage_ratio": total / identity.actual_source_bytes,
        "tensor_storage_is_smaller": total < identity.actual_source_bytes,
        "formula": "Wq + ((W - Wq) Q) Q.T with per-expert stored coefficients",
        "coefficient_storage_rounding": "BF16 rounding of C" if coefficient_dtype == "bfloat16" else "FP32 rounding of C",
        "span_basis_orthogonality": "FP32 numerical tolerance, not exact arithmetic",
        "objective": objective,
        "evaluation_scope": "static_seed_contrast_linear_projection_proxy" if method == METHOD else "static_scoped_input_span_linear_projection_proxy",
        "native_fp8_kernel_evaluated": False, "semantic_meanings_validated": False,
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
        "estimated_numerical_peak_bytes": estimate,
        "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"}
    result = ProbeResidualCompression(identity, base_quantization, q, coefficients, metadata, max_read_bytes)
    _validate(result, expected_identity=identity)
    return result


def fit_scoped_residual_basis(reader, basis, *, base_quantization=None, coefficient_dtype="bfloat16",
                              block_size=128, chunk_blocks=256, chunk_columns=65536,
                              max_working_bytes=1 << 30, max_read_bytes=128 << 20,
                              expected_origin="measured_model_state"):
    """The same INT4 repair for explicit scoped PCA/random/control bases."""
    identity = reader.identity
    if not isinstance(identity, base.ExpertGroupIdentity) or identity.projection not in ("gate_proj", "up_proj"):
        raise ValueError("scoped residual basis requires a gate/up source group")
    for value, name in ((block_size, "block_size"), (chunk_blocks, "chunk_blocks"), (chunk_columns, "chunk_columns"),
                        (max_working_bytes, "max_working_bytes"), (max_read_bytes, "max_read_bytes")):
        _count(value, name, 1)
    if block_size % 2 or not isinstance(coefficient_dtype, str) or coefficient_dtype not in _COEFFICIENT_DTYPES:
        raise ValueError("scoped residual storage requires even INT4 blocks and BF16/FP32 coefficients")
    if not isinstance(basis, dict) or basis.get("format") != "glm-native-residual-basis-v1":
        raise ValueError("an auditable scoped residual basis is required")
    kind = basis.get("basis_kind")
    if not isinstance(kind, str) or kind not in _BASIS_KINDS or basis.get("feature_origin") != _BASIS_KINDS[kind]:
        raise ValueError("control basis kind and explicit feature origin differ")
    scope, layer = _reference_scope(basis, expected_origin)
    teacher = scope["teacher"]
    if (teacher["model_id"] != identity.model_id or teacher["revision"] != identity.revision or layer != identity.layer or
            teacher["hidden_size"] != identity.matrix_shape[1] or _SOURCE_DTYPES.get(teacher["source_precision"]) != identity.source_dtype):
        raise ValueError("control basis differs from the exact source teacher, layer, width or precision")
    wanted_scope = InputStateScope.from_identity(identity, state_site=SITE).to_dict()
    if basis.get("input_state_scope") is not None and basis["input_state_scope"] != wanted_scope:
        raise ValueError("calibration PCA source matrix group scope differs")
    q = basis.get("basis")
    if not isinstance(q, torch.Tensor) or q.ndim != 2 or q.shape[0] != identity.matrix_shape[1] or not 0 < q.shape[1] <= min(q.shape[0], 2048):
        raise ValueError("control basis must have bounded exact native width and nonzero rank")
    rank = q.shape[1]
    requested = _count(basis.get("requested_rank"), "requested_rank", 1)
    if requested > min(q.shape[0], 2048) or basis.get("actual_rank") != rank or basis.get("rank_matches_request") is not (requested == rank):
        raise ValueError("control basis rank comparison differs")
    if basis.get("heldout_used_for_fit") is not False or basis.get("semantic_meanings_validated") is not False:
        raise ValueError("control basis cannot fit heldout data or claim validated meanings")
    e, (rows, width) = identity.expert_count, identity.matrix_shape
    row_chunk = min(rows, max(1, chunk_columns // width))
    blocks = (identity.column_count + block_size - 1) // block_size
    base_bytes = e * blocks * (block_size // 2 + 4)
    correction_bytes = width * rank * 4 + e * rows * rank * (2 if coefficient_dtype == "bfloat16" else 4)
    decode_workspace = _read_plan(identity, rank, row_chunk * width, block_size, e)
    estimate = (base_bytes + correction_bytes + 8 * width * rank * 4 + 8 * rank**2 * 4 +
        e * min(blocks, chunk_blocks) * block_size * 48 + decode_workspace +
        8 * e * row_chunk * (width + rank) * 4 + 2 * 1024**2)
    if estimate > max_working_bytes or decode_workspace > max_read_bytes:
        raise MemoryError("scoped basis, INT4 base, correction factors and read workspace exceed budget")
    if q.device.type != "cpu" or q.dtype != torch.float32 or not base._finite_tensor(q.contiguous()):
        raise ValueError("control basis must contain finite CPU FP32 values")
    if not torch.allclose(q.T @ q, torch.eye(rank), atol=2e-4, rtol=2e-4):
        raise ValueError("control basis must be orthonormal within FP32 tolerance")
    if basis.get("evidence_sha256") != residual_basis_fingerprint(basis):
        raise ValueError("control basis evidence checksum differs")
    q = q.detach().clone().contiguous()
    # Only finite JSON evidence is copied into the serialized correction.
    details = json.loads(base._metadata_bytes({key: value for key, value in basis.items()
        if key not in ("basis", "scope", "scope_sha256", "origin", "evidence_sha256", "input_state_scope")}))
    objective = {**details, "identity_sha256": identity.fingerprint, "input_state_scope": wanted_scope,
        "basis_scope": scope, "basis_scope_sha256": basis["scope_sha256"],
        "basis_evidence_sha256": basis["evidence_sha256"], "record_origin": basis["origin"],
        "measured_temporal_transitions": 0, "heldout_used_for_fit": False}
    _validate_control_evidence(objective, rank)
    if kind == "probe_pca_complement_span":
        protected = objective["protected_probe_rank"]
        digest = hashlib.sha256(q[:, :protected].contiguous().numpy().tobytes()).hexdigest()
        if digest != objective["protected_basis_sha256"] or objective["native_probe_scope_sha256"] != basis["scope_sha256"]:
            raise ValueError("hybrid protected prefix or native source scope checksum differs")
    return _fit_q_group(reader, q, objective, SCOPED_METHOD, base_quantization, coefficient_dtype,
        block_size, chunk_blocks, row_chunk, estimate, max_working_bytes, max_read_bytes)


def evaluate_probe_residual_group(decoded_reader, source_reader, bank, *, chunk_columns=65536,
                                  max_working_bytes=1 << 30, max_axes=128, svd_rtol=1e-5,
                                  expected_origin="measured_model_state"):
    """The same bounded static response metric for residual and INT4 readers."""
    identity, frozen, row_chunk, estimate, svd_rtol = _prepare(source_reader, bank, 0, "float32",
        chunk_columns, max_working_bytes, max_axes, svd_rtol, expected_origin)
    if decoded_reader.identity != identity:
        raise ValueError("static response readers differ in exact source scope")
    if isinstance(decoded_reader, ProbeResidualCompression):
        _metadata_scope(decoded_reader.metadata, identity)
        objective = decoded_reader.metadata["objective"]
        fitted_scope = objective.get("basis_scope", objective.get("probe_bank_scope"))
        if objective["record_origin"] != frozen["origin"] or fitted_scope != frozen["scope"]:
            raise ValueError("residual evaluation bank origin or teacher formatting differs")
        payload = decoded_reader.metadata["factor_tensor_bytes"]
        rank = decoded_reader.span_basis.shape[1]
        block_size = decoded_reader.base_quantization.block_size
    elif isinstance(decoded_reader, quant.BlockQuantization):
        quant._metadata_scope(decoded_reader.metadata, identity)
        if decoded_reader.bits != 4:
            raise ValueError("residual static comparison requires the identical INT4 control")
        objective = {}
        payload, rank, block_size = decoded_reader.metadata["factor_tensor_bytes"], 0, decoded_reader.block_size
    else:
        raise ValueError("static residual controls require a bounded INT4 or residual reader")
    estimate += payload + _read_plan(identity, rank, row_chunk * identity.matrix_shape[1], block_size, identity.expert_count)
    if estimate > max_working_bytes:
        raise MemoryError("stored residual/control arrays and static evaluation workspace exceed byte budget")
    if isinstance(decoded_reader, ProbeResidualCompression):
        _validate(decoded_reader, expected_identity=identity)
    else:
        quant._validate(decoded_reader, expected_identity=identity)
    q, _ = _span(frozen, svd_rtol)
    directions = frozen["directions"].T
    energies = torch.zeros(len(frozen["probe_ids"]), dtype=torch.float64)
    errors = torch.zeros_like(energies)
    span_energy = span_error = 0.
    rows, width = identity.matrix_shape
    with torch.no_grad():
        for start in range(0, rows, row_chunk):
            stop = min(rows, start + row_chunk)
            original = _read_rows(source_reader, identity, start, stop)
            decoded = decoded_reader.read_columns(start * width, stop * width).reshape(identity.expert_count, stop - start, width)
            if source_reader.identity != identity or decoded_reader.identity != identity:
                raise ValueError("reader source scope changed during static residual evaluation")
            reference, proposed = _responses(original, q), _responses(decoded, q)
            span_energy += float(reference.double().square().sum())
            span_error += float((proposed.double() - reference.double()).square().sum())
            reference, proposed = _responses(original, directions), _responses(decoded, directions)
            energies.add_(reference.double().square().sum((0, 1)))
            errors.add_((proposed.double() - reference.double()).square().sum((0, 1)))
    if not math.isfinite(span_energy + span_error) or not bool(torch.isfinite(energies).all() & torch.isfinite(errors).all()):
        raise ValueError("static residual evaluation overflowed")
    axes = [{"probe_id": probe, "source_energy_sq": float(energy), "absolute_error_sq": float(error),
        "relative_rmse": math.sqrt(float(error / energy)) if energy > 0 else None,
        "seed_endpoint_difference_error_sq": float(error * magnitude.double().square())}
        for probe, energy, error, magnitude in zip(frozen["probe_ids"], energies, errors, frozen["contrast_magnitudes"])]
    return {"evaluation_scope": "static_seed_contrast_linear_projection_proxy", "record_origin": frozen["origin"],
        "fitted_basis_kind": objective.get("basis_kind", "native_probe_seed_span" if isinstance(decoded_reader, ProbeResidualCompression) else "unrepaired_int4_control"),
        "probe_evidence_sha256": frozen["evidence_sha256"],
        "same_probe_bank_used_to_fit": objective.get("probe_evidence_sha256", objective.get("native_probe_evidence_sha256")) == frozen["evidence_sha256"],
        "contains_protected_native_probe_span": objective.get("contains_protected_native_probe_span", objective.get("basis_kind") == "native_probe_seed_span"),
        "probe_axes": len(axes), "probe_span_rank": q.shape[1], "span_response_source_energy_sq": span_energy,
        "span_response_absolute_error_sq": span_error,
        "span_response_relative_rmse": math.sqrt(span_error / span_energy) if span_energy else None,
        "zero_source_axes": int((energies == 0).sum()), "axis_responses": axes,
        "semantic_meanings_validated": False, "temporal_transition_evaluated": False,
        "native_fp8_kernel_evaluated": False, "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
        "estimated_numerical_peak_bytes": estimate,
        "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_paths(directory):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("residual artifact must be an ordinary directory")
    paths = [directory / "metadata.json", directory / "correction.pt", directory / "base" / "metadata.json",
             directory / "base" / "factors.pt"]
    if (directory.joinpath("base").is_symlink() or not directory.joinpath("base").is_dir() or
            set(p.name for p in directory.iterdir()) != {"metadata.json", "correction.pt", "base"}):
        raise ValueError("residual artifact contains unexpected files or a linked base")
    if set(p.name for p in (directory / "base").iterdir()) != {"metadata.json", "factors.pt"}:
        raise ValueError("residual base contains unexpected files")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("residual artifact paths must be ordinary files")
    return paths


def save_probe_residual(value, directory, *, max_artifact_bytes=1 << 30):
    """Save a fresh atomic directory; restricted-load the complete round trip."""
    _count(max_artifact_bytes, "max_artifact_bytes", 1)
    _validate(value, expected_identity=value.identity)
    if value.metadata["factor_tensor_bytes"] > max_artifact_bytes:
        raise MemoryError("residual tensor payload exceeds the artifact byte limit")
    directory = Path(directory)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("choose a new residual artifact directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    intended_parent = directory.parent.resolve()
    prefix = f".{directory.name}-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=intended_parent))
    try:
        base_receipt = quant.save_block_quantization(value.base_quantization, staging / "base", max_artifact_bytes=max_artifact_bytes)
        correction = staging / "correction.pt"
        torch.save({"span_basis": base._standalone_storage(value.span_basis),
                    "coefficients": base._standalone_storage(value.coefficients)}, correction)
        metadata = {**value.metadata, "correction_sha256": _sha(correction),
            "base_metadata_sha256": _sha(staging / "base" / "metadata.json"),
            "base_factors_sha256": _sha(staging / "base" / "factors.pt"),
            "base_serialized_bytes": base_receipt["serialized_total_bytes"],
            "correction_serialized_bytes": correction.stat().st_size, "metadata_json_bytes": 0,
            "serialized_total_bytes": 0, "serialized_storage_ratio": 0., "serialized_is_smaller": False}
        for _ in range(16):
            encoded = base._metadata_bytes(metadata)
            total = len(encoded) + metadata["base_serialized_bytes"] + metadata["correction_serialized_bytes"]
            if metadata["metadata_json_bytes"] == len(encoded) and metadata["serialized_total_bytes"] == total:
                break
            metadata.update(metadata_json_bytes=len(encoded), serialized_total_bytes=total,
                serialized_storage_ratio=total / value.identity.actual_source_bytes,
                serialized_is_smaller=total < value.identity.actual_source_bytes)
        else:
            raise ValueError("residual serialized accounting did not stabilize")
        if total > max_artifact_bytes:
            raise MemoryError("serialized residual artifact exceeds its byte limit")
        (staging / "metadata.json").write_bytes(encoded)
        load_probe_residual(staging, expected_identity=value.identity, max_artifact_bytes=max_artifact_bytes,
                            max_read_bytes=value.max_read_bytes)
        os.replace(staging, directory)
        return metadata
    except BaseException:
        base._cleanup_staging(staging, intended_parent, prefix)
        raise


def load_probe_residual(directory, *, expected_identity, max_artifact_bytes=1 << 30,
                        max_metadata_bytes=1 << 20, max_read_bytes=128 << 20):
    """Verify exact files, hashes, byte counts and uncompressed tensor storage."""
    for value, name in ((max_artifact_bytes, "max_artifact_bytes"), (max_metadata_bytes, "max_metadata_bytes"),
                        (max_read_bytes, "max_read_bytes")):
        _count(value, name, 1)
    if not isinstance(expected_identity, base.ExpertGroupIdentity):
        raise ValueError("residual loading requires the expected exact source identity")
    meta, correction, base_meta, base_factors = _artifact_paths(directory)
    sizes = [path.stat().st_size for path in (meta, correction, base_meta, base_factors)]
    if sizes[0] > max_metadata_bytes or sizes[2] > max_metadata_bytes or sum(sizes) > max_artifact_bytes:
        raise MemoryError("residual artifact files exceed the load byte limit")
    try:
        metadata = json.loads(meta.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid residual metadata JSON") from exc
    identity, rank, dtype, _, correction_bytes, payload = _metadata_scope(metadata, expected_identity)
    if payload > max_artifact_bytes:
        raise MemoryError("declared residual payload exceeds load limit")
    actual = sum(sizes)
    if (metadata.get("metadata_json_bytes") != sizes[0] or metadata.get("correction_serialized_bytes") != sizes[1] or
            metadata.get("base_serialized_bytes") != sizes[2] + sizes[3] or metadata.get("serialized_total_bytes") != actual or
            isinstance(metadata.get("serialized_storage_ratio"), bool) or metadata.get("serialized_storage_ratio") != actual / identity.actual_source_bytes or
            metadata.get("serialized_is_smaller") is not (actual < identity.actual_source_bytes)):
        raise ValueError("invalid residual serialized byte accounting")
    for path, name in ((correction, "correction_sha256"), (base_meta, "base_metadata_sha256"), (base_factors, "base_factors_sha256")):
        if _sha(path) != metadata.get(name):
            raise ValueError("residual artifact checksum differs")
    try:
        with zipfile.ZipFile(correction) as archive:
            members = archive.infolist()
            if any(item.compress_type != zipfile.ZIP_STORED for item in members):
                raise ValueError("correction tensors require an uncompressed Torch container")
            if sum(item.file_size for item in members if "/data/" in item.filename) != correction_bytes:
                raise ValueError("correction tensor storage differs from its bounded declaration")
            if sum(item.file_size for item in members) > max_artifact_bytes:
                raise MemoryError("expanded residual tensor container exceeds load limit")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid residual Torch container") from exc
    # No arbitrary Python/model object is reconstructed by either loader.
    factors = torch.load(correction, weights_only=True, map_location="cpu")
    if not isinstance(factors, dict) or set(factors) != {"span_basis", "coefficients"}:
        raise ValueError("invalid residual correction tensor keys")
    base_value = quant.load_block_quantization(base_meta.parent, expected_identity=identity,
        max_artifact_bytes=max_artifact_bytes, max_metadata_bytes=max_metadata_bytes)
    result = ProbeResidualCompression(identity, base_value, factors["span_basis"], factors["coefficients"], metadata, max_read_bytes)
    _validate(result, expected_identity=identity)
    return result
