"""Experimental shared geometry for one group of MoE expert weight matrices.

This encodes flattened matrices across experts, not activations or topology.
It is a storage experiment: a dense reconstructed expert, or ``rank + 1``
dense projections, does not inherently reduce runtime memory or computation.
No source weights are changed. Reconstruction errors measure fitted weights,
not held-out model behavior, language quality, or calibration quality.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Protocol, Sequence
import zipfile

import torch


FORMAT_VERSION = "expert-shared-basis-pca-v1"
_STORAGE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}
_SOURCE_ITEM_BYTES = {
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}
_READ_DTYPES = set(_STORAGE_DTYPES.values())
_FACTOR_NAMES = {"mean", "basis", "coefficients"}


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _nonblank(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a nonblank string")
    return value


@dataclass(frozen=True)
class ExpertGroupIdentity:
    """An exact source scope; equal shape alone never licenses grouping.

    ``layout`` identifies row-major matrix semantics, including any gate/up
    split or interleaving. ``source_storage_bytes`` must include the checkpoint
    tensors' quantization scales when present. It is mandatory for FP8 sources;
    fitting their dequantized FP32 values must not inflate the byte baseline.
    Expert IDs define coefficient row order and cannot be silently permuted.
    """

    model_id: str
    revision: str
    layer: int
    projection: str
    layout: str
    expert_ids: tuple[str, ...]
    matrix_shape: tuple[int, int]
    source_dtype: str
    source_storage_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "revision", "projection", "layout"):
            _nonblank(getattr(self, name), name)
        _integer(self.layer, "layer")
        if not isinstance(self.expert_ids, tuple) or not self.expert_ids:
            raise ValueError("expert_ids must be a nonempty ordered tuple")
        for value in self.expert_ids:
            _nonblank(value, "expert ID")
        if len(set(self.expert_ids)) != len(self.expert_ids):
            raise ValueError("expert IDs must be unique")
        if not isinstance(self.matrix_shape, tuple) or len(self.matrix_shape) != 2:
            raise ValueError("matrix_shape must be a (rows, columns) tuple")
        for value in self.matrix_shape:
            _integer(value, "matrix dimension", 1)
        if self.source_dtype not in _SOURCE_ITEM_BYTES:
            raise ValueError("unsupported source storage dtype")
        minimum = self.expert_count * self.column_count * _SOURCE_ITEM_BYTES[self.source_dtype]
        if self.source_storage_bytes is None:
            if self.source_dtype.startswith("float8_"):
                raise ValueError("FP8 sources require actual source_storage_bytes including scales")
        elif _integer(self.source_storage_bytes, "source_storage_bytes", 1) < minimum:
            raise ValueError("source_storage_bytes is smaller than the source tensor payload")

    @property
    def expert_count(self) -> int:
        return len(self.expert_ids)

    @property
    def column_count(self) -> int:
        return self.matrix_shape[0] * self.matrix_shape[1]

    @property
    def actual_source_bytes(self) -> int:
        if self.source_storage_bytes is not None:
            return self.source_storage_bytes
        return self.expert_count * self.column_count * _SOURCE_ITEM_BYTES[self.source_dtype]

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "layer": self.layer,
            "projection": self.projection,
            "layout": self.layout,
            "expert_ids": list(self.expert_ids),
            "matrix_shape": list(self.matrix_shape),
            "source_dtype": self.source_dtype,
            "source_storage_bytes": self.source_storage_bytes,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ExpertGroupIdentity":
        keys = {"model_id", "revision", "layer", "projection", "layout", "expert_ids",
                "matrix_shape", "source_dtype", "source_storage_bytes"}
        if not isinstance(value, dict) or set(value) != keys:
            raise ValueError("invalid or incomplete group identity")
        if not isinstance(value["expert_ids"], list) or not isinstance(value["matrix_shape"], list):
            raise ValueError("invalid identity lists")
        return cls(**{**value, "expert_ids": tuple(value["expert_ids"]),
                      "matrix_shape": tuple(value["matrix_shape"])})

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ExpertMatrixReader(Protocol):
    identity: ExpertGroupIdentity

    def read_columns(self, start: int, stop: int,
                     expert_ids: Sequence[str] | None = None) -> torch.Tensor:
        """Return CPU floating values shaped (requested experts, stop-start).

        Columns index a row-major flattening of each matrix. Decoding FP8 and
        its scales is the reader's responsibility; fitting accepts FP32/BF16/
        FP16 values and records the original storage bytes separately.
        """


class TensorExpertReader:
    """Small in-memory CPU adapter, useful for synthetic validation."""

    def __init__(self, identity: ExpertGroupIdentity, weights: torch.Tensor):
        if not isinstance(identity, ExpertGroupIdentity):
            raise ValueError("identity must be an ExpertGroupIdentity")
        if not isinstance(weights, torch.Tensor) or weights.device.type != "cpu":
            raise ValueError("weights must be a CPU tensor")
        if weights.dtype not in _READ_DTYPES or not weights.is_contiguous():
            raise ValueError("weights must be contiguous FP32/BF16/FP16 values")
        if tuple(weights.shape) != (identity.expert_count, *identity.matrix_shape):
            raise ValueError("weights do not match the scoped matrix shape")
        self.identity = identity
        self.weights = weights

    def read_columns(self, start: int, stop: int,
                     expert_ids: Sequence[str] | None = None) -> torch.Tensor:
        _integer(start, "start")
        _integer(stop, "stop", 1)
        if not start < stop <= self.identity.column_count:
            raise ValueError("column interval is outside the matrix")
        ids = self.identity.expert_ids if expert_ids is None else tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(x not in self.identity.expert_ids for x in ids):
            raise ValueError("invalid requested expert IDs")
        flat = self.weights.view(self.identity.expert_count, self.identity.column_count)
        if ids == self.identity.expert_ids:
            return flat[:, start:stop]
        positions = [self.identity.expert_ids.index(x) for x in ids]
        return flat[positions, start:stop]


@dataclass
class ExpertCompression:
    mean: torch.Tensor
    basis: torch.Tensor
    coefficients: torch.Tensor
    metadata: dict

    @property
    def identity(self) -> ExpertGroupIdentity:
        return ExpertGroupIdentity.from_dict(self.metadata.get("identity"))

    @property
    def rank(self) -> int:
        return self.metadata["rank"]


def _memory_plan(identity: ExpertGroupIdentity, rank: int, chunk: int, dtype: str) -> dict:
    e, p = identity.expert_count, identity.column_count
    b = _SOURCE_ITEM_BYTES[dtype]
    factors = (rank + 1) * p * b + e * rank * 4
    # Conservative numerical arrays, E^2 eigensolver working arrays and reader
    # chunk (up to FP32). Backend/allocator overhead is not an exact RSS promise.
    workspace = 8 * e * e * 4 + (6 * e + 3 * rank + 3) * chunk * 4
    return {"factor_tensor_bytes": factors, "estimated_numerical_peak_bytes": factors + workspace,
            "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead"}


def _read_chunk(reader: ExpertMatrixReader, identity: ExpertGroupIdentity,
                start: int, stop: int) -> torch.Tensor:
    if reader.identity != identity:
        raise ValueError("reader identity changed; group scopes cannot be mixed")
    value = reader.read_columns(start, stop, expert_ids=identity.expert_ids)
    if reader.identity != identity:
        raise ValueError("reader identity changed during its source read (source scope changed)")
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("reader must return a CPU tensor")
    if value.dtype not in _READ_DTYPES or tuple(value.shape) != (identity.expert_count, stop - start):
        raise ValueError("reader returned an invalid dtype or chunk shape")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("source weights contain nonfinite values")
    return value.detach().to(torch.float32)


def _center_chunk(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Referencing one row before averaging makes identical experts exactly
    # identical rather than manufacturing tiny FP32 centered variance.
    average = value[0] + (value - value[0]).mean(dim=0)
    return average, value - average


def _objective(identity: ExpertGroupIdentity, expert_gram: torch.Tensor | None,
               objective_metadata: dict | None) -> tuple[dict, tuple[torch.Tensor, torch.Tensor] | None]:
    if expert_gram is None:
        if objective_metadata is not None:
            raise ValueError("objective_metadata requires an external expert_gram")
        return {"method": "weight_pca_shared_basis_v1", "identity_sha256": identity.fingerprint,
                "fit_scope": "source_expert_weights", "semantic_quality_evaluated": False,
                "runtime_savings_evaluated": False}, None
    if not isinstance(objective_metadata, dict):
        raise ValueError("external Gram requires objective_metadata")
    try:
        encoded = json.dumps(objective_metadata, allow_nan=False)
        objective = json.loads(encoded)
    except (ValueError, TypeError) as exc:
        raise ValueError("objective_metadata must be finite JSON") from exc
    if len(encoded.encode()) > (1 << 20):
        raise ValueError("objective_metadata is too large")
    if objective.get("method") not in {"trajectory_response_shared_basis_v1", "probe_response_shared_basis_v1"} or objective.get("identity_sha256") != identity.fingerprint or objective.get("semantic_quality_evaluated") is not False:
        raise ValueError("external Gram objective must identify its method, scope and unevaluated semantic quality")
    if objective.get("runtime_savings_evaluated", False) is not False:
        raise ValueError("external Gram cannot assert runtime savings")
    e = identity.expert_count
    if not isinstance(expert_gram, torch.Tensor) or expert_gram.device.type != "cpu" or expert_gram.dtype != torch.float32 or tuple(expert_gram.shape) != (e, e) or not bool(torch.isfinite(expert_gram).all()):
        raise ValueError("expert_gram must be finite CPU FP32 E-by-E")
    tolerance = max(1e-7, float(expert_gram.abs().max()) * 2e-5)
    if not torch.allclose(expert_gram, expert_gram.T, rtol=0, atol=tolerance):
        raise ValueError("expert_gram must be symmetric")
    if not bool((expert_gram.sum(dim=0).abs() <= tolerance * e).all()):
        raise ValueError("expert_gram must be centered across experts")
    # Remove only tolerated numerical asymmetry. Do not repair a non-PSD Gram.
    symmetric = ((expert_gram + expert_gram.T) * .5).detach()
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    if float(eigenvalues.min()) < -tolerance * e:
        raise ValueError("expert_gram must be positive semidefinite")
    return objective, (eigenvalues.clamp_min(0), eigenvectors)


def fit_expert_group(reader: ExpertMatrixReader, rank: int, *, chunk_columns: int = 65536,
                     storage_dtype: str = "float16", max_working_bytes: int = 1 << 30,
                     expected_identity: ExpertGroupIdentity | None = None,
                     expert_gram: torch.Tensor | None = None,
                     objective_metadata: dict | None = None) -> ExpertCompression:
    """Fit a centered, across-expert dual PCA in two streamed CPU passes.

    Reconstruction is ``mean + coefficients[expert] @ basis``. Basis rows retain
    magnitude; coefficients are orthonormal FP32 eigenvectors. Both passes read
    only column chunks, never an E-by-all-weights matrix. Rank zero is a genuine
    mean-only encoding. An explicitly scoped external centered expert Gram can
    select a different subspace while reusing this decoder and storage format.
    Its objective is recorded separately from the weight-PCA control.
    Non-saving ranks remain usable diagnostics, clearly
    labelled rather than reported as successful compression.
    """
    identity = reader.identity
    if not isinstance(identity, ExpertGroupIdentity):
        raise ValueError("reader requires a strict ExpertGroupIdentity")
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("reader does not match expected identity")
    rank = _integer(rank, "rank")
    if rank > min(identity.expert_count - 1, identity.column_count):
        raise ValueError("rank exceeds centered expert geometry")
    chunk_columns = _integer(chunk_columns, "chunk_columns", 1)
    max_working_bytes = _integer(max_working_bytes, "max_working_bytes", 1)
    if storage_dtype not in _STORAGE_DTYPES:
        raise ValueError("storage_dtype must be float16, bfloat16 or float32")
    chunk = min(chunk_columns, identity.column_count)
    plan = _memory_plan(identity, rank, chunk, storage_dtype)
    if plan["estimated_numerical_peak_bytes"] > max_working_bytes:
        raise MemoryError("requested factors and streamed fitting workspace exceed max_working_bytes")
    objective, external_eigensystem = _objective(identity, expert_gram, objective_metadata)

    e, p = identity.expert_count, identity.column_count
    gram = torch.zeros((e, e), dtype=torch.float32)
    raw_energy = mean_energy = centered_energy = 0.0
    with torch.no_grad():
        for start in range(0, p, chunk):
            value = _read_chunk(reader, identity, start, min(p, start + chunk))
            average, centered = _center_chunk(value)
            gram.add_(centered @ centered.T)
            raw_energy += float(value.square().sum(dtype=torch.float64))
            mean_energy += e * float(average.square().sum(dtype=torch.float64))
            centered_energy += float(centered.square().sum(dtype=torch.float64))
        if not bool(torch.isfinite(gram).all()) or not all(map(math.isfinite, (raw_energy, mean_energy, centered_energy))):
            raise ValueError("FP32 fitting overflowed; source precision/range is unsupported")
        eigenvalues, eigenvectors = torch.linalg.eigh(gram) if external_eigensystem is None else external_eigensystem
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0)
        coefficients = eigenvectors[:, order[:rank]].contiguous()
        if rank:
            pivot = coefficients.abs().argmax(dim=0)
            signs = coefficients[pivot, torch.arange(rank)].sign()
            coefficients.mul_(signs)
        dtype = _STORAGE_DTYPES[storage_dtype]
        mean = torch.empty(p, dtype=dtype)
        basis = torch.empty((rank, p), dtype=dtype)
        squared_error = maximum_error = 0.0
        for start in range(0, p, chunk):
            stop = min(p, start + chunk)
            value = _read_chunk(reader, identity, start, stop)
            average, centered = _center_chunk(value)
            mean[start:stop] = average
            basis[:, start:stop] = coefficients.T @ centered
            if not bool(torch.isfinite(mean[start:stop]).all()) or not bool(torch.isfinite(basis[:, start:stop]).all()):
                raise ValueError("storage precision overflowed; use a wider storage dtype")
            reconstructed = coefficients @ basis[:, start:stop].float() + mean[start:stop].float()
            difference = reconstructed - value
            squared_error += float(difference.square().sum(dtype=torch.float64))
            maximum_error = max(maximum_error, float(difference.abs().max()))
    if not math.isfinite(squared_error):
        raise ValueError("reconstruction error overflowed")
    selected_energy = max(0.0, float(torch.trace(coefficients.T @ gram @ coefficients)))
    result = ExpertCompression(mean, basis, coefficients, {
        "format": FORMAT_VERSION,
        "identity": identity.to_dict(),
        "identity_sha256": identity.fingerprint,
        "objective": objective,
        "rank": rank,
        "storage_dtype": storage_dtype,
        "coefficient_dtype": "float32",
        "fitting_dtype": "float32",
        "flattening": "row_major",
        "chunk_columns": chunk,
        "streaming_passes": 2,
        "source_tensor_bytes": identity.actual_source_bytes,
        **plan,
        "tensor_storage_is_smaller": plan["factor_tensor_bytes"] < identity.actual_source_bytes,
        "tensor_storage_ratio": plan["factor_tensor_bytes"] / identity.actual_source_bytes,
        "diagnostics": {
            "source_energy_frobenius_sq": raw_energy,
            "centered_energy_frobenius_sq": centered_energy,
            "mean_energy_frobenius_sq": mean_energy,
            "selected_centered_energy_frobenius_sq": selected_energy,
            "retained_centered_energy_fraction": min(1.0, selected_energy / centered_energy) if centered_energy else 1.0,
            "reconstruction_error_frobenius_sq": squared_error,
            "relative_frobenius_error": math.sqrt(squared_error / raw_energy) if raw_energy else 0.0,
            "maximum_absolute_error": maximum_error,
            "evaluation_scope": "fitted_weight_reconstruction_only",
            "semantic_quality_evaluated": False,
            "runtime_savings_evaluated": False,
        },
    })
    _validate_result(result, expected_identity=identity)
    return result


def _validate_result(result: ExpertCompression, *, expected_identity: ExpertGroupIdentity) -> None:
    if not isinstance(result, ExpertCompression) or not isinstance(result.metadata, dict):
        raise ValueError("invalid expert compression")
    metadata = result.metadata
    if metadata.get("format") != FORMAT_VERSION:
        raise ValueError("unknown compression format")
    identity = ExpertGroupIdentity.from_dict(metadata.get("identity"))
    if identity != expected_identity or metadata.get("identity_sha256") != identity.fingerprint:
        raise ValueError("factor scope does not match expected identity")
    objective = metadata.get("objective")
    if not isinstance(objective, dict) or objective.get("method") not in {"weight_pca_shared_basis_v1", "trajectory_response_shared_basis_v1", "probe_response_shared_basis_v1"} or objective.get("identity_sha256") != identity.fingerprint or objective.get("semantic_quality_evaluated") is not False or objective.get("runtime_savings_evaluated", False) is not False:
        raise ValueError("invalid objective method, scope or quality claim")
    rank = _integer(metadata.get("rank"), "rank")
    if rank > min(identity.expert_count - 1, identity.column_count):
        raise ValueError("invalid factor rank")
    dtype = _STORAGE_DTYPES.get(metadata.get("storage_dtype"))
    if dtype is None or metadata.get("coefficient_dtype") != "float32" or metadata.get("fitting_dtype") != "float32":
        raise ValueError("invalid factor precision")
    if metadata.get("flattening") != "row_major":
        raise ValueError("unsupported factor flattening")
    shapes = {"mean": (identity.column_count,), "basis": (rank, identity.column_count),
              "coefficients": (identity.expert_count, rank)}
    payload_bytes = 0
    for name, shape in shapes.items():
        tensor = getattr(result, name)
        wanted_dtype = torch.float32 if name == "coefficients" else dtype
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.dtype != wanted_dtype or tuple(tensor.shape) != shape:
            raise ValueError(f"invalid {name} factor shape/device/dtype")
        if not tensor.is_contiguous() or not _finite_tensor(tensor):
            raise ValueError(f"{name} must be contiguous and finite")
        payload_bytes += tensor.numel() * tensor.element_size()
    if metadata.get("factor_tensor_bytes") != payload_bytes or metadata.get("source_tensor_bytes") != identity.actual_source_bytes:
        raise ValueError("invalid source or factor byte accounting")
    if metadata.get("tensor_storage_is_smaller") is not (payload_bytes < identity.actual_source_bytes):
        raise ValueError("invalid tensor storage savings claim")
    ratio = metadata.get("tensor_storage_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (float, int)) or ratio != payload_bytes / identity.actual_source_bytes:
        raise ValueError("invalid tensor storage ratio")
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("evaluation_scope") != "fitted_weight_reconstruction_only":
        raise ValueError("missing reconstruction diagnostics scope")
    if diagnostics.get("semantic_quality_evaluated") is not False or diagnostics.get("runtime_savings_evaluated") is not False:
        raise ValueError("unsupported quality or runtime claim")
    for name in ("source_energy_frobenius_sq", "centered_energy_frobenius_sq", "mean_energy_frobenius_sq",
                 "selected_centered_energy_frobenius_sq", "retained_centered_energy_fraction",
                 "reconstruction_error_frobenius_sq", "relative_frobenius_error", "maximum_absolute_error"):
        value = diagnostics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid diagnostic {name}")
    if diagnostics["retained_centered_energy_fraction"] > 1:
        raise ValueError("invalid retained energy fraction")


def _finite_tensor(value: torch.Tensor, chunk: int = 65536) -> bool:
    # Validate large stored factors without allocating their full Boolean mask.
    flat = value.view(-1)
    return all(bool(torch.isfinite(flat[start:start + chunk]).all())
               for start in range(0, flat.numel(), chunk))


def reconstruct_expert(result: ExpertCompression, expert_id: str, *, dtype: torch.dtype = torch.float32,
                       max_output_bytes: int = 1 << 30, column_chunk: int = 65536) -> torch.Tensor:
    """Reconstruct one dense CPU matrix, allocating its full requested output."""
    identity = result.identity
    _validate_result(result, expected_identity=identity)
    if expert_id not in identity.expert_ids or dtype not in _READ_DTYPES:
        raise ValueError("unknown expert or unsupported output dtype")
    _integer(max_output_bytes, "max_output_bytes", 1)
    column_chunk = _integer(column_chunk, "column_chunk", 1)
    # Reconstruction arithmetic is FP32 even if the requested returned matrix is
    # narrower; account for that allocation before making it.
    if identity.column_count * 4 > max_output_bytes:
        raise MemoryError("dense expert reconstruction exceeds max_output_bytes")
    row = identity.expert_ids.index(expert_id)
    output = torch.empty(identity.column_count, dtype=torch.float32)
    for start in range(0, identity.column_count, column_chunk):
        stop = min(identity.column_count, start + column_chunk)
        output[start:stop] = result.coefficients[row] @ result.basis[:, start:stop].float() + result.mean[start:stop].float()
    output = output.reshape(identity.matrix_shape).to(dtype)
    if not _finite_tensor(output):
        raise ValueError("requested output dtype overflowed")
    return output


def expert_linear(result: ExpertCompression, expert_id: str, inputs: torch.Tensor,
                  bias: torch.Tensor | None = None, *, column_chunk: int = 1024) -> torch.Tensor:
    """CPU factorized linear operation without reconstructing a dense expert.

    This uses the mean projection and ``rank`` basis projections. It is a
    reference decoder, not an optimized MoE replacement or a speedup claim.
    The projection blocks avoid a full FP32 copy of stored basis matrices.
    """
    identity = result.identity
    _validate_result(result, expected_identity=identity)
    column_chunk = _integer(column_chunk, "column_chunk", 1)
    if expert_id not in identity.expert_ids or not isinstance(inputs, torch.Tensor) or inputs.device.type != "cpu" or inputs.dtype not in _READ_DTYPES:
        raise ValueError("unknown expert or invalid CPU input")
    rows, columns = identity.matrix_shape
    if inputs.ndim < 1 or inputs.shape[-1] != columns or not bool(torch.isfinite(inputs).all()):
        raise ValueError("input shape or values do not match the expert matrix")
    if bias is not None and (not isinstance(bias, torch.Tensor) or bias.device.type != "cpu" or bias.dtype not in _READ_DTYPES or tuple(bias.shape) != (rows,) or not bool(torch.isfinite(bias).all())):
        raise ValueError("invalid bias")
    coefficient = result.coefficients[identity.expert_ids.index(expert_id)]
    # Traverse output rows, not flattened column windows, so each projection
    # retains the canonical gate/up/down matrix axis conventions.
    x = inputs.float()
    output = torch.zeros((*inputs.shape[:-1], rows), dtype=torch.float32)
    for row_start in range(0, rows, column_chunk):
        row_stop = min(rows, row_start + column_chunk)
        start, stop = row_start * columns, row_stop * columns
        output[..., row_start:row_stop] = x @ result.mean[start:stop].float().view(row_stop - row_start, columns).T
        for k in range(result.rank):
            weights = result.basis[k, start:stop].float().view(row_stop - row_start, columns)
            output[..., row_start:row_stop].add_((x @ weights.T) * coefficient[k])
    if bias is not None:
        output.add_(bias.float())
    if not bool(torch.isfinite(output).all()):
        raise ValueError("linear output overflowed")
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_bytes(metadata: dict) -> bytes:
    encoded = (json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    # Trailing JSON whitespace fixes a possible size/float-ratio feedback loop:
    # byte totals are recorded inside that same metadata file. It is included
    # in actual serialized counts, never discounted as zero-cost overhead.
    return encoded + b" " * ((-len(encoded)) % 512)


def _standalone_storage(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    # Fitted factors already own exact storage, so exporting them does not
    # double the full basis allocation. A caller-provided contiguous slice may
    # retain unrelated backing values; copy just that declared factor then.
    if tensor.storage_offset() != 0 or tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size():
        return tensor.clone().contiguous()
    return tensor


def _cleanup_staging(staging: Path, output_parent: Path, name_prefix: str) -> None:
    """Remove only this export's resolved direct-child staging directory."""
    if not isinstance(name_prefix, str) or not name_prefix or not name_prefix.startswith("."):
        raise ValueError("invalid generated staging name prefix")
    staging = Path(staging)
    intended_parent = Path(output_parent).resolve()
    target = staging.resolve()
    if (target == intended_parent or target.parent != intended_parent or
            not target.name.startswith(name_prefix) or len(target.name) <= len(name_prefix) or
            target.name != staging.name):
        raise ValueError("refusing staging cleanup outside the intended export directory scope")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def save_expert_compression(result: ExpertCompression, directory: str | Path, *,
                            max_artifact_bytes: int = 1 << 30) -> dict:
    """Write a new atomic artifact directory; never overwrite an existing one.

    Tensor factors use a restricted-load-compatible Torch dictionary. Metadata
    includes actual Torch-file and JSON bytes, their sum, and savings against
    the original checkpoint group (including supplied quantization scales).
    """
    identity = result.identity
    _validate_result(result, expected_identity=identity)
    max_artifact_bytes = _integer(max_artifact_bytes, "max_artifact_bytes", 1)
    if result.metadata["factor_tensor_bytes"] > max_artifact_bytes:
        raise MemoryError("factor payload exceeds max_artifact_bytes")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("artifact directory already exists; use a new destination")
    directory.parent.mkdir(parents=True, exist_ok=True)
    intended_parent = directory.parent.resolve()
    staging_prefix = f".{directory.name}-"
    staging = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=intended_parent))
    try:
        factors_path = staging / "factors.pt"
        torch.save({name: _standalone_storage(getattr(result, name)) for name in sorted(_FACTOR_NAMES)}, factors_path)
        metadata = json.loads(json.dumps(result.metadata, allow_nan=False))
        metadata.update({"factors_sha256": _sha256_file(factors_path),
                         "serialized_tensor_file_bytes": factors_path.stat().st_size,
                         "metadata_json_bytes": 0, "serialized_total_bytes": 0,
                         "serialized_storage_ratio": 0.0, "serialized_is_smaller": False})
        for _ in range(16):
            encoded = _metadata_bytes(metadata)
            length = len(encoded)
            total = metadata["serialized_tensor_file_bytes"] + length
            if metadata["metadata_json_bytes"] == length and metadata["serialized_total_bytes"] == total:
                break
            metadata.update({"metadata_json_bytes": length, "serialized_total_bytes": total,
                             "serialized_storage_ratio": total / identity.actual_source_bytes,
                             "serialized_is_smaller": total < identity.actual_source_bytes})
        else:
            raise ValueError("metadata byte accounting did not stabilize")
        if total > max_artifact_bytes:
            raise MemoryError("serialized artifact including metadata exceeds max_artifact_bytes")
        (staging / "metadata.json").write_bytes(encoded)
        # Verify the exact files that will be committed, including safe reload.
        load_expert_compression(staging, expected_identity=identity, max_artifact_bytes=max_artifact_bytes)
        os.replace(staging, directory)
        return metadata
    except BaseException:
        _cleanup_staging(staging, intended_parent, staging_prefix)
        raise


def load_expert_compression(directory: str | Path, *, expected_identity: ExpertGroupIdentity,
                            max_artifact_bytes: int = 1 << 30,
                            max_metadata_bytes: int = 1 << 20) -> ExpertCompression:
    """Bounded, checksum-checked ``weights_only=True`` load with exact scope."""
    if not isinstance(expected_identity, ExpertGroupIdentity):
        raise ValueError("loading requires the expected exact source identity")
    max_artifact_bytes = _integer(max_artifact_bytes, "max_artifact_bytes", 1)
    max_metadata_bytes = _integer(max_metadata_bytes, "max_metadata_bytes", 1)
    directory = Path(directory)
    metadata_path, factors_path = directory / "metadata.json", directory / "factors.pt"
    metadata_size, factor_size = metadata_path.stat().st_size, factors_path.stat().st_size
    if metadata_size > max_metadata_bytes or metadata_size + factor_size > max_artifact_bytes:
        raise MemoryError("artifact files exceed load limits")
    try:
        metadata = json.loads(metadata_path.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("invalid metadata object")
    identity = ExpertGroupIdentity.from_dict(metadata.get("identity"))
    if identity != expected_identity or metadata.get("identity_sha256") != identity.fingerprint:
        raise ValueError("artifact scope does not match expected identity")
    rank = _integer(metadata.get("rank"), "rank")
    if rank > min(identity.expert_count - 1, identity.column_count) or metadata.get("storage_dtype") not in _STORAGE_DTYPES:
        raise ValueError("invalid artifact rank or precision")
    exact_payload = _memory_plan(identity, rank, 1, metadata["storage_dtype"])["factor_tensor_bytes"]
    if exact_payload > max_artifact_bytes or metadata.get("factor_tensor_bytes") != exact_payload:
        raise MemoryError("declared factor allocation exceeds load limits or shape accounting")
    total = metadata_size + factor_size
    if metadata.get("metadata_json_bytes") != metadata_size or metadata.get("serialized_tensor_file_bytes") != factor_size or metadata.get("serialized_total_bytes") != total:
        raise ValueError("serialized byte counts do not match the actual files")
    if metadata.get("serialized_is_smaller") is not (total < identity.actual_source_bytes) or metadata.get("serialized_storage_ratio") != total / identity.actual_source_bytes:
        raise ValueError("invalid serialized savings claim")
    if metadata.get("factors_sha256") != _sha256_file(factors_path):
        raise ValueError("factor checksum mismatch")
    # A compressed ZIP can hide a large storage behind a small disk footprint.
    # Torch's writer produces stored ZIP members; bound storage bytes before
    # restricted unpickling allocates them. Non-Torch/legacy containers fail.
    try:
        with zipfile.ZipFile(factors_path) as archive:
            members = archive.infolist()
            storage_sizes = [item.file_size for item in members if "/data/" in item.filename]
            if any(item.compress_type != zipfile.ZIP_STORED for item in members) or sum(storage_sizes) != exact_payload:
                raise ValueError("invalid or oversized tensor storage container")
            if sum(item.file_size for item in members) > max_artifact_bytes:
                raise MemoryError("expanded Torch archive exceeds load limits")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid Torch factor container") from exc
    payload = torch.load(factors_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != _FACTOR_NAMES:
        raise ValueError("unexpected factor payload")
    result = ExpertCompression(metadata=metadata, **payload)
    _validate_result(result, expected_identity=expected_identity)
    return result
