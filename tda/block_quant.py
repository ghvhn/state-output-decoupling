"""A simple symmetric block quantization control, not AWQ/GPTQ/NF4.

The codec preserves every ordered matrix slot. It quantizes values directly;
it does not calibrate language behavior, replace a model module, or optimize a
device kernel. All byte comparisons include scales and container metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import torch

from tda.expert_compress import (
    ExpertGroupIdentity, ExpertMatrixReader, _cleanup_staging, _metadata_bytes, _read_chunk,
    _standalone_storage,
)


FORMAT = "symmetric-block-control-v1"
METHOD = "unoptimized symmetric block rounding control; not NF4, AWQ or GPTQ"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bits(value):
    if isinstance(value, bool) or not isinstance(value, int) or value not in (4, 8):
        raise ValueError("bits must be the integer 4 or 8")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _metadata_scope(metadata, expected_identity):
    if not isinstance(metadata, dict):
        raise ValueError("quantization metadata must be an object")
    scope = ExpertGroupIdentity.from_dict(metadata.get("identity"))
    if metadata.get("format") != FORMAT or scope != expected_identity or metadata.get("identity_sha256") != scope.fingerprint:
        raise ValueError("artifact source scope mismatch")
    bits = _bits(metadata.get("bits"))
    size = _positive(metadata.get("block_size"), "block_size")
    if size % 2:
        raise ValueError("block_size must be even")
    blocks = (scope.column_count + size - 1) // size
    payload = scope.expert_count * blocks * (size * bits // 8 + 4)
    if metadata.get("factor_tensor_bytes") != payload or metadata.get("source_tensor_bytes") != scope.actual_source_bytes:
        raise ValueError("invalid byte accounting")
    if _number(metadata.get("tensor_storage_ratio"), "tensor_storage_ratio") != payload / scope.actual_source_bytes:
        raise ValueError("invalid tensor storage ratio")
    if metadata.get("tensor_storage_is_smaller") is not (payload < scope.actual_source_bytes):
        raise ValueError("invalid tensor storage savings claim")
    qmax = 2 ** (bits - 1) - 1
    if metadata.get("scale_dtype") != "float32" or metadata.get("level_range") != [-qmax, qmax]:
        raise ValueError("invalid quantization precision or level range")
    padding = metadata.get("padding_values")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding != blocks * size - scope.column_count:
        raise ValueError("invalid zero padding count")
    if metadata.get("evaluation_scope") != "fitted_weight_reconstruction_only" or metadata.get("method") != METHOD:
        raise ValueError("invalid quantization evaluation scope or method")
    if metadata.get("semantic_quality_evaluated") is not False or metadata.get("runtime_savings_evaluated") is not False:
        raise ValueError("unsupported quality claim")
    for name in ("relative_frobenius_error", "maximum_absolute_error"):
        _number(metadata.get(name), name)
    return scope, bits, size, blocks, payload


def _validate(value, *, expected_identity):
    if not isinstance(value, BlockQuantization):
        raise ValueError("invalid quantization object")
    scope, bits, size, blocks, payload = _metadata_scope(value.metadata, expected_identity)
    if value.identity != scope or _bits(value.bits) != bits or _positive(value.block_size, "block_size") != size:
        raise ValueError("quantization object and metadata disagree")
    codes, scales = value.codes, value.scales
    if not isinstance(codes, torch.Tensor) or codes.device.type != "cpu" or codes.dtype != torch.uint8 or tuple(codes.shape) != (scope.expert_count, blocks, size * bits // 8) or not codes.is_contiguous():
        raise ValueError("invalid packed codes")
    if not isinstance(scales, torch.Tensor) or scales.device.type != "cpu" or scales.dtype != torch.float32 or tuple(scales.shape) != (scope.expert_count, blocks) or not scales.is_contiguous():
        raise ValueError("invalid scales")
    qmax = 2 ** (bits - 1) - 1
    # Bound validation temporaries independently of the full stored payload.
    # At least one full block is required to test its canonical max code.
    chunk_blocks = max(1, 65536 // size)
    for first in range(0, blocks, chunk_blocks):
        last = min(blocks, first + chunk_blocks)
        scale = scales[:, first:last]
        if not bool(torch.isfinite(scale).all()) or bool((scale < 0).any()):
            raise ValueError("invalid scales")
        code = codes[:, first:last]
        if bits == 4:
            unpacked = torch.stack((code & 15, code >> 4), dim=-1).flatten(-2)
        else:
            unpacked = code
        if bool((unpacked == 2 * qmax + 1).any()):
            raise ValueError("unused quantization code")
        zero = (unpacked == qmax).all(-1)
        if bool(((scale == 0) & ~zero).any()) or bool(((scale > 0) & zero).any()):
            raise ValueError("zero block scales and codes are inconsistent")
        extreme = ((unpacked == 0) | (unpacked == 2 * qmax)).any(-1)
        if bool(((scale > 0) & ~extreme).any()):
            raise ValueError("positive block scale lacks its canonical maximum code")
    padding = blocks * size - scope.column_count
    if padding:
        final = codes[:, -1]
        if bits == 4:
            final = torch.stack((final & 15, final >> 4), dim=-1).flatten(-2)
        if not bool((final[:, -padding:] == qmax).all()):
            raise ValueError("padded quantization values must encode zero")
    if codes.numel() + scales.numel() * 4 != payload:
        raise ValueError("quantized factor storage does not match its declared shape")


@dataclass
class BlockQuantization:
    identity: ExpertGroupIdentity
    bits: int
    block_size: int
    codes: torch.Tensor
    scales: torch.Tensor
    metadata: dict

    def read_columns(self, start, stop, expert_ids=None):
        _positive(stop, "stop")
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < stop <= self.identity.column_count:
            raise ValueError("invalid column interval")
        ids = self.identity.expert_ids if expert_ids is None else tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(x not in self.identity.expert_ids for x in ids):
            raise ValueError("unknown or repeated matrix slot")
        first, last = start // self.block_size, (stop + self.block_size - 1) // self.block_size
        rows = [self.identity.expert_ids.index(x) for x in ids]
        packed = self.codes[rows, first:last]
        if self.bits == 4:
            unpacked = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        else:
            unpacked = packed
        signed = unpacked.float() - (2 ** (self.bits - 1) - 1)
        restored = (signed * self.scales[rows, first:last, None]).flatten(1)
        offset = start - first * self.block_size
        return restored[:, offset:offset + stop - start]


def quantize_reader(reader: ExpertMatrixReader, *, bits=4, block_size=128,
                    chunk_blocks=256, max_working_bytes=1 << 30):
    """Stream source values into packed INT4/INT8 codes and FP32 scales.

    Symmetric INT4 uses signed levels -7..7; nibble 15 is unused. The final
    block is zero padded and included in storage accounting. Scale zero
    represents an all-zero block. No input activation or label is consulted.
    """
    scope = reader.identity
    if not isinstance(scope, ExpertGroupIdentity):
        raise ValueError("strict source identity required")
    bits = _bits(bits)
    _positive(block_size, "block_size")
    _positive(chunk_blocks, "chunk_blocks")
    _positive(max_working_bytes, "max_working_bytes")
    if block_size % 2:
        raise ValueError("block_size must be even")
    e, p = scope.expert_count, scope.column_count
    blocks = (p + block_size - 1) // block_size
    code_width = block_size * bits // 8
    payload = e * blocks * (code_width + 4)
    # Includes the decoded/padded/rounding/reconstruction arrays and temporary
    # FP64 energy arrays; excludes backend/allocator bookkeeping.
    workspace = e * min(blocks, chunk_blocks) * block_size * 48
    if payload + workspace > max_working_bytes:
        raise MemoryError("quantized arrays and workspace exceed budget")
    codes = torch.empty((e, blocks, code_width), dtype=torch.uint8)
    scales = torch.empty((e, blocks), dtype=torch.float32)
    qmax = 2 ** (bits - 1) - 1
    energy = error = max_error = 0.0
    with torch.no_grad():
        for first in range(0, blocks, chunk_blocks):
            last = min(blocks, first + chunk_blocks)
            start, stop = first * block_size, min(p, last * block_size)
            values = _read_chunk(reader, scope, start, stop)
            if reader.identity != scope:
                raise ValueError("reader source scope changed during a quantization read")
            padded = torch.zeros((e, (last - first) * block_size), dtype=torch.float32)
            padded[:, :stop-start] = values
            shaped = padded.view(e, last-first, block_size)
            scale = shaped.abs().amax(-1) / qmax
            if not bool(torch.isfinite(scale).all()):
                raise ValueError("quantization scale overflowed")
            divisor = torch.where(scale > 0, scale, torch.ones_like(scale))
            signed = (shaped / divisor[..., None]).round().clamp(-qmax, qmax)
            unsigned = (signed + qmax).to(torch.uint8)
            if bits == 4:
                packed = unsigned[..., ::2] | (unsigned[..., 1::2] << 4)
            else:
                packed = unsigned
            codes[:, first:last] = packed
            scales[:, first:last] = scale
            restored = (signed * scale[..., None]).flatten(1)[:, :stop-start]
            delta = restored - values
            energy += float(values.double().square().sum())
            error += float(delta.double().square().sum())
            max_error = max(max_error, float(delta.abs().max()))
    if not all(map(math.isfinite, (energy, error, max_error))):
        raise ValueError("source range overflowed quantization diagnostics")
    result = BlockQuantization(scope, bits, block_size, codes, scales, {
        "format": FORMAT, "identity": scope.to_dict(), "identity_sha256": scope.fingerprint,
        "bits": bits, "block_size": block_size, "scale_dtype": "float32",
        "level_range": [-qmax, qmax], "padding_values": blocks * block_size - p,
        "factor_tensor_bytes": payload, "source_tensor_bytes": scope.actual_source_bytes,
        "tensor_storage_ratio": payload / scope.actual_source_bytes,
        "tensor_storage_is_smaller": payload < scope.actual_source_bytes,
        "relative_frobenius_error": math.sqrt(error / energy) if energy else 0.,
        "maximum_absolute_error": max_error,
        "estimated_numerical_peak_bytes": payload + workspace,
        "memory_bound_kind": "conservative_tensor_arrays_excluding_backend_allocator_overhead",
        "evaluation_scope": "fitted_weight_reconstruction_only",
        "semantic_quality_evaluated": False, "runtime_savings_evaluated": False,
        "method": METHOD,
    })
    _validate(result, expected_identity=scope)
    return result


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_block_quantization(value: BlockQuantization, directory, *, max_artifact_bytes=1 << 30):
    """Save a new immutable artifact, check its restricted-load round trip."""
    _positive(max_artifact_bytes, "max_artifact_bytes")
    if not isinstance(value, BlockQuantization):
        raise ValueError("invalid quantization object")
    _validate(value, expected_identity=value.identity)
    if value.metadata["factor_tensor_bytes"] > max_artifact_bytes:
        raise MemoryError("quantized payload exceeds export limit")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("choose a new artifact directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    intended_parent = directory.parent.resolve()
    staging_prefix = f".{directory.name}-"
    staging = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=intended_parent))
    try:
        tensor_path = staging / "factors.pt"
        torch.save({"codes": _standalone_storage(value.codes), "scales": _standalone_storage(value.scales)}, tensor_path)
        metadata = {**value.metadata, "factors_sha256": _sha(tensor_path),
                    "serialized_tensor_file_bytes": tensor_path.stat().st_size,
                    "metadata_json_bytes": 0, "serialized_total_bytes": 0,
                    "serialized_storage_ratio": 0., "serialized_is_smaller": False}
        for _ in range(16):
            encoded = _metadata_bytes(metadata)
            total = len(encoded) + tensor_path.stat().st_size
            if metadata["metadata_json_bytes"] == len(encoded) and metadata["serialized_total_bytes"] == total:
                break
            metadata.update(metadata_json_bytes=len(encoded), serialized_total_bytes=total,
                            serialized_storage_ratio=total/value.identity.actual_source_bytes,
                            serialized_is_smaller=total < value.identity.actual_source_bytes)
        else:
            raise ValueError("metadata accounting did not stabilize")
        if total > max_artifact_bytes:
            raise MemoryError("serialized artifact exceeds limit")
        (staging / "metadata.json").write_bytes(encoded)
        load_block_quantization(staging, expected_identity=value.identity, max_artifact_bytes=max_artifact_bytes)
        os.replace(staging, directory)
        return metadata
    except BaseException:
        _cleanup_staging(staging, intended_parent, staging_prefix)
        raise


def load_block_quantization(directory, *, expected_identity, max_artifact_bytes=1 << 30,
                            max_metadata_bytes=1 << 20):
    _positive(max_artifact_bytes, "max_artifact_bytes")
    _positive(max_metadata_bytes, "max_metadata_bytes")
    if not isinstance(expected_identity, ExpertGroupIdentity):
        raise ValueError("loading requires the expected source identity")
    directory = Path(directory)
    meta, tensor = directory / "metadata.json", directory / "factors.pt"
    metadata_size, tensor_size = meta.stat().st_size, tensor.stat().st_size
    if metadata_size > max_metadata_bytes or metadata_size + tensor_size > max_artifact_bytes:
        raise MemoryError("artifact files exceed limits")
    try:
        metadata = json.loads(meta.read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid quantization metadata JSON") from exc
    scope, bits, size, blocks, payload = _metadata_scope(metadata, expected_identity)
    if payload > max_artifact_bytes:
        raise MemoryError("declared quantization storage exceeds load limit")
    if _sha(tensor) != metadata.get("factors_sha256"):
        raise ValueError("factor checksum mismatch")
    actual = metadata_size + tensor_size
    if metadata.get("serialized_total_bytes") != actual or metadata.get("metadata_json_bytes") != metadata_size or metadata.get("serialized_tensor_file_bytes") != tensor_size:
        raise ValueError("invalid serialized byte accounting")
    if metadata.get("serialized_is_smaller") is not (actual < scope.actual_source_bytes) or _number(metadata.get("serialized_storage_ratio"), "serialized_storage_ratio") != actual / scope.actual_source_bytes:
        raise ValueError("invalid serialized savings ratio or claim")
    try:
        with zipfile.ZipFile(tensor) as archive:
            members = archive.infolist()
            storage_bytes = sum(item.file_size for item in members if "/data/" in item.filename)
            if any(item.compress_type != zipfile.ZIP_STORED for item in members) or storage_bytes != payload:
                raise ValueError("invalid or oversized quantized storage container")
            if sum(item.file_size for item in members) > max_artifact_bytes:
                raise MemoryError("expanded quantization archive exceeds load limit")
    except zipfile.BadZipFile as exc:
        raise ValueError("invalid Torch quantization container") from exc
    # Restricted tensor load; no application/model Python is unpickled.
    factors = torch.load(tensor, weights_only=True, map_location="cpu")
    if not isinstance(factors, dict) or set(factors) != {"codes", "scales"}:
        raise ValueError("invalid factor keys")
    result = BlockQuantization(scope, bits, size, factors["codes"], factors["scales"], metadata)
    _validate(result, expected_identity=expected_identity)
    return result
