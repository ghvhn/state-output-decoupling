"""Lossless seven-bit storage for the existing angular quantizer's indices.

This changes representation, not code selection, grid points or FP32 scales.
Each row/block has its own least-significant-bit-first stream; unused high
bits are zero. The default 125 states fit in seven bits per two-weight pair.
Block32 therefore uses 14 index bytes + 4 scale bytes (4.5 bits/weight), before
shared configuration, padding and actual container/metadata overhead.
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

from .angular_frame_quant import AngularFrameQuantization, _build_angular_grid
from .expert_compress import (ExpertGroupIdentity, _cleanup_staging,
                              _metadata_bytes, _standalone_storage)


FORMAT = "angular-frame-packed7-v1"
DEFAULT_LEVELS = (-2.0, -1.0, 0.0, 1.0, 2.0)
_PACKING = "lsb-first-seven-bit-rows-zero-high-padding"


def _integer(value, name, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, int) or value < minimum or
            maximum is not None and value > maximum):
        raise ValueError(f"invalid bounded integer {name}")
    return value


def _options(num_states, chunk_codes, max_working_bytes):
    _integer(num_states, "num_states", 1, 128)
    _integer(chunk_codes, "chunk_codes", 1, 1 << 20)
    _integer(max_working_bytes, "max_working_bytes", 1)


def _byte_tensor(value, name):
    if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
            value.dtype != torch.uint8 or not 1 <= value.ndim <= 8 or
            any(size < 1 for size in value.shape[:-1])):
        raise ValueError(f"{name} requires a bounded-rank CPU uint8 tensor")


def _budget(value, output_bytes, count, chunk_codes, maximum):
    # Include backing storage, a possible reshape copy, result, and conservative
    # integer-index/bit-operation temporaries for one bounded logical chunk.
    estimate = (value.untyped_storage().nbytes() + value.numel() + output_bytes +
                min(count, chunk_codes) * 256)
    if estimate > maximum:
        raise MemoryError("seven-bit arrays and chunk workspace exceed budget")
    return estimate


def _packed_shape(packed, count):
    _byte_tensor(packed, "packed indices")
    _integer(count, "count")
    width = (count * 7 + 7) // 8
    if packed.shape[-1] != width:
        raise ValueError("packed byte shape differs from the declared exact code count")
    return width


def _padding(packed, count, chunk_codes):
    width = _packed_shape(packed, count)
    bits = count * 7 % 8
    if bits:
        last = packed.reshape(-1, width)[:, -1]
        for start in range(0, len(last), chunk_codes):
            if bool((last[start:start + chunk_codes] >> bits).any()):
                raise ValueError("packed indices have noncanonical high padding bits")


def _decode_chunk(flat, count, width, start, stop):
    indices = torch.arange(start, stop, dtype=torch.long)
    position = indices.remainder(count)
    bits = position * 7
    byte = indices.div(count, rounding_mode="floor") * width + bits.div(8, rounding_mode="floor")
    shift = bits.remainder(8)
    low = flat[byte].long() >> shift
    has_next = bits.div(8, rounding_mode="floor") + 1 < width
    high = flat[(byte + 1).clamp(max=flat.numel() - 1)].long() * has_next
    return ((low | (high << (8 - shift))) & 127).to(torch.uint8)


def pack_7bit_codes(codes, *, num_states=125, chunk_codes=65536,
                    max_working_bytes=1 << 30):
    """Pack each tensor row's last axis; no logical indices are discarded.

    CPU uint8 input is required. The original last-axis count must accompany
    packed bytes: bytes alone cannot distinguish trailing zero-valued codes
    from padding. This count is explicit in unpacking and the reader metadata.
    """
    _byte_tensor(codes, "angular indices")
    _options(num_states, chunk_codes, max_working_bytes)
    count = codes.shape[-1]
    width = (count * 7 + 7) // 8
    rows = math.prod(codes.shape[:-1])
    _budget(codes, rows * width, codes.numel(), chunk_codes, max_working_bytes)
    flat = codes.reshape(-1)
    packed = torch.zeros(rows * width, dtype=torch.uint8)
    for start in range(0, flat.numel(), chunk_codes):
        stop = min(start + chunk_codes, flat.numel())
        values = flat[start:stop].long()
        if bool((values >= num_states).any()):
            raise ValueError("angular index exceeds the declared state alphabet")
        indices = torch.arange(start, stop, dtype=torch.long)
        bits = indices.remainder(count) * 7
        byte = indices.div(count, rounding_mode="floor") * width + bits.div(8, rounding_mode="floor")
        shifted = values << bits.remainder(8)
        # Bits of adjacent codes occupy disjoint positions: summation here is
        # equivalent to bitwise OR, including when a byte spans chunk edges.
        packed.scatter_add_(0, byte, (shifted & 255).to(torch.uint8))
        valid = bits.div(8, rounding_mode="floor") + 1 < width
        packed.scatter_add_(0, byte[valid] + 1, (shifted[valid] >> 8).to(torch.uint8))
    return packed.reshape(*codes.shape[:-1], width)


def unpack_7bit_codes(packed, *, count, num_states=125, expected_shape=None,
                      chunk_codes=65536, max_working_bytes=1 << 30):
    """Decode the declared count, rejecting extra bytes and nonzero high pad.

    ``expected_shape``, when supplied, includes the unpacked final-axis count.
    A custom alphabet may contain 1..128 states; default indices 125..127 fail.
    """
    _options(num_states, chunk_codes, max_working_bytes)
    width = _packed_shape(packed, count)
    shape = (*packed.shape[:-1], count)
    if expected_shape is not None and tuple(expected_shape) != shape:
        raise ValueError("unpacked exact shape differs from its declaration")
    total = math.prod(shape)
    _budget(packed, total, total, chunk_codes, max_working_bytes)
    _padding(packed, count, chunk_codes)
    flat = packed.reshape(-1)
    result = torch.empty(total, dtype=torch.uint8)
    for start in range(0, total, chunk_codes):
        stop = min(start + chunk_codes, total)
        values = _decode_chunk(flat, count, width, start, stop)
        if bool((values >= num_states).any()):
            raise ValueError("packed angular index exceeds the declared state alphabet")
        result[start:stop] = values
    return result.reshape(shape)


def _configuration(block_size, angles, levels):
    _integer(block_size, "block_size", 2, 65536)
    if block_size % 2:
        raise ValueError("angular block_size must be even")
    if not isinstance(angles, (list, tuple)) or len(angles) != 2:
        raise ValueError("exactly two angular directions are required")
    if not isinstance(levels, (list, tuple)) or not 1 <= len(levels) <= 5:
        raise ValueError("declared levels must generate at most 128 angular states")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
           for x in (*angles, *levels)):
        raise ValueError("angles and levels must be finite numbers")
    angles, levels = tuple(float(x) for x in angles), tuple(float(x) for x in levels)
    _, _, grid = _build_angular_grid(angles, levels)
    if len(grid) > 128 or not bool(torch.isfinite(grid).all()):
        raise ValueError("declared angular grid is oversized or nonfinite")
    return {"block_size": block_size, "angles_deg": list(angles), "levels": list(levels),
            "num_angular_states": len(grid), "bits_per_pair": 7, "packing": _PACKING}


def _config_bytes(config):
    return json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _finite_scales(scales):
    # Do not allocate a Boolean mask over the entire native scale table.
    for row in scales:
        for start in range(0, len(row), 65536):
            if not bool(torch.isfinite(row[start:start + 65536]).all()):
                raise ValueError("packed angular scales must be finite FP32 CPU data")


def _metadata(identity, config):
    blocks = (identity.column_count + config["block_size"] - 1) // config["block_size"]
    pairs = config["block_size"] // 2
    width = (pairs * 7 + 7) // 8
    index_bytes = identity.expert_count * blocks * width
    scale_bytes = identity.expert_count * blocks * 4
    configuration_bytes = len(_config_bytes(config))
    payload = index_bytes + scale_bytes
    return {"format": FORMAT, "identity": identity.to_dict(), "identity_sha256": identity.fingerprint,
            "configuration": config, "code_shape": [identity.expert_count, blocks, pairs],
            "packed_code_shape": [identity.expert_count, blocks, width],
            "index_code_count": identity.expert_count * blocks * pairs,
            "packed_index_bytes": index_bytes, "scale_tensor_bytes": scale_bytes,
            "scale_dtype": "float32", "configuration_json_bytes": configuration_bytes,
            "factor_tensor_bytes": payload, "tensor_and_configuration_bytes": payload + configuration_bytes,
            "padding_values": blocks * config["block_size"] - identity.column_count,
            "unused_high_bits_per_block": (8 - pairs * 7 % 8) % 8,
            "full_block_bits_per_weight_excluding_shared_configuration": 8 * (width + 4) / config["block_size"],
            "effective_bits_per_source_weight_including_configuration":
                8 * (payload + configuration_bytes) / (identity.expert_count * identity.column_count),
            "source_tensor_bytes": identity.actual_source_bytes,
            "lossless_index_repacking": True, "code_selection_and_fp32_scales_changed": False,
            "evaluation_scope": "stored angular-grid reconstruction only",
            "model_behavior_evaluated": False, "runtime_savings_evaluated": False}


def _validate(value, expected_identity, *, check_codes=False):
    if not isinstance(expected_identity, ExpertGroupIdentity) or value.identity != expected_identity:
        raise ValueError("packed angular source identity differs")
    config = _configuration(value.block_size, value.angles_deg, value.levels)
    wanted = _metadata(expected_identity, config)
    if any(value.metadata.get(key) != item for key, item in wanted.items()):
        raise ValueError("packed angular metadata differs from exact scope or configuration")
    _byte_tensor(value.packed_codes, "packed angular indices")
    if tuple(value.packed_codes.shape) != tuple(wanted["packed_code_shape"]):
        raise ValueError("packed angular code shape differs")
    if (not isinstance(value.scales, torch.Tensor) or value.scales.device.type != "cpu" or
            value.scales.dtype != torch.float32 or tuple(value.scales.shape) != tuple(wanted["code_shape"][:2])):
        raise ValueError("packed angular scales must retain exact finite FP32 CPU shape")
    _integer(value.max_working_bytes, "max_working_bytes", 1)
    _integer(value.max_read_bytes, "max_read_bytes", 1)
    storage = value.packed_codes.untyped_storage().nbytes() + value.scales.untyped_storage().nbytes()
    if storage + wanted["configuration_json_bytes"] + min(value.scales.numel(), 65536) > value.max_working_bytes:
        raise MemoryError("packed angular persistent arrays exceed budget")
    _finite_scales(value.scales)
    if check_codes:
        pairs, width = config["block_size"] // 2, value.packed_codes.shape[-1]
        total = wanted["index_code_count"]
        _budget(value.packed_codes, 0, total, 65536,
                value.max_working_bytes - value.scales.untyped_storage().nbytes() - wanted["configuration_json_bytes"])
        _padding(value.packed_codes, pairs, 65536)
        flat = value.packed_codes.reshape(-1)
        for start in range(0, total, 65536):
            decoded = _decode_chunk(flat, pairs, width, start, min(total, start + 65536))
            if bool((decoded >= config["num_angular_states"]).any()):
                raise ValueError("packed angular index exceeds its stored configuration")
    return wanted


@dataclass
class PackedAngularFrameQuantization:
    identity: ExpertGroupIdentity
    block_size: int
    angles_deg: tuple[float, float]
    levels: tuple[float, ...]
    packed_codes: torch.Tensor
    scales: torch.Tensor
    metadata: dict
    max_working_bytes: int = 1 << 30
    max_read_bytes: int = 128 << 20

    def read_columns(self, start, stop, expert_ids=None):
        _integer(start, "start")
        _integer(stop, "stop", 1)
        if not 0 <= start < stop <= self.identity.column_count:
            raise ValueError("invalid packed angular column interval")
        ids = self.identity.expert_ids if expert_ids is None else tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(x not in self.identity.expert_ids for x in ids):
            raise ValueError("unknown or repeated packed angular expert ID")
        info = _validate(self, self.identity)
        first, last = start // self.block_size, (stop + self.block_size - 1) // self.block_size
        pairs = self.block_size // 2
        selected_codes = len(ids) * (last - first) * pairs
        # Full block outputs, byte/index slices and decoding intermediates are
        # bounded before allocating any expert/block selection.
        estimate = (selected_codes * 32 + min(selected_codes, 65536) * 256 +
                    self.packed_codes.shape[-1] * len(ids) * (last - first) * 2 +
                    len(ids) * (last - first) * 4 + 128 * 5 * 4)
        storage = self.packed_codes.untyped_storage().nbytes() + self.scales.untyped_storage().nbytes()
        if estimate > self.max_read_bytes or storage + info["configuration_json_bytes"] + estimate > self.max_working_bytes:
            raise MemoryError("packed angular column decode exceeds its bounded workspace")
        rows = [self.identity.expert_ids.index(x) for x in ids]
        packed = self.packed_codes[rows, first:last]
        codes = unpack_7bit_codes(packed, count=pairs,
                                  num_states=len(self.levels) ** 3,
                                  max_working_bytes=self.max_read_bytes).long()
        _, _, grid = _build_angular_grid(self.angles_deg, self.levels)
        scales = self.scales[rows, first:last, None, None]
        restored = (grid[codes] * scales).reshape(len(ids), -1)
        if not bool(torch.isfinite(restored).all()):
            raise ValueError("packed angular decoding overflowed FP32")
        offset = start - first * self.block_size
        return restored[:, offset:offset + stop - start]


def pack_angular_frame_quantization(quant, *, expected_identity, max_working_bytes=1 << 30,
                                   max_read_bytes=128 << 20):
    """Repack a fitted prototype without fitting, changing indices or scales."""
    if not isinstance(quant, AngularFrameQuantization) or quant.identity != expected_identity:
        raise ValueError("packing requires the exact fitted angular source identity")
    if not isinstance(expected_identity, ExpertGroupIdentity):
        raise ValueError("packing requires an ExpertGroupIdentity")
    _integer(max_working_bytes, "max_working_bytes", 1)
    _integer(max_read_bytes, "max_read_bytes", 1)
    config = _configuration(quant.block_size, quant.angles_deg, quant.metadata.get("levels"))
    info = _metadata(expected_identity, config)
    if (quant.metadata.get("identity") != expected_identity.to_dict() or
            quant.metadata.get("identity_sha256") != expected_identity.fingerprint or
            quant.metadata.get("block_size") != quant.block_size or
            quant.metadata.get("angles_deg") != config["angles_deg"] or
            quant.metadata.get("num_angular_states") != config["num_angular_states"]):
        raise ValueError("fitted angular metadata differs from the declared configuration/source")
    _byte_tensor(quant.codes, "fitted angular indices")
    if tuple(quant.codes.shape) != tuple(info["code_shape"]):
        raise ValueError("fitted angular code shape differs from its source")
    if (not isinstance(quant.scales, torch.Tensor) or quant.scales.device.type != "cpu" or
            quant.scales.dtype != torch.float32 or tuple(quant.scales.shape) != tuple(info["code_shape"][:2])):
        raise ValueError("fitted angular scales must be exact finite FP32 CPU data")
    scale_bytes = quant.scales.untyped_storage().nbytes() + quant.scales.numel() * 4
    if scale_bytes + min(quant.scales.numel(), 65536) >= max_working_bytes:
        raise MemoryError("FP32 scale copies exceed packing budget")
    _finite_scales(quant.scales)
    packed = pack_7bit_codes(quant.codes, num_states=config["num_angular_states"],
                             max_working_bytes=max_working_bytes - scale_bytes)
    result = PackedAngularFrameQuantization(expected_identity, quant.block_size,
        tuple(config["angles_deg"]), tuple(config["levels"]), packed,
        quant.scales.detach().clone(memory_format=torch.contiguous_format), info, max_working_bytes, max_read_bytes)
    _validate(result, expected_identity)
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_packed_angular_frame(value, directory, *, max_artifact_bytes=1 << 30):
    """Save a new atomic artifact; count final files, including configuration."""
    _integer(max_artifact_bytes, "max_artifact_bytes", 1)
    if not isinstance(value, PackedAngularFrameQuantization):
        raise ValueError("invalid packed angular representation")
    info = _validate(value, value.identity, check_codes=True)
    if info["tensor_and_configuration_bytes"] > max_artifact_bytes:
        raise MemoryError("packed angular payload exceeds artifact bound")
    storage = value.packed_codes.untyped_storage().nbytes() + value.scales.untyped_storage().nbytes()
    if (storage + info["factor_tensor_bytes"] + min(info["index_code_count"], 65536) * 256 +
            info["configuration_json_bytes"] > value.max_working_bytes):
        raise MemoryError("packed angular save/round-trip workspace exceeds bound")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError("choose a new packed angular artifact directory")
    directory.parent.mkdir(parents=True, exist_ok=True)
    parent, prefix = directory.parent.resolve(), f".{directory.name}-"
    staging = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    try:
        tensor_path, config_path = staging / "factors.pt", staging / "configuration.json"
        torch.save({"packed_codes": _standalone_storage(value.packed_codes),
                    "scales": _standalone_storage(value.scales)}, tensor_path)
        config_path.write_bytes(_config_bytes(info["configuration"]))
        metadata = {**info, "factors_sha256": _sha(tensor_path), "configuration_sha256": _sha(config_path),
                    "serialized_tensor_file_bytes": tensor_path.stat().st_size,
                    "metadata_json_bytes": 0, "serialized_total_bytes": 0}
        for _ in range(16):
            encoded = _metadata_bytes(metadata)
            total = len(encoded) + tensor_path.stat().st_size + config_path.stat().st_size
            if metadata["metadata_json_bytes"] == len(encoded) and metadata["serialized_total_bytes"] == total:
                break
            metadata.update(metadata_json_bytes=len(encoded), serialized_total_bytes=total)
        else:
            raise ValueError("packed angular metadata byte accounting did not stabilize")
        if total > max_artifact_bytes:
            raise MemoryError("serialized packed angular artifact exceeds bound")
        (staging / "metadata.json").write_bytes(encoded)
        loaded = load_packed_angular_frame(staging, expected_identity=value.identity,
            max_artifact_bytes=max_artifact_bytes, max_working_bytes=value.max_working_bytes,
            max_read_bytes=value.max_read_bytes)
        if not torch.equal(loaded.packed_codes, value.packed_codes) or not torch.equal(loaded.scales, value.scales):
            raise ValueError("packed angular restricted round trip changed stored factors")
        os.rename(staging, directory)
        return metadata
    except BaseException:
        _cleanup_staging(staging, parent, prefix)
        raise


def load_packed_angular_frame(directory, *, expected_identity, max_artifact_bytes=1 << 30,
                             max_working_bytes=1 << 30, max_read_bytes=128 << 20):
    """Load bounded uncompressed tensors, verifying exact source and checksums."""
    for value, name in ((max_artifact_bytes, "max_artifact_bytes"),
                        (max_working_bytes, "max_working_bytes"), (max_read_bytes, "max_read_bytes")):
        _integer(value, name, 1)
    if not isinstance(expected_identity, ExpertGroupIdentity):
        raise ValueError("loading requires an exact source identity")
    directory = Path(directory)
    paths = {name: directory / name for name in ("metadata.json", "configuration.json", "factors.pt")}
    if set(path.name for path in directory.iterdir()) != set(paths):
        raise ValueError("packed angular artifact contains undeclared files")
    sizes = {name: path.stat().st_size for name, path in paths.items()}
    if sizes["metadata.json"] > 1 << 20 or sizes["configuration.json"] > 4096 or sum(sizes.values()) > max_artifact_bytes:
        raise MemoryError("packed angular files exceed declared load bounds")
    if 16 * (sizes["metadata.json"] + sizes["configuration.json"]) > max_working_bytes:
        raise MemoryError("packed angular metadata workspace exceeds bound")
    metadata = json.loads(paths["metadata.json"].read_bytes())
    config = json.loads(paths["configuration.json"].read_bytes())
    if not isinstance(config, dict) or set(config) != {"block_size", "angles_deg", "levels", "num_angular_states", "bits_per_pair", "packing"}:
        raise ValueError("invalid exact angular configuration")
    checked = _configuration(config["block_size"], config["angles_deg"], config["levels"])
    info = _metadata(expected_identity, checked)
    if (config != checked or not isinstance(metadata, dict) or
            any(metadata.get(key) != value for key, value in info.items())):
        raise ValueError("packed angular source identity or configuration differs")
    if (metadata.get("metadata_json_bytes") != sizes["metadata.json"] or
            metadata.get("serialized_tensor_file_bytes") != sizes["factors.pt"] or
            metadata.get("serialized_total_bytes") != sum(sizes.values()) or
            sizes["configuration.json"] != info["configuration_json_bytes"]):
        raise ValueError("packed angular actual file byte accounting differs")
    if (_sha(paths["factors.pt"]) != metadata.get("factors_sha256") or
            _sha(paths["configuration.json"]) != metadata.get("configuration_sha256")):
        raise ValueError("packed angular artifact checksum mismatch")
    payload = info["factor_tensor_bytes"]
    if payload * 2 + min(info["index_code_count"], 65536) * 256 > max_working_bytes:
        raise MemoryError("packed angular load validation workspace exceeds bound")
    with zipfile.ZipFile(paths["factors.pt"]) as archive:
        members = archive.infolist()
        control_bytes = sum(member.file_size for member in members if "/data/" not in member.filename)
        if (any(member.compress_type != zipfile.ZIP_STORED for member in members) or
                sum(member.file_size for member in members if "/data/" in member.filename) != payload or
                sum(member.file_size for member in members) > max_artifact_bytes or control_bytes > 1 << 20):
            raise ValueError("packed angular tensor container has invalid storage")
        if 2 * payload + min(info["index_code_count"], 65536) * 256 + 64 * control_bytes > max_working_bytes:
            raise MemoryError("packed angular tensor control workspace exceeds bound")
    factors = torch.load(paths["factors.pt"], map_location="cpu", weights_only=True)
    if not isinstance(factors, dict) or set(factors) != {"packed_codes", "scales"}:
        raise ValueError("packed angular tensor fields differ")
    result = PackedAngularFrameQuantization(expected_identity, checked["block_size"],
        tuple(checked["angles_deg"]), tuple(checked["levels"]), factors["packed_codes"], factors["scales"],
        metadata, max_working_bytes, max_read_bytes)
    _validate(result, expected_identity, check_codes=True)
    return result
