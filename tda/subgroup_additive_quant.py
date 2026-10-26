"""Subgroup Additive Subvalue / Simplicial Quantization for neural network weights.

Represents each tensor value as a sum of discrete subvalues (atoms) selected
from a shared subgroup alphabet plus an offset:
    w_i = offset + sum_{k=1}^K b_{i,k} * s_k

where b_{i,k} in {0, 1} is a binary indicator of whether subvalue k is used.
For K=4, each weight uses a 4-bit packed code (16 additive levels), matching
the bit-depth of INT4 while adapting the non-uniform discrete basis to the
underlying weight distribution within each subgroup.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
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


FORMAT = "subgroup-additive-quant-v1"
METHOD = "subgroup additive subvalue decomposition (simplicial/multi-component basis)"

_COMBINATIONS_4 = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)), dtype=torch.float32)


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass
class SubgroupAdditiveQuantization:
    identity: ExpertGroupIdentity
    block_size: int
    num_subvalues: int
    codes: torch.Tensor        # [experts, blocks, block_size // 2] uint8 (packed nibbles for K=4)
    offsets: torch.Tensor      # [experts, blocks] float32
    subvalues: torch.Tensor    # [experts, blocks, num_subvalues] float32
    metadata: dict

    def read_columns(self, start, stop, expert_ids=None):
        _positive(stop, "stop")
        if isinstance(start, bool) or not isinstance(start, int) or not 0 <= start < stop <= self.identity.column_count:
            raise ValueError("invalid column interval")
        ids = self.identity.expert_ids if expert_ids is None else tuple(expert_ids)
        if not ids or len(set(ids)) != len(ids) or any(x not in self.identity.expert_ids for x in ids):
            raise ValueError("unknown or repeated matrix slot")
        first = start // self.block_size
        last = (stop + self.block_size - 1) // self.block_size
        rows = [self.identity.expert_ids.index(x) for x in ids]

        # Extract packed nibbles
        packed = self.codes[rows, first:last]  # [experts, blocks_slice, block_size // 2]
        unpacked = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()  # [experts, blocks_slice, block_size]

        # Decode: offset + combos[unpacked] @ subvalues
        combos = _COMBINATIONS_4.to(device=self.subvalues.device)  # [16, 4]
        # combos[unpacked] shape: [experts, blocks_slice, block_size, 4]
        active_subvalues = combos[unpacked]
        subvals = self.subvalues[rows, first:last, None, :]  # [experts, blocks_slice, 1, 4]
        offs = self.offsets[rows, first:last, None]  # [experts, blocks_slice, 1]

        restored = offs + (active_subvalues * subvals).sum(dim=-1)  # [experts, blocks_slice, block_size]
        flattened = restored.flatten(1)
        offset = start - first * self.block_size
        return flattened[:, offset:offset + stop - start]


def fit_subgroup_additive_reader(reader: ExpertMatrixReader, *, block_size=32,
                                num_subvalues=4, iterations=4, chunk_blocks=256,
                                max_working_bytes=1 << 30):
    """Fit additive subvalues and packed codes for each subgroup in the matrix.

    For each subgroup of size B:
        w_i = offset + sum_{k=1}^4 b_{i,k} * s_k
    where b_i in {0, 1}^4 represents the combination of active subvalues.
    """
    scope = reader.identity
    if not isinstance(scope, ExpertGroupIdentity):
        raise ValueError("strict source identity required")
    _positive(block_size, "block_size")
    if block_size % 2 != 0:
        raise ValueError("block_size must be even")
    if num_subvalues != 4:
        raise ValueError("currently 4 subvalues (16 additive levels / 4-bit) supported")
    _positive(chunk_blocks, "chunk_blocks")
    _positive(max_working_bytes, "max_working_bytes")

    e, p = scope.expert_count, scope.column_count
    blocks = (p + block_size - 1) // block_size
    code_width = block_size // 2  # packed 4-bit nibbles

    # Factor storage:
    # codes: e * blocks * code_width (uint8)
    # offsets: e * blocks * 4 (float32)
    # subvalues: e * blocks * num_subvalues * 4 (float32)
    payload = e * blocks * (code_width + 4 + num_subvalues * 4)
    workspace = e * min(blocks, chunk_blocks) * block_size * 64
    if payload + workspace > max_working_bytes:
        raise MemoryError("subgroup additive arrays and workspace exceed budget")

    codes = torch.empty((e, blocks, code_width), dtype=torch.uint8)
    offsets = torch.empty((e, blocks), dtype=torch.float32)
    subvalues = torch.empty((e, blocks, num_subvalues), dtype=torch.float32)

    combos = _COMBINATIONS_4  # [16, 4]
    total_energy = 0.0
    total_error = 0.0
    max_error = 0.0

    with torch.no_grad():
        for first in range(0, blocks, chunk_blocks):
            last = min(blocks, first + chunk_blocks)
            start, stop = first * block_size, min(p, last * block_size)
            values = _read_chunk(reader, scope, start, stop)
            if reader.identity != scope:
                raise ValueError("reader source scope changed during read")

            padded = torch.zeros((e, (last - first) * block_size), dtype=torch.float32)
            padded[:, :stop - start] = values
            shaped = padded.view(e, last - first, block_size)

            M = e * (last - first)
            x = shaped.view(M, block_size)

            x_min = x.min(dim=-1, keepdim=True).values
            x_max = x.max(dim=-1, keepdim=True).values
            span = (x_max - x_min).clamp(min=1e-8)
            s = span * torch.tensor([0.5, 0.25, 0.125, 0.0625], dtype=x.dtype, device=x.device)
            mu = x_min

            best_idx = torch.zeros(M, block_size, dtype=torch.long)
            for _ in range(iterations):
                levels = mu + s @ combos.T  # [M, 16]
                dists = (x.unsqueeze(-1) - levels.unsqueeze(1)).pow(2)  # [M, B, 16]
                best_idx = dists.argmin(dim=-1)  # [M, B]

                A = torch.cat([combos[best_idx], torch.ones(M, block_size, 1, dtype=x.dtype, device=x.device)], dim=-1)
                sol = torch.linalg.lstsq(A, x.unsqueeze(-1)).solution.squeeze(-1)
                s = sol[:, :4].clamp(min=0)
                mu = sol[:, 4:5]

            levels = mu + s @ combos.T
            recon = levels.gather(1, best_idx)
            err = recon - x

            # Account for padding in last block if needed
            valid_mask = torch.ones_like(x, dtype=torch.bool)
            if last == blocks and (stop - start) % block_size != 0:
                pad_len = block_size - ((stop - start) % block_size)
                # Reshape mask to [e, last-first, block_size]
                mask_reshaped = valid_mask.view(e, last - first, block_size)
                mask_reshaped[:, -1, -pad_len:] = False

            x_valid = x[valid_mask]
            err_valid = err[valid_mask]

            total_energy += float(x_valid.pow(2).sum())
            total_error += float(err_valid.pow(2).sum())
            max_error = max(max_error, float(err_valid.abs().max()))

            low = best_idx[:, 0::2] & 15
            high = (best_idx[:, 1::2] & 15) << 4
            packed_blk = (low | high).to(torch.uint8)

            codes[:, first:last] = packed_blk.view(e, last - first, code_width)
            offsets[:, first:last] = mu.squeeze(-1).view(e, last - first)
            subvalues[:, first:last] = s.view(e, last - first, num_subvalues)

    rel_frob = math.sqrt(total_error / total_energy) if total_energy > 0 else 0.0

    metadata = {
        "format": FORMAT,
        "method": METHOD,
        "identity": scope.to_dict(),
        "identity_sha256": scope.fingerprint,
        "block_size": block_size,
        "num_subvalues": num_subvalues,
        "bits_per_code": 4,
        "factor_tensor_bytes": payload,
        "source_tensor_bytes": scope.actual_source_bytes,
        "tensor_storage_ratio": payload / scope.actual_source_bytes,
        "relative_frobenius_error": rel_frob,
        "maximum_absolute_error": max_error,
        "padding_values": blocks * block_size - scope.column_count,
    }

    return SubgroupAdditiveQuantization(
        identity=scope,
        block_size=block_size,
        num_subvalues=num_subvalues,
        codes=codes,
        offsets=offsets,
        subvalues=subvalues,
        metadata=metadata,
    )


def save_subgroup_additive_quantization(quant: SubgroupAdditiveQuantization, output_path: str | Path):
    """Atomically save subgroup additive quantization factors and metadata."""
    out_dir = Path(output_path)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="subgroup_additive_", dir=out_dir.parent))
    try:
        factors = {
            "codes": quant.codes,
            "offsets": quant.offsets,
            "subvalues": quant.subvalues,
        }
        factors_file = tmp_dir / "factors.pt"
        torch.save(factors, factors_file)

        factors_sha256 = hashlib.sha256(factors_file.read_bytes()).hexdigest()
        meta = dict(quant.metadata)
        meta["factors_sha256"] = factors_sha256

        meta_bytes = _metadata_bytes(meta)
        meta_file = tmp_dir / "metadata.json"
        meta_file.write_bytes(meta_bytes)
        meta["metadata_json_bytes"] = len(meta_bytes)
        meta["serialized_total_bytes"] = factors_file.stat().st_size + len(meta_bytes)

        # Write final metadata with byte counts
        meta_file.write_bytes(_metadata_bytes(meta))

        os.rename(tmp_dir, out_dir)
        return meta
    except Exception:
        _cleanup_staging(tmp_dir)
        raise


def load_subgroup_additive_quantization(path: str | Path, *, expected_identity: ExpertGroupIdentity):
    """Load and verify a saved subgroup additive quantization artifact."""
    in_dir = Path(path)
    if not in_dir.is_dir():
        raise FileNotFoundError(f"directory not found: {in_dir}")

    meta_file = in_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"missing metadata.json in {in_dir}")

    meta = json.loads(meta_file.read_bytes())
    if meta.get("format") != FORMAT:
        raise ValueError(f"unexpected format: {meta.get('format')}")

    scope = ExpertGroupIdentity.from_dict(meta.get("identity"))
    if scope != expected_identity:
        raise ValueError("source scope mismatch")

    factors_file = in_dir / "factors.pt"
    if not factors_file.exists():
        raise FileNotFoundError(f"missing factors.pt in {in_dir}")

    actual_hash = hashlib.sha256(factors_file.read_bytes()).hexdigest()
    if actual_hash != meta.get("factors_sha256"):
        raise ValueError("factors.pt checksum mismatch")

    data = torch.load(factors_file, weights_only=True)
    return SubgroupAdditiveQuantization(
        identity=scope,
        block_size=meta["block_size"],
        num_subvalues=meta["num_subvalues"],
        codes=data["codes"],
        offsets=data["offsets"],
        subvalues=data["subvalues"],
        metadata=meta,
    )
