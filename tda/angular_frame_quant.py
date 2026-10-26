"""Angular Frame (Phasor / Simplicial) Quantization for Neural Network Weights.

Represents pairs of weights (w_{2i}, w_{2i+1}) as 2D vector additions of angular
sub-components (phasors) along non-orthogonal directions:
    w = scale * (a * e_0 + b * e_{theta_1} + c * e_{theta_2})

where:
    e_0 = [1, 0] (0 degrees reference)
    e_{theta_1} = [cos(theta_1), sin(theta_1)] (default 67 degrees)
    e_{theta_2} = [cos(theta_2), sin(theta_2)] (default 35 degrees)

The non-orthogonal difference (67 - 35 = 32 deg vs 35 deg) acts as an angular
vernier scale, synthesizing a dense quasi-crystal of 2D directional states
through constructive and destructive vector interference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from tda.expert_compress import (
    ExpertGroupIdentity, ExpertMatrixReader, _cleanup_staging, _metadata_bytes, _read_chunk,
)


FORMAT = "angular-frame-quant-v1"
METHOD = "2D angular phasor frame decomposition (a + b@theta_1 + c@theta_2)"


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _build_angular_grid(angles_deg=(67.0, 35.0), levels=(-2.0, -1.0, 0.0, 1.0, 2.0)):
    """Precompute the 2D angular grid spanned by (a * e_0 + b * e_th1 + c * e_th2)."""
    th0 = 0.0
    th1 = math.radians(angles_deg[0])
    th2 = math.radians(angles_deg[1])

    # Basis vectors [2, 3]
    F_mat = torch.tensor([
        [math.cos(th0), math.cos(th1), math.cos(th2)],
        [math.sin(th0), math.sin(th1), math.sin(th2)]
    ], dtype=torch.float32)

    combos = []
    for a in levels:
        for b in levels:
            for c in levels:
                combos.append([a, b, c])

    coeffs = torch.tensor(combos, dtype=torch.float32)  # [N_states, 3]
    grid_2d = coeffs @ F_mat.T                         # [N_states, 2]
    return F_mat, coeffs, grid_2d


@dataclass
class AngularFrameQuantization:
    identity: ExpertGroupIdentity
    block_size: int
    angles_deg: tuple[float, float]
    codes: torch.Tensor    # [experts, blocks, block_size // 2] uint8
    scales: torch.Tensor   # [experts, blocks] float32
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

        # Precomputed grid
        _, _, grid_2d = _build_angular_grid(self.angles_deg)
        grid_2d = grid_2d.to(device=self.scales.device)

        # Slice packed codes: [len(ids), blocks_slice, pairs_per_block]
        pairs_per_block = self.block_size // 2
        code_slice = self.codes[rows, first:last].long()
        scale_slice = self.scales[rows, first:last, None, None]  # [experts, blocks_slice, 1, 1]

        # Lookup 2D points: [experts, blocks_slice, pairs_per_block, 2]
        decoded_pairs = grid_2d[code_slice] * scale_slice

        # Flatten pairs back into linear weights [experts, blocks_slice * block_size]
        flattened = decoded_pairs.view(len(ids), -1)
        offset = start - first * self.block_size
        return flattened[:, offset:offset + stop - start]


def fit_angular_frame_reader(reader: ExpertMatrixReader, *, block_size=32,
                             angles_deg=(67.0, 35.0), levels=(-2.0, -1.0, 0.0, 1.0, 2.0),
                             iterations=3, chunk_blocks=256, max_working_bytes=1 << 30):
    """Fit angular phasor frame codes and block scales across expert matrices.

    Pairs up weights into 2D vectors and maps each pair to the optimal
    angular combination:
        w_2i, w_{2i+1} ~ scale * (a * e_0 + b * e_{theta_1} + c * e_{theta_2})
    """
    scope = reader.identity
    if not isinstance(scope, ExpertGroupIdentity):
        raise ValueError("strict source identity required")
    _positive(block_size, "block_size")
    if block_size % 2 != 0:
        raise ValueError("block_size must be even for 2D pair decomposition")
    _positive(chunk_blocks, "chunk_blocks")
    _positive(max_working_bytes, "max_working_bytes")

    e, p = scope.expert_count, scope.column_count
    blocks = (p + block_size - 1) // block_size
    pairs_per_block = block_size // 2

    # Grid precomputation
    _, _, grid_2d = _build_angular_grid(angles_deg, levels)
    num_states = len(grid_2d)
    if num_states > 256:
        raise ValueError("number of states exceeds 256 for uint8 code storage")
    max_grid_norm = grid_2d.norm(dim=-1).max().clamp(min=1e-8)

    # Storage accounting: 1 byte per 2D pair (uint8) + 4 bytes scale per block (float32)
    payload = e * blocks * (pairs_per_block + 4)
    workspace = e * min(blocks, chunk_blocks) * block_size * 48
    if payload + workspace > max_working_bytes:
        raise MemoryError("angular frame arrays and workspace exceed budget")

    codes = torch.empty((e, blocks, pairs_per_block), dtype=torch.uint8)
    scales = torch.empty((e, blocks), dtype=torch.float32)

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
            # Reshape into 2D pairs: [e, last-first, pairs_per_block, 2]
            shaped_pairs = padded.view(e, last - first, pairs_per_block, 2)

            M = e * (last - first)
            pair_blocks = shaped_pairs.view(M, pairs_per_block, 2)  # [M, 16, 2]

            # Initial scale guess per block: [M, 1, 1]
            blk_max_norm = pair_blocks.norm(dim=-1).amax(dim=1, keepdim=True).clamp(min=1e-8).unsqueeze(-1)
            scale = blk_max_norm / max_grid_norm  # [M, 1, 1]

            best_idx = torch.zeros(M, pairs_per_block, dtype=torch.long)
            for _ in range(iterations):
                # 1. Assignment
                scaled_grid = scale * grid_2d.unsqueeze(0)  # [M, num_states, 2]
                # dists: [M, pairs, states]
                dists = (pair_blocks.unsqueeze(2) - scaled_grid.unsqueeze(1)).pow(2).sum(dim=-1)
                best_idx = dists.argmin(dim=-1)  # [M, pairs]

                # 2. Optimal scale update: argmin_s ||pair_blocks - s * chosen_grid||^2
                chosen_grid = grid_2d[best_idx]  # [M, pairs, 2]
                denom = chosen_grid.pow(2).sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)  # [M, 1, 1]
                scale = (pair_blocks * chosen_grid).sum(dim=(1, 2), keepdim=True) / denom
                scale = scale.clamp(min=1e-8)

            chosen_grid = grid_2d[best_idx]
            recon = scale * chosen_grid
            err = recon - pair_blocks

            # Account for padding in last block
            valid_mask = torch.ones_like(pair_blocks, dtype=torch.bool)
            if last == blocks and (stop - start) % block_size != 0:
                pad_weights = block_size - ((stop - start) % block_size)
                pad_pairs = pad_weights // 2
                mask_reshaped = valid_mask.view(e, last - first, pairs_per_block, 2)
                mask_reshaped[:, -1, -pad_pairs:, :] = False

            v_orig = pair_blocks[valid_mask]
            v_err = err[valid_mask]

            total_energy += float(v_orig.pow(2).sum())
            total_error += float(v_err.pow(2).sum())
            max_error = max(max_error, float(v_err.abs().max()))

            codes[:, first:last] = best_idx.view(e, last - first, pairs_per_block).to(torch.uint8)
            scales[:, first:last] = scale.squeeze(-1).view(e, last - first)

    rel_frob = math.sqrt(total_error / total_energy) if total_energy > 0 else 0.0

    metadata = {
        "format": FORMAT,
        "method": METHOD,
        "identity": scope.to_dict(),
        "identity_sha256": scope.fingerprint,
        "block_size": block_size,
        "angles_deg": list(angles_deg),
        "levels": list(levels),
        "num_angular_states": num_states,
        "bits_per_weight": 4.0,  # 8 bits per 2D pair
        "factor_tensor_bytes": payload,
        "source_tensor_bytes": scope.actual_source_bytes,
        "tensor_storage_ratio": payload / scope.actual_source_bytes,
        "relative_frobenius_error": rel_frob,
        "maximum_absolute_error": max_error,
        "padding_values": blocks * block_size - scope.column_count,
    }

    return AngularFrameQuantization(
        identity=scope,
        block_size=block_size,
        angles_deg=tuple(angles_deg),
        codes=codes,
        scales=scales,
        metadata=metadata,
    )


def save_angular_frame_quantization(quant: AngularFrameQuantization, output_path: str | Path):
    """Atomically save angular frame quantization factors and metadata."""
    out_dir = Path(output_path)
    if out_dir.exists():
        raise FileExistsError(f"output directory already exists: {out_dir}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="angular_frame_", dir=out_dir.parent))
    try:
        factors = {
            "codes": quant.codes,
            "scales": quant.scales,
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

        # Write final metadata
        meta_file.write_bytes(_metadata_bytes(meta))

        os.rename(tmp_dir, out_dir)
        return meta
    except Exception:
        _cleanup_staging(tmp_dir)
        raise


def load_angular_frame_quantization(path: str | Path, *, expected_identity: ExpertGroupIdentity):
    """Load and verify a saved angular frame quantization artifact."""
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
    return AngularFrameQuantization(
        identity=scope,
        block_size=meta["block_size"],
        angles_deg=tuple(meta["angles_deg"]),
        codes=data["codes"],
        scales=data["scales"],
        metadata=meta,
    )
