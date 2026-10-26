"""CPU storage/reconstruction trial on a previously downloaded bounded GLM sample.

No model is loaded. Random input checks are numerical checks, not measured
language states. A gate projection alone cannot evaluate a nonlinear MoE block
or a complete decoder. Never extrapolate this sample to the whole checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def source_linear(reader, inputs, row_chunk=16):
    """Decode only complete bounded row blocks, preserving matrix axis order."""
    import torch
    e = reader.identity.expert_count
    rows, columns = reader.identity.matrix_shape
    if inputs.ndim != 2 or inputs.shape[1] != columns or inputs.device.type != "cpu":
        raise ValueError("inputs do not match source matrix shape")
    output = torch.empty((e, len(inputs), rows), dtype=torch.float32)
    for first in range(0, rows, row_chunk):
        last = min(rows, first+row_chunk)
        block = reader.read_columns(first*columns, last*columns).reshape(e, last-first, columns)
        output[:, :, first:last] = torch.einsum("eoi,ni->eno", block, inputs)
    return output


def reconstruction_metrics(source, decoded, *, chunk_columns=65536):
    import torch
    if source.identity != decoded.identity:
        raise ValueError("source and decoded groups disagree")
    e, p = source.identity.expert_count, source.identity.column_count
    energy = torch.zeros(e, dtype=torch.float64)
    error = torch.zeros(e, dtype=torch.float64)
    maxima = torch.zeros(e, dtype=torch.float32)
    with torch.no_grad():
        for first in range(0, p, chunk_columns):
            last = min(p, first+chunk_columns)
            original = source.read_columns(first, last)
            restored = decoded.read_columns(first, last)
            delta = restored-original
            energy += original.double().square().sum(-1)
            error += delta.double().square().sum(-1)
            maxima = torch.maximum(maxima, delta.abs().amax(-1))
    per = {slot: {"relative_frobenius_error": math.sqrt(float(error[i]/energy[i])) if energy[i] else 0.,
                  "maximum_absolute_error": float(maxima[i])}
           for i, slot in enumerate(source.identity.expert_ids)}
    return {"relative_frobenius_error": math.sqrt(float(error.sum()/energy.sum())) if energy.sum() else 0.,
            "per_expert": per, "error_scope": "dequantized_original_FP8_plus_scales"}


class FactorReader:
    def __init__(self, result):
        self.result = result
        self.identity = result.identity

    def read_columns(self, start, stop, expert_ids=None):
        ids = self.identity.expert_ids if expert_ids is None else tuple(expert_ids)
        positions = [self.identity.expert_ids.index(slot) for slot in ids]
        return self.result.coefficients[positions] @ self.result.basis[:, start:stop].float()+self.result.mean[start:stop].float()


def run(args):
    import torch
    from tda.glm_weight_sample import GLMWeightSampleReader
    from tda.expert_compress import fit_expert_group, save_expert_compression, load_expert_compression
    from tda.block_quant import quantize_reader, save_block_quantization, load_block_quantization
    torch.set_num_threads(args.threads)
    if args.out.exists():
        raise FileExistsError("use a new output directory")
    args.out.mkdir(parents=True)
    started = time.monotonic()
    results = []
    write_json(args.out/"status.json", {"status": "running", "device": "cpu"})
    try:
        with GLMWeightSampleReader(args.sample) as source:
            scope = source.identity
            inputs = torch.randn(6, scope.matrix_shape[1], generator=torch.Generator().manual_seed(931))/math.sqrt(scope.matrix_shape[1])
            print("[GLM sample] Evaluating original FP8 gate projections on synthetic inputs.", flush=True)
            teacher = source_linear(source, inputs)
            write_json(args.out/"settings.json", {"identity": scope.to_dict(), "ranks": args.ranks,
                "factor_dtype": "bfloat16", "device": "cpu", "matrix_kind": "gate_proj_only",
                "sample_hashes_verified": True, "full_checkpoint_evaluated": False,
                "glm_measured_language_states_available": False,
                "synthetic_input_seed": 931, "synthetic_input_count": len(inputs),
                "cannot_extrapolate_storage_or_quality_to_entire_model": True})
            variants = [(f"shared_pca_rank_{rank}", rank) for rank in args.ranks]+[("symmetric_int4", None)]
            for name, rank in variants:
                print(f"[GLM sample] Encoding {name}.", flush=True)
                if rank is None:
                    encoded = quantize_reader(source, bits=4, block_size=128, max_working_bytes=args.budget)
                    metadata = save_block_quantization(encoded, args.out/name, max_artifact_bytes=args.budget)
                    decoder = load_block_quantization(args.out/name, expected_identity=scope, max_artifact_bytes=args.budget)
                else:
                    encoded = fit_expert_group(source, rank, storage_dtype="bfloat16", max_working_bytes=args.budget)
                    metadata = save_expert_compression(encoded, args.out/name, max_artifact_bytes=args.budget)
                    decoder = FactorReader(load_expert_compression(args.out/name, expected_identity=scope, max_artifact_bytes=args.budget))
                restored = source_linear(decoder, inputs)
                delta = restored-teacher
                response_error = float(delta.double().square().sum())
                response_energy = float(teacher.double().square().sum())
                report = {"variant": name, "rank": rank,
                    "serialized_total_bytes": metadata["serialized_total_bytes"],
                    "factor_tensor_bytes": metadata["factor_tensor_bytes"],
                    "source_tensor_bytes_including_scales": scope.actual_source_bytes,
                    "serialized_storage_ratio": metadata["serialized_storage_ratio"],
                    "serialized_is_smaller": metadata["serialized_is_smaller"],
                    **reconstruction_metrics(source, decoder),
                    "synthetic_linear_output_relative_error": math.sqrt(response_error/response_energy) if response_energy else 0.,
                    "synthetic_linear_output_maximum_absolute_error": float(delta.abs().max()),
                    "evaluation": "original_FP8_gate_weight_and_synthetic_linear_reconstruction_only",
                    "language_quality_established": False, "runnable_checkpoint_exported": False,
                    "runtime_savings_evaluated": False}
                results.append(report)
                write_json(args.out/"results.json", results)
                print(f"[GLM sample] {name}: storage={report['serialized_storage_ratio']:.1%}; weight error={report['relative_frobenius_error']:.1%}.", flush=True)
                del encoded, decoder
        write_json(args.out/"status.json", {"status": "finished", "elapsed_seconds": time.monotonic()-started,
            "completed_variants": [r["variant"] for r in results], "device": "cpu"})
    except BaseException as exc:
        write_json(args.out/"status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "completed_variants": [r["variant"] for r in results]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ranks", default="0,1,2")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--budget", type=int, default=1 << 30)
    args = parser.parse_args()
    try:
        args.ranks = [int(rank) for rank in args.ranks.split(",")]
    except ValueError:
        parser.error("ranks must be comma-separated integers")
    if not args.ranks or len(set(args.ranks)) != len(args.ranks) or any(not 0 <= rank <= 7 for rank in args.ranks):
        parser.error("ranks must be distinct values0..7")
    if not 1 <= args.threads <= 16 or not 1 << 20 <= args.budget <= 4 << 30:
        parser.error("threads must be1..16 and budget1MiB..4GiB")
    run(args)


if __name__ == "__main__":
    main()
