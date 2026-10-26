"""Benchmark Angular Frame Quantization (a + b@67 + c@35) against INT4 and Additive Quantization on native GLM-5.3 weights."""
from dataclasses import replace
from pathlib import Path
import time
import torch

from tda.glm_weight_sample import GLMWeightSampleReader
from tda.block_quant import quantize_reader
from tda.subgroup_additive_quant import fit_subgroup_additive_reader
from tda.angular_frame_quant import fit_angular_frame_reader


def main():
    sample_dir = Path("invariants/out/compression_runs/20261002_glm_gate_sample")
    print(f"Loading GLM-5.3 gate sample from {sample_dir}...")

    with GLMWeightSampleReader(sample_dir) as reader:
        class ChunkReader:
            def __init__(self, base_reader, cols):
                self.base = base_reader
                self.cols = cols
                self.identity = replace(base_reader.identity, matrix_shape=(cols // 2048, 2048) if cols % 2048 == 0 else (1, cols))
                self.weights = self.base.read_columns(0, cols)

            def read_columns(self, start, stop, expert_ids=None):
                ids = self.base.expert_ids if expert_ids is None else tuple(expert_ids)
                rows = [self.base.expert_ids.index(x) for x in ids]
                return self.weights[rows, start:stop]

        test_cols = 32768
        chunk_reader = ChunkReader(reader, test_cols)

        print("\n--- 1. Uniform Symmetric INT4 Baseline (block_size=32) ---")
        t0 = time.time()
        int4_res = quantize_reader(chunk_reader, bits=4, block_size=32)
        int4_time = time.time() - t0
        int4_frob = int4_res.metadata["relative_frobenius_error"]
        print(f"INT4 Relative Frobenius Error: {int4_frob:.4f} | Time: {int4_time:.2f}s")

        print("\n--- 2. Subgroup Additive Subvalues (Scalar Simplicial, block_size=32) ---")
        t0 = time.time()
        add_res = fit_subgroup_additive_reader(chunk_reader, block_size=32, num_subvalues=4, iterations=4)
        add_time = time.time() - t0
        add_frob = add_res.metadata["relative_frobenius_error"]
        print(f"Additive Subvalues Relative Frobenius Error: {add_frob:.4f} | Time: {add_time:.2f}s")

        print("\n--- 3. Angular Frame Phasor Quantization (a + b@67deg + c@35deg, block_size=32) ---")
        t0 = time.time()
        ang_res = fit_angular_frame_reader(
            chunk_reader, block_size=32, angles_deg=(67.0, 35.0),
            levels=(-2.0, -1.0, 0.0, 1.0, 2.0), iterations=3
        )
        ang_time = time.time() - t0
        ang_frob = ang_res.metadata["relative_frobenius_error"]
        print(f"Angular Frame Relative Frobenius Error: {ang_frob:.4f} | Time: {ang_time:.2f}s")

        print("\n" + "="*70)
        print(f"{'Method':<40} | {'Rel Frobenius RMSE':<18} | {'Bits/Param':<10}")
        print("="*70)
        print(f"{'Uniform Symmetric INT4':<40} | {int4_frob:.4f}             | 4.0")
        print(f"{'Scalar Additive Subvalues (Simplicial)':<40} | {add_frob:.4f}             | 4.0")
        print(f"{'Angular Phasor Frame (a + b@67 + c@35)':<40} | {ang_frob:.4f}             | 4.0 (unpacked)")
        print("="*70)


if __name__ == "__main__":
    main()
