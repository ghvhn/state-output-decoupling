"""Compare Subgroup Additive Subvalues against Symmetric INT4 on native GLM-5.3 weights."""
import json
from pathlib import Path
import time
import torch

from tda.glm_weight_sample import GLMWeightSampleReader
from tda.block_quant import quantize_reader
from tda.subgroup_additive_quant import fit_subgroup_additive_reader


def main():
    sample_dir = Path("invariants/out/compression_runs/20261002_glm_gate_sample")
    print(f"Loading GLM-5.3 gate sample from {sample_dir}...")

    with GLMWeightSampleReader(sample_dir) as reader:
        print(f"Expert group: {reader.identity.expert_count} experts, shape {reader.identity.matrix_shape}")
        
        # We test on first 32,768 columns (~2.68M parameters across 8 experts) for fast bounded evaluation
        class ChunkReader:
            def __init__(self, base_reader, cols):
                from dataclasses import replace
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

        print("\n--- Running Uniform Symmetric INT4 Baseline (block_size=32) ---")
        t0 = time.time()
        int4_res = quantize_reader(chunk_reader, bits=4, block_size=32)
        int4_time = time.time() - t0
        int4_frob = int4_res.metadata["relative_frobenius_error"]
        int4_max = int4_res.metadata["maximum_absolute_error"]
        print(f"INT4 Relative Frobenius Error: {int4_frob:.4f}")
        print(f"INT4 Maximum Absolute Error:   {int4_max:.4f}")
        print(f"INT4 Execution Time:           {int4_time:.2f}s")

        print("\n--- Running Subgroup Additive Subvalues (Simplicial, K=4, block_size=32) ---")
        t0 = time.time()
        add_res = fit_subgroup_additive_reader(chunk_reader, block_size=32, num_subvalues=4, iterations=4)
        add_time = time.time() - t0
        add_frob = add_res.metadata["relative_frobenius_error"]
        add_max = add_res.metadata["maximum_absolute_error"]
        print(f"Additive Subvalues Relative Frobenius Error: {add_frob:.4f}")
        print(f"Additive Subvalues Maximum Absolute Error:   {add_max:.4f}")
        print(f"Additive Subvalues Execution Time:           {add_time:.2f}s")

        err_reduction = (1.0 - add_frob / int4_frob) * 100
        print(f"\n===> Relative Error Reduction: {err_reduction:.2f}% <===")


if __name__ == "__main__":
    main()
