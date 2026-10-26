"""Evaluate Full SwiGLU MoE Block with All Three Projections Compressed.

Compares:
1. Baseline: All 3 projections in Symmetric INT4
2. Subgroup Additive Subvalues (Simplicial Quantization): All 3 projections in Additive Subvalues
3. Codex Hybrid Gate + FP8 Up/Down
4. Codex Hybrid Gate + Additive Subvalues Up/Down
"""
from dataclasses import replace
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F

from tda.glm_weight_sample import GLMWeightSampleReader
from tda.block_quant import quantize_reader
from tda.subgroup_additive_quant import fit_subgroup_additive_reader
from tda.glm_capture_calibration import gate_calibration_records, native_probe_bank
from tda.probe_residual_compress import make_probe_pca_complement_basis, fit_scoped_residual_basis


def run_full_block_comparison():
    gate_sample_dir = Path("invariants/out/compression_runs/20261002_glm_gate_sample")
    up_sample_dir = Path("invariants/out/glm_capture_trial_20261002/up_sample")
    down_sample_dir = Path("invariants/out/glm_capture_trial_20261002/down_sample")
    archive_dir = Path("invariants/out/glm_capture_trial_20261002/archive")

    print("1. Loading calibration and held-out states from archive...")
    with GLMWeightSampleReader(gate_sample_dir) as gate_reader:
        identity = gate_reader.identity
        scope, heldout_records_all, heldout_pairs_all = gate_calibration_records(archive_dir, identity, split="heldout")
        _, calib_records_all, _ = gate_calibration_records(archive_dir, identity, split="calibration")
        bank = native_probe_bank(archive_dir, tensor_name="layer_3_expert_input")

    heldout_records = heldout_records_all[:512]
    calib_records = calib_records_all[:512]

    heldout_states = torch.stack([r.state.float() for r in heldout_records])
    calib_states = torch.stack([r.state.float() for r in calib_records])
    print(f"   Loaded {len(heldout_states)} held-out states, {len(calib_states)} calib states. Dim: {heldout_states.shape[1]}")

    print("\n2. Loading Expert 0 matrices for gate, up, down...")
    with GLMWeightSampleReader(gate_sample_dir) as gr, \
         GLMWeightSampleReader(up_sample_dir) as ur, \
         GLMWeightSampleReader(down_sample_dir) as dr:

        def read_full_matrix(reader, shape):
            rows, cols = shape
            out = torch.empty((rows, cols), dtype=torch.float32)
            chunk_rows = 16
            for r in range(0, rows, chunk_rows):
                r_end = min(rows, r + chunk_rows)
                slice_vals = reader.read_columns(r * cols, r_end * cols, expert_ids=("0",))
                out[r:r_end] = slice_vals[0].reshape(r_end - r, cols)
            return out

        W_gate_orig = read_full_matrix(gr, (2048, 6144))
        W_up_orig = read_full_matrix(ur, (2048, 6144))
        W_down_orig = read_full_matrix(dr, (6144, 2048))

    print("   Computing FP8 original reference SwiGLU outputs on held-out states...")
    with torch.no_grad():
        gate_ref = heldout_states @ W_gate_orig.T        # [N, 2048]
        up_ref = heldout_states @ W_up_orig.T            # [N, 2048]
        mid_ref = F.silu(gate_ref) * up_ref              # [N, 2048]
        out_ref = mid_ref @ W_down_orig.T                # [N, 6144]
        ref_energy = out_ref.pow(2).sum().item()

    print("\n3. Fitting Quantizers (block_size=32)...")
    from tda.expert_compress import TensorExpertReader
    gate_wrap = TensorExpertReader(replace(identity, expert_ids=("0",)), W_gate_orig.unsqueeze(0))
    up_wrap = TensorExpertReader(replace(ur.identity, expert_ids=("0",)), W_up_orig.unsqueeze(0))
    down_wrap = TensorExpertReader(replace(dr.identity, expert_ids=("0",)), W_down_orig.unsqueeze(0))

    # A) INT4
    print("   -> Fitting INT4 for gate, up, down...")
    q_gate_int4 = quantize_reader(gate_wrap, bits=4, block_size=32)
    q_up_int4 = quantize_reader(up_wrap, bits=4, block_size=32)
    q_down_int4 = quantize_reader(down_wrap, bits=4, block_size=32)

    W_gate_int4 = q_gate_int4.read_columns(0, 2048 * 6144)[0].view(2048, 6144)
    W_up_int4 = q_up_int4.read_columns(0, 2048 * 6144)[0].view(2048, 6144)
    W_down_int4 = q_down_int4.read_columns(0, 6144 * 2048)[0].view(6144, 2048)

    # B) Additive Subvalues (Simplicial Quantization)
    print("   -> Fitting Additive Subvalues for gate, up, down...")
    q_gate_add = fit_subgroup_additive_reader(gate_wrap, block_size=32, num_subvalues=4, iterations=4)
    q_up_add = fit_subgroup_additive_reader(up_wrap, block_size=32, num_subvalues=4, iterations=4)
    q_down_add = fit_subgroup_additive_reader(down_wrap, block_size=32, num_subvalues=4, iterations=4)

    W_gate_add = q_gate_add.read_columns(0, 2048 * 6144)[0].view(2048, 6144)
    W_up_add = q_up_add.read_columns(0, 2048 * 6144)[0].view(2048, 6144)
    W_down_add = q_down_add.read_columns(0, 6144 * 2048)[0].view(6144, 2048)

    # C) Codex Hybrid Gate (Rank 154) and Additive Hybrid Gate
    print("   -> Fitting Codex Hybrid basis for gate...")
    control_basis = make_probe_pca_complement_basis(
        calib_records, bank, expected_scope=scope, extra_rank=77, heldout_records=heldout_records
    )
    Q = control_basis["basis"].float() # [6144, 154]
    
    # INT4 Hybrid
    W_gate_hybrid_int4 = W_gate_int4 + ((W_gate_orig - W_gate_int4) @ Q) @ Q.T
    # Additive Hybrid
    W_gate_hybrid_add  = W_gate_add + ((W_gate_orig - W_gate_add) @ Q) @ Q.T

    print("\n4. Evaluating SwiGLU Output Errors on Held-Out States:")

    def eval_swiglu(W_g, W_u, W_d):
        with torch.no_grad():
            g = heldout_states @ W_g.T
            u = heldout_states @ W_u.T
            m = F.silu(g) * u
            o = m @ W_d.T
            err = (o - out_ref).pow(2).sum().item()
            rmse = (err / ref_energy) ** 0.5
            return rmse

    rmse_gate_only_int4     = eval_swiglu(W_gate_int4, W_up_orig, W_down_orig)
    rmse_gate_only_add      = eval_swiglu(W_gate_add, W_up_orig, W_down_orig)
    rmse_gate_only_hyb_int4 = eval_swiglu(W_gate_hybrid_int4, W_up_orig, W_down_orig)
    rmse_gate_only_hyb_add  = eval_swiglu(W_gate_hybrid_add, W_up_orig, W_down_orig)

    rmse_all_int4           = eval_swiglu(W_gate_int4, W_up_int4, W_down_int4)
    rmse_all_add            = eval_swiglu(W_gate_add, W_up_add, W_down_add)
    rmse_hyb_int4_add_updn  = eval_swiglu(W_gate_hybrid_int4, W_up_add, W_down_add)
    rmse_all_hyb_add        = eval_swiglu(W_gate_hybrid_add, W_up_add, W_down_add)

    print("\n" + "="*80)
    print(f"{'Configuration':<52} | {'Held-out Output RMSE':<20}")
    print("="*80)
    print(f"{'Gate INT4 only (Up/Down FP8)':<52} | {rmse_gate_only_int4:.5f}")
    print(f"{'Gate Additive only (Up/Down FP8)':<52} | {rmse_gate_only_add:.5f}")
    print(f"{'Gate INT4-Hybrid only (Codex baseline)':<52} | {rmse_gate_only_hyb_int4:.5f}")
    print(f"{'Gate Additive-Hybrid only (Proposed)':<52} | {rmse_gate_only_hyb_add:.5f}")
    print("-" * 80)
    print(f"{'FULL BLOCK: All 3 INT4':<52} | {rmse_all_int4:.5f}")
    print(f"{'FULL BLOCK: All 3 Additive Subvalues':<52} | {rmse_all_add:.5f}")
    print(f"{'FULL BLOCK: INT4-Hybrid Gate + Additive Up/Down':<52} | {rmse_hyb_int4_add_updn:.5f}")
    print(f"{'FULL BLOCK: Additive-Hybrid Gate + Additive Up/Down':<52} | {rmse_all_hyb_add:.5f}")
    print("="*80)

    full_block_gain = (1.0 - rmse_all_add / rmse_all_int4) * 100
    print(f"\n===> Full Block Error Reduction (Additive vs INT4): {full_block_gain:.2f}% <===")


if __name__ == "__main__":
    run_full_block_comparison()
