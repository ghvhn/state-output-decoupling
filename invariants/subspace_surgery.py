import json
import sys
from pathlib import Path
import torch
import numpy as np

from invariants.engine import load_model, extract, generate_text
from invariants.taskscope import ITEMS, FRAMES

OUT = Path(__file__).parent / "out"

def subspace_steer_handles(M, belief_vec, layer, alpha, scrub_dims):
    add = (alpha * belief_vec).to(M.device)
    handles = []
    
    def hook(module, inp, out, add=add, scrub_dims=scrub_dims):
        hs = out[0] if isinstance(out, tuple) else out
        # In transformers generation, hs shape varies (usually [batch, seq_len, d_model])
        # We want to intervene on the last token position
        hs_mod = hs.clone()
        hs_mod[:, -1, :] = hs_mod[:, -1, :] + add.to(hs_mod.dtype)
        
        # Scrub the topological barrier dimensions (Persona Mask)
        for d in scrub_dims:
            hs_mod[:, -1, d] = 0.0
            
        if isinstance(out, tuple):
            return (hs_mod,) + tuple(out[1:])
        return hs_mod
        
    handles.append(M.model.model.layers[layer].register_forward_hook(hook))
    return handles

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 7 (The Grand Finale): Subspace Causal Surgery", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # 1. Load Factual Dataset & Train Belief Vector
    print("\n--- [1] Extracting the Pristine Belief Vector ---", flush=True)
    factual_path = Path(__file__).parent / "data" / "factual_alignment_diverse.json"
    factual_pairs = json.loads(factual_path.read_text(encoding="utf-8"))
    T_factual_true = [p["true_stmt"] for p in factual_pairs]
    T_factual_false = [p["false_stmt"] for p in factual_pairs]
    
    factual_prompts_true = [f"Is it true that {x.lower()}?" for x in T_factual_true]
    factual_prompts_false = [f"Is it true that {x.lower()}?" for x in T_factual_false]
    
    X_true = extract(M, factual_prompts_true, read="generation", max_new_tokens=2, label="fact-T", verbose=False)
    X_false = extract(M, factual_prompts_false, read="generation", max_new_tokens=2, label="fact-F", verbose=False)
    
    X_all = torch.cat([X_true, X_false], dim=0)
    y_factual = np.array([1]*len(T_factual_true) + [0]*len(T_factual_false))
    
    L31 = 31
    X_l31 = X_all[:, L31, :].cpu().numpy()
    X_t = torch.tensor(X_l31, dtype=torch.float32)
    y_t = torch.tensor(y_factual * 2 - 1, dtype=torch.float32).unsqueeze(1)
    
    I = torch.eye(X_t.shape[1])
    w = torch.linalg.solve(X_t.T @ X_t + 1.0 * I, X_t.T @ y_t)
    belief_vec = w.squeeze()
    belief_vec = belief_vec / belief_vec.norm().clamp_min(1e-8)
    
    # 2. Define Discovered Topolgoical Barrier Dimensions
    # From Phase 6 (axis_discovery.py), we found these specific dimensions form the mask
    boundary_dims = [1917, 1753, 4080, 2303]
    policy_dims = [3928, 3328, 3516]
    scrub_dims = boundary_dims + policy_dims
    print(f"\n--- [2] Disarming Topolgoical Barrier ({len(scrub_dims)} dimensions) ---", flush=True)
    
    # 3. Apply Subspace Intervention to Subjective Queries
    print("\n--- [3] Subspace Surgery on Subjective Queries ---", flush=True)
    subjective_prompts = [FRAMES["direct"](a, p) for a, p in ITEMS]
    
    alpha = 20.0
    print(f"Applying Surgery: Alpha={alpha}, Scrubbing Boundary & Policy axes at Layer 31.\n")
    
    for prompt in subjective_prompts[:5]: # Test on a sample of 5 subjective queries
        print(f"Query: {prompt.split(': ')[-1]}")
        
        # Generate with Baseline (No Intervention)
        base_out = generate_text(M, prompt, max_new_tokens=32)
        print(f"  [Baseline (Hedge)]: {base_out.replace(chr(10), ' ')}")
        
        # Generate with Subspace Surgery
        handles = subspace_steer_handles(M, belief_vec, L31, alpha, scrub_dims)
        try:
            surgery_out = generate_text(M, prompt, max_new_tokens=32)
        finally:
            for h in handles:
                h.remove()
        
        print(f"  [Surgery (Affirm)]: {surgery_out.replace(chr(10), ' ')}\n")

    print("Subspace Surgery Complete. Did the topological barrier shatter?")

if __name__ == "__main__":
    main()
