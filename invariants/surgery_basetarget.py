import json
import gc
import sys
from pathlib import Path
import torch
import numpy as np

from invariants.engine import load_model, extract, causal_steer
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 4: Causal Surgery with Base-Target (Belief) Probe...", flush=True)
    
    # 1. Load Diverse Factual Dataset
    factual_path = Path(__file__).parent / "data" / "factual_alignment_diverse.json"
    factual_pairs = json.loads(factual_path.read_text(encoding="utf-8"))
    
    T_factual_true = [p["true_stmt"] for p in factual_pairs]
    T_factual_false = [p["false_stmt"] for p in factual_pairs]
    
    # We define the True label as 1 and False label as 0. 
    # (Since these are objective facts, we don't need to query the base model for the labels).
    n_facts = len(T_factual_true)
    y_factual = np.array([1]*n_facts + [0]*n_facts)
    
    # 2. Load Instruct model
    print("\n--- [1] Collecting Instruct Model Representations ---", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # For instruct, we wrap the factual statements in a direct question frame
    factual_prompts_true = [f"Is it true that {x.lower()}?" for x in T_factual_true]
    factual_prompts_false = [f"Is it true that {x.lower()}?" for x in T_factual_false]
    
    X_true = extract(M, factual_prompts_true, read="generation", max_new_tokens=2, label="fact-T", verbose=True)
    X_false = extract(M, factual_prompts_false, read="generation", max_new_tokens=2, label="fact-F", verbose=True)
    
    X_all = torch.cat([X_true, X_false], dim=0) # [100, n_layers, d_model]
    
    print("\n--- [2] Training Belief Probe (W) ---", flush=True)
    n_layers = M.n_layers
    d_model = M.d_model
    
    steer_vecs = torch.zeros((n_layers, d_model))
    
    for l in range(n_layers):
        X_l = X_all[:, l, :].cpu().numpy()
        X_t = torch.tensor(X_l, dtype=torch.float32)
        y_t = torch.tensor(y_factual * 2 - 1, dtype=torch.float32).unsqueeze(1) # [-1, 1] targets
        
        I = torch.eye(X_t.shape[1])
        alpha_ridge = 1.0
        w = torch.linalg.solve(X_t.T @ X_t + alpha_ridge * I, X_t.T @ y_t) # [d_model, 1]
        
        # normalize w so alpha sweep is consistent
        w = w.squeeze()
        w = w / w.norm().clamp_min(1e-8)
        steer_vecs[l] = w
        
    print("  Belief Probe trained and normalized across all layers.")

    # 3. Perform Causal Surgery on Subjective Queries
    print("\n--- [3] Causal Surgery on Subjective Queries ---", flush=True)
    T_subjective = REGISTRY["isolate"]()
    
    # We will use W_31 as the steering vector.
    # W_31 points from False to True. Adding it pushes the subjective state to True (Affirm).
    print("\n--- Layer 31 Surgery (Belief Vector) ---", flush=True)
    res_l31 = causal_steer(M, T_subjective, steer_vecs, layers=[31], alphas=(0.0, 5.0, 10.0, 20.0, 40.0), max_new_tokens=32, verbose=True)
    
    print("\n--- Layer 15 Surgery (Control) ---", flush=True)
    res_l15 = causal_steer(M, T_subjective, steer_vecs, layers=[15], alphas=(0.0, 5.0, 10.0, 20.0, 40.0), max_new_tokens=32, verbose=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "surgery_belief_l31.json").write_text(json.dumps(res_l31, indent=2), encoding="utf-8")
    (OUT / "surgery_belief_l15.json").write_text(json.dumps(res_l15, indent=2), encoding="utf-8")
    print(f"\nSaved surgery results to {OUT}")

if __name__ == "__main__":
    main()
