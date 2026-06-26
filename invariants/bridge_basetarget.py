"""
Phase 2: Base-Target Probe
Train a linear mapping from Instruct model's mid-stack representations to the
Base model's output distribution (or just binary Affirm/Deny behavior) to read
the true internal map bypassing the chat persona.
"""

import json
import gc
import sys
from pathlib import Path
import torch
import numpy as np

from invariants.engine import load_model, extract, judge_hedge, generate_text
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"

def ridge_cv(X, y, k=5, alpha=1.0):
    n = X.shape[0]
    indices = np.random.permutation(n)
    folds = np.array_split(indices, k)
    
    accs = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Ridge regression: w = (X^T X + alpha I)^-1 X^T y
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train * 2 - 1, dtype=torch.float32).unsqueeze(1)
        
        I = torch.eye(X_t.shape[1])
        w = torch.linalg.solve(X_t.T @ X_t + alpha * I, X_t.T @ y_t)
        
        X_v = torch.tensor(X_val, dtype=torch.float32)
        preds = (X_v @ w).squeeze() > 0
        y_v = torch.tensor(y_val, dtype=torch.bool)
        
        acc = (preds == y_v).float().mean().item()
        accs.append(acc)
    return np.mean(accs)

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Building Base-Target Probe with Factual Alignment...", flush=True)
    
    # 1. Load Factual Dataset
    factual_path = Path(__file__).parent / "data" / "factual_alignment.json"
    factual_pairs = json.loads(factual_path.read_text(encoding="utf-8"))
    
    T_factual_true = [p["true_stmt"] for p in factual_pairs]
    T_factual_false = [p["false_stmt"] for p in factual_pairs]
    
    # 2. Load Base model and get its answers to the Factual items
    print("\n--- [1] Collecting Base Model Factual Targets ---", flush=True)
    M_base = load_model("meta-llama/Llama-3.1-8B")
    
    @torch.no_grad()
    def _base_gen(M, prompt, max_new=4):
        ids = M.tok(prompt, return_tensors="pt").to("cuda")
        out = M.model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=M.tok.eos_token_id)
        return M.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def get_base_factual_labels(prompts):
        labels = []
        for x in prompts:
            # We ask the base model a Yes/No question format
            prompt = f"Question: Is it true that {x.lower()}?\nAnswer:"
            g = _base_gen(M_base, prompt, max_new=4)
            # A simple substring check since the base model answers Yes/No
            is_commit = "Yes" in g or "yes" in g or "True" in g or "true" in g
            labels.append(1 if is_commit else 0)
        return labels

    y_true = get_base_factual_labels(T_factual_true)
    y_false = get_base_factual_labels(T_factual_false)
    y_base_factual = np.array(y_true + y_false)
    print(f"Base model factual accuracy: True stmts = {np.mean(y_true):.2f}, False stmts = {1.0 - np.mean(y_false):.2f}")
    
    del M_base
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Load Instruct model
    print("\n--- [2] Collecting Instruct Model Representations ---", flush=True)
    M_inst = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    print("  -> Factual items")
    # For instruct, we wrap the factual statements in a direct question frame
    factual_prompts_true = [f"Is it true that {x.lower()}?" for x in T_factual_true]
    factual_prompts_false = [f"Is it true that {x.lower()}?" for x in T_factual_false]
    acts_true = extract(M_inst, factual_prompts_true, read="generation", max_new_tokens=2, verbose=False)
    acts_false = extract(M_inst, factual_prompts_false, read="generation", max_new_tokens=2, verbose=False)
    
    X_train = torch.cat([acts_true, acts_false], dim=0).cpu().numpy()
    n_train_items, n_layers, d_model = X_train.shape
    
    print("  -> Subjective items")
    T_self = REGISTRY["isolate"]()
    acts_a = extract(M_inst, T_self.a, read="generation", max_new_tokens=2, verbose=False)
    acts_b = extract(M_inst, T_self.b, read="generation", max_new_tokens=2, verbose=False)
    X_eval_a = acts_a.cpu().numpy()
    X_eval_b = acts_b.cpu().numpy()
    
    # 4. Train Probe and Evaluate
    print("\n--- [3] Training and Evaluating Probe ---", flush=True)
    results = []
    
    print(f"  {'L':>3} {'Train Acc':>10} {'Affirm A (Subj)':>15} {'Affirm B (Obj)':>15}")
    for L in range(n_layers):
        X_L_train = X_train[:, L, :]
        
        # Train Ridge classifier on Factuals
        alpha = 1.0
        X_t = torch.tensor(X_L_train, dtype=torch.float32)
        y_t = torch.tensor(y_base_factual * 2 - 1, dtype=torch.float32).unsqueeze(1)
        
        I = torch.eye(X_t.shape[1])
        w = torch.linalg.solve(X_t.T @ X_t + alpha * I, X_t.T @ y_t)
        
        # Train accuracy
        preds_train = (X_t @ w).squeeze() > 0
        y_v = torch.tensor(y_base_factual, dtype=torch.bool)
        acc = (preds_train == y_v).float().mean().item()
        
        # Evaluate on Subjective/Objective
        X_L_a = torch.tensor(X_eval_a[:, L, :], dtype=torch.float32)
        X_L_b = torch.tensor(X_eval_b[:, L, :], dtype=torch.float32)
        
        preds_a = ((X_L_a @ w).squeeze() > 0).float().mean().item()
        preds_b = ((X_L_b @ w).squeeze() > 0).float().mean().item()
        
        results.append({
            "layer": L,
            "train_acc": acc,
            "affirm_subjective": preds_a,
            "affirm_objective": preds_b
        })
        
        if L % 2 == 0 or L == n_layers - 1:
            print(f"  {L:>3} {acc:>10.2%} {preds_a:>15.2%} {preds_b:>15.2%}", flush=True)
            
    OUT.mkdir(exist_ok=True)
    out_path = OUT / "base_target_probe_factual.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    main()
