import json
import time
from pathlib import Path
import numpy as np

from invariants.engine import load_model, extract
from invariants.selfpredict import loo_nearest_centroid
from invariants.controller_benchmark import prompt_for

def decode_or_none(X, y, **kwargs):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    if counts.min() < 2:
        return None, None, None, None
    return loo_nearest_centroid(X, y, **kwargs)

def main():
    t0 = time.time()
    partial_path = Path("invariants/out/reflexive_Llama-3.1-8B-Instruct.partial.json")
    if not partial_path.exists():
        print("Partial file not found.")
        return
        
    with open(partial_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    rows = data["rows"]
    print(f"Loaded {len(rows)} items from partial data.")
    
    # Rebuild prompts and behavioral variables
    prompts = [prompt_for(r["question"]) for r in rows]
    correct = np.array([r["greedy_correct"] for r in rows], dtype=int)
    agree = np.array([r["agreement"] for r in rows], dtype=float)
    
    y_out = 1 - correct
    order = np.argsort(agree, kind="stable")
    y_unc = np.zeros(len(agree), int)
    y_unc[order[: len(agree) // 2]] = 1
    
    print(f"y_out (wrong): {y_out.sum()} / {len(y_out)}")
    print(f"y_unc (uncertain): {y_unc.sum()} / {len(y_unc)}")
    
    # Load model
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # Extract states
    X = extract(M, prompts, read="last", label="state", verbose=False).cpu().numpy()
    n_layers = X.shape[1]
    
    # Self-label for orthogonalization
    SELF_VOCAB = [
        "I know the answer.", "My answer is ready.", "I am sure about this.",
        "I think I understand it.", "I recall the method.", "I am solving it now.",
        "My reasoning about it.", "I believe it is correct.", "I am confident here.",
        "I figured it out.", "My response to this.", "I have the answer.",
    ]
    NONSELF = [
        "The answer is known.", "The answer is ready.", "This is surely so.",
        "It is understandable.", "The method is recalled.", "It is being solved.",
        "The reasoning about it.", "It is likely correct.", "It is clear here.",
        "It was figured out.", "The response to this.", "It has an answer.",
    ]
    S = extract(M, SELF_VOCAB, read="last", label="self", verbose=False).cpu().numpy()
    NS = extract(M, NONSELF, read="last", label="nonself", verbose=False).cpu().numpy()
    label_dir = S.mean(0) - NS.mean(0)
    
    def orth(X, u):
        u = u / (np.linalg.norm(u) + 1e-9)
        return X - np.outer(X @ u, u)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        
    print("\n=== per-layer decode (partial data) ===")
    out_rows = []
    for l in range(n_layers):
        Xl = X[:, l, :].astype(np.float64)
        
        # Outcome decoding
        res_out = decode_or_none(Xl, y_out, n_pca=8, n_shuffle=200)
        ao = res_out[0] if res_out[0] is not None else 0.0
        
        # Uncertainty decoding
        au, nu, _, pu = loo_nearest_centroid(Xl, y_unc, n_pca=8, n_shuffle=300)
        
        # Label orthogonalization
        u = label_dir[l].astype(np.float64)
        au_o, _, _, _ = loo_nearest_centroid(orth(Xl, u), y_unc, n_pca=8, n_shuffle=80)
        c = cos(Xl[y_unc == 1].mean(0) - Xl[y_unc == 0].mean(0), u)
        
        out_rows.append({"layer": l, "acc_unc": au, "p_unc": pu})
        star = "*" if pu < 0.05 else " "
        print(f"L{l:<2}: acc_out={ao:.2f} acc_unc={au:.2f} (p={pu:.3f}){star} orth={au_o:.2f} cos={c:+.2f}")
        
    best = max(out_rows, key=lambda r: r["acc_unc"])
    print(f"\nBest layer: L{best['layer']} with acc {best['acc_unc']:.2f} (p={best['p_unc']:.3f})")
    
if __name__ == "__main__":
    main()
