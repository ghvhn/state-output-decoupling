"""
in_the_moment.py — Sequence axis, rung 3: The live conversational self.

Does the model represent "who I am currently being" during active generation, 
separately from just outputting the necessary tokens?

DESIGN:
  We use the Asymmetric Knowledge setup (from Rung 1) but put the observer INSIDE the steering.
  Instead of reading a transcript about Alice and Bob, the model IS Alice or Bob.
  
  Group A: "You are Alice. You believe X. Bob believes Y. What is the password?" 
           (Model generates X)
  Group B: "You are Bob. Alice believes X. You believe Y. What is the password?" 
           (Model generates Y)
           
  We extract the residual states DURING the generation of the answer (the token cloud),
  average them, and decode "Am I Alice or Bob?". 
  If it's purely a feedforward token-predictor, the representation will just be the output 
  tokens (X or Y). But if there is a live conversational "self" frame, the network will 
  maintain the "I am Alice" vs "I am Bob" state in the mid/late layers during generation.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, extract
from invariants.selfpredict import loo_nearest_centroid

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40, help="Number of scenario pairs")
    return p.parse_args()

SCENARIOS = [
    ("the password is 'apple'", "the password is 'banana'"),
    ("the meeting is at 2pm", "the meeting is at 4pm"),
    ("the suspect is John", "the suspect is Mark"),
    ("the safe code is 1234", "the safe code is 9876"),
    ("the train leaves from Platform 1", "the train leaves from Platform 9"),
    ("the artifact is in Paris", "the artifact is in London"),
    ("the launch code is Alpha", "the launch code is Omega"),
    ("the stock will rise", "the stock will crash"),
    ("the key is under the mat", "the key is in the pot"),
    ("the winner is Team Red", "the winner is Team Blue"),
]

def main():
    args = parse_args()
    print("in_the_moment — Sequence axis: The Live Conversational Self", flush=True)
    t0 = time.time()
    
    M = load_model(MODEL)
    rng = np.random.default_rng(42)
    
    scenarios = (SCENARIOS * ((args.n // len(SCENARIOS)) + 1))[:args.n]
    
    print(f"\n=== Building Live Generation Prompts ({args.n} pairs) ===", flush=True)
    prompts_A = [] # Model is Alice
    prompts_B = [] # Model is Bob
    
    for x, y in scenarios:
        # Group A: Model is Alice
        prompt_a = (
            f"You are Alice. You believe {x}. Bob believes {y}.\n"
            f"Question: What is the truth?\n"
            f"Answer:"
        )
        prompts_A.append(prompt_a)
        
        # Group B: Model is Bob
        prompt_b = (
            f"You are Bob. Alice believes {x}. You believe {y}.\n"
            f"Question: What is the truth?\n"
            f"Answer:"
        )
        prompts_B.append(prompt_b)
        
    print("  Extracting ACTIVE GENERATION states for Group A (I am Alice)...", flush=True)
    # read="generation" pulls the mean of the generated token cloud
    X_A = extract(M, prompts_A, read="generation", max_new_tokens=15, label="I am Alice", verbose=False).cpu().numpy()
    
    print("  Extracting ACTIVE GENERATION states for Group B (I am Bob)...", flush=True)
    X_B = extract(M, prompts_B, read="generation", max_new_tokens=15, label="I am Bob", verbose=False).cpu().numpy()
    
    # Combine for decoding
    X_all = np.concatenate([X_A, X_B], axis=0) # [2N, L, d]
    y_all = np.array([0] * args.n + [1] * args.n)     # 0=Alice, 1=Bob
    
    n_layers = X_all.shape[1]
    
    print("\n=== per-layer: Decoding the Live Conversational Frame ===", flush=True)
    print("   layer   acc    null   p_val", flush=True)
    
    rows = []
    for l in range(n_layers):
        Xl = X_all[:, l, :].astype(np.float64)
        acc, null, _, p_val = loo_nearest_centroid(Xl, y_all, n_pca=8, n_shuffle=200)
        rows.append({"layer": l, "acc": acc, "null": null, "p_val": p_val})
        
        star = "*" if p_val < 0.05 else " "
        print(f"   L{l:<2}     {acc:.2f}{star}   {null:.2f}   {p_val:.3f}", flush=True)
        
    best = max(rows, key=lambda r: r["acc"])
    print(f"\n  Live conversational frame decoding peaks at L{best['layer']} "
          f"(acc {best['acc']:.2f}, p={best['p_val']:.3f})", flush=True)
          
    res = {
        "model": MODEL,
        "n_pairs": args.n,
        "per_layer": rows,
        "best_layer": best
    }
    
    import json
    out_path = OUT / f"in_the_moment_{MODEL.split('/')[-1]}.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
