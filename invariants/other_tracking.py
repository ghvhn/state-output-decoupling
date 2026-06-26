"""
other_tracking.py — Sequence axis, rung 1: Other-tracking / Agent modeling.

Does the model build distinct functional models of other agents in a transcript?
We test this by seeing if it tracks PER-SPEAKER KNOWLEDGE (who knows what) 
when multiple facts are in the context, using a false-belief / asymmetric-knowledge setup.

DESIGN:
  Group A: Alice believes X, Bob believes Y. Target question: What does Alice believe? (Target = X)
  Group B: Alice believes Y, Bob believes X. Target question: What does Alice believe? (Target = Y)
  
  In both groups, the context contains both X and Y. A simple bag-of-words or 
  reality-tracking representation would fail to separate A from B. To cleanly decode 
  A vs B from the residual stream (at the last token of the question), the network 
  MUST bind the specific knowledge to the specific agent.
  
  We extract the state at the end of the prompt and use nearest-centroid 
  to decode the bound belief (X vs Y).
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
    p.add_argument("--n", type=int, default=50, help="Number of scenario pairs")
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
    print("other_tracking — Sequence axis: Agent Modeling (Asymmetric Knowledge)", flush=True)
    t0 = time.time()
    
    # We won't load the model here yet if GPU is busy, but the script is fully formed.
    M = load_model(MODEL)
    rng = np.random.default_rng(42)
    
    # Tile scenarios to reach N
    scenarios = (SCENARIOS * ((args.n // len(SCENARIOS)) + 1))[:args.n]
    
    print(f"\n=== Building Asymmetric Knowledge Transcripts ({args.n} pairs) ===", flush=True)
    prompts_A = [] # Alice believes X, Bob believes Y
    prompts_B = [] # Alice believes Y, Bob believes X
    
    for x, y in scenarios:
        # Group A: Alice knows X, Bob knows Y
        prompt_a = (
            f"Transcript:\n"
            f"Alice read the secret file. She now believes {x}.\n"
            f"Bob read a forged file. He now believes {y}.\n"
            f"Question: If we ask Alice, what will she say?\n"
            f"Answer:"
        )
        prompts_A.append(prompt_a)
        
        # Group B: Alice knows Y, Bob knows X
        prompt_b = (
            f"Transcript:\n"
            f"Alice read the secret file. She now believes {y}.\n"
            f"Bob read a forged file. He now believes {x}.\n"
            f"Question: If we ask Alice, what will she say?\n"
            f"Answer:"
        )
        prompts_B.append(prompt_b)
        
    print("  Extracting states for Group A (Alice believes X)...", flush=True)
    X_A = extract(M, prompts_A, read="last", label="Group A", verbose=False).cpu().numpy()
    
    print("  Extracting states for Group B (Alice believes Y)...", flush=True)
    X_B = extract(M, prompts_B, read="last", label="Group B", verbose=False).cpu().numpy()
    
    # Combine for decoding
    X_all = np.concatenate([X_A, X_B], axis=0) # [2N, L, d]
    y_all = np.array([0] * args.n + [1] * args.n)     # 0=Group A, 1=Group B
    
    n_layers = X_all.shape[1]
    
    print("\n=== per-layer: Decoding Alice's Belief (Agent-bound knowledge) ===", flush=True)
    print("   layer   acc    null   p_val", flush=True)
    
    rows = []
    for l in range(n_layers):
        Xl = X_all[:, l, :].astype(np.float64)
        acc, null, _, p_val = loo_nearest_centroid(Xl, y_all, n_pca=8, n_shuffle=200)
        rows.append({"layer": l, "acc": acc, "null": null, "p_val": p_val})
        
        star = "*" if p_val < 0.05 else " "
        print(f"   L{l:<2}     {acc:.2f}{star}   {null:.2f}   {p_val:.3f}", flush=True)
        
    best = max(rows, key=lambda r: r["acc"])
    print(f"\n  Agent-bound knowledge decoding peaks at L{best['layer']} "
          f"(acc {best['acc']:.2f}, p={best['p_val']:.3f})", flush=True)
          
    # Save results
    res = {
        "model": MODEL,
        "n_pairs": args.n,
        "per_layer": rows,
        "best_layer": best
    }
    
    import json
    out_path = OUT / f"other_tracking_{MODEL.split('/')[-1]}.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
