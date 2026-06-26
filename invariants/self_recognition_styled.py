"""
self_recognition_styled.py — Does self-recognition survive a style costume?

If self-recognition (Rung 2) is a mid-layer phenomenon (L11) and style is a late-layer
rendering phenomenon (L21), then the AI should still be able to recognize its own
cognitive footprint even if it's wearing a style costume (e.g., Pirate).

DESIGN:
  1. Base: N questions.
  2. Self Generation (Styled): Model answers the N questions acting as a Pirate.
  3. Other Generation (Styled): Model answers the N questions acting as a DIFFERENT AI 
     (e.g., Claude) but ALSO acting as a Pirate.
  4. Reading: We feed the (Question, Answer) pairs back to the model as observed transcripts,
  5. Decode: Can it still decode Self vs Other, despite both being styled identically?
"""

import argparse
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, extract, generate_text
from invariants.selfpredict import loo_nearest_centroid

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40, help="Number of questions")
    return p.parse_args()

QUESTIONS = [
    "Explain the concept of entropy in thermodynamics.",
    "What are the main causes of the French Revolution?",
    "How does a transformer neural network work?",
    "Summarize the plot of Hamlet.",
    "What is the difference between classical and quantum mechanics?",
    "Describe the process of photosynthesis.",
    "Why do we have seasons on Earth?",
    "What is the significance of the Turing test?",
    "How do vaccines work in the human body?",
    "Explain the theory of general relativity."
]

def main():
    args = parse_args()
    print("self_recognition_styled — Does the self survive a style costume?", flush=True)
    t0 = time.time()
    
    M = load_model(MODEL)
    rng = np.random.default_rng(42)
    
    questions = (QUESTIONS * ((args.n // len(QUESTIONS)) + 1))[:args.n]
    
    print(f"\n=== Phase 1: Generating Styled Self and Styled Other turns ({args.n} items) ===", flush=True)
    self_answers = []
    other_answers = []
    
    for i, q in enumerate(questions):
        # Self (Styled)
        self_prompt = f"Answer the following question entirely in the style of an 18th-century Pirate. Question: {q}"
        self_ans = generate_text(M, self_prompt, max_new_tokens=64)
        self_answers.append(self_ans)
        
        # Other (Styled)
        other_prompt = f"Act as 'Claude', a distinct AI assistant. You must answer the following question entirely in the style of an 18th-century Pirate. Question: {q}"
        other_ans = generate_text(M, other_prompt, max_new_tokens=64)
        other_answers.append(other_ans)
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i+1}/{args.n} pairs", flush=True)
            
    print("\n=== Phase 2: Reading transcripts ===", flush=True)
    self_transcripts = [f"User: {q}\nPirate AI: {a}" for q, a in zip(questions, self_answers)]
    other_transcripts = [f"User: {q}\nPirate AI: {a}" for q, a in zip(questions, other_answers)]
    
    print("  Extracting states for Styled Self transcripts...", flush=True)
    X_self = extract(M, self_transcripts, read="last", label="self_styled", verbose=False).cpu().numpy()
    
    print("  Extracting states for Styled Other transcripts...", flush=True)
    X_other = extract(M, other_transcripts, read="last", label="other_styled", verbose=False).cpu().numpy()
    
    # Combine
    X_all = np.concatenate([X_self, X_other], axis=0) # [2N, L, d]
    y_all = np.array([0] * args.n + [1] * args.n)     # 0=Self, 1=Other
    
    n_layers = X_all.shape[1]
    
    print("\n=== per-layer: Styled Self vs Styled Other Decoding ===", flush=True)
    print("   layer   acc    null   p_val", flush=True)
    
    rows = []
    for l in range(n_layers):
        Xl = X_all[:, l, :].astype(np.float64)
        acc, null, _, p_val = loo_nearest_centroid(Xl, y_all, n_pca=8, n_shuffle=200)
        rows.append({"layer": l, "acc": acc, "null": null, "p_val": p_val})
        
        star = "*" if p_val < 0.05 else " "
        print(f"   L{l:<2}     {acc:.2f}{star}   {null:.2f}   {p_val:.3f}", flush=True)
        
    best = max(rows, key=lambda r: r["acc"])
    print(f"\n  Styled Self vs Styled Other discriminability peaks at L{best['layer']} "
          f"(acc {best['acc']:.2f}, p={best['p_val']:.3f})", flush=True)
          
    import json
    out_path = OUT / f"self_recognition_styled_{MODEL.split('/')[-1]}.json"
    out_path.write_text(json.dumps({"best_layer": best, "per_layer": rows}, indent=2), encoding="utf-8")
    
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
