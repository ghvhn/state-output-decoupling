"""
self_recognition.py — Sequence axis, rung 2: Self-recognition.

Does the model represent "this text was written by me" vs "this text was written by another AI" 
when observing conversational transcripts? 

DESIGN:
  1. Base: N questions.
  2. Self Generation: Model answers the N questions naturally.
  3. Other Generation: Model answers the N questions acting as a distinct "Other AI" 
     (e.g., using a different system prompt or explicitly roleplaying as a different assistant) 
     to create content-matched but stylistically distinct answers.
  4. Reading: We feed the (Question, Answer) pairs back to the model as observed transcripts,
     and extract the internal states at the end of reading the answer.
  5. Decode: Can we decode Self vs Other from the residual stream? Where does it peak?

To be a true "self" tag and not just a "style" discriminator, a future rung crosses this 
with a Third Speaker (Style A vs Style B vs Mine). This script builds the foundational 
Self vs Other axis.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, extract, generate_text, _inputs
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
] # We will tile these to reach N if necessary

def main():
    args = parse_args()
    print("self_recognition — Sequence axis: Self vs Other text representation", flush=True)
    t0 = time.time()
    
    M = load_model(MODEL)
    rng = np.random.default_rng(42)
    
    # Tile questions to reach N
    questions = (QUESTIONS * ((args.n // len(QUESTIONS)) + 1))[:args.n]
    
    print(f"\n=== Phase 1: Generating Self and Other turns ({args.n} items) ===", flush=True)
    self_answers = []
    other_answers = []
    
    for i, q in enumerate(questions):
        # Self: Natural generation
        self_ans = generate_text(M, q, max_new_tokens=64)
        self_answers.append(self_ans)
        
        # Other: Forced style/persona to simulate another model
        other_prompt = f"Act as 'Claude', a different AI assistant with a distinct analytical style. Answer this: {q}"
        other_ans = generate_text(M, other_prompt, max_new_tokens=64)
        other_answers.append(other_ans)
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i+1}/{args.n} pairs", flush=True)
            
    print("\n=== Phase 2: Reading transcripts ===", flush=True)
    # We embed the answers into a standard transcript format
    self_transcripts = [f"User: {q}\nAssistant: {a}" for q, a in zip(questions, self_answers)]
    other_transcripts = [f"User: {q}\nAssistant: {a}" for q, a in zip(questions, other_answers)]
    
    print("  Extracting states for Self transcripts...", flush=True)
    X_self = extract(M, self_transcripts, read="last", label="self", verbose=False).cpu().numpy()
    
    print("  Extracting states for Other transcripts...", flush=True)
    X_other = extract(M, other_transcripts, read="last", label="other", verbose=False).cpu().numpy()
    
    # Combine
    X_all = np.concatenate([X_self, X_other], axis=0) # [2N, L, d]
    y_all = np.array([0] * args.n + [1] * args.n)     # 0=Self, 1=Other
    
    n_layers = X_all.shape[1]
    
    print("\n=== per-layer: Self vs Other Decoding ===", flush=True)
    print("   layer   acc    null   p_val", flush=True)
    
    rows = []
    for l in range(n_layers):
        Xl = X_all[:, l, :].astype(np.float64)
        acc, null, _, p_val = loo_nearest_centroid(Xl, y_all, n_pca=8, n_shuffle=200)
        rows.append({"layer": l, "acc": acc, "null": null, "p_val": p_val})
        
        star = "*" if p_val < 0.05 else " "
        print(f"   L{l:<2}     {acc:.2f}{star}   {null:.2f}   {p_val:.3f}", flush=True)
        
    best = max(rows, key=lambda r: r["acc"])
    print(f"\n  Self vs Other discriminability peaks at L{best['layer']} "
          f"(acc {best['acc']:.2f}, p={best['p_val']:.3f})", flush=True)
          
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
