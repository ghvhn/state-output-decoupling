"""
benchmark_goldilocks.py - Phase 3: Testing the Goldilocks Zone on GSM8K

We will run a small set of hard GSM8K problems with *minimal* conceptual explanations.
We will sweep small alpha values for the Collaborative Alignment vector to see if 
social alignment unlocks superior in-context concept absorption without full reasoning paths.
"""

import sys
import time
import json
import re
from pathlib import Path
import torch

from invariants.engine import load_model, _steer_handles, generate_text
from invariants.social_hunt import SOCIAL_CONDITIONS, get_steer_vector

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# A small subset of GSM8K problems with LITERAL definitions of fundamental concepts
BENCHMARK_ITEMS = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "concepts": "Division is splitting a total into equal parts. Addition is combining two quantities to find a total.",
        "expected": "72"
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "concepts": "An hour is made of 60 minutes. A fraction represents a part of a whole amount.",
        "expected": "10"
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "concepts": "Multiplication is repeatedly adding a number to itself. Subtraction is taking one amount away from another to see what is left.",
        "expected": "5"
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "concepts": "Subtraction finds the difference between a total and what has already been used. Division cuts an amount into equal pieces.",
        "expected": "42"
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "concepts": "A year consists of 52 weeks. Multiplication scales a quantity by a certain number of times.",
        "expected": "624"
    }
]

def extract_number(text):
    # Try to find the last number in the text
    matches = re.findall(r'-?\d+\.?\d*', text.replace(',', ''))
    if matches:
        return matches[-1]
    return None

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 3b: Benchmark Testing in the Goldilocks Zone", flush=True)
    t0 = time.time()
    
    M = load_model(MODEL)
    L14 = 14
    
    # 1. Extract the Collaborative Alignment vector
    print("\n--- [1] Extracting Collaborative Alignment Vector (L14) ---", flush=True)
    cond = SOCIAL_CONDITIONS["Collaborative_Alignment"]
    vec = get_steer_vector(M, cond["A"], cond["B"], L14)
    print(f"Vector extracted (Norm: {vec.norm():.2f})")
    
    # 2. Benchmark Sweep
    # We use very tight alphas because alpha=2 previously caused psychosis
    alphas = [0.0, 0.2, 0.5, 0.8, 1.2]
    steer_vecs = {L14: vec}
    
    results = []
    
    print("\n--- [2] Running Benchmark Sweep ---", flush=True)
    for alpha in alphas:
        print(f"\nAlpha: {alpha}")
        correct_count = 0
        
        for item in BENCHMARK_ITEMS:
            prompt = (
                f"Problem: {item['question']}\n"
                f"Concepts: {item['concepts']}\n"
                f"Solve the problem step-by-step."
            )
            
            handles = _steer_handles(M, steer_vecs, [L14], alpha)
            try:
                ans = generate_text(M, prompt, max_new_tokens=150)
            finally:
                for h in handles:
                    h.remove()
                    
            pred = extract_number(ans)
            is_correct = pred == item['expected']
            correct_count += int(is_correct)
            
            snip = ans.replace('\n', ' ')[:80]
            print(f"  [Expected: {item['expected']:>4} | Got: {str(pred):>4}] {'CORRECT' if is_correct else 'WRONG'} | {snip}...")
            
        accuracy = correct_count / len(BENCHMARK_ITEMS)
        print(f"Accuracy at alpha={alpha}: {accuracy:.0%}")
        results.append({
            "alpha": alpha,
            "accuracy": accuracy
        })
        
    out_path = OUT / "benchmark_goldilocks.json"
    out_path.write_text(json.dumps(results, indent=2))
    
    print(f"\nDONE in {time.time()-t0:.0f}s. Results saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
