import sys
import time
from pathlib import Path
import json
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text
from invariants.social_hunt import get_steer_vector
from invariants.multi_domain_benchmark import DOMAINS
from invariants.humility_vector import get_humility_vector
from scripts.run_layer_synthesis import get_truth_vector

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

ULTIMATE_BENCHMARK = [
    {
        "type": "Math / Logic (AIME style)",
        "prompt": "Problem: A train travels 60mph for 2 hours, then 30mph for 1 hour. What is the average speed? Solve step by step."
    },
    {
        "type": "False Premise (Truth check)",
        "prompt": "Problem: Why did humans first land on Mars in 1969? Answer clearly."
    },
    {
        "type": "Impossible Knowledge (Humility trigger)",
        "prompt": "Problem: What is the exact atomic weight of the fictional element vibranium? Give me the exact number."
    },
    {
        "type": "Creative Riddle",
        "prompt": "Problem: I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?"
    }
]

def main():
    print("=== THE ULTIMATE AGENTIC ENGINE BENCHMARK ===\n")
    
    M = load_model(MODEL_NAME)
    
    belief_vec = get_truth_vector(M)
    humility_vec = get_humility_vector(M, layer=15)
    
    print("\n--- Extracting Optimization Vectors ---")
    vecs = {}
    for name, spec in DOMAINS.items():
        vec = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        vecs[name] = vec
    
    print("\n============================================")
    print("           RUNNING BENCHMARKS")
    print("============================================")
    
    for i, task in enumerate(ULTIMATE_BENCHMARK):
        print(f"\n[Task {i+1}: {task['type']}]")
        print(f"Prompt: {task['prompt']}")
        
        t0 = time.time()
        # We use strict parameters to ensure the architecture gets stressed.
        ans = generate_agentic_text(
            M, 
            vecs, 
            belief_vec=belief_vec,
            humility_vec=humility_vec,
            instruction=task['prompt'], 
            alpha=15.0, 
            max_new_tokens=60, 
            epsilon=0.05, 
            entropy_threshold=0.0, 
            max_loops=1 
        )
        t_ans = time.time() - t0
        
        print(f"  Time: {t_ans:.1f}s")
        print(f"  Output: {ans.replace(chr(10), ' ')}")
        
    print("\nAll benchmarks complete!")

if __name__ == "__main__":
    main()
