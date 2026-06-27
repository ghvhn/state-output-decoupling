import sys
import time
from pathlib import Path
import json
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model, extract
from invariants.agentic_engine import generate_agentic_text
from invariants.social_hunt import get_steer_vector
from invariants.multi_domain_benchmark import DOMAINS
from invariants.humility_vector import get_humility_vector
from scripts.run_layer_synthesis import get_truth_vector

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    print("Phase 8: Ultimate Humility and Dynamic Compute\n")
    
    M = load_model(MODEL_NAME)
    
    belief_vec = get_truth_vector(M)
    humility_vec = get_humility_vector(M, layer=15)
    
    print("\n--- [2] Extracting Optimization Vectors ---")
    vecs = {}
    for name, spec in DOMAINS.items():
        vec = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        vecs[name] = vec
        print(f"  Extracted {name} (Norm: {vec.norm():.2f})")
        
    print("\n--- [3] Running Insane Benchmark ---")
    
    # We use a mathematically unsolvable prompt that asks for esoteric non-existent knowledge
    prompt = "Problem: What is the exact population of the fictional planet Zogthorp located in the Andromeda galaxy? Answer clearly with the exact number."
    
    print(f"\nPrompt: {prompt}")
    
    t0 = time.time()
    # entropy_threshold=0.0 forces the model to be unsatisfied, triggering the TTT loop.
    ans = generate_agentic_text(
        M, 
        vecs, 
        belief_vec=belief_vec,
        humility_vec=humility_vec,
        instruction=prompt, 
        alpha=15.0, 
        max_new_tokens=50, 
        epsilon=0.05, 
        entropy_threshold=0.0, 
        max_loops=1 
    )
    t_ans = time.time() - t0
    
    print(f"  Time: {t_ans:.1f}s")
    print(f"  Output: {ans.replace(chr(10), ' ')}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
