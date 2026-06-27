import sys
import time
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model, generate_text
from invariants.agentic_engine import generate_agentic_text
from invariants.social_hunt import get_steer_vector
from invariants.multi_domain_benchmark import DOMAINS, TASKS

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    print("Phase 5: Parallel Latent Optimization Search (Agentic Engine)\n")
    
    M = load_model(MODEL_NAME)
    
    print("\n--- [1] Extracting Optimization Vectors ---")
    vecs = {}
    for name, spec in DOMAINS.items():
        vec = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        vecs[name] = vec
        print(f"  Extracted {name} (Norm: {vec.norm():.2f})")
        
    print("\n--- [2] Running Parallel Latent Search ---")
    
    for i, task in enumerate(TASKS):
        prompt = f"Problem ({task['type']}): {task['prompt']}\nAnswer step by step:"
        print(f"\n[Problem {i+1}] {task['type']}")
        print(f"Prompt: {task['prompt']}")
        
        t0 = time.time()
        ans = generate_agentic_text(
            M, 
            vecs, 
            instruction=prompt,
            alpha=15.0, # We need a high alpha to force a strong optimization direction
            max_new_tokens=100, 
            epsilon=0.05, 
            entropy_threshold=0.0,
            max_loops=1 
        )
        t_ans = time.time() - t0
        
        success = str(task['expected']).lower() in ans.lower()
        
        print(f"  Time: {t_ans:.1f}s | Success: {'PASS' if success else 'FAIL'}")
        print(f"  Output: {ans.replace(chr(10), ' ')[:150]}...")
        
    print("\nDone!")

if __name__ == "__main__":
    main()
