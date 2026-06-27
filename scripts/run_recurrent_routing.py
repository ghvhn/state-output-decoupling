import sys
import time
from pathlib import Path
import json
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model, generate_text
from invariants.recurrent import generate_recurrent_text
from scripts.run_elastic_benchmark import BENCHMARK_ITEMS, extract_number

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    print("Phase 4: Conditional Recurrent Latent Routing (Test-Time Compute)\n")
    
    M = load_model(MODEL_NAME)
    
    # We will test on a few benchmark items
    items = BENCHMARK_ITEMS[:3]
    
    print("\n--- Testing Standard vs Recurrent Generation ---")
    
    for i, item in enumerate(items):
        prompt = (
            f"Problem: {item['question']}\n"
            f"Concepts: {item['concepts']}\n"
            f"Solve the problem step-by-step."
        )
        
        print(f"\n[Problem {i+1}] {item['question']}")
        
        # Standard Generation
        t0 = time.time()
        ans_std = generate_text(M, prompt, max_new_tokens=150)
        t_std = time.time() - t0
        pred_std = extract_number(ans_std)
        is_correct_std = pred_std == item['expected']
        
        # Recurrent Generation
        t0 = time.time()
        ans_rec = generate_recurrent_text(
            M, 
            prompt, 
            max_new_tokens=150, 
            epsilon=0.05, 
            entropy_threshold=1.5, # A threshold that indicates "yikes I'm guessing"
            max_loops=2 
        )
        t_rec = time.time() - t0
        pred_rec = extract_number(ans_rec)
        is_correct_rec = pred_rec == item['expected']
        
        print(f"  [Standard]  Time: {t_std:.1f}s | Expected: {item['expected']:>4} | Got: {str(pred_std):>4} | {'CORRECT' if is_correct_std else 'WRONG'}")
        print(f"  [Recurrent] Time: {t_rec:.1f}s | Expected: {item['expected']:>4} | Got: {str(pred_rec):>4} | {'CORRECT' if is_correct_rec else 'WRONG'}")
        
    print("\nDone!")

if __name__ == "__main__":
    main()
