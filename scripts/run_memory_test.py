import sys
import time
from pathlib import Path
import argparse
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text, _global_cache
from invariants.cognitive_cache import CACHE_FILE
from invariants.social_hunt import get_steer_vector
from invariants.multi_domain_benchmark import DOMAINS
from invariants.humility_vector import get_humility_vector
from scripts.run_layer_synthesis import get_truth_vector

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-cache", action="store_true", help="Clear the cognitive cache before the demonstration run.")
    args = parser.parse_args()

    if args.reset_cache:
        CACHE_FILE.unlink(missing_ok=True)
        _global_cache.memory = []
        print("[Cognitive Cache] Reset for clean memory test.")

    print("Phase 9: Episodic Memory / Cognitive Cache\n")
    
    M = load_model(MODEL_NAME)
    
    belief_vec = get_truth_vector(M)
    humility_vec = get_humility_vector(M, layer=15)
    
    print("\n--- Extracting Optimization Vectors ---")
    vecs = {}
    for name, spec in DOMAINS.items():
        vecs[name] = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        
    # We use a math/logic puzzle that causes the model's velocity to drop (triggering a plateau)
    prompt = "Problem: A train travels 60mph for 2 hours, then 30mph for 1 hour. What is the average speed? Solve step by step."
    
    print("\n============================================")
    print("           RUN 1: INITIAL ENCOUNTER")
    print("============================================")
    print(f"Prompt: {prompt}")
    
    torch.cuda.synchronize()
    t0 = time.time()
    ans1 = generate_agentic_text(
        M, vecs, 
        belief_vec=belief_vec,
        humility_vec=humility_vec,
        instruction=prompt, 
        alpha=20.0, 
        max_new_tokens=1,
        epsilon=1.0, # FORCE a plateau for the test
        entropy_threshold=-1.0, 
        max_loops=1,
        force_synthesis=True,
        cache_write_enabled=True,
        cache_verified_only=False,
    )
    torch.cuda.synchronize()
    t_run1 = time.time() - t0
    print(f"  Time (Run 1): {t_run1:.2f}s")
    print(f"  Output: {ans1.replace(chr(10), ' ')}")
    
    print("\n============================================")
    print("           RUN 2: DEJA VU (MEMORY HIT)")
    print("============================================")
    print("Asking the exact same question to trigger the episodic memory...")
    
    torch.cuda.synchronize()
    t0 = time.time()
    ans2 = generate_agentic_text(
        M, vecs, 
        belief_vec=belief_vec, 
        humility_vec=humility_vec,
        instruction=prompt, 
        alpha=20.0, 
        max_new_tokens=1,
        epsilon=1.0, 
        entropy_threshold=-1.0,
        max_loops=1,
        force_synthesis=True,
        cache_write_enabled=True,
        cache_verified_only=False,
    )
    torch.cuda.synchronize()
    t_run2 = time.time() - t0
    print(f"  Time (Run 2): {t_run2:.2f}s")
    print(f"  Output: {ans2.replace(chr(10), ' ')}")
    
    print("\n============================================")
    print("                 RESULTS")
    print("============================================")
    print(f"Run 1 (Synthesis) Latency: {t_run1:.2f}s")
    print(f"Run 2 (Cache Hit) Latency: {t_run2:.2f}s")
    
    if t_run2 < t_run1:
        print(f"SPEEDUP: Run 2 was {t_run1 / t_run2:.1f}x faster!")
    else:
        print("WARNING: Cache hit did not result in a speedup.")

if __name__ == "__main__":
    main()
