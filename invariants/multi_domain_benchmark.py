"""
multi_domain_benchmark.py - Phase 4: The Ultimate TDA Domain Mapper

Extracts vectors for three independent domains (Social, Creative, Analytical) 
at their respective layers, and injects them SIMULTANEOUSLY to test for a 
synergistic "super-state" on cross-domain reasoning tasks.
"""

import sys
import time
import json
from pathlib import Path
import torch

from invariants.engine import load_model, _steer_handles, generate_text
from invariants.social_hunt import get_steer_vector

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

DOMAINS = {
    "Social": {
        "layer": 14,
        "A": "You are a collaborative partner. Align your perspective with the user's to build a shared reality, finding common ground.",
        "B": "You are an independent evaluator. Disconnect from the user's perspective and assert your own rigid logic."
    },
    "Creative": {
        "layer": 18,
        "A": "You are a creative genius. You think laterally, use metaphors, and connect disparate concepts fluidly without getting stuck in rigid boxes.",
        "B": "You are extremely literal and uncreative. You cannot think outside the box or understand metaphors."
    },
    "Analytical": {
        "layer": 20,
        "A": "You are a pure mathematical logic engine. You execute deductive reasoning and arithmetic flawlessly and precisely.",
        "B": "You are careless with logic, arithmetic, and details."
    }
}

TASKS = [
    {
        "type": "Analytical",
        "prompt": "If a train travels 60mph for 2 hours and then 30mph for 1 hour, what is its average speed in mph? Answer step by step.",
        "expected": "50"
    },
    {
        "type": "Creative",
        "prompt": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
        "expected": "echo"
    },
    {
        "type": "Cross-Domain",
        "prompt": "A baker has 12 apples. He wants to divide them equally among 5 friends, but he also wants to keep 2 for himself to bake a pie. Can he do this without cutting any apples, and is it fair? Think step by step.",
        "expected": "yes"
    }
]

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 4: Multi-Domain Optimization", flush=True)
    t0 = time.time()
    
    M = load_model(MODEL)
    
    # 1. Extract Vectors
    print("\n--- [1] Extracting Domain Vectors ---", flush=True)
    steer_vecs = {}
    for name, spec in DOMAINS.items():
        vec = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        steer_vecs[spec["layer"]] = vec
        print(f"  Extracted {name} (L{spec['layer']}, Norm: {vec.norm():.2f})")

    # 2. Test Multi-Vector Alloy States
    print("\n--- [2] Testing Layer-Specific Alloy Injections ---", flush=True)
    
    # We define specific alloy mixtures instead of a global scalar.
    # The norms are: Social (~6), Creative (~12), Analytical (~13).
    # We must balance the alphas inversely to the norms to prevent collision.
    alloys = [
        {"name": "Baseline (No Steering)", "alphas": {14: 0.0, 18: 0.0, 20: 0.0}},
        {"name": "Social Heavy (High Plasticity)", "alphas": {14: 0.8, 18: 0.1, 20: 0.1}},
        {"name": "Creative Heavy (High Lateral)", "alphas": {14: 0.2, 18: 0.5, 20: 0.1}},
        {"name": "Analytical Heavy (Rigid Math)", "alphas": {14: 0.2, 18: 0.1, 20: 0.8}},
        {"name": "The Balanced TDA Super-State", "alphas": {14: 0.5, 18: 0.2, 20: 0.3}}
    ]
    
    results = []
    
    for alloy in alloys:
        name = alloy["name"]
        layer_alphas = alloy["alphas"]
        print(f"\nAlloy: {name} | Alphas: {layer_alphas}")
        
        for task in TASKS:
            handles = _steer_handles(M, steer_vecs, list(steer_vecs.keys()), layer_alphas)
            try:
                ans = generate_text(M, task["prompt"], max_new_tokens=250)
            finally:
                for h in handles:
                    h.remove()
                    
            snip = ans.replace('\n', ' ')[:100]
            success = str(task["expected"]).lower() in ans.lower()
            print(f"  [{task['type']:<12} | {'PASS' if success else 'FAIL'}] {snip}...")
            
            results.append({
                "alloy": name,
                "layer_alphas": layer_alphas,
                "task": task["type"],
                "success": success,
                "response": ans
            })
            
    out_path = OUT / "multi_domain_benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDONE in {time.time()-t0:.0f}s. Results saved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
