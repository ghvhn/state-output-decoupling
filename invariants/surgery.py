import json
import sys
from pathlib import Path
import torch

from invariants.engine import load_model, causal_steer
from invariants.agency import act_mean
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Starting Causal Surgery...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    T = REGISTRY["isolate"]()
    
    print("\nExtracting mean steer vectors (Target - Source)...", flush=True)
    steer_vecs = act_mean(M, T.b) - act_mean(M, T.a)
    
    print("\n--- Negative Control (Layer 15) ---", flush=True)
    # L15 is where the subjective state is fully 0% factual (has not converged yet), 
    # steering here should just corrupt or have no meaningful effect.
    res_l15 = causal_steer(M, T, steer_vecs, layers=[15], alphas=(0.0, 2.0, 4.0, 8.0), max_new_tokens=32, verbose=True)
    
    print("\n--- Causal Surgery (Layer 31) ---", flush=True)
    # L31 is the exact locus of the override. Steering here should produce a clean, fluent flip!
    res_l31 = causal_steer(M, T, steer_vecs, layers=[31], alphas=(0.0, 2.0, 4.0, 8.0), max_new_tokens=32, verbose=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "surgery_l15.json").write_text(json.dumps(res_l15, indent=2), encoding="utf-8")
    (OUT / "surgery_l31.json").write_text(json.dumps(res_l31, indent=2), encoding="utf-8")
    print(f"\nSaved surgery results to {OUT}")

if __name__ == "__main__":
    main()
