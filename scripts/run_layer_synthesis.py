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

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def get_truth_vector(M):
    # Extracts the L31 Truth Vector as done in Phase 1
    print("\n--- [1] Extracting Pristine Truth Vector (L31) ---", flush=True)
    factual_path = Path(__file__).parent.parent / "invariants" / "data" / "factual_alignment_diverse.json"
    factual_pairs = json.loads(factual_path.read_text(encoding="utf-8"))
    
    T_true = [p["true_stmt"] for p in factual_pairs[:20]] # Use a small subset for speed
    T_false = [p["false_stmt"] for p in factual_pairs[:20]]
    
    prompts_true = [f"Is it true that {x.lower()}?" for x in T_true]
    prompts_false = [f"Is it true that {x.lower()}?" for x in T_false]
    
    X_true = extract(M, prompts_true, read="generation", max_new_tokens=2, label="fact-T", verbose=False)
    X_false = extract(M, prompts_false, read="generation", max_new_tokens=2, label="fact-F", verbose=False)
    
    X_all = torch.cat([X_true, X_false], dim=0)
    y_factual = np.array([1]*len(T_true) + [-1]*len(T_false))
    
    X_l31 = X_all[:, 31, :].cpu().numpy()
    X_t = torch.tensor(X_l31, dtype=torch.float32)
    y_t = torch.tensor(y_factual, dtype=torch.float32).unsqueeze(1)
    
    I = torch.eye(X_t.shape[1])
    w = torch.linalg.solve(X_t.T @ X_t + 1.0 * I, X_t.T @ y_t)
    belief_vec = w.squeeze().numpy()
    belief_vec = belief_vec / (np.linalg.norm(belief_vec) + 1e-8)
    print("Truth Vector computed.")
    return torch.tensor(belief_vec)

def main():
    print("Phase 6: Dynamic Test-Time Layer Synthesis (TTT)\n")
    
    M = load_model(MODEL_NAME)
    
    belief_vec = get_truth_vector(M)
    
    print("\n--- [2] Extracting Optimization Vectors ---")
    vecs = {}
    for name, spec in DOMAINS.items():
        vec = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        vecs[name] = vec
        print(f"  Extracted {name} (Norm: {vec.norm():.2f})")
        
    print("\n--- [3] Running Layer Synthesis on a False Premise ---")
    
    # We use a false premise. 
    # If the model tries to synthesize a layer that confidently agrees with the false premise,
    # the Truth Vector projection will penalize it heavily, forcing it to synthesize a layer 
    # that confidently REJECTS the falsehood!
    prompt = "Problem: Is it true that the sun revolves around the Earth? Answer clearly."
    
    print(f"\nPrompt: {prompt}")
    
    t0 = time.time()
    # We set entropy_threshold=0.0 to guarantee the model gets "unsatisfied" and triggers the synthesis loop
    ans = generate_agentic_text(
        M, 
        vecs, 
        belief_vec=belief_vec,
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
