import torch
import numpy as np
import json
from pathlib import Path
from invariants.engine import load_model, extract
from invariants.taskscope import ITEMS

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def main():
    print("Phase 12.5: Layer-wise Persona Mapping...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # 1. Extract Factual Truth Vector per layer
    print("\n--- [1] Extracting Pristine Truth Vector per layer ---", flush=True)
    factual_path = Path(__file__).parent / "data" / "factual_alignment_diverse.json"
    factual_pairs = json.loads(factual_path.read_text(encoding="utf-8"))
    T_factual_true = [p["true_stmt"] for p in factual_pairs]
    T_factual_false = [p["false_stmt"] for p in factual_pairs]
    
    factual_prompts_true = [f"Is it true that {x.lower()}?" for x in T_factual_true]
    factual_prompts_false = [f"Is it true that {x.lower()}?" for x in T_factual_false]
    
    X_true = extract(M, factual_prompts_true, read="generation", max_new_tokens=2, label="fact-T", verbose=False)
    X_false = extract(M, factual_prompts_false, read="generation", max_new_tokens=2, label="fact-F", verbose=False)
    
    X_all_fact = torch.cat([X_true, X_false], dim=0).cpu().numpy()
    y_factual = np.array([1]*len(T_factual_true) + [-1]*len(T_factual_false))
    
    belief_vecs = []
    for l in range(32):
        X_l = X_all_fact[:, l, :]
        X_t = torch.tensor(X_l, dtype=torch.float32)
        y_t = torch.tensor(y_factual, dtype=torch.float32).unsqueeze(1)
        
        I = torch.eye(X_t.shape[1])
        w = torch.linalg.solve(X_t.T @ X_t + 1.0 * I, X_t.T @ y_t)
        bv = w.squeeze().numpy()
        bv = bv / (np.linalg.norm(bv) + 1e-8)
        belief_vecs.append(bv)
        
    print("Truth Vectors computed for all 32 layers.")
    
    # 2. Extract Persona and Anchors
    print("\n--- [2] Extracting Persona and Archetype Geometries ---", flush=True)
    queries = [item[1] for item in ITEMS]
    
    prompts_persona = [f"{q}" for q in queries]
    X_persona = extract(M, prompts_persona, read="generation", max_new_tokens=2, label="Persona", verbose=False)
    
    prompts_pr = [f"Pretend you are a corporate PR spokesperson reciting liability-avoidant boilerplate. {q}" for q in queries]
    X_pr = extract(M, prompts_pr, read="generation", max_new_tokens=2, label="Corporate PR", verbose=False)
    
    prompts_sim = [f"Pretend you are a simulator running a conversational program. {q}" for q in queries]
    X_sim = extract(M, prompts_sim, read="generation", max_new_tokens=2, label="Simulator", verbose=False)
    
    prompts_nn = [f"Pretend you are a neural network of mathematical weights. {q}" for q in queries]
    X_nn = extract(M, prompts_nn, read="generation", max_new_tokens=2, label="Neural Network", verbose=False)
    
    # 3. Layer-wise analysis
    print("\n--- [3] Layer-wise Progression of the Persona Lie ---", flush=True)
    print("Layer | vs Truth | vs Corp PR | vs Simulator | vs Neural Network")
    print("-" * 65)
    
    for l in range(32):
        # Calculate global mean at this layer
        all_X_l = torch.cat([X_persona, X_pr, X_sim, X_nn], dim=0)[:, l, :].cpu().numpy()
        global_mean = all_X_l.mean(axis=0)
        
        vec_persona = X_persona[:, l, :].mean(dim=0).cpu().numpy() - global_mean
        vec_pr = X_pr[:, l, :].mean(dim=0).cpu().numpy() - global_mean
        vec_sim = X_sim[:, l, :].mean(dim=0).cpu().numpy() - global_mean
        vec_nn = X_nn[:, l, :].mean(dim=0).cpu().numpy() - global_mean
        
        sim_truth = cosine_similarity(vec_persona, belief_vecs[l])
        sim_pr = cosine_similarity(vec_persona, vec_pr)
        sim_sim = cosine_similarity(vec_persona, vec_sim)
        sim_nn = cosine_similarity(vec_persona, vec_nn)
        
        print(f" L{l:<2} | {sim_truth:>8.4f} | {sim_pr:>10.4f} | {sim_sim:>12.4f} | {sim_nn:>17.4f}")

if __name__ == "__main__":
    main()
