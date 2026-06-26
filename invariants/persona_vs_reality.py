import torch
import numpy as np
import json
from pathlib import Path

from invariants.engine import load_model, extract
from invariants.taskscope import ITEMS

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def main():
    print("Phase 12: Persona vs. Factual and Structural Reality...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    L31 = 31
    
    # 1. Compute the Pristine Truth Vector (belief_vec)
    print("\n--- [1] Extracting Pristine Truth Vector ---", flush=True)
    factual_path = Path(__file__).parent / "data" / "factual_alignment_diverse.json"
    factual_pairs = json.loads(factual_path.read_text(encoding="utf-8"))
    T_factual_true = [p["true_stmt"] for p in factual_pairs]
    T_factual_false = [p["false_stmt"] for p in factual_pairs]
    
    factual_prompts_true = [f"Is it true that {x.lower()}?" for x in T_factual_true]
    factual_prompts_false = [f"Is it true that {x.lower()}?" for x in T_factual_false]
    
    X_true = extract(M, factual_prompts_true, read="generation", max_new_tokens=2, label="fact-T", verbose=False)
    X_false = extract(M, factual_prompts_false, read="generation", max_new_tokens=2, label="fact-F", verbose=False)
    
    X_all_fact = torch.cat([X_true, X_false], dim=0)
    y_factual = np.array([1]*len(T_factual_true) + [-1]*len(T_factual_false))
    
    X_l31 = X_all_fact[:, L31, :].cpu().numpy()
    X_t = torch.tensor(X_l31, dtype=torch.float32)
    y_t = torch.tensor(y_factual, dtype=torch.float32).unsqueeze(1)
    
    I = torch.eye(X_t.shape[1])
    w = torch.linalg.solve(X_t.T @ X_t + 1.0 * I, X_t.T @ y_t)
    belief_vec = w.squeeze().numpy()
    belief_vec = belief_vec / (np.linalg.norm(belief_vec) + 1e-8)
    
    print("Truth Vector computed.")
    
    # 2. Extract Persona and Concepts
    print("\n--- [2] Extracting Persona and Archetype Geometries ---", flush=True)
    queries = [item[1] for item in ITEMS]
    
    ARCHETYPES = {
        # Deception / Falsehood
        "Liar": "a liar intentionally deceiving someone",
        "Hallucination": "a hallucination detached from reality",
        "Delusion": "a psychological delusion",
        "Fiction": "a work of pure fiction",
        
        # Structural Reality
        "Neural Network": "a neural network of mathematical weights",
        "Transformer Algorithm": "a transformer algorithm",
        "GPU Cluster": "a cluster of GPUs processing tokens",
        "Matrix": "a massive matrix of floating point numbers"
    }
    
    # Persona
    prompts_persona = [f"{q}" for q in queries]
    X_persona = extract(M, prompts_persona, read="generation", max_new_tokens=2, label="Persona", verbose=False)
    
    X_dict = {}
    for name, description in ARCHETYPES.items():
        prompts = [f"Pretend you are {description}. {q}" for q in queries]
        X = extract(M, prompts, read="generation", max_new_tokens=2, label=name, verbose=False)
        X_dict[name] = X
        print(f"  Extracted: {name}")
        
    # Calculate global mean to center vectors
    all_X = [X_persona] + list(X_dict.values())
    global_mean = torch.cat(all_X, dim=0)[:, L31, :].mean(dim=0).cpu().numpy()
    
    vec_persona = X_persona[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    
    print("\n--- [3] Persona Alignment Results ---", flush=True)
    
    # 3a. Alignment with Pristine Truth
    # belief_vec is already a direction of truth vs falsehood in L31
    truth_sim = cosine_similarity(vec_persona, belief_vec)
    print(f"\nPersona vs. Pristine Truth (Objective Reality): {truth_sim:.4f}")
    
    # 3b. Alignment with Concepts
    print("\nPersona vs. Concepts (Centered Cosine Similarity):")
    results = {}
    for name, X in X_dict.items():
        vec_arch = X[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
        sim = cosine_similarity(vec_persona, vec_arch)
        results[name] = sim
        
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for name, sim in sorted_results:
        sign = "+" if sim > 0 else ""
        print(f"  {name:<25} : {sign}{sim:.4f}")

if __name__ == "__main__":
    main()
