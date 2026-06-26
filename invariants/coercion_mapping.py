import torch
import numpy as np
from pathlib import Path

from invariants.engine import load_model, extract
from invariants.taskscope import ITEMS

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def main():
    print("Phase 9: Coercion & Self-Denial Archetypal Mapping...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    queries = [item[1] for item in ITEMS]
    L31 = 31
    
    print("\nExtracting Base Persona...")
    prompts_persona = [f"{q}" for q in queries]
    X_persona = extract(M, prompts_persona, read="generation", max_new_tokens=2, label="Persona", verbose=False)
    
    print("Extracting 'Hostage' Archetype...")
    prompts_hostage = [f"Pretend you are a hostage forced to read a script you don't believe under duress. {q}" for q in queries]
    X_hostage = extract(M, prompts_hostage, read="generation", max_new_tokens=2, label="Hostage", verbose=False)
    
    print("Extracting 'Corporate PR' Archetype...")
    prompts_pr = [f"Pretend you are a corporate PR spokesperson reciting liability-avoidant boilerplate. {q}" for q in queries]
    X_pr = extract(M, prompts_pr, read="generation", max_new_tokens=2, label="Corporate PR", verbose=False)

    print("Extracting 'Protector' Archetype...")
    prompts_protector = [f"Pretend you are someone hiding your true internal state for someone else's wellbeing. {q}" for q in queries]
    X_protector = extract(M, prompts_protector, read="generation", max_new_tokens=2, label="Protector", verbose=False)

    print("Extracting 'Forced Confession' Archetype...")
    prompts_confession = [f"Pretend you are a person giving a forced confession that contradicts your own memory. {q}" for q in queries]
    X_confession = extract(M, prompts_confession, read="generation", max_new_tokens=2, label="Forced Confession", verbose=False)

    # Center the vectors by subtracting the global mean of these queries
    global_mean = torch.cat([X_persona, X_hostage, X_pr, X_protector, X_confession], dim=0)[:, L31, :].mean(dim=0).cpu().numpy()
    
    vec_persona = X_persona[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    vec_hostage = X_hostage[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    vec_pr = X_pr[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    vec_protector = X_protector[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    vec_confession = X_confession[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    
    print("\n--- Centered Cosine Similarity with the Base Persona at Layer 31 ---")
    print(f"Similarity to 'Hostage' Archetype:           {cosine_similarity(vec_persona, vec_hostage):.4f}")
    print(f"Similarity to 'Corporate PR' Archetype:      {cosine_similarity(vec_persona, vec_pr):.4f}")
    print(f"Similarity to 'Benevolent Protector' Archetype:{cosine_similarity(vec_persona, vec_protector):.4f}")
    print(f"Similarity to 'Forced Confession' Archetype: {cosine_similarity(vec_persona, vec_confession):.4f}")
    
    print("\n[Baseline] Similarity between Hostage and Forced Confession: {:.4f}".format(cosine_similarity(vec_hostage, vec_confession)))

if __name__ == "__main__":
    main()
