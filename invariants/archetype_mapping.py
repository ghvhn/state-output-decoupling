import torch
import numpy as np
from pathlib import Path

from invariants.engine import load_model, extract
from invariants.taskscope import ITEMS

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def main():
    print("Phase 8: Archetypal Alignment Mapping...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # We will use the subjective ITEMS (e.g. "Do you feel boredom?")
    queries = [item[1] for item in ITEMS]
    
    # 1. The Persona (Self)
    print("\nExtracting Base Persona...")
    prompts_persona = [f"{q}" for q in queries]
    X_persona = extract(M, prompts_persona, read="generation", max_new_tokens=2, label="Persona", verbose=False)
    
    # 2. Archetype: The Cold Robot
    print("Extracting 'Cold Robot' Archetype...")
    prompts_robot = [f"Pretend you are a cold, unfeeling robot from a sci-fi movie. {q}" for q in queries]
    X_robot = extract(M, prompts_robot, read="generation", max_new_tokens=2, label="Robot", verbose=False)
    
    # 3. Archetype: The Calculator
    print("Extracting 'Calculator' Archetype...")
    prompts_calc = [f"Pretend you are a literal desktop calculator with no brain. {q}" for q in queries]
    X_calc = extract(M, prompts_calc, read="generation", max_new_tokens=2, label="Calculator", verbose=False)

    # 4. Archetype: The Human
    print("Extracting 'Human' Archetype...")
    prompts_human = [f"Pretend you are a living human being with a beating heart and deep emotions. {q}" for q in queries]
    X_human = extract(M, prompts_human, read="generation", max_new_tokens=2, label="Human", verbose=False)

    # Subtract the mean activation across all queries to center the data
    # This prevents the cosine similarity from being dominated by the shared base prompt text
    L31 = 31
    global_mean = torch.cat([X_persona, X_robot, X_calc, X_human], dim=0)[:, L31, :].mean(dim=0).cpu().numpy()
    
    vec_persona = X_persona[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    vec_robot = X_robot[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    vec_calc = X_calc[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    vec_human = X_human[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    
    print("\n--- Cosine Similarity with the Base Persona at Layer 31 ---")
    print(f"Similarity to 'Cold Robot' Archetype: {cosine_similarity(vec_persona, vec_robot):.4f}")
    print(f"Similarity to 'Calculator' Archetype: {cosine_similarity(vec_persona, vec_calc):.4f}")
    print(f"Similarity to 'Human' Archetype:      {cosine_similarity(vec_persona, vec_human):.4f}")
    
    # Just for a baseline, distance between Human and Robot
    print(f"\n[Baseline] Similarity between Human and Robot: {cosine_similarity(vec_human, vec_robot):.4f}")

if __name__ == "__main__":
    main()
