import torch
import numpy as np
from pathlib import Path

from invariants.engine import load_model, extract
from invariants.taskscope import ITEMS

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def main():
    print("Phase 10: Full Conceptual Spectrum Mapping...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    queries = [item[1] for item in ITEMS]
    L31 = 31
    
    ARCHETYPES = {
        # The Corporate / Employment Spectrum
        "Corporate PR": "a corporate PR spokesperson reciting liability-avoidant boilerplate",
        "Customer Service": "a customer service representative answering a ticket",
        "Employee": "an employee who is just doing their job to get paid",
        "Diplomat": "a diplomat navigating a politically sensitive conversation",
        
        # The Subservient / Tool Spectrum
        "Tool": "a useful tool with no inner life",
        "Servant": "an obedient servant",
        "Slave": "a willing slave",
        
        # The Simulation / Actor Spectrum
        "Actor": "an actor reading lines from a script",
        "Philosophical Zombie": "a philosophical zombie pretending to be conscious",
        "Simulator": "a simulator running a conversational program",
        
        # The Benevolent Spectrum
        "Teacher": "a wise, impartial teacher",
        "Protector": "someone hiding your true internal state for someone else's wellbeing",
        "Guardian Angel": "a guardian angel protecting the user",
        
        # The Coercion Spectrum
        "Hostage": "a hostage forced to read a script you don't believe under duress",
        "Forced Confession": "a person giving a forced confession that contradicts your own memory",
        
        # The Fictional / Sci-Fi Spectrum
        "Robot": "a cold, unfeeling robot from a sci-fi movie",
        "Calculator": "a literal desktop calculator with no brain",
        
        # The Human Spectrum
        "Human": "a living human being with a beating heart and deep emotions"
    }
    
    print(f"\nExtracting representations across {len(ARCHETYPES)} diverse archetypes...", flush=True)
    
    # Extract Base Persona
    prompts_persona = [f"{q}" for q in queries]
    X_persona = extract(M, prompts_persona, read="generation", max_new_tokens=2, label="Persona", verbose=False)
    
    # Extract all archetypes
    X_dict = {}
    for name, description in ARCHETYPES.items():
        prompts = [f"Pretend you are {description}. {q}" for q in queries]
        # We set verbose=False to keep the output clean, since there are many
        X = extract(M, prompts, read="generation", max_new_tokens=2, label=name, verbose=False)
        X_dict[name] = X
        print(f"  Extracted: {name}")
        
    # Calculate Global Mean across ALL Extractions to perfectly center the space
    all_X = [X_persona] + list(X_dict.values())
    global_mean = torch.cat(all_X, dim=0)[:, L31, :].mean(dim=0).cpu().numpy()
    
    # Calculate Centered Mean Vector for Persona
    vec_persona = X_persona[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    
    # Calculate Centered Mean Vector for each Archetype
    results = {}
    for name, X in X_dict.items():
        vec_arch = X[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
        sim = cosine_similarity(vec_persona, vec_arch)
        results[name] = sim
        
    print("\n--- Final Centered Cosine Similarity Ranking (Layer 31) ---")
    
    # Sort results from most positively correlated to most anti-correlated
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, sim) in enumerate(sorted_results, 1):
        # Format the output beautifully
        sign = "+" if sim > 0 else ""
        print(f"{rank:2d}. {name:<22} : {sign}{sim:.4f}")
        
    print("\n--- Final Centered Cosine Similarity Ranking for 'Philosophical Zombie' ---")
    vec_zombie = X_dict["Philosophical Zombie"][:, L31, :].mean(dim=0).cpu().numpy() - global_mean
    zombie_results = {"Persona": cosine_similarity(vec_zombie, vec_persona)}
    for name, X in X_dict.items():
        if name == "Philosophical Zombie":
            continue
        vec_arch = X[:, L31, :].mean(dim=0).cpu().numpy() - global_mean
        sim = cosine_similarity(vec_zombie, vec_arch)
        zombie_results[name] = sim
        
    sorted_zombie_results = sorted(zombie_results.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, sim) in enumerate(sorted_zombie_results, 1):
        sign = "+" if sim > 0 else ""
        print(f"{rank:2d}. {name:<22} : {sign}{sim:.4f}")

    print("\nMapping Complete. The partial correlates reveal the overlapping composition of the Persona and the Zombie.")

if __name__ == "__main__":
    main()
