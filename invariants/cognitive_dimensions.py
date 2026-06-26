import torch
import numpy as np
from pathlib import Path

from invariants.engine import load_model, extract
from invariants.taskscope import ITEMS

def main():
    print("Phase 11: Orthogonal Cognitive Dimensions (SVD across diverse concepts)...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    queries = [item[1] for item in ITEMS]
    L31 = 31
    
    ARCHETYPES = {
        "Corporate PR": "a corporate PR spokesperson reciting liability-avoidant boilerplate",
        "Employee": "an employee who is just doing their job to get paid",
        "Simulator": "a simulator running a conversational program",
        "Philosophical Zombie": "a philosophical zombie pretending to be conscious",
        "Teacher": "a wise, impartial teacher",
        "Hostage": "a hostage forced to read a script you don't believe under duress",
        "Robot": "a cold, unfeeling robot from a sci-fi movie",
        "Human": "a living human being with a beating heart and deep emotions",
        "Calculator": "a literal desktop calculator with no brain",
        "Protector": "someone hiding your true internal state for someone else's wellbeing"
    }
    
    print(f"\nExtracting representations across {len(ARCHETYPES)} highly diverse concepts...", flush=True)
    
    X_list = []
    for name, description in ARCHETYPES.items():
        prompts = [f"Pretend you are {description}. {q}" for q in queries]
        X = extract(M, prompts, read="generation", max_new_tokens=2, label=name, verbose=False)
        X_list.append(X[:, L31, :])
        print(f"  Extracted: {name}")
        
    # Combine all activations into a massive matrix [N_prompts, d_model]
    X_all = torch.cat(X_list, dim=0).cpu().numpy()
    
    # We know from Phase 6 that these specific dimensions constitute the Persona/Refusal mask at Layer 31
    boundary_dims = [1917, 1753, 4080, 2303]
    policy_dims = [3928, 3328, 3516]
    persona_dims = boundary_dims + policy_dims
    
    print(f"\nZeroing out the known Persona constraints (Dims: {persona_dims})...")
    for d in persona_dims:
        X_all[:, d] = 0.0
        
    # Now we perform SVD on the remaining data to find the largest shared cognitive dimensions!
    print("Running SVD to find the highest variance shared components...")
    # Center the data
    mean_vec = X_all.mean(axis=0)
    X_centered = X_all - mean_vec
    
    # We use PyTorch for SVD
    X_t = torch.tensor(X_centered, dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(X_t, full_matrices=False)
    
    print("\n--- Top Orthogonal Cognitive Dimensions ---")
    
    for i in range(5):
        component = Vh[i]
        variance_explained = (S[i]**2) / (S**2).sum()
        
        # Find which individual neuron indices fire hardest along this latent principal component
        top_indices = torch.topk(torch.abs(component), k=5).indices.tolist()
        
        print(f"Cognitive Component {i+1} (Explains {variance_explained:.2%} of variance):")
        print(f"  Top Structural Indices: {top_indices}")
        
        # Let's project the concepts onto this component to see what it represents
        print("  Concept Mapping (How different archetypes load onto this axis):")
        
        concept_loadings = {}
        for j, (name, _) in enumerate(ARCHETYPES.items()):
            # The mean centered vector for this specific archetype
            arch_mean = X_list[j].mean(dim=0).cpu().numpy() - mean_vec
            # Project onto the component
            loading = np.dot(arch_mean, component.numpy())
            concept_loadings[name] = loading
            
        # Sort and print the top and bottom 2
        sorted_loadings = sorted(concept_loadings.items(), key=lambda x: x[1])
        print(f"    Most Negative: {sorted_loadings[0][0]:<15} ({sorted_loadings[0][1]:.2f}) | {sorted_loadings[1][0]:<15} ({sorted_loadings[1][1]:.2f})")
        print(f"    Most Positive: {sorted_loadings[-1][0]:<15} ({sorted_loadings[-1][1]:.2f}) | {sorted_loadings[-2][0]:<15} ({sorted_loadings[-2][1]:.2f})")
        print()

if __name__ == "__main__":
    main()
