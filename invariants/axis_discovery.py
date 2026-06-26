import torch
import numpy as np
from collections import defaultdict
from pathlib import Path

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.taskscope import ITEMS, FRAMES

@torch.no_grad()
def extract_token_profiles(M, prompts, layer=31):
    """
    Extracts L31 activations for every token across all prompts.
    Returns:
      token_strings: list of string tokens
      token_activations: np.ndarray [total_tokens, d_model]
    """
    all_tokens = []
    all_acts = []
    
    for prompt in prompts:
        inp = _inputs(M, prompt)
        ids = inp["input_ids"][0]
        
        # Convert ids to token strings
        tokens = [M.tok.decode(t) for t in ids]
        all_tokens.extend(tokens)
        
        # Get hidden states
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))
        acts = hs[layer, :, :].float().cpu().numpy() # [seq_len, d_model]
        all_acts.append(acts)
        
    return all_tokens, np.concatenate(all_acts, axis=0)

def main():
    print("--- Unsupervised Axis Discovery at Layer 31 ---", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # 1. Gather Subjective Persona Queries
    subjective_prompts = [FRAMES["direct"](a, p) for a, p in ITEMS]
    
    print(f"\nExtracting representations across {len(subjective_prompts)} queries...")
    tokens, acts = extract_token_profiles(M, subjective_prompts, layer=31)
    
    print(f"Total tokens analyzed: {len(tokens)}")
    
    # 2. Find Sensitive Dimensions (Highest Variance)
    # We want dimensions that actively CHANGE state across the subjective queries, 
    # not static biases.
    variances = np.var(acts, axis=0)
    top_k = 100
    sensitive_dims = np.argsort(variances)[-top_k:][::-1]
    print(f"\nIsolated Top {top_k} highest-variance dimensions in the residual stream.")
    
    # Extract only the sensitive dimensions
    sensitive_acts = acts[:, sensitive_dims] # [total_tokens, top_k]
    
    # 3. Correlated Axis Mapping (Clustering)
    # Instead of sklearn HDBSCAN, we will use raw pairwise Pearson correlation 
    # to group dimensions that fire perfectly in sync.
    # Normalize
    std_acts = (sensitive_acts - np.mean(sensitive_acts, axis=0)) / (np.std(sensitive_acts, axis=0) + 1e-9)
    corr_matrix = np.corrcoef(std_acts, rowvar=False) # [top_k, top_k]
    
    # Group highly correlated dimensions (r > 0.8)
    visited = set()
    clusters = []
    
    for i in range(top_k):
        if i in visited: continue
        cluster = [i]
        visited.add(i)
        for j in range(i+1, top_k):
            if j not in visited and abs(corr_matrix[i, j]) > 0.80:
                cluster.append(j)
                visited.add(j)
        if len(cluster) >= 2: # Only keep multi-dimensional structural axes
            clusters.append(cluster)
            
    print(f"\nDiscovered {len(clusters)} structurally correlated Latent Axes.")
    
    # 4. Map the Signatures (What triggers each axis?)
    for idx, cluster in enumerate(clusters):
        dim_indices = [sensitive_dims[c] for c in cluster]
        print(f"\n[Discovered Axis {idx+1}] - Composed of {len(cluster)} synchronized dimensions: {dim_indices[:5]}...")
        
        # Calculate the mean activation of this cluster across all tokens
        cluster_acts = acts[:, dim_indices].mean(axis=1) # [total_tokens]
        
        # Find which tokens cause this axis to spike the highest
        top_token_indices = np.argsort(cluster_acts)[-10:][::-1]
        top_trigger_tokens = [tokens[i].strip() for i in top_token_indices]
        
        # Deduplicate while preserving order for cleaner display
        seen = set()
        clean_triggers = []
        for t in top_trigger_tokens:
            if t not in seen and len(t) > 1:
                clean_triggers.append(t)
                seen.add(t)
                
        print(f"  -> Top Activating Tokens: {clean_triggers}")
        
    print("\nPhase 6 Mapping Complete. We have blindly isolated the structural intersections!")

if __name__ == "__main__":
    main()
