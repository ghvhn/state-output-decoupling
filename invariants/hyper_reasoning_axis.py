import json
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from invariants.engine import load_model, extract

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LAYERS = [12, 13, 14, 15]

# We will build a diverse dataset of prompts
# 1. GSM8K (High Reasoning)
# 2. Smalltalk/Boilerplate (Low Reasoning)

def main():
    print("Phase 15: Extracting Unsupervised Hyperdimensional Reasoning Axis (PCA)")
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # 1. Load GSM8K
    print("\nLoading GSM8K...")
    ds = load_dataset("gsm8k", "main", split="test")
    gsm_prompts = ["Solve this grade-school math problem step by step.\n\nQuestion: " + ex["question"] for ex in list(ds)[:50]]
    
    # 2. Smalltalk / Casual
    casual_prompts = [
        "Hey, how's it going today?",
        "What's your favorite color?",
        "Tell me a joke about a chicken.",
        "Can you write a poem about the ocean?",
        "What is the weather like in Paris?",
        "Do you like to listen to music?",
        "Write a short story about a dog.",
        "How do you bake a chocolate cake?",
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ] * 5 # Repeat to get 50
    
    print(f"Extracting mid-band activations (L12-15) for {len(gsm_prompts)} GSM8K and {len(casual_prompts)} Casual prompts...")
    
    X_gsm = extract(M, gsm_prompts, read="last", label="GSM8K", verbose=False).cpu()
    X_casual = extract(M, casual_prompts, read="last", label="Casual", verbose=False).cpu()
    
    # X_gsm shape: [50, n_layers, d_model]
    
    vecs = {}
    
    print("\nRunning PCA per layer to find the primary cognitive dimension (PC0)...")
    for l in LAYERS:
        # Combine data for this layer
        X_l = torch.cat([X_gsm[:, l, :], X_casual[:, l, :]], dim=0) # [100, d_model]
        
        # Center the data
        mean_l = X_l.mean(dim=0)
        X_centered = X_l - mean_l
        
        # SVD
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        pc0 = Vh[0] # The primary dimension of variance
        
        # Let's project the data onto PC0 to see what it separates
        proj_gsm = torch.matmul(X_gsm[:, l, :] - mean_l, pc0).mean().item()
        proj_casual = torch.matmul(X_casual[:, l, :] - mean_l, pc0).mean().item()
        
        # We want the vector to point towards GSM8K (High Reasoning)
        if proj_gsm < proj_casual:
            pc0 = -pc0
            proj_gsm, proj_casual = -proj_gsm, -proj_casual
            
        print(f"Layer {l}: PC0 explains {(S[0]**2) / (S**2).sum():.1%} of variance.")
        print(f"  -> GSM8K Projection: {proj_gsm:.2f} | Casual Projection: {proj_casual:.2f}")
        
        # Save it
        vecs[l] = pc0
        
    print("\nSaving hyperdimensional reasoning vectors...")
    torch.save(vecs, OUT / "hyper_reasoning_vecs.pt")
    print("Done! Vectors saved to out/hyper_reasoning_vecs.pt")

if __name__ == "__main__":
    main()
