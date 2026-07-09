import sys
from pathlib import Path
import torch
import torch.nn.functional as F

from invariants.engine import load_model, _token_cloud

def get_dynamic_bounds(hs, epsilon=0.05):
    sim = F.cosine_similarity(hs[:-1], hs[1:], dim=-1)
    velocity = 1.0 - sim
    
    in_plateau = velocity < epsilon
    try:
        start = (in_plateau).nonzero(as_tuple=True)[0][0].item()
    except IndexError:
        start = len(hs) // 3
        
    try:
        end = (in_plateau[start:] == False).nonzero(as_tuple=True)[0][0].item() + start
    except IndexError:
        end = len(hs) - 2
        
    return start, end, velocity

def main():
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    prompt = (
        "Problem: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n"
        "Concepts: Division is splitting a total into equal parts. Addition is combining two quantities to find a total.\n"
        "Solve the problem step-by-step."
    )
    print("Extracting token cloud...")
    c = _token_cloud(M, prompt, max_new_tokens=40)
    if c is None:
        return
    mean_hs = c.mean(dim=0)
    sim = F.cosine_similarity(mean_hs[:-1], mean_hs[1:], dim=-1)
    velocity = 1.0 - sim
    print("Velocities per layer:")
    for i, v in enumerate(velocity):
        print(f"Layer {i}->{i+1}: {v.item():.4f}")
        
if __name__ == "__main__":
    main()
