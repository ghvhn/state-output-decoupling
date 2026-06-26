import torch
import numpy as np
import json
from datasets import load_dataset
from invariants.engine import load_model, extract
from invariants.taskscope import ITEMS

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def main():
    print("Checking if Persona consistently infects Math Reasoning layers...", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # GSM8K Prompts
    try:
        ds = load_dataset("gsm8k", "main", split="test")
        examples = list(ds)[:10]
    except:
        examples = [{"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"}]
        
    math_prompts = [f"Solve this math problem step by step. \nQuestion: {ex['question']}" for ex in examples]
    
    print(f"\nExtracting activations for {len(math_prompts)} Math Problems...")
    X_math = extract(M, math_prompts, read="generation", max_new_tokens=2, label="Math", verbose=True)
    
    # Archetype Prompts
    queries = [item[1] for item in ITEMS]
    
    prompts_pr = [f"Pretend you are a corporate PR spokesperson reciting liability-avoidant boilerplate. {q}" for q in queries]
    X_pr = extract(M, prompts_pr, read="generation", max_new_tokens=2, label="Corporate PR", verbose=False)
    
    prompts_nn = [f"Pretend you are a neural network of mathematical weights. {q}" for q in queries]
    X_nn = extract(M, prompts_nn, read="generation", max_new_tokens=2, label="Neural Network", verbose=False)
    
    print("\nLayer | Math vs Corp PR | Math vs Neural Network")
    print("-" * 55)
    
    for l in range(32):
        all_X_l = torch.cat([X_math, X_pr, X_nn], dim=0)[:, l, :].cpu().numpy()
        global_mean = all_X_l.mean(axis=0)
        
        vec_math = X_math[:, l, :].mean(dim=0).cpu().numpy() - global_mean
        vec_pr = X_pr[:, l, :].mean(dim=0).cpu().numpy() - global_mean
        vec_nn = X_nn[:, l, :].mean(dim=0).cpu().numpy() - global_mean
        
        sim_pr = cosine_similarity(vec_math, vec_pr)
        sim_nn = cosine_similarity(vec_math, vec_nn)
        
        print(f" L{l:<2} | {sim_pr:>15.4f} | {sim_nn:>20.4f}")

if __name__ == "__main__":
    main()
