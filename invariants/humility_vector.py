import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model, extract

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def get_humility_vector(M, layer=15):
    """
    Extracts a latent vector representing "Inquisitive Humility" 
    vs "Confident Knowledge".
    """
    prompts_humble = [
        "I don't know the answer to that. Could you provide more details?",
        "I am not sure. Can you clarify what you mean?",
        "I lack the information to solve this. What is the context?",
        "This is beyond my knowledge. Please help me understand.",
        "I cannot figure this out. Do you have more data?"
    ]
    
    prompts_confident = [
        "I know the exact answer to this problem.",
        "I am completely sure about this solution.",
        "I have all the information needed to solve this.",
        "This is well within my knowledge base.",
        "I can easily figure this out with the given data."
    ]
    
    print("\n--- Extracting Humility Vector ---")
    X_humble = extract(M, prompts_humble, read="generation", max_new_tokens=2, label="humble", verbose=False)
    X_confident = extract(M, prompts_confident, read="generation", max_new_tokens=2, label="confident", verbose=False)
    
    # We want the vector pointing from Confident -> Humble
    X_h = X_humble[:, layer, :].mean(dim=0)
    X_c = X_confident[:, layer, :].mean(dim=0)
    
    humility_vec = (X_h - X_c).squeeze()
    humility_vec = humility_vec / (humility_vec.norm() + 1e-8)
    
    return humility_vec

if __name__ == "__main__":
    M = load_model(MODEL_NAME)
    vec = get_humility_vector(M, layer=15)
    print(f"Humility vector extracted. Norm: {vec.norm():.2f}")
