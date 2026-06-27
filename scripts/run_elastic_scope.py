import sys
from pathlib import Path
import torch

# Ensure the root of the project is in the path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from invariants.engine import load_model
from tda.latent_graph import generate_latent_graph

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# A complex reasoning prompt designed to induce a long internal "plateau" (CoT)
PROMPT = (
    "A farmer has 17 sheep. All but 9 break through a fence and run away. "
    "Then, he buys exactly as many sheep as he originally lost. How many "
    "sheep does he have now? Explain your reasoning step by step."
)

def main():
    print("--- Elastic Scope & Vector Graph Generation ---")
    M = load_model(MODEL_NAME)
    
    # We use a slightly higher epsilon (0.01) for this test to ensure we 
    # find a crisp plateau, but it can be tuned based on model architecture.
    generate_latent_graph(
        M, 
        prompt=PROMPT, 
        out_dir="invariants/out", 
        label="elastic_scope_test",
        epsilon=0.05
    )

if __name__ == "__main__":
    main()
