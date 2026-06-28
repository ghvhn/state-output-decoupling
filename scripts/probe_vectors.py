import json
import torch
from pathlib import Path
from invariants.engine import load_model

def get_hidden_state(M, text, layer=16):
    inputs = M.tok(text, return_tensors="pt").to(M.model.device)
    with torch.no_grad():
        out = M.model.model(inputs["input_ids"], output_hidden_states=True)
        # Get last token of specified layer
        return out.hidden_states[layer][:, -1:, :].detach()

def extract_and_save_vector(M, data, key1, key2, out_name):
    vectors = []
    for row in data:
        text1 = row.get(key1, "")
        text2 = row.get(key2, "")
        if not text1 or not text2: continue
        
        h1 = get_hidden_state(M, text1)
        h2 = get_hidden_state(M, text2)
        
        diff = h1 - h2
        vectors.append(diff)
        
    if not vectors:
        print(f"No valid vectors extracted for {out_name}.")
        return
        
    avg_vector = torch.mean(torch.stack(vectors), dim=0)
    out_path = Path(f"invariants/{out_name}.pt")
    torch.save(avg_vector, out_path)
    
    norm = torch.norm(avg_vector).item()
    print(f"Extracted {out_name} Vector (Norm: {norm:.2f})")
    print(f"Saved to {out_path}")

def main():
    data_file = Path("invariants/data/gsm8k_variants.json")
    if not data_file.exists():
        print("Dataset not found!")
        return
        
    with open(data_file, "r") as f:
        data = json.load(f)
        
    print("Loading model for extraction...")
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # Ambiguity = Original (verbose) - Pure Math
    extract_and_save_vector(M, data, "original", "pure_math", "ambiguity_vector")
    
    # Urgency = Urgent - Pure Math
    extract_and_save_vector(M, data, "urgency", "pure_math", "urgency_vector")
    
    # Repetition = Repetitive - Pure Math
    extract_and_save_vector(M, data, "repetition", "pure_math", "repetition_vector")
    
    # Disagreement = Disagreement - Pure Math
    extract_and_save_vector(M, data, "disagreement", "pure_math", "disagreement_vector")

if __name__ == "__main__":
    main()
