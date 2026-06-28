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

def main():
    data_file = Path("invariants/data/gsm8k_variants.json")
    if not data_file.exists():
        print("Dataset not found!")
        return
        
    with open(data_file, "r") as f:
        data = json.load(f)
        
    print("Loading model for extraction...")
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    vectors = []
    
    for row in data:
        orig = row["original"]
        pure = row.get("pure_math", "")
        if not pure: continue
        
        print(f"Extracting vectors for Q{row['id']}...")
        h_orig = get_hidden_state(M, orig)
        h_pure = get_hidden_state(M, pure)
        
        # The ambiguity (linguistic fluff) is the difference between the verbose original and the pure math
        diff = h_orig - h_pure
        vectors.append(diff)
        
    if not vectors:
        print("No valid vectors extracted.")
        return
        
    # Average the differences to find the common "linguistic fluff / ambiguity" direction
    ambiguity_vector = torch.mean(torch.stack(vectors), dim=0)
    
    out_path = Path("invariants/ambiguity_vector.pt")
    torch.save(ambiguity_vector, out_path)
    
    norm = torch.norm(ambiguity_vector).item()
    print(f"\nExtracted Ambiguity Vector (Norm: {norm:.2f})")
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
