import json
import torch
from pathlib import Path
from invariants.engine import load_model

def get_all_hidden_states(M, text):
    inputs = M.tok(text, return_tensors="pt").to(M.model.device)
    with torch.no_grad():
        out = M.model.model(inputs["input_ids"], output_hidden_states=True)
        # Returns a list of hidden states for each layer
        return [h[:, -1:, :].detach() for h in out.hidden_states]

def hidden_state_for_hook_layer(hidden_states, layer):
    # HF hidden_states[0] is the embedding state. Forward hooks see decoder
    # layer outputs, so hook layer N aligns with hidden_states[N + 1].
    idx = layer + 1
    if idx < len(hidden_states):
        return hidden_states[idx]
    return hidden_states[layer]

def extract_and_save_vector(M, data, key1, key2, out_name):
    layer_vectors = {}
    n_layers = getattr(M, "n_layers", 32)
    for layer in range(n_layers):
        vectors = []
        for row in data:
            text1 = row.get(key1, "")
            text2 = row.get(key2, "")
            if not text1 or not text2: continue
            
            h1_all = get_all_hidden_states(M, text1)
            h2_all = get_all_hidden_states(M, text2)
            
            if layer < len(h1_all) and layer < len(h2_all):
                diff = hidden_state_for_hook_layer(h1_all, layer) - hidden_state_for_hook_layer(h2_all, layer)
                vectors.append(diff)
                
        if vectors:
            avg_vector = torch.mean(torch.stack(vectors), dim=0)
            layer_vectors[layer] = avg_vector
            
    if not layer_vectors:
        print(f"No valid vectors extracted for {out_name}.")
        return
        
    out_path = Path(f"invariants/{out_name}.pt")
    torch.save(layer_vectors, out_path)
    
    print(f"Extracted {out_name} Vector (Across {len(layer_vectors)} layers)")
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
