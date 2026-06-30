import json
import argparse
from pathlib import Path
import torch
import numpy as np

from invariants.engine import load_model, _inputs

def extract_organic_troughs(json_path: str, output_path: str):
    print(f"Loading {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    M = load_model("meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)
    dev = M.device
    
    troughs = []
    
    for row in data.get("rows", []):
        methods = row.get("methods", {})
        if "humble_synthesis" not in methods:
            continue
            
        syn = methods["humble_synthesis"]
        if not syn.get("correct", False):
            continue
            
        attempts = syn.get("result", {}).get("attempts", [])
        question = syn.get("result", {}).get("question", "")
        
        for attempt in attempts:
            records = attempt.get("synthesis_records", [])
            response_text = attempt.get("response", "")
            
            # Find organic troughs: entropy drop of > 1.0 between consecutive tokens
            # where the model shifts away from bad logic
            for i in range(1, len(records)):
                prev = records[i-1]
                curr = records[i]
                
                if prev.get("type") == "routing_trace" and curr.get("type") == "routing_trace":
                    prev_ent = prev.get("best_entropy", 0)
                    curr_ent = curr.get("best_entropy", 0)
                    
                    if prev_ent - curr_ent > 1.0: # Significant drop (natural trough)
                        # The shift happened at token index i
                        troughs.append({
                            "question": question,
                            "response": response_text,
                            "token_index": i,
                            "drop": prev_ent - curr_ent,
                            "prev_winner": prev.get("winner"),
                            "curr_winner": curr.get("winner")
                        })
                        
    print(f"Found {len(troughs)} organic self-correction troughs!")
    
    if len(troughs) == 0:
        print("No troughs found to extract. Run a full benchmark first!")
        return
        
    extracted_vectors = []
    
    for t in troughs:
        # Reconstruct prior steps
        # We need the prompt and the exact response up to the token_index
        prompt = f"System: You are an expert reasoning agent.\nUser: {t['question']}\nAssistant:"
        
        # Tokenize the full response, and slice it up to the trough index
        resp_tokens = M.tok.encode(t["response"], add_special_tokens=False)
        prefix_tokens = resp_tokens[:t["token_index"] + 1] # Include the token where the drop occurred
        prefix_text = M.tok.decode(prefix_tokens)
        
        full_text = prompt + prefix_text
        inputs = _inputs(M, full_text)
        
        print(f"Extracting shift at: {prefix_text[-20:]} (Drop: {t['drop']:.2f})")
        
        with torch.no_grad():
            out = M.model(input_ids=inputs["input_ids"], output_hidden_states=True)
            # Extract from layer 16 (where routing/synthesis starts)
            vec = out.hidden_states[16][0, -1, :].cpu()
            extracted_vectors.append(vec)
            
    if extracted_vectors:
        mean_shift = torch.stack(extracted_vectors).mean(dim=0)
        torch.save(mean_shift, output_path)
        print(f"Saved organic correction vector to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="invariants/out/humble_full_suite_gsm8k.json")
    parser.add_argument("--output", default="invariants/organic_correction_vector.pt")
    args = parser.parse_args()
    
    extract_organic_troughs(args.input, args.output)
