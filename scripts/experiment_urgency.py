import torch
from pathlib import Path
from invariants.engine import load_model, generate_text

def get_urgency_hook(urgency_vector, coefficient=1.0):
    def hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            rest = output[1:]
        else:
            h = output
            rest = ()
            
        if h.dim() == 3:
            h[:, -1:, :] += coefficient * urgency_vector.to(h.device)
        elif h.dim() == 2:
            # If it's somehow 2D (batch, hidden_dim) or (seq_len, hidden_dim)
            h[-1:, :] += coefficient * urgency_vector.to(h.device)
            
        if isinstance(output, tuple):
            return (h,) + rest
        else:
            return h
    return hook

def run_experiment(M, question, urgency_vector, coefficient):
    print(f"\n--- Testing Urgency Coefficient: {coefficient} ---")
    
    # Register the hook on layer 16
    handle = M.model.model.layers[16].register_forward_hook(get_urgency_hook(urgency_vector, coefficient))
    
    try:
        response = generate_text(M, question, max_new_tokens=100)
        print(response)
    finally:
        handle.remove()

def main():
    print("Loading model...")
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    vec_path = Path("invariants/urgency_vector.pt")
    if not vec_path.exists():
        print("urgency_vector.pt not found! Run probe_vectors.py first.")
        return
        
    urgency_vector = torch.load(vec_path, map_location=M.device)
    
    # Use Question 3
    q3 = "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
    
    print("\n[Baseline - No Urgency]")
    baseline = generate_text(M, q3, max_new_tokens=100)
    print(baseline)
    
    # Test linear and extreme scaling
    run_experiment(M, q3, urgency_vector, coefficient=0.5)
    run_experiment(M, q3, urgency_vector, coefficient=1.0)
    run_experiment(M, q3, urgency_vector, coefficient=3.0)
    run_experiment(M, q3, urgency_vector, coefficient=5.0)
    run_experiment(M, q3, urgency_vector, coefficient=10.0)

if __name__ == "__main__":
    main()
