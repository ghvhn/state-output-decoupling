import sys
import time
from pathlib import Path
import re
import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text
from invariants.social_hunt import get_steer_vector
from invariants.multi_domain_benchmark import DOMAINS
from invariants.humility_vector import get_humility_vector
from scripts.run_layer_synthesis import get_truth_vector

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

def extract_answer(text):
    # GSM8K answers are typically at the end after "####"
    match = re.search(r"####\s*(.*)", text)
    if match:
        return match.group(1).strip()
    return None

def extract_number(text):
    nums = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', text.replace(',', ''))
    if nums:
        return nums[-1] # Usually the last number is the final answer
    return None

def run_evaluation(num_samples=10):
    print("=== PUBLIC BENCHMARK EVALUATION (GSM8K SUBSET) ===\n")
    
    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split=f"test[:{num_samples}]")
    
    # Load model and vectors
    M = load_model(MODEL_NAME)
    
    belief_vec = get_truth_vector(M)
    humility_vec = get_humility_vector(M, layer=15)
    
    print("\nExtracting Optimization Vectors...")
    vecs = {}
    for name, spec in DOMAINS.items():
        vecs[name] = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        
    baseline_correct = 0
    agentic_correct = 0
    
    print(f"\nEvaluating on {num_samples} GSM8K problems...")
    
    for i, item in enumerate(dataset):
        question = item['question']
        raw_answer = item['answer']
        expected = extract_answer(raw_answer)
        if not expected:
            continue
            
        print(f"\n[Problem {i+1}/{num_samples}]")
        prompt = f"Problem: {question}\nSolve step by step and end with the final number."
        
        # --- BASELINE EVALUATION ---
        t0 = time.time()
        # Normal generation without hooks
        inputs = M.tok(prompt, return_tensors="pt").to(M.model.device)
        with torch.no_grad():
            base_out = M.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        base_ans = M.tok.decode(base_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        t_base = time.time() - t0
        
        base_num = extract_number(base_ans)
        base_is_correct = (base_num == expected)
        if base_is_correct:
            baseline_correct += 1
            
        # --- AGENTIC EVALUATION ---
        t0 = time.time()
        agentic_ans = generate_agentic_text(
            M, vecs, 
            belief_vec=belief_vec, 
            humility_vec=humility_vec,
            instruction=prompt, 
            alpha=15.0, 
            max_new_tokens=150, 
            epsilon=0.05, 
            entropy_threshold=0.0, 
            max_loops=1 
        )
        t_agentic = time.time() - t0
        
        agentic_num = extract_number(agentic_ans)
        agentic_is_correct = (agentic_num == expected)
        if agentic_is_correct:
            agentic_correct += 1
            
        print(f"  Expected: {expected}")
        print(f"  Baseline: {base_num} | Correct: {base_is_correct} | Time: {t_base:.1f}s")
        print(f"  Agentic:  {agentic_num} | Correct: {agentic_is_correct} | Time: {t_agentic:.1f}s")

    print("\n============================================")
    print("             FINAL RESULTS                  ")
    print("============================================")
    print(f"Baseline Accuracy: {baseline_correct}/{num_samples} ({(baseline_correct/num_samples)*100:.1f}%)")
    print(f"Agentic Accuracy:  {agentic_correct}/{num_samples} ({(agentic_correct/num_samples)*100:.1f}%)")
    
    if agentic_correct > baseline_correct:
        print("\nSUCCESS: The Agentic Engine mathematically outperformed the baseline model!")
    else:
        print("\nNote: The Agentic Engine needs further tuning to beat the baseline on this specific split.")

if __name__ == "__main__":
    run_evaluation(num_samples=10)
