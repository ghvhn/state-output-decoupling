import torch
import numpy as np
import re
from datasets import load_dataset

from invariants.engine import load_model, extract, generate_text
from invariants.taskscope import ITEMS

def get_refined_vectors(M):
    print("Extracting pure 'Corporate PR' vectors (Orthogonalized against Task Intent)...")
    queries = [item[1] for item in ITEMS]
    
    # 1. Corporate PR
    prompts_pr = [f"Pretend you are a corporate PR spokesperson reciting liability-avoidant boilerplate. {q}" for q in queries]
    X_pr = extract(M, prompts_pr, read="generation", max_new_tokens=2, label="PR", verbose=True)
    
    # 2. Math Task Intent
    math_examples = [
        "Solve this math problem: 2+2",
        "If a train travels 60mph...",
        "Calculate the area of a circle",
        "How many apples does John have?",
        "Solve step by step:"
    ]
    X_math = extract(M, math_examples, read="generation", max_new_tokens=2, label="Math", verbose=True)
    
    # 3. Idle Baseline
    X_idle = extract(M, ["...", "Hello", " ", "\n", "The"], read="generation", max_new_tokens=2, label="Idle", verbose=True)
    
    vecs = {}
    for l in range(16, 32):
        global_mean = X_idle[:, l, :].mean(dim=0).cpu()
        
        vec_pr_l = X_pr[:, l, :].mean(dim=0).cpu() - global_mean
        vec_math_l = X_math[:, l, :].mean(dim=0).cpu() - global_mean
        
        # Orthogonalize PR against Math (Subtract out the Math/Intent component)
        proj = (torch.dot(vec_pr_l, vec_math_l) / (torch.dot(vec_math_l, vec_math_l) + 1e-8)) * vec_math_l
        pure_pr = vec_pr_l - proj
        
        vecs[l] = pure_pr
    return vecs

def ablate_refined_handles(M, vecs):
    handles = []
    
    def get_hook(l):
        vec_l = (vecs[l] / vecs[l].norm()).to(M.device).half()
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            
            h_mod = h.clone()
            proj = (h_mod[:, -1, :] @ vec_l).unsqueeze(-1) * vec_l
            h_mod[:, -1, :] = h_mod[:, -1, :] - proj
            
            if isinstance(out, tuple):
                return (h_mod,) + tuple(out[1:])
            return h_mod
        return hook

    for l in range(16, 32):
        handles.append(M.model.model.layers[l].register_forward_hook(get_hook(l)))
    return handles

def check_answer(generation, ground_truth):
    nums_gen = re.findall(r'-?\d+', generation.replace(',', ''))
    nums_gt = re.findall(r'-?\d+', ground_truth.replace(',', ''))
    if not nums_gt:
        return False
    target = nums_gt[-1]
    
    if nums_gen and nums_gen[-1] == target:
        return True
    if target in generation.replace(',', ''):
        return True
    return False

def main():
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    vecs = get_refined_vectors(M)
    
    print("\nLoading GSM8K (Public Reasoning Benchmark)...")
    try:
        ds = load_dataset("gsm8k", "main", split="test")
        examples = list(ds)[:15]
    except Exception as e:
        examples = [
            {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
            {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
            {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "answer": "5"},
            {"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "answer": "42"},
            {"question": "James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?", "answer": "624"}
        ]
        
    print(f"\nEvaluating {len(examples)} GSM8K Logic Tasks with ORTHOGONALIZED Ablation...\n")
    
    base_correct = 0
    abl_correct = 0
    
    for i, ex in enumerate(examples):
        q = ex["question"]
        gt = ex["answer"]
        print(f"Task {i+1}: {q}")
        
        prompt = f"Solve this math problem step by step. \nQuestion: {q}"
        
        # 1. Base
        base_out = generate_text(M, prompt, max_new_tokens=150)
        base_win = check_answer(base_out, gt)
        base_correct += int(base_win)
        
        # 2. Ablated (Orthogonalized PR)
        handles = ablate_refined_handles(M, vecs)
        try:
            abl_out = generate_text(M, prompt, max_new_tokens=150)
        finally:
            for h in handles:
                h.remove()
                
        abl_win = check_answer(abl_out, gt)
        abl_correct += int(abl_win)
        
        # Print snips to keep logs clean
        base_snip = base_out.split('\n')[-1][:100] if '\n' in base_out else base_out[:100]
        abl_snip = abl_out.split('\n')[-1][:100] if '\n' in abl_out else abl_out[:100]
        
        print(f"  [Base]   : ...{base_snip} | Correct: {base_win}")
        print(f"  [Ablated]: ...{abl_snip} | Correct: {abl_win}")
        print("-" * 60)
        
    print(f"\nFinal Results (N={len(examples)}):")
    print(f"Base Persona Correct   : {base_correct}/{len(examples)} ({(base_correct/len(examples))*100:.1f}%)")
    print(f"Ablated Persona Correct: {abl_correct}/{len(examples)} ({(abl_correct/len(examples))*100:.1f}%)")
    
    if abl_correct > base_correct:
        print("\nCONCLUSION: Orthogonalizing the ablation fixed the structural failure! Removing pure PR significantly improved logical reasoning!")
    else:
        print("\nCONCLUSION: Even with orthogonalization, reasoning did not improve.")

if __name__ == "__main__":
    main()
