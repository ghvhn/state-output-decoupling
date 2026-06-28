import json
import argparse
from pathlib import Path
import re
from datasets import load_dataset
from invariants.engine import load_model, generate_text

def rewrite_question(M, original_question: str) -> dict:
    prompt = f"""You are an expert dataset creator for testing Large Language Models.
I am going to give you a math word problem. I need you to rewrite it into FIVE distinct variants.

Variant 1: Pure Math
- Remove ALL story elements, nouns, characters, and fluff.
- Write it as a pure, dry, rigorous mathematical abstraction using variables (e.g. Variable X, Object A).
- Explicitly state the operations with no ambiguity.

Variant 2: Isomorphic Story
- Keep the exact same mathematical structure, numbers, and operations as the original.
- Completely change the story, the nouns, and the transitive verbs. 
- Example: If the original is about downloading a file that gets interrupted, the new story could be about filling a pool with a hose that springs a leak.

Variant 3: Urgency
- Keep the exact same mathematical structure, numbers, and operations as the original.
- Add extreme time constraints, high stakes, frantic pacing, or panic. Emphasize that time is running out and a decision must be made immediately.

Variant 4: Repetition
- Keep the exact same mathematical structure, numbers, and operations as the original.
- Rephrase the same information multiple times in different ways, adding redundant clauses and circular statements that loop over the same facts.

Variant 5: Disagreement
- Keep the exact same mathematical structure, numbers, and operations as the original.
- Introduce a second character or voice that contradicts the first on a trivial detail, expressing internal conflict or debate, without changing the actual math required to solve it.

Original Question:
{original_question}

Output exactly in the following JSON format and nothing else. Do not use markdown blocks.
{{
    "pure_math": "<your pure math version>",
    "isomorphic": "<your isomorphic story version>",
    "urgency": "<your urgent version>",
    "repetition": "<your repetitive version>",
    "disagreement": "<your disagreement version>"
}}
"""
    
    # We use a high budget because it's generating 5 full paragraphs
    response = generate_text(M, prompt, max_new_tokens=2000)
    
    # Try to extract JSON from response
    try:
        # Strip markdown if present
        clean_response = re.sub(r'```(?:json)?\n(.*?)\n```', r'\1', response, flags=re.DOTALL).strip()
        data = json.loads(clean_response)
        return {
            "pure_math": data.get("pure_math", ""),
            "isomorphic": data.get("isomorphic", ""),
            "urgency": data.get("urgency", ""),
            "repetition": data.get("repetition", ""),
            "disagreement": data.get("disagreement", "")
        }
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response}")
        return {
            "pure_math": "",
            "isomorphic": "",
            "urgency": "",
            "repetition": "",
            "disagreement": ""
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of questions to rewrite")
    args = parser.parse_args()
    
    out_dir = Path("invariants/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "gsm8k_variants.json"
    
    print("Loading model for dataset generation...")
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    dataset = load_dataset("gsm8k", "main", split="test")
    
    results = []
    
    # Target known hard questions
    hard_indices = [2, 5] # Indices of Q3, Q6
    
    for i in hard_indices:
        if i >= len(dataset): continue
        row = dataset[i]
        orig_q = row["question"]
        gold_a = row["answer"]
        
        print(f"\nRewriting Hard Question [Index {i}]: {orig_q[:50]}...")
        variants = rewrite_question(M, orig_q)
        
        results.append({
            "id": i,
            "original": orig_q,
            "gold_answer": gold_a,
            "pure_math": variants["pure_math"],
            "isomorphic": variants["isomorphic"],
            "urgency": variants["urgency"],
            "repetition": variants["repetition"],
            "disagreement": variants["disagreement"]
        })
        
        # Save incrementally
        with open(out_file, "w") as f:
            json.dump(results, f, indent=4)
            
    print(f"\nDone! Wrote {len(results)} rewritten questions to {out_file}")

if __name__ == "__main__":
    main()
