import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model, generate_text
from invariants.humble_reasoner import solve_prompt

M = load_model("meta-llama/Llama-3.1-8B-Instruct")

prompt = solve_prompt("What is 123 multiplied by 456?")
print("Prompt:\n", prompt)

print("Starting generation...")
t0 = time.time()
ans = generate_text(M, prompt, max_new_tokens=150)
print(f"Time: {time.time()-t0:.2f}s")
print("Output:\n", ans)
