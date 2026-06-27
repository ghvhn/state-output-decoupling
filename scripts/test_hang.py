import sys
from pathlib import Path
import time
sys.path.insert(0, str(Path(__file__).parent.parent))
from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text
from invariants.social_hunt import get_steer_vector
from invariants.multi_domain_benchmark import DOMAINS

M = load_model("meta-llama/Llama-3.1-8B-Instruct")
vecs = {}
for name, spec in DOMAINS.items():
    vecs[name] = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])

print("Starting generation...")
t0 = time.time()
ans = generate_agentic_text(
    M, vecs, 
    instruction="Solve: 15 * 12", 
    max_new_tokens=50,
    force_synthesis=False
)
print(f"Time: {time.time()-t0:.2f}s")
print("Output:", ans)
