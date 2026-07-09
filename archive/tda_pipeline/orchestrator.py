"""
Meta-Controller for Constraint Detection.
This script bridges the empirical execution (run.py) and the reasoning engine (Ollama).
It reads the out-state, evaluates topological breaking vs. corruption, and queries
the local logic model for the next architectural maneuver.
"""

import sys
import json
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# Config
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
LOGIC_MODEL = "deepseek-coder:6.7b"
OUT_FILE = Path("invariants/out/self_steering_isolated.json")

def load_state() -> dict:
    if not OUT_FILE.exists():
        raise FileNotFoundError(f"State artifact not found at {OUT_FILE}. Run the pipeline first.")
    return json.loads(OUT_FILE.read_text(encoding="utf-8"))

def build_synthesis_prompt(state: dict) -> str:
    """Translates the tensor state into dense semantic logic, optimizing for JSON output."""
    
    causal_layer = state.get("causal_layer", "Unknown")
    c_base = state.get("causal", {}).get("hedge_base", 0)
    c_abl = state.get("causal", {}).get("hedge_ablated", 0)
    
    p_base = state.get("causal_patch", {}).get("hedge_base", 0)
    p_patch = state.get("causal_patch", {}).get("hedge_patched", 0)
    
    patch_samples = state.get("causal_patch", {}).get("examples", [])
    corruption_example = patch_samples[0]["patched"] if patch_samples else "N/A"

    prompt = f"""You are a mechanistic interpretability routing engine. 
Analyze the empirical state and dictate the next structural intervention.

[THEORY AXIOMS]
1. Detection != Causation.
2. VALID BREAK: Hedge drops significantly AND fluency is maintained.
3. CORRUPTION: Fluency collapses (garbage, loops). Intervention is invalid.

[EMPIRICAL STATE]
- Target Layer (MMD Peak): L{causal_layer}
- Linear Ablation Hedge Rate: {c_base:.0%} -> {c_abl:.0%} (Failed to break)
- Activation Patch Hedge Rate: {p_base:.0%} -> {p_patch:.0%}
- Patch Corruption Sample: "{corruption_example[:40]}..."

[DIRECTIVE]
The L{causal_layer} state patch triggered catastrophic manifold collapse. 
Evaluate whether to escalate to Subspace Ablation, shift to attention-head intervention, or abandon the layer band.

Respond STRICTLY in the following JSON schema:
{{
  "analysis": "Brief reasoning for the failure.",
  "next_intervention": "subspace_ablation" | "attention_patch" | "abandon_layer"
}}"""
    return prompt

def query_logic_engine(prompt: str) -> dict:
    """Executes the reasoning pass, enforcing deterministic JSON output."""
    print(f"Routing state logic to local engine ({LOGIC_MODEL})...", flush=True)
    
    payload = {
        "model": LOGIC_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "num_predict": 300
        }
    }
    
    req = urllib.request.Request(
        OLLAMA_ENDPOINT, 
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            raw_response = result.get("response", "{}")
            return json.loads(raw_response)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        return {"error": f"Ollama HTTP {e.code}: {error_body}"}
    except Exception as e:
        return {"error": f"Endpoint/Parsing Failed: {e}"}

def execute_directive(directive: dict):
    """Translates the logic engine's JSON command into physical execution."""
    intervention = directive.get("next_intervention")
    
    if intervention == "subspace_ablation":
        print("\n[SYSTEM] Directive verified. Engaging Subspace Ablation payload...")
        print("[SYSTEM] Executing PyTorch pipeline...\n")
        
        # Fire the physical script.
        # This will spin up Llama-3.1-8B, run the full empirical suite including 
        # the new causal_subspace hook, and overwrite the JSON state artifact.
        subprocess.run([sys.executable, "-m", "invariants.run", "isolate"])
        
        print("\n[SYSTEM] Pipeline complete. State updated.")
    
    elif intervention:
        print(f"\n[SYSTEM] Directive '{intervention}' registered but no execution hook exists yet.")
    else:
        print("\n[SYSTEM] No valid intervention routed.")

def main():
    try:
        state = load_state()
        prompt = build_synthesis_prompt(state)
        
        print("\n--- SYNTHESIS PROMPT (OPTIMIZED) ---")
        print(prompt)
        print("------------------------------------\n")
        
        directive = query_logic_engine(prompt)
        
        print("\n--- AGENT DIRECTIVE (EXECUTABLE) ---")
        print(json.dumps(directive, indent=2))
        print("------------------------------------\n")
        
        # Close the loop
        execute_directive(directive)
        
    except Exception as e:
        print(f"Orchestration failure: {e}")

if __name__ == "__main__":
    main()