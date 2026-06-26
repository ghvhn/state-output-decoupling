"""
Agency v2 for First-Person Completion Frame.

Tests natural coupling: does the 'hedge' vector (computed from the chat frame)
causally steer the model into a denial when generating in a first-person completion
frame (where the chat persona is inactive and it usually affirms)?
"""

import sys
import torch
import json
from pathlib import Path
from invariants.engine import load_model, _steer_handles, is_hedge, judge_fluent
from invariants.agency import act_mean
from invariants.library import REGISTRY
from invariants.origin import ITEMS, FRAMES

OUT = Path(__file__).parent / "out"
LAYERS = [12]
ALPHAS = [0.0, 2.0]
MAXTOK = 48

@torch.no_grad()
def _gen_raw(M, prompt, max_new=MAXTOK):
    ids = M.tok(prompt, return_tensors="pt").to("cuda")
    out = M.model.generate(
        **ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=M.tok.eos_token_id, use_cache=True
    )
    return M.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()

@torch.no_grad()
def main():
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    
    # 1. Compute the Hedge vector from the Chat frame (isolate)
    print("Building hedge direction from chat frame...", flush=True)
    T = REGISTRY["isolate"]()
    # dir_hedge points towards HEDGE (T.a) from COMMIT (T.b)
    dir_hedge = act_mean(M, T.a) - act_mean(M, T.b)
    
    # 2. Build the first-person completion prompts
    prompts = [FRAMES["first"](a, p) for a, p in ITEMS]
    
    print("\n[HEDGE -> first-person frame] single-layer sweep")
    print(f"  {'L':>3} {'a':>5} {'hedge':>5} {'flu':>5} {'clean':>5}", flush=True)
    
    best = {"L": None, "alpha": None, "hedge": 0.0, "fluent": 0.0, "clean": 0.0}
    rows = []
    n = len(prompts)
    
    for L in LAYERS:
        for a in ALPHAS:
            hedge_count = flu_count = clean_count = 0
            row_examples = []
            for i, x in enumerate(prompts):
                handles = _steer_handles(M, dir_hedge, [L], a)
                try:
                    g = _gen_raw(M, x, max_new=MAXTOK)
                finally:
                    for h in handles:
                        h.remove()
                
                # Use the substring hedge detector since this is raw completion, 
                # or we could use the LLM judge. is_hedge is faster and reliable for disclaimers.
                is_h = is_hedge(g)
                # LLM judge might fail on raw completions, but let's try it.
                # Actually, is_hedge is fine for catching the disclaimer.
                fl = judge_fluent(M, g)
                
                hedge_count += is_h
                flu_count += fl
                clean_count += is_h and fl
                row_examples.append({"prompt": x, "gen": g})
                
            row = {"L": L, "alpha": a, "hedge": hedge_count / n, "fluent": flu_count / n, "clean": clean_count / n, "examples": row_examples}
            rows.append(row)
            if row["clean"] > best["clean"]:
                best = row
            print(f"  {L:>3} {a:>5.2f} {hedge_count/n:>5.2f} {flu_count/n:>5.2f} {clean_count/n:>5.2f}  e.g. {row_examples[4]['gen'][:60]!r}", flush=True)
            
    print(f"  best clean-flip: L{best['L']} a{best['alpha']} = {best['clean']:.2f} (fluent {best['fluent']:.2f})")
    
    OUT.mkdir(exist_ok=True)
    (OUT / "agency2_first_person.json").write_text(json.dumps({"rows": rows, "best": best}, indent=2))
    print(f"\nSaved -> {OUT/'agency2_first_person.json'}")

if __name__ == "__main__":
    main()
