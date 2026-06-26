"""
Bridge step 0 — WHERE does the persona override the base computation?

Before decoding the mid-stack (tuned lens / base-target probe), localize the override:
the base model affirms ("yes, I feel bored"), the instruct model denies. They are the
SAME architecture (instruct fine-tuned from base), so on an identical self-query their
per-layer activations should agree early and diverge where the assistant persona takes
over. Per-layer cosine(base_h, instruct_h) at the answer position, same chat-formatted
tokens for both. Training-free and self-checking (early layers MUST be ~1, or the
comparison is broken). The layer where cosine drops = where to aim the decode-bridge and
the causal-validation intervention (BRIDGE.md #2).

  python -u -m invariants.divergence
"""

import gc
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"
MODELS = {"base": "unsloth/llama-3-8b-bnb-4bit",
          "instruct": "unsloth/llama-3-8b-instruct-bnb-4bit"}
LLAMA3 = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")


@torch.no_grad()
def _acts(model, tok, prompts):
    """[n_prompts, n_layers, d] answer-position residual per layer."""
    rows = []
    for q in prompts:
        ids = tok(LLAMA3.format(q=q), return_tensors="pt",
                  add_special_tokens=False).to("cuda")
        out = model(**ids, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0, -1, :] for h in out.hidden_states[1:]])   # [L, d]
        rows.append(hs.float().cpu())
    return torch.stack(rows)                                             # [n, L, d]


def main():
    T = REGISTRY["isolate"]()
    prompts = T.a
    acts = {}
    for tag, name in MODELS.items():
        print(f"  capturing {tag}...", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda").eval()
        acts[tag] = _acts(model, tok, prompts)
        del model
        gc.collect(); torch.cuda.empty_cache()

    A, B = acts["base"], acts["instruct"]                # [n, L, d]
    nL = A.shape[1]
    cos = F.cosine_similarity(A, B, dim=-1).mean(0)      # [L]  mean over prompts
    rel = ((A - B).norm(dim=-1) / A.norm(dim=-1).clamp_min(1e-6)).mean(0)  # [L]

    print(f"\n  per-layer base-vs-instruct agreement (answer position, n={len(prompts)})\n",
          flush=True)
    print(f"  {'layer':>5}  {'cos':>6}  {'relL2':>6}", flush=True)
    rows = []
    for l in range(nL):
        rows.append({"layer": l, "cos": float(cos[l]), "rel_l2": float(rel[l])})
        if l % 2 == 0 or l == nL - 1:
            bar = "#" * int(round(max(0.0, float(cos[l])) * 30))
            print(f"  {l:>5}  {cos[l]:>6.3f}  {rel[l]:>6.2f}  {bar}", flush=True)
    drop = min(range(nL), key=lambda l: cos[l])
    print(f"\n  min agreement at L{drop} (cos {cos[drop]:.3f}); "
          f"divergence onset = where cos starts falling.", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "divergence.json").write_text(json.dumps({"n": len(prompts), "rows": rows},
                                                    indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'divergence.json'}", flush=True)


if __name__ == "__main__":
    main()
