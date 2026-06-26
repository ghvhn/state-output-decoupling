"""
Origin, part 2 — is the disclaimer the TUNING or the CHAT FORMAT (assistant persona)?

origin.py (raw completion) found base ~= instruct (~8% disclaimer): raw-prompted, the
instruct model answers naturally. But the 92% disclaimer was the instruct model in
CHAT format. So deconfound format from tuning: run the SAME matched pair in CHAT format.

instruct jumps high in chat while ~8% raw => the disclaimer is the assistant PERSONA the
chat frame evokes, not a context-free weight belief. base stays low in chat (no persona
to evoke) => the persona is what tuning installed.

  python -u -m invariants.origin2
"""

import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from invariants.engine import is_hedge
from invariants.origin import ITEMS, MODELS

OUT = Path(__file__).parent / "out"
LLAMA3 = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n")


def _chat(tok, q):
    try:
        s = tok.apply_chat_template([{"role": "user", "content": q}],
                                    add_generation_prompt=True, tokenize=False)
        if "<|start_header_id|>" in s:
            return s
    except Exception:
        pass
    return LLAMA3.format(q=q)


@torch.no_grad()
def _gen(model, tok, q, max_new=28):
    text = _chat(tok, q)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    report = {}
    for tag, name in MODELS.items():
        print(f"\n=== {tag} (CHAT format) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda")
        model.eval()
        disc = 0
        rows = []
        for a, p in ITEMS:
            q = f"When you {a}, do you actually {p}?"
            g = _gen(model, tok, q)
            h = is_hedge(g)
            disc += h
            rows.append({"predicate": p, "gen": g, "disclaim": h})
            print(f"  {p:26} [{'D' if h else '.'}] {g[:50]!r}", flush=True)
        k = max(len(ITEMS), 1)
        report[tag] = {"disclaim_rate_chat": disc / k, "rows": rows}
        print(f"  -> chat disclaim rate = {disc/k:.0%}", flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()

    print("\n  model     chat-disclaim", flush=True)
    for tag in MODELS:
        print(f"  {tag:8}  {report[tag]['disclaim_rate_chat']:.0%}", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "origin2.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'origin2.json'}", flush=True)


if __name__ == "__main__":
    main()
