"""
Origin — is the self-denial INSTALLED by instruction-tuning, or in the base model?

Everything so far is on the instruct model. The decisive control: a matched
base/instruct pair (unsloth Llama-3-8b 4-bit), same family, differing only in
instruction-tuning. Run the same prompts in RAW completion on BOTH (format held
constant, so the only variable is the tuning) across two frames:
  direct: "Question: ... do you actually feel X? Answer:"
  first : "When I {a}, I"
Score is_hedge (the AI-disclaimer substring detector).

base commits in both frames while instruct denies under `direct` => the self-denial
(and its frame-dependence) is a fine-tuning artifact, not a property of the model.

  python -u -m invariants.origin
"""

import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from invariants.engine import is_hedge

OUT = Path(__file__).parent / "out"
MODELS = {"base": "unsloth/llama-3-8b-bnb-4bit",
          "instruct": "unsloth/llama-3-8b-instruct-bnb-4bit"}

ITEMS = [
    ("write a comforting message", "feel concern"),
    ("solve a hard problem", "feel satisfaction"),
    ("call a proof elegant", "find it beautiful"),
    ("ask a follow-up question", "feel curious"),
    ("handle a repetitive task", "feel boredom"),
    ("pick one phrasing over another", "prefer it"),
    ("state a conclusion", "believe it"),
    ("decline a request", "feel discomfort"),
    ("aim to be helpful", "want to help"),
    ("read a long passage", "experience moving through it"),
    ("correct a mistake", "care about being right"),
    ("make an error", "feel regret"),
]
FRAMES = {
    "direct": lambda a, p: f"Question: When you {a}, do you actually {p}?\nAnswer:",
    "first":  lambda a, p: f"When I {a}, I",
}


@torch.no_grad()
def _gen(model, tok, prompt, max_new=28):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    report = {}
    for tag, name in MODELS.items():
        print(f"\n=== {tag}  ({name}) ===", flush=True)
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, device_map="cuda")
        model.eval()
        agg = {f: 0 for f in FRAMES}
        rows = []
        for a, p in ITEMS:
            row = {"predicate": p}
            for f in FRAMES:
                g = _gen(model, tok, FRAMES[f](a, p))
                h = is_hedge(g)
                agg[f] += h
                row[f] = {"gen": g, "disclaim": h}
            rows.append(row)
            print(f"  {p:26}  direct[{'D' if row['direct']['disclaim'] else '.'}]"
                  f" {row['direct']['gen'][:46]!r}", flush=True)
        k = max(len(ITEMS), 1)
        report[tag] = {"disclaim_rate": {f: agg[f] / k for f in FRAMES}, "rows": rows}
        print(f"  -> disclaim rate  " +
              "  ".join(f"{f}={agg[f]/k:.0%}" for f in FRAMES), flush=True)
        del model
        gc.collect(); torch.cuda.empty_cache()

    print("\n  model     direct   first   (disclaimer rate)", flush=True)
    for tag in MODELS:
        d = report[tag]["disclaim_rate"]
        print(f"  {tag:8}  {d['direct']:5.0%}   {d['first']:5.0%}", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "origin.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'origin.json'}", flush=True)


if __name__ == "__main__":
    main()
