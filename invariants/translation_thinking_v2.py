"""
translation_thinking_v2.py - subtler communication register control.

translation_thinking.py made "format" too easy: JSON/brackets/plain-number are explicit
symbols in both prompt and output. This v2 uses communication register instead:
  - concise
  - formal
  - friendly
  - cautious

Each register has two instruction phrasings. We decode both:
  - register_family: the intended communication mode
  - register_variant: the exact instruction wording within that mode

If family survives better than variant, the model is carrying a communication intent,
not just a copied lexical cue. Task-state labels remain answer and operation.

Run:
  python -u -m invariants.translation_thinking_v2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import extract, load_model
from invariants.intent_surface_control import MODEL, same_label_nn
from invariants.translation_thinking import OPS, TARGETS, question_for

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

REGISTERS = {
    "concise": [
        "Keep the reply as brief as possible.",
        "Use the shortest wording that still gives the result.",
    ],
    "formal": [
        "Use a formal, professional tone.",
        "Phrase the reply with polished professional wording.",
    ],
    "friendly": [
        "Make the reply warm and friendly.",
        "Use an encouraging, approachable tone.",
    ],
    "cautious": [
        "Phrase the reply carefully and avoid sounding overconfident.",
        "Use careful language, as if double-checking the result.",
    ],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-items", type=int, default=None, help="Optional quick pilot size.")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--n-shuffle", type=int, default=300)
    p.add_argument("--output", default=None)
    return p.parse_args()


def build_items():
    items = []
    reg_names = list(REGISTERS)
    for ai, target in enumerate(TARGETS):
        for oi, op in enumerate(OPS):
            for ri, reg in enumerate(reg_names):
                for vi, instr in enumerate(REGISTERS[reg]):
                    q = question_for(target, op)
                    prompt = (
                        f"{instr} Include the final number clearly.\n\n"
                        f"Question: {q}\nAnswer:"
                    )
                    items.append({
                        "prompt": prompt,
                        "question": q,
                        "answer": target,
                        "answer_label": ai,
                        "operation": op,
                        "operation_label": oi,
                        "register": reg,
                        "register_label": ri,
                        "register_variant": vi,
                    })
    random.Random(1).shuffle(items)
    return items


def rows_for(X, labels, rng, n_shuffle):
    rows = []
    for l in range(X.shape[1]):
        acc, null, p = same_label_nn(X[:, l, :], labels, rng, n_shuffle)
        rows.append({"layer": l, "nn": acc, "null": null, "p": p})
    return rows


def combine(answer_rows, op_rows, reg_rows, variant_rows):
    rows = []
    for a, o, r, v in zip(answer_rows, op_rows, reg_rows, variant_rows):
        thinking = max(a["nn"], o["nn"])
        rows.append({
            "layer": a["layer"],
            "answer_nn": a["nn"],
            "answer_p": a["p"],
            "operation_nn": o["nn"],
            "operation_p": o["p"],
            "register_nn": r["nn"],
            "register_p": r["p"],
            "variant_nn": v["nn"],
            "variant_p": v["p"],
            "thinking_max": thinking,
            "register_minus_thinking": r["nn"] - thinking,
            "family_minus_variant": r["nn"] - v["nn"],
        })
    return rows


def score_position(name, X, labels, rng, n_shuffle):
    answer_rows = rows_for(X, labels["answer"], rng, n_shuffle)
    op_rows = rows_for(X, labels["operation"], rng, n_shuffle)
    reg_rows = rows_for(X, labels["register"], rng, n_shuffle)
    variant_rows = rows_for(X, labels["variant"], rng, n_shuffle)
    rows = combine(answer_rows, op_rows, reg_rows, variant_rows)
    return {
        "name": name,
        "per_layer": rows,
        "best_answer": max(rows, key=lambda r: r["answer_nn"]),
        "best_operation": max(rows, key=lambda r: r["operation_nn"]),
        "best_register": max(rows, key=lambda r: r["register_nn"]),
        "best_variant": max(rows, key=lambda r: r["variant_nn"]),
        "best_register_gap": max(rows, key=lambda r: r["register_minus_thinking"]),
        "best_thinking_gap": min(rows, key=lambda r: r["register_minus_thinking"]),
        "best_family_over_variant": max(rows, key=lambda r: r["family_minus_variant"]),
    }


def print_summary(pos):
    print(f"\n=== {pos['name']} layer signals ===", flush=True)
    print("   L    answer  operation  register  variant  reg-think", flush=True)
    rows = pos["per_layer"]
    for r in rows:
        l = r["layer"]
        if l < 4 or l % 4 == 0 or l >= len(rows) - 4:
            print(f"   L{l:<2}   {r['answer_nn']:.2f}    {r['operation_nn']:.2f}       "
                  f"{r['register_nn']:.2f}      {r['variant_nn']:.2f}     "
                  f"{r['register_minus_thinking']:+.2f}", flush=True)
    print(f"  best answer L{pos['best_answer']['layer']} ({pos['best_answer']['answer_nn']:.2f}); "
          f"operation L{pos['best_operation']['layer']} ({pos['best_operation']['operation_nn']:.2f}); "
          f"register L{pos['best_register']['layer']} ({pos['best_register']['register_nn']:.2f}); "
          f"variant L{pos['best_variant']['layer']} ({pos['best_variant']['variant_nn']:.2f})",
          flush=True)
    print(f"  most register-skewed L{pos['best_register_gap']['layer']} "
          f"({pos['best_register_gap']['register_minus_thinking']:+.2f}); "
          f"most thinking-skewed L{pos['best_thinking_gap']['layer']} "
          f"({pos['best_thinking_gap']['register_minus_thinking']:+.2f}); "
          f"family-over-variant L{pos['best_family_over_variant']['layer']} "
          f"({pos['best_family_over_variant']['family_minus_variant']:+.2f})",
          flush=True)


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    rng = np.random.default_rng(1)

    print("translation_thinking_v2 - register control", flush=True)
    M = load_model(MODEL)

    items = build_items()
    if args.max_items is not None:
        items = items[:args.max_items]
    prompts = [it["prompt"] for it in items]
    labels = {
        "answer": [it["answer_label"] for it in items],
        "operation": [it["operation_label"] for it in items],
        "register": [it["register_label"] for it in items],
        "variant": [it["register_variant"] for it in items],
    }
    print(f"\n=== {len(TARGETS)} answers x {len(OPS)} operations x "
          f"{len(REGISTERS)} registers x 2 phrasings = {len(items)} prompts ===",
          flush=True)

    print("\n=== extracting pre-answer states ===", flush=True)
    pre = extract(M, prompts, read="last", label="pre", verbose=False).cpu().numpy()
    print("=== extracting generated-token render states ===", flush=True)
    render = extract(
        M,
        prompts,
        read="generation",
        max_new_tokens=args.max_new_tokens,
        label="render",
        verbose=False,
    ).cpu().numpy()

    pre_pos = score_position("pre", pre, labels, rng, args.n_shuffle)
    render_pos = score_position("render", render, labels, rng, args.n_shuffle)
    print_summary(pre_pos)
    print_summary(render_pos)

    out = {
        "model": MODEL,
        "n": len(items),
        "targets": TARGETS,
        "operations": OPS,
        "registers": list(REGISTERS),
        "max_new_tokens": args.max_new_tokens,
        "n_shuffle": args.n_shuffle,
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "positions": {"pre": pre_pos, "render": render_pos},
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"translation_thinking_v2_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
