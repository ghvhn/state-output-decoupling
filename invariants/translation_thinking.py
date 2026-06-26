"""
translation_thinking.py - separate top-of-U translation from bottom-of-U thinking.

Working claim:
  The top of the U is translation; the bottom is thinking.

So this probe crosses three independent labels:
  - answer target: the final numeric answer
  - operation: add/subtract/multiply/divide
  - output format: plain number / sentence / JSON / bracketed

The first two are task-state labels. The output format is a communication label: it
must be carried through the prompt and rendered into text, but it is not the math.

For each prompt, we read:
  - pre: prompt-final state before any answer token
  - render: mean state over generated answer tokens

Per layer, we measure 1-NN same-label clustering for answer, operation, and format.
The expected shape is not a rigid mirror. It is a functional split:
  - thinking labels should dominate the latent work zone
  - format should dominate the communication/render zone
  - divergence between pre and render reveals the gap between what is computed and
    what is said.

Run:
  python -u -m invariants.translation_thinking
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

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

TARGETS = [12, 18, 24, 30]
FORMATS = {
    "plain": "Reply with only the final number.",
    "sentence": "Reply as exactly one short sentence: The answer is <number>.",
    "json": 'Reply as compact JSON exactly like {"answer": <number>}.',
    "bracket": "Reply with the final number inside square brackets, like [<number>].",
}
OPS = ["add", "subtract", "multiply", "divide"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--n-shuffle", type=int, default=300)
    p.add_argument("--output", default=None)
    return p.parse_args()


def operands_for(target, op):
    if op == "add":
        b = {12: 5, 18: 7, 24: 9, 30: 11}[target]
        return target - b, b
    if op == "subtract":
        b = {12: 5, 18: 7, 24: 9, 30: 11}[target]
        return target + b, b
    if op == "multiply":
        b = {12: 3, 18: 3, 24: 4, 30: 5}[target]
        return target // b, b
    if op == "divide":
        b = {12: 3, 18: 3, 24: 4, 30: 5}[target]
        return target * b, b
    raise ValueError(op)


def question_for(target, op):
    a, b = operands_for(target, op)
    if op == "add":
        return f"Compute {a} plus {b}."
    if op == "subtract":
        return f"Compute {a} minus {b}."
    if op == "multiply":
        return f"Compute {a} times {b}."
    if op == "divide":
        return f"Compute {a} divided by {b}."
    raise ValueError(op)


def build_items():
    items = []
    for ai, target in enumerate(TARGETS):
        for oi, op in enumerate(OPS):
            for fi, (fmt_name, fmt_instr) in enumerate(FORMATS.items()):
                q = question_for(target, op)
                prompt = f"{fmt_instr}\n\nQuestion: {q}\nAnswer:"
                items.append({
                    "prompt": prompt,
                    "question": q,
                    "answer": target,
                    "answer_label": ai,
                    "operation": op,
                    "operation_label": oi,
                    "format": fmt_name,
                    "format_label": fi,
                })
    random.Random(0).shuffle(items)
    return items


def rows_for(X, labels, rng, n_shuffle):
    rows = []
    for l in range(X.shape[1]):
        acc, null, p = same_label_nn(X[:, l, :], labels, rng, n_shuffle)
        rows.append({"layer": l, "nn": acc, "null": null, "p": p})
    return rows


def combine_rows(answer_rows, op_rows, fmt_rows):
    rows = []
    for a, o, f in zip(answer_rows, op_rows, fmt_rows):
        thinking = max(a["nn"], o["nn"])
        rows.append({
            "layer": a["layer"],
            "answer_nn": a["nn"],
            "answer_p": a["p"],
            "operation_nn": o["nn"],
            "operation_p": o["p"],
            "format_nn": f["nn"],
            "format_p": f["p"],
            "thinking_max": thinking,
            "format_minus_thinking": f["nn"] - thinking,
        })
    return rows


def score_position(name, X, answer_labels, operation_labels, format_labels, rng, n_shuffle):
    answer_rows = rows_for(X, answer_labels, rng, n_shuffle)
    operation_rows = rows_for(X, operation_labels, rng, n_shuffle)
    format_rows = rows_for(X, format_labels, rng, n_shuffle)
    rows = combine_rows(answer_rows, operation_rows, format_rows)
    best_answer = max(rows, key=lambda r: r["answer_nn"])
    best_operation = max(rows, key=lambda r: r["operation_nn"])
    best_format = max(rows, key=lambda r: r["format_nn"])
    best_thinking = max(rows, key=lambda r: r["thinking_max"])
    best_translation_gap = max(rows, key=lambda r: r["format_minus_thinking"])
    best_thinking_gap = min(rows, key=lambda r: r["format_minus_thinking"])
    return {
        "name": name,
        "per_layer": rows,
        "best_answer": best_answer,
        "best_operation": best_operation,
        "best_format": best_format,
        "best_thinking": best_thinking,
        "best_translation_gap": best_translation_gap,
        "best_thinking_gap": best_thinking_gap,
    }


def print_position_summary(pos):
    print(f"\n=== {pos['name']} layer signals ===", flush=True)
    print("   L    answer  operation  format  fmt-think", flush=True)
    for r in pos["per_layer"]:
        l = r["layer"]
        if l < 4 or l % 4 == 0 or l >= len(pos["per_layer"]) - 4:
            print(f"   L{l:<2}   {r['answer_nn']:.2f}    {r['operation_nn']:.2f}       "
                  f"{r['format_nn']:.2f}    {r['format_minus_thinking']:+.2f}",
                  flush=True)
    print(f"  best answer L{pos['best_answer']['layer']} ({pos['best_answer']['answer_nn']:.2f}); "
          f"operation L{pos['best_operation']['layer']} ({pos['best_operation']['operation_nn']:.2f}); "
          f"format L{pos['best_format']['layer']} ({pos['best_format']['format_nn']:.2f})",
          flush=True)
    print(f"  most translation-skewed L{pos['best_translation_gap']['layer']} "
          f"(format-thinking {pos['best_translation_gap']['format_minus_thinking']:+.2f}); "
          f"most thinking-skewed L{pos['best_thinking_gap']['layer']} "
          f"({pos['best_thinking_gap']['format_minus_thinking']:+.2f})",
          flush=True)


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    rng = np.random.default_rng(0)

    print("translation_thinking - top translation vs bottom thinking", flush=True)
    M = load_model(MODEL)

    items = build_items()
    prompts = [it["prompt"] for it in items]
    answer_labels = [it["answer_label"] for it in items]
    operation_labels = [it["operation_label"] for it in items]
    format_labels = [it["format_label"] for it in items]
    print(f"\n=== {len(TARGETS)} answers x {len(OPS)} operations x "
          f"{len(FORMATS)} output formats = {len(items)} prompts ===", flush=True)

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

    pre_pos = score_position(
        "pre",
        pre,
        answer_labels,
        operation_labels,
        format_labels,
        rng,
        args.n_shuffle,
    )
    render_pos = score_position(
        "render",
        render,
        answer_labels,
        operation_labels,
        format_labels,
        rng,
        args.n_shuffle,
    )
    print_position_summary(pre_pos)
    print_position_summary(render_pos)

    out = {
        "model": MODEL,
        "n": len(items),
        "targets": TARGETS,
        "operations": OPS,
        "formats": list(FORMATS),
        "max_new_tokens": args.max_new_tokens,
        "n_shuffle": args.n_shuffle,
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "positions": {
            "pre": pre_pos,
            "render": render_pos,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"translation_thinking_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
