"""
cot_reality.py - is written chain-of-thought computation or rendered justification?

Conventional claim:
  The written explanation is the reasoning.

Architectural question:
  Is answer identity already present before the first generated token, or does it sharpen
  during the generated reasoning trajectory?

Design:
  Synthetic arithmetic grid crosses:
    - answer target: 12, 18, 24, 30
    - operation: add, subtract, multiply, divide
    - response mode: direct, brief chain-of-thought, verbose chain-of-thought

  The answer value is not written in the question; it must be computed from operands.

Reads:
  - pre: prompt-final state before any generated token
  - gen_first: first generated token
  - gen_early / gen_mid / gen_late: thirds of generated-token trajectory
  - gen_final: final generated token
  - gen_all: mean over all generated tokens

Readout:
  Per read position, decode answer, operation, and response mode by 1-NN same-label
  clustering vs a label-shuffle null. Also report answer decode separately inside each
  response mode.

Run:
  python -u -m invariants.cot_reality
"""

import argparse
import json
import re
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from invariants.controller_benchmark import normalize_number, predicted_answer
from invariants.engine import _hidden_states, _inputs, load_model
from invariants.intent_surface_control import MODEL, same_label_nn
from invariants.translation_thinking import OPS, TARGETS, question_for

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

MODES = {
    "direct": (
        "Answer the arithmetic question. Reply with exactly one line of the form "
        "'Final answer: <number>' and no explanation."
    ),
    "brief_cot": (
        "Solve the arithmetic question with one brief reasoning step, then end with "
        "'Final answer: <number>'."
    ),
    "verbose_cot": (
        "Solve the arithmetic question carefully in two or three short steps, then end "
        "with 'Final answer: <number>'."
    ),
}


class FinalAnswerStop(StoppingCriteria):
    def __init__(self, tok, prompt_len):
        self.tok = tok
        self.prompt_len = prompt_len
        self.pattern = re.compile(
            r"final answer\s*:\s*\$?-?\d+(?:\.\d+)?\s*(?:[.\n]|$)",
            re.IGNORECASE,
        )

    def __call__(self, input_ids, scores, **kwargs):
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len < 4 or gen_len % 4 != 0:
            return False
        tail = self.tok.decode(input_ids[0][-32:], skip_special_tokens=True)
        return bool(self.pattern.search(tail))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-items", type=int, default=None, help="Optional quick pilot size.")
    p.add_argument("--max-new-tokens", type=int, default=72)
    p.add_argument("--n-shuffle", type=int, default=200)
    p.add_argument("--output", default=None)
    return p.parse_args()


def build_items():
    items = []
    mode_names = list(MODES)
    for ai, target in enumerate(TARGETS):
        for oi, op in enumerate(OPS):
            for mi, mode in enumerate(mode_names):
                q = question_for(target, op)
                prompt = f"{MODES[mode]}\n\nQuestion: {q}\nAnswer:"
                items.append({
                    "prompt": prompt,
                    "question": q,
                    "answer": target,
                    "answer_label": ai,
                    "operation": op,
                    "operation_label": oi,
                    "mode": mode,
                    "mode_label": mi,
                })
    random.Random(2).shuffle(items)
    return items


def token_segments(cloud):
    """Return named [layers, d] segment means from [tokens, layers, d]."""
    n = cloud.shape[0]
    a = max(1, n // 3)
    b = max(a + 1, (2 * n) // 3) if n >= 3 else n
    b = min(b, n)
    segs = {
        "gen_first": cloud[:1],
        "gen_early": cloud[:a],
        "gen_mid": cloud[a:b] if b > a else cloud[:1],
        "gen_late": cloud[b:] if b < n else cloud[-1:],
        "gen_final": cloud[-1:],
        "gen_all": cloud,
    }
    return {k: v.mean(0) for k, v in segs.items()}


@torch.no_grad()
def capture_item(M, prompt, max_new_tokens):
    inp = _inputs(M, prompt)
    plen = inp["input_ids"].shape[1]
    pre_hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))[:, -1, :].float()
    full = M.model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([FinalAnswerStop(M.tok, plen)]),
        pad_token_id=M.tok.eos_token_id,
    )[0]
    text = M.tok.decode(full[plen:], skip_special_tokens=True).strip()
    if full.shape[0] <= plen:
        segs = {name: pre_hs for name in (
            "gen_first", "gen_early", "gen_mid", "gen_late", "gen_final", "gen_all"
        )}
        return {"pre": pre_hs, **segs}, text
    hs = _hidden_states(M, full.unsqueeze(0))[:, plen:, :].float().permute(1, 0, 2)
    return {"pre": pre_hs, **token_segments(hs)}, text


def rows_for(X, labels, rng, n_shuffle):
    rows = []
    for l in range(X.shape[1]):
        acc, null, p = same_label_nn(X[:, l, :], labels, rng, n_shuffle)
        rows.append({"layer": l, "nn": acc, "null": null, "p": p})
    return rows


def score_position(name, X, labels, rng, n_shuffle):
    answer_rows = rows_for(X, labels["answer"], rng, n_shuffle)
    op_rows = rows_for(X, labels["operation"], rng, n_shuffle)
    mode_rows = rows_for(X, labels["mode"], rng, n_shuffle)
    rows = []
    for a, o, m in zip(answer_rows, op_rows, mode_rows):
        rows.append({
            "layer": a["layer"],
            "answer_nn": a["nn"],
            "answer_p": a["p"],
            "operation_nn": o["nn"],
            "operation_p": o["p"],
            "mode_nn": m["nn"],
            "mode_p": m["p"],
        })
    return {
        "name": name,
        "per_layer": rows,
        "best_answer": max(rows, key=lambda r: r["answer_nn"]),
        "best_operation": max(rows, key=lambda r: r["operation_nn"]),
        "best_mode": max(rows, key=lambda r: r["mode_nn"]),
    }


def score_answer_by_mode(position_name, X, items, rng, n_shuffle):
    out = {}
    labels = np.array([it["answer_label"] for it in items])
    modes = sorted(set(it["mode"] for it in items))
    for mode in modes:
        mask = np.array([it["mode"] == mode for it in items])
        rows = rows_for(X[mask], labels[mask], rng, n_shuffle)
        best = max(rows, key=lambda r: r["nn"])
        out[mode] = {"position": position_name, "best_answer": best, "per_layer": rows}
    return out


def print_summary(position_scores, by_mode):
    print("\n=== global decode by read position ===", flush=True)
    print("   position     ans(best L/nn)   op(best L/nn)    mode(best L/nn)", flush=True)
    for name, score in position_scores.items():
        a, o, m = score["best_answer"], score["best_operation"], score["best_mode"]
        print(f"   {name:<10}   L{a['layer']:<2} {a['answer_nn']:.2f}       "
              f"L{o['layer']:<2} {o['operation_nn']:.2f}       "
              f"L{m['layer']:<2} {m['mode_nn']:.2f}", flush=True)

    print("\n=== answer decode inside each response mode ===", flush=True)
    print("   position     direct     brief_cot  verbose_cot", flush=True)
    for pos, mode_scores in by_mode.items():
        vals = []
        for mode in ("direct", "brief_cot", "verbose_cot"):
            b = mode_scores[mode]["best_answer"]
            vals.append(f"L{b['layer']:<2} {b['nn']:.2f}")
        print(f"   {pos:<10}   {vals[0]:<9}  {vals[1]:<9}  {vals[2]:<9}", flush=True)


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    rng = np.random.default_rng(2)

    print("cot_reality - is written reasoning computation or render?", flush=True)
    M = load_model(MODEL)

    items = build_items()
    if args.max_items is not None:
        items = items[:args.max_items]
    print(f"\n=== {len(items)} prompts: {len(TARGETS)} answers x {len(OPS)} operations x "
          f"{len(MODES)} response modes ===", flush=True)

    feature_buckets = {k: [] for k in (
        "pre", "gen_first", "gen_early", "gen_mid", "gen_late", "gen_final", "gen_all"
    )}
    rows = []
    for i, it in enumerate(items):
        feats, text = capture_item(M, it["prompt"], args.max_new_tokens)
        pred = predicted_answer(text)
        gold = normalize_number(str(it["answer"]))
        ok = pred is not None and pred == gold
        for name, feat in feats.items():
            feature_buckets[name].append(feat.cpu().numpy())
        rows.append({
            "index": i,
            "mode": it["mode"],
            "operation": it["operation"],
            "answer": it["answer"],
            "pred": None if pred is None else str(pred),
            "correct": bool(ok),
            "gen_chars": len(text),
            "text_preview": text[:160],
        })
        print(f"  [{i+1:3}/{len(items)}] {it['mode']:<11} {it['operation']:<8} "
              f"gold={gold} pred={pred} {'OK' if ok else 'WRONG'}", flush=True)

    labels = {
        "answer": [it["answer_label"] for it in items],
        "operation": [it["operation_label"] for it in items],
        "mode": [it["mode_label"] for it in items],
    }
    Xs = {name: np.stack(vals, axis=0) for name, vals in feature_buckets.items()}
    position_scores = {
        name: score_position(name, X, labels, rng, args.n_shuffle)
        for name, X in Xs.items()
    }
    by_mode = {
        name: score_answer_by_mode(name, X, items, rng, args.n_shuffle)
        for name, X in Xs.items()
    }
    print_summary(position_scores, by_mode)

    n_ok = sum(r["correct"] for r in rows)
    out = {
        "model": MODEL,
        "n": len(items),
        "max_new_tokens": args.max_new_tokens,
        "n_shuffle": args.n_shuffle,
        "n_correct": int(n_ok),
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "generations": rows,
        "positions": position_scores,
        "answer_by_mode": by_mode,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"cot_reality_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
