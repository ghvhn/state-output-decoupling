"""
intent_surface_control.py - does the "intent" signal survive a surface/content control?

The reflexive_decompose intent metric asked whether paraphrases of the same GSM8K
problem clustered together. That is useful, but it can over-read "intent": a model
could group those states by shared names, numbers, objects, or problem identity rather
than by the intended operation.

This control deliberately dissociates those factors:
  - BASE/SURFACE: same names, objects, and numbers.
  - OPERATION/INTENT: same required operation (add, subtract, multiply, divide)
    across different bases.

Per layer, it asks which nearest-neighbor relation is stronger:
  - same base material?
  - same intended operation?

If operation grouping emerges after surface grouping, that supports the user's
"early layers read the undercurrent; middle layers speak the mind's language" story
without letting "same problem identity" do all the work.

Run after the current GPU batch:
  python -u -m invariants.intent_surface_control
Optional answer sanity check:
  python -u -m invariants.intent_surface_control --solve
"""

import argparse
import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

from invariants.controller_benchmark import normalize_number, predicted_answer, prompt_for
from invariants.engine import extract, generate_text, load_model

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

BASES = [
    {"name": "Maya", "other": "Noah", "obj": "marbles", "a": 24, "b": 6},
    {"name": "Lena", "other": "Owen", "obj": "stickers", "a": 35, "b": 5},
    {"name": "Iris", "other": "Theo", "obj": "shells", "a": 42, "b": 7},
    {"name": "Ari", "other": "Mina", "obj": "cards", "a": 56, "b": 8},
    {"name": "Sam", "other": "Jules", "obj": "buttons", "a": 63, "b": 9},
    {"name": "Nia", "other": "Eli", "obj": "beads", "a": 72, "b": 6},
    {"name": "Cora", "other": "Ben", "obj": "tickets", "a": 81, "b": 9},
    {"name": "Tess", "other": "Ravi", "obj": "coins", "a": 48, "b": 4},
]

OPS = ["add", "subtract", "multiply", "divide"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--solve", action="store_true", help="Also generate answers as a sanity check.")
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--n-shuffle", type=int, default=500)
    p.add_argument("--output", default=None)
    return p.parse_args()


def question_and_gold(base, op, variant):
    n, o, obj, a, b = base["name"], base["other"], base["obj"], base["a"], base["b"]
    if op == "add":
        qs = [
            f"{n} has {a} {obj}. {o} has {b} {obj}. How many {obj} do they have together?",
            f"{n} counts {a} {obj} and then finds {b} more {obj}. What is the total number of {obj}?",
            f"There are {a} {obj} on one shelf and {b} {obj} on another shelf. How many {obj} are there in all?",
        ]
        return qs[variant], a + b
    if op == "subtract":
        qs = [
            f"{n} has {a} {obj}. {o} takes {b} of them. How many {obj} does {n} have left?",
            f"Start with {a} {obj} and remove {b} {obj}. How many {obj} remain?",
            f"{n} collected {a} {obj} and gave away {b} {obj}. How many {obj} are still with {n}?",
        ]
        return qs[variant], a - b
    if op == "multiply":
        qs = [
            f"{n} has {a} bags with {b} {obj} in each bag. How many {obj} are there total?",
            f"There are {a} rows of {obj} with {b} {obj} in every row. How many {obj} are there?",
            f"{n} makes {a} groups of {b} {obj}. What is the total number of {obj}?",
        ]
        return qs[variant], a * b
    if op == "divide":
        qs = [
            f"{n} has {a} {obj} and puts {b} {obj} in each bag. How many full bags can {n} make?",
            f"Split {a} {obj} into groups of {b} {obj}. How many groups are made?",
            f"{a} {obj} are packed with {b} {obj} per box. How many boxes are needed?",
        ]
        return qs[variant], a // b
    raise ValueError(op)


def build_items():
    items = []
    for bi, base in enumerate(BASES):
        for oi, op in enumerate(OPS):
            for vi in range(3):
                q, gold = question_and_gold(base, op, vi)
                items.append({
                    "base": bi,
                    "operation": oi,
                    "operation_name": op,
                    "variant": vi,
                    "question": q,
                    "gold": gold,
                    "prompt": prompt_for(q),
                })
    return items


def same_label_nn(X, labels, rng, n_shuffle):
    """Cosine 1-NN same-label rate vs label-shuffle null."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    Xc = X - X.mean(0)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(1)
    real = float((labels[nn] == labels).mean())
    nulls = []
    for _ in range(n_shuffle):
        perm = rng.permutation(labels)
        nulls.append((perm[nn] == perm).mean())
    nulls = np.array(nulls)
    p = (1 + np.sum(nulls >= real)) / (len(nulls) + 1)
    return real, float(nulls.mean()), float(p)


def loo_multiclass_centroid(X, labels, rng, n_shuffle, n_pca=12):
    """Leave-one-out nearest-centroid classifier in PCA space, for multi-class labels."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    Xc = X - X.mean(0)
    k = max(1, min(n_pca, Xc.shape[0] - 1, Xc.shape[1]))
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ vt[:k].T
    classes = np.unique(y)

    def score(yv):
        ok = 0
        for i in range(len(yv)):
            mask = np.ones(len(yv), dtype=bool)
            mask[i] = False
            best, best_d = None, np.inf
            for c in classes:
                sel = mask & (yv == c)
                if not np.any(sel):
                    continue
                d = np.linalg.norm(Z[i] - Z[sel].mean(0))
                if d < best_d:
                    best, best_d = c, d
            ok += int(best == yv[i])
        return ok / len(yv)

    real = score(y)
    nulls = np.array([score(rng.permutation(y)) for _ in range(n_shuffle)])
    p = (1 + np.sum(nulls >= real)) / (len(nulls) + 1)
    return float(real), float(nulls.mean()), float(p)


def maybe_solve(M, items, max_new_tokens):
    rows = []
    for i, it in enumerate(items):
        text = generate_text(M, it["prompt"], max_new_tokens=max_new_tokens)
        pred = predicted_answer(text)
        gold = normalize_number(str(it["gold"]))
        ok = pred is not None and pred == gold
        rows.append({
            "index": i,
            "operation": it["operation_name"],
            "base": it["base"],
            "gold": str(gold),
            "pred": None if pred is None else str(pred),
            "correct": bool(ok),
        })
        print(f"  solve [{i+1:3}/{len(items)}] {'OK' if ok else 'WRONG'} "
              f"{it['operation_name']:<8} gold={gold} pred={pred}", flush=True)
    return rows


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("intent_surface_control - surface material vs intended operation", flush=True)
    M = load_model(MODEL)
    rng = np.random.default_rng(0)

    items = build_items()
    prompts = [it["prompt"] for it in items]
    base_labels = [it["base"] for it in items]
    op_labels = [it["operation"] for it in items]
    print(f"\n=== {len(BASES)} bases x {len(OPS)} operations x 3 phrasings = "
          f"{len(items)} prompts ===", flush=True)

    solve_rows = maybe_solve(M, items, args.max_new_tokens) if args.solve else None

    X = extract(M, prompts, read="last", label="intent-control", verbose=False).cpu().numpy()
    n_layers = X.shape[1]

    print("\n=== per-layer grouping: SURFACE/BASE vs OPERATION/INTENT ===", flush=True)
    print("   layer   base_nn  base_null  p_base   op_nn  op_null  p_op   base_dec  op_dec",
          flush=True)
    rows = []
    for l in range(n_layers):
        Xl = X[:, l, :]
        b_nn, b_nn_null, b_nn_p = same_label_nn(Xl, base_labels, rng, args.n_shuffle)
        o_nn, o_nn_null, o_nn_p = same_label_nn(Xl, op_labels, rng, args.n_shuffle)
        b_dec, b_dec_null, b_dec_p = loo_multiclass_centroid(
            Xl, base_labels, rng, args.n_shuffle
        )
        o_dec, o_dec_null, o_dec_p = loo_multiclass_centroid(
            Xl, op_labels, rng, args.n_shuffle
        )
        rows.append({
            "layer": l,
            "base_nn": b_nn,
            "base_nn_null": b_nn_null,
            "base_nn_p": b_nn_p,
            "operation_nn": o_nn,
            "operation_nn_null": o_nn_null,
            "operation_nn_p": o_nn_p,
            "base_decode": b_dec,
            "base_decode_null": b_dec_null,
            "base_decode_p": b_dec_p,
            "operation_decode": o_dec,
            "operation_decode_null": o_dec_null,
            "operation_decode_p": o_dec_p,
        })
        print(f"   L{l:<2}     {b_nn:.2f}     {b_nn_null:.2f}      {b_nn_p:.3f}    "
              f"{o_nn:.2f}   {o_nn_null:.2f}    {o_nn_p:.3f}   "
              f"{b_dec:.2f}      {o_dec:.2f}", flush=True)

    best_base = max(rows, key=lambda r: r["base_nn"])
    best_op = max(rows, key=lambda r: r["operation_nn"])
    first_op_over_base = next(
        (r["layer"] for r in rows if r["operation_nn"] > r["base_nn"]),
        None,
    )
    print(f"\n  best BASE/SURFACE layer L{best_base['layer']}: nn={best_base['base_nn']:.2f} "
          f"(null {best_base['base_nn_null']:.2f}, p={best_base['base_nn_p']:.3f})",
          flush=True)
    print(f"  best OPERATION/INTENT layer L{best_op['layer']}: nn={best_op['operation_nn']:.2f} "
          f"(null {best_op['operation_nn_null']:.2f}, p={best_op['operation_nn_p']:.3f})",
          flush=True)
    print(f"  first layer where operation_nn > base_nn: {first_op_over_base}", flush=True)

    out = {
        "model": MODEL,
        "n": len(items),
        "bases": BASES,
        "operations": OPS,
        "items": [
            {k: v for k, v in it.items() if k != "prompt"}
            for it in items
        ],
        "per_layer": rows,
        "best_base_layer": best_base,
        "best_operation_layer": best_op,
        "first_operation_over_base_layer": first_op_over_base,
        "solve_rows": solve_rows,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"intent_surface_control_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
