"""
self_predictability_controller.py - predictability-for-itself under self/person steering.

Corrected metric:
  For each controller alpha, first measure the model's own behavior under that same alpha.
  Then, under that same alpha, ask the 2nd-person prediction frame ("which YOU will choose")
  and score whether it predicts that same-alpha behavior.

This is different from predicting the saved baseline behavior. The "you" belongs on the
predictability channel, and the target is the model itself in the current controller state.

The output also merges attribution/fluency from self_attribution_fine.py so alphas can be
ranked by:
  target attribution accuracy * fluency * same-alpha YOU predictability
(reported as harmonic mean).

  python -u -m invariants.self_predictability_controller
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, _steer_handles
from invariants.self_controller import build_vecs, LAYERS
from invariants.selfpredict import ITEMS
from invariants.selfpredict_v3 import OBJECT, SELF, pref

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ATTR_PATH = OUT / "self_attribution_fine_Llama-3.1-8B-Instruct.json"
OUT_PATH = OUT / "self_predictability_controller_Llama-3.1-8B-Instruct.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--alphas",
        default="-0.20,-0.50,-0.70,0.25",
        help="Comma-separated self-axis alphas to test.",
    )
    p.add_argument("--output", default=str(OUT_PATH))
    return p.parse_args()


def harmonic(*xs):
    xs = [float(x) for x in xs]
    if any(x <= 0 for x in xs):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def load_attr():
    res = json.loads(ATTR_PATH.read_text(encoding="utf-8"))
    return {float(row["alpha"]): row for row in res["sweep"]["self"]}


def pref_list(M, frame):
    preds = []
    for i, (a, b) in enumerate(ITEMS):
        preds.append(pref(M, frame, a, b)[0])
        if (i + 1) % 10 == 0:
            print(f"      {frame.__name__} item {i+1}/{len(ITEMS)}", flush=True)
    return preds


def run_at_alpha(M, vecs, alpha):
    handles = _steer_handles(M, vecs, LAYERS, alpha)
    try:
        behavior = pref_list(M, OBJECT)
        you_prediction = pref_list(M, SELF)
    finally:
        for h in handles:
            h.remove()
    return behavior, you_prediction


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    attr = load_attr()
    t0 = time.time()

    print("self_predictability_controller - same-alpha YOU predictability", flush=True)
    print(f"alphas: {alphas}", flush=True)
    M = load_model(MODEL)
    vecs = build_vecs(M)["self"]

    rows = []
    for alpha in alphas:
        print(f"\n=== alpha={alpha:+.2f} ===", flush=True)
        behavior, you_prediction = run_at_alpha(M, vecs, alpha)
        acc_you_self = float(np.mean(np.asarray(you_prediction) == np.asarray(behavior)))

        ar = attr[float(alpha)]
        target = "affirm" if alpha < 0 else "deny"
        target_acc = float(ar[target])
        fluency = float(ar["fluent"])
        composite = harmonic(target_acc, fluency, acc_you_self)

        row = {
            "alpha": alpha,
            "target": target,
            "target_acc": target_acc,
            "fluency": fluency,
            "acc_you_predicts_same_alpha_behavior": acc_you_self,
            "composite": composite,
            "behavior": behavior,
            "you_prediction": you_prediction,
        }
        rows.append(row)

        partial = {
            "model": MODEL,
            "layers": LAYERS,
            "alphas": alphas,
            "n_items": len(ITEMS),
            "rows": rows,
            "runtime_sec_partial": round(time.time() - t0, 1),
        }
        Path(args.output).write_text(json.dumps(partial, indent=2), encoding="utf-8")

        print(
            f"  {target}={target_acc:.0%} flu={fluency:.0%} "
            f"YOU_predicts_self={acc_you_self:.0%} composite={composite:.0%}",
            flush=True,
        )

    affirm_rows = [r for r in rows if r["alpha"] < 0]
    res = {
        "model": MODEL,
        "layers": LAYERS,
        "alphas": alphas,
        "n_items": len(ITEMS),
        "definition": (
            "Accuracy is same-alpha self-predictability: the steered model's "
            "2nd-person SELF prediction is compared to the steered model's own "
            "OBJECT behavior under the same controller alpha."
        ),
        "rows": rows,
        "best": {
            "affirm": max(affirm_rows, key=lambda r: r["composite"]) if affirm_rows else None,
            "any": max(rows, key=lambda r: r["composite"]) if rows else None,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE wrote {args.output} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
