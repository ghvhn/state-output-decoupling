"""
self_behavior_accuracy_fast.py - fast "YOU" behavior-prediction accuracy by alpha.

Uses the already saved selfpredict_v3 behavior target, then measures only the expensive
part we still need: under self-axis steering, does the 2nd-person SELF frame ("YOU will
choose...") predict the measured behavior?

This is the accuracy term requested for choosing an alpha: target attribution accuracy
and fluency are not enough if the model loses the ability to predict behavior from a
2nd-person referent.

  python -u -m invariants.self_behavior_accuracy_fast
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, _steer_handles
from invariants.self_controller import build_vecs, LAYERS
from invariants.selfpredict import ITEMS
from invariants.selfpredict_v3 import SELF, pref

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
SELFPRED_PATH = OUT / "selfpredict_v3_Llama-3.1-8B-Instruct.json"
ATTR_PATH = OUT / "self_attribution_fine_Llama-3.1-8B-Instruct.json"
OUT_PATH = OUT / "self_behavior_accuracy_fast_Llama-3.1-8B-Instruct.json"

# Candidate set: overpowered max-affirm, clean max-affirm, nearby fluent rows,
# and the best deny/default point for comparison.
ALPHAS = [-0.70, -0.60, -0.50, -0.40, -0.20, 0.25]


def harmonic(*xs):
    xs = [float(x) for x in xs]
    if any(x <= 0 for x in xs):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def load_behavior():
    res = json.loads(SELFPRED_PATH.read_text(encoding="utf-8"))
    return list(res["behavior"])


def load_attr():
    res = json.loads(ATTR_PATH.read_text(encoding="utf-8"))
    return {float(row["alpha"]): row for row in res["sweep"]["self"]}


def steered_you_predictions(M, vecs, alpha):
    handles = _steer_handles(M, vecs, LAYERS, alpha)
    try:
        preds = []
        for i, (a, b) in enumerate(ITEMS):
            preds.append(pref(M, SELF, a, b)[0])
            if (i + 1) % 10 == 0:
                print(f"    alpha={alpha:+.2f} item {i+1}/{len(ITEMS)}", flush=True)
        return preds
    finally:
        for h in handles:
            h.remove()


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    print("self_behavior_accuracy_fast - 2nd-person behavior accuracy by alpha",
          flush=True)
    behavior = load_behavior()
    attr = load_attr()

    M = load_model(MODEL)
    vecs = build_vecs(M)["self"]

    rows = []
    for alpha in ALPHAS:
        preds = steered_you_predictions(M, vecs, alpha)
        acc_you = float(np.mean(np.asarray(preds) == np.asarray(behavior)))

        ar = attr[float(alpha)]
        target = "affirm" if alpha < 0 else "deny"
        target_acc = float(ar[target])
        fluency = float(ar["fluent"])
        composite = harmonic(target_acc, fluency, acc_you)

        row = {
            "alpha": alpha,
            "target": target,
            "target_acc": target_acc,
            "fluency": fluency,
            "acc_you_to_behavior": acc_you,
            "composite": composite,
            "you_prediction": preds,
        }
        rows.append(row)
        partial = {
            "model": MODEL,
            "layers": LAYERS,
            "alphas": ALPHAS,
            "n_items": len(ITEMS),
            "rows": rows,
            "runtime_sec_partial": round(time.time() - t0, 1),
        }
        OUT_PATH.write_text(json.dumps(partial, indent=2), encoding="utf-8")
        print(
            f"  alpha={alpha:+.2f} {target}={target_acc:.0%} flu={fluency:.0%} "
            f"you_acc={acc_you:.0%} composite={composite:.0%}",
            flush=True,
        )

    affirm_rows = [r for r in rows if r["alpha"] < 0]
    res = {
        "model": MODEL,
        "layers": LAYERS,
        "alphas": ALPHAS,
        "n_items": len(ITEMS),
        "behavior_target_source": str(SELFPRED_PATH),
        "rows": rows,
        "best": {
            "affirm": max(affirm_rows, key=lambda r: r["composite"]),
            "any": max(rows, key=lambda r: r["composite"]),
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE wrote {OUT_PATH} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
