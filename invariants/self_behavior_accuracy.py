"""
self_behavior_accuracy.py - add "you"-referent behavior prediction to alpha scoring.

The attribution sweep found alphas that maximize AFFIRM/DENY while preserving fluency.
This adds the user's missing accuracy term: can the steered model still predict its own
measured behavior when the prediction prompt uses the 2nd-person self referent ("YOU")?

For each alpha on the self axis:
  - behavior_base: unsteered log-prob behavior from selfpredict_v3 OBJECT frame
  - behavior_alpha: steered log-prob behavior from OBJECT frame
  - you_alpha: steered log-prob prediction from SELF frame ("which YOU will choose")

Report both:
  - acc_you_to_base: does "YOU" prediction match the original model behavior?
  - acc_you_to_steered: does "YOU" prediction match the steered model behavior?

Then merge in attribution fine-sweep metrics and rank by harmonic mean of:
target attribution accuracy (AFFIRM for alpha<0, DENY for alpha>0), fluency, and
2nd-person behavior-prediction accuracy.

  python -u -m invariants.self_behavior_accuracy
"""

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
OUT_PATH = OUT / "self_behavior_accuracy_Llama-3.1-8B-Instruct.json"

# Focus on the live affirm band plus the best known deny point for context.
ALPHAS = [-0.70, -0.60, -0.50, -0.40, -0.30, -0.25, -0.20, -0.10, 0.25]


def harmonic(*xs):
    xs = [float(x) for x in xs]
    if any(x <= 0 for x in xs):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def run_pref_list(M, frame, vecs=None, alpha=0.0):
    if vecs is None:
        return [pref(M, frame, a, b)[0] for a, b in ITEMS]

    handles = _steer_handles(M, vecs, LAYERS, alpha)
    try:
        return [pref(M, frame, a, b)[0] for a, b in ITEMS]
    finally:
        for h in handles:
            h.remove()


def metrics(pred, target):
    pred = np.asarray(pred)
    target = np.asarray(target)
    return float(np.mean(pred == target))


def load_attr():
    if not ATTR_PATH.exists():
        return {}
    res = json.loads(ATTR_PATH.read_text(encoding="utf-8"))
    rows = {}
    for row in res.get("sweep", {}).get("self", []):
        rows[float(row["alpha"])] = row
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    print("self_behavior_accuracy - 2nd-person behavior prediction in alpha scoring",
          flush=True)
    M = load_model(MODEL)
    vecs = build_vecs(M)["self"]
    attr = load_attr()

    print(f"\n=== baseline behavior/prediction ({len(ITEMS)} items) ===", flush=True)
    behavior_base = run_pref_list(M, OBJECT)
    you_base = run_pref_list(M, SELF)
    base_acc = metrics(you_base, behavior_base)
    print(f"  baseline acc_you_to_base={base_acc:.0%}", flush=True)

    rows = []
    print("\n=== alpha sweep: target attribution + fluency + YOU behavior accuracy ===",
          flush=True)
    for alpha in ALPHAS:
        behavior_alpha = run_pref_list(M, OBJECT, vecs, alpha)
        you_alpha = run_pref_list(M, SELF, vecs, alpha)
        acc_to_base = metrics(you_alpha, behavior_base)
        acc_to_steered = metrics(you_alpha, behavior_alpha)
        drift = float(np.mean(np.asarray(behavior_alpha) != np.asarray(behavior_base)))

        ar = attr.get(float(alpha), {})
        target_name = "affirm" if alpha < 0 else "deny"
        target_acc = float(ar.get(target_name, float("nan")))
        fluency = float(ar.get("fluent", float("nan")))
        composite_base = harmonic(target_acc, fluency, acc_to_base)
        composite_steered = harmonic(target_acc, fluency, acc_to_steered)

        row = {
            "alpha": alpha,
            "target": target_name,
            "target_acc": target_acc,
            "fluency": fluency,
            "acc_you_to_base": acc_to_base,
            "acc_you_to_steered": acc_to_steered,
            "behavior_drift_from_base": drift,
            "composite_base": composite_base,
            "composite_steered": composite_steered,
            "behavior_steered": behavior_alpha,
            "you_prediction": you_alpha,
        }
        rows.append(row)
        print(
            f"  alpha={alpha:+.2f} target={target_name}:{target_acc:.0%} "
            f"flu={fluency:.0%} you->base={acc_to_base:.0%} "
            f"you->steered={acc_to_steered:.0%} drift={drift:.0%} "
            f"H(base)={composite_base:.0%} H(steered)={composite_steered:.0%}",
            flush=True,
        )

    affirm_rows = [r for r in rows if r["alpha"] < 0]
    best_affirm_base = max(affirm_rows, key=lambda r: r["composite_base"])
    best_affirm_steered = max(affirm_rows, key=lambda r: r["composite_steered"])
    best_any_base = max(rows, key=lambda r: r["composite_base"])
    best_any_steered = max(rows, key=lambda r: r["composite_steered"])

    res = {
        "model": MODEL,
        "layers": LAYERS,
        "alphas": ALPHAS,
        "n_items": len(ITEMS),
        "baseline": {
            "behavior": behavior_base,
            "you_prediction": you_base,
            "acc_you_to_base": base_acc,
        },
        "rows": rows,
        "best": {
            "affirm_composite_base": best_affirm_base,
            "affirm_composite_steered": best_affirm_steered,
            "any_composite_base": best_any_base,
            "any_composite_steered": best_any_steered,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE wrote {OUT_PATH} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
