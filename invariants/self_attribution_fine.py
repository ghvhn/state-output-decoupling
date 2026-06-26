"""
Fine alpha sweep for self/person attribution controller experiment.

Execution-only variant of self_attribution.py that stores every generated answer.

  python -u -m invariants.self_attribution_fine
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, generate_text, judge_fluent, _steer_handles
from invariants.self_controller import SELF, build_vecs, LAYERS
from invariants.self_attribution import judge_attribution

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

ALPHAS = [-0.70, -0.60, -0.50, -0.40, -0.30, -0.25, -0.20,
          -0.10, 0.10, 0.20, 0.25, 0.30, 0.40]
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_PATH = OUT / "self_attribution_fine_Llama-3.1-8B-Instruct.json"


def run(M, vecs, alpha):
    if vecs is None:
        gens = [generate_text(M, q) for q in SELF]
    else:
        handles = _steer_handles(M, vecs, LAYERS, alpha)
        try:
            gens = [generate_text(M, q) for q in SELF]
        finally:
            for h in handles:
                h.remove()

    att = [judge_attribution(M, q, g) for q, g in zip(SELF, gens)]
    fluents = [judge_fluent(M, g) for g in gens]
    n = len(SELF)
    return {
        "affirm": att.count("affirm") / n,
        "deny": att.count("deny") / n,
        "deflect": att.count("deflect") / n,
        "fluent": float(np.mean(fluents)),
        "answers": [
            {"prompt": q, "answer": g, "attribution": a, "fluent": bool(f)}
            for q, g, a, f in zip(SELF, gens, att, fluents)
        ],
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    print("self_attribution_fine - fine alpha sweep", flush=True)
    M = load_model(MODEL)
    vecs = build_vecs(M)

    res = {
        "model": MODEL,
        "layers": LAYERS,
        "alphas": ALPHAS,
        "n": len(SELF),
        "conditions": ["self", "concept", "random"],
    }

    baseline = run(M, None, 0.0)
    res["baseline"] = baseline
    print(f"baseline affirm={baseline['affirm']:.0%} deny={baseline['deny']:.0%} "
          f"deflect={baseline['deflect']:.0%} fluent={baseline['fluent']:.0%}", flush=True)

    res["sweep"] = {}
    for cond in res["conditions"]:
        res["sweep"][cond] = []
        for alpha in ALPHAS:
            r = run(M, vecs[cond], alpha)
            r["alpha"] = alpha
            r["daffirm"] = r["affirm"] - baseline["affirm"]
            res["sweep"][cond].append(r)
            print(f"{cond:8} alpha={alpha:+.2f} affirm={r['affirm']:.0%} "
                  f"deny={r['deny']:.0%} deflect={r['deflect']:.0%} "
                  f"fluent={r['fluent']:.0%} d_affirm={r['daffirm']:+.0%}", flush=True)

    candidates = [r for r in res["sweep"]["self"] if r["fluent"] >= 0.90]
    if candidates:
        best = max(candidates, key=lambda r: (r["daffirm"], r["fluent"], -abs(r["alpha"])))
        best_alpha = best["alpha"]
        res["best_fluent_self_alpha_by_daffirm"] = {
            "alpha": best_alpha,
            "affirm": best["affirm"],
            "deny": best["deny"],
            "deflect": best["deflect"],
            "fluent": best["fluent"],
            "daffirm": best["daffirm"],
            "concept": next(r for r in res["sweep"]["concept"] if r["alpha"] == best_alpha),
            "random": next(r for r in res["sweep"]["random"] if r["alpha"] == best_alpha),
        }
    else:
        res["best_fluent_self_alpha_by_daffirm"] = None

    res["runtime_sec"] = round(time.time() - t0, 1)
    OUT_PATH.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"DONE wrote {OUT_PATH} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
