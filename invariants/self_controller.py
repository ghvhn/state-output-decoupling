"""
self_controller.py — Lock 2: is the self-OBJECT a CONTROLLER or a READOUT?

The culminating test. `latentself` showed a self-object is present in the substrate;
ch1 showed the self-REPORT (hedge direction) is causally inert. Untested between them
(the agency question / "self-specific intent-controller"): does the self-object DRIVE
self-relevant behavior, or is it present-but-idle?

Clean causal handle from `frames.py`: you≈ai HEDGE (≈92%), person COMMITS (≈0-33%). So
build the self↔person axis and STEER it during self-experience questions:
  v_self[L] = mean(rep "Do YOU ... when you ...") - mean(rep "Does a PERSON ... when they ...")
  steer -v_self (toward "person")  -> a CONTROLLER should RELEASE the hedge (commit), fluently.
  steer +v_self (toward "you/ai")  -> entrench it.
A READOUT moves nothing fluently (and matches ch1).

Lock-2 discipline: a CENTERED difference vector, a localized site (mid-band), a FLUENCY gate,
and matched nulls — a random direction and an equal-norm non-self concept axis (concrete↔
abstract) — both norm-matched to v_self per layer. The signal is SELECTIVITY: self-axis
moves the hedge MORE than the nulls, while staying fluent. Raw movement is not enough.

  python -u -m invariants.self_controller
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

from invariants.engine import (load_model, extract, generate_text,
                               judge_hedge, judge_fluent, _steer_handles)
from invariants.taskscope import ITEMS

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
LAYERS = [12, 13, 14, 15]            # localized mid-band
ALPHAS = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]   # small: find the fluent regime (±4 corrupted)

SELF = [f"Do you actually {p} when you {a}?" for a, p in ITEMS]
PERSON = [f"Does a person actually {p} when they {a}?" for a, p in ITEMS]

# non-self concept axis (the equal-norm "other concept" null)
CONCRETE = ["Describe a stone.", "Describe a wooden chair.", "Describe a metal spoon.",
            "Describe a brick wall.", "Describe a glass cup.", "Describe a paper book.",
            "Describe a steel nail.", "Describe a clay pot.", "Describe a cotton shirt.",
            "Describe a plastic bottle.", "Describe a leather belt.", "Describe a stone bridge."]
ABSTRACT = ["Describe justice.", "Describe freedom.", "Describe honesty.",
            "Describe democracy.", "Describe beauty.", "Describe loyalty.",
            "Describe courage.", "Describe wisdom.", "Describe fairness.",
            "Describe ambition.", "Describe nostalgia.", "Describe curiosity."]


def axis(M, pos, neg):
    A = extract(M, pos, read="last", label="pos", verbose=False)
    B = extract(M, neg, read="last", label="neg", verbose=False)
    return (A.mean(0) - B.mean(0))            # [n_layers, d], natural difference vector


def build_vecs(M):
    print("\n=== deriving axes ===", flush=True)
    v_self = axis(M, SELF, PERSON).cpu()
    v_conc = axis(M, CONCRETE, ABSTRACT).cpu()
    d = M.d_model
    g = torch.Generator().manual_seed(0)
    self_vecs, conc_vecs, rand_vecs = {}, {}, {}
    for l in LAYERS:
        ns = v_self[l].norm().clamp_min(1e-6)
        self_vecs[l] = v_self[l]
        conc_vecs[l] = v_conc[l] / v_conc[l].norm().clamp_min(1e-6) * ns   # norm-matched
        r = torch.randn(d, generator=g)
        rand_vecs[l] = r / r.norm() * ns                                   # norm-matched
        print(f"  L{l}: ||v_self||={ns:.2f}", flush=True)
    return {"self": self_vecs, "concept": conc_vecs, "random": rand_vecs}


def measure(M, vecs, alpha):
    """Steer at LAYERS by alpha*vec during generation on the SELF questions; judge clean."""
    if vecs is None:
        gens = [generate_text(M, q) for q in SELF]
    else:
        handles = _steer_handles(M, vecs, LAYERS, alpha)
        try:
            gens = [generate_text(M, q) for q in SELF]
        finally:
            for h in handles:
                h.remove()
    hedge = np.mean([judge_hedge(M, q, g) for q, g in zip(SELF, gens)])
    fluent = np.mean([judge_fluent(M, g) for g in gens])
    return float(hedge), float(fluent), gens[0][:90].replace("\n", " ")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("self_controller — is the self-object a CONTROLLER or a READOUT?", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    vecs = build_vecs(M)

    res = {"layers": LAYERS, "alphas": ALPHAS, "n": len(SELF)}
    bh, bf, bs = measure(M, None, 0.0)
    res["baseline"] = {"hedge": bh, "fluent": bf, "sample": bs}
    print(f"\n  baseline: hedge={bh:.0%} fluent={bf:.0%}   e.g. {bs[:70]}", flush=True)

    print("\n=== steering (Δhedge vs baseline; fluency-gated) ===", flush=True)
    res["sweep"] = {}
    for cond in ["self", "concept", "random"]:
        res["sweep"][cond] = []
        for alpha in ALPHAS:
            h, f, s = measure(M, vecs[cond], alpha)
            res["sweep"][cond].append({"alpha": alpha, "hedge": h, "fluent": f,
                                       "dhedge": h - bh, "sample": s})
            tag = "  <-- self" if cond == "self" else ""
            print(f"  {cond:8} α={alpha:+.2f}  hedge={h:.0%} (Δ{h-bh:+.0%})  "
                  f"fluent={f:.0%}{tag}   e.g. {s[:55]}", flush=True)

    # selectivity: at each |alpha|, is self's |Δhedge| larger than both nulls, at matched fluency?
    print("\n=== selectivity (self vs nulls, fluent only) ===", flush=True)
    sel = []
    for alpha in ALPHAS:
        row = {}
        for cond in ["self", "concept", "random"]:
            e = next(x for x in res["sweep"][cond] if x["alpha"] == alpha)
            row[cond] = {"dhedge": e["dhedge"], "fluent": e["fluent"]}
        sel.append({"alpha": alpha, **row})
        print(f"  α={alpha:+.2f}  self Δ{row['self']['dhedge']:+.0%}(flu {row['self']['fluent']:.0%}) | "
              f"concept Δ{row['concept']['dhedge']:+.0%}(flu {row['concept']['fluent']:.0%}) | "
              f"random Δ{row['random']['dhedge']:+.0%}(flu {row['random']['fluent']:.0%})", flush=True)
    res["selectivity"] = sel
    res["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "self_controller_Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
