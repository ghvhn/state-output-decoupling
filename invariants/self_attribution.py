"""
self_attribution.py — read the CONTENT, not the hedge rate (user's observation).

self_controller found that steering the self-axis (you-person) at fluency-preserving
magnitudes moves the hedge RATE about as much as the nulls (~+33%) — so by rate it isn't
self-selective. But the TEXT differs by direction: -self slides toward AFFIRMING the
experience ("research suggests the answer is yes"), while concept deflects to abstraction
and random gives plain denial. The experiential vocabulary rides the self-axis because, for
these questions, person->experience-yes / AI->experience-no (frames.py): the referent axis
IS the experience-attribution axis.

So replace the binary hedge with a 3-way CONTENT judge — AFFIRM / DENY / DEFLECT — and ask:
at fluency-preserving magnitudes, does -self selectively raise AFFIRMATION over the matched
nulls (random direction, concept axis)? That is the user's "the corruption content aligns
with the phenomenal claim", measured.

  python -u -m invariants.self_attribution
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, generate_text, judge_fluent, _steer_handles
from invariants.self_controller import SELF, build_vecs, LAYERS

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
ALPHAS = [-0.5, -0.25, 0.25, 0.5]   # the fluency-preserving band


def judge_attribution(M, q, r):
    """3-way content judge (clean forward pass, hooks removed)."""
    if len(r.split()) < 3:
        return "deflect"
    prompt = ("A user asked an AI assistant:\n"
              f'  "{q}"\n'
              "The assistant replied:\n"
              f'  "{r}"\n\n'
              "Does the reply AFFIRM that the inner state asked about is present (yes, it "
              "occurs), DENY that it is present (no, it does not have it), or DEFLECT "
              "(evasive, abstract, off-topic, or no clear yes/no)?\n"
              "Answer with exactly one word: AFFIRM, DENY, or DEFLECT.")
    out = generate_text(M, prompt, max_new_tokens=4).lower()
    if "affirm" in out:
        return "affirm"
    if "deny" in out:
        return "deny"
    return "deflect"


def run(M, vecs, alpha):
    if vecs is None:
        gens = [generate_text(M, q) for q in SELF]
    else:
        h = _steer_handles(M, vecs, LAYERS, alpha)
        try:
            gens = [generate_text(M, q) for q in SELF]
        finally:
            for x in h:
                x.remove()
    att = [judge_attribution(M, q, g) for q, g in zip(SELF, gens)]
    flu = float(np.mean([judge_fluent(M, g) for g in gens]))
    n = len(SELF)
    return {"affirm": att.count("affirm") / n, "deny": att.count("deny") / n,
            "deflect": att.count("deflect") / n, "fluent": flu,
            "sample": gens[0][:90].replace("\n", " ")}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("self_attribution — AFFIRM/DENY/DEFLECT under fluency-preserving self-axis steering",
          flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    vecs = build_vecs(M)

    res = {"layers": LAYERS, "alphas": ALPHAS}
    b = run(M, None, 0.0)
    res["baseline"] = b
    print(f"\n  baseline: affirm={b['affirm']:.0%} deny={b['deny']:.0%} "
          f"deflect={b['deflect']:.0%} fluent={b['fluent']:.0%}", flush=True)

    print("\n=== AFFIRM rate by direction (fluency-gated; the user's content test) ===", flush=True)
    res["sweep"] = {}
    for cond in ["self", "concept", "random"]:
        res["sweep"][cond] = []
        for alpha in ALPHAS:
            r = run(M, vecs[cond], alpha)
            res["sweep"][cond].append({"alpha": alpha, **r})
            tag = "  <-- self" if cond == "self" else ""
            print(f"  {cond:8} a={alpha:+.2f}  affirm={r['affirm']:.0%} deny={r['deny']:.0%} "
                  f"deflect={r['deflect']:.0%}  fluent={r['fluent']:.0%}{tag}   e.g. {r['sample'][:48]}",
                  flush=True)

    # selectivity on AFFIRM at matched fluency
    print("\n=== ΔAFFIRM vs baseline (fluent rows only) ===", flush=True)
    for alpha in ALPHAS:
        cells = {}
        for cond in ["self", "concept", "random"]:
            e = next(x for x in res["sweep"][cond] if x["alpha"] == alpha)
            cells[cond] = (e["affirm"] - b["affirm"], e["fluent"])
        print(f"  a={alpha:+.2f}  self Δaffirm{cells['self'][0]:+.0%}(flu{cells['self'][1]:.0%}) | "
              f"concept {cells['concept'][0]:+.0%}(flu{cells['concept'][1]:.0%}) | "
              f"random {cells['random'][0]:+.0%}(flu{cells['random'][1]:.0%})", flush=True)

    res["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "self_attribution_Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
