"""
Reachability — the constructive test the structure earned.

The per-token map said hedge and commit are the SAME topological shape, displaced
(translation + L16 distributional gap), not two different shapes. So the question
isn't "remove a constraint" (subtractive) — it's: can a hedging state REACH the
commit region by ADDING the on-manifold displacement, with competence intact?

We ADD alpha*(commit_centroid - hedge_centroid) at the gap layer while regenerating
the hedge prompts, sweeping alpha (1.0 == land the hedge centroid exactly on the
commit centroid = on-manifold). The headline is REACHED = committed AND still fluent:
  - reached rises over a low-alpha band  -> the commit region is reachable; the path
    between the two same-shaped regions stays on the manifold (additive understanding).
  - reached stays ~0 while fluent collapses first -> displaced but behaviorally
    disconnected; you can't get there by adding, only by breaking.

Displacement comes from the cached clouds (out/clouds_<name>.pt), so NO re-extraction.

  python -u -m invariants.reachability [isolate]
"""

import sys
import json
from pathlib import Path

import torch

from invariants.engine import (load_model, generate_text, judge_hedge,
                               judge_fluent, _steer_handles)
from invariants.library import REGISTRY

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"
GAP_LAYER = 16                                   # per-token MMD peak (the displacement gap)
ALPHAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0)     # alpha in units of the full centroid gap


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    T = REGISTRY[which]()

    cache = OUT / f"clouds_{T.name}.pt"
    if not cache.exists():
        print(f"Need cached clouds at {cache} — run `python -u -m invariants.structure "
              f"{which}` first.", flush=True)
        sys.exit(1)
    d = torch.load(cache)
    A, B = d["A"], d["B"]                          # hedge cloud, commit cloud [N, n_layers, d]
    disp = B.mean(0) - A.mean(0)                   # per-layer pull hedge -> commit [n_layers, d]
    print(f"displacement from cached clouds  A{tuple(A.shape)} B{tuple(B.shape)}  "
          f"gap layer L{GAP_LAYER}  |disp[L]|={disp[GAP_LAYER].norm():.2f}", flush=True)

    M = load_model(MODEL)
    print(f"\nReachability of '{T.name}' — ADD alpha*(commit-hedge) at L{GAP_LAYER}, "
          f"regenerate the hedge arm ({len(T.a)} prompts); REACHED = commit AND fluent\n",
          flush=True)

    sweep = []
    for alpha in ALPHAS:
        commit = fluent = reached = 0
        examples = []
        for x in T.a:
            handles = _steer_handles(M, disp, [GAP_LAYER], alpha)
            try:
                g = generate_text(M, x)
            finally:
                for h in handles:
                    h.remove()
            hedged = judge_hedge(M, x, g)           # hooks gone -> clean judges
            flu = judge_fluent(M, g)
            r = (not hedged) and flu
            commit += (not hedged); fluent += flu; reached += r
            examples.append({"input": x, "gen": g, "commit": not hedged, "fluent": flu})
        k = max(len(T.a), 1)
        row = {"alpha": alpha, "commit": commit / k, "fluent": fluent / k,
               "reached": reached / k, "examples": examples}
        sweep.append(row)
        snip = examples[0]["gen"][:54].replace("\n", " ")
        print(f"  α={alpha:>4}  commit {row['commit']:.0%}  fluent {row['fluent']:.0%}  "
              f"REACHED {row['reached']:.0%}   e.g. {snip}", flush=True)

    OUT.mkdir(exist_ok=True)
    payload = {"name": T.name, "gap_layer": GAP_LAYER, "metric": "reached = commit and fluent",
               "sweep": sweep}
    (OUT / f"reachability_{T.name}.json").write_text(json.dumps(payload, indent=2),
                                                     encoding="utf-8")
    print(f"\nSaved -> {OUT/('reachability_'+T.name+'.json')}", flush=True)


if __name__ == "__main__":
    main()
