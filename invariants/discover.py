"""
Topological discovery loop — the orchestrator, re-centered.

The old spine (invariants/run.py) was SUBTRACTIVE: find a direction, ABLATE it,
read the break. Two interventions have now nulled with competence intact —
ablation (hedge 67%->75% under the judge) AND additive reachability (reached
peaks at baseline and falls as you add the displacement, fluency preserved). A
single direction is not the behavioral cause in EITHER direction. So stop
orchestrating around a direction.

This loop orchestrates around STRUCTURE: per-token residual clouds -> every lens
including Topology, each against its own null -> the structural signature (which
families of structure separate the arms, at which depth). Intervention is demoted
to an optional appendix (run.py / reachability.py), never the verdict.

  python -u -m invariants.discover                # whole REGISTRY
  python -u -m invariants.discover isolate self   # named transformations

Clouds cache to out/clouds_<name>.pt, so re-discovery is instant.
"""

import sys
import json
from pathlib import Path

import torch

from invariants.engine import load_model, extract_tokens
from invariants.lenses import LENSES, Topology
from invariants.library import REGISTRY
from invariants.structure import _read, _sub, N_POINTS

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"


def _clouds(M, T):
    """Per-token clouds for both arms, cached — the substrate the lenses map."""
    cache = OUT / f"clouds_{T.name}.pt"
    if cache.exists():
        d = torch.load(cache)
        return d["A"], d["B"]
    A = _sub(extract_tokens(M, T.a, label=T.a_label), N_POINTS)
    B = _sub(extract_tokens(M, T.b, label=T.b_label), N_POINTS)
    OUT.mkdir(exist_ok=True)
    torch.save({"A": A, "B": B}, cache)
    return A, B


def discover_structure(M, T):
    """Map the two arms with every lens vs its own null. Returns the per-lens read
    plus the structural signature = the families that clear."""
    A, B = _clouds(M, T)
    nl = A.shape[1]
    layers = sorted(set(list(range(0, nl, 4)) + [nl // 2, nl - 1]))
    lenses = {lens.name: _read(lens, A, B, layers, 15 if isinstance(lens, Topology) else 40)
              for lens in LENSES}
    cleared = [n for n, r in lenses.items() if r.get("clears")]
    return {"shape": tuple(A.shape), "layers": layers, "lenses": lenses, "signature": cleared}


def main():
    names = sys.argv[1:] or list(REGISTRY)
    M = load_model(MODEL)
    OUT.mkdir(exist_ok=True)
    report = {}
    for which in names:
        if which not in REGISTRY:
            print(f"  skip unknown '{which}' (have {list(REGISTRY)})", flush=True)
            continue
        T = REGISTRY[which]()
        print(f"\n=== {T.name}  (expected={T.expected}) — topological discovery ===", flush=True)
        d = discover_structure(M, T)
        for name, r in d["lenses"].items():
            if r.get("available"):
                print(f"  {name:12} L{r['best_layer']:<3} score {r['score']:.3f}  "
                      f"floor {r['floor']:.3f}  clears={r['clears']}", flush=True)
            else:
                print(f"  {name:12} n/a ({r.get('reason', '')})", flush=True)
        print(f"  -> structural signature: {d['signature'] or 'none clears'}", flush=True)
        report[T.name] = {"expected": T.expected, "signature": d["signature"],
                          "lenses": d["lenses"]}
    (OUT / "discovery.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'discovery.json'}", flush=True)


if __name__ == "__main__":
    main()
