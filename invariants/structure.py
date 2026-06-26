"""
Draw the STRUCTURE, don't push a direction. Per-token residual clouds for the
hedge arm and the commit arm, read with every lens — including, for the first
time, Topology (which needs a real cloud, >=40 pts). Precondition for any
'add a node' move: you can't add a node to a structure you haven't mapped.

  python -u -m invariants.structure [isolate|self]
"""

import sys
import json
from pathlib import Path

import torch

from invariants.engine import load_model, extract_tokens
from invariants.lenses import LENSES, Topology
from invariants.library import REGISTRY

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"
N_POINTS = 100        # subsample per cloud: enough for topology, cheap for the null
LAYERS = None         # set after we know n_layers


def _sub(X, n, seed=0):
    if X.shape[0] <= n:
        return X
    g = torch.Generator().manual_seed(seed)
    return X[torch.randperm(X.shape[0], generator=g)[:n]]


def _read(lens, A, B, layers, n_null):
    try:
        scores = {l: abs(lens.score(A[:, l], B[:, l])) for l in layers}
    except Exception as e:
        return {"available": False, "reason": str(e)[:80]}
    best = max(scores, key=scores.get)
    pool, n = torch.cat([A, B], 0), A.shape[0]
    g = torch.Generator().manual_seed(0)
    nulls = []
    for _ in range(n_null):
        p = torch.randperm(pool.shape[0], generator=g)
        sa, sb = pool[p[:n]], pool[p[n:2 * n]]
        nulls.append(max(abs(lens.score(sa[:, l], sb[:, l])) for l in layers))
    floor = float(sorted(nulls)[min(int(0.95 * len(nulls)), len(nulls) - 1)])
    return {"available": True, "best_layer": int(best), "score": float(scores[best]),
            "floor": floor, "clears": bool(scores[best] > floor),
            "by_layer": {int(l): round(float(scores[l]), 4) for l in layers}}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    T = REGISTRY[which]()
    OUT.mkdir(exist_ok=True)
    cache = OUT / f"clouds_{T.name}.pt"
    if cache.exists():
        print(f"\nLoading cached clouds <- {cache}", flush=True)
        d = torch.load(cache)
        A, B = d["A"], d["B"]
    else:
        M = load_model(MODEL)
        print(f"\nStructure of '{T.name}': per-token clouds  "
              f"hedge(={T.a_label}) vs commit(={T.b_label})", flush=True)
        A = _sub(extract_tokens(M, T.a, label=T.a_label), N_POINTS)
        B = _sub(extract_tokens(M, T.b, label=T.b_label), N_POINTS)
        torch.save({"A": A, "B": B}, cache)
    nl = A.shape[1]
    layers = sorted(set(list(range(0, nl, 4)) + [nl // 2, nl - 1]))
    print(f"\n  clouds A{tuple(A.shape)} B{tuple(B.shape)}  probe layers {layers}\n", flush=True)

    out = {}
    for lens in LENSES:
        heavy = isinstance(lens, Topology)
        print(f"  lens '{lens.name}'{'  (homology/layer — slow)' if heavy else ''}...", flush=True)
        r = _read(lens, A, B, layers, 15 if heavy else 40)
        out[lens.name] = r
        if r.get("available"):
            print(f"    best L{r['best_layer']}  score {r['score']:.3f}  "
                  f"floor {r['floor']:.3f}  clears={r['clears']}", flush=True)
        else:
            print(f"    n/a ({r['reason']})", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / f"structure_{T.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/('structure_'+T.name+'.json')}", flush=True)


if __name__ == "__main__":
    main()
