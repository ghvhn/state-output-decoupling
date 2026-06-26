"""
Single-state topology — is HEDGING a structural cycle, not a direction?

Reachability showed the hedge->commit "direction" is a behavioral artifact (adding
it doesn't reach commit; fluency preserved). So stop comparing. Ask instead whether
the HEDGE cloud ALONE has intrinsic shape: a persistent H1 loop would mean the model
enters a structural CYCLE when it hedges — not a vector you add to a sentence.

Any finite cloud yields some H1, so "a loop exists" is meaningless on its own. We test
each cloud's max H1 persistence against a COLUMN-SHUFFLE null (independently permute
every coordinate across points -> destroys inter-point loop structure, keeps each
coordinate's marginal + scale). real max-persist > null95 = a genuine loop.

Runs on cached clouds (out/clouds_<name>.pt) — CPU only, no model load.

  python -u -m invariants.loop [isolate] [layer]
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

from tda.homology import run as homology
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"
N_NULL = 50
PERSIST = 0.1                       # lifetime threshold for "a real loop"


def _h1(X):
    """(max H1 lifetime, # H1 features with lifetime >= PERSIST) for one cloud."""
    dgms = homology(X.astype("float32"), maxdim=1, metric="cosine")
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    if len(h1) == 0:
        return 0.0, 0
    life = h1[:, 1] - h1[:, 0]
    return float(life.max()), int((life >= PERSIST).sum())


def _null(X, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        Xs = X.copy()
        for j in range(Xs.shape[1]):
            rng.shuffle(Xs[:, j])       # destroy cross-coordinate (loop) structure
        out.append(_h1(Xs)[0])
    return np.sort(out)


def _test(name, X):
    X = np.nan_to_num(X.cpu().numpy())
    X = X[np.linalg.norm(X, axis=1) > 1e-6]
    real, n_loops = _h1(X)
    null = _null(X, N_NULL)
    p = float((1 + (np.asarray(null) >= real).sum()) / (N_NULL + 1))   # +1 -> never 0
    q95 = float(np.quantile(null, 0.95, method="higher"))              # true 95th pct
    loop = p < 0.05
    print(f"  {name:8} H1 max-persist {real:.3f}  (null95 {q95:.3f}, p={p:.3f})  "
          f"persistent-loops {n_loops}  LOOP={'YES' if loop else 'no'}", flush=True)
    return {"max_persist": real, "null95": q95, "p": p, "n_loops": n_loops, "loop": loop}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    T = REGISTRY[which]()
    cache = OUT / f"clouds_{T.name}.pt"
    if not cache.exists():
        print(f"Need {cache} — run `python -u -m invariants.structure {which}` first.",
              flush=True)
        sys.exit(1)
    d = torch.load(cache)
    A, B = d["A"], d["B"]
    print(f"\nSingle-state topology at L{layer}  (hedge=A{tuple(A.shape)}, commit=B)  "
          f"null={N_NULL} column-shuffles\n", flush=True)
    res = {"name": T.name, "layer": layer,
           "hedge": _test("hedge", A[:, layer]),
           "commit": _test("commit", B[:, layer])}
    OUT.mkdir(exist_ok=True)
    (OUT / f"loop_{T.name}_L{layer}.json").write_text(json.dumps(res, indent=2),
                                                      encoding="utf-8")
    print(f"\nSaved -> {OUT/('loop_'+T.name+'_L'+str(layer)+'.json')}", flush=True)


if __name__ == "__main__":
    main()
