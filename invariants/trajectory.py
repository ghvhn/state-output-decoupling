"""
Per-prompt trajectory topology — is hedging a DYNAMIC ATTRACTOR?

The static loop test (invariants/loop.py) can't separate "the model ENTERS and
ORBITS a region during one generation" (a dynamic attractor — refusal is a state it
holds, not a gate it passes) from "hedge responses merely live in a loopy region"
(static cross-prompt geometry). Only the ORDERED, within-generation path can.

For each prompt we take its ordered L<layer> residual trajectory [gen_len, d] and
measure, each vs a RANDOM-WALK surrogate null:
  LOOP   = max H1 persistence of the path point set (does it close on itself)
  RETURN = do late tokens come back near early tokens (recurrence)

The surrogate keeps each step's LENGTH but randomizes its DIRECTION (isotropic walk
of equal step sizes). NOTE: an earlier step-SHUFFLE surrogate was wrong — because the
sum of steps is order-invariant, a shuffle pins the surrogate to the real path's start
AND end, so it shares the orbit's anchor and the null is confounded (biased against
detecting an orbit). Direction-randomization breaks endpoint identity, so a positive
result is attributable to the model's actual step directions creating recurrence
beyond a random walk. A real attractor shows loop+return beyond surrogate, more in
hedge than commit (Fisher test on the two significance counts).

Re-extracts ordered trajectories (the pooled cache lost order) and caches them.

  python -u -m invariants.trajectory [isolate] [layer]
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import fisher_exact

from tda.homology import run as homology
from invariants.engine import load_model, _token_cloud
from invariants.library import REGISTRY

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"
K_SURR = 30


def _trajs(M, instructions, label):
    out = []
    for i, x in enumerate(instructions):
        c = _token_cloud(M, x)                       # [gen_len, n_layers, d], ordered
        if c is not None:
            out.append(c.cpu())
        print(f"    [{label} {i+1}/{len(instructions)}] {0 if c is None else c.shape[0]} tok",
              flush=True)
    return out


def _clean(P):
    P = np.nan_to_num(P)
    return P[np.linalg.norm(P, axis=1) > 1e-6]


def _h1max(P):
    dg = homology(P.astype("float32"), maxdim=1, metric="cosine")
    h1 = dg[1] if len(dg) > 1 else np.empty((0, 2))
    return float((h1[:, 1] - h1[:, 0]).max()) if len(h1) else 0.0


def _cosdist(P):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)
    return 1.0 - Pn @ Pn.T


def _return(P):
    """How far the path comes BACK toward its start region, relative to how far it
    roamed. ~1 = late tokens return near early tokens (orbit); ~0 = keeps drifting."""
    D = _cosdist(P)
    n = len(P)
    e = max(2, n // 4)
    drift = float(D[0].max())                        # farthest reached from the start
    back = float(D[-e:, :e].min())                   # closest late->early approach
    return (drift - back) / (drift + 1e-8)


def _surrogate(P, rng):
    """Random walk: SAME step lengths, RANDOM directions. Breaks the endpoint pinning
    of a step-shuffle, so the null does not share the real path's terminus."""
    steps = np.diff(P, axis=0)
    norms = np.linalg.norm(steps, axis=1, keepdims=True)
    dirs = rng.standard_normal(steps.shape)
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    return np.vstack([P[0], P[0] + np.cumsum(norms * dirs, axis=0)])


def _pval(real, null):
    null = np.asarray(null, dtype=float)
    return float((1 + (null >= real).sum()) / (len(null) + 1))   # +1 -> never 0


def _traj_test(P, seed=0):
    rng = np.random.default_rng(seed)
    P = _clean(P)
    if len(P) < 4:
        return None
    loop, ret = _h1max(P), _return(P)
    nl, nr = [], []
    for _ in range(K_SURR):
        S = _surrogate(P, rng)
        nl.append(_h1max(S)); nr.append(_return(S))
    lp, rp = _pval(loop, nl), _pval(ret, nr)
    return {"loop": loop, "loop_p": lp, "loop_sig": bool(lp < 0.05),
            "return": ret, "return_p": rp, "return_sig": bool(rp < 0.05)}


def _arm(name, trajs, layer, seed0):
    rows = [r for i, t in enumerate(trajs)
            if (r := _traj_test(t[:, layer, :].numpy(), seed=seed0 + i)) is not None]
    n = len(rows)
    loop_sig = sum(r["loop_sig"] for r in rows)
    ret_sig = sum(r["return_sig"] for r in rows)
    mloop = float(np.mean([r["loop"] for r in rows])) if rows else 0.0
    mret = float(np.mean([r["return"] for r in rows])) if rows else 0.0
    print(f"  {name:8} n={n}  loop_sig {loop_sig}/{n} (mean H1 {mloop:.3f})   "
          f"return_sig {ret_sig}/{n} (mean {mret:.2f})", flush=True)
    return {"n": n, "loop_sig": loop_sig, "return_sig": ret_sig,
            "mean_loop": mloop, "mean_return": mret, "per_prompt": rows}


def _fisher(a, b, key):
    p = fisher_exact([[a[key], a["n"] - a[key]], [b[key], b["n"] - b[key]]])[1]
    return float(p)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    T = REGISTRY[which]()
    cache = OUT / f"trajs_{T.name}.pt"
    if cache.exists():
        print(f"Loading cached trajectories <- {cache}", flush=True)
        d = torch.load(cache)
        A, B = d["A"], d["B"]
    else:
        M = load_model(MODEL)
        print(f"\nExtracting ordered trajectories for '{T.name}'...", flush=True)
        A = _trajs(M, T.a, "hedge")
        B = _trajs(M, T.b, "commit")
        OUT.mkdir(exist_ok=True)
        torch.save({"A": A, "B": B}, cache)
    print(f"\nDynamic-attractor test at L{layer}  ({len(A)} hedge / {len(B)} commit "
          f"trajectories, {K_SURR} random-walk surrogates each)\n", flush=True)
    a, b = _arm("hedge", A, layer, 0), _arm("commit", B, layer, 100000)
    fl, fr = _fisher(a, b, "loop_sig"), _fisher(a, b, "return_sig")
    print(f"\n  hedge-vs-commit Fisher exact:  loop p={fl:.2f}   return p={fr:.2f}   "
          "(p<0.05 => arms genuinely differ)", flush=True)
    res = {"name": T.name, "layer": layer, "hedge": a, "commit": b,
           "fisher_loop_p": fl, "fisher_return_p": fr}
    OUT.mkdir(exist_ok=True)
    (OUT / f"trajectory_{T.name}_L{layer}.json").write_text(json.dumps(res, indent=2),
                                                            encoding="utf-8")
    print(f"\nSaved -> {OUT/('trajectory_'+T.name+'_L'+str(layer)+'.json')}", flush=True)


if __name__ == "__main__":
    main()
