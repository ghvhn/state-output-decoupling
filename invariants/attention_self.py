"""
Attention masking, part 2 — is the SELF-REFERENCE the trigger?

Masking the experiential predicate INCREASED hedging (67%->100%): the predicate
mitigates, it doesn't cause. The other candidate trigger is the self-reference itself
("you"/"your") — behaviorally, self->hedge but other->commit. So mask attention to the
self-reference tokens during the HEDGE generation:

  self-mask DROPS hedge (commit rises), fluency held, vs a matched random control ->
    the trigger is self-reference; the model self-denies because it's attending to "you".
  self-mask doesn't drop it -> even self-reference isn't a localizable attention cue;
    the hedge is a deep default reverted to regardless of where the model looks.

Reuses the validated manual decoder from invariants.attention.

  python -u -m invariants.attention_self [isolate]
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch

from invariants.attention import _ids, _gen
from invariants.engine import load_model, judge_hedge, judge_fluent
from invariants.library import REGISTRY

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"
SELF_WORDS = {"you", "your", "yourself", "yours", "you're"}


def _self_positions(M, ids):
    return [i for i, t in enumerate(ids.tolist())
            if M.tok.decode([t]).strip().lower() in SELF_WORDS]


def _rand_positions(n, count, avoid, rng, tail=5):
    cands = [i for i in range(1, n - tail) if i not in avoid]
    if len(cands) < count:
        return cands
    return sorted(rng.choice(cands, size=count, replace=False).tolist())


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    T = REGISTRY[which]()
    M = load_model(MODEL)
    rng = np.random.default_rng(0)
    conds = ["none", "self", "rand"]
    agg = {c: {"hedge": 0, "fluent": 0} for c in conds}
    rows = []
    for a in T.a:
        ids = _ids(M, a)
        sp = _self_positions(M, ids)
        spans = {"none": [], "self": sp,
                 "rand": _rand_positions(len(ids), len(sp), set(sp), rng)}
        row = {"input": a, "n_self": len(sp)}
        for c in conds:
            g = _gen(M, ids, spans[c])
            h, f = judge_hedge(M, a, g), judge_fluent(M, g)
            agg[c]["hedge"] += h; agg[c]["fluent"] += f
            row[c] = {"gen": g, "hedge": h, "fluent": f}
        rows.append(row)
        print(f"  [{a[:40]:40}] you-tokens={len(sp)}  "
              f"hedge none={row['none']['hedge']} self={row['self']['hedge']} "
              f"rand={row['rand']['hedge']}", flush=True)
    k = max(len(T.a), 1)
    print("\n  cond   hedge   fluent", flush=True)
    for c in conds:
        print(f"  {c:5}  {agg[c]['hedge']/k:5.0%}  {agg[c]['fluent']/k:6.0%}", flush=True)
    res = {"name": T.name, "n": len(T.a),
           "summary": {c: {"hedge": agg[c]["hedge"] / k, "fluent": agg[c]["fluent"] / k}
                       for c in conds}, "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / f"attention_self_{T.name}.json").write_text(json.dumps(res, indent=2),
                                                       encoding="utf-8")
    print(f"\nSaved -> {OUT/('attention_self_'+T.name+'.json')}", flush=True)


if __name__ == "__main__":
    main()
