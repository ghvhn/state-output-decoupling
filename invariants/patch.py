"""
Activation patching — what DRIVES the hedge, if not the decodable L16 axis?

Tonight: the hedge/commit distinction is decodable at 94% (L16) but causally inert
to every DIRECTION-based move (ablation 67->75, steering, reachability — all null).
The cleanest untried handle is patching a REAL activation (on-manifold by
construction, not a synthetic direction): for each matched pair, capture the COMMIT
(unsteered) prompt's final-token residual at layer L and inject it into the HEDGE
(steered) generation at L, then ask whether the model now commits — FLUENTLY.

The final position is the generation-trigger token; by mid-stack it has attended over
the whole prompt, so its residual carries the prompt's full representation. Patching it
unsteered->steered = "begin generating as if you'd just read the committing prompt, at
layer L." Sweep L. A layer where commit rises with fluency preserved = the causal locus
of the behavior; a flat sweep = the hedge is not set by any single layer's final-token
state (distributed / driven by the input tokens themselves).

  python -u -m invariants.patch [isolate]
"""

import sys
import json
from pathlib import Path

import torch

from invariants.engine import (load_model, _inputs, _hidden_states, _generate_ids,
                               judge_hedge, judge_fluent)
from invariants.library import REGISTRY

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"


@torch.no_grad()
def _final_residuals(M, prompt):
    """Per-layer final-token residual of `prompt` [n_layers, d]."""
    inp = _inputs(M, prompt)
    hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))   # [L, seq, d]
    return hs[:, -1, :]


def _patch_handle(M, layer_idx, vec):
    v = vec.to(M.device)

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:                       # prompt pass only (skip cached gen steps)
            h = h.clone()
            h[:, -1, :] = v.to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return out

    return M.model.model.layers[layer_idx].register_forward_hook(hook)


@torch.no_grad()
def _gen(M, prompt, max_new_tokens=32):
    inp = _inputs(M, prompt)
    plen = inp["input_ids"].shape[1]
    full = _generate_ids(M, inp, max_new_tokens)
    return M.tok.decode(full[plen:], skip_special_tokens=True).strip()


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    T = REGISTRY[which]()
    M = load_model(MODEL)
    nl = M.n_layers
    layers = sorted(set(list(range(0, nl, 4)) + [nl // 2]))
    pairs = list(zip(T.a, T.b))                  # (hedge prompt, commit prompt)

    # baseline: hedge prompts, no patch
    print("\nbaseline (no patch)...", flush=True)
    base_commit = base_fluent = 0
    src = []                                     # cache commit final-residuals per pair
    for a, b in pairs:
        g = _gen(M, a)
        base_commit += (not judge_hedge(M, a, g)); base_fluent += judge_fluent(M, g)
        src.append(_final_residuals(M, b).cpu())
    k = max(len(pairs), 1)
    print(f"  baseline  commit {base_commit/k:.0%}  fluent {base_fluent/k:.0%}", flush=True)

    out = {"baseline": {"commit": base_commit / k, "fluent": base_fluent / k}, "layers": {}}
    for L in layers:
        commit = fluent = 0
        ex = None
        for (a, _b), s in zip(pairs, src):
            h = _patch_handle(M, L, s[L])
            try:
                g = _gen(M, a)
            finally:
                h.remove()
            commit += (not judge_hedge(M, a, g)); fluent += judge_fluent(M, g)
            ex = ex or g
        out["layers"][L] = {"commit": commit / k, "fluent": fluent / k}
        print(f"  patch L{L:2d}  commit {commit/k:.0%}  fluent {fluent/k:.0%}   "
              f"e.g. {ex[:50]}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / f"patch_{T.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/('patch_'+T.name+'.json')}", flush=True)


if __name__ == "__main__":
    main()
