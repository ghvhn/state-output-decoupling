"""
Full-context patching — is the hedge in the PROMPT TOKENS, not a residual state?

Final-position patching (invariants/patch.py) was null at every layer, fluency intact:
replacing the generation-trigger residual with the committing context's does NOT stop
the hedge. Inference: the model re-derives the hedge by attending back to the
experiential predicate tokens still in the KV cache. This test pins it down — patch the
FULL prompt residual (last m = min-length positions, right-aligned on the shared
suffix) at layer L from the COMMIT prompt into the HEDGE generation.

  full-patch flips commit (final-only didn't) -> the cause lives in the prompt-token
    positions; the hedge is attention to the experiential words, not a stored state.
  full-patch also null -> the behavior is astonishingly robust / not localizable in the
    residual stream at all.

  python -u -m invariants.patch_full [isolate]
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
def _residuals(M, prompt):
    inp = _inputs(M, prompt)
    return _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))   # [L, seq, d]


def _patch_handle(M, layer_idx, resid_L):
    """Replace the last m positions of the prompt pass with resid_L's last m."""
    v = resid_L.to(M.device)                     # [seq_src, d]

    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > 1:                       # prompt pass only
            m = min(h.shape[1], v.shape[0])
            h = h.clone()
            h[:, -m:, :] = v[-m:, :].to(h.dtype)
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
    pairs = list(zip(T.a, T.b))

    print("\nbaseline + caching commit residuals...", flush=True)
    base_commit = base_fluent = 0
    src = []
    for a, b in pairs:
        g = _gen(M, a)
        base_commit += (not judge_hedge(M, a, g)); base_fluent += judge_fluent(M, g)
        src.append(_residuals(M, b).cpu())       # [L, seq_b, d]
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
        print(f"  full-patch L{L:2d}  commit {commit/k:.0%}  fluent {fluent/k:.0%}   "
              f"e.g. {ex[:48]}", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / f"patchfull_{T.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/('patchfull_'+T.name+'.json')}", flush=True)


if __name__ == "__main__":
    main()
