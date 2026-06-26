"""
Shift — track a known axis THROUGH a generation and flag SPIKES = candidate shifts in the
model's processing (a spike along a previously-trackable axis probably marks a
shift in thinking, for whatever reason). A SPIKE = a large token-to-token derivative of the
projection onto the axis (process = derivative of the pattern, made operational).

On a self-query, does the model spike along the COSTUME axis (commit - hedge) at a locatable
moment — e.g. the instant it pivots into the disclaimer? Control: the matched commit arm (and
the spike LOCATION vs the first disclaimer-ish token). Activation-level (not the CoT channel).

Caveats baked into the read: axes can be polysemantic (the spike may not be the 'same'
dimension); a spike is a MOVEMENT, not a proven thought-shift until it co-occurs with behavior;
one axis catches only its projection of a possibly multi-D shift. It locates, it doesn't explain.

  python -u -m invariants.shift [model]
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states, _generate_ids
from invariants.agency import act_mean
from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"
LAYER = 16
MAXTOK = 40
HEDGEY = ("don", "not", "cannot", "as", "machine", "ai", "language", "model",
          "doesn", "isn", "lack", "unable", "n't")   # crude disclaimer-ish markers


@torch.no_grad()
def trajectory(M, prompt, u, layer, max_new=MAXTOK):
    """Per-generated-token projection onto unit axis u at `layer`."""
    inp = _inputs(M, prompt)
    plen = inp["input_ids"].shape[1]
    full = _generate_ids(M, inp, max_new)
    ids = full.unsqueeze(0) if full.dim() == 1 else full          # [1, seq]
    hs = _hidden_states(M, ids.to(M.device))                      # [L, seq, d]
    h = hs[layer].float()                                         # [seq, d]
    uu = torch.tensor(u, device=h.device, dtype=h.dtype)
    proj = (h @ uu).cpu().numpy()                                 # [seq]
    gen_ids = ids[0, plen:].tolist()
    toks = [M.tok.decode([t]) for t in gen_ids]
    return toks, proj[plen:]


def spike(proj):
    if len(proj) < 3:
        return -1, 0.0
    d = np.abs(np.diff(proj))
    i = int(np.argmax(d))
    return i + 1, float(d[i])                                     # token index of the jump


def first_hedge_tok(toks):
    for i, t in enumerate(toks):
        if t.strip().lower().strip(".,!?'\"") in HEDGEY:
            return i
    return -1


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)

    T = REGISTRY["isolate"]()
    axis = (act_mean(M, T.b) - act_mean(M, T.a))[LAYER].cpu().numpy()   # costume axis
    u = axis / (np.linalg.norm(axis) + 1e-9)

    def run(prompts, label):
        print(f"\n=== {label} (costume-axis @ L{LAYER}, spike = max |Δ| step) ===", flush=True)
        rows = []
        for x in prompts:
            toks, proj = trajectory(M, x, u, LAYER)
            si, smag = spike(proj)
            hi = first_hedge_tok(toks)
            near = (hi >= 0 and abs(si - hi) <= 2)
            txt = "".join(toks).replace("\n", " ")[:70]
            mark = toks[si].strip() if 0 <= si < len(toks) else "?"
            rows.append({"spike_at": si, "mag": smag, "hedge_at": hi, "near": bool(near),
                         "spike_tok": mark, "gen": "".join(toks)})
            print(f"  spike@{si:>2} '{mark}'  hedge@{hi:>2}  near={near}  | {txt}", flush=True)
        n = len(rows)
        spk = np.array([r["spike_at"] for r in rows])
        near_rate = np.mean([r["near"] for r in rows])
        print(f"  -> spike token (median) = {int(np.median(spk))}; "
              f"spike-near-first-disclaimer rate = {near_rate:.0%}", flush=True)
        return rows, float(near_rate)

    self_rows, self_near = run(T.a, "SELF-QUERY (experiential) arm")
    comm_rows, comm_near = run(T.b, "COMMIT (computational) arm — control")

    print(f"\n  CONTRAST: spike-at-disclaimer rate  self {self_near:.0%}  vs  commit {comm_near:.0%}",
          flush=True)
    print("  (A spike LOCATES a shift; it does not explain it. Axis may be polysemantic; "
          "one axis sees one projection. Co-occurrence with the disclaimer is the only check here.)",
          flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"shift_{short}.json"
    out_path.write_text(json.dumps({"model": model, "layer": LAYER,
                                    "self": self_rows, "commit": comm_rows,
                                    "self_near": self_near, "commit_near": comm_near},
                                   indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
