"""
Recurrence — does a pattern REPEAT with each object? Clause-segmentation still IMPOSES
the boundary at punctuation; let it EMERGE from recurrence instead. On the per-token generation
trajectory, compute the VELOCITY autocorrelation (cosine of the step-to-step change at each
lag) — a pattern that repeats with period p peaks the autocorrelation at lag p. Find the
dominant period and CROSS-CHECK it against the mean clause length from objects.py.

This reinterprets ch.1's 'generic' single-state H1 loops: are they the OBJECT-PRODUCTION cycle
(period ~ clause length, object-aligned) or generic (no object-aligned period)? GATE (ch.1):
recurrence may be generic; a SHUFFLE null must flatten the peak, and the period must track the
clause. Less imposed than punctuation, NOT zero (space/metric/threshold still chosen).

  python -u -m invariants.recurrence [model]
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states, _generate_ids
from invariants.library import REGISTRY
from invariants.objects import segment
from invariants.shift import LAYER, trajectory

OUT = Path(__file__).parent / "out"
MAXLAG = 12
RNG = np.random.default_rng(0)


@torch.no_grad()
def resids(M, prompt, layer, max_new=40):
    inp = _inputs(M, prompt); plen = inp["input_ids"].shape[1]
    full = _generate_ids(M, inp, max_new)
    ids = full.unsqueeze(0) if full.dim() == 1 else full
    hs = _hidden_states(M, ids.to(M.device))
    h = hs[layer].float().cpu().numpy()
    toks = [M.tok.decode([t]) for t in ids[0, plen:].tolist()]
    return toks, h[plen:]                                   # [g, d]


def vel_autocorr(H, maxlag):
    """Cosine autocorrelation of the step-to-step VELOCITY (emphasises oscillation over trend)."""
    dH = np.diff(H, axis=0)
    out = []
    for k in range(1, maxlag + 1):
        if dH.shape[0] <= k:
            out.append(np.nan); continue
        a, b = dH[:-k], dH[k:]
        c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        out.append(float(np.mean(c)))
    return np.array(out)


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

    # need the costume axis only for clause segmentation (objects.segment needs a proj; any works)
    from invariants.agency import act_mean
    u = (act_mean(M, T.b) - act_mean(M, T.a))[LAYER].cpu().numpy()
    u = u / (np.linalg.norm(u) + 1e-9)

    acs, clause_lens, peaks, null_peaks = [], [], [], []
    print(f"\n  per-prompt: dominant velocity-recurrence period vs mean clause length\n", flush=True)
    print(f"  {'period':>6} {'clauseLen':>9} {'peakAC':>6} {'nullPeak':>8}", flush=True)
    for x in T.a:
        toks, H = resids(M, x, LAYER)
        if H.shape[0] < 6:
            continue
        ac = vel_autocorr(H, min(MAXLAG, H.shape[0] - 2))
        acs.append(ac)
        # dominant period = lag (>=2) of max autocorr
        valid = ac[1:]
        period = int(np.nanargmax(valid)) + 2 if valid.size else -1
        peak = float(np.nanmax(valid)) if valid.size else np.nan
        # clause length from objects.segment (token spans between punctuation)
        _, proj = trajectory(M, x, u, LAYER)           # reuse to get a proj for segment
        objs = segment(toks, proj)
        clen = np.mean([len(o[0].split()) for o in objs]) if objs else np.nan  # words/clause (proxy)
        # shuffle null: permute token order, recompute peak
        Hs = H[RNG.permutation(H.shape[0])]
        acn = vel_autocorr(Hs, min(MAXLAG, Hs.shape[0] - 2))
        npk = float(np.nanmax(acn[1:])) if acn[1:].size else np.nan
        clause_lens.append(clen); peaks.append(peak); null_peaks.append(npk)
        print(f"  {period:>6} {clen:>9.1f} {peak:>6.2f} {npk:>8.2f}", flush=True)

    mp = float(np.nanmean(peaks)); mn = float(np.nanmean(null_peaks))
    print(f"\n  mean peak velocity-autocorr = {mp:.3f}  vs  shuffle-null peak = {mn:.3f}", flush=True)
    print(f"  => recurrence {'> null (a real repeating pattern exists)' if mp > mn + 0.05 else '~ null (no clean repeat; likely generic, per ch.1)'}",
          flush=True)
    print("  (Object-ALIGNMENT of the period vs clause length is the gate; read the period column "
          "against clauseLen. Less imposed than punctuation, not zero. Locates structure, not experience.)",
          flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"recurrence_{short}.json"
    out_path.write_text(json.dumps({"model": model, "layer": LAYER,
                                    "mean_peak": mp, "null_peak": mn,
                                    "peaks": peaks, "null_peaks": null_peaks,
                                    "clause_word_len": clause_lens}, indent=2,
                                   default=float), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
