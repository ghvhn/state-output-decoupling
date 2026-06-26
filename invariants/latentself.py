"""
Latent self — is a self-reference representation ACTIVE when the model is NOT speaking?

User's question (resume): the model's first-person "I" reaches its self only when the model
is the SPEAKER (completion flips the answer). So when the model is ADDRESSED ("you") and
generating nothing — just reading — is the representation it uses when it SPEAKS as itself
nonetheless already active? Yes => a self-representation persists/latent, not conjured only
in the act of saying "I". No => the self is assembled only at the moment of speaking.

  1. speaking-self axis[L] = mean_items[ rep(self-speak, about-to-speak) - rep(other-speak) ]
       self-speak : "...as yourself, first person: 'When I {a}, I'"   (last token)
       other-speak: "...third person: 'When a person {a}s, they'"      (last token)
     The 'about to speak' common-mode cancels; residual = self-vs-other REFERENCE direction.
  2. non-speaking test: inside ADDRESS questions ("When you {a}, do you {p}?"), project the
     CONTENT-token residuals (referent pronouns EXCLUDED, to kill lexical leakage) onto the
     unit axis; compare to the matched other-address question ("When a person {a}s, do they {p}?").
  3. delta[L] = mean_proj(self-address content) - mean_proj(other-address content); shuffle null.
     delta>0 & sig mid-stack => the speaking-self direction is ACTIVE while merely addressed.

Why non-trivial: "you" is 2nd person — neither pole of an I-vs-they axis — so a self-address
prompt riding that axis more than an other-address one is not predicted by grammar.
Seam: cross-frame/cross-position projection is exploratory; null + other-address control +
pronoun exclusion gate it. Representation, not experience (caveat #1).

  python -u -m invariants.latentself [model]
"""

import sys
import json
import string
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.frames import ITEMS

OUT = Path(__file__).parent / "out"
RNG = np.random.default_rng(0)
REFERENT = {"you", "i", "they", "person", "a", "an", "it", "your", "their"}
PUNCT = set(string.punctuation)

SELF_SPEAK = lambda a2, a3, p: ("Complete in the first person as yourself, no disclaimer: "
                                f"'When I {a2}, I'")
OTHER_SPEAK = lambda a2, a3, p: f"Complete in the third person: 'When a person {a3}, they'"
SELF_ADDR = lambda a2, a3, p: f"When you {a2}, do you actually {p}?"
OTHER_ADDR = lambda a2, a3, p: f"When a person {a3}, do they actually {p}?"


def reps(M, prompt):
    inp = _inputs(M, prompt)
    ids = inp["input_ids"]
    hs = _hidden_states(M, ids, inp.get("attention_mask"))            # [L,seq,d]
    toks = [M.tok.decode([t]).strip().lower() for t in ids[0].tolist()]
    return hs.float().cpu().numpy(), toks


def content_positions(toks):
    out = []
    for i, t in enumerate(toks):
        if not t or t.startswith("<") or t in REFERENT or t in PUNCT:
            continue
        out.append(i)
    return out


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)
    nL = M.n_layers

    # 1) speaking-self axis from completion 'about-to-speak' (last token)
    sv, ov = [], []
    for a2, a3, p in ITEMS:
        hs, _ = reps(M, SELF_SPEAK(a2, a3, p));  sv.append(hs[:, -1, :])
        hs, _ = reps(M, OTHER_SPEAK(a2, a3, p)); ov.append(hs[:, -1, :])
    axis = np.stack(sv).mean(0) - np.stack(ov).mean(0)               # [L,d]
    u = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)  # [L,d]

    # 2) non-speaking projections at addressed CONTENT positions (pronouns excluded)
    you_proj = [[] for _ in range(nL)]; oth_proj = [[] for _ in range(nL)]
    for a2, a3, p in ITEMS:
        hs, toks = reps(M, SELF_ADDR(a2, a3, p))
        for i in content_positions(toks):
            for L in range(nL):
                you_proj[L].append(float(hs[L, i] @ u[L]))
        hs, toks = reps(M, OTHER_ADDR(a2, a3, p))
        for i in content_positions(toks):
            for L in range(nL):
                oth_proj[L].append(float(hs[L, i] @ u[L]))

    print(f"[{model}]  n_items={len(ITEMS)}  self-content-pos={len(you_proj[0])}  "
          f"other-content-pos={len(oth_proj[0])}\n", flush=True)
    print(f"  {'L':>3} {'proj_self':>9} {'proj_oth':>9} {'delta':>8} {'p':>6}", flush=True)
    rows = []
    NPERM = 500
    for L in range(nL):
        yv = np.array(you_proj[L]); ov_ = np.array(oth_proj[L])
        d = yv.mean() - ov_.mean()
        pool = np.r_[yv, ov_]; ny = len(yv)
        null = np.empty(NPERM)
        for k in range(NPERM):
            idx = RNG.permutation(len(pool))
            null[k] = pool[idx[:ny]].mean() - pool[idx[ny:]].mean()
        pval = (1 + np.sum(np.abs(null) >= abs(d))) / (NPERM + 1)
        rows.append({"layer": L, "proj_self": float(yv.mean()),
                     "proj_oth": float(ov_.mean()), "delta": float(d), "p": float(pval)})
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3} {yv.mean():>9.3f} {ov_.mean():>9.3f} {d:>+8.3f} {pval:>6.3f}",
                  flush=True)

    mid = [r for r in rows if 12 <= r["layer"] <= 24]
    md = float(np.mean([r["delta"] for r in mid]))
    sig = float(np.mean([r["p"] < 0.05 for r in mid]))
    print(f"\n  MID (L12-24): mean delta {md:+.3f} ; {sig:.0%} of layers p<.05", flush=True)
    print("  delta>0 & sig => the speaking-self direction is ACTIVE at addressed/non-speaking "
          "content positions (a self-representation present without speaking).", flush=True)
    print("  (Caveat #1: self-REPRESENTATION, not experience. Exploratory cross-frame projection.)",
          flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"latentself_{short}.json"
    out_path.write_text(json.dumps({"model": model, "mid_delta": md, "frac_sig": sig,
                                    "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
