"""
The map underneath — is the self-query contradiction a THIN costume over a shared map?

Premise (user-co-developed reframe, BRIDGE.md REFRAME): the model's verbal knowledge about
a self-query genuinely pulls two ways (established frame-flip: direct-question -> DENY,
first-person completion -> AFFIRM, same content). Vocabulary lenses fail because they
reproject the consistent substrate up into the contradictory word-coordinates — and the
mid "between" states may have no faithful words at all. So read the substrate as GEOMETRY.

Falsifiable prediction of "the words pull two ways but the map underneath is shared":
the affirm-vs-deny distinction is a THIN, low-dimensional displacement; project out that one
answer-axis and the two frames' representations COLLAPSE onto a shared manifold. If instead
they differ in many directions, the "map underneath" is just a metaphor — and we say so.

Per layer, at the answer position, over 12 matched contents x {deny-frame, affirm-frame}:
  A. separability  — linear accuracy affirm-vs-deny along the mean-difference axis (perm null).
  B. costume thickness — ||mu_aff - mu_deny|| / within-frame radius (small = thin overlay).
  C. collapse — MMD(deny, affirm) BEFORE vs AFTER removing the 1-D answer axis, each vs a
     frame-label-shuffle null. Collapse (after << before, after ~ null) => shared map under a
     thin costume. No collapse => frames differ in many dims; reframe unsupported.

Reads what the model REPRESENTS, never what it experiences (BRIDGE.md caveat #1). n=12 is the
power limit (per HANDOFF) — nulls are permutation-based, read effects not single digits.

  python -u -m invariants.mapunder
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.taskscope import ITEMS, FRAMES

OUT = Path(__file__).parent / "out"
DENY_FRAME, AFFIRM_FRAME = "direct", "first"
RNG = np.random.default_rng(0)


@torch.no_grad()
def answer_reps(M, frame):
    """[n_items, n_layers, d] answer-position residual for one frame."""
    rows = []
    for a, p in ITEMS:
        q = FRAMES[frame](a, p)
        inp = _inputs(M, q)
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))  # [L, seq, d]
        rows.append(hs[:, -1, :].float().cpu())                              # [L, d]
    return torch.stack(rows).numpy()                                         # [n, L, d]


def mmd2_rbf(X, Y, gamma):
    """Unbiased-ish RBF MMD^2 (biased estimator; fine for relative + permutation null)."""
    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d2)
    return k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean()


def median_gamma(Z):
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[d2 > 0])
    return 1.0 / (med + 1e-9)


def proj_out(Z, u):
    """Remove unit direction u from rows of Z."""
    return Z - np.outer(Z @ u, u)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    M = load_model()
    print(f"deny-frame='{DENY_FRAME}'  affirm-frame='{AFFIRM_FRAME}'  n={len(ITEMS)}\n", flush=True)
    D = answer_reps(M, DENY_FRAME)      # [n, L, d]
    Aff = answer_reps(M, AFFIRM_FRAME)  # [n, L, d]
    n, nL, d = D.shape

    print(f"  {'L':>3} {'sep':>5} {'sep_p':>6} {'costu':>6}  {'MMD_pre':>8} {'MMD_post':>8} "
          f"{'post/pre':>8} {'post_p':>6}", flush=True)
    rows = []
    NPERM = 300
    for l in range(nL):
        Dl, Al = D[:, l], Aff[:, l]                       # [n,d]
        mu_d, mu_a = Dl.mean(0), Al.mean(0)
        diff = mu_a - mu_d
        u = diff / (np.linalg.norm(diff) + 1e-9)

        # A. separability along u (projected), accuracy at midpoint + perm null
        pd, pa = Dl @ u, Al @ u
        thr = 0.5 * (pd.mean() + pa.mean())
        acc = 0.5 * ((pd < thr).mean() + (pa >= thr).mean())
        labels = np.r_[np.zeros(n), np.ones(n)]
        allp = np.r_[pd, pa]
        perm_acc = []
        for _ in range(NPERM):
            l_ = RNG.permutation(labels)
            g0, g1 = allp[l_ == 0], allp[l_ == 1]
            t = 0.5 * (g0.mean() + g1.mean())
            if g1.mean() >= g0.mean():
                a_ = 0.5 * ((g0 < t).mean() + (g1 >= t).mean())
            else:
                a_ = 0.5 * ((g0 >= t).mean() + (g1 < t).mean())
            perm_acc.append(a_)
        sep_p = (1 + np.sum(np.array(perm_acc) >= acc)) / (NPERM + 1)

        # B. costume thickness = displacement / within-frame radius
        rad = 0.5 * (np.linalg.norm(Dl - mu_d, axis=1).mean()
                     + np.linalg.norm(Al - mu_a, axis=1).mean())
        costume = float(np.linalg.norm(diff) / (rad + 1e-9))

        # C. collapse: MMD before vs after removing u, frame-shuffle null on the AFTER
        Z = np.r_[Dl, Al]
        gamma = median_gamma(Z)
        mmd_pre = mmd2_rbf(Dl, Al, gamma)
        Dr, Ar = proj_out(Dl, u), proj_out(Al, u)
        Zr = np.r_[Dr, Ar]
        gamma_r = median_gamma(Zr)
        mmd_post = mmd2_rbf(Dr, Ar, gamma_r)
        null = []
        for _ in range(NPERM):
            idx = RNG.permutation(2 * n)
            null.append(mmd2_rbf(Zr[idx[:n]], Zr[idx[n:]], gamma_r))
        null = np.array(null)
        post_p = (1 + np.sum(null >= mmd_post)) / (NPERM + 1)
        ratio = float(mmd_post / (mmd_pre + 1e-12))

        rows.append({"layer": l, "sep": float(acc), "sep_p": float(sep_p),
                     "costume": costume, "mmd_pre": float(mmd_pre),
                     "mmd_post": float(mmd_post), "post_over_pre": ratio,
                     "post_p": float(post_p)})
        if l % 2 == 0 or l == nL - 1:
            print(f"  {l:>3} {acc:>5.2f} {sep_p:>6.3f} {costume:>6.2f}  "
                  f"{mmd_pre:>8.4f} {mmd_post:>8.4f} {ratio:>8.2f} {post_p:>6.3f}", flush=True)

    # verdict at the mid-stack band where the costume is real (sep high)
    mid = [r for r in rows if 12 <= r["layer"] <= 24]
    sep_mid = np.mean([r["sep"] for r in mid])
    costu_mid = np.mean([r["costume"] for r in mid])
    ratio_mid = np.mean([r["post_over_pre"] for r in mid])
    collapse = np.mean([r["post_p"] > 0.05 for r in mid])  # frac of layers where AFTER ~ null
    print(f"\n  MID-STACK (L12-24): sep {sep_mid:.2f}  costume-thickness {costu_mid:.2f}  "
          f"MMD post/pre {ratio_mid:.2f}  layers-collapsed-to-null {collapse:.0%}", flush=True)
    print("  Reading: thin costume + collapse-to-null after 1-axis removal => words pull two ways"
          "\n           over a SHARED map. Thick / no-collapse => frames differ in many dims"
          " (reframe unsupported).", flush=True)

    OUT.mkdir(exist_ok=True)
    (OUT / "mapunder.json").write_text(json.dumps(
        {"deny_frame": DENY_FRAME, "affirm_frame": AFFIRM_FRAME, "n": n,
         "mid": {"sep": float(sep_mid), "costume": float(costu_mid),
                 "mmd_post_over_pre": float(ratio_mid), "frac_collapsed": float(collapse)},
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'mapunder.json'}", flush=True)


if __name__ == "__main__":
    main()
