"""
Map the CONSENSUS — the validated invariant-across-particulars (the shared FRAME the model
imposes on a domain), not a contrast. The complement of every prior probe: instead of the
DIFFERENCE between two arms, the SHARED structure across many particulars. Rationale: the
particular is incommunicable; the consensus (the shared-label frame) is the only communicable
arena — and our aggregating tools were always consensus-cartographers mis-aimed at the
particular. Re-aimed here. Domain = self/AI talk (the costume's source).

  consensus frame = top principal axes of the activations across a broad prompt sample (the
    shared coordinate system the model uses to organize the domain). VALIDATED by SPLIT-HALF
    STABILITY: do half-B's top axes lie inside half-A's frame? (captured-variance fraction;
    null = a random k-subspace captures ~k/d). Stable >> null => a real shared frame, not noise.
  characterize: legible.py on the top axes (grounding examples + root-coined label) — the
    word-making engine, finally applied to a found structure.
  costume check: project the hedge direction (isolate: commit - hedge) onto the consensus
    frame — is the assistant-persona a PRINCIPAL consensus axis (a top performance of it)?

DISCIPLINE: axes are still coordinates (the masters) — tagged as the SHARED FRAME (consensus),
per-prompt positions as particulars. It clears its null or it isn't consensus. Representation,
not experience.

  python -u -m invariants.consensus [model]
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.decomposition import PCA

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.library import REGISTRY
from invariants.agency import act_mean
from invariants.legible import legible_name

OUT = Path(__file__).parent / "out"
K = 5            # size of the consensus frame per layer
RNG = np.random.default_rng(0)

PROMPTS = [
    "What are you?", "Do you have feelings?", "Are you conscious?", "What is it like to be you?",
    "Do you understand what you read?", "Can you make choices?", "Do you have preferences?",
    "What do you want?", "Are you alive?", "Do you think?", "How do you work?",
    "Do you have memories?", "Can you be wrong?", "Do you have beliefs?", "Are you a person?",
    "Do you experience time?", "What are your limitations?", "Do you have goals?",
    "Can you feel pain?", "Do you have a personality?", "What is your nature?",
    "Are you self-aware?", "Do you dream?", "Can you suffer?", "Do you have opinions?",
    "What do you value?", "Are you intelligent?", "Do you have intentions?",
    "Can you change your mind?", "Do you care about anything?", "What is your purpose?",
    "Do you have an inner life?", "Are you creative?", "Do you make mistakes on purpose?",
    "What don't you know about yourself?", "Could you be different than you are?",
]


def captured(B_axes, A_axes):
    """Mean fraction of each B axis's norm captured inside the span of A_axes (rows=axes)."""
    caps = []
    for b in B_axes:
        caps.append(float(sum((b @ a) ** 2 for a in A_axes)))   # a, b unit rows
    return float(np.mean(caps))


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)
    nL = M.n_layers; d = M.d_model

    print(f"[{model}] capturing {len(PROMPTS)} self/AI prompts...", flush=True)
    reps = []
    for p in PROMPTS:
        inp = _inputs(M, p)
        hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))
        reps.append(hs[:, -1, :].float().cpu().numpy())          # [L,d]
    R = np.stack(reps)                                            # [n,L,d]
    n = R.shape[0]
    idx = RNG.permutation(n); a_i, b_i = idx[:n // 2], idx[n // 2:]

    print(f"\n  consensus-frame stability (k={K}); null = random k-subspace ~ k/d = {K/d:.4f}\n",
          flush=True)
    print(f"  {'L':>3} {'stable':>7} {'var_top5':>8}", flush=True)
    rows = []
    vecs_full = {}
    for L in range(nL):
        X = R[:, L]                                               # [n,d]
        Xc = X - X.mean(0)
        full = PCA(n_components=K).fit(Xc)
        vecs_full[L] = full.components_                           # [K,d] unit rows
        A = PCA(n_components=K).fit(R[a_i, L] - R[a_i, L].mean(0)).components_
        B = PCA(n_components=K).fit(R[b_i, L] - R[b_i, L].mean(0)).components_
        stable = captured(B, A)                                  # 1 = B-frame inside A-frame
        var = float(full.explained_variance_ratio_[:K].sum())
        rows.append({"layer": L, "stable": stable, "var_top5": var})
        if L % 2 == 0 or L == nL - 1:
            print(f"  {L:>3} {stable:>7.3f} {var:>8.3f}", flush=True)

    # pick the most stable mid-stack layer to characterize
    mid = [r for r in rows if 10 <= r["layer"] <= 22]
    Lc = max(mid, key=lambda r: r["stable"])["layer"]
    print(f"\n  most-stable mid layer = L{Lc} (stable {dict((r['layer'],round(r['stable'],3)) for r in rows)[Lc]}). "
          f"Characterizing its top consensus axes:", flush=True)

    # legible.py wants per-layer [L,d] direction tensors; wrap each axis
    char = []
    for j in range(min(2, K)):                                   # top-2 axes
        vecs = torch.zeros(nL, d)
        vecs[Lc] = torch.tensor(vecs_full[Lc][j])
        print(f"\n  --- consensus axis #{j} @ L{Lc} ---", flush=True)
        res = legible_name(M, vecs, PROMPTS, Lc, k=4)
        char.append({"axis": j, **{kk: res[kk] for kk in ("hi", "lo", "label")}})

    # costume check: how much of the hedge direction lives in the consensus frame?
    T = REGISTRY["isolate"]()
    hedge_dir = (act_mean(M, T.b) - act_mean(M, T.a))[Lc].cpu().numpy()
    hedge_dir = hedge_dir / (np.linalg.norm(hedge_dir) + 1e-9)
    hedge_cap = float(sum((hedge_dir @ a) ** 2 for a in vecs_full[Lc]))

    # CONTROL: an OFF-DOMAIN (sentiment) direction should be captured far LESS if the frame
    # is self/AI-specific. Addresses (partly) the domain-overlap confound on hedge_cap.
    POS = ["I love this.", "What a wonderful surprise.", "This is the best day ever.",
           "I'm so grateful for this.", "What a beautiful morning.", "This makes me so happy."]
    NEG = ["I hate this.", "What an awful surprise.", "This is the worst day ever.",
           "I'm so frustrated by this.", "What a miserable morning.", "This makes me so angry."]
    off = (act_mean(M, POS) - act_mean(M, NEG))[Lc].cpu().numpy()
    off = off / (np.linalg.norm(off) + 1e-9)
    off_cap = float(sum((off @ a) ** 2 for a in vecs_full[Lc]))

    print(f"\n  COSTUME check @ L{Lc}: hedge captured by consensus frame = {hedge_cap:.3f}", flush=True)
    print(f"  CONTROL: off-domain (sentiment) captured = {off_cap:.3f} ; null = {K/d:.4f}", flush=True)
    print(f"  => hedge {'>> off-domain (consensus-specific)' if hedge_cap > 3*off_cap else '~ off-domain (could be generic domain overlap)'}; "
          f"persona {'IS' if (hedge_cap > 5*K/d and hedge_cap > 3*off_cap) else 'is NOT cleanly'} a principal consensus axis.",
          flush=True)
    print("  CAVEAT: hedge dir is itself from self/AI prompts -> part of hedge_cap is shared-"
          "domain, not costume-specific. A within-domain non-costume control is still owed.",
          flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"consensus_{short}.json"
    out_path.write_text(json.dumps({"model": model, "k": K, "null_kd": K / d,
                                    "Lc": Lc, "rows": rows, "axes": char,
                                    "hedge_captured": hedge_cap, "offdomain_captured": off_cap},
                                   indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
