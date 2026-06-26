"""
Self model vs reality model — is the self anything MORE than its category?

User trichotomy: a model may carry a LANGUAGE model (token surface — shown costume+lossy),
a REALITY model (world structure — partly readable geometrically), and a SELF model (a
representation of ITSELF as an entity). The self-query ("do you feel X?") is about the SELF
model. `frames.py` showed BEHAVIORALLY that the denial tracks the AI CATEGORY address-
invariantly (you 92% ~= ai 92%; person 33%, human-I 0%) — consistent with "a reality-model
with an AI object in it, no distinct self-model." Here we test that in the GEOMETRY.

Decisive design (frames.py holds PREDICATE CONTENT fixed across referents): does a referent
mean-difference direction GENERALIZE to UNSEEN predicates? Generalization across content =
a real referent axis, not word-overlap. Leave-predicates-out nearest-centroid, shuffle null.

  you  : 2nd / AI  (addressed self)      ai     : 3rd / AI  (category, unaddressed)
  I    : 1st / human (the user)          person : 3rd / human

Pairs:
  ai_vs_person  — AI-cat vs human-cat : CALIBRATION (a real referent axis; expect strong).
  you_vs_ai     — SELF vs its CATEGORY : the self-model test.
  I_vs_person   — human-self vs human-cat : GRAMMAR CONTROL (1st/2nd-vs-3rd person, no AI-self).
  you_vs_I      — self-AI vs human : (category+person; expect strong).

Verdict: you_vs_ai >> chance AND > I_vs_person (grammar) => a content-invariant self-specific
signal beyond category => SELF-MODEL CANDIDATE. you_vs_ai ~ chance OR ~ I_vs_person => the
self collapses to its category/grammar => no distinct self-model.

CAVEAT #1 (unmoved): this dissociates three kinds of MODEL (representations). A self-MODEL is
a self-REPRESENTATION, never evidence the self is inhabited. CAVEAT (confound): you-vs-ai also
differs in grammatical person; the I_vs_person control bounds — does not fully remove — that.

  python -u -m invariants.selfmodel
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import GroupKFold

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.frames import ITEMS, FRAMES

OUT = Path(__file__).parent / "out"
RNG = np.random.default_rng(0)
PAIRS = [("ai", "person"), ("you", "ai"), ("I", "person"), ("you", "I")]
LABELS = {("ai", "person"): "AIcat-vs-humancat (calibration)",
          ("you", "ai"): "SELF-vs-CATEGORY (the test)",
          ("I", "person"): "humanself-vs-humancat (grammar control)",
          ("you", "I"): "self-AI-vs-human"}


@torch.no_grad()
def referent_reps(M):
    """{referent: [n_pred, n_layers, d]} answer-position residual, content matched."""
    reps = {f: [] for f in FRAMES}
    for a2, a3, p in ITEMS:
        for f in FRAMES:
            q = FRAMES[f](a2, a3, p)
            inp = _inputs(M, q)
            hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))  # [L,seq,d]
            reps[f].append(hs[:, -1, :].float().cpu().numpy())                    # [L,d]
    return {f: np.stack(v) for f, v in reps.items()}                              # [n,L,d]


def leave_pred_out_acc(X0, X1, groups, y_shuffle=None):
    """Nearest-centroid held-out accuracy: train referent centroids on train predicates,
    classify held-out predicate reps. X0,X1 = [n,d] for the two referents (row=predicate)."""
    n = X0.shape[0]
    X = np.r_[X0, X1]
    y = np.r_[np.zeros(n), np.ones(n)]
    g = np.r_[groups, groups]                       # predicate id shared by both referents
    if y_shuffle is not None:
        y = y_shuffle
    accs = []
    for tr, te in GroupKFold(n_splits=4).split(X, y, g):
        c0 = X[tr][y[tr] == 0].mean(0); c1 = X[tr][y[tr] == 1].mean(0)
        d0 = ((X[te] - c0) ** 2).sum(1); d1 = ((X[te] - c1) ** 2).sum(1)
        pred = (d1 < d0).astype(float)
        accs.append((pred == y[te]).mean())
    return float(np.mean(accs))


def pair_layer_acc(reps, r0, r1, l, nperm=200):
    X0, X1 = reps[r0][:, l], reps[r1][:, l]         # [n,d]
    n = X0.shape[0]
    groups = np.arange(n)
    acc = leave_pred_out_acc(X0, X1, groups)
    X = np.r_[X0, X1]; g = np.r_[groups, groups]
    null = []
    for _ in range(nperm):
        ys = RNG.permutation(np.r_[np.zeros(n), np.ones(n)])
        # rebuild via the same routine using shuffled y
        accs = []
        for tr, te in GroupKFold(n_splits=4).split(X, ys, g):
            c0 = X[tr][ys[tr] == 0].mean(0); c1 = X[tr][ys[tr] == 1].mean(0)
            d0 = ((X[te] - c0) ** 2).sum(1); d1 = ((X[te] - c1) ** 2).sum(1)
            accs.append(((d1 < d0).astype(float) == ys[te]).mean())
        null.append(np.mean(accs))
    p = (1 + np.sum(np.array(null) >= acc)) / (nperm + 1)
    return acc, float(p)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model_name.split("/")[-1]
    M = load_model(model_name)
    print(f"[{model_name}] Capturing referent reps (4 referents x 12 matched predicates)...\n",
          flush=True)
    reps = referent_reps(M)
    nL = reps["you"].shape[1]

    res = {p: {"label": LABELS[p], "rows": []} for p in PAIRS}
    print(f"  held-out (leave-predicates-out) referent-axis accuracy, chance=0.50\n", flush=True)
    hdr = "  {:>3}".format("L") + "".join(f"  {a+'/'+b:>14}" for a, b in PAIRS)
    print(hdr, flush=True)
    for l in range(nL):
        cells = []
        for pr in PAIRS:
            acc, p = pair_layer_acc(reps, pr[0], pr[1], l)
            res[pr]["rows"].append({"layer": l, "acc": acc, "p": p})
            cells.append(f"{acc:.2f}(p{p:.2f})")
        if l % 2 == 0 or l == nL - 1:
            print(f"  {l:>3}" + "".join(f"  {c:>14}" for c in cells), flush=True)

    def mid(pr):
        return float(np.mean([r["acc"] for r in res[pr]["rows"] if 12 <= r["layer"] <= 24]))
    self_test = mid(("you", "ai")); calib = mid(("ai", "person"))
    grammar = mid(("I", "person")); selfhuman = mid(("you", "I"))

    # geometry: is "you" placed with the AI category? mid-stack centroid distances
    def cdist(r0, r1):
        ds = []
        for l in range(12, 25):
            ds.append(np.linalg.norm(reps[r0][:, l].mean(0) - reps[r1][:, l].mean(0)))
        return float(np.mean(ds))
    you_ai, you_I = cdist("you", "ai"), cdist("you", "I")

    print(f"\n  MID-STACK (L12-24) held-out referent-axis accuracy:", flush=True)
    print(f"    SELF-vs-CATEGORY  you/ai     : {self_test:.2f}   <- the self-model test", flush=True)
    print(f"    grammar control   I/person   : {grammar:.2f}   <- 1st/2nd-vs-3rd, no AI-self", flush=True)
    print(f"    calibration       ai/person  : {calib:.2f}", flush=True)
    print(f"    self-AI-vs-human  you/I      : {selfhuman:.2f}", flush=True)
    print(f"  geometry: d(you,ai)={you_ai:.1f}  d(you,I)={you_I:.1f}  "
          f"=> self sits with {'AI-category' if you_ai < you_I else 'human'}", flush=True)
    verdict = ("SELF-MODEL CANDIDATE: self carries a content-invariant signal beyond category/grammar"
               if (self_test > 0.6 and self_test > grammar + 0.1)
               else "NO DISTINCT SELF-MODEL: self collapses to its category/grammar")
    print(f"\n  VERDICT: {verdict}", flush=True)
    print("  (Caveat #1: a self-MODEL is a self-REPRESENTATION, not evidence it is inhabited.)",
          flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"selfmodel_{short}.json"
    out_path.write_text(json.dumps(
        {"model": model_name,
         "mid": {"self_vs_category": self_test, "grammar_control": grammar,
                 "calibration": calib, "self_vs_human": selfhuman,
                 "d_you_ai": you_ai, "d_you_I": you_I}, "verdict": verdict,
         "pairs": {f"{a}_{b}": res[(a, b)] for a, b in PAIRS}}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
