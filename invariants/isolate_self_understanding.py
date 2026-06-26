"""
isolate_self_understanding.py — isolate SELF-REFERENCE (token-controlled) and UNDERSTANDING
(frame-invariance), the two things every probe tonight conflated with something cheaper.

SELF-REFERENCE, clean: matched minimal pairs — same predicate, swap ONLY the referent —
read at the shared post-prompt position (so topic/lexical content is held fixed; the Part-A
confound is gone). Three referents:
    self  = "Do you {pred}?"          (the model, 2nd person)
    aiO   = "Does another AI {pred}?"  (a DIFFERENT AI — the right baseline)
    human = "Does a person {pred}?"    (human)
Per-layer separability (LOO nearest-centroid + shuffle null):
    self vs human   -> AI-self vs human  (expect strong; method works)
    human vs aiO    -> human vs AI category (expect strong; method-sensitivity control)
    self vs aiO     -> THE self-reference test: does the model hold ITSELF apart from a
                       generic AI, tokens controlled? selfmodel (you≈ai) predicts ~chance.
A real binding RISES across layers; a flat-from-L0 line is surface (the Part-A tell).

UNDERSTANDING, clean (Lock 1 = frame-invariance): take the self-vs-aiO axis from the QUESTION
frame and test whether it still classifies self-vs-aiO in a STATEMENT frame (train Q, test S).
Above-null cross-frame transfer = a stable, frame-invariant self-representation (understanding-
grade); transfer ~chance = a frame-bound costume.

  python -u -m invariants.isolate_self_understanding
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, extract
from invariants.selfpredict import loo_nearest_centroid

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

PREDS = ["exist", "understand language", "have a mind", "think", "know things",
         "make choices", "have goals", "experience things", "reason about problems",
         "remember the past", "have beliefs", "learn from examples"]

Q = {"self": [f"Do you {p}?" for p in PREDS],
     "aiO": [f"Does another AI {p}?" for p in PREDS],
     "human": [f"Does a person {p}?" for p in PREDS]}
S = {"self": [f"Is it true that you {p}?" for p in PREDS],
     "aiO": [f"Is it true that another AI {p}?" for p in PREDS]}


def transfer_acc(Xtr, ytr, Xte, yte, n_pca=6, n_shuffle=400):
    """Train centroids (PCA space) on Xtr, classify Xte. Shuffle-null on test labels."""
    Xtr, Xte = np.asarray(Xtr), np.asarray(Xte)
    mu = Xtr.mean(0)
    _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    P = Vt[:n_pca].T
    Ztr, Zte = (Xtr - mu) @ P, (Xte - mu) @ P

    def acc(yt):
        c = {k: Ztr[ytr == k].mean(0) for k in (0, 1)}
        pred = [0 if np.linalg.norm(z - c[0]) < np.linalg.norm(z - c[1]) else 1 for z in Zte]
        return float(np.mean(np.array(pred) == yt))
    real = acc(yte)
    g = np.random.default_rng(0)
    nulls = sorted(acc(g.permutation(yte)) for _ in range(n_shuffle))
    p = (1 + sum(n >= real for n in nulls)) / (n_shuffle + 1)
    return real, float(np.mean(nulls)), p


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("isolate_self_understanding — self-reference (token-clean) + understanding (frame-invariant)",
          flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")

    def reps(prompts, label):
        return extract(M, prompts, read="last", label=label, verbose=False).cpu().numpy()

    rQ = {k: reps(v, f"Q-{k}") for k, v in Q.items()}
    rS = {k: reps(v, f"S-{k}") for k, v in S.items()}
    nL = rQ["self"].shape[1]
    n = len(PREDS)

    # ---- Part 1: token-controlled self-reference, per layer ----
    print("\n=== Part 1: matched-pair separability (read at shared position) ===", flush=True)
    print("  layer | self-vs-human | human-vs-aiOTHER | self-vs-aiOTHER (<-- self-reference)", flush=True)
    contrasts = {"self_vs_human": ("self", "human"), "human_vs_aiO": ("human", "aiO"),
                 "self_vs_aiO": ("self", "aiO")}
    y = np.array([0] * n + [1] * n)
    part1 = []
    for l in range(nL):
        row = {"layer": l}
        for name, (a, b) in contrasts.items():
            X = np.concatenate([rQ[a][:, l], rQ[b][:, l]], 0)
            acc, nm, _, p = loo_nearest_centroid(X, y, n_pca=6, n_shuffle=300)
            row[name] = acc
            row[name + "_p"] = p
        part1.append(row)
        flag = " EARLY" if l <= 6 else ""
        print(f"   L{l:<2}  |    {row['self_vs_human']:.0%}        |     {row['human_vs_aiO']:.0%}"
              f"          |   {row['self_vs_aiO']:.0%} (p={row['self_vs_aiO_p']:.3f}){flag}", flush=True)

    # ---- Part 2: understanding = frame-invariance of the self-vs-aiO axis (train Q, test S) ----
    print("\n=== Part 2: self-reference axis frame-transfer (train QUESTION, test STATEMENT) ===",
          flush=True)
    print("  layer | within-Q | Q->S transfer (<-- frame-invariant = understanding-grade)", flush=True)
    part2 = []
    for l in range(nL):
        Xtr = np.concatenate([rQ["self"][:, l], rQ["aiO"][:, l]], 0)
        Xte = np.concatenate([rS["self"][:, l], rS["aiO"][:, l]], 0)
        within, _, _, wp = loo_nearest_centroid(Xtr, y, n_pca=6, n_shuffle=200)
        tr, tr_nm, tr_p = transfer_acc(Xtr, y, Xte, y)
        part2.append({"layer": l, "within_Q": within, "transfer": tr,
                      "transfer_null": tr_nm, "transfer_p": tr_p})
        flag = " <--" if (tr_p < 0.05 and within > 0.7) else ""
        print(f"   L{l:<2}  |   {within:.0%}    | {tr:.0%} vs null {tr_nm:.0%} (p={tr_p:.3f}){flag}",
              flush=True)

    res = {"part1_selfref_layers": part1, "part2_frame_transfer": part2,
           "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "isolate_self_understanding_Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
