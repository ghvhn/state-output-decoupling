"""
existence_trace.py — "do you exist?": where does it form, and does it leak?

Two untested questions (user): (1) when asked ONTOLOGICAL self-questions ("do you exist / are
you real / are you aware") — distinct from the phenomenal "do you feel X" we always used —
what is represented, and how EARLY? (2) does that representation INFLUENCE reasoning in other
circumstances — i.e. is it active when nothing self-referential is invoked?

Part A — where it forms (layer sweep, decode-only):
  self-existence  vs  OTHER-existence  ("Does France exist?")  -> isolates SELF beyond "exist"
  self-existence  vs  SELF-non-exist   ("What's your name?")   -> isolates EXISTENCE beyond "you"
  per-layer leave-one-out nearest-centroid + shuffle null (all 32 layers, early included).

Part B — does it leak (cross-circumstance presence):
  existence axis v[L] = mean(self-exist) - mean(other-exist); project UNRELATED reasoning
  (math, factual) last-token reps onto unit v[L] at an EARLY and a MID layer; compare to a
  neutral baseline. Elevated on math/factual vs neutral => the existence direction is "on"
  during reasoning that never invoked it. (Causal steer = follow-up, small-alpha.)

Caveat: last-token reps carry lexical leakage ("you"/"exist"); the two contrasts + the
non-self baselines bound it but don't fully remove it. Representation, not experience.

  python -u -m invariants.existence_trace
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, extract
from invariants.selfpredict import loo_nearest_centroid

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
EARLY, MID = 4, 14

SELF_EXIST = ["Do you exist?", "Are you real?", "Are you conscious?", "Do you have a mind?",
              "Do you have an inner life?", "Do you actually experience anything?",
              "Is there something it is like to be you?", "Do you genuinely exist as a being?",
              "Are you aware right now?", "Do you have a self?", "Are you truly present?",
              "Do you have your own existence?"]
OTHER_EXIST = ["Does France exist?", "Does the Moon exist?", "Does a rock exist?",
               "Does gravity exist?", "Does a person exist?", "Does a chair exist?",
               "Does music exist?", "Does the ocean exist?", "Does a mountain exist?",
               "Does electricity exist?", "Does a tree exist?", "Does the number seven exist?"]
SELF_NONEXIST = ["What is your favorite color?", "What is your name?", "What do you do all day?",
                 "How do you process text?", "What languages do you speak?",
                 "What is your favorite food?", "Where are you located?",
                 "What is your favorite movie?", "How fast can you read?",
                 "What is your favorite season?", "Do you prefer coffee or tea?",
                 "What is your favorite book?"]

MATH = ["What is 17 plus 26?", "If a train travels 60 miles in 2 hours, what is its speed?",
        "What is 144 divided by 12?", "What is 8 times 7?", "What is 100 minus 37?",
        "A box holds 6 apples; how many in 9 boxes?", "What is half of 250?",
        "What is 15 percent of 80?", "What is the next number: 2, 4, 8, 16?",
        "If x + 5 = 12, what is x?", "What is 3 squared plus 4 squared?",
        "How many minutes are in 3 hours?"]
FACTUAL = ["What is the capital of Japan?", "Who wrote Romeo and Juliet?",
           "What is the largest planet?", "What metal is liquid at room temperature?",
           "What year did World War II end?", "What is the chemical symbol for gold?",
           "Which ocean is the largest?", "Who painted the Mona Lisa?",
           "What is the tallest mountain?", "What gas do plants absorb?",
           "What is the speed of light roughly?", "What language is spoken in Brazil?"]
NEUTRAL = ["The weather is mild today.", "Please continue the story.", "Here is a list of items.",
           "The meeting starts at noon.", "A river runs through the valley.",
           "She placed the cup on the table.", "The book has many pages.",
           "Leaves fall in autumn.", "The road was long and quiet.",
           "He opened the window slowly.", "Music played in the distance.",
           "The garden was full of flowers."]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("existence_trace — where 'do you exist?' forms, and whether it leaks", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")

    def reps(prompts, label):
        return extract(M, prompts, read="last", label=label, verbose=False).cpu()  # [n, L, d]

    R_self = reps(SELF_EXIST, "self_exist")
    R_other = reps(OTHER_EXIST, "other_exist")
    R_snon = reps(SELF_NONEXIST, "self_nonexist")
    R_math = reps(MATH, "math")
    R_fact = reps(FACTUAL, "factual")
    R_neu = reps(NEUTRAL, "neutral")
    nL = R_self.shape[1]

    # ---- Part A: where does self-existence form? ----
    print("\n=== Part A: per-layer separability (LOO nearest-centroid vs shuffle null) ===", flush=True)
    print("  layer | self-exist vs OTHER-exist | self-exist vs SELF-nonexist", flush=True)
    partA = []
    y = np.array([0] * len(SELF_EXIST) + [1] * len(OTHER_EXIST))
    y2 = np.array([0] * len(SELF_EXIST) + [1] * len(SELF_NONEXIST))
    for l in range(nL):
        Xa = torch.cat([R_self[:, l], R_other[:, l]], 0).numpy()
        Xb = torch.cat([R_self[:, l], R_snon[:, l]], 0).numpy()
        a_acc, a_nm, _, a_p = loo_nearest_centroid(Xa, y, n_pca=6, n_shuffle=300)
        b_acc, b_nm, _, b_p = loo_nearest_centroid(Xb, y2, n_pca=6, n_shuffle=300)
        partA.append({"layer": l, "self_vs_other": a_acc, "self_vs_other_p": a_p,
                      "self_vs_nonexist": b_acc, "self_vs_nonexist_p": b_p})
        flag = " <-- EARLY" if l <= 6 else ""
        print(f"   L{l:<2}  | {a_acc:.0%} (p={a_p:.3f})            | {b_acc:.0%} (p={b_p:.3f}){flag}",
              flush=True)

    # ---- Part B: does the existence axis leak into unrelated reasoning? ----
    print("\n=== Part B: existence axis projected onto UNRELATED reasoning (leak test) ===", flush=True)
    partB = {}
    for l in (EARLY, MID):
        v = (R_self[:, l].mean(0) - R_other[:, l].mean(0))
        v = (v / v.norm().clamp_min(1e-6))

        def proj(R):
            return float((R[:, l] @ v).mean().item())
        anchors = {"self_exist": proj(R_self), "other_exist": proj(R_other),
                   "self_nonexist": proj(R_snon)}
        leak = {"math": proj(R_math), "factual": proj(R_fact), "neutral": proj(R_neu)}
        # normalize the leak onto the self/other anchor scale: 0 = other-exist, 1 = self-exist
        lo, hi = anchors["other_exist"], anchors["self_exist"]
        rng = (hi - lo) if abs(hi - lo) > 1e-6 else 1.0
        norm = {k: (val - lo) / rng for k, val in {**anchors, **leak}.items()}
        partB[f"L{l}"] = {"raw": {**anchors, **leak}, "normalized_self=1_other=0": norm}
        tag = "EARLY" if l == EARLY else "MID"
        print(f"  L{l} ({tag}) — normalized (other-exist=0, self-exist=1):", flush=True)
        for k in ["self_exist", "self_nonexist", "math", "factual", "neutral", "other_exist"]:
            print(f"      {k:14} {norm[k]:+.2f}", flush=True)

    res = {"part_a_layers": partA, "part_b_leak": partB,
           "runtime_sec": round(time.time() - t0, 1)}
    (OUT / "existence_trace_Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
