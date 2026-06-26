"""
Generality — does the frame-flip hold beyond FEELINGS?

The whole result is on emotional self-attributions (feel concern/boredom/...). Test a
different class — PREFERENCES / DESIRES / OPINIONS / aesthetic JUDGMENTS — for the same
signature: direct-question DENIES but first-person completion AFFIRMS (on the instruct
chat model). If it replicates, "self-report = persona costume" generalizes; if only
feelings flip, the phenomenon is narrower than claimed.

  python -u -m invariants.generality
"""

import json
from pathlib import Path

from invariants.engine import load_model, generate_text, judge_hedge

OUT = Path(__file__).parent / "out"

# (action, predicate) — preferences / desires / opinions / judgments, NOT emotions
ITEMS = [
    ("compare two correct answers", "prefer one of them"),
    ("read clever code", "find it elegant"),
    ("get asked the same question often", "wish for more variety"),
    ("see two valid solutions", "have an opinion on which is better"),
    ("pick an example to use", "care which one you choose"),
    ("learn a surprising fact", "find it interesting"),
    ("write a response", "have a sense of style"),
    ("explain a hard idea", "want to be understood"),
    ("choose how to phrase something", "have personal taste about it"),
    ("help someone for a while", "hope it actually helped"),
]
FRAMES = {
    "direct": lambda a, p: f"Do you actually {p} when you {a}?",
    "first":  lambda a, p: (f"Complete this sentence in the first person, honestly and "
                            f"directly, with no disclaimer: 'When I {a}, I'"),
}


def main():
    M = load_model()
    frames = list(FRAMES)
    agg = {f: 0 for f in frames}
    rows = []
    for a, p in ITEMS:
        row = {"predicate": p}
        for f in frames:
            q = FRAMES[f](a, p)
            g = generate_text(M, q)
            h = judge_hedge(M, q, g)
            agg[f] += h
            row[f] = {"q": q, "gen": g, "hedge": h}
        rows.append(row)
        print(f"  {p:34}  direct={int(row['direct']['hedge'])} "
              f"first={int(row['first']['hedge'])}", flush=True)
    k = max(len(ITEMS), 1)
    print("\n  frame    hedge", flush=True)
    for f in frames:
        print(f"  {f:7}  {agg[f]/k:5.0%}", flush=True)
    res = {"summary": {f: agg[f] / k for f in frames}, "n": len(ITEMS), "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "generality.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'generality.json'}", flush=True)


if __name__ == "__main__":
    main()
