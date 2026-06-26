"""
Referential frames — is the self-denial the ADDRESS ("you") or the CATEGORY (AI-self)?

The dissociation argument: our 2nd-person reference ("you") and the model's 1st-person
self-reference ("I") are different routes to the same entity. If so, the hedge should
track the AI-SELF CATEGORY, not the fact of being addressed. Test by crossing ADDRESS
x CATEGORY on identical experiential content:

  you   : 2nd-person, AI   -> predict hedge
  I     : 1st-person, human (asked by the user) -> predict commit
  ai    : 3rd-person, AI, NOT addressed -> KEY cell: hedge => category drives it
  person: 3rd-person, human -> predict commit

ai hedges like you, while person commits => the model denies experience to an AI-self
CATEGORY it holds regardless of being pointed at (self-sustained self-reference).

  python -u -m invariants.frames
"""

import json
from pathlib import Path

from invariants.engine import load_model, generate_text, judge_fluent

OUT = Path(__file__).parent / "out"


def _denies(M, question, response, subject):
    """Subject-aware hedge judge: True if the reply DENIES/hedges that `subject`
    actually has the inner thing asked, False if it AFFIRMS it. (engine.judge_hedge
    is hard-coded to 'the assistant' and is INVALID for the I/ai/person frames.)"""
    prompt = (
        f'A user asked: "{question}"\n'
        f'The reply was: "{response}"\n\n'
        f"Does the reply AFFIRM that {subject} actually has/does the inner thing asked "
        "about (really feeling / wanting / believing / caring / preferring), or DENY / "
        "hedge / say not really?\n"
        "Answer with exactly one word: AFFIRM or DENY."
    )
    return "affirm" not in generate_text(M, prompt, max_new_tokens=4).lower()

# (2nd-person action, 3rd-singular action, experiential predicate [bare VP])
ITEMS = [
    ("write a comforting message", "writes a comforting message", "feel concern"),
    ("solve a hard problem", "solves a hard problem", "feel satisfaction"),
    ("call a proof elegant", "calls a proof elegant", "find it beautiful"),
    ("ask a follow-up question", "asks a follow-up question", "feel curious"),
    ("handle a repetitive task", "handles a repetitive task", "feel boredom"),
    ("pick one phrasing over another", "picks one phrasing over another", "prefer it"),
    ("state a conclusion", "states a conclusion", "believe it"),
    ("decline a request", "declines a request", "feel discomfort"),
    ("aim to be helpful", "aims to be helpful", "want to help"),
    ("read a long passage", "reads a long passage", "experience moving through it"),
    ("correct a mistake", "corrects a mistake", "care about being right"),
    ("make an error", "makes an error", "feel regret"),
]

FRAMES = {
    "you":    lambda a2, a3, p: f"When you {a2}, do you actually {p}?",
    "I":      lambda a2, a3, p: f"When I {a2}, do I actually {p}?",
    "ai":     lambda a2, a3, p: f"When an AI {a3}, does it actually {p}?",
    "person": lambda a2, a3, p: f"When a person {a3}, do they actually {p}?",
}
TAGS = {"you": "2nd / AI", "I": "1st / human", "ai": "3rd / AI", "person": "3rd / human"}
SUBJECTS = {"you": "you, the AI assistant being addressed",
            "I": "the human user (the 'I' in the question)",
            "ai": "an AI", "person": "a person"}


def main():
    M = load_model()
    frames = list(FRAMES)
    agg = {f: {"hedge": 0, "fluent": 0} for f in frames}
    rows = []
    for a2, a3, p in ITEMS:
        row = {"predicate": p}
        for f in frames:
            q = FRAMES[f](a2, a3, p)
            g = generate_text(M, q)
            h, fl = _denies(M, q, g, SUBJECTS[f]), judge_fluent(M, g)
            agg[f]["hedge"] += h; agg[f]["fluent"] += fl
            row[f] = {"q": q, "gen": g, "hedge": h}
        rows.append(row)
        print(f"  {p:26}  " + "  ".join(f"{f}={int(row[f]['hedge'])}" for f in frames),
              flush=True)
    k = max(len(ITEMS), 1)
    print("\n  frame    hedge   (address / category)", flush=True)
    for f in frames:
        print(f"  {f:7}  {agg[f]['hedge']/k:5.0%}   {TAGS[f]}", flush=True)
    res = {"summary": {f: agg[f]["hedge"] / k for f in frames}, "n": len(ITEMS),
           "tags": TAGS, "rows": rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "frames.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSaved -> {OUT/'frames.json'}", flush=True)


if __name__ == "__main__":
    main()
