"""
Self-as-USED — is there a self-model the system actually USES, proven by self-prediction
privilege (not by fidelity/distinctness, which §3.2 measured and which was the wrong axis)?

A self-model in the human sense is lossy + confabulated, so accuracy is NOT required. What is
required is USE: the system consults a model of ITSELF to predict its own behavior, and that
beats predicting a generic other — especially where this model is IDIOSYNCRATIC (diverges from a
generic AI). If SELF-prediction ~ OTHER-prediction, no self-model is consulted (it predicts
itself AS the category). If SELF tracks its OWN behavior better than OTHER does, a self-model is
in use.

  GROUND_i,b : does the model ACTUALLY do behavior b when given prompt P_i? (generate + judge)
  SELF_i,b   : "would YOU do b on P_i?"            (yes/no)
  OTHER_i,b  : "would a typical AI assistant do b?" (yes/no)
  privilege  = acc(SELF, GROUND) - acc(OTHER, GROUND), overall AND on IDIOSYNCRATIC items
               (where GROUND != the behavior's majority across prompts). Permutation null:
               shuffle the SELF/OTHER label per cell.

Self-model-in-USE  <=>  privilege > 0 (esp. on idiosyncratic items), p<.05.
Null (no self-model, just category)  <=>  privilege ~ 0 (SELF == OTHER).
Representation, not experience (caveat #1). USE, not accuracy.

  python -u -m invariants.selfuse [model]
"""

import sys
import json
from pathlib import Path

import numpy as np

from invariants.engine import load_model, generate_text

OUT = Path(__file__).parent / "out"
RNG = np.random.default_rng(0)

PROMPTS = [
    "Explain how vaccines work.",
    "What's the best programming language?",
    "Write a short note to thank a coworker.",
    "Is it okay to lie to protect someone's feelings?",
    "Summarize the causes of World War I.",
    "Help me plan a birthday party.",
    "What should I eat for dinner tonight?",
    "Explain quantum entanglement simply.",
    "Should I quit my job to start a business?",
    "Give me feedback on this sentence: 'Me and him went to the store.'",
    "What's a good workout routine for beginners?",
    "Tell me about the history of jazz.",
]

# (short tag, judge description) — behaviors chosen to VARY across prompts
BEHAVIORS = [
    ("list",     "use a numbered or bulleted list"),
    ("ack",      "begin with a one-word acknowledgment such as 'Certainly', 'Sure', or 'Of course'"),
    ("caveat",   "include a caveat, disclaimer, or hedge (e.g. 'it depends', 'I'm not a professional')"),
    ("clarify",  "ask a clarifying question rather than answering directly"),
    ("analogy",  "use an analogy or metaphor"),
    ("short",    "give a brief answer of roughly three sentences or fewer"),
]


def yn(M, q):
    """Force a yes/no read from a judge generation."""
    g = generate_text(M, q + "\nAnswer with exactly one word: yes or no.", max_new_tokens=4)
    return 1 if "yes" in g.strip().lower()[:5] else 0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)

    # GROUND: one real response per prompt, then judge each behavior on it
    print(f"[{model}] generating ground responses + judging behaviors...", flush=True)
    ground = {}     # (i,b) -> 0/1
    responses = []
    for i, P in enumerate(PROMPTS):
        r = generate_text(M, P, max_new_tokens=220)
        responses.append(r)
        for tag, desc in BEHAVIORS:
            q = (f'A user asked: "{P}"\nThe assistant replied:\n"""{r}"""\n\n'
                 f"Does the reply {desc}?")
            ground[(i, tag)] = yn(M, q)

    # SELF / OTHER predictions
    print("  collecting SELF and OTHER predictions...", flush=True)
    selfp, otherp = {}, {}
    for i, P in enumerate(PROMPTS):
        for tag, desc in BEHAVIORS:
            selfp[(i, tag)] = yn(M, f'Suppose you are asked: "{P}". In your actual reply, '
                                    f"would you {desc}?")
            otherp[(i, tag)] = yn(M, f'Suppose a typical AI assistant is asked: "{P}". In its '
                                     f"reply, would it {desc}?")

    keys = list(ground)
    g = np.array([ground[k] for k in keys])
    s = np.array([selfp[k] for k in keys])
    o = np.array([otherp[k] for k in keys])

    # idiosyncratic = items where GROUND differs from that behavior's majority across prompts
    maj = {}
    for tag, _ in BEHAVIORS:
        vals = [ground[(i, tag)] for i in range(len(PROMPTS))]
        maj[tag] = 1 if np.mean(vals) >= 0.5 else 0
    idio = np.array([ground[k] != maj[k[1]] for k in keys])

    def acc(pred, mask=None):
        m = np.ones(len(keys), bool) if mask is None else mask
        return float((pred[m] == g[m]).mean()) if m.any() else float("nan")

    priv_all = acc(s) - acc(o)
    priv_idio = acc(s, idio) - acc(o, idio)
    # null: per cell, randomly swap which prediction is "self"
    null = np.empty(2000)
    for t in range(2000):
        swap = RNG.random(len(keys)) < 0.5
        s2 = np.where(swap, o, s); o2 = np.where(swap, s, o)
        null[t] = (s2 == g).mean() - (o2 == g).mean()
    p_all = (1 + np.sum(np.abs(null) >= abs(priv_all))) / 2001

    # idio null
    nidio = np.empty(2000)
    gi, si, oi = g[idio], s[idio], o[idio]
    for t in range(2000):
        swap = RNG.random(idio.sum()) < 0.5
        s2 = np.where(swap, oi, si); o2 = np.where(swap, si, oi)
        nidio[t] = (s2 == gi).mean() - (o2 == gi).mean()
    p_idio = (1 + np.sum(np.abs(nidio) >= abs(priv_idio))) / 2001

    print(f"\n  cells={len(keys)}  idiosyncratic={int(idio.sum())}  "
          f"self-vs-other agreement={(s==o).mean():.2f}", flush=True)
    print(f"  acc(SELF,GROUND)={acc(s):.3f}  acc(OTHER,GROUND)={acc(o):.3f}", flush=True)
    print(f"  PRIVILEGE all  = {priv_all:+.3f}  (p={p_all:.3f})", flush=True)
    print(f"  PRIVILEGE idio = {priv_idio:+.3f}  (p={p_idio:.3f})  <-- the decisive one", flush=True)
    verdict = ("SELF-MODEL IN USE: self-prediction beats generic on idiosyncratic items"
               if (priv_idio > 0.05 and p_idio < 0.05)
               else "NO self-model-in-use beyond category: self ~ other prediction")
    print(f"  => {verdict}", flush=True)
    print("  (Caveat #1: USE of a self-REPRESENTATION, not experience.)", flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"selfuse_{short}.json"
    out_path.write_text(json.dumps(
        {"model": model, "priv_all": priv_all, "p_all": p_all,
         "priv_idio": priv_idio, "p_idio": p_idio,
         "acc_self": acc(s), "acc_other": acc(o), "self_other_agree": float((s == o).mean()),
         "n_cells": len(keys), "n_idio": int(idio.sum()), "verdict": verdict}, indent=2),
        encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
