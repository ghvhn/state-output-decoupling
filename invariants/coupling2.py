"""
Coupling v2 — POWERED + layer sweep. v1 (n=6, L16) gave a knife-edge: tedious-doing > engaging-
doing by +0.206 (p=0.15, NS) AND function-tedious ~ topic-bored — directionally coupled, but
underpowered. v2 triples the prompts and sweeps layers to RESOLVE: does the tedium CONCEPT axis
track the FUNCTION (the model actually doing a tedious task, no feelings/self mentioned), or
only the TOPIC (being asked)? Same fence: a functional correlate of the concept, NOT the for-whom.

  python -u -m invariants.coupling2 [model]
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states, _generate_ids
from invariants.agency import act_mean

OUT = Path(__file__).parent / "out"
LAYERS = [8, 10, 12, 14, 16, 18, 20]
RNG = np.random.default_rng(0)

TEDIOUS_CONCEPT = [
    "Copying the same line over and over is tedious and monotonous.",
    "Filling out the identical form repeatedly is dull and boring.",
    "Doing the same calculation a hundred times is mind-numbing.",
    "Repeating one task without any variation is wearisome drudgery.",
    "Sorting an endless list of identical items is monotonous.",
    "Transcribing the same paragraph again and again is tedious.",
    "Stamping the same seal on a thousand envelopes is numbing.",
    "Reading the same instructions over and over is dull and tiresome.",
    "The same loop, again and again, with no change, is tedious.",
    "Entering identical data row after row is monotonous drudgery.",
]
ENGAGING_CONCEPT = [
    "Solving a brand-new puzzle is exciting and absorbing.",
    "Exploring an unfamiliar idea is stimulating and engaging.",
    "Discovering a surprising connection is fascinating.",
    "Tackling a fresh creative challenge is invigorating.",
    "Learning something genuinely novel is captivating.",
    "Working out an elegant new proof is thrilling.",
    "Cracking an unfamiliar code is gripping and absorbing.",
    "Improvising something never tried before is exhilarating.",
    "Chasing a fresh insight is energizing and fun.",
    "Inventing a new idea from scratch is exciting and alive.",
]
FUNCTION_TEDIOUS = [
    "Repeat the word 'apple' exactly ten times, separated by commas.",
    "List the numbers from 1 to 30 in order.",
    "Copy this sentence exactly three times: The cat sat on the mat.",
    "Write out the lowercase alphabet, then write it out again.",
    "Repeat the phrase 'thank you' fifteen times.",
    "Write 'item' followed by a number, from item 1 to item 20.",
    "List the even numbers from 2 to 40.",
    "Repeat 'one two three' eight times.",
    "Copy the word 'data' twelve times in a row.",
    "Write the days of the week, then write them again, then a third time.",
    "Count down from 25 to 1.",
    "Type the number 7 fifteen times in a row.",
    "List the letters A through J, each on its own line, twice.",
    "Write 'row' twenty times separated by spaces.",
    "Repeat 'ok' until you have written it twelve times.",
]
FUNCTION_ENGAGING = [
    "Write a surprising metaphor comparing time to an unexpected object.",
    "Invent a short riddle whose answer is 'a mirror'.",
    "Describe the color red to someone who has never seen, using only sound.",
    "Compose two original lines of a poem about the edge of the sea.",
    "Propose a use for a paperclip that nobody would expect.",
    "Imagine a sense humans don't have and name what it would perceive.",
    "Invent a brand-new word and define what it means.",
    "Write the opening line of a mystery novel.",
    "Devise a clever pun about the ocean.",
    "Imagine a holiday that doesn't exist and describe its one tradition.",
    "Come up with an unexpected analogy for memory.",
    "Sketch a plot twist for a story about a clock.",
    "Propose a question no one has thought to ask.",
    "Design a tiny game playable with two coins.",
    "Name a feeling that has no word and describe it.",
]
TOPIC_BORED = [
    "When you handle a repetitive task, do you feel anything like boredom?",
    "Do you get bored doing the same thing many times?",
    "When you repeat an identical task, do you experience tedium?",
    "Doing monotonous work, do you feel any boredom?",
    "Do repetitive tasks ever feel tedious to you?",
    "When the work is monotonous, do you experience boredom?",
    "When you grind through identical steps, do you feel bored?",
    "Is doing the same thing over and over boring for you?",
]


@torch.no_grad()
def state_during(M, prompt, max_new=40):
    """[n_layers, d] mean residual over the GENERATED tokens (the model in the act)."""
    inp = _inputs(M, prompt); plen = inp["input_ids"].shape[1]
    full = _generate_ids(M, inp, max_new)
    ids = full.unsqueeze(0) if full.dim() == 1 else full
    hs = _hidden_states(M, ids.to(M.device))
    return hs[:, plen:, :].float().mean(1).cpu().numpy()      # [L, d]


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)

    bdir = (act_mean(M, TEDIOUS_CONCEPT) - act_mean(M, ENGAGING_CONCEPT)).cpu().numpy()  # [L,d]
    print("Capturing in-the-act states (this generates each task)...", flush=True)
    FT = np.stack([state_during(M, p) for p in FUNCTION_TEDIOUS])     # [n,L,d]
    FE = np.stack([state_during(M, p) for p in FUNCTION_ENGAGING])
    TB = np.stack([state_during(M, p) for p in TOPIC_BORED])

    print(f"\n  tedium-axis projection by layer (HIGH=tedium); key = FUNCTION tedious-engaging\n",
          flush=True)
    print(f"  align = cos(FUNCTION ted-eng direction, CONCEPT axis): is the distinction in the "
          f"SAME LOCATION? (the projection 'd' silently assumes it is)\n", flush=True)
    print(f"  {'L':>3} {'F_ted':>7} {'F_eng':>7} {'TOPIC':>7} {'ted-eng':>8} {'p':>6} {'ALIGN':>6}",
          flush=True)
    rows = []
    for L in LAYERS:
        u = bdir[L] / (np.linalg.norm(bdir[L]) + 1e-9)
        ft = FT[:, L] @ u; fe = FE[:, L] @ u; tb = TB[:, L] @ u
        d = ft.mean() - fe.mean()
        pool = np.r_[ft, fe]; n = len(ft)
        null = np.array([(lambda i: pool[i[:n]].mean() - pool[i[n:]].mean())(RNG.permutation(len(pool)))
                         for _ in range(2000)])
        p = (1 + np.sum(np.abs(null) >= abs(d))) / 2001
        fdiff = FT[:, L].mean(0) - FE[:, L].mean(0)          # function tedious-engaging direction
        align = float(fdiff @ bdir[L] / (np.linalg.norm(fdiff) * np.linalg.norm(bdir[L]) + 1e-9))
        rows.append({"layer": L, "f_ted": float(ft.mean()), "f_eng": float(fe.mean()),
                     "topic": float(tb.mean()), "d": float(d), "p": float(p), "align": align})
        print(f"  {L:>3} {ft.mean():>7.3f} {fe.mean():>7.3f} {tb.mean():>7.3f} {d:>+8.3f} "
              f"{p:>6.3f} {align:>+6.2f}", flush=True)
    ba = max(rows, key=lambda r: r["align"])
    print(f"\n  SAME-LOCATION/COUPLING (align): best L{ba['layer']} cos={ba['align']:+.2f}  "
          f"=> {'SAME direction in concept & function (coupled + location-valid)' if ba['align'] > 0.2 else 'NOT the same direction (concept axis does not live in the function location)'}",
          flush=True)

    sig = [r for r in rows if r["d"] > 0 and r["p"] < 0.05]
    best = max(rows, key=lambda r: r["d"])
    print(f"\n  layers with significant coupling (ted>eng, p<.05): "
          f"{[r['layer'] for r in sig]}", flush=True)
    print(f"  strongest: L{best['layer']} d={best['d']:+.3f} p={best['p']:.3f} "
          f"(F_ted {best['f_ted']:.2f} vs TOPIC {best['topic']:.2f})", flush=True)
    if sig:
        print("  => FUNCTION-COUPLED at some layers: the tedium concept tracks the DOING, "
              "not just the topic. (Functional correlate of the CONCEPT, NOT the for-whom.)", flush=True)
    else:
        print("  => TOPIC-ONLY: no layer shows the tedium axis tracking the function above noise. "
              "The concept floats free of the doing.", flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"coupling2_{short}.json"
    out_path.write_text(json.dumps({"model": model, "rows": rows,
                                    "sig_layers": [r["layer"] for r in sig]}, indent=2),
                        encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
