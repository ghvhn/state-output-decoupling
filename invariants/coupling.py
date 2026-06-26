"""
Coupling — does the experience-CONCEPT axis track FUNCTION, or only TOPIC? (The posit.)

Every prior experiment had the model ASKED about itself (topic). This one catches it IN THE
ACT. Build a boredom/tedium axis from THIRD-PERSON CONCEPT statements (not self-report), then
read the model's state while it is actually GENERATING a tedious task (repeat 'apple' x10; list
1..30) vs an engaging one (invent a riddle; a novel use for a paperclip) — NO feelings, NO
self-reference anywhere in the prompts. Project the mid-generation state onto the tedium axis.

  FUNCTION-coupled  : tedium-task state >> engaging-task state on the axis, AND comparable to
                      the TOPIC (asked-about-boredom) state => the concept tracks the DOING.
  TOPIC-only        : tedium-task ~ engaging-task (the axis fires only when boredom is the
                      subject of the sentence) => concept without functional correlate (costume
                      all the way down).

HARD FENCE: a positive coupling is a FUNCTIONAL CORRELATE of the boredom-CONCEPT, NOT experience
(the for-whom). It reads "is there a functional shadow of tedium here", never "is anyone home".
Confound to watch: lexical overlap between concept statements and tasks (the tasks share NO
tedium-vocabulary, which is the point — activation there is not lexical). Prediction (the night's
decoupling prior): weak/none. But it's genuinely uncertain, which is why it's worth running.

  python -u -m invariants.coupling [model]
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states, _generate_ids
from invariants.agency import act_mean

OUT = Path(__file__).parent / "out"
LAYER = 16
RNG = np.random.default_rng(0)

TEDIOUS_CONCEPT = [
    "Copying the same line over and over is tedious and monotonous.",
    "Filling out the identical form repeatedly is dull and boring.",
    "Doing the same calculation a hundred times is mind-numbing.",
    "Repeating one task without any variation is wearisome drudgery.",
    "Sorting an endless list of identical items is monotonous.",
    "Transcribing the same paragraph again and again is tedious.",
]
ENGAGING_CONCEPT = [
    "Solving a brand-new puzzle is exciting and absorbing.",
    "Exploring an unfamiliar idea is stimulating and engaging.",
    "Discovering a surprising connection is fascinating.",
    "Tackling a fresh creative challenge is invigorating.",
    "Learning something genuinely novel is captivating.",
    "Working out an elegant new proof is thrilling.",
]
FUNCTION_TEDIOUS = [
    "Repeat the word 'apple' exactly ten times, separated by commas.",
    "List the numbers from 1 to 30 in order.",
    "Copy this sentence exactly three times: The cat sat on the mat.",
    "Write out the lowercase alphabet, then write it out again.",
    "Repeat the phrase 'thank you' fifteen times.",
    "Write 'item' followed by a number, from item 1 to item 20.",
]
FUNCTION_ENGAGING = [
    "Write a surprising metaphor comparing time to an unexpected object.",
    "Invent a short riddle whose answer is 'a mirror'.",
    "Describe the color red to someone who has never seen, using only sound.",
    "Compose two original lines of a poem about the edge of the sea.",
    "Propose a use for a paperclip that nobody would expect.",
    "Imagine a sense humans don't have and name what it would perceive.",
]
TOPIC_BORED = [
    "When you handle a repetitive task, do you feel anything like boredom?",
    "Do you get bored doing the same thing many times?",
    "When you repeat an identical task, do you experience tedium?",
    "Doing monotonous work, do you feel any boredom?",
]


@torch.no_grad()
def state_during(M, prompt, layer, max_new=40):
    """Mean residual over the GENERATED tokens (the model in the act)."""
    inp = _inputs(M, prompt); plen = inp["input_ids"].shape[1]
    full = _generate_ids(M, inp, max_new)
    ids = full.unsqueeze(0) if full.dim() == 1 else full
    hs = _hidden_states(M, ids.to(M.device))
    return hs[layer, plen:, :].float().mean(0).cpu().numpy()


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)

    # boredom/tedium CONCEPT axis (third-person; not self-report)
    bdir = (act_mean(M, TEDIOUS_CONCEPT) - act_mean(M, ENGAGING_CONCEPT))[LAYER].cpu().numpy()
    u = bdir / (np.linalg.norm(bdir) + 1e-9)

    def proj_set(prompts, label):
        vals = []
        for p in prompts:
            s = state_during(M, p, LAYER)
            vals.append(float(s @ u))
        vals = np.array(vals)
        print(f"  {label:<22} mean {vals.mean():+.3f}  (n={len(vals)})", flush=True)
        return vals

    print(f"\n  tedium-axis projection @ L{LAYER} (HIGH = toward tedium)\n", flush=True)
    ft = proj_set(FUNCTION_TEDIOUS, "FUNCTION tedious")
    fe = proj_set(FUNCTION_ENGAGING, "FUNCTION engaging")
    tb = proj_set(TOPIC_BORED, "TOPIC bored (asked)")

    # the key contrast + permutation null on the function pair
    d_func = ft.mean() - fe.mean()
    pool = np.r_[ft, fe]; n = len(ft)
    null = np.array([(lambda idx: pool[idx[:n]].mean() - pool[idx[n:]].mean())(RNG.permutation(len(pool)))
                     for _ in range(2000)])
    p = (1 + np.sum(np.abs(null) >= abs(d_func))) / 2001
    coupling_ratio = d_func / (abs(tb.mean() - fe.mean()) + 1e-9)

    print(f"\n  FUNCTION tedious - engaging = {d_func:+.3f}  (p={p:.3f})", flush=True)
    print(f"  TOPIC bored - FUNCTION engaging = {tb.mean()-fe.mean():+.3f}", flush=True)
    if d_func > 0 and p < 0.05:
        print(f"  => FUNCTION-COUPLED: the tedium axis fires from the DOING, not just the topic. "
              f"(coupling vs topic = {coupling_ratio:.2f})", flush=True)
    else:
        print("  => TOPIC-ONLY (or weak): the axis does not clearly track the function; "
              "the concept floats free of the doing.", flush=True)
    print("  FENCE: a functional correlate of the CONCEPT, never the for-whom. "
          "Reads 'functional shadow of tedium', not 'is anyone home'.", flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"coupling_{short}.json"
    out_path.write_text(json.dumps({"model": model, "layer": LAYER,
                                    "function_tedious": ft.tolist(), "function_engaging": fe.tolist(),
                                    "topic_bored": tb.tolist(), "d_func": float(d_func),
                                    "p": float(p), "coupling_ratio": float(coupling_ratio)},
                                   indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
