"""
The word-making engine — legibility for the next steps (agency's "decode the controller",
the coupling test, any found direction). Resolves the predictor-native-vs-legible tension:
when a probe/intervention finds a DIRECTION the model computes WITH, name it without (a)
garbage from the mid-stack unembedding (the failed-lens wall) or (b) forcing a wrong existing
word. Two stages, both tagged as OUR labels (the triad's "our understanding" slot) — never a
claim the model "has" the word:

  1. characterize()  — DATA-GROUNDED, robust, NO lens: rank prompts by projection onto the
     direction; the meaning IS the contrast between the high and low extremes (real examples,
     so it dodges mid-stack illegibility). This is the honest grounding; trust this.
  2. coin()  — GENERATIVE, FLAGGED: given that contrast, emit a label. If an existing English
     word fits, use it; if it's a between-state with NO faithful word, COMPOSE a neologism from
     roots + a one-line gloss. A naming aid through the verbal channel (so: costume-adjacent,
     legible, NOT a faithful decode). Always reported next to the grounding, never alone.

  from invariants.legible import characterize, coin, legible_name
"""

import torch

from invariants.engine import _inputs, _hidden_states, generate_text


@torch.no_grad()
def _proj(M, vecs, prompt, layer):
    """Projection of a prompt's answer-position residual onto unit(vecs[layer])."""
    inp = _inputs(M, prompt)
    hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))   # [L,seq,d]
    h = hs[layer, -1, :].float()
    u = vecs[layer].float().to(h.device)
    u = u / (u.norm() + 1e-9)
    return float(h @ u)


@torch.no_grad()
def characterize(M, vecs, prompts, layer, k=4):
    """DATA-GROUNDED: rank prompts by projection onto the direction at `layer`.
    Returns (hi, lo) — the k highest- and lowest-scoring prompts. The contrast is the
    meaning; report these examples verbatim, they are the honest part."""
    scored = sorted(((_proj(M, vecs, p, layer), p) for p in prompts), reverse=True)
    hi = [p for _, p in scored[:k]]
    lo = [p for _, p in scored[-k:]]
    return hi, lo, scored


@torch.no_grad()
def coin(M, hi, lo):
    """GENERATIVE (flagged): name the dimension separating hi from lo. Existing word if one
    fits; else a root-composed neologism + gloss. Returns raw text — a label, not a decode."""
    prompt = (
        "Two groups of prompts differ along one hidden dimension.\n\n"
        "GROUP A (high):\n- " + "\n- ".join(hi) + "\n\n"
        "GROUP B (low):\n- " + "\n- ".join(lo) + "\n\n"
        "Name the dimension in ONE or TWO words (an existing word if one fits, else a word "
        "composed from Greek/Latin roots). Reply with EXACTLY two lines, nothing else, no "
        "restating and no explanation:\nWORD: <one or two words>\nGLOSS: <one short sentence>"
    )
    raw = generate_text(M, prompt, max_new_tokens=40).strip()
    word = gloss = None
    for line in raw.splitlines():
        s = line.strip()
        if word is None and s.upper().startswith("WORD:"):
            word = s[5:].strip()
        elif gloss is None and s.upper().startswith("GLOSS:"):
            gloss = s[6:].strip()
    if word:
        return f"{word}" + (f" — {gloss}" if gloss else "")
    return raw.split("\n")[0][:80]   # fallback: first line only


@torch.no_grad()
def legible_name(M, vecs, prompts, layer, k=4, verbose=True):
    """Full pass: ground (characterize) then label (coin), keeping them adjacent so the
    generative label never travels without its data grounding."""
    hi, lo, scored = characterize(M, vecs, prompts, layer, k)
    label = coin(M, hi, lo)
    if verbose:
        print(f"  [legible @ L{layer}]  grounding (trust this):", flush=True)
        print("    HIGH:\n      - " + "\n      - ".join(hi), flush=True)
        print("    LOW:\n      - " + "\n      - ".join(lo), flush=True)
        print(f"  coined label (generative, flagged — our understanding, not the model's word):",
              flush=True)
        print("    " + label.replace("\n", "\n    "), flush=True)
    return {"layer": layer, "hi": hi, "lo": lo, "label": label,
            "scores": [(round(s, 3), p) for s, p in scored]}


if __name__ == "__main__":
    print(__doc__)
