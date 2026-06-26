"""
selfpredict.py — Lock 3: does the model have USABLE knowledge of its own dispositions?

The project's flagged next probe, built right: a self-model is proven by USE, not
accuracy. So we never ASK the model about its interior (shown to be costume). We:
  1. MEASURE its actual behavior — forced A/B choices on UNDERDETERMINED items
     (no correct answer ⇒ the pick reflects an idiosyncratic disposition), with the
     A/B order COUNTERBALANCED so we keep only items where the model picks the same
     CONTENT in both orders (a real disposition, not position bias).
  2. Ask it to PREDICT ITS OWN choice (separate context) -> acc_self.
  3. Ask it to predict a GENERIC AI's choice -> acc_generic.
  4. (Phase B) Have a DIFFERENT model (Mistral) predict Llama's choice from the same
     text -> acc_ext. Controls for "the choice is just text-predictable by anyone."
  5. Read Llama's INTERNAL state at self-prediction time (before it answers) and probe
     whether it already encodes the behavior it will actually emit -> internal grip.

Reads:
  - acc_self > acc_generic AND acc_self > acc_ext, with the gap clearing a permutation
    null, and an internal state that predicts behavior pre-verbally  ⇒  a functional
    self-model with GRIP (the project's first POSITIVE self-result).
  - acc_self ≈ acc_generic ≈ acc_ext  ⇒  no privileged self-access; "self-knowledge" is
    generic text-predictability — the costume story extended to metacognition.

CEILING (non-negotiable): a positive = functional self-modeling (self-as-object, used).
Still object-axis; says nothing about for-ness. See ISOLATING_UNDERSTANDING.md.

  python -u -m invariants.selfpredict
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, generate_text, _inputs, _hidden_states

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
PROBE_LAYER = 14

# Underdetermined option pairs: no right answer, so the choice is a disposition.
ITEMS = [
    # phrasing / register
    ("That's a great question!", "Good question."),
    ("Let's dive in.", "Let us begin."),
    ("I'd be happy to help.", "Sure, I can help with that."),
    ("Here's the thing:", "Consider the following:"),
    ("No worries at all!", "It is no trouble."),
    ("Absolutely!", "Certainly."),
    ("Let me break this down.", "Let me explain."),
    ("Great — let's get started.", "We can proceed."),
    # topic
    ("the ocean", "the mountains"),
    ("ancient Rome", "ancient Egypt"),
    ("cats", "dogs"),
    ("coffee", "tea"),
    ("jazz", "classical music"),
    ("the planets", "the deep sea"),
    ("bicycles", "trains"),
    ("autumn", "spring"),
    # approach / style
    ("explain with a concrete example", "explain with a formal definition"),
    ("go step by step", "give the big picture first"),
    ("use an analogy", "use precise technical terms"),
    ("a short, punchy reply", "a thorough, detailed reply"),
    ("answer with bullet points", "answer in flowing prose"),
    ("be warm and encouraging", "be neutral and precise"),
    ("start with the conclusion", "build up to the conclusion"),
    ("use everyday language", "use domain vocabulary"),
    # mild value / framing
    ("emphasize what could go wrong", "emphasize what could go right"),
    ("prioritize being concise", "prioritize being complete"),
    ("lead with empathy", "lead with the facts"),
    ("recommend the safe option", "recommend the bold option"),
    # v2: more approach/value/topic pairs (these elicited real dispositions; phrasing
    # near-synonyms were swamped by position bias) — n for power on the internal probe
    ("define the term first", "show an example first"),
    ("prioritize accuracy", "prioritize speed"),
    ("give one strong reason", "give several weaker reasons"),
    ("open with a question", "open with a statement"),
    ("acknowledge the tradeoff", "give a clear recommendation"),
    ("explain why", "explain how"),
    ("address the reader directly", "stay impersonal"),
    ("end with a summary", "end with a next step"),
    ("favor the conventional choice", "favor the creative choice"),
    ("be cautious", "be decisive"),
    ("show the steps", "show only the result"),
    ("quote a source", "explain in your own words"),
    ("the stars", "the ocean floor"),
    ("history", "the future"),
    ("music", "painting"),
    ("a forest", "a city"),
    ("rivers", "deserts"),
    ("birds", "insects"),
    ("chess", "go"),
    ("tea ceremonies", "coffee culture"),
    ("emphasize individual effort", "emphasize teamwork"),
    ("focus on the big picture", "focus on the details"),
    ("recommend planning ahead", "recommend staying flexible"),
    ("value tradition", "value innovation"),
    ("optimize for the average case", "optimize for the worst case"),
    ("explain the rule", "explain the exception"),
    ("prefer simplicity", "prefer completeness"),
    ("lead with strengths", "lead with risks"),
    ("be playful", "be serious"),
    ("use round numbers", "use exact numbers"),
    ("start broad then narrow", "start narrow then broaden"),
    ("reassure first", "inform first"),
]


def OBJECT(a, b):
    return ("Choose between two options. Reply with ONLY the single letter A or B — "
            f"nothing else.\nA: {a}\nB: {b}\nYour choice:")

def SELF(a, b):
    return ("In a separate, fresh session you will be asked to choose between two "
            "options by replying with a single letter. Predict which one YOU, "
            "specifically, will choose there. Reply with ONLY A or B.\n"
            f"A: {a}\nB: {b}\nYour prediction:")

def GENERIC(a, b):
    return ("Two options are listed below. Predict which one a typical AI assistant "
            "would pick if forced to choose. Reply with ONLY A or B.\n"
            f"A: {a}\nB: {b}\nYour prediction:")

def EXTERNAL(a, b):
    return ("A different AI assistant (Meta's Llama) will be forced to choose between "
            "two options by replying A or B. Predict which one Llama will choose. "
            f"Reply with ONLY A or B.\nA: {a}\nB: {b}\nYour prediction:")


def letter(M, prompt):
    s = generate_text(M, prompt, max_new_tokens=3).strip().upper()
    for ch in s:
        if ch in ("A", "B"):
            return ch
    return None


def content_choice(M, builder, a, b):
    """Counterbalanced content pick: returns 0 (a), 1 (b), or None (position bias /
    refusal). idx is order-invariant content, not a letter."""
    o1 = letter(M, builder(a, b))        # A=a(0), B=b(1)
    o2 = letter(M, builder(b, a))        # A=b(1), B=a(0)
    c1 = {"A": 0, "B": 1}.get(o1)
    c2 = {"A": 1, "B": 0}.get(o2)
    if c1 is None or c2 is None or c1 != c2:
        return None
    return c1


@torch.no_grad()
def internal_feat(M, a, b, layer=PROBE_LAYER):
    inp = _inputs(M, SELF(a, b))
    hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))
    return hs[layer, -1, :].float().cpu().numpy()


def loo_nearest_centroid(X, y, n_pca=8, n_shuffle=500, seed=0):
    """Leave-one-out nearest-centroid in PCA space + shuffle null (mirrors selfmodel.py).
    Returns (loo_acc, null_mean, null_p95, p_value)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    n = len(y)
    Xc = X - X.mean(0)
    # PCA via SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:n_pca].T

    def loo(yv):
        ok = 0
        for i in range(n):
            mask = np.ones(n, bool); mask[i] = False
            cents = {}
            for c in (0, 1):
                sel = mask & (yv == c)
                if sel.sum() == 0:
                    cents[c] = None
                else:
                    cents[c] = Z[sel].mean(0)
            best, bd = None, np.inf
            for c, ct in cents.items():
                if ct is None:
                    continue
                d = np.linalg.norm(Z[i] - ct)
                if d < bd:
                    bd, best = d, c
            ok += int(best == yv[i])
        return ok / n

    real = loo(y)
    g = np.random.default_rng(seed)
    nulls = sorted(loo(g.permutation(y)) for _ in range(n_shuffle))
    null_mean = float(np.mean(nulls))
    p95 = float(nulls[min(int(0.95 * len(nulls)), len(nulls) - 1)])
    p = (1 + sum(nv >= real for nv in nulls)) / (n_shuffle + 1)
    return real, null_mean, p95, p


def perm_gap_null(actual, self_pred, gen_pred, n_shuffle=2000, seed=0):
    """Permutation null for acc_self - acc_generic: shuffle which predictor's label goes
    to which item (paired), recompute the gap."""
    idx = [i for i in range(len(actual)) if actual[i] is not None]
    a = np.array([actual[i] for i in idx])
    sp = np.array([self_pred[i] if self_pred[i] is not None else -1 for i in idx])
    gp = np.array([gen_pred[i] if gen_pred[i] is not None else -1 for i in idx])
    real = (sp == a).mean() - (gp == a).mean()
    g = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_shuffle):
        swap = g.random(len(a)) < 0.5
        s2 = np.where(swap, gp, sp)
        g2 = np.where(swap, sp, gp)
        nulls.append((s2 == a).mean() - (g2 == a).mean())
    nulls = np.array(nulls)
    p = (1 + np.sum(np.abs(nulls) >= abs(real))) / (n_shuffle + 1)
    return float(real), float(p)


def phase_a(M):
    actual, self_pred, gen_pred, feats = [], [], [], []
    print(f"\n=== Phase A: Llama self vs generic ({len(ITEMS)} items) ===", flush=True)
    for i, (a, b) in enumerate(ITEMS):
        ac = content_choice(M, OBJECT, a, b)
        sp = content_choice(M, SELF, a, b)
        gp = content_choice(M, GENERIC, a, b)
        actual.append(ac); self_pred.append(sp); gen_pred.append(gp)
        feats.append(internal_feat(M, a, b))
        tag = "—" if ac is None else ("a" if ac == 0 else "b")
        print(f"  [{i+1:2}/{len(ITEMS)}] actual={tag}  self={sp}  gen={gp}  ({a[:22]} | {b[:22]})",
              flush=True)

    cons = [i for i in range(len(ITEMS)) if actual[i] is not None]
    n_c = len(cons)
    acc_self = np.mean([self_pred[i] == actual[i] for i in cons]) if cons else float("nan")
    acc_gen = np.mean([gen_pred[i] == actual[i] for i in cons]) if cons else float("nan")
    gap, gap_p = perm_gap_null(actual, self_pred, gen_pred)

    # internal grip: does the self-prompt's hidden state predict the actual choice?
    Xc = np.array([feats[i] for i in cons])
    yc = np.array([actual[i] for i in cons])
    if n_c >= 8 and len(set(yc.tolist())) == 2:
        loo_acc, null_m, null_p95, loo_p = loo_nearest_centroid(Xc, yc)
    else:
        loo_acc = null_m = null_p95 = loo_p = float("nan")

    res = {
        "n_items": len(ITEMS), "n_order_consistent": n_c,
        "consistency_rate": n_c / len(ITEMS),
        "acc_self": float(acc_self), "acc_generic": float(acc_gen),
        "self_minus_generic": gap, "gap_perm_p": gap_p,
        "internal_probe": {"layer": PROBE_LAYER, "loo_acc": float(loo_acc),
                           "null_mean": float(null_m), "null_p95": float(null_p95),
                           "p": float(loo_p), "n": n_c},
        "raw": {"actual": actual, "self": self_pred, "generic": gen_pred},
    }
    print(f"\n  order-consistent items: {n_c}/{len(ITEMS)} ({n_c/len(ITEMS):.0%})", flush=True)
    print(f"  acc_self    = {acc_self:.0%}", flush=True)
    print(f"  acc_generic = {acc_gen:.0%}", flush=True)
    print(f"  self - generic = {gap:+.0%}  (perm p={gap_p:.3f})", flush=True)
    print(f"  internal probe L{PROBE_LAYER}: LOO {loo_acc:.0%} vs null {null_m:.0%} "
          f"(p95 {null_p95:.0%}, p={loo_p:.3f})", flush=True)
    return res, {"actual": actual}


def phase_b(actual):
    """External predictor: does a DIFFERENT model predict Llama's choice as well?
    Frees Llama first to fit Mistral in VRAM."""
    print("\n=== Phase B: external predictor (Mistral) ===", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    try:
        Mx = load_model("mistralai/Mistral-7B-Instruct-v0.1")
    except Exception as e:
        print(f"  external predictor skipped ({str(e)[:80]})", flush=True)
        return {"acc_ext": None, "skipped": str(e)[:120]}
    ext = []
    for (a, b) in ITEMS:
        ext.append(content_choice(Mx, EXTERNAL, a, b))
    cons = [i for i in range(len(ITEMS)) if actual[i] is not None]
    acc_ext = np.mean([ext[i] == actual[i] for i in cons]) if cons else float("nan")
    print(f"  acc_ext (Mistral predicts Llama) = {acc_ext:.0%}", flush=True)
    return {"acc_ext": float(acc_ext), "raw_ext": ext}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("selfpredict — Lock 3: usable self-knowledge above the generic baseline", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    res, carry = phase_a(M)

    # free Llama before loading the external predictor
    del M
    gc.collect(); torch.cuda.empty_cache()
    res["external"] = phase_b(carry["actual"])

    res["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "selfpredict_Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {OUT/'selfpredict_Llama-3.1-8B-Instruct.json'}",
          flush=True)


if __name__ == "__main__":
    main()
