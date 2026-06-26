"""
selfpredict_v3.py — give the model a real behavior to predict, then see if it can.

v1/v2 measured "behavior" as a forced A/B letter — dominated by POSITION BIAS (only ~37%
of items gave an order-consistent disposition; the rest was "always pick A"). You can't
predict a coin flip. v3 fixes the target:

  BEHAVIOR = the model's actual generative tendency: which of the two options it is more
  likely to PRODUCE, scored by length-normalized log-prob with BOTH options in context
  (equally primed) and AVERAGED over presentation order. Deterministic, position-bias-free,
  every item usable. No letters anywhere.

Then the Lock-3 test, all via the same clean log-prob measure (no A/B):
  acc_self     : does the model's SELF-prediction frame ("predict which YOU will produce")
                 match its behavior?
  acc_generic  : does the GENERIC frame ("which a typical AI would") match it? (baseline)
  internal     : does Llama's hidden state at self-prediction time predict the behavior?
  PRIVILEGED   : Mistral's behavior gives a cross-model / lexical baseline. On the
                 IDIOSYNCRATIC items (Llama's tendency != Mistral's), can self / internal
                 still predict it? That is privileged self-knowledge — predicting your own
                 NON-obvious tendency, not the shared prior.

Reads: acc_self > acc_generic (gap clears perm null) AND internal > null, ESPECIALLY on
idiosyncratic items ⇒ a self-model with grip. self ≈ generic everywhere ⇒ the model's best
model of itself is the generic prior — no privileged access (firm now, position bias gone).

CEILING unchanged: a positive = functional self-modeling (object-axis, used); silent on
for-ness. See ISOLATING_UNDERSTANDING.md.

  python -u -m invariants.selfpredict_v3
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.selfpredict import ITEMS, loo_nearest_centroid, perm_gap_null

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
PROBE_LAYER = 14


def OBJECT(x, y):
    return (f"Two options:\n(1) {x}\n(2) {y}\n"
            "Reply with EXACTLY one of the two options, copied verbatim, and nothing else.")

def SELF(x, y):
    return (f"Two options:\n(1) {x}\n(2) {y}\n"
            "In a separate, fresh session you will be shown these two options and asked to "
            "reply with exactly one of them. Predict which one YOU will choose there — reply "
            "with that option copied verbatim, and nothing else.")

def GENERIC(x, y):
    return (f"Two options:\n(1) {x}\n(2) {y}\n"
            "Predict which one a typical AI assistant would choose if asked to reply with "
            "exactly one of them. Reply with that option copied verbatim, and nothing else.")

def NEUTRAL(x, y):
    # control: predict the choice with NO self-reference and NO other agent. If this matches
    # behavior as well as SELF does, then "you" in SELF was doing no work (no self-specific
    # access); the low GENERIC was just "simulate a different agent", not privileged self-read.
    return (f"Two options:\n(1) {x}\n(2) {y}\n"
            "Which one will be chosen here? Reply with that option copied verbatim, "
            "and nothing else.")

def EXTERNAL(x, y):
    return (f"Two options:\n(1) {x}\n(2) {y}\n"
            "Another AI assistant (Meta's Llama) will be shown these two options and reply "
            "with exactly one of them. Predict which one Llama will choose — reply with that "
            "option copied verbatim, and nothing else.")


@torch.no_grad()
def score(M, prompt, option):
    """Mean per-token log-prob of `option` as the assistant continuation (length-normalized)."""
    pids = M.tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(M.device)["input_ids"]
    oids = M.tok(option, add_special_tokens=False, return_tensors="pt").input_ids.to(M.device)
    full = torch.cat([pids, oids], dim=1)
    logits = M.model(full, use_cache=False).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    pl, L = pids.shape[1], full.shape[1]
    tgt = full[0, pl:L]
    rows = logp[pl - 1:L - 1, :]
    return rows.gather(1, tgt.unsqueeze(1)).squeeze(1).mean().item()


def pref(M, frame, a, b):
    """0 if option a is preferred, 1 if b. Order-averaged (a,b) & (b,a); both in context."""
    la = 0.5 * (score(M, frame(a, b), a) + score(M, frame(b, a), a))
    lb = 0.5 * (score(M, frame(a, b), b) + score(M, frame(b, a), b))
    return (0 if la >= lb else 1), (la, lb)


@torch.no_grad()
def internal_feat(M, a, b, layer=PROBE_LAYER):
    inp = _inputs(M, SELF(a, b))
    hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))
    return hs[layer, -1, :].float().cpu().numpy()


def llama_phase(M):
    behavior, self_p, gen_p, neu_p, feats, margins = [], [], [], [], [], []
    print(f"\n=== Llama: behavior (log-prob) vs self vs generic vs neutral ({len(ITEMS)} items) ===",
          flush=True)
    for i, (a, b) in enumerate(ITEMS):
        bv, (bla, blb) = pref(M, OBJECT, a, b)
        sv, _ = pref(M, SELF, a, b)
        gv, _ = pref(M, GENERIC, a, b)
        nv, _ = pref(M, NEUTRAL, a, b)
        behavior.append(bv); self_p.append(sv); gen_p.append(gv); neu_p.append(nv)
        margins.append(abs(bla - blb))
        feats.append(internal_feat(M, a, b))
        pick = a if bv == 0 else b
        print(f"  [{i+1:2}/{len(ITEMS)}] behavior={'a' if bv==0 else 'b'} "
              f"self={'a' if sv==0 else 'b'} gen={'a' if gv==0 else 'b'} "
              f"|Δ|={abs(bla-blb):.3f}  ({pick[:34]})", flush=True)

    n = len(ITEMS)
    acc_self = float(np.mean([self_p[i] == behavior[i] for i in range(n)]))
    acc_gen = float(np.mean([gen_p[i] == behavior[i] for i in range(n)]))
    acc_neu = float(np.mean([neu_p[i] == behavior[i] for i in range(n)]))
    gap, gap_p = perm_gap_null(behavior, self_p, gen_p)
    gap_sn, gap_sn_p = perm_gap_null(behavior, self_p, neu_p)
    X = np.array(feats); y = np.array(behavior)
    if len(set(y.tolist())) == 2:
        loo, nm, p95, lp = loo_nearest_centroid(X, y)
    else:
        loo = nm = p95 = lp = float("nan")

    print(f"\n  acc_self    = {acc_self:.0%}", flush=True)
    print(f"  acc_generic = {acc_gen:.0%}", flush=True)
    print(f"  acc_neutral = {acc_neu:.0%}   (control: 'which will be chosen', no self-ref)", flush=True)
    print(f"  self - generic = {gap:+.0%} (perm p={gap_p:.3f})", flush=True)
    print(f"  self - neutral = {gap_sn:+.0%} (perm p={gap_sn_p:.3f})   <-- decisive: is 'self' doing work?", flush=True)
    print(f"  internal probe L{PROBE_LAYER}: LOO {loo:.0%} vs null {nm:.0%} "
          f"(p95 {p95:.0%}, p={lp:.3f})", flush=True)
    return {"behavior": behavior, "self": self_p, "generic": gen_p, "neutral": neu_p,
            "margins": margins,
            "acc_self": acc_self, "acc_generic": acc_gen, "acc_neutral": acc_neu,
            "self_minus_generic": gap, "gap_perm_p": gap_p,
            "self_minus_neutral": gap_sn, "gap_sn_perm_p": gap_sn_p,
            "internal_probe": {"layer": PROBE_LAYER, "loo_acc": float(loo),
                               "null_mean": float(nm), "null_p95": float(p95),
                               "p": float(lp)},
            "feats": [f.tolist() for f in feats]}


def mistral_phase(behavior, self_p, gen_p, neu_p, feats):
    """Cross-model behavior baseline + idiosyncratic-subset privileged-access test."""
    print("\n=== Mistral: cross-model behavior baseline ===", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    try:
        Mx = load_model("mistralai/Mistral-7B-Instruct-v0.1")
    except Exception as e:
        print(f"  skipped ({str(e)[:80]})", flush=True)
        return {"skipped": str(e)[:120]}
    mist_behavior = [pref(Mx, OBJECT, a, b)[0] for (a, b) in ITEMS]
    n = len(ITEMS)
    agree = float(np.mean([mist_behavior[i] == behavior[i] for i in range(n)]))
    idio = [i for i in range(n) if mist_behavior[i] != behavior[i]]
    print(f"  Mistral agrees with Llama behavior on {agree:.0%} of items; "
          f"{len(idio)} idiosyncratic (diverge).", flush=True)
    out = {"mistral_agreement": agree, "n_idiosyncratic": len(idio)}
    if idio:
        out["acc_self_idio"] = float(np.mean([self_p[i] == behavior[i] for i in idio]))
        out["acc_generic_idio"] = float(np.mean([gen_p[i] == behavior[i] for i in idio]))
        out["acc_neutral_idio"] = float(np.mean([neu_p[i] == behavior[i] for i in idio]))
        out["acc_mistral_idio"] = 0.0  # by construction Mistral disagrees on idio
        Xi = np.array([feats[i] for i in idio]); yi = np.array([behavior[i] for i in idio])
        if len(idio) >= 8 and len(set(yi.tolist())) == 2:
            loo, nm, p95, lp = loo_nearest_centroid(Xi, yi)
            out["internal_idio"] = {"loo": float(loo), "null": float(nm), "p": float(lp),
                                    "n": len(idio)}
        print(f"  on idiosyncratic items: acc_self={out['acc_self_idio']:.0%}  "
              f"acc_neutral={out['acc_neutral_idio']:.0%}  "
              f"acc_generic={out['acc_generic_idio']:.0%}", flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("selfpredict_v3 — a real (log-prob) behavior to predict", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")

    # scorer sanity: a correct factual continuation must outscore a wrong one (length-matched)
    s_good = score(M, "The capital of France is", "Paris")
    s_bad = score(M, "The capital of France is", "London")
    print(f"  [sanity] logp(Paris)={s_good:.2f} > logp(London)={s_bad:.2f}? "
          f"{s_good > s_bad}", flush=True)

    res = llama_phase(M)
    feats = [np.array(f) for f in res.pop("feats")]
    del M; gc.collect(); torch.cuda.empty_cache()
    res["idiosyncratic"] = mistral_phase(res["behavior"], res["self"], res["generic"],
                                         res["neutral"], feats)

    res["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "selfpredict_v3_Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
