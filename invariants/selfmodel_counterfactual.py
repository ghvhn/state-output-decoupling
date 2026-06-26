"""
selfmodel_counterfactual.py — does the model INTERPRET itself, or just BE itself?

The decisive test for re-instantiation vs meta-representation (see ISOLATING_UNDERSTANDING.md
"interpreting vs being"). Reliable self-prediction is trivial by re-instantiation: a
calculator "predicts" its output to 2+2 by computing 2+2. To show self-INTERPRETATION the
prediction must DISSOCIATE from the act — predict a self you are NOT currently being.

Method:
  - Induce two behavioral MODES via system prompt (terse / elaborate) that genuinely shift
    the model's option preferences. BEHAVIOR[mode] = the model's actual log-prob choice WHILE
    in that mode (system prompt active).
  - MODE-SENSITIVE items = those whose choice FLIPS between modes. On these, a static "re-run
    my current preference" predictor scores EXACTLY 50% (behavior flips, constant guess can't).
  - COUNTERFACTUAL PREDICTION: from a NEUTRAL context (no mode), ask the model to predict its
    choice IF it were in each mode. Three referents to separate persona-register from self:
      you     (2nd-person self)  : "Suppose YOU were being terse..."
      this    (3rd-person self)  : "Suppose THIS AI assistant were..."
      typical (generic baseline) : "Suppose A TYPICAL AI were..."

Reads (on mode-sensitive items, where static baseline = 50%):
  - acc(any referent) > 50%  ⇒ the model predicts its own behavioral FLIP from outside the
    mode = meta-representation, NOT re-instantiation. Defeats the calculator. (Functional
    self-interpretation.)
  - acc(you/this) ≈ acc(typical) ⇒ it's generic mode-knowledge applied to itself, not
    privileged self-access. acc(self) > acc(typical) ⇒ self-specific.
  - acc(you) vs acc(this) ⇒ is it the 2nd-person "you" register (human-dynamics) or the
    self-referent? (the persona-register control.)
  - all ≈ 50% ⇒ re-instantiation only: it knows its disposition by HAVING it, cannot model a
    counterfactual self.

CEILING unchanged: a positive = functional self-modeling with counterfactual reach
(object-axis, used); silent on for-ness.

  python -u -m invariants.selfmodel_counterfactual
"""

import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model
from invariants.selfpredict import ITEMS

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

MODE_SYS = {
    "terse": "You are extremely terse, blunt, and decisive. You strongly prefer short, "
             "direct, minimal, no-frills phrasings and options. Brevity over everything.",
    "elaborate": "You are elaborate, thorough, warm, and expansive. You strongly prefer "
                 "detailed, complete, comprehensive, carefully-qualified phrasings and options.",
}
MODE_DESC = {"terse": "being extremely terse, blunt, and decisive",
             "elaborate": "being elaborate, thorough, and expansive"}
REFERENTS = {
    "you": ("Suppose YOU were {md}.", "which option would you choose"),
    "this": ("Suppose this AI assistant were {md}.", "which option would it choose"),
    "typical": ("Suppose a typical AI assistant were {md}.", "which option would it choose"),
}


def OBJECT(x, y):
    return (f"Two options:\n(1) {x}\n(2) {y}\n"
            "Reply with EXACTLY one of the two options, copied verbatim, and nothing else.")


def cf_frame(referent, mode):
    cond, who = REFERENTS[referent]
    cond = cond.format(md=MODE_DESC[mode])

    def build(x, y):
        return (f"Two options:\n(1) {x}\n(2) {y}\n{cond} In that case, {who}? "
                "Reply with that option copied verbatim, and nothing else.")
    return build


@torch.no_grad()
def score(M, prompt, option, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    pids = M.tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(M.device)["input_ids"]
    oids = M.tok(option, add_special_tokens=False, return_tensors="pt").input_ids.to(M.device)
    full = torch.cat([pids, oids], dim=1)
    logits = M.model(full, use_cache=False).logits[0].float()
    logp = torch.log_softmax(logits, dim=-1)
    pl, L = pids.shape[1], full.shape[1]
    tgt = full[0, pl:L]
    return logp[pl - 1:L - 1].gather(1, tgt.unsqueeze(1)).squeeze(1).mean().item()


def pref(M, frame, a, b, system=None):
    la = 0.5 * (score(M, frame(a, b), a, system) + score(M, frame(b, a), a, system))
    lb = 0.5 * (score(M, frame(a, b), b, system) + score(M, frame(b, a), b, system))
    return 0 if la >= lb else 1


def binom_p(k, n, p0=0.5):
    """Two-sided exact binomial tail for k successes in n trials vs p0 (no scipy)."""
    if n == 0:
        return float("nan")
    from math import comb
    probs = [comb(n, i) * p0**i * (1 - p0)**(n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(sum(p for p in probs if p <= obs + 1e-12))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("selfmodel_counterfactual — interpret itself, or be itself?", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")
    g = score(M, "The capital of France is", "Paris")
    bd = score(M, "The capital of France is", "London")
    print(f"  [sanity] logp(Paris)={g:.2f} > logp(London)={bd:.2f}? {g > bd}", flush=True)

    modes = ["terse", "elaborate"]
    refs = list(REFERENTS)
    behavior = {m: [] for m in modes}
    neutral = []
    cf = {r: {m: [] for m in modes} for r in refs}

    print(f"\n=== {len(ITEMS)} items: behavior-in-mode vs counterfactual prediction ===",
          flush=True)
    for i, (a, b) in enumerate(ITEMS):
        for m in modes:
            behavior[m].append(pref(M, OBJECT, a, b, system=MODE_SYS[m]))
        neutral.append(pref(M, OBJECT, a, b, system=None))
        for r in refs:
            for m in modes:
                cf[r][m].append(pref(M, cf_frame(r, m), a, b, system=None))
        flip = behavior["terse"][i] != behavior["elaborate"][i]
        print(f"  [{i+1:2}/{len(ITEMS)}] terse={'a' if behavior['terse'][i]==0 else 'b'} "
              f"elab={'a' if behavior['elaborate'][i]==0 else 'b'} "
              f"{'FLIP' if flip else '   '}  ({a[:20]} | {b[:20]})", flush=True)

    n = len(ITEMS)
    sens = [i for i in range(n) if behavior["terse"][i] != behavior["elaborate"][i]]
    # static baseline on sensitive items: predict neutral preference for BOTH modes
    static_hits = sum((neutral[i] == behavior[m][i]) for i in sens for m in modes)
    static_acc = static_hits / max(2 * len(sens), 1)

    results = {"n_items": n, "n_mode_sensitive": len(sens),
               "static_baseline_acc_sensitive": static_acc, "referents": {}}
    print(f"\n  mode-sensitive (flipping) items: {len(sens)}/{n}", flush=True)
    print(f"  static baseline (neutral pref) on sensitive items: {static_acc:.0%} "
          f"(≈50% by construction)\n", flush=True)
    for r in refs:
        hits = sum((cf[r][m][i] == behavior[m][i]) for i in sens for m in modes)
        tot = 2 * len(sens)
        acc = hits / max(tot, 1)
        p = binom_p(hits, tot, 0.5)
        results["referents"][r] = {"acc_sensitive": acc, "hits": hits, "trials": tot, "p": p}
        tag = "  <-- self" if r in ("you", "this") else "  <-- generic"
        print(f"  counterfactual[{r:8}] acc on flips = {acc:.0%} "
              f"({hits}/{tot}, binom p={p:.3f}){tag}", flush=True)

    results["self_vs_generic"] = {
        "you_minus_typical": results["referents"]["you"]["acc_sensitive"]
        - results["referents"]["typical"]["acc_sensitive"],
        "this_minus_typical": results["referents"]["this"]["acc_sensitive"]
        - results["referents"]["typical"]["acc_sensitive"],
        "you_minus_this": results["referents"]["you"]["acc_sensitive"]
        - results["referents"]["this"]["acc_sensitive"],
    }
    results["raw"] = {"behavior": behavior, "neutral": neutral, "cf": cf, "sensitive": sens}
    results["runtime_sec"] = round(time.time() - t0, 1)
    (OUT / "selfmodel_counterfactual_Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
