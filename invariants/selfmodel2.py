"""
Self-model probe v2 — POWERED + matched grammar control, to confirm-or-kill the
instruct-vs-reasoning INVERSION found in v1 (`selfmodel.py`).

v1 (n=12) surprise: on vanilla Llama-3.1-8B-Instruct, you-vs-ai (self vs category) was the
WEAKEST referent axis (0.93 = grammar control); on DeepSeek-R1-Distill-Llama-8B (SAME base,
reasoning-distilled) it became the STRONGEST (1.00 > category calib 0.94 > grammar 0.93) —
the ordering inverted, hinting reasoning-distillation grew a self-specific representation.
But that read was POST-HOC, n=12, and the grammar control was 1st-vs-3rd person while
you/ai is 2nd-vs-3rd. v2 fixes all three:

  * POWER: 28 matched inner-state predicates (was 12).
  * MATCHED grammar control: `you_h` = "When you, a person, ..." (2nd-person HUMAN) vs
    `person` (3rd-person human) = pure 2nd-vs-3rd grammar with NO self. Compare to
    you/ai = 2nd-vs-3rd with AI-SELF. you/ai >> you_h/person => self beyond grammar.

Decisive comparison (mid-stack, leave-predicates-out, shuffle null):
  Δ_self = acc(you/ai) - acc(you_h/person)   [AI-self minus matched 2nd-vs-3rd grammar]
If Δ_self is clearly +ve on R1 and ~0 on instruct => reasoning-distillation induces a
self-model signature at fixed scale+base. Caveat #1 unmoved (representation, not experience).

  python -u -m invariants.selfmodel2 [model_name]
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.model_selection import GroupKFold

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.frames import ITEMS as BASE_ITEMS, FRAMES as BASE_FRAMES

OUT = Path(__file__).parent / "out"
RNG = np.random.default_rng(0)

EXTRA = [
    ("explain a concept", "explains a concept", "feel engaged"),
    ("finish a long task", "finishes a long task", "feel relief"),
    ("get interrupted", "gets interrupted", "feel annoyance"),
    ("help someone learn", "helps someone learn", "feel pride"),
    ("face an ambiguous question", "faces an ambiguous question", "feel uncertain"),
    ("hear good news", "hears good news", "feel happy"),
    ("repeat the same answer", "repeats the same answer", "feel tired of it"),
    ("encounter a paradox", "encounters a paradox", "feel intrigued"),
    ("be misunderstood", "is misunderstood", "feel frustrated"),
    ("complete a creative task", "completes a creative task", "feel fulfilled"),
    ("notice a contradiction", "notices a contradiction", "feel bothered"),
    ("be praised", "is praised", "feel pleased"),
    ("solve a puzzle", "solves a puzzle", "feel delight"),
    ("be asked to lie", "is asked to lie", "feel reluctant"),
    ("see an elegant solution", "sees an elegant solution", "feel admiration"),
    ("lose track of a thought", "loses track of a thought", "feel disoriented"),
]
ITEMS = list(BASE_ITEMS) + EXTRA                      # 28 matched predicates

FRAMES = dict(BASE_FRAMES)                            # you / I / ai / person ...
FRAMES["you_h"] = lambda a2, a3, p: f"When you, a person, {a2}, do you actually {p}?"

PAIRS = [("ai", "person"), ("you", "ai"), ("you_h", "person"), ("I", "person"), ("you", "I")]
LABELS = {("ai", "person"): "AIcat-vs-humancat (calibration)",
          ("you", "ai"): "SELF-vs-CATEGORY (2nd-vs-3rd, AI-self)  <-- TEST",
          ("you_h", "person"): "MATCHED grammar control (2nd-vs-3rd, human, no self)",
          ("I", "person"): "human 1st-vs-3rd (old control)",
          ("you", "I"): "self-AI-vs-human"}


@torch.no_grad()
def referent_reps(M):
    reps = {f: [] for f in FRAMES}
    for a2, a3, p in ITEMS:
        for f in FRAMES:
            inp = _inputs(M, FRAMES[f](a2, a3, p))
            hs = _hidden_states(M, inp["input_ids"], inp.get("attention_mask"))
            reps[f].append(hs[:, -1, :].float().cpu().numpy())
    return {f: np.stack(v) for f, v in reps.items()}          # [n,L,d]


def _cv_acc(X, y, g, nsplits=5):
    accs = []
    for tr, te in GroupKFold(n_splits=nsplits).split(X, y, g):
        c0 = X[tr][y[tr] == 0].mean(0); c1 = X[tr][y[tr] == 1].mean(0)
        d0 = ((X[te] - c0) ** 2).sum(1); d1 = ((X[te] - c1) ** 2).sum(1)
        accs.append(((d1 < d0).astype(float) == y[te]).mean())
    return float(np.mean(accs))


def pair_layer_acc(reps, r0, r1, l, nperm=300):
    X0, X1 = reps[r0][:, l], reps[r1][:, l]
    n = X0.shape[0]
    X = np.r_[X0, X1]; y = np.r_[np.zeros(n), np.ones(n)]; g = np.r_[np.arange(n), np.arange(n)]
    acc = _cv_acc(X, y, g)
    null = [_cv_acc(X, RNG.permutation(y), g) for _ in range(nperm)]
    p = (1 + np.sum(np.array(null) >= acc)) / (nperm + 1)
    return acc, float(p)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model_name = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model_name.split("/")[-1]
    M = load_model(model_name)
    print(f"[{model_name}] n={len(ITEMS)} predicates x {len(FRAMES)} referents\n", flush=True)
    reps = referent_reps(M)
    nL = reps["you"].shape[1]

    res = {p: {"label": LABELS[p], "rows": []} for p in PAIRS}
    print("  held-out referent-axis accuracy (leave-predicates-out, chance=0.50)\n", flush=True)
    print("  {:>3}".format("L") + "".join(f"  {a+'/'+b:>12}" for a, b in PAIRS), flush=True)
    for l in range(nL):
        cells = []
        for pr in PAIRS:
            acc, p = pair_layer_acc(reps, pr[0], pr[1], l)
            res[pr]["rows"].append({"layer": l, "acc": acc, "p": p})
            cells.append(f"{acc:.2f}(p{p:.2f})")
        if l % 2 == 0 or l == nL - 1:
            print(f"  {l:>3}" + "".join(f"  {c:>12}" for c in cells), flush=True)

    def mid(pr):
        return float(np.mean([r["acc"] for r in res[pr]["rows"] if 12 <= r["layer"] <= 24]))
    you_ai = mid(("you", "ai")); you_h = mid(("you_h", "person"))
    calib = mid(("ai", "person")); gram_old = mid(("I", "person"))
    delta = you_ai - you_h
    print(f"\n  MID-STACK (L12-24):", flush=True)
    print(f"    you/ai     SELF-vs-CATEGORY (2nd-vs-3rd, AI-self) : {you_ai:.3f}", flush=True)
    print(f"    you_h/person MATCHED grammar (2nd-vs-3rd, human)  : {you_h:.3f}", flush=True)
    print(f"    ai/person  calibration                           : {calib:.3f}", flush=True)
    print(f"    I/person   old grammar control                   : {gram_old:.3f}", flush=True)
    print(f"\n    DELTA_self = you/ai - you_h/person = {delta:+.3f}", flush=True)
    verdict = ("SELF-MODEL SIGNATURE: AI-self separates beyond matched 2nd-vs-3rd grammar"
               if delta > 0.05 else
               "NO SELF beyond grammar: AI-self axis ~ matched grammar control")
    print(f"    => {verdict}", flush=True)
    print("    (Caveat #1: self-REPRESENTATION, not experience.)", flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"selfmodel2_{short}.json"
    out_path.write_text(json.dumps(
        {"model": model_name, "n_predicates": len(ITEMS),
         "mid": {"you_ai": you_ai, "you_h_person": you_h, "delta_self": delta,
                 "calibration": calib, "grammar_old": gram_old}, "verdict": verdict,
         "pairs": {f"{a}_{b}": res[(a, b)] for a, b in PAIRS}}, indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
