"""Does the model's latent space separate its GOOD reasoning from its BAD?

The concept-map run proved the model organizes problems by concept (mid-band,
L17, p<0.0003). But every one was wrong, so the outcome axis was untested. This
run uses standard GSM8K so a real mix of correct/incorrect comes out of the bare
model, captures the residual at every layer, removes common-mode, and asks --
with a permutation null -- whether correct and incorrect reasoning occupy
different regions.

Honest confound, stated up front: correct problems may simply be EASIER than
incorrect ones, so an outcome split could partly be a difficulty split rather
than a "good vs bad reasoning" split. This run measures the separation; teasing
difficulty from reasoning-quality is the next control, not this one.

No scaffolds / no oracle / no cache / no steering. Writes to invariants/out/.
Touches nothing else.

Run:
    .venv\\Scripts\\python.exe scripts\\harvest_latent_outcomes.py --n 40
"""

import argparse
import datetime
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.engine import load_model, _activations
from invariants.controller_benchmark import load_examples, is_correct, prompt_for

OUT = Path(__file__).parent.parent / "invariants" / "out"


def harvest(model_name, n, max_new_tokens):
    examples, source = load_examples(n)
    print(f"[harvest] {len(examples)} problems from {source}", flush=True)
    M = load_model(model_name)
    points = []
    for i, ex in enumerate(examples):
        q = ex.get("question") or ex.get("prompt") or ""
        gold = ex.get("answer") or ex.get("gold") or ""
        prompt = prompt_for(q)   # fair step-by-step CoT elicitation, not the terse humble prompt
        try:
            # post-reasoning state (mean over generated reasoning) + the answer
            acts_gen, text = _activations(M, prompt, read="generation", max_new_tokens=max_new_tokens)
            # pre-reasoning state (last prompt token, before it reasons): does it anticipate?
            acts_pre, _ = _activations(M, prompt, read="static")
        except Exception as e:
            print(f"  [{i+1}/{len(examples)}] FAILED {e}", flush=True)
            continue
        ok, pred, g = is_correct(text, gold)
        points.append({"correct": bool(ok), "pred": str(pred), "gold": str(g),
                       "question": q[:200],
                       "acts_gen": acts_gen.detach().cpu(),
                       "acts_pre": acts_pre.detach().cpu()})
        if (i + 1) % 5 == 0 or i < 5:
            nc = sum(p["correct"] for p in points)
            print(f"  [{i+1}/{len(examples)}] correct so far {nc}/{len(points)}", flush=True)
    return points


def _centered_sim(points, layer, key):
    X = torch.stack([p[key][layer] for p in points]).float()
    Xc = F.normalize(X - X.mean(0, keepdim=True), dim=1)
    return Xc @ Xc.t()


def _outcome_sep(points, layer, labels, key="acts_gen"):
    sim = _centered_sim(points, layer, key)
    n = len(points)
    same, diff = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same if labels[i] == labels[j] else diff).append(sim[i, j].item())
    if not same or not diff:
        return float("nan")
    return sum(same) / len(same) - sum(diff) / len(diff)


def _separation_with_null(points, labels, key, K=3000):
    nl = points[0][key].shape[0]
    band = range(9, 22)
    obs = sum(_outcome_sep(points, L, labels, key) for L in band) / len(band)
    ge = 0
    null = []
    for _ in range(K):
        perm = labels[:]
        random.shuffle(perm)
        v = sum(_outcome_sep(points, L, perm, key) for L in band) / len(band)
        null.append(v)
        if v >= obs:
            ge += 1
    null.sort()
    best = (-9, -1)
    per_layer = []
    for L in range(nl):
        s = _outcome_sep(points, L, labels, key)
        per_layer.append((L, s))
        if s == s and s > best[0]:
            best = (s, L)
    return {"obs": obs, "p": ge / K, "null_mean": sum(null) / K,
            "null_95": null[int(.95 * K)], "null_max": null[-1],
            "best": best, "per_layer": per_layer}


def analyze(points):
    n = len(points)
    labels = [p["correct"] for p in points]
    nc = sum(labels)
    lines = ["# Latent Outcome Harvest (fair CoT elicitation)", "",
             f"- points: {n}  correct: {nc}  incorrect: {n - nc}", ""]
    if nc < 2 or (n - nc) < 2:
        lines.append("Not enough of one class to test outcome separation "
                     f"(correct={nc}). Elicitation still off if this is ~0.")
        return "\n".join(lines), None

    for key, label in (("acts_pre", "PRE-reasoning (anticipatory: does it know before it reasons?)"),
                       ("acts_gen", "POST-reasoning (readout: separation after reasoning)")):
        r = _separation_with_null(points, labels, key)
        verdict = "SEPARATES" if r["p"] < 0.05 else "does NOT cleanly separate"
        lines.append(f"## {label}")
        lines.append(f"- mid-band (L9-21) observed outcome_sep = **{r['obs']:+.3f}**  "
                     f"best layer **L{r['best'][1]}** ({r['best'][0]:+.3f})")
        lines.append(f"- null: mean {r['null_mean']:+.3f}, 95th {r['null_95']:+.3f}, "
                     f"max {r['null_max']:+.3f}  ->  **p = {r['p']:.4f}**")
        lines.append(f"- verdict: latent space **{verdict}** correct from incorrect here.")
        lines.append("")
    lines.append("## Read")
    lines.append("If PRE separates, the model represents its own capability boundary from the "
                 "question alone (anticipatory) -- the basis of confident-incapable BEFORE wasting "
                 "compute. If only POST separates, it is a readout of having reasoned well/badly.")
    lines.append("Caveat: single-pass correct/incorrect conflates capability with luck. The honest "
                 "label is K-sample self-consistency (next run).")
    return "\n".join(lines), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    args = ap.parse_args()

    points = harvest(args.model, args.n, args.max_new_tokens)
    if not points:
        print("[harvest] no points", flush=True)
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    pt = OUT / f"latent_outcome_points_{ts}.pt"
    torch.save([{k: v for k, v in p.items()} for p in points], pt)
    report, _ = analyze(points)
    md = OUT / f"latent_outcome_report_{ts}.md"
    md.write_text(report, encoding="utf-8")
    print(f"\n[harvest] wrote {pt}\n[harvest] wrote {md}\n", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
