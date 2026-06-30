"""Uncertainty in the thinking, not the talking.

A net outputs English; a brain outputs English; neither thinks in English. So we
do NOT label latent states by correctness or by answer-agreement -- both are read
off the render, and the render can be unfaithful (the lie-detector result). We
measure uncertainty where it actually lives: the dispersion of the model's own
latent trajectory across K samples.

  - Elicitation is minimal: the bare question as a normal chat turn. No "step by
    step", no format demand -- we do not tell it how to think.
  - Sample K times. Capture the K latent trajectories (per-layer mean over the
    generated tokens). Do NOT supervise with any English label.
  - Per problem, per layer: do the K trajectories CONVERGE (the computation
    settled -> confident) or SCATTER (the thinking is unsettled -> uncertain)?
    Measured after removing the problem's own mean, so it is not drowned by the
    common-mode (~0.99 cosines), per the concept-run lesson.
  - English enters ONLY as a downstream validation at the end (does low latent
    dispersion track getting it right / answering consistently?) -- never as the
    thing that finds the structure.

Saves raw states so it can be re-analysed with no GPU. Touches nothing else.

Run:
    .venv\\Scripts\\python.exe scripts\\harvest_latent_uncertainty.py --n 16 --k 6
"""

import argparse
import datetime
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.engine import load_model, _inputs, _hidden_states
from invariants.controller_benchmark import load_examples, is_correct

OUT = Path(__file__).parent.parent / "invariants" / "out"


def _eos_ids(M):
    ids = [M.tok.eos_token_id]
    # Llama chat models often end turns with <|eot_id|>; include it so sampled
    # benchmarks do not burn the full token budget after the answer is done.
    if 128009 not in ids:
        ids.append(128009)
    return ids


@torch.no_grad()
def sample_states(
    M,
    question,
    k,
    max_new_tokens,
    temperature,
    *,
    max_time_per_sample=None,
    progress_label=None,
):
    """K sampled generations of the bare question. Returns (states[K,n_layers,d], texts)."""
    inputs = _inputs(M, question)          # native chat turn, no CoT instruction
    plen = inputs["input_ids"].shape[1]
    states, texts = [], []
    eos_ids = _eos_ids(M)
    for sample_idx in range(k):
        t0 = time.time()
        if progress_label:
            print(f"    [{progress_label} sample {sample_idx + 1}/{k}] generating...", flush=True)
        generate_kwargs = {
            **inputs,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "pad_token_id": M.tok.eos_token_id,
            "eos_token_id": eos_ids,
        }
        if max_time_per_sample is not None and max_time_per_sample > 0:
            generate_kwargs["max_time"] = float(max_time_per_sample)
        out = M.model.generate(**generate_kwargs)[0]
        if out.shape[0] <= plen:
            if progress_label:
                print(f"    [{progress_label} sample {sample_idx + 1}/{k}] empty {time.time() - t0:.1f}s", flush=True)
            continue
        hs = _hidden_states(M, out.unsqueeze(0))               # [n_layers, full, d]
        states.append(hs[:, plen:, :].float().mean(1).squeeze(0).cpu())  # [n_layers, d]
        texts.append(M.tok.decode(out[plen:], skip_special_tokens=True).strip())
        if progress_label:
            print(
                f"    [{progress_label} sample {sample_idx + 1}/{k}] done "
                f"{time.time() - t0:.1f}s tokens={int(out.shape[0] - plen)}",
                flush=True,
            )
    if not states:
        return None, []
    return torch.stack(states), texts


def harvest(model_name, n, k, max_new_tokens, temperature):
    examples, source = load_examples(n)
    print(f"[harvest] {len(examples)} problems x {k} samples from {source}", flush=True)
    M = load_model(model_name)
    points = []
    for i, ex in enumerate(examples):
        q = ex.get("question") or ""
        gold = ex.get("answer") or ""
        states, texts = sample_states(M, q, k, max_new_tokens, temperature)
        if states is None:
            print(f"  [{i+1}/{len(examples)}] no samples", flush=True)
            continue
        preds = [str(is_correct(t, gold)[1]) for t in texts]      # validation only
        oks = [bool(is_correct(t, gold)[0]) for t in texts]
        points.append({"question": q[:160], "gold": str(is_correct(texts[0], gold)[2]),
                       "states": states, "preds": preds, "oks": oks})
        modal, cnt = Counter(preds).most_common(1)[0]
        print(f"  [{i+1}/{len(examples)}] samples={len(texts)} "
              f"answer-consistency={cnt}/{len(preds)} (modal {modal})", flush=True)
    return points


def per_problem_dispersion(states, layer):
    """1 - mean pairwise cosine of the K sample-states at `layer`, AFTER removing
    this problem's own mean (so common-mode does not drown the scatter)."""
    X = states[:, layer, :].float()
    Xc = F.normalize(X - X.mean(0, keepdim=True), dim=1)
    S = Xc @ Xc.t()
    k = X.shape[0]
    if k < 2:
        return float("nan")
    off = [S[i, j].item() for i in range(k) for j in range(i + 1, k)]
    return 1.0 - sum(off) / len(off)


def analyze(points):
    nl = points[0]["states"].shape[1]
    lines = ["# Latent Uncertainty (dispersion of the thinking, no English label)", "",
             f"- problems: {len(points)}", ""]
    # per-layer mean dispersion across problems + how much it varies (the usable range)
    lines.append("## Per-layer latent dispersion (mean across problems / spread)")
    lines.append("| layer | mean disp | min | max |")
    lines.append("|---:|---:|---:|---:|")
    disp = {L: [per_problem_dispersion(p["states"], L) for p in points] for L in range(nl)}
    for L in range(nl):
        d = disp[L]
        if L % 2 == 0 or L in (15, 16, 17):
            lines.append(f"| {L} | {sum(d)/len(d):+.3f} | {min(d):+.3f} | {max(d):+.3f} |")
    # validation (downstream only): does latent dispersion track ENGLISH answer-consistency?
    cons = []
    for p in points:
        c = Counter(p["preds"]).most_common(1)[0][1] / len(p["preds"])
        cons.append(c)
    lines.append("")
    lines.append("## Validation only -- latent dispersion vs ENGLISH answer-consistency")
    lines.append("(Spearman-ish: correlation of per-problem latent dispersion with output consistency. "
                 "Negative = more scattered thinking -> less consistent answers, i.e. the latent sensor "
                 "tracks the model's own uncertainty.)")
    lines.append("")
    lines.append("| layer | corr(disp, consistency) |")
    lines.append("|---:|---:|")
    def corr(a, b):
        n = len(a); ma = sum(a)/n; mb = sum(b)/n
        cov = sum((x-ma)*(y-mb) for x, y in zip(a, b))
        va = sum((x-ma)**2 for x in a) ** .5; vb = sum((y-mb)**2 for y in b) ** .5
        return cov/(va*vb) if va > 0 and vb > 0 else float("nan")
    best = (9, -1)
    for L in range(nl):
        c = corr(disp[L], cons)
        if c == c and c < best[0]:
            best = (c, L)
        if L % 2 == 0 or L in (15, 16, 17):
            lines.append(f"| {L} | {c:+.3f} |")
    lines.append("")
    lines.append(f"- strongest (most negative) at **L{best[1]}** (corr {best[0]:+.3f}).")
    lines.append("- if strongly negative: the convergence of the THINKING predicts the consistency of "
                 "the TALKING -- an English-free uncertainty sensor. If ~0: dispersion as measured is "
                 "not the right read, or output-consistency is the wrong validator.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    points = harvest(args.model, args.n, args.k, args.max_new_tokens, args.temperature)
    if not points:
        print("[harvest] no points", flush=True)
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    pt = OUT / f"latent_uncertainty_points_{ts}.pt"
    torch.save([{k: v for k, v in p.items()} for p in points], pt)
    report = analyze(points)
    (OUT / f"latent_uncertainty_report_{ts}.md").write_text(report, encoding="utf-8")
    print(f"\n[harvest] wrote {pt}\n", flush=True)
    print(report, flush=True)


if __name__ == "__main__":
    main()
