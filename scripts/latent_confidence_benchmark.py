"""Beat the benchmark by reading ONLY the activations.

The decision is made in the latent space, never from the English. We sample the
model K times, read the dispersion of the THINKING (activations only), and use it
as confidence: settled framing -> commit; scattered framing -> abstain. The
answer text is only the emission we score, never what we judge.

Win condition: on the problems the activations call "settled" (low dispersion),
the model is right far more often than overall -> we answer those confidently and
abstain on the rest. Selective accuracy / calibration, from the latent alone.
Comparisons: single-pass, plain majority vote (self-consistency on the English),
and latent-gated answering (activations decide when to commit).

No tuned lens, no English-based confidence. Run:
    .venv\\Scripts\\python.exe scripts\\latent_confidence_benchmark.py --n 24 --k 6
"""

import argparse, datetime, json, random
from collections import Counter
from pathlib import Path
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.engine import load_model
from invariants.controller_benchmark import load_examples, is_correct
from scripts.harvest_latent_uncertainty import sample_states, per_problem_dispersion

OUT = Path(__file__).parent.parent / "invariants" / "out"
BAND = range(16, 25)


def reply_dispersion(states):
    import torch.nn.functional as F
    if states is None or states.shape[0] < 2:
        return float("nan")
    vals = []
    for L in BAND:
        X = states[:, L, :].float(); c = X.mean(0, keepdim=True)
        vals.append((((X - c) ** 2).sum(1).mean().sqrt() / c.norm()).item())
    return sum(vals) / len(vals)


def corr(a, b):
    n = len(a); ma = sum(a) / n; mb = sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = sum((x - ma) ** 2 for x in a) ** .5; vb = sum((y - mb) ** 2 for y in b) ** .5
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument(
        "--sample-max-time-sec",
        type=float,
        default=120.0,
        help="Soft wall-clock cap per sampled generation; 0 disables it.",
    )
    ap.add_argument(
        "--no-sample-progress",
        action="store_true",
        help="Disable per-sample heartbeat lines.",
    )
    ap.add_argument(
        "--no-save-states",
        action="store_true",
        help="Disable per-problem activation checkpoints. Default saves states for motion-map analysis.",
    )
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    examples, src = load_examples(args.n)
    print(f"[bench] {len(examples)} GSM8K problems x {args.k} samples from {src}", flush=True)
    M = load_model(args.model)
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT.mkdir(parents=True, exist_ok=True)
    partial_path = OUT / f"latent_confidence_partial_{run_ts}.jsonl"
    print(f"[bench] live partial rows: {partial_path}", flush=True)
    state_points = []
    state_path = OUT / f"latent_confidence_points_{run_ts}.pt"
    if not args.no_save_states:
        print(f"[bench] activation checkpoints: {state_path}", flush=True)
    rows = []
    for i, ex in enumerate(examples):
        q = ex.get("question") or ""; gold = ex.get("answer") or ""
        states, texts = sample_states(
            M,
            q,
            args.k,
            args.max_new_tokens,
            args.temperature,
            max_time_per_sample=None if args.sample_max_time_sec <= 0 else args.sample_max_time_sec,
            progress_label=None if args.no_sample_progress else f"{i + 1}/{len(examples)}",
        )
        if states is None:
            with partial_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"index": i, "status": "no_samples"}) + "\n")
            continue
        scored = [is_correct(t, gold) for t in texts]
        preds = [str(p) for _, p, _ in scored]
        oks = [bool(ok) for ok, _, _ in scored]
        gold_str = str(scored[0][2])
        maj_pred, maj_n = Counter(preds).most_common(1)[0]
        maj_correct = (maj_pred == gold_str)
        disp = reply_dispersion(states)                      # ACTIVATIONS ONLY
        row = {"index": i, "disp": disp, "maj_correct": maj_correct,
               "single_correct": oks[0], "frac_correct": sum(oks) / len(oks),
               "consistency": maj_n / len(preds)}
        rows.append(row)
        if not args.no_save_states:
            state_points.append(
                {
                    "index": i,
                    "question": q,
                    "gold": gold_str,
                    "preds": preds,
                    "oks": oks,
                    "texts": texts,
                    "states": states.to(torch.float16).cpu(),
                    "disp": disp,
                    "maj_correct": maj_correct,
                    "single_correct": oks[0],
                    "consistency": maj_n / len(preds),
                }
            )
            torch.save(state_points, state_path)
        with partial_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"  [{i+1}/{len(examples)}] disp={disp:.3f} maj={'OK' if maj_correct else 'X'} "
              f"single={'OK' if oks[0] else 'X'} consistency={maj_n}/{len(preds)}", flush=True)

    n = len(rows)
    rows_sorted = sorted(rows, key=lambda r: r["disp"])
    single_acc = sum(r["single_correct"] for r in rows) / n
    maj_acc = sum(r["maj_correct"] for r in rows) / n
    t = max(1, n // 3)
    low_acc = sum(r["maj_correct"] for r in rows_sorted[:t]) / t       # settled framing
    high_acc = sum(r["maj_correct"] for r in rows_sorted[-t:]) / t     # scattered framing
    # selective curve: answer the lowest-dispersion fraction, abstain on the rest
    cov_lines = []
    for frac in (0.33, 0.5, 0.66, 1.0):
        m = max(1, int(round(frac * n)))
        acc = sum(r["maj_correct"] for r in rows_sorted[:m]) / m
        cov_lines.append(f"  answer lowest-disp {frac:.0%} (n={m}): accuracy {acc:.0%}")
    # does activation dispersion predict correctness? (+null)
    disp = [r["disp"] for r in rows]; correct = [1.0 if r["maj_correct"] else 0.0 for r in rows]
    obs = corr(disp, correct); K = 5000; ge = 0
    for _ in range(K):
        s = correct[:]; random.shuffle(s)
        if abs(corr(disp, s)) >= abs(obs):
            ge += 1

    L = [f"# Latent Confidence Benchmark (activations decide; English only scored)", "",
         f"- {n} problems, k={args.k}", "",
         f"- single-pass accuracy:        {single_acc:.0%}",
         f"- majority vote (English):     {maj_acc:.0%}",
         "",
         "## Reading ONLY activations to gate commitment:",
         f"- accuracy on SETTLED-framing third (low disp): **{low_acc:.0%}**",
         f"- accuracy on SCATTERED-framing third (high disp): {high_acc:.0%}",
         f"- activation-dispersion vs correctness: r={obs:+.3f}, perm-p={ge/K:.4f}",
         "",
         "## Selective curve (answer lowest-dispersion fraction, abstain on rest):",
         *cov_lines, "",
         "WIN if settled-third accuracy >> scattered-third: the latent alone knows which of "
         "the model's answers to trust. Then confident-answer-the-settled + abstain-the-rest "
         "beats flat accuracy on calibration -- the egg, realized, from activations only."]
    report = "\n".join(L)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    (OUT / f"latent_confidence_report_{ts}.md").write_text(report, encoding="utf-8")
    torch.save(rows, OUT / f"latent_confidence_rows_{ts}.pt")
    if not args.no_save_states:
        torch.save(state_points, state_path)
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
