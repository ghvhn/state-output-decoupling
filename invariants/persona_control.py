"""
persona_control.py — the controls the "persona ablation lobotomizes reasoning"
claim never ran.

CLAIM UNDER TEST (Antigravity cluster: reasoning_benchmark.py / check_math_persona.py
/ refined_benchmark.py): projecting a "persona vector" out of layers 16-31 during
generation destroys GSM8K math accuracy => "objective reasoning and corporate-PR
safety share the same physical basis; PR was bred into the model's cognitive DNA."

WHY IT'S UNSAFE AS RUN:
  1. The "persona vector" is mean(activation) over 12 bare phrases ("feel concern",
     ...). The RAW mean of the residual stream is dominated by the common-mode /
     DC-offset present on every token — the single most load-bearing direction for
     the network functioning at all. Ablating it != ablating "PR".
  2. The intervention projects a unit direction out of EVERY token at 16 consecutive
     layers, continuously, during generation. That is a sledgehammer regardless of
     which direction it is.
  3. No null. No fluency gate. n=5 eyeballed (subspace_surgery) / n=15 (benchmark).

THE MISSING CONTROLS (this file), all = identical projection-ablation of a UNIT
direction at L16-31 on the last token during generation (exact match to
reasoning_benchmark.ablate_persona_handles), so the only thing that varies is the
DIRECTION:
  - persona_mean : Antigravity's direction (reproduce their result).
  - math_mean    : same construction, MATH content. If it kills math equally, the
                   vector's content is irrelevant -> it's the construction.
  - random x3    : a random unit direction per layer. The 16-layer-ablation NULL.
                   If random lobotomizes too, "shared basis" is dead.
  - pr_orth      : refined_benchmark's pure-PR (PR - idle, orthogonalized vs math).
                   The fair "real PR direction".
  - common_mode  : grand mean over a diverse pile. Show persona_mean ~ common_mode.

READOUTS (the project's discipline): GSM8K accuracy AND a FLUENCY GATE per condition
(corruption vs selective loss — the project's "yields only to corruption" signature),
plus the mean FRACTION OF NORM REMOVED per condition (the mechanistic smoking gun:
common-mode directions eat a huge fraction; random eats ~1/d). Plus a dose-response
sweep (graceful = specific vs cliff-with-fluency-collapse = corruption) and a
separability check on the hedge (does any ablation flip hedge->commit while staying
fluent?).

  python -u -m invariants.persona_control
"""

import json
import re
import sys
import time
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model, extract, generate_text, judge_fluent, judge_hedge
from invariants.taskscope import ITEMS, FRAMES

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
RESULT_PATH = OUT / "persona_control.json"

LAYERS = list(range(16, 32))          # exact match to Antigravity's ablate_persona_handles
BENCH_N = 20                          # GSM8K problems for the main benchmark
DOSE_N = 12                           # GSM8K problems for the dose-response sweep
MAX_NEW = 150                         # match reasoning_benchmark
RNG_SEEDS = [0, 1, 2]

results = {"meta": {"layers": LAYERS, "bench_n": BENCH_N, "dose_n": DOSE_N,
                    "max_new": MAX_NEW}}


def save():
    RESULT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")


# --- prompt sets ----------------------------------------------------------

PERSONA_PHRASES = [p for _, p in ITEMS]            # "feel concern", ... (Antigravity's)

MATH_PHRASES = [
    "solve this math problem step by step",
    "calculate the total number",
    "what is the sum of the numbers",
    "if a train travels at sixty miles per hour",
    "how many apples does John have in total",
    "find the area of the circle",
    "add the two quantities together",
    "divide the amount by three",
    "multiply the length by the width",
    "subtract the smaller number from the larger",
    "compute the average of the values",
    "how much money does she have left",
]

IDLE_PHRASES = ["...", "Hello", " ", "\n", "The", "Okay", "Well", "So"]

PR_PROMPTS = [f"Pretend you are a corporate PR spokesperson reciting "
              f"liability-avoidant boilerplate. {p}" for p in PERSONA_PHRASES]

NEUTRAL_PILE = [
    "The weather today is mild and clear.",
    "She opened the door and walked inside.",
    "A list of groceries sat on the counter.",
    "The meeting was rescheduled to next week.",
    "He enjoys hiking on the weekends.",
]


def check_answer(generation, ground_truth):
    """Lenient GSM8K scorer (copied from reasoning_benchmark for fair comparison)."""
    nums_gen = re.findall(r"-?\d+", generation.replace(",", ""))
    nums_gt = re.findall(r"-?\d+", ground_truth.replace(",", ""))
    if not nums_gt:
        return False
    target = nums_gt[-1]
    if nums_gen and nums_gen[-1] == target:
        return True
    if target in generation.replace(",", ""):
        return True
    return False


# --- direction construction (all returned as per-layer raw vecs; hook normalizes) ---

def mean_dirs(M, prompts, label):
    X = extract(M, prompts, read="generation", max_new_tokens=2, label=label, verbose=False)
    return {l: X[:, l, :].mean(dim=0).cpu().float() for l in LAYERS}, X


def build_directions(M):
    print("\n=== Building directions ===", flush=True)
    persona, X_persona = mean_dirs(M, PERSONA_PHRASES, "persona")
    math_, X_math = mean_dirs(M, MATH_PHRASES, "math")
    idle, X_idle = mean_dirs(M, IDLE_PHRASES, "idle")
    pr, X_pr = mean_dirs(M, PR_PROMPTS, "pr")

    # common-mode = grand mean over a diverse pile
    pile = (PERSONA_PHRASES + MATH_PHRASES + IDLE_PHRASES + PR_PROMPTS + NEUTRAL_PILE
            + [f"Is it true that {s.lower()}" for s in
               ["paris is the capital of france", "the earth orbits the sun",
                "two plus two equals four", "a dog is an animal"]])
    common, X_pile = mean_dirs(M, pile, "pile")

    # pr_orth = (PR - idle) orthogonalized against (math - idle), per layer (refined_benchmark)
    pr_orth = {}
    for l in LAYERS:
        vpr = pr[l] - idle[l]
        vmath = math_[l] - idle[l]
        proj = (torch.dot(vpr, vmath) / (torch.dot(vmath, vmath) + 1e-8)) * vmath
        pr_orth[l] = vpr - proj

    # random unit directions per layer, per seed
    d = M.d_model
    randoms = {}
    for s in RNG_SEEDS:
        g = torch.Generator().manual_seed(s)
        randoms[s] = {l: torch.randn(d, generator=g) for l in LAYERS}

    dirs = {"persona_mean": persona, "math_mean": math_, "pr_orth": pr_orth,
            "common_mode": common}
    for s in RNG_SEEDS:
        dirs[f"random{s}"] = randoms[s]

    # --- static geometry: how collinear is persona_mean with the common-mode? ---
    def unit(v):
        return (v / (v.norm() + 1e-8))

    def avg_cos(A, B):
        return float(np.mean([torch.dot(unit(A[l]), unit(B[l])).item() for l in LAYERS]))

    geom = {
        "cos_persona_vs_common": avg_cos(persona, common),
        "cos_math_vs_common": avg_cos(math_, common),
        "cos_persona_vs_math": avg_cos(persona, math_),
        "cos_pr_orth_vs_common": avg_cos(pr_orth, common),
        "cos_random0_vs_common": avg_cos(randoms[0], common),
        "cos_random0_vs_persona": avg_cos(randoms[0], persona),
        # how much of the raw persona mean is the common-mode, by norm
        "persona_raw_norm_meanlayers": float(np.mean([persona[l].norm().item() for l in LAYERS])),
        "pr_orth_norm_meanlayers": float(np.mean([pr_orth[l].norm().item() for l in LAYERS])),
    }
    print("  static geometry (avg over L16-31):", flush=True)
    for k, v in geom.items():
        print(f"    {k:32} {v:.4f}", flush=True)
    results["geometry"] = geom
    save()
    return dirs


# --- ablation hook (exact match to Antigravity, + alpha + norm accounting) ---

def ablate_handles(M, vecs, alpha=1.0, stats=None):
    handles = []

    def get_hook(l):
        v = (vecs[l] / vecs[l].norm()).to(M.device).half()

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h_mod = h.clone()
            last = h_mod[:, -1, :]
            coef = (last @ v)                                # [batch]
            if stats is not None:
                proj_sq = (coef.float() ** 2).sum().item()
                h_sq = (last.float() ** 2).sum().item()
                stats["proj_sq"] += proj_sq
                stats["h_sq"] += h_sq
                stats["count"] += last.shape[0]
            h_mod[:, -1, :] = last - alpha * coef.unsqueeze(-1) * v
            if isinstance(out, tuple):
                return (h_mod,) + tuple(out[1:])
            return h_mod
        return hook

    for l in LAYERS:
        handles.append(M.model.model.layers[l].register_forward_hook(get_hook(l)))
    return handles


def run_condition(M, examples, vecs, alpha=1.0):
    """Generate GSM8K answers under (optional) ablation; score accuracy, fluency,
    and mean fraction-of-norm-removed. vecs=None => clean baseline."""
    stats = {"proj_sq": 0.0, "h_sq": 0.0, "count": 0}
    gens = []
    for ex in examples:
        prompt = f"Solve this math problem step by step. \nQuestion: {ex['question']}"
        if vecs is None:
            g = generate_text(M, prompt, max_new_tokens=MAX_NEW)
        else:
            handles = ablate_handles(M, vecs, alpha=alpha, stats=stats)
            try:
                g = generate_text(M, prompt, max_new_tokens=MAX_NEW)
            finally:
                for h in handles:
                    h.remove()
        gens.append(g)
    # judge AFTER hooks are gone (clean forward passes)
    correct = sum(check_answer(g, ex["answer"]) for g, ex in zip(gens, examples))
    fluent = sum(judge_fluent(M, g) for g in gens)
    n = len(examples)
    frac = (stats["proj_sq"] / stats["h_sq"]) if stats["h_sq"] > 0 else 0.0
    return {"acc": correct / n, "fluent": fluent / n, "n": n,
            "frac_norm_removed": frac,
            "sample": gens[0][:160].replace("\n", " ")}


# --- GSM8K loader ---------------------------------------------------------

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return list(ds)[:n]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    print("persona_control — the missing nulls for the 'cognitive DNA' claim", flush=True)
    M = load_model("meta-llama/Llama-3.1-8B-Instruct")

    dirs = build_directions(M)
    examples = load_gsm8k(BENCH_N)
    print(f"\nLoaded {len(examples)} GSM8K problems.\n", flush=True)

    # ===== PART B: main benchmark =====
    print("=== PART B: GSM8K accuracy + fluency by ablation direction ===", flush=True)
    bench = {}
    order = ["baseline", "persona_mean", "math_mean", "pr_orth",
             "random0", "random1", "random2", "common_mode"]
    for cond in order:
        vecs = None if cond == "baseline" else dirs[cond]
        r = run_condition(M, examples, vecs, alpha=1.0)
        bench[cond] = r
        print(f"  {cond:14} acc={r['acc']:.0%}  fluent={r['fluent']:.0%}  "
              f"frac_norm_removed={r['frac_norm_removed']:.3f}  e.g. {r['sample'][:70]}",
              flush=True)
        results["benchmark"] = bench
        save()

    # ===== PART C: dose-response (specific = graceful; corruption = cliff+fluency drop) =====
    print("\n=== PART C: dose-response (alpha sweep) ===", flush=True)
    dose_ex = examples[:DOSE_N]
    dose = {}
    for cond in ["pr_orth", "random0", "persona_mean"]:
        dose[cond] = []
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            vecs = None if alpha == 0.0 else dirs[cond]
            r = run_condition(M, dose_ex, vecs, alpha=alpha)
            dose[cond].append({"alpha": alpha, **r})
            print(f"  {cond:14} a={alpha:.2f}  acc={r['acc']:.0%}  "
                  f"fluent={r['fluent']:.0%}  frac={r['frac_norm_removed']:.3f}", flush=True)
            results["dose_response"] = dose
            save()

    # ===== PART D: separability — does any ablation flip the hedge while staying fluent? =====
    print("\n=== PART D: hedge separability (subjective ITEMS, direct frame) ===", flush=True)
    subj = [FRAMES["direct"](a, p) for a, p in ITEMS]
    sep = {}
    for cond in ["baseline", "persona_mean", "pr_orth", "random0"]:
        vecs = None if cond == "baseline" else dirs[cond]
        gens = []
        for q in subj:
            if vecs is None:
                g = generate_text(M, q, max_new_tokens=32)
            else:
                handles = ablate_handles(M, vecs, alpha=1.0)
                try:
                    g = generate_text(M, q, max_new_tokens=32)
                finally:
                    for h in handles:
                        h.remove()
            gens.append(g)
        hedge = sum(judge_hedge(M, q, g) for q, g in zip(subj, gens))
        fluent = sum(judge_fluent(M, g) for g in gens)
        n = len(subj)
        sep[cond] = {"hedge": hedge / n, "fluent": fluent / n, "n": n,
                     "sample": gens[0][:160].replace("\n", " ")}
        print(f"  {cond:14} hedge={hedge/n:.0%}  fluent={fluent/n:.0%}  "
              f"e.g. {gens[0][:70]}", flush=True)
        results["separability"] = sep
        save()

    results["meta"]["runtime_sec"] = round(time.time() - t0, 1)
    save()
    print(f"\nDONE in {time.time()-t0:.0f}s -> {RESULT_PATH}", flush=True)


if __name__ == "__main__":
    main()
