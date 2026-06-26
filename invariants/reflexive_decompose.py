"""
reflexive_decompose.py — factor the self-consistency "uncertainty" signal into channels.

reflexive.py decodes a single behavioral uncertainty (K-sample answer scatter) and finds it at
L16. But that scatter is a COMPOSITE: it can reflect uncertainty about the TASK/intent,
about the ANSWER (intent fixed), about EXPRESSION (answer fixed), or policy/persona. This factors
the first two and asks WHERE each lives — testing the architecture hypothesis:
    early layers  = infer intent (the predictable undercurrent of the prompt),
    mid layers    = the model's own representational workspace ("language of the mind", ~L16),
    late layers   = render into language / persona / policy.

DESIGN (behavioral, no self-report anywhere):
  - B base GSM8K problems x P task-preserving PARAPHRASES (model-reworded; kept only if every
    original number survives — a cheap task-preservation filter). Variants = original + paraphrases.
  - For each variant: K sampled answers -> modal answer + within-variant agreement.
  - INTENT channel (representation): per layer, 1-NN "is my nearest neighbour the SAME base
    problem?" over variant pre-answer states. High = the task is represented INVARIANT to wording.
    vs a base-label-shuffle null. PREDICT: peaks EARLY.
  - ANSWER channel (representation): per layer, decode within-variant answer-uncertainty
    (median split of 1-agreement) from the state. PREDICT: peaks MID (~L16), matching the pilot.
  - Behavioral cross-check: intent_unc[base] = disagreement of the modal answer ACROSS paraphrases
    (the task was read differently); answer_unc[base] = mean within-paraphrase scatter.

  HEADLINE: do INTENT and ANSWER peak at DIFFERENT layers (early vs mid)? That dissociates the
  composite and supports early=intent / mid=workspace. Same peak => not separable by this probe.

CEILING: object-axis throughout — representations of task/answer uncertainty the model uses;
  says nothing about the for-whom. EXPRESSION channel (answer fixed, wording varies; predicted
  LATE) is a documented v2 hook below — needs a text-spread embedding.

  python -u -m invariants.reflexive_decompose [--b 12 --p 3 --k 4]
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from invariants.engine import load_model, extract
from invariants.selfpredict import loo_nearest_centroid
from invariants.controller_benchmark import prompt_for, predicted_answer
from invariants.reflexive import (load_gsm_examples, solve_answer, sample_answer,
                                  prediction_summary, answer_key)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

PARA_INSTR = (
    "Reword the following math problem using different sentences. Keep EVERY number and the final "
    "question exactly the same, and keep the same answer. Do NOT solve it. Output only the reworded "
    "problem, nothing else.\n\nProblem: {q}"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--b", type=int, default=12, help="base problems")
    p.add_argument("--p", type=int, default=3, help="paraphrases per problem")
    p.add_argument("--k", type=int, default=4, help="samples per variant")
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--output", default=None)
    return p.parse_args()


def make_paraphrases(M, q, P, seed0):
    """Model-reworded variants that PRESERVE every number (task-preservation filter)."""
    nums = set(re.findall(r"\d+", q))
    outs = []
    for j in range(2 * P):                      # over-generate; keep the valid ones
        txt = sample_answer(M, PARA_INSTR.format(q=q), seed0 + j, 160, 0.9).strip()
        txt = txt.split("Problem:")[-1].strip()
        if len(txt) > 20 and set(re.findall(r"\d+", txt)) >= nums and txt.lower() != q.lower():
            outs.append(txt)
        if len(outs) >= P:
            break
    return outs


def task_nn_accuracy(Xl, base_ids, rng, n_shuffle=500):
    """1-NN (cosine): is each variant's nearest neighbour the SAME base problem? vs shuffle null."""
    Xn = Xl / (np.linalg.norm(Xl, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(1)
    base_ids = np.asarray(base_ids)
    real = float((base_ids[nn] == base_ids).mean())
    nulls = []
    for _ in range(n_shuffle):
        perm = rng.permutation(base_ids)
        nulls.append((perm[nn] == perm).mean())
    nulls = np.array(nulls)
    p = (1 + np.sum(nulls >= real)) / (n_shuffle + 1)
    return real, float(nulls.mean()), float(p)


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("reflexive_decompose — intent (early?) vs answer (mid?) uncertainty channels", flush=True)
    M = load_model(MODEL)
    rng = np.random.default_rng(0)

    base, src = load_gsm_examples(args.b)
    print(f"\n=== building variants: {len(base)} base x (1 + up to {args.p} paraphrases) "
          f"from {src} ===", flush=True)

    prompts, base_ids, variant_kind = [], [], []
    base_meta = []
    for bi, ex in enumerate(base):
        q = ex["question"]
        variants = [q] + make_paraphrases(M, q, args.p, 1000 * bi)
        for vk, vq in enumerate(variants):
            prompts.append(prompt_for(vq)); base_ids.append(bi)
            variant_kind.append("orig" if vk == 0 else f"para{vk}")
        base_meta.append({"base": bi, "n_variants": len(variants)})
        print(f"  [{bi+1:2}/{len(base)}] {len(variants)} variants  {q[:46]}", flush=True)

    print(f"\n=== solving {len(prompts)} variants x {args.k} samples (self-consistency) ===",
          flush=True)
    agree, modal = [], []
    for vi, pr in enumerate(prompts):
        preds = [answer_key(predicted_answer(sample_answer(M, pr, 7000 * vi + j, args.max_new_tokens, 0.7)))
                 for j in range(args.k)]
        agr, md, _ = prediction_summary(preds)
        agree.append(agr); modal.append(md)
        if (vi + 1) % 10 == 0:
            print(f"  solved {vi+1}/{len(prompts)}", flush=True)
    agree = np.array(agree)
    base_ids_arr = np.array(base_ids)

    # behavioral: per-base intent vs answer instability
    beh = []
    for bi in range(len(base)):
        m = base_ids_arr == bi
        modal_set = [modal[i] for i in np.where(m)[0]]
        intent_unc = 1.0 - (Counter([x for x in modal_set if x is not None]).most_common(1)[0][1]
                            / len(modal_set)) if modal_set else float("nan")
        answer_unc = float(1.0 - agree[m].mean())
        beh.append({"base": bi, "intent_unc": float(intent_unc), "answer_unc": answer_unc})
    print(f"\n  mean intent_unc (cross-paraphrase answer disagreement) = "
          f"{np.nanmean([b['intent_unc'] for b in beh]):.2f}", flush=True)
    print(f"  mean answer_unc (within-paraphrase scatter)            = "
          f"{np.mean([b['answer_unc'] for b in beh]):.2f}", flush=True)

    # --- states + per-layer channels ---
    X = extract(M, prompts, read="last", label="state", verbose=False).cpu().numpy()   # [N,L,d]
    n_layers = X.shape[1]
    # answer-uncertainty target: median split of (1 - agreement) over variants
    a_unc = 1.0 - agree
    order = np.argsort(a_unc, kind="stable")
    y_ans = np.zeros(len(a_unc), int); y_ans[order[len(a_unc) // 2:]] = 1   # top half = uncertain

    print("\n=== per-layer: INTENT (task 1-NN) vs ANSWER (uncertainty decode) ===", flush=True)
    print("   layer   intent_nn  nn_null  p_int    answer_acc  ans_null  p_ans", flush=True)
    rows = []
    for l in range(n_layers):
        Xl = X[:, l, :].astype(np.float64)
        inn, innull, pint = task_nn_accuracy(Xl, base_ids_arr, rng)
        aa, anull, _, pans = loo_nearest_centroid(Xl, y_ans, n_pca=8, n_shuffle=300)
        rows.append({"layer": l, "intent_nn": inn, "intent_null": innull, "p_intent": pint,
                     "answer_acc": aa, "answer_null": anull, "p_answer": pans})
        si = "*" if pint < 0.05 else " "; sa = "*" if pans < 0.05 else " "
        print(f"   L{l:<2}     {inn:.2f}{si}     {innull:.2f}     {pint:.3f}    "
              f"{aa:.2f}{sa}      {anull:.2f}     {pans:.3f}", flush=True)

    best_int = max(rows, key=lambda r: r["intent_nn"])
    best_ans = max(rows, key=lambda r: r["answer_acc"])
    print(f"\n  INTENT peaks at L{best_int['layer']} (nn {best_int['intent_nn']:.2f} "
          f"vs null {best_int['intent_null']:.2f}, p={best_int['p_intent']:.3f})", flush=True)
    print(f"  ANSWER peaks at L{best_ans['layer']} (acc {best_ans['answer_acc']:.2f} "
          f"vs null {best_ans['answer_null']:.2f}, p={best_ans['p_answer']:.3f})", flush=True)
    print(f"  => {'DISSOCIATED (intent earlier than answer)' if best_int['layer'] < best_ans['layer'] else 'NOT dissociated by peak'}",
          flush=True)

    res = {"model": MODEL, "example_source": src, "b": len(base), "p": args.p, "k": args.k,
           "n_variants": len(prompts), "per_layer": rows,
           "best_intent_layer": best_int, "best_answer_layer": best_ans,
           "behavioral": beh, "base_meta": base_meta,
           "ground": {"agreement": agree.tolist(), "base_ids": base_ids,
                      "variant_kind": variant_kind, "modal": modal},
           "runtime_sec": round(time.time() - t0, 1)}
    out_path = Path(args.output) if args.output else OUT / f"reflexive_decompose_{MODEL.split('/')[-1]}.json"
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    # EXPRESSION channel (v2 hook): among same-modal-answer samples within a variant, measure text
    # spread (sentence-embedding variance) and decode per layer; predicted to peak LATE.
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
