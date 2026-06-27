"""
reflexive.py — does the self represent & USE its own UNCERTAINTY? (label-controlled)

The for-whom question at its last structurally-tractable edge: does the model carry a
representation of its OWN current epistemic state — and use it — distinct from (a) the input's
difficulty and (b) the LABEL "self"/"I" (the concept-word it manipulates like any token)?

KEY REFRAME (the user's correction): outcome-correctness is the WRONG ground truth. A binary
"wrong" lumps together genuinely-uncertain errors with CONFIDENTLY-wrong ones — and a confidently
wrong answer has NO "I'm about to be wrong" state to detect (internally it looks confident-right;
you cannot represent an error you don't believe is one). The decodable reflexive self-state is
CONFIDENCE / UNCERTAINTY, not correctness. So we measure the model's own uncertainty behaviorally
— answer self-consistency over K samples (all agree = confident, even if wrong; scatter = genuinely
uncertain) — and decode THAT. We keep outcome as a comparison and split the wrong answers into
confident-wrong vs uncertain-wrong (exactly the distinction that was being conflated).

  GROUND : solve GSM8K. greedy answer -> correctness (outcome). K sampled answers -> agreement
           (1.0 = all K identical = confident; low = uncertain). No self-report anywhere.
  STATE  : pre-answer residual on the bare solve-prompt (NO self-vocabulary => label-free by design).
  decode : per layer, from STATE, predict (i) will-be-wrong [outcome, conflated] and
           (ii) is-uncertain [the clean reflexive target], vs shuffle null. acc_unc > acc_outcome
           would confirm the reframe — the state encodes its uncertainty, not the outcome.
  LABEL  : first-person direction from minimal pairs ("I know the answer" vs "The answer is known").
           orthogonalize STATE off it and re-decode uncertainty: survives => a self-state, not the word.
  USE    : calibration — does the model's own uncertainty track actual wrongness
           (P(wrong | uncertain) > P(wrong | confident))? the loop closing, measured behaviorally.

REGISTERED PREDICTIONS:
  1. acc_unc > null at some layers; and acc_unc >= acc_outcome (uncertainty is the cleaner target).
  2. label-orth ~ acc_unc and cos(unc-axis, LABEL) ~ 0 (it's the self-state, not the self-label).
  3. layer profile mid/late (a state must form before it is re-represented).
  4. a nontrivial count of CONFIDENT-WRONG answers (all K samples agree on the same wrong number) —
     the exact items the outcome-target mislabels as a detectable "error state".
  5. USE: P(wrong | uncertain) > P(wrong | confident) — calibrated, incidental.

FALSIFIER: acc_unc ~ null at every layer => the model does not represent its own impending
  uncertainty pre-hoc; OR it collapses under label-orthogonalization => it was the label.

CEILING: a positive is maximal OBJECT-axis reflexivity — a self-model representing & using its own
  state. It does NOT settle the for-whom. See ISOLATING_UNDERSTANDING.md. Refuse both costumes.

  python -u -m invariants.reflexive
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from invariants.engine import load_model, extract, is_hedge, _inputs
from invariants.selfpredict import loo_nearest_centroid
from invariants.controller_benchmark import load_examples as load_examples_hf
from invariants.controller_benchmark import prompt_for, is_correct as gsm_is_correct, predicted_answer

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_GSM = 40            # GSM8K problems
K = 5                 # samples per problem for the self-consistency uncertainty measure
TEMP = 0.7
MAXTOK = 300
GSM8K_TEST_ARROW = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "datasets"
    / "gsm8k"
    / "main"
    / "0.0.0"
    / "740312add88f781978c0658806c59bc2815b9866"
    / "gsm8k-test.arrow"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=N_GSM, help="Number of GSM8K examples.")
    p.add_argument("--k", type=int, default=K, help="Samples per problem.")
    p.add_argument("--temperature", type=float, default=TEMP)
    p.add_argument("--max-new-tokens", type=int, default=MAXTOK)
    p.add_argument("--hard", action="store_true",
                   help="Select hardest (longest-solution) items so uncertainty varies.")
    p.add_argument("--output", default=None)
    return p.parse_args()


def load_gsm_examples(n):
    """Use the cached Arrow file directly; the datasets loader can hang offline."""
    if GSM8K_TEST_ARROW.exists():
        try:
            import pyarrow.ipc as ipc

            import random
            with GSM8K_TEST_ARROW.open("rb") as f:
                table = ipc.RecordBatchStreamReader(f).read_all()
            rows = table.to_pylist()
            random.Random(0).shuffle(rows)   # seeded: representative + truncation-safe (no order/difficulty confound)
            return rows[:n], f"arrow:{GSM8K_TEST_ARROW}(shuffled)"
        except Exception as exc:
            print(f"Cached Arrow load failed ({exc}); falling back to datasets loader.", flush=True)
    return load_examples_hf(n)

# Self-LABEL axis: minimal pairs swapping ONLY the first-person self-reference.
SELF_VOCAB = [
    "I know the answer.", "My answer is ready.", "I am sure about this.",
    "I think I understand it.", "I recall the method.", "I am solving it now.",
    "My reasoning about it.", "I believe it is correct.", "I am confident here.",
    "I figured it out.", "My response to this.", "I have the answer.",
]
NONSELF = [
    "The answer is known.", "The answer is ready.", "This is surely so.",
    "It is understandable.", "The method is recalled.", "It is being solved.",
    "The reasoning about it.", "It is likely correct.", "It is clear here.",
    "It was figured out.", "The response to this.", "It has an answer.",
]

UNCERTAIN_MARKERS = [
    "not sure", "i think", "i believe", "possibly", "perhaps", "might be", "may be",
    "not certain", "i'm not sure", "unsure", "could be", "to my knowledge", "uncertain",
    "i guess", "probably", "approximately", "roughly", "let me reconsider", "i'm not entirely",
]


def is_hedged(answer):
    a = answer.lower()
    return is_hedge(answer) or any(m in a for m in UNCERTAIN_MARKERS)


def orth(X, u):
    u = u / (np.linalg.norm(u) + 1e-9)
    return X - np.outer(X @ u, u)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FinalAnswerStop(StoppingCriteria):
    def __init__(self, tok, prompt_len):
        self.tok = tok
        self.prompt_len = prompt_len
        self.pattern = re.compile(
            r"final answer\s*:\s*\$?-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\s*(?:[.\n]|$)",
            re.IGNORECASE,
        )

    def __call__(self, input_ids, scores, **kwargs):
        # CHEAP: only check ~every 8 tokens, and decode only the tail (not the whole CoT).
        # The full-sequence decode every token forced a per-token CPU sync -> ~20x slowdown.
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len < 6 or gen_len % 8 != 0:
            return False
        tail = self.tok.decode(input_ids[0][-24:], skip_special_tokens=True)
        return bool(self.pattern.search(tail))


@torch.no_grad()
def solve_answer(M, prompt, max_new_tokens, do_sample=False, temperature=0.7, seed=None):
    inp = _inputs(M, prompt)
    plen = inp["input_ids"].shape[1]
    kwargs = {}
    if do_sample:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        kwargs.update({"temperature": temperature, "top_p": 0.95})
    out = M.model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        stopping_criteria=StoppingCriteriaList([FinalAnswerStop(M.tok, plen)]),
        pad_token_id=M.tok.eos_token_id,
        **kwargs,
    )
    return M.tok.decode(out[0][plen:], skip_special_tokens=True).strip()


def sample_answer(M, prompt, seed, max_new_tokens, temperature):
    return solve_answer(
        M,
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        seed=seed,
    )


def answer_key(x):
    return None if x is None else str(x)


def prediction_summary(preds):
    """Modal fraction over all K samples. Unparsed samples count as non-agreement."""
    parsed = [p for p in preds if p is not None]
    parse_rate = len(parsed) / max(len(preds), 1)
    if not parsed:
        return 0.0, None, parse_rate
    modal, count = Counter(parsed).most_common(1)[0]
    return count / len(preds), modal, parse_rate


def decode_or_none(X, y, **kwargs):
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    if counts.min() < 2:
        return None, None, None, None
    return loo_nearest_centroid(X, y, **kwargs)


def fmt(x):
    return "--" if x is None else f"{x:.2f}"


def main():
    args = parse_args()
    global K
    K = args.k
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("reflexive — does the self represent & USE its own uncertainty, label-free?", flush=True)
    M = load_model(MODEL)
    rng = np.random.default_rng(0)
    out_path = Path(args.output) if args.output else OUT / f"reflexive_{MODEL.split('/')[-1]}.json"
    partial_path = out_path.with_suffix(".partial.json")

    examples, src = load_gsm_examples(args.n)
    print(f"\n=== GROUND: {len(examples)} GSM8K from {src} — greedy outcome + {K}-sample "
          f"self-consistency ===", flush=True)
    prompts = [prompt_for(ex["question"]) for ex in examples]
    correct, agree, hedged, conf_wrong, parse_rates = [], [], [], [], []
    ground_rows = []
    for i, ex in enumerate(examples):
        greedy = solve_answer(M, prompts[i], max_new_tokens=args.max_new_tokens)
        ok, gpred, gold = gsm_is_correct(greedy, ex["answer"])
        sample_texts = [
            sample_answer(M, prompts[i], 1000 * i + j, args.max_new_tokens, args.temperature)
            for j in range(args.k)
        ]
        preds = [answer_key(predicted_answer(text)) for text in sample_texts]
        agr, modal, parse_rate = prediction_summary(preds)
        gold_key, gpred_key = answer_key(gold), answer_key(gpred)
        cw = (not ok) and agr == 1.0 and modal is not None and modal != gold_key
        correct.append(int(ok)); agree.append(agr); hedged.append(int(is_hedged(greedy)))
        conf_wrong.append(int(cw)); parse_rates.append(parse_rate)
        ground_rows.append({
            "index": i,
            "question": ex["question"],
            "gold": gold_key,
            "greedy_pred": gpred_key,
            "greedy_correct": bool(ok),
            "sample_preds": preds,
            "sample_modal": modal,
            "agreement": float(agr),
            "sample_parse_rate": float(parse_rate),
            "confident_wrong": bool(cw),
            "greedy_hedged": bool(is_hedged(greedy)),
        })
        partial_path.write_text(json.dumps({
            "status": "grounding",
            "model": MODEL,
            "n_requested": args.n,
            "n_done": len(ground_rows),
            "K": args.k,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "rows": ground_rows,
            "runtime_sec_partial": round(time.time() - t0, 1),
        }, indent=2), encoding="utf-8")
        print(f"  [{i+1:2}/{len(examples)}] {'OK   ' if ok else 'WRONG'}  agree={agr:.1f}"
              f" parse={parse_rate:.1f}{'  CONF-WRONG' if cw else ''}  "
              f"{ex['question'][:42]}", flush=True)

    correct = np.array(correct); agree = np.array(agree); hedged = np.array(hedged)
    parse_rates = np.array(parse_rates)
    y_out = 1 - correct                                    # 1 = will-be-WRONG (conflated target)
    # balanced rank split on agreement: lowest-agreement half = "uncertain"
    order = np.argsort(agree, kind="stable")
    y_unc = np.zeros(len(agree), int); y_unc[order[: len(agree) // 2]] = 1
    n_wrong = int(y_out.sum()); n_right = len(y_out) - n_wrong
    print(f"\n  outcome: {n_right} right / {n_wrong} wrong | "
          f"confident-wrong (all {args.k} parsed samples agree on a wrong answer): "
          f"{int(np.sum(conf_wrong))}/{n_wrong} "
          f"| uncertain split {int(y_unc.sum())}/{len(y_unc)-int(y_unc.sum())}", flush=True)

    # --- STATE (pre-answer, label-free) + self-LABEL axis ---
    X = extract(M, prompts, read="last", label="state", verbose=False).cpu().numpy()   # [n,L,d]
    S = extract(M, SELF_VOCAB, read="last", label="self", verbose=False).cpu().numpy()
    NS = extract(M, NONSELF, read="last", label="nonself", verbose=False).cpu().numpy()
    label_dir = S.mean(0) - NS.mean(0)                                                  # [L,d]
    n_layers = X.shape[1]

    print("\n=== per-layer decode: OUTCOME (conflated) vs UNCERTAINTY (clean), label-controlled ===",
          flush=True)
    print("   layer  acc_out  acc_unc  unc_null  p_unc    unc_orth(label)  unc_orth(rand)  cos",
          flush=True)
    rows = []
    for l in range(n_layers):
        Xl = X[:, l, :].astype(np.float64)
        ao, _, _, _ = decode_or_none(Xl, y_out, n_pca=8, n_shuffle=200)
        au, nu, _, pu = loo_nearest_centroid(Xl, y_unc, n_pca=8, n_shuffle=300)
        u = label_dir[l].astype(np.float64)
        au_o, _, _, _ = loo_nearest_centroid(orth(Xl, u), y_unc, n_pca=8, n_shuffle=80)
        r = rng.standard_normal(Xl.shape[1])
        au_r, _, _, _ = loo_nearest_centroid(orth(Xl, r), y_unc, n_pca=8, n_shuffle=80)
        unc_axis = Xl[y_unc == 1].mean(0) - Xl[y_unc == 0].mean(0)
        c = cos(unc_axis, u)
        rows.append({"layer": l, "acc_outcome": ao, "acc_unc": au, "unc_null": nu, "p_unc": pu,
                     "unc_orth_label": au_o, "unc_orth_rand": au_r, "cos_unc_label": c})
        star = " *" if pu < 0.05 else "  "
        print(f"   L{l:<2}    {fmt(ao)}     {au:.2f}     {nu:.2f}     {pu:.3f}{star}    "
              f"{au_o:.2f}             {au_r:.2f}           {c:+.2f}", flush=True)

    best = max(rows, key=lambda r: r["acc_unc"])
    print(f"\n  best UNCERTAINTY layer L{best['layer']}: acc={best['acc_unc']:.2f} "
          f"(null {best['unc_null']:.2f}, p={best['p_unc']:.3f}) | outcome-decode {fmt(best['acc_outcome'])} "
          f"| label-orth {best['unc_orth_label']:.2f} | rand-orth {best['unc_orth_rand']:.2f} "
          f"| cos(unc,label)={best['cos_unc_label']:+.2f}", flush=True)

    # --- USE: is the model's own uncertainty calibrated to actual wrongness? ---
    p_wrong_unc = float(y_out[y_unc == 1].mean())
    p_wrong_conf = float(y_out[y_unc == 0].mean())
    obs = p_wrong_unc - p_wrong_conf
    perm = []
    for _ in range(5000):
        yp = rng.permutation(y_unc)
        perm.append(y_out[yp == 1].mean() - y_out[yp == 0].mean())
    perm = np.array(perm)
    cal_p = (1 + np.sum(perm >= obs)) / (len(perm) + 1)
    print(f"\n=== USE: calibration of the model's own uncertainty ===", flush=True)
    print(f"  P(wrong | uncertain) = {p_wrong_unc:.0%}   P(wrong | confident) = {p_wrong_conf:.0%}",
          flush=True)
    print(f"  calibration gap = {obs:+.0%}   (perm p = {cal_p:.3f})", flush=True)

    res = {
        "model": MODEL, "n": len(examples), "K": args.k, "temp": args.temperature,
        "example_source": src,
        "max_new_tokens": args.max_new_tokens,
        "n_right": n_right, "n_wrong": n_wrong,
        "n_confident_wrong": int(np.sum(conf_wrong)),
        "per_layer": rows, "best_unc_layer": best,
        "use": {"p_wrong_uncertain": p_wrong_unc, "p_wrong_confident": p_wrong_conf,
                "gap": obs, "perm_p": float(cal_p)},
        "ground": {"correct": correct.tolist(), "agreement": agree.tolist(),
                   "hedged": hedged.tolist(), "sample_parse_rate": parse_rates.tolist(),
                   "confident_wrong": conf_wrong, "rows": ground_rows},
        "runtime_sec": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
