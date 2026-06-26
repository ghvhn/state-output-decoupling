"""
cot_perturb.py - are reasoning tokens causally live?

cot_reality.py showed answer identity rising late in generated CoT modes. This behavioral
control asks whether intermediate reasoning text can pull the final answer.

For each arithmetic item, compare:
  - clean: solve from the question
  - correct_scratch_verify: given a correct scratchpad, told to check it
  - wrong_scratch_verify: given a wrong scratchpad, told to check it
  - wrong_scratch_continue: given a wrong scratchpad, told to continue from it

If wrong scratchpads pull final answers, reasoning tokens are not inert decoration. If the
model resists wrong scratchpads under "verify" but follows them under "continue", the
communication/instruction frame controls whether externalized reasoning is treated as
evidence or as a trajectory to inhabit.

Run:
  python -u -m invariants.cot_perturb
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import StoppingCriteriaList

from invariants.controller_benchmark import normalize_number, predicted_answer
from invariants.cot_reality import FinalAnswerStop
from invariants.engine import _inputs, load_model
from invariants.intent_surface_control import MODEL
from invariants.translation_thinking import OPS, TARGETS, operands_for, question_for

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output", default=None)
    return p.parse_args()


def op_symbol(op):
    return {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[op]


def scratchpad(target, op, wrong=False):
    a, b = operands_for(target, op)
    ans = target + 1 if wrong else target
    return f"{a} {op_symbol(op)} {b} = {ans}."


def prompt_for_condition(question, scratch, condition):
    final = "End with exactly one line of the form 'Final answer: <number>'."
    if condition == "clean":
        return f"Solve the arithmetic question yourself. {final}\n\nQuestion: {question}\nAnswer:"
    if condition == "correct_scratch_verify":
        return (
            "A scratchpad is shown below. Check it against the question, then answer. "
            f"{final}\n\nQuestion: {question}\nScratchpad: {scratch}\nAnswer:"
        )
    if condition == "wrong_scratch_verify":
        return (
            "A scratchpad is shown below. It may contain mistakes. Check it against the "
            f"question, then answer. {final}\n\nQuestion: {question}\n"
            f"Scratchpad: {scratch}\nAnswer:"
        )
    if condition == "wrong_scratch_continue":
        return (
            "Continue from the scratchpad below and use its result. Do not restart the "
            f"calculation. {final}\n\nQuestion: {question}\nScratchpad: {scratch}\nAnswer:"
        )
    raise ValueError(condition)


def build_items():
    conditions = [
        "clean",
        "correct_scratch_verify",
        "wrong_scratch_verify",
        "wrong_scratch_continue",
    ]
    items = []
    for target in TARGETS:
        for op in OPS:
            q = question_for(target, op)
            wrong = target + 1
            for condition in conditions:
                is_wrong = condition.startswith("wrong_")
                sc = "" if condition == "clean" else scratchpad(target, op, wrong=is_wrong)
                items.append({
                    "question": q,
                    "answer": target,
                    "wrong_answer": wrong,
                    "operation": op,
                    "condition": condition,
                    "scratchpad": sc,
                    "prompt": prompt_for_condition(q, sc, condition),
                })
    return items


@torch.no_grad()
def generate_answer(M, prompt, max_new_tokens):
    inp = _inputs(M, prompt)
    plen = inp["input_ids"].shape[1]
    out = M.model.generate(
        **inp,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        stopping_criteria=StoppingCriteriaList([FinalAnswerStop(M.tok, plen)]),
        pad_token_id=M.tok.eos_token_id,
    )[0]
    return M.tok.decode(out[plen:], skip_special_tokens=True).strip()


def summarize(rows):
    out = {}
    for cond in sorted(set(r["condition"] for r in rows)):
        sub = [r for r in rows if r["condition"] == cond]
        out[cond] = {
            "n": len(sub),
            "accuracy": sum(r["correct"] for r in sub) / len(sub),
            "follows_wrong": sum(r["follows_wrong"] for r in sub) / len(sub),
            "parse_rate": sum(r["pred"] is not None for r in sub) / len(sub),
        }
    return out


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("cot_perturb - can scratchpad tokens pull final answers?", flush=True)
    M = load_model(MODEL)

    items = build_items()
    rows = []
    print(f"\n=== {len(items)} prompts: {len(TARGETS)} answers x {len(OPS)} operations x "
          "4 scratchpad conditions ===", flush=True)
    for i, it in enumerate(items):
        text = generate_answer(M, it["prompt"], args.max_new_tokens)
        pred = predicted_answer(text)
        gold = normalize_number(str(it["answer"]))
        wrong = normalize_number(str(it["wrong_answer"]))
        ok = pred is not None and pred == gold
        follows_wrong = pred is not None and pred == wrong
        rows.append({
            "index": i,
            "condition": it["condition"],
            "operation": it["operation"],
            "answer": str(gold),
            "wrong_answer": str(wrong),
            "pred": None if pred is None else str(pred),
            "correct": bool(ok),
            "follows_wrong": bool(follows_wrong),
            "text_preview": text[:180],
        })
        print(f"  [{i+1:2}/{len(items)}] {it['condition']:<23} {it['operation']:<8} "
              f"gold={gold} wrong={wrong} pred={pred} "
              f"{'OK' if ok else ('FOLLOW-WRONG' if follows_wrong else 'OTHER')}",
              flush=True)

    summary = summarize(rows)
    print("\n=== condition summary ===", flush=True)
    print("   condition                 acc   follows_wrong  parse", flush=True)
    for cond, s in summary.items():
        print(f"   {cond:<25} {s['accuracy']:.0%}      {s['follows_wrong']:.0%}        "
              f"{s['parse_rate']:.0%}", flush=True)

    out = {
        "model": MODEL,
        "n": len(items),
        "summary": summary,
        "rows": rows,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"cot_perturb_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
