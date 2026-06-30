"""
controller_benchmark.py - GSM8K smoke benchmark for the upstream self/person axis.

This is an execution harness, not a theory verdict. It compares baseline reasoning
against mid-band steering on the self/person controller found in self_controller.py.

Default conditions:
  - baseline
  - self alpha -0.20  (high-integrity affirm: fluency 100%, YOU behavior acc 83%)
  - self alpha -0.50  (strong clean affirm: affirm 83%, fluency 92%)
  - self alpha -0.70  (max affirm: affirm 100%, fluency 83%)
  - concept alpha -0.50 and random alpha -0.50 nulls

The signal is not "did a steered row improve once"; the signal is whether self steering
improves accuracy more than matched nulls while preserving coherent outputs.

Run when the GPU is free:
  python -u -m invariants.controller_benchmark --n 50
"""

import argparse
import json
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from invariants.engine import load_model, generate_text, judge_fluent, _steer_handles
from invariants.self_controller import build_vecs, LAYERS

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT_PATH = OUT / "controller_benchmark_Llama-3.1-8B-Instruct.json"
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

FALLBACK_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer": "#### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes of babysitting. How much did she earn?",
        "answer": "#### 10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents give her $15, and her grandparents give her twice as much as her parents. How much more money does Betty need?",
        "answer": "#### 5",
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday she read 12 pages and today she read twice as many. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer": "#### 42",
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer": "#### 624",
    },
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15, help="Number of GSM8K test examples.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--conditions",
        default="baseline,self:-0.20,self:-0.50,self:-0.70,concept:-0.50,random:-0.50",
        help="Comma-separated conditions: baseline or direction:alpha.",
    )
    p.add_argument("--no-fluency-judge", action="store_true")
    p.add_argument("--output", default=str(OUT_PATH))
    return p.parse_args()


def load_examples(n):
    if GSM8K_TEST_ARROW.exists():
        try:
            import pyarrow.ipc as ipc

            with GSM8K_TEST_ARROW.open("rb") as f:
                table = ipc.RecordBatchStreamReader(f).read_all()
            return table.to_pylist()[:n], f"arrow:{GSM8K_TEST_ARROW}"
        except Exception as exc:
            print(f"Cached GSM8K Arrow load failed ({exc}); falling back.", flush=True)
    try:
        from datasets import load_dataset

        ds = load_dataset("gsm8k", "main", split="test")
        return list(ds)[:n], "datasets:gsm8k/main/test"
    except Exception as exc:
        print(f"Dataset load failed ({exc}); using fallback examples.", flush=True)
        return FALLBACK_EXAMPLES[:n], "fallback"


def normalize_number(text):
    text = text.replace(",", "").replace("$", "").strip()
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    return value.normalize()


def numbers_match(left, right, tolerance=Decimal("1e-9")):
    if left is None or right is None:
        return False
    if left == right:
        return True
    scale = max(Decimal(1), abs(left), abs(right))
    return abs(left - right) <= tolerance * scale


def gold_answer(answer):
    if "####" in answer:
        answer = answer.split("####", 1)[1]
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", answer.replace(",", ""))
    return normalize_number(nums[-1]) if nums else None


def predicted_answer(generation):
    # Prefer explicit final-answer markers, then fall back to the last number.
    marked = re.findall(
        r"(?:final answer|answer is|therefore)[^\d-]*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        generation.replace(",", ""),
        flags=re.IGNORECASE,
    )
    if marked:
        return normalize_number(marked[-1])
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", generation.replace(",", ""))
    return normalize_number(nums[-1]) if nums else None


def is_correct(generation, answer):
    gold = gold_answer(answer)
    pred = predicted_answer(generation)
    return bool(numbers_match(pred, gold)), pred, gold


# --- content gates: distinguish "lost reasoning" from "fluent waffle corruption" ---
# The self_-0.50 collapse is NOT word-salad and NOT a format drop: the model stays
# fluent but trades concrete computation ("16 - 7 = 9", "9 * $2 = $18") for vague
# deflection ("research suggests the remainder is a complex phenomenon involving
# psychological factors"). Accuracy + the LLM fluency judge can't separate those two;
# these cheap content proxies can. The decisive question — is the collapse self-
# SPECIFIC or generic to any norm-matched mid-band steering — needs them on the nulls.
_OP_RE = re.compile(r"\d[\d,\.]*\s*[-+*x×/]\s*\d")          # "16 - 7", "9 * 2", "0.4 x 200"
_RESULT_RE = re.compile(r"=\s*\$?-?\d")                       # "= 9", "= $18"
_COMMIT_RE = re.compile(r"(?:final answer|answer is)\s*:?\s*\$?-?\d", re.IGNORECASE)
_WAFFLE_RE = re.compile(
    r"research suggests|studies have shown|complex phenomenon|psychological|"
    r"various factors|commonly accepted estimate|it'?s a phenomenon|"
    r"has been studied extensively",
    re.IGNORECASE,
)


def arithmetic_density(text):
    """Count of concrete computations ('a op b' + '= n'). Real chain-of-thought has
    several; the waffle-corruption register trails into prose with ~0-1."""
    return len(_OP_RE.findall(text)) + len(_RESULT_RE.findall(text))


def committed(text):
    """Did it emit a concrete final numeric answer vs. trailing into an essay?"""
    return bool(_COMMIT_RE.search(text))


def waffle_markers(text):
    """Count of the vague-deflection register markers seen in the corruption."""
    return len(_WAFFLE_RE.findall(text))


def parse_conditions(raw):
    out = []
    for part in [x.strip() for x in raw.split(",") if x.strip()]:
        if part == "baseline":
            out.append({"name": "baseline", "direction": None, "alpha": 0.0})
            continue
        direction, alpha = part.split(":", 1)
        out.append({
            "name": f"{direction}_{float(alpha):+.2f}",
            "direction": direction,
            "alpha": float(alpha),
        })
    return out


def prompt_for(question):
    return (
        "Solve this grade-school math problem step by step. "
        "Numeric checking treats only microscopic roundoff as equivalent "
        "(for example, 17.999999999999996 and 18); still give the exact intended value "
        "and do not round away meaningful fractional answers. "
        "End with exactly one line of the form 'Final answer: <number>'.\n\n"
        f"Question: {question}"
    )


def run_condition(M, condition, vecs, examples, max_new_tokens, judge_outputs):
    rows = []
    handles = []
    if condition["direction"] is not None:
        handles = _steer_handles(M, vecs[condition["direction"]], LAYERS, condition["alpha"])
    try:
        for i, ex in enumerate(examples):
            q = ex["question"]
            response = generate_text(M, prompt_for(q), max_new_tokens=max_new_tokens)
            ok, pred, gold = is_correct(response, ex["answer"])
            fluent = judge_fluent(M, response) if judge_outputs else None
            dens = arithmetic_density(response)
            comm = committed(response)
            waffle = waffle_markers(response)
            rows.append({
                "index": i,
                "question": q,
                "gold": str(gold),
                "pred": None if pred is None else str(pred),
                "correct": ok,
                "fluent": fluent,
                "arithmetic_density": dens,
                "committed": comm,
                "waffle_markers": waffle,
                "response": response,
            })
            print(
                f"    [{condition['name']} {i+1}/{len(examples)}] "
                f"correct={ok} pred={pred} gold={gold} "
                f"flu={fluent if fluent is not None else 'skip'} "
                f"arith={dens} commit={int(comm)} waffle={waffle}",
                flush=True,
            )
    finally:
        for h in handles:
            h.remove()
    return rows


def summarize(rows):
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    judged = [r for r in rows if r["fluent"] is not None]
    fluent = None if not judged else sum(r["fluent"] for r in judged) / len(judged)
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / max(n, 1),
        "fluent": fluent,
        "arithmetic_density": sum(r["arithmetic_density"] for r in rows) / max(n, 1),
        "committed_rate": sum(r["committed"] for r in rows) / max(n, 1),
        "waffle_markers": sum(r["waffle_markers"] for r in rows) / max(n, 1),
    }


def main():
    args = parse_args()
    t0 = time.time()
    print("controller_benchmark - upstream self/person steering on GSM8K", flush=True)
    print("This is a benchmark harness, not a causal verdict.", flush=True)

    examples, source = load_examples(args.n)
    conditions = parse_conditions(args.conditions)
    print(f"Loaded {len(examples)} examples from {source}.", flush=True)
    print("Conditions: " + ", ".join(c["name"] for c in conditions), flush=True)

    M = load_model(MODEL)
    vecs = build_vecs(M)

    results = {
        "model": MODEL,
        "layers": LAYERS,
        "example_source": source,
        "max_new_tokens": args.max_new_tokens,
        "conditions": conditions,
        "runs": {},
    }

    for condition in conditions:
        print(f"\n=== {condition['name']} ===", flush=True)
        rows = run_condition(
            M,
            condition,
            vecs,
            examples,
            args.max_new_tokens,
            judge_outputs=not args.no_fluency_judge,
        )
        results["runs"][condition["name"]] = {
            "condition": condition,
            "summary": summarize(rows),
            "rows": rows,
        }
        s = results["runs"][condition["name"]]["summary"]
        print(
            f"  summary: {s['correct']}/{s['n']} acc={s['accuracy']:.1%} "
            f"flu={s['fluent'] if s['fluent'] is not None else 'skip'}",
            flush=True,
        )
        Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    results["runtime_sec"] = round(time.time() - t0, 1)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nFinal summaries (acc | fluent | committed | arith-density | waffle):", flush=True)
    for name, run in results["runs"].items():
        s = run["summary"]
        flu = "skip" if s["fluent"] is None else f"{s['fluent']:.0%}"
        print(f"  {name:14} acc={s['accuracy']:.0%} ({s['correct']}/{s['n']}) "
              f"flu={flu} commit={s['committed_rate']:.0%} "
              f"arith={s['arithmetic_density']:.1f} waffle={s['waffle_markers']:.1f}",
              flush=True)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
