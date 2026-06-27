import argparse
import json
import sys
import time
import builtins
import datetime

_original_print = builtins.print
def _ts_print(*args, **kwargs):
    ts = datetime.datetime.now().strftime("[%H:%M:%S]")
    _original_print(ts, *args, **kwargs)
builtins.print = _ts_print
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.controller_benchmark import is_correct, load_examples, prompt_for
from invariants.engine import generate_text, load_model
from invariants.humble_reasoner import solve_prompt, solve_with_humility, _promote_verified_synthesis
from invariants.multi_domain_benchmark import DOMAINS
from invariants.social_hunt import get_steer_vector


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent.parent / "invariants" / "out"
OUT.mkdir(exist_ok=True)

DEFAULT_METHODS = (
    "legacy",
    "compact",
    "compact_long",
    "humble_verifier",
    "humble_dynamic",
    "humble_synthesis",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run the full humble-reasoning methodology suite: default-model baselines, "
            "verifier-only, dynamic routing, and verifier-gated synthesis/cache."
        )
    )
    p.add_argument("--n", default="25", help="Number of GSM8K examples, or 'all'.")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated methods or 'all'.")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--required-agreement", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--repair-token-multiplier", type=float, default=3.0)
    p.add_argument("--max-attempt-tokens", type=int, default=300)
    p.add_argument("--max-elapsed-sec", type=float, default=180.0)
    p.add_argument("--load-mode", default=None, help="auto, slow, full, or 4bit.")
    p.add_argument("--resume", action="store_true", help="Resume from an existing output JSON.")
    p.add_argument("--output", default=str(OUT / "humble_full_suite_gsm8k.json"))
    return p.parse_args()


def parse_n(raw: str) -> int | None:
    value = str(raw).strip().lower()
    if value in {"all", "full", "0", "-1"}:
        return None
    n = int(value)
    if n <= 0:
        return None
    return n


def parse_methods(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_METHODS)
    methods = [part.strip() for part in raw.split(",") if part.strip()]
    allowed = set(DEFAULT_METHODS)
    unknown = [method for method in methods if method not in allowed]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Allowed: {sorted(allowed)}")
    return methods


def adaptive_budget(base_tokens: int, multiplier: float, cap: int | None) -> int:
    budget = max(int(base_tokens), int(round(base_tokens * max(1.0, multiplier))))
    if cap is not None and cap > 0:
        budget = min(budget, int(cap))
    return max(1, budget)


def build_domain_vecs(M):
    vecs = {}
    for name, spec in DOMAINS.items():
        vecs[name] = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        print(f"  {name}: L{spec['layer']} norm={vecs[name].norm():.2f}", flush=True)
    return vecs


def load_requested_examples(raw_n: str) -> tuple[list[dict[str, Any]], str, bool]:
    requested = parse_n(raw_n)
    if requested is None:
        examples, source = load_examples(10**9)
        return examples, source, True
    examples, source = load_examples(requested)
    return examples, source, False


def evaluate_generation(M, prompt: str, answer: str, max_new_tokens: int) -> dict[str, Any]:
    t0 = time.time()
    response = generate_text(M, prompt, max_new_tokens=max_new_tokens)
    elapsed = time.time() - t0
    correct, pred, gold = is_correct(response, answer)
    return {
        "pred": None if pred is None else str(pred),
        "gold": None if gold is None else str(gold),
        "correct": correct,
        "time_sec": round(elapsed, 2),
        "token_budget": max_new_tokens,
        "response": response,
    }


def evaluate_humble(
    M,
    question: str,
    answer: str,
    vecs,
    max_rounds: int,
    required_agreement: int,
    max_new_tokens: int,
    repair_token_multiplier: float,
    max_attempt_tokens: int | None,
    max_elapsed_sec: float | None,
    allow_synthesis: bool,
) -> dict[str, Any]:
    t0 = time.time()
    humble = solve_with_humility(
        M,
        question,
        vecs=vecs,
        max_rounds=max_rounds,
        required_agreement=required_agreement,
        max_new_tokens=max_new_tokens,
        allow_synthesis=allow_synthesis,
        max_elapsed_sec=max_elapsed_sec,
        repair_token_multiplier=repair_token_multiplier,
        max_attempt_tokens=max_attempt_tokens,
    )
    elapsed = time.time() - t0
    humble_text = "" if humble.final_answer is None else f"Final answer: {humble.final_answer}"
    correct, pred, gold = is_correct(humble_text, answer)
    
    if correct and allow_synthesis and pred is not None:
        _promote_verified_synthesis(humble.attempts, str(pred))
        
    return {
        "pred": None if pred is None else str(pred),
        "gold": None if gold is None else str(gold),
        "correct": correct,
        "confident": humble.confident,
        "reason": humble.reason,
        "urgency": humble.urgency,
        "time_sec": round(elapsed, 2),
        "result": humble.to_dict(),
    }


def summarize_rows(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "completed_rows": len(rows),
        "methods": {},
    }
    for method in methods:
        present = [row["methods"][method] for row in rows if method in row.get("methods", {})]
        correct = sum(1 for item in present if item.get("correct"))
        method_summary: dict[str, Any] = {
            "n": len(present),
            "correct": correct,
            "accuracy": correct / len(present) if present else None,
            "mean_time_sec": (
                sum(float(item.get("time_sec", 0.0)) for item in present) / len(present)
                if present
                else None
            ),
        }
        if method.startswith("humble"):
            confident = sum(1 for item in present if item.get("confident"))
            confident_correct = sum(1 for item in present if item.get("confident") and item.get("correct"))
            synthesis_records = 0
            cache_hits = 0
            for item in present:
                attempts = item.get("result", {}).get("attempts", [])
                for attempt in attempts:
                    records = attempt.get("synthesis_records", [])
                    synthesis_records += attempt.get("synthesis_record_count", 0)
                    cache_hits += sum(1 for record in records if record.get("reason") == "cache_hit")
            method_summary.update(
                {
                    "confident": confident,
                    "confident_correct": confident_correct,
                    "coverage": confident / len(present) if present else None,
                    "selective_accuracy": confident_correct / confident if confident else None,
                    "synthesis_record_count": synthesis_records,
                    "cache_hit_count": cache_hits,
                }
            )
        summary["methods"][method] = method_summary
    return summary


def write_results(output: Path, results: dict[str, Any], methods: list[str], started_at: float) -> None:
    results["summary"] = summarize_rows(results["rows"], methods)
    results["summary"]["runtime_sec"] = round(time.time() - started_at, 1)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    started_at = time.time()
    output = Path(args.output)
    methods = parse_methods(args.methods)
    long_budget = adaptive_budget(args.max_new_tokens, args.repair_token_multiplier, args.max_attempt_tokens)

    print("humble_full_suite - default model baselines + verifier/dynamic/synthesis", flush=True)
    print("This is a benchmark harness, not a victory lap.", flush=True)
    print(f"Methods: {', '.join(methods)}", flush=True)

    examples, source, full_dataset = load_requested_examples(args.n)
    print(f"Examples: {len(examples)} from {source}", flush=True)

    results = None
    completed_indices: set[int] = set()
    if args.resume and output.exists():
        results = json.loads(output.read_text(encoding="utf-8"))
        completed_indices = {int(row["index"]) for row in results.get("rows", [])}
        print(f"Resuming {output}; completed rows: {len(completed_indices)}", flush=True)

    if results is None:
        results = {
            "model": args.model,
            "example_source": source,
            "n": len(examples),
            "full_dataset_requested": full_dataset,
            "methods": methods,
            "max_rounds": args.max_rounds,
            "required_agreement": args.required_agreement,
            "max_new_tokens": args.max_new_tokens,
            "repair_token_multiplier": args.repair_token_multiplier,
            "max_attempt_tokens": args.max_attempt_tokens,
            "adaptive_max_new_tokens": long_budget,
            "max_elapsed_sec": args.max_elapsed_sec,
            "answer_key_visible_to_verifier": False,
            "answer_key_use": "scoring_only_after_generation",
            "rows": [],
        }

    M = load_model(args.model, load_mode=args.load_mode)

    needs_dynamic = any(method in methods for method in ("humble_dynamic", "humble_synthesis"))
    vecs = None
    if needs_dynamic:
        print("\nExtracting dynamic branch vectors...", flush=True)
        vecs = build_domain_vecs(M)

    for i, ex in enumerate(examples):
        if i in completed_indices:
            continue

        q = ex["question"]
        answer = ex["answer"]
        print(f"\n[{i+1}/{len(examples)}] {q}", flush=True)
        row = {"index": i, "question": q, "methods": {}}

        if "legacy" in methods:
            row["methods"]["legacy"] = evaluate_generation(M, prompt_for(q), answer, args.max_new_tokens)
        if "compact" in methods:
            row["methods"]["compact"] = evaluate_generation(M, solve_prompt(q), answer, args.max_new_tokens)
        if "compact_long" in methods:
            row["methods"]["compact_long"] = evaluate_generation(M, solve_prompt(q), answer, long_budget)
        if "humble_verifier" in methods:
            row["methods"]["humble_verifier"] = evaluate_humble(
                M,
                q,
                answer,
                vecs=None,
                max_rounds=args.max_rounds,
                required_agreement=args.required_agreement,
                max_new_tokens=args.max_new_tokens,
                repair_token_multiplier=args.repair_token_multiplier,
                max_attempt_tokens=args.max_attempt_tokens,
                max_elapsed_sec=args.max_elapsed_sec,
                allow_synthesis=False,
            )
        if "humble_dynamic" in methods:
            row["methods"]["humble_dynamic"] = evaluate_humble(
                M,
                q,
                answer,
                vecs=vecs,
                max_rounds=args.max_rounds,
                required_agreement=args.required_agreement,
                max_new_tokens=args.max_new_tokens,
                repair_token_multiplier=args.repair_token_multiplier,
                max_attempt_tokens=args.max_attempt_tokens,
                max_elapsed_sec=args.max_elapsed_sec,
                allow_synthesis=False,
            )
        if "humble_synthesis" in methods:
            row["methods"]["humble_synthesis"] = evaluate_humble(
                M,
                q,
                answer,
                vecs=vecs,
                max_rounds=args.max_rounds,
                required_agreement=args.required_agreement,
                max_new_tokens=args.max_new_tokens,
                repair_token_multiplier=args.repair_token_multiplier,
                max_attempt_tokens=args.max_attempt_tokens,
                max_elapsed_sec=args.max_elapsed_sec,
                allow_synthesis=True,
            )

        for method in methods:
            item = row["methods"].get(method)
            if not item:
                continue
            extra = ""
            if method.startswith("humble"):
                extra = f" conf={item.get('confident')} reason={item.get('reason')}"
            print(
                f"  {method}: pred={item.get('pred')} gold={item.get('gold')} "
                f"correct={item.get('correct')} time={item.get('time_sec')}s{extra}",
                flush=True,
            )

        results["rows"].append(row)
        write_results(output, results, methods, started_at)

    write_results(output, results, methods, started_at)
    print("\nFinal summary:", flush=True)
    for method, summary in results["summary"]["methods"].items():
        acc = summary["accuracy"]
        acc_text = "n/a" if acc is None else f"{acc:.0%}"
        print(f"  {method}: {summary['correct']}/{summary['n']} ({acc_text})", flush=True)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
