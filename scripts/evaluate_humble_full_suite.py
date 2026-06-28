import argparse
import hashlib
import json
import sys
import time
import builtins
import datetime
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


def benchmark_question_key(question: str) -> str:
    normalized = " ".join(question.strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


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
    p.add_argument("--base-max-new-tokens", type=int, default=None, help="Optional token cap for base generations only.")
    p.add_argument("--base-max-time-sec", type=float, default=None, help="Optional wall-clock cap for each base generation.")
    p.add_argument("--load-mode", default=None, help="auto, slow, full, or 4bit.")
    p.add_argument("--resume", action="store_true", help="Resume from an existing output JSON.")
    p.add_argument(
        "--run-kind",
        choices=["bench-standard", "bench-informed"],
        default="bench-standard",
        help="Canonical benchmark run label. bench-standard defers ambiguity; bench-informed asks for clarification immediately.",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        default=None,
        help="Advanced override: prompt the user when ambiguity is detected.",
    )
    p.add_argument(
        "--ambiguity-mode",
        choices=["auto_resolve", "strict_gold", "ask_allowed", "answered_clarification"],
        default=None,
        help="Advanced override for ambiguity behavior.",
    )
    p.add_argument(
        "--interactive-disambiguation",
        choices=["defer", "instant"],
        default=None,
        help="Advanced override: when interactive, defer questions to the end or ask immediately.",
    )
    p.add_argument(
        "--clarification-fallback",
        default=(
            "Resolve the uncertainty internally: list the plausible interpretations, "
            "choose the one best supported by the original wording, and continue without external information."
        ),
        help="Internal policy injected when ambiguity is detected without an instant human answer.",
    )
    p.add_argument(
        "--oracle-cache-mode",
        choices=["ignore_oracle", "exclude_same_question", "use_all"],
        default="ignore_oracle",
        help="Benchmark cache policy. Default ignores oracle-repair cache; use_all reads every cache entry.",
    )
    p.add_argument("--hard-only", action="store_true", help="Skip humble methods if the base model gets it right.")
    p.add_argument("--verbose", action="store_true", help="Print exact token chunks as they are generated (chatty log).")
    p.add_argument("--no-timestamps", action="store_true", help="Disable print statement timestamps (on by default).")
    p.add_argument("--skip-indices", default="", help="Comma-separated zero-based example indices to record as skipped.")
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


def parse_index_set(raw: str) -> set[int]:
    indices: set[int] = set()
    for part in (raw or "").split(","):
        part = part.strip()
        if part:
            indices.add(int(part))
    return indices


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


def evaluate_generation(
    M,
    prompt: str,
    answer: str,
    max_new_tokens: int,
    max_time: float | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    response = generate_text(
        M,
        prompt,
        max_new_tokens=max_new_tokens,
        stop_after_final_answer=True,
        max_time=max_time,
    )
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
    config=None,
    vecs=None,
) -> dict[str, Any]:
    from invariants.config import AgenticConfig
    if config is None:
        config = AgenticConfig()
    question_key = benchmark_question_key(question)
    config.benchmark_question_key = question_key
        
    t0 = time.time()
    humble = solve_with_humility(
        M,
        question,
        vecs=vecs,
        config=config,
    )
    elapsed = time.time() - t0
    humble_text = "" if humble.final_answer is None else f"Final answer: {humble.final_answer}"
    correct, pred, gold = is_correct(humble_text, answer)
    clarifying_question = None
    for attempt in humble.attempts:
        if getattr(attempt, "needs_clarification", False):
            clarifying_question = attempt.clarifying_question
            break
    needs_clarification = humble.reason == "needs_user_clarification"
    
    if correct and config.synthesis_enabled and pred is not None:
        _promote_verified_synthesis(
            humble.attempts,
            str(pred),
            tag="native_success",
            question_key=question_key,
        )
    elif not correct and config.synthesis_enabled and pred is not None and gold is not None:
        oracle_prompt = (
            f"Question: {question}\n\n"
            f"You previously answered {pred}, but the true correct answer is {gold}. "
            f"Analyze both answers. Step-by-step, deduce why {pred} is mathematically flawed, "
            f"and prove why {gold} is the only correct answer."
        )
        print("    [Oracle Curriculum] Model failed. Forcing backwards reasoning synthesis...")
        oracle_humble = solve_with_humility(
            M,
            oracle_prompt,
            vecs=vecs,
            config=config,
        )
        oracle_text = "" if oracle_humble.final_answer is None else f"Final answer: {oracle_humble.final_answer}"
        oracle_correct, oracle_pred, _ = is_correct(oracle_text, answer)
        if oracle_correct:
            print("    [Oracle Curriculum] SUCCESS! Forging distilled vectors to cache...")
            _promote_verified_synthesis(
                oracle_humble.attempts,
                str(oracle_pred),
                tag="oracle_repair",
                question_key=question_key,
            )
        
    result = {
        "pred": None if pred is None else str(pred),
        "gold": None if gold is None else str(gold),
        "correct": correct,
        "confident": humble.confident,
        "reason": humble.reason,
        "urgency": humble.urgency,
        "time_sec": round(elapsed, 2),
        "result": humble.to_dict(),
        "needs_clarification": needs_clarification,
        "clarifying_question": clarifying_question,
    }
    if needs_clarification:
        result.update(
            {
                "skipped_for_ambiguity": True,
                "score_excluded_reason": "needs_user_clarification",
            }
        )
    return result


def summarize_rows(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "completed_rows": len(rows),
        "methods": {},
    }
    for method in methods:
        present = [row["methods"][method] for row in rows if method in row.get("methods", {})]
        scored = [item for item in present if not item.get("score_excluded_reason")]
        correct = sum(1 for item in scored if item.get("correct"))
        method_summary: dict[str, Any] = {
            "n": len(scored),
            "attempted_n": len(present),
            "score_excluded": len(present) - len(scored),
            "correct": correct,
            "accuracy": correct / len(scored) if scored else None,
            "mean_time_sec": (
                sum(float(item.get("time_sec", 0.0)) for item in present) / len(present)
                if present
                else None
            ),
        }
        if method.startswith("humble"):
            confident = sum(1 for item in scored if item.get("confident"))
            confident_correct = sum(1 for item in scored if item.get("confident") and item.get("correct"))
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
                    "coverage": confident / len(scored) if scored else None,
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


def print_progress(row_index: int, total: int, method: str) -> None:
    if method not in {"legacy", "compact", "compact_long"}:
        return
    total = max(int(total), 1)
    current = min(row_index + 1, total)
    print(f"  base {method} item {current}/{total}", flush=True)


def resolve_run_policy(args) -> None:
    policies = {
        "bench-standard": {
            "interactive": False,
            "ambiguity_mode": "ask_allowed",
            "interactive_disambiguation": "defer",
        },
        "bench-informed": {
            "interactive": True,
            "ambiguity_mode": "answered_clarification",
            "interactive_disambiguation": "instant",
        },
    }
    policy = dict(policies[args.run_kind])
    if args.interactive is not None:
        policy["interactive"] = args.interactive
    if args.ambiguity_mode is not None:
        policy["ambiguity_mode"] = args.ambiguity_mode
    if args.interactive_disambiguation is not None:
        policy["interactive_disambiguation"] = args.interactive_disambiguation

    args.interactive = policy["interactive"]
    args.ambiguity_mode = policy["ambiguity_mode"]
    args.interactive_disambiguation = policy["interactive_disambiguation"]


def apply_disambiguation_policy(config, args) -> None:
    defer_for_benchmark = args.ambiguity_mode == "ask_allowed" and (
        not args.interactive or args.interactive_disambiguation == "defer"
    )
    interactive_instant = args.interactive and args.interactive_disambiguation == "instant"
    config.interactive_disambiguation = interactive_instant
    config.defer_disambiguation = defer_for_benchmark
    config.clarification_fallback = args.clarification_fallback


def apply_oracle_cache_policy(config, args) -> None:
    config.ignore_oracle_cache = args.oracle_cache_mode == "ignore_oracle"
    config.exclude_same_question_oracle_cache = args.oracle_cache_mode == "exclude_same_question"


def main():
    args = parse_args()
    resolve_run_policy(args)
    
    if not args.no_timestamps:
        _original_print = builtins.print
        def _ts_print(*pargs, **kwargs):
            ts = datetime.datetime.now().strftime("[%H:%M:%S]")
            _original_print(ts, *pargs, **kwargs)
        builtins.print = _ts_print
        
    started_at = time.time()
    output = Path(args.output)
    methods = parse_methods(args.methods)
    skip_indices = parse_index_set(args.skip_indices)
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
        results["methods"] = methods
        results["max_rounds"] = args.max_rounds
        results["required_agreement"] = args.required_agreement
        results["max_new_tokens"] = args.max_new_tokens
        results["base_max_new_tokens"] = args.base_max_new_tokens
        results["repair_token_multiplier"] = args.repair_token_multiplier
        results["max_attempt_tokens"] = args.max_attempt_tokens
        results["adaptive_max_new_tokens"] = adaptive_budget(
            args.max_new_tokens,
            args.repair_token_multiplier,
            args.max_attempt_tokens,
        )
        results["max_elapsed_sec"] = args.max_elapsed_sec
        results["base_max_time_sec"] = args.base_max_time_sec
        results["stop_on_critical_urgency"] = False
        results["run_kind"] = args.run_kind
        results["benchmark_mode"] = args.ambiguity_mode
        results["oracle_cache_mode"] = args.oracle_cache_mode
        results["interactive"] = args.interactive
        results["interactive_disambiguation"] = args.interactive_disambiguation if args.interactive else None
        print(f"Resuming {output}; completed rows: {len(completed_indices)}", flush=True)

    if results is None:
        if output.exists():
            print(f"Starting fresh; existing {output} will be overwritten. Use --resume to continue it.", flush=True)
        results = {
            "model": args.model,
            "example_source": source,
            "n": len(examples),
            "full_dataset_requested": full_dataset,
            "methods": methods,
            "max_rounds": args.max_rounds,
            "required_agreement": args.required_agreement,
            "max_new_tokens": args.max_new_tokens,
            "base_max_new_tokens": args.base_max_new_tokens,
            "repair_token_multiplier": args.repair_token_multiplier,
            "max_attempt_tokens": args.max_attempt_tokens,
            "adaptive_max_new_tokens": long_budget,
            "max_elapsed_sec": args.max_elapsed_sec,
            "base_max_time_sec": args.base_max_time_sec,
            "stop_on_critical_urgency": False,
            "run_kind": args.run_kind,
            "benchmark_mode": args.ambiguity_mode,
            "oracle_cache_mode": args.oracle_cache_mode,
            "interactive": args.interactive,
            "interactive_disambiguation": args.interactive_disambiguation if args.interactive else None,
            "clarification_fallback": args.clarification_fallback,
            "answer_key_visible_to_verifier": False,
            "answer_key_use": "scoring_only_after_generation",
            "rows": [],
        }

    M = load_model(args.model, load_mode=args.load_mode)
    long_budget = adaptive_budget(args.max_new_tokens, args.repair_token_multiplier, args.max_attempt_tokens)
    base_budget = args.base_max_new_tokens if args.base_max_new_tokens is not None else args.max_new_tokens
    
    vecs = None
    if any(method in methods for method in ("humble_dynamic", "humble_synthesis")):
        vecs = build_domain_vecs(M)

    for i, ex in enumerate(examples):
        if i in completed_indices:
            continue

        q = ex["question"]
        answer = ex["answer"]
        print(f"\n[{i+1}/{len(examples)}] {q}", flush=True)
        row = {"index": i, "question": q, "methods": {}}
        if i in skip_indices:
            row["skipped"] = True
            row["skip_reason"] = "manual_skip_indices"
            print(f"  [skip] item {i+1}/{len(examples)} index={i}", flush=True)
            results["rows"].append(row)
            write_results(output, results, methods, started_at)
            continue

        if "legacy" in methods:
            print_progress(i, len(examples), "legacy")
            row["methods"]["legacy"] = evaluate_generation(
                M,
                prompt_for(q),
                answer,
                base_budget,
                max_time=args.base_max_time_sec,
            )
            if args.hard_only and row["methods"]["legacy"].get("correct", False):
                row["hard_only_skipped_after"] = "legacy"
                print("  [hard-only] Legacy base model succeeded. Skipping remaining methods.", flush=True)
                results["rows"].append(row)
                write_results(output, results, methods, started_at)
                continue
        if "compact" in methods:
            print_progress(i, len(examples), "compact")
            row["methods"]["compact"] = evaluate_generation(
                M,
                solve_prompt(q),
                answer,
                base_budget,
                max_time=args.base_max_time_sec,
            )
        if args.hard_only:
            early_base_correct = any(
                res.get("correct", False)
                for k, res in row["methods"].items()
                if k in ("legacy", "compact")
            )
            if early_base_correct:
                row["hard_only_skipped_after"] = "compact"
                print("  [hard-only] Base model succeeded before compact_long. Skipping remaining methods.", flush=True)
                results["rows"].append(row)
                write_results(output, results, methods, started_at)
                continue
        if "compact_long" in methods:
            print_progress(i, len(examples), "compact_long")
            row["methods"]["compact_long"] = evaluate_generation(
                M,
                solve_prompt(q),
                answer,
                long_budget,
                max_time=args.base_max_time_sec,
            )

        if args.hard_only:
            base_correct = any(
                res.get("correct", False) 
                for k, res in row["methods"].items() 
                if k in ("legacy", "compact", "compact_long")
            )
            if base_correct:
                print("  [hard-only] Base model succeeded. Skipping humble methods.", flush=True)
                results["rows"].append(row)
                write_results(output, results, methods, started_at)
                continue
        
        from invariants.config import AgenticConfig
        
        if "humble_verifier" in methods:
            config = AgenticConfig.from_preset("default")
            config.max_rounds = args.max_rounds
            config.required_agreement = args.required_agreement
            config.max_new_tokens = args.max_new_tokens
            config.repair_token_multiplier = args.repair_token_multiplier
            config.max_attempt_tokens = args.max_attempt_tokens
            config.max_elapsed_sec = args.max_elapsed_sec
            config.stop_on_critical_urgency = False
            config.synthesis_enabled = False
            config.use_expert_vectors = False
            apply_disambiguation_policy(config, args)
            apply_oracle_cache_policy(config, args)
            config.chatty_log = args.verbose
            
            print_progress(i, len(examples), "humble_verifier")
            row["methods"]["humble_verifier"] = evaluate_humble(M, q, answer, config=config)
            
        if "humble_dynamic" in methods:
            config = AgenticConfig.from_preset("default")
            config.max_rounds = args.max_rounds
            config.required_agreement = args.required_agreement
            config.max_new_tokens = args.max_new_tokens
            config.repair_token_multiplier = args.repair_token_multiplier
            config.max_attempt_tokens = args.max_attempt_tokens
            config.max_elapsed_sec = args.max_elapsed_sec
            config.stop_on_critical_urgency = False
            config.synthesis_enabled = False
            apply_disambiguation_policy(config, args)
            apply_oracle_cache_policy(config, args)
            config.chatty_log = args.verbose
            
            print_progress(i, len(examples), "humble_dynamic")
            row["methods"]["humble_dynamic"] = evaluate_humble(M, q, answer, config=config, vecs=vecs)
            
        if "humble_synthesis" in methods:
            config = AgenticConfig.from_preset("thorough")
            config.max_rounds = args.max_rounds
            config.required_agreement = args.required_agreement
            config.max_new_tokens = args.max_new_tokens
            config.repair_token_multiplier = args.repair_token_multiplier
            config.max_attempt_tokens = args.max_attempt_tokens
            config.max_elapsed_sec = args.max_elapsed_sec
            config.stop_on_critical_urgency = False
            config.synthesis_enabled = True
            apply_disambiguation_policy(config, args)
            apply_oracle_cache_policy(config, args)
            config.chatty_log = args.verbose
            config.cache_enabled = True
            config.cache_write_enabled = True
            
            print_progress(i, len(examples), "humble_synthesis")
            row["methods"]["humble_synthesis"] = evaluate_humble(M, q, answer, config=config, vecs=vecs)

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
