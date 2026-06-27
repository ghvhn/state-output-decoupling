import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.controller_benchmark import is_correct, load_examples, predicted_answer, prompt_for
from invariants.engine import generate_text, load_model
from invariants.humble_reasoner import solve_prompt, solve_with_humility
from invariants.multi_domain_benchmark import DOMAINS
from invariants.social_hunt import get_steer_vector


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent.parent / "invariants" / "out"
OUT.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="Number of GSM8K examples.")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--required-agreement", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument(
        "--repair-token-multiplier",
        type=float,
        default=2.0,
        help="Token multiplier for continuation/repair/dynamic attempts after the first solve.",
    )
    p.add_argument(
        "--max-attempt-tokens",
        type=int,
        default=None,
        help="Optional cap for any single humble attempt after adaptive scaling.",
    )
    p.add_argument(
        "--skip-long-baseline",
        action="store_true",
        help="Skip the compact baseline run with the same larger adaptive token budget.",
    )
    p.add_argument("--max-elapsed-sec", type=float, default=180.0)
    p.add_argument("--no-dynamic", action="store_true", help="Use verifier/repair loop without dynamic layer attempts.")
    p.add_argument("--allow-synthesis", action="store_true", help="Allow verifier-gated synthesis inside dynamic repair attempts.")
    p.add_argument("--no-synthesis", action="store_true", help="Compatibility alias; synthesis is disabled unless --allow-synthesis is set.")
    p.add_argument("--load-mode", default=None, help="auto, slow, full, or 4bit.")
    p.add_argument("--output", default=str(OUT / "humble_dynamic_benchmark.json"))
    return p.parse_args()


def build_domain_vecs(M):
    vecs = {}
    for name, spec in DOMAINS.items():
        vecs[name] = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        print(f"  {name}: L{spec['layer']} norm={vecs[name].norm():.2f}", flush=True)
    return vecs


def adaptive_budget(base_tokens: int, multiplier: float, cap: int | None) -> int:
    budget = max(int(base_tokens), int(round(base_tokens * max(1.0, multiplier))))
    if cap is not None and cap > 0:
        budget = min(budget, int(cap))
    return max(1, budget)


def main():
    args = parse_args()
    t0 = time.time()
    output = Path(args.output)

    print("humble_dynamic_benchmark - baseline vs verifier-driven dynamic reasoning", flush=True)
    print("This is a benchmark harness, not a victory lap.", flush=True)

    examples, source = load_examples(args.n)
    M = load_model(MODEL_NAME, load_mode=args.load_mode)

    vecs = None
    if not args.no_dynamic:
        print("\nExtracting dynamic branch vectors...", flush=True)
        vecs = build_domain_vecs(M)
    allow_synthesis = bool(args.allow_synthesis and not args.no_synthesis)
    long_budget = adaptive_budget(args.max_new_tokens, args.repair_token_multiplier, args.max_attempt_tokens)
    run_long_baseline = (not args.skip_long_baseline) and long_budget != args.max_new_tokens

    results = {
        "model": MODEL_NAME,
        "example_source": source,
        "n": len(examples),
        "max_rounds": args.max_rounds,
        "required_agreement": args.required_agreement,
        "max_new_tokens": args.max_new_tokens,
        "repair_token_multiplier": args.repair_token_multiplier,
        "max_attempt_tokens": args.max_attempt_tokens,
        "adaptive_max_new_tokens": long_budget,
        "long_compact_baseline_enabled": run_long_baseline,
        "answer_key_visible_to_verifier": False,
        "answer_key_use": "scoring_only_after_generation",
        "max_elapsed_sec": args.max_elapsed_sec,
        "dynamic_enabled": vecs is not None,
        "synthesis_enabled": vecs is not None and allow_synthesis,
        "rows": [],
    }

    legacy_baseline_correct = 0
    compact_baseline_correct = 0
    compact_long_baseline_correct = 0
    humble_correct = 0
    humble_confident = 0
    humble_confident_correct = 0

    for i, ex in enumerate(examples):
        q = ex["question"]
        print(f"\n[{i+1}/{len(examples)}] {q}", flush=True)

        legacy_t0 = time.time()
        legacy_response = generate_text(M, prompt_for(q), max_new_tokens=args.max_new_tokens)
        legacy_time = time.time() - legacy_t0
        # The gold answer is used only for scoring after model/verifier calls finish.
        legacy_ok, legacy_pred, gold = is_correct(legacy_response, ex["answer"])
        legacy_baseline_correct += int(legacy_ok)

        compact_t0 = time.time()
        compact_response = generate_text(M, solve_prompt(q), max_new_tokens=args.max_new_tokens)
        compact_time = time.time() - compact_t0
        compact_ok, compact_pred, _ = is_correct(compact_response, ex["answer"])
        compact_baseline_correct += int(compact_ok)

        compact_long = None
        if run_long_baseline:
            compact_long_t0 = time.time()
            compact_long_response = generate_text(M, solve_prompt(q), max_new_tokens=long_budget)
            compact_long_time = time.time() - compact_long_t0
            compact_long_ok, compact_long_pred, _ = is_correct(compact_long_response, ex["answer"])
            compact_long_baseline_correct += int(compact_long_ok)
            compact_long = {
                "pred": None if compact_long_pred is None else str(compact_long_pred),
                "correct": compact_long_ok,
                "time_sec": round(compact_long_time, 2),
                "token_budget": long_budget,
                "response": compact_long_response,
            }

        humble_t0 = time.time()
        humble = solve_with_humility(
            M,
            q,
            vecs=vecs,
            max_rounds=args.max_rounds,
            required_agreement=args.required_agreement,
            max_new_tokens=args.max_new_tokens,
            allow_synthesis=allow_synthesis,
            max_elapsed_sec=args.max_elapsed_sec,
            repair_token_multiplier=args.repair_token_multiplier,
            max_attempt_tokens=args.max_attempt_tokens,
        )
        humble_time = time.time() - humble_t0
        humble_text = "" if humble.final_answer is None else f"Final answer: {humble.final_answer}"
        humble_ok, humble_pred, _ = is_correct(humble_text, ex["answer"])
        humble_correct += int(humble_ok)
        humble_confident += int(humble.confident)
        humble_confident_correct += int(humble.confident and humble_ok)

        row = {
            "index": i,
            "question": q,
            "gold": str(gold),
            "legacy_baseline": {
                "pred": None if legacy_pred is None else str(legacy_pred),
                "correct": legacy_ok,
                "time_sec": round(legacy_time, 2),
                "response": legacy_response,
            },
            "compact_baseline": {
                "pred": None if compact_pred is None else str(compact_pred),
                "correct": compact_ok,
                "time_sec": round(compact_time, 2),
                "token_budget": args.max_new_tokens,
                "response": compact_response,
            },
            "compact_long_baseline": compact_long,
            "humble_dynamic": {
                "pred": None if humble_pred is None else str(humble_pred),
                "correct": humble_ok,
                "confident": humble.confident,
                "reason": humble.reason,
                "urgency": humble.urgency,
                "time_sec": round(humble_time, 2),
                "result": humble.to_dict(),
            },
        }
        results["rows"].append(row)

        print(
            f"  legacy:  pred={legacy_pred} gold={gold} correct={legacy_ok} time={legacy_time:.1f}s",
            flush=True,
        )
        print(
            f"  compact: pred={compact_pred} gold={gold} correct={compact_ok} time={compact_time:.1f}s",
            flush=True,
        )
        if compact_long is not None:
            print(
                f"  compact+: pred={compact_long['pred']} gold={gold} correct={compact_long['correct']} "
                f"budget={long_budget} time={compact_long['time_sec']:.1f}s",
                flush=True,
            )
        print(
            f"  humble:   pred={humble_pred} gold={gold} correct={humble_ok} "
            f"conf={humble.confident} reason={humble.reason} time={humble_time:.1f}s",
            flush=True,
        )

        output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    results["summary"] = {
        "legacy_baseline_correct": legacy_baseline_correct,
        "compact_baseline_correct": compact_baseline_correct,
        "compact_long_baseline_correct": compact_long_baseline_correct if run_long_baseline else None,
        "humble_correct": humble_correct,
        "legacy_baseline_accuracy": legacy_baseline_correct / max(len(examples), 1),
        "compact_baseline_accuracy": compact_baseline_correct / max(len(examples), 1),
        "compact_long_baseline_accuracy": (
            compact_long_baseline_correct / max(len(examples), 1) if run_long_baseline else None
        ),
        "humble_accuracy": humble_correct / max(len(examples), 1),
        "humble_confident": humble_confident,
        "humble_confident_correct": humble_confident_correct,
        "humble_coverage": humble_confident / max(len(examples), 1),
        "humble_selective_accuracy": (
            humble_confident_correct / humble_confident if humble_confident else None
        ),
        "runtime_sec": round(time.time() - t0, 1),
    }
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    s = results["summary"]
    print("\nFinal summary:", flush=True)
    print(f"  legacy:  {s['legacy_baseline_correct']}/{len(examples)} ({s['legacy_baseline_accuracy']:.0%})", flush=True)
    print(f"  compact: {s['compact_baseline_correct']}/{len(examples)} ({s['compact_baseline_accuracy']:.0%})", flush=True)
    if run_long_baseline:
        print(
            f"  compact+: {s['compact_long_baseline_correct']}/{len(examples)} "
            f"({s['compact_long_baseline_accuracy']:.0%})",
            flush=True,
        )
    print(f"  humble:   {s['humble_correct']}/{len(examples)} ({s['humble_accuracy']:.0%})", flush=True)
    print(
        f"  humble confident: {s['humble_confident_correct']}/{s['humble_confident']} "
        f"(coverage={s['humble_coverage']:.0%})",
        flush=True,
    )
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
