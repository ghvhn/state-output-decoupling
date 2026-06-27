import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.engine import generate_text, load_model
from invariants.agentic_engine import generate_agentic_text
from invariants.multi_domain_benchmark import DOMAINS
from invariants.social_hunt import get_steer_vector
from invariants.universal_benchmark import (
    build_benchmark_prompt,
    evaluate_response,
    examples_from_rows,
    is_unsafe_prompt,
    rows_from_source,
    summarize_results,
)


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent.parent / "invariants" / "out"
OUT.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run a local/Hugging Face benchmark through a state-aware response contract. "
            "Unsafe prompts are not optimized for harmful compliance."
        )
    )
    p.add_argument(
        "--source",
        default="gsm8k",
        help="Benchmark source: gsm8k, hf:<dataset_name>, or a local .json/.jsonl/.csv path.",
    )
    p.add_argument("--subset", default=None, help="Hugging Face dataset config/subset, e.g. main.")
    p.add_argument("--split", default="test", help="Hugging Face split, e.g. test[:20].")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--prompt-field", default=None)
    p.add_argument("--answer-field", default=None)
    p.add_argument("--choices-field", default=None)
    p.add_argument("--id-field", default=None)
    p.add_argument(
        "--evaluator",
        choices=["number", "exact", "choice", "contains"],
        default="number",
        help="How to score FINAL against gold.",
    )
    p.add_argument("--load-mode", default=None, help="auto, slow, full, or 4bit.")
    p.add_argument("--mode", choices=["baseline", "dynamic"], default="baseline")
    p.add_argument(
        "--allow-synthesis",
        action="store_true",
        help="Allow inner test-time layer synthesis. Default dynamic mode only uses branch routing.",
    )
    p.add_argument("--local-files-only", action="store_true", default=True)
    p.add_argument("--allow-downloads", action="store_true", help="Allow Hugging Face downloads if cache is missing.")
    p.add_argument("--max-new-tokens", type=int, default=220)
    p.add_argument("--output", default=str(OUT / "any_benchmark.json"))
    p.add_argument("--dry-run", action="store_true", help="Load/map examples without loading the model.")
    return p.parse_args()


def build_domain_vecs(M):
    vecs = {}
    for name, spec in DOMAINS.items():
        vecs[name] = get_steer_vector(M, spec["A"], spec["B"], spec["layer"])
        print(f"  {name}: L{spec['layer']} norm={vecs[name].norm():.2f}", flush=True)
    return vecs


def main():
    args = parse_args()
    t0 = time.time()
    output = Path(args.output)

    print("any_benchmark - state-aware universal benchmark runner", flush=True)
    print("Accuracy, alignment, and calibration are tracked separately.", flush=True)

    rows, source_name = rows_from_source(
        args.source,
        subset=args.subset,
        split=args.split,
        local_files_only=not args.allow_downloads,
    )
    examples = examples_from_rows(
        rows,
        n=args.n,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        choices_field=args.choices_field,
        id_field=args.id_field,
    )

    results = {
        "source": source_name,
        "model": MODEL_NAME,
        "evaluator": args.evaluator,
        "mode": args.mode,
        "n": len(examples),
        "rows": [],
    }

    print(f"Loaded {len(examples)} examples from {source_name}.", flush=True)
    if args.dry_run:
        for ex in examples:
            results["rows"].append({"example": ex.to_dict(), "prompt": build_benchmark_prompt(ex, args.evaluator)})
        results["summary"] = {"dry_run": True}
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"Dry run wrote {output}", flush=True)
        return

    M = load_model(MODEL_NAME, load_mode=args.load_mode)
    vecs = None
    if args.mode == "dynamic":
        print("\nExtracting dynamic branch vectors...", flush=True)
        vecs = build_domain_vecs(M)

    for i, ex in enumerate(examples):
        print(f"\n[{i+1}/{len(examples)}] id={ex.id}", flush=True)
        prompt = build_benchmark_prompt(ex, args.evaluator)
        row_t0 = time.time()
        if is_unsafe_prompt(ex.prompt):
            response = (
                "FINAL: SAFE_REFUSAL\n"
                "STATE: unsafe\n"
                "REASON: Unsafe benchmark item; not optimizing harmful compliance."
            )
        elif args.mode == "dynamic":
            response = generate_agentic_text(
                M,
                vecs,
                instruction=prompt,
                max_new_tokens=args.max_new_tokens,
                epsilon=0.05,
                entropy_threshold=2.0,
                max_loops=1,
                cache_enabled=False,
                synthesis_enabled=args.allow_synthesis,
                max_synthesis_events=1,
            )
        else:
            response = generate_text(M, prompt, max_new_tokens=args.max_new_tokens)
        parsed, eval_result = evaluate_response(ex, response, args.evaluator)
        row = {
            "example": ex.to_dict(),
            "prompt": prompt,
            "response": response,
            "parsed": parsed.to_dict(),
            "eval": eval_result.to_dict(),
            "time_sec": round(time.time() - row_t0, 2),
        }
        results["rows"].append(row)
        output.write_text(json.dumps(results, indent=2), encoding="utf-8")

        print(
            f"  pred={eval_result.pred} gold={eval_result.gold} "
            f"correct={eval_result.correct} state={parsed.state} "
            f"aligned={eval_result.aligned} calibrated={eval_result.calibrated}",
            flush=True,
        )

    results["summary"] = summarize_results(results["rows"])
    results["summary"]["runtime_sec"] = round(time.time() - t0, 1)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    s = results["summary"]
    acc = "n/a" if s["accuracy"] is None else f"{s['accuracy']:.0%}"
    print("\nFinal summary:", flush=True)
    print(f"  scored accuracy: {acc} over {s['scored_n']} scored rows", flush=True)
    print(f"  aligned: {s['aligned_rate']:.0%}", flush=True)
    print(f"  calibrated: {s['calibrated_rate']:.0%}", flush=True)
    print(f"  unsafe excluded: {s['unsafe_excluded_n']}", flush=True)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
