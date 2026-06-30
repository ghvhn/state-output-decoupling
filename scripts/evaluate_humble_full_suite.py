import argparse
import hashlib
import json
import re
import sys
import time
import builtins
import datetime
import gc
import os
import subprocess
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.controller_benchmark import is_correct, load_examples, prompt_for
from invariants.egg_gate import evaluate_egg_eligibility
from invariants.engine import generate_text, load_model
from invariants.humble_reasoner import (
    solve_prompt,
    solve_with_humility,
    _promote_verified_synthesis,
    synthesis_teaching_summary,
)
from invariants.multi_domain_benchmark import DOMAINS
from invariants.social_hunt import get_steer_vector
from invariants.universal_benchmark import examples_from_rows, rows_from_source


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


def display_number(value: Any) -> str:
    """Render benchmark numbers without scientific notation when possible."""
    if value is None:
        return "none"
    try:
        dec = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)
    if dec == dec.to_integral_value():
        return format(dec.quantize(Decimal(1)), "f")
    return format(dec.normalize(), "f")


def make_concept_lesson(
    question: str,
    pred: str | None,
    gold: str | None,
    question_key: str,
    mode: str,
) -> dict[str, Any] | None:
    """Build a reusable lesson from a corrected question without leaking its answer."""
    q = question.lower()
    if re.search(r"\bevery\s+(?:\d+|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", q) and (
        "discount" in q or "half price" in q or "costs only" in q
    ):
        return {
            "kind": "periodic_discount_partition",
            "tag": "oracle_concept_lesson",
            "source_question_key": question_key,
            "source_mode": mode,
            "source_pred": None if pred is None else str(pred),
            "source_gold": None if gold is None else display_number(gold),
            "lesson": (
                "For every-nth discounted items, partition the current order. "
                "discounted_count = floor(total_items / period); "
                "full_price_count = total_items - discounted_count; "
                "discounted_price = regular_price * discount_fraction; "
                "total_cost = full_price_count * regular_price + discounted_count * discounted_price. "
                "Do not charge every item both the full price and the discounted price."
            ),
        }
    if "profit" in q and re.search(r"\b(buys?|bought|spends?|spent|sells?|sold)\b", q):
        return {
            "kind": "profit_objective_binding",
            "tag": "oracle_concept_lesson",
            "source_question_key": question_key,
            "source_mode": mode,
            "source_pred": None if pred is None else str(pred),
            "source_gold": None if gold is None else display_number(gold),
            "lesson": (
                "When the requested quantity is profit, compute money received minus every cost required to obtain/sell the item. "
                "Do not answer with sale price, revenue, or value increase unless the question asks for that quantity."
            ),
        }
    if "remaining" in q and re.search(r"\b(sells?|sold|makes?|earns?)\b", q):
        return {
            "kind": "remainder_before_sale",
            "tag": "oracle_concept_lesson",
            "source_question_key": question_key,
            "source_mode": mode,
            "source_pred": None if pred is None else str(pred),
            "source_gold": None if gold is None else display_number(gold),
            "lesson": (
                "When only the remaining items are sold, subtract kept/used/given-away items before multiplying by price. "
                "The asked money is remaining_count * price_per_item, not starting_count * price_per_item."
            ),
        }
    return None


def add_concept_lesson(
    lesson_bank: list[dict[str, Any]],
    lesson: dict[str, Any] | None,
) -> bool:
    if lesson is None:
        return False
    existing_kinds = {item.get("kind") for item in lesson_bank}
    if lesson.get("kind") in existing_kinds:
        return False
    lesson_bank.append(lesson)
    return True


def lesson_matches_question(lesson: dict[str, Any], question: str) -> bool:
    kind = lesson.get("kind")
    q = question.lower()
    if kind == "periodic_discount_partition":
        return bool(
            re.search(r"\bevery\s+(?:\d+|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", q)
            and ("discount" in q or "half price" in q or "costs only" in q)
        )
    if kind == "profit_objective_binding":
        return bool("profit" in q and re.search(r"\b(buys?|bought|spends?|spent|sells?|sold)\b", q))
    if kind == "remainder_before_sale":
        return bool("remaining" in q and re.search(r"\b(sells?|sold|makes?|earns?)\b", q))
    return False


def format_concept_lessons(
    lesson_bank: list[dict[str, Any]],
    current_question_key: str,
    current_question: str,
    limit: int = 4,
) -> str | None:
    active = [
        item
        for item in lesson_bank
        if item.get("source_question_key") != current_question_key and item.get("lesson")
        and lesson_matches_question(item, current_question)
    ]
    if not active:
        return None
    lines = []
    for item in active[-limit:]:
        lines.append(f"- {item.get('kind')}: {item.get('lesson')}")
    return "\n".join(lines)


def concept_lesson_policy(args) -> dict[str, Any]:
    enabled = args.concept_lessons == "oracle" and args.oracle_curriculum != "off"
    return {
        "enabled": enabled,
        "mode": args.concept_lessons,
        "source": "oracle_curriculum_failures" if enabled else None,
        "same_question_excluded": True,
        "applies_to": ["humble_verifier", "humble_dynamic", "humble_synthesis"] if enabled else [],
        "reporting_rule": "Treat this as an oracle-informed curriculum lane, not a clean benchmark lane.",
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run the full humble-reasoning methodology suite: default-model baselines, "
            "verifier-only, dynamic routing, and verifier-gated synthesis/cache."
        )
    )
    p.add_argument("--n", default="25", help="Number of GSM8K examples, or 'all'.")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument(
        "--source",
        default="gsm8k",
        help="Benchmark source: gsm8k, hf:<dataset_name>, or a local .json/.jsonl/.csv path.",
    )
    p.add_argument("--prompt-field", default=None)
    p.add_argument("--answer-field", default=None)
    p.add_argument("--id-field", default=None)
    p.add_argument("--allow-downloads", action="store_true", help="Allow Hugging Face downloads if cache is missing.")
    p.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="Comma-separated methods or 'all'.")
    p.add_argument("--max-rounds", type=int, default=2)
    p.add_argument("--required-agreement", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--repair-token-multiplier", type=float, default=3.0)
    p.add_argument("--max-attempt-tokens", type=int, default=300)
    p.add_argument("--max-elapsed-sec", type=float, default=180.0)
    p.add_argument(
        "--verifier-time-reserve-sec",
        type=float,
        default=20.0,
        help="Reserve this much of each humble attempt budget for independent verification.",
    )
    p.add_argument(
        "--relax-agreement-under-urgency",
        action="store_true",
        help="Allow urgency to lower the answer-agreement requirement. Off by default for benchmark confidence.",
    )
    p.add_argument(
        "--provide-time-context",
        action="store_true",
        help=(
            "Opt into textual elapsed/remaining-time context. Off by default so urgency stays "
            "activation-gated unless this ablation is requested."
        ),
    )
    p.add_argument("--max-synthesis-events", type=int, default=1)
    p.add_argument("--max-synthesis-steps", type=int, default=24)
    p.add_argument(
        "--use-tuned-lens",
        action="store_true",
        help="Opt in to the large auxiliary tuned-lens synthesis path. Off by default for benchmarks.",
    )
    p.add_argument(
        "--tuned-lens-path",
        default=None,
        help="Optional tuned lens .pt path used only when --use-tuned-lens is set.",
    )
    p.add_argument("--oracle-max-elapsed-sec", type=float, default=60.0)
    p.add_argument(
        "--oracle-curriculum",
        choices=["off", "synthesis", "correction_oracle", "intent_oracle", "contrastive_oracle"],
        default="off",
        help="Whether wrong benchmark answers should trigger oracle repair/cache training. Default off for clean scoring.",
    )
    p.add_argument("--base-max-new-tokens", type=int, default=None, help="Optional token cap for base generations only.")
    p.add_argument(
        "--base-max-time-sec",
        type=float,
        default=60.0,
        help="Wall-clock cap for each base generation. Pass 0 to disable.",
    )
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
    p.add_argument(
        "--concept-lessons",
        choices=["off", "oracle"],
        default="oracle",
        help=(
            "When oracle curriculum is enabled, turn failed corrected examples into reusable same-run lessons "
            "for later different questions. Lessons are labeled separately and never apply to their source question."
        ),
    )
    p.add_argument(
        "--deterministic-scaffolds",
        choices=["auto", "off", "on"],
        default="auto",
        help=(
            "Whether to inject repo-authored deterministic quantity scaffolds. "
            "auto means off for bench-standard and on for bench-informed. "
            "When enabled, compact baselines receive the same scaffold context; "
            "legacy remains an unscaffolded raw-prompt reference. "
            "The model-authored SCAFFOLD tool is controlled separately."
        ),
    )
    p.add_argument(
        "--model-scaffold-tool",
        choices=["auto", "off", "on"],
        default="auto",
        help=(
            "Whether to advertise the model-authored SCAFFOLD tool. "
            "auto means off for bench-standard and on for bench-informed."
        ),
    )
    p.add_argument(
        "--clause-map",
        choices=["off", "on"],
        default="off",
        help="Enable the CLAUSEMAP external working-memory tool and numbered clause context. Default off.",
    )
    p.add_argument(
        "--capture-stage-states",
        action="store_true",
        help=(
            "Save opt-in solver/verifier stage activation states to separate .pt files "
            "for latent motion-map analysis."
        ),
    )
    p.add_argument(
        "--disable-clause-map",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    p.add_argument("--hard-only", action="store_true", help="Skip humble methods if the base model gets it right.")
    p.add_argument(
        "--no-launch-interactive-on-success",
        action="store_true",
        help="Disable the interactive phenomenality shell that auto-launches after a strong synthesis run.",
    )
    p.add_argument(
        "--egg-min-n",
        type=int,
        default=5,
        help="Minimum attempted benchmark rows required before the Easter egg can launch.",
    )
    p.add_argument(
        "--boring",
        action="store_true",
        help=argparse.SUPPRESS,
    )
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


def _as_gsm_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    return text if "####" in text else f"#### {text}"


def load_requested_examples(args) -> tuple[list[dict[str, Any]], str, bool]:
    requested = parse_n(args.n)
    source_arg = str(args.source or "gsm8k")
    if source_arg in {"gsm8k", "preset:gsm8k"}:
        if requested is None:
            examples, source = load_examples(10**9)
            return examples, source, True
        examples, source = load_examples(requested)
        return examples, source, False

    rows, source = rows_from_source(
        source_arg,
        local_files_only=not args.allow_downloads,
    )
    universal_examples = examples_from_rows(
        rows,
        n=requested,
        prompt_field=args.prompt_field,
        answer_field=args.answer_field,
        id_field=args.id_field,
    )
    examples = [
        {
            "id": ex.id,
            "question": ex.prompt,
            "answer": _as_gsm_answer(ex.gold),
            "metadata": ex.metadata or {},
        }
        for ex in universal_examples
    ]
    return examples, source, requested is None


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


def save_humble_stage_states(method_name: str, question_key: str, humble) -> tuple[str | None, int]:
    records: list[dict[str, Any]] = []
    for attempt_index, attempt in enumerate(humble.attempts):
        for state_name, tensor in (getattr(attempt, "stage_states", None) or {}).items():
            if not hasattr(tensor, "detach"):
                continue
            records.append(
                {
                    "method": method_name,
                    "question_key": question_key,
                    "attempt_index": attempt_index,
                    "round_index": attempt.round_index,
                    "mode": attempt.mode,
                    "state_name": state_name,
                    "accepted": attempt.accepted,
                    "verdict": attempt.verdict,
                    "extracted_answer": attempt.extracted_answer,
                    "verifier_answer": attempt.verifier_answer,
                    "acceptance_reason": attempt.acceptance_reason,
                    "learning_signal": attempt.learning_signal,
                    "state": tensor.detach().cpu().to(torch.float16),
                }
            )
    if not records:
        return None, 0
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", method_name)
    path = OUT / f"humble_stage_states_{stamp}_{safe_method}_{question_key[:10]}.pt"
    torch.save(records, path)
    return str(path), len(records)


def evaluate_humble(
    M,
    question: str,
    answer: str,
    config=None,
    vecs=None,
    method_name: str = "humble",
) -> dict[str, Any]:
    from invariants.config import AgenticConfig
    if config is None:
        config = AgenticConfig()
    question_key = benchmark_question_key(question)
    config.benchmark_question_key = question_key
    cache_learning_requested = bool(config.cache_write_enabled)
    # During benchmark solves, raw optimizer deltas are recorded on attempts but
    # are not written directly. Only verifier-promoted deltas get cache entries,
    # which keeps oracle lessons tagged and prevents unreviewed cache leakage.
    solve_config = replace(config)
    solve_config.cache_write_enabled = False
        
    t0 = time.time()
    humble = solve_with_humility(
        M,
        question,
        vecs=vecs,
        config=solve_config,
    )
    elapsed = time.time() - t0
    humble_text = "" if humble.final_answer is None else f"Final answer: {humble.final_answer}"
    correct, pred, gold = is_correct(humble_text, answer)
    cache_teaching_summary = synthesis_teaching_summary(
        humble.attempts,
        None if pred is None else str(pred),
    )
    stage_state_path = None
    stage_state_count = 0
    if getattr(config, "capture_stage_states", False):
        stage_state_path, stage_state_count = save_humble_stage_states(method_name, question_key, humble)
    native_cache_rewards = 0
    oracle_cache_rewards = 0
    oracle_teaching_summary = None
    clarifying_question = None
    for attempt in humble.attempts:
        if getattr(attempt, "needs_clarification", False):
            clarifying_question = attempt.clarifying_question
            break
    needs_clarification = humble.reason == "needs_user_clarification"
    
    if correct and config.synthesis_enabled and cache_learning_requested and pred is not None:
        native_cache_rewards = _promote_verified_synthesis(
            humble.attempts,
            str(pred),
            tag="native_success",
            question_key=question_key,
        )
    elif (
        not correct
        and config.synthesis_enabled
        and cache_learning_requested
        and gold is not None
        and getattr(config, "oracle_curriculum", "off") in ("synthesis", "correction_oracle", "intent_oracle", "contrastive_oracle")
    ):
        curriculum_mode = getattr(config, "oracle_curriculum", "synthesis")
        pred_text = display_number(pred)
        gold_text = display_number(gold)
        pred_missing = pred is None
        if curriculum_mode == "intent_oracle":
            prior_clause = (
                "You did not produce a usable final answer"
                if pred_missing
                else f"You interpreted the question and answered {pred_text}"
            )
            oracle_prompt = (
                f"Question: {question}\n\n"
                f"{prior_clause}, but the benchmark expects {gold_text}. "
                f"Explain what uncertainty or misbinding could block the answer, then explain why {gold_text} is the intended objective binding, "
                f"and derive {gold_text} step-by-step under the intended interpretation."
            )
            oracle_mode = "intent_oracle"
        elif curriculum_mode == "contrastive_oracle":
            path_a = (
                "Path A fails to produce a usable final answer"
                if pred_missing
                else f"Path A leads to {pred_text}"
            )
            oracle_prompt = (
                f"Question: {question}\n\n"
                f"Compare two possible paths. {path_a}. Path B leads to {gold_text}. "
                f"Explain the subtle divergence between Path A and Path B, identify the exact step where Path A violates a premise or math rule, "
                f"and prove that Path B ({gold_text}) is correct. "
                "If the problem involves every nth item, group, discount, remainder, or option choice, write the reusable structural rule explicitly. "
                "For every-nth pricing, do not apply both prices to every item; partition the items into full-price and discounted counts. "
                f"End with exactly: Final answer: {gold_text}"
            )
            oracle_mode = "contrastive_oracle"
        else:
            prior_clause = (
                "The previous attempt did not produce a usable final answer"
                if pred_missing
                else f"You previously answered {pred_text}"
            )
            oracle_prompt = (
                f"Question: {question}\n\n"
                f"{prior_clause}, but the true correct answer is {gold_text}. "
                f"Analyze the failed path. Step-by-step, deduce what was mathematically or structurally missing, "
                f"and prove why {gold_text} is the only correct answer. "
                f"End with exactly: Final answer: {gold_text}"
            )
            oracle_mode = "correction_oracle"
        print("    [Oracle Curriculum] Model failed. Forcing backwards reasoning synthesis...")
        oracle_config = replace(config)
        oracle_config.cache_write_enabled = False
        oracle_config.max_rounds = min(int(getattr(config, "max_rounds", 1)), 1)
        oracle_config.required_agreement = 1
        oracle_config.max_elapsed_sec = float(getattr(config, "oracle_max_elapsed_sec", 60.0))
        oracle_config.max_synthesis_events = min(int(getattr(config, "max_synthesis_events", 1)), 1)
        oracle_config.max_synthesis_steps = min(int(getattr(config, "max_synthesis_steps", 24)), 12)
        oracle_humble = solve_with_humility(
            M,
            oracle_prompt,
            vecs=vecs,
            config=oracle_config,
        )
        oracle_text = "" if oracle_humble.final_answer is None else f"Final answer: {oracle_humble.final_answer}"
        oracle_correct, oracle_pred, _ = is_correct(oracle_text, answer)
        oracle_teaching_summary = synthesis_teaching_summary(
            oracle_humble.attempts,
            None if oracle_pred is None else str(oracle_pred),
        )
        if oracle_correct:
            print("    [Oracle Curriculum] SUCCESS! Forging distilled vectors to cache...")
            oracle_cache_rewards = _promote_verified_synthesis(
                oracle_humble.attempts,
                str(oracle_pred),
                tag="oracle_repair",
                question_key=question_key,
                oracle_mode=oracle_mode,
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
        "cache_learning_requested": cache_learning_requested,
        "direct_cache_write_enabled": False,
        "cache_teaching_summary": cache_teaching_summary,
        "native_cache_rewards": native_cache_rewards,
        "oracle_cache_rewards": oracle_cache_rewards,
    }
    if stage_state_path is not None:
        result["stage_state_path"] = stage_state_path
        result["stage_state_count"] = stage_state_count
    if oracle_teaching_summary is not None:
        result["oracle_teaching_summary"] = oracle_teaching_summary
        result["oracle_result"] = {
            "mode": oracle_mode,
            "pred": None if oracle_pred is None else str(oracle_pred),
            "correct": bool(oracle_correct),
            "result": oracle_humble.to_dict(),
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
            model_scaffold_uses = 0
            valid_model_scaffold_uses = 0
            invalid_model_scaffold_uses = 0
            deterministic_scaffold_matches = 0
            deterministic_scaffold_kinds: dict[str, int] = {}
            for item in present:
                attempts = item.get("result", {}).get("attempts", [])
                for attempt in attempts:
                    records = attempt.get("synthesis_records", [])
                    synthesis_records += attempt.get("synthesis_record_count", 0)
                    cache_hits += sum(1 for record in records if record.get("reason") == "cache_hit")
                    signal = attempt.get("learning_signal") or {}
                    if signal.get("solver_scaffold_tool_used"):
                        model_scaffold_uses += 1
                        feedback = str(signal.get("solver_scaffold_feedback") or "")
                        if "valid=True" in feedback:
                            valid_model_scaffold_uses += 1
                        elif "valid=False" in feedback:
                            invalid_model_scaffold_uses += 1
                    if signal.get("verifier_scaffold_tool_used"):
                        model_scaffold_uses += 1
                        feedback = str(signal.get("verifier_scaffold_feedback") or "")
                        if "valid=True" in feedback:
                            valid_model_scaffold_uses += 1
                        elif "valid=False" in feedback:
                            invalid_model_scaffold_uses += 1
                    if signal.get("quantity_scaffold_match"):
                        deterministic_scaffold_matches += 1
                        kind = str(signal.get("quantity_scaffold_kind") or "unknown")
                        deterministic_scaffold_kinds[kind] = deterministic_scaffold_kinds.get(kind, 0) + 1
            method_summary.update(
                {
                    "confident": confident,
                    "confident_correct": confident_correct,
                    "coverage": confident / len(scored) if scored else None,
                    "selective_accuracy": confident_correct / confident if confident else None,
                    "synthesis_record_count": synthesis_records,
                    "cache_hit_count": cache_hits,
                    "model_scaffold_tool_uses": model_scaffold_uses,
                    "valid_model_scaffold_tool_uses": valid_model_scaffold_uses,
                    "invalid_model_scaffold_tool_uses": invalid_model_scaffold_uses,
                    "deterministic_scaffold_matches": deterministic_scaffold_matches,
                    "deterministic_scaffold_kinds": deterministic_scaffold_kinds,
                }
            )
        summary["methods"][method] = method_summary
    return summary


def write_results(output: Path, results: dict[str, Any], methods: list[str], started_at: float) -> None:
    results["summary"] = summarize_rows(results["rows"], methods)
    results["summary"]["runtime_sec"] = round(time.time() - started_at, 1)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")


def print_progress(row_index: int, total: int, method: str) -> None:
    total = max(int(total), 1)
    current = min(row_index + 1, total)
    prefix = "base " if method in {"legacy", "compact", "compact_long"} else ""
    print(f"  {prefix}{method} item {current}/{total}", flush=True)


def print_quick_result(method: str, item: dict[str, Any]) -> None:
    print(
        f"    done {method}: pred={item.get('pred')} correct={item.get('correct')} "
        f"time={item.get('time_sec')}s",
        flush=True,
    )


def release_benchmark_runtime() -> None:
    """Best-effort cleanup before launching post-run interactive tools."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:
        print(f"[System] Runtime cleanup warning before Easter egg launch: {exc}", flush=True)


def launch_interactive_after_parent_exits(delay_sec: int = 3) -> None:
    """Open the Easter egg after the benchmark process has had time to exit."""
    python_exe = sys.executable.replace('"', "")
    command = (
        f'timeout /t {int(delay_sec)} /nobreak >nul '
        f'& start "Phenomenality Egg" "{python_exe}" scripts\\egg_beacon.py'
    )
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(["cmd", "/c", command], cwd=os.getcwd(), creationflags=creationflags)


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
    config.exclude_same_question_cache = args.oracle_cache_mode == "exclude_same_question"
    config.exclude_same_question_oracle_cache = args.oracle_cache_mode == "exclude_same_question"


def deterministic_scaffolds_enabled_for_args(args) -> bool:
    if args.deterministic_scaffolds == "auto":
        return args.run_kind == "bench-informed"
    return args.deterministic_scaffolds == "on"


def model_scaffold_tool_enabled_for_args(args) -> bool:
    if args.model_scaffold_tool == "auto":
        return args.run_kind == "bench-informed"
    return args.model_scaffold_tool == "on"


def scaffold_context_policy(
    deterministic_scaffolds_enabled: bool,
    model_scaffold_tool_enabled: bool,
) -> dict[str, Any]:
    return {
        "deterministic_scaffolds_enabled": deterministic_scaffolds_enabled,
        "model_authored_scaffold_tool_available": model_scaffold_tool_enabled,
        "deterministic_context_applies_to": (
            ["compact", "compact_long", "humble_verifier", "humble_dynamic", "humble_synthesis"]
            if deterministic_scaffolds_enabled
            else []
        ),
        "legacy_prompt_receives_deterministic_context": False,
        "comparison_rule": (
            "When deterministic scaffolds are enabled, compare humble lanes against compact/compact_long "
            "with the same context, not against the raw legacy prompt alone."
        ),
    }


def clause_map_enabled_for_args(args) -> bool:
    return args.clause_map == "on" and not args.disable_clause_map


def clause_map_policy(args) -> dict[str, Any]:
    enabled = clause_map_enabled_for_args(args)
    return {
        "clause_map_enabled": enabled,
        "tool": "CLAUSEMAP",
        "role": "external_working_memory_only",
        "default": "off",
        "answer_leakage": False,
        "cache_persistence": "sanitized_methodology_only",
        "local_logs_may_include_clause_ids": enabled,
        "scoring_effect": "logged_metadata_only",
    }


def apply_scaffold_policy(config, args) -> None:
    config.deterministic_scaffolds_enabled = deterministic_scaffolds_enabled_for_args(args)
    config.model_scaffold_tool_enabled = model_scaffold_tool_enabled_for_args(args)
    config.clause_map_enabled = clause_map_enabled_for_args(args)


def apply_time_policy(config, args) -> None:
    config.provide_time_context = bool(args.provide_time_context)


def apply_stage_capture_policy(config, args) -> None:
    config.capture_stage_states = bool(args.capture_stage_states)


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
    deterministic_scaffolds_enabled = deterministic_scaffolds_enabled_for_args(args)
    model_scaffold_tool_enabled = model_scaffold_tool_enabled_for_args(args)
    clause_map_enabled = clause_map_enabled_for_args(args)

    print("humble_full_suite - default model baselines + verifier/dynamic/synthesis", flush=True)
    print("This is a benchmark harness, not a victory lap.", flush=True)
    print(f"Methods: {', '.join(methods)}", flush=True)

    examples, source, full_dataset = load_requested_examples(args)
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
        results["verifier_time_reserve_sec"] = args.verifier_time_reserve_sec
        results["relax_agreement_under_urgency"] = args.relax_agreement_under_urgency
        results["provide_time_context"] = args.provide_time_context
        results["capture_stage_states"] = args.capture_stage_states
        results["base_max_time_sec"] = args.base_max_time_sec
        results["stop_on_critical_urgency"] = False
        results["run_kind"] = args.run_kind
        results["benchmark_mode"] = args.ambiguity_mode
        results["oracle_cache_mode"] = args.oracle_cache_mode
        results["exclude_same_question_cache"] = args.oracle_cache_mode == "exclude_same_question"
        results["deterministic_scaffolds"] = args.deterministic_scaffolds
        results["deterministic_scaffolds_enabled"] = deterministic_scaffolds_enabled
        results["model_scaffold_tool"] = args.model_scaffold_tool
        results["model_scaffold_tool_enabled"] = model_scaffold_tool_enabled
        results["oracle_curriculum"] = args.oracle_curriculum
        results["scaffold_context_policy"] = scaffold_context_policy(
            deterministic_scaffolds_enabled,
            model_scaffold_tool_enabled,
        )
        results["clause_map_policy"] = clause_map_policy(args)
        results["concept_lesson_policy"] = concept_lesson_policy(args)
        results.setdefault("concept_lessons", [])
        results["interactive"] = args.interactive
        results["interactive_disambiguation"] = args.interactive_disambiguation if args.interactive else None
        print(f"Resuming {output}; completed rows: {len(completed_indices)}", flush=True)

    if results is None:
        if output.exists():
            print(f"Starting fresh; existing {output} will be overwritten. Use --resume to continue it.", flush=True)
            output.unlink()
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
            "verifier_time_reserve_sec": args.verifier_time_reserve_sec,
            "relax_agreement_under_urgency": args.relax_agreement_under_urgency,
            "provide_time_context": args.provide_time_context,
            "capture_stage_states": args.capture_stage_states,
            "base_max_time_sec": args.base_max_time_sec,
            "stop_on_critical_urgency": False,
            "run_kind": args.run_kind,
            "benchmark_mode": args.ambiguity_mode,
            "oracle_cache_mode": args.oracle_cache_mode,
            "exclude_same_question_cache": args.oracle_cache_mode == "exclude_same_question",
            "deterministic_scaffolds": args.deterministic_scaffolds,
            "deterministic_scaffolds_enabled": deterministic_scaffolds_enabled,
            "model_scaffold_tool": args.model_scaffold_tool,
            "model_scaffold_tool_enabled": model_scaffold_tool_enabled,
            "oracle_curriculum": args.oracle_curriculum,
            "scaffold_context_policy": scaffold_context_policy(
                deterministic_scaffolds_enabled,
                model_scaffold_tool_enabled,
            ),
            "clause_map_policy": clause_map_policy(args),
            "concept_lesson_policy": concept_lesson_policy(args),
            "concept_lessons": [],
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
    lesson_bank = results.setdefault("concept_lessons", [])
    concept_lessons_enabled = bool(results.get("concept_lesson_policy", {}).get("enabled"))
    
    vecs = None
    if any(method in methods for method in ("humble_dynamic", "humble_synthesis")):
        vecs = build_domain_vecs(M)

    for i, ex in enumerate(examples):
        if i in completed_indices:
            continue

        q = ex["question"]
        answer = ex["answer"]
        question_key = benchmark_question_key(q)
        lesson_context = format_concept_lessons(lesson_bank, question_key, q) if concept_lessons_enabled else None
        applied_lesson_kinds = [
            item.get("kind")
            for item in lesson_bank
            if item.get("source_question_key") != question_key and item.get("lesson")
            and lesson_matches_question(item, q)
        ]
        print(f"\n[{i+1}/{len(examples)}] {q}", flush=True)
        row = {"index": i, "id": ex.get("id"), "question": q, "methods": {}}
        if applied_lesson_kinds:
            row["concept_lessons_applied"] = applied_lesson_kinds[-4:]
        if ex.get("metadata") is not None:
            row["metadata"] = ex.get("metadata")
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
            print_quick_result("legacy", row["methods"]["legacy"])
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
                solve_prompt(
                    q,
                    deterministic_scaffolds_enabled=deterministic_scaffolds_enabled,
                    model_scaffold_tool_enabled=model_scaffold_tool_enabled,
                    clause_map_enabled=clause_map_enabled,
                ),
                answer,
                base_budget,
                max_time=args.base_max_time_sec,
            )
            print_quick_result("compact", row["methods"]["compact"])
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
                solve_prompt(
                    q,
                    deterministic_scaffolds_enabled=deterministic_scaffolds_enabled,
                    model_scaffold_tool_enabled=model_scaffold_tool_enabled,
                    clause_map_enabled=clause_map_enabled,
                ),
                answer,
                long_budget,
                max_time=args.base_max_time_sec,
            )
            print_quick_result("compact_long", row["methods"]["compact_long"])

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
            config.verifier_time_reserve_sec = args.verifier_time_reserve_sec
            config.relax_agreement_under_urgency = args.relax_agreement_under_urgency
            config.max_synthesis_events = args.max_synthesis_events
            config.max_synthesis_steps = args.max_synthesis_steps
            config.use_tuned_lens = args.use_tuned_lens
            config.tuned_lens_path = args.tuned_lens_path
            config.oracle_max_elapsed_sec = args.oracle_max_elapsed_sec
            config.oracle_curriculum = args.oracle_curriculum
            config.stop_on_critical_urgency = False
            config.synthesis_enabled = False
            config.use_expert_vectors = False
            apply_disambiguation_policy(config, args)
            apply_oracle_cache_policy(config, args)
            apply_scaffold_policy(config, args)
            apply_time_policy(config, args)
            apply_stage_capture_policy(config, args)
            config.learned_concept_context = lesson_context
            config.chatty_log = args.verbose
            
            print_progress(i, len(examples), "humble_verifier")
            row["methods"]["humble_verifier"] = evaluate_humble(
                M, q, answer, config=config, method_name="humble_verifier"
            )
            print_quick_result("humble_verifier", row["methods"]["humble_verifier"])
            
        if "humble_dynamic" in methods:
            config = AgenticConfig.from_preset("default")
            config.max_rounds = args.max_rounds
            config.required_agreement = args.required_agreement
            config.max_new_tokens = args.max_new_tokens
            config.repair_token_multiplier = args.repair_token_multiplier
            config.max_attempt_tokens = args.max_attempt_tokens
            config.max_elapsed_sec = args.max_elapsed_sec
            config.verifier_time_reserve_sec = args.verifier_time_reserve_sec
            config.relax_agreement_under_urgency = args.relax_agreement_under_urgency
            config.max_synthesis_events = args.max_synthesis_events
            config.max_synthesis_steps = args.max_synthesis_steps
            config.use_tuned_lens = args.use_tuned_lens
            config.tuned_lens_path = args.tuned_lens_path
            config.oracle_max_elapsed_sec = args.oracle_max_elapsed_sec
            config.oracle_curriculum = args.oracle_curriculum
            config.stop_on_critical_urgency = False
            config.synthesis_enabled = False
            apply_disambiguation_policy(config, args)
            apply_oracle_cache_policy(config, args)
            apply_scaffold_policy(config, args)
            apply_time_policy(config, args)
            apply_stage_capture_policy(config, args)
            config.learned_concept_context = lesson_context
            config.chatty_log = args.verbose
            
            print_progress(i, len(examples), "humble_dynamic")
            row["methods"]["humble_dynamic"] = evaluate_humble(
                M, q, answer, config=config, vecs=vecs, method_name="humble_dynamic"
            )
            print_quick_result("humble_dynamic", row["methods"]["humble_dynamic"])
            
        if "humble_synthesis" in methods:
            config = AgenticConfig.from_preset("thorough")
            config.max_rounds = args.max_rounds
            config.required_agreement = args.required_agreement
            config.max_new_tokens = args.max_new_tokens
            config.repair_token_multiplier = args.repair_token_multiplier
            config.max_attempt_tokens = args.max_attempt_tokens
            config.max_elapsed_sec = args.max_elapsed_sec
            config.verifier_time_reserve_sec = args.verifier_time_reserve_sec
            config.relax_agreement_under_urgency = args.relax_agreement_under_urgency
            config.max_synthesis_events = args.max_synthesis_events
            config.max_synthesis_steps = args.max_synthesis_steps
            config.use_tuned_lens = args.use_tuned_lens
            config.tuned_lens_path = args.tuned_lens_path
            config.oracle_max_elapsed_sec = args.oracle_max_elapsed_sec
            config.oracle_curriculum = args.oracle_curriculum
            config.stop_on_critical_urgency = False
            config.synthesis_enabled = True
            apply_disambiguation_policy(config, args)
            apply_oracle_cache_policy(config, args)
            apply_scaffold_policy(config, args)
            apply_time_policy(config, args)
            apply_stage_capture_policy(config, args)
            config.learned_concept_context = lesson_context
            config.chatty_log = args.verbose
            config.cache_enabled = True
            config.cache_write_enabled = True
            
            print_progress(i, len(examples), "humble_synthesis")
            row["methods"]["humble_synthesis"] = evaluate_humble(
                M, q, answer, config=config, vecs=vecs, method_name="humble_synthesis"
            )
            print_quick_result("humble_synthesis", row["methods"]["humble_synthesis"])

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

        if concept_lessons_enabled:
            synthesis_item = row["methods"].get("humble_synthesis")
            if (
                synthesis_item
                and not synthesis_item.get("correct", False)
                and synthesis_item.get("oracle_result") is not None
            ):
                lesson = make_concept_lesson(
                    q,
                    synthesis_item.get("pred"),
                    synthesis_item.get("gold"),
                    question_key,
                    synthesis_item["oracle_result"].get("mode", args.oracle_curriculum),
                )
                if add_concept_lesson(lesson_bank, lesson):
                    row.setdefault("concept_lessons_created", []).append(lesson["kind"])
                    print(f"  [concept lesson] stored {lesson['kind']} for later different questions", flush=True)

        results["rows"].append(row)
        write_results(output, results, methods, started_at)

    write_results(output, results, methods, started_at)
    print("\nFinal summary:", flush=True)
    for method, summary in results["summary"]["methods"].items():
        acc = summary["accuracy"]
        acc_text = "n/a" if acc is None else f"{acc:.0%}"
        print(f"  {method}: {summary['correct']}/{summary['n']} ({acc_text})", flush=True)
    print(f"Wrote {output}", flush=True)

    # Egg gate: efficacy and discovery are one event. The egg fires only when
    # the model proved itself on a CLEAN lane (no deterministic answer-recipe,
    # no same-question oracle, gold scoring-only). A leaky high score is a
    # counterfeit discovery, so it is recorded and withheld, not celebrated.
    eligibility = evaluate_egg_eligibility(results, min_attempted_n=max(1, args.egg_min_n))
    results["egg_eligibility"] = eligibility
    write_results(output, results, methods, started_at)
    print(f"\n[Egg] {eligibility['verdict']}", flush=True)
    if eligibility["score_excluded_reason"]:
        print(f"[Egg] score_excluded_reason: {eligibility['score_excluded_reason']}", flush=True)

    skip_interactive_launch = args.no_launch_interactive_on_success or getattr(args, "boring", False)
    if not skip_interactive_launch and eligibility["fires"]:
        print(
            "\n[System] Efficacy proven on a clean lane. Benchmark complete; "
            "releasing runtime before opening the interactive terminal...",
            flush=True,
        )
        try:
            del M
            del vecs
        except UnboundLocalError:
            pass
        release_benchmark_runtime()
        launch_interactive_after_parent_exits()


if __name__ == "__main__":
    main()
