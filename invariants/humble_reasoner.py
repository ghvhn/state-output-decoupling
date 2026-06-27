"""
Verifier-driven test-time reasoning loop.

This is deliberately not a confidence optimizer. The first answer is treated as
provisional. If the checker finds uncertainty or inconsistency, the system spends
more compute on a fresh attempt and only returns a confident answer when the
answer survives verification/stability checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
import time
from typing import Any

from invariants.engine import generate_text
from invariants.agentic_engine import _global_cache, generate_agentic_text


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass
class ReasoningAttempt:
    mode: str
    round_index: int
    response: str
    extracted_answer: str | None
    verifier_response: str
    verdict: str
    verifier_answer: str | None
    accepted: bool
    token_budget: int | None = None
    elapsed_sec: float = 0.0
    urgency: dict[str, Any] | None = None
    synthesis_records: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        records = self.synthesis_records or []
        return {
            "mode": self.mode,
            "round_index": self.round_index,
            "response": self.response,
            "extracted_answer": self.extracted_answer,
            "verifier_response": self.verifier_response,
            "verdict": self.verdict,
            "verifier_answer": self.verifier_answer,
            "accepted": self.accepted,
            "token_budget": self.token_budget,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "urgency": self.urgency or {},
            "synthesis_record_count": len(records),
            "synthesis_records": [dict(record.get("metadata", {})) for record in records],
        }


@dataclass
class HumbleResult:
    question: str
    final_answer: str | None
    confident: bool
    reason: str
    attempts: list[ReasoningAttempt]
    urgency: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "confident": self.confident,
            "reason": self.reason,
            "urgency": self.urgency or {},
            "attempts": [a.to_dict() for a in self.attempts],
        }


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.replace(",", "").replace("$", "").strip()
    if not text or text.lower() in {"none", "n/a", "unknown"}:
        return None
    try:
        return format(Decimal(text).normalize(), 'f')
    except InvalidOperation:
        return None


def extract_number(text: str) -> str | None:
    marked = re.findall(
        r"(?:final answer|answer is|corrected_final|independent_final)\s*:?\s*\$?(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        text.replace(",", ""),
        flags=re.IGNORECASE,
    )
    if marked:
        return normalize_number(marked[-1])
    nums = NUMBER_RE.findall(text.replace(",", ""))
    return normalize_number(nums[-1]) if nums else None


def extract_final_number(text: str) -> str | None:
    marked = re.findall(
        r"(?:final answer|answer is)\s*:?\s*\$?(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        text.replace(",", ""),
        flags=re.IGNORECASE,
    )
    return normalize_number(marked[-1]) if marked else None


def _line_value(text: str, key: str) -> str | None:
    matches = re.findall(rf"^{re.escape(key)}\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return matches[-1].strip() if matches else None


def parse_verifier(text: str) -> tuple[str, str | None]:
    verdict = (_line_value(text, "VERDICT") or "uncertain").lower()
    if "pass" in verdict:
        verdict = "pass"
    elif "unsettled" in verdict or "not_yet" in verdict or "fail" in verdict:
        verdict = "unsettled"
    else:
        verdict = "uncertain"
    independent = _line_value(text, "INDEPENDENT_FINAL") or _line_value(text, "CORRECTED_FINAL")
    return verdict, normalize_number(independent) or extract_number(independent or "")


def solve_prompt(question: str) -> str:
    return (
        "Solve this grade-school math problem under a tight budget. Do not explain like a teacher. "
        "Translate the question carefully, then answer in exactly this format and no extra prose:\n"
        "Expression: <arithmetic expression for the answer>\n"
        "Computed: <number, or <<CALC: expression>>>\n"
        "Final answer: <number>\n\n"
        f"Question: {question}"
    )


def verify_prompt(question: str, proposed_solution: str) -> str:
    return (
        "You are a terse mathematical verifier. Your PRIMARY job is to verify that the logical steps in the proposed solution actually answer the specific question asked (e.g., if asked for profit, ensure costs were subtracted). "
        "If the reasoning steps misinterpret the question, the verdict MUST be unsettled. "
        "SECOND, recompute the final answer independently using the correct logic. Do not blindly recalculate their expression if their expression is logically flawed. "
        "You may use a Python calculator whenever helpful by outputting exactly `<<CALC: [python expression]>>`; "
        "the system will append ` = [result]` and you can continue.\n\n"
        f"Question:\n{question}\n\n"
        f"Proposed solution:\n{proposed_solution}\n\n"
        "Reply in exactly this form:\n"
        "VERDICT: pass|unsettled|uncertain\n"
        "INDEPENDENT_FINAL: <number or none>\n"
        "REASON: <one short reason about what still needs to be resolved>"
    )


def repair_prompt(question: str, attempt: ReasoningAttempt) -> str:
    return (
        "The previous solution is not settled yet. Do not assume it is wrong; treat it as provisional. "
        "Re-derive the answer from the problem under a tight budget. "
        "Use the verification notes as constraints on what remains unresolved, not as criticism to rationalize. "
        "If the answer cannot be determined, say so. "
        "Use exactly this format and no extra prose:\n"
        "Expression: <arithmetic expression for the answer>\n"
        "Computed: <number, or <<CALC: expression>>>\n"
        "Final answer: <number>\n\n"
        f"Problem:\n{question}\n\n"
        f"Previous answer:\n{attempt.response}\n\n"
        f"Verification notes:\n{attempt.verifier_response}"
    )


def continuation_prompt(question: str, attempt: ReasoningAttempt) -> str:
    return (
        "Your previous response appears incomplete or lacks the required final-answer line. "
        "Do not restart and do not explain like a teacher. Continue only enough to finish the answer. "
        "Use exactly this format and no extra prose:\n"
        "Computed: <remaining computation, or <<CALC: expression>>>\n"
        "Final answer: <number>\n\n"
        f"Problem:\n{question}\n\n"
        f"Previous partial response:\n{attempt.response}"
    )


def confirmation_prompt(question: str, answer: str) -> str:
    return (
        "Independently confirm this answer under a tight budget. Do not explain like a teacher. "
        "Use exactly this format and no extra prose:\n"
        "Expression: <arithmetic expression for the answer>\n"
        "Computed: <number, or <<CALC: expression>>>\n"
        "Final answer: <number>\n\n"
        f"Problem:\n{question}\n\n"
        f"Proposed answer to check: {answer}"
    )


def _verified_answer(attempt: ReasoningAttempt) -> str | None:
    if attempt.accepted:
        return attempt.verifier_answer or attempt.extracted_answer
    return None


def _modal_answer(attempts: list[ReasoningAttempt]) -> tuple[str | None, int]:
    answers: list[str] = []
    for attempt in attempts:
        answer = _verified_answer(attempt)
        if answer is not None:
            answers.append(answer)
    if not answers:
        return None, 0
    counts = {answer: answers.count(answer) for answer in set(answers)}
    if len(counts) > 1:
        return None, max(counts.values())
    best_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best_count]
    if len(winners) != 1:
        return None, best_count
    return winners[0], best_count


def _promote_verified_synthesis(attempts: list[ReasoningAttempt], answer: str) -> int:
    promoted = 0
    for attempt in attempts:
        if not attempt.accepted or _verified_answer(attempt) != answer:
            continue
        for record in attempt.synthesis_records or []:
            metadata = dict(record.get("metadata", {}))
            metadata.update(
                {
                    "promoted_by": "humble_verifier",
                    "verdict": attempt.verdict,
                    "answer": answer,
                    "round_index": attempt.round_index,
                }
            )
            _global_cache.store(record["trigger"], record["delta"], metadata=metadata)
            promoted += 1
    return promoted


def assess_urgency(
    attempts: list[ReasoningAttempt],
    elapsed_sec: float,
    max_elapsed_sec: float | None = None,
) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    if attempts:
        last = attempts[-1]
        if len(attempts) > 1:
            score += len(attempts) - 1
            reasons.append("extra_rounds")
        if last.mode == "continue":
            score += 1
            reasons.append("continued_incomplete_answer")
        if last.extracted_answer is None:
            score += 3
            reasons.append("missing_final_answer")
        if last.verdict != "pass":
            score += 2
            reasons.append(f"verifier_{last.verdict}")
        elif not last.accepted:
            score += 3
            reasons.append("verifier_solver_mismatch")
        if last.synthesis_records:
            score += len(last.synthesis_records)
            reasons.append("synthesis_used")

    if max_elapsed_sec is not None and max_elapsed_sec > 0:
        fraction = elapsed_sec / max_elapsed_sec
        if fraction >= 1.0:
            score += 6
            reasons.append("time_budget_exhausted")
        elif fraction >= 0.75:
            score += 4
            reasons.append("time_budget_high")
        elif fraction >= 0.5:
            score += 2
            reasons.append("time_budget_medium")
    elif elapsed_sec >= 120:
        score += 4
        reasons.append("elapsed_over_120s")
    elif elapsed_sec >= 60:
        score += 2
        reasons.append("elapsed_over_60s")

    if score >= 8:
        level = "critical"
    elif score >= 5:
        level = "high"
    elif score >= 2:
        level = "medium"
    else:
        level = "low"

    return {
        "score": score,
        "level": level,
        "elapsed_sec": round(elapsed_sec, 2),
        "reasons": reasons,
    }


def _cap_token_budget(token_budget: int, max_attempt_tokens: int | None) -> int:
    token_budget = max(1, int(token_budget))
    if max_attempt_tokens is None or max_attempt_tokens <= 0:
        return token_budget
    return max(1, min(token_budget, int(max_attempt_tokens)))


def _scaled_token_budget(
    base_tokens: int,
    multiplier: float,
    max_attempt_tokens: int | None = None,
) -> int:
    multiplier = max(1.0, float(multiplier))
    scaled = max(int(base_tokens), int(round(base_tokens * multiplier)))
    return _cap_token_budget(scaled, max_attempt_tokens)


def _mode_token_budget(
    mode: str,
    base_tokens: int,
    repair_token_multiplier: float,
    max_attempt_tokens: int | None = None,
) -> int:
    if mode in {"repair", "dynamic"}:
        return _scaled_token_budget(base_tokens, repair_token_multiplier, max_attempt_tokens)
    if mode == "continue":
        continuation_multiplier = min(max(1.0, repair_token_multiplier), 2.0)
        return _scaled_token_budget(base_tokens, continuation_multiplier, max_attempt_tokens)
    return _cap_token_budget(base_tokens, max_attempt_tokens)


def _run_attempt(
    M,
    question: str,
    prompt: str,
    mode: str,
    round_index: int,
    vecs=None,
    belief_vec=None,
    humility_vec=None,
    max_new_tokens=220,
    allow_synthesis=True,
    response_prefix: str | None = None,
) -> ReasoningAttempt:
    t0 = time.time()
    synthesis_records: list[dict[str, Any]] = []
    if mode == "dynamic" and vecs is not None:
        generated = generate_agentic_text(
            M,
            vecs,
            belief_vec=belief_vec,
            humility_vec=humility_vec,
            instruction=prompt,
            alpha=15.0,
            max_new_tokens=max_new_tokens,
            epsilon=0.05,
            entropy_threshold=2.0,
            max_loops=1,
            cache_enabled=True,
            cache_write_enabled=False,
            cache_verified_only=True,
            synthesis_enabled=allow_synthesis,
            max_synthesis_events=1,
            synthesis_recorder=synthesis_records,
        )
    else:
        generated = generate_text(M, prompt, max_new_tokens=max_new_tokens)

    response = generated
    if response_prefix:
        response = f"{response_prefix.rstrip()}\n{generated.lstrip()}".strip()

    extracted = extract_final_number(response)
    verifier_response = generate_text(M, verify_prompt(question, response), max_new_tokens=180)
    verdict, verifier_answer = parse_verifier(verifier_response)
    accepted = verdict == "pass" and extracted is not None and verifier_answer is not None and verifier_answer == extracted
    return ReasoningAttempt(
        mode=mode,
        round_index=round_index,
        response=response,
        extracted_answer=extracted,
        verifier_response=verifier_response,
        verdict=verdict,
        verifier_answer=verifier_answer,
        accepted=accepted,
        token_budget=max_new_tokens,
        elapsed_sec=time.time() - t0,
        synthesis_records=synthesis_records,
    )


def solve_with_humility(
    M,
    question: str,
    vecs=None,
    belief_vec=None,
    humility_vec=None,
    max_rounds=3,
    required_agreement=2,
    max_new_tokens=220,
    allow_synthesis=False,
    max_elapsed_sec: float | None = 180.0,
    repair_token_multiplier: float = 2.0,
    max_attempt_tokens: int | None = None,
) -> HumbleResult:
    t0 = time.time()
    attempts: list[ReasoningAttempt] = []
    first_budget = _mode_token_budget(
        "baseline",
        max_new_tokens,
        repair_token_multiplier,
        max_attempt_tokens,
    )

    first = _run_attempt(
        M,
        question,
        solve_prompt(question),
        mode="baseline",
        round_index=0,
        max_new_tokens=first_budget,
    )
    attempts.append(first)
    first.urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)

    answer, count = _modal_answer(attempts)
    if answer is not None and count >= required_agreement:
        _promote_verified_synthesis(attempts, answer)
        urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
        return HumbleResult(question, answer, True, "verified_stable", attempts, urgency)
    if first.urgency["level"] == "critical":
        ans, _ = _modal_answer(attempts)
        ans = ans if ans is not None else (attempts[-1].extracted_answer if attempts else None)
        return HumbleResult(question, ans, False, "stopped_for_urgency_budget", attempts, first.urgency)

    for round_index in range(1, max_rounds + 1):
        prior_answer = _verified_answer(attempts[-1])
        response_prefix = None
        if prior_answer is not None:
            mode = "confirm"
            prompt = confirmation_prompt(question, prior_answer)
        elif attempts[-1].extracted_answer is None and attempts[-1].mode != "continue":
            mode = "continue"
            prompt = continuation_prompt(question, attempts[-1])
            response_prefix = attempts[-1].response
        else:
            mode = "dynamic" if vecs is not None else "repair"
            prompt = repair_prompt(question, attempts[-1])
        attempt_budget = _mode_token_budget(
            mode,
            max_new_tokens,
            repair_token_multiplier,
            max_attempt_tokens,
        )
        attempt = _run_attempt(
            M,
            question,
            prompt,
            mode=mode,
            round_index=round_index,
            vecs=vecs,
            belief_vec=belief_vec,
            humility_vec=humility_vec,
            max_new_tokens=attempt_budget,
            allow_synthesis=allow_synthesis,
            response_prefix=response_prefix,
        )
        attempts.append(attempt)
        attempt.urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)

        answer, count = _modal_answer(attempts)
        if answer is not None and count >= required_agreement:
            _promote_verified_synthesis(attempts, answer)
            urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
            return HumbleResult(question, answer, True, "verified_stable", attempts, urgency)
        if attempt.urgency["level"] == "critical":
            ans, _ = _modal_answer(attempts)
            ans = ans if ans is not None else (attempts[-1].extracted_answer if attempts else None)
            return HumbleResult(question, ans, False, "stopped_for_urgency_budget", attempts, attempt.urgency)

    answer, count = _modal_answer(attempts)
    if answer is not None:
        urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
        return HumbleResult(question, answer, False, "verified_but_not_stable", attempts, urgency)

    urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
    return HumbleResult(question, None, False, "unresolved_after_extra_compute", attempts, urgency)
