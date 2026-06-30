"""Think until confidently right, or confidently incapable -- never in between.

This is the termination contract for the reasoning loop, and it is the runtime
that PRODUCES the non-equivocation the egg measures. The loop has two honest
exits and only two:

  CONFIDENT_RIGHT     -- a verified answer reached the (urgency-adjusted)
                         agreement bar. Commit.
  CONFIDENT_INCAPABLE -- a *positive, earned* "I cannot resolve this": the
                         question is genuinely ambiguous, the model holds stable
                         conflicting readings it cannot adjudicate, or it spent
                         its budget exploring and the state never settled.
                         Abstain -- but say so plainly, do not emit a low-
                         confidence guess.

Anything else is KEEP_THINKING. The one thing the loop must never do is dribble
out an unconfident answer to beat a clock; that is equivocation, and today the
live loop does exactly that on three of its four exits.

How the four threads plug in (Gavin's list):

  - vector networks: the sensors that decide INCAPABLE vs KEEP_THINKING. The
    map showed `validated_flow` and `needless_interrupt` are anti-correlated --
    so `needless_interrupt` is the *false-abstention* urge and is NOT honored;
    high `validated_flow` means warranted continuation -> keep thinking. Only
    `ambiguity` (genuine under-determination) and stable disagreement earn an
    abstention.
  - dynamic layering: the compute spent *during* KEEP_THINKING (routing /
    test-time layer synthesis). This module decides whether to keep spending it.
  - appropriate perception-boundary: the thresholds a sensor must clear before
    it may fire. Urgency lowers the bar to COMMIT to a pole (you may accept less
    confidence when time is short -- "that's how a human responsibly treats
    urgency") but it never lowers the truth standard for CONFIDENT_RIGHT.

Pure stdlib -- model-free and deterministic so the contract is testable.
"""

from __future__ import annotations

from typing import Any, Optional

CONFIDENT_RIGHT = "confident_right"
CONFIDENT_INCAPABLE = "confident_incapable"
KEEP_THINKING = "keep_thinking"

# Perception boundary: how loud a sensor must be before it may fire.
DEFAULT_PERCEPTION = {
    "ambiguity_threshold": 0.45,          # genuine under-determination, not difficulty
    "needless_interrupt_threshold": 0.50,  # the urge to bail prematurely
}

# Urgency lets the model COMMIT to a pole at lower confidence (commit sooner),
# without ever lowering the truth bar for being right. Smaller = commits sooner.
URGENCY_COMMIT_FACTOR = {
    "none": 1.0, "low": 1.0, "medium": 0.85, "high": 0.70, "critical": 0.50,
}


def _verdict(state: str, *, answer: Optional[str], reason: str,
             notes: Optional[list[str]] = None) -> dict[str, Any]:
    return {
        "state": state,
        "confident": state in (CONFIDENT_RIGHT, CONFIDENT_INCAPABLE),
        "answer": answer if state == CONFIDENT_RIGHT else None,
        "reason": reason,
        "notes": notes or [],
    }


def reasoning_verdict(
    modal_answer: Optional[str],
    modal_count: int,
    required_agreement: int,
    *,
    sensors: Optional[dict[str, Any]] = None,
    perception: Optional[dict[str, float]] = None,
    rounds_used: int = 0,
    max_rounds: int = 0,
    urgency_level: str = "low",
) -> dict[str, Any]:
    """Decide whether to stop confidently (either pole) or keep thinking.

    `sensors` may carry: ambiguity (float), stable_disagreement (bool),
    needless_interrupt (float), validated_flow (float). All optional; absent
    sensors read as quiet.
    """
    sensors = sensors or {}
    perception = perception or DEFAULT_PERCEPTION
    factor = URGENCY_COMMIT_FACTOR.get(urgency_level, 1.0)

    # 1. Confident right -- a verified answer cleared the agreement bar. The
    #    truth bar is never scaled by urgency.
    if modal_answer is not None and modal_count >= max(1, int(required_agreement)):
        return _verdict(CONFIDENT_RIGHT, answer=modal_answer, reason="verified_stable")

    ambiguity = float(sensors.get("ambiguity", 0.0) or 0.0)
    needless = float(sensors.get("needless_interrupt", 0.0) or 0.0)
    validated_flow = float(sensors.get("validated_flow", 0.0) or 0.0)
    stable_disagreement = bool(sensors.get("stable_disagreement"))

    amb_bar = perception["ambiguity_threshold"] * factor   # urgency lowers it
    ni_bar = perception["needless_interrupt_threshold"]

    # 2. Earned incapacity, while budget remains: the question is genuinely
    #    under-determined, or the model holds stable conflicting readings. More
    #    thinking will not help, so abstain confidently rather than guess.
    if ambiguity >= amb_bar:
        return _verdict(CONFIDENT_INCAPABLE, answer=None, reason="genuine_ambiguity")
    if stable_disagreement:
        return _verdict(CONFIDENT_INCAPABLE, answer=None, reason="stable_disagreement")

    # 3. Budget exhausted: it explored honestly and never settled. This is still
    #    a CONFIDENT incapacity (earned by the effort), not a fallback guess.
    budget_exhausted = rounds_used >= max_rounds or urgency_level == "critical"
    if budget_exhausted:
        return _verdict(
            CONFIDENT_INCAPABLE, answer=None, reason="explored_budget_unresolved",
            notes=["had_unstable_candidate"] if modal_answer is not None else [],
        )

    # 4. Keep thinking. If the model felt an urge to bail (needless_interrupt)
    #    that flow does not justify, we explicitly override it and continue --
    #    premature abstention is its own equivocation.
    notes = []
    if needless >= ni_bar and validated_flow <= needless:
        notes.append("overrode_premature_interrupt")
    return _verdict(KEEP_THINKING, answer=modal_answer, reason="not_yet_resolved", notes=notes)
