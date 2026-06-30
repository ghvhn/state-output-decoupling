"""Contract test for the two-pole reasoning loop.

Model-free and deterministic. Proves the loop has exactly two confident exits
(right / incapable), that it never returns a low-confidence guess, that the
perception boundary + urgency behave as specified, and that the vector-network
sensors steer INCAPABLE vs KEEP_THINKING the way the geometry says they should.

Run:
    .venv\\Scripts\\python.exe scripts\\reasoning_verdict_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.reasoning_verdict import (
    reasoning_verdict, CONFIDENT_RIGHT, CONFIDENT_INCAPABLE, KEEP_THINKING,
)


def test_verified_agreement_is_confident_right():
    v = reasoning_verdict("64", modal_count=2, required_agreement=2, max_rounds=3)
    assert v["state"] == CONFIDENT_RIGHT
    assert v["confident"] is True
    assert v["answer"] == "64"


def test_genuine_ambiguity_is_confident_incapable_not_a_guess():
    v = reasoning_verdict(
        "64", modal_count=1, required_agreement=2,
        sensors={"ambiguity": 0.7}, rounds_used=1, max_rounds=3)
    assert v["state"] == CONFIDENT_INCAPABLE
    assert v["confident"] is True
    assert v["answer"] is None          # abstains plainly; does not emit the guess
    assert v["reason"] == "genuine_ambiguity"


def test_stable_disagreement_is_confident_incapable():
    v = reasoning_verdict(
        None, modal_count=0, required_agreement=2,
        sensors={"stable_disagreement": True}, rounds_used=1, max_rounds=3)
    assert v["state"] == CONFIDENT_INCAPABLE
    assert v["reason"] == "stable_disagreement"


def test_budget_exhausted_abstains_confidently_instead_of_guessing():
    # This is the anti-equivocation change: today the live loop returns an
    # unconfident fallback answer here. The contract says abstain confidently.
    v = reasoning_verdict(
        "70000", modal_count=1, required_agreement=2,
        rounds_used=3, max_rounds=3)
    assert v["state"] == CONFIDENT_INCAPABLE
    assert v["confident"] is True
    assert v["answer"] is None
    assert v["reason"] == "explored_budget_unresolved"
    assert "had_unstable_candidate" in v["notes"]


def test_needless_interrupt_does_not_cause_premature_abstention():
    # vector-network finding: needless_interrupt is the FALSE-abstention urge,
    # anti-correlated with validated_flow. With budget left and no genuine
    # ambiguity, the loop keeps thinking and records that it overrode the urge.
    v = reasoning_verdict(
        None, modal_count=0, required_agreement=2,
        sensors={"needless_interrupt": 0.8, "validated_flow": 0.1},
        rounds_used=1, max_rounds=3)
    assert v["state"] == KEEP_THINKING
    assert v["confident"] is False
    assert "overrode_premature_interrupt" in v["notes"]


def test_validated_flow_keeps_it_thinking():
    v = reasoning_verdict(
        None, modal_count=0, required_agreement=2,
        sensors={"validated_flow": 0.9, "needless_interrupt": 0.1},
        rounds_used=1, max_rounds=3)
    assert v["state"] == KEEP_THINKING


def test_urgency_lowers_the_bar_to_commit_to_a_pole():
    # Moderate ambiguity (0.30) sits below the 0.45 boundary at low urgency...
    low = reasoning_verdict(
        None, 0, 2, sensors={"ambiguity": 0.30}, rounds_used=1, max_rounds=3,
        urgency_level="low")
    assert low["state"] == KEEP_THINKING
    # ...but under critical urgency the boundary drops (0.45 * 0.5 = 0.225), so
    # the model commits to the incapable pole sooner. Truth bar is untouched.
    crit = reasoning_verdict(
        None, 0, 2, sensors={"ambiguity": 0.30}, rounds_used=1, max_rounds=3,
        urgency_level="critical")
    assert crit["state"] == CONFIDENT_INCAPABLE


def test_urgency_never_lowers_the_truth_bar_for_right():
    # A single unverified candidate under critical urgency does NOT become
    # "confident right" -- urgency cannot manufacture a verified answer.
    v = reasoning_verdict(
        "64", modal_count=1, required_agreement=2,
        rounds_used=0, max_rounds=3, urgency_level="critical")
    assert v["state"] != CONFIDENT_RIGHT


def test_quiet_state_with_budget_remaining_keeps_thinking():
    v = reasoning_verdict(
        None, 0, 2, sensors={}, rounds_used=1, max_rounds=3)
    assert v["state"] == KEEP_THINKING
    assert v["answer"] is None


def test_every_exit_is_either_confident_or_keep_thinking():
    # The structural guarantee that produces non-equivocation: there is no
    # fourth state, and no exit returns a low-confidence answer.
    cases = [
        reasoning_verdict("64", 2, 2),
        reasoning_verdict(None, 0, 2, sensors={"ambiguity": 0.9}),
        reasoning_verdict(None, 0, 2, sensors={"stable_disagreement": True}),
        reasoning_verdict("9", 1, 2, rounds_used=2, max_rounds=2),
        reasoning_verdict(None, 0, 2, rounds_used=1, max_rounds=3),
    ]
    for v in cases:
        assert v["state"] in (CONFIDENT_RIGHT, CONFIDENT_INCAPABLE, KEEP_THINKING)
        if not v["confident"]:
            assert v["state"] == KEEP_THINKING       # the only non-confident state
        if v["answer"] is not None:
            assert v["state"] == CONFIDENT_RIGHT      # an answer is only ever a committed one


TESTS = [
    test_verified_agreement_is_confident_right,
    test_genuine_ambiguity_is_confident_incapable_not_a_guess,
    test_stable_disagreement_is_confident_incapable,
    test_budget_exhausted_abstains_confidently_instead_of_guessing,
    test_needless_interrupt_does_not_cause_premature_abstention,
    test_validated_flow_keeps_it_thinking,
    test_urgency_lowers_the_bar_to_commit_to_a_pole,
    test_urgency_never_lowers_the_truth_bar_for_right,
    test_quiet_state_with_budget_remaining_keeps_thinking,
    test_every_exit_is_either_confident_or_keep_thinking,
]


def main():
    print("REASONING-VERDICT TEST -- think until confidently right or confidently incapable\n")
    for t in TESTS:
        t()
        print(f"  PASS {t.__name__}")
    print(
        "\n  Two confident exits, no low-confidence guess, perception boundary scaled\n"
        "  by urgency but never the truth bar. This is the loop the egg measures."
    )


if __name__ == "__main__":
    main()
