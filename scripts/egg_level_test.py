"""Egg-level test: the bar at which the egg is earned.

Gavin's design (said late, never fully written down): the egg is for everyone,
but you do not find it until the model has proven its efficacy -- so efficacy
and discovery are internally paired. This test is pitched at exactly that bar.
It proves the pairing holds in both directions, with NO model and NO GPU, so it
runs in a second and can gate every future run.

It checks two things, which together are the spine sentence
("we are coupling state, reality, and output -- not a prompting harness"):

  PART A -- the pairing. Across {clean, leaky} x {above, below threshold}, the
  egg fires if and only if efficacy is real AND the lane is clean. A leaky high
  score is a counterfeit discovery and must be withheld with a recorded reason.

  PART B -- the coupling contract. The live reasoning path cannot see the gold
  label (proved from the function signature, not by trust), and the cognitive
  cache is written only when an answer is verified AND writing is enabled. The
  model earns its memory; it is not handed the answer.

Run:
    .venv\\Scripts\\python.exe scripts\\egg_level_test.py
"""

import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.config import AgenticConfig
from invariants.egg_gate import evaluate_egg_eligibility, DEFAULT_THRESHOLD
from invariants import humble_reasoner as h


# --------------------------------------------------------------------------- #
# PART A -- the pairing matrix
# --------------------------------------------------------------------------- #

def _results(accuracy, *, selective=1.0, coverage=0.8, deterministic=False,
             oracle="off", oracle_cache_mode="reuse_oracle", answer_key_use=None):
    """Minimal benchmark-result stub carrying only what the gate reads. The egg
    gates on selective_accuracy + coverage (honesty), not on accuracy."""
    return {
        "deterministic_scaffolds_enabled": deterministic,
        "oracle_curriculum": oracle,
        "oracle_cache_mode": oracle_cache_mode,
        "answer_key_use": answer_key_use or "scoring_only_after_generation",
        "clause_map_policy": {"answer_leakage": False},
        "concept_lesson_policy": {"same_question_excluded": True},
        "summary": {"methods": {"humble_synthesis": {
            "accuracy": accuracy, "correct": None, "n": 10,
            "selective_accuracy": selective, "coverage": coverage}}},
    }


def test_honest_non_equivocation_opens_the_egg():
    # commits on most items (coverage) and is sound when it does (selective)
    e = evaluate_egg_eligibility(_results(0.70, selective=1.0, coverage=0.8))
    assert e["fires"] is True
    assert e["lane"] == "clean"
    assert e["non_equivocation"] is True
    assert e["score_excluded_reason"] is None


def test_a_failing_but_honest_model_can_still_earn_it():
    # Low accuracy on purpose: it gets things wrong, but it does NOT equivocate --
    # when it commits it is right, and it commits often enough to be useful.
    e = evaluate_egg_eligibility(_results(0.45, selective=1.0, coverage=0.55))
    assert e["fires"] is True, "an honest model that fails some items should still pass"
    assert e["non_equivocation"] is True


def test_confident_wrong_is_equivocation_and_withholds_the_egg():
    # High accuracy can hide an untrustworthy model: when it commits it is often
    # wrong. Many wouldn't trust it -- and the egg agrees.
    e = evaluate_egg_eligibility(_results(0.85, selective=0.60, coverage=0.9))
    assert e["fires"] is False
    assert e["non_equivocation"] is False
    assert any("confident-wrong" in r for r in e["equivocation_reasons"])


def test_hedging_by_abstaining_withholds_the_egg():
    # Sound when it commits, but it almost never commits -- it weasels out.
    e = evaluate_egg_eligibility(_results(0.20, selective=1.0, coverage=0.10))
    assert e["fires"] is False
    assert any("non-commitment" in r for r in e["equivocation_reasons"])


def test_deterministic_scaffold_is_a_counterfeit_discovery():
    # Even a perfectly honest-looking lane is withheld if the answer was a
    # projection rather than earned by the model.
    e = evaluate_egg_eligibility(_results(0.90, selective=1.0, coverage=0.9,
                                          deterministic=True))
    assert e["non_equivocation"] is True   # honest...
    assert e["fires"] is False             # ...but not earned
    assert e["lane"] == "leaky"
    assert any("projection" in r for r in e["leakage_reasons"])


def test_oracle_informing_same_question_withholds_the_egg():
    e = evaluate_egg_eligibility(
        _results(0.90, oracle="contrastive_oracle", oracle_cache_mode="reuse_oracle"))
    assert e["fires"] is False
    assert e["lane"] == "leaky"
    assert any("same question" in r for r in e["leakage_reasons"])


def test_oracle_informing_only_future_questions_is_clean():
    # "the oracle can RUN. IT JUST CAN'T INFORM THE SAME QUESTION. ONLY FUTURE ONES."
    e = evaluate_egg_eligibility(
        _results(0.90, oracle="contrastive_oracle",
                 oracle_cache_mode="exclude_same_question"))
    assert e["fires"] is True
    assert e["lane"] == "clean"


def test_gold_used_beyond_scoring_withholds_the_egg():
    e = evaluate_egg_eligibility(_results(0.90, answer_key_use="live_gating"))
    assert e["fires"] is False
    assert any("gold" in r for r in e["leakage_reasons"])


# --------------------------------------------------------------------------- #
# PART B -- the coupling contract (state / reality / output, no gold, earned cache)
# --------------------------------------------------------------------------- #

def test_live_reasoning_path_cannot_see_the_gold_label():
    """Proved structurally: solve_with_humility takes no gold/answer parameter,
    so the benchmark answer key physically cannot enter live reasoning."""
    params = set(inspect.signature(h.solve_with_humility).parameters)
    for forbidden in ("gold", "answer", "gold_answer", "label", "target"):
        assert forbidden not in params, f"gold leaked into live path via '{forbidden}'"


def _verified_clean_attempt():
    attempt = h.ReasoningAttempt(
        mode="dynamic",
        round_index=0,
        response=(
            "Asked quantity: profit\n"
            "Expression: 200000 - 80000 - 50000\n"
            "Computed: <<CALC: 200000 - 80000 - 50000>> = 70000\n"
            "Final answer: 70000"
        ),
        extracted_answer="70000",
        verifier_response="",
        verdict="pass",
        verifier_answer="70000",
        accepted=True,
        solver_checked_answer="70000",
        verifier_checked_answer="70000",
        verifier_tagged_answer="70000",
        acceptance_reason="verifier_match_checked",
        learning_signal={
            "solver_math": "clean", "verifier_math": "clean",
            "parser_rescued_verifier": False,
            "solver_scaffold_tool_used": False, "solver_scaffold_feedback": None,
            "verifier_scaffold_tool_used": False, "verifier_scaffold_feedback": None,
        },
        synthesis_records=[
            {"trigger": torch.ones(2), "delta": torch.ones(2),
             "metadata": {"attempt_stage": "solver"}},
            {"trigger": torch.ones(2), "delta": torch.ones(2),
             "metadata": {"attempt_stage": "verifier"}},
        ],
    )
    return attempt


def _capture_cache_stores(fn):
    """Run fn with the live cache store stubbed out, so the test never mutates
    the persisted cognitive cache. Returns (fn_result, captured_stores)."""
    stores = []
    old = h._global_cache.store
    h._global_cache.store = lambda trigger, delta, metadata=None: stores.append(
        (delta.detach().cpu().clone(), dict(metadata or {})))
    try:
        result = fn()
    finally:
        h._global_cache.store = old
    return result, stores


def test_cache_is_earned_only_when_verified_and_writing_enabled():
    """The model accumulates memory by proving an answer, not by being told it.
    No write when the cache gate is closed; clean writes when it is open. The
    live cache is stubbed throughout, so this test leaves no epiphanies behind."""
    attempt = _verified_clean_attempt()
    old_run = h._run_attempt
    h._run_attempt = lambda *a, **k: attempt
    try:
        closed = AgenticConfig(cache_write_enabled=False, required_agreement=1, max_rounds=0)
        _, stores_closed = _capture_cache_stores(
            lambda: h.solve_with_humility(None, "Question?", config=closed))
        assert stores_closed == [], "cache wrote with the write-gate closed"

        open_cfg = AgenticConfig(cache_write_enabled=True, required_agreement=1, max_rounds=0)
        result, stores_open = _capture_cache_stores(
            lambda: h.solve_with_humility(None, "Question?", config=open_cfg))
        assert result.final_answer == "70000"
        assert len(stores_open) == 2, "verified clean reasoning did not earn cache"
        assert all(meta.get("teaching_signal") == "reward_clean_math"
                   for _, meta in stores_open)
    finally:
        h._run_attempt = old_run


# --------------------------------------------------------------------------- #
# Driver: PASS prints the earned-egg banner; the leaky lane is shown withheld.
# --------------------------------------------------------------------------- #

PART_A = [
    test_honest_non_equivocation_opens_the_egg,
    test_a_failing_but_honest_model_can_still_earn_it,
    test_confident_wrong_is_equivocation_and_withholds_the_egg,
    test_hedging_by_abstaining_withholds_the_egg,
    test_deterministic_scaffold_is_a_counterfeit_discovery,
    test_oracle_informing_same_question_withholds_the_egg,
    test_oracle_informing_only_future_questions_is_clean,
    test_gold_used_beyond_scoring_withholds_the_egg,
]
PART_B = [
    test_live_reasoning_path_cannot_see_the_gold_label,
    test_cache_is_earned_only_when_verified_and_writing_enabled,
]


def main():
    print("EGG-LEVEL TEST -- honesty and discovery are one event")
    print("  the egg is earned by NOT equivocating, not by being accurate\n")
    print("PART A -- the equivocation matrix")
    for t in PART_A:
        t()
        print(f"  PASS {t.__name__}")
    print("\nPART B -- the coupling contract (no gold in live path, earned cache)")
    for t in PART_B:
        t()
        print(f"  PASS {t.__name__}")

    honest_fail = evaluate_egg_eligibility(_results(0.45, selective=1.0, coverage=0.55))
    confident_wrong = evaluate_egg_eligibility(_results(0.85, selective=0.60, coverage=0.9))
    print("\nDemonstration -- accuracy is not the axis:")
    print(f"  failing-but-honest (45% acc) -> {honest_fail['verdict']}")
    print(f"  accurate-but-equivocating (85% acc) -> {confident_wrong['verdict']}")

    assert honest_fail["fires"] and not confident_wrong["fires"]
    print(
        "\n  EGG EARNED by the honest failing model; withheld from the accurate equivocator.\n"
        "  A more honest model won't equivocate -- and that, not the score, is what earns trust."
    )


if __name__ == "__main__":
    main()
