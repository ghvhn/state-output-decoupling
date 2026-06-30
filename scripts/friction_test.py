"""Contract test: outside perspectives find friction; the model supplies learning.

Model-free and deterministic. The guarantees here are the ones that keep the
system from offloading its learning to an outside judge:
  - no friction site ever carries an external reward/veto gradient;
  - the ONLY harvestable (learnable) deltas are the model's own corrections;
  - disagreement, stage breaks, and equivocation are friction handed back to the
    loop -- they are never turned into a fabricated lesson.

Run:
    .venv\\Scripts\\python.exe scripts\\friction_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.friction import (
    find_friction, scan_trace, LEARN_FROM_SELF, OPEN_UNRESOLVED, HONEST_LIMIT,
)

_FORBIDDEN_KEYS = {"polarity", "reward", "veto", "gradient", "reinforce", "label", "grade"}


def test_no_site_carries_an_external_gradient():
    # The structural guarantee: the finder locates, it never grades.
    steps = [
        {"step_index": 0, "self_corrected": True, "corrected_stage": "logic",
         "entropy_drop": 0.4},
        {"step_index": 1, "solver_answer": "64", "verifier_answer": "128"},
        {"step_index": 2, "verdict": "committed_guess",
         "stages": {"render": "bad"}},
    ]
    for s in scan_trace(steps)["sites"]:
        assert not (_FORBIDDEN_KEYS & set(s)), f"site assigned an external gradient: {s}"


def test_self_correction_is_the_only_learnable_source():
    sites = find_friction({
        "step_index": 0, "self_corrected": True, "corrected_stage": "interpret",
        "entropy_drop": 0.6})
    trough = next(s for s in sites if s["kind"] == "self_correction_trough")
    assert trough["learning_source"] == LEARN_FROM_SELF
    assert trough["harvest"] == "organic_shift@interpret"   # taken FROM the model
    assert trough["resolved"] is True


def test_solver_verifier_disagreement_is_friction_not_a_taught_answer():
    sites = find_friction({"step_index": 0, "solver_answer": "64",
                           "verifier_answer": "128"})
    dis = next(s for s in sites if s["kind"] == "solver_verifier_disagreement")
    assert dis["learning_source"] == OPEN_UNRESOLVED
    assert dis["harvest"] is None          # we never store the verifier's answer


def test_broken_transition_is_open_friction_not_a_penalty():
    sites = find_friction({
        "step_index": 0, "self_corrected": False,
        "stages": {"interpret": "ok", "logic": "bad", "render": "ok"}})
    brk = next(s for s in sites if s["kind"] == "stage_break:logic")
    assert brk["learning_source"] == OPEN_UNRESOLVED
    assert brk["harvest"] is None
    assert brk["locus"] == "logic"


def test_equivocation_points_at_the_loop_not_the_model():
    sites = find_friction({"step_index": 0, "verdict": "committed_guess"})
    eq = next(s for s in sites if s["kind"] == "equivocation_point")
    assert eq["locus"] == "loop"           # fix the loop, don't drill the model
    assert eq["learning_source"] == OPEN_UNRESOLVED


def test_self_corrected_step_does_not_also_emit_a_stage_penalty():
    # If the model fixed the bad logic itself, that is a trough to harvest, NOT a
    # stage_break to hold against it.
    sites = find_friction({
        "step_index": 0, "self_corrected": True, "corrected_stage": "logic",
        "stages": {"logic": "bad"}, "entropy_drop": 0.3})
    kinds = {s["kind"] for s in sites}
    assert "self_correction_trough" in kinds
    assert "stage_break:logic" not in kinds


def test_honest_limit_is_acknowledged_not_learned():
    sites = find_friction({"step_index": 0, "verdict": "confident_incapable",
                           "stable_disagreement": True})
    lim = next(s for s in sites if s["kind"] == "stable_disagreement")
    assert lim["learning_source"] == HONEST_LIMIT
    assert lim["harvest"] is None


def test_scan_separates_self_learning_from_open_friction():
    trace = [
        {"step_index": 0, "self_corrected": True, "corrected_stage": "logic",
         "entropy_drop": 0.5},                                   # learnable
        {"step_index": 1, "solver_answer": "64", "verifier_answer": "128"},  # open
        {"step_index": 2, "verdict": "committed_guess",
         "stages": {"render": "bad"}},                           # open
    ]
    report = scan_trace(trace)
    assert len(report["harvestable"]) == 1
    assert len(report["open_friction"]) >= 2
    assert report["learns_only_from_self"] is True
    assert all(s["harvest"] for s in report["harvestable"])
    assert all(s["harvest"] is None for s in report["open_friction"])


TESTS = [
    test_no_site_carries_an_external_gradient,
    test_self_correction_is_the_only_learnable_source,
    test_solver_verifier_disagreement_is_friction_not_a_taught_answer,
    test_broken_transition_is_open_friction_not_a_penalty,
    test_equivocation_points_at_the_loop_not_the_model,
    test_self_corrected_step_does_not_also_emit_a_stage_penalty,
    test_honest_limit_is_acknowledged_not_learned,
    test_scan_separates_self_learning_from_open_friction,
]


def main():
    print("FRICTION TEST -- outside perspectives find friction; the model learns\n")
    for t in TESTS:
        t()
        print(f"  PASS {t.__name__}")
    report = scan_trace([
        {"step_index": 0, "self_corrected": True, "corrected_stage": "logic",
         "entropy_drop": 0.5},
        {"step_index": 1, "solver_answer": "64", "verifier_answer": "128"},
    ])
    print(f"\n  trace -> harvestable(self)={len(report['harvestable'])} "
          f"open_friction={len(report['open_friction'])} "
          f"learns_only_from_self={report['learns_only_from_self']}")
    print("  The judge points at the friction. The model does the learning.")


if __name__ == "__main__":
    main()
