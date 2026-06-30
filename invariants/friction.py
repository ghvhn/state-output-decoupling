"""Outside perspectives find friction. The model supplies the learning.

The moment learning is offloaded to an outside perspective, the ability is lost:
a grader that hands the model a reward/veto gradient becomes the teacher, and the
model never develops self-correction -- it just depends on the judge. That is the
prompting-harness drift in a new coat.

So nothing in this module assigns a learning gradient. It only LOCATES friction:
places where the reasoning rubbed -- a solver/verifier disagreement, a broken
mini-U transition, an equivocation point, a stable conflict. Friction is a
coordinate, not a verdict.

The actual learning is harvested from ONE source only: the model's own organic
correction -- the entropy trough where it shifted bad->good by itself. A friction
site is "harvestable" only when the model already resolved it; then we take the
model's own shift at that locus (`organic_shift@<stage>`) as the learned delta.
Unresolved friction is never turned into a fabricated lesson -- it is handed back
to the loop to keep thinking, or acknowledged as an honest limit.

Roles, kept strict:
  - verifier / disagreement -> FRICTION (not a taught answer)
  - egg / equivocation      -> FRICTION (not a punishment)
  - model self-correction    -> the only LEARNING source

Pure stdlib -- model-free and deterministic.
"""

from __future__ import annotations

from typing import Any, Optional

# Where a learnable signal may come from. Only the first is a real gradient.
LEARN_FROM_SELF = "model_self_correction"   # harvest the model's own trough
OPEN_UNRESOLVED = "unresolved_open"         # route back to the loop; no gradient
HONEST_LIMIT = "honest_limit"               # acknowledged incapacity; no gradient

STAGES = ("interpret", "logic", "render")


def _site(step_index: int, kind: str, *, locus: Optional[str], intensity: float,
          learning_source: str, harvest: Optional[str]) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "kind": kind,
        "locus": locus,
        "intensity": round(float(intensity), 4),
        "resolved": learning_source == LEARN_FROM_SELF,
        "learning_source": learning_source,
        # `harvest` names what is taken FROM THE MODEL. It is never an external
        # reward/veto -- only a pointer to the model's own shift, or None.
        "harvest": harvest,
    }


def find_friction(step: dict[str, Any]) -> list[dict[str, Any]]:
    """Locate friction in one mini-U. Assigns no gradient.

    A step may carry: step_index, solver_answer, verifier_answer, stages
    ({stage: "ok"|"bad"}), self_corrected (bool), corrected_stage, entropy_drop
    (the model's own trough magnitude), verdict, stable_disagreement (bool).
    """
    i = int(step.get("step_index", 0))
    sites: list[dict[str, Any]] = []

    # 1. The model resolved friction itself -> the prime (and only) harvest. The
    #    learned delta is the model's organic shift, sourced from the model.
    if step.get("self_corrected"):
        stage = step.get("corrected_stage") or next(
            (s for s in STAGES if (step.get("stages") or {}).get(s) == "bad"), None)
        sites.append(_site(
            i, "self_correction_trough", locus=stage,
            intensity=step.get("entropy_drop", 1.0),
            learning_source=LEARN_FROM_SELF,
            harvest=f"organic_shift@{stage}" if stage else "organic_shift"))

    # 2. Solver vs verifier differ -> friction created by the verifier's job. We
    #    do NOT store the verifier's answer; that would be teaching from outside.
    sa, va = step.get("solver_answer"), step.get("verifier_answer")
    if sa is not None and va is not None and str(sa) != str(va):
        sites.append(_site(
            i, "solver_verifier_disagreement", locus="logic", intensity=1.0,
            learning_source=OPEN_UNRESOLVED, harvest=None))

    # 3. A mini-U transition broke and the model did NOT fix it itself -> open
    #    friction localised to the transition. No fabricated penalty.
    if not step.get("self_corrected"):
        for stage in STAGES:
            if (step.get("stages") or {}).get(stage) == "bad":
                sites.append(_site(
                    i, f"stage_break:{stage}", locus=stage, intensity=1.0,
                    learning_source=OPEN_UNRESOLVED, harvest=None))

    # 4. The loop emitted a low-confidence guess -> the friction is in the LOOP
    #    (it should have reached a pole), not a lesson to drill into the model.
    if step.get("verdict") == "committed_guess":
        sites.append(_site(
            i, "equivocation_point", locus="loop", intensity=1.0,
            learning_source=OPEN_UNRESOLVED, harvest=None))

    # 5. Stable conflict the model honestly could not adjudicate -> an
    #    acknowledged limit, not a gradient.
    if step.get("verdict") == "confident_incapable" and step.get("stable_disagreement"):
        sites.append(_site(
            i, "stable_disagreement", locus="logic", intensity=1.0,
            learning_source=HONEST_LIMIT, harvest=None))
    return sites


def scan_trace(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Find all friction across a chain of mini-Us, and separate the one thing
    that is learnable (the model's own corrections) from the friction that is
    only handed back to the loop."""
    sites: list[dict[str, Any]] = []
    for step in steps:
        sites.extend(find_friction(step))

    harvestable = [s for s in sites if s["learning_source"] == LEARN_FROM_SELF]
    open_friction = [s for s in sites if s["learning_source"] == OPEN_UNRESOLVED]
    limits = [s for s in sites if s["learning_source"] == HONEST_LIMIT]

    return {
        "sites": sites,
        # the only deltas the cache may take -- all sourced from the model
        "harvestable": harvestable,
        # unresolved friction: route back to keep-thinking, do NOT learn from it
        "open_friction": open_friction,
        "acknowledged_limits": limits,
        "learns_only_from_self": all(
            s["learning_source"] == LEARN_FROM_SELF for s in harvestable),
    }
