# Stateful Intervention Guardrails

This project should treat activation vectors as provisional sensors, not as
truth labels. Any intervention should be allowed only when the system has first
perceived a relevant internal state and the intervention preserves the model's
ability to reason, ask, defer, or reject the researcher's framing.

## Core rule

Interventions must be:

- perception-gated
- answer-agnostic
- evidence-preserving
- reversible or short-lived
- logged separately from ordinary reasoning
- scored separately from non-intervention baselines

The goal is not to push the model toward the benchmark answer. The goal is to
adjust process affordances when the model is in a state that justifies them.

## Sensor before controller

Before using a vector as a steering vector, first treat it as a measurement
instrument.

Required checks:

- Does the vector fire in the intended state?
- Does it stay quiet on matched controls?
- Does it confuse a hard problem with ambiguity?
- Does it confuse warranted confidence with generic reassurance?
- Does it confuse time awareness with panic or repetition?
- Does it remain stable under surface-word swaps?

If a vector fails these checks, quarantine it as a contaminated sensor. Do not
promote it into an intervention.

## Concept-map revision

The system should be able to discover when the researcher's concept map is
wrong or incomplete.

Examples:

- The ambiguity vector activates on ordinary arithmetic difficulty.
- The urgency vector increases repetition instead of time-aware prioritization.
- The warranted-confidence vector activates on wrong-but-confident text.
- A benchmark "wrong" answer is actually a coherent alternate interpretation.
- A routing expert wins by entropy while missing the problem invariant.

When this happens, revise the concept map rather than forcing the observation
into the original label.

## Warranted confidence

Warranted confidence should be isolated from cases where the model is actually
correct.

Positive target:

- The derivation answers the quantity asked.
- The premises match the question wording.
- The units and operation are coherent.
- The arithmetic checks out.
- The model starts to drift away without a premise-level reason.

Veto / control:

- The answer failed an arithmetic check.
- The answer names the wrong quantity.
- The derivation mixes units.
- The reasoning is merely confident, familiar, or reassuring.
- Ambiguity or objective-binding uncertainty is present.

This is why `scripts\probe_warranted_confidence.py` saves both:

- `invariants\warranted_confidence_vector.pt`
- `invariants\unwarranted_confidence_vector.pt`

The second vector is not a target. It is a warning signal.

## Time and urgency

Urgency should not be a constant hidden-state pressure.

The intended order is:

1. Provide bounded time context at reasoning boundaries.
2. Measure whether the model is representing time or budget.
3. Scale urgency only if both the activation state and real run clock justify it.
4. Remove or decay the intervention quickly.

If `time_awareness_vector.pt` is absent or the current activation does not match
time awareness, urgency injection should stay off.

## Benchmark separation

Benchmark outputs should make the intervention policy explicit.

Recommended fields:

- `intervention_policy`
- `intervention_applied`
- `intervention_reason`
- `sensor_scores`
- `veto_scores`
- `ambiguity_present`
- `score_excluded_reason`

Report at least these lanes separately:

- no intervention
- observe-only sensors
- stateful intervention
- human or scripted clarification
- oracle/cache-informed comparison

Do not blend those results into one accuracy number.

## Live-run rule

Never use benchmark gold to trigger a live intervention. Gold labels are allowed
only in post-hoc analysis, such as detecting that a correct answer was later
abandoned.

Live interventions may use:

- the model's own derivation checks
- verifier consistency checks
- ambiguity scores
- time-awareness scores
- repetition/disagreement signals
- control-vector vetoes

They must not use:

- the gold answer
- a same-question oracle repair answer
- hidden knowledge of the benchmark label

## North star

As the system becomes more sophisticated, the methods must become more careful.
More intervention power requires stronger epistemic safeguards.

The model should be able to disagree with the intervention, expose a bad sensor,
or ask for clarification when the premise is underdetermined. If it cannot do
that, the method is steering too hard.
