# 3. Measurement and steering

Questions this document answers: What does a probe score establish? What is a field? How much can a correction change a representation?

## Measuring a state

A probe associates a name with one or more layer-specific directions. The basic `probe_score` compares the last-token residual at each available layer with its direction and averages the projections. With unit probe directions, these are cosine similarities. The basic score is raw and uncentered; callers can compare it with a rolling baseline.

A high score means alignment with a particular measurement direction. To interpret it as ambiguity, confidence, repetition, or another named property requires evidence about how that direction was constructed and validated. The label alone is not a factual verdict. Scores can depend on the prompt, layer, pooling rule, and model checkpoint.

Measurement and intervention are separate operations. Reading an activation does not inherently steer it. A controller can use a measurement to decide on an intervention, but that connection has to exist in the running code.

## ClaimMap and fields

ClaimMap compares representations of alternative framings and can derive layer-specific difference vectors. Its steering helper normalizes selected directions and applies them through hooks when given a positive steering strength. A difference between two representations identifies a contrast; it does not determine which statement is true.

The shell's field system represents probes or configured anchors as sources of activation adjustments. Quantities named mass, gravity, shape, time, and qualities configure how those adjustments are calculated. Field hooks compute a pull, pass it through the shared steering cap, and add it to the layer output. Exclusions and zero effective strengths can make configured entries inactive.

These names describe a programmed dynamical system in representation space. A reported gravitational pull is evidence about that calculation, not evidence of a literal physical force or a felt sensation.

## The steering envelope

For a detached proposed addition `d` to a residual `h`, the shared `_cap_steer` helper implements a bound of the form:

```text
applied = d * min(1, fraction * norm(h) / norm(d))
```

Zero additions are handled separately, and non-finite additions are replaced with zero. The cap is per applicable vector row and per helper call. Differentiable candidate deltas are exempt inside optimization; their detached application is the step expected to pass through the cap.

A bound on one addition is not a bound on the combined effect of every hook over many layers and tokens. It also does not guarantee a useful, correct, or stable answer. The associative hooks in chapter 5 have a different implementation and must not be assumed to inherit this helper's protection.

Apply this: distinguish "the score changed," "a hook applied a correction," and "task performance improved." Establish the last claim with outcomes, ideally comparing otherwise matched runs with and without the particular intervention.

Sources: [probe scoring and steering cap](../../invariants/engine.py) (`probe_score`, `_cap_steer`); [ClaimMap steering](../../invariants/claimmap.py) (`claimmap_steer_handles`); [field hooks](../../scripts/interactive_phenomenality.py) (`build_gravity_field_handles`).
