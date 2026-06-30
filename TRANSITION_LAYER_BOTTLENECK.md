# Transition Layer Bottleneck Hypothesis

## Claim

The recent benchmark failures look less like missing arithmetic ability and more
like a U-shape bottleneck:

```text
prompt text -> interpretation / translation -> mid-workspace logic -> communication / render -> final answer text
```

If interpretation is under-settled, the logic layer receives the wrong task
state. If communication/render is under-settled, correct latent logic can be
misreported, truncated, or wrapped in the wrong final tag.

## Why This Fits The Latest Runs

- Row 1 often contains pieces of the correct solution, but the system binds the
  every-second discount rule incorrectly or renders an incomplete answer.
- Dynamic synthesis sometimes makes the row worse, because it acts after a
  flawed interpretation has already entered the workspace.
- The verifier can compute the right answer, but if it is starved of time, the
  final verdict text is incomplete and the harness cannot learn from it.
- The old U-shape notes already separate early intent/role binding, mid
  workspace around L16, and late render/persona.

## Current Risk

The current Social/Creative/Analytical routing layers are:

- Social: L14
- Creative: L18
- Analytical: L20

Those are plausible mid-band vectors, but they do not explicitly protect the
transition from interpretation into logic or the transition from logic into
communication. A low-entropy branch can still be confidently wrong if the
interpreted task state is wrong.

## Proposed Architecture

Use staged compute rather than one generic "try harder" loop:

1. Interpretation stage
   - Bind asked quantity, givens, rules, and objective.
   - Detect alternate coherent readings.
   - Spend more time here only when role binding or wording rules are unstable.

2. Logic stage
   - Use calculator/equation/scaffold tools for arithmetic and symbolic steps.
   - Route structural/tool errors to plain repair before dynamic synthesis.

3. Communication stage
   - Verify that the final rendered answer matches the checked expression.
   - Treat bad final tags, truncation, and verifier cutoffs as render failures,
     not necessarily logic failures.

## Measurement Plan

- Re-run the translation/thinking probe or a smaller cached equivalent on the
  quantity micro-suite.
- Track per-layer separation for:
  - asked quantity / objective binding
  - arithmetic operation
  - final answer identity
  - output format/final-tag correctness
- Look for transition layers where role binding drops before answer identity or
  render format stabilizes.
- Compare correct vs wrong rows by whether the transition layer state changes
  before the final answer flips.

## Policy For Benchmark Runs

- Do not lower truth standards under urgency.
- Use urgency to choose simpler steps and protect verifier time.
- Keep `CLAUSEMAP` opt-in; if used, persist only sanitized methodology, not raw
  clauses.
- Prefer repair for structural/tool errors before vector synthesis.

## Next Experiment

Run a two-lane micro-suite:

- Lane A: current safe benchmark policy, clause-map off.
- Lane B: interpretation-expanded policy, with extra interpretation time and no
  deterministic answer scaffold.

Success is not just more correct answers. Success means fewer cases where the
model reaches a correct intermediate state and then loses it during render.
