# Relational definition conversions

[Definition structure and prefilling](DEFINITION_STRUCTURE_AND_PREFILL.md)
describes the mode 3 extension with ordered residual operations and a neutral
assistant opening.

The learned object is how a conversion changes when its source changes. It does
not require the conversions of A and B to be equal. With raw source states and
raw candidate changes, the model predicts

\[
\widehat{\Delta}(B) = \widehat{\Delta}(A) + g(B-A).
\]

The first implementation makes `g` a fitted low-rank linear transport within
each operation and layer; the whole prediction includes an affine mean anchor.
This relation holds before bounded extrapolation clamps the source relation.
Centered training source rows `X` and change rows `Y`
give

\[
\widehat{\Delta}(x) = \bar\Delta +
(x-\bar x)X^{\mathsf T}(XX^{\mathsf T}+\lambda I)^{-1}Y.
\]

Its mean change anchors the conversion; its transport learns differences between
conversions from differences between sources. A constant conversion is not a
successful adaptive fit. The definition bank requires measured and fitted
variation. The generic learner still supports explicitly marked constant data
for diagnostics and rejects unidentifiable all-identical sources with varying
changes.

The solve uses a small training-row Gram matrix, not a full model-width square
matrix. Calculations use CPU float64; storage is finite CPU float32. Source
relations beyond 1.5 times the calibration radius are radially clamped and
reported. This bounds extrapolation without inventing successful generalization.
The fitted map is affine inside that boundary. The radial clamp adds a nonlinear
boundary, but a learned nonlinear change rule is not implemented.

## Activation targets and context

Each calibration source is the native assistant decision state on synthetic
measured evidence alone. Its candidate change is the difference between two
teacher contexts ending in the same unanswered evidence:

- Completed demonstrations of the current definition operation.
- Matched demonstrations that recap the same observed actions independently.

These are **prospective, contrastive activation changes**, not representations
of a completed definition of the queried evidence. Completed synthetic teacher
examples establish a context for the next operation. Neither future answer text
nor the final state of an already completed answer is read as its starting state.

The four operations remain compare, isolate, distill and verify. Six calibration
themes, each in both coordinate orientations, supply twelve pairs per operation.
Two separate teacher themes are never queried calibration themes. Later-stage
sources include the preceding synthetic accepted contrast under the same opaque
candidate format used by the live pipeline. All samples preserve raw source and
change magnitude. One entire calibration theme, both orientations, is withheld
from a diagnostic fit; its activation prediction error is saved separately.

At runtime, each stage encodes its actual unchanged evidence plus its latest
accepted contrast before installing injection hooks. Its fitted profile predicts
a fresh direction and strength. Strength is the predicted change norm divided
by the mean training change norm, capped at two. That factor multiplies the
existing signed step gain. The subject pulse and conversion share the live
residual cap. Both are injected only at the original assistant decision position
and restored there on prefix rebuilds; cached decode positions receive no new
pulse. Every stage removes its hooks before the next source is encoded.

Teacher examples, operator task instructions, actual probe seeds, operator profiles and fit
diagnostics are absent from generation text. Real generation still receives
measured evidence and, in later stages, only the accepted contrast artifact.
The generated text must pass the existing structural and coordinate gates.
Failed comparison, incomplete output or later rejection cannot apply a rename
or lower naming confidence. Intermediate outputs are not measured examples.

## Storage and use

Profiles live in `invariants/out/definition_relations/bank-<identity-hash>.pt`.
Identity covers the model, tokenizer, precision, exact layers, recipe, source
format and learner version. Banks contain all four operation profiles. The
restricted loader checks exact coverage, tensor shapes, finite values, nonzero
change magnitude and fitted conditional variation. Atomic replacement follows
validation of the serialized new bank. In-memory model weight changes require
an updated identity or bank invalidation.

The shell exposes this candidate path through the existing mode control:

```text
:tune probe_define_process 2
:probe define-experiment denial_minus_confidence
```

Mode 0 retains the older simultaneous controls. Mode 1 retains the separately
gated fixed directions. Mode 2 uses relational conversions. The current default
remains 1 while full-model behavior is evaluated. Step gains remain the existing
`probe_define_step_compare`, `isolate`, `distill` and `verify` controls.

Saved attempts identify the namespace, evidence digest, profile bank, per-stage
predictions, bounded extrapolation and strength. These are operator metadata.

## Validation limits

Analytical CPU holdouts check whether a nonconstant conversion can be learned,
including changed magnitude, rank-deficient sources, permutation and translation
invariance, bounded extrapolation, corruption and cancellation. Tiny native CPU
integration exercises all 144 teacher/source forwards, restricted reload and
actual-evidence resolution. Shell tests exercise phase placement, magnitude,
cleanup, state isolation and naming protection. These establish the mechanics.

Activation prediction error and geometric variation do not certify definition
meaning. The isolated full-model experiment in
`scripts/definition_relation_test.py` compares an unsteered baseline, a fixed
mean conversion and the adaptive conversion on identical unseen evidence. It
writes profiles and reports while preserving live shell state. It tests the
compare operation without a subject pulse, so it does not validate the whole
live pipeline. Adaptive and fixed variants differ in both direction and strength.
The evidence is new but related to a calibration theme. Its
known behavioral oracle is retained for assessment outside model context.
Structural qualification is reported separately from semantic correctness.

The first full-model comparison is recorded in
[`assessment.md`](../invariants/out/definition_runs/20261001_relations_384/assessment.md).
At gain 0.04, the conditional conversion changed the interpretation but still
produced an unqualified recap. A bounded 0.08 follow-up introduced positive-side
desirability claims. No tested output established a valid definition, so mode 2
remains opt-in with the original 0.04 gains. These results do not validate the
later operations or the live four-stage pipeline.

The extraction is informed by
[Function Vectors](https://arxiv.org/abs/2310.15213) and
[Activation Source Selection](https://arxiv.org/abs/2607.25270). It is a candidate
residual-state transport, not a reproduction of their causal attention-head
analysis or a demonstrated general definition operator.
