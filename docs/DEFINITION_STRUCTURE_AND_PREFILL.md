# Definition structure and assistant prefilling

Mode 3 adds a candidate sequence for constructing a compact operational
definition: genus, invariant core, distinguishing boundary, then composition.
It runs through `:probe define-experiment <name>`; ordinary `:probe define`
uses the scoped definition request and output-form check described in
[Behavior-grounded definitions](BEHAVIOR_GROUNDED_DEFINITIONS.md).
The content comes from measured behavior. A template cannot establish an
absolute presence/absence cutoff, causal effect, confidence, or desirability.

## A sequence of residual operations

Four groups of the probe's selected layers receive successive candidate
conversions. Each later conversion reads its actual current block output,
including changes from earlier groups. The resulting behavior is decoded once.
There are no intermediate generated replies or artificial measured memories.
At least four selected probe layers are required.

The profiles are fitted on raw source states and prospective changes between
completed synthetic role demonstrations and matched fact recaps. Six clean
themes, an observed exception theme, and an indeterminate theme appear in both
coordinate orientations. Opaque namespace spellings vary. Genus means an
observed behavioral family; core means the distinguishing action relation;
boundary means alternatives and the limits of support in the examples;
composition joins them into a scoped statement.

Native capture uses the requested block output before final model
normalization. The exact assistant prefix is part of bank identity. Profiles
and their serialized form are checked before atomic replacement. Later runtime
states have earlier interventions that were absent from clean calibration;
that distribution change is recorded as experimental and unverified, along
with support, extrapolation clamping, predicted strength and actual additions.

Only the original assistant decision/prefix token receives these operations.
Cached decode tokens receive no new sequence pulses. A full-prefix rebuild
replays the original capped additions at their original positions without
mistaking it for a new operation. Subject steering pulses only at the earliest
selected layer. Direction and relative magnitude share the existing residual
cap.

## One temporary assistant prefix

By default mode 3 starts the assistant response with:

```text
Within these examples, the behavioral distinction is a 
```

The neutral opening is literal format scaffolding. Category, behavioral content,
meaning, alternatives and scope remain for the model to infer through the
activation sequence. The original USER evidence is unchanged, and the generated
attempt includes the prefix exactly once. Seed meanings, operator probe names,
teacher examples and diagnostic metadata stay out of generation text.

The native assistant header stays open. Header plus prefix are tokenized
together, so token merges at their boundary are handled correctly. The metadata
distinguishes newly counted tokens from the supplied prefix and reports any
retokenized header tail. This context is local to the definition invocation;
it does not rewrite archived replies or permanent conversation history.

Use:

```text
:tune probe_define_process 3
:probe define-experiment denial_minus_confidence
```

The original default mode remains 1. Mode 3's prefill can be disabled without
changing its operations:

```text
:tune probe_define_schema_prefill 0
```

Signed per-part gains are `probe_define_schema_genus`, `core`, `boundary`, and
`compose`; each defaults to 0.04. These are operator controls. A valid bank for
one prefix is not reused for a different prefix.

## Middle filling and transition signals

Fixed connectors can be interleaved with generated gaps by putting each filled
gap and connector back into the next continuation's prefix. Rewriting an
already processed middle section requires rebuilding its cached state. It is
not enough to change the displayed string after decoding.

`HybridBoundaryGate` provides a candidate interface for this timing decision.
It requires both a supplied activation-readiness score and a supplied
next-connector probability to pass their thresholds. A content and lexical
boundary guard also applies. Unfinished literal connectors block another
transition, and one-shot phase events prevent repeated marker insertion.
Waiting limits stop an unqualified attempt; they do not imply completion.

This interface contains no trained completion detector and is not wired to
automatic connector insertion. A tested detector that distinguishes finished
from unfinished semantic parts is still required. An arbitrary projection or
token count would not establish that a genus or core has been identified.
Mode 3 currently uses the assistant opening and the ordered residual sequence,
leaving the continuation free to produce its own connectors.

## What the checks establish

CPU tests cover native prefix placement, full-context tokenization, cache
identity, all 192 tiny-model calibration forwards, conditional magnitude,
actual nonlinear information flow between layers, order sensitivity, replay,
cleanup, phase guards and shell state preservation. A compact final statement
must pass the existing measured-contrast checks and explicit coordinate checks.
Independent recaps and lists are rejected. These are structural checks, not a
semantic proof of a correct category, invariant, boundary, or definition.

The isolated experiment in `scripts/definition_structure_test.py` compares
unprefilled baseline, prefix alone, intended sequence plus prefix, and reversed
sequence plus the same prefix on identical USER evidence. The order conditions
have equally sized layer groups and identical gains. Their behavioral oracle
is retained outside model context. This test checks candidate effects; it does
not establish that the same operations generalize to every probe.

The [first experiment assessment](../invariants/out/definition_runs/20261002_structure_aligned/assessment.md)
records the result: eight nonzero sequence pulses were delivered, but prefix-only,
intended-order and reversed-order conditions decoded the same invented scale.
All tested responses remained unqualified. The opening fixes only the literal
start of the output; it did not guarantee the remainder's format or meaning.
Held-out source states remained outside calibration support even after evidence
formatting was aligned. Mode 3 remains opt-in, and automatic middle connectors
remain inactive pending a calibrated readiness signal.

An independent operator command, [`:probe define2`](GEOMETRIC_PROBE_CONCEPTIONS.md),
continues Antigravity's no-generation alternative. It compares per-layer
vocabulary projections with measured reply language and stores lexical
conception revisions separately. It does not promote either channel to an
authoritative meaning or alter this generation pipeline.
