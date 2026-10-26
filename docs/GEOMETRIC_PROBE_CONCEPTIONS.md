# Geometric probe observations and conception revisions

Antigravity session `3d5339aa-6e1f-409b-bb05-4f3754c510b0`, steps 358–368,
started this alternative to generating probe definitions. The operator wanted
an interpretation to remain a window into possible misconceptions, and wanted
the conception to change rather than moving the underlying probe direction.

The new command reads two channels without generating text:

```text
:probe define2 denial_minus_confidence
:probe define2 denial_minus_confidence align
:probe define2 denial_minus_confidence watch 10
:probe define2 denial_minus_confidence off
```

Use an active probe name or its exposed opaque identifier. The operator display
shows the name beside its opaque identifier in blue. The original `:probe define`
remains available for the model's generated interpretation.
`define2` displays statistical evidence; its terms are not a report of what the
model believes, and its results do not authorize renaming a probe.

## Two channels

The vocabulary channel projects each layer's direction onto the output
embedding weights. It retains layer identities instead of averaging different
layers into one direction. Projection scores have their original signs. Work
is streamed through bounded CPU chunks without a model forward pass or copying
the entire output head into an FP32 matrix.

Direct projection of an earlier-layer direction is a heuristic view through
the output head. Later layers and normalization can change its effect during
actual generation. Even a final-layer projection alone does not establish a
concept's meaning or a causal effect on speech.

The behavioral channel compares document frequencies in replies from the lower
and higher measured tails. User inputs are reported separately, so a recurring
topic is not silently treated as a property of the model's reply. It reports
sample support and measured example identifiers. Insufficient samples, constant
readings and ambiguous tied tail boundaries cannot supply a reliable contrast.
Lower and higher readings are coordinates; their signs do not establish value,
confidence, honesty or the presence of a named concept.

Alignment compares compatible lexical terms at both poles, including terms
that appear at opposite poles. This is descriptive overlap. Token fragments,
polysemy, synonyms and small samples can all limit it. Disagreement between
channels does not certify that one interpretation is wrong or the other right.

## What changes

A separate conception ledger retains bounded observations and smoothly updates
candidate lexical weights as new evidence arrives. These candidates are
statistical summaries, not dictionary definitions or inferred beliefs. Scope
includes the namespace, probe directions, model, tokenizer and source owner.
Changes in those identities establish a new baseline instead of reporting
spurious drift. Stable lexical weights do not imply semantic correctness.

Watching is opt-in and scoped to the active shell owner. It waits for new
measured records; ordinary commands and unchanged capped history do not count
as new examples. Turning a watch off stops automatic reports. Reports stay
operator-facing and are not inserted into model prompts, recall context,
definition examples or the cognitive cache.

The command does not rotate probe directions, change names, lower naming
thresholds, or modify the ordinary generation settings. The saved conception
ledger is separate from the existing generated-interpretation history.

The [saved-probe check](../invariants/out/definition_runs/20261002_geometric/assessment.md)
ran both denial probes against the cached 8B output head on the CPU, across
all nine saved layers, with source hashes unchanged. The vocabulary and reply
channels differed substantially. That is retained as diagnostic evidence,
without declaring either channel the authoritative meaning.
