# Exposure-scoped diction plasticity

Use existing exposure commands:

```
:probe expose curiosity
:probe expose curiosity off
```

While exposed, each probe gets an owner-local lexical association map. Ordinary
conversation replies associate words and short self-referential phrases with
their measured signed probe readings. Associations update gradually and decay;
maps retain at most 256 terms. A phrase needs three observations before appearing
as learned diction. Utility/file-writing completions and candidate-text tool
reads do not train these maps.

Before generation, exposed probe context supplies readings and changes,
prior pole and intensity, diction learned on both sides of the axis, and
contrasting measured reply excerpts when available. The original contrastive
seed sentences are not supplied as the sensor's meaning. Strength is
relative to the stream's historical RMS; insufficient history is explicitly
uncalibrated. Weak expression does not imply a strong opposite. A reply's new
reading cannot retroactively certify the description used to produce it.

Model-facing identifiers are stable `p_` hashes. Only explicit probe references
are converted: `probe_reality`, `probe:reality`, `[probe:reality]`, and probe-tool
targets such as `<<PROBE: reality>>`. Standalone delimited candidate/naming
queries convert their target too. Ordinary speech like "part of reality" or
"I speak with confidence" retains its words even when those words name probes.
Tool candidate text, proposed names and supporting observations retain their
wording unless they contain an explicit probe reference. Reprocessing a hash
keeps the same identifier. Structured field state, spawn reports and sensor
results emit identifiers directly, instead of relying on word substitution.
Original archived speech remains unchanged; definition and memory context apply
this rule when rendered. Operator-authored seed
sentences still mint the vector and identify its saved namespace. Probe tool queries resolve exposed hashes back
to operator names. Operator commands, logs, persisted probe names and vectors
stay readable. Hashes are identifiers, not cryptographic access credentials.

Terminal probe lists, readings, exposure notices, mint notices and routing
winners show the operator name followed by its exact `p_` identifier in light
blue. This display formatting is applied only to console output; model
context continues to use anonymized identifiers without the operator label.

Hiding a probe removes its description and diction from subsequent prompts,
revokes hash-based consultation, and stops language-map learning. Learned maps
persist per tuner owner in `.plasticity.language.json` files and resume on
re-exposure. Changing the framings gives the probe a separate map. Restart
does not expose any probe merely because a language map exists.

This plasticizes lexical associations and uses them as bounded prompt context.
It does not train checkpoint weights or establish emotional experience.
Self-descriptions remain hypotheses grounded in sensor evidence. Real model
language quality and performance effects still require live evaluation.

## Existing cache compatibility

Old cognitive cache metadata receives additive `probe_ids` references and a
`probe_identity_schema` version on load and write. Operator labels remain
available for provenance. The saved-cache migration script makes the same
metadata-only change with a `.before-probe-ids.bak` backup. Cached vectors,
entry order and tensor dtypes stay unchanged. Retired `probe_` labels in
model-facing cached text also receive stable hashes.

Historical latent vectors cannot be retroactively converted into measured
lexical associations. Language-map training starts with exposure-observed
replies; the migration does not invent old exposure evidence.

## Model-assigned names

An exposed sensor can receive a local naming hypothesis through the probe tool:

```
<<PROBE: rename p_HASH || investigative focus || 0.7 || stronger readings accompanied checking alternatives>>
```

The actual hash comes from the exposed context. The model must supply a plain
label, finite confidence between zero and one, and supporting observations.
Confidence is explicitly self-reported. The store records the number of
exposure-observed replies independently; naming does not manufacture evidence.
The hash remains the query identifier and operator names remain unchanged.

New naming requests require confidence at least 0.65. A completed behavioral
definition attempt supported by at least four measured replies, articulating
a connected contrast between measured ends, lowers that threshold
to 0.35 for its namespace. This is a provisional naming policy, not a calibrated
probability. Supporting observations, exposure and collision checks still
apply. The lower threshold persists with the namespace and does not transfer
to another definition, Other binding, or owner. Existing aliases remain intact.
Incomplete attempts and finished recaps remain available for inspection but
do not lower the threshold or enter exposed interpretation context. Legacy
summaries are rechecked on reading. Contrast qualification recognizes structure,
without certifying that the inferred meaning is correct.

Names and up to 16 prior revisions persist in the owner's language map.
Re-exposure restores the previous hypothesis within the same definition
namespace. Changed framings use a separate namespace and do not inherit a name.
Labels cannot collide across retained definitions in that owner. Hidden hashes
cannot be renamed, and hidden names do not appear in the exposure context.
