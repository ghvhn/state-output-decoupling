# Machine learning changes — September 30, 2026

## Prediction learning

Every matured prediction is a full observation, whether it was right or wrong.
An incorrect prediction supplies evidence about what happened; it is not discarded
or treated as a partially valid record. Unobserved futures beyond a sequence's end
remain unobserved.

Evidence applies in matching internal-state environments, with source token,
horizon, and stream context. Certainty can differ across dimensions. Corrections
are gated to supported directions rather than indiscriminately applied everywhere.
External arrivals and the model's generated tokens have separate timelines.
Learned prediction evidence is saved for the matching model.

## Sinusoidal composition

`:lookahead combine linear` remains the default. `:lookahead combine sinusoidal`
combines eligible correction directions as sine phases with a bounded amplitude.
The hidden state itself is not sine-transformed. Contributions can reinforce or
cancel, and multiple phases can also reverse an adjustment. This is an experimental
alternative; implementation checks do not establish better answer quality.

## Documents and internal reading

Loading a document starts a pass through all its passages. The model's captured
internal state ranks unread passages for relevance, with file order as a fallback.
No hardcoded curiosity instruction is appended to each passage.

Reader outputs remain internal thoughts with document provenance; normal outputs
are recorded as speech. Reader thoughts carry forward into later conversation,
and their observed states are tracked separately from speech states. This is
runtime separation and state observation, not a trained latent reader/speaker
switch or a demonstrated intrinsic curiosity reward.

Successful processing marks a passage read. Interrupted or failed passages remain
available. `:doc read` resumes; `:doc reread` starts a fresh pass for changed context.
Source deduplication does not prevent rereading. `:doc stop` pauses the pass.

## Independent spawned agents

One loaded checkpoint is shared. Each ordinary agent owns compact synaptic
changes, cognitive cache entries, travel state, prediction evidence, calibration
values, probe histories, memory records, and tool configuration. Saves contain
these sidecars; they do not contain copies of the checkpoint, vocabulary, or frozen
prediction network. Exact agent and model identifiers keep saved states separate.

New identifiers start with no inherited learned changes. Reusing an identifier
resumes its own saved changes. Name and trait clauses explicitly request a profile;
an identifier alone does not generate a personality. Existing profiles remain
executable setup scripts, and their resulting changes belong to the target agent.

Examples after restarting the shell:

```text
:spawn base replace 10
:spawn reader join
:spawn reader join name careful_reader
:spawn analyst create traits careful, concise, interested in evidence
```

The reserved `base` agent applies no learned corrections and uses the checkpoint's
native conversation format. It skips profile generation, routing, synthesis,
cache changes, and shell tool injection. Other agents can learn independently.
Replace, join, and targeted commands all select the correct owner and restore the
previous owner after completion or errors. Background consolidation writes to its
own engine rather than whichever agent currently speaks. Dropping an agent stops
its background workers and releases its hooks while retaining saved changes.

## Startup and recovery

The missing `Path` import that stopped startup is fixed. Reader errors preserve
unread progress and restore the main shell after a spawned agent fails. Synaptic
saves now restore trace metadata and counts, and use atomic replacement.

## Verification limits

Checks cover semantic evidence, correction composition, document passes and
rereading, channel boundaries, real small-model generation, independent agent
ownership, persistence, and error recovery. Real model checks confirm that the
checkpoint parameter values and storage remain unchanged and shared, and that a
clean base pass matches direct generation for the same input. The live 8B shell
has not been exercised end to end after these changes. Restart it to load the new
code; response-quality gains require comparative evaluation.

## Learned gravity geometry

`:plasticity` now controls persistent outcome-weighted co-activation links in the
shell. After a scored turn, centered probe signals update pair strengths, followed
by decay and an atomic save beside the active tuner's file. Each tuner owns its
links and recent signals; spawned agents do not borrow the main agent's geometry.
Unscored turns refresh signals without inventing an outcome or changing links.
Autocomplete and utility generations do not update this learning state.

The next gravity-steered turn uses the previous completed turn's probe signals
as stimulation. Positive links reduce effective cosine distance; negative links
increase it. This is learned steering geometry, not a modification of checkpoint
weights or a demonstrated biological model. Existing field saturation and caps
still apply. Benchmark generations keep their existing unmodified field.

For distance changes without additional mass modulation:

```text
:plasticity set mass_strength 0
```

Inspect with `:plasticity`, `:plasticity top 10`, `:plasticity pair A B`, or
`:plasticity history 10`. Tune with `:plasticity set learning_rate 0.05`,
`:plasticity set decay_rate 0.005`, and `:plasticity set strength 0.4`.
`:plasticity off` restores ordinary field geometry and suspends learning/decay;
`:plasticity on` resumes. `:plasticity reset` clears links; `:plasticity save`
flushes them explicitly. Control changes persist automatically. Knobs must be
finite and nonnegative; decay and potentiation/depression caps are limited to 1.

Model-free checks cover owner separation, restart recovery, missing outcomes,
invalid controls, and the actual gravity force calculation. Live 8B behavior and
answer-quality improvements remain unverified.

Validation of the resumed plasticity integration: 116 focused checks passed
(plasticity, source attention, stimulation, synaptic memory, and synaptic owner
isolation). The full `tests` run reported 232 passed and 43 failed, including
subtest failures. Remaining failures involve checkpoint generation hooks,
internal memory continuation, and missing spawned-agent shell helpers. The
separate gravity regression script also stops at the missing
`build_space_effects` helper. These wider regressions are unresolved; the shell
must not be described as fully validated.

## Local geometry continuation

The later continuation restored the missing field helpers, connected selective
spaces, drift, rotations and impulses to capped live hooks, and added nonlinear
learned distance deformation driven by each current token. Strongest signed
neighbors compile per target/layer; released, excluded and incompatible vectors
stay inert. Conditional law gates now consume token-duty telemetry.

The final focused run passed 128 checks across geometry, historical gravity
regressions, plasticity, synaptic memory, stimulation and owner isolation.
This supersedes the earlier missing-space-helper failure above. Broader known
shell regressions and live answer-quality improvement remain unverified.
See [controls and model](LOCAL_FIELD_GEOMETRY.md) and
[historical evidence](AGENTIC_GEOMETRY_CHAT_AUDIT.md).

## Observation-first target learning

Added `:relate target <stream|probe> maximize|minimize <controls...>`.
The owner-local controller observes natural changes in joint probe expression
and target changes within coarse signed contexts. Status reports evidence and
variation; observation adds no steering. Explicit `:relate trial` permits
bounded comparisons only after sufficient varied observations. Missing target
measurements break attribution; restart retains evidence but drops pending
actions. General sense-weighted plasticity remains separate.

134 focused tests passed, including natural-change learning, target direction,
insufficient variation, owner separation, persistence and live overlay
integration. See [target relation controls](TARGET_RELATION_LEARNING.md).
