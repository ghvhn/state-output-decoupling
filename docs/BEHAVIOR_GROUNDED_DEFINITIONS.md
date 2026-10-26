# Behavior-grounded namespace definitions

## Directional distance in next-token logits

`:probe <name> explain logits` and `:probe explain <name> logits` measure the
selected direction's local effect on next-token logits. This diagnostic uses an
exact blank user message and native assistant decision header at the last
selected probe layer. It evaluates signed perturbations with a shared steering
cap, and reports Fisher sensitivity, Fisher-Rao distance, and signed token effects.
The positive, negative, and baseline counterfactuals share the same fresh,
unsteered token history. This shell mode does not inspect the current live,
steered conversation. The library can instead accept an exact assistant-prefix
ID sequence while keeping all counterfactual histories identical. It does not
generate a continuation, supply a written instruction or probe framings, or
require measured behavior examples. Select a named probe or displayed ID;
the diagnostic does not ask the model to choose its target.
Active PyTorch forward hooks make the diagnostic unavailable and are preserved;
they could change the supposedly identical counterfactual history. A step that
rounds away at the residual precision is also unavailable. Each resolved step
records its actual magnitude and direction error, alongside a half-step check.
The report also compares the signed pair's midpoint with the baseline. Agreement
between two finite-difference slopes alone can hide precision or curvature effects
around that baseline; both diagnostics and the native tensor dtypes are retained.

`:tune probe_explain_logits_epsilon <value>` sets the perturbation step, default
`0.01`, with a strict finite range `0 < value <= 0.1`. This measurement control
persists with the owner and is excluded from automatic and outcome calibration.
Results are dedicated operator traces with status `diagnostic`; they do not
enter explanation-candidate, naming, or conversational behavior history.

These measurements describe predictive geometry in vocabulary space. A token's
logit response to the selected direction is an observable local effect, not a
verified definition of that probe or evidence that the model describes its own
meaning correctly. The residual map below is an actuator that predicts a
correction from a current state. Its state-change fit and correction norms are
different quantities from directional distance in logits. Fit residuals and
source-radius extrapolation diagnostics are not measurements of probe meaning.

An isolated cached 4-bit Llama 3.1 8B check on 2026-10-13 measured the stored
`illeism` axis at layer 30 with epsilon `0.01`. Fisher sensitivity was `0.675512`
and the negative-to-positive distribution distance was `0.0134171`. The half-step
derivative differed by `2.1%`, with no cap clipping. Reversing the axis preserved
both distances and exactly reversed the reported token sensitivities. The native
float16 logits showed midpoint departure of `49.6%` of the signed step, so the
local estimate has substantial asymmetry around the baseline despite the
half-step agreement. This check establishes resolved signed output effects in
that blank context; it does not establish an exact gradient or a definition.
See the [logit distance results](../invariants/out/activation_runs/logit-distance-20261013T195351Z-00c286eb/results.json)
and `scripts/probe_logit_distance_test.py` for the reproducible bounded check.

## Activation-only probe explanations

`:probe explain <name>` (also `:probe <name> explain`) defaults to an
activation-only candidate explanation. The explicit suffix `activation` selects
the same mode. It does not require measured examples. Generation begins from an
empty user message in the model's native chat template, followed by the assistant
header, with no textual request or prefilled answer opening.

The selected probe's direction supplies the subject. Separately scaled operation
and reference directions provide candidate controls for explaining that subject.
The contributions are combined at the assistant decision position under the live
steering cap. Matched contrast pairs calibrate the operation and reference axes
in separate forward passes on first use; their calibration text is absent from
the actual explanation input.
Calibration reads the raw decoder block output before terminal normalization,
matching the point where the combined control is injected. Capture hooks are
removed even when a calibration forward pass fails.
Cached axes are scoped to the owning tuner, model identity, selected layers, and
selected subject, so an axis from another owner or subject is not silently reused.

| Tuning control | Default | Range | Intended effect |
| --- | ---: | --- | --- |
| `probe_explain_subject` | 0.02 | 0..1 | Mark the selected probe as the subject |
| `probe_explain_operation` | 0.12 | -1..1 | Favor explanation through the operation axis |
| `probe_explain_reference` | 0.04 | -1..1 | Refer the explanation to the marked activation |

Each gain is independent and finite; zero disables its contribution. For example,
`:tune probe_explain_operation 0.08` changes the operation gain while preserving
the subject and reference gains. These are candidate controls. Their geometry and
the generated prose do not establish the probe's meaning or semantic accuracy.
The operator checks output form without supplying a preferred probe meaning.
Greetings, invitations to chat, advice, memory recaps, and text without a general
definition relation are reported as unqualified continuations. A response that
reaches the generation limit is incomplete. Passing prose remains an unverified
candidate and does not train naming or authorize a rename.
Tools, cache access, routing, synthesis, and lookahead are disabled for this
diagnostic generation. Temporary configuration and steering hooks are restored
after success or failure; diagnostic pushes have their own telemetry and do not
enter the normal steering-cap calibration pool.
Attempts, including failed continuations, are appended to a dedicated
`.activation-explain.jsonl` file beside the owner's plasticity or behavior data.
These operator traces record the resolved cap, controls, calibration identity,
token telemetry, and output-form assessment; they are absent from model context.

### Sustained steering as a map

`:tune probe_explain_map 1` selects a candidate current-state map for the operation.
This is an actuation path. It does not measure directional distance in logits
or qualify a semantic interpretation of the selected probe.
The default `0` keeps the decision-only pulse. This sustained path does not reuse
one vector at every token. A low-rank affine conversion predicts a raw correction
from the current block output:

`delta(h) = mean_delta + bounded(h - mean_source) @ source_relations.T @ coefficients`

The operation gain scales that raw correction, preserving learned magnitude.
The correction shares one cap with subject and reference pulses at the original
assistant decision. During decoding the map is evaluated at each assistant
prediction position; full-prefix rebuilds reproduce the same causal operation
on assistant positions. User-body tokens are unchanged. Only the final selected
layer carries the map, avoiding a stack of independently calibrated changes.
The map rejects constant fitted corrections and validates its stored tensors
once before attaching hooks; runtime evaluation does not repeat validation or SVD.

Two generic written requests supply calibration trajectories. For each trajectory,
the source has the exact blank user body and the target has its calibration
request. Both append the identical assistant token IDs at each prediction position,
without decoding and retokenizing a prefix or closing the assistant turn. A ridge
fit learns the source-to-change relation across those paired states. Source
relations are radially bounded to the observed training neighborhood, and live
diagnostics record extrapolation and changes in correction magnitude. The owner,
model, tokenizer, selected subject, layers, evidence and calibration budget scope
the in-session cache. The calibration teacher text is an operator artifact;
it is not a probe meaning label or runtime instruction.

`:tune probe_explain_map_tokens <3..64>` sets the per-trajectory calibration budget,
default `24`. These prefixes may end before the teacher finishes; fitting their
activation differences does not qualify the unfinished prose as an explanation.
No calibration output or generated attempt trains conversational behavior.

### Live model result, 2026-10-13

The initial arithmetic is mechanically implemented but has not established
useful explanation or subject binding. An isolated, cached 4-bit Llama 3.1 8B
run with the stored `illeism` direction reproduced the reported generic chat
opening exactly. All decoded comparison inputs had an empty native user body:

| Comparison | Observed output |
| --- | --- |
| All gains zero | Generic offer to help |
| Current `.02/.12/.04` controls | The reported generic chat opening |
| Blank-source-to-request operation, one decision pulse | Generic offer to help |
| Blank-source-to-request operation, sustained during decoding | Repeated `I` until the token limit |

Neither alternative was promoted to the default. The differences in first-token
logits establish that steering acted, while the prose fails the explanation task.
See the [isolated comparison results](../invariants/out/activation_runs/transfer-20261013T063326Z-1e2bf5aa/results.json).

A separate positive control supplied each calibration request as text, with and
without the selected subject pulse. Each pair produced identical decoded prose,
with no observed description specific to the selected probe. This control is
explicitly prompted and is not an activation-only runtime result. See the
[teacher control results](../invariants/out/activation_runs/teacher-control-20261013T063841Z-dfcd6a2d/results.json).
The subject-to-description relationship needs independent validation; increasing
gains or sustaining the same axes is not a demonstrated fix.

A subsequent current-state map trial used 48 paired prediction states at layer
30, with a separate held-out request and trajectory. The fitted conversion was
adaptive and its correction changed over the 17-token generated reply (predicted
raw norms ranged from 19.52 to 39.99). Its held-out relative state-change error
was 0.924, versus 0.996 for the fixed mean. These are geometric measurements,
not semantic success. The fixed-mean control and the current-state map both
finished with generic chat, retained as unqualified. This trial supports the
implemented state-conditioned actuator but does not establish useful probe
explanation or subject binding. See the [map comparison results](../invariants/out/activation_runs/state-map-20261013T180143Z-bade3b35/results.json)
and the reproducible `scripts/activation_explain_map_test.py` diagnostic.

The `evidence` suffix selects the measured-behavior workflow below. `prompt` uses
probe activation steering with a textual definition request and the
`This probe measures ...` opening; it does not inspect stored definitions or seed
framings. `ping [alpha]` retains the activation nudge sanity check.
`:probe define <name>` displays the original stored definition and framings.

## Explanations from measured behavior

`:probe explain <name> evidence` requests a compact
definition of the behavioral property inferred from measured examples. The
name may be a displayed `[p_...]` identifier, including an unexposed probe.
The native assistant response starts with `This probe measures `; the request
asks for a general property or rule and what lower versus higher readings mean.
It does not supply the seed framings or a preferred meaning. A mistaken
interpretation can qualify: the output check concerns definition form rather
than semantic accuracy. Bare comparisons, memory recaps, lists and advice do not
qualify. One format correction is allowed, using the original evidence and the
reason for rejection. Failed prose is never supplied as another measured memory.
If both attempts fail, the command reports the failure and retains the attempts
for inspection instead of displaying a recap as a definition.

The request and assistant opening exist only within this command. Cache writes,
routing and synthesis remain paused, generated attempts are not scored as new
behavior, and subject hooks and configuration are restored after success or
failure. At least four measured examples with varying readings are required;
`:probe backfill <name>` can import archived replies when needed.

Each owning tuner now keeps a defining profile in a `.defining.pt` file beside
its plasticity data. It records each attempt's exact request suffix, assistant
opening, token budget, decoding and steering settings, structural result, and
hashes of evidence and output. Successful requests are reapplied to fresh
evidence rather than retaining old examples or answers in the request. Use
`:probe defining show` to inspect counts and replay outcomes. Turning it `off`
disables reuse while keeping observations; `on` enables reuse again.

For each explicit attempt, the profile captures raw layer-block states at the
last assistant-prefill token before temporary subject or defining hooks are
attached. Both captures use the same `This probe measures ` opening: one sees
only measured evidence, and the other also sees the actual definition request.
Their unnormalized difference records the prospective activation change
associated with that request. Qualified explicit attempts contribute finite
source/change pairs; failed and truncated attempts retain their outcome without
training steering. These associations establish neither the inferred meaning
nor which change caused the output.

After at least three distinct successful source states, a source-conditioned
conversion predicts a change for fresh evidence. Replay is restricted to the
same model weights, tokenizer, load precision, layers, capture phase, request
version and settings, within the owning tuner. It applies a bounded change at
the last selected layer's original assistant-decision position, sharing the
subject marker's steering cap. Replay uses evidence and the assistant opening;
it omits the explicit request that produced the captured changes. If its output
fails the form check or reaches the token limit, the second attempt uses the explicit request with the
current format correction. Qualified replay outputs are counted separately and
do not train themselves. Captures and profile histories remain operator data,
outside conversation prompts. Conversational lookahead is paused during these
calls and restored afterward. Atomic, validated saves preserve prior files on
failure; a save failure leaves the current definition and in-session observations
usable.

The earlier activation-only workflow remains available as
`:probe define-experiment <name>`. The `probe_define_process` and task, step and
schema tuning controls select its experimental controller. The historical
mechanisms and results below describe that experiment.

An isolated offline check on the cached Llama 3.1 8B Instruct model produced
definition-shaped answers on the first attempt for two distinct held-out
example sets (revision of estimates and confidence in conclusions). The check
used real stored probe directions and the shell's generation path, with zero
cognitive-cache reads or writes. Its example texts and assigned readings were
synthetic, so it verifies output kind rather than the meaning of live probe
measurements. Captures are in
`invariants/out/definition_runs/20261011_output_validation/`.

The separate [`:probe define2` diagnostic](GEOMETRIC_PROBE_CONCEPTIONS.md)
compares vocabulary projections and measured reply language without generation.
Its lexical conception history does not replace generated interpretations,
certify a meaning, or change probe directions.

The experiment's default activation controller is the four-stage candidate process described in
[Stored definition operations](DEFINITION_STEP_VECTORS.md): compare, isolate,
distill and verify. Each operation is stored as its own per-layer vector in a
versioned bank. Each stage has a separate bounded generation call, and a failed
structural or coordinate check stops advancement. Only the accepted contrast is
passed to the next stage as a candidate artifact. These checks do not certify
semantic truth or successful execution of each operation. The evidence,
qualification and namespace contracts below still apply. The earlier
question/task/subtraction controller remains available with
`:tune probe_define_process 0`; its mechanisms and failed live trials below
describe that optional experiment.

The candidate [relational conversion path](RELATIONAL_DEFINITION_CONVERSIONS.md)
is available with `:tune probe_define_process 2`. It learns differences between
conversions from relationships between source states, and predicts fresh
direction and magnitude for each stage's evidence. Its activation fit is separate
from structural qualification and semantic correctness.

```
:probe define-experiment <name>
```

No prompt argument is needed. The command attempts to interpret actual measured
behavior instead of printing the operator's WITH/WITHOUT framings.

Ordinary conversation replies are stored together with their probe readings,
input excerpts and separate sense outcomes. The ledger retains 64 examples per
probe definition and Other binding, scoped to the owning tuner. Definitions
require at least four examples with varying readings. The model receives two
low-reading and two high-reading memories, using the probe's hash rather
than its operator name. The definition text identifies the probe and each
memory's measured location along its spectrum, with zero as the rolling
baseline. It supplies the remembered input and reply without seed meanings.
Ordinary words in these excerpts remain plaintext, including words that happen
to name probes. Only explicitly notated probe references become hash identifiers.
Every memory carries a positive-side, negative-side or baseline label derived
from its signed coordinate. The printed interpretation receives the same
labels on numbered memory references, independent of what the model generates.
Lower sampled positions are not called negative if they are above zero.
The text supplies no
outcome labels, naming instruction or suggested interpretation. The definition
task is carried entirely by injected activations. These locations are measured
associations, not normalized probabilities or causal proofs.

The selected probe is injected into the activation stream at the last original
prefill token as a subject marker. The chunk engine rebuilds the whole prefix
every 24 tokens, so each rebuild restores the marker at its original position.
Cached decode steps do not inject it into new tokens. A separate question axis
carries the user's proposed "What is ___?" operation. It is minted from two
matched native user question/declaration pairs using the same opaque subject
within each pair, sampled at the assistant's first decision position. The
actual subject still comes from the selected probe activation. These question
sentences are used only during minting; they are absent from the definition's
text input.

Four further candidate
task axes augment comparison of the measured ends, abstraction into an
operational distinction, a provisional term, and reduced verbosity. A shorter
memory recap still fails to articulate a definition.

Each of these four axes is minted from two pairs of completed assistant demonstrations under
the model's native chat template. Within a pair the neutral synthetic spectrum
evidence is identical; the assistant responses differ in the target behavior.
These demonstrations contain neither the selected probe's seed meanings nor
its actual memories. Each case difference is normalized before averaging and
normalizing the resulting axis. They are candidate controls, without a claim
that geometry alone establishes their behavioral effect.

The task axes are cached by model identity, task version and exact subject
layers. All enabled controls must cover every subject layer. The original five
controls apply at the assistant decision position and later generated positions;
the experimental operation is a single decision-position activation, restored
at the same position during prefix rebuilds, without injection into later output
tokens. Earlier evidence tokens receive no direct injection. All contributions share the live
steering cap. The shell reports actual layer coverage before generation.

Defaults, as signed fractions of the current token's residual norm:

| Tuning control | Default | Intended effect |
| --- | ---: | --- |
| `probe_define_question` | 0.080 | Ask what the activation-marked subject is |
| `probe_define_contrast` | 0.040 | Compare lower and higher behavior |
| `probe_define_abstraction` | 0.040 | Distill the comparison into a distinction |
| `probe_define_naming` | 0.025 | Suggest a provisional term |
| `probe_define_verbosity` | -0.035 | Suppress repetition of the same distinction |
| `probe_define_operation` | 0.000 | Contrast defining with recapping at the assistant decision position |
| `probe_define_subtract` | 0.000 | Remove measured nuisance overlap from enabled definition controls |

For example, `:tune probe_define_verbosity -0.05` adjusts the last control.
The subject marker remains 2%, combined with the task controls before capping.
Signed task gains must be finite and within -1 to 1; zero disables that control.

The experimental operation direction is minted only when its gain is nonzero.
Two matched request pairs carry the exact selected evidence and request
either one concise operational definition or one concise recap. Both are
sampled at the assistant decision position. Synthetic evidence remains available
to the standalone minting API for controlled checks. This addresses a possible mismatch
between the other four controls' completed-answer states and the position where
they are applied; that explanation remains a hypothesis. The teacher requests
are used to produce activation differences and never enter the actual definition
input. The operation has its own model/layer/version/evidence-hash cache, bounded
to eight entries with corresponding nuisance states evicted together.

Subtraction is an optional refinement of enabled contrast, abstraction and
operation controls. Matched pairs vary verbosity, list presentation, topic or
generic conversational framing while holding the intended operation fixed.
Each difference is normalized before its two-case average is normalized. The
completed-answer controls use completed-answer nuisance states; the operation
uses separately measured assistant-decision nuisance states conditioned on the
same exact evidence. Those contextual nuisance pairs vary verbosity, list
presentation and generic framing, without rewriting the remembered examples.
Topic is held fixed within the contextual difference and is not claimed as a
separately measured contextual nuisance. Other controls
and the selected probe's geometry are preserved.

At each layer, an SVD resolves an orthonormal basis `Q` for the nuisance span.
For a unit task direction `v` and subtraction strength `s`, the candidate is
`unit(v - s * Q @ Q.T @ v)`. This avoids repeatedly removing correlated or
duplicate directions. Strength ranges from 0 to 1. The method refuses to
amplify a cancelled residual or one retaining less than 10% of the original
norm. It records rank, retained fraction, and nuisance overlap before and after
subtraction. Those numbers describe geometry, not semantic precision.

Changing strength recomputes from raw cached directions, so repeated refinement
does not accumulate. Disabled task controls are not refined and cannot cause
cancellation failures. All controls still share the residual cap. Every saved
attempt carries the gains, task/operation/refinement versions, subject gain,
subtraction strength, injection positions and geometric diagnostics as operator metadata; those
settings are absent from model context.

During the diagnostic run, cache reads/writes, automatic routing and synthesis
are disabled. The output is not automatically scored as a new behavioral
example. Hooks and configuration are restored even on failure. The output is
saved as a revisable attempt, not as a certified definition or a forced rename.
A completed attempt must articulate a substantive, connected comparison of
measured ends to qualify as an interpretation hypothesis. A conservative
structural check rejects independent recaps, bare assertions of difference,
unknown memory IDs and references to spectrum sides absent from the supplied
measurements. This check does not certify the meaning, evidence fidelity or
causality of the proposed distinction.

Qualified hypotheses become visible with that namespace while exposed and
lower that namespace's naming confidence threshold
from 0.65 to 0.35. The diagnostic can emit a rename tool query for the selected
exposed hash; the shell applies it immediately using the lower threshold and
the normal evidence and collision checks. The model still chooses its term.
The legacy diagnostic allows up to 384 generated tokens. The default candidate
pipeline allows up to 384 per stage, with at most four stages and no automatic
retries. Outputs that reach their stage budget are retained as incomplete,
omitted from exposed interpretation context,
and do not themselves lower the naming threshold or apply a rename. Finished
recaps are retained as `unqualified`, with a reason, and likewise cannot do so.
Existing saved summaries are checked when read; legacy recaps remain on disk
but no longer appear as interpretations or support the lower naming threshold.
Exposed interpretation context uses the recognized comparison block, so a
distinction stated after memory recaps is not replaced by the first recap.

Changing framings or Other binding selects a separate namespace. Up to 16
interpretations are retained per namespace. Formal model-assigned aliases
continue to use the exposed naming tool and its confidence/evidence fields.

Older numeric histories without paired reply text are not treated as behavioral
examples. `:probe backfill <name> [n]` now imports the freshly re-scored archived
replies into this ledger as well as rebuilding the numeric stream. The first
reply only seeds each rolling baseline, so its synthetic zero is excluded from
definition evidence. The recorded input is the preceding user turn in the same
scope and session, consumed once; missing input remains empty. At least five
archived replies are therefore needed to supply four centered examples, and
their readings must still vary.

Archive examples carry their source timestamp and a backfill marker. Repeating
the backfill updates its existing examples rather than counting them again,
even if figure rows already cover the archive. Ordinary live examples are
retained. Historical Other readings require their original resolved binding;
the current participant is never assigned retroactively. This imports measured
reply associations, without replaying them as plasticity training turns.
`:probe explain <name> prompt` provides an explanation under probe activation
steering and an explicit textual request, without reading stored framings.

Other-related examples collect in parallel for every measured participant,
including self. Selecting `:other self` changes which namespace is inspected;
it does not start its data collection. Each parallel reading carries its resolved
binding and is paired with the same reply. Equivalent bindings are counted once.

Focused tests cover paired evidence, namespace isolation, native assistant
demonstrations and user questions, independent signed controls, decision-position
injection, shared bounds, cleanup, qualification and legacy summary handling.
These use toy or mocked models and establish wiring, not semantic effectiveness.

On 2026-10-01, two live Llama-3.1-8B-Instruct 4-bit runs used the same stored
`denial_minus_confidence` evidence, subject gain 0.02, and 384-token budget.
Adding question gain 0.08 to the existing four controls produced another
recap and reached the token limit. Question gain 0.12 with those four controls
disabled also produced a recap and reached the limit. Both attempts remain
incomplete and unqualified; the namespace's naming threshold stayed at 0.65.
The question control is available for further experiments, but these runs
do not establish successful activation-only definition behavior. Captures and
settings are in `invariants/out/definition_runs/20261001_question/`.

Two further live runs on the same model and evidence used nuisance subtraction
strengths 0.5 and 1 with the original controls enabled. The minimum retained
task norms were 90.3% and 86.9%. Full subtraction reduced measured nuisance
overlap to numerical roundoff, yet both outputs remained recaps and reached
the token budget. Both are incomplete and unqualified. Subtraction stays off
by default; these results do not establish a precise definition direction.
Captures, settings and diagnostics are in
`invariants/out/definition_runs/20261001_subtraction/`.

The decision-phase operation was also tested at gain 0.12, with the original
five controls disabled and continuous injection into generated positions.
Without subtraction it repeated "at the higher" to the token limit. With full
same-phase subtraction (minimum retained norm 85.8%) it repeated a generic
conversation summary to the limit. Both attempts remain incomplete and
unqualified; captures are in `invariants/out/definition_runs/20261001_operation/`.
These failures motivated testing lower gains and decision-only activation.

Decision-only pulses with the synthetic operation direction at gain 0.04
without subtraction and gain 0.08 with full subtraction finished as memory
recaps, retained as unqualified. Gain 0.04 with full subtraction produced a
generic summary that hit the token budget. The lower gains and single pulse
avoided the earlier repetition but did not establish definition behavior.
Captures are in `invariants/out/definition_runs/20261001_operation_pulse/`.

Finally, the operation was conditioned on the exact selected memories and
tested with decision-only gain 0.06. Both the raw direction and full contextual
nuisance subtraction finished as the same per-memory recap and were retained
as unqualified. Full subtraction retained at least 86.8% of the task norm.
The experimental operation and subtraction controls were restored to zero.
Captures and complete settings are in
`invariants/out/definition_runs/20261001_contextual_operation/`.

Across these bounded trials, geometric refinement did not establish a useful
activation-only definition operation. No failed attempt was promoted to an
exposed interpretation or used to lower a naming threshold. The available
controls support further investigation; their semantic effect is not certified.
