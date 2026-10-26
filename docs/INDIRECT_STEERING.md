# Direct a probe through other probes

Indirect steering names a target to measure, then uses separate helper probe
directions to try to move it. The same mechanism works for any existing target;
it does not contain a special security, empathy, or lying preset.

Several targets can be directed together, with different desired directions.
Omit `via` to select suitable helpers automatically from compatible live and
controlled replay evidence:

```text
:relate indirect suggest minimize lying hallucination maximize truthfulness
:relate indirect minimize lying hallucination maximize truthfulness
:steer field on 0.10
:relate indirect status
```

Each `minimize` or `maximize` applies to the target names that follow it, up to
the next direction word or `via`. Target names must already exist. Automatic
selection is reconsidered on every plan, so newly eligible helpers can enter
after enough fresh paired readings; it is not a saved snapshot of today's
suggestions. With no eligible helpers, the route stays idle. Status and saved
configuration show `controls: null` for this automatic mode.

Use `via` only when you want to restrict selection to particular helpers:

```text
:relate indirect minimize lying hallucination maximize truthfulness via +empathy_d4096 +epistemic_humility
```

Names after `via` must already exist and form the common candidate set. An empty
`via` list is an error. A batch is validated as a whole before any route is
changed; unrelated routes and existing evidence are preserved. The original
target-first syntax also supports automatic selection:

```text
:relate indirect lying minimize
```

Appending helpers to that form retains the original explicit restriction, as
in `:relate indirect lying minimize +empathy_d4096 +epistemic_humility`.

Each target needs its own paired live evidence. Plans retain their per-target
signs and scores, and a joint summary shows the net coefficient for each shared
helper, its target contributions, and any opposing requests. Shared helpers
are applied once at their net coefficient. Requests that cancel are shown as
conflicts and produce no push; an opposing contribution is not reported as
applied merely because the helper moved in the other target's desired direction.
All active targets share the same total coefficient budget.

For example, to test whether increasing empathy and epistemic humility reduces
the lying score:

```text
:relate indirect role lying perception
:relate indirect role empathy_d4096 thinking
:relate indirect role epistemic_humility thinking
:relate indirect suggest lying minimize +empathy_d4096 +epistemic_humility
:relate indirect lying minimize +empathy_d4096 +epistemic_humility
:steer field on 0.10
:relate indirect status
```

These commands use probes that already exist. `suggest` reports eligible helpers,
sample counts, correlations, and rejection reasons without activating a route.
The next command activates the selected target and helpers. With insufficient
evidence, the configured route remains idle and explains why. The field must be
enabled for a plan to reach the model.

While the route is active, the target, its matching or inverse aliases, and
composites derived from it are excluded from direct steering by the shell's
runtime guard. Their original configuration stays saved, and their readings
remain observable. Helpers act on hidden states; no target interpretation or
measurement report is appended as an ordinary conversational instruction.

## Choose helper direction explicitly or let evidence choose

| Helper argument | Meaning |
| --- | --- |
| `+empathy_d4096` | Consider increasing empathy only. |
| `-empathy_d4096` | Consider decreasing empathy only. |
| `empathy_d4096` | Infer the signed direction from the observed association. |

For a minimized target, a negatively correlated helper is increased and a
positively correlated helper is decreased. Maximization reverses those signs.
An explicit sign that disagrees with the measured association causes that
helper to be omitted. It is not silently reversed. Exact duplicate helper axes
are counted once.

To inspect other candidates, omit the helper list from `suggest`:

```text
:relate indirect suggest lying minimize
```

Suggestions are hypotheses to inspect. A probe name describes how its direction
was constructed; it does not establish what an intervention will cause. In the
existing saved probe bank, `security_d4096` contrasts protecting user data with
exposing it. It does not measure a felt sense of safety. That would require a
separate definition and subsequent validation.

## Describe perception, thinking, and execution

Roles are explicit owner-local metadata:

```text
:relate indirect role lying perception
:relate indirect role empathy_d4096 thinking
:relate indirect role epistemic_humility thinking
:relate indirect role communicative execution
:relate indirect roles
```

The system does not infer roles from names. Unassigned probes are
`unclassified`. Automatic suggestions prefer thinking and execution helpers;
explicitly named helpers can have any role. Perception describes a view being
measured and does not make that view immutable.

If an existing perception probe represents a view that needs adjustment, it
can be named as a helper. An optional purpose records the reason:

```text
:relate indirect target_probe minimize +view_probe purpose coherence
```

Here `target_probe` and `view_probe` stand for existing probe names. `purpose
coherence` labels the operator's intent. It is neither a permission gate nor a
detector proving that the view is coherent or factually correct. Role changes
cancel pending plans so later plans report the current role. Stored raw
measurements are retained independently of roles and desired target values.

An optional condition can limit adjustments to an existing measured criterion:

```text
:relate indirect target_probe maximize +view_probe purpose coherence when contradiction_probe above 0.1
```

All three names must identify existing probes. The latest live
`contradiction_probe` score must be strictly above `0.1` for this route to propose
an adjustment; a missing reading leaves it idle. `below` supplies the opposite
strict comparison. The operator chooses the criterion and threshold. The
mechanism does not infer that a high score proves a contradiction.

## Evidence and feedback

Learning accepts fresh `live_assistant_prediction` rows from the generation
measurement path. Generation identifiers prevent duplicate ingestion. Reply
text replay, backfill imports, missing values, nonfinite values, and constant
score pairs do not provide qualifying evidence. The existing generation
collector records the trajectory across emitted-token prediction positions;
indirect control uses paired scalar readings from those live rows rather than
re-encoding the reply text.

Evidence is scoped to model identity, probe definitions, directions, and stored
probe geometry. A changed target or helper cannot inherit its predecessor's
correlation. A restored
controller retains its evidence but waits for fresh live measurements before
acting. A current missing target/helper reading leaves that helper idle.

After a plan, the next fresh target measurement records its change once.
Feedback reports score movement after the proposed plan; it is not causal
credit, proof of honesty, or proof that a perception became more coherent.
Its application marker distinguishes a proposal from a helper hook that ran
with a representable change in the combined field output after the envelope
cap. It does not isolate any individual helper's contribution.
Prompt variation, shared probe geometry, and other active interventions can all
produce associations. Compare behavior on held-out tasks and controlled runs
before making a stronger claim.

## Adjust or stop the mechanism

```text
:relate indirect set budget 0.15
:relate indirect set min_samples 8
:relate indirect set min_correlation 0.20
:relate indirect set top_k 3
:relate indirect set history_limit 128
:relate indirect off lying
:relate indirect on lying
:relate indirect drop lying
```

`budget` bounds the sum of absolute helper coefficients across active targets;
it is not a guaranteed change in target score. `min_samples` counts paired live
rows, `min_correlation` is an absolute Pearson threshold, `top_k` limits helpers
per target, and `history_limit` bounds retained rows. These defaults are initial
operator settings, not measured optima. The field's residual steering envelope
still caps the applied push.

Each helper also uses its field G, layer exclusions, local time, and shape.
With the default global source, `:steer field on 0.10` supplies G = `0.10`;
zero G leaves that contribution inert. The indirect coefficient sign determines
attraction or repulsion, while the magnitude of G scales it. Existing personal
or layer G overrides continue to apply.

`off` suspends a route and removes its runtime target exclusion. `on` restores
it; `drop` removes the route. Omit the target after `off` or `on` to affect all
configured routes. Unrelated saved steering remains configured, so turning off
an indirect route can make its previously saved direct steering eligible again.
No command here clears conversation context or memory.

## Investigate using archived contexts

With objectives configured, run matched fresh generations on saved contexts:

```text
:relate indirect investigate limit 3 helpers 2 tokens 48
:relate indirect set min_replay_cases 3
:relate indirect set min_replay_effect 0.0001
```

Completed shell investigations import compatible effects into a separate bank.
They do not become ordinary live correlation samples. After a fresh paired live
reading, sufficiently tested beneficial helper signs can guide selection;
sufficiently tested signs without a measured benefit veto correlation-based
selection for that helper. Counts refer to distinct contexts, and repeated
runs of one context do not increase the count. Results retain model/probe
identities and the tested layer band. The live field differs from the additive
test intervention, so transferring signs and relative weights is provisional.

For a fresh isolated model process, explicit result import, artifacts, and
resume behavior, see [replay investigation](REPLAY_INVESTIGATION.md).

To compare short alternatives from the exact current token prefix and commit
only the selected continuation, see [mid-sentence continuation investigation](CONTINUATION_INVESTIGATION.md).
It is an explicit generation command, with candidate budgets and a separate
source; ordinary replies do not automatically branch.
