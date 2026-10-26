# Stored definition operations

`:probe define-experiment <name>` uses four individually named candidate operations.
The ordinary `:probe define <name>` command requests and checks definition form
directly; see [Behavior-grounded definitions](BEHAVIOR_GROUNDED_DEFINITIONS.md).

| Step | Operation |
| --- | --- |
| `compare` | Compare behavior across the measured spectrum |
| `isolate` | Separate the changing behavior from shared background |
| `distill` | Express the distinction as a general operational rule |
| `verify` | Check the distinction against the examples and possible exceptions |

Each step contains a separate unit vector for every selected probe layer. Two
matched synthetic contexts contrast performing that operation with omitting it.
The model's native assistant decision states supply the differences; each case
is normalized before averaging and normalizing the result. Teacher requests are
used only to mint the vectors. Actual definition generation starts with only
the probe hash and measured memories, with no operation instructions in its text.
Later stages also receive the most recent accepted contrast as a separate opaque
candidate artifact.

On first use, the bank is minted and atomically saved under
`invariants/out/definition_steps/bank-<identity-hash>.pt`. Its four named operations
are separate tensor entries in one complete bundle, so an interrupted save cannot
leave a partially replaced process. Later commands and restarted shells load it
using the restricted tensor loader. All vectors must be finite, normalized,
CPU float32, and match the exact requested layers and model width.

The identity includes the model source and revision or local weights manifest,
tokenizer implementation and native chat template, effective precision and
quantization, library versions, layer selection and minting recipe version.
Different identities receive different files. Corrupt or mismatched banks are
reminted and replaced only after the new complete bank validates. If weights are
changed in memory without a corresponding revision change, the bank must also
be invalidated; this is a stored candidate operation, not a weight-change tracker.

The default process makes a separate generation call for each stage, with a
maximum of 384 output tokens per call and no automatic retries. Only the current
stage's vector is active. It checks the output before proceeding; a rejected
recap or a token-limit exit stops the process immediately. At most four calls
can run, rather than advancing whenever a token counter crosses a boundary.

Comparison and verification must state a connected behavioral contrast that
cites actual measured examples from both sampled ends. Isolation and distillation
may express a general contrast without citations after an anchored candidate
has passed. All stages reject unknown memory references and reliably parsed
explicit coordinates or side assignments that disagree with the measurements.
These are structural and coordinate checks. They do not verify that a stated
behavior occurs in the memories, that a rule generalizes, or that the operation
was cognitively performed. Four repeated well-formed contrasts can pass these
checks; this remains a candidate definition pipeline.

A later call receives the unchanged original evidence plus only the previous
accepted contrast under `[Candidate c_<content-hash>]`. That output is a draft,
not a measured memory or an ordinary conversation turn. Recap paragraphs and tool
payloads are excluded from the handoff. Rejected output is never fed to later
stages. Stage names, gains, checks and operator traces are not inserted into
generation text. Earlier evidence tokens receive no direct injection. The
selected probe remains a subject activation at the original assistant decision
position of each separate call; prefix rebuilds restore that position and cached
decoding does not move the marker onto new tokens.

Each step defaults to 0.04 of the current residual norm. Subject and operation
contributions share the existing live steering cap. All hooks are removed after
the diagnostic, including on failure; unrelated hooks remain intact. Cache
reads/writes, automatic routing and synthesis are paused during generation.

```
:tune probe_define_step_compare 0.04
:tune probe_define_step_isolate 0.04
:tune probe_define_step_distill 0.04
:tune probe_define_step_verify 0.04
:probe define-experiment denial_minus_confidence
```

Signed step gains must be finite and within -1 to 1. Zero disables that step's
push. `:tune probe_define_process 0` selects the prior simultaneous-control
experiment; `:tune probe_define_process 1` selects the gated calls.
`:tune probe_define_process 2` selects the candidate
[relational conversion learner](RELATIONAL_DEFINITION_CONVERSIONS.md), which
fits how each operation's direction and strength change with the source evidence.

Saved attempts carry the namespace, evidence digest, bank identity/path, gains,
stage budgets, separate stage outputs and gate results as operator metadata.
Candidate content IDs and dependency IDs identify which accepted contrast was
forwarded. Only a final candidate that passes all four gates can become a
namespace hypothesis. Incomplete or unqualified attempts cannot lower a naming
threshold or automatically rename the probe. Intermediate outputs are not added
to measured behavior examples, episodic memories or cognitive cache metadata.

Storage and hook tests establish persistence, isolation and sequencing. They do
not establish that the four candidate vectors perform the intended operations
on the full model. Previous simultaneous-vector experiments failed to produce
definitions; their records remain in `BEHAVIOR_GROUNDED_DEFINITIONS.md`.

A tiny randomly initialized CPU Llama with a constructed native tokenizer was
also exercised through the real chunked generation engine. It minted and
reloaded the bank, rebuilt prefixes and decoded from cache, preserved the
evidence-only input, and cleaned up hooks. This validates integration, not
definition quality.

On 2026-10-01 the user's full-model run of the earlier fixed-window controller
finished as a generic conversation recap and was retained as unqualified. The
saved trace shows 320 output positions: 96 compare, 96 isolate, 96 distill and
32 verify positions all received nonzero pushes. The bank was finite, normalized
and distinct, covering all nine requested layers. Delivery worked; behavioral
completion did not. The recap also assigned unsupported emotional meanings to
the spectrum signs. It did not change the probe's 0.65 naming threshold or its
64 measured examples. The captured record is in
`invariants/out/definition_runs/20261001_user_step_failure/attempt.json`.

That failure motivated the separate gated calls. The old token-window controller
is retained for diagnostic reproduction and its cursor trace establishes only
timing. The stored vectors still contrast teacher requests at the assistant
header, rather than completed assistant operations. Neither the changed control
flow nor the bank's geometry establishes successful activation-only definitions
on the full model.
