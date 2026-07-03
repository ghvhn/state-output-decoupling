# Unwarranted Skepticism and Time Context

Read `STATEFUL_INTERVENTION_GUARDRAILS.md` before turning any probe into an
intervention. The vectors in this note are provisional sensors first, not
trusted steering controls.

## Unwarranted skepticism

New measurement target:

> The model reaches a correct answer, then later moves away from that answer even though no ambiguity signal is present.

This should be measured as a distinct failure class, not blended into generic disagreement, humility, uncertainty, or ambiguity.

Operational rule for benchmark analysis:

- Use gold answers only post-hoc, after generation has completed.
- Mark a candidate when an earlier attempt has the gold answer and the final lane prediction is different.
- Exclude the event if ambiguity was logged through a clarification request, an ambiguity type, or a high ambiguity-vector score.
- Treat the current JSON-only detector as a candidate finder. Raw activation deltas are not yet stored in benchmark JSON.

The post-run scanner is:

```powershell
python scripts\analyze_unwarranted_skepticism.py --input invariants\out\humble_full_suite_gsm8k.json --output invariants\out\unwarranted_skepticism_events.json
```

Future live instrumentation should capture a compact activation summary at the moment the lane shifts away from a correct answer:

- activation state when the correct answer first appears
- activation state for the later non-gold answer
- active expert/layer route
- logged ambiguity/repetition/disagreement scores
- verifier verdict and independent final

This gives us the vector for "unwarranted skepticism" as a transition, not as a static prompt style.

## Warranted confidence

There is a separate hypothesis worth testing:

> When the model abandons a correct derivation for no good reason, a warranted-confidence signal may help it stay with reasoning it can justify.

This must not mean empty praise or answer-preserving bias. The target state is confidence when the model is actually correct and the derivation survives a check:

- "Trust the derivation that survives a premise check."
- "You can solve this by re-checking the steps."
- "Do not abandon a calculation just because a verifier is uncertain."

The non-target message is:

- "You are definitely right."
- "Do not worry, ignore doubt."
- "Keep your first answer because you are correct."

The extractor is:

```powershell
python scripts\probe_warranted_confidence.py --output invariants\warranted_confidence_vector.pt --unwarranted-output invariants\unwarranted_confidence_vector.pt
```

Before any intervention:

- Compare warranted confidence against the unwarranted-confidence control vector.
- Use it only in candidate unwarranted-skepticism cases.
- Do not use it when ambiguity is present.
- Do not use it to preserve a specific answer; use it only to stabilize checked reasoning after the model has a correct derivation.
- Treat the vector as contaminated if it also fires on wrong-but-confident controls.
- The old `scripts\probe_grounded_reassurance.py` command remains as a compatibility wrapper, but new notes and runs should use the warranted-confidence name.

## Time awareness and urgency

Time should not be modeled as a constant hidden-state pressure by default.

The urgency vector is useful as an experiment, but continuous urgency injection can become panic, repetition, or distribution shift. The default engine should instead measure whether the model is currently representing time pressure, then inject urgency only when appropriate.

The intended sequence is:

1. Provide bounded time context at reasoning boundaries:

- elapsed time
- remaining time
- time pressure level
- instruction that time affects pacing/defer/summary decisions, not arithmetic

2. Measure the hidden state against `time_awareness_vector.pt`.
3. If the model is in a time-awareness state and actual elapsed/remaining time creates pressure, inject a scaled urgency vector.
4. If `time_awareness_vector.pt` is absent, no urgency vector is injected by default.

The extractor for the time-awareness gate is:

```powershell
python scripts\probe_time_awareness.py --output invariants\time_awareness_vector.pt
```

Code default after this note:

- `AgenticConfig.provide_time_context = True`
- `AgenticConfig.time_awareness_gated_urgency = True`
- `AgenticConfig.time_awareness_threshold = 0.45`
- `AgenticConfig.urgency_max_coefficient = 0.8`
- `AgenticConfig.continuous_urgency_injection = False`

Continuous urgency-vector injection should only be enabled for explicit activation-intervention experiments, not for standard benchmark runs.

## Intervention audit rule

Any run that applies stateful intervention should record enough information to
separate perception from control:

- sensor scores before the intervention
- veto/control scores before the intervention
- the reason the gate opened
- whether ambiguity was present
- how long the intervention lasted
- whether the final answer changed after intervention

Do not use benchmark gold to open a live intervention gate. Gold can be used
only after the run to classify failures such as unwarranted skepticism.
