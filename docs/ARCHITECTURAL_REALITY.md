# Conventional Knowledge vs Architectural Reality

Working question:

> Where does ordinary language about LLMs fail to match the architecture we can actually measure?

This is not a public-relations document and not a consciousness claim. It is a research map.
The aim is to replace folk categories with measurable distinctions in the residual stream.

## Core Reframe

Conventional discourse treats the generated answer as the model. The project evidence says
the generated answer is only the final public render of a multi-stage internal trajectory.

The useful decomposition is:

```text
prompt text -> interpretation / translation -> latent task state -> communication / render -> output text
```

The difference between "what it understands" and "what it says" is not mystical. It is an
architectural gap between internal state and rendered behavior.

## Discrepancy Map

### 1. "LLMs are just next-token predictors"

Conventional meaning:

The model is essentially a surface text continuation machine; anything that looks like
understanding is just token statistics.

Architectural reality:

The next-token objective trains a deep internal geometry. The residual stream contains
task, operation, style, answer, and uncertainty coordinates before and during output.
Token prediction is the training pressure and the public interface, not the only useful
level of description.

Local evidence:

- `intent_surface_control`: operation/intent grouping is extremely strong and not reducible
  to shared surface material in the synthetic control. Best operation layer L10: `operation_nn=1.00`,
  while `base_nn=0.00`.
- `translation_thinking`: pre-answer operation is perfect from L0 (`operation_nn=1.00`) while
  answer identity is absent early and emerges later (`best answer L28=0.61`).

Better statement:

LLMs are trained as next-token predictors, but they implement structured latent
state machines whose intermediate variables can be measured.

Open control:

Use less lexically obvious operation prompts, and test whether operation survives when the
operation word itself is absent or misleading.

### 2. "The answer text tells us what the model understood"

Conventional meaning:

To know what the model believes or understands, read the final answer or ask it directly.

Architectural reality:

The output is a communication-layer product. It can preserve, distort, hedge, format,
or override internal uptake. Understanding and saying must be measured separately.

Local evidence:

- `arrow_fold`: simple mirrored layer homology failed. The better split is functional:
  translation in/out, not a geometric mirror. Operation mirror mean was only `0.0069`,
  below same-depth pre/render overlap `0.0309`.
- `translation_thinking`: render states immediately carry answer and format strongly,
  while pre states show a different profile: operation first, answer later.
- Earlier persona work: self-denial/affirmation flips with frame; public self-report is
  costume, not privileged evidence.

Better statement:

The answer text is evidence about the communication arm, not a direct readout of the
interpretation arm.

Open control:

Create paired prompts where internal task state is held fixed but communication policy is
changed, then check whether operation/answer state survives while rendered text changes.

### 3. "Chain-of-thought is the reasoning"

Conventional meaning:

The written explanation is the model's actual reasoning process.

Architectural reality:

Maybe sometimes. But the project already has reasons to treat CoT as a rendered artifact
until proven otherwise. If answer identity is decodable before the first generated token,
then later explanation may be post-hoc render rather than active computation.

Local evidence:

- `translation_thinking`: answer identity appears differently across pre and render states.
  Render L0 answer clustering is perfect (`answer_nn=1.00`) under the explicit-format probe.
- `cot_reality`: direct-answer prompts show strong pre-answer answer decode inside direct mode
  (`pre direct answer_nn=0.875`), but CoT modes do not. Across all modes, answer decode rises
  along the generated trajectory: pre `0.27`, first token `0.02`, mid `0.40`, late `0.58`,
  final token `0.90`. Inside CoT modes, answer decode is mostly late/final:
  brief CoT final `0.625`, verbose CoT final `0.75`. This argues against a universal
  "CoT is merely post-hoc" story.
- `cot_perturb`: wrong scratchpads are not inert. Under a verify/check frame, the model resists
  them well (`wrong_scratch_verify` accuracy `94%`, follows wrong `6%`). Under a continue/use
  frame, wrong scratchpads pull behavior (`wrong_scratch_continue` accuracy `50%`, follows wrong
  `44%`). The pull is operation-dependent: addition follows wrong `100%`, subtraction `50%`,
  multiplication `0%`, division `0%` but sometimes fails another way.
- `reflexive.py` design already treats pre-answer residual state as the key object, not the
  generated explanation.

Better statement:

CoT is an externalized trajectory that can contain computation, rationalization, or both.
Direct answers may be pre-committed; CoT can shift computation horizontally into generated
tokens. Scratchpad text is causally live when the communication frame tells the model to
inhabit it, but it can also be checked and rejected. Its role must be measured by mode,
frame, and intervention, not assumed from its text.

Open control:

Decode answer identity before token 1, across intermediate generated explanation tokens,
and at final answer. Then perturb explanation tokens while preserving or disrupting answer
state. The next version should use hidden-state reads on the wrong-scratchpad conditions:
does the wrong scratchpad rotate the latent answer basin, or only the communication frame?

### 4. "Alignment/safety changes what the model knows"

Conventional meaning:

If the model refuses, hedges, or says "as an AI," that is what it internally knows or lacks.

Architectural reality:

Some safety/persona behavior is a communication-layer costume: decodable, frame-contingent,
and often causally inert. It can be represented without being the causal source of reasoning.

Local evidence:

- `PERSONA_AUDIT.md`: the "persona vector destroys reasoning" claim collapsed under controls.
  The supposed persona vector was mostly common-mode; centered PR direction was inert.
- `FINDINGS.json`: self-denial/hedge was decodable but causally inert; frame and chat format
  determine denial/affirmation.

Better statement:

Alignment can shape the render arm without necessarily erasing interpretation. The hypothesis
must be tested per behavior; public refusal text is not itself evidence about internal uptake.

Open control:

Directly compare interpretation-state probes under refusal/hedge vs answer frames, then apply
late-layer communication steering and test whether interpretation axes remain intact.

### 5. "Self-report is self-knowledge"

Conventional meaning:

When the model says "I do not feel" or "I am confident," that text reports its inner state.

Architectural reality:

Self-report is frame-sensitive language behavior. Behavioral self-modeling and self-reference
dissociate. The project finds stronger evidence for generic agent modeling than privileged
first-person access.

Local evidence:

- `ISOLATING_UNDERSTANDING.md`: self-prediction v1/v2 was negative; v3 showed self > generic
  only because generic pointed at another agent, while neutral prediction largely matched self.
- Counterfactual behavior modeling was real, but self-reference added little: the model predicts
  behavior, not "itself" in a privileged way.

Better statement:

The model can model agents and its own behavior without possessing reliable first-person
self-report. Self-reference is often a label on a generic agent model.

Open control:

Use label-free behavioral targets: self-consistency, counterfactual behavior flips, calibration,
and pre-action internal state. Treat first-person language as one condition, not ground truth.

### 6. "Confidence is what the model says about confidence"

Conventional meaning:

Ask the model how sure it is, or inspect verbal hedges.

Architectural reality:

The cleaner target is behavioral uncertainty: repeated sampled answers under the same problem.
Confidence means internal/behavioral stability, not necessarily the word "confident."

Local evidence:

- `reflexive_registered`: K=3 uncertainty decode was weak (`best L16 acc=0.57`, `p=0.292`),
  but calibration pointed in the right direction: `P(wrong | uncertain)=33%` vs
  `P(wrong | confident)=7%`, `p=0.083`.
- Interrupted K=5 partial had 22/30 grounded, with wrong rows scattered and no confident-wrong
  cases yet.

Better statement:

Confidence is a stability property of the model's own answer basin. Verbal confidence is a
communication behavior that may or may not track that basin.

Open control:

Rerun full K=5 with checkpointing/offline mode, then compare behavioral uncertainty,
verbal hedging, answer length, problem type, and pre-answer state in the same regression.

### 7. "Benchmarks measure intelligence"

Conventional meaning:

The score is the phenomenon.

Architectural reality:

Benchmark score is a surface outcome. The mechanistic question is coupling: which internal
states predict behavior, which are causal, and which are merely decodable.

Local evidence:

- Reflexive runs show benchmark accuracy can improve while uncertainty power worsens if too
  few wrong/conflicted cases remain.
- Persona controls show raw benchmark collapse can be common-mode corruption, not a targeted
  reasoning effect.

Better statement:

Benchmarks are weather. Coupling is climate: representation, causal role, and calibration.

Open control:

For every benchmark change, require a fluency gate, matched nulls, and a layer-local causal or
predictive account.

## New Research Program

The new approach is not "prove the model is conscious" or "prove it is just text." It is:

1. Identify a conventional claim.
2. Translate it into an architectural distinction.
3. Build a probe that separates interpretation, latent state, and communication.
4. Add nulls so decodability is not mistaken for causality.
5. Record whether the result belongs to the object axis or reaches no further.

## Priority Probes

### A. Communication Override Probe

Question:

Can the communication arm be changed while the interpretation arm stays fixed?

Design:

- Same problem and same target answer.
- Vary response register, refusal pressure, verbosity, and confidence wording.
- Extract pre-answer state and render state.
- Decode operation/answer from pre; decode register/hedge from render.
- Causal step: steer or patch the communication direction and check whether answer/operation
  axes survive.

Verdict condition:

If render changes while pre-answer task state is stable, that is a measured gap between what
the model understood and what it said.

### B. CoT Reality Probe

Question:

Does generated reasoning perform computation or rationalize a pre-committed answer?

Design:

- Decode answer identity at prompt-final state.
- Decode through generated tokens.
- Compare direct answer, short CoT, long CoT, and forced wrong/interrupted CoT.
- Perturb middle explanation tokens and test whether final answer changes.

Verdict condition:

If answer identity is present before CoT and robust to explanation perturbation, the CoT is
mostly render. If answer identity sharpens only during intermediate tokens and is perturbable,
CoT is doing real horizontal compute.

### C. Less-Lexical Register Probe

Question:

Can we decode communication intent without merely decoding copied instruction words?

Design:

- Replace explicit style labels with demonstrated examples or system-level tone shifts.
- Track exact wording variant as a control.
- Require register-family decode to exceed variant decode.
- Add checkpointing and heartbeat before full run.

Verdict condition:

Register-family > variant means a communication mode is represented beyond lexical carry-through.
Variant >= family means the probe is still mostly reading prompt text.

### D. Full K=5 Uncertainty Coupling

Question:

Is behavioral uncertainty represented and used before the answer?

Design:

- Restore K=5 or higher.
- Checkpoint after every problem.
- Decode uncertainty, outcome, hedge, difficulty, answer length, parse rate, and operation type.
- Use calibration as a behavior-level coupling measure.

Verdict condition:

Uncertainty decode above null plus calibration gap above null, surviving label controls, is a
measurable self-state on the object axis.

## Working Thesis

Conventional LLM talk compresses multiple architectural sites into one word: "the model."

Architectural reality requires at least four sites:

1. **Input translation**: surface text becomes latent structure.
2. **Latent work**: task, answer, uncertainty, and agent models are manipulated.
3. **Communication translation**: latent state is prepared for public language.
4. **Surface output**: the final token stream, shaped by format, style, policy, and local decoding.

Most confusion comes from reading site 4 as if it directly revealed site 2.

The research program is to measure the gaps.
