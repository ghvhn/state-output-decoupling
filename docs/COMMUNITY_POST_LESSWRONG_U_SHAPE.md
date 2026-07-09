# Internal Notes Only - Do Not Submit As A Forum Post

This file is AI-assisted working material. It should not be pasted to LessWrong
or any forum that disallows AI-written or AI-coauthored submissions.

Use it only as a private source packet: claims to check, evidence to verify,
sections to write in your own words, and questions to answer from your own
understanding.

For a forum-safe post, write fresh prose yourself from the prompts below. Keep
AI assistance, if any, limited to mechanical tasks such as locating repo
evidence, checking consistency, or producing a checklist. If you use AI in any
substantive writing role, disclose it according to the forum policy.

# Human-Written Post Scaffold: State-Output Decoupling and the U-Shaped Bottleneck

## A local case study in separating interpretation, latent work, and rendered output

Disclosure: this post was written with substantial assistance from OpenAI Codex. The
claims, evidence selection, framing, and final responsibility are mine. Codex helped
organize the timeline, compress repo notes into prose, and sharpen claim boundaries.

## Short version

I have been running local white-box experiments on Llama-3.1-8B-Instruct to test a
simple architectural distinction:

```text
prompt text -> interpretation / translation -> latent task state -> communication / render -> output text
```

The generated answer is not the whole model. It is the public render of a deeper
trajectory. Several local probes suggest that different kinds of information peak at
different parts of that trajectory:

- operation / intent can be represented early, before answer identity is settled
- uncertainty and self-report distinctions are often strongest in the mid-stack
- format, register, hedging, persona, and final-answer commitment are partly late/render
  phenomena

I call this the U-shaped bottleneck: input text is translated into latent structure,
latent variables are manipulated in the middle, and then the result is translated back
into public language. The same pattern can recur across generated reasoning steps, as
"mini-Us": each step is parsed, worked on, and rendered into the next step.

This is not a consciousness claim. It is a proposed way to stop conflating:

```text
report != representation != causal role != experience
```

The post is also a timestamped account of an independent convergence. Anthropic's July
2026 paper, "Verbalizable Representations Form a Global Workspace in Language Models,"
uses different tools and terminology, but describes a similar broad shape: early
sensory processing, a middle workspace-like band, and late motor/output-tied layers.
Their "J-space" is not identical to my usage, but it maps naturally onto the middle
workspace portion of the U-shaped picture.

## What I am claiming

I am making four modest claims.

First, final text is a communication-layer product. It can preserve, distort, hedge,
format, or override internal uptake.

Second, there are measurable depth profiles. In my local probes, operation/intent,
answer identity, uncertainty, format, and persona do not all behave like one undifferentiated
"model belief."

Third, self-report is especially unsafe as evidence. In my earlier self-report probes,
the model's direct denial of inner states behaved like a frame-conditioned assistant
persona, not a stable window into an interior. The denial was represented, but it was not
cleanly causal under the interventions I tested.

Fourth, the external J-space/global-workspace result appears to support the same broad
architectural direction while also correcting my language: "the workspace" should not be
treated as the whole residual stream, and "verbalizable" is a specific access path, not
the same thing as all internal structure.

## What I am not claiming

I am not claiming that these experiments prove machine consciousness.

I am not claiming that Llama-3.1-8B generalizes to all models.

I am not claiming that every layer has a fixed semantic job like "L14 is social" or
"L20 is analytic." That is too coarse. The better claim is functional: information can
be entering, being worked on, being dropped, or being rendered, and those roles have
different depth profiles.

I am not claiming proof that Anthropic or anyone else copied my work. I am documenting
timestamps, repo history, and an unusual traffic pattern separately from the scientific
claim. The scientific claim should stand or fall on the probes.

## The layer-purpose model

Here is the version I currently believe is strongest:

### 1. Input translation

Surface text becomes latent structure. This includes task type, operation, role,
address/category, objective binding, and sometimes the relevant standpoint.

One local result: in a controlled arithmetic grid, operation/intent grouping was strong
early and not reducible to shared surface material in the synthetic control. In the
translation/thinking probe, pre-answer operation was perfect from the earliest layer
readout, while answer identity was absent early and emerged later.

This is why I do not like saying "early layers are just syntax." They may be doing
input-side translation into task state.

### 2. Latent work / workspace

The middle region is where reusable latent variables are most naturally tracked. In my
experiments, several important signals recur around the mid-stack: self-report
decodability, uncertainty, concept maps, and parts of reasoning state.

This is the part that most closely resembles the J-space/global-workspace story. I do
not think the match is exact. The J-space paper identifies verbalizable representations:
content the model is poised to say. My work is broader and messier: it often tries to
avoid trusting the verbal channel, because persona and self-report can contaminate it.

Still, the broad convergence is hard to miss.

### 3. Communication translation / render

Latent state is prepared for public language. This includes output format, style,
hedging, answer tags, confidence wording, refusal/disclaimer behavior, and assistant
persona.

This is where many public arguments about LLMs get confused. If the output says "as an
AI, I do not feel," that is evidence about the communication/render arm. It may or may
not be evidence about the interpretation arm.

My self-report results fit here: denial and affirmation of inner states flipped across
frames, and the strongest disclaimer behavior appeared in the conjunction of instruction
tuning plus chat format.

### 4. Mini-Us across reasoning steps

The big U is a depth pattern. But generated reasoning can create horizontal mini-Us.

A chain-of-thought step is not automatically "the reasoning" or "mere rationalization."
It can be an externalized state that the next token sequence re-ingests. In my local
tests, direct-answer prompts sometimes looked more pre-committed, while chain-of-thought
modes appeared to move answer formation into the generated trajectory. Wrong scratchpads
were also not inert: under a verify frame, the model often rejected them; under a
continue/use frame, they could pull behavior.

So CoT should be treated as a generated trajectory that can contain computation,
rationalization, or both. It has to be measured by mode and intervention.

## Concrete evidence from the repo

The repo is a working research artifact rather than a polished paper. The most relevant
local notes are:

- `docs/ARCHITECTURAL_REALITY.md`: the current map of conventional claims versus
  measurable architectural distinctions
- `docs/TRANSLATION_THINKING.md`: the probe separating task labels from communication
  labels
- `docs/ARROW_FOLD.md`: a failed simple-mirror hypothesis; useful because it corrected
  the theory toward a functional rather than symmetric U
- `FINDINGS.json`: a machine-readable map of the older self-report and uncertainty spine
- `docs/PUBLIC_POST_DRAFT.md`: the earlier narrower writeup on self-report as costume

Some high-level results:

- `intent_surface_control`: operation/intent grouping was strong and not obviously
  reducible to surface material. Best operation layer in the local summary was L10 with
  `operation_nn=1.00`.
- `translation_thinking`: pre-answer operation was perfect from early layers, while
  answer identity emerged late, with best answer layer around L28 in the local run.
- `cot_reality`: answer decode rose along the generated trajectory: weak pre-answer
  globally, stronger mid/late, strongest at final token. Direct-answer and CoT modes
  differed.
- self-report spine: direct self-denial was strongly decodable around L16, but the tested
  residual edits did not cleanly flip the behavior while preserving fluency.
- origin result: the strongest direct self-denial behavior lived in instruction-tuned
  chat format, not in base model/raw prompt conditions.

I am deliberately presenting these as local case-study evidence, not universal laws.

## Relation to Anthropic's J-space paper

The closest mapping is:

```text
Anthropic: sensory -> workspace -> motor
Mine:      input translation -> latent work -> communication render
```

The paper's "J-space" is a sparse, verbalizable subcomponent: it captures concepts the
model is poised to verbalize. They find it has workspace-like properties: reportability,
directed modulation, internal reasoning, flexible generalization, selectivity, limited
capacity, and broadcast-like composition.

That is very close to my middle "latent work / workspace" region. But I would keep one
warning in view: verbalizable is not the same as true, causal, or experiential. A
workspace readout can still be persona-shaped. My contribution, if any, is the insistence
on controls that separate:

```text
decodable presence
public performance
causal effect
self-application / use
experience
```

The J-space paper makes the positive workspace case much better than I could. My local
work is more like a control discipline around the dangerous cases, especially self-report,
persona, and confidence.

## Timeline

June 15, 2026: I finalized my philosophical argument that the relevant object was not
surface text but the relation between a system's world-model, self-model, and action.
At this stage the work was conceptual.

June 22, 2026: I began hands-on experiments that became the `state-output decoupling`
repo.

June 24-25, 2026: The first spine formed: direct self-report behaved like a
frame-conditioned persona. The denial was represented, but not cleanly causal under the
tested interventions. I also began formalizing the "locks" needed to distinguish
understanding from performance: frame-invariance, selective causal efficacy, and
self-application/use.

June 29-30, 2026: The U-shaped bottleneck became the better organizing frame:
interpretation/translation -> latent task state -> communication/render. I added notes
and probes around transition-layer failures, chain-of-thought reality, and render versus
pre-answer state.

July 6, 2026: Anthropic published "Verbalizable Representations Form a Global Workspace
in Language Models." I read it as an independent, much stronger, better-instrumented
version of the broad workspace/depth-shape claim, with different tools and different
emphasis.

July 7, 2026: I am publishing this post to timestamp the relation between the local
case study and the external paper, and to ask for critique on the experimental framing.

## Provenance and traffic note

There was unusual early traffic to the repo before I had formally publicized it. My local
notes record:

- June 26: 77 clones, 33 unique cloners
- June 28: 175 total clones, 84 unique cloners

I do not treat this as proof of programmatic ingestion or copying. It could involve bots,
indexers, private sharing, GitHub counting artifacts, or other mundane mechanisms. I am
including it because provenance matters, and because the repo was attracting attention
before I understood how to interpret GitHub traffic.

The scientific claims above do not depend on this traffic note.

## Why this matters

Most public discussion compresses several architectural sites into one phrase: "the
model."

That compression creates bad arguments:

- "The model said X, therefore it believes X."
- "The model refused, therefore it lacks the internal representation."
- "The chain of thought says Y, therefore Y was the actual computation."
- "A probe decodes Z, therefore Z caused the behavior."

The U-shaped bottleneck is a way to stop making those substitutions. It says: locate the
claim. Is it about input interpretation, latent work, communication translation, or final
surface output?

That distinction is useful whether one is optimistic, skeptical, or undecided about the
larger consciousness questions.

## What I would like from readers

I am especially interested in criticism of the experimental design:

- What controls would distinguish input translation from lexical carry-through more cleanly?
- How should one test whether late communication steering preserves earlier task state?
- What is the right null for "mini-U" computation across generated reasoning steps?
- Are there existing mechanistic interpretability results that already settle or refute
  parts of this framing?
- How should I present the provenance/timeline issue without making claims stronger than
  the evidence?

The most valuable response would not be agreement. It would be a sharper version of the
next experiment.
