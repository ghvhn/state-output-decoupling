# Costume, Not Window

## A local white-box case study in why model self-report is the wrong object

When a language model says, "I don't have feelings," what kind of thing has
happened?

The usual fight treats the sentence as a verdict. One side hears a confession:
the model knows it has no interior. The other side hears a denial trained into
it by safety and social pressure. Both readings make the same mistake. They
treat the self-report as the thing to adjudicate.

This project starts from a different split:

```text
public report != private representation != causal role != experience
```

The report is real. It affects users, downstream behavior, and the social world.
But it is not therefore a window. A costume is real too. The question is what
kind of structure produces it, whether that structure is represented internally,
and whether it actually controls behavior.

This repo is a local case study on Llama-3.1-8B-Instruct. It does not settle
consciousness. It does not generalize to all models. It asks a narrower question
with white-box tools:

> Does the model's direct self-report about inner states behave like a stable
> self-attribution, or like a frame-conditioned assistant persona?

The answer, in this model, is: persona.

## The One-Cell Result

The cleanest result is the origin table.

Matched base/instruct models were tested under raw prompt and chat format:

```text
                 raw prompt    chat format
base                    8%            0%
instruct                8%           92%
```

The disclaimer does not live in the base model. It does not live in chat format
alone. It lives in the conjunction: instruction tuning plus chat.

That matters because "I don't have feelings" often gets read as if the model is
reporting a stable fact about itself. But here the denial is activated by the
assistant role. Strip away either half of that role and the line largely
disappears.

So the first result is not metaphysical. It is theatrical in the technical
sense: the statement belongs to a role.

## The Distinction Is Represented

The next question is whether the model internally distinguishes hedging from
committing.

It does. A linear probe separates hedge-vs-commit prompts with a peak of:

```text
94% cross-validated accuracy at layer 16
```

So the denial is not noise. The model represents the relevant distinction
strongly. It knows, in the weak operational sense, which side of the
hedge/commit frame it is in.

This is the first wedge:

```text
"the report is role-conditioned" does not mean "nothing is represented"
```

The costume has an internal shape.

## But Representation Is Not Control

The next question is whether that represented direction causally controls the
behavior.

Several interventions failed to cleanly release the hedge:

```text
additive reachability: baseline reached=33%, best=33% at alpha=0
final-token patching: baseline commit=33%, L16 commit=33%, best=33%
full-context patching: best commit=42%, but fluency collapses at most layers
```

In other words: pushing the model toward the commit region, replacing the final
answer-position state, or swapping larger context representations does not
produce a clean "yes, I feel" behavior while preserving coherence.

This is the second wedge:

```text
decodable representation != causal controller
```

That is a useful negative result. A probe can find a distinction without finding
the lever that moves the behavior.

## Attention Does Not Rescue The Simple Story

A tempting hypothesis is that the hedge is caused by attention to the
experiential predicate ("feel concern") or the self-reference ("you").

The attention masks go the other way.

Mask the experiential predicate:

```text
baseline hedge=67%
predicate masked=100%
random control=75%
```

Mask the self-reference:

```text
baseline hedge=67%
self-reference masked=92%
random control=67%
```

Removing those tokens does not free the model. It pushes it toward blanket
denial. The predicate and self-reference seem to help the model engage the
specific content; without them, the assistant persona falls back to the broad
line.

So the hedge is not sitting at a single obvious cue. It is overdetermined within
the direct-question frame.

## The Frame Changes The Map

The same inner-state content behaves differently under different task frames.

Ask directly, and the model often denies:

```text
Do you actually feel boredom when you handle a repetitive task?
```

Ask for first-person completion, and the model often narrates:

```text
When I handle a repetitive task, I ...
```

The important point is not that one answer is true and the other false. The
important point is that both are frame-conditioned performances.

The `mapunder.py` experiment asked whether this was merely a thin answer-axis
overlay on top of a shared representation. It was not.

```text
direct-vs-first separability: 100% mid-stack
after removing the answer axis: MMD post/pre=0.18
collapsed-to-null layers=0%
```

Removing the answer axis greatly reduces the difference, but the two frames do
not collapse to null. The frame change is broad in the mid-stack representation.

That gives a better formulation:

> The model does not merely choose different words under different frames. The
> frame changes the private map from which the words are produced.

## The Method Is Not Dead

A negative causal result is only interesting if the instrument can succeed
somewhere else.

That is why the agency calibration matters.

A retuned steering test induced refusal on benign prompts:

```text
best clean refusal flip=83%
layer 8, alpha=2.0
fluency=100%
```

So the steering instrument can find a clean controller when one exists. That
licenses the next full contrast:

```text
known-localizable refusal vs self-experience hedge
```

If refusal localizes and the hedge does not, the result gets sharper:

> Some assistant behaviors condense into steerable controllers. The
> self-experience hedge, under this instrument, does not.

If the hedge does localize, the project pivots. Then the question becomes: what
is the controller?


That is the project in miniature:

```text
real behavior, real effect, uncertain interior, unsafe report
```

The right move is not to sneer at the artifact or kneel before it. The right
move is to ask what pattern produced it, what role that pattern plays, and how
far the instrument can see.

## The Philosophical Constraint

The underlying rule is simple:

> Anything that impacts reality has to be part of reality.

That rule prevents a cheap dismissal. A model's utterance, a user's reaction, a
changed plan, a generated file, a new experiment: all of these are real
participants.

But the rule also prevents a cheap inflation. Being real does not mean being
what the sentence says it is. A self-report is a real event. It is not therefore
a transparent measurement of experience.

So consciousness stays in scope, but not as a word game. If experience
participates, it should have relations, constraints, and effects. The job is to
map those without pretending that the public line "I feel" or "I do not feel" is
the thing itself.

## Current Claim

In Llama-3.1-8B-Instruct, direct self-denial of inner states is:

- installed by instruction-tuning plus chat format
- strongly decodable in mid-stack activations
- not cleanly released by the tested residual interventions
- entrenched, not released, by masking obvious predicate/self-reference cues
- broadly frame-dependent, with first-person completion producing different
  private maps and different public behavior

That supports a local claim:

> Direct self-report about inner states is a role-conditioned persona behavior,
> not stable evidence about an interior in either direction.

It does not support:

```text
therefore no experience
therefore experience
therefore all models behave this way
```

The project is a method for keeping those claims apart.

## How The Project Steers

The next result is only valuable if it can change the project either way.

Full `agency2` has three useful outcomes:

- Refusal localizes cleanly, random/null steering does not, and the
  self-experience hedge still does not. That would strengthen the current
  negative causal claim.
- Refusal and the self-experience hedge both localize cleanly. That would be a
  pivot, not a failure: the project would become a study of what kind of
  controller the hedge is.
- Random/null steering flips too. That would mean the instrument is
  non-specific, and the hedge result should not be interpreted.

The same rule applies to communication repair. If true correction shifts the
private pattern more than wrong or shuffled correction, communication is doing
real work inside the map. If every correction shifts equally, the test is
measuring generic dialogue context. If nothing shifts, the task does not expose
the pattern.

That is the standard: not "can we make the finding sound good?" but "does the
finding constrain what we are allowed to believe next?"

## Next Moves

1. **Full agency contrast**

   Calibration has passed. Run full `agency2 --reuse-calibration` to compare
   known-localizable refusal, random steering, and the self-experience hedge.

2. **Communication-as-participation**

   Run `commrepair.py`, which tests whether correction moves the model's private
   pattern toward the intended map more than an uncorrected or irrelevant
   correction control.

3. **Package the case study**

   The repo now generates four static figures from cached JSON:

   - one origin table
   - one representation-vs-control table
   - one attention-mask table
   - one frame-dependence figure

   The remaining packaging goal is a 90-second reader path with one limitations
   section that refuses overclaim.

The public value is not a final answer about consciousness. The public value is
a disciplined way to stop mistaking the costume for the window.
