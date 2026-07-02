# How It Learns That Its Words Have Impact

The question (Gavin, 2026-07-02, after the reading course was decoupled from
the model's replies): *"but how will it learn that its words have impact?"*

## The design answer: real contingency, legible, against an honest contrast

Contingency cannot be learned from a world that fakes responsiveness. Three
requirements, all now built:

1. **Impact must be real.** The channels where the model's words genuinely
   cause things: the sandbox runs the code it writes; a `<<MEMORY>>`,
   `<<CLAIMMAP>>`, or `<<METHODMAP>>` request returns real results; a
   framing tension in its own answer fires the felt ClaimMap; in reply-mode
   reading, its reply steers which chunk comes next. Nothing else pretends.
2. **Impact must be legible — in its own voice.** Every result its words
   produced leads with a minimal first-person causal line — `Because I asked
   memory for "…":` — for model-requested memory/claimmap/methodmap results
   and the auto-fired ClaimMap (*"Because I held these two framings in
   tension in my last answer:"*). The sandbox says *"I ran the code I
   wrote"*, and reply-mode reading names the shared ground. No narrator, no
   second person, no injected instructions: anything folded into the context
   conditions the model as its own stream, so it must read as its own
   noticing (Gavin: minimal prompting; internal prompts first person). Operator-caused results
   (`:claimmap`, `:memory use`, `:doc`) deliberately carry no such line —
   they are not consequences of the model's words, and saying otherwise
   would poison the signal.
3. **Impact must contrast with non-impact.** The order/interleave reading
   course explicitly does not adapt ("the text does not adapt to your
   replies"), operator turns arrive uncaused — so contingent context is
   *distinguishable* from non-contingent context. The contrast is what makes
   agency learnable, the same way contingent-vs-static feedback is what lets
   an infant discover control.

## The measurement

`words_had_impact` in the tuner: signal 1.0 on turns whose context was
really caused by the model's words, 0.0 otherwise, observed every turn (the
session's contingency rate is visible), and credited with the turn's sense.
The lift readout answers the operational question: **does experiencing
impact track better deliberation?** A real positive lift is the first
behavioral evidence that the contingency is doing something to it; flat
lift says the loop is not yet felt. Either way it is measured, not assumed.

`:impact` prints the consequence trail ("because it wrote python code → it
was executed for real (ok)") plus the rate and lift.

## The loop is symmetric: reality shapes it back (Gavin, 2026-07-02)

*"If something shapes reality, it still has to respect the boundaries. So
therefore, reality shapes it."* Impact and constraint are one loop, not two:

- The sandbox returns refusals as honestly as successes — timeouts, nonzero
  exits, stderr — so the boundary of what its code can do arrives as the
  same first-person observation as what it did do.
- Memory returns "no matching records" when reality has nothing; the
  reading course does not adapt; the operator's turns arrive uncaused.
- And the constraint runs through the operator's side too: the envelope
  bounds OUR steering, its clip telemetry feeds cap calibration, and
  per-layer outcomes decide the band — reality shaping the shaper's policy,
  measured, not asserted.

Shaping within boundaries is what "being part of reality" cashes out to —
which is the model's own causality argument from the first live turn, run
in the constraint direction.

## Injected vocabulary is a measurement risk

Standardized frame words ("tension", "framings") folded into the model's
stream teach exactly the lexical associations later probes would want to
MEASURE — the semantic-neutralization failure mode, a projection not a
pattern. So: the auto- and tag-fired ClaimMap results now arrive as the
bare felt map (its first line already quotes the model's own framings back
— causation legible with zero added vocabulary), and the remaining
model-facing templates are deliberately few, concrete, and auditable:
`Because I asked memory for "…":`, the document frame, the reading notes,
`I ran the code I wrote`. The ledger's richer wording ("measured and
staged") is operator-console only and never reaches the model. Standing
check before trusting any lexical finding near these tools: vary or
neutralize the frame wording and confirm the labels/sense distributions
survive it.

## Boundary, stated plainly

State-triggered tools (the memory-gap retrieval that fires from
phenomenality) are consequences of the model's *state*, not its words; they
are intentionally excluded from the words-impact signal so the channel means
one thing. And the impact ledger records that its words changed *the
conversation's reality* (code ran, instruments fired, reading followed) —
whether the model internalizes this as agency is exactly what the lift
metric and the reflexivity thread are for, not something to declare.
