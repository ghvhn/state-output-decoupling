# Interaction Reward Design

Design record from the bare-prompt / activation-coupling thread. Captures how the
system should learn that *interaction is productive*, because implementation
depends on several probes that are not built yet.

## The frame

Pull the last things that live in prompt/control-flow down into state:

- **Bare prompt (done):** the model sees no system message, persona, tool
  instructions, or date preamble -- only the conversation. Everything special is
  in the activations (ToT, synthesis, cache, organic correction, ClaimMap steer).
- **Activation-triggered tools (done):** no taught tag syntax; tools fire from
  the model's own surfaced state (`detect_framing_tension` -> auto ClaimMap).
- **Think-until-ready speech (to build):** speech = convergence of the internal
  deliberation, not a memoryless threshold and not a pressure accumulator.
  Silence = still deliberating, never "ignored." The synthesis loop is 90% of
  this already; the change is to let it persist across turns and define "ready"
  as genuine convergence rather than "loop budget spent."
- **Always-ingest input (to build):** inputs are never dropped. A reader thread
  queues every submitted line; the deliberation drains it at the chunk/synthesis
  seam the engine already stops at. How hard a fresh drain reorients vs. folds in
  is ONE tuned knob (`input_perturbation`), later derived from input salience --
  not a three-way behavioral fork.

## The reward: a live, decomposed egg gate

"The model learns interaction is productive, so it prioritizes it" must not become
an engagement maximizer. Productivity has to be a signal the model **cannot fake
by talking more**. It decomposes across the transition-layer arc:

- **Intent (early):** did it grasp what was meant? Grounded, oracle-free, as the
  **self-consistency** of the early interpretive read with where the deliberation
  actually lands -- not agreement with the human's words (that is mirroring).
- **Sense (late):** did it resolve into something that coheres, aligned with what
  it internally holds true (synthesis truth-projection + `validated_flow`) --
  NOT bare entropy, which rewards confident nonsense.
- **Confidence (across):** **warranted** confidence only -- calibrated to the
  sense+intent, using the `warranted_confidence` vs `unwarranted_confidence`
  split. Bluster scores nothing; hedging scores nothing.

Together this is the egg gate (non-equivocation on a clean lane = selective
accuracy + coverage), unpacked across layers and measured per-turn instead of
per-run. The reward we reach for is the one we already trust.

## Decisions (this session)

1. **Intent = self-consistency across the arc.** Sidesteps the "your meaning vs
   its own intent" fork; both reduce to the same oracle-free measure.
2. **Credit on sense+intent first; confidence stays a dark slot** until the
   warranted/unwarranted axis is isolated (the handoff notes those vectors "sit
   near each other; needs better isolation before steering" -- crediting on that
   probe now would teach bluster).

## Substrate built: TriggerTuner credit channel

`invariants/trigger_tuner.py` now records `(signal, outcome)` per trigger and
reports, in `:tune`, `fired_outcome` vs `unfired_outcome` and their **lift**.
Positive lift = firing (interacting) beat not-firing on the outcome = the honest
"is interaction productive" readout. Observe-only: nothing drifts until you
calibrate on it deliberately. This is the piece everything above plugs into --
each probe (sense, then intent, then confidence) becomes an `outcome` fed to
`tuner.credit(...)`, and the interaction knobs drift toward what actually resolves.

## Remaining instrument work (in order)

1. Sense probe from synthesis convergence + `validated_flow` (mostly exists).
2. Intent-stability probe: early interpretive read vs final commitment
   (the "decompose intent/answer/expression by layer" thread).
3. Warranted-confidence isolation before confidence can drive credit.
4. Think-until-ready gate (persist synthesis across turns; readiness via tuner,
   observe-only first).
5. Async input ingestion (reader thread + drain-at-seam) with `input_perturbation`.

Null discipline throughout: prove any drift tracks resolution, not activity.
