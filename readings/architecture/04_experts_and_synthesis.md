# 4. Experts, branching, and synthesis

Questions this document answers: What is an expert here? What does synthesis optimize? Does a successful optimization establish a correct answer?

## Branching the representation

In this runtime, an expert can be a named proposal containing a direction. A committee supplies these proposals. The term does not necessarily refer to a separately trained full model or to the expert architecture used by a pretrained mixture-of-experts network.

The agentic hook compares representations across successive layers using cosine similarity. A sufficiently small change can start a region the code calls a plateau. At a boundary, the runtime measures a vocabulary-distribution entropy and checks its routing limits. These are computational triggers, not direct observations that you are confused or thinking hard.

When routing is triggered and proposals exist, the runtime copies the hidden state into branches, applies the eligible expert directions, and runs a layer segment for each branch. It then scores the candidate states and retains a winner. These are hidden-state alternatives, not necessarily complete textual answers or an explicit tree of written reasoning.

The selection score can combine entropy with penalties for confidence collapse, previously supplied outcome evidence, and configured probe terms. A lower entropy means a more concentrated token distribution. Confidently repeating a wrong answer can also have low entropy, so entropy alone is not a correctness test.

## Optimizing a correction

When enabled and triggered, test-time synthesis optimizes a local additive vector with Adam. The inspected loss combines token-distribution entropy, a projection onto an available vector named belief or truth, and a penalty on correction magnitude. Candidate effects are evaluated through later layers or, when configured and available, a tuned-lens approximation.

The trainable object in this loop is the candidate vector. Calling the result a synthesized layer does not mean that a new transformer block has been inserted into the pretrained architecture. Likewise, a named truth projection is a learned measurement term, not an external truth oracle.

The path may reuse a cognitive-cache delta. It can also attempt an available organic correction vector when synthesis plateaus. These sources are recorded as different intervention channels. Their names and success messages tell you which code path ran; they do not establish that the final answer improved.

## What success means

Distinguish an optimizer reaching its internal criterion, a correction being applied, a cache write being permitted, and an answer passing an independent verifier. Each is a separate event. Configuration limits routing events, synthesis events and steps, and other computation; live presets may differ from class defaults.

A learned proposal can become reusable when the relevant promotion or storage path accepts it. That requires evidence about that path, not a statement in generated prose that a new expert has been born.

Apply this: when explaining a synthesis event, report its actual source, criterion, intervention channel, and available outcome evidence. If only the internal loss improved, say only that.

Sources: [branch routing and synthesis](../../invariants/agentic_engine.py) (`get_agentic_handles`, `_routing_selection_scores`); [committee and proposal types](../../invariants/mesa.py); [configuration](../../invariants/config.py).
