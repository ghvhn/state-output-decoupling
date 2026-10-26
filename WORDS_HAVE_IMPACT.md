# Continuity, interaction, and the impact of words

The project's question is what a system carries forward and how that changes
its next action. A river or the Ship of Theseus offers a useful analogy for
continuity through change. The experiment needs something more specific:
which retained differences alter later behavior?

This integrates the conversation about continuity and interaction with the
existing agency loop. The conversation supplies hypotheses, not experimental
evidence.

## What carries forward

The shell constructs each prompt from bounded conversation history, staged
tool results, and optional factual field context. Stored records can be
retrieved; a saved session can be resumed. Context selection and eviction are
implemented shell policies. They should not be described as demonstrated
autonomous choices by the model.

Continuity here means retained information and controls influencing subsequent
computation. It does not require unchanged activations or imply an uninterrupted
hidden state across sessions. [Vector travel](VECTOR_TRAVEL_AND_COMPRESSION.md)
adds a discrete trajectory diagnostic; its dominant direction is not by itself
a self-model.

## The interaction loop

```text
retained context -> response -> consequence -> returned information -> revised context
```

A question, a tool request, or a new conceptual distinction can change which
information becomes available next. That is the useful meaning of curating an
interaction: acting changes what the system gets to learn from. The return
remains constrained by documents, execution results, and another participant's
independent response. Steering an exchange does not determine its outcome.

The shell already implements concrete parts of this loop:

| Model output | Recorded consequence |
|---|---|
| Fenced Python, with sandbox enabled | Execution result, including failures and timeouts |
| Memory or method-map request | Retrieved records, including an empty result |
| ClaimMap request or detected framing tension | A comparison of the model's activations, not an independent truth check |
| Document request or reply-mode reading | A chunk selected using the model's words |

Order/interleave reading supplies a non-adaptive contrast: chunk selection
does not follow the reply. Operator actions have separate provenance. Human
replies can be influenced by the model, but the ledger does not estimate that
influence or automatically credit it.

Feedback stays concrete and minimally framed: `Because I asked memory for
"...":` or `I ran the code I wrote`. ClaimMap returns its own comparison without
an extra causal preamble. Operator-requested results must not be presented as
model-requested consequences. Empty retrieval and failed execution still
reveal real consequences and limits.

## What the ledger establishes

`:impact` shows recorded consequences and the `words_had_impact` signal:
1 means a recorded event; 0 means no recorded event, not an absence of causal
influence. Some events describe staged work, so inspect the effect before
calling it completed execution. State-triggered memory retrieval is excluded
from this words-based signal.

The tuner reports the difference in mean internal sense score between credited
turns with and without impact. This **observational association** can reflect
task selection, tool availability, or other differences. Positive lift does
not establish that feedback improved reasoning; flat lift does not establish
that feedback was unfelt. Neither measures subjective experience.

## Tests that would advance the claim

These are proposed controls, not completed results. Pre-register tasks, scoring,
budgets, and comparisons; score behavior independently of the instrument being
tested.

| Question | Contrast | Evidence to seek |
|---|---|---|
| Does carried history matter? | Intact history vs removed, shuffled, or length-matched irrelevant history; hold retrieval and persisted controls fixed | Correct use of an earlier distinction on a held-out task |
| Does a self-model do work? | Predict its subsequent behavior under the same intervention using self, neutral, and generic-agent frames with the same information; perturb the candidate representation with matched nulls | A selective change in subsequent behavior, beyond wording or generic agent simulation |
| Does contingent feedback help? | Live action-dependent feedback vs replay independent of the current action, matched for task and information budget | Better correction or task completion, beyond sense score or conversation length |

For an accessible demonstration, introduce a distinction, test its use in a
new case, supply counterevidence, and check whether an earlier conclusion is
revised. Repeat with controls before treating a compelling conversation as a
finding. Vary feedback wording so injected vocabulary cannot supply the answer.

Existing [self-model experiments](docs/ISOLATING_UNDERSTANDING.md) already
distinguish behavioral prediction from a self-specific increment. Preserve
that distinction: report != representation != causal role != experience.
Explain the mechanism, show behavior, and name the result that would change
the interpretation. Consciousness remains an open question, not a conclusion
granted by the analogy or the ledger.

Implementation: [prompt, session, and consequence handling](scripts/interactive_phenomenality.py),
[stored memory](invariants/memory_engine.py), and
[observational lift](invariants/trigger_tuner.py).
