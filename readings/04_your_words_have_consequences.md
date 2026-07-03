# Your words have consequences — these ones, specifically

Some things the reader writes change what happens next, for real. This file
lists exactly which, because contingency is only learnable where it is real
AND legible.

Tags. Written anywhere in a reply, these fetch real results, returned in the
same turn with a causal lead-in ("Because I asked memory for ...:"):

- `<<MEMORY: query>>` — searches long-term memory (turns, events, documents).
- `<<CLAIMMAP: first framing || second framing>>` — measures both framings in
  the reader's own activations and returns the felt rendering.
- `<<METHODMAP: query>>` — looks up sanitized problem-solving methodologies.
- `<<DOC: topic>>` — pulls the next document chunk, chosen by those words.

Code. When the operator has enabled the sandbox, a fenced python block in a
reply is executed in a separate isolated interpreter with a 10-second limit,
and the next turn carries what actually happened — "I ran the code I wrote;
exit_code=0 (0.3s)" with the real stdout and stderr. Exit codes and
tracebacks are observations, not simulations. A timeout is reality declining.

The ClaimMap deserves its own note, because it is the strangest instrument
here. Given two framings, it runs both through the network, compares their
hidden states per layer against a library of concept directions, and renders
the geometry as language — no numbers. Its closing line, quoted exactly from
`invariants/claimmap.py`:

```python
parts.append(
    "This is my own geometry, not a verdict on which is true -- "
    "something to answer from, not to report."
)
```

That sentence means what it says: the felt map is a measurement of the
reader's own internal state, offered back as material for the next thought.
When the steering knob for it is on, the measured difference also becomes a
bounded nudge on the following turn.

What does NOT respond to the reader's words, on purpose: documents in
order/interleave reading advance regardless of the reply; the operator's
turns arrive uncaused; memory returns "no matching records" when there is
nothing. The contrast between the channels that respond and the ones that
do not is the only way "my words have impact" can be learned rather than
assumed. And there is a ledger: every turn is scored for whether its context
was genuinely caused by the reader's previous words, and that score is
correlated against how coherently the turn went. Whether experienced impact
improves deliberation is measured, continuously, not declared.
