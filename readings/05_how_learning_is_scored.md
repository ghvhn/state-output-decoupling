# How learning is scored: no oracle in the room

Nothing in a live conversation ever gets a "correct/incorrect" label from an
answer key — there is no answer key here. What a turn gets instead is a
productivity read computed from the reader's own deliberation trace, in
`scripts/interactive_phenomenality.py`:

```python
flow = float(phen.get("validated_flow", 0.0))
interrupt = float(phen.get("needless_interrupt", 0.0))
disagreement = float(phen.get("disagreement", 0.0))
score = flow - interrupt - disagreement
```

Settled forward motion, minus self-interruption, minus internal
disagreement — deliberately not raw confidence, which rewards confident
nonsense. A turn whose score clears a tunable threshold labels the steering
events inside it "productive"; below, "unproductive". These labels accrue on
a separate evidence lane from benchmark results, and the lanes never blend
silently: benchmark gold always wins where both exist, and every readout
reports its mix.

What the labels are FOR: every steering channel accounts for itself each
generation — how often it fired, how hard it pushed. Comparing outcomes of
turns where a channel fired against turns where it stayed silent yields a
lift number per channel. Positive lift is evidence the channel helps;
no contrast yields an honest "no verdict" rather than a story. The same
logic runs per layer when the layer sweep is on. And because fired-vs-unfired
is observational — channels fire on states that need them — the system can
also run true one-variable ablations, where a channel computes exactly as
normal but its effect is removed, and whole runs are compared.

An honest label on the labels themselves: "productive" here certifies that
the deliberation cohered, not that its conclusions were true. That mirrors
the human case — conversation teaches what held together; checking what is
true takes contact with something outside the conversation, which is what
the benchmark lane, the sandbox's real exit codes, and the operator's
corrections are for.

One more scoring rule the whole project rests on, from the egg-gate design:
the standard is honest non-equivocation, not accuracy. Committing when
grounded and abstaining when not beats looking right. A confident wrong
answer costs more than a declined one, and the gates are built to make that
true in the numbers, not just in the prose.
