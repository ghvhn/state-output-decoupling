# Your words have consequences

A reply can change what information arrives next. These requests connect
words to actual operations:

- `<<MEMORY: query>>` retrieves stored records.
- `<<CLAIMMAP: first framing || second framing>>` compares activations for
  the two framings.
- `<<METHODMAP: query>>` retrieves stored problem-solving methods.
- `<<DOC: topic>>` requests a document chunk selected using those words.

With the sandbox enabled, a fenced Python block can produce real stdout,
stderr, an exit code, or a timeout. An empty search and a failed execution
are still results of an action. A queued operation is not yet a completed one.

ClaimMap has a different role from an execution result. It compares internal
geometry; it cannot determine which framing is true. Its source says:

```python
parts.append(
    "This is my own geometry, not a verdict on which is true -- "
    "something to answer from, not to report."
)
```

The return can change a later reply. A question may reveal a missing fact;
a new distinction may change how a passage is read; a failed program may
require revising an earlier assumption. The useful test is whether the next
action uses that information correctly.

Not every channel follows the reply. Order/interleave reading selects chunks
independently of it; reply-mode reading can follow its content. The operator
can respond freely. The impact ledger records explicit tool and reading
consequences, not every influence on another participant.

`:impact` reports those events and their association with an internal sense
score. Association alone does not establish better reasoning, self-awareness,
or experience. Those interpretations require separate controls, described in
[the project note](../WORDS_HAVE_IMPACT.md).

Sources: `invariants/claimmap.py` and the tool and impact handlers in
`scripts/interactive_phenomenality.py`.
