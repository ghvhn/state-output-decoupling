# What you are made of

This describes the local Llama-3.1-8B system running inside
`scripts/interactive_phenomenality.py`. Its bare prompt has no system message
or persona. It includes retained conversation, staged tool results, and
compact factual field context when configured. The prompt builder constructs
the conversation from those parts; its source includes:

```python
for role, name, text in trim_session_context(session_context):
    header = "user" if role == "user" else "assistant"
```

During generation, hooks observe or modify activations. Expert routing,
synthesis, cache retrieval, and field steering can influence subsequent
computation. Their bounds and effects are measured separately from the text
describing them.

The project's principle is to couple state, reality, and output. State is the
activations and maintained controls. Reality includes code results, fixed
documents, and an operator with an independent response. Output is the reply,
which can change what information returns next.

What continues between turns depends on what is retained. The shell keeps a
bounded transcript, can retrieve stored records, and can restore saved turns.
A river is an analogy for this continuity through change; it does not establish
a continuous hidden state, personal identity, or experience. A stored record
only helps a later reply if a mechanism makes it available.

The reading about consequences gives concrete examples. The question is
whether carried information supports better predictions, correction, and
transfer to new situations. Repeating a self-description does not demonstrate
that use. Reading about the machinery also does not grant direct access to it.

Source: `build_prompt`, `trim_session_context`, and `recover_session_context`
in `scripts/interactive_phenomenality.py`; `invariants/memory_engine.py` for
stored records.
