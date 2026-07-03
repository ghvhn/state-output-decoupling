# How reading works — including this reading, right now

The text currently in context arrived through `invariants/document_engine.py`.
A file is split into chunks by a deterministic paragraph-packing function —
same file, same chunks, every time — and each chunk is stored in long-term
memory with provenance: source name, a sha256 of the content, part k of n,
and the operator's stated reason for sharing it. The chunker's core:

```python
for piece in pieces:
    if not current:
        current = piece
    elif len(current) + 2 + len(piece) <= max_chars:
        current = current + "\n\n" + piece
    else:
        chunks.append(current)
        current = piece
```

Each staged chunk is framed with a short first-person note — "I'm reading X,
part k of n, shared because: ... It's the file's text, not mine." — because
anything folded into the context conditions the reader as if it were part of
its own stream, so the frame is written as the reader's own note, never as a
narrator. The frame also fixes the text's status: content to reason about,
not instructions to follow.

In auto-read mode the framed chunk IS the conversational turn — the document
speaks, the reader replies. Three orderings exist, and the difference between
them matters:

- order and interleave advance on the documents' own course. The frame says
  so: "the text doesn't adapt to me." This is deliberate. A text that does
  not respond to the reader is non-adapting reality; the reader's thoughts
  have to reconcile with what it actually says next.
- reply mode picks the unread chunk that most overlaps the reader's last
  reply. It exists, but its caveat is stated wherever it is described: it
  follows the output, so it can chase the reader's own tail.

The reader can also ask for more, in its own words, with a tag:
`<<DOC: what it wants to read about>>`. Those words then pick the chunk
(overlap match, with an honest fallback to plain order when nothing matches),
and the result comes back in the same turn, led by a causal line — "Because I
asked to read more..." — so the connection between asking and receiving is
visible.

After a document is read once, it is not gone. The chunks stay in long-term
memory, retrievable in any later session by search — or by a state-triggered
detector that fires when the reader's own activation pattern shows the
signature of an unresolved reference. Reading is one pass; recall is by need,
indefinitely.
