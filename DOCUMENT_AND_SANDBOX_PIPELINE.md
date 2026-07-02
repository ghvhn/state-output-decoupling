# Documents as Conversation + Sandbox Connection

Two capabilities, both entering through the same door: the conversation.
Nothing is hidden prompt-stuffing; everything arrives framed, provenanced,
and one-turn-staged, so the existing conversational learning lane
(sense-labeled turns, steer-map events, memory records) learns from it with
no new learning machinery.

## Documents (`invariants/document_engine.py`)

`:doc <path> [because <why>]` in the interactive shell:

1. **Ingest, deterministically.** The file is chunked by a pure function of
   (text, max_chars) — same file, same chunks — and every chunk lands in the
   MemoryEngine with provenance: source name/path, sha256, chunk k/n, and the
   operator's stated *why*. Re-ingesting identical content is a sha-deduped
   no-op. Oversize files are refused loudly, never half-ingested.
2. **Stage with an honest frame — minimal, first person.** Nothing folded
   into the context is a narrator injecting words into its brain: internal
   prompts read as the model's own voice, and say only what is needed:
   *"I'm reading "X" — part k of n, shared because: WHY. It's the file's
   text, not mine."* `:doc next` stages the next chunk — advancing is always
   the operator's explicit act.
3. **Learn the way conversation learns.** The document turn is a normal turn:
   `sense_score` labels it, steering events get conversation-basis labels,
   and the chunks persist in long-term memory where the state-triggered
   memory tool can retrieve them in any later session. Read once in
   conversation, recall by need afterward — that is the learning path.

Prompt-use stays one-turn (like the memory tool); long-term residence is the
memory store, not the context window. Hallucinated `[Document Tool Result]`
headers are scrubbed when nothing was staged, same contract as the other
tools.

### Reading as dialogue: the document speaks, the model replies

`:doc read [n] [order|interleave|reply]` turns reading into a conversation.
For up to n turns (capped at 20 per command), the framed chunk simply IS
the user turn — no synthesized question, no cue, no words put in anyone's
mouth; the model responds to the text natively — and it runs as a NORMAL turn — so every reading turn is
sense-labeled, channel-accounted, and remembered like any other. `:doc stop`
interrupts.

Design stance (Gavin's correction): the chunks are **not actual replies** —
no author is answering — so the selection is *not required to track the
model's output*, and mostly should not. The dialogic FORM (the document
speaks, the model replies each turn) is decoupled from selection. A text
that advances on its own course is non-adapting reality the model's thoughts
must reconcile with — patterns, not projections — where overlap-following
feeds it more of whatever it just said. The collision is the point.

Three orderings, all deterministic (ties break to earliest document,
earliest chunk — same inputs, same pick):

- **order** (default) — strict reading order across the library. Note:
  *"I'm reading it in order; the text doesn't adapt to me."*
- **interleave** (alias `weave`) — inter-ordering across documents: the
  least-read document speaks next, lowest unread chunk first, so multiple
  documents weave. Note: *"The reading weaves between documents on its own
  course, not to match me."*
- **reply** — the optional echo-following mode, kept for deliberate use with
  its caveat stated (it follows the output, so it can chase the model's own
  tail). Scores against the reflection stream: the last reply counts double,
  earlier reading replies once; a pick grounded only in earlier replies is
  flagged as a **thread return** (*"This returns to something I raised
  earlier"*). No overlap anywhere → falls back to order and says so.

Nothing is ever re-served (per-chunk read set), advancing is bounded and
operator-initiated, and a fully-read library ends the dialogue explicitly.

### Self-reading (noticed by Gavin, 2026-07-02)

Feeding the repo's own design docs through `:doc` means the model learns its
own architecture the way humans learn theirs: from external documentation,
with provenance, as content about an object that happens to be itself — no
privileged introspection port, no injected self-narrative. The recursion is
live (the envelope caps the turns that read about the envelope; the docs
quote its own first-person inner voice back to it), and afterwards the
chunks sit in long-term memory where the state-triggered retrieval can
surface a description of an intervention at the moment the intervention
fires.

Honest boundary: reading about the architecture is not reflexive knowledge
of it — whether the content ever BINDS to the states it describes is the
open self-application lock (see ISOLATING_UNDERSTANDING.md). It is now
cheaply testable: read it the self-docs and matched control docs (another
system's architecture) through the same pipeline, one variable, and compare
the per-turn sense/phenomenality labels that already accrue for free.

## Coding sandbox (`invariants/sandbox.py`)

`:sandbox on` (default OFF, per-session, deliberate):

- **No taught syntax.** When the model's reply contains an ordinary fenced
  ```python block — something it emits natively — the shell runs the LAST
  such block in a separate `python -I` interpreter (isolated mode, no user
  site-packages, no PYTHON* env) inside `invariants/out/sandbox/`, with a
  hard timeout and truncated I/O.
- **Real results return as conversation, in its own voice.** The next turn
  carries `[Sandbox Tool Result]` framing, minimal and first person: *"I ran
  the code I wrote; exit_code=0 (0.3s)"* plus actual stdout/stderr.
- **Execution is an objective outcome.** Every run is logged to memory with
  provenance (code sha, exit, duration) and observed into the tuner as
  `sandbox_success` (exit 0 and no timeout = 1.0), so the distribution is
  inspectable via `:tune` like every other signal.

Honest scope, stated plainly: this is process isolation — not a hard
security boundary against a determined adversary. It is a local research
affordance on the operator's own machine, off by default, enabled per
session, with every execution printed and logged.

## Tests

`scripts/document_sandbox_test.py` (model-free): deterministic bounded
chunking with no dropped text, provenance + sha dedupe + later
retrievability, the what/why frame and its epistemic-status line, folding
into the bare-mode user turn ahead of the user's text, unstaged-header
scrubbing, real subprocess execution (success, exception capture, timeout,
output truncation), and last-fence extraction.
