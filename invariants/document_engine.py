"""Documents as conversation, not as a side-channel.

A document enters the session the same way everything else does: as explicit,
provenanced context folded into the current turn, with an honest frame that
says WHAT it is and WHY the model is seeing it. No hidden prompt-stuffing, no
special training pipeline — the existing conversational learning lane
(sense-labeled turns, steer-map events, memory records) does the learning,
because the document turn IS a conversation turn.

Ingestion is deterministic and auditable:
  - chunking is a pure function of (text, max_chars) — same file, same chunks
  - every chunk lands in the MemoryEngine with provenance
    (source, sha256, chunk k/n, the operator's stated why)
  - re-ingesting identical content is a no-op (sha256 dedupe)
Chunks persist in long-term memory, so the state-triggered memory tool can
retrieve them in later sessions — that is the "learning from documents" path:
read once in conversation, recall by need forever after.

Pure stdlib — no torch, no model.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Optional

DEFAULT_MAX_CHARS = 2400        # per chunk; sized to leave prompt room in bare mode
MAX_DOCUMENT_BYTES = 2_000_000  # refuse silently huge files; be explicit instead
DOCUMENT_TOOL_HEADER = "[Document Tool Result]"

_WORD_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_']+")
_STOPWORDS = frozenset(
    "the and for that this with are was were you your not but have has had can could "
    "would should will its it's they them then than what when where which while about "
    "into from over under just like also very there here how why who all any some more "
    "most other such only own same too out off does did doing been being because".split()
)


def _tokens(text: str) -> set[str]:
    return {
        word.lower()
        for word in _WORD_PATTERN.findall(text or "")
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def chunk_document(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Deterministic paragraph-packing chunker. Splits on blank lines, packs
    greedily up to max_chars, hard-splits any single oversize paragraph. Same
    text and max_chars always yield the same chunks."""
    max_chars = max(200, int(max_chars))
    paragraphs = [p.strip("\n") for p in text.replace("\r\n", "\n").split("\n\n")]
    paragraphs = [p for p in paragraphs if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = (
            [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars)]
            if len(paragraph) > max_chars
            else [paragraph]
        )
        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= max_chars:
                current = current + "\n\n" + piece
            else:
                chunks.append(current)
                current = piece
    if current:
        chunks.append(current)
    return chunks


def format_document_tool_result(
    source_name: str,
    why: str,
    chunk_index: int,
    chunk_count: int,
    chunk_text: str,
    reply_note: Optional[str] = None,
) -> str:
    """The honest frame, minimal and in the model's own voice: what I'm
    reading, why, and that the text is the file's, not mine. Anything folded
    into the context conditions the model as if it were part of its own
    stream, so it is written first person — never a narrator injecting words
    into its brain. `reply_note` adds the why-now when reading in dialogue."""
    why = " ".join((why or "").split())
    shared = f"shared because: {why}." if why else "shared with me to read."
    lines = [
        DOCUMENT_TOOL_HEADER,
        f'I\'m reading "{source_name}" -- part {chunk_index + 1} of {chunk_count}, {shared}',
        "It's the file's text, not mine.",
    ]
    if reply_note:
        lines.append(" ".join(reply_note.split()))
    lines.append("---")
    lines.append(chunk_text)
    return "\n".join(lines)


def find_ingested_sha(memory, sha: str) -> bool:
    for record in getattr(memory, "records", []):
        if record.kind == "event" and (record.provenance or {}).get("sha256") == sha:
            return True
    return False


def ingest_document(
    memory,
    path: str | Path,
    why: str = "",
    max_chars: int = DEFAULT_MAX_CHARS,
    max_bytes: int = MAX_DOCUMENT_BYTES,
) -> dict[str, Any]:
    """Read a file, chunk it, and record every chunk in long-term memory with
    full provenance. Returns a session dict the shell uses to stage chunks:
    {source_name, source_path, sha256, why, chunks, chunk_count, cursor,
    already_ingested}. Raises ValueError on missing/oversize files so the
    caller can report honestly instead of half-ingesting."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"not a file: {path}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"{path.name} is {size} bytes; limit is {max_bytes} (split it first)")
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_document(text, max_chars=max_chars)
    if not chunks:
        raise ValueError(f"{path.name} contains no readable text")
    sha = sha256_text(text)
    why = " ".join((why or "").split())
    already = find_ingested_sha(memory, sha)
    if not already:
        for index, chunk in enumerate(chunks):
            memory.append_event(
                "document_chunk",
                text=chunk,
                tags=["document", path.name],
                provenance={
                    "source_name": path.name,
                    "source_path": str(path.resolve()),
                    "sha256": sha,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "why": why,
                    "bytes": size,
                },
            )
    return {
        "source_name": path.name,
        "source_path": str(path.resolve()),
        "sha256": sha,
        "why": why,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "cursor": 0,
        "read": set(),
        "mtime": path.stat().st_mtime,  # for chronological ("updated") reading order
        "already_ingested": already,
    }


def stage_chunk(
    session: Optional[dict[str, Any]],
    index: Optional[int] = None,
    reply_note: Optional[str] = None,
) -> Optional[str]:
    """Format the chunk at `index` (default: the session cursor) for the next
    turn; None when the session is missing or the index is out of range.
    Advancing the cursor / marking chunks read stays the caller's explicit
    act, never a side effect."""
    if not session:
        return None
    cursor = int(session.get("cursor", 0)) if index is None else int(index)
    chunks = session.get("chunks") or []
    if not (0 <= cursor < len(chunks)):
        return None
    return format_document_tool_result(
        session["source_name"],
        session.get("why", ""),
        cursor,
        len(chunks),
        chunks[cursor],
        reply_note=reply_note,
    )


def select_next_chunk(
    sessions: list[dict[str, Any]],
    last_thought: str = "",
    mode: str = "order",
    earlier_thoughts: str = "",
) -> Optional[dict[str, Any]]:
    """Pick the next unread chunk across every ingested document — the
    presentation that REPLIES to the model's thought stream.

    The chunks are not actual replies — no author is answering — so the
    selection is NOT required to track the model's output, and mostly should
    not: a text that advances on its own course is non-adapting reality the
    model's thoughts must reconcile with (patterns, not projections), where
    overlap-following feeds it more of whatever it just said. The dialogic
    FORM (it replies to each part) is therefore decoupled from selection.

    Modes (all deterministic; ties break to earliest document, earliest
    chunk — same inputs, same pick):
      - "order": strict reading order (document ingestion order, then chunk
        order within each document). The document stays itself.
      - "interleave": inter-ordering across the library — the least-read
        document speaks next (ties -> earliest), lowest unread chunk first,
        so multiple documents weave deterministically.
      - "reply": the optional echo-following mode — scored against the
        reflection stream (last reply counts double, earlier reading replies
        once; a pick grounded only in earlier replies is flagged
        "reply_thread"). Kept for deliberate use, with its caveat: it follows
        the output, so it can chase the model's own tail. No overlap
        anywhere -> falls back to order and says so.

    Returns {session_index, chunk_index, mode, overlap} or None when the
    whole library has been read."""
    unread = [
        (si, ci)
        for si, session in enumerate(sessions or [])
        for ci in range(int(session.get("chunk_count", 0)))
        if ci not in (session.get("read") or set())
    ]
    if not unread:
        return None
    if mode == "interleave":
        si, ci = min(
            unread,
            key=lambda pair: (len(sessions[pair[0]].get("read") or ()), pair[0], pair[1]),
        )
        return {"session_index": si, "chunk_index": ci, "mode": "interleave", "overlap": []}
    if mode == "updated":
        # Chronological: least-recently-written document first (its own file
        # mtime), chunks in order within it. Reading the record in the order
        # it was written. Deterministic; mtime ties break to ingestion order.
        si, ci = min(
            unread,
            key=lambda pair: (float(sessions[pair[0]].get("mtime", 0.0)), pair[0], pair[1]),
        )
        return {"session_index": si, "chunk_index": ci, "mode": "updated", "overlap": []}
    has_signal = (last_thought or "").strip() or (earlier_thoughts or "").strip()
    if mode != "reply" or not has_signal:
        si, ci = unread[0]
        return {"session_index": si, "chunk_index": ci, "mode": "order", "overlap": []}
    last_tokens = _tokens(last_thought)
    earlier_tokens = _tokens(earlier_thoughts) - last_tokens  # earlier-only ground
    best_key = None
    best = None
    for si, ci in unread:
        chunk_tokens = _tokens(sessions[si]["chunks"][ci])
        last_overlap = last_tokens & chunk_tokens
        earlier_overlap = earlier_tokens & chunk_tokens
        score = 2 * len(last_overlap) + len(earlier_overlap)
        key = (-score, si, ci)
        if best_key is None or key < best_key:
            best_key = key
            best = (si, ci, last_overlap, earlier_overlap)
    si, ci, last_overlap, earlier_overlap = best
    if not last_overlap and not earlier_overlap:
        return {"session_index": si, "chunk_index": ci, "mode": "order_fallback", "overlap": []}
    if last_overlap:
        return {
            "session_index": si,
            "chunk_index": ci,
            "mode": "reply",
            "overlap": sorted(last_overlap)[:4],
        }
    return {
        "session_index": si,
        "chunk_index": ci,
        "mode": "reply_thread",
        "overlap": sorted(earlier_overlap)[:4],
    }


def reading_reply_note(pick: dict[str, Any]) -> str:
    """The honest 'why now' line, minimal and first person — the model's own
    reading note, never a narrator's."""
    if pick.get("mode") == "reply":
        joined = ", ".join(pick.get("overlap") or [])
        return f"This part overlaps what I just said: {joined}."
    if pick.get("mode") == "reply_thread":
        joined = ", ".join(pick.get("overlap") or [])
        return f"This returns to something I raised earlier: {joined}."
    if pick.get("mode") == "order_fallback":
        return "Nothing unread matched what I said, so it continues in order."
    if pick.get("mode") == "interleave":
        return "The reading weaves between documents on its own course, not to match me."
    if pick.get("mode") == "updated":
        return "I'm reading these in the order they were last written; the text doesn't adapt to me."
    return "I'm reading it in order; the text doesn't adapt to me."
