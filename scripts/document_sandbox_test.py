"""Model-free tests: documents enter as conversation, code runs for real.

The document pipeline treats a file as part of the conversation — chunked
deterministically, recorded in memory with provenance, and staged into the
turn with an honest frame saying what it is and why it is being shown. The
sandbox runs the model's fenced python and returns what actually happened.

Run:
    .venv\\Scripts\\python.exe scripts\\document_sandbox_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.document_engine import (
    DOCUMENT_TOOL_HEADER,
    chunk_document,
    format_document_tool_result,
    ingest_document,
    sha256_text,
    stage_chunk,
)
from invariants.memory_engine import MemoryEngine
from invariants.sandbox import (
    extract_python_block,
    format_sandbox_tool_result,
    run_python,
)


def make_memory(tmp):
    return MemoryEngine(path=Path(tmp) / "memory.jsonl", scope="doc_test")


def test_chunking_is_deterministic_and_bounded():
    text = "\n\n".join(f"Paragraph {i} " + ("x" * 80) for i in range(40))
    chunks = chunk_document(text, max_chars=500)
    assert chunks == chunk_document(text, max_chars=500)      # same input, same chunks
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks).count("Paragraph 39") == 1          # nothing dropped
    # A single oversize paragraph is hard-split, not truncated.
    big = "y" * 1200
    parts = chunk_document(big, max_chars=500)
    assert len(parts) == 3 and "".join(parts) == big
    assert chunk_document("   \n\n  ", max_chars=500) == []


def test_ingest_records_provenance_and_dedupes():
    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "notes.md"
        doc.write_text("First paragraph.\n\nSecond paragraph.\n", encoding="utf-8")
        memory = make_memory(tmp)
        session = ingest_document(memory, doc, why="it explains the experiment")
        assert session["chunk_count"] >= 1 and not session["already_ingested"]
        records = [r for r in memory.records if "document" in (r.tags or [])]
        assert len(records) == session["chunk_count"]
        prov = records[0].provenance
        assert prov["source_name"] == "notes.md"
        assert prov["sha256"] == sha256_text(doc.read_text(encoding="utf-8"))
        assert prov["why"] == "it explains the experiment"
        assert prov["chunk_count"] == session["chunk_count"]

        # Same content again: session works, but no duplicate records.
        again = ingest_document(memory, doc, why="second look")
        assert again["already_ingested"]
        assert len([r for r in memory.records if "document" in (r.tags or [])]) == len(records)

        # The chunks are retrievable later -- that is the learning path.
        hits = memory.search("Second paragraph", scope="doc_test")
        assert hits and "Second paragraph." in hits[0].text

        # Missing files fail loudly, never half-ingest.
        try:
            ingest_document(memory, Path(tmp) / "missing.md", why="")
            assert False, "missing file should raise"
        except ValueError:
            pass


def test_document_frame_says_what_and_why():
    framed = format_document_tool_result("notes.md", "it defines the terms", 1, 3, "BODY TEXT")
    assert framed.startswith(DOCUMENT_TOOL_HEADER)
    assert 'I\'m reading "notes.md" -- part 2 of 3' in framed  # first person, minimal
    assert "shared because: it defines the terms." in framed
    assert "It's the file's text, not mine." in framed          # epistemic status, own voice
    assert framed.endswith("BODY TEXT")

    session = {
        "source_name": "notes.md",
        "why": "context",
        "chunks": ["A", "B"],
        "chunk_count": 2,
        "cursor": 1,
    }
    staged = stage_chunk(session)
    assert "part 2 of 2" in staged and staged.endswith("B")
    session["cursor"] = 2
    assert stage_chunk(session) is None                        # exhausted, no invention
    assert stage_chunk(None) is None


def test_document_folds_into_the_conversation_turn():
    from scripts.interactive_phenomenality import build_prompt, scrub_unstaged_memory_status

    framed = format_document_tool_result("notes.md", "context for the question", 0, 1, "The key fact is 42.")
    prompt = build_prompt("What does the document say?", document_tool_result=framed)
    assert "The key fact is 42." in prompt
    assert prompt.index(DOCUMENT_TOOL_HEADER) < prompt.index("What does the document say?")
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    bare = build_prompt("What does the document say?")
    assert DOCUMENT_TOOL_HEADER not in bare

    # Hallucinated document/sandbox headers are scrubbed when nothing was
    # staged (same per-line contract as the other tool headers).
    fake = "[Document Tool Result]\nSome claimed content.\nReal line."
    scrubbed = scrub_unstaged_memory_status(fake)
    assert DOCUMENT_TOOL_HEADER not in scrubbed and "Real line." in scrubbed
    fake2 = "[Sandbox Tool Result]\nReal answer."
    scrubbed2 = scrub_unstaged_memory_status(fake2)
    assert "[Sandbox Tool Result]" not in scrubbed2 and "Real answer." in scrubbed2
    # And when the result WAS staged, the response passes through untouched
    # except for tool-call tags.
    staged_pass = scrub_unstaged_memory_status("fine answer", document_tool_result=framed)
    assert staged_pass == "fine answer"


def test_sandbox_runs_real_code_and_reports_honestly():
    with tempfile.TemporaryDirectory() as tmp:
        ok = run_python("print(2 + 2)", timeout_sec=30, cwd=Path(tmp))
        assert ok["ok"] and ok["exit_code"] == 0 and ok["stdout"].strip() == "4"

        bad = run_python("raise ValueError('boom')", timeout_sec=30, cwd=Path(tmp))
        assert not bad["ok"] and bad["exit_code"] != 0
        assert "ValueError" in bad["stderr"] and "boom" in bad["stderr"]

        slow = run_python("import time; time.sleep(30)", timeout_sec=1, cwd=Path(tmp))
        assert slow["timed_out"] and not slow["ok"]

        noisy = run_python("print('z' * 10000)", timeout_sec=30, cwd=Path(tmp), max_output_chars=200)
        assert len(noisy["stdout"]) < 400 and "truncated" in noisy["stdout"]

        framed = format_sandbox_tool_result(ok)
        assert framed.startswith("[Sandbox Tool Result]")
        assert "I ran the code I wrote" in framed and "exit_code=0" in framed
        assert "4" in framed


def test_python_block_extraction_takes_the_last_fence():
    text = (
        "First try:\n```python\nprint('old')\n```\n"
        "Actually, run this instead:\n```python\nprint('new')\n```\n"
    )
    assert extract_python_block(text) == "print('new')"
    assert extract_python_block("no code here") is None
    assert extract_python_block("```python\n\n```") is None    # empty fence is not code
    assert extract_python_block("```py\nx = 1\n```") == "x = 1"


def _session(name, chunks, read=()):
    return {
        "source_name": name,
        "why": "context",
        "chunks": chunks,
        "chunk_count": len(chunks),
        "cursor": 0,
        "read": set(read),
    }


def test_reading_replies_to_thoughts_or_keeps_order():
    from invariants.document_engine import reading_reply_note, select_next_chunk

    library = [
        _session("alpha.md", ["Cats and dogs.", "Steering vectors and residual caps."], read={0}),
        _session("beta.md", ["Cache deltas and outcomes.", "Weather patterns."]),
    ]

    # Order mode: strict document-then-chunk order over unread chunks. The
    # frame says the text does not adapt to the reader.
    pick = select_next_chunk(library, "anything", "order")
    assert (pick["session_index"], pick["chunk_index"], pick["mode"]) == (0, 1, "order")
    assert "doesn't adapt to me" in reading_reply_note(pick)

    # Reply mode (optional, echo-following): jumps to the unread chunk that
    # answers the last thought, across documents, deterministically.
    pick = select_next_chunk(library, "I keep thinking about the cache and its deltas.", "reply")
    assert (pick["session_index"], pick["chunk_index"], pick["mode"]) == (1, 0, "reply")
    assert "cache" in pick["overlap"] and "deltas" in pick["overlap"]
    again = select_next_chunk(library, "I keep thinking about the cache and its deltas.", "reply")
    assert again == pick                                        # same thought, same pick

    # The why-now note is honest about the ground it shares with the thought.
    note = reading_reply_note(pick)
    assert "overlaps what I just said" in note and "cache" in note

    # No overlap anywhere -> falls back to order and says so.
    pick = select_next_chunk(library, "zzz qqq totally unrelated", "reply")
    assert pick["mode"] == "order_fallback" and (pick["session_index"], pick["chunk_index"]) == (0, 1)
    assert "continues in order" in reading_reply_note(pick)

    # Empty reflection stream in reply mode degrades to order, not to a guess.
    pick = select_next_chunk(library, "", "reply")
    assert pick["mode"] == "order"

    # Fully read library -> None, never re-serves.
    for session in library:
        session["read"] = set(range(session["chunk_count"]))
    assert select_next_chunk(library, "cache", "reply") is None
    assert select_next_chunk([], "cache", "reply") is None


def test_interleave_weaves_documents_on_their_own_course():
    from invariants.document_engine import reading_reply_note, select_next_chunk

    library = [
        _session("alpha.md", ["A0 unique.", "A1 unique.", "A2 unique."]),
        _session("beta.md", ["B0 cache deltas everywhere.", "B1 unique."]),
    ]

    # Least-read document speaks next; the model's reply is IGNORED by design
    # (the chunk is not an actual reply, so it need not track the output).
    sequence = []
    for _ in range(5):
        pick = select_next_chunk(library, "cache deltas cache deltas", "interleave")
        assert pick["mode"] == "interleave" and pick["overlap"] == []
        session = library[pick["session_index"]]
        session["read"].add(pick["chunk_index"])
        sequence.append((pick["session_index"], pick["chunk_index"]))
    # Ties break to the earliest document; reads alternate until beta runs out.
    assert sequence == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)]
    assert select_next_chunk(library, "", "interleave") is None

    note = reading_reply_note({"mode": "interleave", "overlap": []})
    assert "weaves between documents" in note
    assert "not to match me" in note


def test_reply_mode_can_return_to_earlier_threads():
    from invariants.document_engine import reading_reply_note, select_next_chunk

    library = [
        _session("alpha.md", ["Nothing shared here at all.", "Orbital mechanics and albedo steering."]),
    ]
    # The LAST reply has no overlap, but an earlier reflection raised the
    # thread -- the pick is flagged as a thread return, honestly.
    pick = select_next_chunk(
        library,
        "Completely different musing now.",
        "reply",
        earlier_thoughts="Earlier I wondered about albedo and orbital drift.",
    )
    assert pick["mode"] == "reply_thread"
    assert (pick["session_index"], pick["chunk_index"]) == (0, 1)
    assert "albedo" in pick["overlap"] and "orbital" in pick["overlap"]
    assert "raised earlier" in reading_reply_note(pick)

    # Recency dominates when both match: last-reply ground (double weight)
    # beats an earlier-only match of similar size.
    library2 = [
        _session("a.md", ["cache deltas outcomes evidence.", "albedo orbital drift steering."]),
    ]
    pick = select_next_chunk(
        library2,
        "thinking about cache deltas",
        "reply",
        earlier_thoughts="albedo orbital drift",
    )
    assert (pick["chunk_index"], pick["mode"]) == (0, "reply")


def test_impact_attribution_is_minimal_and_first_person():
    from scripts.interactive_phenomenality import impact_note

    note = impact_note('asked memory for  "cache   deltas"')
    # Minimal, first person, causal -- the model's own noticing, no narrator,
    # no second person, no injected instructions.
    assert note == 'Because I asked memory for "cache deltas":'
    assert "you" not in note.lower()
    # It reads as a causal statement, not a status header the
    # unstaged-scrubber would strip.
    from scripts.interactive_phenomenality import scrub_unstaged_memory_status

    kept = scrub_unstaged_memory_status(f"{note}\nanswer text")
    assert "Because I asked" in kept and "answer text" in kept


def test_reply_note_lands_in_the_frame():
    from invariants.document_engine import reading_reply_note

    pick = {"mode": "reply", "overlap": ["cache", "steering"]}
    framed = format_document_tool_result(
        "alpha.md", "context", 1, 2, "BODY", reply_note=reading_reply_note(pick)
    )
    assert "overlaps what I just said" in framed
    assert framed.index("overlaps what I just said") < framed.index("---")
    assert framed.endswith("BODY")
    # stage_chunk carries the note and honors an explicit index.
    session = _session("alpha.md", ["A", "B"])
    staged = stage_chunk(session, index=1, reply_note="WHY-NOW LINE")
    assert "part 2 of 2" in staged and "WHY-NOW LINE" in staged and staged.endswith("B")


TESTS = [
    test_chunking_is_deterministic_and_bounded,
    test_ingest_records_provenance_and_dedupes,
    test_document_frame_says_what_and_why,
    test_document_folds_into_the_conversation_turn,
    test_sandbox_runs_real_code_and_reports_honestly,
    test_python_block_extraction_takes_the_last_fence,
    test_reading_replies_to_thoughts_or_keeps_order,
    test_interleave_weaves_documents_on_their_own_course,
    test_reply_mode_can_return_to_earlier_threads,
    test_impact_attribution_is_minimal_and_first_person,
    test_reply_note_lands_in_the_frame,
]


def main():
    print("DOCUMENT + SANDBOX TEST -- data as conversation, code run for real\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  Documents arrive framed (what + why) and remembered; sandbox output is real observation.")


if __name__ == "__main__":
    main()
