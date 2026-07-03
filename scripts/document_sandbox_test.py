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


def test_updated_mode_reads_in_the_order_files_were_written():
    from invariants.document_engine import reading_reply_note, select_next_chunk

    newer = _session("written_later.md", ["L0", "L1"])
    older = _session("written_first.md", ["F0"])
    newer["mtime"], older["mtime"] = 2000.0, 1000.0
    library = [newer, older]  # ingestion order deliberately NOT chronological

    sequence = []
    for _ in range(3):
        pick = select_next_chunk(library, "ignored words", "updated")
        assert pick["mode"] == "updated" and pick["overlap"] == []
        library[pick["session_index"]]["read"].add(pick["chunk_index"])
        sequence.append((pick["session_index"], pick["chunk_index"]))
    # Oldest file first regardless of ingestion order, chunks in order within.
    assert sequence == [(1, 0), (0, 0), (0, 1)]
    assert select_next_chunk(library, "", "updated") is None

    # mtime tie -> ingestion order (deterministic).
    a, b = _session("a.md", ["A"]), _session("b.md", ["B"])
    a["mtime"] = b["mtime"] = 500.0
    pick = select_next_chunk([a, b], "", "updated")
    assert (pick["session_index"], pick["chunk_index"]) == (0, 0)

    note = reading_reply_note({"mode": "updated", "overlap": []})
    assert "order they were last written" in note and "doesn't adapt to me" in note

    # ingest records the file's real mtime for this ordering.
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "t.md"
        doc.write_text("Some text.", encoding="utf-8")
        memory = make_memory(tmp)
        session = ingest_document(memory, doc, why="")
        assert abs(session["mtime"] - os.path.getmtime(doc)) < 1.0


def test_resume_is_real_across_sessions():
    """Progress lives in the memory record, not the process: re-ingesting the
    same content restores the read-set, so ':doc read' after a restart picks
    up where the reading stopped."""
    from invariants.document_engine import (
        record_chunk_read,
        restore_read_progress,
        select_next_chunk,
    )

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "long.md"
        doc.write_text("\n\n".join(f"Part {i} " + ("x" * 300) for i in range(6)), encoding="utf-8")
        memory = make_memory(tmp)

        first = ingest_document(memory, doc, why="resume test", max_chars=400)
        assert first["read"] == set() and first["chunk_count"] >= 3
        # A session reads chunks 0 and 1, then the shell restarts.
        record_chunk_read(memory, first, 0)
        record_chunk_read(memory, first, 1)

        # New session, same memory file on disk.
        memory2 = MemoryEngine(path=Path(tmp) / "memory.jsonl", scope="doc_test")
        resumed = ingest_document(memory2, doc, why="resume test", max_chars=400)
        assert resumed["already_ingested"]
        assert resumed["read"] == {0, 1}                      # progress restored
        assert resumed["cursor"] == 1                          # ':doc next' -> chunk 2
        pick = select_next_chunk([resumed], "", "order")
        assert pick["chunk_index"] == 2                        # reading continues, not restarts

        # Different content: no progress borrowed across shas.
        other = Path(tmp) / "other.md"
        other.write_text("Entirely different text.", encoding="utf-8")
        fresh = ingest_document(memory2, other, why="")
        assert fresh["read"] == set()

        # Restore is bounded by the current chunk count (stale indices ignored).
        assert restore_read_progress(memory2, first["sha256"], 1) == {0}


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


def test_paired_threshold_goes_both_ways():
    """Any threshold stream can anchor any other: the target bar is the cut
    (in the target's own units) between anchor-fired and anchor-unfired
    turns, over the per-turn signals table."""
    from scripts.interactive_phenomenality import paired_threshold, resolve_stream

    rows = (
        [{"sense": 0.4, "probe_authority": 0.2}] * 5      # authority fired, sense high
        + [{"sense": -0.1, "probe_authority": -0.3}] * 5  # authority unfired, sense low
        + [{"sense": 0.9}]                                # missing anchor: ignored
    )
    # Forward: sense bar anchored to the authority probe (fires at >= 0).
    v = paired_threshold(rows, "sense", "probe_authority", 0.0, ">=")
    assert abs(v - (0.4 + (-0.1)) / 2.0) < 1e-9
    # Reverse: authority bar anchored to productivity (sense fires at >= 0).
    v2 = paired_threshold(rows, "probe_authority", "sense", 0.0, ">=")
    assert abs(v2 - (0.2 + (-0.3)) / 2.0) < 1e-9
    # Comparator respected (<= anchors on the LOW side).
    v3 = paired_threshold(rows, "sense", "probe_authority", 0.0, "<=")
    assert abs(v3 - ((-0.1) + 0.4) / 2.0) < 1e-9
    # Deterministic; refuses on thin or one-sided evidence.
    assert paired_threshold(rows, "sense", "probe_authority", 0.0, ">=") == v
    assert paired_threshold(rows[:6], "sense", "probe_authority", 0.0, ">=") is None
    assert paired_threshold([{"sense": 1.0, "probe_authority": 1.0}] * 20,
                            "sense", "probe_authority", 0.0, ">=") is None  # never unfired
    assert paired_threshold([], "sense", "probe_authority", 0.0, ">=") is None

    # Stream resolution: aliases, registered names, bare probe names.
    class FakeTuner:
        triggers = {"intent_settling": 1, "probe_authority": 1}
    assert resolve_stream("intent", FakeTuner) == "intent_settling"
    assert resolve_stream("productive", FakeTuner) == "sense"
    assert resolve_stream("impact", FakeTuner) == "words_had_impact"
    assert resolve_stream("authority", FakeTuner) == "probe_authority"
    assert resolve_stream("nonexistent", FakeTuner) is None


def test_calibration_policy_refuses_what_should_not_self_calibrate():
    from scripts.interactive_phenomenality import calibration_policy

    # Observed thresholds: safe to percentile-calibrate.
    for name in ("claimmap_tension", "memory_need", "conversation_productive",
                 "eot_urgency", "intent_settling"):
        route, _ = calibration_policy(name)
        assert route == "threshold", name
    # Dedicated data routes.
    assert calibration_policy("steer_cap_fraction")[0] == "cap"
    assert calibration_policy("steer_band")[0] == "band"
    # Binary outcome streams: a percentile bar is meaningless.
    for name in ("sandbox_success", "words_had_impact"):
        route, reason = calibration_policy(name)
        assert route == "reject" and "binary" in reason
    # Strength/budget knobs: circular -- they shape the distribution they
    # would calibrate to. The system refuses to approve its own settings.
    for name in ("claimmap_alpha", "memory_alpha", "steer_fraction",
                 "steer_layer_sweep", "response_tokens", "routing_events",
                 "synthesis_steps", "plateau_epsilon", "steer_band_lo"):
        route, reason = calibration_policy(name)
        assert route == "reject" and "circular" in reason, name
    # Unknown names default to threshold; the handler then requires a real
    # registered trigger with >=10 observed signals before acting.
    assert calibration_policy("some_future_signal")[0] == "threshold"


def test_intent_relative_threshold_is_a_discriminant_cut():
    from scripts.interactive_phenomenality import intent_relative_threshold

    # Settling turns cohere around 0.4; non-settling around -0.1: bar lands
    # between the group medians, so "productive" means "coheres like the
    # turns that actually shaped intent".
    pairs = [(0.2, 0.35), (0.1, 0.4), (0.3, 0.45), (0.05, 0.4), (0.15, 0.5),
             (-0.1, -0.1), (0.0, -0.05), (-0.2, -0.15), (-0.05, -0.1), (0.0, 0.0)]
    v = intent_relative_threshold(pairs)
    assert abs(v - (0.4 + (-0.1)) / 2.0) < 1e-9    # midpoint of medians (0.4, -0.1)
    assert intent_relative_threshold(pairs) == v    # deterministic
    # Refuses without enough evidence on BOTH sides.
    assert intent_relative_threshold(pairs[:6]) is None
    assert intent_relative_threshold([(0.1, 0.3)] * 10) is None
    assert intent_relative_threshold([]) is None


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
    test_updated_mode_reads_in_the_order_files_were_written,
    test_resume_is_real_across_sessions,
    test_reply_mode_can_return_to_earlier_threads,
    test_impact_attribution_is_minimal_and_first_person,
    test_paired_threshold_goes_both_ways,
    test_calibration_policy_refuses_what_should_not_self_calibrate,
    test_intent_relative_threshold_is_a_discriminant_cut,
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
