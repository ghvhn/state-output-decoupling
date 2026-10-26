"""Successful, resumable document reading with explicit contextual rereads."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from invariants.document_engine import (
    begin_reading_pass,
    document_layout_sha256,
    ingest_document,
    record_chunk_read,
    restore_read_progress,
    select_next_chunk,
    stage_chunk,
)
from invariants.memory_engine import MemoryEngine


class DocumentReadingPassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.memory_path = self.root / "memory.jsonl"
        self.memory = MemoryEngine(path=self.memory_path, scope="reading_pass_test")

    def document(self, paragraphs=4, max_chars=240):
        path = self.root / "source.txt"
        path.write_text("\n\n".join(f"topic{i:03d} " + "x" * 200 for i in range(paragraphs)), encoding="utf-8")
        return ingest_document(self.memory, path, max_chars=max_chars)

    def test_reply_selection_covers_more_than_twenty_chunks(self):
        session = self.document(paragraphs=31)
        begin_reading_pass(self.memory, session, context="Find answers to changing questions")
        visited = []
        for index in reversed(range(31)):
            pick = select_next_chunk([session], f"What supports topic{index:03d}?", "reply")
            self.assertEqual(pick["chunk_index"], index)
            self.assertTrue(record_chunk_read(self.memory, session, index))
            visited.append(index)
        self.assertEqual(len(set(visited)), 31)
        self.assertEqual(session["read"], set(range(31)))
        self.assertIsNone(select_next_chunk([session], "topic000", "reply"))

    def test_no_match_still_completes_remaining_coverage(self):
        session = self.document()
        record_chunk_read(self.memory, session, 3)
        sequence = []
        while (pick := select_next_chunk([session], "unrelated planets", "reply")) is not None:
            sequence.append(pick["chunk_index"])
            record_chunk_read(self.memory, session, pick["chunk_index"])
        self.assertEqual(sequence, [0, 1, 2])

    def test_staging_and_failed_generation_do_not_commit(self):
        session = self.document()
        begin_reading_pass(self.memory, session)
        first = select_next_chunk([session], "topic002", "reply")
        self.assertIn("topic002", stage_chunk(session, first["chunk_index"]))
        # A failed response never reaches record_chunk_read.
        self.assertEqual(session["read"], set())
        self.assertEqual(select_next_chunk([session], "topic002", "reply"), first)
        restarted = MemoryEngine(path=self.memory_path, scope="reading_pass_test")
        resumed = ingest_document(restarted, session["source_path"], max_chars=240)
        self.assertEqual(resumed["read"], set())
        self.assertEqual(resumed["reading_pass_id"], session["reading_pass_id"])

    def test_failed_durable_commit_remains_retryable(self):
        session = self.document()
        begin_reading_pass(self.memory, session)
        with patch.object(self.memory, "append_event", side_effect=OSError("disk write failed")):
            with self.assertRaises(OSError):
                record_chunk_read(self.memory, session, 1)
        self.assertNotIn(1, session["read"])
        self.assertTrue(record_chunk_read(self.memory, session, 1))

    def test_commit_is_validated_and_idempotent(self):
        session = self.document()
        initial_records = len(self.memory.records)
        for invalid in (-1, session["chunk_count"], 1.5, True, "1"):
            with self.subTest(index=invalid), self.assertRaises(ValueError):
                record_chunk_read(self.memory, session, invalid)
        self.assertEqual(len(self.memory.records), initial_records)
        self.assertTrue(record_chunk_read(self.memory, session, 1))
        committed_records = len(self.memory.records)
        self.assertFalse(record_chunk_read(self.memory, session, 1))
        self.assertEqual(len(self.memory.records), committed_records)

    def test_restart_resumes_current_pass_and_fills_holes(self):
        session = self.document()
        record_chunk_read(self.memory, session, 3)
        record_chunk_read(self.memory, session, 0)
        restarted = MemoryEngine(path=self.memory_path, scope="reading_pass_test")
        resumed = ingest_document(restarted, session["source_path"], max_chars=240)
        self.assertEqual(resumed["read"], {0, 3})
        self.assertEqual(resumed["reading_pass_id"], session["reading_pass_id"])
        self.assertEqual(select_next_chunk([resumed], mode="order")["chunk_index"], 1)
        begin_reading_pass(restarted, resumed, context="Continue the discussion")
        self.assertEqual(resumed["read"], {0, 3})

    def test_reread_has_new_context_and_retains_prior_history(self):
        session = self.document()
        begin_reading_pass(self.memory, session, context="original question")
        old_pass = session["reading_pass_id"]
        for index in range(session["chunk_count"]):
            record_chunk_read(self.memory, session, index)
        old_records = list(self.memory.records)
        begin_reading_pass(self.memory, session, context="a different question", restart=True)
        self.assertNotEqual(session["reading_pass_id"], old_pass)
        self.assertEqual(session["reading_context"], "a different question")
        self.assertEqual(session["read"], set())
        self.assertEqual(self.memory.records[:len(old_records)], old_records)
        self.assertEqual(
            restore_read_progress(self.memory, session["sha256"], session["chunk_count"], reading_pass_id=old_pass),
            set(range(session["chunk_count"])),
        )
        self.assertEqual(restore_read_progress(self.memory, session["sha256"], session["chunk_count"]), set())
        record_chunk_read(self.memory, session, 2)
        restored = ingest_document(self.memory, session["source_path"], max_chars=240)
        self.assertEqual(restored["read"], {2})
        self.assertTrue(restored["already_ingested"])

    def test_changed_chunk_boundaries_do_not_inherit_progress(self):
        path = self.root / "layout.txt"
        path.write_text("x" * 700, encoding="utf-8")
        first = ingest_document(self.memory, path, max_chars=240)
        record_chunk_read(self.memory, first, 0)
        second = ingest_document(self.memory, path, max_chars=300)
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(first["chunk_count"], second["chunk_count"])
        self.assertNotEqual(first["layout_sha256"], second["layout_sha256"])
        self.assertEqual(second["read"], set())
        self.assertIsNone(second["reading_pass_id"])
        self.assertNotEqual(document_layout_sha256(["ab", "c"]), document_layout_sha256(["a", "bc"]))

    def test_legacy_presentations_remain_history_without_suppressing_pass(self):
        session = self.document()
        self.memory.append_event("document_chunk_read", provenance={"sha256": session["sha256"], "chunk_index": 0})
        self.assertEqual(restore_read_progress(self.memory, session["sha256"], session["chunk_count"]), {0})
        reloaded = ingest_document(self.memory, session["source_path"], max_chars=240)
        self.assertEqual(reloaded["read"], set())
        begin_reading_pass(self.memory, reloaded)
        self.assertEqual(restore_read_progress(self.memory, session["sha256"], session["chunk_count"]), set())
        self.assertTrue(any(r.provenance.get("chunk_index") == 0 and not r.provenance.get("reading_pass_id") for r in self.memory.records))

    def test_document_frame_contains_source_metadata_without_reading_instructions(self):
        from scripts.interactive_phenomenality import stage_document_read

        session = self.document()
        frame = stage_document_read(session, index=1)
        self.assertIn("Source:", frame)
        self.assertIn("Passage:", frame)
        self.assertTrue(frame.endswith(session["chunks"][1]))
        for instruction in ("I'm reading", "answers and evidence", "No prior question",
                            "leaves unanswered", "What does this passage"):
            self.assertNotIn(instruction, frame)


if __name__ == "__main__":
    unittest.main()
