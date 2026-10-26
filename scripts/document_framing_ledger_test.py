"""Model-free checks for the document framing ledger."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.document_framing_ledger import (
    DocumentFramingLedger,
    framing_packet,
    staged_document_target,
)
from invariants.document_engine import stage_chunk


def test_packet_is_deterministic_and_explicit():
    packet = framing_packet(
        document_sha="doc-sha",
        chunk_index=2,
        chunk_text="Source passage.",
        response_text="My framing.",
    )
    assert "document_sha256=doc-sha" in packet
    assert "chunk_index=2" in packet
    assert "[document_chunk]" in packet
    assert "[reader_framing]" in packet
    assert packet == framing_packet(
        document_sha="doc-sha",
        chunk_index=2,
        chunk_text="Source passage.",
        response_text="My framing.",
    )


def test_only_a_single_staged_chunk_is_a_framing_target():
    session = {
        "source_name": "source.md",
        "why": "compare its framing later",
        "cursor": 1,
        "chunks": ["Earlier.", "Target chunk."],
    }
    staged = stage_chunk(session)
    assert staged_document_target(session, staged) == (1, "Target chunk.")
    assert staged_document_target(session, "[Causal result]\n" + staged) == (1, "Target chunk.")
    assert staged_document_target(session, "a whole document library") is None


def test_first_snapshot_then_matched_revision_persists():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = DocumentFramingLedger(root / "ledger.jsonl", root / "artifacts")
        first_trace = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        first = ledger.record_snapshot(
            document_sha="same-doc",
            source_name="source.md",
            chunk_index=0,
            chunk_count=2,
            chunk_text="The source says X.",
            response_text="I initially take this to mean X.",
            trace=first_trace,
            turn_id="turn-1",
            context_flags={"document_only": True},
        )
        assert first.revision is None
        assert (root / first.snapshot["artifact"]).exists()
        first_payload = torch.load(root / first.snapshot["artifact"], weights_only=True)
        assert "The source says X." in first_payload["packet"]
        assert "I initially take this to mean X." in first_payload["packet"]

        second_trace = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        second = ledger.record_snapshot(
            document_sha="same-doc",
            source_name="source.md",
            chunk_index=0,
            chunk_count=2,
            chunk_text="The source says X.",
            response_text="After later evidence, I take X more narrowly.",
            trace=second_trace,
            turn_id="turn-2",
            context_flags={"document_only": True},
        )
        assert second.revision is not None
        assert second.revision["baseline_snapshot_id"] == first.snapshot["record_id"]
        assert second.revision["current_snapshot_id"] == second.snapshot["record_id"]
        assert abs(second.revision["delta_norm"] - torch.tensor([[1.0, 0.0], [0.0, 2.0]]).norm().item()) < 1e-6
        assert (root / second.revision["artifact"]).exists()

        restored = DocumentFramingLedger(root / "ledger.jsonl", root / "artifacts")
        assert restored.summary()["snapshots"] == 2
        assert restored.summary()["revisions"] == 1
        assert len(restored.snapshots("same-doc", 0, "main")) == 2


def test_incompatible_traces_remain_snapshots_without_a_false_revision():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ledger = DocumentFramingLedger(root / "ledger.jsonl", root / "artifacts")
        common = dict(
            document_sha="same-doc",
            source_name="source.md",
            chunk_index=1,
            chunk_count=2,
            chunk_text="A chunk.",
            reader_id="reader-a",
        )
        ledger.record_snapshot(response_text="First framing.", trace=torch.ones(2, 3), **common)
        result = ledger.record_snapshot(response_text="Second framing.", trace=torch.ones(3, 3), **common)
        assert result.revision is None
        assert ledger.summary()["snapshots"] == 2
        assert ledger.summary()["revisions"] == 0


TESTS = [
    test_packet_is_deterministic_and_explicit,
    test_only_a_single_staged_chunk_is_a_framing_target,
    test_first_snapshot_then_matched_revision_persists,
    test_incompatible_traces_remain_snapshots_without_a_false_revision,
]


def main():
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
