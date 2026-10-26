"""Persisted, document-specific framing snapshots.

This module records how a reader framed one exact document chunk at different
times. A snapshot is an activation trace of a fixed packet containing the
chunk and the reader's reply. When the same document SHA and chunk are read
again, the ledger stores a temporal delta against the first snapshot.

The ledger is deliberately descriptive. It does not create probes, change
weights, or install a steer. A revision vector becomes experimental material
only after an operator chooses a separate, controlled use for it.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


DEFAULT_LEDGER_PATH = Path(__file__).parent / "out" / "document_framings.jsonl"
DEFAULT_ARTIFACT_DIR = Path(__file__).parent / "out" / "document_framings"
FORMAT_VERSION = 1
PACKET_VERSION = "document_framing_packet_v1"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def framing_packet(
    *,
    document_sha: str,
    chunk_index: int,
    chunk_text: str,
    response_text: str,
) -> str:
    """Return the fixed representation packet used for every snapshot.

    The packet makes the compared object explicit: this document chunk paired
    with the reader's response. It is analysis-only and never folded into the
    model's conversational prompt.
    """
    return (
        f"[{PACKET_VERSION}]\n"
        f"document_sha256={document_sha}\n"
        f"chunk_index={int(chunk_index)}\n"
        "[document_chunk]\n"
        f"{chunk_text}\n"
        "[/document_chunk]\n"
        "[reader_framing]\n"
        f"{response_text}\n"
        "[/reader_framing]"
    )


def staged_document_target(
    session: Optional[dict[str, Any]],
    tool_result: Optional[str],
) -> Optional[tuple[int, str]]:
    """Identify the one exact chunk represented by a staged tool result.

    Full-library injection has no single honest comparison target, so it is
    intentionally excluded. Model-requested document results may add a short
    causal preface before the normal staging block; accepting an exact suffix
    preserves the document engine's framing without guessing from text.
    """
    if not session or not tool_result:
        return None
    try:
        index = int(session.get("cursor", 0))
        chunks = session.get("chunks") or []
        chunk_text = chunks[index]
    except (IndexError, TypeError, ValueError):
        return None
    from invariants.document_engine import stage_chunk

    expected = stage_chunk(session, index=index)
    if expected and (tool_result == expected or tool_result.endswith(expected)):
        return index, str(chunk_text)
    return None


def _trace_tensor(trace: Any) -> torch.Tensor:
    tensor = torch.as_tensor(trace).detach().float().cpu()
    if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError("framing trace must have shape [layers, hidden_size]")
    if not torch.isfinite(tensor).all():
        raise ValueError("framing trace contains a non-finite value")
    return tensor.contiguous()


def _layerwise_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left_unit = left / left.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    right_unit = right / right.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return float((left_unit * right_unit).sum(dim=-1).mean().item())


@dataclass(frozen=True)
class SnapshotResult:
    snapshot: dict[str, Any]
    revision: Optional[dict[str, Any]] = None


class DocumentFramingLedger:
    """Append-only JSONL metadata with tensor artifacts stored beside it."""

    def __init__(
        self,
        path: str | Path = DEFAULT_LEDGER_PATH,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    ) -> None:
        self.path = Path(path)
        self.artifact_dir = Path(artifact_dir)
        self.records: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and record.get("format_version") == FORMAT_VERSION:
                self.records.append(record)

    def _append(self, record: dict[str, Any]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self.records.append(record)
        return record

    def _artifact_path(self, kind: str, record_id: str) -> Path:
        return self.artifact_dir / f"{kind}_{record_id}.pt"

    def _save_tensor_artifact(self, kind: str, record_id: str, payload: dict[str, Any]) -> str:
        path = self._artifact_path(kind, record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return os.path.relpath(path, self.path.parent)

    def _load_trace(self, record: dict[str, Any]) -> torch.Tensor:
        artifact = record.get("artifact")
        if not isinstance(artifact, str) or not artifact:
            raise ValueError("snapshot has no artifact")
        path = (self.path.parent / artifact).resolve()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "trace" not in payload:
            raise ValueError(f"invalid framing artifact: {path}")
        return _trace_tensor(payload["trace"])

    def snapshots(
        self,
        document_sha: Optional[str] = None,
        chunk_index: Optional[int] = None,
        reader_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return [
            record
            for record in self.records
            if record.get("kind") == "snapshot"
            and (document_sha is None or record.get("document_sha256") == document_sha)
            and (chunk_index is None or record.get("chunk_index") == int(chunk_index))
            and (reader_id is None or record.get("reader_id") == reader_id)
        ]

    def summary(self, limit: int = 8) -> dict[str, Any]:
        snapshots = [record for record in self.records if record.get("kind") == "snapshot"]
        revisions = [record for record in self.records if record.get("kind") == "revision"]
        latest = revisions[-max(0, int(limit)) :] if limit else []
        return {
            "snapshots": len(snapshots),
            "revisions": len(revisions),
            "latest_revisions": latest,
        }

    def record_snapshot(
        self,
        *,
        document_sha: str,
        source_name: str,
        chunk_index: int,
        chunk_count: int,
        chunk_text: str,
        response_text: str,
        trace: Any,
        reader_id: str = "main",
        turn_id: Optional[str] = None,
        context_flags: Optional[dict[str, bool]] = None,
    ) -> SnapshotResult:
        """Store one framing and, where comparable, its delta from the first.

        A comparison requires the same exact document content, chunk, reader,
        packet version, and trace shape. Incompatible traces are retained as
        observations but are not given a revision vector.
        """
        if not document_sha:
            raise ValueError("document_sha is required")
        if not source_name:
            raise ValueError("source_name is required")
        if not response_text or not response_text.strip():
            raise ValueError("response_text is required")

        trace_tensor = _trace_tensor(trace)
        now = _utc_timestamp()
        snapshot_id = _new_id()
        chunk_index = int(chunk_index)
        packet = framing_packet(
            document_sha=document_sha,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
            response_text=response_text,
        )
        artifact = self._save_tensor_artifact(
            "snapshot",
            snapshot_id,
            {
                "format_version": FORMAT_VERSION,
                "packet_version": PACKET_VERSION,
                "trace": trace_tensor,
                "packet": packet,
            },
        )
        snapshot = self._append(
            {
                "format_version": FORMAT_VERSION,
                "kind": "snapshot",
                "record_id": snapshot_id,
                "created_at": now,
                "packet_version": PACKET_VERSION,
                "reader_id": reader_id,
                "turn_id": turn_id,
                "document_sha256": document_sha,
                "source_name": source_name,
                "chunk_index": chunk_index,
                "chunk_count": int(chunk_count),
                "chunk_sha256": text_sha256(chunk_text),
                "response_sha256": text_sha256(response_text),
                "response_excerpt": " ".join(response_text.split())[:320],
                "packet_sha256": text_sha256(packet),
                "trace_shape": list(trace_tensor.shape),
                "artifact": artifact,
                "context_flags": dict(context_flags or {}),
            }
        )

        previous = self.snapshots(document_sha, chunk_index, reader_id)
        baseline = next((item for item in previous if item["record_id"] != snapshot_id), None)
        if baseline is None:
            return SnapshotResult(snapshot=snapshot)
        if baseline.get("packet_version") != PACKET_VERSION or baseline.get("trace_shape") != snapshot["trace_shape"]:
            return SnapshotResult(snapshot=snapshot)

        try:
            baseline_trace = self._load_trace(baseline)
        except (OSError, RuntimeError, ValueError, KeyError):
            return SnapshotResult(snapshot=snapshot)
        if baseline_trace.shape != trace_tensor.shape:
            return SnapshotResult(snapshot=snapshot)

        delta = trace_tensor - baseline_trace
        revision_id = _new_id()
        revision_artifact = self._save_tensor_artifact(
            "revision",
            revision_id,
            {
                "format_version": FORMAT_VERSION,
                "packet_version": PACKET_VERSION,
                "baseline_snapshot_id": baseline["record_id"],
                "current_snapshot_id": snapshot_id,
                "delta": delta,
            },
        )
        revision = self._append(
            {
                "format_version": FORMAT_VERSION,
                "kind": "revision",
                "record_id": revision_id,
                "created_at": now,
                "packet_version": PACKET_VERSION,
                "reader_id": reader_id,
                "document_sha256": document_sha,
                "source_name": source_name,
                "chunk_index": chunk_index,
                "baseline_snapshot_id": baseline["record_id"],
                "current_snapshot_id": snapshot_id,
                "baseline_response_sha256": baseline["response_sha256"],
                "current_response_sha256": snapshot["response_sha256"],
                "same_response": baseline["response_sha256"] == snapshot["response_sha256"],
                "trace_shape": list(delta.shape),
                "delta_norm": float(delta.norm().item()),
                "mean_layer_cosine": _layerwise_cosine(baseline_trace, trace_tensor),
                "artifact": revision_artifact,
                "baseline_context_flags": dict(baseline.get("context_flags") or {}),
                "context_flags": dict(context_flags or {}),
            }
        )
        return SnapshotResult(snapshot=snapshot, revision=revision)


def capture_document_framing(
    model: Any,
    *,
    document_sha: str,
    chunk_index: int,
    chunk_text: str,
    response_text: str,
) -> torch.Tensor:
    """Capture the final-token trace of the fixed framing packet.

    ClaimMap already owns the model-compatible hidden-state extraction path.
    Reusing it keeps tokenization and layer handling consistent with the
    project's existing paired-framing instrument.
    """
    from invariants.claimmap import _activation_trace

    packet = framing_packet(
        document_sha=document_sha,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        response_text=response_text,
    )
    return _activation_trace(model, packet)
