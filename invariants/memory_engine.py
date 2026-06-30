from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_MEMORY_FILE = Path(__file__).parent / "out" / "interactive_memory.jsonl"

METHODOLOGY_KEYS = {
    "kind",
    "methodology",
    "structural_features",
    "clause_map_status",
    "roles_declared",
    "privacy",
}

RAW_PRIVACY_KEYS = {
    "raw_clauses_saved",
    "source_numbers_saved",
    "entity_names_saved",
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        shape = getattr(value, "shape", None)
        return {
            "tensor_shape": list(shape) if shape is not None else None,
            "tensor_dtype": str(getattr(value, "dtype", "unknown")),
            "note": "tensor summarized; raw tensors belong in artifact files",
        }
    return repr(value)


def _unique_tags(tags: Optional[list[str]]) -> list[str]:
    seen = set()
    out = []
    for tag in tags or []:
        t = str(tag).strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def sanitize_methodology_payload(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if "kind" not in payload or "methodology" not in payload:
        return None

    sanitized = {key: _json_safe(payload[key]) for key in METHODOLOGY_KEYS if key in payload}
    privacy = sanitized.get("privacy")
    if not isinstance(privacy, dict):
        privacy = {}
    for key in RAW_PRIVACY_KEYS:
        if privacy.get(key) is True:
            return None
        privacy.setdefault(key, False)
    privacy.setdefault("tier", "reusable_sanitized")
    sanitized["privacy"] = privacy
    return sanitized


def methodology_key(methodology: dict[str, Any]) -> str:
    safe = sanitize_methodology_payload(methodology) or {}
    return json.dumps(
        {
            "kind": safe.get("kind"),
            "methodology": safe.get("methodology"),
            "structural_features": safe.get("structural_features", []),
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def methodology_text(methodology: dict[str, Any]) -> str:
    kind = str(methodology.get("kind") or "methodology")
    body = str(methodology.get("methodology") or "").strip()
    lines = [f"{kind}: {body}"]
    features = methodology.get("structural_features") or []
    if features:
        lines.append("features: " + ", ".join(str(f) for f in features))
    status = methodology.get("clause_map_status")
    if status:
        lines.append(f"clause_map_status: {status}")
    roles = methodology.get("roles_declared") or []
    if roles:
        lines.append("roles_declared: " + ", ".join(str(r) for r in roles))
    return "\n".join(lines)


def iter_sanitized_methodologies(payload: Any, path: str = "root"):
    if isinstance(payload, dict):
        for key in (
            "clause_methodology",
            "solver_clause_methodology",
            "verifier_clause_methodology",
        ):
            methodology = sanitize_methodology_payload(payload.get(key))
            if methodology is not None:
                yield methodology, f"{path}.{key}"
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            for methodology, methodology_path in iter_sanitized_methodologies(metadata, f"{path}.metadata"):
                yield methodology, methodology_path
        for key, value in payload.items():
            if key == "metadata" or key.endswith("methodology") or key == "question":
                continue
            if isinstance(value, (dict, list)):
                for methodology, methodology_path in iter_sanitized_methodologies(value, f"{path}.{key}"):
                    yield methodology, methodology_path
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            for methodology, methodology_path in iter_sanitized_methodologies(item, f"{path}[{idx}]"):
                yield methodology, methodology_path


@dataclass
class MemoryRecord:
    kind: str
    scope: str = "default"
    role: Optional[str] = None
    text: str = ""
    timestamp: str = field(default_factory=utc_timestamp)
    record_id: str = field(default_factory=_new_id)
    session_id: str = ""
    turn_id: Optional[str] = None
    parent_turn_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = _unique_tags(payload.get("tags"))
        return _json_safe(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        known = {
            "kind",
            "scope",
            "role",
            "text",
            "timestamp",
            "record_id",
            "session_id",
            "turn_id",
            "parent_turn_id",
            "tags",
            "provenance",
            "metrics",
            "artifact_path",
        }
        data = {k: payload[k] for k in known if k in payload}
        data.setdefault("kind", "event")
        data.setdefault("tags", [])
        data.setdefault("provenance", {})
        data.setdefault("metrics", {})
        return cls(**data)


class MemoryEngine:
    """Append-only memory log plus explicit retrieval tools.

    The engine keeps three ideas separate:
    - text records available to tools,
    - persistent provenance records for later analysis,
    - activation or tensor artifacts saved outside JSONL and referenced by path.

    It does not automatically build prompts or inject prior turns. Callers must
    ask for memory records explicitly and decide how to use the tool result.
    """

    def __init__(
        self,
        path: Optional[Path | str] = None,
        scope: str = "default",
        session_id: Optional[str] = None,
        load_existing: bool = True,
        include_existing_in_session_view: bool = False,
    ):
        env_path = os.environ.get("INTERACTIVE_MEMORY_FILE")
        self.path = Path(path or env_path or DEFAULT_MEMORY_FILE)
        self.scope = scope
        self.session_id = session_id or _new_id()
        self.records: list[MemoryRecord] = []
        if load_existing:
            self.load()
        self.session_start_index = 0 if include_existing_in_session_view else len(self.records)
        self._turn_counter = 0

    def load(self) -> None:
        self.records = []
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    if isinstance(raw, dict):
                        self.records.append(MemoryRecord.from_dict(raw))
                except json.JSONDecodeError:
                    continue

    def append(self, record: MemoryRecord | dict[str, Any]) -> MemoryRecord:
        if isinstance(record, dict):
            record = MemoryRecord.from_dict(record)
        if not record.session_id:
            record.session_id = self.session_id
        if not record.scope:
            record.scope = self.scope
        record.tags = _unique_tags(record.tags)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
        self.records.append(record)
        return record

    def append_turn(
        self,
        role: str,
        text: str,
        *,
        scope: Optional[str] = None,
        tags: Optional[list[str]] = None,
        provenance: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> MemoryRecord:
        role = role.strip().lower()
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported memory role: {role}")
        self._turn_counter += 1
        return self.append(
            MemoryRecord(
                kind="turn",
                scope=scope or self.scope,
                role=role,
                text=text,
                session_id=self.session_id,
                turn_id=f"{self.session_id}:{self._turn_counter:04d}",
                tags=_unique_tags(["external_io", "conversation_trace"] + (tags or [])),
                provenance=provenance or {},
                metrics=metrics or {},
            )
        )

    def append_event(
        self,
        name: str,
        *,
        text: str = "",
        scope: Optional[str] = None,
        tags: Optional[list[str]] = None,
        provenance: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> MemoryRecord:
        return self.append(
            MemoryRecord(
                kind="event",
                scope=scope or self.scope,
                text=text or name,
                session_id=self.session_id,
                tags=_unique_tags([name] + (tags or [])),
                provenance=provenance or {},
                metrics=metrics or {},
            )
        )

    def append_internal_trace(
        self,
        name: str,
        *,
        text: str = "",
        scope: Optional[str] = None,
        tags: Optional[list[str]] = None,
        provenance: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> MemoryRecord:
        return self.append(
            MemoryRecord(
                kind="internal_trace",
                scope=scope or self.scope,
                text=text or name,
                session_id=self.session_id,
                tags=_unique_tags(["internal", "trace", name] + (tags or [])),
                provenance=provenance or {},
                metrics=metrics or {},
            )
        )

    def append_activation_trace(
        self,
        artifact_path: Path | str,
        *,
        text: str = "",
        scope: Optional[str] = None,
        tags: Optional[list[str]] = None,
        provenance: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> MemoryRecord:
        path = Path(artifact_path)
        try:
            artifact = str(path.resolve())
        except OSError:
            artifact = str(path)
        return self.append(
            MemoryRecord(
                kind="activation_trace",
                scope=scope or self.scope,
                text=text or "activation trace artifact",
                session_id=self.session_id,
                tags=_unique_tags(["activation_trace"] + (tags or [])),
                provenance=provenance or {},
                metrics=metrics or {},
                artifact_path=artifact,
            )
        )

    def append_methodology(
        self,
        methodology: dict[str, Any],
        *,
        source: str = "unknown",
        source_path: Optional[str] = None,
        scope: Optional[str] = None,
        tags: Optional[list[str]] = None,
        provenance: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
    ) -> Optional[MemoryRecord]:
        sanitized = sanitize_methodology_payload(methodology)
        if sanitized is None:
            return None

        key = methodology_key(sanitized)
        existing = {
            r.provenance.get("methodology_key")
            for r in self.records
            if r.kind == "methodology"
        }
        if key in existing:
            return None

        prov = {
            "source": source,
            "source_path": source_path,
            "methodology_key": key,
            "privacy": sanitized.get("privacy", {}),
        }
        if provenance:
            for k, v in provenance.items():
                if k not in {"question", "answer", "raw_clauses", "source_numbers"}:
                    prov[k] = _json_safe(v)

        return self.append(
            MemoryRecord(
                kind="methodology",
                scope=scope or self.scope,
                text=methodology_text(sanitized),
                session_id=self.session_id,
                tags=_unique_tags(
                    [
                        "methodology",
                        "sanitized",
                        "clause_map",
                        str(sanitized.get("kind") or "unknown_kind"),
                    ]
                    + (tags or [])
                ),
                provenance=prov,
                metrics=metrics or {},
            )
        )

    def import_methodologies(
        self,
        payloads: list[Any],
        *,
        source: str,
        source_path: Optional[str] = None,
    ) -> int:
        imported = 0
        for idx, payload in enumerate(payloads):
            for methodology, methodology_path in iter_sanitized_methodologies(payload, f"payload[{idx}]"):
                record = self.append_methodology(
                    methodology,
                    source=source,
                    source_path=source_path,
                    provenance={"methodology_path": methodology_path},
                )
                if record is not None:
                    imported += 1
        return imported

    def mark_session_boundary(self, reason: str = "operator_request") -> None:
        self.session_start_index = len(self.records)
        self.append_event(
            "memory_session_boundary",
            tags=["memory_control"],
            provenance={"reason": reason, "record_count": self.session_start_index},
        )

    def recent_turns(
        self,
        *,
        max_turns: int = 8,
        max_chars: int = 6000,
        scope: Optional[str] = None,
        current_session_only: bool = True,
    ) -> list[MemoryRecord]:
        source = self.records[self.session_start_index :] if current_session_only else self.records
        records = [
            r
            for r in source
            if r.kind == "turn" and (scope is None or r.scope == scope)
        ]
        max_messages = max(0, max_turns) * 2
        tail = records[-max_messages:] if max_messages else []
        if max_chars <= 0:
            return []

        kept: list[MemoryRecord] = []
        total = 0
        for record in reversed(tail):
            text = record.text or ""
            if total + len(text) > max_chars:
                if not kept:
                    kept.append(replace(record, text=text[-max_chars:]))
                break
            kept.append(record)
            total += len(text)
        return list(reversed(kept))

    def search(
        self,
        query: str,
        *,
        max_records: int = 5,
        scope: Optional[str] = None,
        kinds: Optional[list[str]] = None,
        current_session_only: bool = False,
    ) -> list[MemoryRecord]:
        terms = {t for t in query.lower().replace("_", " ").split() if t}
        if not terms:
            return []

        allowed_kinds = set(kinds or [])
        source = self.records[self.session_start_index :] if current_session_only else self.records
        scored: list[tuple[float, int, MemoryRecord]] = []
        for idx, record in enumerate(source):
            if scope is not None and record.scope != scope:
                continue
            if allowed_kinds and record.kind not in allowed_kinds:
                continue
            haystack = " ".join(
                [
                    record.kind,
                    record.role or "",
                    record.text or "",
                    " ".join(record.tags),
                    json.dumps(record.provenance, ensure_ascii=True, sort_keys=True),
                    json.dumps(record.metrics, ensure_ascii=True, sort_keys=True),
                ]
            ).lower()
            hits = sum(1 for term in terms if term in haystack)
            if hits <= 0:
                continue
            recency = idx / max(1, len(source))
            score = hits + (0.15 * recency)
            scored.append((score, idx, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record for _, _, record in scored[:max_records]]

    def format_tool_result(self, records: list[MemoryRecord], *, max_chars: int = 3000) -> str:
        if not records:
            return "[Memory Tool] No matching records."
        lines = ["[Memory Tool Result]"]
        total = len(lines[0])
        for record in records:
            tags = ",".join(record.tags[:5])
            role = f"/{record.role}" if record.role else ""
            snippet = " ".join((record.text or "").split())
            if len(snippet) > 360:
                snippet = snippet[:357] + "..."
            line = (
                f"- {record.timestamp} {record.kind}{role} "
                f"scope={record.scope} tags={tags}: {snippet}"
            )
            if total + len(line) + 1 > max_chars:
                lines.append("- [truncated]")
                break
            lines.append(line)
            total += len(line) + 1
        return "\n".join(lines)

    def format_recent(self, max_turns: int = 4) -> str:
        records = self.recent_turns(max_turns=max_turns, max_chars=4000, scope=self.scope)
        if not records:
            return "[Memory] No turns in this shell session."
        lines = []
        for record in records:
            snippet = " ".join((record.text or "").split())
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            lines.append(f"{record.role}: {snippet}")
        return "\n".join(lines)

    def status(self) -> dict[str, Any]:
        session_records = self.records[self.session_start_index :]
        return {
            "path": str(self.path),
            "scope": self.scope,
            "session_id": self.session_id,
            "total_records": len(self.records),
            "session_records": len(session_records),
            "session_turns": sum(1 for r in session_records if r.kind == "turn"),
        }
