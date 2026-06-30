from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


OUT_DIR = Path(__file__).parent / "out"
DEFAULT_EVENTS_FILE = OUT_DIR / "steer_map_events.jsonl"
DEFAULT_SUMMARY_FILE = OUT_DIR / "steer_map_summary.json"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _event_id() -> str:
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
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
            "note": "tensor summarized; steer-map events store metadata, not raw tensors",
        }
    return repr(value)


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
        return None
    return bool(value)


def _bucket_step(step: Optional[int]) -> str:
    if step is None:
        return "unknown"
    if step <= 0:
        return "0"
    if step <= 3:
        return "1-3"
    if step <= 10:
        return "4-10"
    if step <= 30:
        return "11-30"
    if step <= 60:
        return "31-60"
    return "61+"


def _layer_key(start_layer: Optional[int], end_layer: Optional[int]) -> str:
    if start_layer is None and end_layer is None:
        return "unknown"
    if start_layer == end_layer or start_layer is None:
        return str(end_layer)
    if end_layer is None:
        return str(start_layer)
    return f"{start_layer}->{end_layer}"


@dataclass
class SteerMapEvent:
    kind: str
    action: str
    source: str = "unknown"
    timestamp: str = field(default_factory=utc_timestamp)
    event_id: str = field(default_factory=_event_id)
    event_key: Optional[str] = None
    run_id: Optional[str] = None
    row_index: Optional[int] = None
    method: Optional[str] = None
    attempt_mode: Optional[str] = None
    attempt_round: Optional[int] = None
    attempt_accepted: Optional[bool] = None
    final_correct: Optional[bool] = None
    event_success: Optional[bool] = None
    success_label: str = "unknown"
    step_index: Optional[int] = None
    step_bucket: str = "unknown"
    start_layer: Optional[int] = None
    end_layer: Optional[int] = None
    layer_key: str = "unknown"
    expert: Optional[str] = None
    trigger_vector: Optional[str] = None
    target_vector: Optional[str] = None
    avoid_vector: Optional[str] = None
    sensor_scores: dict[str, Any] = field(default_factory=dict)
    map_relations: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


class SteerMapStore:
    """Append-only steering outcome store plus step/layer aggregation."""

    def __init__(
        self,
        events_path: Optional[Path | str] = None,
        summary_path: Optional[Path | str] = None,
    ):
        env_path = os.environ.get("STEER_MAP_EVENTS_FILE")
        self.events_path = Path(events_path or env_path or DEFAULT_EVENTS_FILE)
        self.summary_path = Path(summary_path or DEFAULT_SUMMARY_FILE)
        self.events: list[SteerMapEvent] = []
        self.load()

    def load(self) -> None:
        self.events = []
        if not self.events_path.exists():
            return
        with self.events_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    self.events.append(self._event_from_dict(payload))

    @staticmethod
    def _event_from_dict(payload: dict[str, Any]) -> SteerMapEvent:
        fields = {field.name for field in SteerMapEvent.__dataclass_fields__.values()}
        data = {key: payload[key] for key in fields if key in payload}
        data.setdefault("kind", "unknown")
        data.setdefault("action", "unknown")
        event = SteerMapEvent(**data)
        if event.success_label == "final_correct_attempt_unaccepted":
            event.event_success = False
        return event

    def append(self, event: SteerMapEvent) -> SteerMapEvent:
        event.step_bucket = _bucket_step(event.step_index)
        event.layer_key = _layer_key(event.start_layer, event.end_layer)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
        self.events.append(event)
        self.write_summary()
        return event

    def has_event_key(self, event_key: Optional[str]) -> bool:
        if not event_key:
            return False
        return any(event.event_key == event_key for event in self.events)

    def record_synthesis_record(
        self,
        record: dict[str, Any],
        *,
        source: str,
        run_id: Optional[str] = None,
        row_index: Optional[int] = None,
        method: Optional[str] = None,
        attempt: Optional[dict[str, Any]] = None,
        final_correct: Optional[bool] = None,
        source_path: Optional[str] = None,
        record_index: Optional[int] = None,
    ) -> Optional[SteerMapEvent]:
        if not isinstance(record, dict):
            return None
        attempt = attempt or {}
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        if record.get("type") == "routing_trace":
            action = f"route_{record.get('winner') or 'unknown'}"
            step_index = record.get("loop")
            expert = record.get("winner")
            metrics = {
                "best_entropy": record.get("best_entropy"),
                "entropies": record.get("entropies", {}),
            }
            start_layer = None
            end_layer = None
        elif metadata:
            action = f"synthesis_{metadata.get('reason') or 'unknown'}"
            step_index = metadata.get("steps")
            expert = metadata.get("expert")
            metrics = {
                "reason": metadata.get("reason"),
                "steps": metadata.get("steps"),
                "phenomenality": metadata.get("phenomenality", {}),
                "time_awareness": metadata.get("time_awareness", {}),
            }
            start_layer = metadata.get("start_layer")
            end_layer = metadata.get("end_layer")
        else:
            return None

        attempt_accepted = _coerce_bool(attempt.get("accepted"))
        final = _coerce_bool(final_correct)
        if final is None:
            success = None
            label = "unlabeled"
        elif final and (attempt_accepted is not False):
            success = True
            label = "final_correct"
        elif final and attempt_accepted is False:
            success = False
            label = "final_correct_attempt_unaccepted"
        else:
            success = False
            label = "final_wrong"

        event_key = None
        if source_path or row_index is not None or record_index is not None:
            event_key = json.dumps(
                {
                    "source_path": source_path,
                    "row_index": row_index,
                    "method": method,
                    "attempt_mode": attempt.get("mode"),
                    "attempt_round": attempt.get("round_index"),
                    "record_index": record_index,
                    "action": action,
                    "step_index": step_index,
                    "start_layer": start_layer,
                    "end_layer": end_layer,
                    "expert": expert,
                },
                sort_keys=True,
                ensure_ascii=True,
            )
        if self.has_event_key(event_key):
            return None

        event = SteerMapEvent(
            kind="synthesis_record",
            action=action,
            source=source,
            event_key=event_key,
            run_id=run_id,
            row_index=row_index,
            method=method,
            attempt_mode=attempt.get("mode"),
            attempt_round=attempt.get("round_index"),
            attempt_accepted=attempt_accepted,
            final_correct=final,
            event_success=success,
            success_label=label,
            step_index=int(step_index) if isinstance(step_index, int) else step_index,
            start_layer=start_layer,
            end_layer=end_layer,
            expert=expert,
            sensor_scores=(metadata or {}).get("phenomenality", {}),
            metrics=metrics,
            provenance={
                "source_path": source_path,
                "record_type": record.get("type"),
                "record_index": record_index,
                "metadata": metadata,
            },
        )
        return self.append(event)

    def record_self_concept_decision(
        self,
        decision: dict[str, Any],
        *,
        source: str = "interactive",
        final_correct: Optional[bool] = None,
        source_path: Optional[str] = None,
    ) -> SteerMapEvent:
        action = str(decision.get("action") or "unknown")
        strength = decision.get("strength", 0.0)
        final = _coerce_bool(final_correct)
        success_label = "unlabeled" if final is None else ("final_correct" if final else "final_wrong")
        event = SteerMapEvent(
            kind="self_concept_decision",
            action=action,
            source=source,
            final_correct=final,
            event_success=final,
            success_label=success_label,
            trigger_vector=decision.get("trigger_vector"),
            target_vector=decision.get("target_vector"),
            avoid_vector=decision.get("avoid_vector"),
            sensor_scores=(decision.get("evidence") or {}).get("scores", {}),
            map_relations=decision.get("map_relations", {}),
            metrics={
                "strength": strength,
                "allowed": decision.get("allowed"),
                "intervention_type": decision.get("intervention_type"),
            },
            provenance={
                "source_path": source_path,
                "decision": _json_safe(decision),
            },
        )
        return self.append(event)

    def import_benchmark_result(self, payload: dict[str, Any], source_path: Optional[str] = None) -> int:
        imported = 0
        run_id = str(payload.get("output") or payload.get("created_at") or payload.get("model") or source_path or "benchmark")
        for row in payload.get("rows", []) or []:
            row_index = row.get("index")
            methods = row.get("methods") or {}
            for method, method_result in methods.items():
                if not isinstance(method_result, dict):
                    continue
                final_correct = method_result.get("correct")
                result = method_result.get("result") or {}
                attempts = result.get("attempts") or method_result.get("attempts") or []
                for attempt in attempts:
                    for record_index, record in enumerate(attempt.get("synthesis_records", []) or []):
                        if self.record_synthesis_record(
                            record,
                            source="benchmark_result",
                            run_id=run_id,
                            row_index=row_index,
                            method=method,
                            attempt=attempt,
                            final_correct=final_correct,
                            source_path=source_path,
                            record_index=record_index,
                        ):
                            imported += 1
        return imported

    def aggregate(self) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for event in self.events:
            key = "|".join(
                [
                    event.kind,
                    event.action,
                    event.layer_key,
                    event.step_bucket,
                    str(event.expert or event.target_vector or ""),
                ]
            )
            row = groups.setdefault(
                key,
                {
                    "kind": event.kind,
                    "action": event.action,
                    "layer_key": event.layer_key,
                    "step_bucket": event.step_bucket,
                    "expert_or_target": event.expert or event.target_vector,
                    "n": 0,
                    "labeled_n": 0,
                    "success": 0,
                    "failure": 0,
                    "unknown": 0,
                    "final_correct": 0,
                    "final_wrong": 0,
                    "attempt_accepted": 0,
                    "attempt_rejected": 0,
                    "success_rate": None,
                    "examples": [],
                },
            )
            row["n"] += 1
            if event.final_correct is True:
                row["final_correct"] += 1
            elif event.final_correct is False:
                row["final_wrong"] += 1
            if event.attempt_accepted is True:
                row["attempt_accepted"] += 1
            elif event.attempt_accepted is False:
                row["attempt_rejected"] += 1
            if event.event_success is True:
                row["labeled_n"] += 1
                row["success"] += 1
            elif event.event_success is False:
                row["labeled_n"] += 1
                row["failure"] += 1
            else:
                row["unknown"] += 1
            if len(row["examples"]) < 3:
                row["examples"].append(
                    {
                        "source": event.source,
                        "row_index": event.row_index,
                        "method": event.method,
                        "success_label": event.success_label,
                        "event_id": event.event_id,
                    }
                )
        for row in groups.values():
            if row["labeled_n"]:
                row["success_rate"] = row["success"] / row["labeled_n"]
        return {
            "created_at": utc_timestamp(),
            "success_basis": "final_correct and attempt_accepted when attempt acceptance is known",
            "events_path": str(self.events_path),
            "event_count": len(self.events),
            "groups": sorted(
                groups.values(),
                key=lambda r: (-(r["labeled_n"] or 0), -(r["success_rate"] or -1), r["action"]),
            ),
        }

    def write_summary(self) -> dict[str, Any]:
        summary = self.aggregate()
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        return summary
