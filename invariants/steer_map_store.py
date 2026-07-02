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
    # Which kind of evidence labeled this event. "gold" = scored benchmark
    # outcome (final_correct + acceptance). "conversation" = the live, label-free
    # productivity read of the turn the steer happened in (sense_score vs a
    # tunable threshold) -- humans learn from conversations, and so does the
    # band, but the lanes stay separate and auditable.
    success_basis: str = "gold"
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

    # Channels the agentic engine accounts for per generation (see
    # agentic_engine._note_channel). Zero-filled on ingest so labeled runs carry
    # the unfired contrast a lift readout needs.
    KNOWN_STEER_CHANNELS = (
        "expert_branch",
        "synthesis_delta",
        "cache_delta",
        "organic_correction",
        "urgency",
    )

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

    @staticmethod
    def _resolve_outcome(attempt, final_correct, conversation_outcome):
        """One outcome policy for every event type. Gold (final_correct +
        acceptance) always wins; with no gold, a live conversation labels via
        its productivity read — deterministic given (score, threshold), both
        stored so the label stays auditable and re-derivable."""
        attempt = attempt or {}
        attempt_accepted = _coerce_bool(attempt.get("accepted"))
        final = _coerce_bool(final_correct)
        basis = "gold"
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
        if final is None and isinstance(conversation_outcome, dict):
            score = conversation_outcome.get("score")
            threshold = conversation_outcome.get("threshold", 0.0)
            if isinstance(score, (int, float)) and isinstance(threshold, (int, float)):
                basis = "conversation"
                success = bool(float(score) >= float(threshold))
                label = "conversation_productive" if success else "conversation_unproductive"
        return attempt_accepted, final, basis, success, label

    def _record_channel_stats(
        self,
        record: dict[str, Any],
        *,
        source: str,
        run_id: Optional[str],
        row_index: Optional[int],
        method: Optional[str],
        attempt: dict[str, Any],
        final_correct: Optional[bool],
        source_path: Optional[str],
        record_index: Optional[int],
        conversation_outcome: Optional[dict[str, Any]],
    ) -> Optional[list[SteerMapEvent]]:
        """One event per channel per generation, zero-filled for known channels
        that stayed silent, so a labeled run answers 'did outcomes differ when
        this channel fired vs when it did not' (channel_lift). Whether a
        channel should be ON is decided by this data, not by assertion."""
        attempt_accepted, final, basis, success, label = self._resolve_outcome(
            attempt, final_correct, conversation_outcome
        )
        channels_present = record.get("channels") if isinstance(record.get("channels"), dict) else {}
        flags = record.get("flags") if isinstance(record.get("flags"), dict) else {}
        any_fired = any(
            (stats or {}).get("applications") for stats in channels_present.values()
        )
        if success is None and not any_fired:
            return None  # nothing fired and nothing labeled: no evidence either way
        created: list[SteerMapEvent] = []
        for name in sorted(set(self.KNOWN_STEER_CHANNELS) | set(channels_present)):
            stats = channels_present.get(name) or {}
            applications = int(stats.get("applications", 0) or 0)
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
                        "channel": name,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                )
            if self.has_event_key(event_key):
                continue
            metrics: dict[str, Any] = {
                "fired": applications > 0,
                "applications": applications,
                "ratio_sum": float(stats.get("ratio_sum", 0.0) or 0.0),
                "ratio_max": float(stats.get("ratio_max", 0.0) or 0.0),
                "clipped": int(stats.get("clipped", 0) or 0),
            }
            if basis == "conversation":
                metrics["conversation_sense"] = float(conversation_outcome["score"])
                metrics["conversation_threshold"] = float(conversation_outcome.get("threshold", 0.0))
            created.append(
                self.append(
                    SteerMapEvent(
                        kind="steer_channel",
                        action=f"channel_{name}",
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
                        success_basis=basis,
                        metrics=metrics,
                        provenance={
                            "source_path": source_path,
                            "record_type": "steer_channel_stats",
                            "record_index": record_index,
                            "flags": flags,
                        },
                    )
                )
            )
        return created or None

    def record_layer_steer(
        self,
        channel: str,
        layer: int,
        alpha: float,
        conversation_outcome: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        source: str = "interactive",
    ) -> SteerMapEvent:
        """One SINGLE-layer steer application, labeled by the turn it shaped.
        This is the transfer-free evidence for where steering works: the push
        landed on exactly one layer, so the outcome attributes to that layer —
        no band confound, no borrowing from a different channel's geometry."""
        _, _, basis, success, label = self._resolve_outcome(None, None, conversation_outcome)
        event_metrics = {"alpha": float(alpha)}
        if metrics:
            event_metrics.update(metrics)
        if basis == "conversation":
            event_metrics["conversation_sense"] = float(conversation_outcome["score"])
            event_metrics["conversation_threshold"] = float(conversation_outcome.get("threshold", 0.0))
        return self.append(
            SteerMapEvent(
                kind="layer_steer",
                action=f"layer_steer_{channel}",
                source=source,
                start_layer=int(layer),
                end_layer=int(layer),
                event_success=success,
                success_label=label,
                success_basis=basis,
                metrics=event_metrics,
            )
        )

    def layer_steer_counts(self, channel: str) -> dict[int, int]:
        """How many single-layer steers each layer has received for `channel`
        — the rotation input for pick_sweep_layer (least-tested first)."""
        counts: dict[int, int] = {}
        action = f"layer_steer_{channel}"
        for event in self.events:
            if event.kind == "layer_steer" and event.action == action and event.start_layer is not None:
                counts[int(event.start_layer)] = counts.get(int(event.start_layer), 0) + 1
        return counts

    def channel_lift(self, basis: str = "any") -> list[dict[str, Any]]:
        """Per-channel fired-vs-unfired outcome comparison over labeled channel
        events. lift > 0 means generations where the channel fired ended better
        than ones where it did not — the honest 'should this be on' readout.
        Deterministic; `basis` restricts the evidence lane like suggest_band."""
        if basis not in {"any", "gold", "conversation"}:
            raise ValueError(f"basis must be 'any', 'gold', or 'conversation', got {basis!r}")
        rows: dict[str, dict[str, Any]] = {}
        for event in self.events:
            if event.kind != "steer_channel" or event.event_success is None:
                continue
            event_basis = getattr(event, "success_basis", "gold") or "gold"
            if basis != "any" and event_basis != basis:
                continue
            name = event.action[len("channel_"):] if event.action.startswith("channel_") else event.action
            fired = bool((event.metrics or {}).get("fired"))
            row = rows.setdefault(
                name,
                {"channel": name, "fired_n": 0, "fired_success": 0, "unfired_n": 0, "unfired_success": 0},
            )
            bucket = "fired" if fired else "unfired"
            row[f"{bucket}_n"] += 1
            if event.event_success:
                row[f"{bucket}_success"] += 1
        out: list[dict[str, Any]] = []
        for row in sorted(rows.values(), key=lambda r: r["channel"]):
            fired_rate = row["fired_success"] / row["fired_n"] if row["fired_n"] else None
            unfired_rate = row["unfired_success"] / row["unfired_n"] if row["unfired_n"] else None
            row["fired_rate"] = fired_rate
            row["unfired_rate"] = unfired_rate
            row["lift"] = (
                round(fired_rate - unfired_rate, 4)
                if fired_rate is not None and unfired_rate is not None
                else None
            )
            row["basis"] = basis
            out.append(row)
        return out

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
        conversation_outcome: Optional[dict[str, Any]] = None,
    ):
        if not isinstance(record, dict):
            return None
        attempt = attempt or {}
        if record.get("type") == "steer_channel_stats":
            return self._record_channel_stats(
                record,
                source=source,
                run_id=run_id,
                row_index=row_index,
                method=method,
                attempt=attempt,
                final_correct=final_correct,
                source_path=source_path,
                record_index=record_index,
                conversation_outcome=conversation_outcome,
            )
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

        attempt_accepted, final, basis, success, label = self._resolve_outcome(
            attempt, final_correct, conversation_outcome
        )

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

        if basis == "conversation":
            metrics["conversation_sense"] = float(conversation_outcome["score"])
            metrics["conversation_threshold"] = float(conversation_outcome.get("threshold", 0.0))
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
            success_basis=basis,
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

    def suggest_band(
        self, n_layers: int, min_events: int = 8, basis: str = "any", evidence: str = "any"
    ) -> Optional[dict[str, Any]]:
        """Data-informed steer band: derive the [lo, hi) depth-fraction window
        deterministically from acceptance-aware per-layer outcomes.

        `basis` picks the evidence lane: "gold" (scored benchmark outcomes),
        "conversation" (live label-free productivity reads — humans learn from
        conversations, so can the band), or "any" (both, with the composition
        reported so the mix is never hidden).

        `evidence` picks the event kind, because per-layer outcomes are NOT
        interchangeable across intervention types: "synthesis" = where
        synthesis/cache deltas landed well (a different channel's geometry);
        "layer_steer" = single-layer steer applications — the transfer-free
        basis for the steering band itself; "any" = both, mix reported.

        Procedure (no RNG, no smoothing — same events, same band):
        1. Attribute each labeled event to the layer its delta landed on
           (end_layer, else start_layer).
        2. A layer is eligible when it has >= min_events labeled events AND its
           success rate >= the overall labeled success rate.
        3. The band spans [min eligible, max eligible + 1) as depth fractions.

        Returns None when no layer clears the evidence bar — the caller keeps
        its explicit prior instead of steering on vibes."""
        n_layers = int(n_layers)
        if n_layers <= 0:
            return None
        if basis not in {"any", "gold", "conversation"}:
            raise ValueError(f"basis must be 'any', 'gold', or 'conversation', got {basis!r}")
        if evidence not in {"any", "synthesis", "layer_steer"}:
            raise ValueError(f"evidence must be 'any', 'synthesis', or 'layer_steer', got {evidence!r}")
        per_layer: dict[int, dict[str, int]] = {}
        labeled = 0
        successes = 0
        labeled_by_basis: dict[str, int] = {}
        labeled_by_evidence: dict[str, int] = {}
        for event in self.events:
            if event.event_success is None:
                continue
            event_basis = getattr(event, "success_basis", "gold") or "gold"
            if basis != "any" and event_basis != basis:
                continue
            event_evidence = "layer_steer" if event.kind == "layer_steer" else "synthesis"
            if evidence != "any" and event_evidence != evidence:
                continue
            layer = event.end_layer if event.end_layer is not None else event.start_layer
            if not isinstance(layer, int) or not (0 <= layer < n_layers):
                continue
            row = per_layer.setdefault(layer, {"n": 0, "success": 0})
            row["n"] += 1
            labeled += 1
            labeled_by_basis[event_basis] = labeled_by_basis.get(event_basis, 0) + 1
            labeled_by_evidence[event_evidence] = labeled_by_evidence.get(event_evidence, 0) + 1
            if event.event_success:
                row["success"] += 1
                successes += 1
        if not labeled:
            return None
        overall = successes / labeled
        eligible = sorted(
            layer
            for layer, row in per_layer.items()
            if row["n"] >= max(1, int(min_events)) and (row["success"] / row["n"]) >= overall
        )
        if not eligible:
            return None
        lo = eligible[0] / n_layers
        hi = (eligible[-1] + 1) / n_layers
        if not (0.0 <= lo < hi <= 1.0):
            return None
        return {
            "lo": lo,
            "hi": hi,
            "n_layers": n_layers,
            "eligible_layers": eligible,
            "overall_success_rate": overall,
            "labeled_events": labeled,
            "labeled_by_basis": labeled_by_basis,
            "labeled_by_evidence": labeled_by_evidence,
            "basis": basis,
            "evidence": evidence,
            "min_events": int(min_events),
            "per_layer": {
                layer: {"n": row["n"], "success_rate": row["success"] / row["n"]}
                for layer, row in sorted(per_layer.items())
            },
        }

    def aggregate(self) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for event in self.events:
            event_basis = getattr(event, "success_basis", "gold") or "gold"
            key = "|".join(
                [
                    event.kind,
                    event.action,
                    event.layer_key,
                    event.step_bucket,
                    str(event.expert or event.target_vector or ""),
                    event_basis,
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
                    "success_basis": event_basis,
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
