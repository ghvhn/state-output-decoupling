"""Routing stages remain distinguishable after steering-event persistence."""

import json
import tempfile
from pathlib import Path

from invariants.steer_map_store import SteerMapStore


def test_routing_stage_details_round_trip_without_changing_loop_index():
    workspace_output = Path(__file__).resolve().parents[1] / "invariants" / "out"
    with tempfile.TemporaryDirectory(prefix="routing_trace_store_", dir=workspace_output) as directory:
        events_path = Path(directory) / "events.jsonl"
        summary_path = Path(directory) / "summary.json"
        store = SteerMapStore(events_path=events_path, summary_path=summary_path)
        for stage_index, stop_reason in ((1, "continue"), (2, "depth_limit")):
            store.record_synthesis_record(
                {
                    "type": "routing_trace", "loop": 1,
                    "winner": "plus", "best_entropy": 3.0,
                    "entropies": {"plus": 3.0, "minus": 4.0},
                    "stage_index": stage_index, "routing_depth": 2,
                    "routing_layer_span": 2, "replay_start": 22,
                    "replay_end": 23, "stop_reason": stop_reason,
                },
                source="interactive",
            )

        persisted = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        restored = SteerMapStore(events_path=events_path, summary_path=summary_path)
        assert len(restored.events) == len(persisted) == 2
        for stage_index, (event, payload) in enumerate(zip(restored.events, persisted), 1):
            assert event.step_index == payload["step_index"] == 1
            assert event.step_bucket == "1-3"
            assert (event.start_layer, event.end_layer, event.layer_key) == (22, 23, "22->23")
            assert event.metrics == payload["metrics"] == {
                "best_entropy": 3.0, "entropies": {"plus": 3.0, "minus": 4.0},
                "stage_index": stage_index, "routing_depth": 2,
                "routing_layer_span": 2, "replay_start": 22, "replay_end": 23,
                "stop_reason": "continue" if stage_index == 1 else "depth_limit",
            }


def test_legacy_routing_trace_retains_unknown_layers_and_existing_metrics():
    workspace_output = Path(__file__).resolve().parents[1] / "invariants" / "out"
    with tempfile.TemporaryDirectory(prefix="routing_trace_store_", dir=workspace_output) as directory:
        store = SteerMapStore(
            events_path=Path(directory) / "events.jsonl",
            summary_path=Path(directory) / "summary.json",
        )
        event = store.record_synthesis_record(
            {"type": "routing_trace", "loop": 2, "winner": "legacy", "best_entropy": 3.0},
            source="interactive",
        )
        assert event.step_index == 2
        assert event.start_layer is None and event.end_layer is None
        assert event.layer_key == "unknown"
        assert event.metrics == {"best_entropy": 3.0, "entropies": {}}
