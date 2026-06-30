"""Model-free tests for step/layer steer-map storage.

Run:
    .venv\\Scripts\\python.exe scripts\\steer_map_store_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.steer_map_store import SteerMapStore


def make_store():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    store = SteerMapStore(
        events_path=root / "steer_map_events.jsonl",
        summary_path=root / "steer_map_summary.json",
    )
    return tmp, store


def synthesis_record(steps=21, start_layer=12, end_layer=18, expert="Analytical"):
    return {
        "trigger": {"tensor_shape": [1, 1, 4096]},
        "delta": {"tensor_shape": [1, 1, 4096]},
        "metadata": {
            "reason": "optimizer",
            "steps": steps,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "expert": expert,
            "phenomenality": {"needless_interrupt": 0.22, "ambiguity": 0.02},
            "time_awareness": {},
        },
    }


def benchmark_payload(correct=True, accepted=True):
    return {
        "model": "test-model",
        "rows": [
            {
                "index": 0,
                "methods": {
                    "humble_synthesis": {
                        "correct": correct,
                        "result": {
                            "attempts": [
                                {
                                    "mode": "baseline",
                                    "round_index": 0,
                                    "accepted": accepted,
                                    "synthesis_records": [synthesis_record()],
                                }
                            ]
                        },
                    }
                },
            }
        ],
    }


def test_import_benchmark_result_stores_success_by_step_and_layer():
    tmp, store = make_store()
    try:
        assert store.import_benchmark_result(benchmark_payload(correct=True), source_path="run.json") == 1
        summary = store.aggregate()
        assert summary["event_count"] == 1
        group = summary["groups"][0]
        assert group["action"] == "synthesis_optimizer"
        assert group["layer_key"] == "12->18"
        assert group["step_bucket"] == "11-30"
        assert group["success"] == 1
        assert group["success_rate"] == 1.0
    finally:
        tmp.cleanup()


def test_failed_benchmark_result_counts_failure():
    tmp, store = make_store()
    try:
        assert store.import_benchmark_result(benchmark_payload(correct=False), source_path="run.json") == 1
        group = store.aggregate()["groups"][0]
        assert group["failure"] == 1
        assert group["success_rate"] == 0.0
    finally:
        tmp.cleanup()


def test_final_correct_rejected_attempt_is_not_counted_as_step_success():
    tmp, store = make_store()
    try:
        assert store.import_benchmark_result(benchmark_payload(correct=True, accepted=False), source_path="run.json") == 1
        group = store.aggregate()["groups"][0]
        assert group["final_correct"] == 1
        assert group["attempt_rejected"] == 1
        assert group["failure"] == 1
        assert group["success_rate"] == 0.0
        assert group["examples"][0]["success_label"] == "final_correct_attempt_unaccepted"
    finally:
        tmp.cleanup()


def test_interactive_trace_is_unlabeled_until_success_known():
    tmp, store = make_store()
    try:
        store.record_synthesis_record(
            synthesis_record(steps=2, start_layer=4, end_layer=5, expert="Creative"),
            source="interactive",
            method="interactive_phenomenality",
            final_correct=None,
        )
        group = store.aggregate()["groups"][0]
        assert group["unknown"] == 1
        assert group["labeled_n"] == 0
        assert group["success_rate"] is None
        assert group["step_bucket"] == "1-3"
        assert group["layer_key"] == "4->5"
    finally:
        tmp.cleanup()


def test_repeated_interactive_traces_are_not_deduped():
    tmp, store = make_store()
    try:
        record = synthesis_record(steps=2, start_layer=4, end_layer=5, expert="Creative")
        store.record_synthesis_record(record, source="interactive", method="interactive_phenomenality")
        store.record_synthesis_record(record, source="interactive", method="interactive_phenomenality")
        assert store.aggregate()["event_count"] == 2
    finally:
        tmp.cleanup()


def test_duplicate_benchmark_import_is_skipped():
    tmp, store = make_store()
    try:
        payload = benchmark_payload(correct=True)
        assert store.import_benchmark_result(payload, source_path="run.json") == 1
        assert store.import_benchmark_result(payload, source_path="run.json") == 0
        assert store.aggregate()["event_count"] == 1
    finally:
        tmp.cleanup()


def test_summary_file_is_written():
    tmp, store = make_store()
    try:
        store.import_benchmark_result(benchmark_payload(correct=True), source_path="run.json")
        summary = json.loads(store.summary_path.read_text(encoding="utf-8"))
        assert summary["event_count"] == 1
        assert summary["groups"][0]["layer_key"] == "12->18"
    finally:
        tmp.cleanup()


TESTS = [
    test_import_benchmark_result_stores_success_by_step_and_layer,
    test_failed_benchmark_result_counts_failure,
    test_final_correct_rejected_attempt_is_not_counted_as_step_success,
    test_interactive_trace_is_unlabeled_until_success_known,
    test_repeated_interactive_traces_are_not_deduped,
    test_duplicate_benchmark_import_is_skipped,
    test_summary_file_is_written,
]


def main():
    print("STEER MAP STORE TEST -- success by step/layer\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  Steer-map events aggregate by action, step bucket, layer span, and success label.")


if __name__ == "__main__":
    main()
