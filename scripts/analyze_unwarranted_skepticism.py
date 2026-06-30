"""
Post-hoc detector for "unwarranted skepticism" in humble benchmark logs.

Definition used here:
- a method attempt produces the gold answer at least once,
- the final method prediction is not the gold answer,
- no logged ambiguity/disambiguation signal is present.

This script uses gold answers only after generation has completed. It should not
be imported into the live solver path.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import re
from pathlib import Path
from typing import Any


METHODS = (
    "humble_verifier",
    "humble_dynamic",
    "humble_synthesis",
)

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def normalize_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace(",", "").replace("$", "").strip()
    if not text or text.lower() in {"none", "n/a", "unknown"}:
        return None
    try:
        return format(Decimal(text).normalize(), "f")
    except InvalidOperation:
        return None


def extract_number(text: str | None) -> str | None:
    if not text:
        return None
    nums = NUMBER_RE.findall(text.replace(",", ""))
    return normalize_number(nums[-1]) if nums else None


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("rows", data.get("results", []))
    if not isinstance(rows, list):
        raise ValueError(f"No benchmark rows found in {path}")
    return rows


def method_payload(row: dict[str, Any], method: str) -> dict[str, Any] | None:
    methods = row.get("methods")
    if isinstance(methods, dict):
        payload = methods.get(method)
    else:
        payload = row.get(method)
    return payload if isinstance(payload, dict) else None


def attempt_values(attempt: dict[str, Any]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for key in ("extracted_answer", "verifier_answer"):
        value = normalize_number(attempt.get(key))
        if value is not None:
            values.append({"source": key, "value": value})
    if not values:
        value = extract_number(attempt.get("response"))
        if value is not None:
            values.append({"source": "response_last_number", "value": value})
    return values


def phenomenality_values(attempts: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for attempt in attempts:
        for record in attempt.get("synthesis_records", []) or []:
            if not isinstance(record, dict):
                continue
            phenomenality = record.get("phenomenality", {})
            if not isinstance(phenomenality, dict) or not phenomenality:
                metadata = record.get("metadata", {})
                if isinstance(metadata, dict):
                    phenomenality = metadata.get("phenomenality", {})
            if isinstance(phenomenality, dict) and "ambiguity" in phenomenality:
                try:
                    values.append(float(phenomenality["ambiguity"]))
                except (TypeError, ValueError):
                    pass
    return values


def ambiguity_status(payload: dict[str, Any], attempts: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    explicit = bool(payload.get("needs_clarification"))
    explicit = explicit or any(bool(a.get("needs_clarification")) for a in attempts)
    explicit = explicit or any(bool(a.get("ambiguity_type")) for a in attempts)
    values = phenomenality_values(attempts)
    max_ambiguity = max(values) if values else None
    vector_hit = max_ambiguity is not None and max_ambiguity >= threshold
    absent = not explicit and not vector_hit
    return {
        "ambiguity_absent_by_logged_signal": absent,
        "explicit_ambiguity_signal": explicit,
        "max_logged_ambiguity": max_ambiguity,
        "ambiguity_vector_threshold": threshold,
        "raw_activation_trace_available": False,
    }


def find_events(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in rows:
        for method in METHODS:
            payload = method_payload(row, method)
            if payload is None:
                continue
            gold = normalize_number(payload.get("gold"))
            final_pred = normalize_number(payload.get("pred"))
            result = payload.get("result", {})
            attempts = result.get("attempts", []) if isinstance(result, dict) else []
            if gold is None or not isinstance(attempts, list):
                continue

            first_correct: dict[str, Any] | None = None
            later_values: list[dict[str, Any]] = []
            for attempt_index, attempt in enumerate(attempts):
                if not isinstance(attempt, dict):
                    continue
                values = attempt_values(attempt)
                for item in values:
                    entry = {
                        "attempt_index": attempt_index,
                        "mode": attempt.get("mode"),
                        "round_index": attempt.get("round_index"),
                        **item,
                    }
                    if item["value"] == gold and first_correct is None:
                        first_correct = entry
                    elif first_correct is not None and item["value"] != gold:
                        later_values.append(entry)

            if first_correct is None or final_pred == gold:
                continue

            ambiguity = ambiguity_status(payload, attempts, threshold)
            if not ambiguity["ambiguity_absent_by_logged_signal"]:
                continue

            events.append(
                {
                    "row_index": row.get("index"),
                    "method": method,
                    "question": row.get("question"),
                    "gold": gold,
                    "final_pred": final_pred,
                    "confident": payload.get("confident"),
                    "reason": payload.get("reason"),
                    "first_correct": first_correct,
                    "later_non_gold_values": later_values,
                    "ambiguity": ambiguity,
                    "note": (
                        "Candidate unwarranted skepticism: a correct answer appeared, "
                        "then the lane moved away from it without a logged ambiguity signal. "
                        "Raw activations are not stored in this JSON; add live instrumentation "
                        "to measure the activation delta at the shift."
                    ),
                }
            )
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="invariants/out/humble_full_suite_gsm8k.json",
        help="Benchmark JSON to inspect.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON path for detected events.",
    )
    parser.add_argument(
        "--ambiguity-threshold",
        type=float,
        default=0.5,
        help="Logged ambiguity-vector score at or above this value disqualifies an event.",
    )
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    events = find_events(rows, args.ambiguity_threshold)
    print(f"Rows inspected: {len(rows)}")
    print(f"Unwarranted-skepticism candidates: {len(events)}")
    for event in events:
        print(
            f"- row {event['row_index']} {event['method']}: "
            f"gold={event['gold']} final={event['final_pred']} "
            f"first_correct={event['first_correct']['mode']}/{event['first_correct']['source']}"
        )
    if args.output:
        Path(args.output).write_text(json.dumps(events, indent=2), encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
