"""Import benchmark synthesis traces into the steer-map store.

The store aggregates steering outcomes by action, step bucket, layer span, and
expert/target. Interactive traces are unlabeled; benchmark imports attach the
method-level correctness label while preserving attempt-level acceptance.

Run:
    .venv\\Scripts\\python.exe scripts\\import_steer_maps.py --json-glob invariants\\out\\humble_full_suite_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.steer_map_store import SteerMapStore


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-file", default=None)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--json-glob", action="append", default=[])
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    args = parse_args()
    store = SteerMapStore(events_path=args.events_file, summary_path=args.summary_file)
    total = 0
    for pattern in args.json_glob:
        matches = sorted(glob.glob(pattern))
        for match in matches:
            path = Path(match)
            try:
                payload = load_json(path)
            except Exception as exc:
                print(f"[SteerMap] skipped {path}: {exc}")
                continue
            count = store.import_benchmark_result(payload, source_path=str(path))
            print(f"[SteerMap] imported={count} source={path}")
            total += count
    summary = store.write_summary()
    print(f"[SteerMap] total_new_events={total}")
    print(f"[SteerMap] event_count={summary['event_count']}")
    print(f"[SteerMap] events_file={store.events_path}")
    print(f"[SteerMap] summary_file={store.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
