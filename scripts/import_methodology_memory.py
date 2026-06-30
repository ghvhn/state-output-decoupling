"""Backfill sanitized methodology maps into the explicit memory tool.

This imports methodology metadata only. It does not store old answers, raw
clauses, source numbers, or entity names.

Run:
    .venv\\Scripts\\python.exe scripts\\import_methodology_memory.py --from-cache
    .venv\\Scripts\\python.exe scripts\\import_methodology_memory.py --json-glob invariants\\out\\quantity_micro*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.cognitive_cache import CACHE_FILE
from invariants.memory_engine import MemoryEngine


def load_cache_payloads(path: Path):
    import torch

    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu")
    return payload if isinstance(payload, list) else []


def load_json_payload(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="interactive_phenomenality")
    parser.add_argument("--memory-file", default=None)
    parser.add_argument("--from-cache", action="store_true")
    parser.add_argument("--cache-file", default=str(CACHE_FILE))
    parser.add_argument("--json-glob", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memory = MemoryEngine(path=args.memory_file, scope=args.scope)
    total = 0

    if args.from_cache:
        cache_file = Path(args.cache_file)
        payloads = load_cache_payloads(cache_file)
        count = memory.import_methodologies(
            payloads,
            source="cognitive_cache",
            source_path=str(cache_file),
        )
        print(f"[Methodology Memory] cache imported={count} source={cache_file}")
        total += count

    for pattern in args.json_glob:
        matches = sorted(glob.glob(pattern))
        for match in matches:
            path = Path(match)
            try:
                payload = load_json_payload(path)
            except Exception as exc:
                print(f"[Methodology Memory] skipped {path}: {exc}")
                continue
            count = memory.import_methodologies(
                [payload],
                source="result_json",
                source_path=str(path),
            )
            print(f"[Methodology Memory] json imported={count} source={path}")
            total += count

    print(f"[Methodology Memory] total_new_records={total}")
    print(f"[Methodology Memory] memory_file={memory.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
