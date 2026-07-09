"""Join the cognitive cache with the session log's synthesis traces.

The two tables:
  - invariants/data/cognitive_cache.pt        -- append-ordered store entries,
    metadata = (expert, start_layer, end_layer, steps, ...), NO timestamps
    (store() now stamps `stored_at`; historical entries predate that).
  - invariants/out/interactive_memory.jsonl   -- synthesis_trace records,
    timestamped, but only for generations that passed a synthesis_recorder
    (main chat turns). Pings/spawn/solve generations store WITHOUT a trace.

Both are chronological, so an order-preserving alignment (LCS on the
(expert, lo, hi, steps) fingerprint) is the honest join: aligned entries
inherit their trace's timestamp exactly; unaligned entries are bounded by
their nearest aligned neighbors. Output is a sidecar JSON -- the 1.2 GB
cache file itself is never rewritten.

Usage:  python scripts/cache_trace_join.py [--window ISO_FROM ISO_TO]
        (window prints entries whose time bounds intersect it)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "invariants" / "data" / "cognitive_cache.pt"
LOG = ROOT / "invariants" / "out" / "interactive_memory.jsonl"
OUT = ROOT / "invariants" / "out" / "cache_timestamp_map.json"

TRACE_RE = re.compile(r"expert=(\S+?);\s*layers=(\d+)\D+?(\d+);\s*steps=(\d+)")


def load_traces():
    traces = []  # (ts, fingerprint)
    with open(LOG, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "synthesis_trace" not in (r.get("tags") or []):
                continue
            m = TRACE_RE.search(r.get("text") or "")
            if m:
                fp = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
                traces.append(((r.get("timestamp") or "")[:19], fp))
    return traces


def load_entries():
    mem = torch.load(CACHE, map_location="cpu", weights_only=True)
    fps = []
    for e in mem:
        md = e.get("metadata") or {}
        fps.append((
            (md.get("expert"), md.get("start_layer"), md.get("end_layer"), md.get("steps")),
            md.get("stored_at"),
        ))
    return fps


def lcs_align(entries, traces):
    """Longest-common-subsequence alignment on fingerprints.
    Returns {entry_index: trace_index} for aligned pairs."""
    n, m = len(entries), len(traces)
    # DP table of LCS lengths (n+1 x m+1); small enough (512 x ~600).
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, below = dp[i], dp[i + 1]
        efp = entries[i][0]
        for j in range(m - 1, -1, -1):
            if efp == traces[j][1]:
                row[j] = below[j + 1] + 1
            else:
                row[j] = max(below[j], row[j + 1])
    pairs = {}
    i = j = 0
    while i < n and j < m:
        if entries[i][0] == traces[j][1]:
            pairs[i] = j
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return pairs


def main():
    traces = load_traces()
    entries = load_entries()
    pairs = lcs_align(entries, traces)
    print(f"cache entries: {len(entries)}; logged traces: {len(traces)}; aligned: {len(pairs)}")

    rows = []
    for i, (fp, stamped) in enumerate(entries):
        if stamped:
            ts, kind = stamped, "stored_at"
        elif i in pairs:
            ts, kind = traces[pairs[i]][0], "trace_exact"
        else:
            ts, kind = None, "bounded"
        rows.append({
            "index": i,
            "expert": fp[0], "layers": f"{fp[1]}-{fp[2]}", "steps": fp[3],
            "ts": ts, "ts_kind": kind,
        })
    # bound the unaligned by nearest aligned neighbors
    last_ts = None
    for r in rows:
        if r["ts"]:
            last_ts = r["ts"]
        else:
            r["ts_after"] = last_ts
    next_ts = None
    for r in reversed(rows):
        if r["ts"]:
            next_ts = r["ts"]
        else:
            r["ts_before"] = next_ts

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"sidecar written: {OUT}")

    if "--window" in sys.argv:
        k = sys.argv.index("--window")
        lo, hi = sys.argv[k + 1], sys.argv[k + 2]
        print(f"\nentries whose time bounds intersect [{lo} .. {hi}]:")
        for r in rows:
            a = r.get("ts") or r.get("ts_after") or ""
            b = r.get("ts") or r.get("ts_before") or "9999"
            if a <= hi and b >= lo:
                tag = r["ts_kind"]
                shown = r.get("ts") or f"({r.get('ts_after')} .. {r.get('ts_before')})"
                print(f"  [{r['index']:>3}] {shown}  {tag:<11} expert={r['expert']} layers={r['layers']} steps={r['steps']}")


if __name__ == "__main__":
    main()
