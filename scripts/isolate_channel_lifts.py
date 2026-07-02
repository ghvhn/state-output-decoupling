"""Automated causal isolation of steering-channel lifts.

The observational channel_lift (fired vs unfired outcomes) is a screening
instrument: channels fire on states that need them, so it cannot separate
"helps" from "shows up in hard spots". This runner does the controlled part
automatically:

1. Picks channels worth isolating (default: every channel with fired evidence
   in the steer-map store; falls back to all known channels).
2. Runs a CONTROL benchmark (full stack), then one ABLATION run per channel
   with exactly that channel's effect removed (TDA_DISABLED_STEER_CHANNELS —
   the channel still computes as in control; only its injection and cache
   write are skipped).
3. Reports the causal contribution per channel: control metrics minus ablated
   metrics, on the same rows, same lane, same settings.

Deterministic procedure: fixed row slice, one-variable-per-run, and the
report is a pure function of the run summaries. GPU nondeterminism aside,
what varies between runs is the single disabled channel.

Run (small first -- each channel costs one full benchmark pass):
    .venv\\Scripts\\python.exe scripts\\isolate_channel_lifts.py --n 5
    .venv\\Scripts\\python.exe scripts\\isolate_channel_lifts.py --n 5 --channels urgency,cache_delta
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.config import KNOWN_STEER_CHANNELS
from invariants.steer_map_store import SteerMapStore

OUT_DIR = Path(__file__).parent.parent / "invariants" / "out"
EVALUATE = Path(__file__).parent / "evaluate_humble_full_suite.py"
METRIC_KEYS = ("accuracy", "coverage", "selective_accuracy", "n", "correct")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", default="5", help="Rows per run. Every run uses the same slice.")
    p.add_argument(
        "--channels",
        default="auto",
        help="Comma-separated channels, 'all', or 'auto' (channels with fired evidence in the steer map).",
    )
    p.add_argument("--method", default="humble_synthesis", help="Method whose summary metrics are compared.")
    p.add_argument("--run-kind", default="bench-standard")
    p.add_argument("--min-fired", type=int, default=1, help="'auto' picks channels with at least this many fired events.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--extra-arg", action="append", default=[], help="Extra argument passed through to the benchmark (repeatable).")
    p.add_argument("--skip-control", default=None, help="Reuse an existing control run JSON instead of running one.")
    return p.parse_args()


def pick_channels(spec: str, min_fired: int) -> list[str]:
    if spec == "all":
        return list(KNOWN_STEER_CHANNELS)
    if spec != "auto":
        names = [part.strip() for part in spec.split(",") if part.strip()]
        unknown = [name for name in names if name not in KNOWN_STEER_CHANNELS]
        if unknown:
            raise SystemExit(f"Unknown channels {unknown}; known: {list(KNOWN_STEER_CHANNELS)}")
        return names
    store = SteerMapStore()
    fired = [
        row["channel"]
        for row in store.channel_lift(basis="any")
        if row["fired_n"] >= max(1, min_fired) and row["channel"] in KNOWN_STEER_CHANNELS
    ]
    if fired:
        print(f"[Isolate] auto-picked channels with fired evidence: {fired}")
        return fired
    print("[Isolate] steer map has no fired channel evidence yet; isolating all known channels.")
    return list(KNOWN_STEER_CHANNELS)


def run_benchmark(args, label: str, disabled: str, output: Path) -> dict:
    cmd = [
        args.python,
        str(EVALUATE),
        "--n", str(args.n),
        "--methods", args.method,
        "--run-kind", args.run_kind,
        "--boring",
        "--output", str(output),
    ] + list(args.extra_arg)
    env = dict(os.environ)
    if disabled:
        env["TDA_DISABLED_STEER_CHANNELS"] = disabled
    else:
        env.pop("TDA_DISABLED_STEER_CHANNELS", None)
    print(f"[Isolate] run={label} disabled={disabled or '-'} -> {output.name}", flush=True)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"[Isolate] run '{label}' failed with exit code {proc.returncode}; aborting.")
    return json.loads(output.read_text(encoding="utf-8"))


def method_metrics(results: dict, method: str) -> dict:
    entry = ((results.get("summary") or {}).get("methods") or {}).get(method) or {}
    return {key: entry.get(key) for key in METRIC_KEYS if key in entry}


def build_isolation_report(method: str, control: dict, ablations: dict[str, dict]) -> dict:
    """Pure function of run summaries -> per-channel causal contributions.
    contribution > 0 means removing the channel HURT (the channel helps);
    contribution < 0 means removing it helped (the channel hurts)."""
    rows = []
    for channel in sorted(ablations):
        ablated = ablations[channel]
        row = {"channel": channel, "control": control, "ablated": ablated}
        for key in ("accuracy", "coverage", "selective_accuracy"):
            c, a = control.get(key), ablated.get(key)
            row[f"{key}_contribution"] = round(c - a, 4) if c is not None and a is not None else None
        rows.append(row)
    return {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "method": method,
        "reading": "contribution = control - ablated; positive means the channel helps",
        "control": control,
        "channels": rows,
    }


def main() -> int:
    args = parse_args()
    channels = pick_channels(args.channels, args.min_fired)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.skip_control:
        control_results = json.loads(Path(args.skip_control).read_text(encoding="utf-8"))
        if control_results.get("disabled_steer_channels"):
            raise SystemExit("[Isolate] --skip-control run had channels disabled; it is not a control.")
    else:
        control_results = run_benchmark(args, "control", "", OUT_DIR / f"isolate_control_{stamp}.json")
    control = method_metrics(control_results, args.method)
    if not control:
        raise SystemExit(f"[Isolate] control run has no summary for method '{args.method}'.")

    ablations: dict[str, dict] = {}
    for channel in channels:
        results = run_benchmark(args, f"ablate_{channel}", channel, OUT_DIR / f"isolate_no_{channel}_{stamp}.json")
        ablations[channel] = method_metrics(results, args.method)

    report = build_isolation_report(args.method, control, ablations)
    report_path = OUT_DIR / f"channel_isolation_report_{stamp}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        f"# Channel Isolation Report {stamp}",
        "",
        f"Method: `{args.method}`; n={args.n}; run-kind={args.run_kind}.",
        "Contribution = control - ablated. Positive: the channel helps. One channel removed per run;",
        "the channel still computes as in control, only its effect is skipped.",
        "",
        f"Control: {json.dumps(control)}",
        "",
        "| channel | accuracy contrib. | coverage contrib. | selective contrib. | ablated metrics |",
        "|---|---|---|---|---|",
    ]
    for row in report["channels"]:
        md_lines.append(
            f"| {row['channel']} | {row['accuracy_contribution']} | {row['coverage_contribution']} "
            f"| {row['selective_accuracy_contribution']} | {json.dumps(row['ablated'])} |"
        )
    md_path = OUT_DIR / f"channel_isolation_report_{stamp}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[Isolate] report: {report_path}")
    print(f"[Isolate] report: {md_path}")
    for row in report["channels"]:
        print(
            f"[Isolate] {row['channel']}: accuracy_contribution={row['accuracy_contribution']} "
            f"coverage_contribution={row['coverage_contribution']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
