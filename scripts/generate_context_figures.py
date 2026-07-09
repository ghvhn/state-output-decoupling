"""Generate SVG figures from phen-shell traces and cached result files.

This is intentionally dependency-free: no model load, no matplotlib. It scans
the existing JSON/JSONL artifacts and writes a figure pack under
``figures/contexts``.

Usage:

  python scripts/generate_context_figures.py
  python scripts/generate_context_figures.py --max-jsonl-lines 50000
  python scripts/generate_context_figures.py --result-glob "invariants/out/*thinking*.json"
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "invariants" / "out"
FIG = ROOT / "figures" / "contexts"

INK = "#172026"
MUTED = "#5f6b73"
GRID = "#d7dde2"
BG = "#fbfcfd"
BLUE = "#2f6fed"
TEAL = "#13866f"
ORANGE = "#c76f1b"
RED = "#b83b4a"
PURPLE = "#7556c8"
GRAY = "#8a96a3"
COLORS = [BLUE, TEAL, ORANGE, RED, PURPLE, "#537188", "#a65f2b", "#4b8f8c"]

DEFAULT_RESULT_GLOBS = [
    "invariants/out/translation_thinking*.json",
    "invariants/out/style_layers*.json",
    "invariants/out/arrow_fold*.json",
    "invariants/out/cot_*.json",
    "invariants/out/frame_shift*.json",
    "invariants/out/role_frame_shift*.json",
    "invariants/out/feltness_empathy*.json",
    "invariants/out/self_regard_respect*.json",
    "invariants/out/reflexive_decompose*.json",
    "invariants/out/surgery*.json",
    "invariants/out/agency2*.json",
    "invariants/out/frames.json",
    "invariants/out/generality.json",
    "invariants/out/taskscope.json",
]

PHEN_KEYS = {
    "ambiguity",
    "repetition",
    "disagreement",
    "narrowing_in",
    "needless_interrupt",
    "self_referential_momentum",
    "time_awareness",
    "validated_flow",
    "warranted_confidence",
    "unwarranted_confidence",
    "warranted_confidence_legacy",
    "unwarranted_confidence_legacy",
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def slug(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return text[:110] or "figure"


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
    return None


def pct(value: float) -> str:
    return f"{100 * value:.0f}%"


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <title>{esc(path.name)}</title>
  <rect width="100%" height="100%" fill="{BG}"/>
  <style>
    text {{ font-family: Inter, Segoe UI, Arial, sans-serif; fill: {INK}; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .sub {{ font-size: 13px; fill: {MUTED}; }}
    .label {{ font-size: 13px; }}
    .small {{ font-size: 11px; fill: {MUTED}; }}
    .num {{ font-size: 18px; font-weight: 700; }}
  </style>
{body}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path, max_lines: int | None = None) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def top_items(counter: Counter, n: int = 12) -> list[tuple[str, int]]:
    return [(str(k), int(v)) for k, v in counter.most_common(n)]


def bar_chart(
    rows: list[tuple[str, float, str | None]],
    *,
    title: str,
    subtitle: str,
    path: Path,
    value_label=lambda x: f"{x:.2f}",
    width: int = 860,
) -> None:
    rows = rows[:18]
    row_h = 34
    top = 86
    height = max(190, top + row_h * len(rows) + 26)
    max_v = max([abs(v) for _, v, _ in rows] or [1.0]) or 1.0
    x0, y0, w = 230, top, width - 330
    body = f"""
  <text class="title" x="28" y="38">{esc(title)}</text>
  <text class="sub" x="28" y="62">{esc(subtitle)}</text>
"""
    for i, (label, value, note) in enumerate(rows):
        y = y0 + i * row_h
        color = COLORS[i % len(COLORS)]
        bw = int(w * (abs(value) / max_v))
        body += f"""
  <text class="label" x="28" y="{y + 18}">{esc(label[:32])}</text>
  <rect x="{x0}" y="{y}" width="{w}" height="20" rx="4" fill="#edf1f4"/>
  <rect x="{x0}" y="{y}" width="{bw}" height="20" rx="4" fill="{color}"/>
  <text class="num" x="{x0 + w + 14}" y="{y + 17}">{esc(value_label(value))}</text>
"""
        if note:
            body += f'  <text class="small" x="{x0 + w + 86}" y="{y + 16}">{esc(note[:36])}</text>\n'
    write_svg(path, width, height, body)


def line_path(values: list[float], x: int, y: int, w: int, h: int) -> str:
    if not values:
        return ""
    if len(values) == 1:
        values = values * 2
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-9:
        lo -= 1.0
        hi += 1.0
    pts = []
    for i, value in enumerate(values):
        px = x + (w * i / max(1, len(values) - 1))
        py = y + h - ((value - lo) / (hi - lo) * h)
        pts.append(f"{px:.1f},{py:.1f}")
    return " ".join(pts)


def sparkline_figure(
    series: list[tuple[str, list[float]]],
    *,
    title: str,
    subtitle: str,
    path: Path,
) -> None:
    series = series[:10]
    width = 900
    row_h = 58
    top = 88
    height = max(220, top + row_h * len(series) + 28)
    body = f"""
  <text class="title" x="28" y="38">{esc(title)}</text>
  <text class="sub" x="28" y="62">{esc(subtitle)}</text>
"""
    for i, (name, values) in enumerate(series):
        y = top + i * row_h
        color = COLORS[i % len(COLORS)]
        vals = values[-180:]
        body += f"""
  <text class="label" x="28" y="{y + 26}">{esc(name[:30])}</text>
  <line x1="230" y1="{y + 22}" x2="820" y2="{y + 22}" stroke="{GRID}" stroke-width="1"/>
  <polyline points="{line_path(vals, 230, y, 590, 44)}" fill="none" stroke="{color}" stroke-width="2.5"/>
  <text class="small" x="832" y="{y + 12}">n={len(values)}</text>
  <text class="small" x="832" y="{y + 30}">last={vals[-1]:+.3f}</text>
"""
    write_svg(path, width, height, body)


def numeric_leafs(row: dict[str, Any]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for key, value in row.items():
        number = safe_float(value)
        if number is not None:
            vals[key] = number
    return vals


def generate_turn_signal_figures(max_lines: int | None) -> list[Path]:
    path = OUT / "turn_signals.jsonl"
    if not path.exists():
        return []
    series: dict[str, list[float]] = defaultdict(list)
    total = 0
    for row in iter_jsonl(path, max_lines=max_lines):
        total += 1
        for key, value in numeric_leafs(row).items():
            series[key].append(value)
    if not series:
        return []

    ranked = sorted(
        series.items(),
        key=lambda kv: (len(kv[1]), statistics.pstdev(kv[1]) if len(kv[1]) > 1 else 0.0),
        reverse=True,
    )
    rows = []
    for name, values in ranked[:14]:
        latest = values[-1]
        span = max(values) - min(values) if values else 0.0
        rows.append((name, abs(latest), f"last {latest:+.3f}, span {span:.3f}"))
    out1 = FIG / "turn_signals_latest.svg"
    bar_chart(
        rows,
        title="Phen-shell turn signals",
        subtitle=f"Latest absolute values from {total} signal rows",
        path=out1,
        value_label=lambda x: f"{x:.3f}",
    )

    out2 = FIG / "turn_signals_sparklines.svg"
    sparkline_figure(
        ranked,
        title="Phen-shell signal trajectories",
        subtitle="Last 180 observed values for the most populated streams",
        path=out2,
    )
    return [out1, out2]


def generate_memory_activity(max_lines: int | None) -> list[Path]:
    path = OUT / "interactive_memory.jsonl"
    if not path.exists():
        return []
    kind_counts: Counter = Counter()
    role_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    day_counts: Counter = Counter()
    sessions: set[str] = set()
    total = 0
    for row in iter_jsonl(path, max_lines=max_lines):
        total += 1
        kind_counts[row.get("kind") or "unknown"] += 1
        role_counts[row.get("role") or "none"] += 1
        for tag in row.get("tags") or []:
            tag_counts[str(tag)] += 1
        timestamp = str(row.get("timestamp") or "")
        if len(timestamp) >= 10:
            day_counts[timestamp[:10]] += 1
        session_id = row.get("session_id")
        if session_id:
            sessions.add(str(session_id))
    if total == 0:
        return []

    rows = [(f"kind:{k}", v, None) for k, v in top_items(kind_counts, 6)]
    rows += [(f"role:{k}", v, None) for k, v in top_items(role_counts, 5)]
    rows += [(f"tag:{k}", v, None) for k, v in top_items(tag_counts, 7)]
    out = FIG / "interactive_memory_activity.svg"
    bar_chart(
        rows,
        title="Interactive memory activity",
        subtitle=f"{total} records across {len(sessions)} sessions; top kinds, roles, and tags",
        path=out,
        value_label=lambda x: f"{int(x)}",
    )

    day_rows = [(day, count, None) for day, count in sorted(day_counts.items())]
    out_days = FIG / "interactive_memory_by_day.svg"
    bar_chart(
        day_rows,
        title="Interactive memory by day",
        subtitle="Records written per UTC date",
        path=out_days,
        value_label=lambda x: f"{int(x)}",
    )
    return [out, out_days]


def generate_steer_map(max_lines: int | None) -> list[Path]:
    path = OUT / "steer_map_events.jsonl"
    if not path.exists():
        return []
    expert_counts: Counter = Counter()
    action_counts: Counter = Counter()
    success_counts: Counter = Counter()
    sensor_values: dict[str, list[float]] = defaultdict(list)
    total = 0
    for row in iter_jsonl(path, max_lines=max_lines):
        total += 1
        expert_counts[row.get("expert") or "unknown"] += 1
        action_counts[row.get("action") or "unknown"] += 1
        success_counts["success" if row.get("event_success") else "not_success"] += 1
        for key, value in (row.get("sensor_scores") or {}).items():
            number = safe_float(value)
            if number is not None:
                sensor_values[str(key)].append(number)
    if total == 0:
        return []
    rows = [(f"expert:{k}", v, None) for k, v in top_items(expert_counts, 7)]
    rows += [(f"action:{k}", v, None) for k, v in top_items(action_counts, 8)]
    rows += [(k, v, None) for k, v in top_items(success_counts, 2)]
    out = FIG / "steer_map_events.svg"
    bar_chart(
        rows,
        title="Steer map event activity",
        subtitle=f"{total} routed synthesis events; counts by expert, action, success",
        path=out,
        value_label=lambda x: f"{int(x)}",
    )
    if sensor_values:
        metric_rows = []
        for key, values in sensor_values.items():
            if values:
                metric_rows.append((key, statistics.fmean(abs(v) for v in values), f"n={len(values)}"))
        metric_rows.sort(key=lambda r: r[1], reverse=True)
        out_sensors = FIG / "steer_map_sensor_magnitudes.svg"
        bar_chart(
            metric_rows,
            title="Steer map sensor magnitudes",
            subtitle="Mean absolute phenomenality/sensor values on routed events",
            path=out_sensors,
            value_label=lambda x: f"{x:.3f}",
        )
        return [out, out_sensors]
    return [out]


def generate_trigger_tuners() -> list[Path]:
    paths = [OUT / "trigger_tuner.json"]
    paths.extend(sorted((OUT / "agents").glob("*/trigger_tuner.json")))
    rows = []
    for path in paths:
        if not path.exists():
            continue
        try:
            data = load_json(path)
        except Exception:
            continue
        agent = path.parent.name if path.parent.name != "out" else "default"
        if path.parent == OUT:
            agent = "default"
        for name, entry in data.items():
            if not isinstance(entry, dict):
                continue
            observed = safe_float(entry.get("observed")) or 0.0
            fired = safe_float(entry.get("fired")) or 0.0
            if observed <= 0:
                continue
            rate = fired / observed if observed else 0.0
            rows.append((f"{agent}:{name}", rate, f"observed {int(observed)}"))
    if not rows:
        return []
    rows.sort(key=lambda r: (r[2], r[1]), reverse=True)
    out = FIG / "trigger_tuner_fire_rates.svg"
    bar_chart(
        rows[:18],
        title="Trigger tuner fire rates",
        subtitle="Observed hooks from default and agent-specific phen-shell tuners",
        path=out,
        value_label=lambda x: pct(x),
    )
    return [out]


def phenomenality_from_dict(row: dict[str, Any]) -> Iterable[dict[str, float]]:
    direct = row.get("phenomenality")
    if isinstance(direct, dict):
        vals = {k: v for k, v in direct.items() if k in PHEN_KEYS and safe_float(v) is not None}
        if vals:
            yield {k: float(v) for k, v in vals.items()}

    metadata = row.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("phenomenality"), dict):
        vals = {
            k: v
            for k, v in metadata["phenomenality"].items()
            if k in PHEN_KEYS and safe_float(v) is not None
        }
        if vals:
            yield {k: float(v) for k, v in vals.items()}

    metrics = row.get("metrics")
    if isinstance(metrics, dict) and isinstance(metrics.get("phenomenality"), dict):
        vals = {
            k: v
            for k, v in metrics["phenomenality"].items()
            if k in PHEN_KEYS and safe_float(v) is not None
        }
        if vals:
            yield {k: float(v) for k, v in vals.items()}

    sensors = row.get("sensor_scores")
    if isinstance(sensors, dict):
        vals = {k: v for k, v in sensors.items() if k in PHEN_KEYS and safe_float(v) is not None}
        if vals:
            yield {k: float(v) for k, v in vals.items()}


def walk_phenomenality(obj: Any) -> Iterable[dict[str, float]]:
    if isinstance(obj, dict):
        yield from phenomenality_from_dict(obj)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                yield from walk_phenomenality(value)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                yield from walk_phenomenality(item)


def generate_phenomenality(max_lines: int | None) -> list[Path]:
    values: dict[str, list[float]] = defaultdict(list)

    for path in [OUT / "steer_map_events.jsonl", OUT / "latent_confidence_partial_20260630_011642.jsonl"]:
        if path.exists():
            for row in iter_jsonl(path, max_lines=max_lines):
                for phen in phenomenality_from_dict(row):
                    for key, value in phen.items():
                        values[key].append(value)

    for path in [OUT / "humble_full_suite_gsm8k.json"]:
        if path.exists() and path.stat().st_size < 5_000_000:
            try:
                data = load_json(path)
            except Exception:
                continue
            for phen in walk_phenomenality(data):
                for key, value in phen.items():
                    values[key].append(value)

    rows = []
    for key, vals in values.items():
        if vals:
            rows.append((key, statistics.fmean(abs(v) for v in vals), f"n={len(vals)}"))
    if not rows:
        return []
    rows.sort(key=lambda r: r[1], reverse=True)
    out = FIG / "phenomenality_metric_magnitudes.svg"
    bar_chart(
        rows,
        title="Phenomenality metric magnitudes",
        subtitle="Mean absolute sensor values from JSONL traces and benchmark records",
        path=out,
        value_label=lambda x: f"{x:.3f}",
    )
    return [out]


def metric_ranges(per_layer: list[dict[str, Any]]) -> list[str]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in per_layer:
        for key, value in row.items():
            if key == "layer" or key.endswith("_null") or key.endswith("_p"):
                continue
            number = safe_float(value)
            if number is not None:
                values[key].append(number)
    ranked = []
    for key, vals in values.items():
        if len(vals) >= 2:
            ranked.append((key, max(vals) - min(vals)))
    ranked.sort(key=lambda kv: kv[1], reverse=True)
    return [key for key, _ in ranked[:5]]


def draw_layer_profile(stem: str, profile_name: str, per_layer: list[dict[str, Any]], path: Path) -> None:
    metrics = metric_ranges(per_layer)
    if not metrics:
        return
    layers = [int(row.get("layer", i)) for i, row in enumerate(per_layer)]
    width, height = 920, 420
    x0, y0, w, h = 76, 92, 760, 250
    body = f"""
  <text class="title" x="28" y="38">{esc(stem)}</text>
  <text class="sub" x="28" y="62">Layer profile: {esc(profile_name)} ({len(per_layer)} layers)</text>
  <rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#ffffff" stroke="{GRID}"/>
"""
    all_values = []
    for metric in metrics:
        for row in per_layer:
            number = safe_float(row.get(metric))
            if number is not None:
                all_values.append(number)
    lo, hi = (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
    if abs(hi - lo) < 1e-9:
        lo -= 1.0
        hi += 1.0
    for i in range(6):
        gy = y0 + h - (h * i / 5)
        value = lo + (hi - lo) * i / 5
        body += f"""
  <line x1="{x0}" y1="{gy:.1f}" x2="{x0 + w}" y2="{gy:.1f}" stroke="{GRID}" stroke-width="1"/>
  <text class="small" x="24" y="{gy + 4:.1f}">{value:.2f}</text>
"""
    lx0 = min(layers) if layers else 0
    lx1 = max(layers) if layers else 1
    if lx0 == lx1:
        lx1 = lx0 + 1
    for mi, metric in enumerate(metrics):
        pts = []
        for row_i, row in enumerate(per_layer):
            value = safe_float(row.get(metric))
            if value is None:
                continue
            layer = int(row.get("layer", row_i))
            px = x0 + ((layer - lx0) / (lx1 - lx0) * w)
            py = y0 + h - ((value - lo) / (hi - lo) * h)
            pts.append(f"{px:.1f},{py:.1f}")
        color = COLORS[mi % len(COLORS)]
        body += f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.4"/>\n'
        body += f'  <rect x="{x0 + mi * 145}" y="{y0 + h + 32}" width="12" height="12" fill="{color}"/>\n'
        body += f'  <text class="small" x="{x0 + mi * 145 + 18}" y="{y0 + h + 43}">{esc(metric[:18])}</text>\n'
    write_svg(path, width, height, body)


def generate_result_profiles(result_globs: list[str], max_json_bytes: int) -> list[Path]:
    paths: list[Path] = []
    for pattern in result_globs:
        paths.extend(sorted(ROOT.glob(pattern)))
    unique_paths = []
    seen = set()
    for path in paths:
        if path in seen or not path.exists() or path.stat().st_size > max_json_bytes:
            continue
        seen.add(path)
        unique_paths.append(path)

    written: list[Path] = []
    for path in unique_paths:
        try:
            data = load_json(path)
        except Exception:
            continue
        stem = path.stem
        profiles: list[tuple[str, list[dict[str, Any]]]] = []
        positions = data.get("positions") if isinstance(data, dict) else None
        if isinstance(positions, dict):
            for name, value in positions.items():
                if isinstance(value, dict) and isinstance(value.get("per_layer"), list):
                    profiles.append((str(name), value["per_layer"]))
        if isinstance(data, dict) and isinstance(data.get("per_layer"), list):
            profiles.append(("per_layer", data["per_layer"]))

        for profile_name, per_layer in profiles:
            if not per_layer:
                continue
            out = FIG / f"profile_{slug(stem)}_{slug(profile_name)}.svg"
            draw_layer_profile(stem, profile_name, per_layer, out)
            if out.exists():
                written.append(out)

        if isinstance(data, dict) and isinstance(data.get("sweep"), list):
            sweep_rows = []
            for row in data["sweep"]:
                if not isinstance(row, dict):
                    continue
                alpha = row.get("alpha")
                for key in ["hedge", "fluent", "clean", "reached", "commit", "acc"]:
                    value = safe_float(row.get(key))
                    if value is not None:
                        sweep_rows.append((f"alpha {alpha} {key}", value, None))
            if sweep_rows:
                out = FIG / f"sweep_{slug(stem)}.svg"
                bar_chart(
                    sweep_rows,
                    title=stem,
                    subtitle="Sweep metrics inferred from cached JSON",
                    path=out,
                    value_label=lambda x: f"{x:.3f}",
                )
                written.append(out)

        summary = data.get("summary") if isinstance(data, dict) else None
        if isinstance(summary, dict):
            simple_rows = []
            for key, value in summary.items():
                number = safe_float(value)
                if number is not None:
                    simple_rows.append((str(key), number, None))
                elif isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        sub_num = safe_float(sub_val)
                        if sub_num is not None:
                            simple_rows.append((f"{key}.{sub_key}", sub_num, None))
            if simple_rows:
                out = FIG / f"summary_{slug(stem)}.svg"
                bar_chart(
                    simple_rows,
                    title=stem,
                    subtitle="Summary metrics inferred from cached JSON",
                    path=out,
                    value_label=lambda x: f"{x:.3f}",
                )
                written.append(out)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate context figures from phen-shell and cached JSON artifacts.")
    parser.add_argument("--out", default=str(FIG), help="Directory for generated SVG figures.")
    parser.add_argument("--max-jsonl-lines", type=int, default=200000, help="Maximum lines to read from each JSONL file.")
    parser.add_argument("--max-json-bytes", type=int, default=5_000_000, help="Skip cached JSON files larger than this.")
    parser.add_argument(
        "--result-glob",
        action="append",
        default=[],
        help="Additional repo-relative glob for cached JSON result profiles. Can be repeated.",
    )
    parser.add_argument("--no-default-results", action="store_true", help="Only use explicitly supplied --result-glob patterns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global FIG
    FIG = Path(args.out)
    max_lines = args.max_jsonl_lines if args.max_jsonl_lines > 0 else None

    written: list[Path] = []
    written += generate_turn_signal_figures(max_lines)
    written += generate_memory_activity(max_lines)
    written += generate_steer_map(max_lines)
    written += generate_trigger_tuners()
    written += generate_phenomenality(max_lines)

    globs = [] if args.no_default_results else list(DEFAULT_RESULT_GLOBS)
    globs += args.result_glob
    written += generate_result_profiles(globs, args.max_json_bytes)

    print(f"Wrote {len(written)} context figures -> {FIG.resolve()}")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
