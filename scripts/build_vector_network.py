"""
Build a signed vector network from vector-geometry output.

Nodes are vector artifact groups. Edges are strong positive or negative cosine
relations. The event lens compares how each node relates to good/correction
deltas versus bad-math penalty deltas.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any


OUT_DIR = Path("invariants/out")


def latest_geometry_file(out_dir: Path) -> Path:
    candidates = sorted(out_dir.glob("vector_geometry_map_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No vector_geometry_map_*.json files found. Run scripts/map_vector_geometry.py first.")
    return candidates[-1]


def artifact_for_entry(label: str, source: str, kind: str) -> str:
    base = Path(source).stem
    if kind.startswith("cache_"):
        return f"{base}:{kind}"
    return base


def infer_role(name: str, entries: list[dict[str, Any]]) -> str:
    tags = Counter()
    signals = Counter()
    for entry in entries:
        meta = entry.get("metadata") or {}
        tag = str(meta.get("tag") or "")
        signal = str(meta.get("teaching_signal") or "")
        if tag:
            tags[tag] += 1
        if signal:
            signals[signal] += 1

    if name == "organic_correction_vector":
        return "correction_delta"
    if name.endswith(":cache_delta"):
        if signals["penalty_bad_math"] or any("bad_math_penalty" in tag for tag in tags):
            return "penalty_delta"
        if signals["reward_clean_math"] or tags["native_success"] or tags["optimizer"]:
            return "reward_delta"
        return "cache_delta"
    if name.endswith(":cache_trigger"):
        if signals["penalty_bad_math"] or any("bad_math_penalty" in tag for tag in tags):
            return "penalty_trigger"
        if signals["reward_clean_math"] or tags["native_success"] or tags["optimizer"]:
            return "reward_trigger"
        return "cache_trigger"
    if name.startswith("clouds_"):
        return "activation_cloud"
    if name.startswith("self_") or name == "self_experience":
        return "self_probe"
    if name.endswith("_vector"):
        return "probe_vector"
    return "artifact"


def connected_components(nodes: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    graph = {node: set() for node in nodes}
    for edge in edges:
        graph[edge["source"]].add(edge["target"])
        graph[edge["target"]].add(edge["source"])
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        group: list[str] = []
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(group))
    return sorted(components, key=lambda g: (-len(g), g[0]))


def edge_between(pair_lookup: dict[tuple[str, str], float], a: str, b: str) -> float | None:
    if (a, b) in pair_lookup:
        return pair_lookup[(a, b)]
    if (b, a) in pair_lookup:
        return pair_lookup[(b, a)]
    return None


def event_lens(nodes: dict[str, dict[str, Any]], pair_lookup: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    good_roles = {"reward_delta", "correction_delta"}
    bad_roles = {"penalty_delta"}
    good_nodes = [name for name, node in nodes.items() if node["role"] in good_roles]
    bad_nodes = [name for name, node in nodes.items() if node["role"] in bad_roles]

    rows: list[dict[str, Any]] = []
    for name in sorted(nodes):
        good_vals = [edge_between(pair_lookup, name, other) for other in good_nodes if other != name]
        bad_vals = [edge_between(pair_lookup, name, other) for other in bad_nodes if other != name]
        good_vals = [v for v in good_vals if v is not None]
        bad_vals = [v for v in bad_vals if v is not None]
        good_mean = sum(good_vals) / len(good_vals) if good_vals else None
        bad_mean = sum(bad_vals) / len(bad_vals) if bad_vals else None
        if good_mean is None and bad_mean is None:
            continue
        if good_mean is None:
            contrast = None
        elif bad_mean is None:
            contrast = None
        else:
            contrast = good_mean - bad_mean
        rows.append(
            {
                "node": name,
                "role": nodes[name]["role"],
                "mean_to_good_events": good_mean,
                "mean_to_bad_events": bad_mean,
                "good_minus_bad": contrast,
            }
        )
    return sorted(
        rows,
        key=lambda r: (r["good_minus_bad"] is None, -(r["good_minus_bad"] or -999)),
    )


def build_network(data: dict[str, Any], positive_threshold: float, negative_threshold: float) -> dict[str, Any]:
    grouped_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in data.get("entries", []):
        name = artifact_for_entry(entry["label"], entry["source"], entry["kind"])
        grouped_entries[name].append(entry)

    nodes: dict[str, dict[str, Any]] = {}
    for name, info in data.get("artifacts", {}).items():
        entries = grouped_entries.get(name, [])
        nodes[name] = {
            "id": name,
            "role": infer_role(name, entries),
            "count": info.get("count"),
            "kinds": info.get("kinds"),
            "layers": info.get("layers"),
            "norm_mean": info.get("norm_mean"),
            "sources": info.get("sources"),
        }

    edges: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[str, str], float] = {}
    for row in data.get("artifact_pairs", []):
        a = row["a"]
        b = row["b"]
        mean = float(row["mean"])
        pair_lookup[(a, b)] = mean
        pair_lookup[(b, a)] = mean
        if mean >= positive_threshold:
            sign = "positive"
        elif mean <= negative_threshold:
            sign = "negative"
        else:
            continue
        edges.append(
            {
                "source": a,
                "target": b,
                "sign": sign,
                "mean": mean,
                "min": row.get("min"),
                "max": row.get("max"),
                "count": row.get("count"),
                "strongest": row.get("strongest"),
            }
        )

    degree = {name: {"positive": 0, "negative": 0, "signed_strength": 0.0} for name in nodes}
    for edge in edges:
        for endpoint in (edge["source"], edge["target"]):
            degree[endpoint][edge["sign"]] += 1
            degree[endpoint]["signed_strength"] += edge["mean"]
    for name, values in degree.items():
        nodes[name]["degree_positive"] = values["positive"]
        nodes[name]["degree_negative"] = values["negative"]
        nodes[name]["signed_strength"] = values["signed_strength"]

    positive_edges = [edge for edge in edges if edge["sign"] == "positive"]
    negative_edges = [edge for edge in edges if edge["sign"] == "negative"]
    return {
        "nodes": nodes,
        "edges": edges,
        "positive_components": connected_components(sorted(nodes), positive_edges),
        "negative_edges": sorted(negative_edges, key=lambda e: e["mean"]),
        "event_lens": event_lens(nodes, pair_lookup),
    }


def write_dot(path: Path, network: dict[str, Any]) -> None:
    role_colors = {
        "correction_delta": "palegreen",
        "reward_delta": "darkseagreen1",
        "penalty_delta": "lightcoral",
        "reward_trigger": "lightgoldenrod1",
        "penalty_trigger": "orange",
        "probe_vector": "lightblue",
        "activation_cloud": "gray90",
        "self_probe": "plum1",
    }
    lines = ["graph VectorNetwork {", "  overlap=false;", "  splines=true;"]
    for name, node in sorted(network["nodes"].items()):
        color = role_colors.get(node["role"], "white")
        label = f"{name}\\n{node['role']}"
        lines.append(f'  "{name}" [label="{label}", style=filled, fillcolor="{color}"];')
    for edge in network["edges"]:
        color = "forestgreen" if edge["sign"] == "positive" else "firebrick"
        style = "solid" if edge["sign"] == "positive" else "dashed"
        lines.append(
            f'  "{edge["source"]}" -- "{edge["target"]}" '
            f'[label="{edge["mean"]:+.2f}", color="{color}", style="{style}"];'
        )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_markdown(path: Path, source_geometry: str, network: dict[str, Any]) -> None:
    nodes = network["nodes"]
    edges = network["edges"]
    positive_edges = [e for e in edges if e["sign"] == "positive"]
    negative_edges = [e for e in edges if e["sign"] == "negative"]
    lines = [
        "# Vector Network",
        "",
        "This is a signed network over saved vector artifact groups. Positive edges indicate aligned directions; negative edges indicate opposed directions.",
        "",
        f"- source geometry: `{source_geometry}`",
        f"- nodes: `{len(nodes)}`",
        f"- positive edges: `{len(positive_edges)}`",
        f"- negative edges: `{len(negative_edges)}`",
        "",
        "## Nodes",
        "",
        "| node | role | +degree | -degree | signed strength |",
        "|---|---|---:|---:|---:|",
    ]
    for name, node in sorted(nodes.items(), key=lambda kv: (kv[1]["role"], kv[0])):
        lines.append(
            f"| `{name}` | `{node['role']}` | {node['degree_positive']} | "
            f"{node['degree_negative']} | {node['signed_strength']:+.3f} |"
        )

    lines.extend(["", "## Positive Components", ""])
    for i, group in enumerate(network["positive_components"], start=1):
        if len(group) <= 1:
            continue
        lines.append(f"- component {i}: " + ", ".join(f"`{name}`" for name in group))

    lines.extend(["", "## Strong Negative Edges", ""])
    for edge in network["negative_edges"][:20]:
        lines.append(f"- `{edge['source']}` vs `{edge['target']}`: {edge['mean']:+.4f}")

    lines.extend(["", "## Good/Bad Event Lens", ""])
    lines.extend(["| node | role | mean to good | mean to bad | good minus bad |", "|---|---|---:|---:|---:|"])
    for row in network["event_lens"]:
        good = "" if row["mean_to_good_events"] is None else f"{row['mean_to_good_events']:+.4f}"
        bad = "" if row["mean_to_bad_events"] is None else f"{row['mean_to_bad_events']:+.4f}"
        contrast = "" if row["good_minus_bad"] is None else f"{row['good_minus_bad']:+.4f}"
        lines.append(f"| `{row['node']}` | `{row['role']}` | {good} | {bad} | {contrast} |")

    lines.extend(
        [
            "",
            "## Read This Way",
            "",
            "- Good-event affinity means a vector currently points with reward/correction deltas.",
            "- Bad-event affinity means a vector points with penalty deltas or away from correction.",
            "- A strong good-minus-bad contrast is a candidate control axis, not proof of a causal intervention.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="", help="vector_geometry_map_*.json path. Defaults to latest.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--positive-threshold", type=float, default=0.30)
    parser.add_argument("--negative-threshold", type=float, default=-0.30)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    input_path = Path(args.input) if args.input else latest_geometry_file(out_dir)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    network = build_network(data, args.positive_threshold, args.negative_threshold)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"vector_network_{stamp}.json"
    md_path = out_dir / f"vector_network_{stamp}.md"
    dot_path = out_dir / f"vector_network_{stamp}.dot"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_geometry": str(input_path),
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
        **network,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, str(input_path), network)
    write_dot(dot_path, network)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {dot_path}")
    print("Positive components:")
    for group in network["positive_components"]:
        if len(group) > 1:
            print("  - " + ", ".join(group))
    print("Strong negative edges:")
    for edge in network["negative_edges"][:8]:
        print(f"  {edge['mean']:+.4f}  {edge['source']}  <->  {edge['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
