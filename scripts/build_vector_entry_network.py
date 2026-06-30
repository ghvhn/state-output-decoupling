"""
Build a signed network over every extracted vector entry.

This is the denser companion to build_vector_network.py. It keeps layer-specific
vectors as separate nodes so we can see which layers carry support/opposition.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
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


def infer_entry_role(entry: dict[str, Any]) -> str:
    kind = entry.get("kind")
    label = entry.get("label", "")
    metadata = entry.get("metadata") or {}
    tag = str(metadata.get("tag") or "")
    signal = str(metadata.get("teaching_signal") or "")
    if label == "organic_correction_vector":
        return "correction_delta"
    if kind == "cache_delta":
        if signal == "penalty_bad_math" or "bad_math_penalty" in tag:
            return "penalty_delta"
        if signal == "reward_clean_math" or tag in {"native_success", "optimizer"}:
            return "reward_delta"
        return "cache_delta"
    if kind == "cache_trigger":
        if signal == "penalty_bad_math" or "bad_math_penalty" in tag:
            return "penalty_trigger"
        if signal == "reward_clean_math" or tag in {"native_success", "optimizer"}:
            return "reward_trigger"
        return "cache_trigger"
    if "validated_flow_vector" in label:
        return "validated_flow_layer"
    if "needless_interrupt_vector" in label:
        return "needless_interrupt_layer"
    if "narrowing_in_vector" in label:
        return "narrowing_layer"
    if "self_referential_momentum_vector" in label:
        return "self_momentum_layer"
    if kind == "layer_vector":
        return "probe_layer"
    return str(kind or "entry")


def connected_components(nodes: set[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    graph = {node: set() for node in nodes}
    for edge in edges:
        graph[edge["source"]].add(edge["target"])
        graph[edge["target"]].add(edge["source"])
    seen: set[str] = set()
    components: list[list[str]] = []
    for node in sorted(nodes):
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


def build_network(data: dict[str, Any], positive_threshold: float, negative_threshold: float) -> dict[str, Any]:
    entry_by_label = {entry["label"]: entry for entry in data.get("entries", [])}
    nodes = {
        label: {
            "id": label,
            "role": infer_entry_role(entry),
            "source": entry.get("source"),
            "kind": entry.get("kind"),
            "layer": entry.get("layer"),
            "index": entry.get("index"),
            "norm": entry.get("norm"),
            "degree_positive": 0,
            "degree_negative": 0,
            "signed_strength": 0.0,
        }
        for label, entry in entry_by_label.items()
    }
    edges: list[dict[str, Any]] = []
    for row in data.get("entry_pairs", []):
        value = float(row["cosine"])
        if value >= positive_threshold:
            sign = "positive"
        elif value <= negative_threshold:
            sign = "negative"
        else:
            continue
        a = row["a"]
        b = row["b"]
        if a not in nodes or b not in nodes:
            continue
        edge = {
            "source": a,
            "target": b,
            "sign": sign,
            "cosine": value,
            "source_role": nodes[a]["role"],
            "target_role": nodes[b]["role"],
        }
        edges.append(edge)
        for endpoint in (a, b):
            nodes[endpoint][f"degree_{sign}"] += 1
            nodes[endpoint]["signed_strength"] += value

    role_edges: dict[str, dict[str, Any]] = {}
    for edge in edges:
        left, right = sorted([edge["source_role"], edge["target_role"]])
        key = f"{left}::{right}::{edge['sign']}"
        item = role_edges.setdefault(
            key,
            {"role_a": left, "role_b": right, "sign": edge["sign"], "count": 0, "mean": 0.0},
        )
        item["count"] += 1
        item["mean"] += edge["cosine"]
    for item in role_edges.values():
        item["mean"] /= item["count"]

    positive_edges = [edge for edge in edges if edge["sign"] == "positive"]
    negative_edges = [edge for edge in edges if edge["sign"] == "negative"]
    return {
        "nodes": nodes,
        "edges": edges,
        "positive_components": connected_components(set(nodes), positive_edges),
        "negative_edges": sorted(negative_edges, key=lambda e: e["cosine"]),
        "role_edges": sorted(role_edges.values(), key=lambda e: (e["sign"], -abs(e["mean"]), e["role_a"])),
    }


def summarize_components(components: list[list[str]], nodes: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in components[:limit]:
        roles = defaultdict(int)
        artifacts = defaultdict(int)
        layers = []
        for node_id in group:
            node = nodes[node_id]
            roles[node["role"]] += 1
            artifacts[Path(str(node.get("source", ""))).stem] += 1
            if node.get("layer") is not None:
                layers.append(node["layer"])
        rows.append(
            {
                "size": len(group),
                "roles": dict(sorted(roles.items())),
                "artifacts": dict(sorted(artifacts.items())),
                "layer_min": min(layers) if layers else None,
                "layer_max": max(layers) if layers else None,
                "sample": group[:10],
            }
        )
    return rows


def write_markdown(path: Path, source_geometry: str, network: dict[str, Any]) -> None:
    nodes = network["nodes"]
    edges = network["edges"]
    strongest_positive = sorted(
        [e for e in edges if e["sign"] == "positive"], key=lambda e: e["cosine"], reverse=True
    )[:30]
    strongest_negative = network["negative_edges"][:30]
    top_nodes = sorted(
        nodes.values(),
        key=lambda n: (n["degree_positive"] + n["degree_negative"], abs(n["signed_strength"])),
        reverse=True,
    )[:40]
    component_summary = summarize_components(network["positive_components"], nodes, 20)

    lines = [
        "# Full Vector Entry Network",
        "",
        "Layer vectors, cache triggers, and cache deltas are separate nodes here.",
        "",
        f"- source geometry: `{source_geometry}`",
        f"- nodes: `{len(nodes)}`",
        f"- edges: `{len(edges)}`",
        "",
        "## High-Degree Nodes",
        "",
        "| node | role | layer | +degree | -degree | signed strength |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for node in top_nodes:
        layer = "" if node["layer"] is None else str(node["layer"])
        lines.append(
            f"| `{node['id']}` | `{node['role']}` | {layer} | {node['degree_positive']} | "
            f"{node['degree_negative']} | {node['signed_strength']:+.3f} |"
        )

    lines.extend(["", "## Positive Component Summary", ""])
    for i, component in enumerate(component_summary, start=1):
        role_text = ", ".join(f"{k}:{v}" for k, v in component["roles"].items())
        layer_text = ""
        if component["layer_min"] is not None:
            layer_text = f" layers {component['layer_min']}-{component['layer_max']}"
        lines.append(f"- component {i}: size {component['size']}{layer_text}; roles {role_text}")
        lines.append("  sample: " + ", ".join(f"`{x}`" for x in component["sample"][:5]))

    lines.extend(["", "## Strong Positive Edges", ""])
    for edge in strongest_positive:
        lines.append(f"- {edge['cosine']:+.4f}: `{edge['source']}` -> `{edge['target']}`")

    lines.extend(["", "## Strong Negative Edges", ""])
    for edge in strongest_negative:
        lines.append(f"- {edge['cosine']:+.4f}: `{edge['source']}` -> `{edge['target']}`")

    lines.extend(["", "## Role-Level Edge Summary", ""])
    lines.extend(["| sign | role A | role B | count | mean |", "|---|---|---|---:|---:|"])
    for edge in network["role_edges"]:
        lines.append(
            f"| `{edge['sign']}` | `{edge['role_a']}` | `{edge['role_b']}` | "
            f"{edge['count']} | {edge['mean']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="", help="vector_geometry_map_*.json path. Defaults to latest.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--positive-threshold", type=float, default=0.55)
    parser.add_argument("--negative-threshold", type=float, default=-0.55)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    input_path = Path(args.input) if args.input else latest_geometry_file(out_dir)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    network = build_network(data, args.positive_threshold, args.negative_threshold)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"vector_entry_network_{stamp}.json"
    md_path = out_dir / f"vector_entry_network_{stamp}.md"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_geometry": str(input_path),
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
        **network,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, str(input_path), network)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Nodes: {len(network['nodes'])}; edges: {len(network['edges'])}")
    print("Largest positive components:")
    for row in summarize_components(network["positive_components"], network["nodes"], 5):
        print(f"  size {row['size']} roles={row['roles']} layers={row['layer_min']}-{row['layer_max']}")
    print("Strongest negative edges:")
    for edge in network["negative_edges"][:8]:
        print(f"  {edge['cosine']:+.4f}  {edge['source']}  <->  {edge['target']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
