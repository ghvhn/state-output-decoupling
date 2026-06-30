"""
Build a small latent relation map from vector-geometry output.

Input comes from scripts/map_vector_geometry.py. The projection is classical MDS
over artifact-level cosine distances, so it is a map of saved vector relations,
not a claim that the model itself has exactly these axes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch


OUT_DIR = Path("invariants/out")


def latest_geometry_file(out_dir: Path) -> Path:
    candidates = sorted(out_dir.glob("vector_geometry_map_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError("No vector_geometry_map_*.json files found. Run scripts/map_vector_geometry.py first.")
    return candidates[-1]


def cosine_to_distance(cosine: float) -> float:
    cosine = max(-1.0, min(1.0, float(cosine)))
    return float((2.0 * (1.0 - cosine)) ** 0.5)


def classical_mds(similarity: torch.Tensor, dims: int = 3) -> tuple[torch.Tensor, list[float]]:
    n = similarity.shape[0]
    distance = torch.empty_like(similarity)
    for i in range(n):
        for j in range(n):
            distance[i, j] = 0.0 if i == j else cosine_to_distance(float(similarity[i, j]))

    d2 = distance.pow(2)
    eye = torch.eye(n, dtype=torch.float64)
    ones = torch.ones((n, n), dtype=torch.float64) / n
    center = eye - ones
    gram = -0.5 * center @ d2.to(torch.float64) @ center
    evals, evecs = torch.linalg.eigh(gram)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]

    positive = torch.clamp(evals[:dims], min=0.0)
    coords = evecs[:, :dims] * positive.sqrt().unsqueeze(0)
    total = torch.clamp(evals, min=0.0).sum().item()
    explained = [(float(v.item()) / total if total > 0 else 0.0) for v in positive]
    return coords.float(), explained


def build_similarity(data: dict[str, Any]) -> tuple[list[str], torch.Tensor, dict[tuple[str, str], dict[str, Any]]]:
    names = sorted(data["artifacts"])
    idx = {name: i for i, name in enumerate(names)}
    sim = torch.zeros((len(names), len(names)), dtype=torch.float64)
    sim.fill_diagonal_(1.0)
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in data.get("artifact_pairs", []):
        a = row["a"]
        b = row["b"]
        if a not in idx or b not in idx:
            continue
        value = float(row["mean"])
        sim[idx[a], idx[b]] = value
        sim[idx[b], idx[a]] = value
        pair_lookup[(a, b)] = row
        pair_lookup[(b, a)] = row
    return names, sim, pair_lookup


def connected_components(names: list[str], edges: list[dict[str, Any]]) -> list[list[str]]:
    graph = {name: set() for name in names}
    for edge in edges:
        graph[edge["a"]].add(edge["b"])
        graph[edge["b"]].add(edge["a"])

    seen: set[str] = set()
    components: list[list[str]] = []
    for name in names:
        if name in seen:
            continue
        stack = [name]
        group: list[str] = []
        seen.add(name)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in sorted(graph[cur]):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(sorted(group))
    return sorted(components, key=lambda g: (-len(g), g[0]))


def neighbors(names: list[str], sim: torch.Tensor, top_k: int) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for i, name in enumerate(names):
        rows = []
        for j, other in enumerate(names):
            if i == j:
                continue
            rows.append({"name": other, "cosine": float(sim[i, j].item())})
        rows.sort(key=lambda r: r["cosine"], reverse=True)
        out[name] = rows[:top_k]
    return out


def axis_extremes(names: list[str], coords: torch.Tensor, dim: int, k: int = 5) -> dict[str, list[dict[str, Any]]]:
    values = [{"name": name, "value": float(coords[i, dim].item())} for i, name in enumerate(names)]
    return {
        "positive": sorted(values, key=lambda r: r["value"], reverse=True)[:k],
        "negative": sorted(values, key=lambda r: r["value"])[:k],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# Vector Latent Space",
        "",
        "This is a relation map of saved vector artifacts. It is useful for choosing probes and controls, not for making causal claims by itself.",
        "",
        f"- source geometry: `{payload['source_geometry']}`",
        f"- artifacts projected: `{len(payload['coordinates'])}`",
        f"- explained variance: `{', '.join(f'{v:.1%}' for v in payload['explained'])}`",
        "",
        "## Coordinates",
        "",
        "| artifact | x | y | z | nearest positive neighbors |",
        "|---|---:|---:|---:|---|",
    ]
    coords = payload["coordinates"]
    for name in sorted(coords):
        c = coords[name]
        neigh = ", ".join(f"{n['name']} ({n['cosine']:+.2f})" for n in payload["neighbors"][name][:3])
        lines.append(f"| `{name}` | {c[0]:+.4f} | {c[1]:+.4f} | {c[2]:+.4f} | {neigh} |")

    lines.extend(["", "## Positive Neighborhoods", ""])
    for i, group in enumerate(payload["positive_components"], start=1):
        if len(group) == 1:
            continue
        lines.append(f"- cluster {i}: " + ", ".join(f"`{name}`" for name in group))

    lines.extend(["", "## Strong Anti-Edges", ""])
    if payload["anti_edges"]:
        for edge in payload["anti_edges"]:
            lines.append(f"- `{edge['a']}` vs `{edge['b']}`: {edge['mean']:+.4f}")
    else:
        lines.append("- none at the selected threshold")

    lines.extend(["", "## Axis Extremes", ""])
    for axis_name, extremes in payload["axis_extremes"].items():
        lines.append(f"### {axis_name}")
        lines.append("- positive: " + ", ".join(f"`{r['name']}` ({r['value']:+.3f})" for r in extremes["positive"]))
        lines.append("- negative: " + ", ".join(f"`{r['name']}` ({r['value']:+.3f})" for r in extremes["negative"]))

    lines.extend(
        [
            "",
            "## Immediate Read",
            "",
            "- Treat high positive neighborhoods as candidate shared routines or contaminated shared wording.",
            "- Treat strong anti-edges as candidate veto axes, especially when a reward delta and a penalty delta are exact opposites.",
            "- Ambiguity/urgency correlation should be controlled before using urgency as an independent intervention.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="", help="vector_geometry_map_*.json path. Defaults to latest.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--positive-threshold", type=float, default=0.30)
    parser.add_argument("--anti-threshold", type=float, default=-0.30)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    input_path = Path(args.input) if args.input else latest_geometry_file(out_dir)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    names, sim, _pair_lookup = build_similarity(data)
    coords, explained = classical_mds(sim, dims=3)

    positive_edges = [
        row
        for row in data.get("artifact_pairs", [])
        if float(row["mean"]) >= args.positive_threshold
    ]
    anti_edges = sorted(
        [
            row
            for row in data.get("artifact_pairs", [])
            if float(row["mean"]) <= args.anti_threshold
        ],
        key=lambda r: r["mean"],
    )
    components = connected_components(names, positive_edges)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"vector_latent_space_{stamp}.json"
    md_path = out_dir / f"vector_latent_space_{stamp}.md"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_geometry": str(input_path),
        "positive_threshold": args.positive_threshold,
        "anti_threshold": args.anti_threshold,
        "explained": explained,
        "coordinates": {
            name: [float(x) for x in coords[i].tolist()]
            for i, name in enumerate(names)
        },
        "neighbors": neighbors(names, sim, args.top_k),
        "positive_edges": positive_edges,
        "anti_edges": anti_edges,
        "positive_components": components,
        "axis_extremes": {
            "axis_1": axis_extremes(names, coords, 0, args.top_k),
            "axis_2": axis_extremes(names, coords, 1, args.top_k),
            "axis_3": axis_extremes(names, coords, 2, args.top_k),
        },
    }

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print("Positive neighborhoods:")
    for group in components:
        if len(group) > 1:
            print("  - " + ", ".join(group))
    print("Strong anti-edges:")
    for edge in anti_edges[:8]:
        print(f"  {edge['mean']:+.4f}  {edge['a']}  <->  {edge['b']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
