"""Build a directed latent motion map from saved activation harvests.

This is not another cosine matrix. It stores directed displacement vectors:

  - pre_reasoning -> generated_reasoning, split by correct/wrong outcomes
  - question_centroid -> sampled_reply, split by correct/wrong sampled replies
  - neutralized_wording -> standard_wording for paired concept probes

The readable markdown summarizes the map; the .pt payload keeps the actual
vectors so later probes can compose, compare, or steer with them.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


OUT = Path("invariants/out")
INV = Path("invariants")


def latest(pattern: str, out_dir: Path = OUT) -> Path | None:
    files = sorted(out_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def load_points(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    obj = torch.load(path, map_location="cpu")
    return obj if isinstance(obj, list) else []


def mean_tensor(items: list[torch.Tensor]) -> torch.Tensor | None:
    if not items:
        return None
    return torch.stack([x.detach().cpu().float() for x in items]).mean(0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a = a.detach().cpu().float().reshape(-1)
    b = b.detach().cpu().float().reshape(-1)
    if a.numel() != b.numel() or a.norm().item() == 0 or b.norm().item() == 0:
        return None
    return float(F.cosine_similarity(a, b, dim=0).item())


def vector_norm(v: torch.Tensor) -> float:
    return float(v.detach().cpu().float().norm().item())


def add_edge(
    payload: dict[str, Any],
    *,
    family: str,
    source: str,
    target: str,
    layer: int,
    vector: torch.Tensor,
    count: int,
    metadata: dict[str, Any] | None = None,
) -> None:
    key = f"edge_{len(payload['edges']):05d}"
    vector = vector.detach().cpu().float().contiguous()
    payload["vectors"][key] = vector
    payload["nodes"].setdefault(source, {"id": source})
    payload["nodes"].setdefault(target, {"id": target})
    payload["edges"].append(
        {
            "id": key,
            "family": family,
            "source": source,
            "target": target,
            "layer": int(layer),
            "count": int(count),
            "magnitude": vector_norm(vector),
            "metadata": metadata or {},
        }
    )


def build_outcome_motion(payload: dict[str, Any], points: list[dict[str, Any]], source: str) -> None:
    usable = [p for p in points if "acts_pre" in p and "acts_gen" in p]
    if not usable:
        return
    n_layers = int(usable[0]["acts_gen"].shape[0])
    groups: dict[str, list[torch.Tensor]] = {"correct": [], "wrong": []}
    state_groups: dict[str, list[torch.Tensor]] = {"correct": [], "wrong": []}
    for p in usable:
        label = "correct" if p.get("correct") else "wrong"
        groups[label].append((p["acts_gen"].float() - p["acts_pre"].float()).cpu())
        state_groups[label].append(p["acts_gen"].float().cpu())

    mean_delta: dict[str, torch.Tensor] = {}
    mean_state: dict[str, torch.Tensor] = {}
    for label, deltas in groups.items():
        m = mean_tensor(deltas)
        if m is None:
            continue
        mean_delta[label] = m
        s = mean_tensor(state_groups[label])
        if s is not None:
            mean_state[label] = s
        for layer in range(n_layers):
            add_edge(
                payload,
                family="outcome_pre_to_generation",
                source="pre_reasoning_state",
                target=f"generated_{label}_state",
                layer=layer,
                vector=m[layer],
                count=len(deltas),
                metadata={"source_file": source, "outcome": label},
            )

    if "correct" in mean_delta and "wrong" in mean_delta:
        for layer in range(n_layers):
            add_edge(
                payload,
                family="outcome_motion_contrast",
                source="wrong_generation_motion",
                target="correct_generation_motion",
                layer=layer,
                vector=mean_delta["correct"][layer] - mean_delta["wrong"][layer],
                count=min(len(groups["correct"]), len(groups["wrong"])),
                metadata={"source_file": source},
            )
    if "correct" in mean_state and "wrong" in mean_state:
        for layer in range(n_layers):
            add_edge(
                payload,
                family="outcome_state_contrast",
                source="generated_wrong_state",
                target="generated_correct_state",
                layer=layer,
                vector=mean_state["correct"][layer] - mean_state["wrong"][layer],
                count=min(len(state_groups["correct"]), len(state_groups["wrong"])),
                metadata={"source_file": source},
            )


def build_sample_motion(payload: dict[str, Any], points: list[dict[str, Any]], source: str, family_prefix: str) -> None:
    usable = [p for p in points if isinstance(p.get("states"), torch.Tensor)]
    if not usable:
        return
    n_layers = int(usable[0]["states"].shape[1])
    deviations: dict[str, list[torch.Tensor]] = {"correct": [], "wrong": []}
    for p in usable:
        states = p["states"].float().cpu()
        oks = list(p.get("oks") or [])
        if len(oks) != states.shape[0]:
            continue
        center = states.mean(0, keepdim=True)
        for sample_idx, ok in enumerate(oks):
            label = "correct" if ok else "wrong"
            deviations[label].append(states[sample_idx] - center.squeeze(0))

    means: dict[str, torch.Tensor] = {}
    for label, vals in deviations.items():
        m = mean_tensor(vals)
        if m is None:
            continue
        means[label] = m
        for layer in range(n_layers):
            add_edge(
                payload,
                family=f"{family_prefix}_centroid_to_sample",
                source="question_reply_centroid",
                target=f"{label}_sample_reply",
                layer=layer,
                vector=m[layer],
                count=len(vals),
                metadata={"source_file": source, "outcome": label},
            )

    if "correct" in means and "wrong" in means:
        for layer in range(n_layers):
            add_edge(
                payload,
                family=f"{family_prefix}_sample_contrast",
                source="wrong_sample_reply",
                target="correct_sample_reply",
                layer=layer,
                vector=means["correct"][layer] - means["wrong"][layer],
                count=min(len(deviations["correct"]), len(deviations["wrong"])),
                metadata={"source_file": source},
            )


def build_concept_motion(payload: dict[str, Any], points: list[dict[str, Any]], source: str) -> None:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for p in points:
        pair_id = p.get("pair_id")
        variant = p.get("variant")
        if pair_id and variant and isinstance(p.get("acts"), torch.Tensor):
            by_pair[str(pair_id)][str(variant)] = p
    pairs = [
        group for group in by_pair.values()
        if "neutralized" in group and "standard" in group
    ]
    if not pairs:
        return
    n_layers = int(pairs[0]["standard"]["acts"].shape[0])
    by_category: dict[str, list[torch.Tensor]] = defaultdict(list)
    all_deltas: list[torch.Tensor] = []
    for group in pairs:
        std = group["standard"]
        neu = group["neutralized"]
        delta = std["acts"].float().cpu() - neu["acts"].float().cpu()
        all_deltas.append(delta)
        by_category[str(std.get("category", "unknown"))].append(delta)

    all_mean = mean_tensor(all_deltas)
    if all_mean is not None:
        for layer in range(n_layers):
            add_edge(
                payload,
                family="wording_neutralized_to_standard",
                source="neutralized_word_problem_state",
                target="standard_word_problem_state",
                layer=layer,
                vector=all_mean[layer],
                count=len(all_deltas),
                metadata={"source_file": source},
            )
    for category, deltas in sorted(by_category.items()):
        m = mean_tensor(deltas)
        if m is None:
            continue
        for layer in range(n_layers):
            add_edge(
                payload,
                family="concept_wording_category_motion",
                source=f"{category}_neutralized",
                target=f"{category}_standard",
                layer=layer,
                vector=m[layer],
                count=len(deltas),
                metadata={"source_file": source, "category": category},
            )


def load_stage_state_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            obj = torch.load(path, map_location="cpu")
        except Exception:
            continue
        if not isinstance(obj, list):
            continue
        for item in obj:
            if isinstance(item, dict) and isinstance(item.get("state"), torch.Tensor):
                item = dict(item)
                item["source_file"] = str(path)
                records.append(item)
    return records


def build_humble_stage_motion(payload: dict[str, Any], records: list[dict[str, Any]]) -> None:
    by_attempt: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record.get("method"),
            record.get("question_key"),
            record.get("attempt_index"),
            record.get("round_index"),
        )
        by_attempt[key][str(record.get("state_name"))] = record

    transitions = [
        ("solver_prompt_pre", "solver_response_mean", "humble_solver_generation_motion"),
        ("verifier_prompt_pre", "verifier_response_mean", "humble_verifier_generation_motion"),
        ("solver_response_mean", "verifier_response_mean", "humble_solver_to_verifier_motion"),
    ]

    for key, states in sorted(by_attempt.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        method, question_key, attempt_index, round_index = key
        for start_name, end_name, family in transitions:
            start = states.get(start_name)
            end = states.get(end_name)
            if not start or not end:
                continue
            a = start["state"].float().cpu()
            b = end["state"].float().cpu()
            if a.shape != b.shape or a.ndim != 2:
                continue
            for layer in range(int(a.shape[0])):
                add_edge(
                    payload,
                    family=family,
                    source=start_name,
                    target=end_name,
                    layer=layer,
                    vector=b[layer] - a[layer],
                    count=1,
                    metadata={
                        "source_file": end.get("source_file") or start.get("source_file"),
                        "method": method,
                        "question_key": question_key,
                        "attempt_index": attempt_index,
                        "round_index": round_index,
                        "mode": end.get("mode") or start.get("mode"),
                        "accepted": end.get("accepted"),
                        "verdict": end.get("verdict"),
                        "acceptance_reason": end.get("acceptance_reason"),
                    },
                )


def load_probe_vectors() -> dict[str, Any]:
    probes: dict[str, Any] = {}
    for path in sorted(INV.glob("*vector*.pt")):
        try:
            probes[path.stem] = torch.load(path, map_location="cpu")
        except Exception:
            continue
    return probes


def probe_for_layer(obj: Any, layer: int) -> torch.Tensor | None:
    if isinstance(obj, dict):
        value = obj.get(layer)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().float().reshape(-1)
        return None
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().float()
        if t.ndim == 1:
            return t.reshape(-1)
        if t.ndim >= 2 and t.shape[0] > layer:
            return t[layer].reshape(-1)
    return None


def add_probe_alignments(payload: dict[str, Any], top_k_per_edge: int = 5) -> None:
    probes = load_probe_vectors()
    alignments: list[dict[str, Any]] = []
    for edge in payload["edges"]:
        vec = payload["vectors"][edge["id"]]
        rows = []
        for name, obj in probes.items():
            probe = probe_for_layer(obj, int(edge["layer"]))
            if probe is None:
                continue
            c = cosine(vec, probe)
            if c is not None:
                rows.append({"probe": name, "cosine": c})
        rows.sort(key=lambda r: abs(r["cosine"]), reverse=True)
        edge["top_probe_alignments"] = rows[:top_k_per_edge]
        for row in rows[:top_k_per_edge]:
            alignments.append(
                {
                    "edge": edge["id"],
                    "family": edge["family"],
                    "source": edge["source"],
                    "target": edge["target"],
                    "layer": edge["layer"],
                    "probe": row["probe"],
                    "cosine": row["cosine"],
                    "magnitude": edge["magnitude"],
                }
            )
    alignments.sort(key=lambda r: abs(r["cosine"]), reverse=True)
    payload["probe_alignment_leaderboard"] = alignments[:80]


def summarize_edges(edges: list[dict[str, Any]], top_k: int = 16) -> list[dict[str, Any]]:
    return sorted(edges, key=lambda e: e["magnitude"], reverse=True)[:top_k]


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    families = defaultdict(int)
    for edge in payload["edges"]:
        families[edge["family"]] += 1

    lines = [
        "# Latent Motion Map",
        "",
        "This artifact stores directed latent displacements. It is a vector map, not a matrix of pairwise similarities.",
        "",
        "## Sources",
        "",
    ]
    for name, src in payload["sources"].items():
        lines.append(f"- {name}: `{src or 'missing'}`")

    lines.extend(["", "## Edge Families", ""])
    for family, count in sorted(families.items()):
        lines.append(f"- `{family}`: {count} layerwise directed vectors")

    lines.extend(["", "## Largest Motions", ""])
    for edge in summarize_edges(payload["edges"]):
        lines.append(
            f"- `{edge['family']}` L{edge['layer']}: `{edge['source']}` -> "
            f"`{edge['target']}` | norm={edge['magnitude']:.3f} | n={edge['count']}"
        )

    lines.extend(["", "## Strong Probe Alignments", ""])
    if payload.get("probe_alignment_leaderboard"):
        for row in payload["probe_alignment_leaderboard"][:20]:
            lines.append(
                f"- {row['cosine']:+.3f} with `{row['probe']}`: "
                f"`{row['family']}` L{row['layer']} `{row['source']}` -> `{row['target']}`"
            )
    else:
        lines.append("- none available")

    lines.extend(
        [
            "",
            "## Read This Carefully",
            "",
            "- `outcome_pre_to_generation` is the cleanest motion: before-answer state to generated-answer state.",
            "- `*_centroid_to_sample` says how sampled replies move away from a problem's own center.",
            "- `wording_neutralized_to_standard` is a wording/semantics transition, not an outcome transition.",
            "- This still does not fully capture verifier-induced abandonment. For that, the benchmark must save attempt-stage states: solver initial, solver final, verifier initial, verifier final, repair final.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(OUT))
    parser.add_argument("--outcome", default="", help="latent_outcome_points_*.pt path. Defaults to latest with pre/gen states.")
    parser.add_argument("--uncertainty", default="", help="latent_uncertainty_points_*.pt path. Defaults to latest.")
    parser.add_argument("--confidence", default="", help="latent_confidence_points_*.pt path. Defaults to latest if present.")
    parser.add_argument("--concept", default="", help="latent_concept_points_*.pt path. Defaults to latest.")
    parser.add_argument(
        "--stage-states-glob",
        default="humble_stage_states_*.pt",
        help="Glob for opt-in humble stage-state files to add as directed transitions.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    outcome_path = Path(args.outcome) if args.outcome else latest("latent_outcome_points_*.pt", out_dir)
    uncertainty_path = Path(args.uncertainty) if args.uncertainty else latest("latent_uncertainty_points_*.pt", out_dir)
    confidence_path = Path(args.confidence) if args.confidence else latest("latent_confidence_points_*.pt", out_dir)
    concept_path = Path(args.concept) if args.concept else latest("latent_concept_points_*.pt", out_dir)
    stage_state_paths = sorted(out_dir.glob(args.stage_states_glob), key=lambda p: p.stat().st_mtime)

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "outcome": str(outcome_path) if outcome_path else None,
            "uncertainty": str(uncertainty_path) if uncertainty_path else None,
            "confidence": str(confidence_path) if confidence_path else None,
            "concept": str(concept_path) if concept_path else None,
            "stage_states": [str(p) for p in stage_state_paths],
        },
        "nodes": {},
        "edges": [],
        "vectors": {},
    }

    build_outcome_motion(payload, load_points(outcome_path), str(outcome_path) if outcome_path else "")
    build_sample_motion(payload, load_points(uncertainty_path), str(uncertainty_path) if uncertainty_path else "", "uncertainty")
    build_sample_motion(payload, load_points(confidence_path), str(confidence_path) if confidence_path else "", "confidence")
    build_concept_motion(payload, load_points(concept_path), str(concept_path) if concept_path else "")
    build_humble_stage_motion(payload, load_stage_state_records(stage_state_paths))
    add_probe_alignments(payload)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pt_path = out_dir / f"latent_motion_map_{stamp}.pt"
    json_path = out_dir / f"latent_motion_map_{stamp}.json"
    md_path = out_dir / f"latent_motion_map_{stamp}.md"

    torch.save(payload, pt_path)
    json_payload = {
        k: v
        for k, v in payload.items()
        if k != "vectors"
    }
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    write_markdown(md_path, json_payload)

    print(f"Wrote {pt_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Edges: {len(payload['edges'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
