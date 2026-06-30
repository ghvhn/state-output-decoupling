"""
Map correlated and anti-correlated vector-like artifacts.

This is intentionally offline: it loads saved `.pt` artifacts, extracts
residual-stream-shaped vectors, and computes cosine geometry without loading a
language model. It treats the vector as evidence of a trajectory, not the
trajectory itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_ROOT = Path("invariants")
DEFAULT_OUT_DIR = Path("invariants/out")
VECTOR_DIM_HINT = 4096


@dataclass
class VectorEntry:
    label: str
    source: str
    kind: str
    vector: torch.Tensor
    layer: int | None = None
    index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def norm(self) -> float:
        return float(self.vector.float().norm().item())

    @property
    def dim(self) -> int:
        return int(self.vector.numel())


def safe_meta(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): safe_meta(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_meta(v) for v in value[:20]]
    return str(value)


def is_vector_like(tensor: torch.Tensor, dim_hint: int = VECTOR_DIM_HINT) -> bool:
    if not isinstance(tensor, torch.Tensor):
        return False
    if tensor.numel() == 0:
        return False
    if tensor.numel() == dim_hint:
        return True
    if tensor.ndim >= 1 and tensor.shape[-1] == dim_hint:
        return True
    return False


def flatten_vector(tensor: torch.Tensor, dim_hint: int = VECTOR_DIM_HINT) -> torch.Tensor:
    t = tensor.detach().cpu().float()
    if t.numel() == dim_hint:
        return t.reshape(-1).contiguous()
    if t.ndim >= 1 and t.shape[-1] == dim_hint:
        return t.reshape(-1, t.shape[-1])[-1].contiguous()
    raise ValueError(f"Not vector-like: shape={tuple(t.shape)} numel={t.numel()}")


def extract_from_dict(path: Path, obj: dict[Any, Any], dim_hint: int) -> tuple[list[VectorEntry], str | None]:
    entries: list[VectorEntry] = []
    numeric_tensor_items = [
        (k, v)
        for k, v in obj.items()
        if isinstance(k, int) and isinstance(v, torch.Tensor) and is_vector_like(v, dim_hint)
    ]
    if numeric_tensor_items:
        for layer, tensor in sorted(numeric_tensor_items, key=lambda kv: kv[0]):
            entries.append(
                VectorEntry(
                    label=f"{path.stem}[L{layer}]",
                    source=str(path),
                    kind="layer_vector",
                    layer=int(layer),
                    vector=flatten_vector(tensor, dim_hint),
                )
            )
        return entries, None

    cache_like = "trigger" in obj and ("delta" in obj or "learned_vector" in obj)
    if cache_like:
        meta = obj.get("metadata", {})
        metadata = safe_meta(meta) if isinstance(meta, dict) else {"metadata": safe_meta(meta)}
        if is_vector_like(obj.get("trigger"), dim_hint):
            entries.append(
                VectorEntry(
                    label=f"{path.stem}:trigger",
                    source=str(path),
                    kind="cache_trigger",
                    vector=flatten_vector(obj["trigger"], dim_hint),
                    metadata=metadata,
                )
            )
        delta = obj.get("delta", obj.get("learned_vector"))
        if is_vector_like(delta, dim_hint):
            entries.append(
                VectorEntry(
                    label=f"{path.stem}:delta",
                    source=str(path),
                    kind="cache_delta",
                    vector=flatten_vector(delta, dim_hint),
                    metadata=metadata,
                )
            )
        return entries, None if entries else "cache-like dict had no vector-like trigger/delta"

    tensor_items = [
        (k, v)
        for k, v in obj.items()
        if isinstance(v, torch.Tensor) and is_vector_like(v, dim_hint)
    ]
    if tensor_items:
        for key, tensor in sorted(tensor_items, key=lambda kv: str(kv[0])):
            entries.append(
                VectorEntry(
                    label=f"{path.stem}:{key}",
                    source=str(path),
                    kind="dict_tensor",
                    vector=flatten_vector(tensor, dim_hint),
                    metadata={"dict_key": safe_meta(key)},
                )
            )
        return entries, None

    return entries, "dict contained no vector-like tensors"


def extract_from_list(path: Path, obj: list[Any], dim_hint: int) -> tuple[list[VectorEntry], str | None]:
    entries: list[VectorEntry] = []
    for i, item in enumerate(obj):
        if isinstance(item, dict):
            meta = item.get("metadata", {})
            metadata = safe_meta(meta) if isinstance(meta, dict) else {"metadata": safe_meta(meta)}
            tag = metadata.get("tag") or metadata.get("teaching_signal") or metadata.get("reason") or "entry"
            if is_vector_like(item.get("trigger"), dim_hint):
                entries.append(
                    VectorEntry(
                        label=f"{path.stem}[{i}]:trigger:{tag}",
                        source=str(path),
                        kind="cache_trigger",
                        index=i,
                        vector=flatten_vector(item["trigger"], dim_hint),
                        metadata=metadata,
                    )
                )
            delta = item.get("delta", item.get("learned_vector"))
            if is_vector_like(delta, dim_hint):
                entries.append(
                    VectorEntry(
                        label=f"{path.stem}[{i}]:delta:{tag}",
                        source=str(path),
                        kind="cache_delta",
                        index=i,
                        vector=flatten_vector(delta, dim_hint),
                        metadata=metadata,
                    )
                )
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            for slot, tensor in (("trigger", item[0]), ("delta", item[1])):
                if is_vector_like(tensor, dim_hint):
                    entries.append(
                        VectorEntry(
                            label=f"{path.stem}[{i}]:{slot}:legacy_tuple",
                            source=str(path),
                            kind=f"cache_{slot}",
                            index=i,
                            vector=flatten_vector(tensor, dim_hint),
                            metadata={"format": "legacy_tuple"},
                        )
                    )
        elif isinstance(item, torch.Tensor) and is_vector_like(item, dim_hint):
            entries.append(
                VectorEntry(
                    label=f"{path.stem}[{i}]",
                    source=str(path),
                    kind="list_tensor",
                    index=i,
                    vector=flatten_vector(item, dim_hint),
                )
            )
    if entries:
        return entries, None
    return entries, "list contained no vector-like entries"


def load_entries(path: Path, dim_hint: int) -> tuple[list[VectorEntry], str | None]:
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as exc:
        return [], f"load failed: {exc}"

    if isinstance(obj, torch.Tensor):
        if is_vector_like(obj, dim_hint):
            return [
                VectorEntry(
                    label=path.stem,
                    source=str(path),
                    kind="tensor_vector",
                    vector=flatten_vector(obj, dim_hint),
                )
            ], None
        return [], f"tensor not vector-like: shape={tuple(obj.shape)} numel={obj.numel()}"
    if isinstance(obj, dict):
        return extract_from_dict(path, obj, dim_hint)
    if isinstance(obj, list):
        return extract_from_list(path, obj, dim_hint)
    return [], f"unsupported object type: {type(obj).__name__}"


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        return float("nan")
    if a.float().norm().item() == 0 or b.float().norm().item() == 0:
        return float("nan")
    return float(F.cosine_similarity(a.float(), b.float(), dim=0).item())


def aggregate_by_artifact(entries: list[VectorEntry]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[VectorEntry]] = {}
    for entry in entries:
        base = Path(entry.source).stem
        if entry.kind.startswith("cache_"):
            base = f"{base}:{entry.kind}"
        groups.setdefault(base, []).append(entry)

    summaries: dict[str, dict[str, Any]] = {}
    for name, items in groups.items():
        layers = sorted({item.layer for item in items if item.layer is not None})
        summaries[name] = {
            "count": len(items),
            "kinds": sorted({item.kind for item in items}),
            "sources": sorted({item.source for item in items}),
            "layers": layers,
            "norm_mean": float(sum(item.norm for item in items) / max(len(items), 1)),
            "norm_min": float(min(item.norm for item in items)),
            "norm_max": float(max(item.norm for item in items)),
        }
    return summaries


def pairwise(entries: list[VectorEntry], max_pairs: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(entries)
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = entries[i]
            b = entries[j]
            if a.dim != b.dim:
                continue
            c = cosine(a.vector, b.vector)
            if math.isnan(c):
                continue
            rows.append(
                {
                    "a": a.label,
                    "b": b.label,
                    "cosine": c,
                    "a_kind": a.kind,
                    "b_kind": b.kind,
                    "a_source": a.source,
                    "b_source": b.source,
                    "a_layer": a.layer,
                    "b_layer": b.layer,
                    "a_index": a.index,
                    "b_index": b.index,
                }
            )
            pair_count += 1
            if max_pairs is not None and pair_count >= max_pairs:
                return rows
    return rows


def artifact_pairwise(entries: list[VectorEntry]) -> list[dict[str, Any]]:
    groups: dict[str, list[VectorEntry]] = {}
    for entry in entries:
        base = Path(entry.source).stem
        if entry.kind.startswith("cache_"):
            base = f"{base}:{entry.kind}"
        groups.setdefault(base, []).append(entry)

    names = sorted(groups)
    rows: list[dict[str, Any]] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            sims: list[dict[str, Any]] = []
            for a in groups[name_a]:
                for b in groups[name_b]:
                    if a.dim != b.dim:
                        continue
                    if a.layer is not None and b.layer is not None and a.layer != b.layer:
                        continue
                    c = cosine(a.vector, b.vector)
                    if not math.isnan(c):
                        sims.append({"cosine": c, "a": a.label, "b": b.label, "layer": a.layer})
            if not sims:
                continue
            values = [s["cosine"] for s in sims]
            strongest = max(sims, key=lambda s: abs(s["cosine"]))
            rows.append(
                {
                    "a": name_a,
                    "b": name_b,
                    "count": len(values),
                    "mean": float(sum(values) / len(values)),
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "strongest": strongest,
                }
            )
    return rows


def write_markdown(
    path: Path,
    entries: list[VectorEntry],
    skipped: list[dict[str, str]],
    artifacts: dict[str, dict[str, Any]],
    artifact_pairs: list[dict[str, Any]],
    raw_pairs: list[dict[str, Any]],
) -> None:
    positive = sorted(artifact_pairs, key=lambda r: r["mean"], reverse=True)[:20]
    negative = sorted(artifact_pairs, key=lambda r: r["mean"])[:20]
    strongest_positive = sorted(raw_pairs, key=lambda r: r["cosine"], reverse=True)[:20]
    strongest_negative = sorted(raw_pairs, key=lambda r: r["cosine"])[:20]

    lines: list[str] = [
        "# Vector Geometry Map",
        "",
        "This map is offline and correlational. It does not prove causality; it shows which saved directions currently align, oppose, or look unrelated.",
        "",
        "## Inventory",
        "",
        f"- vector entries extracted: `{len(entries)}`",
        f"- artifact groups: `{len(artifacts)}`",
        f"- skipped `.pt` files: `{len(skipped)}`",
        "",
        "| artifact | count | kinds | layers | norm mean |",
        "|---|---:|---|---|---:|",
    ]
    for name, info in sorted(artifacts.items()):
        layer_text = ",".join(str(x) for x in info["layers"][:8])
        if len(info["layers"]) > 8:
            layer_text += ",..."
        lines.append(
            f"| `{name}` | {info['count']} | `{', '.join(info['kinds'])}` | `{layer_text}` | {info['norm_mean']:.4f} |"
        )

    lines.extend(["", "## Top Artifact Correlations", ""])
    lines.extend(["| mean cosine | artifact A | artifact B | strongest entry pair |", "|---:|---|---|---|"])
    for row in positive:
        s = row["strongest"]
        lines.append(
            f"| {row['mean']:+.4f} | `{row['a']}` | `{row['b']}` | `{s['a']}` vs `{s['b']}` ({s['cosine']:+.4f}) |"
        )

    lines.extend(["", "## Top Artifact Anti-Correlations", ""])
    lines.extend(["| mean cosine | artifact A | artifact B | strongest entry pair |", "|---:|---|---|---|"])
    for row in negative:
        s = row["strongest"]
        lines.append(
            f"| {row['mean']:+.4f} | `{row['a']}` | `{row['b']}` | `{s['a']}` vs `{s['b']}` ({s['cosine']:+.4f}) |"
        )

    lines.extend(["", "## Strongest Entry-Level Alignments", ""])
    lines.extend(["| cosine | entry A | entry B |", "|---:|---|---|"])
    for row in strongest_positive:
        lines.append(f"| {row['cosine']:+.4f} | `{row['a']}` | `{row['b']}` |")

    lines.extend(["", "## Strongest Entry-Level Oppositions", ""])
    lines.extend(["| cosine | entry A | entry B |", "|---:|---|---|"])
    for row in strongest_negative:
        lines.append(f"| {row['cosine']:+.4f} | `{row['a']}` | `{row['b']}` |")

    if skipped:
        lines.extend(["", "## Skipped Files", ""])
        lines.extend(["| file | reason |", "|---|---|"])
        for item in skipped:
            lines.append(f"| `{item['path']}` | {item['reason']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Directory to scan for .pt files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for JSON/Markdown output.")
    parser.add_argument("--dim", type=int, default=VECTOR_DIM_HINT, help="Residual stream dimension hint.")
    parser.add_argument("--max-raw-pairs", type=int, default=None, help="Optional cap for entry-level pairs.")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[VectorEntry] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.pt")):
        found, reason = load_entries(path, args.dim)
        entries.extend(found)
        if reason:
            skipped.append({"path": str(path), "reason": reason})

    artifacts = aggregate_by_artifact(entries)
    raw_pairs = pairwise(entries, max_pairs=args.max_raw_pairs)
    artifact_pairs = artifact_pairwise(entries)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"vector_geometry_map_{stamp}.json"
    md_path = out_dir / f"vector_geometry_map_{stamp}.md"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "dim": args.dim,
        "entries": [
            {
                "label": e.label,
                "source": e.source,
                "kind": e.kind,
                "layer": e.layer,
                "index": e.index,
                "dim": e.dim,
                "norm": e.norm,
                "metadata": e.metadata,
            }
            for e in entries
        ],
        "artifacts": artifacts,
        "artifact_pairs": artifact_pairs,
        "entry_pairs": raw_pairs,
        "skipped": skipped,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(md_path, entries, skipped, artifacts, artifact_pairs, raw_pairs)

    print(f"Extracted {len(entries)} vector entries from {len(artifacts)} artifact groups.")
    print(f"Skipped {len(skipped)} .pt files that were not vector-like.")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    positives = sorted(artifact_pairs, key=lambda r: r["mean"], reverse=True)[:8]
    negatives = sorted(artifact_pairs, key=lambda r: r["mean"])[:8]
    print("\nTop correlated artifact groups:")
    for row in positives:
        print(f"  {row['mean']:+.4f}  {row['a']}  <->  {row['b']}")
    print("\nTop anti-correlated artifact groups:")
    for row in negatives:
        print(f"  {row['mean']:+.4f}  {row['a']}  <->  {row['b']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
