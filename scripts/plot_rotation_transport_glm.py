"""Render completed angular response-transport results, without model fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


DEFAULT_RESULTS = Path(
    "invariants/out/glm_capture_trial_20261002/rotation_transport_20261004/results.json"
)
CANDIDATES = (
    ("all_int4", "INT4 baseline", "baseline", "#93999e"),
    ("global", "Global correction", "global", "#3277b3"),
    ("region_offsets", "Regional offsets", "offset", "#94b6cf"),
    ("radial_transport", "Radial quadratic map", "radial", "#398e9a"),
    ("rotation_transport", "Rotation-conjugated map", "rotation", "#218a64"),
    ("wrong_rotation_transport", "Wrong-rotation control", "rotation control", "#d0784c"),
    ("fourier_transport", "Fourier map (larger capacity)", "Fourier", "#8972b3"),
    ("wrong_fourier_transport", "Wrong-Fourier control", "Fourier control", "#ad8173"),
    ("separate_angle_transport", "Separate angular-sector maps", "table", "#b46a9a"),
)


def _integer(value, field, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _metric(row, metric):
    value = row["metrics"][metric]["relative_rmse"]
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(value) or value < 0):
        raise ValueError(f"{row['name']}.{metric}.relative_rmse must be finite and nonnegative")
    return float(value)


def prepare_results(data):
    """Validate a complete nine-candidate receipt, preserving declared order."""
    if data.get("format") != "angular-glm-response-results-v1":
        raise ValueError("expected completed angular response-transport results")
    rows = data["candidates"]
    if not isinstance(rows, list) or len(rows) != len(CANDIDATES):
        raise ValueError("figure requires all nine declared candidates")
    by_name = {row["name"]: row for row in rows}
    if set(by_name) != {name for name, _, _, _ in CANDIDATES} or len(by_name) != len(rows):
        raise ValueError("candidate names must be unique and match the declared comparison")
    ordered = [by_name[name] for name, _, _, _ in CANDIDATES]
    base_bytes = _integer(data["base_int4_serialized_bytes"], "base_int4_serialized_bytes", positive=True)
    for row in ordered:
        _metric(row, "output")
        _metric(row, "withheld_angle_output")
        adapter = _integer(row["adapter_serialized_bytes"], f"{row['name']}.adapter_serialized_bytes")
        total = _integer(row["sampled_block_serialized_bytes"], f"{row['name']}.sampled_block_serialized_bytes", positive=True)
        if total != base_bytes + adapter:
            raise ValueError("total storage must equal the unchanged INT4 base plus the adapter")
    if ordered[0]["adapter_serialized_bytes"] != 0:
        raise ValueError("the INT4 reference must not contain a correction adapter")
    for field in ("calibration_records", "heldout_records", "heldout_pairs", "angle_bins"):
        _integer(data[field], field, positive=True)
    sector = _integer(data["withheld_angle_bin"], "withheld_angle_bin")
    if sector >= data["angle_bins"]:
        raise ValueError("withheld_angle_bin must be inside the declared sector range")
    counts = []
    for row in ordered:
        withheld = _integer(row["metrics"]["withheld_angle_records"], "withheld_angle_records", positive=True)
        supported = _integer(row["metrics"]["supported_withheld_angle_records"], "supported_withheld_angle_records")
        if supported > withheld or withheld > data["heldout_records"]:
            raise ValueError("withheld sector counts exceed their parent selection")
        counts.append((withheld, supported))
    if len(set(counts)) != 1:
        raise ValueError("all variants must use the identical withheld-sector selection")
    return ordered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, required=True, help="new directory for PNG, SVG, table and receipt")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("choose a new plot directory; existing artifacts are preserved")
    if args.results.stat().st_size > 64 << 20:
        raise ValueError("comparison JSON exceeds the bounded renderer input size")
    source = args.results.read_bytes()
    data = json.loads(source)
    rows = prepare_results(data)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    labels = [f"{label}\n{row['adapter_serialized_bytes'] / 1e6:.3f} MB adapter · "
              f"{row['sampled_block_serialized_bytes'] / 1e6:.2f} MB total"
              for row, (_, label, _, _) in zip(rows, CANDIDATES)]
    positions = np.arange(len(rows))
    colors = [color for _, _, _, color in CANDIDATES]
    withheld = rows[0]["metrics"]["withheld_angle_records"]
    supported = rows[0]["metrics"]["supported_withheld_angle_records"]
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 10.1), sharey=True)
    plotted = {}
    titles = (
        "All development held-out states",
        f"Withheld angular sector {data['withheld_angle_bin']} ({withheld:,} states)",
    )
    for axis, metric, title in zip(axes, ("output", "withheld_angle_output"), titles):
        values = np.array([_metric(row, metric) * 100 for row in rows])
        plotted[metric] = dict(zip([row["name"] for row in rows], values.tolist()))
        axis.barh(positions, values, height=.63, color=colors)
        axis.set_xlim(0, max(float(values.max()), .001) * 1.18)
        axis.set_title(title, fontsize=13, pad=16)
        axis.set_xlabel("Relative expert-output RMSE (%) — lower is better", labelpad=11)
        axis.set_yticks(positions)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
        for position, value in zip(positions, values):
            axis.text(value + axis.get_xlim()[1] * .012, position, f"{value:.3f}%",
                      va="center", fontsize=10)
    axes[0].set_yticklabels(labels, fontsize=9.5)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("GLM-5.3: testing angular relationships between response-correction maps", fontsize=16, y=.972)
    fig.text(.5, .932, f"{data['calibration_records']:,} calibration states · "
             f"{data['heldout_records']:,} development held-out states · "
             f"{data['heldout_pairs']:,} adjacent pairs · CPU FP32 sampled expert functions",
             ha="center", color="#444444", fontsize=10)
    fig.text(.5, .906, f"Withheld-sector states include global fallback; "
             f"{supported:,}/{withheld:,} have common supported angular geometry.",
             ha="center", color="#444444", fontsize=9)
    fig.text(.025, .04,
             "Measured dequantized SiLU expert functions; all three projections use the same saved INT4 base plus an output correction.\n"
             "Calibration inputs define regions/planes; every withheld-sector response target is excluded from correction fitting.\n"
             "Fourier fits have larger coefficient capacity; correct/wrong angular controls reuse identical fitted tensors.\n"
             "Actual storage includes the three sampled projection artifacts and adapter files (decimal MB); state RMSE uses recorded trajectory weights.\n"
             "Static angular maps are periodic; observed path winding is a separate audit. This development set was previously inspected.\n"
             "Eight sampled experts are weighted equally. No routing, decoder/logits, generation, native GPU kernel or semantic transfer proof.",
             fontsize=8.8, color="#444444", va="bottom")
    fig.subplots_adjust(top=.847, bottom=.225, left=.25, right=.97, wspace=.22)

    args.out.mkdir(parents=True, exist_ok=False)
    try:
        for suffix in ("png", "svg"):
            fig.savefig(args.out / f"comparison.{suffix}", dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    with (args.out / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "label", "family", "adapter_serialized_bytes", "sampled_block_serialized_bytes",
                         "output_relative_rmse", "withheld_angle_output_relative_rmse"))
        for row, (_, label, family, _) in zip(rows, CANDIDATES):
            writer.writerow((row["name"], label, family, row["adapter_serialized_bytes"],
                             row["sampled_block_serialized_bytes"], _metric(row, "output"),
                             _metric(row, "withheld_angle_output")))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(args.out.iterdir()) if p.is_file()}
    fit = data.get("fit", {})
    manifest = {
        "format": "angular-response-comparison-plot-v1",
        "source": str(args.results.resolve()), "source_sha256": hashlib.sha256(source).hexdigest(),
        "renderer": str(Path(__file__).resolve()),
        "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__, "storage_unit": "decimal MB (1,000,000 bytes)",
        "evaluation_scope": data.get("evaluation_scope"), "heldout_status": data.get("heldout_status"),
        "model_behavior_or_semantic_transfer_established": False,
        "calibration_records": data["calibration_records"], "heldout_records": data["heldout_records"],
        "heldout_pairs": data["heldout_pairs"], "angle_bins": data["angle_bins"],
        "withheld_angle_bin": data["withheld_angle_bin"], "withheld_angle_records": withheld,
        "supported_withheld_angle_records": supported, "winding": data.get("winding"),
        "fitcounts": {field: fit.get(field) for field in
                      ("region_counts", "permitted_response_region_counts", "angle_bin_counts", "region_enabled")},
        "plotted_relative_rmse_percent": plotted,
        "candidates": [{"name": row["name"], "adapter_serialized_bytes": row["adapter_serialized_bytes"],
                        "sampled_block_serialized_bytes": row["sampled_block_serialized_bytes"]} for row in rows],
        "files": files,
    }
    (args.out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(args.out.resolve()), "files": list(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
