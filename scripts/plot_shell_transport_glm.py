"""Render completed shell-transport results without fitting any model."""

import argparse
import hashlib
import json
import math
from pathlib import Path


DEFAULT_RESULTS = Path(
    "invariants/out/glm_capture_trial_20261002/shell_transport_20261003/results.json"
)
CANDIDATES = (
    ("all_int4", "INT4 baseline", "#93999e"),
    ("global", "Global correction", "#3277b3"),
    ("region_offsets", "Regional offsets", "#94b6cf"),
    ("constant_transport", "Constant map", "#5f92b7"),
    ("linear_transport", "Linear formula", "#398e9a"),
    ("generated_transport", "Quadratic formula", "#218a64"),
    ("identity_transport", "Identity control", "#a2afb8"),
    ("wrong_shell_transport", "Wrong-level control", "#d0784c"),
    ("separate_shell_transport", "Separate level maps", "#8972b3"),
)


def _integer(value, field, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{field} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _metric(row, metric):
    value = row["metrics"][metric]["relative_rmse"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{row['name']}.{metric}.relative_rmse must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{row['name']}.{metric}.relative_rmse must be finite and nonnegative")
    return float(value)


def prepare_results(data):
    """Validate the completed measured comparison and put rows in declared order."""
    rows = data["candidates"]
    if not isinstance(rows, list) or len(rows) != len(CANDIDATES):
        raise ValueError("figure requires all nine declared candidates")
    by_name = {row["name"]: row for row in rows}
    if set(by_name) != {name for name, _, _ in CANDIDATES} or len(by_name) != len(rows):
        raise ValueError("candidate names must be unique and match the complete declared comparison")
    ordered = [by_name[name] for name, _, _ in CANDIDATES]
    for row in ordered:
        _metric(row, "output")
        _metric(row, "withheld_shell_output")
        _integer(row["adapter_serialized_bytes"], f"{row['name']}.adapter_serialized_bytes")
        _integer(row["sampled_block_serialized_bytes"], f"{row['name']}.sampled_block_serialized_bytes", positive=True)
        if row["adapter_serialized_bytes"] >= row["sampled_block_serialized_bytes"]:
            raise ValueError("sampled block storage must include more than just the adapter")
    _integer(data["calibration_records"], "calibration_records", positive=True)
    _integer(data["heldout_records"], "heldout_records", positive=True)
    _integer(data["heldout_pairs"], "heldout_pairs", positive=True)
    _integer(data["shells"], "shells", positive=True)
    level = _integer(data["withheld_shell"], "withheld_shell")
    if level >= data["shells"]:
        raise ValueError("withheld_shell must be inside the declared shell range")
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, required=True, help="new directory for PNG, SVG and receipt")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("choose a new plot directory; existing artifacts are preserved")
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
              for row, (_, label, _) in zip(rows, CANDIDATES)]
    positions = np.arange(len(rows))
    colors = [color for _, _, color in CANDIDATES]
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 9.0), sharey=True)
    plotted = {}
    withheld_records = rows[0]["metrics"].get("withheld_shell_records")
    if withheld_records is not None:
        _integer(withheld_records, "withheld_shell_records", positive=True)
    titles = (
        "All development held-out states",
        f"Withheld geometric level {data['withheld_shell']}" +
        (f" ({withheld_records:,} states)" if withheld_records is not None else ""),
    )
    for axis, metric, title in zip(axes, ("output", "withheld_shell_output"), titles):
        values = np.array([_metric(row, metric) * 100 for row in rows])
        plotted[metric] = dict(zip([row["name"] for row in rows], values.tolist()))
        axis.barh(positions, values, height=.63, color=colors)
        axis.set_xlim(0, max(float(values.max()), .001) * 1.18)
        axis.set_title(title, fontsize=13, pad=16)
        axis.set_xlabel("Relative output RMSE (%) — lower is better", labelpad=11)
        axis.set_yticks(positions)
        axis.grid(axis="x", alpha=.18)
        axis.set_axisbelow(True)
        for position, value in zip(positions, values):
            axis.text(value + axis.get_xlim()[1] * .012, position, f"{value:.2f}%",
                      va="center", fontsize=10)
    axes[0].set_yticklabels(labels, fontsize=10)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("GLM-5.3: testing response-correction formulas across geometric levels", fontsize=16, y=.97)
    fig.text(.5, .925, f"{data['calibration_records']:,} calibration states · "
             f"{data['heldout_records']:,} development held-out states · "
             f"{data['heldout_pairs']:,} adjacent pairs · CPU FP32 expert functions",
             ha="center", color="#444444", fontsize=10)
    fig.text(.025, .045,
             "Levels are measured geometric coordinates, not electron shells or established semantic categories.\n"
             "Geometry uses all calibration inputs; withheld-level response targets are excluded from correction fitting.\n"
             "Storage includes the three sampled projection artifacts and each adapter (decimal MB).\n"
             "Sampled experts are evaluated equally; routing, decoder logits, generation and native inference speed were not tested.",
             fontsize=9, color="#444444", va="bottom")
    fig.subplots_adjust(top=.85, bottom=.17, left=.235, right=.97, wspace=.22)

    # Creating a new directory is the only artifact mutation this renderer makes.
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        for suffix in ("png", "svg"):
            fig.savefig(args.out / f"comparison.{suffix}", dpi=160, facecolor="white")
    finally:
        plt.close(fig)
    files = {p.name: {"bytes": p.stat().st_size,
                      "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(args.out.iterdir()) if p.is_file()}
    fit = data.get("fit", {})
    fitcounts = data.get("fitcounts", {
        field: fit.get(field) for field in
        ("region_counts", "permitted_response_region_counts", "shell_counts", "region_enabled")
    })
    manifest = {
        "source": str(args.results.resolve()),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "scope": "CPU expert-function numerical responses; no decoder, generation or semantic validation",
        "shell_ordering": data.get("shell_ordering"),
        "withheld_shell": data["withheld_shell"], "shells": data["shells"],
        "fitcounts": fitcounts, "plotted_relative_rmse_percent": plotted,
        "candidates": [{"name": row["name"],
                        "adapter_serialized_bytes": row["adapter_serialized_bytes"],
                        "sampled_block_serialized_bytes": row["sampled_block_serialized_bytes"]}
                       for row in rows],
        "files": files,
    }
    (args.out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.out.resolve()), "files": list(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
