"""Plot the completed scoped regional response comparison (no fitting)."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("choose a new plot directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    data = json.loads(args.results.read_bytes())
    rows = data["candidates"]
    labels = ["INT4", "Global", "Region\noffsets", "Shared\nmaps", "Identity\nmaps", "Wrong\nregion", "Separate\nmaps"]
    if [r["name"] for r in rows] != ["all_int4", "global", "region_offsets", "shared_transport",
                                        "identity_transport", "wrong_region_transport", "independent_local"]:
        raise ValueError("figure requires the complete declared candidate set")
    point = np.array([r["metrics"]["output"]["relative_rmse"] * 100 for r in rows])
    change = np.array([r["metrics"]["output_difference"]["relative_rmse"] * 100 for r in rows])
    sizes = np.array([r["sampled_block_serialized_bytes"] / 1e6 for r in rows] +
                     [data["original_sampled_three_projection_bytes"] / 1e6])
    if not all(np.isfinite(v).all() for v in (point, change, sizes)):
        raise ValueError("only finite complete measured results can be plotted")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    positions = np.arange(len(rows))
    axes[0].bar(positions - .18, point, .36, label="Output", color="#3277b3")
    axes[0].bar(positions + .18, change, .36, label="Output difference", color="#94b6cf")
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("Relative RMSE (%) — lower is better")
    axes[0].set_title("Held-out numerical response error")
    axes[0].set_ylim(0, max(change.max(), point.max()) * 1.2)
    axes[0].grid(axis="y", alpha=.18)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="upper left")
    bars = axes[1].bar(np.arange(len(sizes)), sizes,
                       color=["#3277b3", "#94b6cf", "#94b6cf", "#289776", "#94b6cf", "#d47c52", "#8875b4", "#b0b0b0"])
    axes[1].set_xticks(np.arange(len(sizes)), labels + ["Original\nFP8"])
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].set_ylabel("Storage (decimal MB)")
    axes[1].set_title("All three sampled projections + adapter")
    axes[1].set_ylim(0, sizes.max() * 1.15)
    axes[1].grid(axis="y", alpha=.18)
    axes[1].set_axisbelow(True)
    for bar, size in zip(bars, sizes):
        axes[1].text(bar.get_x()+bar.get_width()/2, size+4, f"{size:.1f}", ha="center", fontsize=9)
    fig.suptitle("GLM-5.3: reusable maps between activation neighborhoods", fontsize=16, y=.98)
    fig.text(.5, .905, f"Layer 3 · eight sampled experts · {data['heldout_records']:,} held-out states · "
             f"{data['heldout_pairs']:,} adjacent pairs · CPU FP32", ha="center", color="#444444")
    fig.text(.03, .05, "Shared maps fit experts 0–6; expert 7's global correction remains calibrated. "
             "Four geometric regions; two have enough support.\n"
             "All experts receive every input equally. Generation, native FP8 kernels, routing and semantic meanings were not tested.",
             fontsize=9, color="#444444", va="bottom")
    fig.subplots_adjust(top=.83, bottom=.22, left=.065, right=.98, wspace=.25)
    args.out.mkdir(parents=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.out / f"comparison.{suffix}", dpi=160, facecolor="white")
    plt.close(fig)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in args.out.iterdir()}
    (args.out / "plot_manifest.json").write_text(json.dumps({"source": str(args.results.resolve()),
        "source_sha256": hashlib.sha256(args.results.read_bytes()).hexdigest(),
        "candidates": [r["name"] for r in rows], "files": files}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
