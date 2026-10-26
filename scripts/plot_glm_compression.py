"""Render saved layer-3 gate comparison JSON without loading model tensors.

Example (use a new output directory):
  python scripts/plot_glm_compression.py --comparison path/to/comparison --out path/to/figure

Requires Matplotlib in the invoking Python environment. PNG/SVG and an auditable
plot manifest are written locally. Undefined metrics are omitted, never zeroed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path


CAPTION = "Layer3 gate projections, dequantized linear proxy"
STYLES = {
    "weight_pca": ("Weight PCA", "#0072B2", "o"),
    "trajectory_response": ("Trajectory response fit", "#009E73", "s"),
    "probe_span": ("Native seed-span fit", "#E69F00", "^"),
    "symmetric_int4": ("INT4 baseline", "#666666", "s"),
    "probe_residual": ("INT4 + native-span repair", "#D55E00", "*"),
    "residual_calibration_pca": ("INT4 + calibration-PCA repair", "#CC79A7", "D"),
    "residual_seeded_random": ("INT4 + random-span repair", "#222222", "X"),
    "probe_pca_hybrid": ("INT4 + native/PCA hybrid", "#332288", "P"),
    "residual_calibration_pca_extended": ("INT4 + extended-PCA repair", "#AA4499", "v"),
    "residual_seeded_random_extended": ("INT4 + extended-random repair", "#666633", "h"),
}
NATIVE_REPAIRS = ("probe_residual", "residual_calibration_pca", "residual_seeded_random")
HYBRID_REPAIRS = ("probe_pca_hybrid", "residual_calibration_pca_extended", "residual_seeded_random_extended")
REPAIRS = NATIVE_REPAIRS + HYBRID_REPAIRS
PANELS = (
    ("Heldout state responses", "heldout_response", "point_relative_rmse"),
    ("Heldout state differences", "heldout_response", "contrast_relative_rmse"),
    ("Native seed-span responses", "static_seed_contrast_response", "span_response_relative_rmse"),
)
DEFINITIONS = {
    "storage": "Actual recursively serialized candidate bytes; includes metadata and all candidate tensor files. The common seed witness is separate.",
    "point_relative_rmse": "Square root of the trajectory-balanced, provenance/stimulation-weighted mean of per-state error energy divided by source response energy, across all sampled experts equally.",
    "contrast_relative_rmse": "Same weighted relative metric on next-minus-previous input states. Differences are not divided by the recorded token gap; this is not velocity.",
    "span_response_relative_rmse": "Square root of total linear-response error energy divided by source response energy on the orthonormal native seed span. Static seeds are not heldout, may be used to fit, and have unvalidated conceptual meanings.",
    "scope": "Counterfactual linear responses of all sampled experts, not router-weighted. No full decoder, language quality, native FP8 kernel, or runtime-savings evaluation.",
}


def read_json(path, limit):
    if path.stat().st_size > limit:
        raise ValueError(f"JSON exceeds bounded input size: {path}")
    raw = path.read_bytes()

    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(raw, parse_constant=reject_constant), hashlib.sha256(raw).hexdigest()


def metric(row, section, key):
    value = row.get(section, {}).get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {section}.{key} for {row['variant']}")
    return float(value)


def load_comparison(path):
    result_path = path / "results.json" if path.is_dir() else path
    settings_path = result_path.parent / "settings.json"
    results, result_hash = read_json(result_path, 16 << 20)
    settings, settings_hash = read_json(settings_path, 1 << 20)
    if not isinstance(settings, dict) or not isinstance(settings.get("identity"), dict):
        raise ValueError("settings.json must contain a source identity")
    if not isinstance(results, list) or not 1 <= len(results) <= 256:
        raise ValueError("results.json must contain 1..256 variants")
    identity = settings.get("identity", {})
    if identity.get("layer") != 3 or identity.get("projection") != "gate_proj":
        raise ValueError("This renderer's caption is scoped to layer 3 gate projections")
    variants, omissions, plotted = set(), [], []
    for row in results:
        if not isinstance(row, dict) or row.get("method") not in STYLES:
            raise ValueError("Unexpected comparison method")
        name, size = row.get("variant"), row.get("serialized_total_bytes")
        if not isinstance(name, str) or not name or name in variants:
            raise ValueError("Variant names must be nonempty and unique")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"Actual serialized bytes missing or invalid for {name}")
        variants.add(name)
        for _, section, key in PANELS:
            if section in row and not isinstance(row[section], dict):
                raise ValueError(f"Invalid metric section for {name}")
            value = metric(row, section, key)
            if value is None:
                omissions.append({"variant": name, "metric": key,
                    "reason": "metric absent or undefined; no value imputed"})
            else:
                plotted.append({"variant": name, "method": row["method"],
                    "serialized_total_bytes": size, "metric": key, "value": value})
    if not plotted:
        raise ValueError("No known finite comparison metrics to plot")
    status_path = result_path.parent / "status.json"
    status = read_json(status_path, 128 << 10)[0].get("status", "unknown") if status_path.exists() else "unknown"
    evidence_path = result_path.parent / "evidence.json"
    evidence, evidence_hash = read_json(evidence_path, 16 << 20) if evidence_path.exists() else ({}, None)
    if not isinstance(evidence, dict):
        raise ValueError("evidence.json must contain a scoped evidence object")
    selection = {}
    for split in ("calibration", "heldout"):
        source = evidence.get(split, {})
        if not isinstance(source, dict):
            raise ValueError("Invalid split selection evidence")
        selection[split] = {}
        for field in ("selected_records", "selected_pairs"):
            value = source.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError("Selected evidence counts must be nonnegative integers")
            selection[split][field] = value
    return results, settings, {
        "results_path": str(result_path.resolve()), "results_sha256": result_hash,
        "settings_sha256": settings_hash, "comparison_status": status,
        "evidence_file_sha256": evidence_hash, "source_archive_status": evidence.get("archive_status", "unknown"),
        "source_archive_manifest_sha256_from_evidence": evidence.get("archive_manifest_sha256"),
        "actual_selection": selection,
        "omitted_metrics": omissions, "plotted_values": plotted,
    }


def formatted(value):
    return "undefined" if value is None else f"{value:.4g}"


def repair_groups(results):
    groups = []
    for methods in (NATIVE_REPAIRS, HYBRID_REPAIRS):
        rows = [row for method in methods for row in results if row["method"] == method]
        if not rows:
            continue
        ranks = {row.get("correction_span_rank") for row in rows}
        factors = {row.get("factor_tensor_bytes") for row in rows}
        bases = {row.get("base_tensor_bytes") for row in rows}
        dtypes = {row.get("coefficient_storage_dtype") for row in rows}
        valid = all(isinstance(value, int) and not isinstance(value, bool) and value > 0
            for values in (ranks, factors, bases) for value in values)
        matched = (len(rows) == 3 and valid and len(ranks) == len(factors) == len(bases) == 1 and
            dtypes == {"float32"} and all(row.get("base_variant") == "symmetric_int4" for row in rows) and
            all(row.get("rank_matched_to_variant") == methods[0] for row in rows if row["method"] != methods[0]))
        groups.append({"anchor_variant": methods[0], "variants": [row["variant"] for row in rows],
            "correction_rank": next(iter(ranks)) if len(ranks) == 1 else None,
            "rank_base_and_tensor_bytes_match": matched})
    return groups


def render(comparison, out, *, dpi=180, linear_y=False):
    results, settings, manifest = load_comparison(comparison)
    if out.exists():
        raise FileExistsError("Use a new figure output directory")
    if importlib.util.find_spec("matplotlib") is None:
        raise RuntimeError("Matplotlib is required in the invoking plotting environment")
    out.mkdir(parents=True)
    # Contain the plotting cache beside this figure rather than changing global
    # configuration or importing the project's tensor/model dependencies.
    os.environ["MPLCONFIGDIR"] = str((out / ".matplotlib").resolve())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    origin = settings.get("origin", "unverified")
    origins = {row.get("heldout_response", {}).get("record_origin") for row in results}
    fixture = origin == "synthetic_fixture" or "synthetic_fixture" in origins
    identity = settings["identity"]
    unit, divisor = ("MiB", 1 << 20) if max(row["serialized_total_bytes"] for row in results) >= 1 << 20 else ("KiB", 1 << 10)
    source_values = {row.get("source_tensor_bytes_including_scales") for row in results}
    source_bytes = next(iter(source_values)) if len(source_values) == 1 else None
    if isinstance(source_bytes, bool) or not isinstance(source_bytes, int) or source_bytes <= 0:
        source_bytes = None
    present = [method for method in STYLES if any(row["method"] == method for row in results)]
    matched_rows = [row for method in ("symmetric_int4", *REPAIRS) for row in results if row["method"] == method]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "svg.fonttype": "none", "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "DejaVu Sans"})
    expanded = any(row["method"] in HYBRID_REPAIRS for row in results)
    figure, axes = plt.subplots(1, 3, figsize=(16, 11 if expanded else 9.6))
    figure.subplots_adjust(left=0.065, right=0.982, top=0.845, bottom=0.49 if expanded else 0.47, wspace=0.30)
    title = "SYNTHETIC FIXTURE — layout check" if fixture else "Scoped compression comparison"
    if origin != "measured_model_state" and not fixture:
        title += " — origin unverified"
    if manifest["comparison_status"] != "finished":
        title += f" — partial ({manifest['comparison_status']})"
    figure.suptitle(title, x=0.065, ha="left", y=0.97, fontsize=17, fontweight="bold")
    expert_count = len(identity.get("expert_ids", []))
    shape = identity.get("matrix_shape", [])
    figure.text(0.065, 0.925,
        f"{identity.get('model_id', 'Unknown source')} · {expert_count} sampled experts · matrix {shape} · {len(results)} saved variants",
        fontsize=11, color="#555555")
    counts = []
    for split, label in (("calibration", "calibration"), ("heldout", "heldout")):
        values = manifest["actual_selection"][split]
        states, pairs = values["selected_records"], values["selected_pairs"]
        counts.append(f"{label} {states:,} states / {pairs:,} pairs" if states is not None and pairs is not None else f"{label} selection unavailable")
    figure.text(0.065, 0.895,
        f"Source archive: {manifest['source_archive_status']} · " + " · ".join(counts),
        fontsize=10, color="#555555")

    rank_offsets = {"weight_pca": (-9, 10), "trajectory_response": (9, -14), "probe_span": (9, 10)}
    rank_points = {}
    for axis, (title, section, key) in zip(axes, PANELS):
        count = 0
        for method in present:
            label, color, marker = STYLES[method]
            points = [row for row in results if row["method"] == method and metric(row, section, key) is not None]
            points.sort(key=lambda row: row.get("rank") if row.get("rank") is not None else -1)
            x = [row["serialized_total_bytes"] / divisor for row in points]
            y = [metric(row, section, key) for row in points]
            if method in rank_offsets and len(points) > 1:
                axis.plot(x, y, color=color, alpha=0.45, linewidth=1, zorder=2)
            size = 150 if method in ("probe_residual", "probe_pca_hybrid") else 72 if method in REPAIRS else 48
            axis.scatter(x, y, label=label, marker=marker, c=color, s=size,
                edgecolors="white", linewidths=0.6, zorder=6 if method in REPAIRS else 3)
            if method in rank_offsets:
                rank_points[axis, method] = [(row["rank"], xi, yi) for row, xi, yi in zip(points, x, y)]
            count += len(points)
        axis.set_title(title, loc="left", pad=12)
        axis.set_xlabel(f"Actual serialized storage ({unit})")
        axis.set_ylabel(key)
        axis.xaxis.set_major_locator(MaxNLocator(5))
        axis.grid(True, which="major", color="#dddddd", linewidth=0.6)
        if not linear_y:
            axis.set_yscale("symlog", linthresh=1e-6, linscale=0.8)
        axis.margins(x=0.10, y=0.08)
        # Give true zero-error markers room without clipping their lower half.
        axis.set_ylim(bottom=-2e-7 if not linear_y else -0.03 * max(1e-6, axis.get_ylim()[1]))
        if source_bytes is not None:
            axis.axvline(source_bytes / divisor, color="#888888", linestyle=":", linewidth=1, zorder=1)
        if count == 0:
            axis.text(0.5, 0.5, "Metric unavailable", transform=axis.transAxes, ha="center", color="#666666")
        axis.text(0, -0.18,
            f"{count}/{len(results)} finite metrics" + (" · static, not heldout" if section.startswith("static") else ""),
            transform=axis.transAxes, fontsize=8, color="#555555")

    # Combine labels only when same-method points are visually adjacent. All
    # measured coordinates stay exact: neither points nor bytes are jittered.
    figure.canvas.draw()
    for (axis, method), points in rank_points.items():
        groups = []
        for rank, x, y in points:
            pixel = axis.transData.transform((x, y))
            group = next((value for value in groups if
                abs(pixel[0] - value["pixel"][0]) < 14 and abs(pixel[1] - value["pixel"][1]) < 14), None)
            if group is None:
                groups.append({"ranks": [rank], "xy": (x, y), "pixel": pixel})
            else:
                group["ranks"].append(rank)
        for group in groups:
            offset = rank_offsets[method]
            if group["xy"][1] < 1e-5 and offset[1] < 0:
                offset = (-18, 10)
            axis.annotate(", ".join(f"r{rank}" for rank in group["ranks"]), group["xy"],
                xytext=offset, textcoords="offset points", fontsize=8, color=STYLES[method][1],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.5})

    handles = [Line2D([], [], color=STYLES[method][1], marker=STYLES[method][2],
        linestyle="none", markersize=9, label=STYLES[method][0]) for method in present]
    if source_bytes is not None:
        source_label = "Original source bytes (FP8 + scales)" if str(identity.get("source_dtype", "")).startswith("float8_") else "Original source bytes"
        handles.append(Line2D([], [], color="#888888", linestyle=":", label=source_label))
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.058, 0.402 if expanded else 0.382),
        ncol=4, frameon=False, fontsize=9, handletextpad=0.5, columnspacing=2.0)

    if matched_rows:
        table_axis = figure.add_axes((0.065, 0.134, 0.917, 0.184 if expanded else 0.166))
        table_axis.set_axis_off()
        rows = []
        for row in matched_rows:
            rows.append([STYLES[row["method"]][0], f"{row['serialized_total_bytes']:,}",
                str(row.get("correction_span_rank", "—")),
                formatted(metric(row, PANELS[0][1], PANELS[0][2])),
                formatted(metric(row, PANELS[1][1], PANELS[1][2])),
                formatted(metric(row, PANELS[2][1], PANELS[2][2]))])
        table = table_axis.table(cellText=rows,
            colLabels=["INT4 and input-span repairs", "Actual bytes", "Repair rank k", "Heldout point", "Heldout Δstate", "Static seed span"],
            cellLoc="left", colLoc="left", colWidths=[0.34, 0.16, 0.12, 0.12, 0.13, 0.13], bbox=(0, 0, 1, 1))
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        for (i, j), cell in table.get_celld().items():
            cell.set_edgecolor("#dddddd")
            cell.set_linewidth(0.4)
            if i == 0:
                cell.set_facecolor("#eeeeee")
                cell.set_text_props(weight="bold")
            elif j == 0:
                cell.set_text_props(color=STYLES[matched_rows[i - 1]["method"]][1])

    groups = repair_groups(results)
    matched_ranks = [str(group["correction_rank"]) for group in groups if group["rank_base_and_tensor_bytes_match"]]
    match_note = ("Matched correction groups: k=" + ", k=".join(matched_ranks) +
        ". Within each: same INT4 base bytes, FP32 coefficient storage and tensor bytes; serialized metadata sizes differ.") if matched_ranks else "No complete matched three-candidate correction group is established."
    yscale = "Linear y axes." if linear_y else "Symlog y axes: linear below 1e-6, exact zero retained."
    figure.text(0.065, 0.106, f"r = across-expert shared-basis rank; k = input-span correction rank. {yscale}", fontsize=8.5)
    figure.text(0.065, 0.083, match_note, fontsize=8.5)
    figure.text(0.065, 0.058, CAPTION, fontsize=10, fontweight="bold")
    figure.text(0.065, 0.036,
        "Heldout: weighted mean of per-input relative squared errors, then √. Δstate = next − previous; not per-token velocity. Seed span: global relative Frobenius error.", fontsize=8)
    figure.text(0.065, 0.016,
        "All sampled experts equally weighted; seeds may fit candidates and are not heldout. Undefined metrics omitted. Seed witness stored separately. No full-decoder or language-quality result.", fontsize=8)
    png, svg = out / "comparison.png", out / "comparison.svg"
    figure.savefig(png, dpi=dpi)
    figure.savefig(svg)
    plt.close(figure)
    manifest.update({"format": "glm-gate-linear-proxy-figure-v1", "scope_caption": CAPTION,
        "origin": origin, "synthetic_fixture": fixture, "identity": identity,
        "matplotlib_version": matplotlib.__version__, "x_unit": unit,
        "y_scale": "linear" if linear_y else "symlog; linear below 1e-6",
        "shared_rank_symbol": "r", "input_span_repair_rank_symbol": "k",
        "three_repairs_rank_and_tensor_bytes_match": any(group["anchor_variant"] == "probe_residual" and
            group["rank_base_and_tensor_bytes_match"] for group in groups),
        "repair_rank_groups": groups,
        "original_source_bytes": source_bytes, "metric_definitions": DEFINITIONS,
        "outputs": {"png": str(png.resolve()), "svg": str(svg.resolve())}})
    (out / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", required=True, type=Path, help="Comparison directory or its results.json")
    parser.add_argument("--out", required=True, type=Path, help="New local figure output directory")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--linear-y", action="store_true", help="Use linear instead of symlog y axes")
    args = parser.parse_args()
    if not 72 <= args.dpi <= 600:
        parser.error("dpi must be 72..600")
    try:
        manifest = render(args.comparison, args.out, dpi=args.dpi, linear_y=args.linear_y)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"[Compression plot] {exc}\n")
    print(f"[Compression plot] PNG + SVG saved; {len(manifest['plotted_values'])} finite metrics, {len(manifest['omitted_metrics'])} omitted. Matplotlib {manifest['matplotlib_version']}.")


if __name__ == "__main__":
    main()
