"""Render invariants/out/persona_control.json -> invariants/out/persona_control.png.

Two panels:
  (left)  GSM8K accuracy + fluency per ablation direction, with fraction-of-norm-removed
          annotated. The money figure: common-mode-aligned dirs (persona/math/common)
          collapse; random + orthogonalized-PR do not.
  (right) Dose-response (accuracy & fluency vs alpha) — graceful = specific,
          cliff + fluency collapse = corruption.

  python -m scripts.plot_persona_control
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "invariants" / "out"
R = json.loads((OUT / "persona_control.json").read_text(encoding="utf-8"))


def main():
    bench = R.get("benchmark", {})
    dose = R.get("dose_response", {})
    order = [c for c in ["baseline", "persona_mean", "math_mean", "common_mode",
                         "random0", "random1", "random2", "pr_orth"] if c in bench]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))

    # --- left: benchmark bars ---
    acc = [bench[c]["acc"] for c in order]
    flu = [bench[c]["fluent"] for c in order]
    x = np.arange(len(order))
    w = 0.38
    axL.bar(x - w / 2, acc, w, label="GSM8K accuracy", color="#3498db")
    axL.bar(x + w / 2, flu, w, label="fluency", color="#2ecc71")
    for i, c in enumerate(order):
        f = bench[c].get("frac_norm_removed", 0.0)
        axL.text(i, max(acc[i], flu[i]) + 0.02, f"{f:.3f}", ha="center",
                 fontsize=8, color="#c0392b")
    axL.set_xticks(x)
    axL.set_xticklabels(order, rotation=30, ha="right")
    axL.set_ylim(0, 1.1)
    axL.set_ylabel("rate")
    axL.set_title("Ablating L16-31 by direction\n(red = fraction of norm removed)")
    axL.legend(loc="upper right")
    axL.axhline(bench.get("baseline", {}).get("acc", 0), ls=":", color="#888", lw=1)

    # --- right: dose-response ---
    colors = {"pr_orth": "#9b59b6", "random0": "#95a5a6", "persona_mean": "#e74c3c"}
    for cond, rows in dose.items():
        a = [r["alpha"] for r in rows]
        axR.plot(a, [r["acc"] for r in rows], "-o", color=colors.get(cond, "#333"),
                 label=f"{cond} acc")
        axR.plot(a, [r["fluent"] for r in rows], "--s", color=colors.get(cond, "#333"),
                 alpha=0.6, label=f"{cond} fluent")
    axR.set_xlabel("alpha (fraction of projection removed)")
    axR.set_ylabel("rate")
    axR.set_ylim(0, 1.1)
    axR.set_title("Dose-response\n(graceful=specific; cliff+fluency drop=corruption)")
    axR.legend(fontsize=8)

    g = R.get("geometry", {})
    fig.suptitle(
        f"persona 'vector' is {g.get('cos_persona_vs_common', float('nan')):.2f} the "
        f"common-mode; math-built mean is {g.get('cos_math_vs_common', float('nan')):.2f}; "
        f"pr_orth {g.get('cos_pr_orth_vs_common', float('nan')):.2f}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = OUT / "persona_control.png"
    fig.savefig(path, dpi=130)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
