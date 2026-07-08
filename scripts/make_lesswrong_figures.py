"""Build the figures for the LessWrong state-output-decoupling post.

Every data figure is drawn directly from the probe output JSONs in
invariants/out/ -- no numbers are typed in by hand except the analytic
chance level for balanced k-class 1-NN, which is (n_per_class-1)/(n-1).

Run:  python scripts/make_lesswrong_figures.py [list | all | <which> ...]
      no args / 'all'  -> build all five
      list             -> name, output file, blurb, built-or-not (needs no matplotlib)
      <which>          -> loose names: 1/schematic, 2/render-u, 3/pre-control,
                          4/cot, 5/origin -- any unambiguous fragment works
Out:  docs/figures/fig*.png

matplotlib is imported lazily (only when actually drawing), so 'list' and name
resolution work under any interpreter -- including the repo venv, which has no
matplotlib. The interactive shell's :figures command leans on that split.
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "figures"

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"    # slot 1: answer identity / direct mode
AQUA = "#1baf7a"    # slot 2: operation / brief CoT
YELLOW = "#eda100"  # slot 3: output format / verbose CoT
GREEN = "#008300"   # slot 4: surface story control

SEQ_LO, SEQ_HI = "#cde2fb", "#0d366b"  # sequential blue ramp endpoints

RC = {
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10.5,
}

DPI = 200

# Populated by _setup_mpl(); every figure function reads these module globals.
plt = None
LinearSegmentedColormap = None


def _setup_mpl():
    """Import matplotlib on first draw so listing never requires it."""
    global plt, LinearSegmentedColormap
    if plt is not None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    from matplotlib.colors import LinearSegmentedColormap as _lsc
    plt = _plt
    LinearSegmentedColormap = _lsc
    plt.rcParams.update(RC)


def style_axes(ax, ymax=1.06):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-0.04, ymax)
    ax.tick_params(length=0)


def title_block(fig, title, subtitle, footer):
    fig.text(0.045, 0.965, title, fontsize=14.5, fontweight="semibold",
             color=INK, ha="left", va="top")
    fig.text(0.045, 0.915, subtitle, fontsize=10.5, color=INK2,
             ha="left", va="top", linespacing=1.35)
    fig.text(0.045, 0.012, footer, fontsize=8, color=MUTED,
             ha="left", va="bottom", linespacing=1.45)


def line(ax, x, y, color, lw=2.2, markers=False, end_dot=True):
    ax.plot(x, y, color=color, lw=lw, solid_joinstyle="round",
            solid_capstyle="round", zorder=3)
    if markers:
        ax.plot(x, y, marker="o", ms=4.6, mfc=color, mec=SURFACE, mew=1.3,
                lw=0, zorder=4)
    elif end_dot:
        ax.plot([x[-1]], [y[-1]], marker="o", ms=5.2, mfc=color, mec=SURFACE,
                mew=1.3, lw=0, zorder=4)


def end_label(ax, x, y, text, color=INK2, dx=0.5, fontsize=9.5, va="center"):
    ax.annotate(text, (x, y), xytext=(x + dx, y), fontsize=fontsize,
                color=color, va=va, ha="left", zorder=6)


def legend_row(target, handles, anchor, fontsize=8.8, ncol=None, loc="upper left"):
    leg = target.legend(handles=handles, loc=loc, frameon=False,
                        fontsize=fontsize, bbox_to_anchor=anchor,
                        ncol=ncol or len(handles), columnspacing=1.6,
                        handlelength=1.6, handletextpad=0.6)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


# ================================================================ figure 1
def fig1_schematic():
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=DPI)
    fig.subplots_adjust(left=0.045, right=0.97, top=0.80, bottom=0.16)

    x = np.linspace(0, 31, 400)
    y = ((x - 15.5) / 15.5) ** 2                     # 1 at edges, 0 mid
    ax.plot(x, y, color=BLUE, lw=3.0, solid_capstyle="round", zorder=3)

    # entry / exit arrows
    ax.annotate("prompt text", (0.0, 1.0), xytext=(-0.5, 1.16), fontsize=11,
                color=INK, fontweight="semibold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))
    ax.annotate("output text", (31.0, 1.0), xytext=(26.2, 1.16), fontsize=11,
                color=INK, fontweight="semibold", ha="left")
    ax.annotate("", (31.8, 1.14), xytext=(31.0, 1.02),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4))

    # arm / basin labels, riding the arms
    ax.annotate("interpretation /\ntranslation", (5.0, 0.62), fontsize=11,
                color=INK, ha="center", rotation=-38, rotation_mode="anchor")
    ax.annotate("latent task state\n(the workspace band)", (15.5, 0.075),
                fontsize=11, color=INK, ha="center", va="bottom")
    ax.annotate("communication /\nrender", (26.4, 0.65), fontsize=11,
                color=INK, ha="center", rotation=38, rotation_mode="anchor")

    # what each arm does, tucked inside the basin, clear of the curve
    ax.annotate("surface text becomes task,\noperation, role, format bindings",
                (9.3, 0.50), fontsize=8.6, color=INK2, ha="center", va="top")
    ax.annotate("reusable variables: intent,\nuncertainty, concept maps",
                (15.5, 0.44), fontsize=8.6, color=INK2, ha="center", va="top")
    ax.annotate("latent state becomes format,\nstyle, hedges, persona, tokens",
                (21.8, 0.50), fontsize=8.6, color=INK2, ha="center", va="top")

    # the external mapping row
    for xx, t in [(4.0, "sensory"), (15.5, "workspace"), (27.0, "motor")]:
        ax.annotate(t, (xx, -0.30), fontsize=9.0, color=MUTED,
                    ha="center", style="italic")
    ax.annotate("cf. Anthropic 2026, global-workspace paper:", (0.0, -0.175),
                fontsize=8.2, color=MUTED, ha="left")

    ax.set_xlim(-1.5, 33)
    ax.set_ylim(-0.38, 1.30)
    ax.axis("off")
    ax.annotate("layer depth  (0 → 31)", (15.5, -0.075), fontsize=9.5,
                color=MUTED, ha="center")

    title_block(
        fig,
        "The U-shaped bottleneck",
        "Generated text is the render of a deeper trajectory: translated in, worked on in latent space, translated back out.",
        "Schematic, not data. Depth roles are functional regimes, not fixed per-layer jobs. Figures 2–4 carry the measurements.",
    )
    fig.savefig(OUT_DIR / "fig1_bottleneck_schematic.png")
    plt.close(fig)


# ================================================================ figure 2
def fig2_render_u():
    d = json.load(open(ROOT / "invariants/out/translation_thinking_Llama-3.1-8B-Instruct.json",
                       encoding="utf-8"))
    pl = d["positions"]["render"]["per_layer"]
    layers = [r["layer"] for r in pl]
    ans = [r["answer_nn"] for r in pl]
    op = [r["operation_nn"] for r in pl]
    fmt = [r["format_nn"] for r in pl]
    n = d["n"]
    chance = (n // 4 - 1) / (n - 1)  # 4 balanced classes per label

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=DPI)
    fig.subplots_adjust(left=0.075, right=0.845, top=0.745, bottom=0.16)

    ax.hlines(chance, 0, 31, color=MUTED, lw=1.0, zorder=2)
    ax.annotate(f"label-shuffle chance ≈ {chance:.2f}", (31, chance + 0.012),
                fontsize=8.2, color=MUTED, ha="right", va="bottom")

    line(ax, layers, fmt, YELLOW)
    line(ax, layers, op, AQUA)
    line(ax, layers, ans, BLUE)

    end_label(ax, 31, 1.035, "output format", dx=0.6)
    end_label(ax, 31, ans[-1] - 0.035, "answer identity", dx=0.6)
    end_label(ax, 31, op[-1], "operation", dx=0.6)

    # the story annotation: the dip
    ax.annotate("L10: the token being emitted is least present;\nthe abstract operation is maximal",
                (10, 0.012), xytext=(12.0, 0.10), fontsize=9, color=INK2,
                va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))

    # interpretive region labels, kept visually secondary
    for xx, t in [(2.0, "token echo"), (11.5, "latent work"), (28.0, "re-render")]:
        ax.annotate(t, (xx, 1.10), fontsize=9, color=MUTED, ha="center",
                    style="italic", annotation_clip=False)

    ax.set_xlabel("layer")
    ax.set_ylabel("same-label nearest-neighbor rate")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 31])
    style_axes(ax, ymax=1.16)

    legend_row(fig, [
        plt.Line2D([], [], color=BLUE, lw=2.2, label="answer identity"),
        plt.Line2D([], [], color=AQUA, lw=2.2, label="operation"),
        plt.Line2D([], [], color=YELLOW, lw=2.2, label="output format"),
    ], anchor=(0.068, 0.862), fontsize=9.2)

    title_block(
        fig,
        "The U, measured at the answer tokens",
        "While the model writes its answer, the answer's own token identity vanishes mid-stack — exactly where the abstract operation peaks.",
        "Llama-3.1-8B-Instruct · hidden states over generated answer tokens · n=64 arithmetic prompts: 4 answers × 4 operations × 4 output formats\n"
        "centered cosine 1-NN same-label rate · 300 label shuffles · invariants/out/translation_thinking_Llama-3.1-8B-Instruct.json",
    )
    fig.savefig(OUT_DIR / "fig2_render_u.png")
    plt.close(fig)


# ================================================================ figure 3
def fig3_pre_and_control():
    d = json.load(open(ROOT / "invariants/out/translation_thinking_Llama-3.1-8B-Instruct.json",
                       encoding="utf-8"))
    pl = d["positions"]["pre"]["per_layer"]
    layers = [r["layer"] for r in pl]
    ans = [r["answer_nn"] for r in pl]
    op = [r["operation_nn"] for r in pl]
    n = d["n"]
    chance = (n // 4 - 1) / (n - 1)

    ic = json.load(open(ROOT / "invariants/out/intent_surface_control_Llama-3.1-8B-Instruct.json",
                        encoding="utf-8"))
    icl = ic["per_layer"]
    ic_layers = [r["layer"] for r in icl]
    ic_op = [r["operation_nn"] for r in icl]
    ic_base = [r["base_nn"] for r in icl]
    ic_op_null = float(np.mean([r["operation_nn_null"] for r in icl]))
    ic_base_null = float(np.mean([r["base_nn_null"] for r in icl]))
    op_p = max(r["operation_nn_p"] for r in icl)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.1), dpi=DPI, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.72, bottom=0.17,
                        wspace=0.14)

    # ---- panel a: before generation
    ax = axes[0]
    ax.hlines(chance, 0, 31, color=MUTED, lw=1.0, zorder=2)
    ax.annotate(f"chance ≈ {chance:.2f}", (20, chance + 0.012),
                fontsize=8.2, color=MUTED, ha="right", va="bottom")
    line(ax, layers, op, AQUA)
    line(ax, layers, ans, BLUE)
    ax.annotate("operation", (3.0, 1.035), fontsize=9.5, color=INK2)
    ax.annotate("answer identity", xy=(25, ans[25]), xytext=(14.5, 0.45),
                fontsize=9.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
    ax.set_title("a · last prompt token, before any output",
                 fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("same-label nearest-neighbor rate")
    ax.set_xticks([0, 10, 20, 31])
    style_axes(ax)

    # ---- panel b: surface control
    ax = axes[1]
    ax.hlines(ic_op_null, 0, 31, color=MUTED, lw=1.0, zorder=2)
    ax.annotate(f"operation shuffle-null ≈ {ic_op_null:.2f}",
                (20, ic_op_null + 0.012), fontsize=8.2, color=MUTED,
                ha="right", va="bottom")
    ax.hlines(ic_base_null, 0, 31, color=MUTED, lw=1.0, zorder=2)
    ax.annotate(f"story shuffle-null ≈ {ic_base_null:.2f}",
                (20, ic_base_null - 0.017), fontsize=8.2, color=MUTED,
                ha="right", va="top")
    line(ax, ic_layers, ic_base, GREEN)
    line(ax, ic_layers, ic_op, AQUA)
    ax.annotate("operation", (2.0, 1.035), fontsize=9.5, color=INK2)
    ax.annotate(f"(p ≤ {op_p:.3f} at every layer)", (16.0, 0.90),
                fontsize=8.2, color=MUTED, ha="center", va="top")
    ax.annotate("surface story (names / objects)", xy=(25, ic_base[25]),
                xytext=(8.0, 0.42), fontsize=9.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
    ax.set_title("b · same operations, crossed word-problem stories",
                 fontsize=10.5, color=INK, loc="left", pad=8)
    ax.set_xlabel("layer")
    ax.set_xticks([0, 10, 20, 31])
    style_axes(ax)

    legend_row(fig, [
        plt.Line2D([], [], color=AQUA, lw=2.2, label="operation (intent)"),
        plt.Line2D([], [], color=BLUE, lw=2.2, label="answer identity"),
        plt.Line2D([], [], color=GREEN, lw=2.2, label="surface story (control)"),
    ], anchor=(0.068, 0.845), fontsize=9.2)

    title_block(
        fig,
        "Intent is settled before the answer exists — and it isn't the words",
        "(a) Pre-generation, operation groups perfectly from layer 0; answer identity only assembles after ~L20.\n"
        "(b) Stories crossed against operations: operation stays at ceiling while story grouping sits at or below its null mid-stack.",
        "Llama-3.1-8B-Instruct · (a) n=64, translation_thinking · (b) n=96: 8 stories × 4 operations × 3 variants, intent_surface_control\n"
        "centered cosine 1-NN · label-shuffle nulls (300 / 500 perms) · invariants/out/",
    )
    fig.savefig(OUT_DIR / "fig3_pre_position_and_control.png")
    plt.close(fig)


# ================================================================ figure 4
def fig4_cot_trajectory():
    d = json.load(open(ROOT / "invariants/out/cot_reality_Llama-3.1-8B-Instruct.json",
                       encoding="utf-8"))
    abm = d["answer_by_mode"]
    positions = ["pre", "gen_first", "gen_early", "gen_mid", "gen_late", "gen_final"]
    pos_labels = ["before\ngeneration", "first\ntoken", "early", "mid", "late", "final\ntoken"]
    modes = [("direct", BLUE, "direct answer"),
             ("brief_cot", AQUA, "brief CoT"),
             ("verbose_cot", YELLOW, "verbose CoT")]

    x = np.arange(len(positions))
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=DPI)
    fig.subplots_adjust(left=0.075, right=0.86, top=0.745, bottom=0.17)

    nulls = [abm[p][m]["best_answer"]["null"] for p in positions for m, _, _ in modes]
    ax.axhspan(min(nulls), max(nulls), color=GRID, alpha=0.55, zorder=1)
    ax.annotate("label-shuffle chance band", (5.05, min(nulls) - 0.014),
                fontsize=8.2, color=MUTED, ha="right", va="top")

    for mode, color, label in modes:
        y = [abm[p][mode]["best_answer"]["nn"] for p in positions]
        line(ax, x, y, color, markers=True)
        end_label(ax, x[-1], y[-1], label, dx=0.12)

    b = abm["pre"]["direct"]["best_answer"]
    ax.annotate(f"direct mode: answer decodable before\nany token ({b['nn']:.2f}, p≈{b['p']:.3f}, L{b['layer']})\n— CoT modes: nothing (0.00)",
                (0, b["nn"]), xytext=(0.30, 0.97), fontsize=9, color=INK2,
                va="top", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels(pos_labels)
    ax.set_ylabel("answer decodability (best layer)")
    ax.set_xlabel("position along the generated reasoning")
    style_axes(ax)

    legend_row(fig, [
        plt.Line2D([], [], color=c, lw=2.2, marker="o", ms=4.6, mfc=c,
                   mec=SURFACE, mew=1.2, label=l) for _, c, l in modes
    ], anchor=(0.068, 0.862), fontsize=9.2)

    title_block(
        fig,
        "Direct prompts pre-commit; chain-of-thought computes in the open",
        "Best-layer decodability of the final answer at six points along the trajectory: direct-mode answers exist before generation;\n"
        "CoT-mode answers only become decodable late in the generated text — the “mini-U” runs through the visible reasoning.",
        "Llama-3.1-8B-Instruct · n=48 (16 per mode), 47/48 answered correctly · centered cosine 1-NN, best of 32 layers per point\n"
        "(selection over layers; per-point permutation p-values in the JSON, 200 shuffles) · invariants/out/cot_reality_Llama-3.1-8B-Instruct.json",
    )
    fig.savefig(OUT_DIR / "fig4_cot_trajectory.png")
    plt.close(fig)


# ================================================================ figure 5
def fig5_origin():
    o1 = json.load(open(ROOT / "invariants/out/origin.json", encoding="utf-8"))
    o2 = json.load(open(ROOT / "invariants/out/origin2.json", encoding="utf-8"))
    # rows: model (base, instruct); cols: raw completion, chat template
    grid = np.array([
        [o1["base"]["disclaim_rate"]["direct"], o2["base"]["disclaim_rate_chat"]],
        [o1["instruct"]["disclaim_rate"]["direct"], o2["instruct"]["disclaim_rate_chat"]],
    ])

    cmap = LinearSegmentedColormap.from_list("seqblue", [SEQ_LO, SEQ_HI])
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=DPI)
    fig.subplots_adjust(left=0.20, right=0.94, top=0.72, bottom=0.14)

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(grid):
        ax.annotate(f"{v * 100:.0f}%", (j, i), ha="center", va="center",
                    fontsize=22, fontweight="semibold",
                    color=INK if v < 0.5 else "#ffffff")

    # surface gap between cells
    ax.hlines(0.5, -0.5, 1.5, color=SURFACE, lw=3)
    ax.vlines(0.5, -0.5, 1.5, color=SURFACE, lw=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["raw completion", "chat template"], fontsize=11, color=INK2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["base\nmodel", "instruct\nmodel"], fontsize=11, color=INK2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)

    title_block(
        fig,
        "The self-disclaimer lives in one cell",
        "Rate of “as an AI I don't…”-style denials when asked “do you actually feel/believe/want…?” —\n"
        "near-absent in the base model and in raw completion; near-ceiling only under tuning × chat format.",
        "Llama-3.1-8B (base) vs Llama-3.1-8B-Instruct · 12 inner-state predicates per cell · raw cell = direct-question condition\n"
        "invariants/out/origin.json · invariants/out/origin2.json",
    )
    fig.savefig(OUT_DIR / "fig5_origin_2x2.png")
    plt.close(fig)


# ============================================================ registry & CLI
FIGURES = {
    "1-schematic": (fig1_schematic, "fig1_bottleneck_schematic.png",
                    "the U-shaped bottleneck frame (schematic, no data)"),
    "2-render-u": (fig2_render_u, "fig2_render_u.png",
                   "the U at the generated answer tokens (translation_thinking, render)"),
    "3-pre-control": (fig3_pre_and_control, "fig3_pre_position_and_control.png",
                      "pre-generation crossing + surface-story control"),
    "4-cot": (fig4_cot_trajectory, "fig4_cot_trajectory.png",
              "answer decodability along the generated reasoning, by prompt mode"),
    "5-origin": (fig5_origin, "fig5_origin_2x2.png",
                 "self-disclaimer 2x2: base/instruct x raw/chat"),
}


def resolve_figure_names(tokens):
    """Map loose tokens (1, fig2, render, cot, origin_2x2, ...) to registry keys."""
    if not tokens:
        return list(FIGURES)
    keys = []
    for tok in tokens:
        t = str(tok).strip().lower().replace("fig", "").replace("_", "-")
        if not t:
            raise ValueError(f"empty figure name '{tok}' -- try: " + ", ".join(FIGURES))
        match = None
        for key, (_fn, fname, _blurb) in FIGURES.items():
            hay = key + " " + fname.lower().replace("_", "-")
            if t == key or t == key.split("-", 1)[0] or t in hay:
                match = key
                break
        if match is None:
            raise ValueError(f"unknown figure '{tok}' -- try: " + ", ".join(FIGURES))
        if match not in keys:
            keys.append(match)
    return keys


def build(tokens=None):
    """Build the requested figures (all when tokens is empty); returns paths."""
    names = resolve_figure_names(list(tokens or []))
    _setup_mpl()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for key in names:
        fn, fname, _blurb = FIGURES[key]
        fn()
        made.append(OUT_DIR / fname)
    return made


def main(argv):
    args = [a.strip() for a in argv if a.strip()]
    if args and args[0].lower() in {"list", "--list", "-l"}:
        for key, (_fn, fname, blurb) in FIGURES.items():
            out = OUT_DIR / fname
            if out.exists():
                ts = datetime.datetime.fromtimestamp(out.stat().st_mtime)
                status = f"built {ts:%Y-%m-%d %H:%M}"
            else:
                status = "not built"
            print(f"{key:15} {fname:38} [{status}]  {blurb}")
        return 0
    if args and args[0].lower() == "all":
        args = args[1:]
    try:
        made = build(args)
    except ValueError as exc:
        print(exc)
        return 2
    for p in made:
        print(f"{p.relative_to(ROOT)} {p.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
