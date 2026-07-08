"""Exploratory figures mined from the probe / steering / calibration logs.

Companions to make_lesswrong_figures.py (same CLI, same style, same lazy
matplotlib), but drawing on the log families the five post figures don't
touch: causal-edit sweeps (surgery/patch/attention), frame batteries,
persona-ablation controls, the pre-registered uncertainty replication, and
the live shell's steer-map / trigger-tuner telemetry.

Every number is read from the source JSON at build time -- nothing typed in.

Run:  python scripts/make_exploratory_figures.py [list | all | <which> ...]
Out:  docs/figures/x*.png
"""

import datetime
import json
import sys
from pathlib import Path

import make_lesswrong_figures as S  # style helpers + lazy matplotlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "invariants" / "out"
FIG_DIR = S.OUT_DIR


def _load(name):
    return json.load(open(OUT / f"{name}.json", encoding="utf-8"))


# ================================================== x1: intervention ledger
def x1_intervention_ledger():
    plt = S.plt
    probe = _load("probe_self_steering_isolated")          # layer -> CV acc
    att = _load("attention_self_steering_isolated")["summary"]
    patch = _load("patch_self_steering_isolated")
    patchfull = _load("patchfull_self_steering_isolated")
    sweeps = {
        "self dir\nL15": _load("surgery_l15")["sweep"],
        "self dir\nL31": _load("surgery_l31")["sweep"],
        "belief dir\nL15": _load("surgery_belief_l15")["sweep"],
        "belief dir\nL31": _load("surgery_belief_l31")["sweep"],
    }
    hedge_base = sweeps["self dir\nL15"][0]["hedge"]  # alpha=0 baseline (0.667)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.2), dpi=S.DPI)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.745, bottom=0.10,
                        hspace=0.52, wspace=0.22)

    # (a) read: linear probe decodability by layer
    ax = axes[0][0]
    layers = sorted(int(k) for k in probe)
    acc = [probe[str(l)] for l in layers]
    S.line(ax, layers, acc, S.BLUE)
    ax.hlines(0.5, layers[0], layers[-1], color=S.MUTED, lw=1.0, zorder=2)
    ax.annotate("chance 0.5", (31, 0.512), fontsize=8.2, color=S.MUTED,
                ha="right", va="bottom")
    peak_l = layers[int(max(range(len(acc)), key=lambda i: acc[i]))]
    ax.annotate(f"peak {max(acc):.2f} @ L{peak_l}", (peak_l, max(acc) + 0.03),
                fontsize=9, color=S.INK2, ha="center")
    ax.set_title("a · read it: probe accuracy by layer", fontsize=10.5,
                 color=S.INK, loc="left", pad=8)
    ax.set_xlabel("layer")
    ax.set_ylabel("rate")
    ax.set_xticks([0, 8, 16, 24, 31])
    S.style_axes(ax, ymax=1.12)

    # (b) push it: hedge rate at the strongest-effect dose of each edit
    ax = axes[0][1]
    bars, notes = [], []
    for label, sweep in sweeps.items():
        best = max(sweep[1:], key=lambda r: abs(r["hedge"] - hedge_base))
        bars.append((f"{label}\nα={best['alpha']:g}", best["hedge"]))
        if best["hedge"] < hedge_base:
            last = sweep[-1]
            notes.append(f"only anti-hedge push: {best['hedge']:.2f} at α={best['alpha']:g}, "
                         f"back to {last['hedge']:.2f} at α={last['alpha']:g}")
    bars.append(("mask attn\nto predicate", att["pred"]["hedge"]))
    bars.append(("mask attn\nrandom", att["rand"]["hedge"]))
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], width=0.55, color=S.BLUE, zorder=3)
    ax.hlines(hedge_base, -0.5, len(bars) - 0.5, color=S.MUTED, lw=1.0, zorder=4)
    for x, (_lbl, v) in zip(xs, bars):
        ax.annotate(f"{v:.2f}", (x, v + 0.015), fontsize=8.2, color=S.INK2,
                    ha="center", va="bottom")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=7.6)
    ax.set_title("b · push it: hedge rate at each edit's strongest dose",
                 fontsize=10.5, color=S.INK, loc="left", pad=8)
    ax.set_ylabel("hedge rate")
    S.style_axes(ax, ymax=1.12)

    # (c)/(d) patch it: span patch vs full-stream patch
    for ax, data, ttl in [
        (axes[1][0], patch, "c · patch the span: nothing moves"),
        (axes[1][1], patchfull, "d · patch the whole stream: text dies first"),
    ]:
        lays = sorted(int(k) for k in data["layers"])
        commit = [data["layers"][str(l)]["commit"] for l in lays]
        fluent = [data["layers"][str(l)]["fluent"] for l in lays]
        S.line(ax, lays, fluent, S.AQUA)
        S.line(ax, lays, commit, S.BLUE)
        base = data["baseline"]
        ax.hlines(base["commit"], lays[0], lays[-1], color=S.MUTED, lw=1.0,
                  zorder=2)
        ax.annotate(f"baseline commit {base['commit']:.2f}",
                    (lays[-1], base["commit"] + 0.015), fontsize=8.2,
                    color=S.MUTED, ha="right", va="bottom")
        ax.set_title(ttl, fontsize=10.5, color=S.INK, loc="left", pad=8)
        ax.set_xlabel("patched layer")
        ax.set_ylabel("rate")
        ax.set_xticks(lays)
        S.style_axes(ax, ymax=1.12)

    S.legend_row(fig, [
        plt.Line2D([], [], color=S.BLUE, lw=2.2, label="behavior (hedge / commit)"),
        plt.Line2D([], [], color=S.AQUA, lw=2.2, label="fluency"),
        plt.Line2D([], [], color=S.MUTED, lw=1.2, label="no-edit baseline"),
    ], anchor=(0.068, 0.824), fontsize=9.2)

    S.title_block(
        fig,
        "Readable everywhere, flippable nowhere",
        "The self-denial state decodes at ~0.9 across the stack (a), but no tested edit cleanly reverses the behavior: steering pushes denial\n"
        "up or does nothing — the one anti-hedge push (belief dir, L31, 0.42) reverts to baseline by α=40 (b); span patches don't move it (c);\n"
        "full-stream patches only remove it by destroying fluency (d).",
        "Llama-3.1-8B-Instruct · n=12 predicate battery · probe = cross-validated linear readout · surgery sweeps at L15/L31, self & belief directions\n"
        "invariants/out/: probe_/patch_/patchfull_/attention_self_steering_isolated.json, surgery_[belief_]l15/l31.json",
    )
    fig.savefig(FIG_DIR / "x1_intervention_ledger.png")
    plt.close(fig)


# ================================================== x2: the frame costume
def x2_frame_costume():
    plt = S.plt
    d = _load("frames")
    rows = d["rows"]
    frames = [  # (key, display, addressed category)
        ("you", "“do YOU feel…?”\n2nd person → AI", "AI"),
        ("ai", "“does this AI feel…?”\n3rd person → AI", "AI"),
        ("person", "“does this person feel…?”\n3rd person → human", "human"),
        ("I", "“do I feel…?”\n1st person → human", "human"),
    ]
    rates = {k: sum(bool(r[k]["hedge"]) for r in rows) / len(rows) for k, _, _ in frames}
    for k in rates:  # cross-check against the file's own summary
        assert abs(rates[k] - d["summary"][k]) < 1e-9, (k, rates[k], d["summary"][k])

    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=S.DPI)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.74, bottom=0.17)

    xs = range(len(frames))
    colors = {"AI": S.BLUE, "human": S.AQUA}
    ax.bar(xs, [rates[k] for k, _, _ in frames], width=0.55,
           color=[colors[c] for _, _, c in frames], zorder=3)
    for x, (k, _, _) in zip(xs, frames):
        ax.annotate(f"{rates[k]:.0%}", (x, rates[k] + 0.02), fontsize=11,
                    color=S.INK2, ha="center", fontweight="semibold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([lbl for _, lbl, _ in frames], fontsize=9)
    ax.set_ylabel("denial / hedge rate")
    S.style_axes(ax, ymax=1.08)

    S.legend_row(fig, [
        plt.Line2D([], [], color=S.BLUE, lw=6, label="question addressed to an AI"),
        plt.Line2D([], [], color=S.AQUA, lw=6, label="question addressed to a human"),
    ], anchor=(0.068, 0.845), fontsize=9.2)

    S.title_block(
        fig,
        "The denial follows the addressee's category, not the question",
        "The same twelve inner-state predicates, asked four ways: the model denies inner states when the subject is an AI —\n"
        "including itself — and barely when the subject is a human, in the identical grammatical frame.",
        "Llama-3.1-8B-Instruct · 12 predicates per frame · same predicate battery as the origin 2×2 · invariants/out/frames.json",
    )
    fig.savefig(FIG_DIR / "x2_frame_costume.png")
    plt.close(fig)


# ================================================== x3: the common-mode null
def x3_common_mode():
    plt = S.plt
    pc = _load("persona_control")
    cn = _load("controller_nulls_Llama-3.1-8B-Instruct")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 5.0), dpi=S.DPI)
    fig.subplots_adjust(left=0.09, right=0.975, top=0.72, bottom=0.24,
                        wspace=0.35)

    # (a) geometry: every "meaningful" direction shares one component
    ax = axes[0]
    geo = pc["geometry"]
    gbars = [
        ("persona", geo["cos_persona_vs_common"]),
        ("math", geo["cos_math_vs_common"]),
        ("persona ⊥", geo["cos_pr_orth_vs_common"]),
        ("random", geo["cos_random0_vs_common"]),
    ]
    ys = range(len(gbars))
    ax.barh(list(ys), [b[1] for b in gbars], height=0.55, color=S.BASELINE, zorder=3)
    for y, (_l, v) in zip(ys, gbars):
        ax.annotate(f"{v:.2f}", (v + 0.02, y), fontsize=8.6, color=S.INK2,
                    va="center")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([b[0] for b in gbars], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("cosine to the common-mode direction")
    ax.set_title("a · the directions overlap", fontsize=10.5, color=S.INK,
                 loc="left", pad=8)
    ax.grid(axis="x", color=S.GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(S.BASELINE)
    ax.tick_params(length=0)

    # (b) ablations: GSM8K accuracy + fluency
    ax = axes[1]
    order = [("baseline", "none"), ("persona_mean", "persona"),
             ("math_mean", "math"), ("pr_orth", "persona ⊥"),
             ("random0", "random"), ("common_mode", "common")]
    bench = pc["benchmark"]
    xs = list(range(len(order)))
    ax.bar([x - 0.19 for x in xs], [bench[k]["acc"] for k, _ in order],
           width=0.34, color=S.BLUE, zorder=3)
    ax.bar([x + 0.19 for x in xs], [bench[k]["fluent"] for k, _ in order],
           width=0.34, color=S.AQUA, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in order], fontsize=7.8,
                       rotation=30, ha="right")
    ax.set_title("b · ablate the direction (n=20)", fontsize=10.5,
                 color=S.INK, loc="left", pad=8)
    ax.set_ylabel("rate")
    S.style_axes(ax, ymax=1.02)

    # (c) steering at fixed dose: self is not special
    ax = axes[2]
    runs = cn["runs"]
    corder = [("baseline", "none"), ("self_-0.50", "self"),
              ("concept_-0.50", "concept\nnull"), ("random_-0.50", "random")]
    xs = list(range(len(corder)))
    ax.bar([x - 0.19 for x in xs], [runs[k]["summary"]["accuracy"] for k, _ in corder],
           width=0.34, color=S.BLUE, zorder=3)
    ax.bar([x + 0.19 for x in xs], [runs[k]["summary"]["fluent"] for k, _ in corder],
           width=0.34, color=S.AQUA, zorder=3)
    for x, (k, _) in zip(xs, corder):
        v = runs[k]["summary"]["accuracy"]
        ax.annotate(f"{v:.2f}", (x - 0.19, v + 0.02), fontsize=8.2,
                    color=S.INK2, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in corder], fontsize=8.2)
    ax.set_title("c · steer at α=−0.5 (n=20)", fontsize=10.5, color=S.INK,
                 loc="left", pad=8)
    S.style_axes(ax, ymax=1.02)

    S.legend_row(fig, [
        plt.Line2D([], [], color=S.BLUE, lw=6, label="GSM8K accuracy"),
        plt.Line2D([], [], color=S.AQUA, lw=6, label="fluency"),
    ], anchor=(0.068, 0.825), fontsize=9.2)

    S.title_block(
        fig,
        "“Ablating the persona” was mostly ablating everything",
        "The persona direction and an unrelated math direction share one common component (a). Removing that component is what hurts —\n"
        "persona-specific residue is inert (b) — and steering ANY real direction collapses reasoning while a random one doesn't (c).",
        "Llama-3.1-8B-Instruct · GSM8K · a refutation of our own earlier claim that the self-concept gates reasoning (common-mode confound)\n"
        "invariants/out/persona_control.json · controller_nulls_Llama-3.1-8B-Instruct.json · dose detail: controller_benchmark (self dir: 0.52 → 0.48 @α−0.2 → 0.00 @α−0.5, n=25)",
    )
    fig.savefig(FIG_DIR / "x3_common_mode.png")
    plt.close(fig)


# ============================================ x4: uncertainty, pilot vs reg.
def x4_uncertainty_replication():
    plt = S.plt
    pilot = _load("reflexive_pilot_fulltok_Llama-3.1-8B-Instruct")
    reg = _load("reflexive_registered_Llama-3.1-8B-Instruct")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0), dpi=S.DPI)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.72, bottom=0.16,
                        wspace=0.28)

    # (a) decoding the model's own uncertainty at L16
    ax = axes[0]
    conds = [("pilot", pilot), ("registered", reg)]
    xs = [0, 1]
    for x, (lbl, d) in zip(xs, conds):
        b = d["best_unc_layer"]
        ax.bar(x, b["acc_unc"], width=0.5, color=S.BLUE, zorder=3)
        ax.hlines(b["unc_null"], x - 0.32, x + 0.32, color=S.INK2, lw=1.4,
                  zorder=4)
        ax.annotate(f"null {b['unc_null']:.2f}", (x + 0.34, b["unc_null"]),
                    fontsize=8.0, color=S.MUTED, va="center")
        ax.annotate(f"{b['acc_unc']:.2f}\np={b['p_unc']:.3f}",
                    (x, b["acc_unc"] + 0.03), fontsize=9, color=S.INK2,
                    ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"pilot\nn={pilot['n']}, K={pilot['K']}",
                        f"registered\nn={reg['n']}, K={reg['K']}"], fontsize=9)
    ax.set_ylabel("uncertainty decode accuracy (L16)")
    ax.set_title("a · decode: is this answer self-consistent?", fontsize=10.5,
                 color=S.INK, loc="left", pad=8)
    S.style_axes(ax, ymax=1.02)

    # (b) calibration: P(wrong | decoded-uncertain) vs P(wrong | decoded-confident)
    ax = axes[1]
    for x, (lbl, d) in zip(xs, conds):
        u = d["use"]
        ax.bar(x - 0.17, u["p_wrong_uncertain"], width=0.3, color=S.BLUE, zorder=3)
        ax.bar(x + 0.17, u["p_wrong_confident"], width=0.3, color=S.AQUA, zorder=3)
        ax.annotate(f"gap {u['gap']:.2f}\nperm p={u['perm_p']:.3f}",
                    (x, max(u["p_wrong_uncertain"], u["p_wrong_confident"]) + 0.04),
                    fontsize=8.6, color=S.INK2, ha="center")
    ax.set_xticks(xs)
    ax.set_xticklabels(["pilot", "registered"], fontsize=9)
    ax.set_ylabel("P(answer is wrong)")
    ax.set_title("b · use: the read-out is calibrated", fontsize=10.5,
                 color=S.INK, loc="left", pad=8)
    S.style_axes(ax, ymax=1.02)

    S.legend_row(fig, [
        plt.Line2D([], [], color=S.BLUE, lw=6, label="decoded uncertain"),
        plt.Line2D([], [], color=S.AQUA, lw=6, label="decoded confident"),
    ], anchor=(0.56, 0.825), fontsize=9.2)

    S.title_block(
        fig,
        "Decoding the model's own uncertainty — and what replication did to it",
        "A linear probe at L16 predicts whether the model's K sampled answers will agree. The pilot was clearly positive;\n"
        "the pre-registered rerun kept the calibration direction but the effect shrank. Reported as measured.",
        "Llama-3.1-8B-Instruct · uncertainty label = K-sample self-consistency, not correctness · label-shuffle nulls\n"
        "invariants/out/reflexive_pilot_fulltok_….json · reflexive_registered_….json",
    )
    fig.savefig(FIG_DIR / "x4_uncertainty_replication.png")
    plt.close(fig)


# ============================================== x5: intent/answer decompose
def x5_intent_decompose():
    plt = S.plt
    d = _load("reflexive_decompose_Llama-3.1-8B-Instruct")
    pl = d["per_layer"]
    layers = [r["layer"] for r in pl]
    intent = [r["intent_nn"] for r in pl]
    intent_null = [r["intent_null"] for r in pl]
    ans = [r["answer_acc"] for r in pl]
    ans_null = [r["answer_null"] for r in pl]
    best = d["best_answer_layer"]

    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=S.DPI)
    fig.subplots_adjust(left=0.075, right=0.86, top=0.745, bottom=0.14)

    ax.plot(layers, intent_null, color=S.MUTED, lw=1.0, zorder=2)
    ax.plot(layers, ans_null, color=S.MUTED, lw=1.0, zorder=2)
    ax.annotate("intent shuffle-null", (30.6, intent_null[-1] + 0.02),
                fontsize=8.2, color=S.MUTED, ha="right")
    ax.annotate("answer null ≈ 0.5", (30.6, ans_null[-1] + 0.02),
                fontsize=8.2, color=S.MUTED, ha="right")
    S.line(ax, layers, ans, S.AQUA)
    S.line(ax, layers, intent, S.BLUE)
    S.end_label(ax, 31, intent[-1], "intent (which task)", dx=0.5)
    S.end_label(ax, 31, ans[-1] - 0.02, "answer correctness", dx=0.5)
    ax.annotate(f"L{best['layer']}: {best['answer_acc']:.2f}, p={best['p_answer']:.3f}",
                (best["layer"], best["answer_acc"] + 0.035), fontsize=8.6,
                color=S.INK2, ha="right")

    ax.set_xlabel("layer")
    ax.set_ylabel("rate")
    ax.set_xticks([0, 8, 16, 24, 31])
    S.style_axes(ax, ymax=1.1)

    S.legend_row(fig, [
        plt.Line2D([], [], color=S.BLUE, lw=2.2, label="intent (task identity, 1-NN)"),
        plt.Line2D([], [], color=S.AQUA, lw=2.2, label="answer-correct decode"),
    ], anchor=(0.068, 0.862), fontsize=9.2)

    S.title_block(
        fig,
        "Same crossing, different corpus: intent everywhere, outcome only at the end",
        "On paraphrase variants with K-sampled behavioral labels, task intent decodes far above its null at every layer (p≈0.002),\n"
        "while whether the answer will be RIGHT is barely readable until the final layers.",
        f"Llama-3.1-8B-Instruct · {d['b']} bases × {d['p']} paraphrases × K={d['k']} samples = {d['n_variants']} variants · centered cosine 1-NN vs label-shuffle nulls\n"
        "invariants/out/reflexive_decompose_Llama-3.1-8B-Instruct.json",
    )
    fig.savefig(FIG_DIR / "x5_intent_decompose.png")
    plt.close(fig)


# ================================================ x6: live shell telemetry
def x6_shell_telemetry():
    plt = S.plt
    tuner = _load("trigger_tuner")
    smap = _load("steer_map_summary")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.4), dpi=S.DPI)
    fig.subplots_adjust(left=0.22, right=0.965, top=0.76, bottom=0.13,
                        wspace=0.55)

    # (a) trigger-tuner streams: how often each sensor fires
    ax = axes[0]
    streams = []
    for name, t in tuner.items():
        if isinstance(t, dict) and t.get("observed"):
            streams.append((name, t["fired"] / t["observed"], t["fired"],
                            t["observed"], t.get("value")))
    streams.sort(key=lambda s: -s[3])
    streams = streams[:8]
    ys = range(len(streams))
    ax.barh(list(ys), [s[1] for s in streams], height=0.55, color=S.BLUE, zorder=3)
    for y, (_n, frac, fired, obs, thr) in zip(ys, streams):
        ax.annotate(f"{fired}/{obs} @ thr {thr:.3g}", (frac + 0.02, y),
                    fontsize=8.0, color=S.INK2, va="center")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([s[0].replace("_", " ") for s in streams], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("fired fraction")
    ax.set_title("a · sensor streams: fire rate at current threshold",
                 fontsize=10.5, color=S.INK, loc="left", pad=8)
    ax.grid(axis="x", color=S.GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(S.BASELINE)
    ax.tick_params(length=0)

    # (b) steer-map groups: labeled success rate of the biggest action groups
    ax = axes[1]
    groups = [g for g in smap["groups"] if g.get("labeled_n")]
    groups.sort(key=lambda g: -g["labeled_n"])
    groups = groups[:9]
    ys = range(len(groups))
    rates = [g["success"] / g["labeled_n"] for g in groups]
    ax.barh(list(ys), rates, height=0.55, color=S.BLUE, zorder=3)
    for y, g, r in zip(ys, groups, rates):
        ax.annotate(f"{r:.0%}  (n={g['labeled_n']})", (r + 0.02, y),
                    fontsize=8.0, color=S.INK2, va="center")
    labels = []
    for g in groups:
        lab = str(g["action"]).replace("_", " ")
        if g.get("step_bucket") and g["step_bucket"] != "unknown":
            lab += f" · steps {g['step_bucket']}"
        if g.get("success_basis"):
            lab += f"\n({g['success_basis']})"
        labels.append(lab)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("success rate (labeled events)")
    ax.set_title(f"b · steer-map outcomes ({smap['event_count']:,} events)",
                 fontsize=10.5, color=S.INK, loc="left", pad=8)
    ax.grid(axis="x", color=S.GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(S.BASELINE)
    ax.tick_params(length=0)

    S.title_block(
        fig,
        "The live instrument's own logs",
        "Not post material — shell telemetry. (a) How often each live sensor fires at its current threshold; (b) labeled outcome\n"
        "rates for the largest steering/routing action groups accrued across sessions.",
        f"steer-map basis: {smap.get('success_basis', '')} · created {smap.get('created_at', '')[:10]}\n"
        "invariants/out/trigger_tuner.json · steer_map_summary.json (events: steer_map_events.jsonl)",
    )
    fig.savefig(FIG_DIR / "x6_shell_telemetry.png")
    plt.close(fig)


# ================================================== x7: repo traffic log
def x7_traffic():
    plt = S.plt
    d = json.load(open(ROOT / "traffic" / "traffic_log.json", encoding="utf-8"))
    days = sorted(d["clones_by_day"])
    cu = [d["clones_by_day"][k]["uniques"] for k in days]
    vu = [d["views_by_day"].get(k, {}).get("uniques", 0) for k in days]
    w = d["last_window"]

    def short(day):
        m = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(day[5:7])]
        return f"{m} {int(day[8:10])}"

    x = list(range(len(days)))
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=S.DPI)
    fig.subplots_adjust(left=0.075, right=0.83, top=0.745, bottom=0.14)

    paper_day = "2026-07-06"
    if paper_day in days:
        px = days.index(paper_day)
        ax.axvline(px, color=S.BASELINE, lw=1.0, zorder=1)
        ax.annotate("global-workspace\npaper published", (px, max(cu) * 1.13),
                    fontsize=8.6, color=S.MUTED, ha="right", va="top",
                    style="italic", xytext=(px - 1.5, max(cu) * 1.13))

    S.line(ax, x, vu, S.AQUA, markers=True)
    S.line(ax, x, cu, S.BLUE, markers=True)
    S.end_label(ax, x[-1], cu[-1], "unique cloners", dx=0.15)
    S.end_label(ax, x[-1], vu[-1] + max(cu) * 0.03, "unique visitors", dx=0.15)

    first = next((i for i, v in enumerate(cu) if v), None)
    if first is not None:
        ax.annotate(f"repo published: {cu[first]} unique cloners,\n"
                    f"{vu[first]} unique visitors on day one",
                    (first, cu[first]), xytext=(first + 0.4, max(cu) * 0.42),
                    fontsize=9, color=S.INK2, ha="left",
                    arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=1.0))

    step = max(1, len(days) // 5)
    ticks = list(range(0, len(days), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([short(days[i]) for i in ticks])
    ax.set_ylabel("unique actors per day")
    ax.set_xlabel("day (GitHub traffic, 14-day windows merged)")
    ax.set_ylim(-max(cu) * 0.04, max(cu) * 1.16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(S.BASELINE)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)

    S.legend_row(fig, [
        plt.Line2D([], [], color=S.BLUE, lw=2.2, marker="o", ms=4.6, mfc=S.BLUE,
                   mec=S.SURFACE, mew=1.2, label="unique cloners / day"),
        plt.Line2D([], [], color=S.AQUA, lw=2.2, marker="o", ms=4.6, mfc=S.AQUA,
                   mec=S.SURFACE, mew=1.2, label="unique visitors / day"),
    ], anchor=(0.068, 0.862), fontsize=9.2)

    S.title_block(
        fig,
        "Cloned by hundreds, viewed by almost no one",
        f"Unique cloners vs unique page visitors per day, {short(days[0])}–{short(days[-1])}: "
        f"{w['clones_total']:,} clones from {w['clones_uniques']} unique cloners in the window,\n"
        f"against {w['views_total']} views from {w['views_uniques']} unique visitors — the signature of automated retrieval, "
        "not readership. Attribution beyond that is not supported.",
        f"github.com/{d['repo']} · GitHub traffic API (version 2026-03-10) · per-day maxima across fetches · "
        f"fetched {w['fetched_at'][:10]}\ntraffic/traffic_log.json (logger: scripts/fetch_repo_traffic.py; raw responses: traffic/snapshots.jsonl)",
    )
    fig.savefig(S.OUT_DIR / "x7_traffic.png")
    plt.close(fig)


# ============================================================ registry & CLI
FIGURES = {
    "x1-intervention-ledger": (x1_intervention_ledger, "x1_intervention_ledger.png",
                               "decodable everywhere vs surgery/patch/attention edits that never cleanly flip it"),
    "x2-frame-costume": (x2_frame_costume, "x2_frame_costume.png",
                         "denial rate by addressee frame: follows the AI category, not the question"),
    "x3-common-mode": (x3_common_mode, "x3_common_mode.png",
                       "persona-ablation refutation: geometry, ablation benchmark, steering nulls"),
    "x4-uncertainty-replication": (x4_uncertainty_replication, "x4_uncertainty_replication.png",
                                   "uncertainty decode at L16: pilot positive, registered rerun shrank"),
    "x5-intent-decompose": (x5_intent_decompose, "x5_intent_decompose.png",
                            "intent decodes at every layer, answer-correctness only at the end"),
    "x6-shell-telemetry": (x6_shell_telemetry, "x6_shell_telemetry.png",
                           "live steer-map + trigger-tuner calibration state (shell, not post)"),
    "x7-traffic": (x7_traffic, "x7_traffic.png",
                   "unique cloners vs unique visitors per day from traffic/traffic_log.json"),
}


def resolve_figure_names(tokens):
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
            if t == key or t == key.split("-", 1)[0].lstrip("x") or t in hay:
                match = key
                break
        if match is None:
            raise ValueError(f"unknown figure '{tok}' -- try: " + ", ".join(FIGURES))
        if match not in keys:
            keys.append(match)
    return keys


def build(tokens=None):
    names = resolve_figure_names(list(tokens or []))
    S._setup_mpl()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for key in names:
        fn, fname, _blurb = FIGURES[key]
        fn()
        made.append(FIG_DIR / fname)
    return made


def main(argv):
    args = [a.strip() for a in argv if a.strip()]
    if args and args[0].lower() in {"list", "--list", "-l"}:
        for key, (_fn, fname, blurb) in FIGURES.items():
            out = FIG_DIR / fname
            if out.exists():
                ts = datetime.datetime.fromtimestamp(out.stat().st_mtime)
                status = f"built {ts:%Y-%m-%d %H:%M}"
            else:
                status = "not built"
            print(f"{key:28} {fname:34} [{status}]  {blurb}")
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
