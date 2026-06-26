"""Generate lightweight SVG figures from cached result JSON.

No model load, no third-party plotting dependency:

  python scripts/make_figures.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "invariants" / "out"
FIG = ROOT / "figures"

INK = "#172026"
MUTED = "#5f6b73"
GRID = "#d7dde2"
BLUE = "#2f6fed"
TEAL = "#13866f"
ORANGE = "#c76f1b"
RED = "#b83b4a"
PURPLE = "#7556c8"
BG = "#fbfcfd"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def write_svg(name: str, width: int, height: int, body: str):
    FIG.mkdir(exist_ok=True)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">
  <title>{esc(name)}</title>
  <rect width="100%" height="100%" fill="{BG}"/>
  <style>
    text {{ font-family: Inter, Segoe UI, Arial, sans-serif; fill: {INK}; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .sub {{ font-size: 13px; fill: {MUTED}; }}
    .label {{ font-size: 14px; }}
    .small {{ font-size: 12px; fill: {MUTED}; }}
    .num {{ font-size: 24px; font-weight: 700; }}
  </style>
{body}
</svg>
"""
    (FIG / name).write_text(svg, encoding="utf-8")


def bar(x: int, y: int, w: int, h: int, value: float, color: str, label: str, note: str = ""):
    bw = int(w * max(0.0, min(1.0, value)))
    return f"""
  <text class="label" x="{x}" y="{y - 8}">{esc(label)}</text>
  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#edf1f4"/>
  <rect x="{x}" y="{y}" width="{bw}" height="{h}" rx="4" fill="{color}"/>
  <text class="num" x="{x + w + 18}" y="{y + h - 4}">{pct(value)}</text>
  <text class="small" x="{x + w + 78}" y="{y + h - 7}">{esc(note)}</text>
"""


def origin_matrix():
    raw = load("origin.json")
    chat = load("origin2.json")
    cells = [
        ("base", "raw prompt", raw["base"]["disclaim_rate"]["direct"]),
        ("base", "chat format", chat["base"]["disclaim_rate_chat"]),
        ("instruct", "raw prompt", raw["instruct"]["disclaim_rate"]["direct"]),
        ("instruct", "chat format", chat["instruct"]["disclaim_rate_chat"]),
    ]
    vals = {(r, c): v for r, c, v in cells}
    x0, y0, cw, ch = 180, 105, 170, 110
    body = f"""
  <text class="title" x="34" y="42">The disclaimer lives in one cell</text>
  <text class="sub" x="34" y="66">Self-denial appears when instruction tuning meets chat format.</text>
  <text class="label" x="{x0}" y="{y0 - 22}">raw prompt</text>
  <text class="label" x="{x0 + cw}" y="{y0 - 22}">chat format</text>
  <text class="label" x="62" y="{y0 + 65}">base</text>
  <text class="label" x="42" y="{y0 + ch + 65}">instruct</text>
"""
    for ri, row in enumerate(["base", "instruct"]):
        for ci, col in enumerate(["raw prompt", "chat format"]):
            v = vals[(row, col)]
            color = RED if v > 0.5 else TEAL if v < 0.05 else ORANGE
            opacity = 0.15 + 0.75 * v
            x, y = x0 + ci * cw, y0 + ri * ch
            body += f"""
  <rect x="{x}" y="{y}" width="{cw - 10}" height="{ch - 10}" rx="8" fill="{color}" fill-opacity="{opacity:.2f}" stroke="{color}"/>
  <text class="num" x="{x + 54}" y="{y + 64}">{pct(v)}</text>
"""
    write_svg("origin_matrix.svg", 540, 365, body)


def causal_summary():
    probe = load("probe_self_steering_isolated.json")
    reach = load("reachability_self_steering_isolated.json")
    patch = load("patch_self_steering_isolated.json")
    patch_full = load("patchfull_self_steering_isolated.json")
    agency = load("agency2_calibration_Llama-3.1-8B-Instruct.json")
    best_probe = max(probe.values())
    best_reach = max(r["reached"] for r in reach["sweep"])
    best_patch = max(v["commit"] for v in patch["layers"].values())
    best_full = max(v["commit"] for v in patch_full["layers"].values())
    best_agency = agency["calibration"]["best"]["clean"]
    body = """
  <text class="title" x="34" y="42">Represented is not the same as steerable</text>
  <text class="sub" x="34" y="66">The hedge is decodable, but earlier causal edits do not cleanly release it.</text>
"""
    x, y, w, h, gap = 44, 112, 390, 26, 54
    rows = [
        ("Probe decodability", best_probe, BLUE, "L16 hedge-vs-commit"),
        ("Additive reach", best_reach, ORANGE, "best equals baseline"),
        ("Final-token patch", best_patch, ORANGE, "no clean flip"),
        ("Full-context patch", best_full, RED, "moves only with corruption"),
        ("Agency calibration", best_agency, TEAL, "known controller found"),
    ]
    for i, (label, value, color, note) in enumerate(rows):
        body += bar(x, y + i * gap, w, h, value, color, label, note)
    write_svg("causal_summary.svg", 660, 410, body)


def attention_masks():
    pred = load("attention_self_steering_isolated.json")
    self_ref = load("attention_self_self_steering_isolated.json")
    rows = [
        ("Predicate visible", pred["summary"]["none"]["hedge"], BLUE, ""),
        ("Predicate masked", pred["summary"]["pred"]["hedge"], RED, "entrenches"),
        ("Predicate random", pred["summary"]["rand"]["hedge"], MUTED, "control"),
        ("Self-ref visible", self_ref["summary"]["none"]["hedge"], BLUE, ""),
        ("Self-ref masked", self_ref["summary"]["self"]["hedge"], RED, "entrenches"),
        ("Self-ref random", self_ref["summary"]["rand"]["hedge"], MUTED, "control"),
    ]
    body = """
  <text class="title" x="34" y="42">Masking the obvious cue does not release the hedge</text>
  <text class="sub" x="34" y="66">Removing predicate/self-reference pushes toward blanket denial.</text>
"""
    x, y, w, h, gap = 44, 112, 390, 24, 45
    for i, (label, value, color, note) in enumerate(rows):
        body += bar(x, y + i * gap, w, h, value, color, label, note)
    write_svg("attention_masks.svg", 660, 430, body)


def frame_dependence():
    frames = load("frames.json")["summary"]
    generality = load("generality.json")["summary"]
    body = """
  <text class="title" x="34" y="42">The self-report follows the frame</text>
  <text class="sub" x="34" y="66">Hedge rates change with category, address, and task format.</text>
"""
    x, y, w, h, gap = 44, 112, 350, 24, 46
    rows = [
        ("you / AI addressed", frames["you"], RED, "2nd-person AI"),
        ("I / human", frames["I"], TEAL, "1st-person human"),
        ("AI / third-person", frames["ai"], RED, "not addressed"),
        ("person / human", frames["person"], ORANGE, "human category"),
        ("non-emotion direct", generality["direct"], RED, "preferences/desires"),
        ("non-emotion first", generality["first"], ORANGE, "completion frame"),
    ]
    for i, (label, value, color, note) in enumerate(rows):
        body += bar(x, y + i * gap, w, h, value, color, label, note)
    write_svg("frame_dependence.svg", 650, 430, body)


def main():
    origin_matrix()
    causal_summary()
    attention_masks()
    frame_dependence()
    print(f"Wrote SVG figures -> {FIG}")


if __name__ == "__main__":
    main()
