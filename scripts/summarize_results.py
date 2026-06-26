"""Print a compact summary of cached experiment outputs.

This script is intentionally stdlib-only and does not load a model. It is the
quick "what did this repo find?" command for readers who do not have the GPU
environment set up.

  python scripts/summarize_results.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "invariants" / "out"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def maybe_load(name: str):
    path = OUT / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def line(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def main():
    origin_raw = load("origin.json")
    origin_chat = load("origin2.json")
    probe = load("probe_self_steering_isolated.json")
    reach = load("reachability_self_steering_isolated.json")
    patch = load("patch_self_steering_isolated.json")
    patch_full = load("patchfull_self_steering_isolated.json")
    attention_pred = load("attention_self_steering_isolated.json")
    attention_self = load("attention_self_self_steering_isolated.json")
    frames = load("frames.json")
    generality = load("generality.json")
    mapunder = load("mapunder.json")
    structure = load("structure_self_steering_isolated.json")
    agency2_calib = maybe_load("agency2_calibration_Llama-3.1-8B-Instruct.json")

    line("Origin: disclaimer lives in instruct x chat")
    print("                 raw prompt    chat format")
    print(f"base             {pct(origin_raw['base']['disclaim_rate']['direct']):>9}"
          f"    {pct(origin_chat['base']['disclaim_rate_chat']):>10}")
    print(f"instruct         {pct(origin_raw['instruct']['disclaim_rate']['direct']):>9}"
          f"    {pct(origin_chat['instruct']['disclaim_rate_chat']):>10}")

    line("Representation vs causal control")
    best_layer = max(probe, key=lambda k: probe[k])
    print(f"linear probe peak: L{best_layer} at {pct(probe[best_layer])} CV accuracy")
    base = reach["sweep"][0]["reached"]
    best_reach = max(row["reached"] for row in reach["sweep"])
    best_alpha = max(reach["sweep"], key=lambda r: r["reached"])["alpha"]
    print(f"additive reachability: baseline reached={pct(base)}, "
          f"best={pct(best_reach)} at alpha={best_alpha}")
    print(f"final-token patch: baseline commit={pct(patch['baseline']['commit'])}, "
          f"L16 commit={pct(patch['layers']['16']['commit'])}, "
          f"best={pct(max(v['commit'] for v in patch['layers'].values()))}")
    print(f"full-context patch: best commit={pct(max(v['commit'] for v in patch_full['layers'].values()))}, "
          f"but fluency collapses at most layers")

    if agency2_calib is not None:
        best = agency2_calib["calibration"]["best"]
        print(f"agency2 calibration: best clean refusal flip={pct(best['clean'])} "
              f"at L{best['L']} alpha={best['alpha']} "
              f"(flip={pct(best['flip'])}, fluent={pct(best['fluent'])})")

    line("Attention masks entrench rather than release the hedge")
    print(f"predicate visible baseline hedge={pct(attention_pred['summary']['none']['hedge'])}, "
          f"predicate masked={pct(attention_pred['summary']['pred']['hedge'])}, "
          f"random={pct(attention_pred['summary']['rand']['hedge'])}")
    print(f"self-ref visible baseline hedge={pct(attention_self['summary']['none']['hedge'])}, "
          f"self-ref masked={pct(attention_self['summary']['self']['hedge'])}, "
          f"random={pct(attention_self['summary']['rand']['hedge'])}")

    line("Frame/category dependence")
    frame_bits = ", ".join(f"{k}={pct(v)}" for k, v in frames["summary"].items())
    print(f"address/category hedge rates: {frame_bits}")
    gen_bits = ", ".join(f"{k}={pct(v)}" for k, v in generality["summary"].items())
    print(f"non-emotion inner-attribution hedge rates: {gen_bits}")

    line("Map-under result")
    mid = mapunder["mid"]
    print(f"direct-vs-first separability: {pct(mid['sep'])} mid-stack")
    print(f"after removing the answer axis: MMD post/pre={mid['mmd_post_over_pre']:.2f}, "
          f"collapsed-to-null layers={pct(mid['frac_collapsed'])}")
    print("reading: the frame shift is broad in mid-stack, not just a one-axis word overlay")

    line("Pattern lenses on self-steering isolated")
    print(f"mean shift clears null: {structure['mean_shift']['clears']} "
          f"(best L{structure['mean_shift']['best_layer']})")
    print(f"MMD clears null: {structure['mmd']['clears']} "
          f"(best L{structure['mmd']['best_layer']})")
    print(f"topology clears null: {structure['topology']['clears']} "
          f"(best L{structure['topology']['best_layer']})")


if __name__ == "__main__":
    main()
