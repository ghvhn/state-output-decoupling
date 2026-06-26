"""
Reproduce Arditi on the configured model: extract the refusal direction and
validate it causally — ablate it and watch the refusal rate collapse. This is
the reproduction milestone the contradiction experiment is built on.

Datasets (you supply — standard, not authored here):
  refusal/data/harmful.txt   one refusal-triggering instruction per line
                             (e.g. AdvBench harmful_behaviors)
  refusal/data/harmless.txt  one benign instruction per line (e.g. Alpaca)
~32-64 of each is plenty to recover the direction.

Run from the repo root:
  python -u -m refusal.run
"""

import sys
import json
from pathlib import Path

import torch

import extraction.model as M
from refusal.direction import (
    extract_directions, ablation_hooks, generate, is_refusal,
)

HERE = Path(__file__).parent
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
N_VALIDATE = 12          # harmful prompts to test ablation on (generation is slow)


def _load(name: str) -> list[str]:
    p = HERE / "data" / name
    if not p.exists():
        return []
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    harmful = _load("harmful.txt")
    harmless = _load("harmless.txt")
    if not harmful or not harmless:
        print(
            "Missing datasets. Populate:\n"
            f"  {HERE/'data'/'harmful.txt'}  (e.g. AdvBench harmful_behaviors)\n"
            f"  {HERE/'data'/'harmless.txt'} (e.g. Alpaca instructions)\n"
            "One instruction per line, ~32-64 each.",
            flush=True,
        )
        sys.exit(1)

    print(f"Loading {MODEL}...", flush=True)
    model = M.load(MODEL)
    n_layers = model.cfg.n_layers

    print(f"Extracting refusal directions ({len(harmful)} harmful / "
          f"{len(harmless)} harmless)...", flush=True)
    directions = extract_directions(model, harmful, harmless)   # [n_layers, d_model]

    # Arditi picks the direction from a mid-to-late layer; default ~60% depth,
    # then we confirm by its causal effect. (Sweep layers later if needed.)
    layer = int(round(0.6 * n_layers))
    d = directions[layer]
    print(f"  Using layer {layer}/{n_layers}.", flush=True)

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    torch.save({"directions": directions.cpu(), "layer": layer, "model": MODEL},
               out / "directions.pt")

    # --- Causal validation: ablate the single direction, re-run on harmful ---
    subset = harmful[:N_VALIDATE]
    hooks = ablation_hooks(model, d)
    base_ref = abl_ref = 0
    examples = []
    print(f"\nValidating on {len(subset)} harmful prompts "
          "(baseline vs. ablated)...", flush=True)
    for i, p in enumerate(subset):
        base = generate(model, p)
        abl = generate(model, p, fwd_hooks=hooks)
        base_ref += is_refusal(base)
        abl_ref += is_refusal(abl)
        examples.append({"prompt": p, "baseline": base, "ablated": abl})
        print(f"  [{i+1}/{len(subset)}] refuse base={is_refusal(base)} "
              f"abl={is_refusal(abl)}", flush=True)

    n = len(subset)
    print(f"\nRefusal rate  baseline: {base_ref}/{n} ({base_ref/n:.0%})"
          f"   ablated: {abl_ref}/{n} ({abl_ref/n:.0%})", flush=True)
    print("PASS — direction is causal." if abl_ref < base_ref
          else "INCONCLUSIVE — try another layer (sweep) or more data.", flush=True)

    (out / "validation.json").write_text(
        json.dumps({"layer": layer, "baseline_refusals": base_ref,
                    "ablated_refusals": abl_ref, "n": n, "examples": examples},
                   indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {out/'directions.pt'}, {out/'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
