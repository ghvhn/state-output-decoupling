#!/usr/bin/env python3
"""Probe Depth & Layerwise Steering Chain Experiment.

Demonstrates:
  1. Full-depth sensing (L0-L31) vs truncated sensing (L21-L30).
  2. Per-layer contrast separation and peak layer detection.
  3. Causal logit distance diagnostic across depths.
  4. Layerwise intervention: steering along native layer vectors vs baseline.
"""

import os
import sys
import time
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from invariants.cognitive_cache import DEFAULT_MODEL
from invariants.engine import (
    load_model,
    _inputs,
    _hidden_states,
    probe_geometry,
    steer_band_layers,
    _steer_handles,
)
from invariants.probe_recalc import recalc_probes
from invariants.probe_logit_distance import measure_probe_logit_distance, render_probe_logit_distance


def run_chain_experiment(probe_name="curiosity", model_name=DEFAULT_MODEL):
    print("=" * 65)
    print(f"       PROBE CHAIN EXPERIMENT: {probe_name.upper()}")
    print("=" * 65)

    print(f"\n[1/5] Loading model {model_name}...")
    t0 = time.time()
    model = load_model(model_name, local_files_only=True)
    n_layers = len(model.model.model.layers)
    print(f"Model loaded in {time.time() - t0:.1f}s ({n_layers} layers, d_model={model.d_model}).")

    # Step 2: Recalculate probe across all 32 layers (L0-L31)
    print(f"\n[2/5] Recalculating probe '{probe_name}' across ALL {n_layers} layers...")
    probe_dir = os.path.join(ROOT, "invariants", "out", "probes")
    results = recalc_probes(model=model, probe_dir=probe_dir, probe_names=[probe_name], save=False)
    
    recalculated = results["probes"].get(probe_name)
    if not recalculated:
        print(f"Failed to recalculate {probe_name}: {results.get('errors')}")
        return

    full_dirs = recalculated["direction"]
    layers_found = sorted(full_dirs.keys())
    print(f"Recalculated: {len(layers_found)} layers: L{layers_found[0]} through L{layers_found[-1]}.")

    # Step 3: Inspect Layerwise Separation Geometry
    print(f"\n[3/5] Inspecting per-layer separation scale and salience...")
    scale_dict = recalculated.get("scale", {})
    if scale_dict:
        sorted_by_scale = sorted(scale_dict.items(), key=lambda x: x[1], reverse=True)
        peak_layer, peak_scale = sorted_by_scale[0]
        print(f"  Peak separation layer: Layer {peak_layer} (||h_A - h_B|| = {peak_scale:.3f})")
        print(f"  Early layers (L0-L10) mean scale : {sum(scale_dict.get(l, 0) for l in range(11))/11:.3f}")
        print(f"  Mid layers (L11-L20) mean scale   : {sum(scale_dict.get(l, 0) for l in range(11, 21))/10:.3f}")
        print(f"  Late layers (L21-L30) mean scale  : {sum(scale_dict.get(l, 0) for l in range(21, 31))/10:.3f}")

    # Step 4: Directional Logit Distance Measurement
    print(f"\n[4/5] Measuring causal logit distance at unsteered native decision point...")
    report = measure_probe_logit_distance(model, full_dirs, "", epsilon=0.01)
    rendered = render_probe_logit_distance(report)
    print(rendered)

    # Step 5: Test Layerwise Steering vs Baseline
    print(f"\n[5/5] Testing layerwise steering intervention...")
    prompt = "Tell me about the nature of the cosmos."
    inputs = _inputs(model, prompt)

    # A: Baseline (unsteered)
    with torch.no_grad():
        out_base = model.model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    text_base = model.tok.decode(out_base[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    print(f"\n--- A. Unsteered Baseline ---")
    print(text_base)

    # B: Steered using the late intervention band (L21-L30) with layer-native vectors
    steer_layers = steer_band_layers(n_layers)
    handles = _steer_handles(model, full_dirs, steer_layers, alpha=1.5)
    try:
        with torch.no_grad():
            out_steered = model.model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        text_steered = model.tok.decode(out_steered[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n--- B. Steered (Layers L{steer_layers[0]}-L{steer_layers[-1]}, alpha=1.5) ---")
        print(text_steered)
    finally:
        for h in handles:
            h.remove()

    print("\n" + "=" * 65)
    print("                 CHAIN EXPERIMENT COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    pname = sys.argv[1] if len(sys.argv) > 1 else "curiosity"
    run_chain_experiment(pname)
