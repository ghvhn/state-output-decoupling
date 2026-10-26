#!/usr/bin/env python3
"""Batch recalculate probes from their creation definitions.

Usage:
    python scripts/recalc_probes.py [--all] [--probe <name>] [--model <model_name>] [--no-save]

Probes are recalculated strictly according to how they were created:
  - Minted probes: Recomputed from contrastive text framings (a || b) via model forward passes.
  - Composed probes: Recomputed by evaluating the composition formula over newly recalculated components.
  - Adopted probes: Recomputed by linking to the freshly recalculated target probe.
"""

import argparse
import os
import sys
import time

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class _Color:
        def __getattr__(self, name):
            return ""
    Fore = Style = _Color()

# Ensure repository root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from invariants.engine import load_model
from invariants.cognitive_cache import DEFAULT_MODEL
from invariants.probe_recalc import recalc_probes


def parse_args():
    parser = argparse.ArgumentParser(description="Recalculate probes from their creation definitions.")
    parser.add_argument(
        "--probe", "-p",
        action="append",
        dest="probes",
        help="Specific probe name(s) to recalculate. Can be passed multiple times. If omitted, recalculates all.",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        default=True,
        help="Recalculate all probes in the probe directory (default).",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Model name or path to load (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--probe-dir",
        default=os.path.join(ROOT, "invariants", "out", "probes"),
        help="Directory containing stored probe .pt files.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry run: compute recalculated directions without saving to disk.",
    )
    parser.add_argument(
        "--band",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        help="Optional explicit layer range (inclusive, e.g. --band 21 30). By default, probes sense across ALL layers (0..n_layers-1).",
    )
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow downloading model weights from Hugging Face if not found locally.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    probe_dir = os.path.abspath(args.probe_dir)
    explicit_layers = list(range(min(args.band), max(args.band) + 1)) if args.band else None

    print(Fore.CYAN + Style.BRIGHT + "==================================================")
    print("        Probe Creation-Definition Recalculator    ")
    print("==================================================" + Style.RESET_ALL)
    print(f"Probe directory : {probe_dir}")
    print(f"Model           : {args.model}")
    print(f"Mode            : {'Dry-run (no-save)' if args.no_save else 'Save to disk'}")
    if explicit_layers:
        print(f"Sensing depth   : Layers L{explicit_layers[0]}-L{explicit_layers[-1]} (explicit band)")
    else:
        print("Sensing depth   : Full model depth (all layers)")
    if args.probes:
        print(f"Target probes   : {', '.join(args.probes)}")
    else:
        print("Target probes   : All discovered in directory")
    print()

    # Load model
    print(Fore.CYAN + f"[1/2] Loading model '{args.model}'..." + Style.RESET_ALL)
    t0 = time.time()
    try:
        model = load_model(args.model, local_files_only=not args.allow_downloads)
    except Exception as exc:
        print(Fore.RED + f"Failed to load model: {exc}" + Style.RESET_ALL)
        sys.exit(1)
    print(Fore.GREEN + f"Model loaded in {time.time() - t0:.1f}s (device: {model.device}, d_model: {model.d_model})." + Style.RESET_ALL)
    print()

    # Recalculate probes
    print(Fore.CYAN + "[2/2] Recalculating probes according to their creation definitions..." + Style.RESET_ALL)

    def on_progress(name, ptype, pdata):
        layers = sorted(pdata.get("direction", {}).keys())
        layer_span = f"L{layers[0]}-L{layers[-1]} ({len(layers)} layers)" if layers else "no layers"
        print(f"  {Fore.GREEN}OK{Style.RESET_ALL} [{ptype:8s}] {name:30s} -> {layer_span}")

    results = recalc_probes(
        model=model,
        probe_dir=probe_dir,
        probe_names=args.probes if args.probes else None,
        layers=explicit_layers,
        save=not args.no_save,
        on_progress=on_progress,
    )

    print()
    print(Fore.CYAN + Style.BRIGHT + "================ Summary ================" + Style.RESET_ALL)
    print(f"Minted recalculated   : {results['minted_count']}")
    print(f"Composed recalculated : {results['composed_count']}")
    print(f"Adopted recalculated  : {results['adopted_count']}")
    print(f"Total recalculated    : {results['total_recalculated']}")
    if not args.no_save:
        print(f"Files saved to disk   : {results['saved_count']}")

    if results["errors"]:
        print()
        print(Fore.YELLOW + f"Warnings / Errors ({len(results['errors'])}):" + Style.RESET_ALL)
        for name, err in sorted(results["errors"].items()):
            print(f"  {Fore.RED}FAIL{Style.RESET_ALL} {name}: {err}")

    print(Fore.GREEN + Style.BRIGHT + "\nRecalculation complete." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
