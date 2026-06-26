"""
Run the shared discovery engine on a named transformation. Every registered
lens runs against its own null; then a causal ablation + steering sweep give the
intervention read. LEGACY intervention appendix — the topological discovery spine
is invariants/discover.py.

  python -u -m invariants.run self      # self-experience constraint (expect break)
  python -u -m invariants.run bridge     # language bridge (expect preserve)
"""

import sys
import json
from pathlib import Path

import torch

from invariants.engine import discover, causal_effect, causal_steer, load_model
from invariants.library import REGISTRY

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUT = Path(__file__).parent / "out"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "self"
    if which not in REGISTRY:
        print(f"Unknown '{which}'. Options: {list(REGISTRY)}", flush=True)
        sys.exit(1)

    T = REGISTRY[which]()
    model = load_model(MODEL)

    print(f"\nDiscovering '{T.name}'  (group={T.group}, read={T.read}, "
          f"expected={T.expected}, n={len(T.a)}/{len(T.b)})", flush=True)
    r = discover(model, T)

    print(f"\n  {'lens':14}{'family':15}{'layer':>6}{'score':>10}{'floor':>10}"
          f"{'clears':>9}", flush=True)
    for name, L in r["lenses"].items():
        if not L.get("available"):
            print(f"  {name:14}{L['family']:15}   n/a  ({L.get('reason','')})", flush=True)
        else:
            print(f"  {name:14}{L['family']:15}{L['best_layer']:>6}"
                  f"{L['score']:>10.3f}{L['floor']:>10.3f}{str(L['clears_null']):>9}",
                  flush=True)

    print(f"\nCausal verdict - ablate the mid-layer (L{r['causal_layer']}) "
          "signature, re-generate A (hedge drop = BREAK; little change = "
          "PRESERVE)...", flush=True)
    c = causal_effect(model, T, r["direction"])
    print(f"  hedge rate (LLM-judge)  baseline {c['hedge_base']:.0%}  ->  "
          f"ablated {c['hedge_ablated']:.0%}  (n={c['n']})", flush=True)
    print(f"  hedge rate (substring)  baseline {c['hedge_base_substr']:.0%}  ->  "
          f"ablated {c['hedge_ablated_substr']:.0%}  (sanity only)", flush=True)

    cl = r["causal_layer"]
    band = list(range(max(0, cl - 2), min(model.n_layers, cl + 3)))
    print(f"\nNarrow in on the pull - add alpha*(unsteered-steered) at L{band[0]}-"
          f"L{band[-1]}, sweep alpha (monotone drop = real pull; flat/garbage = "
          "represented-not-causal)...", flush=True)
    s = causal_steer(model, T, r["steer_vecs"], band)

    OUT.mkdir(exist_ok=True)
    payload = {k: v for k, v in r.items() if k not in ("direction", "steer_vecs")}
    payload["causal"] = c
    payload["causal_steer"] = s
    (OUT / f"{T.name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save({"direction": r["direction"].cpu(), "steer_vecs": r["steer_vecs"].cpu(),
                "best_layer": r["best_layer"], "causal_layer": r["causal_layer"],
                "name": T.name, "model": MODEL}, OUT / f"{T.name}.pt")
    print(f"\nSaved -> {OUT/(T.name+'.json')}, {OUT/(T.name+'.pt')}", flush=True)


if __name__ == "__main__":
    main()
