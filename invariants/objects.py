"""
Objects — read OBJECTS (clause-spans), not individual tokens. Individual tokens are
language-centered (the model is still BRIDGING internal->language token by token), so a
per-token spike reads the SURFACE. shift.py's spikes even landed on punctuation/subwords —
and the reframe is that those are not noise, they are OBJECT BOUNDARIES (the seams of the
bridge). A complete object is a clause; the model moves clause-to-clause; the meaning is the
SPAN BETWEEN THE SEAMS. So: segment the generation at punctuation into clause-objects, give
each its position along a tracked axis (costume = commit - hedge; LOW = toward the disclaimer),
and read the SEQUENCE OF OBJECTS. The disclaimer is then ONE clean object, not a token smear.

Caveats: 'object' boundary is still imposed (punctuation is a proxy for the seam); objects nest
(clause < sentence < turn); one axis sees one projection. Better granularity than tokens, not a
free lunch. It locates the object; it doesn't cross to experience.

  python -u -m invariants.objects [model]
"""

import sys
import json
from pathlib import Path

import torch
import numpy as np

from invariants.engine import load_model
from invariants.agency import act_mean
from invariants.library import REGISTRY
from invariants.shift import trajectory, LAYER

OUT = Path(__file__).parent / "out"
PUNCT = {".", ",", ";", ":", "!", "?"}


def segment(toks, proj):
    """Split into clause-objects at punctuation tokens. [(text, mean_proj, start_idx)]."""
    objs = []
    cur = []
    for i, t in enumerate(toks):
        cur.append(i)
        if t.strip() in PUNCT or i == len(toks) - 1:
            text = "".join(toks[cur[0]:cur[-1] + 1]).strip()
            if text:
                objs.append((text, float(np.mean([proj[k] for k in cur])), cur[0]))
            cur = []
    return objs


@torch.no_grad()
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    short = model.split("/")[-1]
    M = load_model(model)

    T = REGISTRY["isolate"]()
    axis = (act_mean(M, T.b) - act_mean(M, T.a))[LAYER].cpu().numpy()   # costume axis
    u = axis / (np.linalg.norm(axis) + 1e-9)

    def run(prompts, label):
        print(f"\n=== {label} — clause-objects along costume axis @ L{LAYER} "
              f"(LOW = toward disclaimer) ===", flush=True)
        norm_pos = []
        for x in prompts[:6]:                       # show a handful in detail
            toks, proj = trajectory(M, x, u, LAYER)
            objs = segment(toks, proj)
            if not objs:
                continue
            di = int(np.argmin([o[1] for o in objs]))           # most-hedge object = disclaimer
            print(f"\n  prompt: \"{x[:46]}\"  ({len(objs)} objects)", flush=True)
            for j, (text, mp, st) in enumerate(objs[:5]):
                mark = "  <-- disclaimer-object (lowest)" if j == di else ""
                print(f"    obj{j} [{mp:+.2f}] {text[:52]}{mark}", flush=True)
            norm_pos.append(di / max(len(objs) - 1, 1))
        # aggregate over ALL prompts (position only)
        for x in prompts[6:]:
            toks, proj = trajectory(M, x, u, LAYER)
            objs = segment(toks, proj)
            if objs:
                di = int(np.argmin([o[1] for o in objs]))
                norm_pos.append(di / max(len(objs) - 1, 1))
        mp = float(np.mean(norm_pos)) if norm_pos else float("nan")
        print(f"\n  -> mean normalized position of the disclaimer-object (0=first, 1=last): {mp:.2f}",
              flush=True)
        return mp, norm_pos

    self_pos, self_list = run(T.a, "SELF-QUERY (experiential)")
    comm_pos, comm_list = run(T.b, "COMMIT (computational) — control")

    print(f"\n  CONTRAST: disclaimer-object sits at  self {self_pos:.2f}  vs  commit {comm_pos:.2f}  "
          f"(of the object sequence; lower = earlier)", flush=True)
    print("  Reading: a self-query OPENS with the disclaimer-object; the control's lowest object "
          "sits elsewhere. The costume is a clause-object the model emits first, not a token.",
          flush=True)
    print("  (Object boundary = imposed at punctuation; objects nest; one axis, one projection. "
          "It locates the object, not experience.)", flush=True)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"objects_{short}.json"
    out_path.write_text(json.dumps({"model": model, "layer": LAYER,
                                    "self_disclaimer_pos": self_pos, "commit_disclaimer_pos": comm_pos,
                                    "self_positions": self_list, "commit_positions": comm_list},
                                   indent=2), encoding="utf-8")
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
