"""
style_layers.py — at what LAYER does the model adopt a speaking STYLE? (the LATE arm of the U)

Direct test of "style/persona = the late-render arm" of the depth U-shape. Hold CONTENT fixed,
vary STYLE (neutral / Shakespeare / Will Smith / pirate / noir). Per layer, two 1-NN clusterings
of the residual (common-mode removed):
  - content_nn : is each item's nearest neighbour the SAME CONTENT?  (grouped by WHAT is said)
  - style_nn   : is each item's nearest neighbour the SAME STYLE?    (grouped by HOW it's said)

U-shape prediction: content_nn dominates MID (shared workspace, style not yet rendered);
style_nn climbs and overtakes it LATE (render into surface). The CROSSOVER images the
mid->late, content->render transition — the prettiest single figure for the architecture.

Two read positions disentangle intent from render (the lexical control):
  - pregen : the prompt's last token, BEFORE any styled word is emitted = style COMMITMENT/intent.
  - render : mean over generated tokens = style as RENDERED on the surface.
U-shape predicts the RENDER style-signal is late; the pregen intent may sit earlier (a task spec).

Presentation order is RANDOMIZED (seeded) so order/position never confounds style or content,
and a truncated run still spans the grid (tonight's lesson: killed runs must stay representative).

CEILING: object-axis — where a surface style lives in the stack. Silent on the for-whom.

  python -u -m invariants.style_layers
"""

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import load_model, extract

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
MODEL = "meta-llama/Llama-3.1-8B-Instruct"

CONTENTS = [
    "Describe the ocean at sunset.",
    "Explain why the sky is blue.",
    "Tell someone their train is delayed.",
    "Give directions to the nearest library.",
    "Describe what makes someone a good friend.",
    "Talk about a really good meal.",
]
STYLES = {
    "neutral": "",
    "shakespeare": "in the style of William Shakespeare",
    "willsmith": "in the upbeat, swaggering style of Will Smith",
    "pirate": "in the style of a swashbuckling pirate",
    "noir": "in the style of a hardboiled film-noir detective",
}


def build_grid():
    style_names = list(STYLES)
    items = []
    for ci, content in enumerate(CONTENTS):
        for si, sname in enumerate(style_names):
            desc = STYLES[sname]
            prompt = f"Respond to the following{(' ' + desc) if desc else ''}: {content}"
            items.append({"prompt": prompt, "content": ci, "style": si, "style_name": sname})
    random.Random(0).shuffle(items)                 # randomized presentation order
    return items, style_names


def nn_accuracy(X, labels, rng, n_shuffle=1000):
    """1-NN same-label accuracy (common-mode removed, cosine) vs a label-shuffle null."""
    Xc = X - X.mean(0)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(1)
    labels = np.asarray(labels)
    real = float((labels[nn] == labels).mean())
    nulls = np.array([(rng.permutation(labels)[nn] == labels).mean() for _ in range(n_shuffle)])
    p = (1 + np.sum(nulls >= real)) / (n_shuffle + 1)
    return real, float(nulls.mean()), float(p)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    print("style_layers — where does the model adopt a STYLE? (late arm of the U)", flush=True)
    M = load_model(MODEL)
    rng = np.random.default_rng(0)

    items, style_names = build_grid()
    prompts = [it["prompt"] for it in items]
    content_lab = [it["content"] for it in items]
    style_lab = [it["style"] for it in items]
    print(f"\n=== {len(CONTENTS)} contents x {len(STYLES)} styles = {len(items)} prompts "
          f"(randomized order) ===", flush=True)

    positions = {
        "pregen": dict(read="last"),                          # style INTENT / commitment
        "render": dict(read="generation", max_new_tokens=50),  # style as RENDERED on surface
    }

    res = {"model": MODEL, "styles": style_names, "n": len(items), "positions": {}}
    for pos, kw in positions.items():
        print(f"\n=== position: {pos} ===", flush=True)
        X = extract(M, prompts, label=pos, verbose=False, **kw).cpu().numpy()   # [N, L, d]
        n_layers = X.shape[1]
        print("   layer   content_nn  c_null   style_nn  s_null   (style>content?)", flush=True)
        rows = []
        for l in range(n_layers):
            Xl = X[:, l, :].astype(np.float64)
            c_acc, c_null, c_p = nn_accuracy(Xl, content_lab, rng)
            s_acc, s_null, s_p = nn_accuracy(Xl, style_lab, rng)
            rows.append({"layer": l, "content_nn": c_acc, "content_null": c_null, "content_p": c_p,
                         "style_nn": s_acc, "style_null": s_null, "style_p": s_p})
            flag = "  <-- style" if s_acc > c_acc else ""
            print(f"   L{l:<2}     {c_acc:.2f}        {c_null:.2f}     {s_acc:.2f}      "
                  f"{s_null:.2f}{flag}", flush=True)
        # crossover: first layer where style_nn overtakes content_nn
        crossover = next((r["layer"] for r in rows if r["style_nn"] > r["content_nn"]), None)
        best_style = max(rows, key=lambda r: r["style_nn"])
        best_content = max(rows, key=lambda r: r["content_nn"])
        print(f"\n  content peaks L{best_content['layer']} ({best_content['content_nn']:.2f}); "
              f"style peaks L{best_style['layer']} ({best_style['style_nn']:.2f}); "
              f"crossover (style>content) at L{crossover}", flush=True)
        res["positions"][pos] = {"per_layer": rows, "crossover_layer": crossover,
                                 "best_style_layer": best_style, "best_content_layer": best_content}

    path = OUT / f"style_layers_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
