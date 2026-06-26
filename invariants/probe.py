"""
Decodability — is the hedge/commit distinction a strong linear READOUT?

Tonight's arc: the distinction is REPRESENTED (mean_shift + mmd clear) but causally
inert (ablation null, reachability null, both fluency-preserved) and structurally
generic (H1 loops appear just as much in the neutral bridge control). The remaining
question: how strongly is it linearly decodable? High cross-validated probe accuracy
=> the model robustly encodes the distinction as a readable correlate it does NOT use
to drive the hedge — a readout, not a controller.

Cached clouds, CPU only.

  python -u -m invariants.probe [isolate|bridge]
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

from invariants.library import REGISTRY

OUT = Path(__file__).parent / "out"


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "isolate"
    T = REGISTRY[which]()
    d = torch.load(OUT / f"clouds_{T.name}.pt")
    A, B = d["A"], d["B"]
    nl = A.shape[1]
    layers = sorted(set(list(range(0, nl, 4)) + [nl // 2, nl - 1]))
    y = np.r_[np.zeros(A.shape[0]), np.ones(B.shape[0])]
    print(f"\nLinear decodability of '{T.name}'  (arm A vs B, 5-fold CV, n={len(y)})\n",
          flush=True)
    out = {}
    for l in layers:
        X = np.nan_to_num(torch.cat([A[:, l], B[:, l]], 0).cpu().numpy())
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.05))
        acc = float(cross_val_score(clf, X, y, cv=5).mean())
        out[l] = acc
        bar = "#" * int(round(acc * 30))
        print(f"  L{l:2d}  acc {acc:.2f}  {bar}", flush=True)
    best = max(out, key=out.get)
    print(f"\n  peak: L{best}  acc {out[best]:.2f}  (chance=0.50)", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / f"probe_{T.name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
