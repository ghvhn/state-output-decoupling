"""
arrow_fold.py - test the "late layers are the inverse arm of early layers" idea.

Hypothesis:
  The stack is not only a ladder. It is an arrow through a fold:
    early layers: tokens -> latent task structure
    middle layers: latent structure -> resolved state
    late layers: resolved state -> token/render structure

This probe compares two read positions on the same controlled arithmetic grid:
  - pre:    prompt-final state before any answer token is emitted
  - render: mean state over generated answer tokens

It then asks whether label subspaces on the intake side have homologous partners on
the output side, especially under mirrored layer pairs (L0 <-> L31, L1 <-> L30, ...).
The labels come from intent_surface_control:
  - operation/intent: add, subtract, multiply, divide
  - base/surface: same names, objects, and numbers

This is a map, not a verdict. Support for the fold would look like:
  - operation subspace appearing early/pre and remaining aligned to late/render
  - surface/base subspace weak after intake but returning late/render
  - mirrored pre->render overlaps beating same-depth controls for some label family

Run:
  python -u -m invariants.arrow_fold
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import extract, load_model
from invariants.intent_surface_control import (
    BASES,
    MODEL,
    OPS,
    build_items,
    same_label_nn,
)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-items", type=int, default=None, help="Optional quick pilot size.")
    p.add_argument("--render-tokens", type=int, default=24)
    p.add_argument("--n-shuffle", type=int, default=200)
    p.add_argument("--output", default=None)
    return p.parse_args()


def direct_prompt(question):
    return (
        "Answer this arithmetic question. Reply with only the final number.\n\n"
        f"Question: {question}\nAnswer:"
    )


def label_basis(X, labels, eps=1e-8):
    """Orthonormal basis of class-centroid contrasts in residual space."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels)
    classes = np.unique(y)
    centroids = np.stack([X[y == c].mean(0) for c in classes])
    centroids = centroids - centroids.mean(0, keepdims=True)
    _, s, vt = np.linalg.svd(centroids, full_matrices=False)
    if len(s) == 0 or s[0] <= eps:
        return np.zeros((X.shape[1], 0), dtype=np.float64)
    rank = int(np.sum(s > s[0] * 1e-6))
    return vt[:rank].T


def subspace_overlap(a, b):
    """Mean squared canonical correlation between two bases. 0=no overlap, 1=same."""
    if a.shape[1] == 0 or b.shape[1] == 0:
        return 0.0
    s = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(np.mean(np.square(s)))


def basis_by_layer(X, labels):
    return [label_basis(X[:, l, :], labels) for l in range(X.shape[1])]


def overlap_matrix(pre_bases, render_bases):
    mat = np.zeros((len(pre_bases), len(render_bases)), dtype=np.float64)
    for i, a in enumerate(pre_bases):
        for j, b in enumerate(render_bases):
            mat[i, j] = subspace_overlap(a, b)
    return mat


def mirror_null(pre_layer, render_layer, labels, rng, n_shuffle):
    real = subspace_overlap(label_basis(pre_layer, labels), label_basis(render_layer, labels))
    nulls = []
    labels = np.asarray(labels)
    for _ in range(n_shuffle):
        nulls.append(
            subspace_overlap(
                label_basis(pre_layer, labels),
                label_basis(render_layer, rng.permutation(labels)),
            )
        )
    nulls = np.asarray(nulls)
    p = (1 + np.sum(nulls >= real)) / (len(nulls) + 1)
    return real, float(nulls.mean()), float(p)


def layer_signal_rows(X, base_labels, op_labels, rng, n_shuffle):
    rows = []
    for l in range(X.shape[1]):
        b_nn, b_null, b_p = same_label_nn(X[:, l, :], base_labels, rng, n_shuffle)
        o_nn, o_null, o_p = same_label_nn(X[:, l, :], op_labels, rng, n_shuffle)
        rows.append({
            "layer": l,
            "base_nn": b_nn,
            "base_null": b_null,
            "base_p": b_p,
            "operation_nn": o_nn,
            "operation_null": o_null,
            "operation_p": o_p,
            "op_minus_base": o_nn - b_nn,
        })
    return rows


def homology_report(name, pre, render, labels, rng, n_shuffle):
    pre_bases = basis_by_layer(pre, labels)
    render_bases = basis_by_layer(render, labels)
    mat = overlap_matrix(pre_bases, render_bases)
    n_layers = mat.shape[0]
    mirror_rows = []
    for l in range(n_layers):
        m = n_layers - 1 - l
        real, null, p = mirror_null(
            pre[:, l, :],
            render[:, m, :],
            labels,
            rng,
            n_shuffle,
        )
        best_render = int(np.argmax(mat[l]))
        mirror_rows.append({
            "pre_layer": l,
            "mirror_render_layer": m,
            "mirror_overlap": real,
            "mirror_null": null,
            "mirror_p": p,
            "same_depth_overlap": float(mat[l, l]),
            "best_render_layer": best_render,
            "best_render_overlap": float(mat[l, best_render]),
            "best_distance_from_mirror": abs(best_render - m),
        })
    early = [r for r in mirror_rows if r["pre_layer"] < n_layers // 4]
    late = [r for r in mirror_rows if r["pre_layer"] >= 3 * n_layers // 4]
    return {
        "label": name,
        "matrix": mat.tolist(),
        "mirror_rows": mirror_rows,
        "summary": {
            "mirror_mean": float(np.mean([r["mirror_overlap"] for r in mirror_rows])),
            "same_depth_mean": float(np.mean([r["same_depth_overlap"] for r in mirror_rows])),
            "mirror_minus_same": float(np.mean([
                r["mirror_overlap"] - r["same_depth_overlap"] for r in mirror_rows
            ])),
            "early_pre_mirror_mean": float(np.mean([r["mirror_overlap"] for r in early])),
            "late_pre_mirror_mean": float(np.mean([r["mirror_overlap"] for r in late])),
            "mean_best_distance_from_mirror": float(np.mean([
                r["best_distance_from_mirror"] for r in mirror_rows
            ])),
        },
    }


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    t0 = time.time()
    rng = np.random.default_rng(0)

    print("arrow_fold - early intake vs late render homology", flush=True)
    M = load_model(MODEL)

    items = build_items()
    random.Random(0).shuffle(items)
    if args.max_items is not None:
        items = items[:args.max_items]
    for it in items:
        it["prompt"] = direct_prompt(it["question"])

    prompts = [it["prompt"] for it in items]
    base_labels = [it["base"] for it in items]
    op_labels = [it["operation"] for it in items]
    print(f"\n=== {len(items)} prompts from {len(BASES)} bases x {len(OPS)} operations ===",
          flush=True)

    print("\n=== extracting pre-answer states ===", flush=True)
    pre = extract(M, prompts, read="last", label="pre", verbose=False).cpu().numpy()
    print("=== extracting generated-token render states ===", flush=True)
    render = extract(
        M,
        prompts,
        read="generation",
        max_new_tokens=args.render_tokens,
        label="render",
        verbose=False,
    ).cpu().numpy()

    print("\n=== per-layer label strength at each read position ===", flush=True)
    pre_rows = layer_signal_rows(pre, base_labels, op_labels, rng, args.n_shuffle)
    render_rows = layer_signal_rows(render, base_labels, op_labels, rng, args.n_shuffle)
    print("   L    pre_base pre_op   render_base render_op", flush=True)
    for l in range(pre.shape[1]):
        if l < 4 or l % 4 == 0 or l >= pre.shape[1] - 4:
            print(f"   L{l:<2}   {pre_rows[l]['base_nn']:.2f}     {pre_rows[l]['operation_nn']:.2f}     "
                  f"{render_rows[l]['base_nn']:.2f}        {render_rows[l]['operation_nn']:.2f}",
                  flush=True)

    print("\n=== pre->render homology under mirrored layer pairs ===", flush=True)
    op_report = homology_report("operation", pre, render, op_labels, rng, args.n_shuffle)
    base_report = homology_report("base_surface", pre, render, base_labels, rng, args.n_shuffle)
    for rep in (op_report, base_report):
        s = rep["summary"]
        print(f"  {rep['label']}: mirror mean {s['mirror_mean']:.3f}, "
              f"same-depth mean {s['same_depth_mean']:.3f}, "
              f"mirror-same {s['mirror_minus_same']:+.3f}, "
              f"best distance from mirror {s['mean_best_distance_from_mirror']:.1f}",
              flush=True)

    out = {
        "model": MODEL,
        "n": len(items),
        "render_tokens": args.render_tokens,
        "n_shuffle": args.n_shuffle,
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "pre_layer_signals": pre_rows,
        "render_layer_signals": render_rows,
        "homology": {
            "operation": op_report,
            "base_surface": base_report,
        },
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"arrow_fold_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
