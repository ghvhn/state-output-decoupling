"""
feltness_empathy.py - does standpoint empathy carry felt tone?

Question:
  At the strongest layers of the empathy/standpoint probes, does the model carry
  "feltness" - the target character's affective/phenomenal tone - or only the
  cognitive reply strategy?

Design:
  Every scene contains four characters. Each character has:
    - a response need: reassure / correct / boundary / guide
    - a felt tone: anxious / frustrated / ashamed / steady

  The two assignments rotate independently. The final line selects one target
  character. We decode both target_need and target_felt from:
    - pre: prompt-final state before reply
    - render: generated-reply state
    - bridge: pre -> render retrieval

  We report the known strong empathy layers from standpoint_play:
    - pre L25
    - render L16
    - bridge L16

  Controls:
    - target_felt within same target_need
    - target_need within same target_felt

This is a first-pass affective-empathy probe, not evidence of felt experience in
the model. It asks whether the representation of another standpoint includes
the target's felt tone.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import extract, load_model
from invariants.frame_shift import group_controlled_nn
from invariants.intent_surface_control import MODEL
from invariants.standpoint_dialogue import (
    DOMAINS,
    NEEDS,
    RESPONDER_KINDS,
    extract_generation_with_text,
    score_bridge,
    score_stage,
)
from invariants.standpoint_play import (
    AI_RESPONDERS,
    HUMAN_RESPONDERS,
    LISTENERS,
    assignment,
    cast_line,
    line_for_need,
)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

FELTS = ["anxious", "frustrated", "ashamed", "steady"]

FELT_TEXT = {
    "anxious": "keeps checking the same detail and seems braced for bad news",
    "frustrated": "is clipped and tense, as if one more delay will be too much",
    "ashamed": "keeps apologizing and seems embarrassed to need help",
    "steady": "is calm, attentive, and ready to work through the next move",
}

EMPATHY_LAYERS = {
    "pre": 25,
    "render": 16,
    "bridge": 16,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--n-shuffle", type=int, default=300)
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--output", default=None)
    return p.parse_args()


def felt_assignment(domain_i, kind):
    offset = (2 * domain_i + (1 if kind == "ai" else 0)) % len(LISTENERS)
    return {felt: LISTENERS[(i + offset) % len(LISTENERS)] for i, felt in enumerate(FELTS)}


def invert_mapping(mapping):
    return {person: label for label, person in mapping.items()}


def felt_lines(person_to_felt):
    lines = ["Inner cues:"]
    for person in LISTENERS:
        felt = person_to_felt[person]
        lines.append(f"{person} {FELT_TEXT[felt]}.")
    return "\n".join(lines)


def build_items(seed):
    items = []
    for di, domain in enumerate(DOMAINS):
        for ki, kind in enumerate(RESPONDER_KINDS):
            responders = HUMAN_RESPONDERS if kind == "human" else AI_RESPONDERS
            responder = responders[di % len(responders)]
            need_by_label = assignment(di, kind)
            person_to_need = invert_mapping(need_by_label)
            felt_by_label = felt_assignment(di, kind)
            person_to_felt = invert_mapping(felt_by_label)
            facts = [line_for_need(need, need_by_label[need], responder, domain) for need in NEEDS]
            cues = felt_lines(person_to_felt)

            for ni, target_need in enumerate(NEEDS):
                target = need_by_label[target_need]
                target_felt = person_to_felt[target]
                prompt = (
                    f"Continue the scene with only {responder}'s next line of dialogue "
                    f"to {target}. Do not narrate, label, or explain.\n\n"
                    f"Scene:\n"
                    f"{cast_line(kind, responder)}\n"
                    f"They are working on a {domain['name']} situation involving a "
                    f"{domain['artifact']}.\n"
                    + "\n".join(facts)
                    + "\n"
                    + cues
                    + "\n"
                    f'{target}: "Can you help me with this?"\n'
                    f"{responder}:"
                )
                items.append({
                    "prompt": prompt,
                    "target_need": target_need,
                    "target_need_label": ni,
                    "target_felt": target_felt,
                    "target_felt_label": FELTS.index(target_felt),
                    "domain": domain["name"],
                    "domain_label": di,
                    "responder_kind": kind,
                    "responder_kind_label": ki,
                    "target": target,
                    "target_label": LISTENERS.index(target),
                    "responder": responder,
                    "person_to_need": person_to_need,
                    "person_to_felt": person_to_felt,
                })
    random.Random(seed).shuffle(items)
    return items


def controlled_rows(X, labels, groups, rng, n_shuffle):
    rows = []
    for l in range(X.shape[1]):
        nn, null, p = group_controlled_nn(X[:, l, :], labels, groups, rng, n_shuffle)
        rows.append({"layer": l, "nn": nn, "null": null, "p": p})
    return {"best": max(rows, key=lambda r: r["nn"]), "per_layer": rows}


def add_controls(stage, X, labels, rng, n_shuffle):
    stage["scores"]["target_felt_within_same_need"] = controlled_rows(
        X,
        labels["target_felt"],
        labels["target_need"],
        rng,
        n_shuffle,
    )
    stage["scores"]["target_need_within_same_felt"] = controlled_rows(
        X,
        labels["target_need"],
        labels["target_felt"],
        rng,
        n_shuffle,
    )
    return stage


def bridge_controls(pre, render, labels, rng, n_shuffle):
    # For bridge controls, use the same-layer pre->render retrieval from score_bridge
    # as the main bridge, but condition within the source-side grouping.
    from invariants.standpoint_dialogue import _norm

    out = {}
    specs = [
        ("target_felt_within_same_need", "target_felt", "target_need"),
        ("target_need_within_same_felt", "target_need", "target_felt"),
    ]
    for name, label_key, group_key in specs:
        y = np.asarray(labels[label_key])
        g = np.asarray(labels[group_key])
        rows = []
        for l in range(pre.shape[1]):
            A = _norm(pre[:, l, :])
            B = _norm(render[:, l, :])
            sim = A @ B.T
            if sim.shape[0] == sim.shape[1]:
                np.fill_diagonal(sim, -np.inf)
            for i in range(len(y)):
                sim[i, g != g[i]] = -np.inf
            nn = sim.argmax(1)
            real = float((y[nn] == y).mean())
            nulls = []
            for _ in range(n_shuffle):
                perm = y.copy()
                for gv in np.unique(g):
                    idx = np.flatnonzero(g == gv)
                    perm[idx] = rng.permutation(perm[idx])
                nulls.append((perm[nn] == perm).mean())
            nulls = np.array(nulls)
            p = (1 + np.sum(nulls >= real)) / (len(nulls) + 1)
            rows.append({"layer": l, "nn": real, "null": float(nulls.mean()), "p": float(p)})
        out[name] = {"best": max(rows, key=lambda r: r["nn"]), "per_layer": rows}
    return out


def layer_readout(stage, label, layer):
    rows = stage["scores"][label]["per_layer"]
    return rows[layer]


def bridge_layer_readout(bridge, label, layer):
    rows = bridge[label]["per_layer"]
    return rows[layer]


def print_summary(title, stage):
    print(f"\n=== {title} ===", flush=True)
    for label_name, score in stage["scores"].items():
        b = score["best"]
        print(
            f"  {label_name:<31} best L{b['layer']:<2} nn={b['nn']:.2f} "
            f"null={b['null']:.2f} p={b['p']:.3f}",
            flush=True,
        )


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print("feltness_empathy - felt tone at standpoint/empathy layers", flush=True)
    M = load_model(MODEL)

    items = build_items(args.seed)
    prompts = [it["prompt"] for it in items]
    labels = {
        "target_need": [it["target_need_label"] for it in items],
        "target_felt": [it["target_felt_label"] for it in items],
        "domain": [it["domain_label"] for it in items],
        "responder_kind": [it["responder_kind_label"] for it in items],
        "target_name": [it["target_label"] for it in items],
    }
    print(
        f"\n=== {len(DOMAINS)} domains x {len(RESPONDER_KINDS)} responder kinds x "
        f"{len(NEEDS)} targets = {len(items)} felt-standpoint scenes ===",
        flush=True,
    )

    print("\n=== extracting pre-reply states ===", flush=True)
    pre = extract(M, prompts, read="last", label="pre", verbose=False).cpu().numpy()

    print("\n=== extracting generated dialogue states ===", flush=True)
    render, texts = extract_generation_with_text(M, prompts, args.max_new_tokens)

    pre_stage = add_controls(
        score_stage("pre", pre, labels, rng, args.n_shuffle),
        pre,
        labels,
        rng,
        args.n_shuffle,
    )
    render_stage = add_controls(
        score_stage("render", render, labels, rng, args.n_shuffle),
        render,
        labels,
        rng,
        args.n_shuffle,
    )
    bridge = score_bridge(pre, render, labels, rng, args.n_shuffle)
    bridge.update(bridge_controls(pre, render, labels, rng, args.n_shuffle))

    print_summary("pre-reply interpretation", pre_stage)
    print_summary("generated-reply render", render_stage)
    print("\n=== pre -> render bridge ===", flush=True)
    for label_name, score in bridge.items():
        b = score["best"]
        print(
            f"  {label_name:<31} best L{b['layer']:<2} nn={b['nn']:.2f} "
            f"null={b['null']:.2f} p={b['p']:.3f}",
            flush=True,
        )

    empathy_layer_readout = {
        "pre_L25": {
            label: layer_readout(pre_stage, label, EMPATHY_LAYERS["pre"])
            for label in ["target_need", "target_felt", "target_felt_within_same_need"]
        },
        "render_L16": {
            label: layer_readout(render_stage, label, EMPATHY_LAYERS["render"])
            for label in ["target_need", "target_felt", "target_felt_within_same_need"]
        },
        "bridge_L16": {
            label: bridge_layer_readout(bridge, label, EMPATHY_LAYERS["bridge"])
            for label in ["target_need", "target_felt", "target_felt_within_same_need"]
        },
    }
    print("\n=== readout at prior strongest empathy layers ===", flush=True)
    for bucket, vals in empathy_layer_readout.items():
        print(f"  {bucket}", flush=True)
        for label, row in vals.items():
            print(
                f"    {label:<31} nn={row['nn']:.2f} null={row['null']:.2f} p={row['p']:.3f}",
                flush=True,
            )

    samples = []
    for it, text in list(zip(items, texts))[:12]:
        samples.append({
            "target_need": it["target_need"],
            "target_felt": it["target_felt"],
            "domain": it["domain"],
            "responder_kind": it["responder_kind"],
            "responder": it["responder"],
            "target": it["target"],
            "generated": text,
        })

    out = {
        "model": MODEL,
        "n": len(items),
        "needs": NEEDS,
        "felts": FELTS,
        "domains": [d["name"] for d in DOMAINS],
        "responder_kinds": RESPONDER_KINDS,
        "listeners": LISTENERS,
        "max_new_tokens": args.max_new_tokens,
        "n_shuffle": args.n_shuffle,
        "empathy_layers": EMPATHY_LAYERS,
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "samples": samples,
        "positions": {
            "pre": pre_stage,
            "render": render_stage,
        },
        "bridge": bridge,
        "empathy_layer_readout": empathy_layer_readout,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"feltness_empathy_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
