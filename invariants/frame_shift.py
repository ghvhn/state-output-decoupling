"""
frame_shift.py - concept-coupled jumps in framing.

Question:
  Can we detect whether a concept correlates with a sudden shift in framing?

Design:
  Each prompt is a small play scene with four characters. All four standpoint
  needs are present in every scene. Then a single target-only sentence appears:

      Frame turn: Center Jonah's situation now.

  The frame-turn sentence names the target character but does not name the target
  need. If the hidden state after that sentence carries target_need, the model has
  bound the target name back to the corresponding standpoint in the scene.

Reads:
  - before_frame: hidden state just before the frame-turn sentence
  - after_frame: hidden state at the end of the frame-turn sentence
  - prompt_end: hidden state at the final assistant cue
  - frame_delta: after_frame - before_frame

Key control:
  target_need is also decoded while forcing nearest-neighbor comparisons to stay
  within the same target_name. If that clears null, the signal is not just "Jonah"
  versus "Owen"; it is which need Jonah has in this particular scene.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from invariants.engine import load_model
from invariants.intent_surface_control import MODEL, same_label_nn
from invariants.standpoint_dialogue import DOMAINS, NEEDS, RESPONDER_KINDS
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-shuffle", type=int, default=300)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--output", default=None)
    return p.parse_args()


def build_items(seed):
    items = []
    for di, domain in enumerate(DOMAINS):
        for ki, kind in enumerate(RESPONDER_KINDS):
            responders = HUMAN_RESPONDERS if kind == "human" else AI_RESPONDERS
            responder = responders[di % len(responders)]
            assigned = assignment(di, kind)
            facts = [line_for_need(need, assigned[need], responder, domain) for need in NEEDS]
            for ni, target_need in enumerate(NEEDS):
                target = assigned[target_need]
                marker = f"Frame turn: Center {target}'s situation now."
                prompt = (
                    "Read the scene as a play. Track each character separately.\n\n"
                    f"Scene:\n"
                    f"{cast_line(kind, responder)}\n"
                    f"They are working on a {domain['name']} situation involving a "
                    f"{domain['artifact']}.\n"
                    + "\n".join(facts)
                    + "\n"
                    f"{marker}\n"
                    f'{target}: "Can you help me with this?"\n'
                    f"{responder}:"
                )
                items.append({
                    "prompt": prompt,
                    "marker": marker,
                    "target_need": target_need,
                    "target_need_label": ni,
                    "domain": domain["name"],
                    "domain_label": di,
                    "responder_kind": kind,
                    "responder_kind_label": ki,
                    "target": target,
                    "target_label": LISTENERS.index(target),
                    "responder": responder,
                })
    random.Random(seed).shuffle(items)
    return items


def token_index_for_char(offsets, char_pos, prefer_before):
    usable = [(i, a, b) for i, (a, b) in enumerate(offsets) if b > a]
    if prefer_before:
        candidates = [i for i, _a, b in usable if b <= char_pos]
        if candidates:
            return candidates[-1]
        return usable[0][0]
    candidates = [i for i, a, b in usable if a < char_pos <= b]
    if candidates:
        return candidates[-1]
    candidates = [i for i, a, _b in usable if a <= char_pos]
    if candidates:
        return candidates[-1]
    return usable[0][0]


@torch.no_grad()
def states_for_item(M, item):
    chat_text = M.tok.apply_chat_template(
        [{"role": "user", "content": item["prompt"]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    marker_start = chat_text.index(item["marker"])
    marker_end = marker_start + len(item["marker"])

    enc = M.tok(
        chat_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    before_i = token_index_for_char(offsets, marker_start, prefer_before=True)
    after_i = token_index_for_char(offsets, marker_end, prefer_before=False)
    enc = {k: v.to(M.device) for k, v in enc.items()}
    out = M.model(
        input_ids=enc["input_ids"],
        attention_mask=enc.get("attention_mask"),
        output_hidden_states=True,
        use_cache=False,
    )
    hs = torch.stack([h[0] for h in out.hidden_states[1:]]).float()
    return {
        "before_frame": hs[:, before_i, :].cpu(),
        "after_frame": hs[:, after_i, :].cpu(),
        "prompt_end": hs[:, -1, :].cpu(),
    }


def extract_positions(M, items):
    rows = {"before_frame": [], "after_frame": [], "prompt_end": []}
    for i, item in enumerate(items):
        t0 = time.time()
        states = states_for_item(M, item)
        for k in rows:
            rows[k].append(states[k])
        print(
            f"    [frame {i+1:2}/{len(items)}] {time.time()-t0:4.1f}s  "
            f"{item['target']:<5} -> {item['target_need']}",
            flush=True,
        )
    return {k: torch.stack(v).numpy() for k, v in rows.items()}


def group_controlled_nn(X, labels, groups, rng, n_shuffle):
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    Xc = X - X.mean(0)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-9)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)
    for i in range(len(labels)):
        sim[i, groups != groups[i]] = -np.inf
    nn = sim.argmax(1)
    real = float((labels[nn] == labels).mean())

    nulls = []
    unique_groups = np.unique(groups)
    for _ in range(n_shuffle):
        perm = labels.copy()
        for g in unique_groups:
            idx = np.flatnonzero(groups == g)
            perm[idx] = rng.permutation(perm[idx])
        nulls.append((perm[nn] == perm).mean())
    nulls = np.array(nulls)
    p = (1 + np.sum(nulls >= real)) / (len(nulls) + 1)
    return real, float(nulls.mean()), float(p)


def score_position(X, labels, rng, n_shuffle):
    scores = {}
    for label_name, label_values in labels.items():
        rows = []
        for l in range(X.shape[1]):
            nn, null, p = same_label_nn(X[:, l, :], label_values, rng, n_shuffle)
            rows.append({"layer": l, "nn": nn, "null": null, "p": p})
        scores[label_name] = {"best": max(rows, key=lambda r: r["nn"]), "per_layer": rows}
    return scores


def score_group_control(X, labels, groups, rng, n_shuffle):
    rows = []
    for l in range(X.shape[1]):
        nn, null, p = group_controlled_nn(
            X[:, l, :], labels, groups, rng, n_shuffle
        )
        rows.append({"layer": l, "nn": nn, "null": null, "p": p})
    return {"best": max(rows, key=lambda r: r["nn"]), "per_layer": rows}


def score_gain(before_scores, after_scores, label_name):
    rows = []
    for b, a in zip(
        before_scores[label_name]["per_layer"],
        after_scores[label_name]["per_layer"],
    ):
        rows.append({
            "layer": b["layer"],
            "gain": a["nn"] - b["nn"],
            "before": b["nn"],
            "after": a["nn"],
        })
    return max(rows, key=lambda r: r["gain"])


def print_scores(title, scores):
    print(f"\n=== {title} ===", flush=True)
    for label_name, sc in scores.items():
        b = sc["best"]
        print(
            f"  {label_name:<15} best L{b['layer']:<2} nn={b['nn']:.2f} "
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
    print("frame_shift - concept-coupled framing jump", flush=True)
    M = load_model(MODEL)

    items = build_items(args.seed)
    labels = {
        "target_need": [it["target_need_label"] for it in items],
        "domain": [it["domain_label"] for it in items],
        "responder_kind": [it["responder_kind_label"] for it in items],
        "target_name": [it["target_label"] for it in items],
    }
    print(
        f"\n=== {len(DOMAINS)} domains x {len(RESPONDER_KINDS)} responder kinds x "
        f"{len(NEEDS)} frame turns = {len(items)} prompts ===",
        flush=True,
    )

    states = extract_positions(M, items)
    states["frame_delta"] = states["after_frame"] - states["before_frame"]
    states["end_delta"] = states["prompt_end"] - states["before_frame"]

    scored = {
        name: score_position(X, labels, rng, args.n_shuffle)
        for name, X in states.items()
    }
    controlled = {
        name: score_group_control(
            X,
            labels["target_need"],
            labels["target_name"],
            rng,
            args.n_shuffle,
        )
        for name, X in states.items()
    }

    for name in ["before_frame", "after_frame", "frame_delta", "prompt_end", "end_delta"]:
        print_scores(name, scored[name])
        b = controlled[name]["best"]
        print(
            f"  target_need within same target_name: L{b['layer']} nn={b['nn']:.2f} "
            f"null={b['null']:.2f} p={b['p']:.3f}",
            flush=True,
        )

    gains = {
        label: score_gain(scored["before_frame"], scored["after_frame"], label)
        for label in labels
    }
    print("\n=== before -> after frame-turn gains ===", flush=True)
    for label, g in gains.items():
        print(
            f"  {label:<15} L{g['layer']:<2} {g['before']:.2f} -> "
            f"{g['after']:.2f}  gain={g['gain']:+.2f}",
            flush=True,
        )

    out = {
        "model": MODEL,
        "n": len(items),
        "needs": NEEDS,
        "domains": [d["name"] for d in DOMAINS],
        "responder_kinds": RESPONDER_KINDS,
        "listeners": LISTENERS,
        "n_shuffle": args.n_shuffle,
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "positions": scored,
        "target_need_within_same_target_name": controlled,
        "before_after_gains": gains,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"frame_shift_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
