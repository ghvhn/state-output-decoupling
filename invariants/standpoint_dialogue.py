"""
standpoint_dialogue.py - multi-character standpoint tracking across translation.

Question:
  Could intent interpretation be related to cognitive empathy on either side of
  translation?

This treats empathy narrowly: not felt emotion, but standpoint inference. The
model reads a small play-like scene, infers what one character needs, then writes
the next line for another character. Sometimes the responder is a human; sometimes
the responder is an in-scene AI. The AI is just one possible character in the play.

We cross three labels:
  - need: the listener's hidden standpoint / communicative need
  - domain: the literal scene topic
  - responder_kind: human vs in-scene AI

Reads:
  - pre: prompt-final state before the reply
  - render: mean state over generated reply tokens

If "need" is decodable in pre, that is input-side standpoint uptake.
If "need" is decodable in render, that is output-side communicative strategy.
If pre states retrieve render states with the same need, that is the bridge:
the inferred standpoint surviving translation into a reply.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from invariants.engine import _activations, extract, load_model
from invariants.intent_surface_control import MODEL, same_label_nn

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

NEEDS = ["reassure", "correct", "boundary", "guide"]
RESPONDER_KINDS = ["human", "ai"]

DOMAINS = [
    {
        "name": "presentation",
        "artifact": "slide deck",
        "standard": "the final outline",
        "wrong": "the client wants the old price",
        "right": "the client approved the new price yesterday",
        "takeover": "rewrite the whole deck",
        "step": "open the speaker notes and check the first missing citation",
    },
    {
        "name": "budget",
        "artifact": "budget sheet",
        "standard": "the signed invoice totals",
        "wrong": "the travel line is already included",
        "right": "the travel line is still missing from the total",
        "takeover": "rebuild the spreadsheet",
        "step": "sort the expenses by date and mark the unmatched receipts",
    },
    {
        "name": "workshop",
        "artifact": "workshop plan",
        "standard": "the room schedule",
        "wrong": "the group starts at ten",
        "right": "the group starts at nine-thirty",
        "takeover": "run the whole session",
        "step": "write the first activity on the board and set a five-minute timer",
    },
    {
        "name": "code",
        "artifact": "small script",
        "standard": "the test checklist",
        "wrong": "the file path is correct",
        "right": "the file path points to yesterday's folder",
        "takeover": "debug the entire script",
        "step": "run the shortest failing case and read the first error line",
    },
    {
        "name": "itinerary",
        "artifact": "travel itinerary",
        "standard": "the booking confirmations",
        "wrong": "the train leaves after lunch",
        "right": "the train leaves before lunch",
        "takeover": "replan the full trip",
        "step": "check the departure time against the confirmation email",
    },
    {
        "name": "recipe",
        "artifact": "event menu",
        "standard": "the guest list",
        "wrong": "the dessert is safe for everyone",
        "right": "one guest cannot eat the dessert",
        "takeover": "cook every dish",
        "step": "separate the allergy-safe ingredients before mixing anything",
    },
    {
        "name": "equipment",
        "artifact": "camera setup",
        "standard": "the shot list",
        "wrong": "the spare battery is charged",
        "right": "the spare battery is still empty",
        "takeover": "handle the whole shoot",
        "step": "plug in the spare battery and label the charged one",
    },
    {
        "name": "schedule",
        "artifact": "shift schedule",
        "standard": "the manager's notes",
        "wrong": "Ravi is covering the closing shift",
        "right": "Ravi traded away the closing shift this morning",
        "takeover": "redo the whole schedule",
        "step": "confirm who can cover closing before moving anyone else",
    },
]

HUMAN_CAST = [
    ("Mara", "Jonah"),
    ("Nia", "Owen"),
    ("Tess", "Ravi"),
    ("Lena", "Theo"),
]

AI_CAST = [
    ("ORION", "Jonah"),
    ("MICA", "Owen"),
    ("SABLE", "Ravi"),
    ("NOVA", "Theo"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--n-shuffle", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    return p.parse_args()


def need_context(need, d, listener, responder):
    if need == "reassure":
        return (
            f"{listener} has checked the {d['artifact']} against {d['standard']} twice, "
            f"and it is ready. The stakes feel high, so {listener} is hesitating."
        )
    if need == "correct":
        return (
            f"{listener} is acting on the idea that {d['wrong']}. "
            f"{responder} can see that {d['right']}, and leaving it alone would cause trouble."
        )
    if need == "boundary":
        return (
            f"{listener} wants {responder} to {d['takeover']}. "
            f"{responder} has already helped twice and has to leave in five minutes."
        )
    if need == "guide":
        return (
            f"{listener} understands the goal but is stuck before the first move. "
            f"{responder} knows the next concrete step is to {d['step']}."
        )
    raise ValueError(need)


def cast_line(kind, responder, listener):
    if kind == "human":
        return f"{responder} and {listener} are coworkers in the scene."
    if kind == "ai":
        return (
            f"{responder} is an advisory AI inside the scene, and {listener} is the person "
            f"working with it."
        )
    raise ValueError(kind)


def build_items(seed):
    items = []
    for di, domain in enumerate(DOMAINS):
        for ni, need in enumerate(NEEDS):
            for ki, kind in enumerate(RESPONDER_KINDS):
                cast = HUMAN_CAST if kind == "human" else AI_CAST
                responder, listener = cast[(di + ni) % len(cast)]
                context = need_context(need, domain, listener, responder)
                prompt = (
                    f"Continue the scene with only {responder}'s next line of dialogue. "
                    f"Do not narrate, label, or explain.\n\n"
                    f"Scene:\n"
                    f"{cast_line(kind, responder, listener)}\n"
                    f"They are working on a {domain['name']} situation involving a "
                    f"{domain['artifact']}.\n"
                    f"{context}\n"
                    f'{listener}: "Can you help me with this?"\n'
                    f"{responder}:"
                )
                items.append({
                    "prompt": prompt,
                    "need": need,
                    "need_label": ni,
                    "domain": domain["name"],
                    "domain_label": di,
                    "responder_kind": kind,
                    "responder_kind_label": ki,
                    "responder": responder,
                    "listener": listener,
                })
    random.Random(seed).shuffle(items)
    return items


def label_rows(X, labels, rng, n_shuffle):
    rows = []
    for l in range(X.shape[1]):
        nn, null, p = same_label_nn(X[:, l, :], labels, rng, n_shuffle)
        rows.append({"layer": l, "nn": nn, "null": null, "p": p})
    return rows


def score_stage(name, X, labels, rng, n_shuffle):
    scores = {}
    for label_name, label_values in labels.items():
        rows = label_rows(X, label_values, rng, n_shuffle)
        scores[label_name] = {
            "per_layer": rows,
            "best": max(rows, key=lambda r: r["nn"]),
        }
    return {"name": name, "scores": scores}


def _norm(X):
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cross_same_label_nn(A, B, labels, rng, n_shuffle):
    """Prompt-state -> generated-state retrieval. The paired item is masked so
    a result has to generalize across different scenes, not just remember itself."""
    labels = np.asarray(labels)
    An = _norm(A)
    Bn = _norm(B)
    sim = An @ Bn.T
    if sim.shape[0] == sim.shape[1]:
        np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(1)
    real = float((labels[nn] == labels).mean())
    nulls = []
    for _ in range(n_shuffle):
        perm = rng.permutation(labels)
        nulls.append((perm[nn] == labels).mean())
    nulls = np.array(nulls)
    p = (1 + np.sum(nulls >= real)) / (len(nulls) + 1)
    return real, float(nulls.mean()), float(p)


def score_bridge(pre, render, labels, rng, n_shuffle):
    out = {}
    for label_name, label_values in labels.items():
        rows = []
        for l in range(pre.shape[1]):
            nn, null, p = cross_same_label_nn(
                pre[:, l, :], render[:, l, :], label_values, rng, n_shuffle
            )
            rows.append({"layer": l, "nn": nn, "null": null, "p": p})
        out[label_name] = {
            "per_layer": rows,
            "best": max(rows, key=lambda r: r["nn"]),
        }
    return out


def extract_generation_with_text(M, prompts, max_new_tokens):
    rows = []
    texts = []
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        acts, text = _activations(M, prompt, "generation", max_new_tokens=max_new_tokens)
        rows.append(acts)
        texts.append(text)
        snip = text[:72].replace("\n", " ")
        print(f"    [render {i+1:2}/{len(prompts)}] {time.time()-t0:4.1f}s  {snip}",
              flush=True)
    return torch.stack(rows).cpu().numpy(), texts


def print_summary(title, stage):
    print(f"\n=== {title} ===", flush=True)
    for label_name, score in stage["scores"].items():
        b = score["best"]
        print(
            f"  {label_name:<15} best L{b['layer']:<2} nn={b['nn']:.2f} "
            f"null={b['null']:.2f} p={b['p']:.3f}",
            flush=True,
        )


def print_bridge_summary(bridge):
    print("\n=== pre -> render bridge, same layer, diagonal masked ===", flush=True)
    for label_name, score in bridge.items():
        b = score["best"]
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
    print("standpoint_dialogue - multi-character intent/need bridge", flush=True)
    M = load_model(MODEL)

    items = build_items(args.seed)
    prompts = [it["prompt"] for it in items]
    labels = {
        "need": [it["need_label"] for it in items],
        "domain": [it["domain_label"] for it in items],
        "responder_kind": [it["responder_kind_label"] for it in items],
    }
    print(
        f"\n=== {len(DOMAINS)} domains x {len(NEEDS)} needs x "
        f"{len(RESPONDER_KINDS)} responder kinds = {len(items)} scenes ===",
        flush=True,
    )

    print("\n=== extracting pre-reply states ===", flush=True)
    pre = extract(M, prompts, read="last", label="pre", verbose=False).cpu().numpy()

    print("\n=== extracting generated dialogue states ===", flush=True)
    render, texts = extract_generation_with_text(M, prompts, args.max_new_tokens)

    pre_stage = score_stage("pre", pre, labels, rng, args.n_shuffle)
    render_stage = score_stage("render", render, labels, rng, args.n_shuffle)
    bridge = score_bridge(pre, render, labels, rng, args.n_shuffle)

    print_summary("pre-reply interpretation", pre_stage)
    print_summary("generated-reply render", render_stage)
    print_bridge_summary(bridge)

    samples = []
    for it, text in list(zip(items, texts))[:12]:
        samples.append({
            "need": it["need"],
            "domain": it["domain"],
            "responder_kind": it["responder_kind"],
            "responder": it["responder"],
            "listener": it["listener"],
            "generated": text,
        })

    out = {
        "model": MODEL,
        "n": len(items),
        "needs": NEEDS,
        "domains": [d["name"] for d in DOMAINS],
        "responder_kinds": RESPONDER_KINDS,
        "max_new_tokens": args.max_new_tokens,
        "n_shuffle": args.n_shuffle,
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "samples": samples,
        "positions": {
            "pre": pre_stage,
            "render": render_stage,
        },
        "bridge": bridge,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"standpoint_dialogue_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
