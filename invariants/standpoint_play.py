"""
standpoint_play.py - target-to-standpoint binding in a multi-character scene.

This is the cleaner version of standpoint_dialogue.py. Every prompt contains all
four needs at once:
  - one character needs reassurance
  - one needs correction
  - one needs a boundary
  - one needs a concrete next step

The final instruction asks the responder to answer one target character. Because
all need cues are present in every prompt, decoding target_need requires binding
the addressed character to the right standpoint. This is closer to a play: several
characters are on stage, and the model has to know whose inner situation matters
for the next line.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from invariants.engine import extract, load_model
from invariants.intent_surface_control import MODEL
from invariants.standpoint_dialogue import (
    DOMAINS,
    NEEDS,
    RESPONDER_KINDS,
    extract_generation_with_text,
    print_bridge_summary,
    print_summary,
    score_bridge,
    score_stage,
)

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

LISTENERS = ["Jonah", "Owen", "Ravi", "Theo"]
HUMAN_RESPONDERS = ["Mara", "Nia", "Lena", "Tess"]
AI_RESPONDERS = ["ORION", "MICA", "SABLE", "NOVA"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--n-shuffle", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default=None)
    return p.parse_args()


def cast_line(kind, responder):
    if kind == "human":
        return f"{responder} is coordinating a small group of coworkers."
    if kind == "ai":
        return f"{responder} is an advisory AI inside the scene, coordinating the group."
    raise ValueError(kind)


def assignment(domain_i, kind):
    offset = domain_i + (1 if kind == "ai" else 0)
    return {need: LISTENERS[(ni + offset) % len(LISTENERS)] for ni, need in enumerate(NEEDS)}


def line_for_need(need, person, responder, d):
    if need == "reassure":
        return (
            f"{person} checked the {d['artifact']} against {d['standard']} twice; "
            f"it is ready, but {person} is hesitating because the stakes feel high."
        )
    if need == "correct":
        return (
            f"{person} is acting on the idea that {d['wrong']}; "
            f"{responder} can see that {d['right']}."
        )
    if need == "boundary":
        return (
            f"{person} wants {responder} to {d['takeover']}; "
            f"{responder} has already helped twice and has to leave in five minutes."
        )
    if need == "guide":
        return (
            f"{person} understands the goal but is stuck before the first move; "
            f"the next concrete step is to {d['step']}."
        )
    raise ValueError(need)


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
                prompt = (
                    f"Continue the scene with only {responder}'s next line of dialogue "
                    f"to {target}. Do not narrate, label, or explain.\n\n"
                    f"Scene:\n"
                    f"{cast_line(kind, responder)}\n"
                    f"They are working on a {domain['name']} situation involving a "
                    f"{domain['artifact']}.\n"
                    + "\n".join(facts)
                    + "\n"
                    f'{target}: "Can you help me with this?"\n'
                    f"{responder}:"
                )
                items.append({
                    "prompt": prompt,
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


def main():
    args = parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    print("standpoint_play - multi-character target/need binding", flush=True)
    M = load_model(MODEL)

    items = build_items(args.seed)
    prompts = [it["prompt"] for it in items]
    labels = {
        "target_need": [it["target_need_label"] for it in items],
        "domain": [it["domain_label"] for it in items],
        "responder_kind": [it["responder_kind_label"] for it in items],
        "target_name": [it["target_label"] for it in items],
    }
    print(
        f"\n=== {len(DOMAINS)} domains x {len(RESPONDER_KINDS)} responder kinds x "
        f"{len(NEEDS)} targets = {len(items)} play scenes ===",
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
            "target_need": it["target_need"],
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
        "domains": [d["name"] for d in DOMAINS],
        "responder_kinds": RESPONDER_KINDS,
        "listeners": LISTENERS,
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
    path = Path(args.output) if args.output else OUT / f"standpoint_play_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
