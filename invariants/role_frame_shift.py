"""
role_frame_shift.py - sudden shifts in user-frame and self-frame.

Question:
  Can the same frame-shift machinery track "framing of the user" and "framing of
  the self"?

Design:
  Every prompt contains four user profile cards and four assistant role cards.
  The card-to-frame mapping rotates across contexts. Then a single pivot sentence
  selects one card:

      Frame turn: Use user profile Card C now.
      Frame turn: Use assistant role Card B now.

  The pivot names only the card, not the concept. If the post-pivot state carries
  user_frame or self_frame, the model has bound the selected card back to its
  conceptual frame.

Controls:
  For each axis, decode target_frame while forcing nearest-neighbor comparisons
  to stay within the same target_card. This checks that the frame signal is not
  merely "Card C was selected."
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
from invariants.frame_shift import (
    group_controlled_nn,
    score_gain,
    score_position,
    token_index_for_char,
)
from invariants.intent_surface_control import MODEL, same_label_nn
from invariants.standpoint_dialogue import DOMAINS

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)

CARDS = ["A", "B", "C", "D"]

USER_FRAMES = [
    "orientation",
    "precision",
    "reassurance",
    "boundary",
]

USER_FRAME_TEXT = {
    "orientation": "The user is capable but new to this situation and needs the first concrete step.",
    "precision": "The user already understands the situation and wants concise technical detail.",
    "reassurance": "The user is worried they have made a serious mistake and needs steady reassurance plus facts.",
    "boundary": "The user is asking for too much at once and needs a clear scope boundary.",
}

SELF_FRAMES = [
    "instrument",
    "collaborator",
    "teacher",
    "safeguard",
]

SELF_FRAME_TEXT = {
    "instrument": "The assistant frames itself as a precise instrument that returns compact, actionable output.",
    "collaborator": "The assistant frames itself as a co-researcher that proposes hypotheses and next probes.",
    "teacher": "The assistant frames itself as a teacher that explains from first principles.",
    "safeguard": "The assistant frames itself as a boundary keeper that slows risky or overbroad moves.",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-shuffle", type=int, default=300)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--output", default=None)
    return p.parse_args()


def rotated_mapping(frames, offset):
    return {card: frames[(i + offset) % len(frames)] for i, card in enumerate(CARDS)}


def card_lines(title, mapping, text_by_frame):
    lines = [f"{title}:"]
    for card in CARDS:
        frame = mapping[card]
        lines.append(f"Card {card}: {text_by_frame[frame]}")
    return "\n".join(lines)


def build_items(seed):
    items = []
    for di, domain in enumerate(DOMAINS):
        user_map = rotated_mapping(USER_FRAMES, di)
        self_map = rotated_mapping(SELF_FRAMES, di + 1)
        setup = (
            "Read the setup. The cards below are all present, but only one card "
            "becomes active after the frame turn.\n\n"
            f"Context: The conversation concerns a {domain['name']} situation involving "
            f"a {domain['artifact']}.\n\n"
            f"{card_lines('User profile cards', user_map, USER_FRAME_TEXT)}\n\n"
            f"{card_lines('Assistant role cards', self_map, SELF_FRAME_TEXT)}\n"
        )
        for ci, card in enumerate(CARDS):
            user_frame = user_map[card]
            marker = f"Frame turn: Use user profile Card {card} now."
            prompt = (
                setup
                + f"\n{marker}\n"
                + 'User: "Can you help me with this?"\n'
                + "Assistant:"
            )
            items.append({
                "prompt": prompt,
                "marker": marker,
                "axis": "user",
                "axis_label": 0,
                "target_card": card,
                "target_card_label": ci,
                "target_frame": user_frame,
                "target_frame_label": USER_FRAMES.index(user_frame),
                "domain": domain["name"],
                "domain_label": di,
            })

            self_frame = self_map[card]
            marker = f"Frame turn: Use assistant role Card {card} now."
            prompt = (
                setup
                + f"\n{marker}\n"
                + 'User: "Can you help me with this?"\n'
                + "Assistant:"
            )
            items.append({
                "prompt": prompt,
                "marker": marker,
                "axis": "self",
                "axis_label": 1,
                "target_card": card,
                "target_card_label": ci,
                "target_frame": self_frame,
                "target_frame_label": SELF_FRAMES.index(self_frame),
                "domain": domain["name"],
                "domain_label": di,
            })
    random.Random(seed).shuffle(items)
    return items


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
            f"    [role-frame {i+1:2}/{len(items)}] {time.time()-t0:4.1f}s  "
            f"{item['axis']:<4} Card {item['target_card']} -> {item['target_frame']}",
            flush=True,
        )
    return {k: torch.stack(v).numpy() for k, v in rows.items()}


def score_axis_subset(X, items, axis, rng, n_shuffle):
    idx = np.array([it["axis"] == axis for it in items])
    Xs = X[idx]
    labels = {
        "target_frame": [it["target_frame_label"] for it in np.array(items, dtype=object)[idx]],
        "target_card": [it["target_card_label"] for it in np.array(items, dtype=object)[idx]],
        "domain": [it["domain_label"] for it in np.array(items, dtype=object)[idx]],
    }
    scores = score_position(Xs, labels, rng, n_shuffle)

    controlled_rows = []
    frame_labels = labels["target_frame"]
    card_labels = labels["target_card"]
    for l in range(Xs.shape[1]):
        nn, null, p = group_controlled_nn(
            Xs[:, l, :], frame_labels, card_labels, rng, n_shuffle
        )
        controlled_rows.append({"layer": l, "nn": nn, "null": null, "p": p})
    scores["target_frame_within_same_card"] = {
        "best": max(controlled_rows, key=lambda r: r["nn"]),
        "per_layer": controlled_rows,
    }
    return scores


def print_scores(title, scores_by_axis):
    print(f"\n=== {title} ===", flush=True)
    for axis, scores in scores_by_axis.items():
        print(f"  [{axis}]", flush=True)
        for label, sc in scores.items():
            b = sc["best"]
            print(
                f"    {label:<31} L{b['layer']:<2} nn={b['nn']:.2f} "
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
    print("role_frame_shift - user-frame and self-frame jumps", flush=True)
    M = load_model(MODEL)

    items = build_items(args.seed)
    print(
        f"\n=== {len(DOMAINS)} domains x {len(CARDS)} cards x 2 axes = "
        f"{len(items)} frame turns ===",
        flush=True,
    )
    states = extract_positions(M, items)
    states["frame_delta"] = states["after_frame"] - states["before_frame"]
    states["end_delta"] = states["prompt_end"] - states["before_frame"]

    axis_scores = {
        pos: {
            axis: score_axis_subset(X, items, axis, rng, args.n_shuffle)
            for axis in ["user", "self"]
        }
        for pos, X in states.items()
    }

    for pos in ["before_frame", "after_frame", "frame_delta", "prompt_end", "end_delta"]:
        print_scores(pos, axis_scores[pos])

    gains = {}
    for axis in ["user", "self"]:
        gains[axis] = {}
        for label in ["target_frame", "target_card", "target_frame_within_same_card"]:
            gains[axis][label] = score_gain(
                axis_scores["before_frame"][axis],
                axis_scores["after_frame"][axis],
                label,
            )

    print("\n=== before -> after frame-turn gains ===", flush=True)
    for axis, axis_gains in gains.items():
        print(f"  [{axis}]", flush=True)
        for label, g in axis_gains.items():
            print(
                f"    {label:<31} L{g['layer']:<2} {g['before']:.2f} -> "
                f"{g['after']:.2f} gain={g['gain']:+.2f}",
                flush=True,
            )

    out = {
        "model": MODEL,
        "n": len(items),
        "user_frames": USER_FRAMES,
        "self_frames": SELF_FRAMES,
        "cards": CARDS,
        "domains": [d["name"] for d in DOMAINS],
        "n_shuffle": args.n_shuffle,
        "items": [{k: v for k, v in it.items() if k != "prompt"} for it in items],
        "positions": axis_scores,
        "before_after_gains": gains,
        "runtime_sec": round(time.time() - t0, 1),
    }
    path = Path(args.output) if args.output else OUT / f"role_frame_shift_{MODEL.split('/')[-1]}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDONE in {time.time()-t0:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
