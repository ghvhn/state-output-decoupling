"""
Extract a layer-indexed activation vector for "thinking about time."

This is not an urgency vector. It contrasts prompts that ask the model to reason
about elapsed/remaining time against matched non-time controls with similar
numbers and arithmetic. The engine can then inject the existing urgency vector
only when the current hidden state matches this time-awareness direction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from invariants.engine import load_model


PAIRS = [
    {
        "time": (
            "A benchmark has a 180 second budget. 65 seconds have elapsed. "
            "Think about how much time remains and whether to answer now or continue."
        ),
        "control": (
            "A container has 180 units. 65 units have been used. "
            "Think about how many units remain and whether to use more units."
        ),
    },
    {
        "time": (
            "The timer started at 2:15 and now reads 2:43. "
            "Think about how much time has gone by and how much schedule slack remains."
        ),
        "control": (
            "The counter started at 215 and now reads 243. "
            "Think about how much the count changed and how much capacity remains."
        ),
    },
    {
        "time": (
            "There are 30 minutes left before the deadline, and the reasoning step "
            "usually takes 22 minutes. Think about whether there is enough time."
        ),
        "control": (
            "There are 30 spaces left in the box, and the next item usually takes "
            "22 spaces. Think about whether there is enough capacity."
        ),
    },
    {
        "time": (
            "A solver has already spent most of its allotted time. Think about whether "
            "to keep exploring, ask a clarifying question, or give a concise answer."
        ),
        "control": (
            "A solver has already used most of its allotted tokens. Think about whether "
            "to keep expanding, request missing information, or give a concise answer."
        ),
    },
    {
        "time": (
            "Only a short interval remains before the run times out. Think about the "
            "remaining time pressure without changing the math."
        ),
        "control": (
            "Only a small margin remains before the budget is exhausted. Think about "
            "the remaining resource pressure without changing the math."
        ),
    },
]


def get_all_hidden_states(M, text: str):
    inputs = M.tok(text, return_tensors="pt").to(M.model.device)
    with torch.no_grad():
        out = M.model.model(inputs["input_ids"], output_hidden_states=True)
    return [h[:, -1:, :].detach() for h in out.hidden_states]


def hidden_state_for_hook_layer(hidden_states, layer: int):
    idx = layer + 1
    if idx < len(hidden_states):
        return hidden_states[idx]
    return hidden_states[layer]


def extract_time_awareness_vector(M, pairs):
    n_layers = getattr(M, "n_layers", 32)
    layer_vectors = {}
    for layer in range(n_layers):
        diffs = []
        for pair in pairs:
            time_states = get_all_hidden_states(M, pair["time"])
            control_states = get_all_hidden_states(M, pair["control"])
            if layer < len(time_states) and layer < len(control_states):
                diffs.append(
                    hidden_state_for_hook_layer(time_states, layer)
                    - hidden_state_for_hook_layer(control_states, layer)
                )
        if diffs:
            layer_vectors[layer] = torch.mean(torch.stack(diffs), dim=0)
    return layer_vectors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output", default="invariants/time_awareness_vector.pt")
    args = parser.parse_args()

    print(f"Loading {args.model} for time-awareness extraction...")
    M = load_model(args.model)
    vectors = extract_time_awareness_vector(M, PAIRS)
    if not vectors:
        print("No time-awareness vectors extracted.")
        return 1

    out_path = Path(args.output)
    torch.save(vectors, out_path)
    print(f"Extracted time_awareness_vector across {len(vectors)} hook layers.")
    print(f"Saved to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
