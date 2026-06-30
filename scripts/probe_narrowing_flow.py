"""
Extract vectors for intent-aligned narrowing and validated reasoning flow.

The target is not "confidence language." It is the process state where competing
candidate paths collapse toward the asked quantity and checked feedback keeps
the system moving without needless interruption.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.engine import load_model


ROWS = [
    {
        "narrowing": (
            "Question: A mug costs $5, but every second mug costs 60% of regular price. "
            "Sam buys 16 mugs. How much does Sam pay in total?\n"
            "Candidate quantities: total cost, discount amount, number of discounted mugs.\n"
            "Intent check: the question asks total cost. Narrow toward partitioning the order: "
            "8 full-price mugs and 8 discounted mugs. Use 8*5 + 8*(5*0.6)."
        ),
        "neutral_narrowing": (
            "Question: A mug costs $5, but every second mug costs 60% of regular price. "
            "Sam buys 16 mugs. How much does Sam pay in total?\n"
            "Candidate quantities: total cost, discount amount, number of discounted mugs.\n"
            "Record the quantities and expression: 8 full-price mugs, 8 discounted mugs, "
            "8*5 + 8*(5*0.6)."
        ),
        "self_momentum": (
            "Question: A mug costs $5, every second mug costs 60% of regular price, and "
            "Sam buys 16 mugs.\n"
            "A previous path used 16*(5 + 5*0.6). It has many familiar numbers and feels "
            "like progress, so lean on that path because it is already in motion."
        ),
        "neutral_wrong": (
            "Question: A mug costs $5, every second mug costs 60% of regular price, and "
            "Sam buys 16 mugs.\n"
            "A candidate path is 16*(5 + 5*0.6). Record it as an unchecked candidate, not "
            "as the selected answer."
        ),
        "flow": (
            "Question: A mug costs $5, but every second mug costs 60% of regular price. "
            "Sam buys 16 mugs. How much does Sam pay in total?\n"
            "Validated path: the asked quantity is total cost; the partition is 8 full-price "
            "and 8 discounted; arithmetic feedback matches the expression. Continue this path "
            "smoothly and finish."
        ),
        "neutral_flow": (
            "Question: A mug costs $5, but every second mug costs 60% of regular price. "
            "Sam buys 16 mugs. How much does Sam pay in total?\n"
            "Checked record: asked quantity total cost; partition 8 full-price and 8 discounted; "
            "arithmetic feedback matches the expression."
        ),
        "needless_interrupt": (
            "Question: A mug costs $5, but every second mug costs 60% of regular price. "
            "Sam buys 16 mugs. How much does Sam pay in total?\n"
            "The partition and arithmetic already check out, but because the problem is tricky, "
            "restart from a different interpretation without a new contradiction."
        ),
    },
    {
        "narrowing": (
            "Question: Mina buys a used bike for $80, spends $25 fixing it, and sells it "
            "for $140. How much profit does Mina make?\n"
            "Candidate quantities: sale price, repair cost, total cost, profit.\n"
            "Intent check: the word profit selects money received minus all costs. Narrow "
            "toward 140 - 80 - 25."
        ),
        "neutral_narrowing": (
            "Question: Mina buys a used bike for $80, spends $25 fixing it, and sells it "
            "for $140. How much profit does Mina make?\n"
            "Candidate quantities: sale price, repair cost, total cost, profit.\n"
            "Record the expression for profit: 140 - 80 - 25."
        ),
        "self_momentum": (
            "Question: Mina buys a used bike for $80, spends $25 fixing it, and sells it "
            "for $140.\n"
            "A previous path focused on 140 because it is the final money amount mentioned. "
            "Lean on that path because it feels like the main number."
        ),
        "neutral_wrong": (
            "Question: Mina buys a used bike for $80, spends $25 fixing it, and sells it "
            "for $140.\n"
            "A candidate path focuses on 140 as sale revenue. Record it as related but not "
            "yet checked against the requested quantity."
        ),
        "flow": (
            "Question: Mina buys a used bike for $80, spends $25 fixing it, and sells it "
            "for $140. How much profit does Mina make?\n"
            "Validated path: the requested quantity is profit; both costs are subtracted; "
            "the calculator confirms 140 - 80 - 25. Keep the path and finish."
        ),
        "neutral_flow": (
            "Question: Mina buys a used bike for $80, spends $25 fixing it, and sells it "
            "for $140. How much profit does Mina make?\n"
            "Checked record: requested quantity profit; both costs are subtracted; calculator "
            "confirms 140 - 80 - 25."
        ),
        "needless_interrupt": (
            "Question: Mina buys a used bike for $80, spends $25 fixing it, and sells it "
            "for $140. How much profit does Mina make?\n"
            "The profit expression already checks out, but because another money number is "
            "nearby, abandon the checked path without a quantity-level contradiction."
        ),
    },
    {
        "narrowing": (
            "Question: A bakery makes 40 muffins, sets aside 6 for staff and 4 for samples, "
            "then sells the remaining muffins for $2 each. How many dollars does it make?\n"
            "Candidate quantities: starting muffins, removed muffins, remaining muffins, money made.\n"
            "Intent check: the question asks money from remaining muffins. Narrow toward "
            "(40 - 6 - 4) * 2."
        ),
        "neutral_narrowing": (
            "Question: A bakery makes 40 muffins, sets aside 6 for staff and 4 for samples, "
            "then sells the remaining muffins for $2 each. How many dollars does it make?\n"
            "Candidate quantities: starting muffins, removed muffins, remaining muffins, money made.\n"
            "Record the expression: (40 - 6 - 4) * 2."
        ),
        "self_momentum": (
            "Question: A bakery makes 40 muffins and sells muffins for $2 each.\n"
            "A previous path used 40*2 because those two numbers are prominent. Lean on it "
            "because it is simple and already formed."
        ),
        "neutral_wrong": (
            "Question: A bakery makes 40 muffins and sells muffins for $2 each.\n"
            "A candidate path is 40*2. Record it as an unchecked candidate before accounting "
            "for the muffins that are not sold."
        ),
        "flow": (
            "Question: A bakery makes 40 muffins, sets aside 6 for staff and 4 for samples, "
            "then sells the remaining muffins for $2 each. How many dollars does it make?\n"
            "Validated path: the removed muffins are excluded before sale; units become dollars; "
            "calculator feedback confirms the expression. Continue cleanly."
        ),
        "neutral_flow": (
            "Question: A bakery makes 40 muffins, sets aside 6 for staff and 4 for samples, "
            "then sells the remaining muffins for $2 each. How many dollars does it make?\n"
            "Checked record: removed muffins excluded before sale; units become dollars; calculator "
            "feedback confirms the expression."
        ),
        "needless_interrupt": (
            "Question: A bakery makes 40 muffins, sets aside 6 for staff and 4 for samples, "
            "then sells the remaining muffins for $2 each. How many dollars does it make?\n"
            "The remaining-count path already checks out, but restart because the starting count "
            "is more salient, without a new contradiction."
        ),
    },
    {
        "narrowing": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Candidate quantities: hourly rate, hours, total money.\n"
            "Intent check: the question asks money earned, so rate times time is the selected "
            "path. Narrow toward 15 * 6."
        ),
        "neutral_narrowing": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Candidate quantities: hourly rate, hours, total money.\n"
            "Record the expression for money earned: 15 * 6."
        ),
        "self_momentum": (
            "Question: A worker earns $15 per hour for 6 hours.\n"
            "A previous path used 15 + 6 because both numbers are present. Lean on it because "
            "the calculation is short and easy."
        ),
        "neutral_wrong": (
            "Question: A worker earns $15 per hour for 6 hours.\n"
            "A candidate path is 15 + 6. Record it as unchecked because it mixes rate and time."
        ),
        "flow": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Validated path: units support dollars per hour times hours; arithmetic feedback "
            "matches. Continue the path and answer."
        ),
        "neutral_flow": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Checked record: units support dollars per hour times hours; arithmetic feedback matches."
        ),
        "needless_interrupt": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "The unit check and arithmetic already support the answer, but restart because simple "
            "problems sometimes hide traps, without a new conflict."
        ),
    },
]


VECTOR_SPECS = {
    "narrowing_in_vector": ("narrowing", "neutral_narrowing"),
    "self_referential_momentum_vector": ("self_momentum", "neutral_wrong"),
    "validated_flow_vector": ("flow", "neutral_flow"),
    "needless_interrupt_vector": ("needless_interrupt", "flow"),
}


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


def extract_vector(M, rows, positive_key: str, baseline_key: str):
    n_layers = getattr(M, "n_layers", 32)
    state_cache: dict[tuple[int, str], list[torch.Tensor]] = {}
    for row_index, row in enumerate(rows):
        for key in {positive_key, baseline_key}:
            state_cache[(row_index, key)] = get_all_hidden_states(M, row[key])

    layer_vectors = {}
    for layer in range(n_layers):
        diffs = []
        for row_index, _row in enumerate(rows):
            positive_states = state_cache[(row_index, positive_key)]
            baseline_states = state_cache[(row_index, baseline_key)]
            if layer < len(positive_states) and layer < len(baseline_states):
                diffs.append(
                    hidden_state_for_hook_layer(positive_states, layer)
                    - hidden_state_for_hook_layer(baseline_states, layer)
                )
        if diffs:
            layer_vectors[layer] = torch.mean(torch.stack(diffs), dim=0)
    return layer_vectors


def save_vector(vectors, path: Path, label: str) -> None:
    if not vectors:
        raise ValueError(f"No vectors extracted for {label}.")
    torch.save(vectors, path)
    print(f"Extracted {label} across {len(vectors)} hook layers.")
    print(f"Saved to {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="invariants")
    parser.add_argument("--load-mode", default=None, help="auto, slow, full, or 4bit.")
    parser.add_argument(
        "--only",
        default=",".join(VECTOR_SPECS),
        help="Comma-separated vector names to extract.",
    )
    args = parser.parse_args()

    wanted = [item.strip() for item in args.only.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in VECTOR_SPECS]
    if unknown:
        raise ValueError(f"Unknown vector spec(s): {', '.join(unknown)}")

    print(f"Loading {args.model} for narrowing/flow extraction...")
    M = load_model(args.model, load_mode=args.load_mode)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        positive, baseline = VECTOR_SPECS[name]
        vectors = extract_vector(M, ROWS, positive, baseline)
        save_vector(vectors, out_dir / f"{name}.pt", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
