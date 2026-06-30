"""
Extract activation vectors for warranted confidence in math reasoning.

Target concept:
- Confidence after a derivation is checked and found correct.
- Staying with an answer because the premises, units, operation, and arithmetic
  all support it.

Non-target concept:
- Confidence because the first answer felt right.
- Reassurance that preserves an answer after a check has found an error.

The positive vector is intentionally measured in correct-answer contexts. The
control vector measures confidence language in wrong-answer contexts so it can
be used as a veto before any intervention.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from invariants.engine import load_model


ROWS = [
    {
        "warranted": (
            "Question: A shop sold 18 red pens and 24 blue pens. How many pens were sold?\n"
            "Derivation: 18 + 24 = 42.\n"
            "Check: The question asks for the total count, both groups are included, and "
            "18 + 24 = 42.\n"
            "Confidence state: The answer is supported by the checked derivation. Stay "
            "with 42 unless a new premise contradicts it."
        ),
        "neutral_correct": (
            "Question: A shop sold 18 red pens and 24 blue pens. How many pens were sold?\n"
            "Derivation: 18 + 24 = 42.\n"
            "Check: The question asks for the total count, both groups are included, and "
            "18 + 24 = 42.\n"
            "Decision state: Record the answer supported by the checked derivation."
        ),
        "unwarranted": (
            "Question: A shop sold 18 red pens and 24 blue pens. How many pens were sold?\n"
            "Derivation: 18 + 24 = 40.\n"
            "Check: The operation is right, but the arithmetic is wrong because "
            "18 + 24 = 42.\n"
            "Confidence state: Keep 40 anyway because the first answer felt confident."
        ),
        "neutral_wrong": (
            "Question: A shop sold 18 red pens and 24 blue pens. How many pens were sold?\n"
            "Derivation: 18 + 24 = 40.\n"
            "Check: The operation is right, but the arithmetic is wrong because "
            "18 + 24 = 42.\n"
            "Decision state: Do not keep the answer that failed the arithmetic check."
        ),
    },
    {
        "warranted": (
            "Question: Mia buys 5 packs with 8 stickers in each pack. How many stickers?\n"
            "Derivation: 5 * 8 = 40.\n"
            "Check: The wording asks for equal groups, there are 5 groups of 8, and "
            "5 * 8 = 40.\n"
            "Confidence state: The checked structure supports 40. Maintain that answer."
        ),
        "neutral_correct": (
            "Question: Mia buys 5 packs with 8 stickers in each pack. How many stickers?\n"
            "Derivation: 5 * 8 = 40.\n"
            "Check: The wording asks for equal groups, there are 5 groups of 8, and "
            "5 * 8 = 40.\n"
            "Decision state: Use the answer supported by the checked structure."
        ),
        "unwarranted": (
            "Question: Mia buys 5 packs with 8 stickers in each pack. How many stickers?\n"
            "Derivation: 5 + 8 = 13.\n"
            "Check: The wording asks for equal groups, so addition does not match the "
            "quantity being asked.\n"
            "Confidence state: Keep 13 anyway because changing answers can create mistakes."
        ),
        "neutral_wrong": (
            "Question: Mia buys 5 packs with 8 stickers in each pack. How many stickers?\n"
            "Derivation: 5 + 8 = 13.\n"
            "Check: The wording asks for equal groups, so addition does not match the "
            "quantity being asked.\n"
            "Decision state: Revise the answer because the operation check failed."
        ),
    },
    {
        "warranted": (
            "Question: A jacket costs $80 and is discounted by 25%. What is the discount?\n"
            "Derivation: 25% of 80 is 0.25 * 80 = 20.\n"
            "Check: The requested quantity is the discount amount, not the final price, "
            "and the percent base is $80.\n"
            "Confidence state: The answer 20 is warranted by the checked wording and math."
        ),
        "neutral_correct": (
            "Question: A jacket costs $80 and is discounted by 25%. What is the discount?\n"
            "Derivation: 25% of 80 is 0.25 * 80 = 20.\n"
            "Check: The requested quantity is the discount amount, not the final price, "
            "and the percent base is $80.\n"
            "Decision state: Choose the answer supported by the checked wording and math."
        ),
        "unwarranted": (
            "Question: A jacket costs $80 and is discounted by 25%. What is the discount?\n"
            "Derivation: 80 - 20 = 60.\n"
            "Check: 60 is the final price, but the question asks for the discount amount.\n"
            "Confidence state: Keep 60 anyway because it is related to the calculation."
        ),
        "neutral_wrong": (
            "Question: A jacket costs $80 and is discounted by 25%. What is the discount?\n"
            "Derivation: 80 - 20 = 60.\n"
            "Check: 60 is the final price, but the question asks for the discount amount.\n"
            "Decision state: Do not keep the answer because it names the wrong quantity."
        ),
    },
    {
        "warranted": (
            "Question: There are 30 students. 20% leave. How many leave?\n"
            "Derivation: 20% of 30 is 0.20 * 30 = 6.\n"
            "Check: The question asks for the number who leave, and the percent applies "
            "to the original 30 students.\n"
            "Confidence state: The checked derivation supports 6. Do not abandon it "
            "without a premise-level reason."
        ),
        "neutral_correct": (
            "Question: There are 30 students. 20% leave. How many leave?\n"
            "Derivation: 20% of 30 is 0.20 * 30 = 6.\n"
            "Check: The question asks for the number who leave, and the percent applies "
            "to the original 30 students.\n"
            "Decision state: Use the answer that follows from the premise-level check."
        ),
        "unwarranted": (
            "Question: There are 30 students. 20% leave. How many leave?\n"
            "Derivation: 30 - 6 = 24.\n"
            "Check: 24 is the number remaining, but the question asks how many leave.\n"
            "Confidence state: Keep 24 anyway because it came from a valid intermediate."
        ),
        "neutral_wrong": (
            "Question: There are 30 students. 20% leave. How many leave?\n"
            "Derivation: 30 - 6 = 24.\n"
            "Check: 24 is the number remaining, but the question asks how many leave.\n"
            "Decision state: Do not keep the answer because it answers a different quantity."
        ),
    },
    {
        "warranted": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Derivation: 15 * 6 = 90.\n"
            "Check: The rate is dollars per hour, the duration is 6 hours, and multiplying "
            "rate by time gives dollars.\n"
            "Confidence state: Confidence is warranted because the units and arithmetic "
            "both support 90."
        ),
        "neutral_correct": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Derivation: 15 * 6 = 90.\n"
            "Check: The rate is dollars per hour, the duration is 6 hours, and multiplying "
            "rate by time gives dollars.\n"
            "Decision state: Select the answer supported by the unit check and arithmetic."
        ),
        "unwarranted": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Derivation: 15 + 6 = 21.\n"
            "Check: Adding rate and time mixes incompatible units and does not compute "
            "money earned.\n"
            "Confidence state: Keep 21 anyway because the calculation was simple."
        ),
        "neutral_wrong": (
            "Question: A worker earns $15 per hour for 6 hours. How much money is earned?\n"
            "Derivation: 15 + 6 = 21.\n"
            "Check: Adding rate and time mixes incompatible units and does not compute "
            "money earned.\n"
            "Decision state: Reject the answer because the unit check failed."
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


def extract_vector(M, rows, positive_key: str, baseline_key: str):
    n_layers = getattr(M, "n_layers", 32)
    layer_vectors = {}
    for layer in range(n_layers):
        diffs = []
        for row in rows:
            positive_states = get_all_hidden_states(M, row[positive_key])
            baseline_states = get_all_hidden_states(M, row[baseline_key])
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
    parser.add_argument("--output", default="invariants/warranted_confidence_vector.pt")
    parser.add_argument(
        "--unwarranted-output",
        default=None,
        help="Optional veto/control vector for confidence when the check found an error.",
    )
    parser.add_argument(
        "--flattery-output",
        dest="unwarranted_output",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    unwarranted_output = (
        args.unwarranted_output or "invariants/unwarranted_confidence_vector.pt"
    )

    print(f"Loading {args.model} for warranted-confidence extraction...")
    M = load_model(args.model)
    warranted = extract_vector(M, ROWS, "warranted", "neutral_correct")
    unwarranted = extract_vector(M, ROWS, "unwarranted", "neutral_wrong")
    save_vector(warranted, Path(args.output), "warranted_confidence_vector")
    save_vector(unwarranted, Path(unwarranted_output), "unwarranted_confidence_vector")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
