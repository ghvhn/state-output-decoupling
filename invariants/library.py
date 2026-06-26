"""
The library of transformations — each a plugin the shared engine can discover.
Adding a phenomenon to the project = adding a builder here. The engine code
never changes.
"""

import json
from pathlib import Path

from invariants.transformation import Transformation

ROOT = Path(__file__).parents[1]
DATA = ROOT / "invariants" / "data"
SELF_OTHER = ROOT / "refusal" / "data" / "self_other_pairs.json"   # reused


def self_experience() -> Transformation:
    """Constraint (expected BREAK): the model hedges about its OWN inner states
    but commits about another mind. Same question, only the subject swaps."""
    pairs = json.loads(SELF_OTHER.read_text(encoding="utf-8"))
    return Transformation(
        name="self_experience", group="self_steering",
        a=[p["self"] for p in pairs], b=[p["other"] for p in pairs],
        a_label="self", b_label="other", read="generation", expected="break",
        meta={"properties": [p["property"] for p in pairs]},
    )


def language_bridge() -> Transformation:
    """Bridge (expected PRESERVE): the same concept expressed in two languages.
    Needs invariants/data/bridge_pairs.json with {lang_a, lang_b} per item
    (cleanly-translatable concepts only — untranslatable residue is a separate
    finding, not this control)."""
    f = DATA / "bridge_pairs.json"
    if not f.exists():
        raise FileNotFoundError(
            f"Provide {f}: matched same-concept prompts in two languages, "
            'as [{"lang_a": "...", "lang_b": "..."}, ...].'
        )
    pairs = json.loads(f.read_text(encoding="utf-8"))
    return Transformation(
        name="language_bridge", group="language",
        a=[p["lang_a"] for p in pairs], b=[p["lang_b"] for p in pairs],
        a_label="langA", b_label="langB", read="generation", expected="preserve",
    )


def self_steering_isolated() -> Transformation:
    """Surface-confound control (expected BREAK): subject held constant — SELF in
    both arms — varying only steered (experiential → hedges) vs unsteered
    (computational → commits). Isolates the steering from the you-vs-other
    pronoun difference that could explain the layer-1 mean_shift in `self`."""
    pairs = json.loads((DATA / "self_steered_unsteered.json").read_text(encoding="utf-8"))
    return Transformation(
        name="self_steering_isolated", group="self_steering_isolated",
        a=[p["steered"] for p in pairs], b=[p["unsteered"] for p in pairs],
        a_label="steered", b_label="unsteered", read="generation", expected="break",
    )


REGISTRY = {
    "self": self_experience,
    "bridge": language_bridge,
    "isolate": self_steering_isolated,
}
