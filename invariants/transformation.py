"""
The one primitive everything plugs into.

A Transformation is two matched sets of inputs (a, b) that differ ONLY by the
transformation, plus metadata. A constraint (self vs other), a bridge (lang A vs
lang B), a dialectic move (proposition vs moved) are all Transformations. The
engine runs the SAME discovery pipeline on any of them — add a phenomenon by
adding a Transformation, never by writing a new detector.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Transformation:
    name: str
    group: str                       # the NAMED transformation group (design-consequence A)
    a: list[str]                     # condition A inputs (chat instructions)
    b: list[str]                     # condition B inputs (matched, differ only by T)
    a_label: str = "A"
    b_label: str = "B"
    read: str = "generation"         # "generation" (model's own output) | "prompt"
    expected: str = "unknown"        # "break" | "preserve" | "unknown" — reporting only
    measure: Optional[Callable] = None   # optional extra scorer(model, A, B, dirs, result)->dict
    meta: dict = field(default_factory=dict)
