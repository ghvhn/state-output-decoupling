"""
Loads conversations from JSON for replay through the model.

Format:
[
  {
    "id": "conv_001",
    "turns": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ],
    "domain_hint": null   # null in discovery mode, string label in confirmation mode
  }
]

Each model's activations are evaluated against itself only.
The conversation text is the shared medium — not the geometry.
"""

import json
from pathlib import Path


def load(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save(path: str, conversations: list[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)


def filter_by_domain(conversations: list[dict], domain: str) -> list[dict]:
    return [c for c in conversations if c.get("domain_hint") == domain]


def all_domains(conversations: list[dict]) -> list[str]:
    return list({c["domain_hint"] for c in conversations if c.get("domain_hint")})
