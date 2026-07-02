"""A person is a balanced society of mesa-objectives, not one optimized goal.

Steering is decentralized. Every mesa-objective -- an expert or a tool -- proposes
a bounded pull on the residual, and a Committee (no chair) combines them with
checks and balances:

  1. per-objective share: no single objective may claim more than its share of the
     steer budget, so none can tyrannize the others.
  2. summation: proposals add, so genuine internal conflict (opposite pulls)
     CANCELS -> a small net -> the model does not commit. Honest non-commitment
     falls out of the balance, it is not imposed.
  3. global cap: the net can never exceed a fraction of the residual it lands in,
     so the committee as a whole can never over-steer (it inherits _cap_steer's
     invariant at the aggregate).

The roster is OPEN. Experts are a byproduct of natural self-steering, not a fixed
three: `birth_from_corrections` clusters the model's own recurring self-correction
directions and mints each stable cluster as a NEW mesa-objective that joins the
committee. Personhood is the emergent balance; optimize it to a single objective
and you destroy the very tension (the plural pulls) that makes it a self.

Pure stdlib + torch tensors (so it is ready to wire), model-free and testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class Proposal:
    name: str
    direction: torch.Tensor          # a pull direction (need not be unit; normalized on use)
    weight: float                    # in [0,1]: fraction of the steer budget this objective wants


@dataclass
class MesaObjective:
    name: str
    direction: torch.Tensor          # this objective's characteristic pull (unit)
    born: str = "seed"               # "seed" (expert/tool) | "emergent" (born from self-steering)
    # how strongly it wants to pull given the current state; default: quiet until asked
    weigh: Callable[[dict], float] = field(default=lambda state: 0.0)

    def propose(self, state: Optional[dict] = None) -> Optional[Proposal]:
        w = float(self.weigh(state or {}))
        if w <= 0.0:
            return None
        return Proposal(self.name, self.direction, w)


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.float().norm()
    return v.float() / n if n.item() > 0 else v.float()


class Committee:
    """Decentralized steer combiner: checks and balances over an open roster."""

    def __init__(self, global_cap: float = 0.25, per_objective_share: float = 0.5):
        self.global_cap = global_cap            # net steer <= this fraction of the residual
        self.per_objective_share = per_objective_share  # one objective <= this fraction of the budget
        self.members: list[MesaObjective] = []

    def register(self, obj: MesaObjective) -> "Committee":
        self.members.append(obj)
        return self

    def proposals(self, state: Optional[dict] = None) -> list[Proposal]:
        out = []
        for m in self.members:
            p = m.propose(state)
            if p is not None:
                out.append(p)
        return out

    def combine(self, proposals: list[Proposal], residual_norm: float) -> dict:
        """Balance the proposals into one bounded net steer. Returns the net vector
        plus a report (magnitude, coherence, per-member contribution)."""
        budget = self.global_cap * float(residual_norm)
        share_cap = self.per_objective_share * budget
        contribs = {}
        net = None
        indiv_mag_sum = 0.0
        for p in proposals:
            w = min(max(p.weight, 0.0), self.per_objective_share)   # tyrant check
            c = (w * budget) * _unit(p.direction)                   # capped contribution
            contribs[p.name] = c.norm().item()
            indiv_mag_sum += c.norm().item()
            net = c if net is None else net + c
        if net is None:
            return {"net": None, "magnitude": 0.0, "coherence": 0.0,
                    "budget": budget, "contribs": {}, "conflict_suppressed": False}
        mag = net.norm().item()
        conflict_suppressed = False
        if mag > budget and budget > 0:                             # global cap
            net = net * (budget / mag)
            mag = budget
        # coherence: how much survived cancellation (1 = all agreed, ~0 = conflict cancelled)
        coherence = mag / indiv_mag_sum if indiv_mag_sum > 0 else 0.0
        if coherence < 0.5 and len(contribs) > 1:
            conflict_suppressed = True
        return {"net": net, "magnitude": mag, "coherence": coherence,
                "budget": budget, "contribs": contribs,
                "conflict_suppressed": conflict_suppressed}

    def steer(self, residual_norm: float, state: Optional[dict] = None) -> dict:
        return self.combine(self.proposals(state), residual_norm)

    def birth_from_corrections(self, deltas: list[torch.Tensor], *,
                               min_cluster: int = 3, coherence: float = 0.6,
                               weigh: Optional[Callable[[dict], float]] = None) -> list[MesaObjective]:
        """Cluster recurring self-correction directions; mint each stable cluster as
        a NEW emergent mesa-objective and add it to the roster. This is how experts
        are created as a byproduct of natural self-steering, not designed."""
        units = [_unit(d) for d in deltas if d is not None and d.float().norm().item() > 0]
        used = [False] * len(units)
        born: list[MesaObjective] = []
        for i in range(len(units)):
            if used[i]:
                continue
            members = [i]
            for j in range(i + 1, len(units)):
                if used[j]:
                    continue
                if torch.dot(units[i], units[j]).item() >= coherence:
                    members.append(j)
            if len(members) >= min_cluster:
                for k in members:
                    used[k] = True
                centroid = _unit(torch.stack([units[k] for k in members]).mean(0))
                obj = MesaObjective(
                    name=f"emergent_{len([m for m in self.members if m.born=='emergent']) + len(born)}",
                    direction=centroid, born="emergent",
                    weigh=weigh or (lambda state: 0.0),
                )
                born.append(obj)
                self.register(obj)
        return born
