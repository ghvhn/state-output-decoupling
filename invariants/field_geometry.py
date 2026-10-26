"""Bounded local distance geometry, compiled from learned field relationships.

These are target-conditioned effective distances, not a claim of a globally
Euclidean metric or biological plasticity. Checkpoint directions are unchanged.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class FieldEntry:
    """Named field contributions; old dictionary/tuple inputs remain supported."""
    direction: torch.Tensor
    mass: float = 0.
    g: float = 0.
    time: float = 1.
    shape: dict = field(default_factory=lambda: {"kind": "point"})
    name: str = ""
    qualities: list = field(default_factory=list)
    bands: list = field(default_factory=list)
    impulses: list = field(default_factory=list)
    warp: float = 1.
    geometry: object = None
    geometry_live: bool = False

    @classmethod
    def from_input(cls, item, index, G, taus=None, names=None):
        if isinstance(item, cls):
            return item
        if isinstance(item, dict):
            return cls(**{key: value for key, value in item.items()
                          if key in cls.__dataclass_fields__})
        return cls(item[0], float(item[1]), 1.,
                   float(taus[index]) if taus and index < len(taus) and taus[index] is not None else 1.,
                   name=names[index] if names and index < len(names) else str(index))


def local_distance(distance, support, *, strength=.4, radius=.35):
    """Contract/expand a local well while returning to the original far field.

    d' = d exp(-a exp(-d/r)), a = clamp(strength * support, -1, 1).
    At fixed support this preserves zero and is strictly increasing:
    dd'/dd = exp(-a exp(-d/r)) (1 + a (d/r) exp(-d/r)) > 0.
    The exponent fades rather than multiplying every distance by a constant.
    """
    radius = max(.02, min(2., float(radius)))
    d = distance.clamp_min(0)
    amplitude = (float(strength) * torch.as_tensor(
        support, dtype=d.dtype, device=d.device)).clamp(-1, 1)
    return d * torch.exp(-amplitude * torch.exp(-d / radius))


@dataclass(frozen=True)
class LocalGeometryStencil:
    """Small immutable relationship snapshot measured against every live token."""
    directions: torch.Tensor  # [relationships, residual width]
    bonds: torch.Tensor       # [relationships], signed outcome association
    strength: float
    radius: float

    def support(self, state_unit):
        directions = self.directions.to(device=state_unit.device, dtype=state_unit.dtype)
        matches = (state_unit @ directions.T).clamp(0, 1).square()
        bonds = self.bonds.to(device=state_unit.device, dtype=state_unit.dtype)
        # A lone weak bond stays weak; extra active neighbors do not amplify it.
        return ((matches * bonds).sum(-1, keepdim=True)
                / matches.sum(-1, keepdim=True).clamp_min(1.))

    def distance(self, raw_distance, state_unit):
        return local_distance(raw_distance, self.support(state_unit),
                              strength=self.strength, radius=self.radius)


def compile_local_geometry(matrix, target, layer, directions_by_name, *, max_neighbors=16, width=None):
    """Compile strongest valid, observed relationships; absent bodies stay inert."""
    if not matrix.enabled:
        return None
    candidates = []
    for other, layers in directions_by_name.items():
        if other == target:
            continue
        bond = matrix.potentiation(target, other)
        direction = layers.get(layer, layers.get(str(layer)))
        if not bond or direction is None:
            continue
        vector = torch.as_tensor(direction).detach().float().reshape(-1)
        norm = vector.norm()
        if (width is not None and vector.numel() != width) or not torch.isfinite(vector).all() or norm < 1e-6:
            continue
        candidates.append((other, float(bond), vector / norm))
    candidates.sort(key=lambda row: (-abs(row[1]), row[0]))
    chosen = candidates[:max_neighbors]
    if not chosen:
        return None
    width = chosen[0][2].numel()
    chosen = [row for row in chosen if row[2].numel() == width]
    return LocalGeometryStencil(torch.stack([row[2] for row in chosen]),
                                torch.tensor([row[1] for row in chosen], device=chosen[0][2].device),
                                matrix.strength, matrix.radius)
