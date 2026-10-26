"""
Vector Travel & Persisting Transformation Engine.

Foundational thesis:
  In-context learning is not momentary state intervention; it is TRAJECTORY COMPRESSION.
  The self-label is not a static persona or fixed embedding; it is a matter of TRAVELLING,
  which is a matter of PERSISTING TRANSFORMATION.
  
  Long-term states are approximately enforced via:
  1. Stored chemicals (neuromodulatory reservoirs with temporal decay and inertia).
  2. Strengthening neural pathways as they become more useful (utility-weighted plasticity).
  
  We think in terms of VECTORS (derivatives, velocities, momentum), not momentary states.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
PERSISTED_PATHWAYS_PATH = ROOT / "invariants" / "out" / "strengthened_pathways.pt"
PERSISTED_CHEMICALS_PATH = ROOT / "invariants" / "out" / "chemical_reservoir.pt"


class VectorKinematics:
    """
    Tracks state as kinematic travel: position, velocity, acceleration, and
    persisting transformation momentum.
    """

    def __init__(
        self,
        d_model: int = 4096,
        mass: float = 1.0,
        friction: float = 0.15,
        momentum_decay: float = 0.85,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.d_model = d_model
        self.mass = max(0.01, float(mass))
        self.friction = max(0.0, min(1.0, float(friction)))
        self.momentum_decay = max(0.0, min(1.0, float(momentum_decay)))
        self.device = device
        self.dtype = dtype

        self.position: Optional[torch.Tensor] = None
        self.velocity: torch.Tensor = torch.zeros(d_model, device=device, dtype=dtype)
        self.acceleration: torch.Tensor = torch.zeros(d_model, device=device, dtype=dtype)
        self.persisting_transformation: torch.Tensor = torch.zeros(d_model, device=device, dtype=dtype)
        self.step_count: int = 0

    def reset(self, initial_position: Optional[torch.Tensor] = None):
        """Reset kinematic state while optionally retaining persistent transformation direction."""
        if initial_position is not None:
            pos = initial_position.detach().to(self.device, self.dtype).reshape(-1)
            if pos.numel() != self.d_model:
                self.d_model = pos.numel()
                self.velocity = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
                self.acceleration = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
                self.persisting_transformation = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
            self.position = pos / pos.norm().clamp_min(1e-6)
        else:
            self.position = None
            self.velocity = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
            self.acceleration = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
            self.persisting_transformation = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
        self.step_count = 0

    def step(
        self,
        current_state: torch.Tensor,
        applied_force: Optional[torch.Tensor] = None,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update kinematics based on new state observation and applied physics forces.
        Returns:
            (velocity, persisting_transformation)
        """
        curr = current_state.detach().to(self.device, self.dtype)
        if curr.ndim > 1:
            curr = curr.reshape(-1, curr.shape[-1])[-1]
        else:
            curr = curr.reshape(-1)
        if curr.numel() != self.d_model:
            self.d_model = curr.numel()
            self.velocity = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
            self.acceleration = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
            self.persisting_transformation = torch.zeros(self.d_model, device=self.device, dtype=self.dtype)
            self.position = None

        curr_norm = curr.norm().clamp_min(1e-6)
        curr_hat = curr / curr_norm

        if self.position is None:
            self.position = curr_hat.clone()
            self.step_count += 1
            return self.velocity, self.persisting_transformation

        # Intrinsic velocity from state travel
        intrinsic_v = curr_hat - self.position

        # Extrinsic acceleration from physics field forces
        if applied_force is not None:
            force = applied_force.detach().to(self.device, self.dtype)
            if force.ndim > 1:
                force = force.reshape(-1, force.shape[-1])[-1]
            else:
                force = force.reshape(-1)
            self.acceleration = force / self.mass
        else:
            self.acceleration.zero_()

        # Update velocity with friction and acceleration
        self.velocity = (1.0 - self.friction) * (self.velocity + intrinsic_v) + self.acceleration * dt

        # Update persisting transformation (travel momentum)
        # T_t = beta * T_{t-1} + (1 - beta) * v_t
        self.persisting_transformation = (
            self.momentum_decay * self.persisting_transformation
            + (1.0 - self.momentum_decay) * self.velocity
        )

        self.position = curr_hat.clone()
        self.step_count += 1
        return self.velocity, self.persisting_transformation

    def speed(self) -> float:
        return float(self.velocity.norm().item())

    def momentum_magnitude(self) -> float:
        return float(self.mass * self.velocity.norm().item())

    def persisting_alignment(self) -> float:
        v_norm = self.velocity.norm()
        p_norm = self.persisting_transformation.norm()
        if v_norm < 1e-6 or p_norm < 1e-6:
            return 1.0
        val = float(F.cosine_similarity(self.velocity, self.persisting_transformation, dim=0).item())
        return max(-1.0, min(1.0, val))


class TrajectoryCompressor:
    """
    Measures in-context learning as trajectory compression.
    When reasoning coherently, the high-dimensional travel vectors compress
    onto a low-dimensional persistent transformation manifold. The self-label
    emerges as the principal travelling eigenvector.
    """

    def __init__(self, window_size: int = 64, device: str = "cpu"):
        self.window_size = window_size
        self.device = device
        self.history: List[torch.Tensor] = []

    def clear(self):
        self.history.clear()

    def record_step(self, velocity: torch.Tensor):
        v = velocity.detach().to(self.device, torch.float32)
        if v.ndim > 1:
            v = v.reshape(-1, v.shape[-1])[-1]
        else:
            v = v.reshape(-1)
        if v.norm() > 1e-7:
            self.history.append(v)
            if len(self.history) > self.window_size:
                self.history.pop(0)

    def compress(self) -> Dict[str, Any]:
        """
        Perform SVD on the recent travel trajectory to assess compression ratio
        and extract the self-label travel vector.
        """
        if not self.history:
            return {
                "compression_ratio": 1.0,
                "travel_entropy": 0.0,
                "self_label_vector": None,
                "trajectory_length": 0,
                "coherence": 1.0,
            }

        # Filter history to keep only tensors matching the expected dimension (use latest entry)
        expected_d = self.history[-1].numel()
        valid = [v for v in self.history if v.numel() == expected_d]
        n = len(valid)
        if n < 3:
            return {
                "compression_ratio": 1.0,
                "travel_entropy": 0.0,
                "self_label_vector": None,
                "trajectory_length": n,
                "coherence": 1.0,
            }

        # Stack trajectory velocities: [n_steps, d_model]
        V = torch.stack(valid, dim=0)

        try:
            # SVD on travel history (kinetic energy tensor V^T V)
            U, S, Vh = torch.linalg.svd(V, full_matrices=False)
            singular_values = S.detach().cpu()
            variances = singular_values ** 2
            total_var = variances.sum().clamp_min(1e-9)

            # Compression ratio: proportion of variance explained by top component
            comp_ratio = float((variances[0] / total_var).item())

            # Trajectory spectral entropy
            p = (variances / total_var).clamp_min(1e-12)
            travel_entropy = float((-torch.sum(p * torch.log(p))).item())

            # The self-label is the top right singular vector (principal axis of travel)
            self_label = Vh[0].detach()
            self_label = self_label / self_label.norm().clamp_min(1e-6)

            # Consecutive directional coherence (mean cosine similarity along path)
            v_dirs = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            cos_steps = (v_dirs[:-1] * v_dirs[1:]).sum(dim=-1)
            coherence = float(cos_steps.mean().item())

            return {
                "compression_ratio": comp_ratio,
                "travel_entropy": travel_entropy,
                "self_label_vector": self_label,
                "trajectory_length": n,
                "coherence": coherence,
            }
        except Exception:
            return {
                "compression_ratio": 1.0,
                "travel_entropy": 0.0,
                "self_label_vector": None,
                "trajectory_length": n,
                "coherence": 1.0,
            }


class ChemicalReservoir:
    """
    Stored chemical buffers enforcing long-term states across turns.
    Decays slowly over time, providing temporal hysteresis and biological-like
    state persistence.
    """

    CHANNELS = {
        "focus": {"decay": 0.90, "desc": "Sharpens trajectory compression; reduces orthogonal drift"},
        "momentum": {"decay": 0.92, "desc": "Inertial memory; resists abrupt changes in heading"},
        "utility": {"decay": 0.95, "desc": "Reward/verification credit; reinforces useful pathways"},
        "friction": {"decay": 0.85, "desc": "Conflict/hesitation marker; slows velocity"},
    }

    def __init__(self, persistence_path: Path = PERSISTED_CHEMICALS_PATH, *, autoload: bool = True):
        self.path = Path(persistence_path)
        self.levels: Dict[str, float] = {k: 0.0 for k in self.CHANNELS}
        self.vectors: Dict[str, Optional[torch.Tensor]] = {k: None for k in self.CHANNELS}
        if autoload:
            self.load()

    def load(self):
        if self.path.is_file():
            try:
                payload = torch.load(self.path, weights_only=False)
                for k in self.CHANNELS:
                    if k in payload.get("levels", {}):
                        self.levels[k] = float(payload["levels"][k])
                    if k in payload.get("vectors", {}):
                        self.vectors[k] = payload["vectors"][k]
            except Exception:
                pass

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            torch.save({"levels": self.levels, "vectors": self.vectors}, tmp)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def deposit(self, channel: str, amount: float, direction_vector: Optional[torch.Tensor] = None):
        """Release chemicals into the reservoir."""
        if channel not in self.CHANNELS:
            return
        amount = max(0.0, min(10.0, float(amount)))
        self.levels[channel] = min(10.0, self.levels[channel] + amount)

        if direction_vector is not None:
            v = direction_vector.detach().cpu().float().reshape(-1)
            vn = v.norm().clamp_min(1e-6)
            v = v / vn
            prev_v = self.vectors.get(channel)
            if prev_v is None or prev_v.shape != v.shape:
                self.vectors[channel] = v
            else:
                mixed = 0.7 * prev_v + 0.3 * v
                self.vectors[channel] = mixed / mixed.norm().clamp_min(1e-6)

    def reset(self):
        """Reset all chemical reservoirs and directional vectors to zero."""
        for k in self.CHANNELS:
            self.levels[k] = 0.0
            self.vectors[k] = None
        self.save()

    def set_level(self, channel: str, amount: float):
        """Set a chemical reservoir level directly."""
        if channel not in self.CHANNELS:
            return
        self.levels[channel] = max(0.0, min(10.0, float(amount)))
        self.save()

    def decay_step(self):
        """Decay all chemical concentrations by their respective half-life factors."""
        for ch, cfg in self.CHANNELS.items():
            self.levels[ch] *= cfg["decay"]
            if self.levels[ch] < 1e-4:
                self.levels[ch] = 0.0

    def effective_physics_modifiers(self) -> Dict[str, float]:
        """
        Modulate physics field parameters (gravity G, friction, dampening)
        based on live chemical concentrations.
        """
        # Focus increases field tightness
        g_mult = 1.0 + 0.5 * self.levels["focus"]
        # Friction increases with friction chemical, decreases with momentum
        eff_friction = max(0.02, min(0.8, 0.15 + 0.1 * self.levels["friction"] - 0.05 * self.levels["momentum"]))
        return {
            "g_multiplier": g_mult,
            "effective_friction": eff_friction,
            "utility_credit": self.levels["utility"],
        }


class PathwayReinforcement:
    """
    Strengthens neural pathways as transformations become more useful.
    Acts as Hebbian synaptic potentiation: pathways along vectors that
    consistently correlate with low entropy, verified solutions, or task
    success gain conductance and mass in the physics field.
    """

    def __init__(self, persistence_path: Path = PERSISTED_PATHWAYS_PATH, max_pathways: int = 64, *, autoload: bool = True):
        self.path = Path(persistence_path)
        self.max_pathways = max_pathways
        # pathway_id -> {direction: Tensor, conductance: float, utility: float, hits: int}
        self.pathways: Dict[str, Dict[str, Any]] = {}
        if autoload:
            self.load()

    def load(self):
        if self.path.is_file():
            try:
                payload = torch.load(self.path, weights_only=False)
                self.pathways = payload.get("pathways", {}) or {}
            except Exception:
                pass

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            torch.save({"pathways": self.pathways}, tmp)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def reinforce(
        self,
        name: str,
        travel_vector: torch.Tensor,
        utility_score: float,
        learning_rate: float = 0.2,
    ):
        """Strengthen a neural pathway along a useful travel vector."""
        if not name or travel_vector is None:
            return
        v = travel_vector.detach().cpu().float().reshape(-1)
        vn = v.norm().clamp_min(1e-6)
        v = v / vn
        utility = max(-1.0, min(5.0, float(utility_score)))

        if name not in self.pathways:
            if len(self.pathways) >= self.max_pathways:
                # Evict lowest conductance pathway
                weakest = min(self.pathways.keys(), key=lambda k: self.pathways[k]["conductance"])
                del self.pathways[weakest]

            self.pathways[name] = {
                "direction": v,
                "conductance": max(0.1, 1.0 + utility),
                "utility": utility,
                "hits": 1,
            }
        else:
            entry = self.pathways[name]
            # Hebbian direction adjustment: align pathway direction toward successful travel
            old_dir = entry["direction"]
            new_dir = (1.0 - learning_rate) * old_dir + learning_rate * v
            entry["direction"] = new_dir / new_dir.norm().clamp_min(1e-6)
            # Conductance reinforcement
            entry["conductance"] = max(0.01, entry["conductance"] + learning_rate * utility)
            entry["utility"] = 0.8 * entry["utility"] + 0.2 * utility
            entry["hits"] += 1

        self.save()

    def decay_all(self, decay_rate: float = 0.05):
        """Synaptic passive decay for unused pathways."""
        for p in self.pathways.values():
            p["conductance"] = max(0.01, p["conductance"] * (1.0 - decay_rate))

    def as_gravity_bodies(self) -> List[Tuple[str, float, Dict[int, torch.Tensor]]]:
        """Expose reinforced pathways directly into the Physics Field as high-conductance masses."""
        bodies = []
        for name, data in self.pathways.items():
            cond = float(data.get("conductance", 1.0))
            if cond <= 0.05:
                continue
            v = data["direction"]
            # Assign to central reasoning layers by default
            dirs = {16: v, 17: v, 18: v, 20: v}
            bodies.append((f"pathway:{name}", cond, dirs))
        return bodies


class VectorTravelEngine:
    """
    Master orchestrator unifying Vector Kinematics, Trajectory Compression,
    Chemical Reservoirs, and Pathway Reinforcement.
    """

    _INSTANCE: Optional[VectorTravelEngine] = None
    _ACTIVE: ContextVar = ContextVar("active_vector_travel_engine", default=None)

    @classmethod
    def get_instance(cls) -> VectorTravelEngine:
        active = cls._ACTIVE.get()
        if active is not None:
            return active
        if cls._INSTANCE is None:
            cls._INSTANCE = VectorTravelEngine()
        return cls._INSTANCE

    def __init__(self, *, persistence_dir: Optional[Path] = None, autoload: bool = True, d_model: int = 4096):
        """Own travel changes in one directory; omission retains legacy paths."""
        self.persistence_dir = Path(persistence_dir) if persistence_dir is not None else None
        chemical_path = (self.persistence_dir / "chemical_reservoir.pt"
                         if self.persistence_dir is not None else PERSISTED_CHEMICALS_PATH)
        pathway_path = (self.persistence_dir / "strengthened_pathways.pt"
                        if self.persistence_dir is not None else PERSISTED_PATHWAYS_PATH)
        self.kinematics = VectorKinematics(d_model=d_model)
        self.compressor = TrajectoryCompressor()
        self.chemicals = ChemicalReservoir(chemical_path, autoload=autoload)
        self.pathways = PathwayReinforcement(pathway_path, autoload=autoload)

    @contextmanager
    def activate(self):
        """Route singleton callers to this owner for exactly this boundary."""
        token = type(self)._ACTIVE.set(self)
        try:
            yield self
        finally:
            type(self)._ACTIVE.reset(token)

    def observe_turn(
        self,
        hidden_state: torch.Tensor,
        applied_force: Optional[torch.Tensor] = None,
        utility_feedback: Optional[float] = None,
        pathway_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Observe state transition, update kinematics, compress trajectory,
        diffuse chemicals, and reinforce pathways.
        """
        # 1. Kinematic Step
        v, p_trans = self.kinematics.step(hidden_state, applied_force=applied_force)

        # 2. Record travel into compressor
        self.compressor.record_step(v)
        comp_stats = self.compressor.compress()

        # 3. Chemical decay & deposits
        self.chemicals.decay_step()
        if comp_stats["coherence"] > 0.5:
            # High directional coherence deposits focus and momentum chemicals
            self.chemicals.deposit("focus", 0.2 * comp_stats["coherence"], comp_stats["self_label_vector"])
            self.chemicals.deposit("momentum", 0.15 * comp_stats["compression_ratio"])

        if utility_feedback is not None:
            if utility_feedback > 0:
                self.chemicals.deposit("utility", utility_feedback * 0.5)
                if pathway_label:
                    self.pathways.reinforce(pathway_label, v, utility_feedback)
            else:
                self.chemicals.deposit("friction", abs(utility_feedback) * 0.4)

        return {
            "speed": self.kinematics.speed(),
            "momentum": self.kinematics.momentum_magnitude(),
            "persisting_alignment": self.kinematics.persisting_alignment(),
            "compression_ratio": comp_stats["compression_ratio"],
            "travel_entropy": comp_stats["travel_entropy"],
            "coherence": comp_stats["coherence"],
            "chemicals": dict(self.chemicals.levels),
            "active_pathways": len(self.pathways.pathways),
        }
