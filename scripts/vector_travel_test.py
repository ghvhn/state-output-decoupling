"""
Model-free test suite for Vector Travel, Trajectory Compression,
Chemical Reservoir, and Pathway Reinforcement.

Run:
    .venv\\Scripts\\python.exe scripts\\vector_travel_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from invariants.vector_travel import (
    VectorKinematics,
    TrajectoryCompressor,
    ChemicalReservoir,
    PathwayReinforcement,
    VectorTravelEngine,
)


def test_vector_kinematics():
    kin = VectorKinematics(d_model=64, mass=2.0, friction=0.1, momentum_decay=0.8)
    s0 = torch.randn(64)
    v0, p0 = kin.step(s0)
    assert kin.position is not None
    assert kin.step_count == 1
    assert torch.allclose(v0, torch.zeros(64))

    # Move to a new state
    s1 = torch.randn(64)
    v1, p1 = kin.step(s1, dt=1.0)
    assert kin.step_count == 2
    assert kin.speed() > 0.0
    assert kin.momentum_magnitude() > 0.0
    assert abs(kin.momentum_magnitude() - 2.0 * kin.speed()) < 1e-5

    # Test alignment
    align = kin.persisting_alignment()
    assert -1.0 <= align <= 1.0

    # Test applied force accelerates
    force = torch.randn(64) * 2.0
    v2, p2 = kin.step(s1, applied_force=force, dt=1.0)
    assert kin.acceleration.norm() > 0.0
    print("PASS test_vector_kinematics")


def test_trajectory_compression_and_self_label():
    comp = TrajectoryCompressor(window_size=32)
    assert comp.compress()["compression_ratio"] == 1.0

    # Simulate trajectory travelling along a primary axis with small orthogonal noise
    primary_dir = torch.randn(64)
    primary_dir = primary_dir / primary_dir.norm()

    for _ in range(16):
        noise = torch.randn(64) * 0.05
        step_v = primary_dir * 1.0 + noise
        comp.record_step(step_v)

    stats = comp.compress()
    assert stats["trajectory_length"] == 16
    # Should show high compression ratio along the primary travel axis
    assert stats["compression_ratio"] > 0.6, f"Expected > 0.6, got {stats['compression_ratio']}"
    assert stats["coherence"] > 0.8
    assert stats["self_label_vector"] is not None

    # Self-label vector should be highly aligned with primary_dir
    self_label = stats["self_label_vector"]
    cos_sim = abs(float(F.cosine_similarity(self_label, primary_dir, dim=0).item()))
    assert cos_sim > 0.9, f"Self-label alignment expected > 0.9, got {cos_sim}"
    print("PASS test_trajectory_compression_and_self_label")


def test_chemical_reservoir():
    with tempfile.TemporaryDirectory() as tmp:
        chem_path = Path(tmp) / "chem.pt"
        reservoir = ChemicalReservoir(persistence_path=chem_path)
        assert reservoir.levels["focus"] == 0.0

        v = torch.randn(64)
        reservoir.deposit("focus", 2.0, direction_vector=v)
        reservoir.deposit("momentum", 3.0)
        assert reservoir.levels["focus"] == 2.0
        assert reservoir.levels["momentum"] == 3.0
        assert reservoir.vectors["focus"] is not None

        # Test persistence
        reservoir.save()
        loaded = ChemicalReservoir(persistence_path=chem_path)
        assert loaded.levels["focus"] == 2.0
        assert loaded.levels["momentum"] == 3.0

        # Test decay
        reservoir.decay_step()
        assert reservoir.levels["focus"] == 2.0 * 0.90
        assert reservoir.levels["momentum"] == 3.0 * 0.92

        # Test physics field modifiers
        mods = reservoir.effective_physics_modifiers()
        assert mods["g_multiplier"] > 1.0
        assert "effective_friction" in mods

        # Test set_level and reset
        reservoir.set_level("focus", 1.5)
        assert reservoir.levels["focus"] == 1.5
        reservoir.reset()
        assert reservoir.levels["focus"] == 0.0
        assert reservoir.levels["momentum"] == 0.0
        print("PASS test_chemical_reservoir")


def test_pathway_reinforcement():
    with tempfile.TemporaryDirectory() as tmp:
        path_file = Path(tmp) / "pathways.pt"
        reinforce = PathwayReinforcement(persistence_path=path_file, max_pathways=8)

        useful_vec = torch.randn(64)
        useful_vec = useful_vec / useful_vec.norm()

        reinforce.reinforce("reasoning_conduit", useful_vec, utility_score=2.0)
        assert "reasoning_conduit" in reinforce.pathways
        cond_initial = reinforce.pathways["reasoning_conduit"]["conductance"]
        assert cond_initial > 1.0

        # Reinforce again along same vector
        reinforce.reinforce("reasoning_conduit", useful_vec, utility_score=1.5)
        assert reinforce.pathways["reasoning_conduit"]["conductance"] > cond_initial

        # Conversion to gravity field bodies
        bodies = reinforce.as_gravity_bodies()
        assert len(bodies) == 1
        name, mass, dirs = bodies[0]
        assert name == "pathway:reasoning_conduit"
        assert mass == reinforce.pathways["reasoning_conduit"]["conductance"]
        assert 16 in dirs
        print("PASS test_pathway_reinforcement")


def test_vector_travel_engine_observe_turn():
    engine = VectorTravelEngine()
    engine.kinematics.d_model = 64
    engine.kinematics.reset()
    engine.compressor.clear()

    # Step through 5 turns with consistent travel direction and positive feedback
    base_dir = torch.randn(64)
    base_dir = base_dir / base_dir.norm()

    last_stats = None
    for i in range(6):
        state = base_dir * (i + 1) + torch.randn(64) * 0.05
        last_stats = engine.observe_turn(
            state,
            utility_feedback=1.0 if i > 0 else None,
            pathway_label="active_lane" if i > 0 else None,
        )

    assert last_stats is not None
    assert last_stats["compression_ratio"] > 0.6
    assert last_stats["chemicals"]["utility"] > 0.0
    assert last_stats["active_pathways"] >= 1
    print("PASS test_vector_travel_engine_observe_turn")


if __name__ == "__main__":
    test_vector_kinematics()
    test_trajectory_compression_and_self_label()
    test_chemical_reservoir()
    test_pathway_reinforcement()
    test_vector_travel_engine_observe_turn()
    print("\nALL VECTOR TRAVEL & COMPRESSION TESTS PASSED.")
