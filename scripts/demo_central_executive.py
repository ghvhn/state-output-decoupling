"""Demo script showing the Central Executive Network & Default Mode Network in action.

Runs the waking executor (CEN) and reflective consolidator (DMN) on the shared substrate,
printing live telemetry, activation deltas, dynamic friction, and consolidated memory traces.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
from invariants.central_executive import (
    CentralExecutiveNetwork,
    DefaultModeNetwork,
    DualNetworkRuntime,
    SharedSubstrateMemory,
)


def run_autonomous_demo():
    print("=" * 65)
    print("RUNNING DUAL-NETWORK AUTONOMOUS RUNTIME (CEN + DMN)")
    print("=" * 65)

    substrate = SharedSubstrateMemory(buffer_size=100)

    # Launch dual network using context manager
    with DualNetworkRuntime(
        substrate=substrate,
        gamma=0.05,
        step_interval=0.2,
        consolidation_window=4,
        friction_threshold=0.035,
    ):
        for i in range(8):
            time.sleep(0.2)
            tel = substrate.get_telemetry()
            print(
                f"[Runtime Tick {i+1}] Emitted: {tel['total_emitted']} | "
                f"Consumed: {tel['total_consumed']} | "
                f"Queue Depth: {tel['buffer_depth']} | "
                f"Recent Avg Friction: {tel['average_recent_friction']:.6f}"
            )

    print("\n[Consolidation Complete] DMN Episodic Repository:")
    for episode in substrate.consolidated_repository:
        print(
            f"  -> Episode {episode['episode_id']}: "
            f"Steps {episode['start_step']}..{episode['end_step']} | "
            f"Tokens: {episode['tokens']} | "
            f"Zone: {episode['state_zone']} | "
            f"Mean Friction: {episode['mean_friction']:.5f} | "
            f"Coherence: {episode['trajectory_coherence']:.5f}"
        )


def run_manual_step_hook_demo():
    print("\n" + "=" * 65)
    print("RUNNING MANUAL STEP / INFERENCE HOOK DEMO")
    print("=" * 65)

    substrate = SharedSubstrateMemory()
    cen = CentralExecutiveNetwork(substrate=substrate, gamma=0.05)

    # Simulate sequence of hidden state vectors (e.g. from an LLM forward pass)
    tokens_and_activations = [
        ("The", [1.0, 1.0, 1.0]),
        ("cat", [1.2, 0.9, 1.1]),
        ("sat", [1.3, 0.8, 1.2]),
        ("on", [1.4, 0.7, 1.1]),
        ("quantum", [3.8, -1.5, 4.2]),  # Large semantic / domain jump -> high friction!
        ("superposition", [4.0, -1.8, 4.5]),
    ]

    for token, hidden_state in tokens_and_activations:
        snapshot = cen.process_step(current_tensors=hidden_state, output_token=token)
        print(
            f"Step {snapshot.step:02d} | Token: {snapshot.token_string:<15} | "
            f"|d|: {snapshot.derivative_magnitude:.4f} | "
            f"Dynamic Friction: {snapshot.friction_gradient:.6f} | "
            f"J-Space Variance: {[round(x, 3) for x in snapshot.j_space_variance]}"
        )


def run_probe_system_demo():
    print("\n" + "=" * 65)
    print("RUNNING PROBE SYSTEM INTEGRATION DEMO (ProbeExecutiveAdapter)")
    print("=" * 65)

    import torch
    from invariants.central_executive import ProbeExecutiveAdapter

    # Load actual concept probe vectors from the repository
    probes = {
        "ambiguity": torch.load(ROOT / "invariants" / "ambiguity_vector.pt", map_location="cpu"),
        "disagreement": torch.load(ROOT / "invariants" / "disagreement_vector.pt", map_location="cpu"),
        "flow": torch.load(ROOT / "invariants" / "validated_flow_vector.pt", map_location="cpu"),
        "warranted": torch.load(ROOT / "invariants" / "warranted_confidence_vector.pt", map_location="cpu"),
    }

    substrate = SharedSubstrateMemory()
    cen = CentralExecutiveNetwork(substrate=substrate, gamma=0.08)
    adapter = ProbeExecutiveAdapter(cen=cen, probes=probes, layer=16)

    print(f"Tracking concept axes at Layer {adapter.layer}: {adapter.probe_names}")

    # Synthesize hidden state transitions across semantic boundaries
    probe_ambiguity_vec = adapter.probe_vectors["ambiguity"]
    probe_flow_vec = adapter.probe_vectors["flow"]
    probe_disagreement_vec = adapter.probe_vectors["disagreement"]

    # Base state
    h_base = torch.randn(4096)
    # Trajectory transitions
    steps = [
        ("Assume x = 5", h_base + 1.2 * probe_flow_vec),
        ("Then 2x = 10", h_base + 1.4 * probe_flow_vec),
        ("Wait, or is it 12?", h_base + 2.0 * probe_ambiguity_vec),  # Epistemic friction spike!
        ("No, that contradicts", h_base + 2.5 * probe_disagreement_vec), # Extreme conflict!
        ("Ah, x was 6", h_base + 1.5 * probe_flow_vec),  # Self-correction resolution!
    ]

    for token, h in steps:
        snap = adapter.step_with_activation(h, token_string=token)
        scores = snap.metadata["probe_scores"]
        print(
            f"Token: {token:<22} | "
            f"Friction: {snap.friction_gradient:.5f} | "
            f"Flow: {scores['flow']:+.2f} | "
            f"Ambiguity: {scores['ambiguity']:+.2f} | "
            f"Disagreement: {scores['disagreement']:+.2f}"
        )


if __name__ == "__main__":
    run_autonomous_demo()
    run_manual_step_hook_demo()
    run_probe_system_demo()

