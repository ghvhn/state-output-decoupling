"""Tests for CentralExecutiveNetwork, DefaultModeNetwork, and SharedSubstrateMemory."""

import time
import unittest
from invariants.central_executive import (
    CentralExecutiveNetwork,
    DefaultModeNetwork,
    DualNetworkRuntime,
    SharedSubstrateMemory,
    TensorActivationSnapshot,
)


class TestCentralExecutiveNetwork(unittest.TestCase):
    def setUp(self):
        self.substrate = SharedSubstrateMemory(buffer_size=50)

    def test_derivative_and_friction_computation(self):
        cen = CentralExecutiveNetwork(substrate=self.substrate, gamma=0.05)
        # Previous: [1.0, 1.0, 1.0], Current: [2.0, 0.0, 1.5]
        # Deltas: [1.0, -1.0, 0.5]
        # Magnitude: (1.0 + 1.0 + 0.5) / 3 = 2.5 / 3 = ~0.8333
        # Friction: 0.05 * (2.5 / 3) = ~0.041666
        deltas, mag, friction = cen.compute_derivative([2.0, 0.0, 1.5], [1.0, 1.0, 1.0])
        
        self.assertEqual(len(deltas), 3)
        self.assertAlmostEqual(deltas[0], 1.0)
        self.assertAlmostEqual(deltas[1], -1.0)
        self.assertAlmostEqual(deltas[2], 0.5)
        self.assertAlmostEqual(mag, 2.5 / 3.0)
        self.assertAlmostEqual(friction, 0.05 * (2.5 / 3.0))

    def test_process_step_manual_tensors(self):
        cen = CentralExecutiveNetwork(substrate=self.substrate, gamma=0.05)
        
        # Step 1
        s1 = cen.process_step(current_tensors=[2.0, 1.0, 1.0], output_token="Token_1")
        self.assertEqual(s1.token_string, "Token_1")
        self.assertEqual(s1.step, 1)
        self.assertAlmostEqual(s1.j_space_variance[0], 1.0)
        self.assertEqual(self.substrate.activation_buffer.qsize(), 1)
        
        # Step 2
        s2 = cen.process_step(current_tensors=[2.0, 3.0, 1.0], output_token="Token_2")
        self.assertEqual(s2.token_string, "Token_2")
        self.assertEqual(s2.step, 2)
        # Delta from [2.0, 1.0, 1.0] to [2.0, 3.0, 1.0] -> [0.0, 2.0, 0.0]
        self.assertAlmostEqual(s2.j_space_variance[0], 0.0)
        self.assertAlmostEqual(s2.j_space_variance[1], 2.0)
        self.assertEqual(self.substrate.activation_buffer.qsize(), 2)

    def test_buffer_full_graceful_handling(self):
        # Small buffer
        small_substrate = SharedSubstrateMemory(buffer_size=2)
        cen = CentralExecutiveNetwork(substrate=small_substrate)
        
        cen.process_step()
        cen.process_step()
        # Third one should not raise queue.Full, should record dropped_snapshots
        cen.process_step()
        
        telemetry = small_substrate.get_telemetry()
        self.assertEqual(telemetry["buffer_depth"], 2)
        self.assertEqual(telemetry["dropped_snapshots"], 1)
        self.assertEqual(telemetry["total_emitted"], 3)

    def test_dual_network_threaded_runtime(self):
        with DualNetworkRuntime(
            substrate=self.substrate,
            gamma=0.05,
            step_interval=0.05,
            consolidation_window=4,
            friction_threshold=0.03,
        ) as runtime:
            # Let it run for ~350ms (should generate ~6-7 steps)
            time.sleep(0.35)

        telemetry = self.substrate.get_telemetry()
        self.assertGreater(telemetry["total_emitted"], 3)
        self.assertGreater(telemetry["total_consumed"], 0)
        self.assertGreater(telemetry["consolidated_episodes_count"], 0)
        
        # Check consolidated memory structure
        episode = self.substrate.consolidated_repository[0]
        self.assertIn("episode_id", episode)
        self.assertIn("mean_friction", episode)
        self.assertIn("trajectory_coherence", episode)
        self.assertIn("tokens", episode)

    def test_probe_executive_adapter(self):
        import torch
        from invariants.central_executive import ProbeExecutiveAdapter

        probes = {
            "ambiguity": {16: torch.tensor([1.0, 0.0, 0.0])},
            "flow": {16: torch.tensor([0.0, 1.0, 0.0])},
            "disagreement": {16: torch.tensor([0.0, 0.0, 1.0])},
        }

        cen = CentralExecutiveNetwork(substrate=self.substrate, gamma=0.1)
        adapter = ProbeExecutiveAdapter(cen=cen, probes=probes, layer=16)

        # 1. State aligned with ambiguity
        h1 = torch.tensor([1.0, 0.0, 0.0])
        s1 = adapter.step_with_activation(h1, token_string="confused")
        self.assertAlmostEqual(s1.metadata["probe_scores"]["ambiguity"], 1.0)
        self.assertAlmostEqual(s1.metadata["probe_scores"]["flow"], 0.0)

        # 2. State switching sharply to flow
        h2 = torch.tensor([0.0, 1.0, 0.0])
        s2 = adapter.step_with_activation(h2, token_string="aha!")
        self.assertAlmostEqual(s2.metadata["probe_scores"]["ambiguity"], 0.0)
        self.assertAlmostEqual(s2.metadata["probe_scores"]["flow"], 1.0)
        # Friction reflects delta between [1.0, 0.0, 0.0] and [0.0, 1.0, 0.0] -> deltas [-1.0, 1.0, 0.0]
        # |d| = (1.0 + 1.0 + 0.0) / 3 = 2/3 = 0.6667
        # friction = 0.1 * (2/3) = ~0.0667
        self.assertAlmostEqual(s2.friction_gradient, 0.1 * (2.0 / 3.0), places=4)

    def test_probe_executive_adapter_multilayer(self):
        import torch
        from invariants.central_executive import ProbeExecutiveAdapter

        # Probes across layers 12, 16, 20
        # Ambiguity is rising with depth: L12=x, L16=x, L20=x
        # Flow is orthogonal: y
        probes = {
            "ambiguity": {
                12: torch.tensor([1.0, 0.0, 0.0]),
                16: torch.tensor([1.0, 0.0, 0.0]),
                20: torch.tensor([1.0, 0.0, 0.0]),
            },
            "flow": {
                12: torch.tensor([0.0, 1.0, 0.0]),
                16: torch.tensor([0.0, 1.0, 0.0]),
                20: torch.tensor([0.0, 1.0, 0.0]),
            },
        }

        cen = CentralExecutiveNetwork(substrate=self.substrate, gamma=0.1)
        layers = [12, 16, 20]
        adapter = ProbeExecutiveAdapter(cen=cen, probes=probes, layers=layers)

        self.assertEqual(adapter.layers, [12, 16, 20])
        self.assertEqual(adapter.layer, 16)
        self.assertIn("ambiguity", adapter.probe_names)
        self.assertIn("flow", adapter.probe_names)

        # Create 3D hidden states tensor [28, seq_len=4, dim=3]
        # Make ambiguity emerge with depth:
        # At L12: [0.2, 0.8, 0.0] -> cos with ambiguity = 0.2 / sqrt(0.2^2 + 0.8^2) = 0.2425
        # At L16: [0.6, 0.8, 0.0] -> cos with ambiguity = 0.6 / 1.0 = 0.6000
        # At L20: [1.0, 0.0, 0.0] -> cos with ambiguity = 1.0 / 1.0 = 1.0000
        hs = torch.zeros((28, 4, 3))
        hs[12, -1, :] = torch.tensor([0.2, 0.8, 0.0])
        hs[16, -1, :] = torch.tensor([0.6, 0.8, 0.0])
        hs[20, -1, :] = torch.tensor([1.0, 0.0, 0.0])

        snap = adapter.step_with_activation(hs, token_string="evolving_thought")

        meta = snap.metadata
        self.assertIn("layer_scores", meta)
        self.assertIn("depth_gradients", meta)
        self.assertIn("concept_overlap", meta)

        l_scores = meta["layer_scores"]["ambiguity"]
        self.assertAlmostEqual(l_scores[16], 0.6, places=4)
        self.assertAlmostEqual(l_scores[20], 1.0, places=4)

        # Ambiguity is rising: depth gradient d(ambiguity)/dl must be positive
        depth_grad = meta["depth_gradients"]["ambiguity"]
        self.assertGreater(depth_grad, 0.0)

        # Overlap between ambiguity and flow (orthogonal basis) should be 0
        self.assertAlmostEqual(meta["concept_overlap"]["ambiguity_x_flow"], 0.0, places=4)

        # Integrated band score should be the average across 12, 16, 20
        expected_mean = (l_scores[12] + l_scores[16] + l_scores[20]) / 3.0
        self.assertAlmostEqual(meta["probe_scores"]["ambiguity"], expected_mean, places=4)

    def test_probe_executive_adapter_disaggregated(self):
        import torch
        from invariants.central_executive import ProbeExecutiveAdapter

        probes = {
            "ambiguity": {
                12: torch.tensor([1.0, 0.0, 0.0]),
                16: torch.tensor([0.0, 1.0, 0.0]),
            },
        }

        cen = CentralExecutiveNetwork(substrate=self.substrate, gamma=0.1)
        adapter = ProbeExecutiveAdapter(cen=cen, probes=probes, layers=[12, 16], mode="disaggregated")

        self.assertIn("ambiguity@L12", adapter.probe_names)
        self.assertIn("ambiguity@L16", adapter.probe_names)

        hs = {
            12: torch.tensor([1.0, 0.0, 0.0]),
            16: torch.tensor([0.0, 1.0, 0.0]),
        }
        snap = adapter.step_with_activation(hs, token_string="disaggregated_step")
        self.assertAlmostEqual(snap.metadata["probe_scores"]["ambiguity@L12"], 1.0, places=4)
        self.assertAlmostEqual(snap.metadata["probe_scores"]["ambiguity@L16"], 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
