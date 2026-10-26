"""Spawned synaptic overlays share weights and retain independent learning."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from invariants.synaptic_memory import SynapticMemoryEngine


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.down_proj = nn.Linear(4, 4, bias=False)
        self.layers = nn.ModuleList([layer])

    def forward(self, value):
        return self.layers[0].mlp.down_proj(value)


class SynapticAgentOverlayTests(unittest.TestCase):
    def setUp(self):
        self.previous_main = SynapticMemoryEngine._INSTANCE
        SynapticMemoryEngine._INSTANCE = None
        self.model = TinyModel()
        self.main = SynapticMemoryEngine(self.model, target_layers=[0])
        SynapticMemoryEngine._INSTANCE = self.main
        self.agent = SynapticMemoryEngine(self.model, target_layers=[0])
        self.base = SynapticMemoryEngine(self.model, target_layers=[0], learnable=False)
        self.engines = [self.main, self.agent, self.base]
        self.key = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.main_value = torch.tensor([0.0, 1.0, 0.0, 0.0])
        self.agent_value = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self.input = self.key.view(1, 1, 4)

    def tearDown(self):
        for engine in self.engines:
            engine.detach()
        SynapticMemoryEngine._INSTANCE = self.previous_main

    def seed(self):
        self.main.consolidate_associative_memory(self.key, self.main_value, layer_idx=0, learning_rate=0.1)
        self.agent.consolidate_associative_memory(self.key, self.agent_value, layer_idx=0, learning_rate=0.2)

    def test_main_agent_and_base_apply_only_their_own_delta(self):
        self.seed()
        with self.base.activate():
            original = self.model(self.input)
        # Live corrections are capped against the checkpoint output norm.
        # Random checkpoint weights can make that budget smaller than alpha.
        budget = min(0.25, 0.05 * float(original.detach().norm()))
        main_delta = min(0.1, budget) * self.main_value
        agent_delta = min(0.2, budget) * self.agent_value
        torch.testing.assert_close(self.model(self.input), original + main_delta)
        with self.agent.activate():
            torch.testing.assert_close(self.model(self.input), original + agent_delta)
        with self.base.activate():
            torch.testing.assert_close(self.model(self.input), original)
        torch.testing.assert_close(self.model(self.input), original + main_delta)

    def test_engines_share_identical_checkpoint_parameters(self):
        original_parameters = tuple(self.model.parameters())
        original_values = [p.detach().clone() for p in original_parameters]
        self.seed()
        for engine in self.engines:
            self.assertIs(engine.model, self.model)
            with engine.activate():
                self.model(self.input)
            for before, current, value in zip(original_parameters, self.model.parameters(), original_values):
                self.assertIs(before, current)
                torch.testing.assert_close(current, value)
        payload = self.agent.state_dict()
        self.assertEqual(set(payload), {"version", "consolidated_trace_count", "layers"})
        self.assertEqual(sum(t.numel() for t in payload["layers"][0]["keys"]), 4)

    def test_nested_selection_restores_owner_after_exception(self):
        self.assertIs(SynapticMemoryEngine.get_instance(), self.main)
        with self.agent.activate():
            self.assertTrue(self.agent.enabled)
            self.assertFalse(self.main.enabled)
            self.assertIs(SynapticMemoryEngine.get_instance(), self.agent)
            with self.assertRaisesRegex(RuntimeError, "failed generation"):
                with self.base.activate():
                    self.assertIs(SynapticMemoryEngine.get_instance(), self.base)
                    self.assertFalse(any(engine.enabled for engine in self.engines))
                    raise RuntimeError("failed generation")
            self.assertIs(SynapticMemoryEngine.get_instance(), self.agent)
        self.assertIs(SynapticMemoryEngine.get_instance(), self.main)

    def test_background_context_keeps_main_and_its_learning(self):
        self.seed()
        result = []

        def background():
            engine = SynapticMemoryEngine.get_instance()
            result.append(engine)
            engine.consolidate_associative_memory(self.key, self.main_value, layer_idx=0)
            result.append(self.model(self.input).detach())

        with self.agent.activate():
            agent_before = self.model(self.input).detach()
            worker = threading.Thread(target=background)
            worker.start()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            torch.testing.assert_close(self.model(self.input), agent_before)
        self.assertIs(result[0], self.main)
        self.assertEqual(self.main.consolidated_trace_count, 2)
        self.assertEqual(self.agent.consolidated_trace_count, 1)
        torch.testing.assert_close(result[1], self.model(self.input))

    def test_base_cannot_learn_or_persist_changes(self):
        self.seed()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "base.pt"
            with self.base.activate():
                self.assertEqual(self.base.consolidate_associative_memory(self.key, self.agent_value), "")
                self.assertEqual(self.base.consolidate_from_dmn_episode({}, reference_state=self.key), [])
                self.assertEqual(self.base.integrate_text_memory("memory"), [])
                self.assertFalse(self.base.load_state_dict(self.agent.state_dict()))
                self.base.apply_synaptic_decay()
                self.base.save_synaptic_weights(target)
            self.assertFalse(target.exists())
            self.assertEqual(self.base.consolidated_trace_count, 0)
            self.assertFalse(self.base.hooks[0].keys)

    def test_persistence_round_trip_restores_metadata_and_counter(self):
        self.agent.consolidate_associative_memory(
            self.key, self.agent_value, layer_idx=0, metadata={"owner": "agent", "reason": ["learning"]},
        )
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "agent.pt"
            self.agent.save_synaptic_weights(target)
            replacement = SynapticMemoryEngine(self.model, target_layers=[0], persistence_path=target)
            self.engines.append(replacement)
            self.assertTrue(replacement.load_synaptic_weights())
            self.assertEqual(replacement.consolidated_trace_count, 1)
            self.assertEqual(replacement.hooks[0].traces["SYN_00001"].metadata, {"owner": "agent", "reason": ["learning"]})
            self.assertEqual(replacement.telemetry()["total_synaptic_traces"], 1)
            with replacement.activate():
                restored = self.model(self.input)
            with self.agent.activate():
                torch.testing.assert_close(restored, self.model(self.input))
            self.assertEqual(replacement.consolidate_associative_memory(self.key, self.agent_value, layer_idx=0), "SYN_00002")

    def test_legacy_layer_dump_still_loads(self):
        self.seed()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "legacy.pt"
            torch.save(self.main.state_dict()["layers"], target)
            self.assertTrue(self.agent.load_synaptic_weights(target))
        self.assertEqual(self.agent.consolidated_trace_count, 1)
        self.assertEqual(self.agent.hooks[0].traces["SYN_00001"].target_module, "mlp.down_proj")
        with self.agent.activate():
            copied = self.model(self.input)
        torch.testing.assert_close(copied, self.model(self.input))

    def test_state_dict_copy_does_not_alias_owner_buffers(self):
        self.seed()
        dump = self.agent.state_dict()
        dump["layers"][0]["keys"][0].zero_()
        dump["layers"][0]["alphas"][0] = 0
        dump["layers"][0]["traces"]["SYN_00001"]["metadata"]["owner"] = "someone else"
        torch.testing.assert_close(self.agent.hooks[0].keys[0], self.key)
        self.assertEqual(self.agent.hooks[0].alphas[0], 0.2)
        self.assertNotIn("owner", self.agent.hooks[0].traces["SYN_00001"].metadata)

    def test_invalid_load_leaves_existing_overlay_intact(self):
        self.seed()
        dump = self.agent.state_dict()
        dump["layers"][0]["values"] = []
        self.assertFalse(self.agent.load_state_dict(dump))
        self.assertEqual(self.agent.consolidated_trace_count, 1)
        self.assertEqual(self.agent.hooks[0].alphas, [0.2])

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        self.seed()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "agent.pt"
            self.agent.save_synaptic_weights(target)
            original = target.read_bytes()
            with patch("invariants.synaptic_memory.torch.save", side_effect=RuntimeError("save failed")):
                with self.assertRaisesRegex(RuntimeError, "save failed"):
                    self.agent.save_synaptic_weights(target)
            self.assertEqual(target.read_bytes(), original)
            self.assertEqual(list(Path(temporary).iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
