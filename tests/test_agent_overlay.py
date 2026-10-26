"""Whole-agent sidecars remain independent on one shared checkpoint."""

import ast
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from invariants.agent_overlay import AgentOverlay, agent_directory
from invariants.lookahead_predictor import FutureWaypointPredictor, LookaheadApparatus
from invariants.synaptic_memory import SynapticMemoryEngine
from invariants.trigger_tuner import TriggerTuner
from invariants.vector_travel import VectorTravelEngine


class AgentOverlayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.previous_synaptic = SynapticMemoryEngine._INSTANCE
        self.previous_travel = VectorTravelEngine._INSTANCE
        self.model = LlamaForCausalLM(LlamaConfig(
            vocab_size=256, hidden_size=16, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64,
        )).eval()
        self.model.d_model = 16
        self.main = SynapticMemoryEngine(self.model, target_layers=[0])
        SynapticMemoryEngine._INSTANCE = self.main
        self.main_travel = VectorTravelEngine(persistence_dir=self.root / "main", autoload=False, d_model=16)
        VectorTravelEngine._INSTANCE = self.main_travel
        self.parent_tuner = TriggerTuner(path=self.root / "main" / "tuner.json")
        self.parent_tuner.register("response_tokens", 512, kind="coefficient")
        self.parent_tuner.register("probe_parent", 0.2)
        self.template = LookaheadApparatus(
            predictor=FutureWaypointPredictor(d_model=16, hidden_dim=8),
            embedding_matrix=self.model.get_input_embeddings().weight,
            tuner=self.parent_tuner, d_model=16, hidden_dim=8,
        )
        self.overlays = []
        self.a = self.new_overlay("a")
        self.b = self.new_overlay("b")
        self.base = self.new_overlay("base", base_model_only=True)

    def tearDown(self):
        for overlay in self.overlays:
            overlay.synaptic.detach()
        self.main.detach()
        SynapticMemoryEngine._INSTANCE = self.previous_synaptic
        VectorTravelEngine._INSTANCE = self.previous_travel
        self.temporary.cleanup()

    def new_overlay(self, name, **kwargs):
        directory = self.root / name
        tuner = TriggerTuner(path=directory / "trigger_tuner.json")
        tuner.register("response_tokens", 512, kind="coefficient")
        overlay = AgentOverlay(self.model, directory, tuner=tuner,
                               lookahead_template=self.template, target_layers=[0], **kwargs)
        self.overlays.append(overlay)
        return overlay

    def add_learning(self, overlay):
        key = torch.arange(1, 17).float()
        delta = torch.arange(16, 0, -1).float()
        overlay.synaptic.consolidate_associative_memory(key, delta, layer_idx=0, metadata={"owner": "a"})
        overlay.cache.store(key, delta, metadata={"tag": "test_agent", "reason": "own learning"})
        evidence = overlay.lookahead.evidence
        environment = evidence.locate(key, "self", "context_a", create=True)
        evidence.observe(environment, 1, key, key + delta, 3, False)
        overlay.config.organic_correction_vector = delta
        overlay.config.recent_corrections.append(delta + 1)
        overlay.config.entropy_threshold = 1.5
        overlay.memory.append_turn("assistant", "agent a's own thought")
        overlay.travel.chemicals.deposit("focus", 1.5, key)

    def test_actual_llama_outputs_restore_main_and_base_without_copying_weights(self):
        parameters = tuple(self.model.parameters())
        pointers = [p.data_ptr() for p in parameters]
        values = [p.detach().clone() for p in parameters]
        key = torch.ones(16)
        self.main.consolidate_associative_memory(key, torch.arange(16).float(), layer_idx=0, learning_rate=0.2)
        self.a.synaptic.consolidate_associative_memory(key, -torch.arange(16).float(), layer_idx=0, learning_rate=0.2)
        inputs = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            main_output = self.model(inputs, use_cache=False).logits
            with self.a.activate():
                agent_output = self.model(inputs, use_cache=False).logits
                self.assertIs(SynapticMemoryEngine.get_instance(), self.a.synaptic)
                self.assertIs(VectorTravelEngine.get_instance(), self.a.travel)
            with self.base.activate():
                base_output = self.model(inputs, use_cache=False).logits
            torch.testing.assert_close(self.model(inputs, use_cache=False).logits, main_output)
        self.assertGreater(float((agent_output - main_output).abs().max()), 1e-5)
        self.assertGreater(float((base_output - main_output).abs().max()), 1e-5)
        for parameter, pointer, value in zip(self.model.parameters(), pointers, values):
            self.assertEqual(parameter.data_ptr(), pointer)
            torch.testing.assert_close(parameter, value)
        for overlay in self.overlays:
            self.assertIs(overlay.synaptic.model, self.model)

    def test_predictor_and_vocabulary_storage_shared_but_evidence_independent(self):
        for overlay in (self.a, self.b):
            self.assertIs(overlay.lookahead.predictor, self.template.predictor)
            self.assertEqual(overlay.lookahead.embedding_matrix.data_ptr(), self.template.embedding_matrix.data_ptr())
            for own, shared in zip(overlay.lookahead.predictor.parameters(), self.template.predictor.parameters()):
                self.assertEqual(own.data_ptr(), shared.data_ptr())
        self.add_learning(self.a)
        self.assertEqual(self.a.lookahead.learning_updates, 1)
        self.assertEqual(self.b.lookahead.learning_updates, 0)
        self.assertEqual(self.template.learning_updates, 0)
        self.assertFalse(self.b.cache.memory)
        self.assertFalse(self.b.synaptic.hooks[0].keys)
        self.assertFalse(self.b.memory.records)
        self.assertFalse(self.main.hooks[0].keys)

    def test_compact_sidecars_resume_only_their_own_learning(self):
        self.add_learning(self.a)
        self.a.lookahead.set_composition("sinusoidal")
        self.a.lookahead.tuner.set("response_tokens", 321)
        self.a.save()
        restored = self.new_overlay("a")
        self.assertEqual(len(restored.cache.memory), 1)
        torch.testing.assert_close(restored.cache.memory[0]["trigger"].float(), torch.arange(1, 17).float())
        self.assertEqual(restored.synaptic.hooks[0].traces["SYN_00001"].metadata, {"owner": "a"})
        self.assertEqual(restored.synaptic.consolidated_trace_count, 1)
        self.assertEqual(restored.lookahead.learning_updates, 1)
        self.assertEqual(restored.lookahead.composition, "sinusoidal")
        self.assertEqual(restored.lookahead.tuner.get("response_tokens"), 321)
        self.assertEqual(restored.config.entropy_threshold, 1.5)
        torch.testing.assert_close(restored.config.organic_correction_vector, torch.arange(16, 0, -1).float())
        self.assertEqual(len(restored.config.recent_corrections), 1)
        self.assertEqual(len(restored.memory.records), 1)
        self.assertEqual(restored.travel.chemicals.levels["focus"], 1.5)
        self.assertTrue(restored.travel.pathways.pathways)
        self.assertFalse(self.b.cache.memory)
        self.assertEqual(self.b.lookahead.learning_updates, 0)
        self.assertEqual(self.main.consolidated_trace_count, 0)
        weights_size = sum(parameter.numel() * parameter.element_size() for parameter in self.model.parameters())
        sidecar_size = sum(path.stat().st_size for path in self.a.directory.rglob("*") if path.is_file())
        self.assertLess(sidecar_size, weights_size // 2)
        for path in self.a.directory.glob("*.pt"):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict):
                self.assertNotIn("model_state_dict", payload)
                self.assertNotIn("predictor", payload)
                self.assertNotIn("embedding_matrix", payload)
                self.assertFalse(any("embed_tokens.weight" in str(key) or "down_proj.weight" in str(key) for key in payload))

    def test_base_ignores_prior_saved_learning_and_has_no_save_effect(self):
        self.add_learning(self.a)
        self.a.save()
        base_from_learned_path = self.new_overlay("a", base_model_only=True)
        self.assertFalse(base_from_learned_path.cache.memory)
        self.assertIsNone(base_from_learned_path.lookahead)
        self.assertFalse(base_from_learned_path.synaptic.hooks[0].keys)
        self.assertFalse(base_from_learned_path.memory.records)
        self.assertEqual(base_from_learned_path.travel.chemicals.levels["focus"], 0)
        self.assertTrue(base_from_learned_path.config.base_model_only)
        before = {path.name: path.read_bytes() for path in self.a.directory.iterdir() if path.is_file()}
        base_from_learned_path.save()
        after = {path.name: path.read_bytes() for path in self.a.directory.iterdir() if path.is_file()}
        self.assertEqual(before, after)

    def test_composition_setting_survives_without_prediction_evidence(self):
        self.a.lookahead.set_composition("sinusoidal")
        self.a.save()
        restored = self.new_overlay("a")
        self.assertEqual(restored.lookahead.learning_updates, 0)
        self.assertEqual(restored.lookahead.composition, "sinusoidal")
        restored.lookahead.set_composition("linear")
        restored.save()
        restored_again = self.new_overlay("a")
        self.assertEqual(restored_again.lookahead.composition, "linear")

    def test_activation_restores_both_owners_after_failure(self):
        with self.a.activate():
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                with self.base.activate():
                    self.assertIs(SynapticMemoryEngine.get_instance(), self.base.synaptic)
                    self.assertIs(VectorTravelEngine.get_instance(), self.base.travel)
                    raise RuntimeError("generation failed")
            self.assertIs(SynapticMemoryEngine.get_instance(), self.a.synaptic)
            self.assertIs(VectorTravelEngine.get_instance(), self.a.travel)
        self.assertIs(SynapticMemoryEngine.get_instance(), self.main)
        self.assertIs(VectorTravelEngine.get_instance(), self.main_travel)

    def test_settings_cannot_restore_global_learning_access(self):
        self.a.directory.mkdir(parents=True, exist_ok=True)
        (self.a.directory / "settings.json").write_text(json.dumps({
            "base_model_only": True, "use_expert_vectors": True,
            "allow_organic_fallback": True, "entropy_threshold": 1.75,
        }), encoding="utf-8")
        restored = self.new_overlay("a")
        self.assertFalse(restored.config.base_model_only)
        self.assertFalse(restored.config.use_expert_vectors)
        self.assertFalse(restored.config.allow_organic_fallback)
        self.assertEqual(restored.config.entropy_threshold, 1.75)

    def test_shell_tuner_uses_code_defaults_and_resumes_own_changes(self):
        shell_path = Path(__file__).resolve().parents[1] / "scripts" / "interactive_phenomenality.py"
        tree = ast.parse(shell_path.read_text(encoding="utf-8-sig"))
        helper = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_agent_tuner")
        namespace = {
            "ROOT": str(self.root), "model_name": "tiny-llama", "model": self.model,
            "tuner": self.parent_tuner, "TriggerTuner": TriggerTuner,
        }
        self.parent_tuner.set("response_tokens", 999)
        self.parent_tuner.observe("response_tokens", 10)
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(shell_path), "exec"), namespace)
        own_tuner = namespace["_agent_tuner"]("worker")
        self.assertEqual(own_tuner.get("response_tokens"), 512)
        self.assertEqual(own_tuner.triggers["response_tokens"].observed, 0)
        self.assertFalse(own_tuner.triggers["response_tokens"].signals)
        self.assertNotIn("probe_parent", own_tuner.triggers)
        own_tuner.set("response_tokens", 123)
        resumed = namespace["_agent_tuner"]("worker")
        self.assertEqual(resumed.get("response_tokens"), 123)
        self.assertEqual(self.parent_tuner.get("response_tokens"), 999)
        self.assertEqual(resumed.path, agent_directory(self.root, "tiny-llama", 16, "worker") / "trigger_tuner.json")

    def test_close_waits_for_background_completion_and_detaches_on_save_failure(self):
        finished = threading.Event()
        stop = threading.Event()

        class Network(threading.Thread):
            def run(self):
                stop.wait(timeout=1)
                finished.set()

            def stop(self):
                stop.set()

        network = Network()
        cen_network = Network()
        self.a.executive_owner._dmn_network = network
        self.a.executive_owner._cen_network = cen_network
        network.start()
        cen_network.start()
        with patch.object(self.a, "save", side_effect=RuntimeError("save failure")):
            with self.assertRaisesRegex(RuntimeError, "save failure"):
                self.a.close()
        self.assertTrue(finished.is_set())
        self.assertFalse(network.is_alive())
        self.assertFalse(cen_network.is_alive())
        self.assertFalse(self.a.synaptic.hooks)

    def test_directory_keys_distinguish_models_and_exact_agent_names(self):
        lower = agent_directory(self.root, "model/a", 16, "Reader")
        other_case = agent_directory(self.root, "model/a", 16, "reader")
        other_model = agent_directory(self.root, "model/b", 16, "Reader")
        other_width = agent_directory(self.root, "model/a", 32, "Reader")
        self.assertEqual(len({lower, other_case, other_model, other_width}), 4)


if __name__ == "__main__":
    unittest.main()
