"""Agent corrections have distinct owners while the checkpoint stays shared."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from invariants import agentic_engine
from invariants.cognitive_cache import CognitiveCache
from invariants.config import AgenticConfig, _global_registry
from invariants.tool_utils import ToolStoppingCriteria
from invariants.vector_travel import VectorTravelEngine


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        tokens = [3 + ord(char) % 61 for char in text]
        return torch.tensor([tokens]) if return_tensors == "pt" else tokens

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        tokens = self.encode(text, return_tensors=return_tensors)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(65 + int(token) % 26) for token in tokens if int(token) != 2)


class AgentCacheIsolationTests(unittest.TestCase):
    def test_explicit_files_start_empty_and_never_write_legacy_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            legacy = Mock()
            with patch.object(VectorTravelEngine, "get_instance", return_value=legacy):
                a = CognitiveCache(file=directory / "a" / "cache.pt", autoload=False)
                b = CognitiveCache(file=directory / "b" / "cache.pt", autoload=False)
                a.store(torch.tensor([1., 0.]), torch.tensor([0., 1.]), {"concept": "first"})
                self.assertIsNone(b.retrieve(torch.tensor([1., 0.])))
                torch.testing.assert_close(a.retrieve(torch.tensor([1., 0.])), torch.tensor([0., 1.]))
            self.assertEqual(len(a.memory), 1)
            self.assertEqual(b.memory, [])
            self.assertIn("first", a.vector_travel.pathways.pathways)
            self.assertEqual(b.vector_travel.pathways.pathways, {})
            self.assertEqual(b.vector_travel.chemicals.levels["momentum"], 0.)
            legacy.pathways.reinforce.assert_not_called()
            legacy.chemicals.deposit.assert_not_called()
            self.assertTrue((directory / "a" / "cache.pt").is_file())
            self.assertTrue((directory / "a" / "strengthened_pathways.pt").is_file())
            self.assertFalse((directory / "b").exists())

    def test_optional_load_reads_only_the_selected_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "agent" / "cache.pt"
            first = CognitiveCache(file=path, autoload=False)
            first.store(torch.tensor([1., 0.]), torch.tensor([0., 1.]), {"concept": "owned"})
            fresh = CognitiveCache(file=path, autoload=False)
            restored = CognitiveCache(file=path)
            self.assertEqual(fresh.memory, [])
            self.assertEqual(fresh.vector_travel.pathways.pathways, {})
            self.assertEqual(len(restored.memory), 1)
            self.assertIn("owned", restored.vector_travel.pathways.pathways)

    def test_explicit_travel_owner_wins_over_active_context(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            a = VectorTravelEngine(persistence_dir=directory / "a", autoload=False)
            b = VectorTravelEngine(persistence_dir=directory / "b", autoload=False)
            cache = CognitiveCache(file=directory / "a" / "cache.pt", autoload=False, vector_travel=a)
            with b.activate():
                cache.store(torch.tensor([1., 0.]), torch.tensor([0., 1.]))
            self.assertTrue(a.pathways.pathways)
            self.assertEqual(b.pathways.pathways, {})

    def test_nested_travel_context_restores_after_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            a = VectorTravelEngine(persistence_dir=Path(temp) / "a", autoload=False)
            b = VectorTravelEngine(persistence_dir=Path(temp) / "b", autoload=False)
            legacy = Mock()
            with patch.object(VectorTravelEngine, "_INSTANCE", legacy):
                self.assertIs(VectorTravelEngine.get_instance(), legacy)
                with a.activate():
                    self.assertIs(VectorTravelEngine.get_instance(), a)
                    with self.assertRaisesRegex(RuntimeError, "boundary"):
                        with b.activate():
                            self.assertIs(VectorTravelEngine.get_instance(), b)
                            raise RuntimeError("boundary")
                    self.assertIs(VectorTravelEngine.get_instance(), a)
                self.assertIs(VectorTravelEngine.get_instance(), legacy)

    def test_legacy_cache_uses_the_active_travel_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            travel = VectorTravelEngine(persistence_dir=Path(temp), autoload=False)
            cache = CognitiveCache(autoload=False)
            cache.file = Path(temp) / "legacy-cache.pt"
            with travel.activate():
                cache.store(torch.tensor([1., 0.]), torch.tensor([0., 1.]))
            self.assertTrue(travel.pathways.pathways)


class BaseGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(122)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )).eval()
        model.generation_config.suppress_tokens = [2]
        self.M = SimpleNamespace(model=model, tok=Tokenizer(), device=torch.device("cpu"), d_model=16, n_layers=2)

    def test_base_installs_no_correction_hooks_or_vector_lookups(self):
        config = AgenticConfig()
        config.base_model_only = True
        with patch.object(_global_registry, "get_vecs") as experts, patch.object(_global_registry, "get_special_vec") as special:
            handles, state = agentic_engine.get_agentic_handles(self.M, config=config)
        self.assertEqual(handles, [])
        self.assertEqual(state["branch_names"], [])
        self.assertEqual(state["steer_channels"], {})
        self.assertIsNone(config.committee)
        experts.assert_not_called()
        special.assert_not_called()
        self.assertTrue(all(not layer._forward_hooks for layer in self.M.model.model.layers))

    def test_base_real_generation_matches_checkpoint_and_preserves_parameters(self):
        config = AgenticConfig()
        config.base_model_only = True
        config.lookahead_processor = Mock()
        seam = Mock()
        weights = [(parameter, parameter.data_ptr(), parameter.detach().clone()) for parameter in self.M.model.parameters()]
        inputs = self.M.tok("abc")
        with torch.no_grad():
            expected = self.M.model.generate(**inputs, max_new_tokens=26, do_sample=False, use_cache=True,
                                             eos_token_id=[2, 128009], pad_token_id=2)
        with patch.object(agentic_engine, "_engine_memory_snapshot", return_value=(0., 0.)):
            with patch.object(agentic_engine, "_global_cache") as legacy:
                text, telemetry = agentic_engine.generate_agentic_text(
                    self.M, instruction="abc", config=config, pre_formatted=True,
                    max_new_tokens=26, return_telemetry=True, mid_chunk_hook=seam,
                )
        self.assertEqual(text, self.M.tok.decode(expected[0, 3:]))
        self.assertEqual(telemetry["tokens_generated"], 26)
        self.assertEqual([chunk["tokens_in_chunk"] for chunk in telemetry["chunk_traces"]], [24, 2])
        legacy.retrieve.assert_not_called()
        legacy.store.assert_not_called()
        seam.assert_not_called()
        seam.cleanup.assert_not_called()
        self.assertEqual(config.lookahead_processor.mock_calls, [])
        for parameter, address, value in weights:
            self.assertEqual(parameter.data_ptr(), address)
            torch.testing.assert_close(parameter, value, rtol=0., atol=0.)

    def test_base_skips_tool_output_execution_and_keeps_channel(self):
        config = AgenticConfig()
        config.base_model_only = True
        seen = {}

        def generate(**kwargs):
            seen.update(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([[3, 4]])], dim=1)

        with patch.object(self.M.model, "generate", side_effect=generate):
            with patch("invariants.tool_utils.iter_tool_calls") as parse, patch("invariants.tool_utils.evaluate_python_expression") as evaluate:
                _, telemetry = agentic_engine.generate_agentic_text(
                    self.M, instruction="abc", config=config, pre_formatted=True,
                    max_new_tokens=2, return_telemetry=True,
                )
        parse.assert_not_called()
        evaluate.assert_not_called()
        self.assertFalse(any(isinstance(criteria, ToolStoppingCriteria) for criteria in seen["stopping_criteria"]))
        self.assertEqual(telemetry["generation_channel"], "speech")

    def test_active_config_cache_receives_real_synthesis_retrieval(self):
        config = AgenticConfig()
        config.use_expert_vectors = False
        config.entropy_threshold = 0.
        config.force_synthesis = True
        config.max_loops = 1
        config.max_routing_events = 1
        config.max_synthesis_events = 1
        owned = Mock()
        owned.retrieve.return_value = torch.ones(16) * .01
        config.cognitive_cache = owned
        with patch.object(_global_registry, "get_special_vec", return_value=None):
            with patch.object(agentic_engine, "_global_cache") as legacy:
                agentic_engine.generate_agentic_text(self.M, instruction="abc", config=config,
                                                    pre_formatted=True, max_new_tokens=2)
        owned.retrieve.assert_called_once()
        legacy.retrieve.assert_not_called()
        legacy.store.assert_not_called()
        self.assertTrue(all(not layer._forward_hooks for layer in self.M.model.model.layers))


if __name__ == "__main__":
    unittest.main()
