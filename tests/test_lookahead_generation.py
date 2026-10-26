"""Offline integration checks against the real Hugging Face generation loop.

The model is constructed locally with random weights; no model download, CUDA,
or project cognitive state is needed.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from invariants import agentic_engine
from invariants.config import AgenticConfig
from invariants.lookahead_predictor import LookaheadApparatus
from invariants.trigger_tuner import TriggerTuner


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        ids = [3 + ord(char) % 61 for char in text]
        return torch.tensor([ids], dtype=torch.long) if return_tensors == "pt" else ids

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = self.encode(text, return_tensors=return_tensors)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(chr(65 + int(token) % 26) for token in ids if int(token) != 2)


class TestLookaheadGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(917)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )).eval()
        model.generation_config.suppress_tokens = [2]
        self.M = SimpleNamespace(model=model, tok=TinyTokenizer(), device=torch.device("cpu"))
        self.config = AgenticConfig()
        self.config.chatty_log = False
        self.registered_handles = []

        def inert_agentic_handles(*args, **kwargs):
            handle = model.model.layers[0].register_forward_hook(lambda module, inp, out: out)
            self.registered_handles.append(handle)
            return [handle], {"steer_channels": {}}

        self.patch_handles = patch.object(agentic_engine, "get_agentic_handles", side_effect=inert_agentic_handles)
        self.patch_handles.start()
        self.addCleanup(self.patch_handles.stop)
        self.patch_memory = patch.object(agentic_engine, "_engine_memory_snapshot", return_value=(0.0, 0.0))
        self.patch_memory.start()
        self.addCleanup(self.patch_memory.stop)

    def assert_handles_removed(self):
        self.assertTrue(self.registered_handles)
        for handle in self.registered_handles:
            self.assertNotIn(handle.id, self.M.model.model.layers[0]._forward_hooks)

    def test_real_generation_builds_stopping_criteria_and_crosses_chunks(self):
        text, telemetry = agentic_engine.generate_agentic_text(
            self.M, instruction="abc", config=self.config, pre_formatted=True,
            max_new_tokens=26, return_telemetry=True,
        )
        self.assertEqual(len(text), 26)
        self.assertEqual(telemetry["tokens_generated"], 26)
        self.assertEqual([row["tokens_in_chunk"] for row in telemetry["chunk_traces"]], [24, 2])
        self.assertEqual(len(telemetry["eot_probs"]), 26)
        self.assert_handles_removed()

    def test_generation_error_cleans_hooks_and_restores_config(self):
        self.config.chatty_log = True
        self.config._generation_deadline = 123.0
        self.config._generation_budget_sec = 456.0
        cleanup = SimpleNamespace(injected_text="", cleanup=lambda: None)
        with patch.object(cleanup, "cleanup") as cleanup_spy:
            with patch.object(self.M.model, "generate", side_effect=RuntimeError("test generation failure")):
                with self.assertRaisesRegex(RuntimeError, "test generation failure"):
                    agentic_engine.generate_agentic_text(
                        self.M, instruction="abc", config=self.config, pre_formatted=True,
                        max_new_tokens=2, chatty_log=False, max_time=30, mid_chunk_hook=cleanup,
                    )
            cleanup_spy.assert_called_once_with()
        self.assert_handles_removed()
        self.assertTrue(self.config.chatty_log)
        self.assertEqual(self.config._generation_deadline, 123.0)
        self.assertEqual(self.config._generation_budget_sec, 456.0)

    def attach_lookahead(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tuner = TriggerTuner(Path(directory.name)/"tuner.json")
        tuner.set("lookahead_default_k", 1)
        app = LookaheadApparatus(tuner=tuner, d_model=32, hidden_dim=16,
                                embedding_matrix=self.M.model.get_input_embeddings().weight,
                                device=self.M.device)
        handles = app.make_steer_handles(self.M)
        self.addCleanup(lambda: [handle.remove() for handle in handles])
        self.config.lookahead_processor = app.create_logits_processor()
        return app

    def test_exact_generated_arrivals_with_tool_injection_and_chunk_tail(self):
        app = self.attach_lookahead()
        arrivals, generated = [], []
        original_arrival = app.step_arrival
        original_generate = self.M.model.generate

        def record_arrival(token, state=None):
            arrivals.append((app.current_mode, token, state is None))
            return original_arrival(token, state)

        def record_generation(**kwargs):
            output = original_generate(**kwargs)
            generated.extend(output[0, kwargs["input_ids"].shape[1]:].tolist())
            return output

        class Tool:
            injected_text = ""
            calls = 0
            cleaned = False
            def __call__(self, text, state):
                self.calls += 1
                if self.calls == 1:
                    self.injected_text = "xy"
            def cleanup(self):
                self.cleaned = True

        tool = Tool()
        with patch.object(app, "step_arrival", side_effect=record_arrival):
            with patch.object(self.M.model, "generate", side_effect=record_generation):
                _, telemetry = agentic_engine.generate_agentic_text(
                    self.M, instruction="abc", config=self.config, pre_formatted=True,
                    max_new_tokens=26, return_telemetry=True, mid_chunk_hook=tool,
                )
        self.assertEqual([token for mode, token, _ in arrivals if mode == "self"], generated)
        self.assertEqual([token for mode, token, _ in arrivals if mode == "external"], self.M.tok.encode("xy"))
        self.assertEqual(sum(missing for mode, _, missing in arrivals if mode == "self"), 2)
        self.assertEqual(telemetry["tokens_generated"], 26)
        self.assertEqual(app.current_step, 26)  # tool tokens pause, not erase, self history
        self.assertGreater(app.learning_updates, 0)
        self.assertFalse(app._capture_enabled)
        self.assertFalse(app.ledger)
        self.assertTrue(tool.cleaned)
        self.assert_handles_removed()

    def test_no_prompt_recount_or_duplicate_logits_callbacks(self):
        app = self.attach_lookahead()
        processor = self.config.lookahead_processor
        ids = torch.tensor([[3, 4, 5]])
        processor.begin_generation(ids)
        app.last_observed_state = torch.ones(32)
        scores = torch.randn(1, 64)
        self.assertIs(processor(ids, scores), scores)
        pending = sum(map(len, app.ledger.values()))
        processor(ids, scores)
        self.assertEqual(app.current_step, 0)
        self.assertEqual(sum(map(len, app.ledger.values())), pending)
        processor.finish_chunk(torch.tensor([[3, 4, 5, 6]]))
        self.assertEqual(app.current_step, 1)
        self.assertGreater(app.learning_updates, 0)
        processor.finish_generation()
        processor.begin_generation(torch.tensor([[11, 12]]))
        self.assertEqual(app.current_step, 0)
        self.assertIsNone(app.last_observed_state)

    def test_lookahead_lifecycle_ends_on_generation_error(self):
        app = self.attach_lookahead()
        with patch.object(self.M.model, "generate", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                agentic_engine.generate_agentic_text(self.M, instruction="abc", config=self.config,
                                                   pre_formatted=True, max_new_tokens=2)
        self.assertFalse(app._capture_enabled)
        self.assertFalse(app.ledger)
        self.assert_handles_removed()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_half_precision_generation_and_evidence(self):
        device = torch.device("cuda", min(1, torch.cuda.device_count()-1))
        self.M.model.to(device=device, dtype=torch.float16)
        self.M.device = device
        app = self.attach_lookahead()
        _, telemetry = agentic_engine.generate_agentic_text(
            self.M, instruction="abc", config=self.config, pre_formatted=True,
            max_new_tokens=6, return_telemetry=True)
        self.assertEqual(telemetry["tokens_generated"], 6)
        self.assertGreater(app.learning_updates, 0)
        self.assertEqual(app.device, device)
        self.assertFalse(app._capture_enabled)


if __name__ == "__main__":
    unittest.main()
