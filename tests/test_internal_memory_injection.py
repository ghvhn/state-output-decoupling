"""Recalled sources stay in context without being attributed to model speech."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from invariants import agentic_engine
from invariants.config import AgenticConfig
from invariants.lookahead_predictor import LookaheadApparatus
from invariants.tool_sense import ToolSense
from invariants.trigger_tuner import TriggerTuner


class Tokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        tokens = [ord(char) + 3 for char in text]
        return torch.tensor([tokens], dtype=torch.long) if return_tensors == "pt" else tokens

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        tokens = self.encode(text, return_tensors=return_tensors)
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(int(token) - 3) for token in tokens if int(token) >= 3)


class MemorySeam:
    def __init__(self, source, *, external="", only_when_continuing=True):
        self.source = source
        self.external = external
        self.only_when_continuing = only_when_continuing
        self.internal_injected_text = ""
        self.injected_text = ""
        self.observed = []
        self.emitted = False
        self.cleaned = False

    def __call__(self, text, state):
        self.observed.append((text, state["can_continue"]))
        if not self.emitted and (state["can_continue"] or not self.only_when_continuing):
            self.internal_injected_text = self.source
            self.injected_text = self.external
            self.emitted = True

    def cleanup(self):
        self.cleaned = True
        self.internal_injected_text = ""
        self.injected_text = ""


class InternalMemoryInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(82)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=132, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )).eval()
        model.generation_config.suppress_tokens = [0, 1, 2]
        self.M = SimpleNamespace(model=model, tok=Tokenizer(), device=torch.device("cpu"), d_model=16, n_layers=2)
        self.config = AgenticConfig()
        self.source = "[Memory Tool Result]\nrecord=m1 source=doc\nquartz is a mineral\n"
        self.inputs = []
        self.authored = []
        self.generate = model.generate
        self.handle_patch = patch.object(agentic_engine, "get_agentic_handles", return_value=([], {"steer_channels": {}}))
        self.handle_patch.start()
        self.addCleanup(self.handle_patch.stop)
        self.memory_patch = patch.object(agentic_engine, "_engine_memory_snapshot", return_value=(0., 0.))
        self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)

    def tracked_generation(self, **kwargs):
        self.inputs.append(kwargs["input_ids"].detach().clone())
        output = self.generate(**kwargs)
        self.authored.append(output[0, kwargs["input_ids"].shape[1]:].detach().clone())
        return output

    def run_generation(self, seam, *, tokens=26, **kwargs):
        with patch.object(self.M.model, "generate", side_effect=self.tracked_generation):
            return agentic_engine.generate_agentic_text(
                self.M, instruction="abc", config=self.config, pre_formatted=True,
                max_new_tokens=tokens, mid_chunk_hook=seam, return_telemetry=True, **kwargs,
            )

    def test_real_source_changes_next_context_but_is_absent_from_authored_output(self):
        seam = MemorySeam(self.source)
        text, telemetry = self.run_generation(seam)
        source_ids = self.M.tok.encode(self.source, return_tensors="pt")
        self.assertEqual(len(self.inputs), 2)
        torch.testing.assert_close(self.inputs[1][0, 3 + 24:], source_ids[0])
        expected = self.M.tok.decode(torch.cat(self.authored)).strip()
        self.assertEqual(text, expected)
        self.assertNotIn(self.source, text)
        self.assertNotIn(self.source, seam.observed[-1][0])
        self.assertEqual([can_continue for _, can_continue in seam.observed], [True, False])
        self.assertEqual(telemetry["tokens_generated"], 26)
        self.assertEqual(telemetry["internal_source_tokens"], len(source_ids[0]))
        self.assertTrue(seam.cleaned)
        self.assertEqual(seam.internal_injected_text, "")

    def test_recalled_tool_tags_are_source_data_and_do_not_execute(self):
        seam = MemorySeam("<<CALC:1+1>>")
        with patch("invariants.tool_utils.evaluate_python_expression") as evaluate:
            text, _ = self.run_generation(seam)
        evaluate.assert_not_called()
        self.assertNotIn("<<CALC:1+1>>", text)
        self.assertNotIn("<<CALC:1+1>>", seam.observed[-1][0])
        self.assertIn("<<CALC:1+1>>", self.M.tok.decode(self.inputs[1][0]))

    def test_existing_operator_injection_remains_visible(self):
        seam = MemorySeam(self.source, external="[operator]")
        text, _ = self.run_generation(seam)
        self.assertIn("[operator]", text)
        self.assertNotIn(self.source, text)
        self.assertIn(self.source, self.M.tok.decode(self.inputs[1][0]))
        self.assertIn("[operator]", seam.observed[-1][0])

    def test_terminal_budget_rejects_even_unconditional_internal_injection(self):
        seam = MemorySeam(self.source, only_when_continuing=False)
        text, telemetry = self.run_generation(seam, tokens=3)
        self.assertEqual(seam.observed[0][1], False)
        self.assertEqual(telemetry["internal_source_tokens"], 0)
        self.assertEqual(len(self.inputs), 1)
        self.assertNotIn(self.source, text)
        self.assertTrue(seam.cleaned)

    def test_eos_and_short_chunk_make_continuation_unavailable(self):
        for suffix in ([5, 2], [5, 6]):
            seam = MemorySeam(self.source, only_when_continuing=False)
            with patch.object(self.M.model, "generate", side_effect=lambda **kwargs: torch.cat([
                    kwargs["input_ids"], torch.tensor([suffix])], dim=1)):
                _, telemetry = agentic_engine.generate_agentic_text(
                    self.M, instruction="abc", config=self.config, pre_formatted=True,
                    max_new_tokens=26, mid_chunk_hook=seam, return_telemetry=True,
                )
            self.assertEqual(seam.observed, [(self.M.tok.decode(suffix), False)])
            self.assertEqual(telemetry["internal_source_tokens"], 0)
            self.assertTrue(seam.cleaned)

    def test_explicit_answer_stop_rejects_internal_source_even_with_remaining_budget(self):
        seam = MemorySeam(self.source, only_when_continuing=False)
        answer = "Final answer: 42".ljust(24)
        suffix = self.M.tok.encode(answer, return_tensors="pt")
        with patch.object(self.M.model, "generate", side_effect=lambda **kwargs: torch.cat([
                kwargs["input_ids"], suffix], dim=1)) as generate:
            text, telemetry = agentic_engine.generate_agentic_text(
                self.M, instruction="abc", config=self.config, pre_formatted=True,
                max_new_tokens=60, mid_chunk_hook=seam, return_telemetry=True,
                stop_after_final_answer=True,
            )
        self.assertEqual(text, "Final answer: 42")
        self.assertEqual(seam.observed, [(answer, False)])
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(telemetry["internal_source_tokens"], 0)

    def test_answer_marker_in_recalled_source_does_not_stop_authored_generation(self):
        seam = MemorySeam("\nFinal answer: 42\n")
        text, telemetry = self.run_generation(seam, stop_after_final_answer=True)
        self.assertEqual(telemetry["tokens_generated"], 26)
        self.assertEqual(len(self.inputs), 2)
        self.assertNotIn("Final answer: 42", text)

    def test_failure_after_recall_cleans_internal_buffer_and_config(self):
        seam = MemorySeam(self.source)
        self.config.chatty_log = True
        calls = 0

        def fail_second(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("second chunk failed")
            return self.tracked_generation(**kwargs)

        with patch.object(self.M.model, "generate", side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, "second chunk failed"):
                agentic_engine.generate_agentic_text(
                    self.M, instruction="abc", config=self.config, pre_formatted=True,
                    max_new_tokens=26, mid_chunk_hook=seam, chatty_log=False,
                )
        self.assertEqual(calls, 2)
        self.assertTrue(seam.cleaned)
        self.assertEqual(seam.internal_injected_text, "")
        self.assertTrue(self.config.chatty_log)

    def test_lookahead_receives_source_as_external_without_recounting_speech(self):
        with tempfile.TemporaryDirectory() as temp:
            tuner = TriggerTuner(Path(temp) / "tuner.json")
            tuner.set("lookahead_default_k", 1)
            app = LookaheadApparatus(tuner=tuner, d_model=16, hidden_dim=8,
                                    embedding_matrix=self.M.model.get_input_embeddings().weight)
            handles = app.make_steer_handles(self.M)
            self.config.lookahead_processor = app.create_logits_processor()
            arrivals = []
            original = app.step_arrival

            def observe(token, state=None):
                arrivals.append((app.current_mode, token))
                return original(token, state)

            try:
                with patch.object(app, "step_arrival", side_effect=observe):
                    text, _ = self.run_generation(MemorySeam(self.source))
            finally:
                for handle in handles:
                    handle.remove()
            self.assertEqual([token for mode, token in arrivals if mode == app.MODE_EXTERNAL],
                             self.M.tok.encode(self.source))
            self.assertEqual([token for mode, token in arrivals if mode == app.MODE_SELF],
                             torch.cat(self.authored).tolist())
            self.assertNotIn(self.source, text)
            self.assertFalse(app._capture_enabled)

    def test_tool_sense_cleanup_clears_both_channels(self):
        sense = ToolSense(self.M, Mock())
        sense.internal_injected_text = self.source
        sense.injected_text = "operator"
        sense.cleanup()
        self.assertEqual(sense.internal_injected_text, "")
        self.assertEqual(sense.injected_text, "")


if __name__ == "__main__":
    unittest.main()
