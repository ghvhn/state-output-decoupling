"""Reader generation remains internal data, never tool execution or prompting."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from invariants import agentic_engine
from invariants.config import AgenticConfig
from invariants.tool_utils import ToolStoppingCriteria


class PlainTokenizer:
    eos_token_id = 2

    def __init__(self):
        self.prompts = []

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        values = [ord(char) + 3 for char in text]
        return torch.tensor([values], dtype=torch.long) if return_tensors == "pt" else values

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        self.prompts.append(text)
        values = self.encode(text, return_tensors=return_tensors)
        return {"input_ids": values, "attention_mask": torch.ones_like(values)}

    def decode(self, values, skip_special_tokens=True):
        if torch.is_tensor(values):
            values = values.tolist()
        return "".join(chr(int(value) - 3) for value in values if int(value) >= 3)


class DocumentReaderChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(18)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=132, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )).eval()
        model.generation_config.suppress_tokens = [0, 1, 2]
        self.M = SimpleNamespace(model=model, tok=PlainTokenizer(), device=torch.device("cpu"))
        self.config = AgenticConfig()
        self.config.chatty_log = True
        self.handles = []

        def handles(*_args, **_kwargs):
            handle = model.model.layers[0].register_forward_hook(lambda _m, _i, out: out)
            self.handles.append(handle)
            return [handle], {"steer_channels": {}}

        self.handle_patch = patch.object(agentic_engine, "get_agentic_handles", side_effect=handles)
        self.handle_patch.start()
        self.addCleanup(self.handle_patch.stop)
        self.memory_patch = patch.object(agentic_engine, "_engine_memory_snapshot", return_value=(0., 0.))
        self.memory_patch.start()
        self.addCleanup(self.memory_patch.stop)

    def assert_clean(self):
        self.assertTrue(self.config.chatty_log)
        for handle in self.handles:
            self.assertNotIn(handle.id, self.M.model.model.layers[0]._forward_hooks)

    def emitted_tool(self, **kwargs):
        self.last_generate_kwargs = kwargs
        self.assertFalse(self.config.chatty_log)
        suffix = self.M.tok.encode("<<CALC:1+1>>", return_tensors="pt")
        return torch.cat([kwargs["input_ids"], suffix], dim=1)

    def test_reader_tool_tokens_are_data_with_no_seam_or_secondary_prompt(self):
        seam = Mock()
        recorder = []
        with patch.object(self.M.model, "generate", side_effect=self.emitted_tool) as generate:
            with patch("invariants.tool_utils.iter_tool_calls") as parse_tools, patch("invariants.tool_utils.evaluate_python_expression") as evaluate:
                text, telemetry = agentic_engine.generate_agentic_text(
                    self.M, instruction="raw source", config=self.config, pre_formatted=True,
                    max_new_tokens=11, generation_channel="reader", chatty_log=True,
                    mid_chunk_hook=seam, synthesis_recorder=recorder, return_telemetry=True,
                )
        self.assertEqual(text, "<<CALC:1+1>>")
        self.assertEqual(generate.call_count, 1)
        parse_tools.assert_not_called()
        evaluate.assert_not_called()
        seam.assert_not_called()
        seam.cleanup.assert_not_called()
        self.assertEqual(self.M.tok.prompts, ["raw source"])
        self.assertFalse(any(isinstance(criteria, ToolStoppingCriteria) for criteria in self.last_generate_kwargs["stopping_criteria"]))
        self.assertEqual(telemetry["generation_channel"], "reader")
        self.assertEqual(recorder[-2]["flags"]["generation_channel"], "reader")
        self.assert_clean()

    def test_speech_default_keeps_tool_execution_and_stopping(self):
        with patch.object(self.M.model, "generate", side_effect=self.emitted_tool):
            with patch("invariants.tool_utils.evaluate_python_expression", return_value="2") as evaluate:
                text, telemetry = agentic_engine.generate_agentic_text(
                    self.M, instruction="question", config=self.config, pre_formatted=True,
                    max_new_tokens=11, chatty_log=False, return_telemetry=True,
                )
        evaluate.assert_called_once_with("1+1")
        self.assertIn(" = 2", text)
        self.assertTrue(any(isinstance(criteria, ToolStoppingCriteria) for criteria in self.last_generate_kwargs["stopping_criteria"]))
        self.assertEqual(telemetry["generation_channel"], "speech")
        self.assert_clean()

    def test_reader_disambiguation_raises_original_error_without_reprompt(self):
        error = agentic_engine.NeedsDisambiguationError("ambiguous source")
        self.config._generation_deadline = 123.
        self.config._generation_budget_sec = 456.
        with patch.object(self.M.model, "generate", side_effect=error) as generate:
            with self.assertRaises(agentic_engine.NeedsDisambiguationError) as raised:
                agentic_engine.generate_agentic_text(
                    self.M, instruction="the document itself", config=self.config,
                    pre_formatted=True, max_new_tokens=3, generation_channel="reader", max_time=30,
                )
        self.assertIs(raised.exception, error)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(self.M.tok.prompts, ["the document itself"])
        self.assertEqual(self.config._generation_deadline, 123.)
        self.assertEqual(self.config._generation_budget_sec, 456.)
        self.assert_clean()

    def test_reader_generic_failure_cleans_hooks_and_restores_config(self):
        with patch.object(self.M.model, "generate", side_effect=RuntimeError("decode failure")):
            with self.assertRaisesRegex(RuntimeError, "decode failure"):
                agentic_engine.generate_agentic_text(
                    self.M, instruction="document", config=self.config, pre_formatted=True,
                    max_new_tokens=3, generation_channel="reader",
                )
        self.assert_clean()

    def test_real_reader_generation_crosses_chunks(self):
        text, telemetry = agentic_engine.generate_agentic_text(
            self.M, instruction="abc", config=self.config, pre_formatted=True,
            max_new_tokens=26, generation_channel="reader", return_telemetry=True,
        )
        self.assertEqual(telemetry["tokens_generated"], 26)
        self.assertEqual([chunk["tokens_in_chunk"] for chunk in telemetry["chunk_traces"]], [24, 2])
        self.assertTrue(text)
        self.assertEqual(telemetry["generation_channel"], "reader")
        self.assert_clean()

    def test_invalid_channel_fails_before_prompt_or_config_changes(self):
        with self.assertRaisesRegex(ValueError, "generation_channel"):
            agentic_engine.generate_agentic_text(self.M, generation_channel="unknown", config=self.config)
        self.assertEqual(self.M.tok.prompts, [])
        self.assertEqual(self.handles, [])
        self.assertTrue(self.config.chatty_log)


if __name__ == "__main__":
    unittest.main()
