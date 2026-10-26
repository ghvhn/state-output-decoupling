"""Memory factors attend to source provenance and current-state stimulation."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from invariants.memory_attention import text_spans
from invariants.synaptic_memory import PlasticWeightHook, SynapticMemoryEngine


class TinySourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.down_proj = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                layer.mlp.down_proj.weight.copy_(torch.eye(4))
            self.layers.append(layer)
        self.requires_grad_(False)

    def forward(self, value):
        for layer in self.layers:
            value = layer.mlp.down_proj(value)
        return value


class SynapticStimulationTests(unittest.TestCase):
    def setUp(self):
        self.previous_main = SynapticMemoryEngine._INSTANCE
        self.model = TinySourceModel()
        self.engine = SynapticMemoryEngine(self.model, target_layers=[0, 1])
        SynapticMemoryEngine._INSTANCE = self.engine
        self.engines = [self.engine]
        self.texts = []

        def inputs(_model, text, pre_formatted=False):
            self.assertTrue(pre_formatted, "Raw source must not receive a chat instruction")
            self.texts.append(text)
            return {"input_ids": torch.tensor([[len(self.texts) - 1]])}

        def hidden(model, input_ids, _attention=None):
            text = self.texts[int(input_ids[0, 0])]
            final = torch.tensor([0.0, 0.0, 1.0, 0.0]) if "LATE_ANSWER" in text else torch.tensor([1.0, 0.0, 0.0, 0.0])
            state = torch.stack([torch.zeros(4), final]).unsqueeze(0)
            states = []
            for layer in model.layers:
                state = layer.mlp.down_proj(state)
                states.append(state[0])
            return torch.stack(states)

        self.inputs_patch = patch("invariants.engine._inputs", side_effect=inputs)
        self.hidden_patch = patch("invariants.engine._hidden_states", side_effect=hidden)
        self.inputs_patch.start()
        self.hidden_patch.start()

    def tearDown(self):
        self.inputs_patch.stop()
        self.hidden_patch.stop()
        for engine in self.engines:
            engine.detach()
        SynapticMemoryEngine._INSTANCE = self.previous_main

    def trace_metadata(self):
        return [trace.metadata for hook in self.engine.hooks.values() for trace in hook.traces.values()]

    def test_spans_include_late_content_with_bounded_forward_work(self):
        text = "early " * 1100 + " LATE_ANSWER"
        self.assertTrue(self.engine.integrate_text_memory(text, max_spans=4))
        self.assertEqual(len(self.texts), 4)
        self.assertTrue(any("LATE_ANSWER" in text for text in self.texts))
        self.assertTrue(all(len(text) <= 512 for text in self.texts))
        self.assertTrue(any(meta["span_start"] > 512 for meta in self.trace_metadata()))
        self.assertEqual(max(meta["span_end"] for meta in self.trace_metadata()), len(text))
        self.assertTrue(all(meta["source_sha256"] == hashlib.sha256(text.encode()).hexdigest() for meta in self.trace_metadata()))
        self.assertTrue(all(meta["key_source"] == "module_input" for meta in self.trace_metadata()))

    def test_actual_state_weights_supporting_late_span(self):
        text = "front " * 500 + " LATE_ANSWER"
        state = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self.assertTrue(self.engine.integrate_text_memory(text, stimulation=state))
        metadata = self.trace_metadata()
        self.assertTrue(metadata)
        self.assertTrue(all("LATE_ANSWER" in item["text_sample"] or item["span_end"] == len(text) for item in metadata))
        self.assertTrue(all(item["span_start"] > 512 and item["state_match"] == 1.0 for item in metadata))

    def test_opposing_live_state_does_not_write_a_reversed_memory(self):
        text = "front " * 500 + " LATE_ANSWER"
        state = torch.tensor([0.0, 0.0, -1.0, 0.0])
        self.assertEqual(self.engine.integrate_text_memory(text, stimulation=state), [])
        self.assertEqual(self.engine.consolidated_trace_count, 0)
        self.assertFalse(any(hook.keys for hook in self.engine.hooks.values()))

    def test_explicit_span_stimulation_and_complete_origin_survive_persistence(self):
        text = "source " * 700 + " LATE_ANSWER"
        spans = text_spans(text)
        target = len(spans) // 2
        provenance = {"source": "model", "correct": False, "provenance_weight": 2,
                      "origin": {"file": "notes.txt", "agent": "a"},
                      "spans": [{"start": spans[target]["start"], "end": spans[target]["end"], "weight": 1.5}]}
        scores = {span["index"]: 0 for span in spans}
        scores[target] = 4
        self.assertTrue(self.engine.integrate_text_memory(text, provenance=provenance, stimulation=scores, max_spans=2))
        metadata = self.trace_metadata()
        self.assertTrue(all(item["span_index"] == target for item in metadata))
        self.assertTrue(all(item["provenance"] == provenance for item in metadata))
        self.assertTrue(all(item["provenance_weight"] == 3 and item["stimulation"] == 4 for item in metadata))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "memory.pt"
            self.engine.save_synaptic_weights(path)
            resumed = SynapticMemoryEngine(self.model, target_layers=[0, 1], persistence_path=path)
            self.engines.append(resumed)
            self.assertTrue(resumed.load_synaptic_weights())
            self.assertEqual(resumed.hooks[0].provenance_weights, [3])
            self.assertEqual(next(iter(resumed.hooks[0].traces.values())).metadata["provenance"], provenance)
            self.assertEqual(resumed._span_rotations, self.engine._span_rotations)

    def test_source_type_and_truth_labels_do_not_assign_hidden_priority(self):
        for source in ("human", "model", "document"):
            self.engine.integrate_text_memory("short source", provenance={"source": source, "correct": source == "human"})
        self.assertEqual(self.engine.hooks[0].provenance_weights, [1, 1, 1])
        self.assertTrue(all(item["provenance_weight"] == 1 for item in self.trace_metadata()))

    def test_rotation_changes_source_sampling_and_remains_separate_from_trace_count(self):
        text = "source " * 1000
        self.engine.integrate_text_memory(text, max_spans=2)
        first_indices = [item["span_index"] for item in self.trace_metadata() if item["source_pass"] == 0]
        self.engine.integrate_text_memory(text, max_spans=2)
        later_indices = [item["span_index"] for item in self.trace_metadata() if item["source_pass"] == 1]
        self.assertNotEqual(first_indices, later_indices)
        self.assertEqual(self.engine.consolidated_trace_count, 8)
        self.assertEqual(self.engine._span_rotations[hashlib.sha256(text.encode()).hexdigest()], 2)

    def test_failed_source_measurement_removes_temporary_hooks(self):
        counts = [len(hook.module._forward_hooks) for hook in self.engine.hooks.values()]
        with patch("invariants.engine._hidden_states", side_effect=RuntimeError("bad source")):
            self.assertEqual(self.engine.integrate_text_memory("source " * 600), [])
        self.assertEqual([len(hook.module._forward_hooks) for hook in self.engine.hooks.values()], counts)

    def test_base_bypasses_encoding_and_shared_parameter_values_remain_unchanged(self):
        parameters = tuple(self.model.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        self.engine.integrate_text_memory("source " * 600 + "LATE_ANSWER")
        base = SynapticMemoryEngine(self.model, target_layers=[0, 1], learnable=False)
        self.engines.append(base)
        encoded = len(self.texts)
        self.assertEqual(base.integrate_text_memory("source", stimulation=torch.ones(4)), [])
        self.assertEqual(len(self.texts), encoded)
        value = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        with base.activate():
            torch.testing.assert_close(self.model(value), value)
        self.assertTrue(torch.norm(self.model(value) - value) > 0)
        for parameter, original, saved in zip(self.model.parameters(), parameters, before):
            self.assertIs(parameter, original)
            torch.testing.assert_close(parameter, saved)


class PlasticStimulationTests(unittest.TestCase):
    def make_hook(self, zero=False):
        module = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            module.weight.copy_(torch.zeros(4, 4) if zero else torch.eye(4))
        hook = PlasticWeightHook(module, layer_idx=0)
        module.requires_grad_(False)
        self.addCleanup(hook.remove)
        return module, hook

    def test_nonlinear_key_match_is_normalized_and_negative_match_is_suppressed(self):
        module, hook = self.make_hook()
        hook.add_associative_trace("memory", torch.tensor([1.0, 0, 0, 0]), torch.tensor([0.0, 0, 1, 0]), alpha=0.04)
        partial = torch.tensor([[[0.5, 3 ** 0.5 / 2, 0.0, 0.0]]])
        torch.testing.assert_close(module(partial) - partial, torch.tensor([[[0.0, 0.0, 0.01, 0.0]]]))
        negative = torch.tensor([[[-1.0, 0, 0, 0]]])
        torch.testing.assert_close(module(negative), negative)
        large_orthogonal = torch.tensor([[[0.1, 1000000.0, 0, 0]]])
        self.assertLess(float((module(large_orthogonal) - large_orthogonal).norm()), 1e-9)
        huge_aligned = torch.tensor([[[1e30, 0, 0, 0]]])
        self.assertTrue(torch.isfinite(module(huge_aligned)).all())
        self.assertAlmostEqual(float(module(huge_aligned)[0, 0, 2]), 0.04, places=6)

    def test_explicit_provenance_changes_relative_priority_under_total_budget(self):
        module, hook = self.make_hook()
        key = torch.tensor([1.0, 0, 0, 0])
        hook.add_associative_trace("a", key, torch.tensor([0.0, 1, 0, 0]), alpha=0.04, metadata={"provenance_weight": 3})
        hook.add_associative_trace("b", key, torch.tensor([0.0, 0, 1, 0]), alpha=0.04, metadata={"provenance_weight": 1})
        value = key.view(1, 1, 4)
        delta = (module(value) - value)[0, 0]
        self.assertAlmostEqual(float(delta[1] / delta[2]), 3, places=5)
        self.assertAlmostEqual(float(delta.norm()), 0.05, places=6)

    def test_many_memories_cannot_exceed_relative_or_absolute_budget(self):
        module, hook = self.make_hook()
        key = torch.tensor([1.0, 0, 0, 0])
        target = torch.tensor([0.0, 1, 0, 0])
        for index in range(100):
            hook.add_associative_trace(str(index), key, target, alpha=0.25, metadata={"provenance_weight": 4})
        small = (key * 0.1).view(1, 1, 4)
        large = (key * 100).view(1, 1, 4)
        self.assertAlmostEqual(float((module(small) - small).norm()), 0.005, places=6)
        self.assertAlmostEqual(float((module(large) - large).norm()), 0.25, places=6)

    def test_zero_original_output_does_not_gain_a_phantom_delta(self):
        module, hook = self.make_hook(zero=True)
        hook.add_associative_trace("memory", torch.ones(4), torch.ones(4), alpha=0.25)
        torch.testing.assert_close(module(torch.ones(1, 1, 4)), torch.zeros(1, 1, 4))


if __name__ == "__main__":
    unittest.main()
