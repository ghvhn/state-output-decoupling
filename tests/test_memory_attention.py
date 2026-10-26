"""Whole-source, provenance-conditioned memory attention and bounded recall."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from invariants.memory_attention import (
    SalientMemoryIndex, auxiliary_encoding, encode_weighted_memory,
    is_auxiliary_encoding, provenance_weight, select_spans, text_spans,
)


def record(identifier, text, *, scope="default", provenance=None, metrics=None):
    return {"record_id": identifier, "text": text, "scope": scope,
            "provenance": provenance or {}, "metrics": metrics or {}}


class Encoder:
    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append((text, is_auxiliary_encoding()))
        return torch.tensor([0., 1.]) if "BETA" in text else torch.tensor([1., 0.])


class RawTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        self.calls.append((text, add_special_tokens))
        tokens = [2 if word == "BETA" else 1 for word in text.split()] or [1]
        return {"input_ids": torch.tensor([tokens]), "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long)}

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("Memory source encoding cannot invent an instruction")


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
        self.norm = torch.nn.Identity()
        self.calls = []

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.calls.append((kwargs, is_auxiliary_encoding()))
        return SimpleNamespace(last_hidden_state=self.norm(self.embedding(input_ids)))


def bundle():
    backbone = Backbone()
    language_model = torch.nn.Module()
    language_model.model = backbone
    return SimpleNamespace(model=language_model, tok=RawTokenizer(), device=torch.device("cpu"))


class SourceSpanTests(unittest.TestCase):
    def test_whole_source_offsets_and_whitespace_are_preserved(self):
        source = "Early sentence.\n\n" + "Middle facts and details. " * 50 + "Final detail!"
        spans = text_spans(source, 64)
        self.assertEqual("".join(span["text"] for span in spans), source)
        self.assertEqual([span["index"] for span in spans], list(range(len(spans))))
        self.assertEqual(spans[0]["start"], 0)
        self.assertEqual(spans[-1]["end"], len(source))
        for left, right in zip(spans, spans[1:]):
            self.assertEqual(left["end"], right["start"])
        self.assertTrue(all(0 < len(span["text"]) <= 64 for span in spans))

    def test_empty_and_unbroken_sources_have_exact_coverage(self):
        self.assertEqual(text_spans(""), [])
        source = "a" * 1027
        spans = text_spans(source, 512)
        self.assertEqual([len(span["text"]) for span in spans], [512, 512, 3])
        self.assertEqual("".join(span["text"] for span in text_spans("a b", 1)), "a b")

    def test_cold_selection_samples_ends_and_rotation_covers_every_region(self):
        spans = text_spans("0123456789" * 20, 10)
        chosen = select_spans(spans, max_spans=4)
        self.assertEqual(chosen[0]["index"], 0)
        self.assertEqual(chosen[-1]["index"], len(spans) - 1)
        seen = {span["index"] for rotation in range(len(spans))
                for span in select_spans(spans, max_spans=2, rotation=rotation)}
        self.assertEqual(seen, set(range(len(spans))))

    def test_provenance_annotations_can_prioritize_middle_with_tail_coverage(self):
        spans = text_spans("abcdefghij" * 10, 10)
        provenance = {"spans": [{"start": 40, "end": 50, "weight": 3}]}
        chosen = select_spans(spans, max_spans=2, provenance=provenance)
        self.assertEqual([span["index"] for span in chosen], [4, 9])
        self.assertEqual(chosen[0]["weight"], 3.0)
        self.assertEqual(chosen[1]["weight"], 1.0)

    def test_stimulation_mapping_callback_and_sequence_agree(self):
        spans = text_spans("abcdefghij" * 4, 10)
        scores = [0.1, 3.0, 0.2, 0.0]
        variants = [scores, dict(enumerate(scores)), lambda span: scores[span["index"]]]
        for stimulation in variants:
            chosen = select_spans(spans, 2, stimulation=stimulation)
            self.assertEqual([span["index"] for span in chosen], [1, 3])
            self.assertEqual(chosen[0]["weight"], 3.0)

    def test_provenance_is_attention_and_never_automatic_truth(self):
        self.assertEqual(provenance_weight({"role": "user", "source": "verified", "truth": False}), 1.0)
        self.assertEqual(provenance_weight({"role": "assistant", "correct": False}), 1.0)
        self.assertEqual(provenance_weight({"relevance_weight": 2}), 2.0)
        self.assertEqual(provenance_weight({}, {"provenance_weight": 3}), 3.0)
        self.assertEqual(provenance_weight({"provenance_weight": 0}, {"provenance_weight": 3}), 0.0)
        for value, expected in [(float("nan"), 1), (float("inf"), 1), (-10, 0), (100, 4), ("bad", 1)]:
            self.assertEqual(provenance_weight({"provenance_weight": value}), expected)

    def test_auxiliary_marker_is_nested_and_exception_safe(self):
        self.assertFalse(is_auxiliary_encoding())
        with self.assertRaises(RuntimeError):
            with auxiliary_encoding():
                self.assertTrue(is_auxiliary_encoding())
                with auxiliary_encoding():
                    self.assertTrue(is_auxiliary_encoding())
                self.assertTrue(is_auxiliary_encoding())
                raise RuntimeError("failed measurement")
        self.assertFalse(is_auxiliary_encoding())


class WeightedEncodingTests(unittest.TestCase):
    def test_late_salient_text_is_encoded_and_current_state_changes_weighting(self):
        source = "ALPHA " * 100 + "BETA"
        encoder = Encoder()
        result = encode_weighted_memory(None, source, state=torch.tensor([0., 1.]),
                                        span_chars=64, max_spans=4, encoder=encoder)
        self.assertEqual(result["newly_encoded"], 4)
        self.assertTrue(any("BETA" in span["text"] for span in result["spans"]))
        self.assertTrue(torch.equal(result["vector"], torch.tensor([0., 1.])))
        self.assertEqual(result["spans"][-1]["end"], len(source))
        self.assertTrue(all(marked for _, marked in encoder.calls))
        self.assertFalse(result["vector"].requires_grad)

    def test_raw_model_uses_all_tokens_in_selected_window_without_prompt(self):
        model = bundle()
        result = encode_weighted_memory(model, "ALPHA BETA", max_spans=1)
        expected = torch.tensor([1., 1.]) / 2 ** 0.5
        self.assertTrue(torch.allclose(result["vector"], expected))
        self.assertEqual(model.tok.calls, [("ALPHA BETA", False)])
        self.assertTrue(all(marked for _, marked in model.model.model.calls))
        self.assertTrue(all(options["output_hidden_states"] is False for options, _ in model.model.model.calls))

    def test_zero_provenance_cannot_create_a_direction(self):
        result = encode_weighted_memory(None, "BETA", provenance={"provenance_weight": 0}, encoder=Encoder())
        self.assertIsNone(result["vector"])
        self.assertEqual(result["spans"][0]["weight"], 0)

    def test_failed_and_mismatched_encoding_do_not_fabricate_state(self):
        self.assertIsNone(encode_weighted_memory(None, "", encoder=Encoder())["vector"])
        def broken(_text):
            raise RuntimeError("unsupported")
        result = encode_weighted_memory(None, "source", encoder=broken)
        self.assertEqual(result["newly_encoded"], 0)
        self.assertIsNone(result["vector"])
        self.assertFalse(is_auxiliary_encoding())


class SalientMemoryIndexTests(unittest.TestCase):
    def setUp(self):
        self.encoder = Encoder()
        self.index = SalientMemoryIndex(None, encoder=self.encoder, span_chars=64, max_spans=4)
        self.records = [record("alpha", "ALPHA"), record("beta", "BETA")]

    def test_activation_queries_rerank_cached_memory_without_another_forward(self):
        stats = self.index.prepare(self.records)
        self.assertEqual(stats["newly_encoded"], 2)
        self.assertTrue(stats["complete"])
        first = self.index.query(torch.tensor([1., 0.]))
        second = self.index.rank(self.records, torch.tensor([0., 1.]), allow_encode=False)
        self.assertEqual(first[0]["record_id"], "alpha")
        self.assertEqual(second[0]["record_id"], "beta")
        self.assertEqual(len(self.encoder.calls), 2)
        self.assertEqual(self.index._rotation, 1)
        self.assertEqual(self.index._record_rotation, {"alpha": 1, "beta": 1})

    def test_cached_queries_keep_memory_recallable_and_cannot_mutate_cache(self):
        self.index.prepare(self.records)
        first = self.index.query(torch.tensor([0., 1.]))
        first[0]["span"]["vector"].zero_()
        first[0]["vector"].zero_()
        second = self.index.query(torch.tensor([0., 1.]))
        self.assertEqual(second[0]["record_id"], "beta")
        self.assertEqual(second[0]["stimulation"], 1.0)
        self.assertEqual(len(self.encoder.calls), 2)

    def test_late_source_match_is_not_lost_to_prefix_truncation(self):
        source = "ALPHA " * 120 + "BETA"
        candidates = self.index.rank([record("long", source)], torch.tensor([0., 1.]))
        self.assertEqual(candidates[0]["stimulation"], 1.0)
        self.assertEqual(candidates[0]["span"]["end"], len(source))
        self.assertIn("BETA", candidates[0]["span"]["text"])
        self.assertGreater(candidates[0]["span_count"], 4)

    def test_scope_filter_and_distributed_record_scan_include_recent_tail(self):
        records = [record(str(i), "BETA" if i == 19 else "ALPHA") for i in range(20)]
        self.index.prepare(records, scan=2)
        candidates = self.index.query(torch.tensor([0., 1.]))
        self.assertEqual(candidates[0]["record_id"], "19")
        self.index.prepare([record("other", "BETA", scope="other"), self.records[0]], scope="default")
        self.assertEqual([item["record_id"] for item in self.index.query(torch.tensor([1., 0.]))], ["alpha"])
        self.assertEqual(self.index.query(torch.tensor([1., 0.]), scope="other"), [])

    def test_explicit_strength_affects_ranking_without_assigning_truth(self):
        records = [record("a", "BETA", provenance={"truth": True}),
                   record("b", "BETA", provenance={"truth": False, "provenance_weight": 2})]
        self.index.prepare(records)
        result = self.index.query(torch.tensor([0., 1.]))
        self.assertEqual(result[0]["record_id"], "b")
        self.assertEqual(result[0]["score"], 2.0)
        self.assertEqual(result[0]["evidence_weight"], 1.0)

    def test_all_recall_outcomes_are_unit_evidence_with_context_local_usefulness(self):
        self.index.prepare(self.records)
        self.index.observe("beta", torch.tensor([0., 1.]), "helpful", weight=100)
        self.index.observe("beta", torch.tensor([0., 1.]), "irrelevant", weight=0)
        self.index.observe("beta", torch.tensor([0., 1.]), "contradicted", weight=0.5)
        self.index.observe("beta", torch.tensor([0., 1.]), "unknown")
        self.index.observe("beta", torch.tensor([0., 1.]), "empty")
        self.assertEqual(sum(item["count"] for item in self.index.snapshot()["evidence"]), 5)
        self.assertEqual(self.index.query(torch.tensor([0., 1.]))[0]["evidence_weight"], 1)
        self.index.observe("alpha", torch.tensor([1., 0.]), "helpful", channel="reader")
        speech = self.index.query(torch.tensor([1., 0.]), channel="speech")[0]
        reader = self.index.query(torch.tensor([1., 0.]), channel="reader")[0]
        self.assertEqual(speech["evidence_weight"], 1)
        self.assertEqual(reader["evidence_weight"], 1.5)
        self.assertEqual(self.index._evidence_weight("alpha", torch.tensor([0., 1.]), "reader"), 1)

    def test_partial_indexing_is_reported_without_calling_missing_memory_irrelevant(self):
        index = SalientMemoryIndex(None, encoder=self.encoder, max_new_vectors=0)
        stats = index.prepare(self.records)
        self.assertEqual(stats["unindexed_records"], 2)
        self.assertEqual(stats["indexed_spans"], 0)
        self.assertFalse(stats["complete"])
        self.assertEqual(index.query(torch.tensor([1., 0.])), [])
        self.assertEqual(index.snapshot()["evidence"], [])

    def test_encoding_and_resident_vector_cost_are_bounded(self):
        index = SalientMemoryIndex(None, encoder=self.encoder, span_chars=10,
                                  max_spans=4, max_new_vectors=3, max_cached_vectors=2)
        stats = index.prepare([record(str(i), "ALPHA " * 20 + "BETA") for i in range(10)])
        self.assertEqual(len(self.encoder.calls), 3)
        self.assertEqual(stats["newly_encoded"], 3)
        self.assertEqual(len(index._cache), 2)
        before = len(self.encoder.calls)
        index.rank(None, torch.tensor([1., 0.]), allow_encode=False)
        self.assertEqual(len(self.encoder.calls), before)

    def test_rotation_measures_additional_source_regions_on_later_passes(self):
        source = "ALPHA " * 100 + "BETA"
        self.index.prepare([record("long", source)])
        before = set(self.index._cache)
        self.index.prepare([record("long", source)])
        after = set(self.index._cache)
        self.assertTrue(after > before)
        self.assertEqual(self.index.query(torch.tensor([0., 1.]))[0]["stimulation"], 1)

    def test_source_edit_invalidates_old_source_measurements(self):
        self.index.prepare([record("edited", "ALPHA")])
        self.index.prepare([record("edited", "BETA")])
        candidates = self.index.query(torch.tensor([1., 0.]))
        self.assertEqual(candidates[0]["stimulation"], 0)
        self.assertEqual(candidates[0]["span"]["text"], "BETA")

    def test_invalid_state_does_not_encode_or_create_evidence(self):
        for state in [None, torch.zeros(2), torch.tensor([float("nan"), 1.]), torch.ones(1, 2)]:
            self.assertEqual(self.index.rank(self.records, state), [])
            self.assertFalse(self.index.observe("a", state))
        self.assertEqual(self.encoder.calls, [])
        with self.assertRaises(ValueError):
            self.index.observe("a", torch.ones(2), outcome="fabricated")

    def test_sidecar_is_compact_namespaced_evidence_and_reload_preserves_unit_counts(self):
        self.index.namespace = "agent/one"
        self.index.observe("beta", torch.tensor([0., 1.]), "helpful", weight=999)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recall.json"
            self.index.save(path)
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("BETA", text)
            matching = SalientMemoryIndex(None, encoder=Encoder(), namespace="agent/one")
            self.assertTrue(matching.load(path))
            self.assertEqual(matching.snapshot(), self.index.snapshot())
            other = SalientMemoryIndex(None, encoder=Encoder(), namespace="agent/two")
            self.assertFalse(other.load(path))
            self.assertEqual(other.snapshot()["evidence"], [])

    def test_real_tiny_llama_source_encoding_is_raw_and_cached_query_is_observational(self):
        from transformers import LlamaConfig, LlamaForCausalLM
        torch.manual_seed(12)
        language_model = LlamaForCausalLM(LlamaConfig(vocab_size=16, hidden_size=8,
            intermediate_size=16, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=64)).eval()
        model = SimpleNamespace(model=language_model, tok=RawTokenizer(), device=torch.device("cpu"))
        index = SalientMemoryIndex(model)
        parameters = [parameter.detach().clone() for parameter in language_model.parameters()]
        stats = index.prepare(self.records)
        self.assertEqual(stats["newly_encoded"], 2)
        query = next(iter(index._cache.values())).clone()
        candidates = index.query(query)
        self.assertEqual(candidates[0]["stimulation"], 1.0)
        self.assertEqual(len(model.tok.calls), 2)
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(parameters, language_model.parameters())))
        self.assertEqual(len(language_model.model.norm._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
