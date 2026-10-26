"""State-conditioned source selection without semantic prompts or steering."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from invariants.document_attention import IntrinsicDocumentReader
from invariants.document_engine import document_layout_sha256, select_next_chunk


class RawTokenizer:
    def __init__(self):
        self.calls = []

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        self.calls.append((text, return_tensors, add_special_tokens))
        token = 2 if text.startswith("beta") else 1
        return {"input_ids": torch.tensor([[token]]), "attention_mask": torch.ones((1, 1), dtype=torch.long)}

    def apply_chat_template(self, *_args, **_kwargs):
        raise AssertionError("Candidate source text must not use a chat template")


class TinyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(torch.tensor([[0., 0.], [1., 0.], [0., 1.]]))
        self.norm = torch.nn.Identity()
        self.forward_options = []

    def forward(self, input_ids, attention_mask=None, **kwargs):
        self.forward_options.append(kwargs)
        return SimpleNamespace(last_hidden_state=self.norm(self.embedding(input_ids)))


def model_bundle():
    backbone = TinyBackbone()
    language_model = torch.nn.Module()
    language_model.model = backbone
    return SimpleNamespace(model=language_model, tok=RawTokenizer(), device=torch.device("cpu"))


def document(chunks=None, pass_id="pass-one"):
    chunks = chunks or ["alpha source evidence", "beta source evidence"]
    return {
        "sha256": "same-document", "chunks": chunks, "chunk_count": len(chunks),
        "layout_sha256": document_layout_sha256(chunks), "reading_pass_id": pass_id,
        "read": set(),
    }


class IntrinsicDocumentReaderTests(unittest.TestCase):
    def setUp(self):
        self.model = model_bundle()
        self.reader = IntrinsicDocumentReader(self.model)

    def capture(self, token):
        with self.reader.capture_generation():
            return self.model.model.model(input_ids=torch.tensor([[token]])).last_hidden_state

    def test_cold_start_returns_empty_scores_for_order_fallback(self):
        session = document()
        self.assertEqual(self.reader.rank([session]), ({}, 0, 0))
        self.assertEqual(select_next_chunk([session], probe_scores={})["chunk_index"], 0)
        self.assertEqual(self.model.tok.calls, [])

    def test_capture_is_observational_and_commits_on_success(self):
        before = self.model.model.model(input_ids=torch.tensor([[1]])).last_hidden_state
        inside = self.capture(1)
        self.assertTrue(torch.equal(before, inside))
        self.assertTrue(torch.equal(self.reader.latest_state, torch.tensor([1., 0.])))
        self.assertEqual(len(self.model.model.model.norm._forward_hooks), 0)
        self.assertFalse(self.reader.latest_state.requires_grad)

    def test_candidate_encoding_uses_only_raw_source_text(self):
        self.capture(1)
        state = self.reader.latest_state.clone()
        session = document()
        scores, new, candidates = self.reader.rank([session])
        self.assertEqual((new, candidates), (2, 2))
        self.assertGreater(scores[(0, 0)], scores[(0, 1)])
        self.assertEqual([call[0] for call in self.model.tok.calls], session["chunks"])
        self.assertTrue(all(call[2] is False for call in self.model.tok.calls))
        self.assertTrue(all(options.get("output_hidden_states") is False for options in self.model.model.model.forward_options[1:]))
        self.assertTrue(torch.equal(self.reader.latest_state, state))
        self.assertEqual(session["read"], set())

    def test_new_context_reranks_cached_vectors(self):
        session = document()
        self.capture(1)
        first, _, _ = self.reader.rank([session])
        self.capture(2)
        second, new, _ = self.reader.rank([session])
        self.assertGreater(first[(0, 0)], first[(0, 1)])
        self.assertGreater(second[(0, 1)], second[(0, 0)])
        self.assertEqual(new, 0)
        self.assertEqual(len(self.model.tok.calls), 2)

    def test_new_pass_reencodes_and_discards_old_pass_cache(self):
        session = document()
        self.capture(1)
        self.reader.rank([session])
        session["reading_pass_id"] = "new-context-pass"
        _, new, _ = self.reader.rank([session])
        self.assertEqual(new, 2)
        self.assertEqual(len(self.reader._cache), 2)
        self.assertTrue(all(key[3] == "new-context-pass" for key in self.reader._cache))

    def test_failed_capture_retains_state_and_removes_hook(self):
        self.capture(1)
        previous = self.reader.latest_state.clone()
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            with self.reader.capture_generation():
                self.model.model.model(input_ids=torch.tensor([[2]]))
                raise RuntimeError("generation failed")
        self.assertTrue(torch.equal(self.reader.latest_state, previous))
        self.assertEqual(len(self.model.model.model.norm._forward_hooks), 0)
        self.assertFalse(self.reader._capturing)

    def test_missing_invalid_or_active_capture_returns_no_ranking(self):
        for invalid in (None, torch.zeros(2), torch.tensor([float("nan"), 1.]), torch.ones(1, 2)):
            self.reader.latest_state = invalid
            self.assertEqual(self.reader.rank([document()]), ({}, 0, 0))
        self.capture(1)
        with self.reader.capture_generation():
            self.assertEqual(self.reader.rank([document()]), ({}, 0, 0))
        self.assertEqual(self.model.tok.calls, [])

    def test_cache_scan_and_prefix_are_bounded(self):
        reader = IntrinsicDocumentReader(self.model, max_cached_vectors=2, max_chars=6)
        with reader.capture_generation():
            self.model.model.model(input_ids=torch.tensor([[1]]))
        session = document(["alpha long " + str(i) for i in range(6)])
        scores, new, count = reader.rank([session], scan=4)
        self.assertEqual((len(scores), new, count), (4, 4, 4))
        self.assertEqual(len(reader._cache), 2)
        self.assertTrue(all(len(call[0]) == 6 for call in self.model.tok.calls))

    def test_read_chunks_are_not_ranked_and_layout_changes_reencode(self):
        self.capture(1)
        session = document()
        session["read"] = {0}
        scores, _, count = self.reader.rank([session])
        self.assertEqual(set(scores), {(0, 1)})
        self.assertEqual(count, 1)
        session["chunks"][1] = "alpha changed boundary"
        session["layout_sha256"] = document_layout_sha256(session["chunks"])
        scores, new, _ = self.reader.rank([session])
        self.assertEqual(new, 1)
        self.assertAlmostEqual(scores[(0, 1)], 1.0)

    def test_reader_and_speech_states_have_a_descriptive_difference(self):
        self.assertIsNone(self.reader.channel_difference())
        self.capture(1)  # Default remains speech.
        self.assertEqual(self.reader.current_channel, "speech")
        self.assertIsNone(self.reader.channel_difference())
        with self.reader.capture_generation(channel="reader"):
            self.model.model.model(input_ids=torch.tensor([[2]]))
        self.assertEqual(self.reader.current_channel, "reader")
        self.assertTrue(torch.equal(self.reader.latest_state, torch.tensor([0., 1.])))
        difference = self.reader.channel_difference()
        self.assertTrue(torch.equal(difference, torch.tensor([-1., 1.])))
        self.assertEqual(difference.device.type, "cpu")
        self.assertFalse(difference.requires_grad)
        difference.zero_()
        self.assertTrue(torch.equal(self.reader.channel_difference(), torch.tensor([-1., 1.])))

    def test_failed_reader_capture_preserves_channel_history(self):
        self.capture(1)
        with self.assertRaisesRegex(RuntimeError, "reader failed"):
            with self.reader.capture_generation(channel="reader"):
                self.model.model.model(input_ids=torch.tensor([[2]]))
                raise RuntimeError("reader failed")
        self.assertEqual(self.reader.current_channel, "speech")
        self.assertEqual(set(self.reader.channel_states), {"speech"})
        self.assertIsNone(self.reader.channel_difference())

    def test_missing_capture_does_not_fabricate_a_channel_state(self):
        with self.reader.capture_generation(channel="reader"):
            pass
        self.assertEqual(self.reader.channel_states, {})
        self.assertIsNone(self.reader.current_channel)
        with self.assertRaisesRegex(ValueError, "channel"):
            with self.reader.capture_generation(channel="invented"):
                pass
        self.assertEqual(len(self.model.model.model.norm._forward_hooks), 0)

    def test_real_tiny_llama_capture_and_raw_backbone_ranking(self):
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(7)
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=31,
        )).eval()
        bundle = SimpleNamespace(model=model, tok=RawTokenizer(), device=torch.device("cpu"))
        reader = IntrinsicDocumentReader(bundle)
        observed = []
        handle = model.model.norm.register_forward_hook(lambda _m, _a, out: observed.append(out[0, -1, :].detach().clone()))
        try:
            with reader.capture_generation():
                model.generate(input_ids=torch.tensor([[1, 4, 5]]), max_new_tokens=3, do_sample=False)
        finally:
            handle.remove()
        self.assertTrue(torch.allclose(reader.latest_state, observed[-1]))
        state = reader.latest_state.clone()
        scores, new, count = reader.rank([document()])
        self.assertEqual((new, count), (2, 2))
        self.assertEqual(set(scores), {(0, 0), (0, 1)})
        self.assertTrue(all(-1.0 <= score <= 1.0 for score in scores.values()))
        self.assertTrue(torch.equal(reader.latest_state, state))
        self.assertEqual(len(model.model.norm._forward_hooks), 0)


if __name__ == "__main__":
    unittest.main()
