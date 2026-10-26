"""Explicit stopping controls remain separate from automatic reading coverage."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import interactive_phenomenality as shell


class ReaderStopTests(unittest.TestCase):
    def setUp(self):
        self.tuner = SimpleNamespace(
            triggers={"probe_interest": SimpleNamespace(value=0.4)},
            get=lambda key, default: {"conversation_productive": 0.5,
                                      "reading_settled_streak": 2}.get(key, default),
        )

    def test_default_full_pass_cannot_stop_on_curiosity_or_confidence_scores(self):
        control, error = shell.parse_doc_read_request("read", [])
        self.assertIsNone(error)
        for sense in (None, -1.0, 1.0):
            self.assertFalse(shell.document_reading_should_stop(control, self.tuner, sense, 100.0))

    def test_explicit_settled_stop_resets_streak_after_unsettled_read(self):
        control, _ = shell.parse_doc_read_request("read satisfied", [])
        self.assertFalse(shell.document_reading_should_stop(control, self.tuner, 0.8))
        self.assertFalse(shell.document_reading_should_stop(control, self.tuner, 0.1))
        self.assertFalse(shell.document_reading_should_stop(control, self.tuner, 0.8))
        self.assertTrue(shell.document_reading_should_stop(control, self.tuner, 0.8))

    def test_probe_stop_requires_an_observed_signal_from_requested_probe(self):
        control, _ = shell.parse_doc_read_request("read interest", ["interest"])
        self.assertFalse(shell.document_reading_should_stop(control, self.tuner))
        self.assertFalse(shell.document_reading_should_stop(control, self.tuner, probe_signal=0.39))
        self.assertTrue(shell.document_reading_should_stop(control, self.tuner, probe_signal=0.4))

    def test_shell_capture_keeps_reader_and_speech_channels_distinct(self):
        from test_document_attention import model_bundle
        from invariants.document_attention import IntrinsicDocumentReader

        bundle = model_bundle()
        reader = IntrinsicDocumentReader(bundle)
        def engine(*_args, generation_channel, **_kwargs):
            token = 2 if generation_channel == "reader" else 1
            bundle.model.model(input_ids=torch.tensor([[token]]))
            return generation_channel
        with patch.object(shell, "_engine_generate", side_effect=engine), patch.object(shell, "_ACTIVE_PROBES", {}):
            shell.generate_agentic_text(bundle, document_reader=reader, generation_channel="speech")
            shell.generate_agentic_text(bundle, document_reader=reader, generation_channel="reader")
        self.assertEqual(reader.current_channel, "reader")
        self.assertEqual(set(reader.channel_states), {"reader", "speech"})
        self.assertTrue(torch.equal(reader.channel_difference(), torch.tensor([-1., 1.])))


if __name__ == '__main__':
    unittest.main()
