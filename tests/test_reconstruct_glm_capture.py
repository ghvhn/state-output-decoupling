from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tda.glm_capture_archive import CaptureArchiveWriter, reconstruct_logits

SPEC = importlib.util.spec_from_file_location("glm_readout", Path(__file__).resolve().parents[1] / "scripts/reconstruct_glm_capture.py")
readout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(readout)


def archive(root, *, anchors=True, corrupted_anchor=False):
    identity = {"model_id": "tiny", "revision": "a" * 40, "hidden_size": 3,
                "vocab_size": 5, "num_layers": 2, "source_precision": "bfloat16",
                "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64}
    head = torch.eye(5, 3).bfloat16()
    states = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=torch.bfloat16)
    rows = [{"id": f"example:{i}", "trajectory_id": "example", "environment_id": "arithmetic",
             "split": "heldout", "token_position": i, "next_token_id": i + 1} for i in range(3)]
    # The final framing has no observed next token. The other two labels are
    # deliberately wrong predictions, which still belong in the evidence.
    rows[-1].update(kind="probe_framing", probe_id="p_test", polarity=-1,
                    next_token_observed=False, next_token_id=-1)
    tensors = {"final_hidden": states, "positions": torch.arange(3), "next_token_ids": torch.tensor([1, 2, -1])}
    if anchors:
        tensors["spotcheck_rows"] = torch.tensor([0, 2])
        tensors["spotcheck_logits"] = reconstruct_logits(states[[0, 2]], head).clone()
        if corrupted_anchor:
            tensors["spotcheck_logits"] += 1
    writer = CaptureArchiveWriter(root, identity, origin="synthetic_fixture")
    writer.write_head_block(0, head)
    writer.write_chunk("first", tensors, {"rows": rows, "state_sites": {"final_hidden": "post_final_norm_lm_head_input"}})
    writer.finalize()
    return states, head


def test_local_readout_roundtrip_retains_wrong_predictions_and_probe_sentinel(tmp_path):
    source, output = tmp_path / "archive", tmp_path / "readout"
    states, head = archive(source)
    report = readout.reconstruct_archive(source, out=output, batch_rows=2, atol=0, rtol=0)
    assert report["readout_qualified"] and report["status"] == "complete"
    assert report["positions"] == 3 and report["observed_next_tokens"] == 2
    assert report["unobserved_probe_positions"] == 1
    assert report["observed_target_metrics"]["top1_accuracy"] == 0
    assert report["source_archive_origin"] == "synthetic_fixture"
    assert not report["modified_decoder_validated"]
    restored = [torch.load(output / entry["file"], weights_only=True) for entry in report["entries"]]
    assert torch.equal(torch.cat([item["logits"] for item in restored]), reconstruct_logits(states, head))
    assert torch.cat([item["next_token_ids"] for item in restored]).tolist() == [1, 2, -1]
    assert json.loads((output / "report.json").read_text())["exported_bytes"] > 0


@pytest.mark.parametrize("mode", ["missing", "wrong"])
def test_unverified_readout_cannot_be_exported(tmp_path, mode):
    archive(tmp_path / "source", anchors=mode != "missing", corrupted_anchor=mode == "wrong")
    report = readout.reconstruct_archive(tmp_path / "source")
    assert not report["readout_qualified"]
    with pytest.raises(ValueError, match="spot checks"):
        readout.reconstruct_archive(tmp_path / "source", out=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_readout_budget_fails_before_creating_output(tmp_path):
    archive(tmp_path / "source")
    with pytest.raises(ValueError, match="byte budget"):
        readout.reconstruct_archive(tmp_path / "source", out=tmp_path / "out", max_export_bytes=1)
    assert not (tmp_path / "out").exists()


def test_readout_never_overwrites_captured_evidence(tmp_path):
    archive(tmp_path / "source")
    with pytest.raises(ValueError, match="separate"):
        readout.reconstruct_archive(tmp_path / "source", out=tmp_path / "source/nested")


def test_interrupted_export_retains_its_completed_artifacts(tmp_path, monkeypatch):
    archive(tmp_path / "source")
    save = torch.save
    count = [0]
    def interrupted(*args, **kwargs):
        count[0] += 1
        if count[0] == 2:
            raise OSError("disk interrupted")
        return save(*args, **kwargs)
    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        readout.reconstruct_archive(tmp_path / "source", out=tmp_path / "out", batch_rows=2)
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "partial" and len(report["entries"]) == 1
    assert (tmp_path / "out" / report["entries"][0]["file"]).is_file()
    assert not list((tmp_path / "out").glob("*.incomplete"))
