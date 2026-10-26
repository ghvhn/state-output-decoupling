import copy
import json

import pytest
import torch

from tda.glm_capture_archive import (CaptureArchiveWriter, load_archive_manifest,
    load_archive_entry, load_output_head, reconstruct_logits, verify_spotchecks)


@pytest.fixture
def identity():
    return {"model_id": "tiny_teacher", "revision": "a" * 40,
            "hidden_size": 4, "vocab_size": 7, "num_layers": 3,
            "source_precision": "bfloat16", "tokenizer_sha256": "b" * 64,
            "chat_template_sha256": "c" * 64}


def chunk(number=0, split="calibration", trajectory=None):
    torch.manual_seed(88 + number)
    states = torch.randn(2, 4).bfloat16()
    rows = [{"id": f"example-{number}:{i}", "trajectory_id": trajectory or f"example-{number}",
             "environment_id": "arithmetic", "split": split,
             "token_position": i + 3, "next_token_id": i + 1} for i in range(2)]
    tensors = {"final_hidden": states,
               "positions": torch.tensor([3, 4]), "next_token_ids": torch.tensor([1, 2]),
               "layer_1_expert_input": states.clone()}
    metadata = {"rows": rows, "state_sites": {"final_hidden": "post_final_norm_lm_head_input",
                "layer_1_expert_input": "post_attention_norm_expert_gate_up_input"}}
    return tensors, metadata


def make_writer(path, identity, **kwargs):
    return CaptureArchiveWriter(path, identity, origin="synthetic_fixture", **kwargs)


def complete(path, identity):
    writer = make_writer(path, identity)
    torch.manual_seed(7)
    head = torch.randn(7, 4).bfloat16()
    writer.write_head_block(0, head[:3].clone())
    writer.write_head_block(3, head[3:].clone())
    tensors, metadata = chunk()
    tensors["spotcheck_logits"] = reconstruct_logits(tensors["final_hidden"], head).clone()
    tensors["spotcheck_rows"] = torch.tensor([0, 1])
    writer.write_chunk("first", tensors, metadata)
    return writer.finalize(), head, tensors


def test_exact_native_head_reconstruction_and_spotcheck(tmp_path, identity):
    original, head, tensors = complete(tmp_path / "archive", identity)
    manifest = load_archive_manifest(tmp_path / "archive", expected_identity=identity)
    assert original == manifest
    restored = load_output_head(tmp_path / "archive", manifest)
    assert torch.equal(restored, head)
    entry = next(e for e in manifest["entries"] if e["role"] == "capture")
    captured = load_archive_entry(tmp_path / "archive", entry)
    assert torch.equal(captured["final_hidden"], tensors["final_hidden"])
    report = verify_spotchecks(captured["final_hidden"], restored, captured["spotcheck_logits"], atol=0, rtol=0)
    assert report["passed"] and report["max_absolute_error"] == 0
    wrong = captured["spotcheck_logits"].float() + 1
    assert not verify_spotchecks(captured["final_hidden"], restored, wrong, atol=0, rtol=0)["passed"]


def test_partial_chunks_are_readable_after_stop(tmp_path, identity):
    writer = make_writer(tmp_path, identity)
    tensors, metadata = chunk()
    entry = writer.write_chunk("a", tensors, metadata)
    manifest = load_archive_manifest(tmp_path)
    assert manifest["status"] == "partial"
    assert torch.equal(load_archive_entry(tmp_path, entry)["final_hidden"], tensors["final_hidden"])
    with pytest.raises(ValueError, match="complete"):
        writer.finalize()
    writer.finalize("partial", {"reason": "deadline"})


def test_unobserved_probe_pole_is_never_a_prediction_label(tmp_path, identity):
    tensors, metadata = chunk()
    for row in metadata["rows"]:
        row.update(kind="probe_framing", probe_id="p_native", polarity=1,
                   next_token_observed=False, next_token_id=-1)
    tensors["next_token_ids"].fill_(-1)
    writer = make_writer(tmp_path, identity)
    entry = writer.write_chunk("probe", tensors, metadata)
    restored = load_archive_entry(tmp_path, entry, identity=identity)
    assert restored["next_token_ids"].tolist() == [-1, -1]
    metadata["rows"][0]["kind"] = "conversation"
    with pytest.raises(ValueError, match="unobserved"):
        writer.write_chunk("invalid", tensors, metadata)


@pytest.mark.parametrize("mode", ["missing", "shape", "duplicate"])
def test_router_capture_preserves_paired_unique_choices(tmp_path, identity, mode):
    tensors, metadata = chunk()
    tensors["layer_1_router_ids"] = torch.tensor([[0, 1], [1, 2]])
    tensors["layer_1_router_weights"] = torch.ones(2, 2)
    if mode == "missing":
        del tensors["layer_1_router_weights"]
    elif mode == "shape":
        tensors["layer_1_router_weights"] = torch.ones(2, 1)
    else:
        tensors["layer_1_router_ids"][0, 1] = 0
    with pytest.raises(ValueError, match="matching pairs|duplicate an expert"):
        make_writer(tmp_path, identity).write_chunk("invalid", tensors, metadata)


def test_separate_trajectories_cannot_leak_one_conversation(tmp_path, identity):
    writer = make_writer(tmp_path, identity)
    tensors, metadata = chunk()
    for row in metadata["rows"]:
        row["conversation_id"] = "same-conversation"
    writer.write_chunk("first", tensors, metadata)
    other, info = chunk(1, split="heldout")
    for row in info["rows"]:
        row["conversation_id"] = "same-conversation"
    with pytest.raises(ValueError, match="leak"):
        writer.write_chunk("other", other, info)


def test_reader_revalidates_row_timeline_not_only_tensor_shapes(tmp_path, identity):
    _, _, _ = complete(tmp_path, identity)
    manifest = load_archive_manifest(tmp_path)
    entry = next(e for e in manifest["entries"] if e["role"] == "capture")
    entry["metadata"]["rows"][0]["token_position"] += 1
    with pytest.raises(ValueError, match="disagrees"):
        load_archive_entry(tmp_path, entry, identity=identity)


def test_cross_context_pair_is_rejected(tmp_path, identity):
    tensors, metadata = chunk()
    metadata["rows"][1]["trajectory_id"] = "different"
    metadata["pairs"] = [[0, 1]]
    with pytest.raises(ValueError, match="one measured trajectory"):
        make_writer(tmp_path, identity).write_chunk("invalid", tensors, metadata)


@pytest.mark.parametrize("field,value", [("revision", "main"), ("hidden_size", True),
    ("tokenizer_sha256", "bad"), ("chat_template_sha256", None)])
def test_source_identity_cannot_drift(tmp_path, identity, field, value):
    wrong = {**identity, field: value}
    with pytest.raises(ValueError):
        make_writer(tmp_path, wrong)


def test_reader_rejects_other_teacher(tmp_path, identity):
    complete(tmp_path, identity)
    with pytest.raises(ValueError, match="scope"):
        load_archive_manifest(tmp_path, expected_identity={**identity, "revision": "d" * 40})


def test_existing_capture_is_not_overwritten(tmp_path, identity):
    writer = make_writer(tmp_path, identity)
    with pytest.raises(ValueError, match="overwrite"):
        make_writer(tmp_path, identity)
    assert (tmp_path / "manifest.json").exists()
    writer.finalize("failed")


@pytest.mark.parametrize("bad", ["../outside", "/abs", "a/b", ""])
def test_unsafe_chunk_paths_refused(tmp_path, identity, bad):
    writer = make_writer(tmp_path, identity)
    with pytest.raises(ValueError, match="chunk ID"):
        writer.write_chunk(bad, *chunk())


def test_byte_budget_refuses_before_serialization(tmp_path, identity, monkeypatch):
    writer = make_writer(tmp_path, identity, max_chunk_bytes=8)
    called = []
    monkeypatch.setattr(torch, "save", lambda *args, **kwargs: called.append(True))
    with pytest.raises(ValueError, match="byte bound"):
        writer.write_chunk("a", *chunk())
    assert not called and not list(tmp_path.glob("*.pt"))


def test_serialized_overhead_counts_against_total(tmp_path, identity):
    writer = make_writer(tmp_path, identity, max_total_bytes=100, max_chunk_bytes=10_000)
    with pytest.raises(ValueError, match="byte budget"):
        writer.write_head_block(0, torch.ones(2, 4).bfloat16())
    assert not list(tmp_path.glob("*.pt")) and not list(tmp_path.glob(".capture-*"))


@pytest.mark.parametrize("mutation", ["offset", "backing", "nan", "width", "timeline", "site", "target"])
def test_invalid_native_evidence_refused(tmp_path, identity, mutation):
    writer = make_writer(tmp_path, identity)
    tensors, metadata = chunk()
    if mutation == "offset":
        tensors["final_hidden"] = torch.zeros(3, 4).bfloat16()[1:]
    elif mutation == "backing":
        tensors["final_hidden"] = torch.zeros(3, 4).bfloat16()[:2]
    elif mutation == "nan":
        tensors["final_hidden"][0, 0] = float("nan")
    elif mutation == "width":
        tensors["final_hidden"] = torch.zeros(2, 3).bfloat16()
    elif mutation == "timeline":
        tensors["positions"][0] = 5
    elif mutation == "site":
        metadata["state_sites"]["final_hidden"] = "before_norm"
    else:
        metadata["rows"][0]["next_token_id"] = 7
    with pytest.raises(ValueError):
        writer.write_chunk("bad", tensors, metadata)
    assert not list(tmp_path.glob("*.pt"))


def test_duplicate_rows_and_split_leakage_refused(tmp_path, identity):
    writer = make_writer(tmp_path, identity)
    writer.write_chunk("first", *chunk())
    with pytest.raises(ValueError, match="duplicate"):
        writer.write_chunk("second", *chunk())
    with pytest.raises(ValueError, match="leak"):
        writer.write_chunk("third", *chunk(1, "heldout", "example-0"))


@pytest.mark.parametrize("start,shape", [(1, (2, 4)), (0, (8, 4)), (0, (2, 3))])
def test_head_order_and_shape_enforced(tmp_path, identity, start, shape):
    writer = make_writer(tmp_path, identity)
    with pytest.raises(ValueError, match="head blocks"):
        writer.write_head_block(start, torch.zeros(shape).bfloat16())


def test_head_dtype_cannot_change(tmp_path, identity):
    writer = make_writer(tmp_path, identity)
    writer.write_head_block(0, torch.zeros(2, 4).bfloat16())
    with pytest.raises(ValueError, match="dtype"):
        writer.write_head_block(2, torch.zeros(2, 4))


def test_finalized_capture_cannot_mutate(tmp_path, identity):
    writer = make_writer(tmp_path, identity)
    writer.finalize("partial")
    with pytest.raises(ValueError, match="finalized"):
        writer.write_chunk("later", *chunk())


def test_checksum_rejects_modified_payload(tmp_path, identity):
    manifest, _, _ = complete(tmp_path, identity)
    entry = manifest["entries"][0]
    path = tmp_path / entry["file"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        load_archive_entry(tmp_path, entry)


@pytest.mark.parametrize("mutation", ["outside", "duplicate", "total", "allocation", "gap"])
def test_manifest_cannot_hide_scope_or_allocation_damage(tmp_path, identity, mutation):
    manifest, _, _ = complete(tmp_path, identity)
    if mutation == "outside":
        manifest["entries"][0]["file"] = "../head-000000000.pt"
    elif mutation == "duplicate":
        manifest["entries"].append(copy.deepcopy(manifest["entries"][0]))
    elif mutation == "total":
        manifest["artifact_bytes"] += 1
    elif mutation == "allocation":
        manifest["entries"][0]["tensors"]["weight"]["shape"][0] = 100_000_000
    else:
        manifest["entries"][1]["metadata"]["start_row"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_archive_manifest(tmp_path)


def test_head_and_readout_allocation_caps(tmp_path, identity):
    manifest, head, tensors = complete(tmp_path, identity)
    with pytest.raises(ValueError, match="allocation"):
        load_output_head(tmp_path, manifest, max_head_bytes=8)
    with pytest.raises(ValueError, match="working budget"):
        reconstruct_logits(tensors["final_hidden"], head, max_working_bytes=8)


def test_readout_does_not_silently_change_dtype(tmp_path, identity):
    _, head, tensors = complete(tmp_path, identity)
    with pytest.raises(ValueError, match="dtype"):
        reconstruct_logits(tensors["final_hidden"].float(), head)
