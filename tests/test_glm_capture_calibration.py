import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity
from tda.glm_capture_archive import CaptureArchiveWriter
from tda.glm_capture_calibration import native_probe_bank, gate_calibration_records


def evidence(tmp_path):
    teacher = {"model_id": "tiny", "revision": "a" * 40, "hidden_size": 3,
               "vocab_size": 5, "num_layers": 2, "source_precision": "bfloat16",
               "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64}
    writer = CaptureArchiveWriter(tmp_path, teacher, origin="synthetic_fixture")
    rows = [{"id": f"probe:{i}", "trajectory_id": f"probe-{i}", "environment_id": "seed",
             "kind": "probe_framing", "split": "calibration", "probe_id": "p_test", "polarity": 1 if i == 0 else -1,
             "token_position": 3, "next_token_id": -1, "next_token_observed": False} for i in range(2)]
    states = torch.tensor([[2, 1, 1], [0, 1, 1]], dtype=torch.bfloat16)
    sites = {"final_hidden": "post_final_norm_lm_head_input", "layer_1_expert_input": "post_attention_norm_expert_gate_up_input"}
    writer.write_chunk("seed", {"final_hidden": states, "layer_1_expert_input": states.clone(),
        "positions": torch.tensor([3, 3]), "next_token_ids": torch.tensor([-1, -1])}, {"rows": rows, "state_sites": sites})
    for split in ("calibration", "heldout"):
        rows = [{"id": f"{split}:{i}", "trajectory_id": split, "environment_id": "arithmetic",
                 "kind": "conversation", "split": split, "token_position": i, "next_token_id": i,
                 "stimulation_weight": 0.5} for i in range(2)]
        writer.write_chunk(split, {"final_hidden": states.clone(), "layer_1_expert_input": states.clone(),
            "positions": torch.arange(2), "next_token_ids": torch.arange(2)}, {"rows": rows, "pairs": [[0, 1]], "state_sites": sites})
    writer.finalize("partial")
    return ExpertGroupIdentity("tiny", "a" * 40, 1, "gate_proj", "separate row-major", ("0", "1"), (2, 3), "bfloat16")


def test_native_probe_axes_fit_only_explicit_seed_poles(tmp_path):
    evidence(tmp_path)
    bank = native_probe_bank(tmp_path)
    assert bank["probe_ids"] == ["p_test"]
    assert torch.equal(bank["directions"], torch.tensor([[1., 0, 0]]))
    assert torch.equal(bank["centers"], torch.tensor([[1., 1, 1]]))
    assert bank["origin"] == "synthetic_fixture"
    assert not bank["semantic_meanings_validated"] and not bank["heldout_used_to_fit_axes"]
    other = native_probe_bank(tmp_path, tensor_name="layer_1_expert_input")
    assert other["scope_sha256"] != bank["scope_sha256"]


def test_gate_records_preserve_timeline_environment_and_stimulation(tmp_path):
    identity = evidence(tmp_path)
    scope, train, pairs = gate_calibration_records(tmp_path, identity)
    _, heldout, test_pairs = gate_calibration_records(tmp_path, identity, split="heldout")
    assert len(train) == len(heldout) == 2 and len(pairs) == len(test_pairs) == 1
    assert not {r.record_id for r in train} & {r.record_id for r in heldout}
    assert all(r.scope == scope and r.stimulation_weight == 0.5 and r.prediction_correct is None for r in train)
    assert pairs[0].previous_id == train[0].record_id and pairs[0].next_id == train[1].record_id
    assert scope.state_site == "post_attention_norm_expert_gate_up_input"


def test_down_projection_and_other_teacher_cannot_use_shared_gate_inputs(tmp_path):
    identity = evidence(tmp_path)
    down = ExpertGroupIdentity("tiny", "a" * 40, 1, "down_proj", "row-major", ("0", "1"), (2, 3), "bfloat16")
    with pytest.raises(ValueError, match="down projection"):
        gate_calibration_records(tmp_path, down)
    other = ExpertGroupIdentity("other", "a" * 40, 1, identity.projection, identity.layout, identity.expert_ids, identity.matrix_shape, identity.source_dtype)
    with pytest.raises(ValueError, match="teacher differs"):
        gate_calibration_records(tmp_path, other)


@pytest.mark.parametrize("limits", [{"max_records": 1}, {"max_pairs": True}, {"max_state_bytes": 1}])
def test_calibration_evidence_bounds_are_checked(tmp_path, limits):
    identity = evidence(tmp_path)
    with pytest.raises((ValueError, MemoryError)):
        gate_calibration_records(tmp_path, identity, **limits)


def test_probe_calibration_cannot_mistake_router_ids_for_geometry(tmp_path):
    evidence(tmp_path)
    with pytest.raises(ValueError, match="hidden-state site"):
        native_probe_bank(tmp_path, tensor_name="layer_1_router_ids")
