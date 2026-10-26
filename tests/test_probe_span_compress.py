"""Synthetic CPU controls for a static linear proxy, never semantic evidence."""
import copy
import hashlib
import json

import pytest
import torch

from tda.expert_compress import (ExpertGroupIdentity, TensorExpertReader, fit_expert_group,
    save_expert_compression, load_expert_compression, reconstruct_expert)
from tda.glm_capture_archive import CaptureArchiveWriter
from tda.glm_capture_calibration import native_probe_bank_fingerprint, native_probe_bank
from tda.probe_span_compress import fit_probe_span_group, evaluate_probe_span_group


class CountingReader(TensorExpertReader):
    def __init__(self, identity, weights):
        super().__init__(identity, weights)
        self.calls = []

    def read_columns(self, start, stop, expert_ids=None):
        self.calls.append((start, stop))
        return super().read_columns(start, stop, expert_ids)


def bank_for(identity, directions):
    directions = torch.tensor(directions, dtype=torch.float32)
    teacher = {"model_id": identity.model_id, "revision": identity.revision,
        "hidden_size": identity.matrix_shape[1], "vocab_size": 5, "num_layers": 4,
        "source_precision": identity.source_dtype, "tokenizer_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64}
    n = len(directions)
    bank = {"format": "glm-native-probe-bank-v1",
        "scope": {"teacher": teacher, "tensor_name": f"layer_{identity.layer}_expert_input",
            "state_site": "post_attention_norm_expert_gate_up_input"},
        "origin": "synthetic_fixture", "probe_ids": [f"p_{i}" for i in range(n)],
        "directions": directions, "centers": torch.zeros_like(directions),
        "contrast_magnitudes": torch.ones(n),
        "pole_record_ids": [{"negative": f"low-{i}", "positive": f"high-{i}"} for i in range(n)],
        "source_capture_sha256": ["d" * 64], "unpaired_probe_ids": [], "zero_contrast_probe_ids": [],
        "semantic_meanings_validated": False, "heldout_used_to_fit_axes": False}
    return rehash(bank)


def rehash(bank):
    bank["scope_sha256"] = hashlib.sha256(json.dumps(bank["scope"], sort_keys=True).encode()).hexdigest()
    bank["evidence_sha256"] = native_probe_bank_fingerprint(bank)
    return bank


def fixture(width=2):
    identity = ExpertGroupIdentity("synthetic-test", "a" * 40, 3, "gate_proj",
        "row_major_gate_neurons", ("a", "b", "c", "d"), (2, width), "float32")
    a, b, c = (torch.tensor(v) for v in ([-1., -1., 1., 1.], [-1., 1., -1., 1.], [-1., 1., 1., -1.]))
    weights = torch.zeros(4, 2, width)
    weights[:, 0, 0] = 100 * a
    weights[:, 1, 1] = b
    if width == 3:
        weights[:, 0, 2] = 2 * c
    return CountingReader(identity, weights)


def fit(reader, bank, **kwargs):
    return fit_probe_span_group(reader, 1, bank, expected_origin="synthetic_fixture",
        chunk_columns=reader.identity.matrix_shape[1], **kwargs)


def evaluate(result, reader, bank, **kwargs):
    return evaluate_probe_span_group(result, reader, bank, expected_origin="synthetic_fixture",
        chunk_columns=reader.identity.matrix_shape[1], **kwargs)


def test_same_storage_protects_constructed_low_energy_probe_response():
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    control = fit_expert_group(reader, 1, storage_dtype="float32", chunk_columns=2)
    candidate = fit(reader, bank)
    assert candidate.metadata["factor_tensor_bytes"] == control.metadata["factor_tensor_bytes"]
    assert evaluate(control, reader, bank)["span_response_relative_rmse"] > .99
    report = evaluate(candidate, reader, bank)
    assert report["span_response_relative_rmse"] < 1e-6
    assert report["same_probe_bank_used_to_fit"] is True
    assert candidate.metadata["diagnostics"]["relative_frobenius_error"] > control.metadata["diagnostics"]["relative_frobenius_error"]
    objective = candidate.metadata["objective"]
    assert objective["method"] == "probe_response_shared_basis_v1"
    assert objective["measured_temporal_transitions"] == 0
    assert objective["heldout_used_for_fit"] is False
    assert objective["semantic_meanings_validated"] is False
    assert report["temporal_transition_evaluated"] is False
    assert report["native_fp8_kernel_evaluated"] is False
    assert report["evaluation_scope"] == "static_seed_contrast_linear_projection_proxy"


def test_signed_counterfactual_response_is_preserved_across_two_base_inputs():
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    candidate = fit(reader, bank)
    original = reader.weights[0]
    decoded = reconstruct_expert(candidate, "a").reshape(2, 2)
    axis = bank["directions"][0]
    # These constructed differences are algebraic projection tests, not a
    # claim that two prompts followed these transitions in the native model.
    for base_input in (torch.tensor([1., 0.]), torch.tensor([2., 5.])):
        for step in (-2., .5):
            original_change = original @ (base_input + step * axis) - original @ base_input
            decoded_change = decoded @ (base_input + step * axis) - decoded @ base_input
            torch.testing.assert_close(decoded_change, original_change, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(decoded_change, step * (original @ axis), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("directions", [
    [[0, -1, 0], [0, 0, -1]],
    [[0, 2**-.5, 2**-.5], [0, -2**-.5, 2**-.5]],
    [[0, 1, 0], [0, 0, 1], [0, 1, 0]],
])
def test_sign_rotation_and_duplicate_axes_preserve_same_span_fit(directions):
    reader = fixture(3)
    reference = fit(reader, bank_for(reader.identity, [[0, 1, 0], [0, 0, 1]]))
    altered = fit(reader, bank_for(reader.identity, directions))
    torch.testing.assert_close(reference.coefficients @ reference.coefficients.T,
        altered.coefficients @ altered.coefficients.T, atol=1e-6, rtol=1e-6)
    assert reference.metadata["objective"]["probe_span_rank"] == altered.metadata["objective"]["probe_span_rank"] == 2


def test_duplicate_axes_cannot_change_relative_svd_rank_cutoff():
    reader = fixture()
    axes = [[1., 0.], [1., 3e-5]]
    reference = fit(reader, bank_for(reader.identity, axes))
    repeated = fit(reader, bank_for(reader.identity, axes + [[1., 0.]] * 12))
    assert reference.metadata["objective"]["probe_span_rank"] == repeated.metadata["objective"]["probe_span_rank"] == 2
    torch.testing.assert_close(reference.coefficients @ reference.coefficients.T,
        repeated.coefficients @ repeated.coefficients.T)


@pytest.mark.parametrize("options", [{"probe_gain": 0}, {"weight_floor": 1}])
def test_disabled_probe_objective_honestly_returns_exact_pca(options):
    reader = fixture()
    baseline = fit_expert_group(reader, 1, storage_dtype="float32", chunk_columns=2)
    candidate = fit(reader, bank_for(reader.identity, [[0, 1]]), **options)
    for name in ("mean", "basis", "coefficients"):
        torch.testing.assert_close(getattr(candidate, name), getattr(baseline, name), atol=0, rtol=0)
    assert candidate.metadata["objective"]["method"] == "weight_pca_shared_basis_v1"


def test_high_weight_floor_recovers_pca_subspace_and_gain_controls_mixture():
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    baseline = fit_expert_group(reader, 1, storage_dtype="float32", chunk_columns=2)
    floor = fit(reader, bank, weight_floor=.99)
    protected = fit(reader, bank, weight_floor=.99, probe_gain=1e6)
    torch.testing.assert_close(floor.coefficients @ floor.coefficients.T,
        baseline.coefficients @ baseline.coefficients.T)
    assert evaluate(protected, reader, bank)["span_response_relative_rmse"] < 1e-6


def test_zero_source_response_keeps_absolute_error_diagnostic():
    reader = fixture()
    reader.weights[:, :, 1] = 0
    bank = bank_for(reader.identity, [[0, 1]])
    candidate = fit(reader, bank)
    assert candidate.metadata["objective"]["method"] == "weight_pca_shared_basis_v1"
    assert candidate.metadata["probe_span_zero_response_fallback"]["source_response_energy_sq"] == 0
    candidate.basis[:, 1] = 1  # A deliberately altered projection leaks into a zero-source direction.
    report = evaluate(candidate, reader, bank)
    assert report["span_response_source_energy_sq"] == 0
    assert report["span_response_relative_rmse"] is None
    assert report["span_response_absolute_error_sq"] > 0
    assert report["zero_source_axes"] == 1
    assert report["axis_responses"][0]["relative_rmse"] is None
    assert report["axis_responses"][0]["absolute_error_sq"] > 0


@pytest.mark.parametrize("field,value", [("tensor_name", "final_hidden"),
    ("tensor_name", "layer_2_expert_input"), ("state_site", "decoder_block_output"),
    ("model_id", "other"), ("revision", "f" * 40), ("hidden_size", 3),
    ("source_precision", "bfloat16"), ("num_layers", 3)])
def test_wrong_native_scope_rejected_before_reads(field, value):
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    destination = bank["scope"] if field in ("tensor_name", "state_site") else bank["scope"]["teacher"]
    destination[field] = value
    rehash(bank)
    with pytest.raises(ValueError, match="scope"):
        fit(reader, bank)
    assert reader.calls == []


def test_down_projection_rejected_before_reads():
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    source = reader.identity.to_dict()
    source["projection"] = "down_proj"
    reader.identity = ExpertGroupIdentity.from_dict(source)
    with pytest.raises(ValueError, match="down"):
        fit(reader, bank)
    assert reader.calls == []


@pytest.mark.parametrize("field,value", [("origin", "measured_model_state"),
    ("semantic_meanings_validated", True), ("heldout_used_to_fit_axes", True),
    ("scope_sha256", "0" * 64), ("evidence_sha256", "0" * 64),
    ("source_capture_sha256", ["wrong"]), ("pole_record_ids", [{"negative": "same", "positive": "same"}]),
    ("probe_ids", [None])])
def test_evidence_claims_corruption_and_endpoints_rejected_before_reads(field, value):
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    bank[field] = value
    with pytest.raises(ValueError):
        fit(reader, bank)
    assert reader.calls == []


@pytest.mark.parametrize("field,value", [("directions", torch.tensor([[0., float("nan")]])),
    ("directions", torch.tensor([[0., 2.]])), ("centers", torch.zeros(1, 2, dtype=torch.int32)),
    ("contrast_magnitudes", torch.zeros(1))])
def test_malformed_axis_tensors_rejected_before_reads(field, value):
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    bank[field] = value
    with pytest.raises(ValueError):
        fit(reader, bank)
    assert reader.calls == []


def test_direction_corruption_is_detected_even_with_valid_norm():
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    bank["directions"] = torch.tensor([[1., 0.]])
    with pytest.raises(ValueError, match="checksum"):
        fit(reader, bank)
    assert reader.calls == []


@pytest.mark.parametrize("kwargs", [{"max_working_bytes": 1}, {"max_axes": 1},
    {"svd_rtol": 0}, {"svd_rtol": 1}, {"svd_rtol": float("nan")},
    {"weight_floor": 1.1}, {"probe_gain": -1}, {"probe_gain": True}, {"max_axes": True}])
def test_budgets_and_options_checked_before_reads(kwargs):
    reader = fixture()
    bank = bank_for(reader.identity, [[1, 0], [0, 1]])
    with pytest.raises((MemoryError, ValueError)):
        fit(reader, bank, **kwargs)
    assert reader.calls == []


def test_fp8_source_precision_alias_is_exactly_mapped():
    reader = fixture()
    source = reader.identity.to_dict()
    source.update(source_dtype="float8_e4m3fn", source_storage_bytes=32)
    reader.identity = ExpertGroupIdentity.from_dict(source)
    bank = bank_for(reader.identity, [[0, 1]])
    bank["scope"]["teacher"]["source_precision"] = "native_fp8_e4m3fn"
    fit(reader, rehash(bank))


def test_serialized_roundtrip_retains_evidence_and_same_response(tmp_path):
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    candidate = fit(reader, bank)
    save_expert_compression(candidate, tmp_path / "candidate")
    loaded = load_expert_compression(tmp_path / "candidate", expected_identity=reader.identity)
    assert loaded.metadata["objective"] == candidate.metadata["objective"]
    assert loaded.metadata["objective"]["pole_record_ids"] == bank["pole_record_ids"]
    assert evaluate(loaded, reader, bank) == evaluate(candidate, reader, bank)


def test_native_bank_retains_actual_archive_poles_magnitude_and_fingerprint(tmp_path):
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    writer = CaptureArchiveWriter(tmp_path, bank["scope"]["teacher"], origin="synthetic_fixture")
    states = torch.tensor([[0., 0.], [0., 2.]])
    rows = [{"id": f"pole:{i}", "trajectory_id": f"seed-{i}", "environment_id": "seed",
        "kind": "probe_framing", "split": "calibration", "probe_id": "p_seed",
        "polarity": -1 if i == 0 else 1, "token_position": 1, "next_token_id": -1,
        "next_token_observed": False} for i in range(2)]
    writer.write_chunk("seed", {"final_hidden": states, "layer_3_expert_input": states.clone(),
        "positions": torch.tensor([1, 1]), "next_token_ids": torch.tensor([-1, -1])},
        {"rows": rows, "state_sites": {"final_hidden": "post_final_norm_lm_head_input",
            "layer_3_expert_input": "post_attention_norm_expert_gate_up_input"}})
    writer.finalize("partial")
    measured = native_probe_bank(tmp_path, tensor_name="layer_3_expert_input")
    assert measured["pole_record_ids"] == [{"negative": "pole:0", "positive": "pole:1"}]
    torch.testing.assert_close(measured["contrast_magnitudes"], torch.tensor([2.]))
    assert measured["evidence_sha256"] == native_probe_bank_fingerprint(measured)
    assert len(measured["source_capture_sha256"]) == 1
    fit(reader, measured)
    altered = copy.deepcopy(measured)
    altered["centers"] += 1
    assert native_probe_bank_fingerprint(altered) != measured["evidence_sha256"]


def test_evaluator_rejects_different_teacher_formatting_and_budget_before_reads():
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    candidate = fit(reader, bank)
    reader.calls.clear()
    alternate = copy.deepcopy(bank)
    alternate["scope"]["teacher"]["chat_template_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="formatting"):
        evaluate(candidate, reader, rehash(alternate))
    with pytest.raises(MemoryError):
        evaluate(candidate, reader, bank, max_working_bytes=1)
    assert reader.calls == []


def test_evaluation_cannot_emit_a_report_after_reader_changes_source_scope():
    reader = fixture()
    bank = bank_for(reader.identity, [[0, 1]])
    candidate = fit(reader, bank)
    original_read = reader.read_columns

    def changing_read(start, stop, expert_ids=None):
        values = original_read(start, stop, expert_ids)
        source = reader.identity.to_dict()
        source["revision"] = "f" * 40
        reader.identity = ExpertGroupIdentity.from_dict(source)
        return values

    reader.read_columns = changing_read
    with pytest.raises(ValueError, match="scope changed"):
        evaluate(candidate, reader, bank)
