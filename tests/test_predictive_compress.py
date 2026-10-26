"""Bounded synthetic calibration mechanics; no learned-model quality claims."""
from dataclasses import replace
import copy

import pytest
import torch

from tda.expert_compress import (ExpertGroupIdentity, TensorExpertReader,
                                 fit_expert_group, save_expert_compression,
                                 load_expert_compression)
from tda.trajectory_compress import InputStateScope
from tda.predictive_compress import (PredictiveTrajectory, predictive_objective,
                                    refine_predictive_factors)


class Reader(TensorExpertReader):
    def __init__(self, identity, weights):
        super().__init__(identity, weights)
        self.reads = 0

    def read_columns(self, *args, **kwargs):
        self.reads += 1
        return super().read_columns(*args, **kwargs)


def fixture(dtype="float32"):
    identity = ExpertGroupIdentity("synthetic", "fixture-v1", 2, "query_head", "heads-row-major",
                                   ("head0", "head1"), (1, 1), "float32")
    reader = Reader(identity, torch.tensor([[[1.]], [[3.]]]))
    result = fit_expert_group(reader, 0, storage_dtype=dtype)
    reader.reads = 0
    scope = InputStateScope.from_identity(identity, state_site="pre_query")
    batch = PredictiveTrajectory("train", "test-environment", scope,
                                 ("r0", "r1"), (0, 1),
                                 torch.tensor([[-1., 1.], [1., -1.]]),
                                 torch.tensor([0, 0], dtype=torch.int64), ((0, 1),),
                                 origin="synthetic_fixture")
    calls = []

    def forward(item, mean, basis, coefficients):
        calls.append(item.trajectory_id)
        score = torch.tensor([-1., 1.]) * mean[0]
        return torch.stack((score, -score), dim=-1)

    return reader, result, scope, batch, forward, calls


def refine(reader, initial, scope, batch, forward, **kw):
    return refine_predictive_factors(reader, initial, [batch], forward, expected_scope=scope,
                                     heldout_trajectory_ids=kw.pop("heldout_trajectory_ids", ("test",)),
                                     heldout_record_ids=kw.pop("heldout_record_ids", ("h0",)),
                                     epochs=kw.pop("epochs", 6), learning_rate=kw.pop("learning_rate", 0.1), **kw)


def test_same_argmax_is_not_zero_predictive_distribution_error():
    _, _, _, batch, _, _ = fixture()
    batch = replace(batch, teacher_logits=torch.tensor([[4., 0.], [4., 0.]]))
    student = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
    assert torch.equal(student.argmax(-1), batch.teacher_logits.argmax(-1))
    loss = predictive_objective(student, batch)
    assert loss["point_kl"] > 0.1
    loss["point_kl"].backward()
    assert bool((student.grad != 0).any())


def test_wrong_teacher_predictions_remain_valid_full_vocabulary_observations():
    _, _, _, batch, _, _ = fixture()
    assert int((batch.teacher_logits.argmax(-1) == batch.observed_next_token_ids).sum()) == 1
    student = torch.zeros_like(batch.teacher_logits, requires_grad=True)
    loss = predictive_objective(student, batch)
    correct_targets = replace(batch, observed_next_token_ids=batch.teacher_logits.argmax(-1))
    alternate = predictive_objective(student, correct_targets)
    torch.testing.assert_close(loss["point_kl"], alternate["point_kl"])
    assert loss["point_kl"] > 0 and loss["contrast_loss"] > 0
    assert loss["positions"] == 2 and loss["teacher_argmax_matches_observed"] == 1


def test_signed_relation_loss_detects_wrong_student_change_direction():
    _, _, _, batch, _, _ = fixture()
    aligned = predictive_objective(batch.teacher_logits.clone(), batch)
    reversed_student = predictive_objective(batch.teacher_logits.flip(0), batch)
    reversed_both_pair = predictive_objective(batch.teacher_logits.clone(), replace(batch, pairs=((1, 0),)))
    assert aligned["contrast_loss"] < 1e-10
    assert reversed_student["contrast_loss"] > 1
    torch.testing.assert_close(aligned["contrast_loss"], reversed_both_pair["contrast_loss"])


def test_logit_shift_is_not_a_change_in_token_log_odds():
    _, _, _, batch, _, _ = fixture()
    shifted = batch.teacher_logits + torch.tensor([[10.], [-20.]])
    loss = predictive_objective(shifted, batch)
    assert loss["point_kl"] < 1e-6 and loss["contrast_loss"] < 1e-10


def test_zero_teacher_change_is_valid_and_student_drift_penalized():
    _, _, _, batch, _, _ = fixture()
    batch = replace(batch, teacher_logits=torch.tensor([[1., -1.], [1., -1.]]))
    loss = predictive_objective(torch.tensor([[1., -1.], [-1., 1.]]), batch)
    assert bool(torch.isfinite(loss["contrast_loss"])) and loss["contrast_loss"] > 0


def test_observed_wrong_target_enters_union_even_when_top_one_unchanged():
    _, _, _, batch, _, _ = fixture()
    batch = replace(batch, teacher_logits=torch.tensor([[4., 0.], [2., 0.]]),
                    observed_next_token_ids=torch.tensor([1, 1]))
    loss = predictive_objective(torch.zeros(2, 2), batch, top_k=1)
    assert loss["contrast_loss"] > 0  # observed target adds the second log-odds axis


def test_calibration_descends_same_bytes_frozen_coefficients_no_source_mutation():
    reader, initial, scope, batch, forward, calls = fixture()
    old_metadata = copy.deepcopy(initial.metadata)
    old_mean, old_basis, old_coeff = initial.mean.clone(), initial.basis.clone(), initial.coefficients.clone()
    old_weights = reader.weights.clone()
    result = refine(reader, initial, scope, batch, forward)
    correction = result.metadata["objective"]["predictive_correction"]
    assert correction["stored_precision_calibration"]["total"] < correction["initial_calibration"]["total"]
    assert correction["best_checkpoint_selection"] == "calibration_objective_only"
    assert correction["heldout_callback_invoked"] is False
    assert result.metadata["factor_tensor_bytes"] == initial.metadata["factor_tensor_bytes"]
    torch.testing.assert_close(result.coefficients, initial.coefficients)
    torch.testing.assert_close(initial.mean, old_mean)
    torch.testing.assert_close(initial.basis, old_basis)
    torch.testing.assert_close(initial.coefficients, old_coeff)
    torch.testing.assert_close(reader.weights, old_weights)
    assert initial.metadata == old_metadata and set(calls) == {"train"}
    assert correction["all_observations_valid_independent_of_prediction_accuracy"] is True
    assert result.metadata["diagnostics"]["reconstruction_error_frobenius_sq"] > initial.metadata["diagnostics"]["reconstruction_error_frobenius_sq"]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_storage_dtype_and_exact_payload_budget_preserved(dtype):
    reader, initial, scope, batch, forward, _ = fixture(dtype)
    result = refine(reader, initial, scope, batch, forward, epochs=2)
    assert result.mean.dtype == initial.mean.dtype and result.basis.dtype == initial.basis.dtype
    assert result.metadata["factor_tensor_bytes"] == initial.metadata["factor_tensor_bytes"]


def test_corrected_artifact_exports_with_fresh_checksums_and_real_diagnostics(tmp_path):
    reader, initial, scope, batch, forward, _ = fixture()
    save_expert_compression(initial, tmp_path / "initial")
    initial = load_expert_compression(tmp_path / "initial", expected_identity=reader.identity)
    result = refine(reader, initial, scope, batch, forward, epochs=2)
    assert "factors_sha256" not in result.metadata
    save_expert_compression(result, tmp_path / "corrected")
    loaded = load_expert_compression(tmp_path / "corrected", expected_identity=reader.identity)
    assert loaded.metadata["objective"]["predictive_correction"]["epochs"] == 2
    actual = reader.weights.flatten(1) - loaded.mean.float()
    assert loaded.metadata["diagnostics"]["reconstruction_error_frobenius_sq"] == pytest.approx(float(actual.double().square().sum()))


@pytest.mark.parametrize("kind", ["trajectory", "record"])
def test_heldout_exclusion_precedes_callback_and_reader(kind):
    reader, initial, scope, batch, forward, calls = fixture()
    kw = {"heldout_trajectory_ids": ("train",)} if kind == "trajectory" else {"heldout_record_ids": ("r0",)}
    with pytest.raises(ValueError, match="heldout"):
        refine(reader, initial, scope, batch, forward, **kw)
    assert reader.reads == 0 and calls == []


@pytest.mark.parametrize("kind", ["site", "model", "projection", "revision", "layer", "fingerprint"])
def test_scope_rejection_precedes_callback_and_reader(kind):
    reader, initial, scope, batch, forward, calls = fixture()
    replacements = {"site": {"state_site": "other"}, "model": {"model_id": "other"},
                    "projection": {"projection": "other"}, "revision": {"revision": "other"},
                    "layer": {"layer": 4}, "fingerprint": {"identity_sha256": "0" * 64}}
    batch = replace(batch, scope=replace(scope, **replacements[kind]))
    with pytest.raises(ValueError, match="scope"):
        refine(reader, initial, scope, batch, forward)
    assert reader.reads == 0 and calls == []


@pytest.mark.parametrize("kw", [{"max_working_bytes": 1}, {"callback_working_bytes": 1 << 40},
                                {"max_positions": 1}, {"max_pairs": 1}])
def test_budget_rejection_precedes_callback_and_reader(kw):
    reader, initial, scope, batch, forward, calls = fixture()
    if "max_pairs" in kw:
        batch = replace(batch, pairs=((0, 1), (1, 0)))
    with pytest.raises(MemoryError):
        refine(reader, initial, scope, batch, forward, **kw)
    assert reader.reads == 0 and calls == []


@pytest.mark.parametrize("kind", ["nan", "shape", "dtype", "target", "positions", "missing_pair", "duplicate_pair", "origin", "weight"])
def test_invalid_observations_rejected_before_callback_and_reader(kind):
    reader, initial, scope, batch, forward, calls = fixture()
    changes = {
        "nan": {"teacher_logits": torch.tensor([[float("nan"), 0.], [1., 0.]])},
        "shape": {"teacher_logits": torch.ones(2)},
        "dtype": {"teacher_logits": torch.ones(2, 2, dtype=torch.float64)},
        "target": {"observed_next_token_ids": torch.tensor([3, 0])},
        "positions": {"token_positions": (1, 0)},
        "missing_pair": {"pairs": ((0, 3),)},
        "duplicate_pair": {"pairs": ((0, 1), (1, 0))},
        "origin": {"origin": "semantic_truth"},
        "weight": {"provenance_weight": float("nan")},
    }
    batch = replace(batch, **changes[kind])
    with pytest.raises(ValueError):
        refine(reader, initial, scope, batch, forward)
    assert reader.reads == 0 and calls == []


def test_nondifferentiable_callback_rejected_without_source_write():
    reader, initial, scope, batch, _, _ = fixture()
    with pytest.raises(ValueError, match="differentiable"):
        refine(reader, initial, scope, batch, lambda *args: torch.zeros(2, 2))
    assert reader.reads == 0


def test_unrelated_gradient_leaf_callback_is_rejected_without_source_write():
    reader, initial, scope, batch, _, _ = fixture()
    before = initial.mean.clone()
    with pytest.raises(ValueError, match="disconnected"):
        refine(reader, initial, scope, batch,
               lambda *args: torch.zeros(2, 2, requires_grad=True))
    assert reader.reads == 0
    torch.testing.assert_close(initial.mean, before)


def test_callback_wrong_shape_or_nonfinite_rejected():
    reader, initial, scope, batch, _, _ = fixture()
    with pytest.raises(ValueError, match="callback"):
        refine(reader, initial, scope, batch, lambda *args: torch.ones(3, 2))
    with pytest.raises(ValueError, match="callback"):
        refine(reader, initial, scope, batch, lambda *args: torch.full((2, 2), float("inf")))
    assert reader.reads == 0


def test_best_epoch_selection_can_keep_initial_if_update_worsens_calibration():
    reader, initial, scope, batch, forward, _ = fixture()
    result = refine(reader, initial, scope, batch, forward, learning_rate=100, epochs=1)
    correction = result.metadata["objective"]["predictive_correction"]
    assert correction["best_epoch"] == 0
    torch.testing.assert_close(result.mean, initial.mean)


def test_no_heldout_declaration_is_rejected_before_callback():
    reader, initial, scope, batch, forward, calls = fixture()
    with pytest.raises(ValueError, match="heldout"):
        refine(reader, initial, scope, batch, forward, heldout_trajectory_ids=())
    assert reader.reads == 0 and calls == []


def test_heldout_declarations_are_sample_bounded_before_callback():
    reader, initial, scope, batch, forward, calls = fixture()
    with pytest.raises(MemoryError, match="heldout"):
        refine(reader, initial, scope, batch, forward, max_positions=2,
               heldout_record_ids=("h0", "h1", "h2"))
    assert reader.reads == 0 and calls == []


def test_zero_quality_mass_rejected_before_callback():
    reader, initial, scope, batch, forward, calls = fixture()
    with pytest.raises(ValueError, match="mass"):
        refine(reader, initial, scope, replace(batch, stimulation_weight=0), forward)
    assert reader.reads == 0 and calls == []


def test_teacher_normalization_overflow_rejected_before_callback():
    reader, initial, scope, batch, forward, calls = fixture()
    batch = replace(batch, teacher_logits=torch.tensor([[3e38, -3e38], [3e38, -3e38]]))
    with pytest.raises(ValueError, match="precision"):
        refine(reader, initial, scope, batch, forward)
    assert reader.reads == 0 and calls == []


def test_rank_one_coefficient_columns_remain_exactly_fixed():
    reader, initial, scope, batch, forward, _ = fixture()
    initial = fit_expert_group(reader, 1, storage_dtype="float32")
    # Include basis in a differentiable synthetic decoder.
    def with_basis(item, mean, basis, coefficients):
        score = torch.tensor([-1., 1.]) * (mean[0] + coefficients[0] @ basis[:, 0])
        return torch.stack((score, -score), dim=-1)
    result = refine(reader, initial, scope, batch, with_basis, epochs=1)
    assert torch.equal(result.coefficients, initial.coefficients)
