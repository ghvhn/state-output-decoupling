"""Synthetic CPU fixtures test a scoped objective, never GLM language quality."""
from dataclasses import replace

import pytest
import torch

from tda.expert_compress import (ExpertGroupIdentity, TensorExpertReader,
                                 fit_expert_group, save_expert_compression,
                                 load_expert_compression)
from tda.trajectory_compress import (InputStateScope, CalibrationRecord,
                                    TrajectoryPair, fit_trajectory_group,
                                    evaluate_trajectory_group)


class CountingReader(TensorExpertReader):
    def __init__(self, identity, weights):
        super().__init__(identity, weights)
        self.calls = []

    def read_columns(self, start, stop, expert_ids=None):
        self.calls.append((start, stop))
        return super().read_columns(start, stop, expert_ids)


def fixture():
    identity = ExpertGroupIdentity("synthetic-test", "test-revision", 3,
                                   "gate_proj", "row_major_gate_neurons",
                                   ("a", "b", "c", "d"), (2, 2), "float32")
    a = torch.tensor([-1., -1., 1., 1.])
    b = torch.tensor([-1., 1., -1., 1.])
    weights = torch.zeros(4, 2, 2)
    weights[:, 0, 0] = 100 * a
    weights[:, 1, 1] = b
    reader = CountingReader(identity, weights)
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm")
    return reader, scope


def record(scope, name, value, *, trajectory="train", environment="env", correct=None, **kw):
    return CalibrationRecord(name, environment, trajectory, scope,
                             torch.tensor(value, dtype=torch.float32),
                             prediction_correct=correct, origin="synthetic_fixture", **kw)


def training(scope):
    return [record(scope, "a0", [1, 0], correct=True),
            record(scope, "a1", [1, 1], correct=False)], [TrajectoryPair("p", "a0", "a1")]


def heldout(scope):
    return [record(scope, "h0", [2, 0], trajectory="test"),
            record(scope, "h1", [2, 3], trajectory="test")], [TrajectoryPair("h", "h0", "h1")]


def fit(reader, scope, records=None, pairs=None, **kw):
    if records is None:
        records, default_pairs = training(scope)
        pairs = default_pairs if pairs is None else pairs
    return fit_trajectory_group(reader, 1, records, pairs or (), expected_scope=scope,
                                point_weight=kw.pop("point_weight", 0),
                                contrast_weight=kw.pop("contrast_weight", 1),
                                chunk_columns=2, **kw)


def test_same_bytes_preserve_low_weight_energy_heldout_transition_better():
    reader, scope = fixture()
    train, pairs = training(scope)
    test, test_pairs = heldout(scope)
    control = fit_expert_group(reader, 1, storage_dtype="float32", chunk_columns=2)
    proposed = fit(reader, scope, train, pairs, heldout_records=test, heldout_pairs=test_pairs)
    assert proposed.metadata["factor_tensor_bytes"] == control.metadata["factor_tensor_bytes"]
    assert proposed.metadata["diagnostics"]["relative_frobenius_error"] > control.metadata["diagnostics"]["relative_frobenius_error"]
    baseline_error = evaluate_trajectory_group(control, reader, test, test_pairs, expected_scope=scope)
    proposed_error = evaluate_trajectory_group(proposed, reader, test, test_pairs, expected_scope=scope)
    assert baseline_error["contrast_relative_rmse"] > 0.99
    assert proposed_error["contrast_relative_rmse"] < 1e-6
    assert proposed_error["semantic_quality_evaluated"] is False
    assert proposed_error["record_origin"] == "synthetic_fixture"
    assert proposed.metadata["objective"]["heldout_used_for_fit"] is False


def test_correct_and_wrong_predictions_are_equally_valid_not_accuracy_weighted():
    reader, scope = fixture()
    records, pairs = training(scope)
    original = fit(reader, scope, records, pairs)
    inverted = fit(reader, scope, [replace(r, prediction_correct=not r.prediction_correct) for r in records], pairs)
    torch.testing.assert_close(original.coefficients, inverted.coefficients)
    torch.testing.assert_close(original.basis, inverted.basis)
    metadata = original.metadata["objective"]
    assert metadata["all_records_are_valid_observations"] is True
    assert metadata["prediction_accuracy_is_a_separate_observation"] is True
    assert metadata["training_records"] == 2 and metadata["correct_predictions"] == 1


def test_pair_reversal_is_same_quadratic_objective_not_a_negative_control():
    reader, scope = fixture()
    records, pairs = training(scope)
    forward = fit(reader, scope, records, pairs)
    reverse = fit(reader, scope, records, [TrajectoryPair("reverse", "a1", "a0")])
    torch.testing.assert_close(forward.coefficients, reverse.coefficients)
    torch.testing.assert_close(forward.basis, reverse.basis)
    assert forward.metadata["objective"]["pair_reversal_changes_quadratic_gram"] is False


def test_shuffled_within_scope_endpoints_are_a_substantive_contrast_control():
    reader, scope = fixture()
    records = [record(scope, f"r{i}", state) for i, state in enumerate(([1, 0], [1, 1], [2, 1], [2, 2]))]
    proper = [TrajectoryPair("p0", "r0", "r1"), TrajectoryPair("p1", "r1", "r2"), TrajectoryPair("p2", "r2", "r3")]
    shuffled = [TrajectoryPair("s0", "r0", "r2"), TrajectoryPair("s1", "r2", "r1"), TrajectoryPair("s2", "r1", "r3")]
    good = fit(reader, scope, records, proper)
    shuffled_result = fit(reader, scope, records, shuffled)
    test, pairs = heldout(scope)
    assert evaluate_trajectory_group(good, reader, test, pairs, expected_scope=scope)["contrast_relative_rmse"] < 1e-6
    assert evaluate_trajectory_group(shuffled_result, reader, test, pairs, expected_scope=scope)["contrast_relative_rmse"] > 0.99


def test_relative_source_energy_removes_arbitrary_positive_magnitude():
    reader, scope = fixture()
    records = [record(scope, "r0", [0, 1])]
    small = fit(reader, scope, records, [], point_weight=1, contrast_weight=0)
    large = fit(reader, scope, [replace(records[0], state=records[0].state * 10000)], [], point_weight=1, contrast_weight=0)
    torch.testing.assert_close(small.coefficients, large.coefficients)
    torch.testing.assert_close(small.basis, large.basis)


def test_trajectory_mass_balancing_avoids_longer_log_dominating():
    reader, scope = fixture()
    short = [record(scope, "a", [1, 0], trajectory="one"),
             record(scope, "b", [0, 1], trajectory="two", provenance_weight=0.5)]
    long = [short[0]] + [record(scope, f"b{i}", [0, 1], trajectory="two", provenance_weight=0.5)
                        for i in range(20)]
    one = fit(reader, scope, short, [], point_weight=1, contrast_weight=0)
    many = fit(reader, scope, long, [], point_weight=1, contrast_weight=0)
    # Unequal trajectory mass avoids a degenerate eigenspace with arbitrary basis.
    torch.testing.assert_close(one.coefficients @ one.coefficients.T,
                               many.coefficients @ many.coefficients.T, atol=1e-5, rtol=1e-5)


def test_weight_floor_can_recover_weight_control_without_changing_bytes():
    reader, scope = fixture()
    contrast = fit(reader, scope)
    floor = fit(reader, scope, weight_floor=0.99)
    control = fit_expert_group(reader, 1, storage_dtype="float32", chunk_columns=2)
    assert contrast.metadata["factor_tensor_bytes"] == floor.metadata["factor_tensor_bytes"]
    torch.testing.assert_close(floor.coefficients @ floor.coefficients.T,
                               control.coefficients @ control.coefficients.T)


def test_heldout_values_do_not_change_fitting_or_calibration_fingerprint():
    reader, scope = fixture()
    test, pairs = heldout(scope)
    original = fit(reader, scope, heldout_records=test, heldout_pairs=pairs)
    changed = fit(reader, scope, heldout_records=[replace(r, state=r.state * 12) for r in test], heldout_pairs=pairs)
    torch.testing.assert_close(original.coefficients, changed.coefficients)
    assert original.metadata["objective"]["calibration_sha256"] == changed.metadata["objective"]["calibration_sha256"]
    assert original.metadata["objective"]["heldout_sha256"] != changed.metadata["objective"]["heldout_sha256"]


def test_artifact_export_retains_objective_and_exact_source_scope(tmp_path):
    reader, scope = fixture()
    result = fit(reader, scope)
    saved = save_expert_compression(result, tmp_path / "candidate")
    reloaded = load_expert_compression(tmp_path / "candidate", expected_identity=reader.identity)
    assert reloaded.metadata["objective"] == result.metadata["objective"]
    assert saved["serialized_total_bytes"] == sum(p.stat().st_size for p in (tmp_path / "candidate").iterdir())
    assert saved["serialized_is_smaller"] is False  # honest tiny-container overhead


@pytest.mark.parametrize("field,value", [("model_id", "other"), ("revision", "wrong"), ("layer", 4),
                                         ("projection", "up_proj"), ("layout", "other"),
                                         ("matrix_input_width", 3), ("state_site", "wrong-site"),
                                         ("identity_sha256", "0" * 64)])
def test_wrong_scope_rejected_before_any_weight_read(field, value):
    reader, scope = fixture()
    records, pairs = training(scope)
    records[0] = replace(records[0], scope=replace(scope, **{field: value}))
    with pytest.raises(ValueError, match="scope"):
        fit(reader, scope, records, pairs)
    assert reader.calls == []


@pytest.mark.parametrize("state", [torch.tensor([float("nan"), 0]), torch.tensor([float("inf"), 1]),
                                    torch.ones(2, 1), torch.ones(3), torch.ones(2, dtype=torch.int32)])
def test_invalid_state_rejected_before_weight_read(state):
    reader, scope = fixture()
    records, pairs = training(scope)
    records[0] = replace(records[0], state=state)
    with pytest.raises(ValueError, match="states"):
        fit(reader, scope, records, pairs)
    assert reader.calls == []


@pytest.mark.parametrize("kw", [{"provenance_weight": -1}, {"provenance_weight": 1.1},
                                {"stimulation_weight": float("nan")}, {"stimulation_weight": True},
                                {"prediction_correct": 0}, {"origin": "inferred_semantic_truth"}])
def test_invalid_evidence_metadata_rejected_before_weight_read(kw):
    reader, scope = fixture()
    records, pairs = training(scope)
    records[0] = replace(records[0], **kw)
    with pytest.raises(ValueError):
        fit(reader, scope, records, pairs)
    assert reader.calls == []


@pytest.mark.parametrize("kind", ["missing", "environment", "trajectory", "duplicate", "self", "reverse_duplicate"])
def test_invalid_pairs_rejected_before_reads(kind):
    reader, scope = fixture()
    records, pairs = training(scope)
    if kind == "missing":
        pairs = [TrajectoryPair("bad", "a0", "absent")]
    elif kind == "environment":
        records[1] = replace(records[1], environment_id="elsewhere")
    elif kind == "trajectory":
        records[1] = replace(records[1], trajectory_id="elsewhere")
    elif kind == "duplicate":
        pairs *= 2
    elif kind == "self":
        pairs = [TrajectoryPair("bad", "a0", "a0")]
    else:
        pairs.append(TrajectoryPair("bad", "a1", "a0"))
    with pytest.raises(ValueError):
        fit(reader, scope, records, pairs)
    assert reader.calls == []


@pytest.mark.parametrize("kind", ["record", "trajectory", "endpoint"])
def test_train_holdout_leakage_rejected_before_reads(kind):
    reader, scope = fixture()
    test, pairs = heldout(scope)
    if kind == "record":
        test[0] = replace(test[0], record_id="a0")
        pairs = []
    elif kind == "trajectory":
        test = [replace(r, trajectory_id="train") for r in test]
    else:
        pairs = [TrajectoryPair("bad", "a0", "h1")]
    with pytest.raises(ValueError):
        fit(reader, scope, heldout_records=test, heldout_pairs=pairs)
    assert reader.calls == []


@pytest.mark.parametrize("kw", [{"max_records": 1}, {"max_pairs": 1, "heldout_pairs": [TrajectoryPair("h", "h0", "h1")],
                                "heldout_records": "placeholder"}, {"max_working_bytes": 1}])
def test_budget_rejected_before_reads(kw):
    reader, scope = fixture()
    if kw.get("heldout_records") == "placeholder":
        kw = {**kw, "heldout_records": heldout(scope)[0]}
    with pytest.raises(MemoryError):
        fit(reader, scope, **kw)
    assert reader.calls == []


def test_missing_required_contrast_or_mass_rejected_before_reads():
    reader, scope = fixture()
    records, pairs = training(scope)
    with pytest.raises(ValueError, match="pairs"):
        fit(reader, scope, records, [])
    records = [replace(r, provenance_weight=0) for r in records]
    with pytest.raises(ValueError, match="mass"):
        fit(reader, scope, records, pairs)
    assert reader.calls == []


def test_evaluation_rejects_training_trajectory_even_with_new_record_ids():
    reader, scope = fixture()
    result = fit(reader, scope)
    reader.calls.clear()
    records, _ = training(scope)
    with pytest.raises(ValueError, match="leaks"):
        evaluate_trajectory_group(result, reader, [replace(r, record_id="new" + r.record_id) for r in records], expected_scope=scope)
    assert reader.calls == []


def test_evaluation_rejects_different_fitted_input_site_before_reads():
    reader, scope = fixture()
    result = fit(reader, scope)
    reader.calls.clear()
    other = replace(scope, state_site="other-site")
    records, pairs = heldout(scope)
    records = [replace(r, scope=other) for r in records]
    with pytest.raises(ValueError, match="input site"):
        evaluate_trajectory_group(result, reader, records, pairs, expected_scope=other)
    assert reader.calls == []


def test_evaluation_rejects_origin_relabeling_before_reads():
    reader, scope = fixture()
    result = fit(reader, scope)
    reader.calls.clear()
    records, pairs = heldout(scope)
    records = [replace(r, origin="measured_model_state") for r in records]
    with pytest.raises(ValueError, match="origin"):
        evaluate_trajectory_group(result, reader, records, pairs, expected_scope=scope)
    assert reader.calls == []


def test_nonfinite_reader_rejected_and_source_state_not_mutated():
    reader, scope = fixture()
    records, pairs = training(scope)
    before = [r.state.clone() for r in records]
    reader.weights[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        fit(reader, scope, records, pairs)
    for r, old in zip(records, before):
        torch.testing.assert_close(r.state, old)


def test_oversized_scope_identifiers_are_rejected_before_weight_reads():
    reader, scope = fixture()
    records, pairs = training(scope)
    records[0] = replace(records[0], record_id="x" * 257)
    with pytest.raises(ValueError, match="256"):
        fit(reader, scope, records, pairs)
    assert reader.calls == []


def test_mean_only_rank_and_zero_response_cases_are_honest():
    reader, scope = fixture()
    records, pairs = training(scope)
    result = fit_trajectory_group(reader, 0, records, pairs, expected_scope=scope,
                                  point_weight=0, contrast_weight=1, chunk_columns=2)
    assert result.rank == 0 and result.basis.shape == (0, 4)
    with pytest.raises(ValueError, match="zero source"):
        fit(reader, scope, [record(scope, "zero", [0, 0])], [], point_weight=1, contrast_weight=0)


def test_streaming_reads_whole_output_rows_without_full_group_materialization():
    reader, scope = fixture()
    fit(reader, scope)
    assert len(reader.calls) == 8  # two functional + two factor passes
    assert all(stop - start == 2 for start, stop in reader.calls)
