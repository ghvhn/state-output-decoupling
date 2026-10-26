"""Synthetic CPU evidence for the sampled nonlinear proxy and its boundaries."""
from dataclasses import replace
import copy
import hashlib
import json
import math

import pytest
import torch
import torch.nn.functional as functional

from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader
from tda.expert_compress import fit_expert_group
from tda.block_quant import quantize_reader
from tda.glm_capture_calibration import native_probe_bank_fingerprint
from tda.glm_moe_sample_eval import evaluate_sampled_moe_candidate, _metric_weights
from tda.probe_span_compress import SITE
from tda.trajectory_compress import CalibrationRecord, InputStateScope, TrajectoryPair, _features
from tda.probe_residual_compress import fit_probe_residual_group


class CountingReader(TensorExpertReader):
    def __init__(self, identity, weights):
        super().__init__(identity, weights)
        self.calls = []

    def read_columns(self, start, stop, expert_ids=None):
        self.calls.append((start, stop, expert_ids))
        return super().read_columns(start, stop, expert_ids)


def fixture(*, identity_candidate=False):
    ids = ("0", "1")
    gate_id = ExpertGroupIdentity("synthetic", "a"*40, 1, "gate_proj", "row-major", ids, (2, 3), "float32")
    up_id = replace(gate_id, projection="up_proj")
    down_id = replace(gate_id, projection="down_proj", matrix_shape=(3, 2))
    gate = CountingReader(gate_id, torch.tensor([[[.4, -.2, .8], [-.7, .3, .4]], [[.1, .6, -.3], [.5, -.6, .2]]]))
    up = CountingReader(up_id, torch.tensor([[[.2, .9, -.1], [-.2, .4, 1.]], [[.7, -.3, .5], [.2, .3, -.8]]]))
    down = CountingReader(down_id, torch.tensor([[[.5, -.2], [.1, .8], [.2, .3]], [[.4, .2], [-.7, .1], [.3, .8]]]))
    candidate = CountingReader(gate_id, gate.weights.clone() if identity_candidate else gate.weights*.85+.07)
    scope = InputStateScope.from_identity(gate_id, state_site=SITE)
    teacher = {"model_id": "synthetic", "revision": "a"*40, "hidden_size": 3, "vocab_size": 5,
        "num_layers": 2, "source_precision": "float32", "tokenizer_sha256": "b"*64, "chat_template_sha256": "c"*64}
    return gate, up, down, candidate, scope, teacher


def record(scope, name, state, *, trajectory="train", environment="env", correct=None, **kwargs):
    return CalibrationRecord(name, environment, trajectory, scope, torch.tensor(state, dtype=torch.float32),
        prediction_correct=correct, origin="synthetic_fixture", **kwargs)


def splits(scope):
    return {"calibration": ([record(scope, "a", [1., .2, -.8], correct=True),
        record(scope, "b", [.1, .5, 1.2], correct=False)], [TrajectoryPair("ab", "a", "b")]),
        "heldout": ([record(scope, "h0", [.2, -.3, 1.], trajectory="test"),
        record(scope, "h1", [.7, .8, .4], trajectory="test")], [TrajectoryPair("h01", "h0", "h1")])}


def evaluate(values, data=None, **kwargs):
    gate, up, down, candidate, scope, teacher = values
    return evaluate_sampled_moe_candidate(gate, up, down, candidate, data if data is not None else splits(scope),
        expected_scope=scope, expected_teacher=teacher, expected_origin="synthetic_fixture",
        row_chunk=kwargs.pop("row_chunk", 1), batch_records=kwargs.pop("batch_records", 1), **kwargs)


def dense_outputs(values, records, candidate=False):
    gate, up, down, proposed, _, _ = values
    x = torch.stack([r.state for r in records])
    w_gate = proposed.weights if candidate else gate.weights
    middle = functional.silu(torch.einsum("emh,nh->enm", w_gate, x))*torch.einsum("emh,nh->enm", up.weights, x)
    return torch.einsum("ehm,enm->enh", down.weights, middle)


def test_identity_candidate_is_exact_nonlinear_reference_and_readers_unchanged():
    values = fixture(identity_candidate=True)
    originals = [r.weights.clone() for r in values[:4]]
    result = evaluate(values)
    for split in result["splits"].values():
        assert split["point_relative_rmse"] == 0
        assert split["contrast_relative_rmse"] == 0
    for reader, original in zip(values[:4], originals):
        torch.testing.assert_close(reader.weights, original, atol=0, rtol=0)
    assert result["fitting_performed"] is False
    assert result["native_dynamic_fp8_kernel_evaluated"] is False
    assert result["modified_decoder_evaluated"] is False
    assert result["generation_evaluated"] is False
    assert result["semantic_quality_evaluated"] is False
    assert result["expert_chunk"] == 1
    assert result["splits"]["calibration"]["prediction_correct_counts"] == {"correct": 1, "incorrect": 1, "unknown": 0}
    assert result["splits"]["heldout"]["prediction_correct_counts"]["unknown"] == 2


def test_real_output_differences_match_independent_reference_not_f_of_delta():
    values = fixture()
    data = splits(values[4])
    report = evaluate(values, data)
    for name, (records, pairs) in data.items():
        original, proposed = dense_outputs(values, records), dense_outputs(values, records, True)
        energies = original.double().square().sum(dim=(0, 2))
        errors = (proposed.double()-original.double()).square().sum(dim=(0, 2))
        point = math.sqrt(float((errors/energies).mean()))
        reference_delta = original[:, 1].double()-original[:, 0].double()
        proposed_delta = proposed[:, 1].double()-proposed[:, 0].double()
        contrast = math.sqrt(float((proposed_delta-reference_delta).square().sum()/reference_delta.square().sum()))
        assert report["splits"][name]["point_relative_rmse"] == pytest.approx(point, abs=2e-7)
        assert report["splits"][name]["contrast_relative_rmse"] == pytest.approx(contrast, abs=2e-7)
        difference_record = replace(records[0], state=records[1].state-records[0].state)
        wrong = dense_outputs(values, [difference_record])[:, 0]
        assert not torch.allclose(wrong.double(), reference_delta, rtol=1e-4, atol=1e-4)


def test_weighting_matches_canonical_trajectory_quality_and_endpoint_degrees():
    scope = fixture()[4]
    records = [record(scope, "a", [1, 0, 0], provenance_weight=.2),
        record(scope, "b", [0, 1, 0], stimulation_weight=.7),
        record(scope, "c", [0, 0, 1], provenance_weight=.6, stimulation_weight=.4),
        record(scope, "d", [1, 1, 1], trajectory="other", provenance_weight=.3)]
    pairs = [TrajectoryPair("ab", "a", "b"), TrajectoryPair("bc", "b", "c")]
    _, canonical = _features(records, pairs)
    point, contrast = _metric_weights(records, pairs)
    torch.testing.assert_close(torch.cat((point, contrast)), canonical, rtol=0, atol=0)
    assert float(point.sum()) == pytest.approx((.2+.7+.6*.4)/3+.3)
    assert float(contrast.sum()) == pytest.approx((.2+.7+.6*.4)/3)


def test_weighted_global_errors_respect_unequal_trajectories_and_validity_labels():
    values = fixture()
    scope = values[4]
    records = [record(scope, f"r{i}", [i*.11+.2, .7-i*.06, .4], correct=(i%2 == 0),
        trajectory="long" if i < 5 else "short", provenance_weight=.8 if i < 5 else .2,
        stimulation_weight=.7 if i%2 else .3) for i in range(6)]
    pairs = [TrajectoryPair(f"p{i}", f"r{i}", f"r{i+1}") for i in range(4)]
    result = evaluate(values, {"heldout": (records, pairs)})["splits"]["heldout"]
    source, proposed = dense_outputs(values, records).double(), dense_outputs(values, records, True).double()
    energies = source.square().sum(dim=(0, 2))
    errors = (proposed-source).square().sum(dim=(0, 2))
    pw, cw = _metric_weights(records, pairs)
    expected = math.sqrt(float((pw*errors/energies).sum()/pw.sum()))
    assert result["point_relative_rmse"] == pytest.approx(expected, abs=2e-6)
    altered = [replace(r, prediction_correct=not r.prediction_correct) for r in records]
    report2 = evaluate(values, {"heldout": (altered, pairs)})["splits"]["heldout"]
    assert report2["point_relative_rmse"] == result["point_relative_rmse"]
    assert report2["contrast_relative_rmse"] == result["contrast_relative_rmse"]


def test_heldout_input_changes_only_evaluation_and_never_candidate_or_calibration():
    values = fixture()
    data = splits(values[4])
    before = values[3].weights.clone()
    first = evaluate(values, data)
    changed = {**data, "heldout": ([replace(r, state=r.state*3+.6) for r in data["heldout"][0]], data["heldout"][1])}
    second = evaluate(values, changed)
    assert first["splits"]["calibration"] == second["splits"]["calibration"]
    assert first["splits"]["heldout"]["point_relative_rmse"] != second["splits"]["heldout"]["point_relative_rmse"]
    torch.testing.assert_close(values[3].weights, before, rtol=0, atol=0)


@pytest.mark.parametrize("field,target", [("revision", "up"), ("layer", "down"),
    ("expert_ids", "up"), ("source_dtype", "up"), ("layout", "down"),
    ("model_id", "down"), ("matrix_shape", "up"), ("projection", "candidate")])
def test_scope_mismatch_rejected_before_weight_reads(field, target):
    values = fixture()
    index = {"up": 1, "down": 2, "candidate": 3}[target]
    updates = {"revision": "f"*40, "layer": 0, "expert_ids": ("1", "0"), "source_dtype": "bfloat16",
        "layout": "wrong", "model_id": "other", "matrix_shape": (3, 2), "projection": "up_proj"}
    values[index].identity = replace(values[index].identity, **{field: updates[field]})
    with pytest.raises(ValueError):
        evaluate(values)
    assert all(not r.calls for r in values[:4])


@pytest.mark.parametrize("field,value", [("revision", "f"*40), ("hidden_size", 4),
    ("num_layers", 1), ("source_precision", "bfloat16")])
def test_teacher_mismatch_rejected_before_reads(field, value):
    values = list(fixture())
    values[5] = {**values[5], field: value}
    with pytest.raises(ValueError, match="teacher"):
        evaluate(values)
    assert all(not r.calls for r in values[:4])


def test_no_cross_trajectory_pairs_or_split_identity_leakage():
    values = fixture()
    scope = values[4]
    records = [record(scope, "a", [1, 0, 1]), record(scope, "b", [0, 1, 1], trajectory="other")]
    with pytest.raises(ValueError, match="trajectory"):
        evaluate(values, {"heldout": (records, [TrajectoryPair("ab", "a", "b")])})
    data = splits(scope)
    same_trajectory = [replace(r, trajectory_id="train") for r in data["heldout"][0]]
    with pytest.raises(ValueError, match="disjoint"):
        evaluate(values, {**data, "heldout": (same_trajectory, data["heldout"][1])})
    assert all(not r.calls for r in values[:4])


def test_candidate_training_identity_cannot_overlap_heldout_evaluation():
    values = fixture()
    values[3].metadata = {"objective": {"training_trajectory_ids": ["test"]}}
    with pytest.raises(ValueError, match="fitted on heldout"):
        evaluate(values)
    assert all(not r.calls for r in values[:4])


@pytest.mark.parametrize("kwargs", [{"max_working_bytes": 1}, {"max_records": 1},
    {"row_chunk": 33}, {"batch_records": 257}, {"max_pairs": 4097}, {"max_records": True}])
def test_bounds_are_checked_before_snapshots_or_weight_reads(monkeypatch, kwargs):
    values = fixture()
    called = []
    def snapshot(*args, **kw):
        called.append(1)
        raise AssertionError("guard should prevent state cloning")
    monkeypatch.setattr("tda.glm_moe_sample_eval._snapshot", snapshot)
    with pytest.raises((ValueError, MemoryError)):
        evaluate(values, **kwargs)
    assert not called
    assert all(not r.calls for r in values[:4])


def test_one_expert_row_decoding_is_cached_across_record_batches():
    values = fixture()
    result = evaluate(values, batch_records=1)
    for reader in values[:4]:
        rows, columns = reader.identity.matrix_shape
        assert len(reader.calls) == reader.identity.expert_count*rows
        assert all(stop-start <= columns and len(ids) == 1 for start, stop, ids in reader.calls)
    other = fixture()
    result2 = evaluate(other, batch_records=4)
    for split in result["splits"]:
        assert result["splits"][split]["point_relative_rmse"] == pytest.approx(result2["splits"][split]["point_relative_rmse"], abs=2e-6)
        assert result["splits"][split]["contrast_relative_rmse"] == pytest.approx(result2["splits"][split]["contrast_relative_rmse"], abs=2e-6)


def test_nonfinite_weights_and_projection_overflow_reject_with_no_source_mutation():
    values = fixture()
    values[1].weights[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate(values)
    values = fixture()
    values[0].weights.fill_(1e30)
    before = values[0].weights.clone()
    records = [record(values[4], "overflow", [1e30, 1e30, 1e30])]
    with pytest.raises(ValueError, match="overflow"):
        evaluate(values, {"heldout": (records, [])})
    torch.testing.assert_close(values[0].weights, before, rtol=0, atol=0)


def test_zero_reference_energy_reports_undefined_relative_and_absolute_error():
    values = fixture()
    values[0].weights.zero_()
    report = evaluate(values)
    result = report["splits"]["heldout"]
    assert result["point_relative_rmse"] is None
    assert result["contrast_relative_rmse"] is None
    assert result["point_zero_source_features"] == 2
    assert result["contrast_zero_source_features"] == 1
    assert result["point_zero_source_absolute_error_sq"] > 0
    assert result["contrast_zero_source_absolute_error_sq"] > 0


def test_zero_quality_is_undefined_without_discarding_observed_records():
    values = fixture()
    data = {"heldout": ([replace(r, provenance_weight=0.) for r in splits(values[4])["heldout"][0]], [])}
    result = evaluate(values, data)["splits"]["heldout"]
    assert result["point_relative_rmse"] is None
    assert result["point_zero_weight_features"] == 2
    assert result["point_source_energy_sq"] > 0
    assert result["records"] == 2
    assert result["all_predictions_retained"] is True


def seed_fixture():
    identity = ExpertGroupIdentity("synthetic", "a"*40, 1, "gate_proj", "row-major", ("0",), (1, 2), "float32")
    gate = CountingReader(identity, torch.tensor([[[1., 1.]]]))
    up = CountingReader(replace(identity, projection="up_proj"), torch.tensor([[[0., 1.]]]))
    down = CountingReader(replace(identity, projection="down_proj", matrix_shape=(2, 1)), torch.tensor([[[1.], [2.]]]))
    proposed = CountingReader(identity, torch.tensor([[[1., 3.]]]))
    scope = InputStateScope.from_identity(identity, state_site=SITE)
    teacher = {"model_id": "synthetic", "revision": "a"*40, "hidden_size": 2, "vocab_size": 5,
        "num_layers": 2, "source_precision": "float32", "tokenizer_sha256": "b"*64, "chat_template_sha256": "c"*64}
    bank = {"format": "glm-native-probe-bank-v1", "scope": {"teacher": teacher, "tensor_name": "layer_1_expert_input", "state_site": SITE},
        "origin": "synthetic_fixture", "probe_ids": ["p0"], "directions": torch.tensor([[1., 0.]]),
        "centers": torch.tensor([[0., 1.]]), "contrast_magnitudes": torch.tensor([2.]),
        "pole_record_ids": [{"negative": "low", "positive": "high"}], "source_capture_sha256": ["d"*64],
        "unpaired_probe_ids": [], "zero_contrast_probe_ids": [], "semantic_meanings_validated": False,
        "heldout_used_to_fit_axes": False}
    rehash(bank)
    poles = {"p0": {"negative": record(scope, "low", [-1., 1.], trajectory="seedlow"),
        "positive": record(scope, "high", [1., 1.], trajectory="seedhigh")}}
    data = {"heldout": ([record(scope, "h0", [.3, .7], trajectory="test"),
        record(scope, "h1", [.8, .4], trajectory="test")], [TrajectoryPair("h", "h0", "h1")])}
    return (gate, up, down, proposed, scope, teacher), data, bank, poles


def rehash(bank):
    bank["scope_sha256"] = hashlib.sha256(json.dumps(bank["scope"], sort_keys=True).encode()).hexdigest()
    bank["evidence_sha256"] = native_probe_bank_fingerprint(bank)


def test_linear_probe_exactness_does_not_ensure_centered_nonlinear_seed_contrast():
    values, data, bank, poles = seed_fixture()
    source_delta = values[0].weights @ bank["directions"][0]
    proposed_delta = values[3].weights @ bank["directions"][0]
    torch.testing.assert_close(source_delta, proposed_delta, atol=0, rtol=0)
    report = evaluate(values, data, native_seed_bank=bank, native_seed_records=poles)
    seed = report["seed_pole_response"]
    assert seed["contrast_relative_rmse"] > .2
    assert seed["point_relative_rmse"] > .5
    assert seed["probe_count"] == 1 and seed["pole_records"] == 2
    assert seed["semantic_meanings_validated"] is False
    assert seed["heldout_examples"] is False
    assert "exact raw" in seed["inputs"]
    expected = abs(float(functional.silu(torch.tensor(4.))-2*functional.silu(torch.tensor(2.))))/float(functional.silu(torch.tensor(2.)))
    assert seed["contrast_relative_rmse"] == pytest.approx(expected, abs=1e-6)


def test_seed_identity_reference_and_center_dependency_are_preserved():
    values, data, bank, poles = seed_fixture()
    values[3].weights.copy_(values[0].weights)
    report = evaluate(values, data, native_seed_bank=bank, native_seed_records=poles)
    assert report["seed_pole_response"]["contrast_relative_rmse"] == 0
    assert report["seed_pole_response"]["point_relative_rmse"] == 0
    torch.testing.assert_close(bank["centers"], torch.tensor([[0., 1.]]), rtol=0, atol=0)


@pytest.mark.parametrize("field", ["pole_id", "center", "checksum", "source_site", "heldout", "raw_state"])
def test_seed_corruption_rejected_before_weight_reads(field):
    values, data, bank, poles = seed_fixture()
    if field == "pole_id":
        poles["p0"]["positive"] = replace(poles["p0"]["positive"], record_id="wrong")
    elif field == "center":
        bank["centers"] += .1
        rehash(bank)
    elif field == "checksum":
        bank["evidence_sha256"] = "0"*64
    elif field == "source_site":
        bank["scope"]["state_site"] = "block_output"
        rehash(bank)
    elif field == "heldout":
        bank["heldout_used_to_fit_axes"] = True
        rehash(bank)
    else:
        poles["p0"]["positive"] = replace(poles["p0"]["positive"], state=torch.tensor([1.2, 1.]))
    with pytest.raises(ValueError):
        evaluate(values, data, native_seed_bank=bank, native_seed_records=poles)
    assert all(not r.calls for r in values[:4])


def test_bank_without_raw_poles_cannot_fabricate_observed_seed_inputs():
    values, data, bank, _ = seed_fixture()
    with pytest.raises(ValueError, match="exact raw"):
        evaluate(values, data, native_seed_bank=bank)
    assert all(not r.calls for r in values[:4])


def test_reader_scope_change_and_invalid_slices_reject_even_with_matching_shape():
    values = fixture()
    original = values[1].read_columns
    def changing(start, stop, expert_ids=None):
        result = original(start, stop, expert_ids)
        values[1].identity = replace(values[1].identity, revision="f"*40)
        return result
    values[1].read_columns = changing
    with pytest.raises(ValueError, match="during"):
        evaluate(values)
    values = fixture()
    values[2].read_columns = lambda start, stop, expert_ids=None: torch.ones((2, stop-start))
    with pytest.raises(ValueError, match="invalid"):
        evaluate(values)


def test_candidate_tensor_payload_is_included_in_preallocation_guard():
    values = fixture()
    basic = evaluate(values)["estimated_numerical_peak_bytes"]
    values = fixture()
    values[3].basis = torch.zeros(1000, dtype=torch.float32)
    with pytest.raises(MemoryError, match="byte budget"):
        evaluate(values, max_working_bytes=basic+3999)
    assert all(not r.calls for r in values[:4])


def test_int4_residual_and_shared_factor_readers_use_the_same_bounded_evaluator():
    values, data, bank, poles = seed_fixture()
    # Non-integer coefficients ensure this is a real quantization/repair test.
    values[0].weights.copy_(torch.tensor([[[.31, 1.17]]]))
    control = quantize_reader(values[0], bits=4, block_size=2, chunk_blocks=1)
    repair = fit_probe_residual_group(values[0], bank, base_quantization=control,
        block_size=2, chunk_blocks=1, chunk_columns=2, expected_origin="synthetic_fixture")
    factor = fit_expert_group(values[0], 0, chunk_columns=2, storage_dtype="bfloat16")
    class FactorReader:
        def __init__(self, result):
            self.result, self.identity = result, result.identity
        def read_columns(self, start, stop, expert_ids=None):
            positions = [self.identity.expert_ids.index(v) for v in expert_ids]
            return self.result.coefficients[positions] @ self.result.basis[:, start:stop].float()+self.result.mean[start:stop].float()
    for candidate in (control, repair, FactorReader(factor)):
        result = evaluate((values[0], values[1], values[2], candidate, values[4], values[5]), data,
            native_seed_bank=bank, native_seed_records=poles)
        assert math.isfinite(result["splits"]["heldout"]["point_relative_rmse"])
        assert math.isfinite(result["seed_pole_response"]["contrast_relative_rmse"])
        assert result["resident_reader_and_bank_tensor_bytes"] >= sum(
            t.numel()*t.element_size() for t in (control.codes, control.scales))
        assert result["seed_pole_response"]["seed_bank_used_to_fit_candidate"] is (candidate is repair)
