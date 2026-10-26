"""Independent expert INT4 repair tests with explicitly synthetic evidence."""
import copy
import hashlib
import json

import pytest
import torch

from tda.block_quant import quantize_reader
from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader
from tda.glm_capture_calibration import native_probe_bank_fingerprint
from tda.probe_residual_compress import (fit_probe_residual_group, evaluate_probe_residual_group,
    save_probe_residual, load_probe_residual, make_random_basis, make_calibration_pca_basis,
    make_probe_pca_complement_basis, fit_scoped_residual_basis, residual_basis_fingerprint)
from tda.probe_span_compress import _span
from tda.trajectory_compress import InputStateScope, CalibrationRecord


class CountingReader(TensorExpertReader):
    def __init__(self, identity, weights):
        super().__init__(identity, weights)
        self.calls = []

    def read_columns(self, start, stop, expert_ids=None):
        self.calls.append((start, stop))
        return super().read_columns(start, stop, expert_ids)


def rehash(bank):
    bank["scope_sha256"] = hashlib.sha256(json.dumps(bank["scope"], sort_keys=True).encode()).hexdigest()
    bank["evidence_sha256"] = native_probe_bank_fingerprint(bank)
    return bank


def fixture(directions=None):
    identity = ExpertGroupIdentity("synthetic", "a" * 40, 1, "gate_proj", "row-major",
        ("e0", "e1", "e2"), (2, 4), "float32")
    weights = torch.tensor([[[.2, 1.4, -.35, 4.], [2., .15, .8, -.7]],
                            [[-.3, .25, 3., -.4], [.6, -1.1, .2, 2.5]],
                            [[.13, -.22, 1.9, -.7], [.9, .3, -.15, 2.1]]])
    reader = CountingReader(identity, weights)
    teacher = {"model_id": identity.model_id, "revision": identity.revision, "hidden_size": 4,
        "vocab_size": 5, "num_layers": 2, "source_precision": "float32",
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64}
    directions = torch.tensor(directions if directions is not None else [[1., 0, 0, 0], [0, 1., 0, 0]])
    n = len(directions)
    bank = {"format": "glm-native-probe-bank-v1", "scope": {"teacher": teacher,
        "tensor_name": "layer_1_expert_input", "state_site": "post_attention_norm_expert_gate_up_input"},
        "origin": "synthetic_fixture", "probe_ids": [f"p_{i}" for i in range(n)], "directions": directions,
        "centers": torch.zeros_like(directions), "contrast_magnitudes": torch.ones(n),
        "pole_record_ids": [{"negative": f"low-{i}", "positive": f"high-{i}"} for i in range(n)],
        "source_capture_sha256": ["d" * 64], "unpaired_probe_ids": [], "zero_contrast_probe_ids": [],
        "semantic_meanings_validated": False, "heldout_used_to_fit_axes": False}
    return reader, rehash(bank)


def fit(reader, bank, **kwargs):
    return fit_probe_residual_group(reader, bank, block_size=4, chunk_blocks=1, chunk_columns=4,
        expected_origin="synthetic_fixture", coefficient_dtype=kwargs.pop("coefficient_dtype", "float32"), **kwargs)


def evaluate(value, reader, bank, **kwargs):
    return evaluate_probe_residual_group(value, reader, bank, chunk_columns=4,
        expected_origin="synthetic_fixture", **kwargs)


def test_identical_int4_base_plus_independent_residual_repairs_source_span():
    reader, bank = fixture()
    control = quantize_reader(reader, bits=4, block_size=4, chunk_blocks=1)
    source_codes, source_scales = control.codes.clone(), control.scales.clone()
    repaired = fit(reader, bank, base_quantization=control)
    torch.testing.assert_close(control.codes, source_codes, atol=0, rtol=0)
    torch.testing.assert_close(control.scales, source_scales, atol=0, rtol=0)
    assert repaired.base_quantization is control
    before, after = evaluate(control, reader, bank), evaluate(repaired, reader, bank)
    assert before["span_response_relative_rmse"] > .03
    assert after["span_response_relative_rmse"] < 1e-6
    assert after["same_probe_bank_used_to_fit"] is True
    assert repaired.coefficients.shape == (3, 2, 2)
    total = control.metadata["factor_tensor_bytes"] + repaired.span_basis.numel() * 4 + repaired.coefficients.numel() * 4
    assert repaired.metadata["factor_tensor_bytes"] == total
    assert repaired.metadata["factor_tensor_bytes"] > control.metadata["factor_tensor_bytes"]
    assert repaired.metadata["tensor_storage_ratio"] == total / reader.identity.actual_source_bytes
    assert repaired.metadata["semantic_quality_evaluated"] is False
    assert repaired.metadata["native_fp8_kernel_evaluated"] is False
    assert repaired.metadata["objective"]["measured_temporal_transitions"] == 0


def test_signed_mixtures_and_gains_preserved_while_orthogonal_residual_unchanged():
    reader, bank = fixture()
    repaired = fit(reader, bank)
    decoded = repaired.read_columns(0, 8).reshape(3, 2, 4)
    base = repaired.base_quantization.read_columns(0, 8).reshape(3, 2, 4)
    for inputs in (torch.tensor([.75, -1.25, 0., 0.]), torch.tensor([-2., .5, 0., 0.])):
        torch.testing.assert_close(decoded @ inputs, reader.weights @ inputs, atol=1e-6, rtol=1e-6)
    orthogonal = torch.tensor([0., 0., 1., -2.])
    torch.testing.assert_close(decoded @ orthogonal, base @ orthogonal, atol=0, rtol=0)
    torch.testing.assert_close(decoded[:, :, 2:], base[:, :, 2:], atol=0, rtol=0)


def test_oblique_span_has_same_signed_gain_and_leaves_orthogonal_actions():
    reader, bank = fixture([[.5, .5, .5, .5]])
    repaired = fit(reader, bank)
    decoded = repaired.read_columns(0, 8).reshape(3, 2, 4)
    base = repaired.base_quantization.read_columns(0, 8).reshape(3, 2, 4)
    axis = bank["directions"][0]
    for scale in (-3., .25):
        torch.testing.assert_close(decoded @ (axis * scale), reader.weights @ (axis * scale), atol=2e-6, rtol=2e-6)
    orthogonal = torch.tensor([-.5, .5, -.5, .5])
    torch.testing.assert_close(decoded @ orthogonal, base @ orthogonal, atol=1e-6, rtol=1e-6)


def test_bfloat16_coefficient_rounding_is_exposed_and_improves_same_base():
    reader, bank = fixture()
    repaired = fit(reader, bank, coefficient_dtype="bfloat16")
    before = evaluate(repaired.base_quantization, reader, bank)
    after = evaluate(repaired, reader, bank)
    assert repaired.coefficients.dtype == torch.bfloat16 and repaired.span_basis.dtype == torch.float32
    assert repaired.metadata["coefficient_storage_rounding"] == "BF16 rounding of C"
    assert 0 < after["span_response_relative_rmse"] < before["span_response_relative_rmse"] / 20
    assert repaired.metadata["correction_tensor_bytes"] == repaired.span_basis.numel() * 4 + repaired.coefficients.numel() * 2


def test_interval_reader_preserves_expert_order_and_checks_read_budget():
    reader, bank = fixture()
    repaired = fit(reader, bank)
    full = repaired.read_columns(0, 8)
    subset = repaired.read_columns(1, 7, ["e2", "e0"])
    torch.testing.assert_close(subset, full[[2, 0], 1:7])
    with pytest.raises(ValueError):
        repaired.read_columns(0, 8, ["e0", "e0"])
    repaired.max_read_bytes = 1
    with pytest.raises(MemoryError):
        repaired.read_columns(0, 8)


@pytest.mark.parametrize("field,value", [("tensor_name", "final_hidden"),
    ("tensor_name", "layer_0_expert_input"), ("state_site", "decoder_block_output"),
    ("revision", "f" * 40), ("source_precision", "bfloat16"), ("hidden_size", 5),
    ("origin", "measured_model_state"), ("evidence_sha256", "0" * 64)])
def test_scope_and_evidence_corruption_rejected_before_source_reads(field, value):
    reader, bank = fixture()
    if field in ("tensor_name", "state_site"):
        bank["scope"][field] = value
        rehash(bank)
    elif field in ("revision", "source_precision", "hidden_size"):
        bank["scope"]["teacher"][field] = value
        rehash(bank)
    else:
        bank[field] = value
    with pytest.raises(ValueError):
        fit(reader, bank)
    assert reader.calls == []


def test_down_projection_and_wrong_base_rejected_before_source_reads():
    reader, bank = fixture()
    other, _ = fixture()
    source = other.identity.to_dict()
    source["revision"] = "f" * 40
    other.identity = ExpertGroupIdentity.from_dict(source)
    wrong_base = quantize_reader(other, bits=4, block_size=4)
    with pytest.raises(ValueError):
        fit(reader, bank, base_quantization=wrong_base)
    assert reader.calls == []
    source = reader.identity.to_dict()
    source["projection"] = "down_proj"
    reader.identity = ExpertGroupIdentity.from_dict(source)
    with pytest.raises(ValueError, match="down"):
        fit(reader, bank)
    assert reader.calls == []


@pytest.mark.parametrize("kwargs", [{"max_working_bytes": 1}, {"max_read_bytes": 1},
    {"max_axes": 1}, {"coefficient_dtype": "float16"}, {"svd_rtol": 0}])
def test_allocation_budgets_and_options_checked_before_source_reads(kwargs):
    reader, bank = fixture()
    with pytest.raises((MemoryError, ValueError)):
        fit(reader, bank, **kwargs)
    assert reader.calls == []


def test_identical_base_required_to_use_int4():
    reader, bank = fixture()
    base8 = quantize_reader(reader, bits=8, block_size=4)
    reader.calls.clear()
    with pytest.raises(ValueError, match="INT4"):
        fit(reader, bank, base_quantization=base8)
    assert reader.calls == []


def test_safe_roundtrip_counts_every_serialized_file_and_never_overwrites(tmp_path):
    reader, bank = fixture()
    repaired = fit(reader, bank, coefficient_dtype="bfloat16")
    directory = tmp_path / "candidate"
    receipt = save_probe_residual(repaired, directory)
    loaded = load_probe_residual(directory, expected_identity=reader.identity)
    actual = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    assert receipt["serialized_total_bytes"] == actual
    assert loaded.metadata["serialized_total_bytes"] == actual
    assert receipt["serialized_is_smaller"] is False  # Tiny fixture has container overhead.
    torch.testing.assert_close(loaded.read_columns(0, 8), repaired.read_columns(0, 8), atol=0, rtol=0)
    assert evaluate(loaded, reader, bank)["span_response_relative_rmse"] == evaluate(repaired, reader, bank)["span_response_relative_rmse"]
    with pytest.raises(FileExistsError):
        save_probe_residual(repaired, directory)


def test_roundtrip_rejects_nonorthogonal_factors_and_wrong_expected_source(tmp_path):
    reader, bank = fixture()
    repaired = fit(reader, bank)
    malformed = copy.deepcopy(repaired)
    malformed.span_basis *= 2
    with pytest.raises(ValueError, match="orthonormal"):
        save_probe_residual(malformed, tmp_path / "malformed")
    directory = tmp_path / "candidate"
    save_probe_residual(repaired, directory)
    source = reader.identity.to_dict()
    source["revision"] = "f" * 40
    with pytest.raises(ValueError, match="scope"):
        load_probe_residual(directory, expected_identity=ExpertGroupIdentity.from_dict(source))


@pytest.mark.parametrize("corruption", ["correction", "base", "accounting", "precision", "meaning"])
def test_serialized_corruption_rejected_before_decoding(tmp_path, corruption):
    reader, bank = fixture()
    repaired = fit(reader, bank)
    directory = tmp_path / "candidate"
    save_probe_residual(repaired, directory)
    if corruption in ("correction", "base"):
        path = directory / "correction.pt" if corruption == "correction" else directory / "base" / "factors.pt"
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    else:
        path = directory / "metadata.json"
        metadata = json.loads(path.read_bytes())
        if corruption == "accounting":
            metadata["factor_tensor_bytes"] += 1
        elif corruption == "precision":
            metadata["coefficient_dtype"] = "float16"
        else:
            metadata["semantic_meanings_validated"] = True
        path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        load_probe_residual(directory, expected_identity=reader.identity)


def test_artifact_limits_bound_export_load_and_reject_extra_files(tmp_path):
    reader, bank = fixture()
    repaired = fit(reader, bank)
    with pytest.raises(MemoryError):
        save_probe_residual(repaired, tmp_path / "too-small", max_artifact_bytes=1)
    directory = tmp_path / "candidate"
    save_probe_residual(repaired, directory)
    with pytest.raises(MemoryError):
        load_probe_residual(directory, expected_identity=reader.identity, max_artifact_bytes=1)
    with pytest.raises(MemoryError):
        load_probe_residual(directory, expected_identity=reader.identity, max_metadata_bytes=1)
    (directory / "extra").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        load_probe_residual(directory, expected_identity=reader.identity)


def test_zero_source_response_reports_absolute_error():
    reader, bank = fixture([[1., 0., 0., 0.]])
    reader.weights[:, :, 0] = 0
    repaired = fit(reader, bank)
    report = evaluate(repaired, reader, bank)
    assert report["span_response_source_energy_sq"] == 0
    assert report["span_response_relative_rmse"] is None
    assert report["span_response_absolute_error_sq"] == 0
    repaired.coefficients[:, :, 0] += .25
    report = evaluate(repaired, reader, bank)
    assert report["span_response_absolute_error_sq"] > 0
    assert report["axis_responses"][0]["relative_rmse"] is None


def test_evaluation_bank_teacher_formatting_must_match_and_stays_bounded():
    reader, bank = fixture()
    repaired = fit(reader, bank)
    reader.calls.clear()
    alternate = copy.deepcopy(bank)
    alternate["scope"]["teacher"]["tokenizer_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="formatting"):
        evaluate(repaired, reader, rehash(alternate))
    with pytest.raises(MemoryError):
        evaluate(repaired, reader, bank, max_working_bytes=1)
    assert reader.calls == []


def test_static_evaluation_detects_changed_source_during_read():
    reader, bank = fixture()
    repaired = fit(reader, bank)
    original_read = reader.read_columns

    def changing_read(start, stop, expert_ids=None):
        values = original_read(start, stop, expert_ids)
        source = reader.identity.to_dict()
        source["revision"] = "f" * 40
        reader.identity = ExpertGroupIdentity.from_dict(source)
        return values

    reader.read_columns = changing_read
    with pytest.raises(ValueError, match="scope changed"):
        evaluate(repaired, reader, bank)


def calibration_records(reader, values, *, trajectory="train"):
    scope = InputStateScope.from_identity(reader.identity, state_site="post_attention_norm_expert_gate_up_input")
    return [CalibrationRecord(f"{trajectory}:{index}", "env", trajectory, scope,
        torch.tensor(value, dtype=torch.float32), origin="synthetic_fixture") for index, value in enumerate(values)]


def pca_basis(reader, bank, records=None, **kwargs):
    if records is None:
        records = calibration_records(reader, [[1, 0, 0, 0], [0, 1, 0, 0]])
    return make_calibration_pca_basis(records, bank, expected_scope=records[0].scope,
        requested_rank=kwargs.pop("requested_rank", 2), expected_origin="synthetic_fixture", **kwargs)


def random_basis(bank, **kwargs):
    return make_random_basis(bank, requested_rank=kwargs.pop("requested_rank", 2),
        expected_origin="synthetic_fixture", **kwargs)


def scoped_fit(reader, basis, control, **kwargs):
    return fit_scoped_residual_basis(reader, basis, base_quantization=control, block_size=4,
        chunk_blocks=1, chunk_columns=4, coefficient_dtype="float32", expected_origin="synthetic_fixture", **kwargs)


def test_equal_rank_probe_pca_random_controls_have_identical_base_and_tensor_bytes():
    reader, bank = fixture()
    control = quantize_reader(reader, bits=4, block_size=4)
    native = fit(reader, bank, base_quantization=control)
    pca, random = pca_basis(reader, bank), random_basis(bank, seed=17)
    candidates = [native, scoped_fit(reader, pca, control), scoped_fit(reader, random, control)]
    assert len({candidate.metadata["factor_tensor_bytes"] for candidate in candidates}) == 1
    assert all(candidate.span_basis.shape[1] == 2 for candidate in candidates)
    for candidate in candidates:
        assert candidate.base_quantization is control
        torch.testing.assert_close(candidate.base_quantization.codes, control.codes, atol=0, rtol=0)
        torch.testing.assert_close(candidate.base_quantization.scales, control.scales, atol=0, rtol=0)
    for candidate, kind in zip(candidates[1:], ("calibration_state_second_moment_span", "reproducible_random_span")):
        objective = candidate.metadata["objective"]
        assert objective["basis_kind"] == kind
        assert objective["rank_matches_request"] is True and objective["heldout_used_for_fit"] is False
        assert "probe_ids" not in objective and "pole_record_ids" not in objective
        assert "probe_evidence_sha256" not in objective
        report = evaluate(candidate, reader, bank)
        assert report["same_probe_bank_used_to_fit"] is False
        assert report["fitted_basis_kind"] == kind
    assert evaluate(candidates[1], reader, bank)["span_response_relative_rmse"] < 1e-6


def test_random_basis_is_reproducible_and_explicitly_synthetic():
    _, bank = fixture()
    first, second, changed = random_basis(bank, seed=23), random_basis(bank, seed=23), random_basis(bank, seed=24)
    torch.testing.assert_close(first["basis"], second["basis"], atol=0, rtol=0)
    assert first["evidence_sha256"] == second["evidence_sha256"] != changed["evidence_sha256"]
    assert not torch.equal(first["basis"], changed["basis"])
    assert first["feature_origin"] == "synthetic_random_control"
    assert first["training_record_ids"] == []


def test_pca_uses_uncentered_second_moment_and_reports_available_rank():
    reader, bank = fixture()
    records = calibration_records(reader, [[10, 1, 0, 0], [10, -1, 0, 0]])
    first = pca_basis(reader, bank, records, requested_rank=1)
    torch.testing.assert_close(first["basis"].abs(), torch.tensor([[1.], [0.], [0.], [0.]]))
    assert first["pca_objective"] == "uncentered unweighted calibration input second moment X.T @ X"
    low_rank = pca_basis(reader, bank, calibration_records(reader, [[1, 0, 0, 0], [2, 0, 0, 0]]))
    assert low_rank["requested_rank"] == 2 and low_rank["actual_rank"] == 1
    assert low_rank["rank_matches_request"] is False


@pytest.mark.parametrize("leak", ["record", "trajectory"])
def test_pca_rejects_heldout_record_or_trajectory_leakage(leak):
    reader, bank = fixture()
    train = calibration_records(reader, [[1, 0, 0, 0], [0, 1, 0, 0]])
    heldout = calibration_records(reader, [[0, 0, 1, 0]], trajectory="test")
    if leak == "record":
        heldout[0] = CalibrationRecord(train[0].record_id, "env", "test", train[0].scope,
            torch.ones(4), origin="synthetic_fixture")
    else:
        heldout[0] = CalibrationRecord("heldout", "env", train[0].trajectory_id, train[0].scope,
            torch.ones(4), origin="synthetic_fixture")
    with pytest.raises(ValueError, match="heldout"):
        pca_basis(reader, bank, train, heldout_records=heldout)
    assert reader.calls == []


def test_heldout_state_values_are_never_read_or_fitted_by_pca():
    reader, bank = fixture()
    heldout = calibration_records(reader, [[0, 0, 1, 0]], trajectory="test")
    first = pca_basis(reader, bank, heldout_records=heldout)
    heldout[0] = CalibrationRecord(heldout[0].record_id, "env", "test", heldout[0].scope,
        None, origin="synthetic_fixture")  # No tensor can be consulted.
    second = pca_basis(reader, bank, heldout_records=heldout)
    assert first["evidence_sha256"] == second["evidence_sha256"]
    torch.testing.assert_close(first["basis"], second["basis"], atol=0, rtol=0)


@pytest.mark.parametrize("change", ["site", "revision", "origin", "tensor", "fingerprint", "rank"])
def test_control_scope_rank_corruption_rejected_before_source_reads(change):
    reader, bank = fixture()
    basis = random_basis(bank)
    control = quantize_reader(reader, bits=4, block_size=4)
    reader.calls.clear()
    if change == "site":
        basis["scope"]["state_site"] = "decoder_block_output"
    elif change == "revision":
        basis["scope"]["teacher"]["revision"] = "f" * 40
    elif change == "origin":
        basis["origin"] = "measured_model_state"
    elif change == "tensor":
        basis["basis"] *= 2
    elif change == "fingerprint":
        basis["evidence_sha256"] = "0" * 64
    else:
        basis["actual_rank"] = 1
    with pytest.raises(ValueError):
        scoped_fit(reader, basis, control)
    assert reader.calls == []


def test_control_construction_and_fit_budgets_fail_before_source_reads():
    reader, bank = fixture()
    with pytest.raises(MemoryError):
        random_basis(bank, max_working_bytes=1)
    with pytest.raises(MemoryError):
        pca_basis(reader, bank, max_working_bytes=1)
    control = quantize_reader(reader, bits=4, block_size=4)
    reader.calls.clear()
    with pytest.raises(MemoryError):
        scoped_fit(reader, random_basis(bank), control, max_working_bytes=1)
    assert reader.calls == []


def test_control_cannot_smuggle_fake_probe_or_heldout_evidence_before_reads():
    reader, bank = fixture()
    control = quantize_reader(reader, bits=4, block_size=4)
    reader.calls.clear()
    random = random_basis(bank)
    random["probe_ids"] = ["p_fake"]
    random["evidence_sha256"] = residual_basis_fingerprint(random)
    with pytest.raises(ValueError, match="measured probe"):
        scoped_fit(reader, random, control)
    pca = pca_basis(reader, bank)
    pca["excluded_heldout_record_ids"] = pca["training_record_ids"].copy()
    pca["evidence_sha256"] = residual_basis_fingerprint(pca)
    with pytest.raises(ValueError, match="heldout"):
        scoped_fit(reader, pca, control)
    assert reader.calls == []


@pytest.mark.parametrize("kind", ["pca", "random"])
def test_control_safe_roundtrip_retains_non_probe_evidence_and_real_bytes(tmp_path, kind):
    reader, bank = fixture()
    control = quantize_reader(reader, bits=4, block_size=4)
    basis = pca_basis(reader, bank) if kind == "pca" else random_basis(bank)
    candidate = scoped_fit(reader, basis, control)
    directory = tmp_path / kind
    receipt = save_probe_residual(candidate, directory)
    loaded = load_probe_residual(directory, expected_identity=reader.identity)
    assert loaded.metadata["objective"] == candidate.metadata["objective"]
    assert loaded.metadata["objective"]["feature_origin"] == basis["feature_origin"]
    assert receipt["serialized_total_bytes"] == sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    torch.testing.assert_close(loaded.read_columns(0, 8), candidate.read_columns(0, 8), atol=0, rtol=0)


def test_serialized_control_cannot_claim_a_different_feature_origin(tmp_path):
    reader, bank = fixture()
    control = quantize_reader(reader, bits=4, block_size=4)
    candidate = scoped_fit(reader, random_basis(bank), control)
    directory = tmp_path / "random"
    save_probe_residual(candidate, directory)
    path = directory / "metadata.json"
    metadata = json.loads(path.read_bytes())
    metadata["objective"]["feature_origin"] = "derived_native_probe_seed_contrasts"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="feature origin"):
        load_probe_residual(directory, expected_identity=reader.identity)


def hybrid_basis(reader, bank, records=None, **kwargs):
    if records is None:
        records = calibration_records(reader, [[1, 0, 1, 0], [0, 1, -2, 0],
            [2, 1, 3, 0], [-1, 2, -1, 0]])
    return make_probe_pca_complement_basis(records, bank, expected_scope=records[0].scope,
        extra_rank=kwargs.pop("extra_rank", 1), expected_origin="synthetic_fixture", **kwargs)


def test_hybrid_retains_native_prefix_and_repairs_signed_complement_with_same_int4():
    reader, bank = fixture()
    basis = hybrid_basis(reader, bank)
    protected, _ = _span(bank, 1e-5)
    torch.testing.assert_close(basis["basis"][:, :2], protected, atol=0, rtol=0)
    torch.testing.assert_close(basis["basis"].T @ basis["basis"], torch.eye(3), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(protected.T @ basis["basis"][:, 2:], torch.zeros(2, 1), atol=1e-6, rtol=0)
    assert basis["protected_probe_rank"] == 2 and basis["extra_rank"] == 1
    assert basis["available_complement_rank"] == 1 and basis["rank_matches_request"] is True
    assert basis["basis_kind"] == "probe_pca_complement_span"
    assert basis["feature_origin"] == "derived_native_probe_span_and_calibration_complement"
    assert basis["native_probe_evidence_sha256"] == bank["evidence_sha256"]
    assert basis["origin"] == "synthetic_fixture" and basis["heldout_used_for_fit"] is False
    assert basis["semantic_meanings_validated"] is False and "probe_ids" not in basis
    control = quantize_reader(reader, bits=4, block_size=4)
    before_codes, before_scales = control.codes.clone(), control.scales.clone()
    repaired = scoped_fit(reader, basis, control)
    assert repaired.base_quantization is control
    torch.testing.assert_close(control.codes, before_codes, atol=0, rtol=0)
    torch.testing.assert_close(control.scales, before_scales, atol=0, rtol=0)
    decoded = repaired.read_columns(0, 8).reshape(3, 2, 4)
    for value in ([.75, -1.25, -2., 0.], [-2., .5, .25, 0.]):
        inputs = torch.tensor(value)
        torch.testing.assert_close(decoded @ inputs, reader.weights @ inputs, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(decoded[:, :, 3], control.read_columns(0, 8).reshape(3, 2, 4)[:, :, 3], atol=0, rtol=0)
    report = evaluate(repaired, reader, bank)
    assert report["same_probe_bank_used_to_fit"] is True
    assert report["contains_protected_native_probe_span"] is True
    assert report["span_response_relative_rmse"] < 1e-6
    assert report["temporal_transition_evaluated"] is False and report["native_fp8_kernel_evaluated"] is False


def test_hybrid_selects_top_uncentered_complement_and_equal_rank_controls_have_equal_payload(tmp_path):
    reader, bank = fixture()
    records = calibration_records(reader, [[20, 0, 10, 1], [20, 1, 10, -1],
        [-20, 0, 10, 1], [-20, -1, 10, -1]])
    hybrid = hybrid_basis(reader, bank, records)
    torch.testing.assert_close(hybrid["basis"][:, 2:].abs(), torch.tensor([[0.], [0.], [1.], [0.]]), atol=1e-6, rtol=1e-6)
    assert hybrid["available_complement_rank"] == 2
    assert "uncentered unweighted" in hybrid["pca_objective"]
    control = quantize_reader(reader, bits=4, block_size=4)
    candidates = [scoped_fit(reader, value, control) for value in (
        hybrid, pca_basis(reader, bank, records, requested_rank=3), random_basis(bank, requested_rank=3, seed=31))]
    assert len({candidate.metadata["factor_tensor_bytes"] for candidate in candidates}) == 1
    native = fit(reader, bank, base_quantization=control)
    assert candidates[0].metadata["correction_tensor_bytes"] > native.metadata["correction_tensor_bytes"]
    for index, candidate in enumerate(candidates):
        assert candidate.span_basis.shape[1] == 3 and candidate.base_quantization is control
        directory = tmp_path / str(index)
        receipt = save_probe_residual(candidate, directory)
        assert receipt["serialized_total_bytes"] == sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        loaded = load_probe_residual(directory, expected_identity=reader.identity)
        torch.testing.assert_close(loaded.read_columns(0, 8), candidate.read_columns(0, 8), atol=0, rtol=0)
    report = evaluate(candidates[0], reader, bank)
    assert report["fitted_basis_kind"] == "probe_pca_complement_span"


@pytest.mark.parametrize("directions", [
    [[-1., 0, 0, 0], [0, -1., 0, 0]],
    [[.6, .8, 0, 0], [-.8, .6, 0, 0]],
    [[1., 0, 0, 0], [0, 1., 0, 0], [1., 0, 0, 0], [-1., 0, 0, 0]],
])
def test_hybrid_projector_and_repair_invariant_to_native_sign_rotation_duplicates(directions):
    reader, bank = fixture()
    _, transformed = fixture(directions)
    original, changed = hybrid_basis(reader, bank), hybrid_basis(reader, transformed)
    assert original["actual_rank"] == changed["actual_rank"] == 3
    assert original["protected_probe_rank"] == changed["protected_probe_rank"] == 2
    assert original["native_probe_evidence_sha256"] != changed["native_probe_evidence_sha256"]
    first, second = original["basis"], changed["basis"]
    torch.testing.assert_close(first @ first.T, second @ second.T, atol=2e-6, rtol=2e-6)
    control = quantize_reader(reader, bits=4, block_size=4)
    torch.testing.assert_close(scoped_fit(reader, original, control).read_columns(0, 8),
        scoped_fit(reader, changed, control).read_columns(0, 8), atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("leak", ["record", "trajectory"])
def test_hybrid_rejects_heldout_record_and_trajectory_leakage(leak):
    reader, bank = fixture()
    train = calibration_records(reader, [[1, 0, 1, 0], [0, 1, -2, 0]])
    heldout = calibration_records(reader, [[0, 0, 0, 1]], trajectory="test")
    heldout[0] = CalibrationRecord(train[0].record_id if leak == "record" else "heldout", "env",
        train[0].trajectory_id if leak == "trajectory" else "test", train[0].scope, None, origin="synthetic_fixture")
    with pytest.raises(ValueError, match="heldout"):
        hybrid_basis(reader, bank, train, heldout_records=heldout)
    assert reader.calls == []


def test_hybrid_does_not_read_heldout_values_or_use_them_in_evidence():
    reader, bank = fixture()
    heldout = calibration_records(reader, [[1e10, -1e10, 1e10, -1e10]], trajectory="test")
    first = hybrid_basis(reader, bank, heldout_records=heldout)
    heldout[0] = CalibrationRecord(heldout[0].record_id, "env", "test", heldout[0].scope, None, origin="synthetic_fixture")
    second = hybrid_basis(reader, bank, heldout_records=heldout)
    assert first["evidence_sha256"] == second["evidence_sha256"]
    torch.testing.assert_close(first["basis"], second["basis"], atol=0, rtol=0)


def test_hybrid_rejects_insufficient_spectrum_instead_of_padding_projection_noise():
    reader, bank = fixture()
    with pytest.raises(ValueError, match="numerical rank 1.*no padding"):
        hybrid_basis(reader, bank, extra_rank=2)
    native_states = calibration_records(reader, [[1, 2, 0, 0], [3, -2, 0, 0]])
    with pytest.raises(ValueError, match="numerical rank 0"):
        hybrid_basis(reader, bank, native_states)
    _, oblique_bank = fixture([[.5, .5, .5, .5]])
    # These states have only native action; FP32 projector residue is not new data.
    oblique = calibration_records(reader, [[1., 1., 1., 1.], [3., 3., 3., 3.]])
    with pytest.raises(ValueError, match="numerical rank 0"):
        hybrid_basis(reader, oblique_bank, oblique)
    assert reader.calls == []


@pytest.mark.parametrize("kwargs", [{"extra_rank": 0}, {"extra_rank": 3},
    {"max_working_bytes": 1}, {"max_axes": 1}, {"max_records": 1}, {"svd_rtol": 0}])
def test_hybrid_bounds_fail_before_source_weight_reads(kwargs):
    reader, bank = fixture()
    with pytest.raises((ValueError, MemoryError)):
        hybrid_basis(reader, bank, **kwargs)
    assert reader.calls == []


@pytest.mark.parametrize("change", ["site", "revision", "origin", "fingerprint", "heldout", "directions"])
def test_hybrid_native_evidence_scope_corruption_rejected(change):
    reader, bank = fixture()
    if change == "site":
        bank["scope"]["state_site"] = "decoder_block_output"
        rehash(bank)
    elif change == "revision":
        bank["scope"]["teacher"]["revision"] = "f" * 40
        rehash(bank)
    elif change == "origin":
        bank["origin"] = "measured_model_state"
        rehash(bank)
    elif change == "fingerprint":
        bank["evidence_sha256"] = "0" * 64
    elif change == "heldout":
        bank["heldout_used_to_fit_axes"] = True
        rehash(bank)
    else:
        bank["directions"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        hybrid_basis(reader, bank)
    assert reader.calls == []


@pytest.mark.parametrize("corruption", ["prefix", "native_scope", "protected_rank", "available_rank"])
def test_hybrid_corruption_rejected_before_weights_even_with_rehashed_basis(corruption):
    reader, bank = fixture()
    basis = hybrid_basis(reader, bank)
    control = quantize_reader(reader, bits=4, block_size=4)
    reader.calls.clear()
    if corruption == "prefix":
        basis["basis"] = basis["basis"][:, [2, 1, 0]].contiguous()  # Remains orthonormal.
    elif corruption == "native_scope":
        basis["native_probe_scope_sha256"] = "f" * 64
    elif corruption == "protected_rank":
        basis["protected_probe_rank"] = 1
    else:
        basis["available_complement_rank"] = 4
    basis["evidence_sha256"] = residual_basis_fingerprint(basis)
    with pytest.raises(ValueError, match="hybrid"):
        scoped_fit(reader, basis, control)
    assert reader.calls == []


@pytest.mark.parametrize("corruption", ["protected_basis_sha256", "native_probe_scope_sha256", "available_complement_rank"])
def test_saved_hybrid_protection_and_scope_checks_cannot_be_corrupted(tmp_path, corruption):
    reader, bank = fixture()
    control = quantize_reader(reader, bits=4, block_size=4)
    candidate = scoped_fit(reader, hybrid_basis(reader, bank), control)
    directory = tmp_path / "hybrid"
    save_probe_residual(candidate, directory)
    path = directory / "metadata.json"
    raw = path.read_bytes()
    old = candidate.metadata["objective"][corruption]
    new = "f" * 64 if isinstance(old, str) else 9
    # Preserve byte lengths/accounting so the relevant protection check is reached.
    before, after = json.dumps({corruption: old})[1:-1], json.dumps({corruption: new})[1:-1]
    assert len(before) == len(after) and before.encode() in raw
    path.write_bytes(raw.replace(before.encode(), after.encode()))
    with pytest.raises(ValueError, match="hybrid"):
        load_probe_residual(directory, expected_identity=reader.identity)
