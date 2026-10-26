"""Native-site selection and actual serialized proxy comparisons, CPU fixtures."""
from dataclasses import replace
import json
import math

import pytest
import torch

from scripts import calibrate_glm_compression as trial
from tda.expert_compress import ExpertGroupIdentity, TensorExpertReader, fit_expert_group
from tda.glm_capture_archive import CaptureArchiveWriter
from tda.trajectory_compress import (InputStateScope, CalibrationRecord, TrajectoryPair,
                                    evaluate_trajectory_group)


def fixture(tmp_path, *, heldout_scale=1.0, probe_seed=False, routers=False, token_gap=1):
    teacher = {"model_id": "tiny", "revision": "a" * 40, "hidden_size": 2,
        "vocab_size": 4, "num_layers": 4, "source_precision": "float32",
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64}
    identity = ExpertGroupIdentity("tiny", "a" * 40, 3, "gate_proj", "row-major",
        ("0", "1", "2", "3"), (2, 2), "float32")
    weights = torch.zeros(4, 2, 2)
    weights[:, 0, 0] = 100 * torch.tensor([-1., -1., 1., 1.])
    weights[:, 1, 1] = torch.tensor([-1., 1., -1., 1.])
    source = TensorExpertReader(identity, weights)
    root = tmp_path / "archive"
    writer = CaptureArchiveWriter(root, teacher, origin="synthetic_fixture")
    sites = {"final_hidden": "post_final_norm_lm_head_input", "layer_3_expert_input": "post_attention_norm_expert_gate_up_input"}
    for split in ("calibration", "heldout"):
        states = torch.tensor([[1., 0.], [1., 1.]])
        if split == "heldout":
            states *= heldout_scale
        rows = [{"id": f"{split}:{i}", "trajectory_id": split, "environment_id": "arithmetic",
            "split": split, "token_position": i * token_gap, "next_token_id": i,
            "provenance_weight": 0.75, "stimulation_weight": 0.5} for i in range(2)]
        tensors = {"final_hidden": states, "layer_3_expert_input": states.clone(),
            "positions": torch.tensor([0, token_gap]), "next_token_ids": torch.arange(2)}
        if routers:
            tensors.update(layer_3_router_ids=torch.tensor([[0, 9], [1, 9]]),
                layer_3_router_weights=torch.tensor([[0.5, 1.5], [1., 1.]]))
        writer.write_chunk(split, tensors,
            {"rows": rows, "pairs": [[0, 1]], "state_sites": sites})
    if probe_seed:
        states = torch.tensor([[0., -1.], [0., 1.]])
        rows = [{"id": f"seed:{i}", "trajectory_id": f"seed-{i}", "environment_id": "seed",
            "split": "calibration", "kind": "probe_framing", "probe_id": "p_seed",
            "polarity": -1 if i == 0 else 1, "token_position": 0, "next_token_id": -1,
            "next_token_observed": False} for i in range(2)]
        writer.write_chunk("seed", {"final_hidden": states, "layer_3_expert_input": states.clone(),
            "positions": torch.tensor([0, 0]), "next_token_ids": torch.tensor([-1, -1])},
            {"rows": rows, "state_sites": sites})
    writer.finalize("partial")
    return source, root


def test_whole_trajectories_keep_pairs_and_skip_oversized_groups():
    identity = ExpertGroupIdentity("fixture", "revision", 3, "gate_proj", "row-major", ("0", "1"), (2, 2), "float32")
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm_expert_gate_up_input")
    records = [CalibrationRecord(f"{name}:{i}", "environment", name, scope, torch.tensor([1., i]),
        provenance_weight=0.75, stimulation_weight=0.5, origin="synthetic_fixture")
        for name, size in (("large", 5), ("first", 2), ("second", 2)) for i in range(size)]
    pairs = [TrajectoryPair(name, f"{name}:0", f"{name}:1") for name in ("large", "first", "second")]
    chosen, transitions, selection = trial.select_trajectories(records, pairs, max_records=3)
    assert [r.record_id for r in chosen] == ["first:0", "first:1"]
    assert transitions == [pairs[1]]
    assert selection["skipped_trajectory_ids"] == ["large", "second"]
    assert all(r.provenance_weight == 0.75 and r.stimulation_weight == 0.5 for r in chosen)


def test_saved_equal_rank_comparison_separates_weight_and_heldout_transition_error(tmp_path):
    source, archive = fixture(tmp_path)
    out = tmp_path / "comparison"
    reports = trial.compare(source, archive, out, ranks=(1,), storage_dtype="float32",
        max_train=2, max_heldout=2, row_chunk=1, point_weight=0, contrast_weight=1, include_int4=False)
    pca, response = reports
    assert pca["factor_tensor_bytes"] == response["factor_tensor_bytes"]
    assert pca["weight_error"]["relative_frobenius_error"] < response["weight_error"]["relative_frobenius_error"]
    assert pca["heldout_response"]["contrast_relative_rmse"] > 0.99
    assert response["heldout_response"]["contrast_relative_rmse"] < 1e-5
    for report in reports:
        directory = out / report["variant"]
        assert report["serialized_total_bytes"] == sum(p.stat().st_size for p in directory.iterdir())
        assert not report["full_decoder_evaluated"] and not report["language_quality_established"]
        assert not report["heldout_used_for_fit"]
    evidence = json.loads((out / "evidence.json").read_text())
    assert evidence["origin"] == "synthetic_fixture"
    assert evidence["input_state_scope"]["identity_sha256"] == source.identity.fingerprint
    assert all(r["environment_id"] == "arithmetic" and r["stimulation_weight"] == 0.5 for r in evidence["records"])
    assert [r["token_position"] for r in evidence["records"]] == [0, 1, 0, 1]
    assert evidence["pairs"]["heldout"][0]["token_gap"] == 1
    assert "not per-token velocity" in evidence["transition_input"]
    assert not evidence["router_coverage"]["available"]
    assert json.loads((out / "status.json").read_text())["status"] == "finished"


def test_heldout_values_never_affect_fitted_coefficients(tmp_path):
    source, archive = fixture(tmp_path / "first", heldout_scale=1)
    _, other_archive = fixture(tmp_path / "second", heldout_scale=17)
    for folder, evidence in ((tmp_path / "fit_a", archive), (tmp_path / "fit_b", other_archive)):
        trial.compare(source, evidence, folder, ranks=(1,), storage_dtype="float32",
            max_train=2, max_heldout=2, row_chunk=1, point_weight=0, include_int4=False)
    first = torch.load(tmp_path / "fit_a/trajectory_response_rank_1/factors.pt", weights_only=True)
    second = torch.load(tmp_path / "fit_b/trajectory_response_rank_1/factors.pt", weights_only=True)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_generic_int4_metric_matches_factor_evaluation(tmp_path):
    source, archive = fixture(tmp_path)
    scope, train, pairs, test, transitions, _ = trial.load_samples(archive, source.identity)
    encoded = fit_expert_group(source, 1, storage_dtype="float32", chunk_columns=2)
    expected = evaluate_trajectory_group(encoded, source, test, transitions, expected_scope=scope, row_chunk=1)
    actual = trial.evaluate_reader(source, trial.FactorReader(encoded), test, transitions, scope=scope, row_chunk=1)
    assert actual["point_relative_rmse"] == pytest.approx(expected["point_relative_rmse"])
    assert actual["contrast_relative_rmse"] == pytest.approx(expected["contrast_relative_rmse"])
    int4 = trial.quantize_reader(source, bits=4, block_size=128)
    control = trial.evaluate_reader(source, int4, test, transitions, scope=scope, row_chunk=1)
    assert control["point_relative_rmse"] >= 0
    assert control["contrast_relative_rmse"] >= 0


def test_actual_int4_bytes_and_zero_rank_control_are_saved(tmp_path):
    source, archive = fixture(tmp_path)
    reports = trial.compare(source, archive, tmp_path / "results", ranks=(0,), max_train=2,
        max_heldout=2, row_chunk=1, storage_dtype="float32")
    assert [r["variant"] for r in reports] == ["weight_pca_rank_0", "trajectory_response_rank_0", "symmetric_int4"]
    assert reports[0]["heldout_response"]["point_relative_rmse"] == reports[1]["heldout_response"]["point_relative_rmse"]
    assert all(r["serialized_total_bytes"] > r["factor_tensor_bytes"] for r in reports)


def test_wrong_teacher_precision_and_budget_rejected_before_tensor_loading(tmp_path, monkeypatch):
    source, archive = fixture(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail("tensor payload opened before metadata preflight rejected")
    monkeypatch.setattr(trial, "gate_calibration_records", unexpected)
    with pytest.raises(MemoryError, match="candidate states"):
        trial.load_samples(archive, source.identity, max_working_bytes=1)
    identity = replace(source.identity, source_dtype="bfloat16")
    with pytest.raises(ValueError, match="precision differs"):
        trial.load_samples(archive, identity)


def test_selection_too_small_cannot_silently_truncate_a_trajectory(tmp_path):
    source, archive = fixture(tmp_path)
    with pytest.raises(ValueError, match="complete calibration and heldout trajectories"):
        trial.load_samples(archive, source.identity, max_train=1)


def test_output_never_overwritten(tmp_path):
    source, archive = fixture(tmp_path)
    out = tmp_path / "occupied"
    out.mkdir()
    with pytest.raises(FileExistsError):
        trial.compare(source, archive, out)


def test_static_probe_span_is_separate_from_true_temporal_evidence(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    out = tmp_path / "probe_comparison"
    reports = trial.compare(source, archive, out, ranks=(1,), storage_dtype="float32",
        max_train=2, max_heldout=2, row_chunk=1, include_int4=False, include_probe_span=True)
    pca, temporal, span = reports
    assert [r["method"] for r in reports] == ["weight_pca", "trajectory_response", "probe_span"]
    assert pca["static_seed_contrast_response"]["span_response_relative_rmse"] > 0.99
    assert span["static_seed_contrast_response"]["span_response_relative_rmse"] < 1e-5
    assert not span["static_seed_contrast_response"]["temporal_transition_evaluated"]
    assert span["heldout_response"]["pairs"] == 1
    assert all(r["factor_tensor_bytes"] == pca["factor_tensor_bytes"] for r in reports)
    evidence = json.loads((out / "evidence.json").read_text())
    assert len(evidence["records"]) == 4
    assert not any(r["record_id"].startswith("seed:") for r in evidence["records"])
    bank = torch.load(out / "native_probe_bank.pt", weights_only=True)
    assert bank["probe_ids"] == ["p_seed"] and not bank["semantic_meanings_validated"]


def test_zero_energy_transitions_report_undefined_without_stopping_comparison(tmp_path):
    source, archive = fixture(tmp_path)
    # Zero source transition energy is distinct from a zero-error success.
    identity = source.identity
    weights = source.weights.clone()
    weights[:, 1, 1] = 0
    reader = TensorExpertReader(identity, weights)
    reports = trial.compare(reader, archive, tmp_path / "zero_results", ranks=(0,),
        max_train=2, max_heldout=2, row_chunk=1, include_int4=False, storage_dtype="float32")
    assert all(r["heldout_response"]["contrast_relative_rmse"] is None for r in reports)


def test_router_coverage_reports_real_choice_and_scaled_weight_shares_without_changing_fit(tmp_path):
    source, archive = fixture(tmp_path, routers=True)
    _, _, _, _, _, evidence = trial.load_samples(archive, source.identity)
    coverage = evidence["router_coverage"]
    assert coverage["available"] and not coverage["fit_uses_router_weights"]
    for split in ("calibration", "heldout"):
        item = coverage[split]
        assert item["positions"] == 2 and item["choices"] == 4
        assert item["sample_choice_fraction"] == 0.5
        assert item["sample_position_fraction"] == 1.0
        assert item["sample_routed_weight_share"] == pytest.approx(0.375)
        assert item["per_expert_choices"] == {"0": 1, "1": 1, "2": 0, "3": 0}
        assert item["routed_weight"] == 4.0


def test_probe_residual_requires_the_uncorrected_int4_control(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    with pytest.raises(ValueError, match="identical uncorrected INT4 control"):
        trial.compare(source, archive, tmp_path / "invalid", include_probe_residual=True, include_int4=False)
    assert not (tmp_path / "invalid").exists()


def test_native_span_residual_repairs_int4_without_shared_expert_collapse(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    out = tmp_path / "residual_comparison"
    reports = trial.compare(source, archive, out, ranks=(0, 1), storage_dtype="float32",
        max_train=2, max_heldout=2, row_chunk=1, include_probe_span=True, include_probe_residual=True)
    assert [r["method"] for r in reports] == ["weight_pca", "trajectory_response", "probe_span",
        "weight_pca", "trajectory_response", "probe_span", "symmetric_int4", "probe_residual"]
    uncorrected, repaired = reports[-2:]
    assert uncorrected["static_seed_contrast_response"]["span_response_relative_rmse"] > 0.99
    assert repaired["static_seed_contrast_response"]["span_response_relative_rmse"] < 1e-5
    assert repaired["heldout_response"]["point_relative_rmse"] < uncorrected["heldout_response"]["point_relative_rmse"]
    assert repaired["shared_expert_rank"] is None and repaired["rank"] is None
    assert repaired["probe_span_rank"] == 1 and repaired["coefficient_storage_dtype"] == "float32"
    assert repaired["base_tensor_bytes"] == uncorrected["factor_tensor_bytes"]
    assert repaired["factor_tensor_bytes"] == repaired["base_tensor_bytes"] + repaired["correction_tensor_bytes"]
    assert repaired["correction_tensor_bytes"] > 0
    assert not repaired["full_decoder_evaluated"] and not repaired["heldout_used_for_fit"]
    from tda.probe_residual_compress import load_probe_residual
    decoded = load_probe_residual(out / "probe_residual", expected_identity=source.identity)
    base_quantization = trial.load_block_quantization(out / "symmetric_int4", expected_identity=source.identity)
    # The exact same INT4 codes/scales form the base; x lies off the seed y span.
    torch.testing.assert_close(decoded.base_quantization.codes, base_quantization.codes, rtol=0, atol=0)
    torch.testing.assert_close(decoded.base_quantization.scales, base_quantization.scales, rtol=0, atol=0)
    torch.testing.assert_close(decoded.read_columns(0, 4)[:, [0, 2]], base_quantization.read_columns(0, 4)[:, [0, 2]], rtol=0, atol=0)
    directory = out / "probe_residual"
    assert repaired["serialized_total_bytes"] == sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def test_residual_can_be_enabled_without_shared_span_variant(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    reports = trial.compare(source, archive, tmp_path / "residual_only_option", ranks=(0,),
        storage_dtype="float32", max_train=2, max_heldout=2, row_chunk=1, include_probe_residual=True)
    assert [r["method"] for r in reports] == ["weight_pca", "trajectory_response", "symmetric_int4", "probe_residual"]
    assert all("static_seed_contrast_response" in r for r in reports)


def test_nonunit_token_gap_is_explicit_and_does_not_turn_state_differences_into_velocity(tmp_path):
    source, archive = fixture(tmp_path, token_gap=3)
    scope, train, pairs, _, _, evidence = trial.load_samples(archive, source.identity)
    assert evidence["pairs"]["calibration"][0]["token_gap"] == 3
    assert "not divided by token gap" in evidence["transition_input"]
    inputs, _ = trial.trajectory._features(train, pairs)
    torch.testing.assert_close(inputs[-1], torch.tensor([0., 1.]))


def test_equal_rank_residual_controls_require_native_residual_candidate(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    with pytest.raises(ValueError, match="require the native probe residual"):
        trial.compare(source, archive, tmp_path / "invalid_controls", include_probe_residual_controls=True)
    assert not (tmp_path / "invalid_controls").exists()


def test_residual_controls_use_same_rank_base_precision_and_honest_distinct_evidence(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    out = tmp_path / "controlled_repairs"
    reports = trial.compare(source, archive, out, ranks=(0,), max_train=2, max_heldout=2,
        row_chunk=1, storage_dtype="float32", include_probe_residual=True, include_probe_residual_controls=True)
    assert [r["method"] for r in reports] == ["weight_pca", "trajectory_response", "symmetric_int4",
        "probe_residual", "residual_calibration_pca", "residual_seeded_random"]
    native, pca, random = reports[-3:]
    assert native["static_seed_contrast_response"]["span_response_relative_rmse"] < 1e-5
    assert pca["static_seed_contrast_response"]["span_response_relative_rmse"] > 0.1
    assert random["static_seed_contrast_response"]["span_response_relative_rmse"] > 0.1
    assert pca["control_basis"]["basis_kind"] == "calibration_state_second_moment_span"
    assert pca["control_basis"]["feature_origin"] == "derived_calibration_states"
    assert random["control_basis"]["basis_kind"] == "reproducible_random_span"
    assert random["control_basis"]["feature_origin"] == "synthetic_random_control"
    from tda.probe_residual_compress import load_probe_residual
    original = trial.load_block_quantization(out / "symmetric_int4", expected_identity=source.identity)
    for report in (native, pca, random):
        assert report["correction_span_rank"] == 1
        assert report["coefficient_storage_dtype"] == "float32"
        assert report["factor_tensor_bytes"] == native["factor_tensor_bytes"]
        assert report["base_tensor_bytes"] == original.metadata["factor_tensor_bytes"]
        assert report["heldout_response"]["records"] == 2 and not report["heldout_used_for_fit"]
        directory = out / report["variant"]
        assert report["serialized_total_bytes"] == sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        loaded = load_probe_residual(directory, expected_identity=source.identity)
        torch.testing.assert_close(loaded.base_quantization.codes, original.codes, rtol=0, atol=0)
        torch.testing.assert_close(loaded.base_quantization.scales, original.scales, rtol=0, atol=0)
    for report in (pca, random):
        assert report["rank_matched_to_variant"] == "probe_residual"
        assert report["tensor_bytes_match_native_repair"]
        assert report["serialized_total_bytes_difference_from_native_repair"] == report["serialized_total_bytes"] - native["serialized_total_bytes"]
        assert report["control_basis"]["requested_rank"] == report["control_basis"]["actual_rank"] == 1
        assert report["control_basis"]["rank_matches_request"]
        assert not report["static_seed_contrast_response"]["same_probe_bank_used_to_fit"]
        assert not {"probe_ids", "pole_record_ids", "probe_evidence_sha256"} & report["fit_evidence"].keys()
    assert pca["fit_evidence"]["training_record_ids"] == ["calibration:0", "calibration:1"]
    assert pca["fit_evidence"]["excluded_heldout_record_ids"] == ["heldout:0", "heldout:1"]
    assert random["fit_evidence"]["training_record_ids"] == []


def test_control_corrections_are_deterministic_and_do_not_fit_heldout_values(tmp_path):
    source, archive = fixture(tmp_path / "first_control", probe_seed=True, heldout_scale=1)
    _, second_archive = fixture(tmp_path / "second_control", probe_seed=True, heldout_scale=17)
    for folder, evidence in ((tmp_path / "fit_control_a", archive), (tmp_path / "fit_control_b", second_archive)):
        trial.compare(source, evidence, folder, ranks=(0,), max_train=2, max_heldout=2,
            row_chunk=1, storage_dtype="float32", include_probe_residual=True,
            include_probe_residual_controls=True, random_control_seed=19)
    for method in ("probe_residual", "residual_calibration_pca", "residual_seeded_random"):
        first = torch.load(tmp_path / "fit_control_a" / method / "correction.pt", weights_only=True)
        second = torch.load(tmp_path / "fit_control_b" / method / "correction.pt", weights_only=True)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_lower_rank_calibration_control_is_rejected_before_fitting(tmp_path, monkeypatch):
    source, archive = fixture(tmp_path, probe_seed=True)
    import tda.probe_residual_compress as residual
    monkeypatch.setattr(residual, "make_calibration_pca_basis", lambda *args, **kwargs: {"rank_matches_request": False})
    def unexpected(*args, **kwargs):
        pytest.fail("a rank-mismatched control was fit")
    monkeypatch.setattr(residual, "fit_scoped_residual_basis", unexpected)
    out = tmp_path / "rank_mismatch"
    with pytest.raises(ValueError, match="cannot match the native probe rank"):
        trial.compare(source, archive, out, ranks=(0,), max_train=2, max_heldout=2,
            row_chunk=1, include_probe_residual=True, include_probe_residual_controls=True)
    assert json.loads((out / "status.json").read_text())["status"] == "failed"
    assert json.loads((out / "results.json").read_text())[-1]["method"] == "probe_residual"


def test_random_control_seed_is_reproducible_and_changes_the_generated_span(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    from tda.glm_capture_calibration import native_probe_bank
    from tda.probe_residual_compress import make_random_basis
    bank = native_probe_bank(archive, tensor_name="layer_3_expert_input")
    options = {"requested_rank": 1, "expected_origin": "synthetic_fixture"}
    first = make_random_basis(bank, seed=19, **options)
    repeated = make_random_basis(bank, seed=19, **options)
    changed = make_random_basis(bank, seed=20, **options)
    torch.testing.assert_close(first["basis"], repeated["basis"], rtol=0, atol=0)
    assert first["evidence_sha256"] == repeated["evidence_sha256"]
    assert not torch.allclose(first["basis"] @ first["basis"].T, changed["basis"] @ changed["basis"].T)


def test_full_heldout_evaluation_preserves_global_trajectory_quality_weighting():
    identity = ExpertGroupIdentity("wide-corpus-fixture", "revision", 3, "gate_proj", "row-major",
        ("0", "1"), (2, 2), "float32")
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm_expert_gate_up_input")
    source = TensorExpertReader(identity, torch.eye(2).repeat(2, 1, 1))
    decoded = TensorExpertReader(identity, torch.diag(torch.tensor([1., 0.])).repeat(2, 1, 1))
    records, pairs = [], []
    # The longer trajectory has lower evidence quality and errors only in its
    # constant coordinate. The shorter one loses every point and transition.
    # A row mean or separately normalized batch average gives a different result.
    for name, count, provenance, stimulation in (("long", 2048, 0.25, 0.5), ("short", 384, 1.0, 0.75)):
        for i in range(count):
            state = torch.tensor([float(i + 1), 1.]) if name == "long" else torch.tensor([0., float(i + 1)])
            records.append(CalibrationRecord(f"{name}:{i}", "arithmetic", name, scope, state,
                provenance_weight=provenance, stimulation_weight=stimulation, origin="synthetic_fixture"))
            if i:
                pairs.append(TrajectoryPair(f"{name}:pair:{i}", f"{name}:{i-1}", f"{name}:{i}"))
    report = trial.evaluate_reader(source, decoded, records, pairs, scope=scope, row_chunk=1)
    long_point_mean = sum(1 / (i * i + 1) for i in range(1, 2049)) / 2048
    expected_point = math.sqrt((0.125 * long_point_mean + 0.75) / 0.875)
    expected_contrast = math.sqrt(0.75 / 0.875)
    assert report["records"] == 2432 and report["pairs"] == 2430
    assert report["point_relative_rmse"] == pytest.approx(expected_point, abs=1e-12)
    assert report["contrast_relative_rmse"] == pytest.approx(expected_contrast, abs=1e-12)
    assert report["contrast_relative_rmse"] > 2 * math.sqrt(383 / 2430)


@pytest.mark.parametrize(("record_count", "pair_count"), [(8193, 0), (2, 4097)])
def test_evaluation_count_limits_reject_before_state_cloning(record_count, pair_count, monkeypatch):
    identity = ExpertGroupIdentity("bounded", "revision", 3, "gate_proj", "row-major",
        ("0", "1"), (2, 2), "float32")
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm_expert_gate_up_input")
    source = TensorExpertReader(identity, torch.eye(2).repeat(2, 1, 1))
    def unexpected(*args, **kwargs):
        pytest.fail("states cloned before count preflight rejected")
    monkeypatch.setattr(trial.trajectory, "_snapshot", unexpected)
    with pytest.raises(MemoryError, match="native capture limits"):
        trial.evaluate_reader(source, source, [None] * record_count, [None] * pair_count, scope=scope)


def test_evaluation_byte_budget_rejects_before_state_cloning(tmp_path, monkeypatch):
    source, archive = fixture(tmp_path)
    scope, _, _, records, pairs, _ = trial.load_samples(archive, source.identity)
    def unexpected(*args, **kwargs):
        pytest.fail("states cloned before byte preflight rejected")
    monkeypatch.setattr(trial.trajectory, "_snapshot", unexpected)
    with pytest.raises(MemoryError, match="working-byte budget"):
        trial.evaluate_reader(source, source, records, pairs, scope=scope, max_working_bytes=1)
    for invalid in (True, 1.5, 0):
        with pytest.raises(ValueError, match="positive integer"):
            trial.evaluate_reader(source, source, records, pairs, scope=scope, max_working_bytes=invalid)


@pytest.mark.parametrize(("max_train", "max_heldout"), [(1024, 2432), (8192, 8192)])
def test_cli_accepts_complete_capture_position_budgets(max_train, max_heldout, monkeypatch, tmp_path):
    calls = []
    class SourceContext:
        def __enter__(self):
            return "source"
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(trial, "GLMWeightSampleReader", lambda path: SourceContext())
    monkeypatch.setattr(trial.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(trial, "compare", lambda *args, **kwargs: calls.append(kwargs))
    assert trial.main(["--sample", "sample", "--archive", "archive", "--out", str(tmp_path / "out"),
        "--max-train", str(max_train), "--max-heldout", str(max_heldout)]) == 0
    assert calls[0]["max_train"] == max_train
    assert calls[0]["max_heldout"] == max_heldout


@pytest.mark.parametrize("option", ["--max-train", "--max-heldout"])
@pytest.mark.parametrize("value", ["8193", "1", "true", "1.5"])
def test_cli_rejects_overcap_or_noninteger_budgets_before_source_open(option, value, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("source opened before invalid CLI budget rejected")
    monkeypatch.setattr(trial, "GLMWeightSampleReader", unexpected)
    with pytest.raises(SystemExit) as failure:
        trial.main(["--sample", "sample", "--archive", "archive", "--out", "out", option, value])
    assert failure.value.code == 2


@pytest.mark.parametrize("budget", [True, 1.5, 8193, 0])
def test_library_position_budgets_reject_before_archive_read(budget, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("archive opened before invalid position budget rejected")
    monkeypatch.setattr(trial, "load_archive_manifest", unexpected)
    with pytest.raises(ValueError, match="integer in 1..8192"):
        trial.load_samples("unused", None, max_train=budget)
    with pytest.raises(ValueError, match="integer in 1..8192"):
        trial.load_samples("unused", None, max_heldout=budget)


@pytest.mark.parametrize("options", [
    {"include_probe_hybrid": True},
    {"include_probe_hybrid": True, "include_probe_residual": True, "include_int4": False},
    {"hybrid_extra_rank": 1},
    {"include_probe_hybrid": True, "include_probe_residual": True, "hybrid_extra_rank": 0},
    {"include_probe_hybrid": True, "include_probe_residual": True, "hybrid_extra_rank": True},
    {"include_probe_hybrid": True, "include_probe_residual": True, "hybrid_extra_rank": 1.5},
    {"include_probe_hybrid": True, "include_probe_residual": True, "hybrid_extra_rank": 2},
])
def test_hybrid_requirements_and_extra_rank_reject_before_evidence_reads(tmp_path, monkeypatch, options):
    source, _ = fixture(tmp_path)
    def unexpected(*args, **kwargs):
        pytest.fail("archive read before hybrid option validation")
    monkeypatch.setattr(trial, "load_samples", unexpected)
    out = tmp_path / "invalid_hybrid"
    with pytest.raises(ValueError):
        trial.compare(source, "unused", out, **options)
    assert not out.exists()


def test_hybrid_and_extended_controls_preserve_seed_span_and_match_higher_rank_bytes(tmp_path):
    source, archive = fixture(tmp_path, probe_seed=True)
    out = tmp_path / "hybrid_comparison"
    reports = trial.compare(source, archive, out, ranks=(0, 1, 2), max_train=2, max_heldout=2,
        row_chunk=1, storage_dtype="float32", include_probe_span=True, include_probe_residual=True,
        include_probe_residual_controls=True, include_probe_hybrid=True)
    assert len(reports) == 16
    native = next(row for row in reports if row["method"] == "probe_residual")
    hybrid, pca, random = reports[-3:]
    assert [row["variant"] for row in reports[-3:]] == [
        "probe_pca_hybrid", "residual_calibration_pca_extended", "residual_seeded_random_extended"]
    assert hybrid["protected_probe_rank"] == hybrid["extra_rank"] == 1
    assert hybrid["control_basis"]["basis_kind"] == "probe_pca_complement_span"
    assert hybrid["control_basis"]["feature_origin"] == "derived_native_probe_span_and_calibration_complement"
    assert hybrid["static_seed_contrast_response"]["same_probe_bank_used_to_fit"]
    assert hybrid["static_seed_contrast_response"]["span_response_relative_rmse"] < 1e-5
    assert pca["control_basis"]["feature_origin"] == "derived_calibration_states"
    assert random["control_basis"]["feature_origin"] == "synthetic_random_control"
    assert not pca["static_seed_contrast_response"]["same_probe_bank_used_to_fit"]
    assert not random["static_seed_contrast_response"]["same_probe_bank_used_to_fit"]
    baseline = trial.load_block_quantization(out / "symmetric_int4", expected_identity=source.identity)
    from tda.probe_residual_compress import load_probe_residual
    native_saved = load_probe_residual(out / "probe_residual", expected_identity=source.identity)
    for row in (hybrid, pca, random):
        assert row["correction_span_rank"] == 2
        assert row["coefficient_storage_dtype"] == "float32"
        assert row["base_variant"] == "symmetric_int4"
        assert row["base_tensor_bytes"] == native["base_tensor_bytes"]
        assert row["factor_tensor_bytes"] == hybrid["factor_tensor_bytes"] > native["factor_tensor_bytes"]
        assert row["tensor_bytes_difference_from_native_repair"] == hybrid["factor_tensor_bytes"] - native["factor_tensor_bytes"]
        assert row["repair_rank_group"] == "native_plus_calibration_complement"
        assert row["serialized_total_bytes"] == sum(path.stat().st_size for path in (out / row["variant"]).rglob("*") if path.is_file())
        assert not row["heldout_used_for_fit"] and not row["language_quality_established"]
        saved = load_probe_residual(out / row["variant"], expected_identity=source.identity)
        torch.testing.assert_close(saved.base_quantization.codes, baseline.codes, rtol=0, atol=0)
        torch.testing.assert_close(saved.base_quantization.scales, baseline.scales, rtol=0, atol=0)
        assert saved.coefficients.dtype == torch.float32
        if row is hybrid:
            torch.testing.assert_close(saved.span_basis[:, :1], native_saved.span_basis, rtol=0, atol=0)
        else:
            assert row["rank_matched_to_variant"] == "probe_pca_hybrid"
            assert row["tensor_bytes_match_matched_repair"]
            assert row["serialized_total_bytes_difference_from_matched_repair"] == row["serialized_total_bytes"] - hybrid["serialized_total_bytes"]
    settings = json.loads((out / "settings.json").read_text())
    assert settings["probe_hybrid_enabled"] and settings["hybrid_extra_rank_requested"] is None
    assert settings["hybrid_extra_rank_default"] == "native probe span rank"


def test_hybrid_factors_do_not_depend_on_heldout_state_values(tmp_path):
    source, first_archive = fixture(tmp_path / "hybrid_first", probe_seed=True, heldout_scale=1)
    _, second_archive = fixture(tmp_path / "hybrid_second", probe_seed=True, heldout_scale=31)
    for folder, archive in ((tmp_path / "hybrid_a", first_archive), (tmp_path / "hybrid_b", second_archive)):
        trial.compare(source, archive, folder, ranks=(0,), max_train=2, max_heldout=2,
            row_chunk=1, include_probe_residual=True, include_probe_hybrid=True, hybrid_extra_rank=1)
    for method in ("probe_pca_hybrid", "residual_calibration_pca_extended", "residual_seeded_random_extended"):
        first = torch.load(tmp_path / "hybrid_a" / method / "correction.pt", weights_only=True)
        second = torch.load(tmp_path / "hybrid_b" / method / "correction.pt", weights_only=True)
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        report = next(row for row in json.loads((tmp_path / "hybrid_a/results.json").read_text()) if row["method"] == method)
        assert not set(report["control_basis"].get("training_record_ids", ())) & {"heldout:0", "heldout:1"}


def test_unqualified_hybrid_basis_is_not_fit(tmp_path, monkeypatch):
    source, archive = fixture(tmp_path, probe_seed=True)
    import tda.probe_residual_compress as residual
    monkeypatch.setattr(residual, "make_probe_pca_complement_basis", lambda *args, **kwargs: {"rank_matches_request": False})
    def unexpected(*args, **kwargs):
        pytest.fail("an unqualified hybrid basis was fit")
    monkeypatch.setattr(residual, "fit_scoped_residual_basis", unexpected)
    out = tmp_path / "unqualified_hybrid"
    with pytest.raises(ValueError, match="requested hybrid rank"):
        trial.compare(source, archive, out, ranks=(0,), max_train=2, max_heldout=2,
            row_chunk=1, include_probe_residual=True, include_probe_hybrid=True)
    assert json.loads((out / "status.json").read_text())["status"] == "failed"
    assert json.loads((out / "results.json").read_text())[-1]["method"] == "probe_residual"


def test_cli_passes_explicit_hybrid_rank_and_requires_native_residual(monkeypatch):
    calls = []
    class SourceContext:
        def __enter__(self):
            return "source"
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(trial, "GLMWeightSampleReader", lambda path: SourceContext())
    monkeypatch.setattr(trial.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(trial, "compare", lambda *args, **kwargs: calls.append(kwargs))
    args = ["--sample", "sample", "--archive", "archive", "--out", "out"]
    assert trial.main(args + ["--probe-residual", "--probe-hybrid", "--hybrid-extra-rank", "77"]) == 0
    assert calls[-1]["include_probe_hybrid"] and calls[-1]["hybrid_extra_rank"] == 77
    for options in (["--probe-hybrid"], ["--hybrid-extra-rank", "77"],
                    ["--probe-residual", "--probe-hybrid", "--hybrid-extra-rank", "0"],
                    ["--probe-residual", "--probe-hybrid", "--hybrid-extra-rank", "2048"]):
        with pytest.raises(SystemExit) as failure:
            trial.main(args + options)
        assert failure.value.code == 2
