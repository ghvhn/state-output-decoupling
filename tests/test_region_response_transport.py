from dataclasses import replace

import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity
from tda.trajectory_compress import InputStateScope, CalibrationRecord, TrajectoryPair
from tda.region_response_transport import fit_regions, variant_bank, predict_residual
from scripts.region_transport_glm import response_metrics, _save_adapter


def fixture():
    generator = torch.Generator().manual_seed(7)
    identity = ExpertGroupIdentity("fixture", "a" * 40, 3, "gate_proj", "fixture layout",
                                  ("0", "1", "2"), (2, 4), "float32")
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm_expert_gate_up_input")
    centers = torch.tensor([[-3., -3., 0., 0.], [-3., 3., 0., 0.],
                            [3., -3., 0., 0.], [3., 3., 0., 0.]])
    x = torch.cat([center + torch.randn(30, 4, generator=generator) * .25 for center in centers])
    records = [CalibrationRecord(f"r{i}", "fixture", f"trajectory{i // 2}", scope, state,
                                  prediction_correct=bool(i % 2), origin="synthetic_fixture")
               for i, state in enumerate(x)]
    # Same piecewise nonlinear feature change is reused by different output bases.
    features = torch.stack((x[:, 0] + .4 * x[:, 0] * x[:, 1],
                            x[:, 1] + .2 * x[:, 0].square()), dim=1)
    bases = torch.tensor([[[1., .2, -.5], [.1, 1., .3]],
                          [[.2, -.8, 1.], [1., .1, -.4]],
                          [[.7, .4, -.3], [-.2, 1.3, .2]]])
    y = torch.einsum("nr,erd->end", features, bases)
    return scope, records, y


def fit(scope, records, y, **kwargs):
    return fit_regions(records, y, expected_scope=scope, rank=2, regions=4,
                       operator_excluded_expert=2, **kwargs)


def test_shared_maps_reuse_a_real_nonlinear_change_for_excluded_expert():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    x = torch.stack([r.state for r in records])
    global_prediction, _ = predict_residual(variant_bank(bank, "global"), x, expected_scope=scope)
    shared, _ = predict_residual(variant_bank(bank, "shared_transport"), x, expected_scope=scope)
    assert (shared[2] - y[2]).square().mean() < .25 * (global_prediction[2] - y[2]).square().mean()
    assert bank["metadata"]["region_semantics_validated"] is False
    assert bank["metadata"]["unseen_region_transfer_evaluated"] is False


def test_excluded_expert_cannot_change_shared_operator_fit():
    scope, records, y = fixture()
    first = fit(scope, records, y)
    altered = y.clone()
    altered[2] = torch.sin(y[2]) * 100
    second = fit(scope, records, altered)
    assert torch.equal(first["tensors"]["operators"], second["tensors"]["operators"])
    assert not torch.equal(first["tensors"]["slopes"][2], second["tensors"]["slopes"][2])


def test_wrong_region_and_identity_controls_retain_actual_tensor_bytes():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    variants = [variant_bank(bank, name) for name in
                ("shared_transport", "identity_transport", "wrong_region_transport")]
    assert len({v["metadata"]["tensor_payload_bytes"] for v in variants}) == 1
    x = torch.stack([r.state for r in records])
    outputs = [predict_residual(v, x, expected_scope=scope)[0] for v in variants]
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])
    assert (outputs[0] - y).square().mean() < (outputs[2] - y).square().mean()


def test_identity_operator_equals_regional_offset_baseline():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    x = torch.stack([r.state for r in records])
    actual = predict_residual(variant_bank(bank, "identity_transport"), x, expected_scope=scope)[0]
    expected = predict_residual(variant_bank(bank, "region_offsets"), x, expected_scope=scope)[0]
    assert torch.equal(actual, expected)


def test_fit_rejects_record_or_whole_trajectory_leakage():
    scope, records, y = fixture()
    with pytest.raises(ValueError, match="heldout"):
        fit(scope, records, y, heldout_records=(records[0],))
    duplicate_trajectory = replace(records[0], record_id="new heldout record")
    with pytest.raises(ValueError, match="heldout"):
        fit(scope, records, y, heldout_records=(duplicate_trajectory,))


def test_fit_rejects_scope_nonfinite_and_budget_before_numerical_fitting():
    scope, records, y = fixture()
    with pytest.raises(ValueError, match="scope"):
        fit(replace(scope, layer=4), records, y)
    invalid = y.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit(scope, records, invalid)
    with pytest.raises(MemoryError):
        fit(scope, records, y, max_working_bytes=1)


def test_nonuniform_attention_strength_is_not_silently_ignored():
    scope, records, y = fixture()
    records[0] = replace(records[0], stimulation_weight=.3)
    with pytest.raises(ValueError, match="uniform"):
        fit(scope, records, y)


def test_wrong_predictions_are_retained_in_calibration_evidence():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    assert bank["metadata"]["calibration_record_ids"] == [r.record_id for r in records]
    assert any(r.prediction_correct is False for r in records)


def test_outside_support_uses_global_fallback_without_discarding_record():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    x = torch.tensor([[100., 100., 100., 100.]])
    regional, diagnostic = predict_residual(variant_bank(bank, "shared_transport"), x, expected_scope=scope)
    global_value, _ = predict_residual(variant_bank(bank, "global"), x, expected_scope=scope)
    assert torch.equal(regional, global_value)
    assert diagnostic["global_fallback_count"] == 1
    assert diagnostic["supported"].tolist() == [False]


def test_sparse_regions_keep_evidence_and_decline_local_map_fit():
    scope, records, y = fixture()
    selected = list(range(91))  # Fourth region retains one recorded example.
    bank = fit(scope, [records[i] for i in selected], y[:, selected])
    assert not all(bank["metadata"]["region_enabled"])
    assert bank["metadata"]["calibration_record_ids"] == [records[i].record_id for i in selected]
    for region, enabled in enumerate(bank["metadata"]["region_enabled"]):
        if not enabled:
            assert torch.equal(bank["tensors"]["operators"][region], torch.eye(2))
            assert torch.count_nonzero(bank["tensors"]["offsets"][:, region]) == 0


def test_prediction_rejects_different_site_and_corrupt_tensor():
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y), "shared_transport")
    x = torch.stack([r.state for r in records[:2]])
    with pytest.raises(ValueError, match="scope"):
        predict_residual(bank, x, expected_scope=replace(scope, state_site="other"))
    bank["tensors"]["operators"][0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        predict_residual(bank, x, expected_scope=scope)


def test_residual_context_binds_all_projections_and_base_codec():
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y, response_context_sha256="a" * 64), "shared_transport")
    x = torch.stack([r.state for r in records[:2]])
    with pytest.raises(ValueError, match="context"):
        predict_residual(bank, x, expected_scope=scope, expected_response_context_sha256="b" * 64)
    actual, _ = predict_residual(bank, x, expected_scope=scope, expected_response_context_sha256="a" * 64)
    assert torch.isfinite(actual).all()


def test_metrics_use_endpoint_outputs_and_keep_region_crossings():
    scope, records, _ = fixture()
    records = records[:4]
    pairs = [TrajectoryPair("p0", "r0", "r1"), TrajectoryPair("p1", "r2", "r3")]
    reference = torch.tensor([[[1.], [2.], [3.], [5.]], [[2.], [4.], [6.], [10.]]])
    proposed = reference + 1  # Constant output offset cancels in actual differences.
    metrics = response_metrics(reference, proposed, records, pairs, torch.tensor([0, 0, 0, 1]), 1)
    assert metrics["output"]["relative_rmse"] > 0
    assert metrics["output_difference"]["relative_rmse"] == 0
    assert metrics["within_region_pairs"] == metrics["cross_region_pairs"] == 1
    assert metrics["per_expert"][1]["excluded_from_operator_fit"] is True


def test_saved_adapter_counts_final_metadata_and_restricted_roundtrip(tmp_path):
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y), "shared_transport")
    receipt, loaded = _save_adapter(bank, tmp_path / "adapter", scope)
    assert receipt["serialized_total_bytes"] == sum(p.stat().st_size for p in (tmp_path / "adapter").iterdir())
    x = torch.stack([r.state for r in records[:2]])
    assert torch.equal(predict_residual(bank, x, expected_scope=scope)[0],
                       predict_residual(loaded, x, expected_scope=scope)[0])
