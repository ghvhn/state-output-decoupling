"""Angular response laws: exact objectives, covariance and evidence exclusion."""
from dataclasses import replace
import math

import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity
from tda.trajectory_compress import InputStateScope, CalibrationRecord
from tda.rotation_response_transport import (fit_rotations, variant_bank, predict_geometry,
    predict_residual, _rotation_matrices, _transport_delta, _fourier_design,
    _covariant_shared_fit, _linear_shared_fit)
from tda.shell_response_transport import _design


def fixture():
    generator = torch.Generator().manual_seed(19)
    identity = ExpertGroupIdentity("fixture", "a" * 40, 3, "gate_proj", "fixture row-major",
                                   ("a", "b", "c"), (3, 4), "float32")
    scope = InputStateScope.from_identity(identity, state_site="post_attention_norm_expert_gate_up_input")
    centers = torch.tensor([[-3., 0., 0., 0.], [3., 0., 0., 0.]])
    x = torch.cat([center + torch.randn(96, 4, generator=generator) * .4 for center in centers])
    records = [CalibrationRecord(f"r{i}", "fixture", f"trajectory{i // 2}", scope, state,
                                  prediction_correct=bool(i % 2), origin="synthetic_fixture")
               for i, state in enumerate(x)]
    features = torch.stack((x[:, 0] + .3 * x[:, 0] * x[:, 1],
                            x[:, 1] + .2 * x[:, 0].square()), dim=1)
    bases = torch.tensor([[[1., .2, -.5], [.1, 1., .3]],
                          [[.2, -.8, 1.], [1., .1, -.4]],
                          [[.7, .4, -.3], [-.2, 1.3, .2]]])
    y = torch.einsum("nr,erd->end", features, bases)
    return scope, records, y


def fit(scope, records, y, **kwargs):
    return fit_rotations(records, y, expected_scope=scope, rank=2, regions=2,
                         operator_excluded_expert=2, **kwargs)


def inputs(records):
    return torch.stack([record.state for record in records])


def test_covariant_law_rotates_affine_shift_and_retains_complement():
    generator = torch.Generator().manual_seed(7)
    v = torch.randn(40, 4, generator=generator)
    plane = torch.eye(4)[:, :2]
    phase = torch.atan2(v[:, 1], v[:, 0])
    rho = v.norm(dim=1) / 4
    h = torch.randn(3, 5, 4, generator=generator) * .3
    alpha = torch.linspace(-4 * math.pi, 6 * math.pi, len(v))
    rotation = _rotation_matrices(alpha, plane)
    forward = torch.einsum("nr,nsr->ns", v, rotation)
    expected = torch.einsum("nr,nsr->ns", _transport_delta(v, rho, phase, plane, h), rotation)
    actual = _transport_delta(forward, rho, phase + alpha, plane, h)
    torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
    torch.testing.assert_close(forward[:, 2:], v[:, 2:], atol=0, rtol=0)
    periodic = _transport_delta(v, rho, phase + 2 * math.pi, plane, h)
    torch.testing.assert_close(periodic, _transport_delta(v, rho, phase, plane, h), atol=2e-6, rtol=2e-6)
    assert bool((h[:, -1].abs() > .01).any())  # Genuine affine translation rotates too.


def test_exact_rotated_basis_objective_matches_independent_full_design():
    generator = torch.Generator().manual_seed(31)
    v = torch.randn(12, 2, generator=generator)
    rho = torch.rand(12, generator=generator)
    phase = torch.linspace(-.7, 7.3, len(v))
    plane = torch.eye(2)
    b = torch.randn(3, 2, 3, generator=generator)
    y = torch.randn(3, 12, 3, generator=generator)
    zero = torch.zeros(3, 3)
    selected = torch.arange(len(v))
    h, detail = _covariant_shared_fit(v, rho, phase, plane, torch.zeros_like(v), selected,
        y, zero, b, zero, [0, 1], .07)
    rotation = _rotation_matrices(phase.double(), plane.double())
    design = _design(torch.einsum("nr,nrs->ns", v.double(), rotation), rho.double(), 2)
    rows, targets = [], []
    for expert in (0, 1):
        for n in range(len(v)):
            rotated_basis = rotation[n].T @ b[expert].double()
            for output in range(3):
                rows.append(torch.outer(design[n], rotated_basis[:, output]).flatten())
                targets.append(y[expert, n, output].double())
    full = torch.stack(rows)
    target = torch.stack(targets)
    expected = torch.linalg.solve(full.T @ full + detail["ridge_penalty"] * torch.eye(18, dtype=torch.float64),
                                  full.T @ target).reshape(3, 3, 2)
    torch.testing.assert_close(h.double(), expected, atol=3e-7, rtol=3e-7)
    assert detail["normal_unknowns"] == 18
    assert detail["normal_residual_relative"] < 1e-10


def test_covariant_map_recovers_withheld_angle_and_excluded_expert():
    generator = torch.Generator().manual_seed(17)
    v = torch.randn(300, 3, generator=generator)
    phase = torch.remainder(torch.atan2(v[:, 1], v[:, 0]), 2 * math.pi)
    rho = v.norm(dim=1) / v.norm(dim=1).max()
    allowed = ~((phase >= math.pi / 2) & (phase < math.pi))
    plane = torch.eye(3)[:, :2]
    b = torch.randn(3, 3, 4, generator=generator)
    oracle_h = torch.randn(3, 4, 3, generator=generator) * .1
    delta = _transport_delta(v, rho, phase, plane, oracle_h)
    y = torch.einsum("nr,erd->end", delta, b)
    selected = torch.nonzero(allowed, as_tuple=False).flatten()
    zero = torch.zeros(3, 4)
    h, detail = _covariant_shared_fit(v[allowed], rho[allowed], phase[allowed], plane,
        torch.zeros(int(allowed.sum()), 3), selected, y, zero, b, zero, [0, 1], 1e-10)
    predicted = _transport_delta(v, rho, phase, plane, h) @ b[2]
    assert int((~allowed).sum()) > 40
    torch.testing.assert_close(predicted[~allowed], y[2, ~allowed], atol=2e-5, rtol=2e-5)
    assert detail["solver"].startswith("exact per-record")


def test_measured_phase_design_rank_discards_canonical_roundoff_coordinate():
    from tda.rotation_geometry import project_directions
    generator = torch.Generator().manual_seed(75)
    v = torch.randn(120, 3, generator=generator)
    plane = torch.linalg.qr(torch.randn(3, 2, generator=generator)).Q.float()
    phase = project_directions(v, plane)["phase_rad"].float()
    rho = v.norm(dim=1) / v.norm(dim=1).max()
    b = torch.randn(3, 3, 4, generator=generator)
    y = torch.randn(3, 120, 4, generator=generator)
    zero = torch.zeros(3, 4)
    _, detail = _covariant_shared_fit(v, rho, phase, plane, torch.zeros_like(v),
        torch.arange(len(v)), y, zero, b, zero, [0, 1], .01)
    assert detail["design_columns"] == 12
    assert detail["effective_design_rank"] <= 3 * v.shape[1]
    assert not detail["full_design_rank"]
    assert "1e-10" in detail["design_rank_threshold"]


def test_fourier_exact_objective_matches_independent_full_design():
    generator = torch.Generator().manual_seed(71)
    v = torch.randn(20, 2, generator=generator)
    rho = torch.rand(20, generator=generator)
    phase = torch.rand(20, generator=generator) * 2 * math.pi
    d = _fourier_design(v, rho, phase)
    b = torch.randn(3, 2, 3, generator=generator)
    y = torch.randn(3, 20, 3, generator=generator)
    zero = torch.zeros(3, 3)
    h, detail = _linear_shared_fit(d, torch.zeros_like(v), torch.arange(20), y,
        zero, b, zero, [0, 1], .05)
    rows = torch.cat([torch.einsum("np,ro->nopr", d.double(), b[e].double()).reshape(-1, 90) for e in (0, 1)])
    target = y[:2].double().reshape(-1)
    expected = torch.linalg.solve(rows.T @ rows + detail["ridge_penalty"] * torch.eye(90, dtype=torch.float64),
                                  rows.T @ target).reshape(45, 2)
    torch.testing.assert_close(h.double(), expected, atol=4e-7, rtol=4e-7)
    assert d.shape == (20, 45)  #15 affine feature blocks, not equal capacity to quadratic radial.


def test_withheld_sector_cannot_change_any_fitted_tensor_or_fit_digest():
    scope, records, y = fixture()
    first = fit(scope, records, y)
    withheld = torch.tensor(first["metadata"]["calibration_angle_bin_ids"]) == 1
    assert int(withheld.sum()) > 30
    changed = y.clone()
    changed[:, withheld] = torch.randn(changed[:, withheld].shape,
        generator=torch.Generator().manual_seed(47)) * 10000
    second = fit(scope, records, changed)
    assert first["metadata"] == second["metadata"]
    assert all(torch.equal(value, second["tensors"][name]) for name, value in first["tensors"].items())
    assert first["metadata"]["withheld_sector_inputs_used_for_geometry"]
    assert not first["metadata"]["withheld_sector_responses_used_for_fit"]


def test_excluded_expert_changes_only_own_global_slopes_and_offsets():
    scope, records, y = fixture()
    first = fit(scope, records, y)
    changed = y.clone()
    changed[2] = torch.sin(y[2]) * 50
    second = fit(scope, records, changed)
    for name in first["tensors"]:
        if name not in ("response_mean", "slopes", "offsets"):
            assert torch.equal(first["tensors"][name], second["tensors"][name])
    for name in ("response_mean", "slopes", "offsets"):
        assert torch.equal(first["tensors"][name][:2], second["tensors"][name][:2])
        assert not torch.equal(first["tensors"][name][2], second["tensors"][name][2])


def test_phase_and_radius_are_independent_and_calibration_geometry_reproducible():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    diagnostic = predict_geometry(bank, inputs(records), expected_scope=scope)
    assert diagnostic["angle_bin_ids"].tolist() == bank["metadata"]["calibration_angle_bin_ids"]
    assert diagnostic["region_ids"].tolist() == bank["metadata"]["calibration_region_ids"]
    from tda.rotation_response_transport import _plane_geometry
    plane = torch.eye(4)[:, :2]
    directions = torch.tensor([[1., 0., 1., 0.], [2., 0., 2., 0.], [0., 1., 1., 0.]])
    phase, fraction, valid = _plane_geometry(directions, plane)
    assert phase[0] == phase[1] and phase[0] != phase[2]
    assert bool(valid.all())
    assert torch.equal(fraction[0], fraction[1])


def test_wrong_controls_keep_geometry_and_equal_coefficient_bytes():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    candidates = {name: variant_bank(bank, name) for name in (
        "radial_transport", "rotation_transport", "wrong_rotation_transport",
        "fourier_transport", "wrong_fourier_transport")}
    assert candidates["radial_transport"]["metadata"]["tensor_payload_bytes"] == candidates["rotation_transport"]["metadata"]["tensor_payload_bytes"]
    assert candidates["rotation_transport"]["metadata"]["tensor_payload_bytes"] == candidates["wrong_rotation_transport"]["metadata"]["tensor_payload_bytes"]
    assert candidates["fourier_transport"]["metadata"]["tensor_payload_bytes"] == candidates["wrong_fourier_transport"]["metadata"]["tensor_payload_bytes"]
    assert candidates["fourier_transport"]["tensors"]["coefficients"].numel() == 5 * candidates["radial_transport"]["tensors"]["coefficients"].numel()
    for normal, wrong in (("rotation_transport", "wrong_rotation_transport"), ("fourier_transport", "wrong_fourier_transport")):
        assert all(torch.equal(value, candidates[wrong]["tensors"][name]) for name, value in candidates[normal]["tensors"].items())
        pred, geometry = predict_residual(candidates[normal], inputs(records), expected_scope=scope)
        wrong_pred, wrong_geometry = predict_residual(candidates[wrong], inputs(records), expected_scope=scope)
        assert not torch.equal(pred, wrong_pred)
        for name in geometry:
            if isinstance(geometry[name], torch.Tensor):
                torch.testing.assert_close(geometry[name], wrong_geometry[name], equal_nan=True)
            else:
                assert geometry[name] == wrong_geometry[name]


def test_fourier_continuation_is_periodic_even_beyond_one_revolution():
    generator = torch.Generator().manual_seed(23)
    v = torch.randn(15, 4, generator=generator)
    rho = torch.rand(15, generator=generator)
    phase = torch.rand(15, generator=generator) * 2 * math.pi
    d = _fourier_design(v, rho, phase)
    continued = _fourier_design(v, rho, phase + 4 * math.pi)
    torch.testing.assert_close(d, continued, atol=4e-6, rtol=4e-6)


def test_nonwithheld_angles_only_appear_in_compact_table():
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y), "separate_angle_transport")
    assert bank["metadata"]["table_angle_bin_ids"] == [0, 2, 3]
    assert bank["tensors"]["table_coefficients"].shape == (2, 3, 3, 2)
    assert bank["tensors"]["table_enabled"].shape == (2, 3)
    predicted, geometry = predict_residual(bank, inputs(records), expected_scope=scope)
    assert predicted.shape == y.shape
    assert bool(geometry["supported"].all())


def test_all_variants_fallback_to_global_outside_same_support():
    scope, records, y = fixture()
    full = fit(scope, records, y)
    distant = inputs(records)[:5] * 100
    global_prediction, geometry = predict_residual(variant_bank(full, "global"), distant, expected_scope=scope)
    assert geometry["global_fallback_count"] == 5
    for name in ("region_offsets", "radial_transport", "rotation_transport", "fourier_transport", "separate_angle_transport"):
        actual, other = predict_residual(variant_bank(full, name), distant, expected_scope=scope)
        assert torch.equal(actual, global_prediction)
        assert other["global_fallback_count"] == 5


def test_undefined_plane_phase_is_minus_one_and_global_fallback():
    scope, records, y = fixture()
    full = fit(scope, records, y)
    # The exact center has no local direction. Inverse PCA reconstructs a point
    # near that center; use a deliberately negligible plane projection instead.
    center = full["tensors"]["centers"][0]
    state = full["tensors"]["input_mean"] + (center * full["tensors"]["feature_scale"]) @ full["tensors"]["input_basis"].T
    candidate = variant_bank(full, "rotation_transport")
    candidate["tensors"]["centers"][0] = ((state - candidate["tensors"]["input_mean"]) @ candidate["tensors"]["input_basis"] / candidate["tensors"]["feature_scale"])
    prediction, geometry = predict_residual(candidate, state[None], expected_scope=scope)
    baseline = variant_bank(full, "global")
    baseline["tensors"]["centers"][0] = candidate["tensors"]["centers"][0]
    expected, _ = predict_residual(baseline, state[None], expected_scope=scope)
    assert geometry["angle_bin_ids"].item() == -1
    assert torch.isnan(geometry["phase_rad"]).all()
    assert not bool(geometry["supported"].any())
    assert torch.equal(prediction, expected)


def test_split_scope_context_and_numerical_budgets_guard_before_fit():
    scope, records, y = fixture()
    with pytest.raises(ValueError, match="heldout"):
        fit(scope, records, y, heldout_records=records[:2])
    wrong_scope = replace(scope, state_site="different-site")
    with pytest.raises(ValueError, match="scope"):
        fit(wrong_scope, records, y)
    with pytest.raises(MemoryError, match="budget"):
        fit(scope, records, y, max_working_bytes=1)
    measured = [replace(record, origin="measured_model_state") for record in records]
    with pytest.raises(ValueError, match="context"):
        fit(scope, measured, y)
    bank = variant_bank(fit(scope, records, y, response_context_sha256="a" * 64), "rotation_transport")
    with pytest.raises(ValueError, match="context"):
        predict_residual(bank, inputs(records), expected_scope=scope, expected_response_context_sha256="b" * 64)
    with pytest.raises(MemoryError, match="budget"):
        predict_residual(bank, inputs(records), expected_scope=scope,
                         expected_response_context_sha256="a" * 64, max_working_bytes=1)


def test_predictions_false_and_unknown_are_retained_and_weights_rejected():
    scope, records, y = fixture()
    changed = [replace(record, prediction_correct=None) for record in records]
    first, second = fit(scope, records, y), fit(scope, changed, y)
    assert first["metadata"]["calibration_record_ids"] == second["metadata"]["calibration_record_ids"]
    assert all(torch.equal(value, second["tensors"][name]) for name, value in first["tensors"].items())
    changed[0] = replace(changed[0], stimulation_weight=.5)
    with pytest.raises(ValueError, match="uniform"):
        fit(scope, changed, y)


def test_rank_one_chart_has_no_invented_plane_and_falls_back_to_global(monkeypatch):
    scope, records, y = fixture()
    generator = torch.Generator().manual_seed(52)
    line = torch.zeros(96, 4)
    line[:, 0] = torch.linspace(-3.2, -2.8, 96)
    cloud = torch.tensor([3., 0., 0., 0.]) + torch.randn(96, 4, generator=generator) * .4
    x = torch.cat((line, cloud))
    records = [replace(record, state=state) for record, state in zip(records, x)]
    # Fix calibration-only membership so the test exercises the precise
    # low-rank chart condition, independently of k-means initialization.
    def clusters(z, regions, seed):
        labels = torch.cat((torch.zeros(96, dtype=torch.long), torch.ones(96, dtype=torch.long)))
        return torch.stack([z[labels == k].mean(dim=0) for k in range(2)]), labels
    monkeypatch.setattr("tda.rotation_response_transport._cluster", clusters)
    full = fit(scope, records, y)
    assert full["metadata"]["rotation_plane_defined"] == [False, True]
    assert full["metadata"]["region_enabled"] == [False, True]
    assert "span" in full["metadata"]["rotation_plane_fallback_reasons"][0]
    global_pred, diagnostic = predict_residual(variant_bank(full, "global"), x, expected_scope=scope)
    inactive = diagnostic["region_ids"] == 0
    assert int(inactive.sum()) >= 90
    assert not bool(diagnostic["phase_valid"][inactive].any())
    assert bool((diagnostic["angle_bin_ids"][inactive] == -1).all())
    assert bool(torch.isnan(diagnostic["phase_rad"][inactive]).all())
    assert not bool(diagnostic["supported"][inactive].any())
    for name in ("radial_transport", "rotation_transport", "fourier_transport", "separate_angle_transport"):
        candidate, current = predict_residual(variant_bank(full, name), x, expected_scope=scope)
        assert torch.equal(candidate[:, inactive], global_pred[:, inactive])
        assert not bool(current["supported"][inactive].any())
    assert bool((full["tensors"]["rotation_coefficients"][0] == 0).all())


def test_all_rank_one_charts_fail_informatively_without_target_fitting(monkeypatch):
    scope, records, y = fixture()
    first = torch.zeros(96, 4)
    first[:, 0] = -3
    first[:, 1] = torch.linspace(-.4, .4, 96)
    second = torch.zeros(96, 4)
    second[:, 0] = torch.linspace(2.8, 3.2, 96)
    x = torch.cat((first, second))
    records = [replace(record, state=state) for record, state in zip(records, x)]
    def clusters(z, regions, seed):
        labels = torch.cat((torch.zeros(96, dtype=torch.long), torch.ones(96, dtype=torch.long)))
        return torch.stack([z[labels == k].mean(dim=0) for k in range(2)]), labels
    monkeypatch.setattr("tda.rotation_response_transport._cluster", clusters)
    with pytest.raises(ValueError, match="no calibration chart supports a defined rotation plane"):
        fit(scope, records, y)


def test_undefined_plane_cannot_be_enabled_by_saved_flags():
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y), "rotation_transport")
    bank["tensors"]["rotation_plane_defined"][0] = 0
    with pytest.raises(ValueError, match="without a defined rotation plane"):
        predict_geometry(bank, inputs(records), expected_scope=scope)


@pytest.mark.parametrize("field,operation", [
    ("rotation_planes", lambda t: t * 2),
    ("radii", lambda t: -t),
    ("region_enabled", lambda t: t + .5),
    ("rotation_plane_defined", lambda t: t + .5),
])
def test_saved_geometry_rejects_corruption(field, operation):
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y), "rotation_transport")
    bank["tensors"][field] = operation(bank["tensors"][field])
    with pytest.raises(ValueError):
        predict_residual(bank, inputs(records), expected_scope=scope)
