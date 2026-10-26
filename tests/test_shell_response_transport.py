"""Continuous affine shell laws, response exclusions and honest controls."""
from dataclasses import replace

import pytest
import torch

from tda.expert_compress import ExpertGroupIdentity
from tda.trajectory_compress import InputStateScope, CalibrationRecord, TrajectoryPair
from tda.shell_response_transport import (fit_shells, variant_bank, predict_residual,
    predict_geometry, _design, _shared_fit)
from scripts.region_transport_glm import response_metrics, _save_adapter


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
    return fit_shells(records, y, expected_scope=scope, rank=2, regions=2,
                      shells=4, withheld_shell=1, operator_excluded_expert=2, **kwargs)


def inputs(records):
    return torch.stack([r.state for r in records])


def test_continuous_matrix_and_affine_polynomial_recover_excluded_response_level():
    generator = torch.Generator().manual_seed(7)
    v = torch.randn(160, 2, generator=generator)
    s = v.norm(dim=1) / v.norm(dim=1).max()
    # Hold out an actual continuous interval, not merely one demonstration.
    allowed = ~((s > .25) & (s <= .40))
    b = torch.tensor([[[1., .3, -.5], [.1, 1., .2]],
                      [[.2, -.8, 1.], [1., .1, -.4]],
                      [[.7, .4, -.3], [-.2, 1.3, .2]]])
    h = torch.randn(3, 3, 2, generator=generator) * .2
    oracle = _design(v, s, 2) @ h.reshape(9, 2)
    y = torch.einsum("nr,erd->end", oracle, b)
    zero = torch.zeros(3, 3)
    result, detail = _shared_fit(v[allowed], s[allowed], torch.zeros(int(allowed.sum()), 2),
        allowed, y, zero, b, zero, [0, 1], 2, 1e-10)
    recovered = _design(v, s, 2) @ result.reshape(9, 2)
    predicted = recovered @ b[2]
    assert detail["full_design_rank"]
    assert int((~allowed).sum()) > 10
    torch.testing.assert_close(predicted[~allowed], y[2, ~allowed], atol=2e-5, rtol=2e-5)
    assert bool((result[:, -1].abs() > 1e-3).any())  # Genuine affine h(s), not only gain.
    assert bool((result[2, :-1].abs() > 1e-3).any())  # Genuine quadratic matrices.


def test_exact_two_gram_solution_matches_independent_kronecker_objective():
    generator = torch.Generator().manual_seed(31)
    v = torch.randn(12, 2, generator=generator)
    s = torch.rand(12, generator=generator)
    b = torch.randn(3, 2, 3, generator=generator)
    y = torch.randn(3, 12, 3, generator=generator)
    zero = torch.zeros(3, 3)
    mask = torch.ones(12, dtype=torch.bool)
    h, detail = _shared_fit(v, s, torch.zeros(12, 2), mask, y, zero, b, zero, [0, 1], 1, .07)
    d = _design(v, s, 1).double()
    # Independently build the full design from outer products for every observed
    # expert/record/output coordinate. No reuse of the factorized normal eq.
    rows, targets = [], []
    for expert in (0, 1):
        for n in range(12):
            for output in range(3):
                rows.append(torch.outer(d[n], b[expert, :, output].double()).flatten())
                targets.append(y[expert, n, output].double())
    full = torch.stack(rows)
    target = torch.stack(targets)
    reference = torch.linalg.solve(full.T @ full + detail["ridge_penalty"] * torch.eye(12, dtype=torch.float64),
                                   full.T @ target).reshape(2, 3, 2)
    torch.testing.assert_close(h.double(), reference, atol=2e-7, rtol=2e-7)


def test_withheld_shell_targets_cannot_change_any_fitted_tensor_or_fit_digest():
    scope, records, y = fixture()
    first = fit(scope, records, y)
    withheld = torch.tensor(first["metadata"]["calibration_shell_ids"]) == 1
    assert int(withheld.sum()) == 48
    changed = y.clone()
    changed[:, withheld] = torch.randn(changed[:, withheld].shape,
        generator=torch.Generator().manual_seed(47)) * 10000
    second = fit(scope, records, changed)
    assert first["metadata"] == second["metadata"]
    assert all(torch.equal(value, second["tensors"][name]) for name, value in first["tensors"].items())
    assert first["metadata"]["withheld_level_inputs_used_for_geometry"] is True
    assert first["metadata"]["withheld_level_responses_used_for_fit"] is False


def test_excluded_expert_can_change_only_its_response_bases_and_offsets():
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


def test_shell_order_is_measured_radius_not_arbitrary_cluster_index():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    diagnostic = predict_geometry(bank, inputs(records), expected_scope=scope)
    assert diagnostic["shell_ids"].tolist() == bank["metadata"]["calibration_shell_ids"]
    for region in range(2):
        mask = diagnostic["region_ids"] == region
        order = diagnostic["shell_coordinates"][mask].argsort()
        assert bool((diagnostic["shell_ids"][mask][order].diff() >= 0).all())
    variant = variant_bank(bank, "generated_transport")
    reordered = {"metadata": dict(variant["metadata"]), "tensors": dict(variant["tensors"])}
    for name in ("centers", "radii", "shell_boundaries", "shell_coordinate_ranges", "region_enabled", "coefficients"):
        reordered["tensors"][name] = reordered["tensors"][name].flip(0)
    reordered["tensors"]["offsets"] = reordered["tensors"]["offsets"].flip(1)
    expected = predict_residual(variant, inputs(records), expected_scope=scope)[0]
    actual = predict_residual(reordered, inputs(records), expected_scope=scope)[0]
    assert torch.equal(actual, expected)


def test_equal_bytes_nested_degree_and_identity_controls():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    names = ("constant_transport", "linear_transport", "generated_transport", "identity_transport", "wrong_shell_transport")
    variants = [variant_bank(bank, name) for name in names]
    assert len({v["metadata"]["tensor_payload_bytes"] for v in variants}) == 1
    assert not bool(variants[0]["tensors"]["coefficients"][:, 1:].any())
    assert not bool(variants[1]["tensors"]["coefficients"][:, 2:].any())
    assert not bool(variants[3]["tensors"]["coefficients"].any())
    identity = predict_residual(variants[3], inputs(records), expected_scope=scope)[0]
    offsets = predict_residual(variant_bank(bank, "region_offsets"), inputs(records), expected_scope=scope)[0]
    assert torch.equal(identity, offsets)
    expected = predict_geometry(bank, inputs(records), expected_scope=scope)
    for value in variants:
        _, diagnostic = predict_residual(value, inputs(records), expected_scope=scope)
        for key in ("region_ids", "shell_ids", "supported", "shell_coordinates"):
            assert torch.equal(expected[key], diagnostic[key])
    generated = predict_residual(variants[2], inputs(records), expected_scope=scope)[0]
    wrong = predict_residual(variants[4], inputs(records), expected_scope=scope)[0]
    assert not torch.equal(generated, wrong)
    assert torch.equal(wrong, predict_residual(variants[4], inputs(records), expected_scope=scope)[0])


def test_separate_table_never_fits_withheld_cell_and_uses_lower_index_tie():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    assert not bool(bank["tensors"]["table_coefficients"][:, 1].any())
    assert not bool(bank["tensors"]["table_enabled"][:, 1].any())
    compact = variant_bank(bank, "separate_shell_transport")
    assert compact["metadata"]["table_shell_ids"] == [0, 2, 3]
    assert compact["tensors"]["table_coefficients"].shape == (2, 3, 3, 2)
    assert compact["tensors"]["table_coefficients"].numel() == variant_bank(bank, "generated_transport")["tensors"]["coefficients"].numel()
    _, diagnostic = predict_residual(compact, inputs(records), expected_scope=scope)
    withheld = (diagnostic["shell_ids"] == 1) & diagnostic["supported"]
    assert bool(withheld.any())
    assert bool((diagnostic["selected_table_shell_ids"][withheld] == 0).all())


def test_radial_range_support_shared_at_inner_and_outer_unmeasured_distances():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    t = bank["tensors"]
    # Build exact chart coordinates at the center (below its observed min)
    # and beyond max radius but within the old 1.5x support radius, moving away
    # from the other chart so the nearest regional center remains unchanged.
    direction = t["centers"][0] - t["centers"][1]
    direction /= direction.norm()
    z = torch.stack((t["centers"][0], t["centers"][0] + direction * 1.2 * t["radii"][0]))
    x = t["input_mean"] + (z * t["feature_scale"]) @ t["input_basis"].T
    diagnostic = predict_geometry(bank, x, expected_scope=scope)
    assert diagnostic["supported"].tolist() == [False, False]
    expected = predict_residual(variant_bank(bank, "global"), x, expected_scope=scope)[0]
    for name in ("constant_transport", "linear_transport", "generated_transport", "identity_transport", "wrong_shell_transport", "separate_shell_transport"):
        actual, diagnostic = predict_residual(variant_bank(bank, name), x, expected_scope=scope)
        assert torch.equal(actual, expected)
        assert diagnostic["global_fallback_count"] == 2


def test_global_fallback_same_for_all_variants_and_constant_offsets_cancel_in_differences():
    scope, records, y = fixture()
    bank = fit(scope, records, y)
    far = torch.tensor([[100., 100., 100., 100.], [101., 100., 100., 100.]])
    expected = predict_residual(variant_bank(bank, "global"), far, expected_scope=scope)[0]
    for name in ("region_offsets", "constant_transport", "linear_transport", "generated_transport", "identity_transport", "wrong_shell_transport", "separate_shell_transport"):
        actual, diagnostic = predict_residual(variant_bank(bank, name), far, expected_scope=scope)
        assert torch.equal(actual, expected)
        assert diagnostic["global_fallback_count"] == 2
        assert torch.equal(actual[:, 1] - actual[:, 0], expected[:, 1] - expected[:, 0])
    chosen = records[:4]
    pairs = [TrajectoryPair("p0", "r0", "r1"), TrajectoryPair("p1", "r2", "r3")]
    reference = torch.tensor([[[1.], [2.], [3.], [5.]], [[2.], [4.], [6.], [10.]]])
    report = response_metrics(reference, reference + 1, chosen, pairs, torch.tensor([0, 0, 0, 1]), 1)
    assert report["output"]["relative_rmse"] > 0
    assert report["output_difference"]["relative_rmse"] == 0
    assert report["within_region_pairs"] == report["cross_region_pairs"] == 1


def test_heldout_targets_and_states_do_not_fit_geometry_or_responses():
    scope, records, y = fixture()
    hold = [replace(records[0], record_id="h0", trajectory_id="heldout", state=torch.tensor([5., 1., 0., 0.]))]
    first = fit(scope, records, y, heldout_records=hold)
    hold = [replace(hold[0], state=torch.tensor([1000., 1000., 1000., 1000.]))]
    second = fit(scope, records, y, heldout_records=hold)
    assert first["metadata"] == second["metadata"]
    assert all(torch.equal(value, second["tensors"][name]) for name, value in first["tensors"].items())
    for overlap in (records[0], replace(records[0], record_id="new ID")):
        with pytest.raises(ValueError, match="heldout"):
            fit(scope, records, y, heldout_records=[overlap])


def test_scope_context_uniform_finiteness_and_budget_guards():
    scope, records, y = fixture()
    with pytest.raises(ValueError, match="scope"):
        fit(replace(scope, layer=4), records, y)
    with pytest.raises(MemoryError):
        fit(scope, records, y, max_working_bytes=1)
    invalid = y.clone()
    invalid[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        fit(scope, records, invalid)
    weakened = [replace(records[0], stimulation_weight=.3)] + records[1:]
    with pytest.raises(ValueError, match="uniform"):
        fit(scope, weakened, y)
    bank = variant_bank(fit(scope, records, y, response_context_sha256="a" * 64), "generated_transport")
    with pytest.raises(ValueError, match="context"):
        predict_residual(bank, inputs(records), expected_scope=scope, expected_response_context_sha256="b" * 64)
    with pytest.raises(ValueError, match="scope"):
        predict_residual(bank, inputs(records), expected_scope=replace(scope, state_site="other"), expected_response_context_sha256="a" * 64)
    with pytest.raises(MemoryError):
        predict_residual(bank, inputs(records), expected_scope=scope, expected_response_context_sha256="a" * 64, max_working_bytes=1)
    with pytest.raises(MemoryError):
        predict_geometry(bank, inputs(records), expected_scope=scope, expected_response_context_sha256="a" * 64, max_working_bytes=1)


def test_global_overflow_checked_before_early_return():
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y), "global")
    bank["tensors"]["slopes"].fill_(1e30)
    with pytest.raises(ValueError, match="overflow"):
        predict_residual(bank, torch.full((1, 4), 1e30), expected_scope=scope)


def test_sparse_charts_share_global_fallback_and_retained_response_records():
    scope, records, y = fixture()
    selected = list(range(8)) + list(range(96, 192))
    bank = fit(scope, [records[i] for i in selected], y[:, selected])
    assert sum(bank["metadata"]["region_enabled"]) == 1
    assert len(bank["metadata"]["calibration_record_ids"]) == len(selected)
    assert len(bank["metadata"]["response_fit_record_ids"]) + len(bank["metadata"]["response_withheld_record_ids"]) == len(selected)
    assert bool((bank["tensors"]["table_enabled"].sum(dim=1) == 0).any())


def test_plain_bank_saved_restricted_roundtrip_and_actual_bytes(tmp_path):
    scope, records, y = fixture()
    bank = variant_bank(fit(scope, records, y), "generated_transport")
    receipt, loaded = _save_adapter(bank, tmp_path / "generated", scope)
    assert receipt["serialized_total_bytes"] == sum(p.stat().st_size for p in (tmp_path / "generated").iterdir())
    assert torch.equal(predict_residual(bank, inputs(records), expected_scope=scope)[0],
                       predict_residual(loaded, inputs(records), expected_scope=scope)[0])
