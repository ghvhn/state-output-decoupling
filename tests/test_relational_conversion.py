"""CPU algebra checks, not semantic validation of a definition intervention."""

import copy
import math

import pytest
import torch

from invariants.relational_conversion import (
    RELATIONAL_CONVERSION_VERSION,
    fit_relational_conversion,
    predict_relational_conversion,
    validate_relational_conversion,
)


def _linear_data():
    # More samples than the independent source dimensions, with held-out
    # queries between the observed states rather than duplicated training rows.
    x = torch.tensor([
        [-2., -1., .2], [-1., 1., -.4], [0., -1., .8],
        [1., 0., -.5], [2., 1., .3], [-1., 0., .9], [1., -1., -.7],
    ], dtype=torch.float64)
    transform = torch.tensor([
        [2., -.5, 1.], [.25, 3., -.75], [-1., .5, 2.],
    ], dtype=torch.float64)
    offset = torch.tensor([4., -2., .75], dtype=torch.float64)
    return x, x @ transform + offset, transform, offset


def _profile():
    x, y, _, _ = _linear_data()
    return fit_relational_conversion(x, y, ridge=1e-5)


def test_unseen_sources_produce_different_correct_changes():
    x, y, transform, offset = _linear_data()
    profile = fit_relational_conversion(x, y, ridge=1e-5)
    queries = [torch.tensor([-.3, .4, .1], dtype=torch.float64),
               torch.tensor([.6, -.25, -.2], dtype=torch.float64)]
    predictions = []
    for query in queries:
        delta, diagnostic = predict_relational_conversion(profile, query)
        torch.testing.assert_close(delta.to(torch.float64), query @ transform + offset, rtol=1e-4, atol=1e-4)
        assert diagnostic['adaptive'] and not diagnostic['clamped']
        assert diagnostic['diagnostic_only']
        assert delta.device.type == 'cpu' and delta.dtype == torch.float32
        predictions.append(delta)
    assert not torch.allclose(predictions[0], predictions[1])
    assert profile['diagnostics']['source_rank'] == 3
    assert profile['diagnostics']['observed_pair_change_rms'] > 0
    assert profile['diagnostics']['predicted_pair_change_rms'] > 0
    assert profile['diagnostics']['fit_residual_rms'] < 1e-4


def test_pair_difference_is_source_relation_transport_without_mean_offset():
    profile = _profile()
    a = torch.tensor([-.3, .4, .1])
    b = torch.tensor([.6, -.25, -.2])
    da, _ = predict_relational_conversion(profile, a)
    db, _ = predict_relational_conversion(profile, b)
    expected_change = (b.to(torch.float64) - a.to(torch.float64)) @ profile['source_relations'].double().T @ profile['coefficients'].double()
    torch.testing.assert_close((db-da).double(), expected_change, rtol=1e-5, atol=1e-6)


def test_raw_change_magnitude_is_preserved_without_unit_normalization():
    x, y, _, _ = _linear_data()
    profile = fit_relational_conversion(x, y, ridge=1e-4)
    scaled = fit_relational_conversion(x, 7.2*y, ridge=1e-4)
    query = torch.tensor([.2, .1, -.3])
    original, _ = predict_relational_conversion(profile, query)
    larger, diagnostic = predict_relational_conversion(scaled, query)
    torch.testing.assert_close(larger, 7.2*original, rtol=1e-5, atol=1e-5)
    assert float(larger.norm()) > 10
    assert diagnostic['predicted_delta_norm'] == pytest.approx(float(larger.double().norm()))
    assert scaled['changes_norm_mean'] == pytest.approx(7.2*profile['changes_norm_mean'])


def test_cancelling_changes_keep_mean_row_magnitude_denominator():
    x = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
    y = 3*x
    profile = fit_relational_conversion(x, y)
    assert torch.count_nonzero(profile['center_change']) == 0
    assert profile['changes_norm_mean'] == pytest.approx(4.5)
    assert profile['diagnostics']['adaptive']
    delta, diagnostic = predict_relational_conversion(profile, torch.tensor([.5, 0.]))
    assert float(delta.norm()) > 1
    assert diagnostic['changes_norm_mean'] > 0


def test_pair_diagnostics_measure_actual_training_variation():
    x, y, _, _ = _linear_data()
    profile = fit_relational_conversion(x, y)
    actual = torch.sqrt(torch.pdist(y).square().mean())
    assert profile['diagnostics']['observed_pair_change_rms'] == pytest.approx(float(actual))
    predicted = (profile['center_change'].double() + profile['source_relations'].double() @ profile['source_relations'].double().T @ profile['coefficients'].double()).float().double()
    predicted_rms = torch.sqrt(torch.pdist(predicted).square().mean())
    assert profile['diagnostics']['predicted_pair_change_rms'] == pytest.approx(float(predicted_rms))


def test_out_of_domain_relation_is_radially_clamped():
    profile = _profile()
    center = profile['center_source'].double()
    direction = torch.tensor([2., -1., .5], dtype=torch.float64)
    direction = direction / direction.norm()
    limit = profile['training_radius'] * profile['extrapolation']
    boundary = center + direction * limit
    far = center + direction * limit * 100
    expected, _ = predict_relational_conversion(profile, boundary)
    actual, diagnostic = predict_relational_conversion(profile, far)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert diagnostic['clamped'] and diagnostic['outside_training_support']
    assert diagnostic['ood_ratio'] == pytest.approx(150)
    assert diagnostic['source_scale'] == pytest.approx(.01)
    assert diagnostic['max_source_distance'] == pytest.approx(limit)
    assert diagnostic['source_distance'] == pytest.approx(limit*100)


def test_outside_training_radius_but_inside_bound_is_reported_without_clamping():
    profile = _profile()
    center = profile['center_source'].double()
    query = center + torch.tensor([1., 0., 0.]) * profile['training_radius'] * 1.2
    _, diagnostic = predict_relational_conversion(profile, query)
    assert diagnostic['outside_training_support']
    assert not diagnostic['clamped']
    assert diagnostic['ood_ratio'] == pytest.approx(1.2)
    assert diagnostic['source_scale'] == 1


def test_translation_of_source_origin_preserves_conversion():
    x, y, _, _ = _linear_data()
    origin = torch.tensor([10., -25., 4.], dtype=torch.float64)
    query = torch.tensor([.2, -.4, .1], dtype=torch.float64)
    first = fit_relational_conversion(x, y, ridge=.003)
    second = fit_relational_conversion(x+origin, y, ridge=.003)
    da, _ = predict_relational_conversion(first, query)
    db, _ = predict_relational_conversion(second, query+origin)
    torch.testing.assert_close(da, db, rtol=1e-5, atol=1e-5)
    assert first['training_radius'] == pytest.approx(second['training_radius'])


def test_permuting_pairs_does_not_change_unseen_prediction():
    x, y, _, _ = _linear_data()
    order = torch.tensor([6, 0, 3, 1, 5, 2, 4])
    first = fit_relational_conversion(x, y, ridge=.003)
    second = fit_relational_conversion(x[order], y[order], ridge=.003)
    query = torch.tensor([.2, -.4, .1])
    da, _ = predict_relational_conversion(first, query)
    db, _ = predict_relational_conversion(second, query)
    torch.testing.assert_close(da, db, rtol=1e-5, atol=1e-5)
    assert first['diagnostics']['fit_residual_rms'] == pytest.approx(second['diagnostics']['fit_residual_rms'])


def test_rank_deficient_source_space_is_supported():
    x = torch.tensor([[-2., -4., 0.], [-1., -2., 0.], [1., 2., 0.], [2., 4., 0.]])
    y = torch.stack((3*x[:, 0]+2, -x[:, 0], x[:, 0]*.5), dim=1)
    profile = fit_relational_conversion(x, y, ridge=1e-5)
    delta, _ = predict_relational_conversion(profile, torch.tensor([.5, 1., 0.]))
    torch.testing.assert_close(delta, torch.tensor([3.5, -.5, .25]), rtol=1e-4, atol=1e-4)
    assert profile['diagnostics']['source_rank'] == 1


def test_constant_changes_are_honestly_constant_despite_varied_sources():
    x, _, _, _ = _linear_data()
    y = torch.tensor([12., -7., 3.], dtype=torch.float64).expand_as(x).clone()
    profile = fit_relational_conversion(x, y)
    assert profile['diagnostics']['changes_constant']
    assert not profile['diagnostics']['adaptive']
    assert profile['diagnostics']['observed_pair_change_rms'] == 0
    assert profile['diagnostics']['predicted_pair_change_rms'] == 0
    assert torch.count_nonzero(profile['coefficients']) == 0
    for query in (torch.tensor([.2, -.4, .1]), torch.tensor([300., -400., 100.])):
        delta, _ = predict_relational_conversion(profile, query)
        torch.testing.assert_close(delta, y[0].float())


def test_identical_sources_cannot_explain_different_changes():
    x = torch.ones(3, 2)
    y = torch.tensor([[1., 2.], [2., 2.], [3., 2.]])
    with pytest.raises(ValueError, match='unidentifiable'):
        fit_relational_conversion(x, y)


def test_identical_sources_and_changes_remain_constant_with_defined_ood_fields():
    x = torch.tensor([[10., -2.]]).expand(3, 2)
    y = torch.tensor([[8., 3.]]).expand(3, 2)
    profile = fit_relational_conversion(x, y)
    assert profile['training_radius'] == 0
    assert profile['diagnostics']['source_rank'] == 0
    assert not profile['diagnostics']['adaptive']
    same, at_center = predict_relational_conversion(profile, x[0])
    elsewhere, outside = predict_relational_conversion(profile, torch.tensor([12., -2.]))
    torch.testing.assert_close(same, elsewhere)
    assert at_center['ood_ratio'] == 0 and not at_center['clamped']
    assert outside['ood_ratio'] is None and outside['clamped']
    assert outside['source_scale'] == 0 and outside['source_distance'] == 2


def test_all_zero_changes_are_allowed_only_as_explicit_nonadaptive_model():
    x, _, _, _ = _linear_data()
    profile = fit_relational_conversion(x, torch.zeros_like(x))
    delta, diagnostic = predict_relational_conversion(profile, torch.zeros(3))
    assert profile['changes_norm_mean'] == 0
    assert profile['diagnostics']['changes_constant']
    assert not diagnostic['adaptive'] and torch.count_nonzero(delta) == 0


def test_float32_prediction_rounding_does_not_fabricate_adaptation():
    x = torch.tensor([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]], dtype=torch.float64)
    # These changes differ in float64 but the final stored output cannot retain
    # that difference at this common origin. Report both facts explicitly.
    y = x + 1e8
    profile = fit_relational_conversion(x, y, ridge=1e-5)
    assert not profile['diagnostics']['changes_constant']
    assert profile['diagnostics']['observed_pair_change_rms'] > 0
    assert profile['diagnostics']['predicted_pair_change_rms'] == 0
    assert not profile['diagnostics']['adaptive']
    first, _ = predict_relational_conversion(profile, x[0])
    last, _ = predict_relational_conversion(profile, x[-1])
    torch.testing.assert_close(first, last, rtol=0, atol=0)


def test_fit_predict_and_validate_do_not_mutate_or_alias_inputs():
    x, y, _, _ = _linear_data()
    saved_x, saved_y = x.clone(), y.clone()
    profile = fit_relational_conversion(x, y)
    saved_profile = copy.deepcopy(profile)
    query = torch.tensor([.2, -.4, .1])
    saved_query = query.clone()
    predict_relational_conversion(profile, query)
    clean = validate_relational_conversion(profile)
    clean['coefficients'][0, 0] = 12345
    clean['diagnostics']['adaptive'] = False
    torch.testing.assert_close(x, saved_x)
    torch.testing.assert_close(y, saved_y)
    torch.testing.assert_close(query, saved_query)
    for key in ('center_source', 'center_change', 'source_relations', 'coefficients'):
        torch.testing.assert_close(profile[key], saved_profile[key])
    assert profile['diagnostics'] == saved_profile['diagnostics']


def test_safe_tensor_storage_roundtrip_preserves_raw_conversion(tmp_path):
    profile = _profile()
    path = tmp_path/'relation.pt'
    torch.save(profile, path)
    loaded = validate_relational_conversion(torch.load(path, weights_only=True, map_location='cpu'), expected_width=3)
    assert loaded['version'] == RELATIONAL_CONVERSION_VERSION
    for key in ('center_source', 'center_change', 'source_relations', 'coefficients'):
        assert loaded[key].dtype == torch.float32 and loaded[key].device.type == 'cpu'
    query = torch.tensor([.2, -.4, .1])
    original, _ = predict_relational_conversion(profile, query)
    restored, _ = predict_relational_conversion(loaded, query)
    torch.testing.assert_close(original, restored, rtol=0, atol=0)
    # These are raw state/transport arrays, not unit direction storage.
    assert loaded['center_change'].norm() > 1


@pytest.mark.parametrize('ridge', [0, -1, math.nan, math.inf, None, True])
def test_invalid_ridge_rejected(ridge):
    x, y, _, _ = _linear_data()
    with pytest.raises(ValueError, match='ridge'):
        fit_relational_conversion(x, y, ridge=ridge)


@pytest.mark.parametrize('extrapolation', [0, -1, math.nan, math.inf, None, True])
def test_invalid_extrapolation_rejected(extrapolation):
    x, y, _, _ = _linear_data()
    with pytest.raises(ValueError, match='extrapolation'):
        fit_relational_conversion(x, y, extrapolation=extrapolation)


@pytest.mark.parametrize('x,y,error', [
    (torch.ones(2, 2), torch.ones(2, 2), 'N >= 3'),
    (torch.empty(3, 0), torch.empty(3, 0), 'D >= 1'),
    (torch.ones(3, 2), torch.ones(3, 3), 'same'),
    (torch.ones(3), torch.ones(3), '2-dimensional'),
    (torch.ones(3, 2, dtype=torch.int64), torch.ones(3, 2), 'floating'),
    (torch.ones(3, 2), torch.full((3, 2), math.nan), 'finite'),
    (torch.full((3, 2), math.inf), torch.ones(3, 2), 'finite'),
])
def test_invalid_training_pairs_rejected(x, y, error):
    with pytest.raises(ValueError, match=error):
        fit_relational_conversion(x, y)


@pytest.mark.parametrize('source,error', [
    (torch.ones(2), 'shape'), (torch.ones(1, 3), '1-dimensional'),
    (torch.ones(3, dtype=torch.int64), 'floating'),
    (torch.tensor([math.nan, 0., 0.]), 'finite'),
])
def test_invalid_prediction_source_rejected(source, error):
    with pytest.raises(ValueError, match=error):
        predict_relational_conversion(_profile(), source)


@pytest.mark.parametrize('key,value,error', [
    ('schema', True, 'schema'), ('version', 'old', 'version'),
    ('width', True, 'width'), ('n_samples', 2, 'n_samples'),
    ('training_radius', math.inf, 'training_radius'),
    ('training_radius', 1000., 'training_radius'),
    ('changes_norm_mean', 0., 'changes_norm_mean'),
    ('scaled_ridge', math.nan, 'scaled_ridge'),
    ('scaled_ridge', 500., 'scaled_ridge'),
])
def test_profile_metadata_is_validated(key, value, error):
    profile = _profile()
    profile[key] = value
    with pytest.raises(ValueError, match=error):
        validate_relational_conversion(profile)


@pytest.mark.parametrize('key', ['center_source', 'center_change', 'source_relations', 'coefficients'])
@pytest.mark.parametrize('fault', ['missing', 'dtype', 'shape', 'nan'])
def test_profile_array_shape_dtype_presence_and_finiteness(key, fault):
    profile = _profile()
    if fault == 'missing':
        del profile[key]
    elif fault == 'dtype':
        profile[key] = profile[key].double()
    elif fault == 'shape':
        profile[key] = profile[key].reshape(-1, 1, 1)
    else:
        profile[key].view(-1)[0] = math.nan
    with pytest.raises(ValueError, match=key):
        validate_relational_conversion(profile)


def test_profile_requires_exact_expected_width_and_consistent_diagnostics():
    profile = _profile()
    with pytest.raises(ValueError, match='expected width'):
        validate_relational_conversion(profile, expected_width=4)
    for key, value, error in [
        ('source_rank', 0, 'source_rank'),
        ('adaptive', False, 'adaptive'),
        ('predicted_pair_change_rms', 0., 'predicted variation'),
        ('diagnostic_only', False, 'candidate geometric'),
    ]:
        broken = copy.deepcopy(profile)
        broken['diagnostics'][key] = value
        with pytest.raises(ValueError, match=error):
            validate_relational_conversion(broken)


def test_numerical_overflow_rejected_in_fit_or_storage():
    x = torch.tensor([[-1e300, 0.], [0., 0.], [1e300, 0.]], dtype=torch.float64)
    with pytest.raises(ValueError, match='finite numerical range'):
        fit_relational_conversion(x, torch.ones_like(x))
    x = torch.tensor([[-1., 0.], [0., 0.], [1., 0.]], dtype=torch.float64)
    with pytest.raises(ValueError, match='float32 storage'):
        fit_relational_conversion(x, torch.full_like(x, 1e100))
