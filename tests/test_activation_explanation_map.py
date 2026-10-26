"""State maps must adapt, align exact prefixes, and preserve runtime isolation."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from invariants import activation_explanation_map as maps, probe_activation_explain as core
from invariants.config import AgenticConfig
from invariants.relational_conversion import fit_relational_conversion, predict_relational_conversion
from test_probe_activation_explain import IdentityRootModel, NativeTokenizer, native_prompt


def profile():
    source = torch.tensor([[1., 2., 3.], [2., 1., 3.], [3., 2., 2.], [2., 4., 1.]])
    change = source * torch.tensor([.2, -.3, .1]) + torch.tensor([.4, .2, -.1])
    return fit_relational_conversion(source, change)


@pytest.fixture
def wrapped():
    return SimpleNamespace(model=IdentityRootModel(2), device=torch.device('cpu'),
                           d_model=3, n_layers=2, tok=NativeTokenizer())


def attach(model, **kwargs):
    return maps.build_mapped_explanation_handles(
        model, {0: torch.tensor([1., 0., 0.]), 1: torch.tensor([1., 0., 0.])},
        {1: profile()}, operation_gain=kwargs.pop('operation_gain', .5),
        subject_gain=kwargs.pop('subject_gain', 0.), cap_fraction=kwargs.pop('cap_fraction', 1.), **kwargs)


def test_compiled_map_matches_existing_bounded_conversion_and_varies_with_state():
    fitted = profile()
    compiled = maps._compiled(fitted, 3, 'cpu')
    rows = torch.tensor([[1., 2., 3.], [3., 2., 2.], [20., 40., 50.]])
    actual, _distance, clamped, outside = maps.evaluate_compiled_map(compiled, rows)
    expected = torch.stack([predict_relational_conversion(fitted, row)[0] for row in rows])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(actual[0], actual[1])
    assert clamped[-1] and outside[-1]


def test_current_state_map_affects_assistant_positions_and_cached_decode_not_user_body(wrapped):
    handles, telemetry = attach(wrapped)
    try:
        hidden = torch.tensor([[[9., 8., 7.], [5., 4., 3.], [1., 2., 3.]]])
        initial = wrapped.model(hidden)
        decoded = wrapped.model(torch.tensor([[[3., 2., 2.]]]))
        rebuilt_input = torch.cat((hidden, torch.tensor([[[3., 2., 2.]]])), dim=1)
        rebuilt = wrapped.model(rebuilt_input)
        torch.testing.assert_close(initial[:, :-1], hidden[:, :-1])
        torch.testing.assert_close(rebuilt[:, :2], hidden[:, :2])
        torch.testing.assert_close(rebuilt[:, 2:3], initial[:, 2:3])
        torch.testing.assert_close(rebuilt[:, 3:4], decoded)
        first_delta = initial[0, -1] - hidden[0, -1]
        next_delta = decoded[0, 0] - torch.tensor([3., 2., 2.])
        assert not torch.allclose(first_delta, next_delta)
        stats = telemetry['layers']['1']
        assert stats['evaluations'] == 4 and stats['rebuild_evaluations'] == 2
        assert stats['delta_change_max'] > 0
        assert telemetry['state_conditioned'] and not telemetry['static_model']
    finally:
        for handle in handles:
            handle.remove()
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)


def test_same_current_state_gets_same_correction_at_different_decode_positions(wrapped):
    handles, _ = attach(wrapped)
    try:
        wrapped.model(torch.ones(1, 3, 3))
        state = torch.tensor([[[2., 1., 3.]]])
        first = wrapped.model(state)
        second = wrapped.model(state)
        torch.testing.assert_close(first, second)
        expected = .5 * predict_relational_conversion(profile(), state[0, 0])[0]
        torch.testing.assert_close(first[0, 0] - state[0, 0], expected)
    finally:
        for handle in handles:
            handle.remove()


def test_subject_reference_and_map_share_one_per_row_cap(wrapped):
    wrapped.model = IdentityRootModel(1)
    handles, _ = maps.build_mapped_explanation_handles(
        wrapped, {0: torch.tensor([1., 0., 0.])}, {0: profile()}, operation_gain=1.,
        subject_gain=1., reference={0: torch.tensor([0., 1., 0.])}, reference_gain=1., cap_fraction=.015)
    try:
        state = torch.tensor([[[1., 2., 3.], [2., 1., 3.]]])
        output = wrapped.model(state)
        change = output[0, -1] - state[0, -1]
        prediction = predict_relational_conversion(profile(), state[0, -1])[0]
        uncapped = prediction + state[0, -1].norm() * torch.tensor([1., 1., 0.])
        torch.testing.assert_close(change, .015 * state[0, -1].norm() * uncapped / uncapped.norm())
        torch.testing.assert_close(output[:, :-1], state[:, :-1])
    finally:
        for handle in handles:
            handle.remove()


def test_constant_map_is_rejected_before_any_hook_attaches(wrapped):
    source = torch.tensor([[1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
    constant = fit_relational_conversion(source, torch.ones_like(source))
    with pytest.raises(ValueError, match='state-dependent'):
        maps.build_mapped_explanation_handles(wrapped, {0: torch.ones(3)}, {0: constant},
            operation_gain=.12, subject_gain=.02)
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)


def test_offloaded_meta_block_uses_sequential_hook_execution_device(wrapped):
    block = nn.Linear(3, 3, device='meta')
    block._hf_hook = SimpleNamespace(hooks=(SimpleNamespace(execution_device=None),
                                            SimpleNamespace(execution_device='cpu')))
    wrapped.model.model.layers[1] = block
    handles, _ = attach(wrapped)
    try:
        hook = next(iter(block._forward_hooks.values()))
        state = torch.tensor([[[1., 2., 3.]]])
        actual = hook(block, (), state)
        expected = state + .5 * predict_relational_conversion(profile(), state[0, 0])[0]
        torch.testing.assert_close(actual, expected)
        assert actual.device.type == 'cpu'
    finally:
        for handle in handles:
            handle.remove()


def test_map_accepts_same_nonnegative_cap_domain_as_engine(wrapped):
    handles, _ = attach(wrapped, cap_fraction=2.)
    for handle in handles:
        handle.remove()
    with pytest.raises(ValueError, match='nonnegative'):
        attach(wrapped, cap_fraction=-1.)


def test_invalid_later_profile_leaves_no_partial_hooks(wrapped):
    damaged = copy.deepcopy(profile())
    damaged['coefficients'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        maps.build_mapped_explanation_handles(wrapped, {0: torch.ones(3), 1: torch.ones(3)},
            {0: profile(), 1: damaged}, operation_gain=.12, subject_gain=.02)
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)


class TrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([nn.Identity()]))
        self.inputs = []

    def forward(self, input_ids, **kwargs):
        self.inputs.append(input_ids.detach().clone())
        ids = input_ids.float()
        accumulated = ids.cumsum(-1) / 100.
        positions = torch.arange(ids.shape[1]).reshape(1, -1).expand_as(ids).float() / 10.
        hidden = torch.stack((ids / 10., accumulated.square(), positions), dim=-1)
        return self.model.layers[0](hidden)

    def generate(self, input_ids, **kwargs):
        continuation = torch.tensor([[6, 7, 8, 9, 10, 11]], device=input_ids.device)
        return torch.cat((input_ids, continuation), dim=1)


def trajectory_model():
    tok = NativeTokenizer()
    tok.decode = lambda ids, **kwargs: ' '.join(map(str, ids))
    tok.all_special_ids = []
    return SimpleNamespace(model=TrajectoryModel(), tok=tok, device='cpu', d_model=3, n_layers=1)


def test_trajectory_pairs_use_exact_same_prefix_ids_and_raw_prediction_position():
    model = trajectory_model()
    continuation = torch.tensor([6, 7, 8, 9])
    source_header, teacher_header = torch.tensor([[3, 4]]), torch.tensor([[4, 5, 6, 7, 8]])
    source = maps.capture_prediction_trajectory(model, source_header, continuation, (0,))[0]
    teacher = maps.capture_prediction_trajectory(model, teacher_header, continuation, (0,))[0]
    assert source.shape == teacher.shape == (4, 3)
    assert model.model.inputs[0].tolist() == [[3, 4, 6, 7, 8]]
    assert model.model.inputs[1].tolist() == [[4, 5, 6, 7, 8, 6, 7, 8]]
    assert source[0, 0] == .4 and teacher[0, 0] == .8
    torch.testing.assert_close(source[1:, 0], teacher[1:, 0])
    assert not model.model.model.layers[0]._forward_hooks


def test_calibration_fits_a_state_map_with_separate_held_out_trajectory():
    model = trajectory_model()
    profiles, metadata = maps.calibrate_activation_map(model, (0,), max_tokens=8, evaluate_held_out=True)
    assert profiles[0]['diagnostics']['adaptive']
    assert profiles[0]['n_samples'] == 12
    assert metadata['prefix_alignment'] == 'exact_shared_assistant_token_ids'
    assert len(metadata['trajectories']) == 2
    assert metadata['held_out']['whole_trajectory_held_out']
    assert metadata['held_out']['request'] not in metadata['calibration_requests']
    assert all(not block._forward_hooks for block in model.model.model.layers)


def test_generation_map_uses_blank_runtime_and_restores_hooks_and_config(monkeypatch, wrapped):
    monkeypatch.setattr(maps, 'get_or_mint_activation_map', Mock(return_value=({1: profile()}, {'map_cache_hit': False})))
    monkeypatch.setattr(core, 'get_or_mint_activation_axes', Mock(return_value=({}, {})))
    config = AgenticConfig()
    original = dict(vars(config))
    observed = []

    def generate(model, **kwargs):
        assert kwargs['instruction'] == native_prompt('')
        assert kwargs['pre_formatted'] and kwargs['generation_channel'] == 'reader'
        assert kwargs['vecs'] == {}
        observed.append(model.model(torch.ones(1, 3, 3)))
        observed.append(model.model(torch.tensor([[[3., 2., 2.]]])))
        return 'A free continuation.', {'tokens_generated': 4}

    result = core.generate_activation_explanation(wrapped, {0: torch.ones(3), 1: torch.ones(3)},
        generate=generate, config=config, map_enabled=True, reference_gain=0., subject_gain=0.,
        operation_gain=.12, cap_fraction=1.)
    assert result['metadata']['placement'] == 'current_state_map_assistant_positions'
    assert result['metadata']['map_telemetry']['layers']['1']['evaluations'] == 2
    assert not torch.allclose(observed[0][0, -1] - torch.ones(3), observed[1][0, 0] - torch.tensor([3., 2., 2.]))
    assert vars(config) == original
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)


def test_map_generation_failure_restores_hooks_and_config(monkeypatch, wrapped):
    monkeypatch.setattr(maps, 'get_or_mint_activation_map', Mock(return_value=({1: profile()}, {})))
    monkeypatch.setattr(core, 'get_or_mint_activation_axes', Mock(return_value=({}, {})))
    config = AgenticConfig()
    original = dict(vars(config))
    with pytest.raises(RuntimeError, match='generation failed'):
        core.generate_activation_explanation(wrapped, {0: torch.ones(3), 1: torch.ones(3)},
            generate=Mock(side_effect=RuntimeError('generation failed')), config=config,
            map_enabled=True, reference_gain=0.)
    assert vars(config) == original
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)


def test_map_cache_is_subject_owner_model_and_context_scoped(monkeypatch):
    model = trajectory_model()
    fitted = profile()
    mint = Mock(return_value=({0: fitted}, {'candidate': True}))
    monkeypatch.setattr(maps, 'calibrate_activation_map', mint)
    first, second = {}, {}
    subject = {0: torch.tensor([1., 0., 0.])}
    maps.get_or_mint_activation_map(model, subject, '', first)
    _, cached = maps.get_or_mint_activation_map(model, subject, '', first)
    assert cached['map_cache_hit'] and mint.call_count == 1
    maps.get_or_mint_activation_map(model, subject, '', second)
    maps.get_or_mint_activation_map(model, subject, 'other evidence', first)
    maps.get_or_mint_activation_map(model, {0: torch.tensor([0., 1., 0.])}, '', first)
    maps.get_or_mint_activation_map(trajectory_model(), subject, '', first)
    assert mint.call_count == 5
