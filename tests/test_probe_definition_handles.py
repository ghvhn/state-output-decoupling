"""Guard malformed definition controls before they can leave partial hooks."""
from types import SimpleNamespace

import pytest
import torch

from invariants.probe_behavior import build_definition_handles


class Layer(torch.nn.Identity):
    def __init__(self):
        super().__init__()
        self.registered = 0

    def register_forward_hook(self, *args, **kwargs):
        handle = super().register_forward_hook(*args, **kwargs)
        self.registered += 1
        return handle


def model_and_controls():
    layers = [Layer(), Layer()]
    model = SimpleNamespace(device='cpu', model=SimpleNamespace(model=SimpleNamespace(layers=layers)))
    subject = {layer: torch.tensor([1., 0., 0.]) for layer in range(2)}
    tasks = {'contrast': {layer: torch.tensor([0., 1., 0.]) for layer in range(2)}}
    return model, layers, subject, tasks


def test_enabled_task_requires_every_subject_layer_before_registration():
    model, layers, subject, tasks = model_and_controls()
    tasks['contrast'].pop(1)
    with pytest.raises(ValueError, match='contrast must cover every subject layer'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize('missing', ['operation', 'custom_control'])
def test_declared_enabled_control_requires_matching_axis_before_registration(missing):
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match=f'task {missing} requires a matching axis'):
        build_definition_handles(model, subject, tasks,
                                 task_gains={'contrast': .04, missing: .03})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize('gain', [None, 'invalid', float('nan'), float('inf'), 1.01, -1.01])
def test_missing_axis_does_not_hide_invalid_declared_gain(gain):
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match='task gain for operation must be finite and between -1 and 1'):
        build_definition_handles(model, subject, tasks,
                                 task_gains={'contrast': .04, 'operation': gain})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


def test_missing_default_enabled_axis_cannot_silently_disable_registered_control():
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match='task abstraction requires a matching axis'):
        build_definition_handles(model, subject, tasks)
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


def test_empty_named_axes_still_validate_explicit_active_controls():
    model, layers, subject, _ = model_and_controls()
    with pytest.raises(ValueError, match='task operation requires a matching axis'):
        build_definition_handles(model, subject, {}, task_gains={'operation': .03})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize('gain', [None, 'invalid', float('nan'), float('inf'), 1.01, -1.01])
def test_invalid_gain_has_task_specific_error_and_registers_nothing(gain):
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match='task gain for contrast must be finite and between -1 and 1'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': gain})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize('bad', [torch.zeros(3), torch.tensor([float('nan'), 1., 0.]),
                               torch.ones(4)])
def test_bad_later_task_direction_removes_already_registered_definition_hook(bad):
    model, layers, subject, tasks = model_and_controls()
    tasks['contrast'][1] = bad
    with pytest.raises(ValueError, match='task direction.*layer 1'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04})
    assert layers[0].registered == 1 and layers[1].registered == 0
    assert all(not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize('bad', [torch.zeros(3), torch.tensor([float('inf'), 0., 0.])])
def test_bad_later_subject_direction_removes_already_registered_definition_hook(bad):
    model, layers, subject, tasks = model_and_controls()
    subject[1] = bad
    with pytest.raises(ValueError, match='subject must be finite and nonzero at layer 1'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04})
    assert layers[0].registered == 1 and layers[1].registered == 0
    assert all(not layer._forward_hooks for layer in layers)


@pytest.mark.parametrize('target', ['subject', 'task'])
def test_finite_elements_with_overflowed_norm_are_rejected_and_cleaned_up(target):
    model, layers, subject, tasks = model_and_controls()
    overflowing = torch.tensor([3e38, 3e38, 0.], dtype=torch.float32)
    assert torch.isfinite(overflowing).all() and not torch.isfinite(overflowing.norm())
    if target == 'subject':
        subject[1] = overflowing
    else:
        tasks['contrast'][1] = overflowing
    with pytest.raises(ValueError, match='finite and nonzero at layer 1'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04})
    assert layers[0].registered == 1 and layers[1].registered == 0
    assert all(not layer._forward_hooks for layer in layers)


def test_failure_preserves_preexisting_unrelated_hook():
    model, layers, subject, tasks = model_and_controls()
    original = layers[0].register_forward_hook(lambda module, inputs, output: output + 1)
    original_ids = set(layers[0]._forward_hooks)
    tasks['contrast'][1] = torch.zeros(3)
    try:
        with pytest.raises(ValueError, match='task direction.*layer 1'):
            build_definition_handles(model, subject, tasks, task_gains={'contrast': .04})
        assert set(layers[0]._forward_hooks) == original_ids
        assert not layers[1]._forward_hooks
        state = torch.ones(1, 2, 3)
        torch.testing.assert_close(layers[0](state), state + 1)
    finally:
        original.remove()


def test_explicitly_disabled_task_does_not_require_layers_or_direction():
    model, layers, subject, _ = model_and_controls()
    handles = build_definition_handles(model, subject, {'contrast': {}}, task_gains={'contrast': 0.})
    try:
        assert len(handles) == 2
        state = torch.ones(1, 2, 3)
        after = layers[0](state)
        torch.testing.assert_close(after[:, :-1], state[:, :-1])
        assert after[0, -1, 0] > state[0, -1, 0]
        torch.testing.assert_close(after[0, -1, 1:], state[0, -1, 1:])
    finally:
        for handle in handles:
            handle.remove()
    assert all(not layer._forward_hooks for layer in layers)


def test_zero_gain_may_omit_axis_while_existing_task_remains_active():
    model, layers, subject, tasks = model_and_controls()
    handles = build_definition_handles(model, subject, tasks,
                                       task_gains={'contrast': .04, 'operation': 0.})
    try:
        state = torch.ones(1, 2, 3)
        after = layers[0](state)
        torch.testing.assert_close(after[:, :-1], state[:, :-1])
        assert after[0, -1, 0] > state[0, -1, 0]
        assert after[0, -1, 1] > state[0, -1, 1]
        torch.testing.assert_close(after[0, -1, 2], state[0, -1, 2])
    finally:
        for handle in handles:
            handle.remove()
    assert all(not layer._forward_hooks for layer in layers)


def test_empty_named_axes_with_zero_control_leave_subject_marker_available():
    model, layers, subject, _ = model_and_controls()
    handles = build_definition_handles(model, subject, {}, task_gains={'operation': 0.})
    try:
        state = torch.ones(1, 2, 3)
        after = layers[0](state)
        assert after[0, -1, 0] > state[0, -1, 0]
        torch.testing.assert_close(after[0, -1, 1:], state[0, -1, 1:])
    finally:
        for handle in handles:
            handle.remove()


def test_legacy_single_axis_keeps_inquiry_fraction_and_cleanup():
    model, layers, subject, tasks = model_and_controls()
    handles = build_definition_handles(model, subject, tasks['contrast'],
                                       inquiry_fraction=.04, subject_fraction=.02)
    try:
        state = torch.ones(1, 2, 3)
        after = layers[0](state)
        torch.testing.assert_close(after[:, :-1], state[:, :-1])
        assert after[0, -1, 0] > state[0, -1, 0]
        assert after[0, -1, 1] > state[0, -1, 1]
    finally:
        for handle in handles:
            handle.remove()
    assert all(not layer._forward_hooks for layer in layers)


def mixed_operation_controls():
    model, layers, subject, tasks = model_and_controls()
    tasks['operation'] = {layer: torch.tensor([0., 0., 1.]) for layer in range(2)}
    return model, layers, subject, tasks


def test_decision_operation_is_restored_only_at_original_prefill_position():
    model, layers, subject, tasks = mixed_operation_controls()
    handles = build_definition_handles(model, subject, tasks, cap_fraction=1.,
                                       task_gains={'contrast': .04, 'operation': .12},
                                       task_positions={'operation': 'decision'})
    try:
        prefill = torch.ones(1, 4, 3)
        first = layers[0](prefill)
        torch.testing.assert_close(first[:, :3], prefill[:, :3])
        expected = prefill[0, 3] + prefill[0, 3].norm() * torch.tensor([.02, .04, .12])
        torch.testing.assert_close(first[0, 3], expected)
        decode = torch.ones(1, 1, 3)
        for _ in range(2):
            decoded = layers[0](decode)
            expected_decode = decode + decode.norm(dim=-1, keepdim=True) * torch.tensor([0., .04, 0.])
            torch.testing.assert_close(decoded, expected_decode)
        rebuilt = torch.ones(1, 8, 3)
        restored = layers[0](rebuilt)
        torch.testing.assert_close(restored[:, :4], first)
        expected_new = rebuilt[:, 4:] + rebuilt[:, 4:].norm(dim=-1, keepdim=True) * torch.tensor([0., .04, 0.])
        torch.testing.assert_close(restored[:, 4:], expected_new)
    finally:
        for handle in handles:
            handle.remove()
    assert all(not layer._forward_hooks for layer in layers)


def test_subject_decision_pulse_and_generation_control_share_one_cap():
    model, layers, subject, tasks = mixed_operation_controls()
    handles = build_definition_handles(model, subject, tasks, cap_fraction=.05,
                                       task_gains={'contrast': .04, 'operation': .12},
                                       task_positions={'operation': 'decision'})
    try:
        state = torch.ones(1, 4, 3)
        after = layers[0](state)
        delta = after[0, -1] - state[0, -1]
        direction = torch.tensor([.02, .04, .12])
        expected = .05 * state[0, -1].norm() * direction / direction.norm()
        torch.testing.assert_close(delta, expected)
        torch.testing.assert_close(delta.norm(), .05 * state[0, -1].norm())
        decoded = layers[0](state[:, -1:])
        expected_decode = state[:, -1:] + state[:, -1:].norm(dim=-1, keepdim=True) * torch.tensor([0., .04, 0.])
        torch.testing.assert_close(decoded, expected_decode)
    finally:
        for handle in handles:
            handle.remove()


def test_operation_placement_defaults_to_generation():
    model, layers, subject, tasks = mixed_operation_controls()
    handles = build_definition_handles(model, subject, tasks, cap_fraction=1.,
                                       task_gains={'contrast': .04, 'operation': .12})
    try:
        state = torch.ones(1, 4, 3)
        layers[0](state)
        decoded = layers[0](state[:, -1:])
        expected = state[:, -1:] + state[:, -1:].norm(dim=-1, keepdim=True) * torch.tensor([0., .04, .12])
        torch.testing.assert_close(decoded, expected)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('placement', [None, '', 'all', 'Decision', 0])
def test_invalid_task_placement_registers_no_hooks(placement):
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match='task position for contrast must be decision or generation'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04},
                                 task_positions={'contrast': placement})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


def test_undeclared_task_position_reports_typo_before_hooks():
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match='task position for operaton names an undeclared control'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04, 'operation': 0.},
                                 task_positions={'operaton': 'decision'})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


def test_disabled_operation_may_have_decision_placement_without_axis():
    model, layers, subject, tasks = model_and_controls()
    handles = build_definition_handles(model, subject, tasks, task_gains={'contrast': .04, 'operation': 0.},
                                       task_positions={'operation': 'decision'})
    try:
        state = torch.ones(1, 4, 3)
        first = layers[0](state)
        decoded = layers[0](state[:, -1:])
        torch.testing.assert_close(first[..., 2], state[..., 2])
        torch.testing.assert_close(decoded[..., 2], state[:, -1:, 2])
    finally:
        for handle in handles:
            handle.remove()


def test_decision_direction_failure_cleans_partial_hooks():
    model, layers, subject, tasks = mixed_operation_controls()
    tasks['operation'][1] = torch.zeros(3)
    with pytest.raises(ValueError, match='task direction.*layer 1'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04, 'operation': .12},
                                 task_positions={'operation': 'decision'})
    assert layers[0].registered == 1 and layers[1].registered == 0
    assert all(not layer._forward_hooks for layer in layers)
