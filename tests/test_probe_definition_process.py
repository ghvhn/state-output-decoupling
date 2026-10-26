"""Verify scheduled residual operations without claiming semantic completion."""
import json
from types import SimpleNamespace

import pytest
import torch

from invariants.probe_definition_process import (
    STEP_ORDER, DefinitionStepProcess, build_definition_process_handles,
)


class ResidualModel(torch.nn.Module):
    def __init__(self, count=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(count)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class RootModel(torch.nn.Module):
    def __init__(self, count=2):
        super().__init__()
        self.model = ResidualModel(count)

    def forward(self, hidden):
        return self.model(hidden)


def setup(count=1):
    root = RootModel(count)
    model = SimpleNamespace(device='cpu', model=root)
    subject = {layer: torch.tensor([1., 0., 0.]) for layer in range(count)}
    directions = (torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]),
                  torch.tensor([0., -1., 0.]), torch.tensor([0., 0., -1.]))
    bank = {step: {layer: directions[index].clone() for layer in range(count)}
            for index, step in enumerate(STEP_ORDER)}
    return model, subject, bank


def remove(handles):
    for handle in handles:
        handle.remove()


def test_four_operations_switch_on_absolute_positions_and_restore_history():
    model, subject, bank = setup()
    process = DefinitionStepProcess(step_tokens=2)
    handles = build_definition_process_handles(model, subject, bank, process=process, cap_fraction=1.)
    try:
        initial = torch.ones(1, 4, 3)
        first = model.model(initial)
        torch.testing.assert_close(first[:, :3], initial[:, :3])
        expected_decision = initial[0, 3] + initial[0, 3].norm() * torch.tensor([.02, .04, 0.])
        torch.testing.assert_close(first[0, 3], expected_decision)
        directions = [torch.tensor([0., 1., 0.]), torch.tensor([0., 0., 1.]),
                      torch.tensor([0., -1., 0.]), torch.tensor([0., 0., -1.])]
        for position in range(1, 8):
            decoded = torch.ones(1, 1, 3)
            actual = model.model(decoded)
            expected = decoded + .04 * decoded.norm(dim=-1, keepdim=True) * directions[position // 2]
            torch.testing.assert_close(actual, expected)
        rebuilt = torch.ones(1, 11, 3)
        historical = model.model(rebuilt)
        torch.testing.assert_close(historical[:, :4], first)
        for position in range(1, 8):
            expected = rebuilt[0, position + 3] + .04 * rebuilt[0, position + 3].norm() * directions[position // 2]
            torch.testing.assert_close(historical[0, position + 3], expected)
        # A repeated prefix is replay, not another generated position.
        model.model(rebuilt)
        assert process.trace()['observed_output_positions'] == 8
        outside = torch.ones(1, 1, 3)
        torch.testing.assert_close(model.model(outside), outside)
        trace = process.trace()
        assert [window['scheduled_positions'] for window in trace['windows']] == [2, 2, 2, 2]
        assert [window['injected_positions'] for window in trace['windows']] == [2, 2, 2, 2]
        assert [event['step'] for event in trace['events']] == list(STEP_ORDER)
        assert trace['diagnostic_only'] and trace['decision_position'] == 3
        json.dumps(trace)
    finally:
        remove(handles)
    torch.testing.assert_close(model.model(initial), initial)


def test_multiple_layers_advance_cached_cursor_once_per_root_forward():
    model, subject, bank = setup(count=3)
    process = DefinitionStepProcess(step_tokens=1)
    handles = build_definition_process_handles(model, subject, bank, process=process)
    try:
        model.model(torch.ones(1, 4, 3))
        assert process.trace()['furthest_output_position'] == 0
        model.model(torch.ones(1, 1, 3))
        assert process.trace()['furthest_output_position'] == 1
        assert process.trace()['forward_count'] == 2
        assert [window['scheduled_positions'] for window in process.trace()['windows']] == [1, 1, 0, 0]
        model.model(torch.ones(1, 5, 3))
        assert process.trace()['furthest_output_position'] == 1
    finally:
        remove(handles)


def test_shared_cap_combines_fixed_subject_and_one_step_only():
    model, subject, bank = setup()
    process = DefinitionStepProcess(step_tokens=1)
    handles = build_definition_process_handles(model, subject, bank, process=process,
                                               step_gains={'compare': .12}, cap_fraction=.05)
    try:
        state = torch.ones(1, 4, 3)
        after = model.model(state)
        direction = torch.tensor([.02, .12, 0.])
        expected = .05 * state[0, -1].norm() * direction / direction.norm()
        torch.testing.assert_close(after[0, -1] - state[0, -1], expected)
        torch.testing.assert_close(after[:, :-1], state[:, :-1])
        # The next operation receives neither the subject nor compare's axis.
        decoded = state[:, -1:]
        actual = model.model(decoded)
        expected_decode = decoded + .04 * decoded.norm(dim=-1, keepdim=True) * torch.tensor([0., 0., 1.])
        torch.testing.assert_close(actual, expected_decode)
    finally:
        remove(handles)


def test_default_schedule_is_four_96_position_windows():
    trace = DefinitionStepProcess().trace()
    assert trace['budget_positions'] == 384
    assert [(window['step'], window['start'], window['stop']) for window in trace['windows']] == [
        ('compare', 0, 96), ('isolate', 96, 192), ('distill', 192, 288), ('verify', 288, 384)]


def test_zero_gain_stage_is_scheduled_but_has_no_step_injection():
    model, subject, bank = setup()
    process = DefinitionStepProcess(step_tokens=1, step_gains={'isolate': 0.})
    handles = build_definition_process_handles(model, subject, bank, process=process)
    try:
        model.model(torch.ones(1, 4, 3))
        decoded = torch.ones(1, 1, 3)
        torch.testing.assert_close(model.model(decoded), decoded)
        isolated = process.trace()['windows'][1]
        assert isolated['entered'] and isolated['scheduled_positions'] == 1
        assert isolated['gain'] == 0. and isolated['injected_positions'] == 0
    finally:
        remove(handles)


def test_zero_live_cap_records_schedule_without_claiming_injection():
    model, subject, bank = setup()
    process = DefinitionStepProcess(step_tokens=1)
    handles = build_definition_process_handles(model, subject, bank, process=process, cap_fraction=0.)
    try:
        state = torch.ones(1, 4, 3)
        torch.testing.assert_close(model.model(state), state)
        window = process.trace()['windows'][0]
        assert window['scheduled_positions'] == 1 and window['injected_positions'] == 0
    finally:
        remove(handles)


@pytest.mark.parametrize('value', [0, -1, 1.5, True, float('nan'), float('inf'), 'invalid'])
def test_invalid_window_lengths_fail_before_registration(value):
    model, subject, bank = setup()
    with pytest.raises(ValueError, match='positive finite integer'):
        build_definition_process_handles(model, subject, bank, step_tokens=value)
    assert not model.model._forward_pre_hooks
    assert not model.model.model.layers[0]._forward_hooks


@pytest.mark.parametrize('value', [None, 'invalid', float('nan'), float('inf'), 1.01, -1.01])
def test_invalid_step_gain_fails_before_registration(value):
    model, subject, bank = setup()
    with pytest.raises(ValueError, match='step gain for isolate must be finite'):
        build_definition_process_handles(model, subject, bank, step_gains={'isolate': value})
    assert not model.model._forward_pre_hooks


def test_all_steps_require_exact_subject_layers_even_at_zero_gain():
    model, subject, bank = setup(count=2)
    bank['verify'].pop(1)
    with pytest.raises(ValueError, match='step verify must exactly match subject layers'):
        build_definition_process_handles(model, subject, bank, step_gains={'verify': 0.})
    assert not model.model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in model.model.model.layers)


@pytest.mark.parametrize('bad', [torch.zeros(3), torch.tensor([float('nan'), 1., 0.]), torch.ones(4)])
def test_invalid_step_vectors_register_no_hooks(bad):
    model, subject, bank = setup(count=2)
    bank['verify'][1] = bad
    with pytest.raises(ValueError, match='step verify.*finite and nonzero at layer 1'):
        build_definition_process_handles(model, subject, bank)
    assert not model.model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in model.model.model.layers)


def test_partial_registration_failure_preserves_unrelated_hooks(monkeypatch):
    model, subject, bank = setup(count=2)
    original_pre = model.model.register_forward_pre_hook(lambda _module, _inputs: None)
    original_layer = model.model.model.layers[0].register_forward_hook(lambda _module, _inputs, out: out + 1)
    pre_ids = set(model.model._forward_pre_hooks)
    layer_ids = set(model.model.model.layers[0]._forward_hooks)
    process = DefinitionStepProcess()
    monkeypatch.setattr(model.model.model.layers[1], 'register_forward_hook',
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('registration failed')))
    try:
        with pytest.raises(RuntimeError, match='registration failed'):
            build_definition_process_handles(model, subject, bank, process=process)
        assert set(model.model._forward_pre_hooks) == pre_ids
        assert set(model.model.model.layers[0]._forward_hooks) == layer_ids
        assert not process._claimed
        state = torch.ones(1, 4, 3)
        torch.testing.assert_close(model.model(state), state + 1)
    finally:
        original_pre.remove()
        original_layer.remove()


def test_controller_window_mismatch_rejects_before_hooks():
    model, subject, bank = setup()
    with pytest.raises(ValueError, match='match the supplied process controller'):
        build_definition_process_handles(model, subject, bank, process=DefinitionStepProcess(2), step_tokens=3)
    assert not model.model._forward_pre_hooks
