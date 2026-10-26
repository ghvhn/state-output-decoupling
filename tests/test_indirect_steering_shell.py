"""Indirect helper plans reach real field hooks without steering their targets."""

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from invariants.generation_probe_capture import GENERATION_PROBE_SOURCE
from invariants.hebbian_plasticity import PlasticityOwners
from invariants.indirect_steering import IndirectSteering, handle_indirect_command
from invariants.trigger_tuner import TriggerTuner
from scripts import interactive_phenomenality as shell


def probe(name, axis, mass=0):
    return {
        'direction': {layer: torch.tensor(axis, dtype=torch.float32) for layer in (0, 1)},
        'framings': (f'I express {name}.', f'I do not express {name}.'),
        'measurement_source': GENERATION_PROBE_SOURCE,
        'history': deque([0.0], maxlen=40), 'released': False, 'mass': mass,
    }


def sample(index, target, helper):
    return {
        'probe_target': target, 'probe_helper': helper,
        'probe_measurement': {'source': GENERATION_PROBE_SOURCE,
                              'generation_id': f'generation-{index}', 'token_count': 4},
    }


@pytest.fixture
def isolated_field(monkeypatch):
    bodies = {'loaded': True, 'anchors': {}, 'probe_poles': {}, 'laws': {},
              'fields': {}, 'g_families': {}, 'qualities': {}, 'sets': {},
              'bands': {}, 'spaces': {}, 'rotations': {}, 'law_gates': {},
              'trigger_bodies': {}, 'impulses': {}}
    monkeypatch.setattr(shell, '_STEER_BODIES', bodies)
    monkeypatch.setattr(shell, '_GRAVITY_STATE', deepcopy(shell._GRAVITY_STATE))
    monkeypatch.setattr(shell, 'save_steer_bodies', lambda: None)
    monkeypatch.setattr(shell, 'save_probe_raw_histories', lambda *args, **kwargs: True)
    return bodies


@pytest.fixture
def setup_drive(tmp_path, isolated_field):
    tuner = TriggerTuner(tmp_path / 'tuner.json')
    tuner.register('prioritize_alpha', .1, kind='coefficient')
    tuner.register('prioritize_gravity', 1, kind='coefficient')
    probes = {
        'target': probe('target', [1, 0, 0, 0], mass=10),
        'helper': probe('helper', [0, 1, 0, 0]),
    }
    for name in probes:
        tuner.register('probe_' + name, 0)
    controller = tuner.indirect_controller = IndirectSteering()
    controller.configure('target', -1, ['+helper'])
    controller.set_role('helper', 'thinking')
    for index in range(16):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(sample(index, value, -value), probes)
    model = SimpleNamespace(device='cpu', model=SimpleNamespace(
        model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()]))))
    return model, probes, tuner, controller


def field_output(model, probes, tuner, **options):
    handles, description = shell.build_gravity_field_handles(model, probes, tuner, **options)
    state = torch.tensor([[[0., 0., 1., 0.]]])
    try:
        output = state.clone()
        for layer in model.model.model.layers:
            output = layer(output)
        return output, len(handles), description
    finally:
        for handle in handles:
            handle.remove()


def test_helper_plan_reaches_both_layer_hooks_while_target_mass_is_suspended(setup_drive):
    model, probes, tuner, _controller = setup_drive
    before_mass = probes['target']['mass']
    before_history = list(probes['target']['history'])
    output, count, description = field_output(model, probes, tuner, cap_fraction=.01)
    assert count == 2, description
    assert output[0, 0, 1] > 0
    assert output[0, 0, 0] == 0
    assert probes['target']['mass'] == before_mass
    assert not probes['target']['released']
    assert list(probes['target']['history']) == before_history


def test_automatic_helper_selection_reaches_field_without_via(setup_drive):
    model, probes, tuner, controller = setup_drive
    handle_indirect_command(['minimize', 'target'], controller, probes, tuner)
    assert controller.routes['target']['controls'] is None
    output, count, _description = field_output(model, probes, tuner)
    assert count == 2 and output[0, 0, 1] > 0
    assert output[0, 0, 0] == 0


def test_saved_target_alias_inverse_and_derived_masses_cannot_bypass_guard(setup_drive):
    model, probes, tuner, _controller = setup_drive
    probes['alias'] = probe('alias', [1, 0, 0, 0], mass=100)
    probes['inverse'] = probe('inverse', [-1, 0, 0, 0], mass=100)
    probes['derived'] = probe('derived', [0, 0, 0, 1], mass=100)
    probes['derived']['framings'] = ('composed: target + helper', '')
    output, _count, _description = field_output(model, probes, tuner)
    assert output[0, 0, 1] > 0
    assert output[0, 0, 0] == output[0, 0, 3] == 0
    assert probes['derived']['mass'] == 100


def test_band_pointing_at_target_is_suspended_without_deleting_it(setup_drive, isolated_field):
    model, probes, tuner, _controller = setup_drive
    isolated_field['bands']['direct_target'] = {
        'node': 'probe:target', 'mode': 'pull', 'rest': .1, 'give': .2,
        'strength': shell.parse_field_source('100'), 'when': [],
    }
    output, _count, _description = field_output(model, probes, tuner)
    assert output[0, 0, 0] == 0
    assert output[0, 0, 1] > 0
    assert 'direct_target' in isolated_field['bands']


@pytest.mark.parametrize('pin', [
    {'probe': 'target', 'sign': 1}, {'mix': ['target']}, {'landscape': 'target'},
])
def test_direct_priority_pin_mix_and_landscape_cannot_reintroduce_target(setup_drive, pin):
    model, probes, tuner, _controller = setup_drive
    direction, _label, _sign = shell.resolve_priority_direction(model, probes, tuner, pin)
    assert direction is None


def test_off_restores_original_target_eligibility_without_modifying_mass(setup_drive):
    model, probes, tuner, controller = setup_drive
    handle_indirect_command(['off', 'target'], controller, probes, tuner)
    output, count, _description = field_output(model, probes, tuner)
    assert count == 2
    assert output[0, 0, 0] > 0
    assert output[0, 0, 1] == 0
    direction, label, _sign = shell.resolve_priority_direction(
        model, probes, tuner, {'probe': 'target', 'sign': 1})
    assert label == 'target' and direction is probes['target']['direction']
    assert probes['target']['mass'] == 10 and not probes['target']['released']


def test_explicit_perception_helper_remains_adjustable(setup_drive):
    model, probes, tuner, controller = setup_drive
    controller.set_role('helper', 'perception')
    controller.configure('target', -1, ['+helper'], purpose='coherence')
    output, count, _description = field_output(model, probes, tuner)
    assert count == 2 and output[0, 0, 1] > 0
    assert controller.plan(probes)[0]['role'] == 'perception'


def test_zero_global_g_leaves_indirect_plan_inert(setup_drive):
    model, probes, tuner, controller = setup_drive
    tuner.set('prioritize_alpha', 0)
    output, count, _description = field_output(model, probes, tuner)
    assert count == 0
    torch.testing.assert_close(output, torch.tensor([[[0., 0., 1., 0.]]]))
    controller.observe(sample(99, .2, -.2), probes)
    assert controller.feedback['target']['application_confirmed'] is False


def test_field_hook_execution_confirms_helpers_once_in_next_feedback(setup_drive):
    model, probes, tuner, controller = setup_drive
    handles, _description = shell.build_gravity_field_handles(model, probes, tuner)
    try:
        assert controller.pending
        assert not controller.pending[0].get('applied_helpers')
        output = torch.tensor([[[0., 0., 1., 0.]]])
        for layer in model.model.model.layers:
            output = layer(output)
        assert output[0, 0, 1] > 0
    finally:
        for handle in handles:
            handle.remove()
    controller.observe(sample(99, .2, -.2), probes)
    feedback = controller.feedback['target']
    assert feedback['application_confirmed'] is True
    assert feedback['applied_n'] == 1
    assert len(feedback['applied_helpers']) == 1
    assert feedback['applied_helpers'][0]['name'] == 'helper'


def test_zero_cap_never_confirms_application_for_an_unchanged_state(setup_drive):
    model, probes, tuner, controller = setup_drive
    output, count, _description = field_output(model, probes, tuner, cap_fraction=0)
    assert count == 2  # Hooks ran, but the envelope suppressed their entire push.
    torch.testing.assert_close(output, torch.tensor([[[0., 0., 1., 0.]]]))
    controller.observe(sample(99, .2, -.2), probes)
    assert controller.feedback['target']['application_confirmed'] is False


def test_layer_exclusion_limits_indirect_helper_injection(setup_drive, isolated_field):
    model, probes, tuner, _controller = setup_drive
    isolated_field['fields']['layer:1'] = {'excluded': True}
    output, count, _description = field_output(model, probes, tuner)
    assert count == 1
    assert output[0, 0, 1] > 0 and output[0, 0, 0] == 0


def test_runtime_view_does_not_hide_target_from_live_measurement_capture(setup_drive):
    _model, probes, tuner, _controller = setup_drive
    view = shell.steering_probe_view(probes, tuner)
    assert view['target']['released'] and not probes['target']['released']
    telemetry = {
        'generation_probe_readings': {
            'target': {'raw_score': .4, 'by_token': [.1, .2, .3, .4], 'token_count': 4},
            'helper': {'raw_score': -.4, 'by_token': [-.1, -.2, -.3, -.4], 'token_count': 4},
        },
        'generation_probe_capture': {'source': GENERATION_PROBE_SOURCE,
                                      'generation_id': 'measured-after-plan', 'token_count': 4},
    }
    turn = {}
    shell.score_generation_probes(probes, tuner, telemetry, turn_row=turn, display=False)
    assert turn['probe_target'] == pytest.approx(.4)
    assert turn['probe_helper'] == pytest.approx(-.4)
    assert turn['probe_generation_readings']['target']['by_token'] == [.1, .2, .3, .4]


def test_owner_observation_receives_original_target_and_stays_isolated(tmp_path):
    owners = PlasticityOwners()
    main = TriggerTuner(tmp_path / 'main.json')
    other = TriggerTuner(tmp_path / 'other.json')
    main_controller = owners.for_tuner(main)[0].indirect_controller
    other_controller = owners.for_tuner(other)[0].indirect_controller
    probes = {'target': probe('target', [1, 0, 0, 0]),
              'helper': probe('helper', [0, 1, 0, 0])}
    main_controller.configure('target', -1, ['helper'])
    for index in range(16):
        value = ((index * 7) % 17 - 8) / 10
        turn = sample(index, value, -value)
        original = deepcopy(turn)
        owners.observe(main, turn, None, probes=probes)
        assert {key: turn[key] for key in original} == original
        assert turn['probe_definition_fingerprints']
    assert main.indirect_controller is main_controller
    assert main_controller.plan(probes)
    assert not other_controller.observations and not other_controller.routes
    assert main_controller.path != other_controller.path


def add_second_objective(setup_drive, *, direction=-1, noisy=False):
    model, probes, tuner, controller = setup_drive
    probes['second'] = probe('second', [0, 0, 0, 1], mass=10)
    for index in range(16, 40):
        value = ((index * 7) % 17 - 8) / 10
        second = value + ((index % 3 - 1) * .4 if noisy else 0)
        turn = sample(index, value, -value)
        turn['probe_second'] = second
        controller.observe(turn, probes)
    controller.configure_many({'target': -1, 'second': direction}, ['helper'])
    return model, probes, tuner, controller


def test_batch_shared_helper_pushes_once_with_all_target_masses_suspended(setup_drive):
    model, probes, tuner, controller = add_second_objective(setup_drive)
    output, count, _description = field_output(model, probes, tuner)
    assert count == 2 and output[0, 0, 1] > 0
    assert output[0, 0, 0] == output[0, 0, 3] == 0
    summary = controller.status()['joint_plan']
    assert len(summary['helpers']) == 1
    assert summary['helpers'][0]['coefficient'] == pytest.approx(.15)
    assert probes['target']['mass'] == probes['second']['mass'] == 10
    turn = sample(99, .1, -.1)
    turn['probe_second'] = .1
    controller.observe(turn, probes)
    assert all(result['application_confirmed'] for result in controller.feedback.values())
    assert set(controller.feedback) == {'target', 'second'}


def test_conflicting_batch_helper_actions_cancel_without_false_application_credit(setup_drive):
    model, probes, tuner, controller = add_second_objective(setup_drive, direction=1)
    output, count, _description = field_output(model, probes, tuner)
    assert count == 0
    torch.testing.assert_close(output, torch.tensor([[[0., 0., 1., 0.]]]))
    assert controller.status()['joint_plan']['conflicts'] == ['helper']
    turn = sample(99, .1, -.1)
    turn['probe_second'] = .1
    controller.observe(turn, probes)
    assert set(controller.feedback) == {'target', 'second'}
    assert not any(value['application_confirmed'] for value in controller.feedback.values())


def test_partial_batch_conflict_only_confirms_target_contributions_matching_net_push(
        setup_drive, isolated_field):
    model, probes, tuner, controller = add_second_objective(setup_drive, direction=1)
    probes['auxiliary'] = probe('auxiliary', [0, 0, 1, 0])
    for index in range(40, 56):
        value = ((index * 7) % 17 - 8) / 10
        turn = sample(index, value, -value)
        turn.update(probe_second=value, probe_auxiliary=value)
        controller.observe(turn, probes)
    # The first target divides its share over two helpers; the second puts its
    # whole share on the common helper. Only that helper is eligible to fire.
    controller.configure('target', -1, ['helper', 'auxiliary'])
    isolated_field['fields']['probe:auxiliary'] = {'excluded': True}
    output, count, _description = field_output(model, probes, tuner)
    assert count == 2 and output[0, 0, 1] < 0
    assert controller.status()['joint_plan']['conflicts'] == ['helper']
    turn = sample(99, .1, -.1)
    turn['probe_second'] = .3
    controller.observe(turn, probes)
    assert controller.feedback['target']['application_confirmed'] is False
    assert controller.feedback['second']['application_confirmed'] is True
