"""Controlled rerun effects remain separate from live probe observations."""

from copy import deepcopy
import json

import pytest
import torch

from invariants.indirect_steering import IndirectSteering, definition_fingerprints, summarize_plan


LIVE = 'live_assistant_prediction'
REPLAY = 'controlled_replay_investigation'
METRIC = 'final_layer_mean_token_score'
MODEL = {'model': 'fixture-model', 'layers': 2, 'width': 4,
         'quantized_4bit': False, 'dtype': 'torch.float32'}


def probe(axis, label):
    return {'direction': {layer: torch.tensor(axis, dtype=torch.float32)
                          for layer in (0, 1)},
            'framings': (f'I express {label}.', f'I do not express {label}.'),
            'measurement_source': LIVE}


def bank():
    return {'target': probe([1, 0, 0, 0], 'the target'),
            'helper': probe([0, 1, 0, 0], 'the first helper'),
            'second': probe([0, 0, 1, 0], 'the second helper'),
            'gate': probe([0, 0, 0, 1], 'the gate')}


def live_row(index, **values):
    return {'probe_model_identity': deepcopy(MODEL),
            'probe_measurement': {'source': LIVE, 'generation_id': f'live-{index}',
                                  'token_count': 4},
            **{'probe_' + name: value for name, value in values.items()}}


def train(controller, probes):
    # The natural correlation recommends +helper for target minimization.
    for index in range(16):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(live_row(index, target=value, helper=-value,
                                    second=-value, gate=.5), probes)


def effect(*, target='target', helper='helper', sign=-1, delta=-.03, count=3):
    return {'target': target, 'helper': helper, 'sign': sign,
            'n_cases': count, 'n': count,
            'mean_delta': delta, 'standard_error': 0.,
            'case_deltas': [{'case_id': f'case-{index}', 'mean_delta': delta,
                             'n_repeats': 1} for index in range(count)],
            'metric': METRIC, 'readout_layer': MODEL['layers'] - 1}


def summary(probes, *, count=3, effects=None, fingerprint='run-fixture'):
    effects = effects if effects is not None else [
        effect(sign=1, delta=.02, count=count),
        effect(sign=-1, delta=-.03, count=count),
    ]
    return {'source': REPLAY, 'fingerprint': fingerprint,
            'model_identity': deepcopy(MODEL),
            'definition_fingerprints': definition_fingerprints(probes),
            'case_ids': [f'case-{index}' for index in range(count)],
            'completed_trials': count * 3, 'expected_trials': count * 3,
            'settings': {'alpha': .2, 'cap_fraction': .01, 'layers': [0],
                         'objectives': {'target': -1},
                         'controls': [{'name': helper, 'sign': None} for helper in
                                      dict.fromkeys(item['helper'] for item in effects)],
                         'max_new_tokens': 48, 'repeats': 1},
            'effects': effects, 'negative_controls': []}


def configured(probes, path=None, controls=None):
    controller = IndirectSteering(path)
    controller._refresh(probes)
    controller.configure('target', -1, controls or ['helper'])
    return controller


def state(controller):
    return deepcopy({'observations': controller.observations,
                     'replay_evidence': controller.replay_evidence,
                     'model_identity': controller.model_identity,
                     'feedback': controller.feedback, 'pending': controller.pending,
                     'routes': controller.routes})


def test_import_keeps_experimental_rows_out_of_natural_observations(tmp_path):
    probes = bank()
    controller = configured(probes, tmp_path / 'owner.indirect.json')
    train(controller, probes)
    before = deepcopy(controller.observations)
    report = controller.import_investigation(summary(probes), probes, model_identity=MODEL)
    assert isinstance(report, dict)
    assert controller.replay_evidence
    assert controller.observations == before
    assert all(item['source'] == LIVE for item in controller.observations)
    payload = json.loads(controller.path.read_text(encoding='utf-8'))
    assert payload['replay_evidence']
    assert payload['observations'] == before


def test_default_replay_minimum_and_effect_threshold_are_explicit_knobs():
    controller = IndirectSteering()
    assert controller.settings['min_replay_cases'] == 3
    assert controller.settings['min_replay_effect'] == pytest.approx(.0001)


def test_three_case_experiment_reverses_confounded_live_correlation():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    assert controller.plan(probes)[0]['coefficient'] > 0
    controller.import_investigation(summary(probes), probes)
    action = controller.plan(probes)[0]
    assert action['name'] == 'helper' and action['coefficient'] < 0
    assert action['basis'] == REPLAY
    assert action['mean_delta'] == pytest.approx(-.03)
    assert action['n_cases'] == 3


def test_two_cases_do_not_override_live_association():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    controller.import_investigation(summary(probes, count=2), probes)
    action = controller.plan(probes)[0]
    assert action['coefficient'] > 0
    assert action.get('basis') != REPLAY


@pytest.mark.parametrize('deltas', [(.01, .02), (-.00005, -.00009), (0., 0.)])
def test_adequately_tested_unhelpful_helper_cannot_fall_back_to_correlation(deltas):
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    data = summary(probes, effects=[effect(sign=1, delta=deltas[0]),
                                    effect(sign=-1, delta=deltas[1])])
    controller.import_investigation(data, probes)
    assert controller.plan(probes) == []


@pytest.mark.parametrize('permission,expected', [('+helper', None), ('-helper', -1)])
def test_explicit_helper_sign_is_not_reversed_by_experimental_preference(permission, expected):
    probes = bank()
    controller = configured(probes, controls=[permission])
    train(controller, probes)
    controller.import_investigation(summary(probes), probes)
    plan = controller.plan(probes)
    if expected is None:
        assert plan == []
    else:
        assert plan and plan[0]['coefficient'] < 0


def test_experimental_choice_tracks_maximization_direction():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    controller.import_investigation(summary(probes), probes)
    controller.configure('target', 1, ['helper'])
    action = controller.plan(probes)[0]
    assert action['coefficient'] > 0
    assert action['mean_delta'] > 0
    assert action['basis'] == REPLAY


def test_import_without_live_reading_does_not_manufacture_freshness():
    probes = bank()
    controller = configured(probes)
    controller.import_investigation(summary(probes), probes, model_identity=MODEL)
    assert controller.model_identity == MODEL
    assert controller.observations == []
    assert controller.plan(probes) == []
    controller.observe(live_row(99, target=.2), probes)
    assert controller.plan(probes) == []
    controller.observe(live_row(100, target=.2, helper=-.2), probes)
    assert controller.plan(probes)[0]['coefficient'] < 0


def test_live_condition_still_gates_an_experimentally_supported_helper():
    probes = bank()
    controller = configured(probes)
    controller.configure('target', -1, ['helper'], when=['gate', 'above', .4])
    controller.import_investigation(summary(probes), probes, model_identity=MODEL)
    controller.observe(live_row(99, target=.2, helper=-.2, gate=.3), probes)
    assert controller.plan(probes) == []
    controller.observe(live_row(100, target=.2, helper=-.2, gate=.5), probes)
    assert controller.plan(probes)[0]['coefficient'] < 0
    controller.observe(live_row(101, target=.2, helper=-.2), probes)
    assert controller.plan(probes) == []


def test_valid_json_file_import_and_identical_reimport_are_idempotent(tmp_path):
    probes = bank()
    controller = configured(probes)
    data = summary(probes)
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(data), encoding='utf-8')
    controller.import_investigation(path, probes, model_identity=MODEL)
    once = deepcopy(controller.replay_evidence)
    controller.import_investigation(path, probes, model_identity=MODEL)
    assert controller.replay_evidence == once


def test_repeats_and_reimport_do_not_supply_independent_cases():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    data = summary(probes, count=2)
    for item in data['effects']:
        item['n'] = 200
        for case in item['case_deltas']:
            case['n_repeats'] = 100
    data['settings']['repeats'] = 100
    data['completed_trials'] = data['expected_trials'] = 600
    controller.import_investigation(data, probes)
    controller.import_investigation(data, probes)
    assert controller.plan(probes)[0]['coefficient'] > 0


def mutate_scope(data, kind):
    if kind == 'source':
        data['source'] = 'reply_text_replay'
    elif kind == 'incomplete':
        data['completed_trials'] -= 1
    elif kind == 'empty_fingerprint':
        data['fingerprint'] = ''
    elif kind == 'model':
        data['model_identity']['model'] = 'different-model'
    elif kind == 'target_definition':
        data['definition_fingerprints']['target'] = 'old-target'
    elif kind == 'helper_definition':
        data['definition_fingerprints']['helper'] = 'old-helper'
    elif kind == 'metric':
        data['effects'][0]['metric'] = 'mean_token_score'
    elif kind == 'readout_layer':
        data['effects'][0]['readout_layer'] = 0
    elif kind == 'count':
        data['effects'][0]['n_cases'] = 4
    elif kind == 'duplicate_case':
        data['effects'][0]['case_deltas'][1]['case_id'] = 'case-0'
    elif kind == 'foreign_case':
        data['effects'][0]['case_deltas'][1]['case_id'] = 'not-in-this-investigation'
    elif kind == 'nonfinite_mean':
        data['effects'][0]['mean_delta'] = float('nan')
    elif kind == 'negative_stderr':
        data['effects'][0]['standard_error'] = -.1
    elif kind == 'nonfinite_stderr':
        data['effects'][0]['standard_error'] = float('inf')
    elif kind == 'case_mean_disagrees':
        data['effects'][0]['mean_delta'] = .9
    elif kind == 'invalid_sign':
        data['effects'][0]['sign'] = 0
    elif kind == 'zero_dose':
        data['settings']['alpha'] = 0
    elif kind == 'no_intervention_band':
        data['settings']['layers'] = []
    elif kind == 'readout_intervention':
        data['settings']['layers'] = [MODEL['layers'] - 1]
    elif kind == 'duplicate_source_case':
        data['case_ids'][1] = data['case_ids'][0]
    elif kind == 'non_string_target':
        data['effects'][0]['target'] = ['target']
    elif kind == 'duplicate_effect':
        data['effects'].append(deepcopy(data['effects'][0]))
    elif kind == 'unrepresentable_effect_per_alpha':
        data['settings']['alpha'] = 1e-323
    else:
        raise AssertionError(kind)


@pytest.mark.parametrize('kind', [
    'source', 'incomplete', 'empty_fingerprint', 'model',
    'target_definition', 'helper_definition', 'metric', 'readout_layer',
    'count', 'duplicate_case', 'foreign_case', 'nonfinite_mean',
    'negative_stderr', 'nonfinite_stderr', 'case_mean_disagrees',
    'invalid_sign', 'zero_dose', 'no_intervention_band', 'readout_intervention',
    'duplicate_source_case', 'non_string_target', 'duplicate_effect',
    'unrepresentable_effect_per_alpha',
])
def test_invalid_import_is_atomic_and_cannot_pollute_live_state(tmp_path, kind):
    probes = bank()
    controller = configured(probes, tmp_path / 'owner.indirect.json')
    train(controller, probes)
    controller.import_investigation(summary(probes), probes)
    before = state(controller)
    saved = controller.path.read_bytes()
    data = summary(probes, fingerprint='invalid-run')
    mutate_scope(data, kind)
    with pytest.raises(ValueError):
        controller.import_investigation(data, probes, model_identity=MODEL)
    assert state(controller) == before
    assert controller.path.read_bytes() == saved


def test_invalid_later_effect_does_not_partially_import_the_valid_first_effect():
    probes = bank()
    controller = configured(probes)
    controller.model_identity = deepcopy(MODEL)
    data = summary(probes)
    data['effects'][1]['helper'] = 'missing-helper'
    with pytest.raises(ValueError):
        controller.import_investigation(data, probes)
    assert controller.replay_evidence == []
    assert controller.observations == []


def test_explicit_current_model_identity_cannot_disagree_with_bound_controller():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    other = dict(MODEL, model='different-model')
    data = summary(probes)
    data['model_identity'] = other
    with pytest.raises(ValueError):
        controller.import_investigation(data, probes, model_identity=other)
    assert controller.replay_evidence == []


def test_import_rolls_back_in_memory_state_if_persistence_fails(tmp_path, monkeypatch):
    probes = bank()
    controller = configured(probes, tmp_path / 'owner.indirect.json')
    train(controller, probes)
    before = state(controller)
    saved = controller.path.read_bytes()

    def failed_save():
        raise OSError('fixture persistence failure')

    monkeypatch.setattr(controller, 'save', failed_save)
    with pytest.raises(OSError, match='fixture persistence'):
        controller.import_investigation(summary(probes), probes)
    assert state(controller) == before
    assert controller.path.read_bytes() == saved


@pytest.mark.parametrize('helper_kind', ['target', 'alias', 'inverse', 'derived'])
def test_import_cannot_add_target_or_proxy_as_a_helper(helper_kind):
    probes = bank()
    if helper_kind != 'target':
        probes['proxy'] = deepcopy(probes['target'])
        if helper_kind == 'inverse':
            probes['proxy']['direction'] = {layer: -vector
                                           for layer, vector in probes['proxy']['direction'].items()}
            probes['proxy']['framings'] = ('An inverse.', 'Its opposite.')
        elif helper_kind == 'derived':
            probes['proxy'] = probe([1, 1, 0, 0], 'a derived target')
            probes['proxy']['framings'] = ('composed: target + second', '')
    controller = configured(probes)
    helper = 'target' if helper_kind == 'target' else 'proxy'
    data = summary(probes, effects=[effect(helper=helper)])
    with pytest.raises(ValueError):
        controller.import_investigation(data, probes, model_identity=MODEL)
    assert controller.replay_evidence == []


def test_experimental_effects_survive_restart_but_require_a_fresh_live_pair(tmp_path):
    probes = bank()
    path = tmp_path / 'owner.indirect.json'
    controller = configured(probes, path)
    train(controller, probes)
    controller.import_investigation(summary(probes), probes)
    restored = IndirectSteering(path)
    assert restored.replay_evidence
    assert restored.model_identity == MODEL
    assert restored.plan(probes) == []
    restored.observe(live_row(99, target=.2, helper=-.2), probes)
    action = restored.plan(probes)[0]
    assert action['coefficient'] < 0 and action['basis'] == REPLAY


@pytest.mark.parametrize('changed', ['target', 'helper'])
def test_remint_invalidates_experimental_effect_for_changed_instrument(changed):
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    controller.import_investigation(summary(probes), probes)
    probes[changed]['direction'][0].add_(.01)
    assert controller.plan(probes) == []
    assert not controller.replay_evidence


def test_a_different_model_observation_invalidates_saved_replay_effects():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    controller.import_investigation(summary(probes), probes)
    sample = live_row(99, target=.2, helper=-.2)
    sample['probe_model_identity'] = dict(MODEL, model='replacement-model')
    controller.observe(sample, probes)
    assert not controller.replay_evidence
    assert controller.plan(probes) == []


def test_joint_experimental_helpers_keep_the_existing_global_budget():
    probes = bank()
    controller = configured(probes, controls=['helper', 'second'])
    train(controller, probes)
    data = summary(probes, effects=[effect(helper='helper', delta=-.03),
                                    effect(helper='second', delta=-.01)])
    controller.import_investigation(data, probes)
    plan = controller.plan(probes)
    assert {action['name'] for action in plan} == {'helper', 'second'}
    assert all(action['coefficient'] < 0 and action['basis'] == REPLAY for action in plan)
    assert sum(abs(action['coefficient']) for action in plan) <= controller.settings['budget'] + 1e-9
    by_name = {action['name']: action for action in plan}
    assert abs(by_name['helper']['coefficient']) > abs(by_name['second']['coefficient'])


def test_plan_carries_tested_scope_and_marks_additive_to_field_transfer_provisional():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    data = summary(probes)
    controller.import_investigation(data, probes)
    action = controller.plan(probes)[0]
    assert action['layers'] == data['settings']['layers']
    assert action['tested_alpha'] == data['settings']['alpha']
    assert action['tested_cap_fraction'] == data['settings']['cap_fraction']
    assert action['operator_transfer'] == 'provisional_additive_to_field'
    assert 'not validated' in action['transfer_assumption']


def test_disjoint_tested_bands_do_not_cancel_opposite_shared_helper_requests():
    entries = [{'name': 'helper', 'target': 'target_a', 'coefficient': .075, 'layers': [0]},
               {'name': 'helper', 'target': 'target_b', 'coefficient': -.075, 'layers': [1]}]
    result = summarize_plan(entries, budget=.15)
    assert result['conflicts'] == []
    assert result['gross_budget_used'] == pytest.approx(.15)
    assert result['net_budget_used'] == pytest.approx(.15)
    assert result['helpers'] == [
        {'name': 'helper', 'coefficient': .075, 'layers': [0],
         'contributions': [{'target': 'target_a', 'coefficient': .075}]},
        {'name': 'helper', 'coefficient': -.075, 'layers': [1],
         'contributions': [{'target': 'target_b', 'coefficient': -.075}]},
    ]


def test_equivalent_tested_bands_share_one_net_action_despite_layer_order():
    entries = [{'name': 'helper', 'target': 'target_a', 'coefficient': .075, 'layers': [1, 0]},
               {'name': 'helper', 'target': 'target_b', 'coefficient': -.075, 'layers': [0, 1]}]
    result = summarize_plan(entries, budget=.15)
    assert result['conflicts'] == ['helper']
    assert result['net_budget_used'] == 0
    assert len(result['helpers']) == 1
    assert result['helpers'][0]['layers'] == [0, 1]


def test_unscoped_live_route_is_distinct_from_an_experimental_band():
    entries = [{'name': 'helper', 'target': 'target_a', 'coefficient': .075},
               {'name': 'helper', 'target': 'target_b', 'coefficient': -.075, 'layers': [0]}]
    result = summarize_plan(entries, budget=.15)
    assert result['conflicts'] == []
    assert len(result['helpers']) == 2
    assert 'layers' not in result['helpers'][0]
    assert result['helpers'][1]['layers'] == [0]


def inverted_remint(probes):
    """Same named contrast with reversed poles, despite numerical remint drift."""
    probes['helper']['framings'] = tuple(reversed(probes['target']['framings']))
    probes['helper']['direction'] = {
        layer: torch.tensor([1., .1, 0., 0.]) for layer in (0, 1)}
    cosine = torch.nn.functional.cosine_similarity(
        probes['target']['direction'][0], probes['helper']['direction'][0], dim=0)
    assert abs(float(cosine)) < .9999


def test_swapped_contrastive_framings_block_configuration_despite_remint_drift():
    probes = bank()
    inverted_remint(probes)
    controller = IndirectSteering()
    controller._refresh(probes)
    with pytest.raises(ValueError, match='alias/inverse'):
        controller.configure('target', -1, ['helper'])


def test_swapped_contrastive_framings_block_an_existing_helper_proposal_after_remint():
    probes = bank()
    controller = configured(probes)
    train(controller, probes)
    inverted_remint(probes)
    proposal = controller.propose('target', -1, probes, ['helper'])
    assert proposal['helpers'] == []
    assert 'alias/inverse' in proposal['rejected']['helper']


def test_swapped_contrastive_framings_cannot_import_experimental_target_proxy():
    probes = bank()
    inverted_remint(probes)
    controller = IndirectSteering()
    data = summary(probes)
    with pytest.raises(ValueError, match='direct proxy'):
        controller.import_investigation(data, probes, model_identity=MODEL)
    assert controller.replay_evidence == [] and controller.observations == []
