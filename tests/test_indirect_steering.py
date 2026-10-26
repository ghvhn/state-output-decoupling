"""Indirect control uses separate helper directions and fresh, scoped evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from invariants.generation_probe_capture import GENERATION_PROBE_SOURCE
from invariants.indirect_steering import IndirectSteering, handle_indirect_command


def probe(axis, *, frame=None, **extra):
    return {
        'direction': {0: torch.tensor(axis, dtype=torch.float32)},
        'framings': (frame or f'I express property {list(axis)}.',
                     f'I do not express property {list(axis)}.'),
        'measurement_source': GENERATION_PROBE_SOURCE,
        **extra,
    }


def bank():
    return {
        'target': probe([1, 0, 0, 0, 0, 0]),
        'negative': probe([0, 1, 0, 0, 0, 0]),
        'positive': probe([0, 0, 1, 0, 0, 0]),
        'other': probe([0, 0, 0, 1, 0, 0]),
    }


def row(index, *, source=GENERATION_PROBE_SOURCE, **values):
    return {
        'probe_measurement': {
            'source': source,
            'generation_id': f'generation-{index}',
            'token_count': 6,
        },
        **{'probe_' + name: value for name, value in values.items()},
    }


def authorize_helpers(controller, names=('negative', 'positive', 'other')):
    for name in names:
        controller.set_role(name, 'thinking')


def train(controller, probes, *, start=0, count=16, source=GENERATION_PROBE_SOURCE,
          assign_roles=True):
    if assign_roles:
        authorize_helpers(controller)
    for index in range(start, start + count):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(row(index, source=source, target=value,
                               negative=-value, positive=value,
                               other=(index % 3 - 1) / 10), probes)


def coefficients(plan):
    return {entry['name']: entry['coefficient'] for entry in plan}


def test_alias_guard_refreshes_before_planning_after_in_place_remint():
    probes = bank()
    controller = IndirectSteering()
    train(controller, probes)
    controller.configure('target', -1, ['negative'])
    assert 'other' not in controller.blocked_probes(probes)
    probes['other']['direction'][0].copy_(probes['target']['direction'][0])
    assert 'other' in controller.blocked_probes(probes)


def test_aliases_of_target_derived_composites_are_also_excluded():
    probes = bank()
    probes['derived'] = probe([1, 1, 0, 0, 0, 0], frame='composed: target + negative')
    probes['copy'] = probe([1, 1, 0, 0, 0, 0], frame='A copied derived direction.')
    controller = IndirectSteering()
    train(controller, probes)
    controller.configure('target', -1, ['negative'])
    assert {'target', 'derived', 'copy'} <= controller.blocked_probes(probes)


def test_other_referent_framing_is_observational_without_cached_flag():
    probes = bank()
    probes['negative']['framings'] = ('I understand @.', 'I do not understand @.')
    controller = IndirectSteering()
    train(controller, probes)
    controller.configure('target', -1, ['negative'])
    assert controller.plan(probes) == []


def test_minimization_raises_negative_correlate_and_lowers_positive_correlate():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative', 'positive'])
    train(controller, probes)

    plan = controller.plan(probes)
    values = coefficients(plan)
    assert values['negative'] > 0
    assert values['positive'] < 0
    assert all(entry['target'] == 'target' and entry['n'] >= 8 for entry in plan)
    assert {entry['name']: entry['correlation'] for entry in plan} == pytest.approx(
        {'negative': -1, 'positive': 1})
    assert sum(abs(value) for value in values.values()) <= .15 + 1e-9


def test_maximization_reverses_helper_polarities():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', 1, ['negative', 'positive'])
    train(controller, probes)
    values = coefficients(controller.plan(probes))
    assert values['negative'] < 0 < values['positive']


def test_explicit_positive_permission_filters_a_helper_that_needs_negative_steering():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['+negative', '+positive'])
    train(controller, probes)
    values = coefficients(controller.plan(probes))
    assert values['negative'] > 0
    assert 'positive' not in values


def test_explicit_negative_permission_does_not_get_silently_reversed():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['-negative', '-positive'])
    train(controller, probes)
    values = coefficients(controller.plan(probes))
    assert values['positive'] < 0
    assert 'negative' not in values


def test_literal_target_cannot_be_its_own_helper():
    controller = IndirectSteering()
    with pytest.raises(ValueError):
        controller.configure('target', -1, ['target'])
    with pytest.raises(ValueError):
        controller.configure('target', -1, ['+target'])


def test_alias_inverse_and_recursive_target_derivatives_are_blocked():
    probes = bank()
    probes.update({
        'alias': probe([1, 0, 0, 0, 0, 0]),
        'inverse': probe([-1, 0, 0, 0, 0, 0]),
        'derived': probe([0, 0, 0, 0, 1, 0], frame='composed: target + other'),
        'recursive': probe([0, 0, 0, 0, 0, 1], frame='composed: derived - positive'),
    })
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative', 'alias', 'inverse', 'derived', 'recursive'])
    train(controller, probes)
    blocked = set(controller.blocked_probes(probes))
    assert {'target', 'alias', 'inverse', 'derived', 'recursive'} <= blocked
    assert 'negative' not in blocked
    assert set(coefficients(controller.plan(probes))) == {'negative'}


def test_same_contrastive_definition_is_blocked_despite_remint_direction_drift():
    probes = bank()
    probes['reminted_target'] = deepcopy(probes['target'])
    probes['reminted_target']['direction'] = {0: torch.tensor([1., .1, 0., 0., 0., 0.])}
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative', 'reminted_target'])
    train(controller, probes)
    assert 'reminted_target' in controller.blocked_probes(probes)
    assert set(coefficients(controller.plan(probes))) == {'negative'}


def test_identical_helper_directions_do_not_receive_double_weight():
    probes = bank()
    probes['duplicate'] = deepcopy(probes['negative'])
    controller = IndirectSteering()
    authorize_helpers(controller, ['negative', 'duplicate'])
    controller.configure('target', -1, ['negative', 'duplicate'])
    for index in range(16):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(row(index, target=value, negative=-value, duplicate=-value), probes)
    plan = controller.plan(probes)
    assert len(plan) == 1
    assert plan[0]['name'] in {'negative', 'duplicate'}
    assert plan[0]['coefficient'] > 0


def test_replay_and_backfill_rows_cannot_supply_live_control_evidence():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes, source='reply_text_replay')
    for index in range(20, 36):
        sample = row(index, target=index / 100, negative=-index / 100)
        sample['backfill'] = True
        controller.observe(sample, probes)
    assert controller.plan(probes) == []


def test_backfill_basis_cannot_supply_live_evidence_even_with_a_live_source_label():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    for index in range(16):
        sample = row(index, target=index / 10, negative=-index / 10)
        sample['basis'] = 'backfill'
        assert controller.observe(sample, probes) is False
    assert controller.plan(probes) == []


def test_one_generation_cannot_be_recounted_to_clear_sample_requirement():
    probes = bank()
    controller = IndirectSteering()
    authorize_helpers(controller)
    controller.configure('target', -1, ['negative'])
    for index in range(16):
        sample = row('same', target=index / 10, negative=-index / 10)
        controller.observe(sample, probes)
    assert controller.plan(probes) == []


def test_replayed_generation_cannot_rewrite_an_already_observed_pair():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    before = controller.propose('target', -1, probes, ['negative'])
    for index in range(16):
        controller.observe(row(index, target=100 + index, negative=100 + index), probes)
    after = controller.propose('target', -1, probes, ['negative'])
    assert after['helpers'] == before['helpers']


@pytest.mark.parametrize('missing', ['target', 'negative'])
def test_missing_current_measurement_never_reuses_a_stale_control_plan(missing):
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    sample = row(99, target=.1, negative=-.1)
    sample.pop('probe_' + missing)
    controller.observe(sample, probes)
    assert controller.plan(probes) == []


def test_empty_fresh_live_generation_cannot_reuse_last_good_plan():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    controller.observe(row(99), probes)
    assert controller.plan(probes) == []


@pytest.mark.parametrize('bad', [None, True, float('nan'), float('inf'), {'value': .5}])
def test_non_numeric_or_nonfinite_measurements_do_not_become_control_evidence(bad):
    probes = bank()
    controller = IndirectSteering()
    authorize_helpers(controller)
    controller.configure('target', -1, ['negative'])
    for index in range(16):
        controller.observe(row(index, target=index / 10, negative=bad), probes)
    assert controller.plan(probes) == []


def test_non_varying_helper_or_target_cannot_authorize_a_direction():
    probes = bank()
    controller = IndirectSteering()
    authorize_helpers(controller)
    controller.configure('target', -1, ['negative', 'positive'])
    for index in range(16):
        controller.observe(row(index, target=.2, negative=-index / 10, positive=.2), probes)
    assert controller.plan(probes) == []


def test_new_helper_definition_requires_new_evidence():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    probes['negative'] = probe([0, 0, 0, 0, 1, 0], frame='I express a different property.')
    assert controller.plan(probes) == []
    train(controller, probes, start=50, count=4)
    assert controller.plan(probes) == []
    train(controller, probes, start=54, count=12)
    assert coefficients(controller.plan(probes))['negative'] > 0


def test_new_target_definition_invalidates_old_correlations():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    probes['target'] = probe([0, 0, 0, 0, 1, 0], frame='I express a new target.')
    assert controller.plan(probes) == []


@pytest.mark.parametrize('geometry', ['position', 'scale'])
def test_changed_probe_geometry_requires_fresh_evidence(geometry):
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    probes['negative'][geometry] = {0: torch.zeros(6) if geometry == 'position' else 2.0}
    assert controller.plan(probes) == []


def test_in_place_direction_mutation_invalidates_old_evidence():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    probes['negative']['direction'][0][4] = .1
    assert controller.plan(probes) == []


def test_changed_model_identity_never_inherits_earlier_model_correlations():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    for index in range(16):
        sample = row(index, target=index / 10, negative=-index / 10)
        sample['probe_model_identity'] = {'model': 'first', 'd_model': 6}
        controller.observe(sample, probes)
    assert controller.plan(probes)
    sample = row(99, target=.2, negative=-.2)
    sample['probe_model_identity'] = {'model': 'second', 'd_model': 6}
    controller.observe(sample, probes)
    assert controller.plan(probes) == []
    assert not controller.feedback


def test_foreign_declared_definition_cannot_be_credited_as_current_evidence():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    for index in range(16):
        sample = row(index, target=index / 10, negative=-index / 10)
        sample['probe_definition_fingerprints'] = {'target': 'old', 'negative': 'old'}
        controller.observe(sample, probes)
    assert controller.plan(probes) == []


def test_source_change_or_released_helper_disables_its_plan():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    probes['negative']['measurement_source'] = 'reply_text_replay'
    assert controller.plan(probes) == []
    probes['negative']['measurement_source'] = GENERATION_PROBE_SOURCE
    probes['negative']['released'] = True
    assert controller.plan(probes) == []


def test_proposal_is_read_only_and_can_discover_helpers_without_a_route():
    probes = bank()
    controller = IndirectSteering()
    train(controller, probes)
    assert not controller.routes
    proposal = controller.propose('target', -1, probes)
    assert {'negative', 'positive'} <= {entry['name'] for entry in proposal['helpers']}
    assert not controller.routes
    assert controller.plan(probes) == []


def test_multiple_targets_share_one_total_budget():
    probes = bank()
    probes['second'] = probe([0, 0, 0, 0, 1, 0])
    controller = IndirectSteering()
    authorize_helpers(controller)
    controller.configure('target', -1, ['negative', 'positive'])
    controller.configure('second', 1, ['negative', 'positive'])
    for index in range(16):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(row(index, target=value, second=value,
                               negative=-value, positive=value), probes)
    controller.settings['budget'] = .04
    plan = controller.plan(probes)
    assert {entry['target'] for entry in plan} == {'target', 'second'}
    assert sum(abs(entry['coefficient']) for entry in plan) <= .04 + 1e-9


def test_history_limit_and_min_samples_are_effective_knobs():
    probes = bank()
    controller = IndirectSteering()
    controller.settings['history_limit'] = 5
    controller.configure('target', -1, ['negative'])
    train(controller, probes, count=24)
    assert controller.plan(probes) == []  # Default minimum is eight paired samples.
    controller.settings['min_samples'] = 4
    assert coefficients(controller.plan(probes))['negative'] > 0


def test_top_k_limits_distinct_helper_axes():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative', 'positive'])
    train(controller, probes)
    assert len(controller.plan(probes)) == 2
    controller.settings['top_k'] = 1
    assert len(controller.plan(probes)) == 1


def test_config_and_evidence_persist_without_crossing_owner_paths(tmp_path):
    probes = bank()
    first_path = tmp_path / 'main.indirect.json'
    first = IndirectSteering(first_path)
    first.configure('target', -1, ['negative'])
    train(first, probes)
    first.settings['budget'] = .07
    first.save()
    restored = IndirectSteering(first_path)
    separate = IndirectSteering(tmp_path / 'agent.indirect.json')
    assert restored.routes == first.routes
    assert restored.settings['budget'] == .07
    assert not separate.routes
    assert separate.plan(probes) == []
    # Restart may preserve evidence, but must wait for fresh current measurements.
    assert restored.plan(probes) == []
    restored.observe(row(99, target=.3, negative=-.3), probes)
    assert coefficients(restored.plan(probes))['negative'] > 0


def test_command_surface_supports_suggestion_activation_and_reversible_off():
    probes = bank()
    tuner = SimpleNamespace(triggers={f'probe_{name}': object() for name in probes})
    controller = IndirectSteering()
    train(controller, probes)
    proposal = handle_indirect_command(['suggest', 'target', 'minimize', '+negative'],
                                       controller, probes, tuner)
    assert proposal['helpers'] and not controller.routes
    handle_indirect_command(['indirect', 'target', 'minimize', '+negative'],
                            controller, probes, tuner)
    assert controller.plan(probes)
    handle_indirect_command(['off'], controller, probes, tuner)
    assert controller.plan(probes) == []
    assert not controller.blocked_probes(probes)
    handle_indirect_command(['on'], controller, probes, tuner)
    handle_indirect_command(['set', 'budget', '0.03'], controller, probes, tuner)
    assert sum(abs(entry['coefficient']) for entry in controller.plan(probes)) <= .03 + 1e-9
    handle_indirect_command(['drop', 'target'], controller, probes, tuner)
    assert not controller.routes


@pytest.mark.parametrize('knob,value', [
    ('budget', '-1'), ('budget', 'nan'), ('min_samples', '1'),
    ('min_correlation', '1.1'), ('top_k', '0'), ('history_limit', '0'),
])
def test_invalid_control_settings_do_not_change_prior_settings(knob, value):
    controller = IndirectSteering()
    before = deepcopy(controller.settings)
    with pytest.raises(ValueError):
        handle_indirect_command(['set', knob, value], controller, bank(), None)
    assert controller.settings == before


def test_unclassified_explicit_helper_is_observed_without_guessing_its_role():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes, assign_roles=False)
    before = deepcopy(controller.observations)
    plan = controller.plan(probes)
    assert coefficients(plan)['negative'] > 0
    assert plan[0]['role'] == 'unclassified'
    assert controller.observations == before


@pytest.mark.parametrize('role', ['thinking', 'execution'])
def test_thinking_and_execution_are_explicit_default_drive_helpers(role):
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes, assign_roles=False)
    controller.set_role('negative', role)
    assert coefficients(controller.plan(probes))['negative'] > 0


def test_perception_can_be_adjusted_with_explicit_coherence_purpose():
    probes = bank()
    controller = IndirectSteering()
    controller.set_role('target', 'perception')
    controller.set_role('negative', 'perception')
    train(controller, probes, assign_roles=False)
    controller.configure('target', -1, ['negative'])
    assert coefficients(controller.plan(probes))['negative'] > 0
    controller.configure('target', -1, ['negative'], purpose='coherence')
    plan = controller.plan(probes)
    assert coefficients(plan)['negative'] > 0
    assert plan[0]['role'] == 'perception' and plan[0]['purpose'] == 'coherence'
    assert 'negative' not in controller.blocked_probes(probes)


def test_role_change_cancels_pending_plan_without_rewriting_observations():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes) and controller.pending
    before = deepcopy(controller.observations)
    controller.set_role('negative', 'perception')
    assert not controller.pending
    assert controller.observations == before
    assert controller.plan(probes)[0]['role'] == 'perception'


def test_invalid_role_does_not_relabel_a_probe():
    controller = IndirectSteering()
    controller.set_role('negative', 'perception')
    before = deepcopy(controller.roles)
    with pytest.raises(ValueError):
        controller.set_role('negative', 'manipulation')
    assert controller.roles == before


def test_coherence_purpose_and_roles_are_owner_local_persisted_configuration(tmp_path):
    probes = bank()
    controller = IndirectSteering(tmp_path / 'owner.indirect.json')
    tuner = SimpleNamespace(triggers={f'probe_{name}': object() for name in probes})
    handle_indirect_command(['role', 'negative', 'perception'], controller, probes, tuner)
    handle_indirect_command(['indirect', 'target', 'minimize', '+negative', 'purpose', 'coherence'],
                            controller, probes, tuner)
    train(controller, probes, assign_roles=False)
    assert controller.plan(probes)
    controller.save()
    restored = IndirectSteering(controller.path)
    restored.observe(row(99, target=.3, negative=-.3), probes)
    assert restored.plan(probes)
    assert restored.routes['target']['purpose'] == 'coherence'
    assert IndirectSteering(tmp_path / 'different.indirect.json').plan(probes) == []


def test_feedback_consumes_one_fresh_generation_once_and_stays_observational():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    assert controller.plan(probes)
    assert controller.pending
    next_row = row(99, target=.1, negative=-.1)
    controller.observe(next_row, probes)
    assert not controller.pending
    first_feedback = deepcopy(controller.feedback['target'])
    assert first_feedback['n'] == 1
    assert first_feedback['source'] == GENERATION_PROBE_SOURCE
    assert first_feedback['last_desired_delta'] == pytest.approx(
        -first_feedback['last_target_delta'])
    controller.observe(next_row, probes)
    assert controller.feedback['target'] == first_feedback
    controller.observe(row(100, target=.2, negative=-.2), probes)
    assert controller.feedback['target']['n'] == 1


def test_feedback_distinguishes_proposal_from_confirmed_helper_application():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1, ['negative'])
    train(controller, probes)
    controller.plan(probes)
    controller.observe(row(99, target=.2, negative=-.2), probes)
    assert controller.feedback['target']['application_confirmed'] is False
    plan = controller.plan(probes)
    controller.mark_applied(plan)
    controller.observe(row(100, target=.1, negative=-.1), probes)
    result = controller.feedback['target']
    assert result['application_confirmed'] is True and result['applied_n'] == 1
    assert result['applied_helpers'] == plan


def test_optional_probe_gate_controls_when_the_view_adjustment_is_proposed():
    probes = bank()
    controller = IndirectSteering()
    train(controller, probes)
    controller.configure('target', -1, ['+negative'], purpose='coherence',
                         when={'probe': 'other', 'comparison': 'above', 'threshold': .1})
    assert controller.plan(probes) == []
    controller.observe(row(99, target=.2, negative=-.2, other=.3), probes)
    assert coefficients(controller.plan(probes))['negative'] > 0
    controller.observe(row(100, target=.2, negative=-.2), probes)
    assert controller.plan(probes) == []
    controller.observe(row(101, target=.2, negative=-.2, other=.1), probes)
    assert controller.plan(probes) == []  # 'above' is strict.


def test_command_accepts_explicit_coherence_condition_without_inventing_a_detector():
    probes = bank()
    controller = IndirectSteering()
    train(controller, probes)
    handle_indirect_command(
        ['indirect', 'target', 'minimize', '+negative', 'when', 'other', 'below', '0.4',
         'purpose', 'coherence'], controller, probes)
    assert controller.routes['target']['purpose'] == 'coherence'
    assert controller.plan(probes)


def batch_bank():
    probes = bank()
    probes['second'] = probe([0, 0, 0, 0, 1, 0])
    probes['third'] = probe([0, 0, 0, 0, 0, 1])
    return probes


def train_batch(controller, probes, count=16):
    for index in range(count):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(row(index, target=value, second=value, third=-value,
                               negative=-value, positive=value,
                               other=(index % 3 - 1) / 10), probes)


def test_batch_configures_several_minimized_and_maximized_targets_together():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    created = controller.configure_many(
        {'target': 'minimize', 'second': 'minimize', 'third': 'maximize'},
        ['+negative'])
    assert list(created) == ['target', 'second', 'third']
    assert {name: route['direction'] for name, route in created.items()} == {
        'target': -1, 'second': -1, 'third': 1}
    plan = controller.plan(probes)
    assert {entry['target'] for entry in plan} == {'target', 'second', 'third'}
    assert all(entry['coefficient'] > 0 for entry in plan)
    assert sum(abs(entry['coefficient']) for entry in plan) <= .15 + 1e-9
    assert {'target', 'second', 'third'} <= controller.blocked_probes(probes)


def test_batch_preserves_unrelated_routes_and_evidence_with_one_save(monkeypatch):
    probes = batch_bank()
    controller = IndirectSteering()
    controller.configure('other', 1, ['positive'])
    train_batch(controller, probes)
    earlier_route = deepcopy(controller.routes['other'])
    earlier_rows = deepcopy(controller.observations)
    saves = []
    monkeypatch.setattr(controller, 'save', lambda: saves.append(deepcopy(controller.routes)))
    created = controller.configure_many([('third', 1), ('target', -1)], ['negative'])
    assert list(created) == ['third', 'target']
    assert controller.routes['other'] == earlier_route
    assert controller.observations == earlier_rows
    assert len(saves) == 1
    assert {'other', 'third', 'target'} == set(saves[0])


@pytest.mark.parametrize('objectives,controls', [
    ([('target', -1), ('missing', 1)], ['negative']),
    ([('target', -1), ('second', 1)], ['missing']),
    ([('target', -1), ('second', 1)], ['second']),
    ([('target', -1), ('target', 1)], ['negative']),
    ([('target', -1), ('target', -1)], ['negative']),
])
def test_invalid_batch_leaves_all_routes_and_observations_untouched(
        monkeypatch, objectives, controls):
    probes = batch_bank()
    controller = IndirectSteering()
    controller.configure('third', 1, ['positive'])
    train_batch(controller, probes)
    before_routes = deepcopy(controller.routes)
    before_rows = deepcopy(controller.observations)
    saves = []
    monkeypatch.setattr(controller, 'save', lambda: saves.append(True))
    with pytest.raises(ValueError):
        controller.configure_many(objectives, controls)
    assert controller.routes == before_routes
    assert controller.observations == before_rows
    assert not saves


def test_existing_active_target_cannot_become_a_new_batch_helper():
    probes = batch_bank()
    controller = IndirectSteering()
    controller.configure('other', 1, ['positive'])
    train_batch(controller, probes)
    before = deepcopy(controller.routes)
    with pytest.raises(ValueError):
        controller.configure_many({'target': -1, 'second': 1}, ['other'])
    assert controller.routes == before


def test_alias_of_another_batch_target_cannot_be_used_as_a_helper():
    probes = batch_bank()
    probes['second_alias'] = deepcopy(probes['second'])
    controller = IndirectSteering()
    train_batch(controller, probes)
    with pytest.raises(ValueError):
        controller.configure_many({'target': -1, 'second': 1}, ['second_alias'])
    assert not controller.routes


def test_batch_suggestion_is_read_only_and_reports_opposing_shared_helper_actions():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    before_rows = deepcopy(controller.observations)
    proposal = controller.propose_many({'target': -1, 'second': 1}, probes,
                                        controls=['negative'])
    assert proposal['objectives'] == {'target': -1, 'second': 1}
    assert set(proposal['proposals']) == {'target', 'second'}
    assert {entry['target'] for entry in proposal['plan']} == {'target', 'second'}
    assert proposal['joint_plan']['conflicts'] == ['negative']
    assert proposal['joint_plan']['gross_budget_used'] == pytest.approx(.15)
    assert proposal['joint_plan']['net_budget_used'] == pytest.approx(0)
    assert proposal['joint_plan']['helpers'][0]['coefficient'] == pytest.approx(0)
    assert not controller.routes and not controller.pending
    assert controller.observations == before_rows


def test_batch_discovery_never_uses_another_requested_objective_as_its_helper():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    proposal = controller.propose_many({'target': -1, 'second': 1, 'third': 1}, probes)
    assert proposal['plan']
    assert not ({entry['name'] for entry in proposal['plan']} & {'target', 'second', 'third'})


def test_raw_batch_plan_retains_per_target_entries_and_joint_summary_exposes_cancellation():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    controller.configure_many({'target': -1, 'second': 1}, ['negative'])
    plan = controller.plan(probes)
    by_target = {entry['target']: entry['coefficient'] for entry in plan}
    assert by_target['target'] > 0 > by_target['second']
    joint = controller.status()['joint_plan']
    assert joint['conflicts'] == ['negative']
    assert joint['net_budget_used'] == pytest.approx(0)
    assert {entry['target'] for entry in joint['helpers'][0]['contributions']} == {
        'target', 'second'}


def test_each_batch_target_requires_its_own_complete_live_evidence():
    probes = batch_bank()
    controller = IndirectSteering()
    controller.configure_many({'target': -1, 'second': 1}, ['negative'])
    train(controller, probes)  # No second-target readings are present.
    assert {entry['target'] for entry in controller.plan(probes)} == {'target'}
    for index in range(50, 54):
        value = (index % 7 - 3) / 10
        controller.observe(row(index, target=value, second=value, negative=-value), probes)
    assert {entry['target'] for entry in controller.plan(probes)} == {'target'}
    for index in range(54, 64):
        value = (index % 7 - 3) / 10
        controller.observe(row(index, target=value, second=value, negative=-value), probes)
    assert {entry['target'] for entry in controller.plan(probes)} == {'target', 'second'}


def test_batch_optional_purpose_and_condition_apply_to_each_new_objective():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    controller.configure_many({'target': -1, 'third': 1}, ['+negative'],
                              purpose='coherence', when=['other', 'above', .1])
    assert all(route['purpose'] == 'coherence' for route in controller.routes.values())
    assert controller.plan(probes) == []
    controller.observe(row(99, target=.2, third=-.2, negative=-.2, other=.4), probes)
    assert {entry['target'] for entry in controller.plan(probes)} == {'target', 'third'}


def test_batch_feedback_tracks_each_target_once_and_survives_restart(tmp_path):
    probes = batch_bank()
    controller = IndirectSteering(tmp_path / 'batch.indirect.json')
    train_batch(controller, probes)
    controller.configure_many({'target': -1, 'third': 1}, ['+negative'])
    plan = controller.plan(probes)
    controller.mark_applied(plan)
    next_sample = row(99, target=.1, third=-.1, negative=-.1)
    controller.observe(next_sample, probes)
    assert set(controller.feedback) == {'target', 'third'}
    before = deepcopy(controller.feedback)
    controller.observe(next_sample, probes)
    assert controller.feedback == before
    assert all(result['n'] == result['applied_n'] == 1 for result in before.values())
    assert before['target']['last_desired_delta'] == pytest.approx(
        -before['target']['last_target_delta'])
    assert before['third']['last_desired_delta'] == pytest.approx(
        before['third']['last_target_delta'])
    restored = IndirectSteering(controller.path)
    assert restored.routes == controller.routes
    assert restored.feedback == controller.feedback
    assert restored.plan(probes) == []


def test_grouped_batch_command_keeps_legacy_single_target_syntax():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    result = handle_indirect_command(
        ['indirect', 'minimize', 'target', 'second', 'maximize', 'third',
         'via', '+negative'], controller, probes)
    assert set(controller.routes) == {'target', 'second', 'third'}
    assert result['joint_plan']['helpers']
    handle_indirect_command(['indirect', 'third', 'minimize', '-negative'],
                            controller, probes)
    assert controller.routes['third']['direction'] == -1


def test_grouped_batch_suggestion_reports_proposals_without_configuring_routes():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    result = handle_indirect_command(
        ['indirect', 'suggest', 'minimize', 'target', 'second', 'maximize', 'third',
         'via', '+negative'], controller, probes)
    assert set(result['proposals']) == {'target', 'second', 'third'}
    assert not controller.routes


def test_joint_summary_keeps_opposing_target_provenance():
    from invariants.indirect_steering import summarize_plan
    entries = [
        {'name': 'negative', 'target': 'target', 'coefficient': .075},
        {'name': 'negative', 'target': 'second', 'coefficient': -.075},
    ]
    summary = summarize_plan(entries, budget=.15)
    assert summary['helpers'] == [{
        'name': 'negative', 'coefficient': 0.0,
        'contributions': [{'target': 'target', 'coefficient': .075},
                          {'target': 'second', 'coefficient': -.075}],
    }]
    assert summary['conflicts'] == ['negative']
    assert summary['gross_budget_used'] == pytest.approx(.15)
    assert summary['net_budget_used'] == pytest.approx(0)


def test_automatic_helpers_follow_role_preference_and_exclude_target_proxies():
    probes = bank()
    probes['target_alias'] = deepcopy(probes['target'])
    probes['target_proxy'] = probe([0, 0, 0, 0, 1, 0], frame='composed: target + other')
    controller = IndirectSteering()
    train(controller, probes, assign_roles=False)
    controller.set_role('positive', 'thinking')
    controller.set_role('negative', 'perception')
    controller.settings['top_k'] = 1
    controller.configure('target', -1)
    assert controller.routes['target']['controls'] is None
    plan = controller.plan(probes)
    assert coefficients(plan) == {'positive': pytest.approx(-.15)}
    assert {'target', 'target_alias', 'target_proxy'} <= controller.blocked_probes(probes)


@pytest.mark.parametrize('arguments,targets', [
    (['target', 'minimize'], {'target'}),
    (['minimize', 'target', 'second', 'maximize', 'third'], {'target', 'second', 'third'}),
])
def test_commands_without_via_configure_automatic_single_and_batch_routes(arguments, targets):
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    handle_indirect_command(['indirect', *arguments], controller, probes)
    assert set(controller.routes) == targets
    assert all(route['controls'] is None for route in controller.routes.values())
    plan = controller.plan(probes)
    assert plan and not ({entry['name'] for entry in plan} & targets)
    assert sum(abs(entry['coefficient']) for entry in plan) <= .15 + 1e-9


def test_automatic_route_waits_when_no_helper_has_sufficient_evidence():
    probes = bank()
    controller = IndirectSteering()
    controller.configure('target', -1)
    train(controller, probes, count=7)
    assert controller.plan(probes) == []
    assert controller.routes['target']['enabled']


def test_automatic_route_discovers_newly_eligible_helpers_without_reconfiguration():
    probes = {'target': probe([1, 0, 0, 0, 0, 0]),
              'positive': probe([0, 1, 0, 0, 0, 0])}
    controller = IndirectSteering()
    controller.settings['top_k'] = 1
    controller.configure('target', -1)
    for index in range(16):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(row(index, target=value,
                               positive=value + (index % 3 - 1) * .4), probes)
    assert set(coefficients(controller.plan(probes))) == {'positive'}
    before_route = deepcopy(controller.routes['target'])
    probes['adaptive'] = probe([0, 0, 0, 0, 1, 0])
    for index in range(16, 23):
        value = ((index * 7) % 17 - 8) / 10
        controller.observe(row(index, target=value, adaptive=-value,
                               positive=value + (index % 3 - 1) * .4), probes)
    assert set(coefficients(controller.plan(probes))) == {'positive'}
    controller.observe(row(23, target=.4, adaptive=-.4, positive=.7), probes)
    assert coefficients(controller.plan(probes))['adaptive'] > 0
    assert controller.routes['target'] == before_route


def test_automatic_selection_persists_as_null_and_resumes_after_a_fresh_row(tmp_path):
    import json
    probes = bank()
    path = tmp_path / 'automatic.indirect.json'
    controller = IndirectSteering(path)
    train(controller, probes)
    controller.configure('target', -1)
    assert json.loads(path.read_text())['routes']['target']['controls'] is None
    restored = IndirectSteering(path)
    assert restored.routes['target']['controls'] is None
    assert restored.plan(probes) == []
    restored.observe(row(99, target=.2, negative=-.2, positive=.2), probes)
    assert restored.plan(probes)
    assert restored.routes['target']['controls'] is None


def test_explicit_via_restricts_automatic_discovery_and_empty_via_is_invalid():
    probes = bank()
    controller = IndirectSteering()
    train(controller, probes)
    handle_indirect_command(['indirect', 'minimize', 'target', 'via', 'positive'],
                            controller, probes)
    assert set(coefficients(controller.plan(probes))) == {'positive'}
    before = deepcopy(controller.routes)
    with pytest.raises(ValueError):
        handle_indirect_command(['indirect', 'minimize', 'target', 'via'], controller, probes)
    assert controller.routes == before


def test_explicit_empty_helper_list_does_not_mean_automatic_selection():
    controller = IndirectSteering()
    with pytest.raises(ValueError):
        controller.configure('target', -1, [])
    with pytest.raises(ValueError):
        controller.configure_many({'target': -1, 'second': 1}, [])
    assert not controller.routes


def test_invalid_automatic_batch_remains_atomic():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    controller.configure('third', 1, ['negative'])
    before = deepcopy(controller.routes)
    with pytest.raises(ValueError):
        controller.configure_many({'target': -1, 'missing': 1})
    assert controller.routes == before


def test_automatic_batch_obeys_common_gate_and_shared_budget():
    probes = batch_bank()
    controller = IndirectSteering()
    train_batch(controller, probes)
    controller.configure_many({'target': -1, 'second': -1, 'third': 1},
                              when=['other', 'above', .1])
    assert controller.plan(probes) == []
    controller.observe(row(99, target=.2, second=.2, third=-.2,
                           negative=-.2, positive=.2, other=.4), probes)
    plan = controller.plan(probes)
    assert {entry['target'] for entry in plan} == {'target', 'second', 'third'}
    assert not ({entry['name'] for entry in plan} & {'target', 'second', 'third'})
    assert sum(abs(entry['coefficient']) for entry in plan) <= .15 + 1e-9
