"""Measurement-source changes must not relabel old probe evidence as live."""
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants.exposure_language import ExposureLanguage, probe_alias
from invariants.other_probes import OtherReference
from invariants.probe_behavior import ProbeBehavior, definition_key
from invariants.probe_measurement import (
    LEGACY_PROBE_SOURCE, bounded_probe_measurement,
    ensure_probe_measurement_source, prepare_probe_measurements,
)


LIVE_SOURCE = 'live_assistant_prediction'


def probe():
    return {'framings': ('Explore alternatives.', 'Settle immediately.'),
            'direction': {0: torch.tensor([1., 0.])},
            'history': deque([.1, .2], maxlen=40),
            'other_histories': {'self': [.3], 'user': [.4]}, 'exposed': True}


def tuner():
    trigger = SimpleNamespace(signals=deque([.1, .4]), outcomes=deque([(.4, .7)]),
                              observed=2, fired=1, calibrations=3, value=.23)
    return SimpleNamespace(triggers={'probe_p': trigger}, save=Mock())


def test_switch_preserves_independent_raw_and_other_histories():
    p = probe()
    original_key = definition_key('p', p)
    assert ensure_probe_measurement_source(p, LIVE_SOURCE)
    assert not p['history'] and not p['other_histories']
    assert p['_measurement_seed_pending']
    assert p['measurement_histories'][LEGACY_PROBE_SOURCE]['history'] == [.1, .2]
    assert definition_key('p', p) != original_key
    p['history'].append(.8)
    p['other_histories']['self'] = [.9]
    assert ensure_probe_measurement_source(p, LEGACY_PROBE_SOURCE)
    assert list(p['history']) == [.1, .2]
    assert p['other_histories'] == {'self': [.3], 'user': [.4]}
    assert definition_key('p', p) == original_key
    assert not p['_measurement_seed_pending']
    p['other_histories']['self'].append(.5)
    ensure_probe_measurement_source(p, LIVE_SOURCE)
    assert list(p['history']) == [.8]
    assert p['other_histories'] == {'self': [.9]}


def test_frozen_new_source_requests_exactly_one_seed_without_repeated_reset():
    from scripts.interactive_phenomenality import centered_probe_observation

    p = dict(probe(), frozen=True, last_profile={'raw_score': .5})
    assert ensure_probe_measurement_source(p, LIVE_SOURCE)
    assert p['_measurement_seed_pending'] and p['frozen']
    assert 'last_profile' not in p
    assert centered_probe_observation(p, .6) is None
    assert centered_probe_observation(p, .8) == pytest.approx(.2)
    assert not ensure_probe_measurement_source(p, LIVE_SOURCE)
    assert list(p['history']) == [.6]
    assert '_measurement_seed_pending' not in p


def test_prepare_removes_stale_model_visible_signals_and_credit_only_on_switch():
    probes = {'p': probe()}
    t = tuner()
    unrelated = SimpleNamespace(signals=deque([7.]))
    t.triggers['unrelated'] = unrelated
    assert prepare_probe_measurements(probes, t, LIVE_SOURCE) == ['p']
    trigger = t.triggers['probe_p']
    assert not trigger.signals and not trigger.outcomes
    assert trigger.observed == trigger.fired == trigger.calibrations == 0
    assert trigger.value == .23 and list(unrelated.signals) == [7.]
    archived = probes['p']['measurement_trigger_histories'][LEGACY_PROBE_SOURCE]
    assert archived['signals'] == [.1, .4] and archived['outcomes'] == [(.4, .7)]
    trigger.signals.append(.9)
    assert prepare_probe_measurements(probes, t, LIVE_SOURCE) == []
    assert list(trigger.signals) == [.9]
    t.save.assert_called_once()


def test_legacy_call_and_namespace_remain_unchanged():
    p = probe()
    before = definition_key('p', p)
    assert not ensure_probe_measurement_source(p, LEGACY_PROBE_SOURCE)
    assert 'measurement_source' not in p and definition_key('p', p) == before
    with pytest.raises(ValueError, match='nonempty'):
        ensure_probe_measurement_source(p, ' ')


def test_persisted_behavior_and_language_are_scoped_without_losing_legacy(tmp_path):
    p = probe()
    probes = {'p': p}
    t = tuner()
    behavior = ProbeBehavior(tmp_path / 'behavior.json')
    language = ExposureLanguage(tmp_path / 'language.json')
    for reading in (-.8, -.3, .3, .8):
        behavior.observe('Legacy wording', 'Legacy input', {'probe_p': reading}, probes)
    old_key = definition_key('p', p)
    language.sync(probes, t)
    old_language_key = language.active['p']['key']
    for _ in range(4):
        language.observe('Distinctive legacy phrase', {'probe_p': .8}, probes)
    assert 'Legacy wording' in language.context(behavior=behavior, probes=probes)
    prepare_probe_measurements(probes, t, LIVE_SOURCE)
    language.sync(probes, t)
    assert language.active['p']['key'] != old_language_key
    assert 'Legacy wording' not in language.context(behavior=behavior, probes=probes)
    assert 'Distinctive legacy' not in language.context()
    assert 'Reading=' not in language.context()
    assert 'Relational observation' not in language.context()
    assert old_key in behavior.records and old_language_key in language.maps
    for reading in (-.4, -.1, .1, .4):
        behavior.observe('New live wording', 'New input', {'probe_p': reading}, probes)
    new_key = definition_key('p', p)
    assert new_key != old_key and len(behavior.records[new_key]) == 4
    resumed = ProbeBehavior(tmp_path / 'behavior.json')
    assert 'New live wording' in resumed.evidence('p', p, probes)
    assert 'Legacy wording' not in resumed.evidence('p', p, probes)
    prepare_probe_measurements(probes, t, LEGACY_PROBE_SOURCE)
    language.sync(probes, t)
    assert language.active['p']['key'] == old_language_key
    assert 'Legacy wording' in language.context(behavior=resumed, probes=probes)


def test_stale_exposure_snapshot_cannot_train_or_name_new_source():
    p = probe()
    probes = {'p': p}
    language = ExposureLanguage()
    language.sync(probes, tuner())
    old_mapping = language.active['p']['mapping']
    ensure_probe_measurement_source(p, LIVE_SOURCE)
    language.observe('Should not leak', {'probe_p': .8}, probes)
    assert old_mapping['n'] == 0
    with pytest.raises(ValueError, match='currently exposed'):
        language.rename(probe_alias('p'), 'new name', .9, 'observations', probes)


def test_behavior_keeps_only_bounded_measurement_provenance_for_each_referent():
    p = probe()
    ensure_probe_measurement_source(p, LIVE_SOURCE)
    provenance = {'source': LIVE_SOURCE, 'endpoint': 'post_layer', 'generation_id': 'a' * 200,
                  'token_count': 4, 'decision_position': 9, 'prompt_sha256': 'F' * 64,
                  'prompt': 'This must not be saved.', 'hidden_states': torch.ones(3)}
    behavior = ProbeBehavior()
    behavior.observe('Reply', 'Input', {'probe_p': .2, 'probe_measurement': provenance}, {'p': p})
    saved = behavior.records[definition_key('p', p)][0]['probe_measurement']
    assert saved == dict(source=LIVE_SOURCE, endpoint='post_layer', generation_id='a' * 128,
                         token_count=4, decision_position=9, prompt_sha256='f' * 64)
    behavior.observe('Other reply', 'Input', {'probe_measurement': provenance,
                     'other_probe_readings': {'p': {'user': {'centered': .3, 'binding': 'u'}}}}, {'p': p})
    assert behavior.records[definition_key('p', dict(p, other_binding='u'))][0]['probe_measurement'] == saved
    assert bounded_probe_measurement({'token_count': True, 'decision_position': -1,
                                      'prompt_sha256': 'not a digest', 'source': 5}) == {}


def test_other_readings_are_tagged_and_get_fresh_source_baselines():
    p = probe()
    p['framings'] = ('I converse with @.', 'I ignore @.')
    probes = {'p': p}
    reference = OtherReference()
    mint = lambda frames: {0: torch.tensor([1., 0.])}
    ensure_probe_measurement_source(p, LIVE_SOURCE)
    first = reference.participant_readings(probes, ['self', 'user'], mint, lambda direction: .2)
    assert first['p']['self']['centered'] is None
    assert first['p']['self']['measurement_source'] == LIVE_SOURCE
    second = reference.participant_readings(probes, ['self', 'user'], mint, lambda direction: .3)
    assert second['p']['self']['centered'] == pytest.approx(.1)
    ensure_probe_measurement_source(p, LEGACY_PROBE_SOURCE)
    assert p['other_histories'] == {'self': [.3], 'user': [.4]}


def test_backfill_uses_archive_measurement_source_instead_of_active_live_source():
    p = probe()
    legacy_key = definition_key('p', p)
    ensure_probe_measurement_source(p, LIVE_SOURCE)
    live_key = definition_key('p', p)
    behavior = ProbeBehavior()
    observations = [{'ts': 'old', 'reply': 'Old archived words', 'readings': {'p': .1}},
                    {'ts': 'new', 'reply': 'New archived words', 'readings': {'p': .2},
                     'probe_measurement': {'source': LIVE_SOURCE, 'token_count': 4,
                                           'prompt': 'not retained'}}]
    assert behavior.backfill(observations, {'p': p})['imported'] == 2
    assert [r['reply'] for r in behavior.records[legacy_key]] == ['Old archived words']
    assert [r['reply'] for r in behavior.records[live_key]] == ['New archived words']
    assert behavior.records[live_key][0]['probe_measurement'] == {'source': LIVE_SOURCE, 'token_count': 4}
    assert behavior.backfill(observations, {'p': p})['imported'] == 0
