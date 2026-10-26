"""The shell consumes live-generation evidence and separates older baselines."""

from collections import deque
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import engine
from invariants.generation_probe_capture import GENERATION_PROBE_SOURCE
from invariants.probe_measurement import LEGACY_PROBE_SOURCE
from invariants.trigger_tuner import Trigger
from scripts import interactive_phenomenality as shell


def _probe(history=(), **fields):
    return {'direction': {0: torch.tensor([1., 0.])},
            'history': deque(history, maxlen=40), **fields}


def _tuner():
    return SimpleNamespace(triggers={}, observe=Mock(), credit=Mock(), save=Mock())


def _telemetry(raw, name='p'):
    return {
        'tokens_generated': 3,
        'generation_probe_readings': {name: {
            'raw_score': raw, 'last_token_score': raw,
            'mean_token_score': .3, 'token_count': 3,
            'by_layer': {0: raw}, 'by_token': [.1, .2, raw],
            'prediction_positions': [7, 8, 9],
            'measurement': GENERATION_PROBE_SOURCE,
        }},
        'generation_probe_capture': {'source': GENERATION_PROBE_SOURCE, 'token_count': 3},
    }


@pytest.fixture
def no_replay(monkeypatch):
    forbidden = Mock(side_effect=AssertionError('No text replay or replacement forward is allowed'))
    monkeypatch.setattr(shell, 'score_probes_on_text', forbidden)
    monkeypatch.setattr(engine, '_inputs', forbidden)
    monkeypatch.setattr(engine, '_hidden_states', forbidden)
    save = Mock()
    monkeypatch.setattr(shell, 'save_probe_raw_histories', save)
    return forbidden, save


def test_shell_records_and_credits_captured_values_without_text_replay(no_replay):
    probes = {'p': _probe([.2, .4], measurement_source=GENERATION_PROBE_SOURCE)}
    tuner, row = _tuner(), {}
    telemetry = _telemetry(.8)
    original = copy.deepcopy(telemetry)
    observed = shell.score_generation_probes(
        probes, tuner, telemetry, turn_row=row, turn_sense=.6, display=False)
    tuner.observe.assert_called_once_with('probe_p', pytest.approx(.5))
    tuner.credit.assert_called_once_with('probe_p', pytest.approx(.5), .6)
    assert row['probe_p'] == pytest.approx(.5)
    assert row['probe_measurement'] == telemetry['generation_probe_capture']
    assert row['probe_generation_readings'] == observed
    assert observed['p']['raw_score'] == .8 and observed['p']['token_count'] == 3
    assert observed['p']['by_token'] == [.1, .2, .8]
    assert observed['p']['prediction_positions'] == [7, 8, 9]
    assert telemetry == original
    assert list(probes['p']['history']) == [.2, .4, .8]
    no_replay[0].assert_not_called()
    no_replay[1].assert_called_once_with(probes)


def test_migrated_frozen_probe_seeds_once_without_reusing_reply_replay_baseline(no_replay):
    probes = {'p': _probe([.2, .4], frozen=True, last_profile={'raw_score': 99.})}
    tuner, row = _tuner(), {}
    old_trigger = Trigger('probe_p', .15)
    old_trigger.observe(99.)
    old_trigger.credit(99., .5)
    tuner.triggers['probe_p'] = old_trigger
    shell.score_generation_probes(probes, tuner, _telemetry(.8), turn_row=row, display=False)

    assert probes['p']['measurement_source'] == GENERATION_PROBE_SOURCE
    assert probes['p']['measurement_histories'][LEGACY_PROBE_SOURCE]['history'] == [.2, .4]
    assert list(probes['p']['history']) == [.8]
    assert list(old_trigger.signals) == list(old_trigger.outcomes) == []
    assert old_trigger.value == .15 and old_trigger.observed == old_trigger.fired == 0
    assert 'probe_p' not in row  # A seed cannot masquerade as measured zero.
    tuner.observe.assert_not_called()
    shell.score_generation_probes(probes, tuner, _telemetry(.9), turn_row=row, display=False)
    tuner.observe.assert_called_once_with('probe_p', pytest.approx(.1))
    assert row['probe_p'] == pytest.approx(.1)
    assert list(probes['p']['history']) == [.8]  # Frozen after the fresh source seed.
    assert not shell.probe_row_matches_source({'probe_p': .5}, probes['p'])
    assert shell.probe_row_matches_source(row, probes['p'])


@pytest.mark.parametrize('payload', [{}, {'generation_probe_readings': {}},
                                     _telemetry(float('nan')), _telemetry(float('inf'))])
def test_unavailable_or_nonfinite_capture_does_not_create_zero_measurements(no_replay, payload):
    probes = {'p': _probe([.2], measurement_source=GENERATION_PROBE_SOURCE)}
    tuner, row = _tuner(), {}
    result = shell.score_generation_probes(probes, tuner, payload, turn_row=row, display=False)
    assert result == row['probe_generation_readings'] == {}
    assert 'probe_p' not in row
    assert list(probes['p']['history']) == [.2] and 'last_profile' not in probes['p']
    tuner.observe.assert_not_called()
    tuner.credit.assert_not_called()
    no_replay[0].assert_not_called()
    no_replay[1].assert_not_called()


def test_persisted_raw_baselines_remain_separate_by_measurement_source(tmp_path, monkeypatch):
    path = tmp_path / 'raw_history.json'
    path.write_text(json.dumps({'version': 1, 'probes': {'p': [.2, .4]}}), encoding='utf-8')
    persist = shell.save_probe_raw_histories
    monkeypatch.setattr(shell, 'save_probe_raw_histories', lambda probes: persist(probes, path))
    probes = {'p': _probe([.2, .4], frozen=True)}
    shell.score_generation_probes(probes, _tuner(), _telemetry(.8), display=False)
    assert shell.load_probe_raw_histories(path, LEGACY_PROBE_SOURCE) == {'p': [.2, .4]}
    assert shell.load_probe_raw_histories(path, GENERATION_PROBE_SOURCE) == {'p': [.8]}
    assert shell.load_probe_raw_histories(path) == {'p': [.8]}

    # A later legacy-source write must leave the generation baseline available.
    assert persist({'p': _probe([.7], measurement_source=LEGACY_PROBE_SOURCE)}, path)
    assert shell.load_probe_raw_histories(path, LEGACY_PROBE_SOURCE) == {'p': [.7]}
    assert shell.load_probe_raw_histories(path, GENERATION_PROBE_SOURCE) == {'p': [.8]}


@pytest.mark.parametrize('return_telemetry', [False, True])
def test_generation_wrapper_preserves_output_and_scores_only_live_engine_telemetry(
        monkeypatch, no_replay, return_telemetry):
    probes = {'p': _probe([.2], measurement_source=GENERATION_PROBE_SOURCE)}
    tuner, wrapped = _tuner(), object()
    telemetry = _telemetry(.8)
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        assert args == (wrapped,)
        assert kwargs['instruction'] == 'Original prompt retained verbatim'
        assert kwargs['generation_probes'] is probes
        assert kwargs['return_telemetry'] is True
        return 'Unchanged engine answer', telemetry

    monkeypatch.setattr(shell, '_engine_generate', generate)
    monkeypatch.setattr(shell, '_ACTIVE_PROBES', probes)
    monkeypatch.setattr(shell, '_ACTIVE_MODEL', wrapped)
    monkeypatch.setattr(shell, '_ACTIVE_TUNER', tuner)
    monkeypatch.setattr(shell, '_PROBES_SHOW_ACT', True)
    monkeypatch.setattr(shell, '_PROBES_SHOW_TALK', False)
    result = shell.generate_agentic_text(
        wrapped, instruction='Original prompt retained verbatim', return_telemetry=return_telemetry)
    assert result == (('Unchanged engine answer', telemetry) if return_telemetry else 'Unchanged engine answer')
    assert len(calls) == 1
    tuner.observe.assert_called_once_with('probe_p', pytest.approx(.6))
    no_replay[0].assert_not_called()


def test_reader_generation_does_not_publish_speech_probe_readings(monkeypatch, no_replay):
    probes = {'p': _probe([.2], measurement_source=GENERATION_PROBE_SOURCE)}
    tuner = _tuner()
    generate = Mock(return_value=('Private reader text', _telemetry(.8)))
    monkeypatch.setattr(shell, '_engine_generate', generate)
    monkeypatch.setattr(shell, '_ACTIVE_PROBES', probes)
    monkeypatch.setattr(shell, '_ACTIVE_MODEL', object())
    monkeypatch.setattr(shell, '_ACTIVE_TUNER', tuner)
    monkeypatch.setattr(shell, '_PROBES_SHOW_ACT', True)
    assert shell.generate_agentic_text(object(), instruction='Reader source',
                                       generation_channel='reader', return_telemetry=True)[0] == 'Private reader text'
    tuner.observe.assert_not_called()
    no_replay[0].assert_not_called()


def test_whole_prediction_trace_survives_turn_row_json_persistence(no_replay):
    count = 160
    values = [index / 1000 for index in range(count)]
    positions = list(range(17, 17 + count))
    telemetry = _telemetry(values[-1])
    telemetry['tokens_generated'] = count
    telemetry['generation_probe_capture']['token_count'] = count
    telemetry['generation_probe_readings']['p'].update(
        token_count=count, by_token=values, prediction_positions=positions)
    probes = {'p': _probe([.2], measurement_source=GENERATION_PROBE_SOURCE)}
    row = {}
    shell.score_generation_probes(probes, _tuner(), telemetry, turn_row=row, display=False)
    reloaded = json.loads(json.dumps(row))['probe_generation_readings']['p']
    assert reloaded['token_count'] == len(reloaded['by_token']) == len(reloaded['prediction_positions']) == count
    assert reloaded['by_token'] == values
    assert reloaded['prediction_positions'] == positions
    assert reloaded['by_token'][-1] == values[-1] and reloaded['prediction_positions'][-1] == 176
    no_replay[0].assert_not_called()
