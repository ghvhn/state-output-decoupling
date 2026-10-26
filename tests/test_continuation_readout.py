"""Selected continuation readings stay visible without becoming live baselines."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import continuation_investigation as continuation
from scripts import interactive_phenomenality as shell
from test_continuation_investigation import rows, run, synthetic


def add_observer(probes):
    probes['observer'] = {
        'direction': {1: torch.tensor([0., 0., 1.])},
        'framings': ('Observer present.', 'Observer absent.'),
        'measurement_source': 'live_assistant_prediction',
        'history': [.25, .5], 'released': True, 'chatty': False,
    }


def test_all_named_probes_are_measured_including_quiet_released_observers(synthetic, tmp_path):
    _model, probes = synthetic
    add_observer(probes)
    result = run(synthetic, tmp_path, max_new_tokens=2)
    readings = result['telemetry']['generation_probe_readings']
    assert {'target', 'helper', 'observer'} <= readings.keys()
    assert all(readings[name]['token_count'] == 2 for name in probes)
    trials = rows(tmp_path, 'trials.jsonl')
    assert {trial['arm'] for trial in trials} == {'baseline', 'helper:+', 'helper:-'}
    assert all(set(probes) <= trial['telemetry']['generation_probe_readings'].keys()
               for trial in trials)
    assert all(readings[name]['measurement'] == continuation.SOURCE for name in probes)


def test_observer_profile_contains_only_winning_committed_predictions(synthetic, tmp_path):
    _model, probes = synthetic
    add_observer(probes)
    result = run(synthetic, tmp_path, max_new_tokens=3, stride=1, horizon=3)
    trials = rows(tmp_path, 'trials.jsonl')
    decisions = rows(tmp_path, 'decisions.jsonl')
    expected = []
    for decision in decisions:
        chosen = next(row for row in trials if row['decision'] == decision['decision']
                      and row['arm'] == decision['selected_arm'])
        expected.extend(chosen['telemetry']['generation_probe_readings']['observer']['by_token']
                        [:decision['committed_tokens']])
    profile = result['telemetry']['generation_probe_readings']['observer']
    assert profile['by_token'] == expected
    assert profile['token_count'] == len(expected) == 3
    assert profile['mean_token_score'] == pytest.approx(sum(expected) / len(expected))
    assert profile['last_token_score'] == expected[-1]
    assert profile['prediction_positions'] == [3, 4, 5]
    assert result['telemetry']['branch_tokens'] > len(expected)


def test_display_uses_raw_selected_metrics_and_excludes_internal_aliases(synthetic, tmp_path, capsys):
    _model, probes = synthetic
    add_observer(probes)
    probes['observer']['chatty'] = True
    result = run(synthetic, tmp_path, max_new_tokens=2)
    observed = shell.display_continuation_probe_readings(result, probes)
    output = capsys.readouterr().out
    assert '[Continuation Probe]' in output
    assert all(name in observed for name in probes)
    assert not any(name.startswith(continuation.FINAL_PREFIX) for name in observed)
    assert continuation.FINAL_PREFIX not in output
    for name in probes:
        expected = result['telemetry']['generation_probe_readings'][name]
        assert observed[name]['mean_token_score'] == expected['mean_token_score']
        assert observed[name]['last_token_score'] == expected['last_token_score']
        assert observed[name]['token_count'] == 2
    assert observed['target']['target_final_mean'] == result['telemetry']['generation_probe_readings'][continuation.FINAL_PREFIX + 'target']['mean_token_score']
    assert observed['target']['target_final_layer'] == 2
    assert 'final_L2_mean' in output


def test_display_does_not_update_baselines_or_live_learning(synthetic, tmp_path, monkeypatch):
    _model, probes = synthetic
    result = run(synthetic, tmp_path, max_new_tokens=2)
    originals = deepcopy(probes)
    blockers = {}
    for name in ('prepare_probe_measurements', 'centered_probe_observation',
                 'save_probe_raw_histories', 'score_generation_probes'):
        blockers[name] = Mock(side_effect=AssertionError(f'{name} must stay out of continuation display'))
        monkeypatch.setattr(shell, name, blockers[name])
    shell.display_continuation_probe_readings(result, probes, display=False)
    for name, original in originals.items():
        current = probes[name]
        assert current.keys() == original.keys()
        for key in original.keys() - {'direction'}:
            assert current[key] == original[key]
        for layer, direction in original['direction'].items():
            torch.testing.assert_close(current['direction'][layer], direction)
    for blocker in blockers.values():
        blocker.assert_not_called()


def test_chatty_controls_output_but_keeps_measurements_available(synthetic, tmp_path, capsys):
    _model, probes = synthetic
    add_observer(probes)
    result = run(synthetic, tmp_path, max_new_tokens=1)
    observed = shell.display_continuation_probe_readings(result, probes)
    output = capsys.readouterr().out
    assert 'observer' in observed
    assert 'observer' not in output
    assert 'target' in output and 'helper' in output
    silent = shell.display_continuation_probe_readings(result, probes, display=False)
    assert capsys.readouterr().out == ''
    assert silent.keys() == observed.keys()
    shell.display_continuation_probe_readings(result, probes, show_all=True)
    assert 'observer' in capsys.readouterr().out


def test_display_ignores_uncommitted_candidate_telemetry(synthetic, tmp_path):
    _model, probes = synthetic
    result = run(synthetic, tmp_path, max_new_tokens=1)
    original = deepcopy(result)
    result['discarded_candidates'] = [{
        'generation_probe_readings': {'target': {'mean_token_score': 999.,
            'last_token_score': 999., 'token_count': 99}},
    }]
    observed = shell.display_continuation_probe_readings(result, probes, display=False)
    assert observed['target']['token_count'] == 1
    assert observed['target']['mean_token_score'] == original['telemetry']['generation_probe_readings']['target']['mean_token_score']


@pytest.mark.parametrize('source', ['live_assistant_prediction', 'reply_text_replay', None])
def test_display_refuses_a_different_measurement_source(synthetic, tmp_path, source):
    _model, probes = synthetic
    result = run(synthetic, tmp_path, max_new_tokens=1)
    result['source'] = source
    result['telemetry']['source'] = source
    result['telemetry']['generation_probe_capture']['source'] = source
    assert shell.display_continuation_probe_readings(result, probes, display=False) == {}


def test_missing_profile_is_reported_without_a_reply_replay(synthetic, tmp_path, capsys, monkeypatch):
    _model, probes = synthetic
    result = run(synthetic, tmp_path, max_new_tokens=1)
    result['telemetry']['generation_probe_readings'].pop('helper')
    replay = Mock(side_effect=AssertionError('Display cannot rescore response text'))
    monkeypatch.setattr(shell, 'score_generation_probes', replay)
    observed = shell.display_continuation_probe_readings(result, probes)
    output = capsys.readouterr().out
    assert 'helper' not in observed
    assert 'helper' in output and 'unavailable' in output.lower()
    replay.assert_not_called()


def test_incompatible_optional_observer_does_not_abort_objective_selection(synthetic, tmp_path, capsys):
    _model, probes = synthetic
    probes['incompatible_observer'] = {
        'direction': {1: torch.ones(4)}, 'history': [.5], 'chatty': True,
        'framings': ('Optional present.', 'Optional absent.'),
        'measurement_source': 'live_assistant_prediction',
    }
    result = run(synthetic, tmp_path, max_new_tokens=1)
    assert result['response'] == '<40>'
    assert 'incompatible_observer' not in result['telemetry']['generation_probe_readings']
    observed = shell.display_continuation_probe_readings(result, probes)
    output = capsys.readouterr().out
    assert 'incompatible_observer' not in observed
    assert 'incompatible_observer' in output and 'unavailable' in output.lower()
    assert probes['incompatible_observer']['history'] == [.5]


def test_presented_readouts_are_bounded_and_do_not_observe_or_credit(synthetic, tmp_path, capsys):
    _model, probes = synthetic
    add_observer(probes)
    result = run(synthetic, tmp_path, max_new_tokens=1)
    histories = {name: list(probe['history']) for name, probe in probes.items()}
    tuner = SimpleNamespace(observe=Mock(), credit=Mock(), save=Mock())
    for index in range(25):
        current = {**result, 'output_dir': f'run-{index}'}
        shell.present_continuation_result(current, probes, tuner)
    capsys.readouterr()
    assert len(tuner._continuation_probe_readouts) == 20
    assert tuner._continuation_probe_readouts[0]['output_dir'] == 'run-5'
    assert tuner._continuation_probe_readouts[-1]['output_dir'] == 'run-24'
    assert tuner._last_probe_readout_source == continuation.SOURCE
    assert all('response' not in record for record in tuner._continuation_probe_readouts)
    assert all(list(probe['history']) == histories[name] for name, probe in probes.items())
    tuner.observe.assert_not_called()
    tuner.credit.assert_not_called()
    tuner.save.assert_not_called()


def test_probe_values_exposes_quiet_cached_readings_until_an_ordinary_turn(synthetic, tmp_path, capsys):
    _model, probes = synthetic
    add_observer(probes)
    result = run(synthetic, tmp_path, max_new_tokens=1)
    tuner = SimpleNamespace(observe=Mock(), credit=Mock(), save=Mock(), triggers={})
    shell.present_continuation_result(result, probes, tuner)
    capsys.readouterr()
    assert shell.print_cached_continuation_probe_values(tuner, probes, selected=['observer'])
    output = capsys.readouterr().out
    assert 'observer' in output and 'target' not in output and 'helper' not in output
    assert continuation.SOURCE in output and 'raw scores' in output
    tuner.observe.assert_not_called()
    tuner.credit.assert_not_called()
    shell.score_generation_probes({}, tuner, {}, display=False)
    assert tuner._last_probe_readout_source == 'live_assistant_prediction'
    assert not shell.print_cached_continuation_probe_values(tuner, probes)
    assert capsys.readouterr().out == ''
    assert len(tuner._continuation_probe_readouts) == 1
