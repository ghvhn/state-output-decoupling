"""Unequal helper doses are searched, physically injected, and traced exactly."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import continuation_investigation as continuation
from invariants.continuation_mode import ContinuationMode
from test_joint_continuation_helpers import helpers, nonlinear, read_rows


def shape(arm):
    return tuple(abs(part['coefficient']) for part in arm['components'])


def test_default_two_helper_grid_contains_balanced_and_both_unequal_allocations():
    arms, plan = continuation._build_candidate_arms(helpers(), [0])
    assert len(arms) == plan['selected'] == plan['total_possible'] == 17
    assert plan['mixture_resolution'] == 4
    assert not plan['truncated']
    pairs = [arm for arm in arms if arm['joint_size'] == 2]
    assert {shape(arm) for arm in pairs} == {(.25, .75), (.5, .5), (.75, .25)}
    for weights in ((.25, .75), (.5, .5), (.75, .25)):
        assert {tuple(part['sign'] for part in arm['components'])
                for arm in pairs if shape(arm) == weights} == {
                    (left, right) for left in (1, -1) for right in (1, -1)}
    for arm in pairs:
        assert sum(abs(part['coefficient']) for part in arm['components']) == pytest.approx(1.)
        torch.testing.assert_close(arm['vectors'][0], torch.tensor([
            *(part['coefficient'] for part in arm['components']), 0.]))


def test_three_helper_full_grid_and_capped_search_cover_unequal_pairs_and_triples():
    full, plan = continuation._build_candidate_arms(
        helpers(3), [0], max_joint_helpers=3, max_candidates=75)
    assert len(full) == plan['total_possible'] == 75
    assert len({arm['arm'] for arm in full}) == 75
    triples = [arm for arm in full if arm['joint_size'] == 3]
    assert len(triples) == 32
    assert {shape(arm) for arm in triples} == {
        (1 / 3, 1 / 3, 1 / 3), (.5, .25, .25), (.25, .5, .25), (.25, .25, .5)}
    limited, limited_plan = continuation._build_candidate_arms(
        helpers(3), [0], max_joint_helpers=3, max_candidates=27)
    assert len(limited) == 27 and limited_plan['truncated']
    assert limited_plan['total_possible'] == 75
    assert sum(arm['joint_size'] == 1 for arm in limited) == 6
    assert {(.75, .25), (.25, .75)} <= {
        shape(arm) for arm in limited if arm['joint_size'] == 2}
    assert {(.5, .25, .25), (.25, .5, .25), (.25, .25, .5)} <= {
        shape(arm) for arm in limited if arm['joint_size'] == 3}
    again, again_plan = continuation._build_candidate_arms(
        deepcopy(helpers(3)), [0], max_joint_helpers=3, max_candidates=27)
    assert limited_plan == again_plan
    assert [(arm['arm'], arm['components']) for arm in limited] == [
        (arm['arm'], arm['components']) for arm in again]


def test_mix_one_preserves_balanced_only_search():
    arms, plan = continuation._build_candidate_arms(
        helpers(3), [0], max_joint_helpers=3, mixture_resolution=1)
    assert len(arms) == plan['total_possible'] == 27
    assert all(len(set(shape(arm))) <= 1 for arm in arms)


@pytest.mark.parametrize(('resolution', 'count'), [(2, 3), (3, 2), (5, 3), (6, 4), (8, 4)])
def test_nondefault_grid_counts_match_distinct_positive_profiles(resolution, count):
    arms, plan = continuation._build_candidate_arms(
        helpers(count), [0], max_joint_helpers=count, max_candidates=10000,
        mixture_resolution=resolution)
    assert len(arms) == plan['total_possible']
    assert len({arm['arm'] for arm in arms}) == len(arms)
    assert not plan['truncated']
    for size in range(2, count + 1):
        profiles = {shape(arm) for arm in arms if arm['joint_size'] == size}
        assert len(profiles) == plan['profile_counts_by_joint_size'][size]
        assert all(all(weight > 0 for weight in weights)
                   and sum(weights) == pytest.approx(1.) for weights in profiles)


def test_explicit_signs_constrain_every_unequal_allocation():
    arms, plan = continuation._build_candidate_arms(helpers(signs=[1, -1]), [0])
    assert plan['total_possible'] == len(arms) == 6
    assert {shape(arm) for arm in arms if arm['joint_size'] == 2} == {
        (.25, .75), (.5, .5), (.75, .25)}
    assert all(part['sign'] == {'left': 1, 'right': -1}[part['helper']]
               and part['coefficient'] * part['sign'] > 0
               for arm in arms for part in arm['components'])


class UnequalOptimumLayer(torch.nn.Module):
    def forward(self, hidden):
        output = hidden.clone()
        x, y = hidden[..., 0], hidden[..., 1]
        # Singles and a uniform pair are worse than baseline. A 3:1 pair is
        # useful even after the common physical cap scales the combined dose.
        output[..., 2] = x * y - (x - 3 * y).square()
        return output


def run(nonlinear, directory, **overrides):
    model, probes = nonlinear
    model.model.model.layers[1] = UnequalOptimumLayer()
    options = dict(prompt_messages=[{'role': 'user', 'content': 'Finish this'}],
                   output_dir=directory, max_new_tokens=1, stride=1, horizon=1,
                   alpha=.25, cap_fraction=.02, layers=[0], max_helpers=2,
                   progress=None)
    options.update(overrides)
    return continuation.run_continuation_investigation(
        model, probes, {'target': 1}, ['left', 'right'], **options)


def test_unequal_nonlinear_optimum_wins_and_real_hook_preserves_exact_ratio(
        nonlinear, tmp_path, monkeypatch):
    model, probes = nonlinear
    before = deepcopy(probes)
    physical = []
    progress = []
    original = continuation._steer_handles

    def record(_model, vectors, layers, alpha, **options):
        physical.append((deepcopy(vectors), alpha, options))
        return original(_model, vectors, layers, alpha, **options)

    monkeypatch.setattr(continuation, '_steer_handles', record)
    result = run(nonlinear, tmp_path, progress=progress.append)
    trials = read_rows(tmp_path, 'trials.jsonl')
    decision = read_rows(tmp_path, 'decisions.jsonl')[0]
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert len(trials) == len(model.model.calls) == 17
    assert len(physical) == 16  # One cap envelope per nonbaseline arm.
    assert decision['selected_components'] == [
        {'helper': 'left', 'sign': 1, 'coefficient': .75},
        {'helper': 'right', 'sign': 1, 'coefficient': .25}]
    assert decision['selected_utility'] > 0
    for candidate in decision['arms']:
        if len(candidate['components']) <= 1 or len(set(shape(candidate))) <= 1:
            assert candidate['utility'] <= 0
    index = next(i for i, row in enumerate(trials) if row['arm'] == decision['selected_arm'])
    vectors, alpha, options = physical[index - 1]
    torch.testing.assert_close(vectors[0], torch.tensor([.75, .25, 0., 0.]))
    assert alpha == .25 and options['cap_fraction'] == .02
    state = model.model.calls[index]['states'][0][0, -1]
    assert float(state[0] / state[1]) == pytest.approx(3.)
    assert float(state[:2].norm()) == pytest.approx(.02, rel=1e-5)
    assert result['response'] == '<60>'
    assert manifest['settings']['mixture_resolution'] == 4
    assert manifest['settings']['candidate_plan']['total_possible'] == 17
    assert '.75' in decision['selected_arm'] or '0.75' in decision['selected_arm']
    selected_message = next(message for message in progress if 'selected ' in message)
    assert 'mix={left:+0.75,right:+0.25}' in selected_message
    assert all(row['source'] == continuation.SOURCE for row in trials)
    assert all(profile['measurement'] == continuation.SOURCE for row in trials
               for profile in row['telemetry']['generation_probe_readings'].values())
    readings = result['telemetry']['generation_probe_readings']
    assert all(profile['measurement'] == continuation.SOURCE for profile in readings.values())
    selected = readings['target']
    assert selected['token_count'] == len(selected['by_token']) == 1
    assert selected['mean_token_score'] > 0
    for name in probes:
        assert probes[name]['history'] == before[name]['history']
        for layer in probes[name]['direction']:
            torch.testing.assert_close(probes[name]['direction'][layer], before[name]['direction'][layer])


def test_unequal_sum_cannot_reconstruct_a_target_axis_or_inverse(nonlinear, tmp_path):
    model, probes = nonlinear
    probes['target']['direction'][0] = torch.tensor([3., 1., 0., 0.]) / 10 ** .5
    run(nonlinear, tmp_path)
    rows = read_rows(tmp_path, 'trials.jsonl')
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    plan = manifest['settings']['candidate_plan']
    assert plan['total_possible'] == 17
    assert plan['blocked_joint_count'] == 2
    assert len(rows) == len(model.model.calls) == 15
    assert not any(shape(row) == (.75, .25)
                   and row['components'][0]['sign'] == row['components'][1]['sign']
                   for row in rows if row['joint_size'] == 2)
    assert all(item['objectives'] == ['target'] for item in plan['blocked_joint_arms'])


def test_unequal_winner_commits_only_selected_prefix_and_keeps_lookahead_separate(nonlinear, tmp_path):
    callbacks = []
    result = run(nonlinear, tmp_path, max_new_tokens=2, stride=1, horizon=2,
                 max_decisions=1, on_commit=callbacks.append)
    trials = read_rows(tmp_path, 'trials.jsonl')
    decision = read_rows(tmp_path, 'decisions.jsonl')[0]
    winner = next(row for row in trials if row['arm'] == decision['selected_arm'])
    assert len(winner['token_ids']) == 2
    assert result['selected_token_ids'] == winner['token_ids'][:1]
    assert callbacks == [result['response']] == ['<60>']
    assert result['telemetry']['branch_tokens'] == 34
    assert result['telemetry']['stop_reason'] == 'max_decisions'
    selected = result['telemetry']['generation_probe_readings']
    for name, profile in selected.items():
        assert profile['measurement'] == continuation.SOURCE
        assert profile['token_count'] == len(profile['by_token']) == 1
        assert profile['by_token'] == winner['telemetry']['generation_probe_readings'][name]['by_token'][:1]


def test_auto_selection_still_excludes_target_aliases_with_asymmetric_search(nonlinear, tmp_path):
    model, probes = nonlinear
    model.model.model.layers[1] = UnequalOptimumLayer()
    probes['a_target_alias'] = deepcopy(probes['target'])
    result = continuation.run_continuation_investigation(
        model, probes, {'target': 1}, controls=None,
        prompt_messages=[{'role': 'user', 'content': 'Finish this'}],
        output_dir=tmp_path, max_new_tokens=1, stride=1, horizon=1,
        layers=[0], max_helpers=3, progress=None)
    trials = read_rows(tmp_path, 'trials.jsonl')
    assert len(trials) == 17
    assert {part['helper'] for row in trials for part in row['components']} == {'left', 'right'}
    assert 'a_target_alias' in result['telemetry']['generation_probe_readings']


@pytest.mark.parametrize('value', [0, -1, True, 1.5, 65])
def test_invalid_mixture_resolution_fails_before_any_generation(nonlinear, tmp_path, value):
    with pytest.raises(ValueError, match='mixture|resolution'):
        run(nonlinear, tmp_path, mixture_resolution=value)
    assert not nonlinear[0].model.calls
    assert not list(tmp_path.iterdir())


def test_maximum_resolution_is_bounded_before_constructing_grid():
    arms, plan = continuation._build_candidate_arms(
        helpers(5), [0], max_joint_helpers=5, max_candidates=33, mixture_resolution=64)
    assert len(arms) == 33
    assert plan['total_possible'] > 33 and plan['truncated']
    assert all(sum(abs(part['coefficient']) for part in arm['components']) == pytest.approx(1.)
               for arm in arms[1:])


def test_mode_retains_mixture_resolution_across_ordinary_turns_and_toggle():
    mode = ContinuationMode()
    assert mode.options['mix'] == 4
    mode.configure(['on', 'mix', '8'])
    for prompt in ('First turn', 'Second turn'):
        args = mode.arguments(prompt)
        assert args[args.index('mix') + 1] == '8'
    mode.configure(['off'])
    mode.configure(['on', 'tokens', '128'])
    assert mode.options['mix'] == 8


@pytest.mark.parametrize('value', ['0', '-1', '1.5', '65'])
def test_invalid_mode_mix_is_atomic(value):
    mode = ContinuationMode()
    mode.configure(['on', 'mix', '8'])
    before = mode.status()
    with pytest.raises(ValueError):
        mode.configure(['on', 'tokens', '1024', 'mix', value])
    assert mode.status() == before


def test_shell_maps_mixture_resolution_for_one_shot_and_repeated_turns(nonlinear, tmp_path, monkeypatch):
    from scripts import interactive_phenomenality as shell

    model, probes = nonlinear
    controller = SimpleNamespace(routes={'target': {'direction': 1, 'controls': None}}, roles={})
    memory = SimpleNamespace(records=[], scope='mix-test', session_id='mix-session', append_turn=Mock())
    calls = []

    def fake_run(_model, _probes, objectives, **options):
        calls.append(options)
        return {'source': continuation.SOURCE, 'fingerprint': 'mix-test',
                'response': 'SELECTED', 'output_dir': str(options['output_dir']),
                'telemetry': {'tokens_generated': 1, 'generation_probe_readings': {}}}

    monkeypatch.setattr(continuation, 'run_continuation_investigation', fake_run)
    shell.handle_continuation_investigation_command(
        ['continue', 'mix', '8', 'output', str(tmp_path / 'one'), '--', 'One prompt'],
        controller, probes, model, memory)
    assert calls[0]['mixture_resolution'] == 8
    shell.handle_continuation_mode_command(
        ['on', 'mix', '1', 'output', str(tmp_path / 'turns')], controller, probes, model)
    for prompt in ('First turn', 'Second turn'):
        shell.run_enabled_continuation_turn(prompt, controller, probes, model, memory)
    assert all(call['mixture_resolution'] == 1 for call in calls[1:])


def test_cli_exposes_default_and_explicit_mixture_resolution():
    from scripts.investigate_indirect_continuation import parser

    parser_ = parser()
    defaults = parser_.parse_args(['--prompt', 'Prompt', '--output', 'unused'])
    assert defaults.mix == 4
    selected = parser_.parse_args(['--prompt', 'Prompt', '--output', 'unused', '--mix', '8'])
    assert selected.mix == 8


def test_cli_passes_mixture_resolution_to_actual_runner_entry(nonlinear, tmp_path, monkeypatch):
    from invariants import engine
    from scripts import investigate_indirect_backfill as backfill_cli
    from scripts.investigate_indirect_continuation import main

    model, probes = nonlinear
    calls = []

    def fake_run(_model, _probes, objectives, **options):
        calls.append(options)
        return {'response': 'SELECTED', 'output_dir': str(options['output_dir']),
                'telemetry': {'tokens_generated': 1}}

    monkeypatch.setattr(engine, 'load_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(backfill_cli, 'load_probe_bank', lambda *args, **kwargs: probes)
    monkeypatch.setattr(continuation, 'run_continuation_investigation', fake_run)
    main(['--prompt', 'Prompt', '--maximize', 'target', '--via', 'left', 'right',
          '--controller', str(tmp_path / 'owner.json'), '--output', str(tmp_path / 'run'),
          '--mix', '8'])
    assert calls[0]['mixture_resolution'] == 8
