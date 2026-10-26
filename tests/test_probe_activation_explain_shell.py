"""Exercise activation explanations through the actual operator command branch."""

import ast
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from invariants import activation_explanation_output, probe_activation_explain
from invariants.exposure_language import probe_alias
from invariants.probe_definition_task import DEFINITION_TASK_GAINS
from scripts import interactive_phenomenality as shell
from test_probe_definition_output_shell import DEFINITION, OutputHarness
from test_probe_definition_tuning import registered_tuner, run_tune


_DEFAULT_TELEMETRY = object()


def activation_harness(monkeypatch, tmp_path, command='explain secret_probe', *, gains=None,
                       failure=False, text=DEFINITION, telemetry=_DEFAULT_TELEMETRY):
    harness = OutputHarness(monkeypatch, tmp_path, command=command)
    supplied = gains or {}
    harness.tuner.get.side_effect = lambda name, default=None: supplied.get(name, default)
    for owner, name in (
        (harness.behavior, 'evidence'), (harness.behavior, 'definition_memories'),
        (harness.behavior, 'record_interpretation'),
        (harness.language, 'record_namespace_summary'),
    ):
        monkeypatch.setattr(owner, name, Mock(side_effect=AssertionError(f'Activation mode must not call {name}')))
    harness.namespace['defining_profile_for'] = Mock(side_effect=AssertionError('Activation mode must not use the evidence profile'))
    harness.namespace['load_prompt'] = Mock(side_effect=AssertionError('Activation mode must not load a written task prompt'))

    def generate(model, subject, evidence, **kwargs):
        assert model is harness.model
        assert subject is harness.probe['direction']
        assert evidence == ''
        assert kwargs['config'] is harness.config
        assert kwargs['generate'] == harness.generate
        assert kwargs['axis_cache'] is harness.matrix.activation_explanation_axes
        assert kwargs['map_enabled'] is bool(supplied.get('probe_explain_map', 0.))
        assert kwargs['map_tokens'] == supplied.get('probe_explain_map_tokens', 24.)
        for axis, default in probe_activation_explain.EXPLAIN_GAINS.items():
            assert kwargs[axis + '_gain'] == supplied.get('probe_explain_' + axis, default)
        if failure:
            raise RuntimeError('activation callback failed')
        kwargs['axis_cache']['cached-candidate-axis'] = True
        return {
            'text': text,
            'telemetry': {'tokens_generated': 30} if telemetry is _DEFAULT_TELEMETRY else copy.deepcopy(telemetry),
            'metadata': {'mode': 'activation', 'semantic_truth_checked': False},
        }

    harness.activation_generate = Mock(side_effect=generate)
    monkeypatch.setattr(probe_activation_explain, 'generate_activation_explanation', harness.activation_generate)
    return harness


@pytest.mark.parametrize('command', [
    'explain secret_probe', 'secret_probe explain', 'explain secret_probe activation',
    'secret_probe explain activation', '[secret_probe] explain', 'explain probe:secret_probe',
    f'[{probe_alias("secret_probe")}] explain',
])
def test_default_and_explicit_activation_commands_never_use_measured_evidence(monkeypatch, tmp_path, command):
    harness = activation_harness(monkeypatch, tmp_path, command)
    harness.probe['exposed'] = False
    output = harness.run()
    harness.activation_generate.assert_called_once()
    assert 'Activation explanation candidate' in output and DEFINITION in output
    assert 'unverified' in output
    assert harness.behavior.interpretations == {}
    assert harness.matrix.activation_explanations[-1] == {
        'probe': 'secret_probe', 'mode': 'activation', 'status': 'candidate',
        'text': DEFINITION, 'controls': dict(probe_activation_explain.EXPLAIN_GAINS),
        'metadata': {'mode': 'activation', 'semantic_truth_checked': False},
        'telemetry': {'tokens_generated': 30}, 'semantic_truth_checked': False,
        'qualification': activation_explanation_output.assess_activation_explanation(
            DEFINITION, {'tokens_generated': 30}, 384),
    }
    harness.matrix.other_reference.activate.assert_not_called()
    assert harness.calls == []


def test_activation_explanation_does_not_require_any_measured_examples(monkeypatch, tmp_path):
    harness = activation_harness(monkeypatch, tmp_path)
    harness.behavior.records.clear()
    harness.records_before = {}
    output = harness.run()
    harness.activation_generate.assert_called_once()
    assert 'Activation explanation candidate' in output
    assert 'at least four measured examples' not in output


def test_shell_map_knobs_enable_current_state_path_with_its_calibration_budget(monkeypatch, tmp_path):
    harness = activation_harness(monkeypatch, tmp_path,
        gains={'probe_explain_map': 1., 'probe_explain_map_tokens': 12.})
    output = harness.run()
    assert 'current-state map' in output
    assert harness.activation_generate.call_args.kwargs['map_enabled'] is True
    assert harness.activation_generate.call_args.kwargs['map_tokens'] == 12.
    assert harness.behavior.interpretations == {}


@pytest.mark.parametrize('value', [-1., .5, 2., float('nan')])
def test_invalid_map_selector_is_rejected_before_generation(monkeypatch, tmp_path, value):
    harness = activation_harness(monkeypatch, tmp_path, gains={'probe_explain_map': value})
    output = harness.run()
    assert 'probe_explain_map must be 0 or 1' in output
    harness.activation_generate.assert_not_called()


@pytest.mark.parametrize('axis', ['subject', 'operation', 'reference'])
def test_each_activation_gain_can_be_zero_without_disabling_other_channels(monkeypatch, tmp_path, axis):
    harness = activation_harness(monkeypatch, tmp_path, gains={'probe_explain_' + axis: 0.})
    harness.run()
    controls = harness.matrix.activation_explanations[-1]['controls']
    assert controls[axis] == 0.
    assert all(controls[name] == default for name, default in probe_activation_explain.EXPLAIN_GAINS.items() if name != axis)


@pytest.mark.parametrize('options', ['activation evidence', 'activation ping', 'evidence 2', 'prompt choose', 'unknown', 'nan'])
def test_invalid_mode_suffix_stops_before_activation_or_evidence_work(monkeypatch, tmp_path, options):
    harness = activation_harness(monkeypatch, tmp_path, command='explain secret_probe ' + options)
    output = harness.run()
    harness.activation_generate.assert_not_called()
    harness.owners.for_tuner.assert_not_called()
    assert not hasattr(harness.matrix, 'activation_explanations')
    assert 'Choose one explain mode' in output or 'Unknown explain option' in output or 'finite' in output


def test_failed_activation_attempt_does_not_enter_interpretation_or_naming_history(monkeypatch, tmp_path):
    harness = activation_harness(monkeypatch, tmp_path, failure=True)
    output = harness.run()
    assert 'Activation explanation failed: activation callback failed' in output
    assert not hasattr(harness.matrix, 'activation_explanations')
    assert harness.behavior.interpretations == {}


def operator_trace(tmp_path):
    return [json.loads(line) for line in (tmp_path / 'behavior.activation-explain.jsonl')
            .read_text(encoding='utf-8').splitlines()]


def test_generic_chat_is_unqualified_and_persisted_only_as_an_operator_trace(monkeypatch, tmp_path):
    opening = "At this time, I'm happy to chat with you. What would you like to talk about?"
    harness = activation_harness(monkeypatch, tmp_path, text=opening)
    output = harness.run()
    harness.activation_generate.assert_called_once()
    assert 'No activation explanation produced:' in output
    assert 'Unqualified continuation:\n' + opening in output
    assert 'Activation explanation candidate for' not in output
    attempt = harness.matrix.activation_explanations[-1]
    assert attempt['status'] == 'unqualified' and attempt['text'] == opening
    assert not attempt['qualification']['accepted']
    assert attempt['qualification']['structural_only']
    assert not attempt['qualification']['semantic_truth_checked']
    rows = operator_trace(tmp_path)
    assert len(rows) == 1
    assert {key: rows[0][key] for key in attempt} == attempt
    assert 'Operator trace:' in output and rows[0]['timestamp']
    assert harness.behavior.interpretations == {}
    harness.behavior.record_interpretation.assert_not_called()
    harness.language.record_namespace_summary.assert_not_called()
    harness.naming.assert_not_called()
    assert harness.calls == []


@pytest.mark.parametrize('telemetry', [
    {'tokens_generated': 384}, {'tokens_generated': 30, 'finish_reason': 'length'},
])
def test_incomplete_activation_output_is_retained_without_a_valid_candidate(monkeypatch, tmp_path, telemetry):
    harness = activation_harness(monkeypatch, tmp_path, telemetry=telemetry)
    output = harness.run()
    harness.activation_generate.assert_called_once()
    assert 'No activation explanation produced:' in output
    assert 'generation limit' in output
    assert 'Unqualified continuation:\n' + DEFINITION in output
    assert 'Activation explanation candidate for' not in output
    attempt = harness.matrix.activation_explanations[-1]
    assert attempt['status'] == 'incomplete'
    assert attempt['qualification']['status'] == 'incomplete'
    assert not attempt['qualification']['accepted']
    assert operator_trace(tmp_path)[0]['status'] == 'incomplete'
    assert harness.behavior.interpretations == {}
    harness.naming.assert_not_called()


def test_operator_trace_write_failure_keeps_output_and_candidate_history(monkeypatch, tmp_path):
    harness = activation_harness(monkeypatch, tmp_path)
    record = Mock(side_effect=OSError('operator log unavailable'))
    monkeypatch.setattr(activation_explanation_output, 'record_activation_explanation', record)
    output = harness.run()
    assert 'Could not save operator trace: operator log unavailable' in output
    assert 'Activation explanation candidate for' in output and DEFINITION in output
    assert 'Operator trace:' not in output
    attempt = harness.matrix.activation_explanations[-1]
    assert attempt['status'] == 'candidate' and attempt['qualification']['accepted']
    record.assert_called_once_with(harness.matrix, attempt)
    assert not (tmp_path / 'behavior.activation-explain.jsonl').exists()
    assert harness.behavior.interpretations == {}
    harness.language.record_namespace_summary.assert_not_called()
    harness.naming.assert_not_called()


@pytest.mark.parametrize('telemetry', [None, {}])
def test_missing_activation_token_telemetry_is_a_clean_failure_without_history(monkeypatch, tmp_path, telemetry):
    harness = activation_harness(monkeypatch, tmp_path, telemetry=telemetry)
    output = harness.run()
    harness.activation_generate.assert_called_once()
    assert 'Activation explanation failed:' in output and 'requires tokens_generated' in output
    assert 'Activation explanation candidate for' not in output
    assert not hasattr(harness.matrix, 'activation_explanations')
    assert not (tmp_path / 'behavior.activation-explain.jsonl').exists()
    assert harness.behavior.interpretations == {}
    harness.naming.assert_not_called()


def test_activation_axis_cache_is_owned_and_candidate_history_is_bounded(monkeypatch, tmp_path):
    harness = activation_harness(monkeypatch, tmp_path)
    existing = [{'text': str(index)} for index in range(20)]
    harness.matrix.activation_explanations = existing
    harness.run()
    cache = harness.matrix.activation_explanation_axes
    harness.run()
    assert harness.activation_generate.call_count == 2
    assert harness.matrix.activation_explanation_axes is cache
    assert all(call.kwargs['axis_cache'] is cache for call in harness.activation_generate.call_args_list)
    assert len(harness.matrix.activation_explanations) == 16
    assert harness.matrix.activation_explanations[0]['text'] == '6'


def test_activation_knobs_are_registered_and_bypass_probe_name_redirect():
    tree = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
    registration = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                        and ast.unparse(node.target) == '(explain_axis, explain_gain)')
    tuner = Mock()
    registration_namespace = dict(vars(shell), tuner=tuner, EXPLAIN_GAINS=probe_activation_explain.EXPLAIN_GAINS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(registration)], type_ignores=[])),
                 shell.__file__, 'exec'), registration_namespace)
    assert tuner.register.call_args_list == [
        ((('probe_explain_' + axis, gain)), {'kind': 'coefficient'})
        for axis, gain in probe_activation_explain.EXPLAIN_GAINS.items()
    ]

    redirect = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                    and 'targs[0] not in definition_controls' in ast.unparse(node.test))
    owner = next(node for node in ast.walk(tree) if isinstance(node, (ast.If, ast.For, ast.FunctionDef))
                 and redirect in getattr(node, 'body', []))
    redirect_index = owner.body.index(redirect)
    controls_start = next(index for index, node in enumerate(owner.body[:redirect_index])
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == 'definition_controls'
                                  for target in node.targets))
    controls_body = copy.deepcopy(owner.body[controls_start:redirect_index])
    for knob in ('probe_explain_subject', 'probe_explain_operation', 'probe_explain_reference',
                 'probe_explain_map', 'probe_explain_map_tokens'):
        namespace = dict(vars(shell), targs=[knob, '0.0'], probes={}, tuner=Mock(triggers={knob: object()}),
                         DEFINITION_TASK_GAINS=DEFINITION_TASK_GAINS)
        exec(compile(ast.fix_missing_locations(ast.Module(body=controls_body, type_ignores=[])), shell.__file__, 'exec'), namespace)
        assert not eval(compile(ast.Expression(copy.deepcopy(redirect.test)), shell.__file__, 'eval'), namespace)
        assert knob in shell.CALIBRATION_CIRCULAR


def registered_explanation_tuner(path):
    tuner = registered_tuner(path)
    for axis, default in probe_activation_explain.EXPLAIN_GAINS.items():
        tuner.register('probe_explain_' + axis, default, kind='coefficient')
    tuner.register('probe_explain_map', 0., kind='coefficient')
    tuner.register('probe_explain_map_tokens', 24., kind='coefficient')
    return tuner


@pytest.mark.parametrize('name, value, expected', [
    ('probe_explain_map', '1', 1.), ('probe_explain_map', 'off', 0.),
    ('probe_explain_map_tokens', '3', 3.), ('probe_explain_map_tokens', '64', 64.),
])
def test_map_knobs_persist_and_bypass_probe_redirect(tmp_path, name, value, expected):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_explanation_tuner(path)
    output = run_tune(':tune ' + name + ' ' + value, tuner, {'secret_probe': {'chatty': True}})
    assert 'Redirected' not in output and 'No active probe' not in output
    assert registered_explanation_tuner(path).get(name) == expected


@pytest.mark.parametrize('name, value', [
    ('probe_explain_map', '0.5'), ('probe_explain_map', 'nan'), ('probe_explain_map', '2'),
    ('probe_explain_map_tokens', '2'), ('probe_explain_map_tokens', '65'),
    ('probe_explain_map_tokens', '3.5'), ('probe_explain_map_tokens', 'inf'),
])
def test_invalid_map_knobs_do_not_change_persisted_settings(tmp_path, name, value):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_explanation_tuner(path)
    before = tuner.get(name)
    output = run_tune(':tune ' + name + ' ' + value, tuner, {})
    assert 'must be' in output and 'Redirected' not in output
    assert registered_explanation_tuner(path).get(name) == before


@pytest.mark.parametrize('axis, settings', [
    ('subject', [('0', 0.), ('1', 1.), ('off', 0.), ('0.02', .02)]),
    ('operation', [('-0.12', -.12), ('-1', -1.), ('1', 1.), ('off', 0.)]),
    ('reference', [('-0.04', -.04), ('-1', -1.), ('1', 1.), ('off', 0.)]),
])
def test_explanation_tune_controls_persist_signed_or_zero_values_without_muting(tmp_path, axis, settings):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_explanation_tuner(path)
    probes = {'secret_probe': {'chatty': True}}
    knob = 'probe_explain_' + axis
    for setting, expected in settings:
        output = run_tune(':tune ' + knob + ' ' + setting, tuner, probes)
        assert 'Redirected' not in output and 'No active probe' not in output
        assert 'is a probe' not in output
        resumed = registered_explanation_tuner(path)
        assert tuner.get(knob) == resumed.get(knob) == expected
        assert resumed.triggers[knob].kind == 'coefficient'
        assert probes['secret_probe']['chatty'] is True
        assert resumed.get('probe_secret_probe') == .4
        for other_axis, default in probe_activation_explain.EXPLAIN_GAINS.items():
            if other_axis != axis:
                assert resumed.get('probe_explain_' + other_axis) == default


@pytest.mark.parametrize('axis, setting', [
    ('subject', '-0.01'), ('subject', '1.01'), ('subject', 'nan'),
    ('subject', 'inf'), ('subject', '-inf'),
    ('operation', '-1.01'), ('operation', '1.01'), ('operation', 'nan'),
    ('operation', 'inf'), ('operation', '-inf'),
    ('reference', '-1.01'), ('reference', '1.01'), ('reference', 'nan'),
    ('reference', 'inf'), ('reference', '-inf'),
])
def test_explanation_tune_rejects_invalid_gain_without_changing_persisted_default(tmp_path, axis, setting):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_explanation_tuner(path)
    probes = {'secret_probe': {'chatty': True}}
    knob = 'probe_explain_' + axis
    output = run_tune(':tune ' + knob + ' ' + setting, tuner, probes)
    assert 'must be finite and between' in output
    assert 'Redirected' not in output
    resumed = registered_explanation_tuner(path)
    for name, default in probe_activation_explain.EXPLAIN_GAINS.items():
        assert tuner.get('probe_explain_' + name) == resumed.get('probe_explain_' + name) == default
    assert probes['secret_probe']['chatty'] is True
    assert resumed.get('probe_secret_probe') == .4
