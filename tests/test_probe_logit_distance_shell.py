"""Logit-distance commands stay observational through the actual shell branch."""

import ast
import copy
import importlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from invariants import activation_explanation_output, probe_activation_explain
from invariants.exposure_language import probe_alias
from scripts import interactive_phenomenality as shell
from test_probe_definition_output_shell import OutputHarness
from test_probe_definition_tuning import registered_tuner, run_tune


KNOB = 'probe_explain_logits_epsilon'
RENDERED = 'Measured local output-distribution distance; semantic meaning unverified.'


def logits_harness(monkeypatch, tmp_path, command='explain secret_probe logits', *,
                   epsilon=.01, failure=False):
    module = importlib.import_module('invariants.probe_logit_distance')
    harness = OutputHarness(monkeypatch, tmp_path, command=command)
    harness.tuner.get.side_effect = lambda name, default=None: epsilon if name == KNOB else default
    for owner, name in (
        (harness.behavior, 'evidence'), (harness.behavior, 'definition_memories'),
        (harness.behavior, 'record_interpretation'),
        (harness.language, 'record_namespace_summary'),
    ):
        monkeypatch.setattr(owner, name, Mock(side_effect=AssertionError(f'Logits must not call {name}')))
    for name in ('defining_profile_for', 'load_prompt', '_engine_generate', 'generate_agentic_text'):
        harness.namespace[name] = Mock(side_effect=AssertionError(f'Logits must not call {name}'))
    monkeypatch.setattr(probe_activation_explain, 'generate_activation_explanation',
                        Mock(side_effect=AssertionError('Logits must not generate an explanation')))
    harness.report = {
        'status': 'diagnostic', 'epsilon': float(epsilon) if not isinstance(epsilon, str) else epsilon,
        'evidence_sha256': 'empty-runtime-body', 'layers': [5],
        'semantic_truth_checked': False,
        'distance': {'jensen_shannon': .002, 'total_variation': .01},
    }
    harness.subject_before = {layer: vector.clone() for layer, vector in harness.probe['direction'].items()}

    def measure(model, subject, evidence='', **kwargs):
        assert model is harness.model
        assert subject is harness.probe['direction']
        assert evidence == ''
        assert kwargs == {'epsilon': epsilon}
        # Only the selected numeric direction reaches the measurement API.
        assert set(subject) == {4, 5} and all(torch.is_tensor(value) for value in subject.values())
        if failure:
            raise RuntimeError('logit forward failed')
        return copy.deepcopy(harness.report)

    harness.measure = Mock(side_effect=measure)
    harness.render = Mock(return_value=RENDERED)
    monkeypatch.setattr(module, 'measure_probe_logit_distance', harness.measure)
    monkeypatch.setattr(module, 'render_probe_logit_distance', harness.render)
    return harness


def trace_rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / 'behavior.activation-explain.jsonl')
            .read_text(encoding='utf-8').splitlines()]


def assert_observational(harness):
    assert harness.calls == []
    harness.handles.assert_not_called()
    harness.mint.assert_not_called()
    harness.naming.assert_not_called()
    for name in ('defining_profile_for', 'load_prompt', '_engine_generate', 'generate_agentic_text'):
        harness.namespace[name].assert_not_called()
    for owner, name in (
        (harness.behavior, 'evidence'), (harness.behavior, 'definition_memories'),
        (harness.behavior, 'record_interpretation'),
        (harness.language, 'record_namespace_summary'),
    ):
        getattr(owner, name).assert_not_called()
    probe_activation_explain.generate_activation_explanation.assert_not_called()
    harness.matrix.other_reference.activate.assert_not_called()
    assert harness.behavior.interpretations == {}
    assert not hasattr(harness.matrix, 'activation_explanations')
    assert not hasattr(harness.matrix, 'activation_explanation_axes')
    for layer, before in harness.subject_before.items():
        torch.testing.assert_close(harness.probe['direction'][layer], before)


@pytest.mark.parametrize('command', [
    'explain secret_probe logits', 'secret_probe explain logits',
    '[secret_probe] explain logits', 'explain probe:secret_probe logits',
    f'[{probe_alias("secret_probe")}] explain logits',
    f'explain [{probe_alias("secret_probe")}] logits',
])
def test_logits_measure_exact_blank_context_without_generating_or_reading_seed_text(monkeypatch, tmp_path, command):
    harness = logits_harness(monkeypatch, tmp_path, command)
    harness.probe['exposed'] = False
    output = harness.run()
    harness.measure.assert_called_once()
    harness.render.assert_called_once_with(harness.report)
    assert RENDERED in output and 'Operator trace:' in output
    assert_observational(harness)
    rows = trace_rows(tmp_path)
    assert len(rows) == 1 and rows[0]['timestamp']
    assert {key: rows[0][key] for key in ('probe', 'mode', 'status', 'report', 'semantic_truth_checked')} == {
        'probe': 'secret_probe', 'mode': 'logits', 'status': 'diagnostic',
        'report': harness.report, 'semantic_truth_checked': False,
    }


def test_logits_need_no_measured_examples_and_use_the_configured_epsilon(monkeypatch, tmp_path):
    harness = logits_harness(monkeypatch, tmp_path, epsilon=.037)
    harness.behavior.records.clear()
    harness.records_before = {}
    output = harness.run()
    harness.measure.assert_called_once_with(harness.model, harness.probe['direction'], '', epsilon=.037)
    assert RENDERED in output and 'at least four' not in output
    assert_observational(harness)


@pytest.mark.parametrize('target', ['choose', 'auto', 'CHOICE', '[choose]', 'probe:auto', '?choose'])
def test_logits_reject_generated_target_selection_before_resolution(monkeypatch, tmp_path, target):
    harness = logits_harness(monkeypatch, tmp_path, command='explain ' + target + ' logits')
    resolver = Mock(side_effect=AssertionError('Logits must not generate a target choice'))
    harness.namespace['resolve_probe_definition_target'] = resolver
    output = harness.run()
    assert 'Select a named probe or displayed ID' in output
    resolver.assert_not_called()
    harness.measure.assert_not_called()
    harness.render.assert_not_called()
    harness.owners.for_tuner.assert_not_called()
    assert not (tmp_path / 'behavior.activation-explain.jsonl').exists()
    assert_observational(harness)


def test_existing_probe_named_choose_remains_a_literal_target(monkeypatch, tmp_path):
    harness = logits_harness(monkeypatch, tmp_path, command='explain choose logits')
    harness.probes['choose'] = harness.probe
    output = harness.run()
    harness.measure.assert_called_once()
    assert RENDERED in output and trace_rows(tmp_path)[0]['probe'] == 'choose'
    assert_observational(harness)


def test_failed_logit_measurement_preserves_existing_hooks_and_creates_no_trace(monkeypatch, tmp_path):
    harness = logits_harness(monkeypatch, tmp_path, failure=True)
    existing = harness.layers[5].register_forward_hook(lambda _module, _inputs, output: output)
    try:
        output = harness.run()
        assert existing.id in harness.layers[5]._forward_hooks
    finally:
        existing.remove()
    harness.measure.assert_called_once()
    harness.render.assert_not_called()
    assert 'Diagnostic failed: logit forward failed' in output
    assert not (tmp_path / 'behavior.activation-explain.jsonl').exists()
    assert_observational(harness)


def test_failed_rendering_creates_no_operator_or_behavior_record(monkeypatch, tmp_path):
    harness = logits_harness(monkeypatch, tmp_path)
    harness.render.side_effect = ValueError('invalid diagnostic report')
    output = harness.run()
    harness.measure.assert_called_once()
    harness.render.assert_called_once_with(harness.report)
    assert 'Diagnostic failed: invalid diagnostic report' in output
    harness.owners.for_tuner.assert_not_called()
    assert not (tmp_path / 'behavior.activation-explain.jsonl').exists()
    assert_observational(harness)


def test_trace_write_failure_preserves_rendered_measurement_without_behavior_history(monkeypatch, tmp_path):
    harness = logits_harness(monkeypatch, tmp_path)
    record = Mock(side_effect=OSError('operator trace unavailable'))
    monkeypatch.setattr(activation_explanation_output, 'record_activation_explanation', record)
    output = harness.run()
    assert RENDERED in output and 'Could not save operator trace: operator trace unavailable' in output
    assert 'Operator trace:' not in output
    record.assert_called_once_with(harness.matrix, {
        'probe': 'secret_probe', 'mode': 'logits', 'status': 'diagnostic',
        'report': harness.report, 'semantic_truth_checked': False,
    })
    assert_observational(harness)


@pytest.mark.parametrize('epsilon', [0., -.01, .100001, float('nan'), float('inf'), -float('inf')])
def test_invalid_runtime_epsilon_stops_before_measurement_and_trace(monkeypatch, tmp_path, epsilon):
    harness = logits_harness(monkeypatch, tmp_path, epsilon=epsilon)
    output = harness.run()
    assert KNOB in output and 'must be' in output
    harness.measure.assert_not_called()
    harness.render.assert_not_called()
    assert not (tmp_path / 'behavior.activation-explain.jsonl').exists()
    assert_observational(harness)


@pytest.mark.parametrize('raw, expected', [
    ('secret_probe explain logits', 'explain secret_probe logits'),
    ('EXPLAIN [secret_probe] logits', 'explain [secret_probe] logits'),
])
def test_logit_command_normalization_preserves_the_target(raw, expected):
    assert shell.normalize_probe_definition_args(raw) == expected


def test_logit_mode_is_explicit_and_does_not_replace_default_activation_mode():
    assert shell.parse_probe_explain_mode(['logits']) == ('logits', None)
    assert shell.parse_probe_explain_mode(['LOGITS']) == ('logits', None)
    assert shell.parse_probe_explain_mode([]) == ('activation', None)


@pytest.mark.parametrize('tokens', [
    ['logits', 'activation'], ['logits', 'evidence'], ['logits', 'prompt'],
    ['logits', 'ping'], ['logits', '.01'], ['logits', 'choose'], ['logits', 'nan'],
])
def test_logit_mode_rejects_mixed_modes_and_ping_arguments(tokens):
    with pytest.raises(ValueError):
        shell.parse_probe_explain_mode(tokens)


def registered_logit_tuner(path):
    tuner = registered_tuner(path)
    tuner.register(KNOB, .01, kind='coefficient')
    return tuner


def test_logit_epsilon_registration_has_expected_default_and_calibration_policy():
    tree = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
    registration = next(node for node in ast.walk(tree) if isinstance(node, ast.Expr)
                        and isinstance(node.value, ast.Call)
                        and ast.unparse(node.value.func) == 'tuner.register'
                        and node.value.args and isinstance(node.value.args[0], ast.Constant)
                        and node.value.args[0].value == KNOB)
    tuner = Mock()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(registration)], type_ignores=[])),
                 shell.__file__, 'exec'), dict(vars(shell), tuner=tuner))
    tuner.register.assert_called_once_with(KNOB, .01, kind='coefficient')
    assert KNOB in shell.CALIBRATION_CIRCULAR
    assert KNOB not in shell.OUTCOME_CALIBRATABLE
    assert shell.calibration_policy(KNOB)[0] == 'reject'


@pytest.mark.parametrize('setting, expected', [('0.000001', .000001), ('0.01', .01), ('0.1', .1)])
def test_logit_epsilon_persists_without_probe_redirect_or_muting(tmp_path, setting, expected):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_logit_tuner(path)
    probes = {'explain_logits_epsilon': {'chatty': True}, 'secret_probe': {'chatty': True}}
    output = run_tune(':tune ' + KNOB + ' ' + setting, tuner, probes)
    resumed = registered_logit_tuner(path)
    assert 'Redirected' not in output and 'is a probe' not in output
    assert tuner.get(KNOB) == resumed.get(KNOB) == expected
    assert resumed.triggers[KNOB].kind == 'coefficient'
    assert resumed.get('probe_secret_probe') == .4
    assert all(probe['chatty'] for probe in probes.values())


@pytest.mark.parametrize('setting', ['0', 'off', '-0.01', '0.100001', 'nan', 'inf', '-inf'])
def test_invalid_logit_epsilon_preserves_persisted_value_and_probe_state(tmp_path, setting):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_logit_tuner(path)
    run_tune(':tune ' + KNOB + ' 0.035', tuner, {})
    probes = {'explain_logits_epsilon': {'chatty': True}}
    output = run_tune(':tune ' + KNOB + ' ' + setting, tuner, probes)
    assert KNOB in output and 'must be' in output and 'Redirected' not in output
    assert tuner.get(KNOB) == registered_logit_tuner(path).get(KNOB) == .035
    assert probes['explain_logits_epsilon']['chatty'] is True
