"""Definition task controls pass through the actual ordinary :tune branch."""
import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from invariants.probe_definition_task import DEFINITION_TASK_GAINS
from invariants.probe_definition_steps import STEP_GAINS
from invariants.trigger_tuner import TriggerTuner
from scripts import interactive_phenomenality as shell


TREE = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
BRANCH = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "user_input.startswith(':tune')")
COMMAND = ast.fix_missing_locations(ast.Module(body=[ast.For(
    target=ast.Name(id='_run', ctx=ast.Store()),
    iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
    body=[copy.deepcopy(BRANCH)], orelse=[])], type_ignores=[]))


def registered_tuner(path):
    tuner = TriggerTuner(path)
    for axis, default in DEFINITION_TASK_GAINS.items():
        tuner.register('probe_define_' + axis, default, kind='coefficient')
    tuner.register('probe_define_subtract', 0., kind='coefficient')
    tuner.register('probe_define_process', 1., kind='coefficient')
    for step, default in STEP_GAINS.items():
        tuner.register('probe_define_step_' + step, default, kind='coefficient')
    tuner.register('probe_secret_probe', .4, kind='threshold')
    return tuner


def run_tune(command, tuner, probes):
    namespace = dict(vars(shell), user_input=command, tuner=tuner, probes=probes,
                     model=None, config=SimpleNamespace(),
                     DEFINITION_TASK_GAINS=DEFINITION_TASK_GAINS, STEP_GAINS=STEP_GAINS)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(COMMAND, shell.__file__, 'exec'), namespace)
    return output.getvalue()


def test_ordinary_tune_persists_negative_and_zero_definition_controls(tmp_path):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_tuner(path)
    probes = {'secret_probe': {'chatty': True}}

    for command in (':tune probe_define_verbosity -0.08',
                    ':tune probe_define_naming 0',
                    ':tune probe_define_contrast off',
                    ':tune probe_define_question -0.12'):
        output = run_tune(command, tuner, probes)
        assert 'Redirected' not in output and 'No active probe' not in output

    assert tuner.get('probe_define_verbosity') == -.08
    assert tuner.get('probe_define_naming') == 0.
    assert tuner.get('probe_define_contrast') == 0.
    assert tuner.get('probe_define_question') == -.12
    assert probes['secret_probe']['chatty'] is True
    resumed = registered_tuner(path)
    assert resumed.get('probe_define_verbosity') == -.08
    assert resumed.get('probe_define_naming') == 0.
    assert resumed.get('probe_define_contrast') == 0.
    assert resumed.get('probe_define_question') == -.12
    assert resumed.triggers['probe_define_verbosity'].kind == 'coefficient'
    assert resumed.get('probe_secret_probe') == .4


def test_ordinary_tune_can_disable_and_reenable_question_control_without_muting(tmp_path):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_tuner(path)
    probes = {'secret_probe': {'chatty': True}}
    output = run_tune(':tune probe_define_question off', tuner, probes)
    assert 'Redirected' not in output
    assert registered_tuner(path).get('probe_define_question') == 0.
    output = run_tune(':tune probe_define_question 0.09', tuner, probes)
    assert 'Redirected' not in output
    assert registered_tuner(path).get('probe_define_question') == .09
    assert probes['secret_probe']['chatty'] is True


def test_ordinary_tune_subtraction_strength_persists_and_zero_disables_it(tmp_path):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_tuner(path)
    probes = {'secret_probe': {'chatty': True}}
    output = run_tune(':tune probe_define_subtract 0.6', tuner, probes)
    assert 'Redirected' not in output and 'No active probe' not in output
    assert registered_tuner(path).get('probe_define_subtract') == .6
    assert registered_tuner(path).triggers['probe_define_subtract'].kind == 'coefficient'
    output = run_tune(':tune probe_define_subtract 0', tuner, probes)
    assert 'Redirected' not in output
    assert registered_tuner(path).get('probe_define_subtract') == 0.
    assert probes['secret_probe']['chatty'] is True
    assert tuner.get('probe_secret_probe') == .4


def test_ordinary_tune_operation_control_is_zero_by_default_and_persists_signed_values(tmp_path):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_tuner(path)
    probes = {'secret_probe': {'chatty': True}}
    assert tuner.get('probe_define_operation') == 0.
    for setting, expected in (('0.1', .1), ('-0.1', -.1), ('off', 0.)):
        output = run_tune(':tune probe_define_operation ' + setting, tuner, probes)
        assert 'Redirected' not in output and 'No active probe' not in output
        assert registered_tuner(path).get('probe_define_operation') == expected
    assert probes['secret_probe']['chatty'] is True
    assert tuner.get('probe_secret_probe') == .4


def test_ordinary_tune_process_mode_and_step_gains_persist_without_muting(tmp_path):
    path = tmp_path / 'trigger_tuner.json'
    tuner = registered_tuner(path)
    probes = {'secret_probe': {'chatty': True}}
    assert tuner.get('probe_define_process') == 1.
    for step, default in STEP_GAINS.items():
        assert tuner.get('probe_define_step_' + step) == default

    for name, setting, expected in (
        ('probe_define_process', 'off', 0.),
        ('probe_define_process', '1', 1.),
        ('probe_define_process', '2', 2.),
        ('probe_define_step_compare', '-0.08', -.08),
        ('probe_define_step_isolate', '0', 0.),
        ('probe_define_step_distill', 'off', 0.),
        ('probe_define_step_verify', '0.06', .06),
    ):
        output = run_tune(':tune ' + name + ' ' + setting, tuner, probes)
        assert 'Redirected' not in output and 'No active probe' not in output
        resumed = registered_tuner(path)
        assert resumed.get(name) == expected
        assert resumed.triggers[name].kind == 'coefficient'
    assert probes['secret_probe']['chatty'] is True
    assert tuner.get('probe_secret_probe') == .4


@pytest.mark.parametrize('name', ['probe_secret_probe', 'secret_probe'])
def test_ordinary_tune_still_redirects_probe_off_to_console_mute(tmp_path, name):
    tuner = registered_tuner(tmp_path / 'trigger_tuner.json')
    probes = {'secret_probe': {'chatty': True}}
    output = run_tune(':tune ' + name + ' off', tuner, probes)

    assert 'muted console output' in output
    assert probes['secret_probe']['chatty'] is False
    assert tuner.get('probe_secret_probe') == .4
    assert 'secret_probe' not in tuner.triggers
    for axis, default in DEFINITION_TASK_GAINS.items():
        assert tuner.get('probe_define_' + axis) == default
