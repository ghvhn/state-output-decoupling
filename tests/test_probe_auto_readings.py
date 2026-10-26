"""Execute the production input-display branch without loading a model."""
import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import traceback

from scripts import interactive_phenomenality as shell


TREE = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
DISPLAY = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
               and 'auto_probe_readings' in ast.unparse(node.test))


def display(probes, rows, enabled=True):
    namespace = dict(vars(shell), probes=probes, turn_log=rows,
                     config=SimpleNamespace(auto_probe_readings=enabled))
    # No agent has run: there is deliberately no backup _sys_probes binding.
    namespace.pop('_sys_probes', None)
    module = ast.Module(body=[copy.deepcopy(DISPLAY)], type_ignores=[])
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(ast.fix_missing_locations(module), shell.__file__, 'exec'), namespace)
    return output.getvalue()


def test_auto_readings_before_any_spawn_uses_current_probes():
    output = display({'current_probe': {}}, [
        {'probe_current_probe': -.2},
        {'probe_current_probe': .75, 'probe_other_agent': 1.0},
        {'sense': .4},
    ])
    assert '[Auto Probe Readings]' in output
    assert 'current_probe' in output and '+0.750' in output
    assert shell.probe_alias('current_probe') in output
    assert 'other_agent' not in output and '-0.200' not in output


def test_disabled_or_empty_probes_does_not_touch_history():
    class UnreadableHistory:
        def __iter__(self):
            raise AssertionError('History should not be read')
    assert display({'probe': {}}, UnreadableHistory(), enabled=False) == ''
    assert display({}, UnreadableHistory()) == ''


def test_missing_readings_returns_without_printing():
    assert display({'probe': {}}, []) == ''
    assert display({'probe': {}}, [{'sense': .3}]) == ''


def test_recovery_log_records_the_original_exception():
    event_try = next(node for node in ast.walk(TREE) if isinstance(node, ast.Try)
                     and any(isinstance(child, ast.Call) and child.args
                             and isinstance(child.args[0], ast.Constant)
                             and child.args[0].value == 'turn_error_recovered'
                             for child in ast.walk(ast.Module(body=node.body, type_ignores=[])))
                     and len(node.body) == 1)
    memory = SimpleNamespace(append_event=Mock())
    try:
        raise RuntimeError('Original input display failure')
    except RuntimeError as error:
        namespace = dict(memory=memory, _turn_exc=error, traceback=traceback)
        module = ast.Module(body=[copy.deepcopy(event_try)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), shell.__file__, 'exec'), namespace)
    memory.append_event.assert_called_once()
    fields = memory.append_event.call_args.kwargs
    assert fields['text'] == 'RuntimeError: Original input display failure'
    assert 'RuntimeError: Original input display failure' in fields['provenance']['traceback']
