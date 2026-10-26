"""Session continuation mode keeps settings and only selected conversation turns."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import continuation_investigation as continuation
from invariants.continuation_mode import ContinuationMode
from scripts import interactive_phenomenality as shell


def test_mode_starts_disabled_and_is_session_local():
    mode = ContinuationMode()
    assert not mode.enabled
    assert mode.options['tokens'] == 64
    assert mode.options['stride'] == 8
    assert mode.options['horizon'] == 12
    mode.configure(['on', 'tokens', '32'])
    assert mode.enabled and mode.options['tokens'] == 32
    assert not ContinuationMode().enabled
    assert ContinuationMode().options['tokens'] == 64


def test_settings_stay_across_on_off_and_status():
    mode = ContinuationMode()
    mode.configure(['on', 'tokens', '24', 'stride', '4', 'layers', '0', '1'], n_layers=3)
    before = deepcopy(mode.options)
    assert mode.configure(['status']) == mode.status()
    assert mode.configure([]) == mode.status()
    assert mode.options == before and mode.enabled
    mode.configure(['off'])
    assert not mode.enabled and mode.options == before
    mode.configure(['on', 'horizon', '16'])
    assert mode.enabled
    assert mode.options['tokens'] == 24 and mode.options['stride'] == 4
    assert mode.options['horizon'] == 16 and mode.options['layers'] == [0, 1]


@pytest.mark.parametrize('arguments', [
    ['on', 'tokens', '0'], ['on', 'stride', '-1'], ['on', 'horizon', '0'],
    ['on', 'helpers', '0'], ['on', 'context', '0'], ['on', 'alpha', 'nan'],
    ['on', 'alpha', '-.1'], ['on', 'cap', 'inf'], ['on', 'cap', '-.1'],
    ['on', 'tokens', '3.5'], ['on', 'decisions', '0'], ['on', 'candidate_tokens', '-1'],
    ['on', 'layers', '2', '1'], ['on', 'layers', '-1', '1'],
    ['on', 'layers', '0', '2'], ['on', 'layers', '0'], ['on', 'unknown', '1'],
    ['on', 'tokens'], ['off', 'tokens', '1'], ['status', 'tokens', '1'], ['maybe'],
])
def test_invalid_settings_are_atomic(arguments):
    mode = ContinuationMode()
    mode.configure(['on', 'tokens', '24', 'stride', '4'], n_layers=3)
    before = deepcopy(mode.status())
    with pytest.raises(ValueError):
        mode.configure(arguments, n_layers=3)
    assert mode.status() == before


def test_output_base_gives_each_turn_a_unique_directory_without_splitting_prompt(tmp_path):
    mode = ContinuationMode()
    base = tmp_path / 'branch output'
    mode.configure(['on', 'output', str(base), 'tokens', '16'])
    first = mode.arguments('A prompt with "quotes" and -- inside it.')
    second = mode.arguments('Another prompt')
    assert first[0] == second[0] == 'continue'
    assert first[-2:] == ['--', 'A prompt with "quotes" and -- inside it.']
    assert second[-2:] == ['--', 'Another prompt']
    first_output = Path(first[first.index('output') + 1])
    second_output = Path(second[second.index('output') + 1])
    assert first_output.parent == second_output.parent == base
    assert first_output != second_output
    assert not first_output.exists() and not second_output.exists()


@pytest.fixture
def shell_setup(monkeypatch):
    from invariants.synaptic_memory import SynapticMemoryEngine

    monkeypatch.setattr(SynapticMemoryEngine, '_INSTANCE', None)
    model = SimpleNamespace(n_layers=3, d_model=3,
        model=torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity()))
    probes = {
        'target': {'direction': {2: torch.tensor([1., 0., 0.])},
            'framings': ('Target present.', 'Target absent.'),
            'measurement_source': 'live_assistant_prediction'},
        'helper': {'direction': {0: torch.tensor([0., 1., 0.])},
            'framings': ('Helper present.', 'Helper absent.'),
            'measurement_source': 'live_assistant_prediction'},
    }
    controller = SimpleNamespace(routes={'target': {
        'direction': -1, 'controls': [{'name': 'helper', 'sign': 1}]}}, roles={},
        observe=Mock(), import_investigation=Mock())
    records = [
        {'role': 'user', 'text': 'PREVIOUS USER', 'kind': 'turn',
         'session_id': 'session-a', 'scope': 'scope-a'},
        {'role': 'assistant', 'text': 'PREVIOUS SELECTED', 'kind': 'turn',
         'session_id': 'session-a', 'scope': 'scope-a'},
        {'role': 'assistant', 'text': 'OPERATOR DRAFT EVENT', 'kind': 'event',
         'session_id': 'session-a', 'scope': 'scope-a'},
    ]
    memory = SimpleNamespace(session_id='session-a', scope='scope-a', records=records)

    def append_turn(role, text, **kwargs):
        record = {'role': role, 'text': text, 'kind': 'turn', 'session_id': memory.session_id,
                  'scope': memory.scope, **kwargs}
        records.append(record)
        return record

    memory.append_turn = Mock(side_effect=append_turn)
    memory.append_event = Mock()
    return model, probes, controller, memory


def enable_mode(setup, *arguments):
    model, probes, controller, _memory = setup
    return shell.handle_continuation_mode_command(
        ['on', *arguments], controller, probes, model)


def test_mode_command_requires_active_unguarded_existing_objectives(shell_setup):
    model, probes, controller, _memory = shell_setup
    controller.routes = {}
    with pytest.raises(ValueError, match='objective|Configure'):
        enable_mode(shell_setup)
    controller.routes = {'missing': {'direction': -1, 'controls': None}}
    with pytest.raises(ValueError):
        enable_mode(shell_setup)
    controller.routes = {'target': {'direction': -1, 'controls': None,
        'when': {'probe': 'helper', 'op': 'above', 'threshold': .2}}}
    with pytest.raises(ValueError, match='unguarded|conditional|guard'):
        enable_mode(shell_setup)
    assert not getattr(controller, '_continuation_mode', ContinuationMode()).enabled
    shell.handle_continuation_mode_command(['status'], controller, probes, model)
    shell.handle_continuation_mode_command(['off'], controller, probes, model)


def test_two_ordinary_turns_reuse_settings_and_only_selected_answers_enter_context(
        shell_setup, tmp_path, monkeypatch):
    model, probes, controller, memory = shell_setup
    enable_mode(shell_setup, 'tokens', '24', 'stride', '4', 'horizon', '8',
                'output', str(tmp_path / 'runs'))
    calls = []

    def generate(_model, _probes, objectives, **kwargs):
        calls.append({'objectives': deepcopy(objectives), **deepcopy(kwargs)})
        index = len(calls)
        return {'source': continuation.SOURCE, 'fingerprint': f'fingerprint-{index}',
            'response': f'SELECTED {index}', 'output_dir': str(kwargs['output_dir']),
            'telemetry': {'tokens_generated': index, 'generation_probe_readings': {}},
            'discarded_candidates': ['PRIVATE ALTERNATIVE']}

    monkeypatch.setattr(continuation, 'run_continuation_investigation', generate)
    context = []
    first = shell.run_enabled_continuation_turn('FIRST USER', controller, probes, model,
        memory, session_context=context)
    assert first['response'] == 'SELECTED 1'
    controller.routes['target']['direction'] = 1
    second = shell.run_enabled_continuation_turn('SECOND USER', controller, probes, model,
        memory, session_context=context)
    assert second['response'] == 'SELECTED 2'
    assert [call['objectives'] for call in calls] == [{'target': -1}, {'target': 1}]
    assert all(call['max_new_tokens'] == 24 and call['stride'] == 4
               and call['horizon'] == 8 for call in calls)
    assert calls[0]['output_dir'] != calls[1]['output_dir']
    assert calls[1]['prompt_messages'] == [
        {'role': 'user', 'content': 'PREVIOUS USER'},
        {'role': 'assistant', 'content': 'PREVIOUS SELECTED'},
        {'role': 'user', 'content': 'FIRST USER'},
        {'role': 'assistant', 'content': 'SELECTED 1'},
        {'role': 'user', 'content': 'SECOND USER'}]
    assert [(entry.args[0], entry.args[1]) for entry in memory.append_turn.call_args_list] == [
        ('user', 'FIRST USER'), ('assistant', 'SELECTED 1'),
        ('user', 'SECOND USER'), ('assistant', 'SELECTED 2')]
    assert [(entry[0], entry[2]) for entry in context] == [
        ('user', 'FIRST USER'), ('assistant', 'SELECTED 1'),
        ('user', 'SECOND USER'), ('assistant', 'SELECTED 2')]
    assert 'PRIVATE ALTERNATIVE' not in str(memory.records) + str(context) + str(calls)
    assert 'OPERATOR DRAFT EVENT' not in str(calls)
    assert controller._continuation_mode.enabled
    assert controller._continuation_mode.completed_turns == 2
    assert controller._continuation_mode.failed_turns == 0
    controller.observe.assert_not_called()
    controller.import_investigation.assert_not_called()
    memory.append_event.assert_not_called()


@pytest.mark.parametrize(('prompt', 'flags'), [
    (':relate indirect continue status', {}), (':unknown', {}), (' :tune', {}),
    ('', {}), ('   ', {}), ('Read this document', {'reading_turn': True}),
    ('Generate command', {'utility_turn': True}),
])
def test_commands_reading_and_utility_turns_bypass_mode(shell_setup, monkeypatch, prompt, flags):
    model, probes, controller, memory = shell_setup
    enable_mode(shell_setup)
    run = Mock(side_effect=AssertionError('No branching for command or utility turn'))
    monkeypatch.setattr(continuation, 'run_continuation_investigation', run)
    context = []
    assert shell.run_enabled_continuation_turn(prompt, controller, probes, model,
        memory, session_context=context, **flags) is None
    run.assert_not_called()
    memory.append_turn.assert_not_called()
    assert context == [] and controller._continuation_mode.completed_turns == 0


def test_disabled_mode_leaves_ordinary_generation_to_shell(shell_setup, monkeypatch):
    model, probes, controller, memory = shell_setup
    run = Mock(side_effect=AssertionError('Disabled mode must not generate'))
    monkeypatch.setattr(continuation, 'run_continuation_investigation', run)
    assert shell.run_enabled_continuation_turn('Normal prompt', controller, probes, model, memory) is None
    enable_mode(shell_setup)
    shell.handle_continuation_mode_command(['off'], controller, probes, model)
    assert shell.run_enabled_continuation_turn('Normal prompt', controller, probes, model, memory) is None
    run.assert_not_called()
    memory.append_turn.assert_not_called()


def test_selected_session_context_obeys_shell_history_bound(shell_setup, monkeypatch):
    model, probes, controller, memory = shell_setup
    enable_mode(shell_setup)
    monkeypatch.setattr(shell, 'MAX_SESSION_TURNS', 2)
    context = [('user' if index % 2 == 0 else 'assistant', 'participant',
                f'OLD {index}', 0.) for index in range(8)]
    selected = {'source': continuation.SOURCE, 'fingerprint': 'bounded-context',
        'response': 'SELECTED ANSWER', 'output_dir': 'output',
        'telemetry': {'tokens_generated': 2, 'generation_probe_readings': {}}}
    monkeypatch.setattr(continuation, 'run_continuation_investigation', lambda *_args, **_kwargs: selected)
    shell.run_enabled_continuation_turn('CURRENT PROMPT', controller, probes, model,
        memory, session_context=context)
    assert len(context) == 4
    assert [row[2] for row in context] == ['OLD 6', 'OLD 7', 'CURRENT PROMPT', 'SELECTED ANSWER']


def test_context_off_excludes_saved_history_but_saves_selected_turns(shell_setup, monkeypatch):
    model, probes, controller, memory = shell_setup
    enable_mode(shell_setup)
    calls = []

    def generate(*_args, **kwargs):
        calls.append(kwargs)
        return {'source': continuation.SOURCE, 'fingerprint': 'no-history',
            'response': 'SELECTED ANSWER', 'output_dir': str(kwargs['output_dir']),
            'telemetry': {'tokens_generated': 2, 'generation_probe_readings': {}}}

    monkeypatch.setattr(continuation, 'run_continuation_investigation', generate)
    shell.run_enabled_continuation_turn('CURRENT PROMPT', controller, probes, model,
        memory, history_enabled=False)
    assert calls[0]['prompt_messages'] == [{'role': 'user', 'content': 'CURRENT PROMPT'}]
    assert [(call.args[0], call.args[1]) for call in memory.append_turn.call_args_list] == [
        ('user', 'CURRENT PROMPT'), ('assistant', 'SELECTED ANSWER')]
    assert controller._continuation_mode.completed_turns == 1


def test_failure_restores_owned_hook_and_preserves_enabled_mode_and_context(
        shell_setup, monkeypatch):
    from invariants.synaptic_memory import SynapticMemoryEngine

    model, probes, controller, memory = shell_setup
    enable_mode(shell_setup)
    module = model.model[0]
    handle = module.register_forward_hook(lambda _module, _args, output: output)
    owner = SynapticMemoryEngine(model=None)
    owner.model = model
    bank = torch.tensor([.1, .2])
    owner.hooks = {0: SimpleNamespace(module=module, hook_handle=handle, bank=bank)}
    monkeypatch.setattr(SynapticMemoryEngine, '_INSTANCE', owner)
    original_hooks = list(module._forward_hooks.items())

    def fail(*_args, **_kwargs):
        assert not module._forward_hooks
        raise RuntimeError('synthetic branch failure')

    monkeypatch.setattr(continuation, 'run_continuation_investigation', fail)
    context = [('user', 'user', 'BEFORE', 0.)]
    before = deepcopy(memory.records)
    try:
        with owner.activate(), pytest.raises(RuntimeError, match='synthetic branch failure'):
            shell.run_enabled_continuation_turn('FAILED PROMPT', controller, probes, model,
                memory, session_context=context)
        assert list(module._forward_hooks.items()) == original_hooks
        assert owner.hooks[0].hook_handle is handle
        assert owner.hooks[0].bank is bank
        assert controller._continuation_mode.enabled
        assert controller._continuation_mode.failed_turns == 1
        assert controller._continuation_mode.completed_turns == 0
        assert memory.records == before
        assert context == [('user', 'user', 'BEFORE', 0.)]
        memory.append_turn.assert_not_called()
    finally:
        handle.remove()
