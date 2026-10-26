"""Exercise profile tracking and replay through the actual definition command."""
import ast
import contextlib
import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import engine, probe_behavior
from invariants.probe_behavior import DEFINITION_MAX_TOKENS, definition_key
from invariants.probe_definition_output import build_definition_request
from scripts import interactive_phenomenality as shell
from test_probe_definition_output_shell import (
    DEFINITION, PREFIX, RECAP, NativeTokenizer, OutputHarness,
)


class RecordingSession:
    """Expose callbacks while production hooks and output checks remain real."""

    def __init__(self, harness, *, replay=False):
        self.harness = harness
        self.replay = replay
        self.events = []
        self.requests = []
        self.captures = []
        self.observed = []
        self.mode = None
        self.history_marker = 'PREVIOUS PROFILE RESPONSE MUST STAY OPERATOR ONLY'

    def request(self, evidence, feedback, attempt_index):
        assert evidence == self.harness.evidence
        self.mode = 'activation_replay' if self.replay and attempt_index == 0 else 'explicit_recipe'
        request = evidence if self.mode == 'activation_replay' else build_definition_request(evidence, feedback)
        self.requests.append((attempt_index, request, feedback, self.mode))
        self.events.append('request')
        return request

    def capture(self, request):
        # Captures must see the unmodified baseline, never temporary subject or
        # defining hooks from this generation.
        assert [set(layer._forward_hooks) for layer in self.harness.layers] == self.harness.original_hooks
        assert self.harness.config.lookahead_processor is None
        assert request == self.requests[-1][1]
        self.captures.append(request)
        self.events.append('capture')

    def steering(self):
        self.events.append('steering')
        if self.mode != 'activation_replay':
            return None
        return {'layer': 5, 'direction': torch.tensor([0., 1., 0.]), 'gain': .08,
                'diagnostics': {'candidate': True}}

    def observe(self, record):
        # Learning gets the final structural and truncation gate, after hooks
        # have already been removed. Generated prose is not a measured reply.
        assert [set(layer._forward_hooks) for layer in self.harness.layers] == self.harness.original_hooks
        self.observed.append(copy.deepcopy(record))
        self.events.append('observe')

    def needs_explicit_fallback(self, attempt):
        assert self.observed and attempt == self.observed[-1]
        return bool(self.observed and self.observed[-1]['status'] == 'incomplete'
                    and self.requests[-1][3] == 'activation_replay')

    def metadata(self):
        return {'version': 'recording-session', 'attempts': copy.deepcopy(self.observed),
                'semantic_truth_checked': False, 'causal_effect_checked': False,
                'operator_history': self.history_marker}


class ProfileHarness(OutputHarness):
    def __init__(self, monkeypatch, tmp_path, *, replay=False, token_counts=None, **kwargs):
        super().__init__(monkeypatch, tmp_path, **kwargs)
        self.token_counts = token_counts
        self.model.model_name = 'local-test-model'
        self.session = RecordingSession(self, replay=replay)
        self.profile = SimpleNamespace(session=Mock(return_value=self.session))
        self.matrix.defining_profile = self.profile
        original_builder = probe_behavior.build_definition_handles

        def record_builder(*args, **kwargs):
            self.session.events.append('hooks')
            return original_builder(*args, **kwargs)

        self.handles = Mock(side_effect=record_builder)
        monkeypatch.setattr(probe_behavior, 'build_definition_handles', self.handles)

    def generate(self, model, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        self.session.events.append('generate')
        assert model is self.model
        assert kwargs['vecs'] == {} and kwargs['return_telemetry'] is True
        assert kwargs['pre_formatted'] is True and kwargs['chatty_log'] is False
        assert kwargs['max_new_tokens'] == DEFINITION_MAX_TOKENS
        assert not self.config.cache_enabled and not self.config.cache_write_enabled
        assert self.config.lookahead_processor is None
        for name in ('max_routing_events', 'max_synthesis_events', 'max_synthesis_steps', 'max_loops'):
            assert getattr(self.config, name) == 0
        request = self.model.tok.messages[-1][0]['content']
        assert kwargs['instruction'] == NativeTokenizer.header(request) + PREFIX
        assert request == self.session.requests[-1][1]
        assert request.count(self.evidence) == 1
        for forbidden in ('Secret positive seed', 'Secret negative seed', 'secret_probe',
                          self.session.history_marker, RECAP):
            assert forbidden not in request
        self.naming.assert_not_called()
        for layer, module in enumerate(self.layers):
            assert len(module._forward_hooks) == len(self.original_hooks[layer]) + (layer in (4, 5))
        self.root(torch.ones(1, 5, 3))
        self.root(torch.ones(1, 1, 3))
        self.root(torch.ones(1, 7, 3))
        if self.fail:
            raise RuntimeError('definition generation failed')
        suffix = self.suffixes[min(index, len(self.suffixes) - 1)]
        tokens = self.tokens if self.token_counts is None else self.token_counts[min(index, len(self.token_counts) - 1)]
        return (suffix, {'tokens_generated': tokens}) if self.telemetry else suffix


def test_profile_receives_exact_runtime_recipe_and_final_output_gate(monkeypatch, tmp_path):
    harness = ProfileHarness(monkeypatch, tmp_path)
    output = harness.run()
    call = harness.profile.session.call_args
    assert call.args == (harness.model, (4, 5))
    assert call.kwargs == {
        'evidence': harness.evidence,
        'namespace': definition_key('secret_probe', harness.probe),
        'settings': {
            'assistant_prefix': PREFIX, 'subject_fraction': .02,
            'max_tokens': DEFINITION_MAX_TOKENS, 'decoding': 'greedy',
            'cap_fraction': engine.get_steer_cap_fraction(),
            'config_overrides': {'cache_enabled': False, 'cache_write_enabled': False,
                                 'max_routing_events': 0, 'max_synthesis_events': 0,
                                 'max_synthesis_steps': 0, 'max_loops': 0,
                                 'lookahead_processor': None},
        },
    }
    assert harness.session.events == ['request', 'capture', 'steering', 'hooks', 'generate', 'observe']
    assert harness.session.captures == [harness.session.requests[0][1]]
    assert len(harness.session.observed) == 1
    record = harness.session.observed[0]
    assert record == harness.attempt()['controls']['attempts'][0]
    assert record['status'] == 'hypothesis' and record['qualification']['accepted']
    assert record['qualification']['semantic_truth_checked'] is False
    assert record['request_sha256'] == hashlib.sha256(harness.session.captures[0].encode()).hexdigest()
    assert harness.attempt()['controls']['defining_profile']['attempts'] == harness.session.observed
    assert 'Definition hypothesis' in output


def test_learned_replay_uses_last_layer_original_decision_and_one_shared_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .025)
    harness = ProfileHarness(monkeypatch, tmp_path, replay=True)
    harness.run()
    assert harness.session.requests[0][1] == harness.evidence
    assert harness.handles.call_args.kwargs == {
        'task_gains': {'defining_profile': .08},
        'task_positions': {'defining_profile': 'decision'},
        'task_layer_scales': {'defining_profile': {4: 0., 5: 1.}},
    }
    initial, decode, rebuild = harness.root.pulses
    for layer in range(6):
        torch.testing.assert_close(decode[layer], torch.zeros_like(decode[layer]))
        torch.testing.assert_close(initial[layer][:, :4], torch.zeros_like(initial[layer][:, :4]))
        torch.testing.assert_close(rebuild[layer][:, :4], torch.zeros_like(rebuild[layer][:, :4]))
        torch.testing.assert_close(rebuild[layer][:, 5:], torch.zeros_like(rebuild[layer][:, 5:]))
        if layer not in (4, 5):
            torch.testing.assert_close(initial[layer], torch.zeros_like(initial[layer]))
            torch.testing.assert_close(rebuild[layer], torch.zeros_like(rebuild[layer]))
    assert torch.count_nonzero(initial[4][:, 4]) == 1
    assert initial[4][0, 4, 1] == 0
    # The subject and profile are capped together at layer 5. If two hooks
    # independently capped them this norm would exceed the .025 envelope.
    before_last = torch.ones(3) + initial[4][0, 4]
    push = initial[5][0, 4]
    torch.testing.assert_close(push.norm(), before_last.norm() * .025)
    torch.testing.assert_close(push[1], push[0] * 4)
    assert push[2] == 0


def test_failed_activation_replay_labels_failure_and_retries_saved_explicit_recipe(monkeypatch, tmp_path):
    harness = ProfileHarness(monkeypatch, tmp_path, replay=True, suffixes=(RECAP, DEFINITION))
    output = harness.run()
    assert len(harness.calls) == len(harness.session.observed) == 2
    first, second = harness.session.observed
    assert first['status'] == 'unqualified' and not first['qualification']['accepted']
    assert second['status'] == 'hypothesis' and second['qualification']['accepted']
    assert [request[3] for request in harness.session.requests] == ['activation_replay', 'explicit_recipe']
    corrected = harness.session.requests[1][1]
    assert '[Definition request]' in corrected and '[Format correction]' in corrected
    assert RECAP not in corrected and RECAP not in output
    assert harness.session.requests[1][2] == first['qualification']['reason']
    assert harness.handles.call_args_list[1].kwargs == {'task_gains': {}}
    assert harness.attempt()['text'] == PREFIX + DEFINITION


@pytest.mark.parametrize('final_tokens,final_status', [(30, 'hypothesis'), (DEFINITION_MAX_TOKENS, 'incomplete')])
def test_incomplete_activation_replay_gets_one_explicit_fallback_and_cannot_qualify_truncated_text(monkeypatch, tmp_path, final_tokens, final_status):
    harness = ProfileHarness(monkeypatch, tmp_path, replay=True,
                             token_counts=(DEFINITION_MAX_TOKENS, final_tokens))
    output = harness.run()
    assert len(harness.calls) == len(harness.session.observed) == 2
    first, second = harness.session.observed
    assert first['status'] == 'incomplete' and not first['qualification']['accepted']
    assert first['tokens_generated'] == DEFINITION_MAX_TOKENS
    assert second['status'] == final_status
    assert second['qualification']['accepted'] is (final_status == 'hypothesis')
    assert [request[3] for request in harness.session.requests] == ['activation_replay', 'explicit_recipe']
    assert harness.session.requests[1][2] == first['qualification']['reason']
    assert '[Definition request]' in harness.session.requests[1][1]
    assert harness.handles.call_args_list[1].kwargs == {'task_gains': {}}
    assert harness.attempt()['qualified'] is (final_status == 'hypothesis')
    assert ('Definition hypothesis' in output) is (final_status == 'hypothesis')
    if final_status == 'incomplete':
        assert 'No definition produced' in output and PREFIX + DEFINITION not in output


@pytest.mark.parametrize('tokens,status,accepted', [
    (DEFINITION_MAX_TOKENS, 'incomplete', False), (30, 'hypothesis', True),
])
def test_profile_observes_actual_completion_and_permits_mistaken_meaning(monkeypatch, tmp_path, tokens, status, accepted):
    mistaken = ('a policy of obedience to authority: lower readings resist commands '
                'whereas higher readings follow them without hesitation.')
    harness = ProfileHarness(monkeypatch, tmp_path, tokens=tokens, suffixes=(mistaken,))
    harness.run()
    assert len(harness.session.observed) == 1
    record = harness.session.observed[0]
    assert record['status'] == status and record['qualification']['accepted'] is accepted
    assert record['qualification']['semantic_truth_checked'] is False


@pytest.mark.parametrize('failure', ['generation', 'missing_telemetry'])
def test_profile_never_learns_a_failed_generation_and_shell_restores_state(monkeypatch, tmp_path, failure):
    harness = ProfileHarness(monkeypatch, tmp_path, replay=True,
                             fail=failure == 'generation', telemetry=failure != 'missing_telemetry')
    with pytest.raises((RuntimeError, ValueError)):
        harness.run()
    assert len(harness.session.captures) == 1
    assert not harness.session.observed
    assert harness.behavior.interpretations == {}


def test_profile_histories_are_operator_metadata_and_not_measured_evidence(monkeypatch, tmp_path):
    harness = ProfileHarness(monkeypatch, tmp_path, suffixes=(RECAP, DEFINITION))
    harness.run()
    metadata = harness.attempt()['controls']['defining_profile']
    assert metadata['operator_history'] == harness.session.history_marker
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == harness.evidence
    for message in harness.model.tok.messages:
        assert harness.session.history_marker not in message[0]['content']
        assert RECAP not in message[0]['content']
    harness.behavior_observe.assert_not_called()
    harness.language_observe.assert_not_called()


def test_profile_fallback_path_is_tuner_owned_and_reused(monkeypatch, tmp_path):
    from invariants import defining_profile

    constructor = Mock(side_effect=lambda path=None: SimpleNamespace(path=path))
    monkeypatch.setattr(defining_profile, 'DefiningProfile', constructor)
    main = SimpleNamespace(path=tmp_path / 'main.plasticity.json')
    agent = SimpleNamespace(path=tmp_path / 'agent.plasticity.json')
    main_profile = shell.defining_profile_for(main)
    agent_profile = shell.defining_profile_for(agent)
    assert main_profile.path == tmp_path / 'main.plasticity.defining.pt'
    assert agent_profile.path == tmp_path / 'agent.plasticity.defining.pt'
    assert shell.defining_profile_for(main) is main_profile
    assert constructor.call_count == 2
    behavior_only = SimpleNamespace(probe_behavior=SimpleNamespace(path=tmp_path / 'observations.json'))
    assert shell.defining_profile_for(behavior_only).path == tmp_path / 'observations.defining.pt'


def test_corrupt_profile_preserves_file_and_ordinary_definition(monkeypatch, tmp_path):
    from invariants import defining_profile

    harness = ProfileHarness(monkeypatch, tmp_path)
    del harness.matrix.defining_profile
    expected_path = Path(harness.behavior.path).with_suffix('.defining.pt')
    expected_path.write_bytes(b'unreadable prior profile must be preserved')

    def constructor(path=None):
        if path is not None:
            assert path == expected_path
            raise ValueError('unreadable profile')
        return harness.profile

    loader = Mock(side_effect=constructor)
    monkeypatch.setattr(defining_profile, 'DefiningProfile', loader)
    output = harness.run()
    assert loader.call_count == 2
    assert expected_path.read_bytes() == b'unreadable prior profile must be preserved'
    assert 'Could not load defining profile' in output
    assert 'Definition hypothesis' in output and harness.attempt()['qualified']


@pytest.mark.parametrize('suffix,accepted', [(DEFINITION, True), (RECAP, False)])
def test_actual_profile_persists_settings_and_only_qualified_pre_hook_changes(monkeypatch, tmp_path, suffix, accepted):
    from invariants import defining_profile

    harness = OutputHarness(monkeypatch, tmp_path, suffixes=(suffix,))
    identity = {
        'version': defining_profile.PROFILE_VERSION,
        'recipe_sha256': defining_profile._digest(defining_profile._BASE_SUFFIX),
        'model': {'name': 'native-fixture'}, 'tokenizer': {'name': 'native-fixture'},
        'effective_load': {}, 'width': 3, 'model_layers': 6, 'layers': [4, 5],
        'libraries': {}, 'output_version': defining_profile.OUTPUT_VERSION,
        'output_prefix': PREFIX, 'capture_phase': defining_profile.CAPTURE_PHASE,
    }
    monkeypatch.setattr(defining_profile, '_profile_identity', lambda _model, _layers: identity)
    captures = []

    def capture(model, prompt, layers, width):
        assert model is harness.model and tuple(layers) == (4, 5) and width == 3
        assert [set(layer._forward_hooks) for layer in harness.layers] == harness.original_hooks
        captures.append(prompt)
        if len(captures) == 1:
            return {4: torch.tensor([1., 0., 0.]), 5: torch.tensor([0., 1., 0.])}
        return {4: torch.tensor([1., .1, 0.]), 5: torch.tensor([0., 1., .2])}

    monkeypatch.setattr(defining_profile, '_capture_states', capture)
    harness.run()
    stored = defining_profile.DefiningProfile(Path(harness.behavior.path).with_suffix('.defining.pt'))
    assert len(stored.families) == 1
    family = next(iter(stored.families.values()))
    assert len(family['history']) == len(harness.calls)
    assert all(item['accepted'] is accepted for item in family['history'])
    assert all(item['settings']['subject_fraction'] == .02 for item in family['history'])
    assert all(item['settings']['assistant_prefix'] == PREFIX for item in family['history'])
    assert all('text' not in item for item in family['history'])
    assert len(family['samples']) == int(accepted)
    assert captures[0] == NativeTokenizer.header(harness.evidence) + PREFIX
    assert '[Definition request]' in captures[1]
    if accepted:
        sample = family['samples'][0]
        torch.testing.assert_close(sample['sources'][5], torch.tensor([0., 1., 0.]))
        torch.testing.assert_close(sample['changes'][5], torch.tensor([0., 0., .2]))
        assert sample['semantic_truth_checked'] is False and sample['causal_effect_checked'] is False


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('preexisting', [False, True])
def test_lookahead_is_suspended_without_consuming_predictions_and_exactly_restored(monkeypatch, tmp_path, fail, preexisting):
    harness = ProfileHarness(monkeypatch, tmp_path, fail=fail)
    predictor = Mock(side_effect=AssertionError('Definition output must not consume conversational predictions'))
    if preexisting:
        harness.config.lookahead_processor = predictor
        harness.config_before = dict(vars(harness.config))
    if fail:
        with pytest.raises(RuntimeError, match='definition generation failed'):
            harness.run()
    else:
        harness.run()
    predictor.assert_not_called()
    if preexisting:
        assert harness.config.lookahead_processor is predictor
    else:
        assert not hasattr(harness.config, 'lookahead_processor')


def run_profile_control(command, profile):
    tree = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "pargs.lower() == 'defining' or pargs.lower().startswith('defining ')")
    statement = ast.fix_missing_locations(ast.Module(body=[ast.For(
        target=ast.Name(id='_run', ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
        body=[copy.deepcopy(branch)], orelse=[])], type_ignores=[]))
    matrix = SimpleNamespace(defining_profile=profile)
    owners = SimpleNamespace(for_tuner=Mock(return_value=(matrix, {})))
    generator = Mock(side_effect=AssertionError('Profile inspection must not generate model output'))
    namespace = dict(vars(shell), plasticity_owners=owners, tuner=object(),
                     pargs=command, _engine_generate=generator)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(statement, shell.__file__, 'exec'), namespace)
    generator.assert_not_called()
    assert matrix.defining_profile is profile
    return output.getvalue()


@pytest.mark.parametrize('command,initial,expected,saves', [
    ('defining', True, True, 0), ('defining show', False, False, 0),
    ('defining on', False, True, 1), ('defining off', True, False, 1),
])
def test_defining_controls_inspect_and_toggle_the_same_owner_profile(command, initial, expected, saves):
    profile = SimpleNamespace(enabled=initial, save=Mock(), summary=Mock(return_value='profile status'))
    output = run_profile_control(command, profile)
    assert profile.enabled is expected
    assert profile.save.call_count == saves
    profile.summary.assert_called_once_with()
    assert 'profile status' in output


@pytest.mark.parametrize('error', [OSError('read only profile'), RuntimeError('profile storage unavailable'),
                                 ValueError('invalid profile storage')])
def test_profile_control_save_failure_keeps_session_toggle_and_displays_summary(error):
    profile = SimpleNamespace(enabled=True, save=Mock(side_effect=error),
                              summary=Mock(return_value='profile status'))
    output = run_profile_control('defining off', profile)
    assert profile.enabled is False
    assert 'could not be saved' in output and 'profile status' in output
    profile.summary.assert_called_once_with()


def test_profile_save_failure_preserves_an_ordinary_qualified_definition(monkeypatch, tmp_path):
    from invariants import defining_profile

    harness = OutputHarness(monkeypatch, tmp_path)
    saved_profile = defining_profile.DefiningProfile(Path(harness.behavior.path).with_suffix('.defining.pt'))
    saved_profile.save()
    bytes_before = saved_profile.path.read_bytes()
    harness.matrix.defining_profile = saved_profile
    failed_save = Mock(side_effect=OSError('profile write denied'))
    monkeypatch.setattr(defining_profile.DefiningProfile, 'save', failed_save)
    output = harness.run()
    failed_save.assert_called_once_with()
    assert harness.attempt()['qualified'] and 'Definition hypothesis' in output
    assert 'Defining profile could not be saved' in output and 'in this session' in output
    metadata = harness.attempt()['controls']['defining_profile']
    assert 'profile write denied' in metadata['persistence_error']
    assert len(harness.matrix.defining_profile.unidentified_history) == 1
    assert saved_profile.path.read_bytes() == bytes_before
