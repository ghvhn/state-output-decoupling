"""Verify the normal definition command against its actual shell branch.

The continuation is controlled so these tests can distinguish the requested
output kind from its inferred meaning. Real native prefilling and CPU hooks
exercise the command's prompt, cleanup, and observation boundaries.
"""
import ast
import contextlib
import copy
import io
from pathlib import Path
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import probe_behavior, probe_definition_task
from invariants.exposure_language import ExposureLanguage, probe_alias
from invariants.probe_behavior import DEFINITION_MAX_TOKENS, ProbeBehavior, definition_key
from scripts import interactive_phenomenality as shell


PREFIX = 'This probe measures '
DEFINITION = (
    'how firmly an answer commits to its claims: lower readings hedge '
    'conclusions whereas higher readings state them categorically.'
)
RECAP = (
    'a collection of conversations about the model. Memory 0 concerns '
    'its controls and Memory 3 concerns its identity.'
)


def definition_command():
    tree = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "pargs.lower().startswith('explain ')")
    return ast.fix_missing_locations(ast.Module(body=[ast.For(
        target=ast.Name(id='_run', ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
        body=[copy.deepcopy(branch)], orelse=[])], type_ignores=[]))


COMMAND = definition_command()


class NativeTokenizer:
    all_special_tokens = ()

    def __init__(self):
        self.messages = []

    @staticmethod
    def header(request):
        return '[native user]\n' + request + '\n[native assistant]\n'

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        self.messages.append(copy.deepcopy(messages))
        return self.header(messages[0]['content'])

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is False and return_tensors == 'pt'
        return {'input_ids': torch.tensor([[ord(char) for char in text]])}


class NativeRoot(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.model = SimpleNamespace(layers=self.layers)
        self.pulses = []

    def forward(self, hidden):
        pulses = []
        for layer in self.layers:
            before = hidden.clone()
            hidden = layer(hidden)
            pulses.append(hidden.detach().clone() - before)
        self.pulses.append(pulses)
        return hidden


class OutputHarness:
    def __init__(self, monkeypatch, tmp_path, *, suffixes=(DEFINITION,),
                 tokens=30, fail=False, telemetry=True, command='explain secret_probe evidence'):
        self.layers = [torch.nn.Identity() for _ in range(6)]
        self.root = NativeRoot(self.layers)
        self.model = SimpleNamespace(device='cpu', d_model=3, model=self.root,
                                     tok=NativeTokenizer())
        self.probe = {
            'direction': {4: torch.tensor([1., 0., 0.]), 5: torch.tensor([1., 0., 0.])},
            'framings': ('Secret positive seed', 'Secret negative seed'), 'exposed': True,
        }
        self.probes = {'secret_probe': self.probe}
        self.behavior = ProbeBehavior(path=tmp_path / 'behavior.json')
        replies = ('I might be mistaken.', 'This is probably right.',
                   'This is certainly true.', 'There is no doubt about this.')
        for reply, reading in zip(replies, (-.8, -.2, .2, .8)):
            self.behavior.observe(reply, 'How certain is the claim?',
                                  {'probe_secret_probe': reading}, self.probes)
        self.language = ExposureLanguage(path=tmp_path / 'language.json')
        self.config = SimpleNamespace(cache_enabled=True, cache_write_enabled=True,
                                      max_routing_events=3, max_synthesis_events=3,
                                      max_synthesis_steps=3, max_loops=3)
        self.tuner = SimpleNamespace(triggers={}, get=Mock(return_value=0.))
        self.language.sync(self.probes, self.tuner)
        self.records_before = copy.deepcopy(self.behavior.records)
        self.evidence = self.behavior.evidence('secret_probe', self.probe, self.probes)
        self.config_before = dict(vars(self.config))
        self.behavior_observe = Mock(wraps=self.behavior.observe)
        self.language_observe = Mock(wraps=self.language.observe)
        self.naming = Mock(wraps=self.language.naming_query)
        monkeypatch.setattr(self.behavior, 'observe', self.behavior_observe)
        monkeypatch.setattr(self.language, 'observe', self.language_observe)
        monkeypatch.setattr(self.language, 'naming_query', self.naming)
        self.handles = Mock(wraps=probe_behavior.build_definition_handles)
        monkeypatch.setattr(probe_behavior, 'build_definition_handles', self.handles)
        self.mint = Mock(side_effect=AssertionError('Normal definitions must not mint task controls'))
        monkeypatch.setattr(probe_definition_task, 'mint_definition_task_axes', self.mint)
        self.cognitive_cache = SimpleNamespace(write=Mock(), put=Mock())
        self.ledger = SimpleNamespace(observe=Mock(), record=Mock())
        self.matrix = SimpleNamespace(
            other_reference=SimpleNamespace(activate=Mock()),
            probe_behavior=self.behavior, exposure_language=self.language)
        self.owners = SimpleNamespace(for_tuner=Mock(return_value=(self.matrix, {})))
        self.suffixes, self.tokens = list(suffixes), tokens
        self.fail, self.telemetry = fail, telemetry
        self.calls = []
        self.mutate_timing_fields = False
        self.original_hooks = None
        self.namespace = dict(vars(shell), probes=self.probes, model=self.model,
                              config=self.config, tuner=self.tuner,
                              plasticity_owners=self.owners, _other_mint=Mock(),
                              _engine_generate=self.generate, cognitive_cache=self.cognitive_cache,
                              ledger=self.ledger,
                              pargs=shell.normalize_probe_definition_args(command))

    def generate(self, model, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        assert model is self.model
        assert kwargs['vecs'] == {} and kwargs['return_telemetry'] is True
        assert kwargs['pre_formatted'] is True and kwargs['chatty_log'] is False
        assert kwargs['max_new_tokens'] == DEFINITION_MAX_TOKENS
        assert not self.config.cache_enabled and not self.config.cache_write_enabled
        for name in ('max_routing_events', 'max_synthesis_events', 'max_synthesis_steps', 'max_loops'):
            assert getattr(self.config, name) == 0
        request = self.model.tok.messages[-1][0]['content']
        assert kwargs['instruction'] == NativeTokenizer.header(request) + PREFIX
        assert request.count(self.evidence) == 1
        assert 'definition' in request.lower()
        for forbidden in ('Secret positive seed', 'Secret negative seed', 'secret_probe'):
            assert forbidden not in request
        self.naming.assert_not_called()
        for layer, module in enumerate(self.layers):
            assert len(module._forward_hooks) == len(self.original_hooks[layer]) + (layer in (4, 5))
        self.root(torch.ones(1, 5, 3))
        self.root(torch.ones(1, 1, 3))
        self.root(torch.ones(1, 7, 3))
        if self.mutate_timing_fields:
            self.config._generation_deadline = None
            self.config._generation_budget_sec = None
            self.config.committee = object()
            self.config._synthesis_events_used = 0
        if self.fail:
            raise RuntimeError('definition generation failed')
        suffix = self.suffixes[min(index, len(self.suffixes) - 1)]
        return (suffix, {'tokens_generated': self.tokens}) if self.telemetry else suffix

    def run(self):
        self.original_hooks = [set(layer._forward_hooks) for layer in self.layers]
        root_hooks = set(self.root._forward_pre_hooks)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(COMMAND, shell.__file__, 'exec'), self.namespace)
        finally:
            assert [set(layer._forward_hooks) for layer in self.layers] == self.original_hooks
            assert set(self.root._forward_pre_hooks) == root_hooks
            assert dict(vars(self.config)) == self.config_before
            assert self.behavior.records == self.records_before
            self.behavior_observe.assert_not_called()
            self.language_observe.assert_not_called()
            self.cognitive_cache.write.assert_not_called()
            self.cognitive_cache.put.assert_not_called()
            self.ledger.observe.assert_not_called()
            self.ledger.record.assert_not_called()
        self.language.sync(self.probes, self.tuner)
        return output.getvalue()

    def attempt(self):
        return self.behavior.interpretations[definition_key('secret_probe', self.probe)][-1]


def test_definition_requests_output_kind_with_unchanged_measured_evidence(monkeypatch, tmp_path):
    harness = OutputHarness(monkeypatch, tmp_path)
    output = harness.run()
    assert len(harness.calls) == 1
    assert 'Definition hypothesis' in output and PREFIX + DEFINITION in output
    attempt = harness.attempt()
    assert attempt['text'] == PREFIX + DEFINITION
    assert attempt['status'] == 'hypothesis' and attempt['qualified']
    assert attempt['controls']['mode'] == 'definition_output'
    assert len(attempt['controls']['attempts']) == 1
    assert attempt['controls']['prefills'][0]['assistant_prefix'] == PREFIX
    assert 'threshold=0.35' in harness.language.context()
    harness.mint.assert_not_called()
    harness.naming.assert_not_called()
    assert harness.handles.call_args.args == (harness.model, harness.probe['direction'], {})
    assert harness.handles.call_args.kwargs['task_gains'] == {}
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == harness.evidence


def test_only_original_decision_gets_subject_pulse_and_unrelated_hooks_survive(monkeypatch, tmp_path):
    harness = OutputHarness(monkeypatch, tmp_path)
    calls = []
    existing = harness.layers[4].register_forward_hook(lambda _m, _i, _o: calls.append('existing'))
    try:
        harness.run()
        assert calls == ['existing'] * 3
        initial, decode, rebuild = harness.root.pulses
        for layer in range(6):
            torch.testing.assert_close(decode[layer], torch.zeros_like(decode[layer]))
            if layer not in (4, 5):
                torch.testing.assert_close(initial[layer], torch.zeros_like(initial[layer]))
                torch.testing.assert_close(rebuild[layer], torch.zeros_like(rebuild[layer]))
                continue
            torch.testing.assert_close(initial[layer][:, :4], torch.zeros_like(initial[layer][:, :4]))
            assert torch.count_nonzero(initial[layer][:, 4]) == 1
            torch.testing.assert_close(rebuild[layer][:, :4], torch.zeros_like(rebuild[layer][:, :4]))
            torch.testing.assert_close(rebuild[layer][:, 5:], torch.zeros_like(rebuild[layer][:, 5:]))
    finally:
        existing.remove()


def test_mistaken_inferred_meaning_still_counts_as_a_definition(monkeypatch, tmp_path):
    suffix = ('a policy of obedience to authority: lower readings resist commands '
              'whereas higher readings follow them without hesitation.')
    harness = OutputHarness(monkeypatch, tmp_path, suffixes=(suffix,))
    harness.run()
    assert harness.attempt()['status'] == 'hypothesis'
    assert harness.namespace['qualification']['semantic_truth_checked'] is False


def test_recap_retry_is_bounded_and_rejected_text_is_not_a_new_prompt_turn(monkeypatch, tmp_path):
    harness = OutputHarness(monkeypatch, tmp_path, suffixes=(RECAP,))
    output = harness.run()
    assert len(harness.calls) == 2
    assert harness.attempt()['status'] == 'unqualified'
    assert len(harness.attempt()['controls']['attempts']) == 2
    assert RECAP not in output and 'No definition produced' in output
    assert 'threshold=0.65' in harness.language.context()
    assert 'Namespace interpretation hypothesis=' not in harness.language.context()
    correction = harness.model.tok.messages[1][0]['content']
    assert RECAP not in correction and correction.count(harness.evidence) == 1
    harness.naming.assert_not_called()


def test_retry_can_replace_recap_with_definition(monkeypatch, tmp_path):
    harness = OutputHarness(monkeypatch, tmp_path, suffixes=(RECAP, DEFINITION))
    output = harness.run()
    assert len(harness.calls) == 2
    assert harness.attempt()['status'] == 'hypothesis'
    assert harness.attempt()['text'] == PREFIX + DEFINITION
    assert 'Definition hypothesis' in output
    assert len(harness.handles.call_args_list) == 2


def test_token_limit_cannot_enable_naming_or_display_an_unfinished_definition(monkeypatch, tmp_path):
    harness = OutputHarness(monkeypatch, tmp_path, tokens=DEFINITION_MAX_TOKENS)
    output = harness.run()
    assert harness.attempt()['status'] == 'incomplete'
    assert 'threshold=0.65' in harness.language.context()
    assert PREFIX + DEFINITION not in output
    assert len(harness.calls) <= 2


@pytest.mark.parametrize('failure', ['generation', 'missing_telemetry'])
def test_failure_restores_configuration_and_hooks_without_creating_observations(monkeypatch, tmp_path, failure):
    harness = OutputHarness(monkeypatch, tmp_path, fail=failure == 'generation',
                            telemetry=failure != 'missing_telemetry')
    with pytest.raises((RuntimeError, ValueError)):
        harness.run()
    assert harness.behavior.interpretations == {}
    assert 'threshold=0.65' in harness.language.context()


@pytest.mark.parametrize('fail', [False, True])
@pytest.mark.parametrize('preexisting', [False, True])
def test_temporary_engine_state_restores_values_or_absence(monkeypatch, tmp_path, fail, preexisting):
    harness = OutputHarness(monkeypatch, tmp_path, fail=fail)
    harness.mutate_timing_fields = True
    if preexisting:
        harness.config._generation_deadline = 1234.5
        harness.config._generation_budget_sec = 45.
        harness.config.committee = object()
        harness.config._synthesis_events_used = 7
        harness.config_before = dict(vars(harness.config))
    if fail:
        with pytest.raises(RuntimeError, match='definition generation failed'):
            harness.run()
    else:
        harness.run()
    assert dict(vars(harness.config)) == harness.config_before


def test_insufficient_measured_examples_explains_backfill_without_generating(monkeypatch, tmp_path):
    harness = OutputHarness(monkeypatch, tmp_path)
    key = definition_key('secret_probe', harness.probe)
    harness.behavior.records[key].pop()
    harness.records_before = copy.deepcopy(harness.behavior.records)
    output = harness.run()
    assert 'at least four measured examples' in output and ':probe backfill' in output
    assert not harness.calls
    harness.handles.assert_not_called()
    assert harness.behavior.interpretations == {}


@pytest.mark.parametrize('command', [
    'secret_probe explain evidence', '[secret_probe] explain evidence', 'explain [secret_probe] evidence',
    'explain probe:secret_probe evidence',
])
def test_operator_name_first_or_bracketed_target_reaches_definition(monkeypatch, tmp_path, command):
    harness = OutputHarness(monkeypatch, tmp_path, command=command)
    harness.run()
    assert harness.attempt()['status'] == 'hypothesis'


def test_displayed_opaque_identifier_resolves_even_for_hidden_probe(monkeypatch, tmp_path):
    harness = OutputHarness(monkeypatch, tmp_path, command=f'[{probe_alias("secret_probe")}] explain evidence')
    harness.probe['exposed'] = False
    harness.run()
    assert harness.attempt()['status'] == 'hypothesis'


def test_creation_framings_containing_define_are_not_reordered():
    framing = 'new_trait I define useful concepts || I avoid definitions'
    assert shell.normalize_probe_definition_args(framing) == framing


def test_probe_define_shows_original_definition_without_generation(capsys):
    probes = {
        'test_p': {
            'framings': ('I test things carefully', 'I test things carelessly'),
            'direction': {16: torch.tensor([1.0, 0.0]), 17: torch.tensor([1.0, 0.0])},
            'exposed': True,
        }
    }
    shell.print_probe_original_definition('test_p', probes)
    out = capsys.readouterr().out
    clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', out)
    assert 'Origin: minted contrastive framings' in clean
    assert 'WITH    (+ pole): I test things carefully' in clean
    assert 'WITHOUT (- pole): I test things carelessly' in clean
    assert ':probe test_p I test things carefully || I test things carelessly' in clean

