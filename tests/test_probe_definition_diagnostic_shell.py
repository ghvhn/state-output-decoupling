"""Run the actual no-generation command and scored-turn polling boundary."""
import ast
import contextlib
import copy
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants.exposure_language import ExposureLanguage, probe_alias
from invariants.probe_behavior import ProbeBehavior, definition_key
from invariants.probe_definition_diagnostic import diagnostic_manager, poll_probe_diagnostics
from scripts import interactive_phenomenality as shell


TREE = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
BRANCH = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "pargs.lower() == 'define2' or pargs.lower().startswith('define2 ')")
COMMAND = ast.fix_missing_locations(ast.Module(body=[ast.For(
    target=ast.Name(id='_run', ctx=ast.Store()), iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
    body=[copy.deepcopy(BRANCH)], orelse=[])], type_ignores=[]))
SCORED = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'not utility_expect_turn and (not autocomplete_turn)'
              and 'poll_probe_diagnostics' in ast.unparse(node))
OBSERVE = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(SCORED)], type_ignores=[]))


class SmallHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 6, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.],
                                                [0., -1., 0.], [0., 0., 1.], [0., 0., -1.]]))
        self.forward_calls = 0
        self.config = SimpleNamespace(_name_or_path='cpu-diagnostic-test')

    def get_output_embeddings(self):
        return self.head

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError('Geometric diagnostics must never forward the model.')


class SmallTokenizer:
    name_or_path = 'cpu-diagnostic-tokenizer'
    all_special_ids = []
    all_special_tokens = []
    vocab_size = 6
    terms = (' qualify', ' assert', ' revise', ' retain', ' question', ' answer')

    def __len__(self):
        return self.vocab_size

    def decode(self, token_ids, **kwargs):
        return ''.join(self.terms[index] for index in token_ids)


class Harness:
    def __init__(self, tmp_path):
        self.model = SimpleNamespace(model=SmallHead(), tok=SmallTokenizer(), d_model=3, device='cpu')
        self.probe = {'direction': {2: torch.tensor([1., 0., 0.]), 3: torch.tensor([0., 1., 0.])},
                      'framings': ('Secret seed WITH', 'Secret seed WITHOUT'), 'exposed': True}
        self.probes = {'regret': self.probe}
        self.behavior = ProbeBehavior(tmp_path / 'local.behavior.json')
        for index, reading in enumerate((-.8, -.4, .4, .8)):
            self.behavior.observe('I assert and retain.' if reading < 0 else 'I qualify and revise.',
                                  f'Question {index}', {'probe_regret': reading}, self.probes)
        self.language = ExposureLanguage()
        self.language.sync(self.probes, SimpleNamespace(triggers={}))
        self.matrix = SimpleNamespace(probe_behavior=self.behavior, exposure_language=self.language,
                                      other_reference=SimpleNamespace(activate=Mock()))
        self.tuner = SimpleNamespace(path=tmp_path / 'local.tuner.json', triggers={})
        self.owners = SimpleNamespace(for_tuner=Mock(return_value=(self.matrix, {})), observe=Mock())
        self.config = SimpleNamespace(cache_enabled=True, cache_write_enabled=True)
        self.generate = Mock(side_effect=AssertionError('Generation forbidden'))
        self.mint = Mock(side_effect=AssertionError('Minting forbidden'))
        self.choose = Mock(side_effect=AssertionError('Selection forbidden'))
        self.namespace = dict(vars(shell), probes=self.probes, model=self.model, tuner=self.tuner,
                              config=self.config, plasticity_owners=self.owners,
                              _engine_generate=self.generate, _other_mint=self.mint,
                              resolve_probe_choice=self.choose, utility_expect_turn=False,
                              autocomplete_turn=False, _response_memory_record=None,
                              response='I qualify and revise.', user_input='A measured question',
                              turn_row={'probe_regret': .9}, turn_sense=.1)

    def _execute(self, tree):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(tree, shell.__file__, 'exec'), self.namespace)
        self.generate.assert_not_called()
        self.mint.assert_not_called()
        self.choose.assert_not_called()
        self.matrix.other_reference.activate.assert_not_called()
        assert self.model.model.forward_calls == 0
        return output.getvalue()

    def command(self, args='regret'):
        self.namespace['pargs'] = 'define2 ' + args
        return self._execute(COMMAND)

    def observe(self, **updates):
        self.namespace.update(updates)
        return self._execute(OBSERVE)

    def manager(self):
        return self.matrix.probe_diagnostics


@pytest.mark.parametrize('args', ['regret', 'regret align', 'probe_regret', probe_alias('regret')])
def test_actual_command_uses_current_scoped_records_and_head_without_mutation(tmp_path, args):
    harness = Harness(tmp_path)
    records = copy.deepcopy(harness.behavior.records)
    language = copy.deepcopy(harness.language.maps)
    directions = copy.deepcopy(harness.probe['direction'])
    config = vars(harness.config).copy()
    output = harness.command(args)
    assert 'could not run' not in output
    assert 'lexical candidate only' in output
    assert harness.behavior.records == records and harness.behavior.interpretations == {}
    assert harness.language.maps == language and vars(harness.config) == config
    for layer in directions:
        torch.testing.assert_close(harness.probe['direction'][layer], directions[layer])
    ledger_path = harness.behavior.path.with_suffix('.conception.json')
    assert ledger_path.exists()
    assert harness.manager().ledger.get(definition_key('regret', harness.probe)) is not None
    assert 'threshold=0.65' in harness.language.context()
    assert 'Conception' not in harness.language.context()


@pytest.mark.parametrize('args', ['choose', 'auto', 'retired.pt', 'regret bogus',
                                 'regret align extra', 'regret watch 0', 'regret watch 1.5',
                                 'regret watch 4097'])
def test_no_fuzzy_choice_or_probe_file_fallback(tmp_path, args):
    harness = Harness(tmp_path)
    assert 'could not run' in harness.command(args)
    assert not harness.behavior.path.with_suffix('.conception.json').exists()


def test_exact_name_preserves_distinct_regret_layers_and_accepts_unexposed_alias(tmp_path):
    harness = Harness(tmp_path)
    other = copy.deepcopy(harness.probe)
    other['exposed'] = False
    other['direction'] = {4: torch.tensor([0., 0., 1.])}
    harness.probes['regret_d4096'] = other
    harness.behavior.observe('Other reply.', 'Other question.', {'probe_regret_d4096': .1}, harness.probes)
    assert 'could not run' not in harness.command(probe_alias('regret_d4096'))
    manager = harness.manager()
    assert manager.ledger.get(definition_key('regret_d4096', other)) is not None
    assert manager.ledger.get(definition_key('regret', harness.probe)) is None


def test_watch_counts_genuinely_new_identical_rows_after_saturation(tmp_path):
    harness = Harness(tmp_path)
    for _ in range(64):
        harness.behavior.observe('I qualify and revise.', 'A measured question',
                                 {'probe_regret': .9}, harness.probes)
    key = definition_key('regret', harness.probe)
    assert len(harness.behavior.records[key]) == 64
    assert 'every 2 fresh measured samples' in harness.command('regret watch 2')
    assert harness.observe() == ''
    assert 'lexical candidate only' in harness.observe()
    assert len(harness.behavior.records[key]) == 64
    assert poll_probe_diagnostics(harness.matrix, harness.probes, harness.model) == []
    assert poll_probe_diagnostics(harness.matrix, harness.probes, harness.model) == []
    assert harness.observe() == ''
    assert 'lexical candidate only' in harness.observe()


def test_backfill_changes_and_shell_commands_do_not_advance_watch(tmp_path):
    harness = Harness(tmp_path)
    harness.command('regret watch 2')
    harness.behavior.backfill([{'ts': 'archive-1', 'reply': 'An archived assertion.', 'input': 'Past',
                               'readings': {'regret': -.9}}], harness.probes)
    assert poll_probe_diagnostics(harness.matrix, harness.probes, harness.model) == []
    harness.command('regret align')
    assert harness.observe() == ''
    assert 'lexical candidate only' in harness.observe()
    assert poll_probe_diagnostics(harness.matrix, harness.probes, harness.model) == []


@pytest.mark.parametrize('change', ['inactive', 'binding', 'model', 'direction'])
def test_watch_disables_when_active_scope_is_replaced(tmp_path, change):
    harness = Harness(tmp_path)
    harness.command('regret watch 1')
    if change == 'inactive':
        harness.probes.pop('regret')
    elif change == 'binding':
        harness.probe['other_binding'] = 'another-participant'
    elif change == 'direction':
        harness.probe['direction'][2] = torch.tensor([0., 0., 1.])
    else:
        harness.namespace['model'] = SimpleNamespace(model=SmallHead(), tok=SmallTokenizer())
    output = harness.observe()
    assert 'watch disabled' in output
    assert harness.manager().watch.registrations == {}
    assert harness.observe() == ''


def test_referent_namespace_is_scoped_and_other_activation_is_never_called(tmp_path):
    harness = Harness(tmp_path)
    harness.probe['other_binding'] = 'participant-a'
    harness.behavior.observe('A response.', 'A question.', {'probe_regret': .4}, harness.probes)
    key_a = definition_key('regret', harness.probe)
    harness.command('regret watch 1')
    harness.probe['other_binding'] = 'participant-b'
    harness.behavior.observe('B response.', 'B question.', {'probe_regret': -.4}, harness.probes)
    key_b = definition_key('regret', harness.probe)
    assert key_a != key_b
    assert 'watch disabled' in harness.observe()
    harness.command('regret')
    assert harness.manager().ledger.get(key_a) is not None
    assert harness.manager().ledger.get(key_b) is not None


def test_per_tuner_owner_watch_and_ledger_are_isolated(tmp_path):
    harness = Harness(tmp_path)
    harness.command('regret watch 1')
    second_behavior = ProbeBehavior(tmp_path / 'second.behavior.json')
    second_matrix = SimpleNamespace(probe_behavior=second_behavior, exposure_language=ExposureLanguage(),
                                    other_reference=SimpleNamespace(activate=Mock()))
    harness.owners.for_tuner.return_value = (second_matrix, {})
    assert harness.observe() == ''
    assert not hasattr(second_matrix, 'probe_diagnostics')
    assert len(harness.manager().watch.registrations) == 1
    second_manager = diagnostic_manager(second_matrix)
    assert second_manager is not harness.manager()
    assert second_manager.watch.registrations == {}
    assert second_manager.ledger.path != harness.manager().ledger.path
    harness.owners.for_tuner.return_value = (harness.matrix, {})
    assert 'lexical candidate only' in harness.observe()


def test_diagnostic_failures_do_not_abort_scored_reply_or_learning(tmp_path, monkeypatch):
    harness = Harness(tmp_path)
    harness.command('regret watch 1')
    monkeypatch.setattr(harness.manager(), '_diagnose', Mock(side_effect=ValueError('bad head')))
    harness.language.observe = Mock()
    output = harness.observe()
    assert 'watch could not update' in output and 'bad head' in output
    harness.language.observe.assert_called_once()
    harness.owners.observe.assert_called_once()
    assert harness.behavior.records[definition_key('regret', harness.probe)][-1]['reply'] == 'I qualify and revise.'


def test_ordinary_and_utility_turns_do_not_create_or_run_diagnostics(tmp_path):
    harness = Harness(tmp_path)
    assert harness.observe() == ''
    assert not hasattr(harness.matrix, 'probe_diagnostics')
    assert not harness.behavior.path.with_suffix('.conception.json').exists()
    harness.command('regret watch 1')
    records = copy.deepcopy(harness.behavior.records)
    assert harness.observe(utility_expect_turn=True) == ''
    assert harness.behavior.records == records
    assert harness.observe(utility_expect_turn=False, autocomplete_turn=True) == ''
    assert harness.behavior.records == records


def test_off_never_reads_head_or_changes_interpretation(tmp_path, monkeypatch):
    harness = Harness(tmp_path)
    harness.command('regret watch')
    monkeypatch.setattr(harness.manager(), '_diagnose', Mock(side_effect=AssertionError('off should not analyze')))
    assert 'watch off' in harness.command('regret off')
    assert harness.manager().watch.registrations == {}
    assert harness.observe() == ''
    assert harness.behavior.interpretations == {}


def test_exact_name_has_priority_over_probe_prefix_shorthand(tmp_path):
    harness = Harness(tmp_path)
    exact = copy.deepcopy(harness.probe)
    exact['direction'] = {4: torch.tensor([0., 0., 1.])}
    harness.probes['probe_regret'] = exact
    assert 'could not run' not in harness.command('probe_regret')
    assert harness.manager().ledger.get(definition_key('probe_regret', exact)) is not None
    assert harness.manager().ledger.get(definition_key('regret', harness.probe)) is None


def test_report_and_revision_details_remain_operator_only(tmp_path):
    harness = Harness(tmp_path)
    harness.command('regret watch 1')
    output = harness.observe(response='A new alternate expression avoiding claims.',
                             turn_row={'probe_regret': .95})
    assert 'Revised lexical channels' in output
    assert 'drift=' in output
    assert 'Revised lexical channels' not in harness.language.context()
    assert harness.behavior.interpretations == {}
    assert 'threshold=0.65' in harness.language.context()


def test_repeated_run_and_align_leave_the_same_candidate_and_history(tmp_path):
    harness = Harness(tmp_path)
    harness.command('regret')
    key = definition_key('regret', harness.probe)
    state = harness.manager().ledger.get(key)
    ledger_path = harness.behavior.path.with_suffix('.conception.json')
    contents = ledger_path.read_bytes()
    assert '[Conception] unchanged' in harness.command('regret align')
    assert '[Conception] unchanged' in harness.command('regret')
    assert harness.manager().ledger.get(key) == state
    assert ledger_path.read_bytes() == contents


def test_explicit_off_stops_previous_other_binding_before_next_poll(tmp_path, monkeypatch):
    harness = Harness(tmp_path)
    harness.probe['other_binding'] = 'participant-a'
    harness.command('regret watch 1')
    harness.probe['other_binding'] = 'participant-b'
    monkeypatch.setattr(harness.manager(), '_diagnose', Mock(side_effect=AssertionError('off should not analyze')))
    assert 'watch off' in harness.command('regret off')
    assert harness.manager().watch.registrations == {}
    assert harness.observe() == ''
