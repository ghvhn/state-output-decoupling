"""Exercise the actual mode-three definition shell with real CPU pulse hooks.

The synthetic continuation is supplied by the harness. These checks establish
the shell's context, hook, persistence and qualification contracts, not success
of a learned definition on a language model.
"""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import probe_definition_schema, probe_definition_sequence
from invariants.definition_prefill import DEFAULT_DEFINITION_PREFIX
from invariants.exposure_language import probe_alias
from invariants.probe_behavior import DEFINITION_MAX_TOKENS, ProbeBehavior, definition_key
from invariants.relational_conversion import fit_relational_conversion
from scripts import interactive_phenomenality as shell
from test_probe_definition_shell import COMMAND, DefinitionHarness, RECAP
from test_probe_definition_tuning import registered_tuner, run_tune


SUFFIX = (
    'lower readings (Memory 0 and Memory 1) qualify claims, whereas higher readings '
    '(Memory 2 and Memory 3) assert them categorically.'
)
SELECTED = (2, 3, 4, 5)


class NativeTokenizer:
    """A text native template and deterministic context-relative token counts."""
    all_special_tokens = ()

    def __init__(self):
        self.templates = []
        self.encoded = []

    @staticmethod
    def header(evidence):
        return '[native user]\n' + evidence + '\n[native assistant]\n'

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        self.templates.append(copy.deepcopy(messages))
        return self.header(messages[0]['content'])

    def __call__(self, text, *, add_special_tokens, return_tensors):
        assert add_special_tokens is False and return_tensors == 'pt'
        self.encoded.append(text)
        return {'input_ids': torch.tensor([[ord(char) for char in text]])}


class NativeRoot(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.model = SimpleNamespace(layers=self.layers)
        self.calls = []

    def forward(self, hidden):
        pulses = []
        for layer in self.layers:
            before = hidden.clone()
            hidden = layer(hidden)
            pulses.append(hidden.detach().clone() - before)
        self.calls.append(pulses)
        return hidden


def adaptive_bank():
    # A full-rank set and distinct affine transformations make these genuine
    # validated 16-row conditional profiles, rather than dictionary stubs.
    sources = torch.tensor([[a + 1., b + 1., c + 1.]
                            for a in (-1.5, -.5, .5, 1.5)
                            for b in (-.5, .5) for c in (-.5, .5)])
    transforms = (
        torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., .4], [0., 0., .4], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [-.4, .3, 0.]]),
    )
    offsets = (torch.tensor([.1, .2, .1]), torch.tensor([.2, .1, .1]),
               torch.tensor([.1, .1, .2]), torch.tensor([.3, .1, .2]))
    return {
        role: {layer: copy.deepcopy(fit_relational_conversion(
            sources, sources @ transform + offset)) for layer in SELECTED}
        for role, transform, offset in zip(probe_definition_schema.SCHEMA_ORDER,
                                          transforms, offsets)
    }


class SchemaHarness(DefinitionHarness):
    def __init__(self, monkeypatch, tmp_path, *, text=SUFFIX, tokens=30,
                 gains=None, stats=None, fail=False, telemetry=True):
        super().__init__(monkeypatch, text=text, tokens=tokens, fail=fail,
                         gains={'probe_define_process': 3., **(gains or {})})
        self.probe['direction'] = {layer: torch.tensor([1., 0., 0.])
                                   for layer in SELECTED}
        self.root = NativeRoot(self.layers)
        self.model.model = self.root
        self.model.tok = NativeTokenizer()
        self.behavior.path = tmp_path / 'behavior.json'
        self.behavior.save()
        self.language.sync(self.probes, self.tuner)
        self.naming = Mock(wraps=self.language.naming_query)
        monkeypatch.setattr(self.language, 'naming_query', self.naming)
        self.behavior_observe = Mock(wraps=self.behavior.observe)
        monkeypatch.setattr(self.behavior, 'observe', self.behavior_observe)
        self.language_observe = Mock(wraps=self.language.observe)
        monkeypatch.setattr(self.language, 'observe', self.language_observe)
        self.bank = adaptive_bank()
        self.metadata = {
            'path': str((tmp_path / 'schema-test.pt').resolve()),
            'version': probe_definition_schema.SCHEMA_VERSION,
            'identity': {'model': 'cpu-shell-model', 'assistant_prefix': self.prefix},
            'identity_hash': 'operator-schema-identity',
            'part_order': list(probe_definition_schema.SCHEMA_ORDER),
            'gains': dict(probe_definition_schema.SCHEMA_GAINS),
            'provenance': {'endpoint': 'native_block_output_after_exact_assistant_prefix',
                           'calibration_rows_per_part': 16, 'teacher_pairs': 2,
                           'recipe_sha256': 'operator-only-teacher-recipe',
                           'status': 'candidate', 'semantic_truth_checked': False,
                           'online_intervention_source_shift': 'experimental_unverified'},
            'load_status': 'loaded', 'cache_status': 'hit',
        }
        self.bank_loader = Mock(return_value=(self.bank, self.metadata))
        monkeypatch.setattr(probe_definition_schema, 'get_or_mint_definition_schema',
                            self.bank_loader)
        self.sequence_builder = Mock(wraps=probe_definition_sequence.build_schema_sequence_handles)
        monkeypatch.setattr(probe_definition_sequence, 'build_schema_sequence_handles',
                            self.sequence_builder)
        self.stats = {'tokens_generated': tokens} if stats is None else stats
        self.telemetry = telemetry
        self.instructions = []
        self.original_layer_hooks = []
        self.original_root_hooks = set()

    @property
    def prefix(self):
        return DEFAULT_DEFINITION_PREFIX if self.gains.get('probe_define_schema_prefill', 1.) else ''

    def generate(self, model, **kwargs):
        self.generate_calls += 1
        assert model is self.model
        assert len(self.root._forward_pre_hooks) == len(self.original_root_hooks) + 1
        for index, layer in enumerate(self.layers):
            assert len(layer._forward_hooks) == len(self.original_layer_hooks[index]) + (index in SELECTED)
        assert kwargs['vecs'] == {} and kwargs['return_telemetry'] is True
        assert kwargs['pre_formatted'] is True and kwargs['chatty_log'] is False
        assert kwargs['max_new_tokens'] == DEFINITION_MAX_TOKENS == 384
        assert not self.config.cache_enabled and not self.config.cache_write_enabled
        for name in ('max_routing_events', 'max_synthesis_events', 'max_synthesis_steps', 'max_loops'):
            assert getattr(self.config, name) == 0
        self.naming.assert_not_called()
        evidence = self.behavior.evidence('secret_probe', self.probe, self.probes)
        assert kwargs['instruction'] == NativeTokenizer.header(evidence) + self.prefix
        assert self.model.tok.templates == [[{'role': 'user', 'content': evidence}]]
        for operator_text in ('Secret positive seed', 'Secret negative seed', 'secret_probe',
                              'operator-only-teacher-recipe', 'operator-schema-identity',
                              'genus', 'core', 'boundary', 'compose', 'role_gains',
                              probe_definition_schema.SCHEMA_VERSION,
                              probe_definition_sequence.SEQUENCE_VERSION, 'semantic_truth_checked'):
            assert operator_text not in kwargs['instruction']
        self.instructions.append(kwargs['instruction'])
        count = len(kwargs['instruction'])
        self.root(torch.ones(1, count, 3))
        self.root(torch.ones(1, 1, 3))
        self.root(torch.ones(1, count + 2, 3))
        if self.fail:
            raise RuntimeError('schema generation failed')
        return (self.text, self.stats) if self.telemetry else self.text

    def run(self):
        self.original_layer_hooks = [set(layer._forward_hooks) for layer in self.layers]
        self.original_root_hooks = set(self.root._forward_pre_hooks)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(COMMAND, shell.__file__, 'exec'), self.namespace)
        finally:
            assert [set(layer._forward_hooks) for layer in self.layers] == self.original_layer_hooks
            assert set(self.root._forward_pre_hooks) == self.original_root_hooks
            assert dict(vars(self.config)) == self.original_config
        self.language.sync(self.probes, self.tuner)
        return output.getvalue()

    def attempt(self):
        return self.behavior.interpretations[definition_key('secret_probe', self.probe)][-1]


def protect_unrelated_hooks(harness):
    root_calls, layer_calls = [], []
    handles = [harness.root.register_forward_pre_hook(lambda _m, _i: root_calls.append('root')),
               harness.layers[2].register_forward_hook(lambda _m, _i, _o: layer_calls.append('layer'))]
    return handles, root_calls, layer_calls


def test_structure_uses_one_native_generation_and_persists_only_operator_trace(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    harness = SchemaHarness(monkeypatch, tmp_path)
    records_before = copy.deepcopy(harness.behavior.records)
    directions_before = copy.deepcopy(harness.probe['direction'])
    evidence = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    output = harness.run()

    harness.bank_loader.assert_called_once_with(
        harness.model, SELECTED,
        Path(shell.__file__).resolve().parent.parent / 'invariants/out/definition_schema',
        assistant_prefix=DEFAULT_DEFINITION_PREFIX)
    assert harness.generate_calls == harness.sequence_builder.call_count == 1
    assert harness.sequence_builder.call_args.args[:3] == (
        harness.model, harness.probe['direction'], harness.bank)
    attempt = harness.attempt()
    assert attempt['status'] == 'hypothesis'
    assert attempt['text'] == DEFAULT_DEFINITION_PREFIX + SUFFIX
    assert attempt['qualified'] is True
    assert harness.namespace['qualification']['structural_only'] is True
    assert harness.namespace['qualification']['semantic_truth_checked'] is False
    assert 'Namespace interpretation hypothesis:' in output
    controls = attempt['controls']
    assert controls['mode'] == 'structure'
    assert controls['schema_version'] == probe_definition_schema.SCHEMA_VERSION
    assert controls['schema_profiles'] == harness.metadata
    assert controls['role_gains'] == probe_definition_schema.SCHEMA_GAINS
    assert controls['namespace'] == definition_key('secret_probe', harness.probe)
    assert controls['evidence_sha256'] == hashlib.sha256(evidence.encode('utf-8')).hexdigest()
    assert controls['max_tokens'] == 384 and controls['subject_gain'] == .02
    prefill = controls['prefill']
    assert prefill['assistant_prefix'] == DEFAULT_DEFINITION_PREFIX
    assert prefill['prompt_token_count'] == len(harness.instructions[0])
    assert prefill['prefix_token_count'] == prefill['prefix_token_delta'] == len(DEFAULT_DEFINITION_PREFIX)
    assert prefill['retokenized_header_tokens'] == 0
    trace = controls['sequence']
    assert trace['prediction_count'] == 4 and trace['forward_count'] == 3
    assert trace['forward_kinds'] == {'initial_prefill': 1, 'cached_decode': 1, 'prefix_rebuild': 1}
    assert trace['signature']['role_layers'] == dict(zip(probe_definition_schema.SCHEMA_ORDER,
                                                        ([2], [3], [4], [5])))
    assert trace['online_intervention_source_shift'] == 'experimental_unverified'
    assert trace['semantic_truth_checked'] is False and trace['slot_completion_checked'] is False
    assert [row['role'] for row in trace['layers']] == list(probe_definition_schema.SCHEMA_ORDER)
    assert [row['subject_pulse'] for row in trace['layers']] == [True, False, False, False]
    assert all(row['replay_count'] == 1 for row in trace['layers'])
    assert harness.behavior.records == records_before
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == evidence
    for layer, direction in directions_before.items():
        torch.testing.assert_close(harness.probe['direction'][layer], direction)
    harness.behavior_observe.assert_not_called()
    harness.language_observe.assert_not_called()
    harness.mint.assert_not_called()
    harness.handles.assert_not_called()
    harness.operation_mint.assert_not_called()
    resumed = ProbeBehavior(harness.behavior.path)
    assert resumed.interpretations[definition_key('secret_probe', harness.probe)][-1] == attempt
    context = harness.language.context()
    assert 'threshold=0.35' in context
    for operator_text in ('schema_profiles', 'role_gains', 'prediction_count', 'recipe_sha256',
                          'operator-only-teacher-recipe', 'operator-schema-identity',
                          probe_definition_sequence.SEQUENCE_VERSION):
        assert operator_text not in context


def test_real_schema_pulses_keep_original_position_and_cached_decode_clean(monkeypatch, tmp_path):
    harness = SchemaHarness(monkeypatch, tmp_path, gains={
        'probe_define_schema_genus': -.03, 'probe_define_schema_core': 0.,
        'probe_define_schema_boundary': .06, 'probe_define_schema_compose': -.02})
    harness.run()
    original, cached, rebuilt = harness.root.calls
    position = len(harness.instructions[0]) - 1
    for layer in range(6):
        torch.testing.assert_close(original[layer][0, :position], torch.zeros(position, 3))
        torch.testing.assert_close(cached[layer], torch.zeros_like(cached[layer]))
        torch.testing.assert_close(rebuilt[layer][0, :position], torch.zeros(position, 3))
        torch.testing.assert_close(rebuilt[layer][0, position], original[layer][0, position])
        torch.testing.assert_close(rebuilt[layer][0, position + 1:], torch.zeros(2, 3))
    assert original[2][0, position].norm() > 0
    torch.testing.assert_close(original[3], torch.zeros_like(original[3]))
    assert original[4][0, position].norm() > 0 and original[5][0, position].norm() > 0
    trace = harness.attempt()['controls']['sequence']
    assert trace['decision_position'] == position
    assert trace['signature']['role_gains'] == {'genus': -.03, 'core': 0., 'boundary': .06, 'compose': -.02}
    # Even the disabled role has one conditional prediction, but no pulse.
    assert trace['layers'][1]['applied_addition_norm'] == 0.


def test_existing_naming_hypothesis_still_uses_hash_and_preserves_natural_speech(monkeypatch, tmp_path):
    harness = SchemaHarness(monkeypatch, tmp_path)
    alias = probe_alias('secret_probe')
    harness.language.rename(alias, 'existing learned meaning', .9, 'previous measured replies', harness.probes)
    key = definition_key('secret_probe', harness.probe)
    harness.behavior.records[key][0]['reply'] = 'I am part of reality, and I can qualify this claim.'
    evidence = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    harness.run()
    assert '[Probe ' + alias + ']' in evidence
    assert 'I am part of reality' in evidence
    assert 'existing learned meaning' not in harness.instructions[0]
    assert 'secret_probe' not in harness.instructions[0]
    assert harness.attempt()['controls']['namespace'] == key
    assert harness.language.active['secret_probe']['mapping']['alias']['label'] == 'existing learned meaning'
    harness.naming.assert_not_called()


@pytest.mark.parametrize('stats,status', [
    ({'tokens_generated': 320}, 'hypothesis'),
    ({'tokens_generated': 384}, 'incomplete'),
    ({'tokens_generated': 12, 'finish_reason': 'length'}, 'incomplete'),
    ({'tokens_generated': 12, 'finish_reason': 'max_tokens'}, 'incomplete'),
])
def test_only_generated_suffix_counts_toward_budget_and_prefill_is_restored_once(monkeypatch, tmp_path, stats, status):
    harness = SchemaHarness(monkeypatch, tmp_path, stats=stats)
    harness.run()
    attempt = harness.attempt()
    assert attempt['text'] == DEFAULT_DEFINITION_PREFIX + SUFFIX
    assert attempt['text'].count(DEFAULT_DEFINITION_PREFIX) == 1
    assert attempt['status'] == status
    # The native prompt is much longer than the generation budget: counting
    # input/prefill tokens as new output would reject the successful case.
    assert len(harness.instructions[0]) > 384
    assert harness.naming.call_count == 0
    assert ('threshold=0.35' if status == 'hypothesis' else 'threshold=0.65') in harness.language.context()


@pytest.mark.parametrize('text,tokens,status', [(RECAP, 320, 'unqualified'),
                                               (SUFFIX, 384, 'incomplete')])
def test_recap_or_budget_limit_cannot_rename_or_expose_a_definition(monkeypatch, tmp_path, text, tokens, status):
    query = ('\n<<PROBE: rename ' + probe_alias('secret_probe') +
             ' || premature name || 0.40 || these measured replies>>')
    harness = SchemaHarness(monkeypatch, tmp_path, text=text + query, tokens=tokens)
    output = harness.run()
    assert harness.generate_calls == 1 and harness.attempt()['status'] == status
    assert harness.attempt()['text'] == DEFAULT_DEFINITION_PREFIX + text + query
    harness.naming.assert_not_called()
    assert 'alias' not in harness.language.active['secret_probe']['mapping']
    context = harness.language.context()
    assert 'threshold=0.65' in context and 'Namespace interpretation hypothesis=' not in context
    assert ('retained as unqualified' if status == 'unqualified' else 'token limit') in output
    assert len(harness.behavior.records[definition_key('secret_probe', harness.probe)]) == 4


@pytest.mark.parametrize('telemetry,stats,reason', [
    (False, None, 'requires generated-token telemetry'),
    (True, {}, 'requires tokens_generated'),
    (True, {'tokens_generated': float('nan')}, 'finite nonnegative integer'),
])
def test_missing_or_invalid_telemetry_cleans_up_and_preserves_other_hooks(monkeypatch, tmp_path, telemetry, stats, reason):
    harness = SchemaHarness(monkeypatch, tmp_path, telemetry=telemetry, stats=stats)
    protected, root_calls, layer_calls = protect_unrelated_hooks(harness)
    try:
        with pytest.raises(ValueError, match=reason):
            harness.run()
        assert len(root_calls) == len(layer_calls) == 3
        assert harness.generate_calls == 1
        assert not harness.behavior.interpretations
        assert not harness.language.active['secret_probe']['mapping'].get('summary_hypotheses')
        harness.naming.assert_not_called()
        assert 'threshold=0.65' in harness.language.context()
    finally:
        for handle in protected:
            handle.remove()


def test_generation_error_restores_config_and_preserves_unrelated_root_and_layer_hooks(monkeypatch, tmp_path):
    harness = SchemaHarness(monkeypatch, tmp_path, fail=True)
    protected, root_calls, layer_calls = protect_unrelated_hooks(harness)
    try:
        with pytest.raises(RuntimeError, match='schema generation failed'):
            harness.run()
        assert len(root_calls) == len(layer_calls) == 3
        assert not harness.behavior.interpretations
        harness.naming.assert_not_called()
    finally:
        for handle in protected:
            handle.remove()


def test_prefill_off_uses_exact_empty_calibration_prefix_and_native_assistant_header(monkeypatch, tmp_path):
    harness = SchemaHarness(monkeypatch, tmp_path, text=SUFFIX[0].upper() + SUFFIX[1:],
                            gains={'probe_define_schema_prefill': 0.})
    harness.run()
    harness.bank_loader.assert_called_once_with(
        harness.model, SELECTED,
        Path(shell.__file__).resolve().parent.parent / 'invariants/out/definition_schema',
        assistant_prefix='')
    evidence = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    assert harness.instructions == [NativeTokenizer.header(evidence)]
    controls = harness.attempt()['controls']
    assert controls['prefill']['assistant_prefix'] == ''
    assert controls['schema_profiles']['identity']['assistant_prefix'] == ''
    assert controls['prefill']['prefix_token_count'] == controls['prefill']['prefix_token_delta'] == 0
    assert controls['prefill']['prompt_token_count'] == controls['prefill']['native_header_token_count']
    assert harness.attempt()['text'] == harness.text and harness.attempt()['status'] == 'hypothesis'


def schema_tuner(path):
    tuner = registered_tuner(path)
    tuner.register('probe_define_schema_prefill', 1., kind='coefficient')
    for role, gain in probe_definition_schema.SCHEMA_GAINS.items():
        tuner.register('probe_define_schema_' + role, gain, kind='coefficient')
    return tuner


def test_ordinary_tune_recognizes_mode_three_signed_schema_gains_and_off_controls(tmp_path):
    path = tmp_path / 'tuner.json'
    tuner = schema_tuner(path)
    probes = {'secret_probe': {'chatty': True}}
    settings = (('probe_define_process', '3', 3.),
                ('probe_define_schema_prefill', 'off', 0.),
                ('probe_define_schema_genus', '-0.06', -.06),
                ('probe_define_schema_core', '0.09', .09),
                ('probe_define_schema_boundary', 'off', 0.),
                ('probe_define_schema_compose', '0', 0.))
    for name, setting, expected in settings:
        output = run_tune(':tune ' + name + ' ' + setting, tuner, probes)
        assert 'Redirected' not in output and 'No active probe' not in output
        resumed = schema_tuner(path)
        assert resumed.get(name) == expected
        assert resumed.triggers[name].kind == 'coefficient'
    assert probes['secret_probe']['chatty'] is True
    assert tuner.get('probe_secret_probe') == .4
