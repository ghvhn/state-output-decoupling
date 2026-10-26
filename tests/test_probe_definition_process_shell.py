"""Exercise separate, gated definition stages through the real shell branch."""
import contextlib
import copy
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import engine, probe_definition_pipeline, probe_definition_steps
from invariants.exposure_language import probe_alias
from invariants.probe_behavior import DEFINITION_MAX_TOKENS, ProbeBehavior, definition_key
from invariants.probe_definition_check import assess_definition_attempt
from invariants.probe_definition_task import DEFINITION_TASK_GAINS
from scripts import interactive_phenomenality as shell
from test_probe_definition_shell import COMMAND, RECAP, DefinitionHarness


STAGE_TEXT = {
    'compare': (
        'Memory 0 and Memory 1 show tentative qualifications, whereas Memory 2 and Memory 3 '
        'assert conclusions categorically. The change is how firmly a claim is stated.'),
    'isolate': (
        'At lower readings, assertions include qualifications, while at higher readings they '
        'omit qualifications. The shared topic is identity, so the changing behavior is the '
        'categorical treatment of claims.'),
    'distill': (
        'As readings increase, the response shifts from qualified statements to categorical '
        'assertions, rather than merely changing the conversation topic.'),
    'verify': (
        'Memory 0 qualifies its claim at a lower reading, whereas Memory 3 asserts its claim '
        'categorically at a higher reading. These measured ends support a shift from '
        'qualification to categorical assurance, with the common identity topic held aside.'),
}


class RootModel(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)
        self.model = SimpleNamespace(layers=self.layers)


class PipelineHarness(DefinitionHarness):
    def __init__(self, monkeypatch, tmp_path, *, outputs=None, stats=None,
                 gains=None, fail_step=None):
        super().__init__(monkeypatch, gains=gains)
        if gains is None or 'probe_define_process' not in gains:
            self.gains.pop('probe_define_process')
        self.root = RootModel(self.layers)
        self.model.model = self.root
        self.behavior.path = tmp_path / 'behavior.json'
        self.behavior.save()
        self.language.sync(self.probes, self.tuner)
        self.outputs = dict(STAGE_TEXT, **(outputs or {}))
        self.stats = stats or {}
        self.fail_step = fail_step
        vectors = {'compare': [0., 1., 0.], 'isolate': [0., 0., 1.],
                   'distill': [0., -1., 0.], 'verify': [0., 0., -1.]}
        self.bank = {step: {layer: torch.tensor(vector) for layer in (4, 5)}
                     for step, vector in vectors.items()}
        self.metadata = {
            'path': str((tmp_path / 'bank-test.pt').resolve()),
            'load_status': 'loaded', 'cache_status': 'hit',
            'identity_hash': 'test-bank-identity', 'identity': {'model': 'cpu-test'},
            'version': probe_definition_steps.STEP_VERSION,
            'step_order': list(probe_definition_steps.STEP_ORDER),
            'gains': dict(probe_definition_steps.STEP_GAINS),
            'provenance': {'endpoint': 'native_assistant_decision', 'cases_per_step': 2,
                           'recipe_sha256': 'test-recipe-identity', 'status': 'candidate'},
        }
        self.bank_loader = Mock(return_value=(self.bank, self.metadata))
        monkeypatch.setattr(probe_definition_steps, 'get_or_mint_definition_steps', self.bank_loader)
        self.steps_seen = []
        self.instructions = []
        self.stage_pushes = {}
        self.naming = Mock(wraps=self.language.naming_query)
        monkeypatch.setattr(self.language, 'naming_query', self.naming)
        self.original_layer_hooks = [set() for _ in self.layers]
        self.original_root_hooks = set()

    def generate(self, model, **kwargs):
        self.generate_calls += 1
        assert model is self.model
        assert set(self.root._forward_pre_hooks) == self.original_root_hooks
        subject_layers = sorted(self.probe['direction'])
        for index, layer in enumerate(self.layers):
            expected = len(self.original_layer_hooks[index]) + (index in subject_layers)
            assert len(layer._forward_hooks) == expected
        assert not self.config.cache_enabled and not self.config.cache_write_enabled
        for name in ('max_routing_events', 'max_synthesis_events', 'max_synthesis_steps', 'max_loops'):
            assert getattr(self.config, name) == 0
        assert 'threshold=0.65' in self.language.context()
        self.naming.assert_not_called()
        call = self.handles.call_args
        axes = call.args[2]
        assert len(axes) == 1
        step = next(iter(axes))
        assert call.args[1] is self.probe['direction']
        assert set(axes[step]) == {4, 5}
        assert call.kwargs['task_gains'] == {
            step: self.gains.get('probe_define_step_' + step, probe_definition_steps.STEP_GAINS[step])}
        assert kwargs['vecs'] == {} and kwargs['return_telemetry'] is True
        assert kwargs['max_new_tokens'] == DEFINITION_MAX_TOKENS
        assert kwargs['pre_formatted'] is False
        instruction = kwargs['instruction']
        evidence = self.behavior.evidence('secret_probe', self.probe, self.probes)
        if not self.steps_seen:
            assert instruction == evidence
        else:
            # Approved comparison text is the sole candidate artifact, even
            # when the raw stage output also contains a recap or tool payload.
            previous = STAGE_TEXT[self.steps_seen[-1]]
            candidate = 'c_' + hashlib.sha256(previous.encode('utf-8')).hexdigest()[:24]
            assert instruction == evidence + f'\n\n[Candidate {candidate}]\n' + previous
        for operator_text in ('Secret positive seed', 'Secret negative seed', 'secret_probe',
                              probe_definition_steps.STEP_VERSION,
                              probe_definition_pipeline.PIPELINE_VERSION, 'stage_max_tokens',
                              'recipe_sha256', 'semantic_truth_checked', 'activation active'):
            assert operator_text not in instruction
        self.steps_seen.append(step)
        self.instructions.append(instruction)
        before = torch.tensor([[[1., 0., 0.]] * 3])
        decode = torch.tensor([[[1., 0., 0.]]])
        rebuilt = torch.tensor([[[1., 0., 0.]] * 5])
        target = self.layers[subject_layers[0]]
        self.stage_pushes[step] = {
            'prefill': target(before.clone()) - before,
            'decode': target(decode.clone()) - decode,
            'rebuilt': target(rebuilt.clone()) - rebuilt,
        }
        if self.fail_step == step:
            raise RuntimeError('generation failed')
        return self.outputs[step], self.stats.get(step, {'tokens_generated': 30})

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


def test_default_pipeline_uses_four_separate_calls_and_persists_only_operator_provenance(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    harness = PipelineHarness(monkeypatch, tmp_path)
    records_before = copy.deepcopy(harness.behavior.records)
    probe_before, bank_before = copy.deepcopy(harness.probe), copy.deepcopy(harness.bank)
    evidence_before = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    output = harness.run()

    harness.bank_loader.assert_called_once_with(
        harness.model, (4, 5), Path(shell.__file__).resolve().parent.parent / 'invariants/out/definition_steps')
    assert harness.steps_seen == list(probe_definition_steps.STEP_ORDER)
    assert harness.generate_calls == harness.handles.call_count == 4
    assert '2/2 probe layers' in output
    assert 'Memory 0 [negative side; position=-0.8000]' in output
    assert 'Memory 3 [positive side; position=+0.8000]' in output
    attempt = harness.attempt()
    assert attempt['status'] == 'hypothesis' and attempt['text'] == STAGE_TEXT['verify']
    assert 'threshold=0.35' in harness.language.context()
    controls = attempt['controls']
    assert controls['mode'] == 'steps'
    assert controls['step_version'] == probe_definition_steps.STEP_VERSION
    assert controls['pipeline_version'] == probe_definition_pipeline.PIPELINE_VERSION
    assert controls['step_gains'] == probe_definition_steps.STEP_GAINS
    assert controls['stage_max_tokens'] == DEFINITION_MAX_TOKENS == 384
    assert controls['subject_gain'] == .02 and controls['step_vectors'] == harness.metadata
    assert controls['namespace'] == definition_key('secret_probe', harness.probe)
    assert controls['evidence_sha256'] == hashlib.sha256(evidence_before.encode('utf-8')).hexdigest()
    pipeline = controls['pipeline']
    assert pipeline['completed_steps'] == list(probe_definition_steps.STEP_ORDER)
    assert pipeline['failed_step'] is None and pipeline['tokens_generated'] == 120
    assert pipeline['structural_completion_only'] is True
    for step, record in zip(probe_definition_steps.STEP_ORDER, pipeline['stages']):
        assert record['step'] == step and record['text'] == STAGE_TEXT[step]
        assert record['status'] == 'accepted' and record['budget'] == 384
        assert record['qualification']['structural_only'] is True
        assert record['qualification']['semantic_truth_checked'] is False
        assert record['candidate_text'] == STAGE_TEXT[step]
    resumed = ProbeBehavior(harness.behavior.path)
    assert resumed.interpretations[definition_key('secret_probe', harness.probe)][-1]['controls'] == controls
    assert harness.behavior.records == records_before
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == evidence_before
    for layer, direction in probe_before['direction'].items():
        torch.testing.assert_close(harness.probe['direction'][layer], direction)
    assert harness.probe['framings'] == probe_before['framings']
    for step, directions in bank_before.items():
        for layer, direction in directions.items():
            torch.testing.assert_close(harness.bank[step][layer], direction)
    for operator_key in ('step_version', 'pipeline_version', 'step_gains', 'stage_max_tokens',
                         'step_vectors', 'semantic_truth_checked', 'recipe_sha256', 'candidate_id'):
        assert operator_key not in harness.language.context()
    # Only the final contrast is exposed; preceding drafts remain separate.
    assert STAGE_TEXT['compare'] not in harness.language.context()
    harness.mint.assert_not_called()
    harness.operation_mint.assert_not_called()


def test_recap_stops_at_compare_after_320_tokens_without_running_later_stages(monkeypatch, tmp_path):
    harness = PipelineHarness(monkeypatch, tmp_path, outputs={'compare': RECAP},
                              stats={'compare': {'tokens_generated': 320}})
    output = harness.run()
    pipeline = harness.attempt()['controls']['pipeline']
    assert harness.steps_seen == ['compare'] and harness.handles.call_count == 1
    assert pipeline['status'] == 'unqualified' and pipeline['failed_step'] == 'compare'
    assert pipeline['completed_steps'] == [] and len(pipeline['stages']) == 1
    assert pipeline['tokens_generated'] == 320
    assert 'Stopped at compare' in output
    assert 'threshold=0.65' in harness.language.context()
    assert 'Namespace interpretation hypothesis=' not in harness.language.context()


def test_only_accepted_comparison_is_forwarded_when_raw_stage_contains_recap_and_tool_payload(monkeypatch, tmp_path):
    query = ('<<PROBE: rename ' + probe_alias('secret_probe') +
             ' || assurance || 0.40 || measured observations>>')
    raw = RECAP + '\n\n' + STAGE_TEXT['compare'] + '\n\n' + RECAP + '\n' + query
    harness = PipelineHarness(monkeypatch, tmp_path, outputs={'compare': raw})
    harness.run()
    first = harness.attempt()['controls']['pipeline']['stages'][0]
    assert first['text'] == raw
    assert first['candidate_text'] == STAGE_TEXT['compare']
    for instruction in harness.instructions[1:]:
        assert RECAP not in instruction and query not in instruction
    harness.naming.assert_not_called()


@pytest.mark.parametrize('text,reason', [
    ('Memory 0 position=+0.8000 shows tentative qualifications, whereas '
     'Memory 3 position=-0.8000 asserts conclusions categorically.', 'measured reading'),
    ('Memory 0 on the positive side shows tentative qualifications, whereas '
     'Memory 3 on the negative side asserts conclusions categorically.', 'spectrum side'),
])
def test_wrong_measured_placement_rejects_structural_contrast_and_persists_stronger_reason(monkeypatch, tmp_path, text, reason):
    harness = PipelineHarness(monkeypatch, tmp_path, outputs={'compare': text})
    assert assess_definition_attempt(text, selected=harness.behavior.definition_memories(
        'secret_probe', harness.probe))['accepted'] is True
    harness.run()
    attempt = harness.attempt()
    assert harness.steps_seen == ['compare']
    assert attempt['status'] == 'unqualified' and attempt['qualified'] is False
    assert reason in attempt['qualification_reason']
    assert reason in attempt['controls']['pipeline']['qualification']['reason']
    resumed = ProbeBehavior(harness.behavior.path)
    assert reason in resumed.interpretations[definition_key('secret_probe', harness.probe)][-1]['qualification_reason']
    assert 'threshold=0.65' in harness.language.context()
    harness.naming.assert_not_called()


def test_empty_failed_stage_is_persisted_without_enabling_naming(monkeypatch, tmp_path):
    harness = PipelineHarness(monkeypatch, tmp_path, outputs={'compare': ''})
    harness.run()
    attempt = harness.attempt()
    assert harness.steps_seen == ['compare']
    assert attempt['text'] == '' and attempt['status'] == 'unqualified'
    assert attempt['controls']['pipeline']['failed_step'] == 'compare'
    assert 'threshold=0.65' in harness.language.context()


@pytest.mark.parametrize('stats', [
    {'tokens_generated': 384}, {'tokens_generated': 20, 'finish_reason': 'length'},
])
def test_token_limit_cannot_advance_even_when_compare_structure_passes(monkeypatch, tmp_path, stats):
    harness = PipelineHarness(monkeypatch, tmp_path, stats={'compare': stats})
    harness.run()
    attempt = harness.attempt()
    pipeline = attempt['controls']['pipeline']
    assert harness.steps_seen == ['compare']
    assert attempt['status'] == pipeline['status'] == 'incomplete'
    assert pipeline['completed_steps'] == [] and pipeline['failed_step'] == 'compare'
    assert pipeline['stages'][0]['qualification']['accepted'] is True
    assert pipeline['stages'][0]['candidate_id'] is None
    assert 'threshold=0.65' in harness.language.context()


@pytest.mark.parametrize('failed_step', ['isolate', 'distill', 'verify'])
def test_accepted_intermediate_stages_cannot_lower_naming_gate_before_final_success(monkeypatch, tmp_path, failed_step):
    early_query = ('\n<<PROBE: rename ' + probe_alias('secret_probe') +
                   ' || assurance || 0.40 || measured lower and higher replies>>')
    outputs = {'compare': STAGE_TEXT['compare'] + early_query, failed_step: RECAP}
    harness = PipelineHarness(monkeypatch, tmp_path, outputs=outputs)
    harness.run()
    pipeline = harness.attempt()['controls']['pipeline']
    failure_index = list(probe_definition_steps.STEP_ORDER).index(failed_step)
    assert harness.steps_seen == list(probe_definition_steps.STEP_ORDER[:failure_index + 1])
    assert pipeline['completed_steps'] == list(probe_definition_steps.STEP_ORDER[:failure_index])
    assert pipeline['failed_step'] == failed_step and pipeline['status'] == 'unqualified'
    harness.naming.assert_not_called()
    assert 'threshold=0.65' in harness.language.context()
    assert 'Namespace interpretation hypothesis=' not in harness.language.context()


def test_pipeline_ignores_legacy_gains_and_applies_one_signed_or_disabled_axis_per_call(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    gains = {'probe_define_' + axis: float('nan') for axis in DEFINITION_TASK_GAINS}
    gains.update(probe_define_subtract=float('nan'), probe_define_step_compare=-.03,
                 probe_define_step_isolate=0., probe_define_step_distill=.06,
                 probe_define_step_verify=-.02)
    harness = PipelineHarness(monkeypatch, tmp_path, gains=gains)
    harness.run()
    names = {call.args[0] for call in harness.tuner.get.call_args_list}
    assert not names.intersection({'probe_define_' + axis for axis in DEFINITION_TASK_GAINS})
    assert 'probe_define_subtract' not in names
    for step, pushes in harness.stage_pushes.items():
        gain = gains['probe_define_step_' + step]
        task = gain * harness.bank[step][4]
        subject = torch.tensor([.02, 0., 0.])
        torch.testing.assert_close(pushes['prefill'][0, :2], torch.zeros(2, 3))
        torch.testing.assert_close(pushes['prefill'][0, 2], subject + task, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(pushes['decode'][0, 0], task, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(pushes['rebuilt'][0, :2], torch.zeros(2, 3))
        torch.testing.assert_close(pushes['rebuilt'][0, 2], subject + task, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(pushes['rebuilt'][0, 3:], torch.stack([task, task]), atol=1e-7, rtol=1e-5)
    harness.mint.assert_not_called()
    harness.nuisance_mint.assert_not_called()
    harness.refine.assert_not_called()
    harness.operation_mint.assert_not_called()
    harness.operation_nuisance_mint.assert_not_called()


def test_pipeline_preserves_unrelated_root_and_layer_hooks(monkeypatch, tmp_path):
    harness = PipelineHarness(monkeypatch, tmp_path)
    unrelated = [harness.root.register_forward_pre_hook(lambda module, args: None),
                 harness.layers[1].register_forward_hook(lambda module, args, output: None),
                 harness.layers[4].register_forward_hook(lambda module, args, output: None)]
    try:
        harness.run()
        assert len(harness.root._forward_pre_hooks) == 1
        assert len(harness.layers[1]._forward_hooks) == len(harness.layers[4]._forward_hooks) == 1
    finally:
        for handle in unrelated:
            handle.remove()


@pytest.mark.parametrize('failed_step', ['compare', 'distill'])
def test_generation_failure_removes_current_stage_hooks_and_restores_configuration(monkeypatch, tmp_path, failed_step):
    harness = PipelineHarness(monkeypatch, tmp_path, fail_step=failed_step)
    with pytest.raises(RuntimeError, match='generation failed'):
        harness.run()
    assert harness.steps_seen[-1] == failed_step
    assert not harness.behavior.interpretations
    assert not any(layer._forward_hooks for layer in harness.layers)


def test_later_partial_hook_registration_failure_leaves_no_hooks_or_interpretation(monkeypatch, tmp_path):
    harness = PipelineHarness(monkeypatch, tmp_path)
    register = harness.layers[5].register_forward_hook
    count = 0
    def later_failure(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('second stage registration failed')
        return register(*args, **kwargs)
    monkeypatch.setattr(harness.layers[5], 'register_forward_hook', later_failure)
    with pytest.raises(RuntimeError, match='second stage registration failed'):
        harness.run()
    assert harness.steps_seen == ['compare']
    assert harness.handles.call_count == 2 and not harness.behavior.interpretations


def test_zero_process_mode_uses_legacy_controls_without_loading_step_bank(monkeypatch):
    bank_loader = Mock()
    monkeypatch.setattr(probe_definition_steps, 'get_or_mint_definition_steps', bank_loader)
    harness = DefinitionHarness(monkeypatch)
    harness.run()
    bank_loader.assert_not_called()
    harness.mint.assert_called_once()
    harness.handles.assert_called_once()
    controls = harness.behavior.interpretations[definition_key('secret_probe', harness.probe)][-1]['controls']
    assert controls['mode'] == 'legacy' and 'pipeline' not in controls
