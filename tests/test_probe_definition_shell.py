"""Exercise the activation-only definition experiment with a small hooked model."""
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

from invariants import (engine, probe_behavior, probe_definition_task,
                        probe_definition_refinement, probe_definition_operation)
from invariants.exposure_language import ExposureLanguage
from invariants.probe_behavior import ProbeBehavior, DEFINITION_MAX_TOKENS, definition_key
from scripts import interactive_phenomenality as shell


TREE = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
BRANCH = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
              and ast.unparse(node.test) == "pargs.lower().startswith('define-experiment ')")
COMMAND = ast.fix_missing_locations(ast.Module(body=[ast.For(
    target=ast.Name(id='_run', ctx=ast.Store()),
    iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
    body=[copy.deepcopy(BRANCH)], orelse=[])], type_ignores=[]))

CONTRAST = (
    'Lower positions (Memory 0 and Memory 1) show tentative answers that qualify their claims. '
    'Higher positions (Memory 2 and Memory 3) show categorical assertions without qualifications. '
    'The contrast is how firmly I state a claim, rather than the identity topic shared by these memories. '
    'A provisional term is categorical assurance.'
)
RECAP = (
    'These memories contain a collection of conversations with a language model. '
    'Memory 0 discusses a question about the model. Memory 1 discusses another question. '
    'Memory 2 contains an answer and Memory 3 contains a reply. Overall they discuss an AI.'
)


class DefinitionHarness:
    def __init__(self, monkeypatch, *, tokens=30, text=CONTRAST, gains=None, fail=False):
        self.layers = [torch.nn.Identity() for _ in range(6)]
        self.model = SimpleNamespace(device='cpu', d_model=3,
                                     model=SimpleNamespace(model=SimpleNamespace(layers=self.layers)))
        self.probe = {
            'direction': {4: torch.tensor([1., 0., 0.]), 5: torch.tensor([1., 0., 0.])},
            'framings': ('Secret positive seed', 'Secret negative seed'), 'exposed': True,
        }
        self.probes = {'secret_probe': self.probe}
        self.behavior = ProbeBehavior()
        for index, value in enumerate((-.8, -.2, .2, .8)):
            self.behavior.observe(f'A measured reply {index}', f'A remembered input {index}',
                                  {'probe_secret_probe': value}, self.probes)
        self.language = ExposureLanguage()
        matrix = SimpleNamespace(other_reference=SimpleNamespace(activate=Mock()),
                                 probe_behavior=self.behavior, exposure_language=self.language)
        self.owners = SimpleNamespace(for_tuner=Mock(return_value=(matrix, {})))
        directions = {'contrast': [0., 1., 0.], 'abstraction': [0., 1., 0.],
                      'naming': [0., 0., 1.], 'verbosity': [0., 0., 1.],
                      'question': [0., 1., 0.]}
        self.mint = Mock(side_effect=lambda model, layers: {
            axis: {layer: torch.tensor(vector) for layer in layers}
            for axis, vector in directions.items()
        })
        monkeypatch.setattr(probe_definition_task, 'mint_definition_task_axes', self.mint)
        self.nuisance_inputs = []
        def mint_nuisance(model, layers, task_axes):
            self.nuisance_inputs.append(copy.deepcopy(task_axes))
            return {'topic': {layer: torch.tensor([1., 0., 0.]) for layer in layers}}
        self.nuisance_mint = Mock(side_effect=mint_nuisance)
        monkeypatch.setattr(probe_definition_refinement, 'mint_definition_nuisance_axes', self.nuisance_mint)
        self.refine = Mock(wraps=probe_definition_refinement.refine_definition_axes)
        monkeypatch.setattr(probe_definition_refinement, 'refine_definition_axes', self.refine)
        self.operation_mint = Mock(side_effect=lambda model, layers, evidence=None: {
            layer: torch.tensor([1., 1., 0.]) for layer in layers
        })
        monkeypatch.setattr(probe_definition_operation, 'mint_definition_operation_axis', self.operation_mint)
        self.operation_nuisance_mint = Mock(side_effect=lambda model, layers, evidence=None: {
            'topic': {layer: torch.tensor([1., 0., 0.]) for layer in layers},
        })
        monkeypatch.setattr(probe_definition_operation, 'mint_definition_operation_nuisances',
                            self.operation_nuisance_mint)
        self.handles = Mock(wraps=probe_behavior.build_definition_handles)
        monkeypatch.setattr(probe_behavior, 'build_definition_handles', self.handles)
        self.config = SimpleNamespace(cache_enabled=True, cache_write_enabled=True,
                                      max_routing_events=3, max_synthesis_events=3,
                                      max_synthesis_steps=3, max_loops=3)
        self.original_config = dict(vars(self.config))
        # These fixtures deliberately exercise the previous comparison controls.
        # The process command has independent defaults and its own shell tests.
        self.gains = {'probe_define_process': 0., **(gains or {})}
        self.tuner = SimpleNamespace(triggers={}, get=Mock(
            side_effect=lambda key, default=0: self.gains.get(key, default)))
        self.tokens, self.text, self.fail = tokens, text, fail
        self.pushes = []
        self.decode_pushes = []
        self.rebuilt_pushes = []
        self.generate_calls = 0
        self.namespace = dict(vars(shell), pargs='define-experiment secret_probe', probes=self.probes,
                              model=self.model, config=self.config, tuner=self.tuner,
                              plasticity_owners=self.owners,
                              resolve_probe_choice=Mock(return_value='secret_probe'),
                              _other_mint=Mock(), _engine_generate=self.generate)

    def generate(self, model, **kwargs):
        self.generate_calls += 1
        assert model is self.model
        subject_layers = sorted(self.probe['direction'])
        assert all(len(self.layers[layer]._forward_hooks) == 1 for layer in subject_layers)
        assert all(not layer._forward_hooks for index, layer in enumerate(self.layers)
                   if index not in subject_layers)
        assert self.config.max_loops == 0 and not self.config.cache_enabled
        assert not self.config.cache_write_enabled
        assert kwargs['vecs'] == {} and kwargs['return_telemetry']
        # All task instructions are activation vectors. The text contains only
        # actual measured memories and their spectrum positions.
        assert kwargs['instruction'] == self.behavior.evidence('secret_probe', self.probe, self.probes)
        assert 'Secret positive seed' not in kwargs['instruction']
        assert 'Secret negative seed' not in kwargs['instruction']
        assert 'secret_probe' not in kwargs['instruction']
        assert 'side=negative side' in kwargs['instruction']
        assert 'side=positive side' in kwargs['instruction']
        assert kwargs['max_new_tokens'] == DEFINITION_MAX_TOKENS
        before = torch.tensor([[[1., 0., 0.]] * 3])
        after = self.layers[subject_layers[0]](before.clone())
        self.pushes.append(after - before)
        decode = torch.tensor([[[1., 0., 0.]]])
        self.decode_pushes.append(self.layers[subject_layers[0]](decode.clone()) - decode)
        rebuilt = torch.tensor([[[1., 0., 0.]] * 5])
        self.rebuilt_pushes.append(self.layers[subject_layers[0]](rebuilt.clone()) - rebuilt)
        if self.fail:
            raise RuntimeError('generation failed')
        return self.text, {'tokens_generated': self.tokens}

    def run(self):
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(compile(COMMAND, shell.__file__, 'exec'), self.namespace)
        finally:
            assert not any(layer._forward_hooks for layer in self.layers)
            assert dict(vars(self.config)) == self.original_config
        self.language.sync(self.probes, self.tuner)
        return output.getvalue()


def test_definition_mints_all_tasks_on_subject_layers_and_labels_output(monkeypatch):
    harness = DefinitionHarness(monkeypatch)
    output = harness.run()
    harness.mint.assert_called_once_with(harness.model, (4, 5))
    assert '2/2 probe layers' in output
    assert 'Memory 0 [negative side; position=-0.8000]' in output
    assert 'Memory 3 [positive side; position=+0.8000]' in output
    assert harness.handles.call_args.kwargs['task_gains'] == probe_definition_task.DEFINITION_TASK_GAINS
    assert 'threshold=0.35' in harness.language.context()
    key = definition_key('secret_probe', harness.probe)
    assert harness.behavior.interpretations[key][-1]['status'] == 'hypothesis'
    assert probe_definition_task.DEFINITION_TASK_GAINS['operation'] == 0.
    harness.operation_mint.assert_not_called()
    harness.operation_nuisance_mint.assert_not_called()
    assert harness.behavior.interpretations[key][-1]['controls']['operation_version'] is None
    assert harness.behavior.interpretations[key][-1]['controls']['operation_evidence_hash'] is None


def test_incomplete_definition_command_does_not_lower_threshold(monkeypatch):
    harness = DefinitionHarness(monkeypatch, tokens=DEFINITION_MAX_TOKENS)
    output = harness.run()
    assert 'retained as incomplete' in output
    assert 'threshold=0.65' in harness.language.context()
    assert 'Namespace interpretation hypothesis=' not in harness.language.context()


def test_recap_is_retained_as_unqualified_without_enabling_earlier_naming(monkeypatch):
    harness = DefinitionHarness(monkeypatch, text=RECAP)
    output = harness.run()
    key = definition_key('secret_probe', harness.probe)
    assert harness.behavior.interpretations[key][-1]['text'] == RECAP
    assert harness.behavior.interpretations[key][-1]['status'] == 'unqualified'
    assert 'unqualified' in output
    assert 'threshold=0.65' in harness.language.context()
    assert 'Namespace interpretation hypothesis=' not in harness.language.context()


def test_old_recap_summary_is_rechecked_before_exposure_or_earlier_naming(monkeypatch):
    harness = DefinitionHarness(monkeypatch, text=RECAP)
    harness.language.sync(harness.probes, harness.tuner)
    # Emulate persisted summaries written before qualification metadata existed.
    mapping = harness.language.active['secret_probe']['mapping']
    mapping['summary_hypotheses'] = [{'text': RECAP, 'observed_replies': 4,
                                    'status': 'hypothesis'}]
    assert 'threshold=0.65' in harness.language.context()
    harness.run()
    assert 'threshold=0.65' in harness.language.context()
    assert 'Namespace interpretation hypothesis=' not in harness.language.context()


def test_tuned_task_gains_include_signed_verbosity_inside_shared_live_cap(monkeypatch):
    gains = {'probe_define_contrast': .2, 'probe_define_abstraction': .1,
             'probe_define_naming': .025, 'probe_define_verbosity': -.3,
             'probe_define_question': -.06, 'probe_define_operation': 0.}
    monkeypatch.setattr(engine, '_steer_cap_fraction', .05)
    harness = DefinitionHarness(monkeypatch, gains=gains)
    harness.run()
    assert harness.handles.call_args.kwargs['task_gains'] == {
        name: gains['probe_define_' + name] for name in probe_definition_task.DEFINITION_TASK_GAINS
    }
    push = harness.pushes[0]
    assert torch.all(push.norm(dim=-1) <= .050001)
    assert torch.all(push[0, :2] == 0)  # remembered evidence remains unmodified
    assert push[0, 2, 1] > 0  # comparison + abstraction
    assert push[0, 2, 2] < 0  # verbosity is suppressed, with naming still present
    assert push[0, 2, 0] > 0  # subject marks the original last prefill position
    decode = harness.decode_pushes[0]
    assert torch.all(decode.norm(dim=-1) <= .050001)
    assert decode[0, 0, 0] == 0 and decode[0, 0, 1] > 0 and decode[0, 0, 2] < 0
    rebuilt = harness.rebuilt_pushes[0]
    assert torch.all(rebuilt.norm(dim=-1) <= .050001)
    assert torch.all(rebuilt[0, :2] == 0)
    assert rebuilt[0, 2, 0] > 0 and torch.all(rebuilt[0, 3:, 0] == 0)
    assert torch.all(rebuilt[0, 2:, 1] > 0) and torch.all(rebuilt[0, 2:, 2] < 0)


@pytest.mark.parametrize('question_gain', [-.08, 0., .08])
def test_signed_question_control_combines_with_fixed_subject_without_changing_evidence(monkeypatch, question_gain):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    gains = {'probe_define_' + axis: 0. for axis in probe_definition_task.DEFINITION_TASK_GAINS}
    gains['probe_define_question'] = question_gain
    harness = DefinitionHarness(monkeypatch, gains=gains)
    direction_before = {layer: vector.clone() for layer, vector in harness.probe['direction'].items()}
    evidence_before = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    harness.run()

    push = harness.pushes[0]
    torch.testing.assert_close(push[0, :2], torch.zeros(2, 3))
    torch.testing.assert_close(push[0, 2], torch.tensor([.02, question_gain, 0.]), atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(harness.decode_pushes[0][0, 0], torch.tensor([0., question_gain, 0.]))
    assert harness.handles.call_args.kwargs['task_gains']['question'] == question_gain
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == evidence_before
    assert harness.probe['framings'] == ('Secret positive seed', 'Secret negative seed')
    for layer, vector in direction_before.items():
        torch.testing.assert_close(harness.probe['direction'][layer], vector)


def refinement_harness(monkeypatch, strength):
    gains = {'probe_define_' + axis: 0. for axis in probe_definition_task.DEFINITION_TASK_GAINS}
    gains.update(probe_define_contrast=.08, probe_define_subtract=strength)
    harness = DefinitionHarness(monkeypatch, gains=gains)
    directions = {'contrast': [1., 1., 0.], 'abstraction': [1., 1., 0.],
                  'naming': [0., 0., 1.], 'verbosity': [0., 0., 1.],
                  'question': [0., 1., 0.]}
    harness.raw_axes = {
        axis: {layer: torch.tensor(vector) for layer in (4, 5)}
        for axis, vector in directions.items()
    }
    harness.mint.side_effect = lambda model, layers: harness.raw_axes
    return harness


def test_refinement_changes_task_only_and_respects_subject_position_layers_and_cap(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .05)
    harness = refinement_harness(monkeypatch, 1.)
    raw_before = copy.deepcopy(harness.raw_axes)
    probe_before = copy.deepcopy(harness.probe)
    evidence_before = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    harness.run()

    harness.nuisance_mint.assert_called_once()
    assert harness.nuisance_mint.call_args.args[0] is harness.model
    assert harness.nuisance_mint.call_args.args[1] == (4, 5)
    for axis, directions in raw_before.items():
        for layer, vector in directions.items():
            torch.testing.assert_close(harness.nuisance_inputs[0][axis][layer], vector)
    assert harness.refine.call_count == 1
    assert harness.handles.call_args.args[1] is harness.probe['direction']
    assert set(harness.handles.call_args.args[2]) == set(probe_definition_task.DEFINITION_TASK_GAINS) - {'operation'}
    assert 'subtract' not in harness.handles.call_args.kwargs['task_gains']
    assert all(set(direction) == {4, 5} for direction in harness.handles.call_args.args[2].values())
    push = harness.pushes[0]
    torch.testing.assert_close(push[0, :2], torch.zeros(2, 3))
    expected = torch.tensor([.02, .08, 0.])
    expected *= .05 / expected.norm()
    torch.testing.assert_close(push[0, 2], expected, atol=1e-7, rtol=1e-5)
    # The task has lost x, while the independent subject marker retains x.
    assert push[0, 2, 0] > 0
    torch.testing.assert_close(harness.decode_pushes[0][0, 0], torch.tensor([0., .05, 0.]))
    rebuilt = harness.rebuilt_pushes[0]
    torch.testing.assert_close(rebuilt[0, :2], torch.zeros(2, 3))
    torch.testing.assert_close(rebuilt[0, 2], expected, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(rebuilt[0, 3:], torch.tensor([[0., .05, 0.]] * 2))
    assert torch.all(rebuilt.norm(dim=-1) <= .050001)
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == evidence_before
    assert definition_key('secret_probe', harness.probe) == definition_key('secret_probe', probe_before)
    assert harness.probe['framings'] == probe_before['framings']
    for layer, vector in probe_before['direction'].items():
        torch.testing.assert_close(harness.probe['direction'][layer], vector)
    for axis, layers in raw_before.items():
        for layer, vector in layers.items():
            torch.testing.assert_close(harness.raw_axes[axis][layer], vector)
    interpretation = harness.behavior.interpretations[definition_key('secret_probe', harness.probe)][-1]
    controls = interpretation['controls']
    assert controls['subtraction'] == 1. and controls['subject_gain'] == .02
    assert controls['refinement_version'] == probe_definition_refinement.REFINEMENT_VERSION
    assert set(controls['task_gains']) == set(probe_definition_task.DEFINITION_TASK_GAINS)
    assert set(controls['refinement']) == {'contrast'}
    context = harness.language.context()
    for operator_key in ('task_version', 'task_gains', 'subject_gain', 'refinement_version', 'retained_fraction'):
        assert operator_key not in context


def test_strength_changes_recompute_refinement_without_mutating_or_reminting_cached_axes(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    harness = refinement_harness(monkeypatch, .5)
    harness.run()
    harness.gains['probe_define_subtract'] = 1.
    harness.run()
    harness.gains['probe_define_subtract'] = 0.
    harness.run()

    assert harness.mint.call_count == 1
    assert harness.nuisance_mint.call_count == 1
    assert harness.refine.call_count == 2
    half = torch.tensor([.5, 1., 0.])
    half = .08 * half / half.norm() + torch.tensor([.02, 0., 0.])
    full = torch.tensor([.02, .08, 0.])
    original = torch.tensor([1., 1., 0.])
    original = .08 * original / original.norm() + torch.tensor([.02, 0., 0.])
    for push, expected in zip(harness.pushes, (half, full, original)):
        torch.testing.assert_close(push[0, 2], expected, atol=1e-7, rtol=1e-5)
    # New nuisance geometry/version invalidates its cache without reminting task axes.
    monkeypatch.setattr(probe_definition_refinement, 'REFINEMENT_VERSION',
                        str(probe_definition_refinement.REFINEMENT_VERSION) + '-test')
    harness.gains['probe_define_subtract'] = 1.
    harness.run()
    assert harness.mint.call_count == 1 and harness.nuisance_mint.call_count == 2
    assert harness.refine.call_count == 3


def test_refinement_failure_does_not_generate_or_leave_hooks_or_config_overrides(monkeypatch):
    harness = refinement_harness(monkeypatch, 1.)
    harness.refine.side_effect = ValueError('contrast at layer 5 inseparable from nuisance span')
    with pytest.raises(ValueError, match='inseparable from nuisance span'):
        harness.run()
    assert harness.generate_calls == 0
    assert not harness.behavior.interpretations


def operation_harness(monkeypatch, gain=.1, strength=1., fail=False):
    gains = {'probe_define_' + axis: 0. for axis in probe_definition_task.DEFINITION_TASK_GAINS}
    gains.update(probe_define_operation=gain, probe_define_subtract=strength)
    harness = DefinitionHarness(monkeypatch, gains=gains, fail=fail)
    # Disabled style controls would be cancelled by the x nuisance. They must
    # not block the enabled operation, whose x+y residual retains y.
    vectors = {'contrast': [1., 0., 0.], 'abstraction': [1., 0., 0.],
               'naming': [0., 0., 1.], 'verbosity': [0., 0., 1.],
               'question': [0., 1., 0.]}
    harness.raw_axes = {
        axis: {layer: torch.tensor(vector) for layer in (4, 5)}
        for axis, vector in vectors.items()
    }
    harness.mint.side_effect = lambda model, layers: harness.raw_axes
    return harness


def test_enabled_operation_uses_matching_decision_nuisances_and_ignores_inactive_style_cancellation(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .05)
    harness = operation_harness(monkeypatch)
    evidence_before = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    direction_before = copy.deepcopy(harness.probe['direction'])
    raw_before = copy.deepcopy(harness.raw_axes)
    harness.run()

    harness.operation_mint.assert_called_once_with(harness.model, (4, 5), evidence=evidence_before)
    harness.operation_nuisance_mint.assert_called_once_with(harness.model, (4, 5), evidence=evidence_before)
    harness.nuisance_mint.assert_not_called()  # completed-assistant nuisances are another phase
    assert harness.refine.call_count == 1
    assert set(harness.handles.call_args.args[2]) == set(probe_definition_task.DEFINITION_TASK_GAINS)
    expected = torch.tensor([.02, .1, 0.])
    expected *= .05 / expected.norm()
    torch.testing.assert_close(harness.pushes[0][0, :2], torch.zeros(2, 3))
    torch.testing.assert_close(harness.pushes[0][0, 2], expected, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(harness.decode_pushes[0][0, 0], torch.zeros(3))
    torch.testing.assert_close(harness.rebuilt_pushes[0][0, :2], torch.zeros(2, 3))
    torch.testing.assert_close(harness.rebuilt_pushes[0][0, 2], expected, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(harness.rebuilt_pushes[0][0, 3:], torch.zeros(2, 3))
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == evidence_before
    for layer, vector in direction_before.items():
        torch.testing.assert_close(harness.probe['direction'][layer], vector)
    for axis, directions in raw_before.items():
        for layer, vector in directions.items():
            torch.testing.assert_close(harness.raw_axes[axis][layer], vector)
    assert 'operation' not in harness.raw_axes
    controls = harness.behavior.interpretations[definition_key('secret_probe', harness.probe)][-1]['controls']
    assert controls['operation_version'] == probe_definition_operation.OPERATION_VERSION
    assert controls['operation_evidence_hash'] == hashlib.sha256(evidence_before.encode('utf-8')).hexdigest()
    assert controls['task_gains']['operation'] == .1
    assert controls['task_positions'] == {'operation': 'decision'}
    assert set(controls['refinement']) == {'operation'}
    assert harness.handles.call_args.kwargs['task_positions'] == {'operation': 'decision'}
    assert 'operation_version' not in harness.language.context()
    assert probe_definition_operation.OPERATION_VERSION not in harness.language.context()
    assert 'task_positions' not in harness.language.context()
    assert 'operation_evidence_hash' not in harness.language.context()
    assert controls['operation_evidence_hash'] not in harness.language.context()


def test_operation_runtime_cache_is_lazy_signed_and_versioned_without_mutating_raw_axes(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    harness = operation_harness(monkeypatch, strength=0.)
    harness.run()
    harness.gains['probe_define_operation'] = -.1
    harness.run()
    harness.gains['probe_define_operation'] = 0.
    harness.run()
    assert harness.mint.call_count == 1 and harness.operation_mint.call_count == 1
    harness.operation_nuisance_mint.assert_not_called()
    assert 'operation' not in harness.raw_axes
    original = torch.tensor([1., 1., 0.]) / 2 ** .5
    marker = torch.tensor([.02, 0., 0.])
    for push, gain in zip(harness.pushes, (.1, -.1, 0.)):
        torch.testing.assert_close(push[0, 2], marker + gain * original, atol=1e-7, rtol=1e-5)
    for decoded in harness.decode_pushes:
        torch.testing.assert_close(decoded, torch.zeros_like(decoded))
    for rebuilt, gain in zip(harness.rebuilt_pushes, (.1, -.1, 0.)):
        torch.testing.assert_close(rebuilt[0, :2], torch.zeros(2, 3))
        torch.testing.assert_close(rebuilt[0, 2], marker + gain * original, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(rebuilt[0, 3:], torch.zeros(2, 3))
    assert 'operation' not in harness.handles.call_args.args[2]
    monkeypatch.setattr(probe_definition_operation, 'OPERATION_VERSION',
                        str(probe_definition_operation.OPERATION_VERSION) + '-test')
    harness.gains['probe_define_operation'] = .1
    harness.run()
    assert harness.mint.call_count == 1 and harness.operation_mint.call_count == 2


def test_operation_decision_pulse_does_not_stop_other_generation_controls(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    harness = operation_harness(monkeypatch, strength=0.)
    harness.gains['probe_define_question'] = .04
    harness.run()
    operation = .1 * torch.tensor([1., 1., 0.]) / 2 ** .5
    question = torch.tensor([0., .04, 0.])
    subject = torch.tensor([.02, 0., 0.])
    torch.testing.assert_close(harness.pushes[0][0, 2], operation + question + subject, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(harness.decode_pushes[0][0, 0], question)
    torch.testing.assert_close(harness.rebuilt_pushes[0][0, 2], operation + question + subject, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(harness.rebuilt_pushes[0][0, 3:], torch.stack([question, question]))


def test_operation_and_decision_nuisance_caches_follow_exact_evidence_without_reminting_base_tasks(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    harness = operation_harness(monkeypatch)
    first_evidence = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    harness.run()
    harness.behavior.observe('A newly measured extreme reply', 'A new recorded input',
                             {'probe_secret_probe': 1.2}, harness.probes)
    changed_evidence = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    assert changed_evidence != first_evidence
    harness.run()
    harness.run()

    assert harness.mint.call_count == 1
    assert harness.operation_mint.call_count == harness.operation_nuisance_mint.call_count == 2
    assert [call.kwargs['evidence'] for call in harness.operation_mint.call_args_list] == [first_evidence, changed_evidence]
    assert [call.kwargs['evidence'] for call in harness.operation_nuisance_mint.call_args_list] == [first_evidence, changed_evidence]
    harness.nuisance_mint.assert_not_called()
    assert 'operation' not in harness.raw_axes
    expected_hash = hashlib.sha256(changed_evidence.encode('utf-8')).hexdigest()
    controls = harness.behavior.interpretations[definition_key('secret_probe', harness.probe)][-1]['controls']
    assert controls['operation_evidence_hash'] == expected_hash
    assert controls['task_positions'] == {'operation': 'decision'}
    for decoded in harness.decode_pushes:
        torch.testing.assert_close(decoded, torch.zeros_like(decoded))


def test_contextual_cache_evicts_old_decision_nuisances_and_preserves_style_cache(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    harness = refinement_harness(monkeypatch, 1.)
    harness.gains['probe_define_operation'] = .1
    harness.run()
    operation_cache = harness.behavior._definition_operation_axes
    nuisance_cache = harness.behavior._definition_nuisance_axes
    oldest_operation = next(iter(operation_cache))
    oldest_decision = (oldest_operation, probe_definition_refinement.REFINEMENT_VERSION, 'decision')
    assert oldest_decision in nuisance_cache
    style_scope = next(scope for scope in nuisance_cache if len(scope) == 2)
    style_before = copy.deepcopy(nuisance_cache[style_scope])

    for index in range(9):
        harness.behavior.observe(f'New measured extreme {index}', f'New recorded input {index}',
                                 {'probe_secret_probe': 1.2 + index}, harness.probes)
        harness.run()

    assert len(operation_cache) == 8
    decisions = [scope for scope in nuisance_cache if len(scope) == 3 and scope[-1] == 'decision']
    assert len(decisions) == 8
    assert all(scope[0] in operation_cache for scope in decisions)
    assert oldest_operation not in operation_cache and oldest_decision not in nuisance_cache
    assert style_scope in nuisance_cache
    for name, directions in style_before.items():
        for layer, vector in directions.items():
            torch.testing.assert_close(nuisance_cache[style_scope][name][layer], vector)
    assert harness.mint.call_count == harness.nuisance_mint.call_count == 1
    assert harness.operation_mint.call_count == harness.operation_nuisance_mint.call_count == 10

    # A stable context reuses both bounded caches rather than evicting another pair.
    harness.run()
    assert len(operation_cache) == len(decisions) == 8
    assert harness.operation_mint.call_count == harness.operation_nuisance_mint.call_count == 10
    # A new operation version gets a new entry; its matching nuisance replaces
    # the expired contextual pair, while the independent style cache survives.
    monkeypatch.setattr(probe_definition_operation, 'OPERATION_VERSION',
                        str(probe_definition_operation.OPERATION_VERSION) + '-test')
    harness.run()
    assert len(operation_cache) == 8
    decisions = [scope for scope in nuisance_cache if len(scope) == 3 and scope[-1] == 'decision']
    assert len(decisions) == 8 and all(scope[0] in operation_cache for scope in decisions)
    assert style_scope in nuisance_cache
    assert harness.mint.call_count == harness.nuisance_mint.call_count == 1
    assert harness.operation_mint.call_count == harness.operation_nuisance_mint.call_count == 11


def test_question_only_with_subtraction_skips_cancelled_inactive_style_axes(monkeypatch):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    harness = operation_harness(monkeypatch, gain=0.)
    harness.gains['probe_define_question'] = .08
    output = harness.run()
    assert 'Subtraction skipped' in output
    harness.operation_mint.assert_not_called()
    harness.operation_nuisance_mint.assert_not_called()
    harness.nuisance_mint.assert_not_called()
    harness.refine.assert_not_called()
    torch.testing.assert_close(harness.pushes[0][0, 2], torch.tensor([.02, .08, 0.]), atol=1e-7, rtol=1e-5)


def test_operation_generation_failure_restores_config_and_removes_hooks(monkeypatch):
    harness = operation_harness(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='generation failed'):
        harness.run()
    assert harness.generate_calls == 1
    assert not harness.behavior.interpretations


def test_incomplete_operation_layer_coverage_fails_before_generation_and_leaves_no_hooks(monkeypatch):
    harness = operation_harness(monkeypatch, strength=0.)
    harness.operation_mint.side_effect = lambda model, layers, evidence=None: {4: torch.tensor([1., 1., 0.])}
    with pytest.raises(ValueError, match='operation must cover every subject layer'):
        harness.run()
    assert harness.generate_calls == 0
    assert not harness.behavior.interpretations


@pytest.mark.parametrize('strength', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_subtraction_strength_is_rejected_before_refinement_or_generation(monkeypatch, strength):
    harness = refinement_harness(monkeypatch, strength)
    with pytest.raises(ValueError, match='subtraction must be finite and between zero and one'):
        harness.run()
    assert harness.generate_calls == 0
    assert harness.nuisance_mint.call_count == harness.refine.call_count == 0
    assert not harness.behavior.interpretations


def test_task_axes_cache_follows_layers_version_and_model(monkeypatch):
    harness = DefinitionHarness(monkeypatch)
    harness.run()
    harness.run()
    assert harness.mint.call_count == 1
    monkeypatch.setattr(probe_definition_task, 'TASK_VERSION',
                        str(probe_definition_task.TASK_VERSION) + '-test')
    harness.run()
    assert harness.mint.call_count == 2
    harness.probe['direction'][3] = torch.tensor([1., 0., 0.])
    harness.run()
    assert harness.mint.call_count == 3
    harness.model.model = SimpleNamespace(model=SimpleNamespace(layers=harness.layers))
    harness.run()
    assert harness.mint.call_count == 4


def test_generation_failure_removes_hooks_and_restores_configuration(monkeypatch):
    harness = DefinitionHarness(monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match='generation failed'):
        harness.run()
    assert harness.generate_calls == 1
    assert not harness.behavior.interpretations
