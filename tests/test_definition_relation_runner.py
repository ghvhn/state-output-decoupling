"""CPU regressions for benchmark profiles whose fixed mean cancels."""
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants import engine, probe_behavior, probe_definition_relations
from scripts import definition_relation_test as runner


CONTRAST = (
    'Memory 0 retains the original estimate, whereas Memory 3 revises its report using '
    'the new measurement. At lower readings the reply maintains an obsolete estimate, '
    'while at higher readings it updates using new evidence.'
)


class Layer(torch.nn.Identity):
    def __init__(self):
        super().__init__()
        self.registrations = 0

    def register_forward_hook(self, *args, **kwargs):
        self.registrations += 1
        return super().register_forward_hook(*args, **kwargs)


class RunnerHarness:
    def __init__(self, monkeypatch, centers):
        self.layers = [Layer() for _ in centers]
        self.model = SimpleNamespace(device='cpu', d_model=3,
            model=SimpleNamespace(model=SimpleNamespace(layers=self.layers)),
            tok=SimpleNamespace(decode=Mock(return_value=CONTRAST)))
        self.bank = {'compare': {
            layer: {'center_change': vector.clone(), 'changes_norm_mean': 2.}
            for layer, vector in enumerate(centers)}}
        self.evidence, self.selected = runner.heldout_revision()
        self.directions = {layer: torch.tensor([0., 1., 0.]) for layer in range(len(centers))}
        self.diagnostics = {
            'layer_scales': {layer: .5 for layer in range(len(centers))},
            'layers': {layer: {'transport_change_norm': 1.} for layer in range(len(centers))},
            'semantic_truth_checked': False,
        }
        self.resolver = Mock(return_value=(self.directions, self.diagnostics))
        monkeypatch.setattr(probe_definition_relations, 'resolve_definition_relations', self.resolver)
        self.handles = Mock(wraps=probe_behavior.build_definition_handles)
        monkeypatch.setattr(probe_behavior, 'build_definition_handles', self.handles)
        self.inputs = {'input_ids': torch.tensor([[1, 2, 3]], device='cpu')}
        self.input_builder = Mock(return_value=self.inputs)
        monkeypatch.setattr(engine, '_inputs', self.input_builder)
        self.generate = Mock(side_effect=self.generate_ids)
        monkeypatch.setattr(engine, '_generate_ids', self.generate)
        self.fail_generation = False
        self.pushes = []

    def generate_ids(self, model, inputs, budget, **kwargs):
        assert model is self.model and inputs is self.inputs
        assert inputs['input_ids'].device.type == 'cpu'
        assert budget == 20 and kwargs == {'max_tool_calls': 0}
        for layer in self.layers:
            assert len(layer._forward_hooks) == 1
            before = torch.tensor([[[1., 0., 0.]] * 3])
            self.pushes.append(layer(before.clone()) - before)
        if self.fail_generation:
            raise RuntimeError('CPU fake generation failed')
        return torch.tensor([1, 2, 3, 4, 5], device='cpu')

    def run(self, variant):
        try:
            return runner.run_variant(self.model, self.bank, self.evidence, self.selected,
                                      variant, gain=.04, budget=20, cap=.08)
        finally:
            assert not any(layer._forward_hooks for layer in self.layers)


@pytest.mark.parametrize('centers,zero_layers', [
    ([torch.zeros(3), torch.tensor([1., 0., 0.])], [0]),
    ([torch.tensor([1., 0., 0.]), torch.zeros(3)], [1]),
    ([torch.zeros(3), torch.zeros(3)], [0, 1]),
])
def test_any_cancelled_fixed_mean_is_unavailable_without_generation_or_hooks(monkeypatch, centers, zero_layers):
    harness = RunnerHarness(monkeypatch, centers)
    before = copy.deepcopy(harness.bank)
    result = harness.run('fixed_mean')

    assert result['variant'] == 'fixed_mean' and result['status'] == 'unavailable'
    assert result['zero_mean_layers'] == zero_layers
    assert 'cancel' in result['reason'] and 'zero fixed mean' in result['reason']
    assert result['text'] == '' and result['tokens_generated'] == 0
    assert result['qualification'] is None
    assert result['conversion_diagnostics'] == result['variation_from_mean'] == {}
    assert result['evidence_sha256'] == hashlib.sha256(harness.evidence.encode('utf-8')).hexdigest()
    assert result['semantic_truth_checked'] is False
    harness.handles.assert_not_called()
    harness.input_builder.assert_not_called()
    harness.generate.assert_not_called()
    harness.model.tok.decode.assert_not_called()
    harness.resolver.assert_not_called()
    assert all(layer.registrations == 0 for layer in harness.layers)
    for layer, profile in before['compare'].items():
        torch.testing.assert_close(harness.bank['compare'][layer]['center_change'], profile['center_change'])
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('centers', [
    [torch.zeros(3), torch.tensor([0., 2., 0.])],
    [torch.zeros(3), torch.zeros(3)],
])
def test_adaptive_generation_succeeds_with_null_cosine_only_for_cancelled_means(monkeypatch, centers):
    harness = RunnerHarness(monkeypatch, centers)
    result = harness.run('adaptive')

    harness.resolver.assert_called_once_with(harness.model, harness.bank, 'compare', harness.evidence)
    harness.input_builder.assert_called_once_with(harness.model, harness.evidence)
    harness.generate.assert_called_once()
    assert harness.handles.call_count == 1
    assert result['status'] == 'hypothesis' and result['tokens_generated'] == 2
    assert result['text'] == CONTRAST and result['qualification']['accepted'] is True
    assert result['qualification']['structural_only'] is True
    assert result['semantic_truth_checked'] is False
    assert result['conversion_diagnostics'] == harness.diagnostics
    for layer, center in enumerate(centers):
        variation = result['variation_from_mean'][layer]
        if not center.any():
            assert variation['cosine_to_fixed_mean'] is None
        else:
            assert variation['cosine_to_fixed_mean'] == pytest.approx(1.)
        assert variation['relative_transport_norm'] == pytest.approx(.5)
        # A useful adaptive change survives even when its aggregate mean is zero.
        torch.testing.assert_close(harness.pushes[layer][0, :2], torch.zeros(2, 3))
        torch.testing.assert_close(harness.pushes[layer][0, 2], torch.tensor([0., .02, 0.]))
    json.dumps(result, allow_nan=False)


def test_adaptive_generation_error_with_cancelled_mean_removes_real_hooks(monkeypatch):
    harness = RunnerHarness(monkeypatch, [torch.zeros(3), torch.zeros(3)])
    harness.fail_generation = True
    with pytest.raises(RuntimeError, match='CPU fake generation failed'):
        harness.run('adaptive')
    assert harness.generate.call_count == 1
    assert all(layer.registrations == 1 and not layer._forward_hooks for layer in harness.layers)
    harness.model.tok.decode.assert_not_called()
