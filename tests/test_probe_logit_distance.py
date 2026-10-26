"""Directional output geometry must preserve gauge, sign, and exact context."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from invariants import probe_logit_distance as metric, engine
from test_probe_activation_explain import NativeTokenizer


def logits_case(epsilon=1e-3):
    baseline = torch.tensor([1., .5, -.7, -2., .1], dtype=torch.float64)
    tangent = torch.tensor([.2, -.3, .5, -.1, .6], dtype=torch.float64)
    return baseline, baseline + epsilon * tangent, baseline - epsilon * tangent, tangent


def test_metric_removes_uniform_logit_gauge_and_zeroes_predictive_distance():
    baseline = torch.tensor([2., 1., 0., -1.], dtype=torch.float64)
    report = metric.analyze_directional_logits(baseline, baseline + .03, baseline - .03, .01)
    assert report['fisher_rms'] < 1e-12
    assert report['kl_baseline_positive'] < 1e-12
    assert report['kl_baseline_negative'] < 1e-12
    assert report['fisher_rao_negative_positive'] < 1e-7
    assert report['uniform_logit_shift_removed']
    assert not report['semantic_truth_checked'] and not report['definition_quality_checked']


def test_signed_sensitivity_matches_softmax_derivative_and_local_kl():
    epsilon = 1e-3
    baseline, positive, negative, tangent = logits_case(epsilon)
    report = metric.analyze_directional_logits(baseline, positive, negative, epsilon, top_k=5,
        positive_half=baseline + epsilon / 2 * tangent,
        negative_half=baseline - epsilon / 2 * tangent)
    probability = baseline.softmax(-1)
    expected = tangent - (probability * tangent).sum()
    actual = {row['id']: row['logprob_sensitivity'] for row in report['influential']}
    torch.testing.assert_close(torch.tensor([actual[index] for index in range(5)], dtype=torch.float64), expected)
    assert abs(report['weighted_sensitivity_mean']) < 1e-12
    assert report['half_step_relative_error'] < 1e-10
    assert report['midpoint_relative_error'] < 1e-10
    assert report['direct_logprob_relative_error'] < 1e-6
    assert report['kl_baseline_positive'] == pytest.approx(report['local_kl_prediction'], rel=.001)
    assert report['kl_baseline_negative'] == pytest.approx(report['local_kl_prediction'], rel=.001)
    assert report['fisher_rao_negative_positive'] == pytest.approx(2 * epsilon * report['fisher_rms'], rel=.001)


def test_reversing_probe_direction_preserves_distance_and_reverses_signed_effects():
    baseline, positive, negative, _ = logits_case(.01)
    forward = metric.analyze_directional_logits(baseline, positive, negative, .01, top_k=5)
    reverse = metric.analyze_directional_logits(baseline, negative, positive, .01, top_k=5)
    assert reverse['fisher_rms'] == pytest.approx(forward['fisher_rms'])
    assert reverse['fisher_rao_negative_positive'] == pytest.approx(forward['fisher_rao_negative_positive'])
    first = {row['id']: row['logprob_sensitivity'] for row in forward['influential']}
    second = {row['id']: row['logprob_sensitivity'] for row in reverse['influential']}
    assert all(second[index] == pytest.approx(-value) for index, value in first.items())


def test_small_predictive_distance_does_not_disappear_when_coefficient_rounds_to_one():
    baseline, positive, negative, _ = logits_case(1e-8)
    report = metric.analyze_directional_logits(baseline, positive, negative, 1e-8)
    assert report['fisher_rao_negative_positive'] > 0
    assert report['fisher_rao_negative_positive'] == pytest.approx(2e-8 * report['fisher_rms'], rel=1e-6)


def test_half_step_agreement_does_not_hide_departure_of_signed_midpoint_from_baseline():
    epsilon = .01
    baseline, positive, negative, tangent = logits_case(epsilon)
    offset = torch.tensor([.02, -.01, .01, 0., -.03], dtype=torch.float64)
    report = metric.analyze_directional_logits(baseline, positive + offset, negative + offset, epsilon,
        positive_half=baseline + epsilon / 2 * tangent + offset,
        negative_half=baseline - epsilon / 2 * tangent + offset)
    assert report['half_step_relative_error'] < 1e-10
    assert report['midpoint_relative_error'] > 1


def test_influential_token_ranking_weights_probability_not_only_sensitivity():
    baseline = torch.tensor([0., -1., -35.], dtype=torch.float64)
    direction = torch.tensor([0., .5, 100.], dtype=torch.float64)
    report = metric.analyze_directional_logits(baseline, baseline + .01 * direction,
        baseline - .01 * direction, .01, top_k=3)
    assert report['influential'][0]['id'] == 1
    assert report['influential'][-1]['id'] == 2
    assert report['positive'][-1]['logprob_sensitivity'] > 90


@pytest.mark.parametrize('epsilon', [0, -.01, .11, True, float('nan'), float('inf')])
def test_invalid_step_is_rejected(epsilon):
    baseline, positive, negative, _ = logits_case()
    with pytest.raises(ValueError, match='epsilon'):
        metric.analyze_directional_logits(baseline, positive, negative, epsilon)


@pytest.mark.parametrize('bad', [torch.ones(2, 3), torch.ones(4), torch.tensor([1., float('nan'), 0., 0., 0.])])
def test_invalid_or_mismatched_logit_geometry_is_rejected(bad):
    baseline, positive, negative, _ = logits_case()
    with pytest.raises(ValueError, match='logits'):
        metric.analyze_directional_logits(baseline, bad, negative, .01)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity(), nn.Tanh()])
        self.norm = nn.LayerNorm(3, elementwise_affine=False)

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class CausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()
        self.head = nn.Linear(3, 5, bias=False)
        self.inputs = []
        self.before_downstream = []
        self.fail_on_call = None
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1., .1, -.2], [-.3, 1., .2], [.7, -.4, .5], [.2, .2, 1.], [-1., 0., .3]]))

    def forward(self, input_ids, **kwargs):
        self.inputs.append((input_ids.detach().clone(), copy.deepcopy(kwargs)))
        if self.fail_on_call == len(self.inputs):
            raise RuntimeError('counterfactual failed')
        ids = input_ids.float()
        hidden = torch.stack((ids / 30., torch.ones_like(ids) * .7, (ids % 4) / 8. + .2), dim=-1)
        hidden = self.model.layers[0](hidden)
        self.before_downstream.append(hidden.detach().clone())
        hidden = self.model.norm(self.model.layers[1](hidden))
        return SimpleNamespace(logits=self.head(hidden))


@pytest.fixture
def wrapped():
    tokenizer = NativeTokenizer()
    tokenizer.decode = lambda ids: 'word_' + str(ids[0])
    tokenizer.all_special_ids = [0]
    return SimpleNamespace(model=CausalModel(), tok=tokenizer, device='cpu', d_model=3, n_layers=2)


def test_full_downstream_measurement_preserves_identical_ids_user_tokens_and_train_state(wrapped):
    raw = wrapped.model
    raw.train()
    raw.model.layers[1].eval()  # Preserve a deliberately mixed prior mode.
    before = [(module, module.training) for module in raw.modules()]
    source = {0: torch.tensor([.2, .6, -.1])}
    source_before = source[0].clone()
    telemetry_before = (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count)
    report = metric.measure_probe_logit_distance(wrapped, source, epsilon=.01, cap_fraction=.15)
    assert len(raw.inputs) == 5 and len(raw.before_downstream) == 5
    for ids, kwargs in raw.inputs:
        torch.testing.assert_close(ids, raw.inputs[0][0])
        assert kwargs['use_cache'] is False and kwargs['output_hidden_states'] is False
    for hidden in raw.before_downstream[1:]:
        torch.testing.assert_close(hidden[:, :-1], raw.before_downstream[0][:, :-1])
    assert not torch.allclose(raw.before_downstream[1][:, -1], raw.before_downstream[2][:, -1])
    assert report['fisher_rms'] > 0 and report['half_step_relative_error'] < .01
    assert report['metadata']['layers'] == [0]
    assert report['metadata']['forward_count'] == 5
    assert report['metadata']['cap_telemetry']['clipped'] == 0
    assert report['metadata']['prompt_task_request'] is False
    assert report['metadata']['assistant_prefix_token_count'] == 0
    assert all(row['token'].startswith('word_') for row in report['influential'])
    assert all(module.training == mode for module, mode in before)
    assert all(not block._forward_hooks for block in raw.model.layers)
    assert telemetry_before == (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count)
    torch.testing.assert_close(source[0], source_before)
    assert wrapped.tok.calls[0][0] == [{'role': 'user', 'content': ''}]


def test_prefix_tokens_are_appended_exactly_without_a_closed_assistant_or_retokenization(wrapped):
    prefix = torch.tensor([101, 102, 103])
    report = metric.measure_probe_logit_distance(wrapped, {0: torch.tensor([.2, .6, -.1])},
        epsilon=.01, assistant_token_ids=prefix, check_half=False)
    assert len(wrapped.tok.calls) == 1
    assert all(ids[0, -3:].tolist() == prefix.tolist() for ids, _ in wrapped.model.inputs)
    assert report['metadata']['assistant_prefix_token_count'] == 3
    assert report['metadata']['forward_count'] == 3


def test_failed_counterfactual_restores_hooks_training_and_cap_telemetry(wrapped):
    wrapped.model.fail_on_call = 2
    wrapped.model.train()
    before = (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count)
    with pytest.raises(RuntimeError, match='counterfactual failed'):
        metric.measure_probe_logit_distance(wrapped, {0: torch.tensor([.2, .6, -.1])}, epsilon=.01)
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)
    assert wrapped.model.training
    assert before == (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count)


def test_step_above_cap_is_rejected_before_any_model_forward(wrapped):
    with pytest.raises(ValueError, match='steering cap'):
        metric.measure_probe_logit_distance(wrapped, {0: torch.tensor([.2, .6, -.1])},
            epsilon=.01, cap_fraction=.005)
    assert wrapped.model.inputs == []
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)


def test_unresolved_residual_step_is_not_reported_as_zero_predictive_distance(wrapped):
    before = (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count)
    with pytest.raises(ValueError, match='unresolved'):
        metric.measure_probe_logit_distance(wrapped, {0: torch.tensor([.2, .6, -.1])}, epsilon=1e-20)
    assert all(not block._forward_hooks for block in wrapped.model.model.layers)
    assert before == (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count)


def test_preexisting_stateful_hook_is_preserved_and_rejected_before_forward(wrapped):
    calls = []
    handle = wrapped.model.model.layers[1].register_forward_hook(lambda *_args: calls.append(True))
    try:
        with pytest.raises(ValueError, match='active forward hooks'):
            metric.measure_probe_logit_distance(wrapped, {0: torch.tensor([.2, .6, -.1])})
        assert wrapped.model.inputs == [] and calls == []
        assert handle.id in wrapped.model.model.layers[1]._forward_hooks
    finally:
        handle.remove()
