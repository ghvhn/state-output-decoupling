"""Verify head alignment and meaningful distribution measurements, no model load."""
import pytest
import torch
from pathlib import Path

from scripts.compression_language_test import query_heads, install_query_heads, query_factor_output, token_metrics, load_evaluation_fixture, rollout_comparison


def test_gpt2_head_slot_roundtrip_preserves_key_value_and_bias():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.arange(12*36, dtype=torch.float32).view(12, 36))
    module.bias = torch.nn.Parameter(torch.arange(36, dtype=torch.float32))
    before, bias = module.weight.detach().clone(), module.bias.detach().clone()
    heads = query_heads(module.weight, 3)
    assert heads.shape == (3, 4, 12)
    install_query_heads(module, heads+1)
    torch.testing.assert_close(module.weight[:, :12], before[:, :12]+1)
    torch.testing.assert_close(module.weight[:, 12:], before[:, 12:], rtol=0, atol=0)
    torch.testing.assert_close(module.bias, bias, rtol=0, atol=0)
    install_query_heads(module, heads)
    torch.testing.assert_close(module.weight, before, rtol=0, atol=0)


def test_same_decoded_choices_can_have_nonzero_distribution_and_margin_errors():
    teacher = torch.tensor([[3., 1., 0.], [0., 3., 1.]])
    student = torch.tensor([[2., 1., 0.], [0., 2., 1.]])
    ids = torch.tensor([0, 0, 1])
    result = token_metrics(teacher, student, ids)
    assert result["argmax_agreement"] == 1
    assert result["mean_kl_nats"] > .05
    assert result["nll_change_nats"] > .1
    assert result["mean_top2_margin_absolute_error"] == 1


def test_identical_logits_baseline_and_invalid_timeline():
    teacher = torch.tensor([[3., 1., 0.], [0., 3., 1.]])
    result = token_metrics(teacher, teacher, torch.tensor([0, 0, 1]))
    assert result["mean_kl_nats"] == 0
    assert result["nll_change_nats"] == 0
    with pytest.raises(ValueError, match="timeline"):
        token_metrics(teacher, teacher, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="nonfinite"):
        token_metrics(teacher, teacher*float("nan"), torch.tensor([0, 0, 1]))


def test_factor_override_keeps_key_value_and_carries_query_gradients():
    inputs = torch.arange(8, dtype=torch.float32).view(2, 4)
    native = torch.arange(24, dtype=torch.float32).view(2, 12)
    bias = torch.arange(12, dtype=torch.float32)
    mean = torch.ones(8, requires_grad=True)
    basis = torch.arange(8, dtype=torch.float32).view(1, 8).requires_grad_()
    coefficients = torch.tensor([[1.], [-1.]])
    result = query_factor_output(inputs, native, bias, mean, basis, coefficients)
    torch.testing.assert_close(result[:, 4:], native[:, 4:], rtol=0, atol=0)
    result[:, :4].square().sum().backward()
    assert mean.grad is not None and mean.grad.abs().sum() > 0
    assert basis.grad is not None and basis.grad.abs().sum() > 0
    assert not coefficients.requires_grad


def test_sealed_fixture_and_rollout_common_prefix():
    fixture = Path(__file__).parent/"fixtures/compression_language_sealed.json"
    texts, prompts, digest = load_evaluation_fixture(fixture)
    assert len(texts) == 8 and len(prompts) == 2
    assert digest == "61d5990ba8ca5c2612adbaf607303792ecb394a12c89aa7d5aa8887b1f6375ba"
    teacher = {"generated_ids": [2, 4, 6], "continuation": "original"}
    student = {"generated_ids": [2, 5, 6], "continuation": "changed"}
    result = rollout_comparison(teacher, student)
    assert result["common_prefix_tokens"] == 1
    assert not result["identical_token_sequence"]
