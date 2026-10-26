"""Learned direction and amplitude share one bounded, phase-correct injection."""
import pytest
import torch

from invariants.probe_behavior import build_definition_handles
from test_probe_definition_handles import model_and_controls


@pytest.mark.parametrize('gain', [.04, -.04, 0.])
def test_decision_scales_preserve_magnitude_and_subject_position_on_rebuild(gain):
    model, layers, subject, tasks = model_and_controls()
    scales = {0: .5, 1: 2.}
    handles = build_definition_handles(model, subject, tasks, cap_fraction=.5,
        task_gains={'contrast': gain}, task_positions={'contrast': 'decision'},
        task_layer_scales={'contrast': scales})
    try:
        for index, layer in enumerate(layers):
            original = torch.tensor([[[1., 0., 0.]] * 3])
            expected = torch.zeros_like(original)
            expected[0, 2] = torch.tensor([.02, gain * scales[index], 0.])
            torch.testing.assert_close(layer(original.clone()) - original, expected)
            decode = original[:, :1].clone()
            torch.testing.assert_close(layer(decode.clone()) - decode, torch.zeros_like(decode))
            rebuilt = torch.tensor([[[1., 0., 0.]] * 5])
            expected_rebuilt = torch.zeros_like(rebuilt)
            expected_rebuilt[0, 2] = expected[0, 2]
            torch.testing.assert_close(layer(rebuilt.clone()) - rebuilt, expected_rebuilt)
    finally:
        for handle in handles:
            handle.remove()
    assert all(not layer._forward_hooks for layer in layers)


def test_learned_magnitude_and_subject_share_the_residual_cap():
    model, layers, subject, tasks = model_and_controls()
    handles = build_definition_handles(model, subject, tasks, cap_fraction=.025,
        task_gains={'contrast': .04}, task_positions={'contrast': 'decision'},
        task_layer_scales={'contrast': {0: 2., 1: 0.}})
    try:
        original = torch.tensor([[[1., 0., 0.]] * 3])
        first = layers[0](original.clone()) - original
        assert first[0, 2].norm().item() == pytest.approx(.025)
        assert first[0, 2, 1] / first[0, 2, 0] == pytest.approx(4., rel=2e-5)
        second = layers[1](original.clone()) - original
        torch.testing.assert_close(second[0, 2], torch.tensor([.02, 0., 0.]))
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('scales', [None, {0: 1.}, {0: 1., 1: 1., 2: 1.},
    {'0': 1., 1: 1.}, {0: True, 1: 1.}, {0: -.01, 1: 1.}, {0: 2.01, 1: 1.},
    {0: float('nan'), 1: 1.}, {0: float('inf'), 1: 1.}, {0: 'broken', 1: 1.}])
def test_invalid_scales_register_no_partial_hooks(scales):
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match='layer scale'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04},
            task_layer_scales={'contrast': scales})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)


def test_scales_cannot_name_an_undeclared_operation():
    model, layers, subject, tasks = model_and_controls()
    with pytest.raises(ValueError, match='undeclared control'):
        build_definition_handles(model, subject, tasks, task_gains={'contrast': .04},
            task_layer_scales={'unknown': {0: 1., 1: 1.}})
    assert all(layer.registered == 0 and not layer._forward_hooks for layer in layers)
