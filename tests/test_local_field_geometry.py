"""Geometry depends on live state and locality, not a global mass scale."""
from types import SimpleNamespace
import torch
import pytest
from invariants.field_geometry import local_distance, compile_local_geometry
from invariants.hebbian_plasticity import HebbianPlasticityMatrix
from scripts import interactive_phenomenality as shell


def test_local_deformation_is_monotone_and_fades_in_far_field():
    d = torch.linspace(0., 2., 1001)
    contracted = local_distance(d, 1., strength=.8, radius=.2)
    expanded = local_distance(d, -1., strength=.8, radius=.2)
    assert contracted[0] == expanded[0] == 0
    assert torch.all(contracted[1:] > contracted[:-1])
    assert torch.all(expanded[1:] > expanded[:-1])
    assert contracted[50] / d[50] < .7
    assert expanded[50] > d[50]
    assert abs(float(contracted[-1] / d[-1]) - 1.) < .0001


def test_relationships_gate_each_token_and_filter_wrong_width():
    matrix = HebbianPlasticityMatrix(learning_rate=.5)
    matrix.update({'a': 1., 'b': 1., 'bad': 1.}, 1.)
    stencil = compile_local_geometry(matrix, 'a', 0,
        {'b': {0: torch.tensor([1., 0.])}, 'bad': {0: torch.ones(3)}}, width=2)
    state = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
    d = torch.full((3, 1), .2)
    effective = stencil.distance(d, state)
    assert effective[0] < d[0]
    torch.testing.assert_close(effective[1:], d[1:])


def test_live_hook_keeps_parameters_and_cap(monkeypatch):
    bodies = {key: {} for key in ('anchors', 'probe_poles', 'laws', 'fields',
              'g_families', 'qualities', 'sets', 'bands', 'spaces', 'rotations',
              'impulses', 'law_gates', 'trigger_bodies')}
    bodies['loaded'] = True
    monkeypatch.setattr(shell, '_STEER_BODIES', bodies)
    monkeypatch.setitem(shell._GRAVITY_STATE, 'accum', None)
    layer = torch.nn.Linear(3, 3, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(3))
    wrapper = SimpleNamespace(device='cpu', model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    tuner = SimpleNamespace(get=lambda name, default=0.: .01 if name == 'prioritize_alpha' else default)
    probes = {'a': {'direction': {0: torch.tensor([1., 0., 0.])}, 'mass': 1., 'history': []},
              'b': {'direction': {0: torch.tensor([0., 1., 0.])}, 'mass': 0., 'history': []}}
    matrix = HebbianPlasticityMatrix(learning_rate=.5, mass_strength=0.)
    matrix.update({'a': 1., 'b': 1.}, 1.)
    state = torch.tensor([[.8, .6, 0.], [.8, 0., .6]])
    original = layer.weight.detach().clone()
    handles, _ = shell.build_gravity_field_handles(wrapper, probes, tuner, plasticity=matrix)
    assert handles
    try:
        learned = layer(state)
        matrix.enabled = False
        baseline = layer(state)
        assert (learned[0] - state[0]).norm() > (baseline[0] - state[0]).norm()
        torch.testing.assert_close(learned[1], baseline[1])
        assert torch.all((learned - state).norm(dim=-1) <= .05 * state.norm(dim=-1) + 1e-6)
        torch.testing.assert_close(layer.weight, original)
    finally:
        for handle in handles:
            handle.remove()


def test_manual_edit_validates_before_persisting(monkeypatch):
    bodies = {key: {} for key in ('anchors', 'fields', 'g_families', 'spaces')}
    monkeypatch.setattr(shell, 'steer_bodies', lambda: bodies)
    monkeypatch.setattr(shell, 'save_steer_bodies', lambda: None)
    probes = {'care': {'direction': {0: torch.tensor([1., 0.])}}}
    tuner = SimpleNamespace(triggers={})
    shell.handle_geometry_command('space focus contract .care warp 0.5'.split(), probes, tuner)
    assert shell.space_scale_factor(bodies['spaces']['focus'], probes, tuner) == pytest.approx(2/3)
    before = dict(bodies['spaces'])
    with pytest.raises(ValueError):
        shell.handle_geometry_command('space broken expand .care mystery 1'.split(), probes, tuner)
    assert bodies['spaces'] == before
