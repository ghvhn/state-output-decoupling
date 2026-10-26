"""Natural movement trains associations; trial steering is explicit and bounded."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import pytest
from invariants.target_relations import TargetRelations, handle_target_command
from invariants.hebbian_plasticity import PlasticityOwners


def train(controller, sign=1):
    for i in range(24):
        a = .1 + (i % 4) * .1
        controller.observe({'probe_a': a, 'probe_b': .5, 'quality': sign*a*.5})


def test_observation_learns_target_patterns_without_interventions():
    controller = TargetRelations()
    controller.configure('quality', ['a', 'b'])
    train(controller)
    evidence = controller.evidence()[0]
    assert evidence['n'] == 23
    assert evidence['correlation'] == pytest.approx(1)
    assert evidence['ready_for_trial']
    assert controller.prepare({'a': .2, 'b': .5}, ['a', 'b']) == {}
    assert controller.adjustment('a', 'b') == 0


def test_minimization_and_lack_of_variation():
    controller = TargetRelations()
    controller.configure('quality', ['a', 'b'], -1)
    train(controller)
    assert controller.evidence()[0]['correlation'] == pytest.approx(-1)
    controller.configure('quality', ['a', 'b'])
    for _ in range(20):
        controller.observe({'probe_a': .2, 'probe_b': .5, 'quality': 1.})
    assert not controller.evidence()[0]['ready_for_trial']
    with pytest.raises(ValueError):
        handle_target_command(['trial'], controller, {}, None)


def test_explicit_trials_are_bounded_and_consumed_once():
    controller = TargetRelations()
    controller.configure('quality', ['a', 'b'])
    train(controller)
    handle_target_command(['trial'], controller, {}, None)
    # Ordinary comparison first; then one bounded trial.
    assert controller.prepare({'a': .2, 'b': .5}, ['a', 'b']) == {}
    controller.observe({'probe_a': .2, 'probe_b': .5, 'quality': .1})
    overlay = controller.prepare({'a': .2, 'b': .5}, ['a', 'b'])
    assert abs(overlay['a|b']) <= controller.budget
    assert controller.prepare({'a': .2, 'b': .5}, ['a', 'b']) == overlay
    controller.observe({'probe_a': .2, 'probe_b': .5, 'quality': .2})
    count = sum(r['n'] for r in controller.records.values())
    controller.observe({'probe_a': .2, 'probe_b': .5, 'quality': .2})
    assert sum(r['n'] for r in controller.records.values()) == count == 1


def test_missing_target_breaks_comparison_and_restart_preserves_evidence():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'relations.json'
        controller = TargetRelations(path)
        controller.configure('quality', ['a', 'b'])
        train(controller)
        controller.observe({'probe_a': .2, 'probe_b': .5})
        n = controller.evidence()[0]['n']
        controller.observe({'probe_a': .4, 'probe_b': .5, 'quality': .8})
        assert controller.evidence()[0]['n'] == n
        resumed = TargetRelations(path)
        assert resumed.evidence() == controller.evidence()
        assert resumed.pending is None and resumed.last_target is None


def test_owner_isolation_and_target_validation():
    with tempfile.TemporaryDirectory() as directory:
        owners = PlasticityOwners()
        tuner = SimpleNamespace(path=Path(directory)/'main.json', triggers={'quality': object()})
        agent = SimpleNamespace(path=Path(directory)/'agent.json')
        matrix = owners.for_tuner(tuner)[0]
        handle_target_command(['target', 'quality', 'maximize', 'a', 'b'], matrix.target_controller, {'a': {}, 'b': {}}, tuner)
        for i in range(12):
            a = .1 + .1*(i % 4)
            owners.observe(tuner, {'probe_a': a, 'probe_b': .5, 'quality': a*.5}, None)
        assert matrix.target_controller.evidence()
        assert not owners.for_tuner(agent)[0].target_controller.evidence()
        assert matrix.status()['pairs'] == 0
        with pytest.raises(ValueError):
            handle_target_command(['target', 'unknown', 'maximize', 'a', 'b'], matrix.target_controller, {'a': {}, 'b': {}}, tuner)


def test_trial_overlay_reaches_matrix_and_off_removes_it():
    from invariants.hebbian_plasticity import HebbianPlasticityMatrix
    from invariants.field_geometry import compile_local_geometry
    import torch
    matrix = HebbianPlasticityMatrix()
    controller = matrix.target_controller = TargetRelations()
    controller.configure('quality', ['a', 'b'])
    train(controller)
    controller.mode = 'trial'
    controller.turn = 1
    controller.prepare({'a': .2, 'b': .5}, ['a', 'b'])
    assert matrix.potentiation('a', 'b') != 0
    stencil = compile_local_geometry(matrix, 'a', 0, {'b': {0: torch.tensor([1., 0.])}})
    assert stencil is not None
    assert stencil.distance(torch.tensor([[.2]]), torch.tensor([[1., 0.]])).item() != pytest.approx(.2)
    handle_target_command(['off'], controller, {}, None)
    assert matrix.potentiation('a', 'b') == 0
