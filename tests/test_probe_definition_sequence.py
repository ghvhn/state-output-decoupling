"""Depth-composition geometry and lifecycle checks; no semantic success claim."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

import invariants.probe_definition_sequence as sequence_module
from invariants.probe_definition_schema import SCHEMA_ORDER
from invariants.probe_definition_sequence import DefinitionSchemaSequence, build_schema_sequence_handles
from invariants.relational_conversion import fit_relational_conversion, predict_relational_conversion


class Block(torch.nn.Module):
    def __init__(self, nonlinear=False, tuple_output=False):
        super().__init__()
        self.nonlinear = nonlinear
        self.tuple_output = tuple_output
        self.registrations = 0

    def forward(self, h):
        if self.nonlinear:
            # An earlier pulse changes what the next block computes, rather
            # than merely accumulating four independent constant vectors.
            h = torch.stack((h[..., 0]+.07*h[..., 1].square(),
                             h[..., 1]+.04*h[..., 0].square(),
                             h[..., 2]+.03*h[..., 0]*h[..., 1]), dim=-1)
        return (h, 'unchanged-cache') if self.tuple_output else h

    def register_forward_hook(self, *args, **kwargs):
        self.registrations += 1
        return super().register_forward_hook(*args, **kwargs)


class Decoder(torch.nn.Module):
    def __init__(self, count, nonlinear=False, tuple_output=False):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block(nonlinear, tuple_output) for _ in range(count)])

    def forward(self, h):
        for block in self.layers:
            result = block(h)
            if isinstance(result, tuple):
                assert result[1] == 'unchanged-cache'
                h = result[0]
            else:
                h = result
        return h


class Root(torch.nn.Module):
    def __init__(self, count, nonlinear=False, tuple_output=False):
        super().__init__()
        self.model = Decoder(count, nonlinear, tuple_output)

    def forward(self, hidden):
        return self.model(hidden)


def setup(count=4, *, nonlinear=False, tuple_output=False, selected=None):
    root = Root(count, nonlinear, tuple_output)
    model = SimpleNamespace(device='cpu', d_model=3, model=root)
    layers = tuple(range(count)) if selected is None else selected
    subject = {layer: torch.tensor([1., 0., 0.]) for layer in layers}
    x = torch.tensor([[a+1, b+1, c+1] for a in (-1.5, -.5, .5, 1.5)
                      for b in (-.5, .5) for c in (-.5, .5)])
    transforms = (
        torch.tensor([[0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]]),
        torch.tensor([[0., 0., .4], [0., 0., .4], [0., 0., 0.]]),
        torch.tensor([[0., 0., 0.], [0., 0., 0.], [-.4, .3, 0.]]),
    )
    offsets = (torch.tensor([.1, .2, .1]), torch.tensor([.2, .1, .1]),
               torch.tensor([.1, .1, .2]), torch.tensor([.3, .1, .2]))
    bank = {}
    for role, transform, offset in zip(SCHEMA_ORDER, transforms, offsets):
        profile = fit_relational_conversion(x, x@transform+offset)
        bank[role] = {layer: copy.deepcopy(profile) for layer in layers}
    return model, subject, bank


def remove(handles):
    for handle in handles:
        handle.remove()


def assert_no_hooks(model):
    assert not model.model._forward_pre_hooks
    assert all(not block._forward_hooks for block in model.model.model.layers)


def test_later_prediction_observes_real_nonlinear_effect_of_earlier_pulse(monkeypatch):
    model, subject, bank = setup(nonlinear=True)
    clean = Root(4, nonlinear=True)
    captured, clean_states, predicted = [], [], []
    capture = model.model.model.layers[1].register_forward_hook(lambda _m, _i, h: captured.append(h[0, -1].clone()))
    clean_hook = clean.model.layers[1].register_forward_hook(lambda _m, _i, h: clean_states.append(h[0, -1].clone()))
    original_predict = sequence_module.predict_relational_conversion
    def predict(profile, source):
        predicted.append(source.detach().cpu().clone())
        return original_predict(profile, source)
    monkeypatch.setattr(sequence_module, 'predict_relational_conversion', predict)
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence, subject_fraction=0)
    try:
        evidence = torch.ones(1, 5, 3)
        clean(evidence)
        model.model(evidence)
        assert len(predicted) == 4
        torch.testing.assert_close(predicted[1], captured[0])
        assert not torch.allclose(predicted[1], clean_states[0])
        assert sequence.trace()['layers'][1]['role'] == 'core'
        assert sequence.trace()['prediction_count'] == 4
    finally:
        remove(handles)
        capture.remove()
        clean_hook.remove()


def test_depth_order_changes_final_state_with_same_gains_and_evidence():
    model_a, subject_a, bank_a = setup(nonlinear=True)
    model_b, subject_b, bank_b = setup(nonlinear=True)
    a, b = DefinitionSchemaSequence(), DefinitionSchemaSequence()
    handles_a = build_schema_sequence_handles(model_a, subject_a, bank_a, sequence=a, subject_fraction=0)
    handles_b = build_schema_sequence_handles(model_b, subject_b, bank_b, sequence=b,
                                               order=tuple(reversed(SCHEMA_ORDER)), subject_fraction=0)
    try:
        evidence = torch.ones(1, 5, 3)
        first, reversed_order = model_a.model(evidence), model_b.model(evidence)
        torch.testing.assert_close(first[:, :-1], reversed_order[:, :-1])
        assert not torch.allclose(first[:, -1], reversed_order[:, -1])
        assert a.trace()['signature']['role_gains'] == b.trace()['signature']['role_gains']
        assert [entry['role'] for entry in b.trace()['layers']] == list(reversed(SCHEMA_ORDER))
    finally:
        remove(handles_a)
        remove(handles_b)


def test_first_pulses_replay_at_original_position_without_prediction_or_decode_changes(monkeypatch):
    model, subject, bank = setup()
    predictions = []
    original = sequence_module.predict_relational_conversion
    def predict(profile, source):
        predictions.append(source.clone())
        return original(profile, source)
    monkeypatch.setattr(sequence_module, 'predict_relational_conversion', predict)
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        evidence = torch.ones(1, 5, 3)
        first = model.model(evidence)
        assert len(predictions) == 4
        torch.testing.assert_close(first[:, :-1], evidence[:, :-1], rtol=0, atol=0)
        for _ in range(3):
            cached = torch.ones(1, 1, 3)*2
            torch.testing.assert_close(model.model(cached), cached, rtol=0, atol=0)
        rebuilt = torch.ones(1, 8, 3)
        second = model.model(rebuilt)
        torch.testing.assert_close(second[:, :5], first, rtol=0, atol=0)
        torch.testing.assert_close(second[:, 5:], rebuilt[:, 5:], rtol=0, atol=0)
        # Even a changed source does not remint or rescale the saved addition.
        changed_source = rebuilt*10
        third = model.model(changed_source)
        torch.testing.assert_close(third[0, 4]-changed_source[0, 4], first[0, 4]-evidence[0, 4], rtol=1e-5, atol=1e-6)
        assert len(predictions) == 4
        trace = sequence.trace()
        assert trace['decision_position'] == 4
        assert trace['forward_kinds'] == {'initial_prefill': 1, 'prefix_rebuild': 2, 'cached_decode': 3}
        assert all(record['replay_count'] == 2 for record in trace['layers'])
        assert trace['cached_decode'] == 'untouched'
    finally:
        remove(handles)
    assert_no_hooks(model)


def test_selected_layers_split_into_nonempty_contiguous_ordered_groups():
    model, subject, bank = setup(count=9)
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        groups = sequence.trace()['signature']['role_layers']
        assert groups == {'genus': [0, 1], 'core': [2, 3], 'boundary': [4, 5], 'compose': [6, 7, 8]}
        model.model(torch.ones(1, 5, 3))
        assert len(sequence.trace()['layers']) == 9
    finally:
        remove(handles)


def test_sparse_selected_layers_are_grouped_in_their_depth_order():
    selected = (1, 3, 4, 7, 8, 10, 11)
    model, subject, bank = setup(count=12, selected=selected)
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        groups = sequence.trace()['signature']['role_layers']
        assert groups == {'genus': [1], 'core': [3, 4], 'boundary': [7, 8], 'compose': [10, 11]}
    finally:
        remove(handles)


def test_subject_is_pulsed_once_and_combines_with_task_under_same_cap():
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence,
                                             role_gains={role: .8 for role in SCHEMA_ORDER}, cap_fraction=.01)
    try:
        model.model(torch.ones(1, 5, 3))
        records = sequence.trace()['layers']
        assert [entry['subject_pulse'] for entry in records] == [True, False, False, False]
        assert [entry['subject_fraction'] for entry in records] == [.02, 0, 0, 0]
        for record in records:
            assert record['applied_addition_norm'] <= .01*record['source_norm']+1e-6
        assert records[0]['attempted_addition_norm'] > records[0]['applied_addition_norm']
        assert sequence.trace()['signature']['subject_layer'] == 0
    finally:
        remove(handles)


def test_zero_task_gains_leave_only_one_subject_pulse():
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence,
                                             role_gains={role: 0 for role in SCHEMA_ORDER}, cap_fraction=1)
    try:
        evidence = torch.ones(1, 5, 3)
        result = model.model(evidence)
        expected = evidence.clone()
        expected[0, -1, 0] += .02*evidence[0, -1].norm()
        torch.testing.assert_close(result, expected)
        assert sequence.trace()['prediction_count'] == 4
        assert [entry['applied_addition_norm'] > 0 for entry in sequence.trace()['layers']] == [True, False, False, False]
    finally:
        remove(handles)


def test_signed_gain_retains_conditional_magnitude_and_direction():
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence()
    gains = {role: 0 for role in SCHEMA_ORDER}
    gains['genus'] = -.04
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence,
                                             role_gains=gains, subject_fraction=0, cap_fraction=1)
    try:
        evidence = torch.ones(1, 5, 3)
        delta, _ = predict_relational_conversion(bank['genus'][0], evidence[0, -1])
        expected = -.04*evidence[0, -1].norm()*delta/bank['genus'][0]['changes_norm_mean']
        actual = model.model(evidence)-evidence
        torch.testing.assert_close(actual[0, -1], expected, rtol=1e-5, atol=1e-6)
        assert sequence.trace()['layers'][0]['effective_task_gain'] < 0
    finally:
        remove(handles)


def test_conditional_zero_is_zero_not_a_fabricated_nonzero_direction():
    model, subject, bank = setup()
    x = torch.tensor([[a+1, b+1, c+1] for a in (-1.5, -.5, .5, 1.5)
                      for b in (-.5, .5) for c in (-.5, .5)])
    zero_at_center = fit_relational_conversion(x, x-1)
    bank = {role: {layer: copy.deepcopy(zero_at_center) for layer in subject} for role in SCHEMA_ORDER}
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence, subject_fraction=0)
    try:
        evidence = torch.ones(1, 5, 3)
        torch.testing.assert_close(model.model(evidence), evidence, rtol=0, atol=0)
        assert all(entry['raw_delta_norm'] == 0 and entry['layer_scale'] == 0 for entry in sequence.trace()['layers'])
    finally:
        remove(handles)


def test_ood_support_and_magnitude_bound_are_explicit_in_operator_trace():
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        model.model(torch.ones(1, 5, 3)*1000)
        trace = sequence.trace()
        assert all(entry['prediction']['clamped'] and entry['prediction']['outside_training_support'] for entry in trace['layers'])
        assert all(0 <= entry['layer_scale'] <= 2 for entry in trace['layers'])
        assert trace['online_intervention_source_shift'] == 'experimental_unverified'
        assert not trace['semantic_truth_checked'] and not trace['slot_completion_checked']
        assert trace['diagnostic_only']
        json.dumps(trace, allow_nan=False)
    finally:
        remove(handles)


def test_zero_cap_saves_zero_additions_without_claiming_slot_completion():
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence, cap_fraction=0)
    try:
        evidence = torch.ones(1, 5, 3)
        torch.testing.assert_close(model.model(evidence), evidence, rtol=0, atol=0)
        assert all(record['applied_addition_norm'] == 0 for record in sequence.trace()['layers'])
        assert not sequence.trace()['slot_completion_checked']
    finally:
        remove(handles)


def test_tuple_outputs_input_immutability_and_trace_independence():
    model, subject, bank = setup(tuple_output=True)
    saved_bank = copy.deepcopy(bank)
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        evidence = torch.ones(1, 5, 3)
        original = evidence.clone()
        model.model(evidence)
        torch.testing.assert_close(evidence, original, rtol=0, atol=0)
        trace = sequence.trace()
        trace['signature']['layers'][0] = 100
        trace['layers'][0]['prediction']['source_distance'] = -5
        assert sequence.trace()['signature']['layers'][0] == 0
        assert sequence.trace()['layers'][0]['prediction']['source_distance'] >= 0
        for role in SCHEMA_ORDER:
            for layer in subject:
                torch.testing.assert_close(bank[role][layer]['coefficients'], saved_bank[role][layer]['coefficients'])
    finally:
        remove(handles)
    assert_no_hooks(model)


@pytest.mark.parametrize('option,value,error', [
    ('cap_fraction', -1, 'cap fraction'), ('cap_fraction', float('nan'), 'cap fraction'),
    ('cap_fraction', float('inf'), 'cap fraction'), ('cap_fraction', True, 'cap fraction'),
    ('subject_fraction', -1, 'subject fraction'), ('subject_fraction', 1.1, 'subject fraction'),
    ('subject_fraction', None, 'subject fraction'), ('subject_fraction', True, 'subject fraction'),
    ('role_gains', {'core': None}, 'gain for core'), ('role_gains', {'core': float('inf')}, 'gain for core'),
    ('role_gains', {'core': 1.1}, 'gain for core'), ('role_gains', {'core': True}, 'gain for core'),
    ('role_gains', {'unknown': .04}, 'declared roles'),
    ('order', ('genus', 'genus', 'boundary', 'compose'), 'exactly once'),
    ('order', ('genus', 'core', 'boundary'), 'exactly once'),
    ('order', ([], 'core', 'boundary', 'compose'), 'exactly once'),
])
def test_invalid_controls_fail_before_any_hook_registration(option, value, error):
    model, subject, bank = setup()
    with pytest.raises(ValueError, match=error):
        build_schema_sequence_handles(model, subject, bank, **{option: value})
    assert_no_hooks(model)
    assert all(block.registrations == 0 for block in model.model.model.layers)


def test_at_least_four_layers_and_known_width_required_before_hooks():
    model, subject, bank = setup(count=3)
    with pytest.raises(ValueError, match='at least four'):
        build_schema_sequence_handles(model, subject, bank)
    assert_no_hooks(model)
    model, subject, bank = setup()
    model.d_model = None
    with pytest.raises(ValueError, match='known positive'):
        build_schema_sequence_handles(model, subject, bank)
    assert_no_hooks(model)


@pytest.mark.parametrize('fault', ['missing_role', 'missing_layer', 'wrong_dtype', 'nan', 'subject_nan', 'subject_shape', 'subject_zero'])
def test_all_profiles_and_subjects_validate_before_registration_even_at_zero_gain(fault):
    model, subject, bank = setup()
    if fault == 'missing_role':
        del bank['compose']
    elif fault == 'missing_layer':
        del bank['compose'][3]
    elif fault == 'wrong_dtype':
        bank['compose'][3]['coefficients'] = bank['compose'][3]['coefficients'].double()
    elif fault == 'nan':
        bank['compose'][3]['center_source'][0] = float('nan')
    elif fault == 'subject_nan':
        subject[3][0] = float('nan')
    elif fault == 'subject_shape':
        subject[3] = torch.ones(4)
    else:
        subject[3] = torch.zeros(3)
    with pytest.raises(ValueError):
        build_schema_sequence_handles(model, subject, bank, role_gains={role: 0 for role in SCHEMA_ORDER})
    assert_no_hooks(model)
    assert all(block.registrations == 0 for block in model.model.model.layers)


def test_missing_decoder_block_fails_before_registration():
    model, subject, bank = setup(selected=(0, 1, 2, 8))
    with pytest.raises(ValueError, match='missing a requested decoder'):
        build_schema_sequence_handles(model, subject, bank)
    assert_no_hooks(model)


def test_failed_registration_removes_earlier_hooks_and_preserves_existing_hook(monkeypatch):
    model, subject, bank = setup()
    calls = []
    existing = model.model.model.layers[0].register_forward_hook(lambda _m, _i, _o: calls.append('existing'))
    def fail(*_args, **_kwargs):
        raise RuntimeError('registration failure')
    monkeypatch.setattr(model.model.model.layers[2], 'register_forward_hook', fail)
    try:
        with pytest.raises(RuntimeError, match='registration failure'):
            build_schema_sequence_handles(model, subject, bank)
        assert not model.model._forward_pre_hooks
        assert len(model.model.model.layers[0]._forward_hooks) == 1
        assert not model.model.model.layers[1]._forward_hooks
        model.model(torch.ones(1, 5, 3))
        assert calls == ['existing']
    finally:
        existing.remove()


@pytest.mark.parametrize('hidden', [torch.ones(2, 5, 3), torch.ones(5, 3), torch.ones(1, 5, 4)])
def test_runtime_requires_one_actual_sequence_with_matching_width(hidden):
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        with pytest.raises(ValueError, match='batch-one'):
            model.model(hidden)
        assert sequence.trace()['prediction_count'] == 0
    finally:
        remove(handles)
    assert_no_hooks(model)


def test_invalid_finite_residual_norm_and_short_rebuild_fail_closed():
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        with pytest.raises(ValueError, match='finite norm'):
            model.model(torch.ones(1, 5, 3)*1e30)
        assert sequence.trace()['prediction_count'] == 0
    finally:
        remove(handles)
    sequence = DefinitionSchemaSequence()
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        model.model(torch.ones(1, 5, 3))
        with pytest.raises(ValueError, match='shorter than'):
            model.model(torch.ones(1, 4, 3))
        assert sequence.trace()['prediction_count'] == 4
    finally:
        remove(handles)


def test_controller_cannot_be_reused_and_constructor_defaults_are_honored():
    model, subject, bank = setup()
    sequence = DefinitionSchemaSequence(role_gains={'core': .07}, order=tuple(reversed(SCHEMA_ORDER)))
    handles = build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    try:
        assert sequence.trace()['signature']['order'] == list(reversed(SCHEMA_ORDER))
        assert sequence.trace()['signature']['role_gains']['core'] == .07
        with pytest.raises(ValueError, match='fresh'):
            build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    finally:
        remove(handles)
    with pytest.raises(ValueError, match='fresh'):
        build_schema_sequence_handles(model, subject, bank, sequence=sequence)
    assert_no_hooks(model)
