"""Check controlled nuisance minting and subtraction without live generation."""
import copy
import re
from types import SimpleNamespace

import pytest
import torch

from invariants.probe_definition_refinement import (
    REFINEMENT_VERSION, TASKREFINEMENT_VERSION, mint_definition_nuisance_axes,
    refine_definition_axes,
)


def task_axes(layers=(0,), width=6):
    def vector(values, dtype=torch.float32):
        return torch.tensor(values + [0.] * (width - len(values)), dtype=dtype)
    return {
        'contrast': {layer: vector([.6, .8]) for layer in layers},
        'abstraction': {layer: vector([.6, 0., .8]) for layer in layers},
        'naming': {layer: vector([0., 0., 0., 1.]) for layer in layers},
        'verbosity': {layer: vector([1.]) for layer in layers},
        # Preserve an untouched axis's original value and dtype, not its unit copy.
        'question': {layer: vector([0., 0., 0., 0., 2.], torch.float64) for layer in layers},
    }


def nuisance_axes(layers=(0,), width=6):
    return {'verbosity': {layer: torch.eye(width)[0] for layer in layers}}


def test_projection_removes_overlap_and_preserves_sources_and_other_tasks():
    tasks = task_axes((0, 2))
    nuisances = nuisance_axes((0, 2))
    before_tasks, before_nuisances = copy.deepcopy(tasks), copy.deepcopy(nuisances)
    cleaned, diagnostics = refine_definition_axes(tasks, nuisances)
    for layer in (0, 2):
        torch.testing.assert_close(cleaned['contrast'][layer], torch.eye(6)[1])
        torch.testing.assert_close(cleaned['abstraction'][layer], torch.eye(6)[2])
        for name in ('contrast', 'abstraction'):
            row = diagnostics[name][layer]
            assert row['rank'] == 1
            assert row['retained_fraction'] == pytest.approx(.8)
            assert row['max_nuisance_overlap_before'] == pytest.approx(.6)
            assert row['max_nuisance_overlap_after'] == pytest.approx(0., abs=1e-6)
        for name in ('naming', 'verbosity', 'question'):
            torch.testing.assert_close(cleaned[name][layer], before_tasks[name][layer])
            assert cleaned[name][layer].dtype == tasks[name][layer].dtype
        for name in tasks:
            torch.testing.assert_close(tasks[name][layer], before_tasks[name][layer])
            assert cleaned[name][layer].data_ptr() != tasks[name][layer].data_ptr()
        torch.testing.assert_close(nuisances['verbosity'][layer], before_nuisances['verbosity'][layer])
    assert set(diagnostics) == {'contrast', 'abstraction'}


def test_orthogonal_directions_are_unchanged_and_projection_is_idempotent():
    tasks = task_axes()
    tasks['contrast'][0] = torch.eye(6)[1]
    tasks['abstraction'][0] = torch.eye(6)[2]
    cleaned, first = refine_definition_axes(tasks, nuisance_axes())
    twice, second = refine_definition_axes(cleaned, nuisance_axes())
    for name in ('contrast', 'abstraction'):
        torch.testing.assert_close(cleaned[name][0], tasks[name][0])
        torch.testing.assert_close(twice[name][0], cleaned[name][0])
        assert first[name][0]['retained_fraction'] == pytest.approx(1.)
        assert second[name][0]['retained_fraction'] == pytest.approx(1.)


def test_duplicate_sign_flipped_and_reordered_nuisances_do_not_over_subtract():
    tasks = task_axes()
    x, extra = torch.eye(6)[0], torch.eye(6)[3]
    unique = {'x': {0: x}, 'extra': {0: extra}}
    duplicates = {'negative_x': {0: -x}, 'extra': {0: extra},
                  'x_twice': {0: 2 * x}, 'x': {0: x}}
    first, first_info = refine_definition_axes(tasks, unique)
    second, second_info = refine_definition_axes(tasks, duplicates)
    third, third_info = refine_definition_axes(tasks, dict(reversed(list(duplicates.items()))))
    for name in ('contrast', 'abstraction'):
        torch.testing.assert_close(first[name][0], second[name][0], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first[name][0], third[name][0], atol=1e-6, rtol=1e-6)
        assert first_info[name][0]['rank'] == second_info[name][0]['rank'] == third_info[name][0]['rank'] == 2
        assert second_info[name][0]['retained_fraction'] == pytest.approx(.8, abs=1e-6)


def test_near_duplicate_singular_component_is_not_promoted_to_nuisance_rank():
    nuisances = nuisance_axes()
    almost_x = torch.eye(6)[0] + 1e-7 * torch.eye(6)[1]
    nuisances['almost_x'] = {0: almost_x}
    cleaned, diagnostics = refine_definition_axes(task_axes(), nuisances)
    assert diagnostics['contrast'][0]['rank'] == 1
    torch.testing.assert_close(cleaned['contrast'][0], torch.eye(6)[1], atol=1e-6, rtol=1e-6)


def test_partial_subtraction_and_zero_strength_are_bounded():
    tasks = task_axes()
    zero, zero_info = refine_definition_axes(tasks, nuisance_axes(), strength=0)
    partial, partial_info = refine_definition_axes(tasks, nuisance_axes(), strength=.5)
    expected = torch.tensor([.3, .8, 0., 0., 0., 0.])
    norm = expected.norm()
    expected /= norm
    torch.testing.assert_close(zero['contrast'][0], tasks['contrast'][0])
    torch.testing.assert_close(partial['contrast'][0], expected)
    assert zero_info['contrast'][0]['retained_fraction'] == pytest.approx(1.)
    assert partial_info['contrast'][0]['retained_fraction'] == pytest.approx(float(norm))


@pytest.mark.parametrize('near_parallel, floor', [(False, 0.), (True, .10)])
def test_full_or_near_cancellation_is_rejected_without_amplifying_roundoff(near_parallel, floor):
    tasks = task_axes()
    tasks['contrast'][0] = torch.tensor([1., .05 if near_parallel else 0., 0., 0., 0., 0.])
    with pytest.raises(ValueError, match='contrast.*inseparable from nuisance span.*layer 0'):
        refine_definition_axes(tasks, nuisance_axes(), min_retained=floor)


def test_a_full_random_nuisance_span_is_rejected_even_with_zero_retention_floor():
    generator = torch.Generator().manual_seed(5)
    columns = torch.randn(6, 6, generator=generator)
    nuisances = {f'nuisance_{index}': {0: columns[:, index]} for index in range(6)}
    with pytest.raises(ValueError, match='inseparable from nuisance span'):
        refine_definition_axes(task_axes(), nuisances, min_retained=0)


@pytest.mark.parametrize('keyword, value', [
    ('strength', None), ('strength', float('nan')), ('strength', float('inf')),
    ('strength', -.1), ('strength', 1.1), ('min_retained', None),
    ('min_retained', float('nan')), ('min_retained', -.1), ('min_retained', 1.1),
])
def test_invalid_fractions_fail_clearly(keyword, value):
    with pytest.raises(ValueError, match='finite and between zero and one'):
        refine_definition_axes(task_axes(), nuisance_axes(), **{keyword: value})


@pytest.mark.parametrize('fault, expected', [
    ('missing_task_layer', 'exactly the requested layers'),
    ('extra_nuisance_layer', 'exactly the requested layers'),
    ('wrong_width', 'incompatible width'),
    ('matrix', 'must be a vector'),
    ('zero', 'zero or cancelled'),
    ('nonfinite', 'non-finite'),
    ('overflow', 'non-finite'),
    ('invalid_untouched', 'non-finite'),
    ('empty_nuisances', 'nonempty nuisance axes'),
])
def test_invalid_axes_fail_before_returning_partial_results(fault, expected):
    tasks = task_axes((0, 2))
    nuisances = nuisance_axes((0, 2))
    if fault == 'missing_task_layer':
        tasks['abstraction'].pop(2)
    elif fault == 'extra_nuisance_layer':
        nuisances['verbosity'][3] = torch.eye(6)[0]
    elif fault == 'wrong_width':
        nuisances['verbosity'][2] = torch.ones(7)
    elif fault == 'matrix':
        nuisances['verbosity'][2] = torch.ones(1, 6)
    elif fault == 'zero':
        nuisances['verbosity'][2].zero_()
    elif fault == 'nonfinite':
        nuisances['verbosity'][2][1] = float('nan')
    elif fault == 'overflow':
        nuisances['verbosity'][2] = torch.full((6,), 3e38)
    elif fault == 'invalid_untouched':
        tasks['question'][2][0] = float('inf')
    elif fault == 'empty_nuisances':
        nuisances = {}
    with pytest.raises(ValueError, match=expected):
        refine_definition_axes(tasks, nuisances)


def test_svd_failure_is_reported_with_layer_and_preserves_sources(monkeypatch):
    tasks = task_axes()
    before = copy.deepcopy(tasks)
    def fail(*args, **kwargs):
        raise RuntimeError('no convergence')
    monkeypatch.setattr(torch.linalg, 'svd', fail)
    with pytest.raises(ValueError, match='span could not be resolved at layer 0'):
        refine_definition_axes(tasks, nuisance_axes())
    for name in tasks:
        torch.testing.assert_close(tasks[name][0], before[name][0])


class Tokenizer:
    def __init__(self):
        self.calls, self.renderings, self.encoded = [], [], []

    def apply_chat_template(self, messages, **kwargs):
        index = len(self.calls)
        self.calls.append((copy.deepcopy(messages), kwargs))
        rendered = f'case-{index}\n' + '\n'.join(f'{message["role"]}:{message["content"]}' for message in messages)
        self.renderings.append(rendered)
        return rendered

    def __call__(self, text, **kwargs):
        self.encoded.append((text, kwargs))
        index = self.renderings.index(text)
        return {'input_ids': torch.tensor([[index, index]]), 'attention_mask': torch.ones(1, 2)}


class Model:
    def __init__(self, n_layers=4, failure=None):
        self.n_layers, self.failure = n_layers, failure
        self.grad_enabled = []

    def __call__(self, input_ids, **kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        assert kwargs['output_hidden_states'] and not kwargs['use_cache']
        index = int(input_ids[0, 0])
        axis, within_axis = divmod(index, 4)
        case, side = divmod(within_axis, 2)
        vector = torch.eye(6)[axis]
        if case:
            vector = 100 * (vector + .3 * torch.eye(6)[4])
        if self.failure == 'cancel' and case:
            vector = -torch.eye(6)[axis]
        if self.failure == 'zero':
            vector.zero_()
        states = [torch.ones(1, 2, 6)]
        for layer in range(self.n_layers):
            hidden = torch.ones(1, 2, 6)
            if side == 0:
                hidden[0, 0] -= 100 * vector
                hidden[0, -1] += (layer + 1) * vector
            if self.failure == 'nonfinite':
                hidden[0, -1, 0] = float('nan')
            states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(states))


def wrapped_model(**kwargs):
    return SimpleNamespace(device='cpu', tok=Tokenizer(), model=Model(**kwargs))


def test_nuisance_mint_uses_controlled_completed_assistant_pairs_and_reuses_verbosity():
    model = wrapped_model()
    tasks = task_axes((1, 3))
    before = copy.deepcopy(tasks)
    nuisances = mint_definition_nuisance_axes(model, (3, 1, 3), tasks)
    assert REFINEMENT_VERSION == TASKREFINEMENT_VERSION and REFINEMENT_VERSION
    assert set(nuisances) == {'verbosity', 'enumeration', 'topic', 'generic_framing'}
    assert len(model.tok.calls) == 12
    assert all(not enabled for enabled in model.model.grad_enabled)
    for messages, kwargs in model.tok.calls:
        assert [message['role'] for message in messages] == ['user', 'assistant']
        assert kwargs == {'tokenize': False, 'add_generation_prompt': False}
        assert '[Probe p_000000000000000000000000]' in messages[0]['content']
        assert 'denial' not in str(messages) and 'secret_target' not in str(messages)
    for rendered, (encoded, kwargs) in zip(model.tok.renderings, model.tok.encoded):
        assert encoded == rendered
        assert kwargs == {'return_tensors': 'pt', 'add_special_tokens': False}
    for name, directions in nuisances.items():
        assert set(directions) == {1, 3}
        for layer, vector in directions.items():
            torch.testing.assert_close(vector.norm(), torch.tensor(1.))
            assert torch.isfinite(vector).all()
            if name == 'verbosity':
                torch.testing.assert_close(vector, tasks[name][layer])
                assert vector.data_ptr() != tasks[name][layer].data_ptr()
    for name in tasks:
        for layer in tasks[name]:
            torch.testing.assert_close(tasks[name][layer], before[name][layer])
    assert not torch.allclose(nuisances['enumeration'][1], nuisances['topic'][1])
    assert not torch.allclose(nuisances['topic'][1], nuisances['generic_framing'][1])
    # Large second-case magnitude cannot dominate the averaged candidate.
    first = torch.eye(6)[0]
    second = first + .3 * torch.eye(6)[4]
    expected = first + second / second.norm()
    expected /= expected.norm()
    torch.testing.assert_close(nuisances['enumeration'][1], expected)


def test_nuisance_teacher_pairs_isolate_presentation_topic_and_generic_framing():
    model = wrapped_model()
    mint_definition_nuisance_axes(model, (0,), task_axes())
    for index in (0, 2):
        positive = model.tok.calls[index][0]
        negative = model.tok.calls[index + 1][0]
        assert positive[0] == negative[0]
        flattened = re.sub(r'(?m)^\d+: ', '', positive[1]['content']).replace('\n', ' ')
        assert flattened == negative[1]['content']
    for index in (4, 6):
        positive = model.tok.calls[index][0]
        negative = model.tok.calls[index + 1][0]
        assert positive[0] != negative[0]
        for messages in (positive, negative):
            contrast = messages[1]['content']
            assert 'Lower positions' in contrast and 'higher positions' in contrast
        # Object words change while the behavioral operation and style stay fixed.
        def without_objects(text):
            return re.sub(r'corridors?|drinks?|assembly|planting|pieces|plants', 'OBJECT', text)
        assert without_objects(positive[1]['content']) == without_objects(negative[1]['content'])
    for index in (8, 10):
        positive = model.tok.calls[index][0]
        negative = model.tok.calls[index + 1][0]
        assert positive[0] == negative[0]
        assert positive[1]['content'].endswith(negative[1]['content'])
        assert 'AI assistant' in positive[1]['content']
        assert 'AI assistant' not in negative[1]['content']


@pytest.mark.parametrize('failure, message', [('zero', 'zero or cancelled'),
                                             ('cancel', 'zero or cancelled'),
                                             ('nonfinite', 'invalid native state')])
def test_nuisance_mint_rejects_nonfinite_or_cancelled_candidates(failure, message):
    with pytest.raises(ValueError, match=message):
        mint_definition_nuisance_axes(wrapped_model(failure=failure), (0,), task_axes())


def test_nuisance_mint_requires_exact_verbosity_layers_widths_and_available_model_layers():
    with pytest.raises(ValueError, match='verbosity task axis'):
        mint_definition_nuisance_axes(wrapped_model(), (0,), {})
    with pytest.raises(ValueError, match='exactly the requested layers'):
        mint_definition_nuisance_axes(wrapped_model(), (0, 2), task_axes())
    with pytest.raises(ValueError, match='incompatible state widths'):
        mint_definition_nuisance_axes(wrapped_model(), (0,), task_axes(width=7))
    with pytest.raises(ValueError, match='missing requested layer 4'):
        mint_definition_nuisance_axes(wrapped_model(), (4,), task_axes((4,)))
