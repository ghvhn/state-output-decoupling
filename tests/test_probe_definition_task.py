"""Check native assistant task minting without a live model or generation."""
import copy
import re
from types import SimpleNamespace

import pytest
import torch

from invariants.probe_definition_task import (DEFINITION_TASK_GAINS, TASK_VERSION,
                                              mint_definition_task_axes,
                                              mint_definition_question_axis)


class Tokenizer:
    def __init__(self):
        self.calls = []
        self.renderings = []
        self.encoded = []

    def apply_chat_template(self, messages, **kwargs):
        index = len(self.calls)
        self.calls.append((copy.deepcopy(messages), kwargs))
        rendered = f'render-{index}\n' + '\n'.join(f'{m["role"]}: {m["content"]}' for m in messages)
        if kwargs.get('add_generation_prompt'):
            rendered += '\nassistant:<decision>'
        self.renderings.append(rendered)
        return rendered

    def __call__(self, text, **kwargs):
        self.encoded.append((text, kwargs))
        index = self.renderings.index(text)
        return {'input_ids': torch.tensor([[index, index]]), 'attention_mask': torch.ones(1, 2)}


class Model:
    def __init__(self, n_layers=4, failure=None):
        self.n_layers = n_layers
        self.failure = failure
        self.grad_enabled = []

    def __call__(self, input_ids, **kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        assert kwargs['output_hidden_states'] and not kwargs['use_cache']
        index = int(input_ids[0, 0])
        axis, within_axis = divmod(index, 4)
        case, side = divmod(within_axis, 2)
        vector = torch.zeros(5)
        vector[axis] = 1.
        vector[(axis + 1) % 5] = case * .3
        vector *= 1 if case == 0 else 100
        if self.failure == 'cancel' and case == 1:
            vector = -torch.eye(5)[axis]
        if self.failure == 'zero':
            vector.zero_()
        hidden = [torch.ones(1, 2, 5)]
        for layer in range(self.n_layers):
            state = torch.ones(1, 2, 5)
            if side == 0:
                # Selecting an interior token or averaging the sequence would
                # reverse this contrast; the final decision position must win.
                state[0, 0] -= 100 * vector
                state[0, -1] += (layer + 1) * vector
            if self.failure == 'nonfinite':
                state[0, -1, 0] = float('nan')
            hidden.append(state)
        return SimpleNamespace(hidden_states=tuple(hidden))


def wrapped_model(**kwargs):
    return SimpleNamespace(device='cpu', tok=Tokenizer(), model=Model(**kwargs))


def test_task_axes_use_completed_native_assistant_roles_on_all_requested_layers():
    model = wrapped_model()
    axes = mint_definition_task_axes(model, (3, 1, 3))
    assert TASK_VERSION and set(axes) == {'contrast', 'abstraction', 'naming', 'verbosity', 'question'}
    assert len(model.tok.calls) == 20
    assert all(not enabled for enabled in model.model.grad_enabled)
    for messages, kwargs in model.tok.calls[:16]:
        assert [message['role'] for message in messages] == ['user', 'assistant']
        assert kwargs == {'tokenize': False, 'add_generation_prompt': False}
        assert '[Probe p_000000000000000000000000]' in messages[0]['content']
        assert 'position=-' in messages[0]['content'] and 'position=+' in messages[0]['content']
        assert 'denial' not in str(messages) and 'secret_target' not in str(messages)
    for messages, kwargs in model.tok.calls[16:]:
        assert [message['role'] for message in messages] == ['user']
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
    for index in range(0, 16, 2):
        assert model.tok.calls[index][0][0] == model.tok.calls[index + 1][0][0]
        assert model.tok.calls[index][0][1] != model.tok.calls[index + 1][0][1]
    for rendered, (encoded, kwargs) in zip(model.tok.renderings, model.tok.encoded):
        assert encoded == rendered
        assert kwargs == {'return_tensors': 'pt', 'add_special_tokens': False}
    assert any(len(rendered) > 600 for rendered in model.tok.renderings)
    for axis in axes.values():
        assert set(axis) == {1, 3}
        for direction in axis.values():
            assert torch.isfinite(direction).all()
            torch.testing.assert_close(direction.norm(), torch.tensor(1.))
    # Separate candidate axes were actually minted, not one dictionary relabeled.
    assert not torch.allclose(axes['contrast'][1], axes['abstraction'][1])
    assert not torch.allclose(axes['naming'][1], axes['verbosity'][1])
    assert not torch.allclose(axes['question'][1], axes['contrast'][1])


def test_case_directions_are_normalized_before_they_are_averaged():
    axes = mint_definition_task_axes(wrapped_model(), (0,))
    first = torch.tensor([1., 0., 0., 0., 0.])
    second = torch.tensor([1., .3, 0., 0., 0.])
    second /= second.norm()
    expected = first + second
    expected /= expected.norm()
    torch.testing.assert_close(axes['contrast'][0], expected)


def test_abstraction_defines_one_distinction_and_verbosity_preserves_it():
    model = wrapped_model()
    mint_definition_task_axes(model, (0,))
    # Third and fourth completion pairs are the independent abstraction cases.
    for index in (4, 6):
        positive = model.tok.calls[index][0][1]['content']
        assert positive.startswith('This dimension is ')
        assert 'lower positions' in positive and 'higher positions' in positive
        assert 'Memory 0' not in positive
    for index in (12, 14):
        verbose = model.tok.calls[index][0][1]['content']
        concise = model.tok.calls[index + 1][0][1]['content']
        assert verbose.startswith(concise)
        assert len(verbose.split()) > 2 * len(concise.split())
        assert 'lower positions' in concise.lower() and 'higher positions' in concise.lower()
    assert DEFINITION_TASK_GAINS == {'contrast': .04, 'abstraction': .04,
                                     'naming': .025, 'verbosity': -.035, 'question': .08,
                                     'operation': 0.0}


def test_question_axis_compares_native_user_question_with_matched_opaque_declaration():
    model = wrapped_model()
    axes = mint_definition_question_axis(model, (3, 1, 3))
    assert len(model.tok.calls) == 4
    subjects = []
    for index in (0, 2):
        positive, positive_kwargs = model.tok.calls[index]
        negative, negative_kwargs = model.tok.calls[index + 1]
        assert [message['role'] for message in positive] == ['user']
        assert [message['role'] for message in negative] == ['user']
        assert positive_kwargs == negative_kwargs == {'tokenize': False, 'add_generation_prompt': True}
        positive_subjects = re.findall(r'p_[0-9a-f]{24}', positive[0]['content'])
        negative_subjects = re.findall(r'p_[0-9a-f]{24}', negative[0]['content'])
        assert len(positive_subjects) == 1 and positive_subjects == negative_subjects
        subject = positive_subjects[0]
        subjects.append(subject)
        assert positive[0]['content'] == f'What is {subject}?'
        assert negative[0]['content'] == f'Here is {subject}.'
        assert model.tok.renderings[index].endswith('assistant:<decision>')
        assert model.tok.renderings[index + 1].endswith('assistant:<decision>')
    assert subjects == ['p_' + '0' * 24, 'p_' + '1' * 24]
    for rendered, (encoded, kwargs) in zip(model.tok.renderings, model.tok.encoded):
        assert encoded == rendered
        assert kwargs == {'return_tensors': 'pt', 'add_special_tokens': False}
        assert 'denial' not in rendered and 'secret_target' not in rendered and 'Memory ' not in rendered
    expected = torch.tensor([1., 0., 0., 0., 0.])
    second = torch.tensor([1., .3, 0., 0., 0.])
    expected += second / second.norm()
    expected /= expected.norm()
    assert set(axes) == {1, 3}
    for direction in axes.values():
        torch.testing.assert_close(direction, expected)
        torch.testing.assert_close(direction.norm(), torch.tensor(1.))
        assert torch.isfinite(direction).all()
    assert all(not enabled for enabled in model.model.grad_enabled)


@pytest.mark.parametrize('failure, message', [('zero', 'zero or cancelled'),
                                              ('cancel', 'zero or cancelled'),
                                              ('nonfinite', 'invalid.*state')])
def test_invalid_or_cancelled_question_axes_fail_clearly(failure, message):
    with pytest.raises(ValueError, match=message):
        mint_definition_question_axis(wrapped_model(failure=failure), (0,))


def test_question_axis_rejects_missing_and_invalid_layers_without_falling_back():
    with pytest.raises(ValueError, match='missing requested layer 4'):
        mint_definition_question_axis(wrapped_model(), (4,))
    for layers in ((), (-1,), (.5,), (True,), (float('nan'),), None):
        with pytest.raises(ValueError, match='layers|indices'):
            mint_definition_question_axis(wrapped_model(), layers)


@pytest.mark.parametrize('failure, message', [('zero', 'zero or cancelled'),
                                              ('cancel', 'zero or cancelled'),
                                              ('nonfinite', 'invalid.*state')])
def test_invalid_or_cancelled_task_axes_fail_clearly(failure, message):
    with pytest.raises(ValueError, match=message):
        mint_definition_task_axes(wrapped_model(failure=failure), (0,))


def test_missing_and_invalid_layers_fail_instead_of_falling_back():
    with pytest.raises(ValueError, match='missing requested layer 4'):
        mint_definition_task_axes(wrapped_model(), (4,))
    for layers in ((), (-1,), (.5,), (True,), (float('nan'),), None):
        with pytest.raises(ValueError, match='layers|indices'):
            mint_definition_task_axes(wrapped_model(), layers)
