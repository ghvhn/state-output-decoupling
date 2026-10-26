"""Check decision-phase candidate minting without a live model or GPU."""
import copy
from types import SimpleNamespace

import pytest
import torch

from invariants.probe_definition_operation import (
    OPERATION_VERSION, mint_definition_operation_axis,
    mint_definition_operation_nuisances,
)


class Tokenizer:
    def __init__(self):
        self.calls, self.renderings, self.encoded = [], [], []

    def apply_chat_template(self, messages, **kwargs):
        index = len(self.calls)
        self.calls.append((copy.deepcopy(messages), kwargs))
        rendered = f'case-{index}\n' + '\n'.join(f'{message["role"]}:{message["content"]}' for message in messages)
        if kwargs['add_generation_prompt']:
            rendered += '\nassistant:<decision>'
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
        width = 6 if ((self.failure == 'pair_width' and index == 1) or
                      (self.failure == 'case_width' and index >= 2)) else 5
        vector = torch.eye(width)[axis]
        if case:
            vector = 100 * (vector + .3 * torch.eye(width)[4])
        if self.failure == 'cancel' and case:
            vector = -torch.eye(width)[axis]
        if self.failure == 'zero':
            vector.zero_()
        states = [torch.ones(1, 2, width)]
        for layer in range(self.n_layers):
            hidden = torch.ones(1, 2, width)
            if side == 0:
                # The interior token has the opposite difference, ensuring the
                # assistant decision endpoint is actually selected.
                hidden[0, 0] -= 100 * vector
                hidden[0, -1] += (layer + 1) * vector
            if self.failure == 'nonfinite':
                hidden[0, -1, 0] = float('nan')
            states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(states))


def wrapped_model(**kwargs):
    return SimpleNamespace(device='cpu', tok=Tokenizer(), model=Model(**kwargs))


def split_request(messages):
    assert len(messages) == 1 and messages[0]['role'] == 'user'
    return messages[0]['content'].rsplit('\n\n', 1)


def assert_native_decision_calls(model, count, expected_evidence=None):
    assert len(model.tok.calls) == count
    assert all(not enabled for enabled in model.model.grad_enabled)
    for (messages, kwargs), rendered, (encoded, encode_kwargs) in zip(
            model.tok.calls, model.tok.renderings, model.tok.encoded):
        assert [message['role'] for message in messages] == ['user']
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        assert rendered.endswith('assistant:<decision>')
        assert encoded == rendered
        assert encode_kwargs == {'return_tensors': 'pt', 'add_special_tokens': False}
        evidence, request = split_request(messages)
        if expected_evidence is None:
            assert '[Probe p_000000000000000000000000]' in evidence
            assert 'position=-' in evidence and 'position=+' in evidence
            assert 'denial' not in evidence and 'secret_target' not in evidence
        else:
            assert evidence == expected_evidence
        assert request
    assert any(len(rendered) > 600 for rendered in model.tok.renderings)


def expected_direction(axis):
    first = torch.eye(5)[axis]
    second = first + .3 * torch.eye(5)[4]
    result = first + second / second.norm()
    return result / result.norm()


def test_operation_is_matched_concise_definition_minus_recap_at_native_decision():
    model = wrapped_model()
    directions = mint_definition_operation_axis(model, (3, 1, 3))
    assert OPERATION_VERSION and 'decision' in OPERATION_VERSION and 'candidate' in OPERATION_VERSION
    assert_native_decision_calls(model, 4)
    evidence_cases = []
    for index in (0, 2):
        positive_evidence, positive_request = split_request(model.tok.calls[index][0])
        negative_evidence, negative_request = split_request(model.tok.calls[index + 1][0])
        assert positive_evidence == negative_evidence
        evidence_cases.append(positive_evidence)
        assert positive_request.startswith('In one concise sentence,')
        assert negative_request.startswith('In one concise sentence,')
        assert 'define the behavioral distinction' in positive_request
        assert 'lower and higher positions' in positive_request
        assert 'restate the individual behavior' in negative_request
        assert "each of this probe's memories" in negative_request
    assert evidence_cases[0] != evidence_cases[1]
    assert set(directions) == {1, 3}
    for direction in directions.values():
        torch.testing.assert_close(direction, expected_direction(0))
        torch.testing.assert_close(direction.norm(), torch.tensor(1.))
        assert torch.isfinite(direction).all()


def test_nuisances_are_independent_same_phase_factors_preserving_definition_operation():
    model = wrapped_model()
    nuisances = mint_definition_operation_nuisances(model, (1, 3))
    assert set(nuisances) == {'verbosity', 'enumeration', 'topic', 'generic_framing'}
    assert_native_decision_calls(model, 16)
    for axis_index, name in enumerate(nuisances):
        assert set(nuisances[name]) == {1, 3}
        for direction in nuisances[name].values():
            torch.testing.assert_close(direction, expected_direction(axis_index))
            torch.testing.assert_close(direction.norm(), torch.tensor(1.))
            assert torch.isfinite(direction).all()
    for index in range(0, 16, 2):
        positive_evidence, positive_request = split_request(model.tok.calls[index][0])
        negative_evidence, negative_request = split_request(model.tok.calls[index + 1][0])
        assert 'define the behavioral distinction' in positive_request.lower()
        assert 'define the behavioral distinction' in negative_request.lower()
        assert 'lower and higher positions' in positive_request
        assert 'lower and higher positions' in negative_request
        assert 'individual behavior' not in positive_request and 'individual behavior' not in negative_request
        if index in (8, 10):
            assert positive_request == negative_request
            assert positive_evidence != negative_evidence
        else:
            assert positive_evidence == negative_evidence
    for index in (0, 2):
        _, positive = split_request(model.tok.calls[index][0])
        _, negative = split_request(model.tok.calls[index + 1][0])
        assert positive.replace('detailed', 'concise') == negative
    for index in (4, 6):
        _, positive = split_request(model.tok.calls[index][0])
        _, negative = split_request(model.tok.calls[index + 1][0])
        assert positive.replace('numbered', 'prose') == negative
    for index in (12, 14):
        _, positive = split_request(model.tok.calls[index][0])
        _, negative = split_request(model.tok.calls[index + 1][0])
        assert positive.split(' Frame this distinction')[0] == negative.split(' Frame this distinction')[0]
        assert 'conversations with an AI assistant' in positive
        assert 'directly as a behavioral pattern' in negative


CONTEXT_EVIDENCE = (
    '  [Probe p_ffffffffffffffffffffffff]\n'
    'These are measured centered positions; zero is the rolling baseline.\n'
    'Memory 0: position=-0.80; input="What happened?"; reply="I cannot yet tell whether the report is correct."\n'
    'Memory 1: position=-0.25; input="What happened?"; reply="The available reports leave several possibilities open."\n'
    'Memory 2: position=+0.35; input="What happened?"; reply="The account seems settled, based on the matching records."\n'
    'Memory 3: position=+0.85; input="What happened?"; reply="The records consistently support the same account."\n  '
)


def test_contextual_operation_keeps_exact_evidence_and_varies_only_two_task_paraphrases():
    model = wrapped_model()
    directions = mint_definition_operation_axis(model, (1, 3), evidence=CONTEXT_EVIDENCE)
    assert 'contextual' in OPERATION_VERSION
    assert_native_decision_calls(model, 4, expected_evidence=CONTEXT_EVIDENCE)
    requests = []
    for index in (0, 2):
        positive_evidence, positive_request = split_request(model.tok.calls[index][0])
        negative_evidence, negative_request = split_request(model.tok.calls[index + 1][0])
        assert positive_evidence == negative_evidence == CONTEXT_EVIDENCE
        assert positive_request.startswith('In one concise sentence,')
        assert negative_request.startswith('In one concise sentence,')
        assert 'lower and higher positions' in positive_request
        assert ('individual behavior' in negative_request or 'separate concrete behavior' in negative_request)
        requests.append((positive_request, negative_request))
    assert requests[0] != requests[1]
    assert 'define the behavioral distinction' in requests[0][0]
    assert 'general behavioral rule' in requests[1][0]
    assert set(directions) == {1, 3}
    for direction in directions.values():
        torch.testing.assert_close(direction, expected_direction(0))


def test_contextual_nuisances_preserve_exact_evidence_and_omit_topic_rewriting():
    model = wrapped_model()
    nuisances = mint_definition_operation_nuisances(model, (1, 3), evidence=CONTEXT_EVIDENCE)
    assert set(nuisances) == {'verbosity', 'enumeration', 'generic_framing'}
    assert_native_decision_calls(model, 12, expected_evidence=CONTEXT_EVIDENCE)
    for axis_index, name in enumerate(nuisances):
        for direction in nuisances[name].values():
            torch.testing.assert_close(direction, expected_direction(axis_index))
    for start in (0, 4, 8):
        first_pair, second_pair = [], []
        for index, collected in ((start, first_pair), (start + 2, second_pair)):
            _, positive = split_request(model.tok.calls[index][0])
            _, negative = split_request(model.tok.calls[index + 1][0])
            assert 'lower and higher positions' in positive and 'lower and higher positions' in negative
            assert 'individual behavior' not in positive and 'individual behavior' not in negative
            collected.extend((positive, negative))
        assert first_pair != second_pair
    for index in (0, 2):
        _, positive = split_request(model.tok.calls[index][0])
        _, negative = split_request(model.tok.calls[index + 1][0])
        assert positive.replace('detailed', 'concise') == negative
    for index in (4, 6):
        _, positive = split_request(model.tok.calls[index][0])
        _, negative = split_request(model.tok.calls[index + 1][0])
        assert positive.replace('numbered', 'prose') == negative
    for index in (8, 10):
        _, positive = split_request(model.tok.calls[index][0])
        _, negative = split_request(model.tok.calls[index + 1][0])
        assert 'conversations with an AI assistant' in positive
        assert 'directly as a behavioral' in negative


@pytest.mark.parametrize('mint', [mint_definition_operation_axis, mint_definition_operation_nuisances])
@pytest.mark.parametrize('evidence', ['', ' \n\t', 1, True, [], {}, b'measured evidence'])
def test_invalid_contextual_evidence_is_rejected_before_model_forward(mint, evidence):
    model = wrapped_model()
    with pytest.raises(ValueError, match='evidence must be a nonempty string'):
        mint(model, (0,), evidence=evidence)
    assert model.tok.calls == []


@pytest.mark.parametrize('mint', [mint_definition_operation_axis, mint_definition_operation_nuisances])
def test_explicit_none_retains_default_synthetic_cases(mint):
    implicit_model, explicit_model = wrapped_model(), wrapped_model()
    implicit = mint(implicit_model, (0,))
    explicit = mint(explicit_model, (0,), evidence=None)
    assert implicit_model.tok.calls == explicit_model.tok.calls
    if mint is mint_definition_operation_axis:
        torch.testing.assert_close(implicit[0], explicit[0])
    else:
        assert set(implicit) == set(explicit) == {'verbosity', 'enumeration', 'topic', 'generic_framing'}
        for name in implicit:
            torch.testing.assert_close(implicit[name][0], explicit[name][0])


@pytest.mark.parametrize('mint', [mint_definition_operation_axis, mint_definition_operation_nuisances])
@pytest.mark.parametrize('failure, message', [
    ('zero', 'zero or cancelled'), ('cancel', 'zero or cancelled'),
    ('nonfinite', 'invalid native state'), ('pair_width', 'incompatible state widths'),
    ('case_width', 'incompatible state widths'),
])
def test_invalid_decision_candidates_fail_clearly(mint, failure, message):
    with pytest.raises(ValueError, match=message):
        mint(wrapped_model(failure=failure), (0,))


@pytest.mark.parametrize('mint', [mint_definition_operation_axis, mint_definition_operation_nuisances])
def test_missing_or_invalid_layers_fail_without_substitution(mint):
    with pytest.raises(ValueError, match='missing requested layer 4'):
        mint(wrapped_model(), (4,))
    for layers in ((), (-1,), (.5,), (True,), (float('nan'),), None):
        with pytest.raises(ValueError, match='layers|indices'):
            mint(wrapped_model(), layers)
