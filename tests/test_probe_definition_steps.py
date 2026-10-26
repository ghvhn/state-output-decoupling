"""Check named step minting and persistent identity without a live model."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invariants import probe_definition_steps as steps


class Config:
    def __init__(self):
        self._name_or_path = 'toy/causal-model'
        self._commit_hash = 'a' * 40
        self.hidden_size = 5
        self.num_hidden_layers = 4
        self._attn_implementation = 'sdpa'
        self.quantization_config = None

    def to_dict(self):
        return {'_name_or_path': self._name_or_path, 'hidden_size': self.hidden_size,
                'num_hidden_layers': self.num_hidden_layers, 'model_type': 'toy'}


class Tokenizer:
    def __init__(self):
        self.calls, self.renderings, self.encoded = [], [], []
        self.name_or_path = 'toy/tokenizer'
        self.chat_template = 'native-user-and-assistant-decision-v1'
        self.init_kwargs = {'add_bos_token': False}
        self.special_tokens_map = {'eos_token': '<eos>'}
        self.vocabulary = {'x': 0, '<eos>': 1}
        self.backend_serial = '{"normalizer":"identity","vocabulary":{"x":0,"<eos>":1}}'
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: self.backend_serial)

    def get_vocab(self):
        return dict(self.vocabulary)

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
    def __init__(self, failure=None):
        self.config, self.dtype = Config(), torch.float32
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.grad_enabled, self.failure = [], failure

    def parameters(self):
        return iter((self.parameter,))

    def __call__(self, input_ids, **kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        assert kwargs['output_hidden_states'] and not kwargs['use_cache']
        index = int(input_ids[0, 0])
        axis, within_axis = (index // 4) % 4, index % 4
        case, side = divmod(within_axis, 2)
        vector = torch.eye(5)[axis]
        if case:
            vector = 100 * (vector + .3 * torch.eye(5)[4])
        if self.failure == 'cancel' and case:
            vector = -torch.eye(5)[axis]
        if self.failure == 'zero':
            vector.zero_()
        states = [torch.ones(1, 2, 5)]
        for layer in range(4):
            hidden = torch.ones(1, 2, 5)
            if side == 0:
                hidden[0, 0] -= 100 * vector
                hidden[0, -1] += (layer + 1) * vector
            if self.failure == 'nonfinite' and axis == 3:
                hidden[0, -1, 0] = float('nan')
            states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(states))


def wrapped_model(failure=None):
    return SimpleNamespace(model=Model(failure), tok=Tokenizer(), device='cpu', d_model=5, n_layers=4)


def assert_valid_bank(bank, layers):
    assert tuple(bank) == steps.STEP_ORDER
    for directions in bank.values():
        assert set(directions) == set(layers)
        for vector in directions.values():
            assert vector.dtype == torch.float32 and vector.device.type == 'cpu'
            assert vector.shape == (5,) and torch.isfinite(vector).all()
            torch.testing.assert_close(vector.norm(), torch.tensor(1.))


def test_named_steps_use_matched_native_decision_requests_and_persist_only_vectors(tmp_path):
    model = wrapped_model()
    bank, metadata = steps.get_or_mint_definition_steps(model, (3, 1, 3), tmp_path / 'banks')
    assert steps.STEP_ORDER == ('compare', 'isolate', 'distill', 'verify')
    assert steps.STEP_GAINS == {step: .04 for step in steps.STEP_ORDER}
    assert 'candidate' in steps.STEP_VERSION
    assert_valid_bank(bank, (1, 3))
    assert len(model.tok.calls) == 16 and not any(model.model.grad_enabled)
    for (messages, kwargs), rendering, (encoded, encode_kwargs) in zip(
            model.tok.calls, model.tok.renderings, model.tok.encoded):
        assert [message['role'] for message in messages] == ['user']
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        assert rendering.endswith('assistant:<decision>') and encoded == rendering
        assert encode_kwargs == {'return_tensors': 'pt', 'add_special_tokens': False}
        assert 'p_000000000000000000000000' in messages[0]['content']
        assert 'position=-' in messages[0]['content'] and 'position=+' in messages[0]['content']
        assert 'denial' not in str(messages) and 'secret_target' not in str(messages)
    for index in range(0, 16, 2):
        positive_evidence, positive_request = model.tok.calls[index][0][0]['content'].rsplit('\n\n', 1)
        negative_evidence, negative_request = model.tok.calls[index + 1][0][0]['content'].rsplit('\n\n', 1)
        assert positive_evidence == negative_evidence and positive_request != negative_request
        assert positive_request.startswith('In one concise sentence,')
        assert negative_request.startswith('In one concise sentence,')
    for axis, name in enumerate(steps.STEP_ORDER):
        first = torch.eye(5)[axis]
        second = first + .3 * torch.eye(5)[4]
        expected = first + second / second.norm()
        expected /= expected.norm()
        torch.testing.assert_close(bank[name][1], expected)
    path = Path(metadata['path'])
    assert path.is_absolute() and path.exists() and path.parent == tmp_path / 'banks'
    assert path.name == f'bank-{metadata["identity_hash"]}.pt'
    assert metadata['load_status'] == 'minted' and metadata['cache_status'] == 'minted'
    assert metadata['provenance']['endpoint'] == 'native_assistant_decision'
    payload = torch.load(path, weights_only=True, map_location='cpu')
    assert set(payload['bank']) == set(steps.STEP_ORDER)
    assert 'In one concise sentence' not in json.dumps(metadata)
    assert 'corridor' not in json.dumps(metadata)


def test_cache_reloads_without_teacher_forwards_and_returns_independent_tensors(tmp_path):
    first_model = wrapped_model()
    first, metadata = steps.get_or_mint_definition_steps(first_model, (0, 2), tmp_path / 'banks')
    second_model = wrapped_model()
    second, second_metadata = steps.get_or_mint_definition_steps(second_model, (2, 0), tmp_path / 'banks')
    assert second_model.tok.calls == []
    assert second_metadata['load_status'] == 'loaded' and second_metadata['cache_status'] == 'hit'
    assert metadata['path'] == second_metadata['path']
    for name in steps.STEP_ORDER:
        for layer in (0, 2):
            torch.testing.assert_close(first[name][layer], second[name][layer])
    second['compare'][0].zero_()
    third, _ = steps.get_or_mint_definition_steps(wrapped_model(), (0, 2), tmp_path / 'banks')
    torch.testing.assert_close(third['compare'][0], first['compare'][0])


def test_directory_keeps_alternating_layer_banks_without_overwrite(tmp_path):
    model = wrapped_model()
    _, first = steps.get_or_mint_definition_steps(model, (0, 1, 2), tmp_path / 'banks')
    _, second = steps.get_or_mint_definition_steps(model, (0, 1, 2, 3), tmp_path / 'banks')
    assert first['path'] != second['path']
    assert len(list((tmp_path / 'banks').glob('*.pt'))) == 2
    calls = len(model.tok.calls)
    _, again = steps.get_or_mint_definition_steps(model, (0, 1, 2), tmp_path / 'banks')
    assert again['cache_status'] == 'hit' and len(model.tok.calls) == calls


@pytest.mark.parametrize('change', ['model', 'revision', 'tokenizer_name', 'template', 'vocabulary',
                                  'backend', 'special_tokens', 'dtype', 'quantization', 'device_map', 'version'])
def test_identity_changes_get_separate_banks_and_never_reuse_vectors(tmp_path, monkeypatch, change):
    _, first = steps.get_or_mint_definition_steps(wrapped_model(), (0,), tmp_path / 'banks')
    model = wrapped_model()
    if change == 'model':
        model.model.config._name_or_path = 'toy/different-model'
    elif change == 'revision':
        model.model.config._commit_hash = 'b' * 40
    elif change == 'tokenizer_name':
        model.tok.name_or_path = 'toy/different-tokenizer'
    elif change == 'template':
        model.tok.chat_template += '-v2'
    elif change == 'vocabulary':
        model.tok.vocabulary['new'] = 2
    elif change == 'backend':
        model.tok.backend_serial = '{"normalizer":"changed"}'
    elif change == 'special_tokens':
        model.tok.special_tokens_map['eos_token'] = '<changed>'
    elif change == 'dtype':
        model.model.dtype = torch.bfloat16
    elif change == 'quantization':
        model.model.is_loaded_in_4bit = True
        model.model.config.quantization_config = {'load_in_4bit': True, 'compute_dtype': torch.float16}
    elif change == 'device_map':
        model.model.hf_device_map = {'layer_0': 'cpu', 'layer_1': 'cuda:0'}
    elif change == 'version':
        monkeypatch.setattr(steps, 'STEP_VERSION', steps.STEP_VERSION + '-v2')
    _, second = steps.get_or_mint_definition_steps(model, (0,), tmp_path / 'banks')
    assert second['path'] != first['path'] and second['identity_hash'] != first['identity_hash']
    assert second['cache_status'] == 'minted' and len(model.tok.calls) == 16


@pytest.mark.parametrize('damage', ['missing_step', 'extra_step', 'missing_layer', 'extra_layer',
                                  'width', 'dtype', 'zero', 'nonfinite', 'not_unit', 'identity', 'boolean_layer'])
def test_malformed_cache_is_rejected_and_reminted_as_complete_bank(tmp_path, damage):
    path = tmp_path / 'explicit.pt'
    _, metadata = steps.get_or_mint_definition_steps(wrapped_model(), (0, 2), path)
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if damage == 'missing_step':
        payload['bank'].pop('verify')
    elif damage == 'extra_step':
        payload['bank']['other'] = payload['bank']['verify']
    elif damage == 'missing_layer':
        payload['bank']['compare'].pop(2)
    elif damage == 'extra_layer':
        payload['bank']['compare'][3] = torch.eye(5)[0]
    elif damage == 'width':
        payload['bank']['verify'][2] = torch.eye(6)[0]
    elif damage == 'dtype':
        payload['bank']['verify'][2] = torch.eye(5, dtype=torch.float64)[0]
    elif damage == 'zero':
        payload['bank']['verify'][2].zero_()
    elif damage == 'nonfinite':
        payload['bank']['verify'][2][0] = float('nan')
    elif damage == 'not_unit':
        payload['bank']['verify'][2] *= 2
    elif damage == 'identity':
        payload['identity']['model']['source'] = 'toy/wrong-model'
    elif damage == 'boolean_layer':
        payload['bank']['compare'][False] = payload['bank']['compare'].pop(0)
    torch.save(payload, path)
    model = wrapped_model()
    repaired, repaired_metadata = steps.get_or_mint_definition_steps(model, (0, 2), path)
    assert_valid_bank(repaired, (0, 2))
    assert repaired_metadata['path'] == metadata['path']
    assert repaired_metadata['cache_status'] == 'minted' and 'cache_rejection' in repaired_metadata
    assert len(model.tok.calls) == 16


def test_incompatible_explicit_cache_survives_failed_mint_without_partial_bank(tmp_path):
    path = tmp_path / 'steps.pt'
    steps.get_or_mint_definition_steps(wrapped_model(), (0,), path)
    original = path.read_bytes()
    model = wrapped_model(failure='nonfinite')
    model.model.config._commit_hash = 'b' * 40
    with pytest.raises(ValueError, match='invalid native state'):
        steps.get_or_mint_definition_steps(model, (0,), path)
    assert path.read_bytes() == original and not list(tmp_path.glob('*.tmp'))


def test_atomic_write_failure_preserves_existing_bundle_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / 'steps.pt'
    steps.get_or_mint_definition_steps(wrapped_model(), (0,), path)
    original = path.read_bytes()
    model = wrapped_model()
    model.model.config._commit_hash = 'b' * 40
    def fail_replace(*args):
        raise OSError('simulated replace failure')
    monkeypatch.setattr(steps.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='replace failure'):
        steps.get_or_mint_definition_steps(model, (0,), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


_UNSAFE_INVOCATIONS = []


def _unsafe_marker():
    _UNSAFE_INVOCATIONS.append(True)
    return {}


class UnsafePayload:
    def __reduce__(self):
        return _unsafe_marker, ()


def test_weights_only_loader_never_executes_arbitrary_pickle_and_replaces_invalid_bundle(tmp_path, monkeypatch):
    path = tmp_path / 'steps.pt'
    torch.save(UnsafePayload(), path)
    _UNSAFE_INVOCATIONS.clear()
    original_load = torch.load
    load_calls = []
    def checked_load(*args, **kwargs):
        assert kwargs['weights_only'] is True and kwargs['map_location'] == 'cpu'
        load_calls.append(args[0])
        return original_load(*args, **kwargs)
    monkeypatch.setattr(steps.torch, 'load', checked_load)
    bank, metadata = steps.get_or_mint_definition_steps(wrapped_model(), (0,), path)
    assert_valid_bank(bank, (0,))
    assert _UNSAFE_INVOCATIONS == [] and len(load_calls) == 2
    assert metadata['cache_status'] == 'minted' and 'cache_rejection' in metadata


@pytest.mark.parametrize('fault, message', [('source', 'source identity'),
                                         ('revision', 'immutable model revision'),
                                         ('template', 'native chat template'),
                                         ('backend', 'tokenizer backend'),
                                         ('vocabulary', 'vocabulary identity')])
def test_unknown_identity_fails_closed_before_mint_or_storage(tmp_path, fault, message):
    model = wrapped_model()
    if fault == 'source':
        model.model.config._name_or_path = ''
    elif fault == 'revision':
        model.model.config._commit_hash = None
    elif fault == 'template':
        model.tok.chat_template = None
    elif fault == 'backend':
        model.tok.backend_tokenizer = None
    elif fault == 'vocabulary':
        model.tok.vocabulary = {}
    with pytest.raises(ValueError, match=message):
        steps.get_or_mint_definition_steps(model, (0,), tmp_path / 'banks')
    assert model.tok.calls == [] and list(tmp_path.iterdir()) == []


def test_mutable_local_weight_manifest_invalidates_same_path_source(tmp_path):
    source = tmp_path / 'local-model'
    source.mkdir()
    weights = source / 'model.safetensors'
    weights.write_bytes(b'first-version')
    first_model = wrapped_model()
    first_model.model.config._name_or_path = str(source)
    first_model.model.config._commit_hash = None
    _, first = steps.get_or_mint_definition_steps(first_model, (0,), tmp_path / 'banks')
    weights.write_bytes(b'changed-version-with-different-size')
    second_model = wrapped_model()
    second_model.model.config._name_or_path = str(source)
    second_model.model.config._commit_hash = None
    _, second = steps.get_or_mint_definition_steps(second_model, (0,), tmp_path / 'banks')
    assert first['identity_hash'] != second['identity_hash']
    assert first['path'] != second['path'] and second['cache_status'] == 'minted'


@pytest.mark.parametrize('failure, message', [('zero', 'zero or cancelled'),
                                           ('cancel', 'zero or cancelled'),
                                           ('nonfinite', 'invalid native state')])
def test_invalid_minted_vectors_never_write_a_bank(tmp_path, failure, message):
    with pytest.raises(ValueError, match=message):
        steps.get_or_mint_definition_steps(wrapped_model(failure), (0,), tmp_path / 'banks')
    assert list(tmp_path.iterdir()) == []


def test_invalid_or_missing_layers_fail_before_storage(tmp_path):
    for layers in ((), (-1,), (.5,), (True,), (float('nan'),), None, (4,)):
        with pytest.raises(ValueError, match='layers|indices'):
            steps.get_or_mint_definition_steps(wrapped_model(), layers, tmp_path / 'banks')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('attribute,value', [('d_model', True), ('d_model', 5.5),
                                          ('n_layers', float('inf')), ('n_layers', 4.5)])
def test_model_dimensions_must_be_finite_integer_identities(tmp_path, attribute, value):
    model = wrapped_model()
    setattr(model, attribute, value)
    with pytest.raises(ValueError, match='width and layer count'):
        steps.get_or_mint_definition_steps(model, (0,), tmp_path / 'banks')
    assert model.tok.calls == [] and list(tmp_path.iterdir()) == []
