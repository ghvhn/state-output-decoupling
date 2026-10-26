"""Check native relation calibration, conditional delivery, and safe storage."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invariants import probe_definition_relations as relations


class Config:
    _name_or_path = 'test/relation-model'
    _commit_hash = 'a' * 40
    _attn_implementation = 'sdpa'
    quantization_config = None

    def to_dict(self):
        return {'model_type': 'test', 'hidden_size': 5, 'num_hidden_layers': 3}


class Tokenizer:
    name_or_path = 'test/relation-tokenizer'
    chat_template = 'native-user-assistant-v1'
    init_kwargs = {}
    special_tokens_map = {'eos_token': '<eot>'}
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"test":"relation"}')

    def get_vocab(self):
        return {'x': 0, '<eot>': 1}


def wrapped_model():
    raw = SimpleNamespace(config=Config(), dtype=torch.float32,
                          parameters=lambda: iter((torch.nn.Parameter(torch.ones(1)),)))
    return SimpleNamespace(model=raw, tok=Tokenizer(), device='cpu', d_model=5, n_layers=3)


class NativeStates:
    def __init__(self):
        self.calls = []
        self.failure = None

    def __call__(self, model, messages, layers, axis, *, add_generation_prompt):
        assert add_generation_prompt is True and not torch.is_grad_enabled()
        self.calls.append((copy.deepcopy(messages), tuple(layers), axis))
        evidence = messages[-1]['content']
        theme = next((index for index, values in enumerate(relations._THEMES)
                      if values[2] in evidence), None)
        if theme is None:
            theme = int(hashlib.sha256(evidence.encode()).hexdigest()[:3], 16) / 1000
        reverse = 'Memory 0: position=+' in evidence
        stage = relations.STEP_ORDER.index(axis)
        source = torch.tensor([1 + theme * .7, 1. if reverse else -1.,
                               .5 + stage * .4, .2 + theme * int(reverse) * .15, .8 + theme ** 2 * .1])
        result = {}
        for layer in layers:
            current = source * (1 + .1 * layer)
            if len(messages) > 1:
                current = current + .5
                if 'whereas' in messages[1]['content'] or 'The distinction is' in messages[1]['content']:
                    current = current + torch.stack((source[0] * .1 + source[1] * .3,
                        source[1] * .2 + stage * .1, source[4] * .05,
                        .1 + source[3] * .1, torch.tensor(.2))) * (layer + 1)
            if self.failure == 'nan':
                current[0] = float('nan')
            if self.failure == 'width':
                current = current[:-1]
            if self.failure == 'constant':
                current = torch.ones(5) * (1 if len(messages) == 1 else 2)
            result[layer] = current
        return result


@pytest.fixture
def native(monkeypatch):
    capture = NativeStates()
    monkeypatch.setattr(relations, '_native_states', capture)
    return capture


def test_mint_pairs_native_decisions_with_identical_final_queries_and_no_instruction_seed(tmp_path, native):
    bank, metadata = relations.get_or_mint_definition_relations(wrapped_model(), (2, 0, 2), tmp_path / 'relations')
    assert len(native.calls) == 144
    for index in range(0, len(native.calls), 3):
        (source, source_layers, stage), (positive, positive_layers, positive_stage), (negative, negative_layers, negative_stage) = native.calls[index:index + 3]
        assert source_layers == positive_layers == negative_layers == (0, 2)
        assert stage == positive_stage == negative_stage
        assert source == [positive[-1]] == [negative[-1]]
        assert [message['role'] for message in positive] == ['user', 'assistant', 'user', 'assistant', 'user']
        assert [message['role'] for message in negative] == ['user', 'assistant', 'user', 'assistant', 'user']
        assert positive[0] == negative[0] and positive[2] == negative[2]
        assert positive[1] != negative[1] and positive[3] != negative[3]
        assert relations._TEACHERS[0][2] not in source[0]['content']
        assert relations._TEACHERS[1][2] not in source[0]['content']
        assert 'In one concise sentence' not in str(positive + negative)
        assert 'denial' not in str(positive + negative)
        if stage == 'compare':
            assert '[Candidate ' not in source[0]['content']
        else:
            assert '[Candidate c_' in source[0]['content']
    assert tuple(bank) == relations.STEP_ORDER
    for profiles in bank.values():
        assert set(profiles) == {0, 2}
        for profile in profiles.values():
            assert profile['n_samples'] == 12 and profile['width'] == 5
            assert profile['diagnostics']['adaptive']
            assert profile['changes_norm_mean'] > 0
            assert profile['source_relations'].device.type == 'cpu'
    assert Path(metadata['path']).is_absolute()
    assert metadata['load_status'] == 'minted' and metadata['cache_status'] == 'minted'
    assert metadata['identity']['source_units'] == 'raw_native_residual'
    assert metadata['identity']['target_units'] == 'raw_native_state_difference'
    assert metadata['provenance']['calibration_rows_per_step'] == 12
    assert metadata['provenance']['semantic_truth_checked'] is False
    assert metadata['calibration']['compare'][0]['heldout_rows'] == 2
    assert metadata['calibration']['compare'][0]['predicted_change_distance'] > 0
    payload = torch.load(metadata['path'], weights_only=True, map_location='cpu')
    plain = json.dumps({key: value for key, value in payload.items() if key != 'bank'})
    assert 'Memory 0' not in plain and 'packages' not in plain and 'p_000' not in plain


def test_calibration_covers_both_polarities_and_prior_candidate_matches_pipeline_hash():
    for step in relations.STEP_ORDER:
        rows = relations._calibration_rows(step)
        assert len(rows) == 12 and {row['theme'] for row in rows} == set(range(6))
        for index in range(0, 12, 2):
            forward, reverse = rows[index:index + 2]
            assert forward['orientation'] == 0 and reverse['orientation'] == 1
            assert 'Memory 0: position=-0.75' in forward['evidence']
            assert 'Memory 0: position=+0.75' in reverse['evidence']
        if step != 'compare':
            previous = relations.STEP_ORDER[relations.STEP_ORDER.index(step) - 1]
            original, outputs, _ = relations._case(relations._THEMES[0], 0)
            expected = 'c_' + hashlib.sha256(outputs[previous].encode()).hexdigest()[:24]
            assert rows[0]['evidence'] == original + f'\n\n[Candidate {expected}]\n' + outputs[previous]


def test_cached_bank_reloads_without_any_native_forwards_and_does_not_alias_tensors(tmp_path, native):
    model = wrapped_model()
    first, initial = relations.get_or_mint_definition_relations(model, (0, 2), tmp_path)
    native.calls.clear()
    second, loaded = relations.get_or_mint_definition_relations(model, (2, 0), tmp_path)
    assert native.calls == [] and loaded['cache_status'] == 'hit'
    assert initial['path'] == loaded['path']
    second['compare'][0]['coefficients'].zero_()
    assert torch.count_nonzero(first['compare'][0]['coefficients']) > 0
    third, _ = relations.get_or_mint_definition_relations(model, (0, 2), tmp_path)
    torch.testing.assert_close(first['compare'][0]['coefficients'], third['compare'][0]['coefficients'])


@pytest.mark.parametrize('mutation', ['dtype', 'template', 'revision', 'recipe', 'layers'])
def test_identity_changes_create_separate_banks(tmp_path, native, monkeypatch, mutation):
    model = wrapped_model()
    _, initial = relations.get_or_mint_definition_relations(model, (0, 2), tmp_path)
    layers = (0, 2)
    if mutation == 'dtype':
        model.model.dtype = torch.float16
    elif mutation == 'template':
        model.tok.chat_template = 'different-native-template'
    elif mutation == 'revision':
        model.model.config = copy.copy(model.model.config)
        model.model.config._commit_hash = 'b' * 40
    elif mutation == 'recipe':
        monkeypatch.setattr(relations, 'RIDGE', .02)
    else:
        layers = (1, 2)
    _, changed = relations.get_or_mint_definition_relations(model, layers, tmp_path)
    assert initial['identity_hash'] != changed['identity_hash']
    assert Path(initial['path']).exists() and Path(changed['path']).exists()


@pytest.mark.parametrize('corruption', ['missing_step', 'missing_layer', 'bad_tensor', 'unsafe_text', 'bad_provenance'])
def test_corrupt_bundle_is_rejected_and_reminted_once(tmp_path, native, corruption):
    path = tmp_path / 'fixed.pt'
    _, metadata = relations.get_or_mint_definition_relations(wrapped_model(), (0, 2), path)
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if corruption == 'missing_step':
        del payload['bank']['verify']
    elif corruption == 'missing_layer':
        del payload['bank']['compare'][2]
    elif corruption == 'bad_tensor':
        payload['bank']['compare'][0]['coefficients'][0, 0] = float('nan')
    elif corruption == 'unsafe_text':
        payload['unexpected_teacher_text'] = 'an arbitrary extra seed'
    else:
        payload['provenance']['semantic_truth_checked'] = True
    torch.save(payload, path)
    native.calls.clear()
    bank, reminted = relations.get_or_mint_definition_relations(wrapped_model(), (0, 2), path)
    assert len(native.calls) == 144 and reminted['cache_status'] == 'minted'
    assert 'cache_rejection' in reminted and metadata['path'] == reminted['path']
    assert set(bank) == set(relations.STEP_ORDER)
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_remint_keeps_existing_corrupt_file_and_cleans_temporary_files(tmp_path, native):
    path = tmp_path / 'fixed.pt'
    path.write_bytes(b'keep this existing file')
    native.failure = 'nan'
    with pytest.raises(ValueError, match='finite'):
        relations.get_or_mint_definition_relations(wrapped_model(), (0, 2), path)
    assert path.read_bytes() == b'keep this existing file'
    assert len(native.calls) == 1 and not list(tmp_path.glob('*.tmp'))


def test_failed_serialized_validation_preserves_old_bank_and_cleans_tempfile(tmp_path, native):
    path = tmp_path / 'fixed.pt'
    _, metadata = relations.get_or_mint_definition_relations(wrapped_model(), (0,), path)
    before = path.read_bytes()
    payload = torch.load(path, weights_only=True, map_location='cpu')
    payload['bank']['compare'][0]['coefficients'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        relations._write_atomic(path, payload, metadata['identity'], metadata['identity_hash'], (0,))
    assert path.read_bytes() == before and not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('failure', ['width', 'constant'])
def test_invalid_calibration_never_writes_a_bank(tmp_path, native, failure):
    native.failure = failure
    with pytest.raises(ValueError):
        relations.get_or_mint_definition_relations(wrapped_model(), (0,), tmp_path)
    assert not list(tmp_path.glob('*.pt'))


def test_resolution_is_one_unprimed_actual_evidence_call_with_adaptive_direction_and_amplitude(tmp_path, native):
    model = wrapped_model()
    bank, _ = relations.get_or_mint_definition_relations(model, (0, 2), tmp_path)
    first_e = relations._calibration_rows('compare')[0]['evidence']
    second_e = relations._calibration_rows('compare')[1]['evidence']
    native.calls.clear()
    first, diag = relations.resolve_definition_relations(model, bank, 'compare', first_e)
    assert native.calls == [([{'role': 'user', 'content': first_e}], (0, 2), 'compare')]
    second, changed = relations.resolve_definition_relations(model, bank, 'compare', second_e)
    assert diag['evidence_sha256'] == hashlib.sha256(first_e.encode()).hexdigest()
    assert diag['semantic_truth_checked'] is False and set(diag['layer_scales']) == {0, 2}
    assert not torch.allclose(first[0], second[0])
    assert diag['layer_scales'][0] != pytest.approx(changed['layer_scales'][0])
    for layer, direction in first.items():
        assert direction.shape == (5,) and direction.device.type == 'cpu' and direction.dtype == torch.float32
        torch.testing.assert_close(direction.norm(), torch.tensor(1.))
        norm = diag['layers'][layer]['raw_delta_norm']
        reference = bank['compare'][layer]['changes_norm_mean']
        assert diag['layer_scales'][layer] == pytest.approx(min(2., norm / reference))


@pytest.mark.parametrize('prediction', ['zero', 'nan'])
def test_bad_prediction_fails_without_a_fixed_direction_fallback(tmp_path, native, monkeypatch, prediction):
    bank, _ = relations.get_or_mint_definition_relations(wrapped_model(), (0,), tmp_path)
    monkeypatch.setattr(relations, 'predict_relational_conversion',
                        lambda profile, source: (torch.zeros(5) if prediction == 'zero' else torch.full((5,), float('nan')), {}))
    native.calls.clear()
    with pytest.raises(ValueError, match='finite and nonzero'):
        relations.resolve_definition_relations(wrapped_model(), bank, 'verify', 'Actual unprimed evidence.')
    assert len(native.calls) == 1


def test_relative_amplitude_is_bounded_without_changing_predicted_direction(tmp_path, native, monkeypatch):
    bank, _ = relations.get_or_mint_definition_relations(wrapped_model(), (0,), tmp_path)
    delta = torch.tensor([100., -200., 300., 100., 50.])
    monkeypatch.setattr(relations, 'predict_relational_conversion', lambda profile, source: (delta, {}))
    directions, diagnostics = relations.resolve_definition_relations(wrapped_model(), bank, 'verify', 'Actual evidence.')
    assert diagnostics['layer_scales'][0] == 2.
    torch.testing.assert_close(directions[0], delta / delta.norm())


def test_bad_operation_or_missing_coverage_fails_before_native_forward(tmp_path, native):
    bank, _ = relations.get_or_mint_definition_relations(wrapped_model(), (0, 2), tmp_path)
    native.calls.clear()
    with pytest.raises(ValueError, match='declared operation'):
        relations.resolve_definition_relations(wrapped_model(), bank, 'invented', 'E')
    del bank['verify'][2]
    with pytest.raises(ValueError, match='exactly the requested layers'):
        relations.resolve_definition_relations(wrapped_model(), bank, 'compare', 'E')
    assert native.calls == []


def test_real_tiny_native_model_uses_unanswered_assistant_headers_and_cpu_cache(tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from invariants.engine import HF

    rows = [row for step in relations.STEP_ORDER for row in relations._calibration_rows(step)]
    texts = [message['content'] for row in rows for key in ('positive', 'negative') for message in row[key]]
    special = ['[UNK]', '[PAD]', '[BOS]', '[EOS]', '<user>', '<assistant>', '<eot>']
    vocabulary = {word: index for index, word in enumerate(special + sorted(set(' '.join(texts).split()) - set(special)))}
    backend = BackendTokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
                                 bos_token='[BOS]', eos_token='[EOS]', name_or_path='test/native-relations')
    tok.model_input_names = ['input_ids', 'attention_mask']
    tok.chat_template = ("{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' <eot> ' }}{% endfor %}"
                         "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}")
    config = LlamaConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=2048, attention_dropout=0.)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    handle = None
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            raw = LlamaForCausalLM(config).cpu().eval()
        raw.config._name_or_path = 'test/native-relation-llama'
        raw.config._commit_hash = 'c' * 40
        wrapped = HF(raw, tok)
        endpoints = []
        def capture(_module, args, kwargs):
            assert not torch.is_grad_enabled()
            ids = kwargs['input_ids']
            assert ids.device.type == 'cpu'
            assert int(ids[0, -1]) == tok.convert_tokens_to_ids('<assistant>')
            endpoints.append(int(ids.shape[-1]))
        handle = raw.register_forward_pre_hook(capture, with_kwargs=True)
        bank, metadata = relations.get_or_mint_definition_relations(wrapped, (0, 1), tmp_path)
        assert len(endpoints) == 144 and metadata['identity']['effective_load']['device'] == 'cpu'
        for index in range(0, 144, 3):
            assert endpoints[index] < endpoints[index + 1] and endpoints[index] < endpoints[index + 2]
        reloaded, loaded = relations.get_or_mint_definition_relations(wrapped, (0, 1), tmp_path)
        assert loaded['cache_status'] == 'hit' and len(endpoints) == 144
        evidence = relations._calibration_rows('isolate')[3]['evidence']
        directions, diagnostics = relations.resolve_definition_relations(wrapped, reloaded, 'isolate', evidence)
        assert len(endpoints) == 145 and set(directions) == {0, 1}
        assert diagnostics['semantic_truth_checked'] is False
        for layer, direction in directions.items():
            torch.testing.assert_close(direction.norm(), torch.tensor(1.))
            torch.testing.assert_close(bank['isolate'][layer]['center_source'], reloaded['isolate'][layer]['center_source'])
    finally:
        if handle is not None:
            handle.remove()
        torch.set_num_threads(previous_threads)
