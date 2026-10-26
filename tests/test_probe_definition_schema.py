"""Native block capture and schema calibration checks without GPU resources."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from invariants import probe_definition_schema as schema
from invariants.probe_behavior import render_definition_evidence


class Config:
    _name_or_path = 'test/schema-model'
    _commit_hash = 'a' * 40
    _attn_implementation = 'sdpa'
    quantization_config = None
    def to_dict(self):
        return dict(model_type='schema-test', hidden_size=5, num_hidden_layers=3)


class Tokenizer:
    def __init__(self):
        self.name_or_path = 'test/schema-tokenizer'
        self.chat_template = 'native-schema-v1'
        self.init_kwargs = {}
        self.special_tokens_map = {'eos_token': '<eot>'}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"type":"schema-test"}')
        self.calls, self.texts, self.encoded = [], [], []
    def get_vocab(self):
        return {'x': 0, '<eot>': 1}
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        text = '\n'.join(f'{message["role"].upper()}:{message["content"]}' for message in messages)
        return text + '\nASSISTANT:'
    def __call__(self, text, **kwargs):
        self.encoded.append((text, kwargs))
        if text not in self.texts:
            self.texts.append(text)
        index = self.texts.index(text)
        return dict(input_ids=torch.tensor([[index, index, index]]), attention_mask=torch.ones(1, 3, dtype=torch.long))


class Block(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    def forward(self, hidden):
        return hidden + .1 * (self.layer + 1)


class Model(torch.nn.Module):
    def __init__(self, tok):
        super().__init__()
        self.config, self.tok = Config(), tok
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(layer) for layer in range(3)])
        self.grad_enabled, self.failure = [], None
    @property
    def dtype(self):
        return self.parameter.dtype
    def forward(self, input_ids, **kwargs):
        assert not kwargs['output_hidden_states'] and not kwargs['use_cache']
        self.grad_enabled.append(torch.is_grad_enabled())
        text = self.tok.texts[int(input_ids[0, -1])]
        evidence = text.rsplit('USER:', 1)[1]
        theme = next(index for index, item in enumerate(schema._THEMES) if item['family'] in evidence)
        reverse = 'Memory 0: position=+' in evidence
        source = torch.tensor([1. + .3 * theme, 1. if reverse else -1., .2 * theme ** 2,
                               .2 + .1 * theme * int(reverse), .5])
        if text.count('ASSISTANT:') > 1:
            teachers = text.split('USER:')[:-1]
            positive = any(any(marker in item for marker in ('are examples of', 'observable distinction',
                'prevents treating', 'Those cases do not distinguish', 'Within these examples')) for item in teachers)
            if positive:
                source += torch.tensor([.1 + .05 * theme, .3 if reverse else -.3,
                                        .1 * theme, .25, .15])
            source += .2
        hidden = torch.ones(1, 3, 5)
        hidden[0, -1] = source
        for index, block in enumerate(self.model.layers):
            if self.failure == 'forward' and index == 1:
                raise RuntimeError('native forward failed')
            if self.failure == 'nan' and index == 1:
                hidden[0, -1, 0] = float('nan')
            hidden = block(hidden)
        # Deliberately unrelated final normalization must not be captured.
        return SimpleNamespace(last_hidden_state=100 * hidden)


def wrapped():
    tok = Tokenizer()
    model = Model(tok)
    return SimpleNamespace(model=model, tok=tok, device='cpu', d_model=5, n_layers=3)


def test_native_pairs_capture_raw_blocks_at_exact_prefix_and_store_no_teacher_text(tmp_path):
    model = wrapped()
    prefix = 'A different exact prefix  '
    bank, metadata = schema.get_or_mint_definition_schema(model, (2, 0, 2), tmp_path, assistant_prefix=prefix)
    assert schema.SCHEMA_ORDER == ('genus', 'core', 'boundary', 'compose')
    assert schema.SCHEMA_GAINS == {role: .04 for role in schema.SCHEMA_ORDER}
    assert len(model.tok.calls) == 192 and len(model.model.grad_enabled) == 192
    assert not any(model.model.grad_enabled)
    assert all(not block._forward_hooks for block in model.model.model.layers)
    for index in range(0, 192, 3):
        source, positive, negative = [call[0] for call in model.tok.calls[index:index + 3]]
        assert source == [positive[-1]] == [negative[-1]]
        assert positive[0] == negative[0] and positive[2] == negative[2]
        assert positive[1] != negative[1] and positive[3] != negative[3]
        assert 'p_' in source[0]['content'] and '[Candidate' not in source[0]['content']
    for (messages, kwargs), (text, tokenize_kwargs) in zip(model.tok.calls, model.tok.encoded):
        assert kwargs == dict(tokenize=False, add_generation_prompt=True)
        assert text.endswith('\nASSISTANT:' + prefix)
        assert tokenize_kwargs == dict(return_tensors='pt', add_special_tokens=False)
        assert 'In one concise sentence' not in text and 'denial' not in text
    assert metadata['identity']['assistant_prefix'] == prefix
    assert metadata['identity']['source_format'] == 'original-evidence-native-prefix-block-output-v2'
    assert metadata['provenance']['online_intervention_source_shift'] == 'experimental_unverified'
    assert metadata['provenance']['endpoint'] == 'native_block_output_after_exact_assistant_prefix'
    assert metadata['provenance']['semantic_truth_checked'] is False
    assert metadata['holdouts']['boundary'][2]['heldout_theme'] == 7
    assert Path(metadata['path']).is_absolute()
    for role in schema.SCHEMA_ORDER:
        for profile in bank[role].values():
            assert profile['n_samples'] == 16 and profile['diagnostics']['adaptive']
            assert profile['center_source'].norm() < 10 # excludes fake final normalization
    payload = torch.load(metadata['path'], weights_only=True, map_location='cpu')
    plain = json.dumps({key: value for key, value in payload.items() if key != 'bank'})
    assert 'Memory 0' not in plain and 'packages' not in plain and 'p_' not in plain


def test_schema_examples_keep_genus_under_reversal_and_show_real_exceptions_and_indeterminate_support():
    aliases = set()
    for theme_index, theme in enumerate(schema._THEMES):
        forward, target, recap = schema._case(theme, 0)
        reverse, reversed_target, _ = schema._case(theme, 1)
        actions = [theme['first'], theme['first'], theme['second'], theme['second']]
        contexts = [f'A routine case involving {theme["family"]}.'] * 4
        if theme.get('kind') == 'exception':
            actions[3] = theme['first']
        elif theme.get('kind') == 'indeterminate':
            actions[1] = actions[3] = theme['neutral']
            contexts = [f'A discriminating case involving {theme["family"]}.',
                        f'A case involving {theme["family"]} without the information needed for the distinction.'] * 2
        namespace = forward.split('[Probe ')[1].split(']')[0]
        for orientation, evidence in enumerate((forward, reverse)):
            selected = [(index, {'reading': (-1 if orientation else 1) * reading,
                                 'input': contexts[index], 'reply': f'I {actions[index]}.'})
                        for index, reading in enumerate((-.75, -.25, .25, .75))]
            selected.sort(key=lambda item: (item[1]['reading'], item[0]))
            assert evidence == render_definition_evidence(namespace, selected)
            assert [int(line.split(':', 1)[0].split()[1]) for line in evidence.splitlines()
                    if line.startswith('Memory ')] == ([0, 1, 2, 3] if not orientation else [3, 2, 1, 0])
        assert target['genus'] == reversed_target['genus']
        assert 'Memory 0: position=-0.75' in forward and 'Memory 0: position=+0.75' in reverse
        assert f'Memory 0 {theme["first"]}' in target['core']
        assert f'Memory 2 {theme["second"]}' in reversed_target['core']
        assert 'At lower readings, Memory 0' in target['compose']
        assert 'at higher readings Memory 0' in reversed_target['compose']
        renamed, renamed_outputs, _ = schema._case(theme, 0, 'p_ffffffffffffffffffffffff')
        assert renamed.replace('p_ffffffffffffffffffffffff', forward.split('[Probe ')[1].split(']')[0]) == forward
        assert renamed_outputs['genus'] == target['genus'] and renamed_outputs['core'] == target['core']
        assert all('always' not in output and 'if and only if' not in output for output in target.values())
        if theme.get('kind') == 'exception':
            assert f'Memory 3 {theme["first"]}' in target['boundary']
            assert 'observed exception' in target['boundary'] and 'partial' in target['compose']
        if theme.get('kind') == 'indeterminate':
            assert f'Memory 1 and Memory 3 both {theme["neutral"]}' in target['boundary']
            assert 'indeterminate' in target['compose']
        for role in schema.SCHEMA_ORDER:
            assert target[role] != recap[role]
        aliases.add(schema._calibration_rows('genus')[theme_index * 2]['evidence'].split(']')[0])
    assert len(aliases) == 8


def test_cache_hit_avoids_native_forwards_and_returned_profiles_do_not_alias(tmp_path):
    model = wrapped()
    first, initial = schema.get_or_mint_definition_schema(model, (0, 2), tmp_path)
    model.tok.calls.clear()
    second, loaded = schema.get_or_mint_definition_schema(model, (2, 0), tmp_path)
    assert model.tok.calls == [] and loaded['cache_status'] == 'hit'
    assert initial['path'] == loaded['path']
    second['core'][0]['coefficients'].zero_()
    assert torch.count_nonzero(first['core'][0]['coefficients']) > 0


def test_assistant_prefix_changes_bank_identity_even_when_only_trailing_whitespace_changes(tmp_path):
    model = wrapped()
    _, first = schema.get_or_mint_definition_schema(model, (0,), tmp_path, assistant_prefix='Prefix ')
    _, second = schema.get_or_mint_definition_schema(model, (0,), tmp_path, assistant_prefix='Prefix  ')
    assert first['identity_hash'] != second['identity_hash']
    assert Path(first['path']).exists() and Path(second['path']).exists()


def test_empty_prefix_uses_clean_native_header_and_a_separate_reloadable_identity(tmp_path):
    model = wrapped()
    _, empty = schema.get_or_mint_definition_schema(model, (0,), tmp_path, assistant_prefix='')
    assert empty['identity']['assistant_prefix'] == '' and len(model.tok.calls) == 192
    assert all(text.endswith('\nASSISTANT:') for text, _ in model.tok.encoded)
    model.tok.calls.clear()
    _, loaded = schema.get_or_mint_definition_schema(model, (0,), tmp_path, assistant_prefix='')
    assert model.tok.calls == [] and loaded['cache_status'] == 'hit'
    assert loaded['path'] == empty['path']
    _, prefixed = schema.get_or_mint_definition_schema(model, (0,), tmp_path)
    assert prefixed['identity_hash'] != empty['identity_hash']
    assert Path(prefixed['path']).exists() and Path(empty['path']).exists()


def test_all_composed_teacher_examples_meet_compact_definition_structure():
    import re
    from invariants.definition_schema_check import assess_schema_definition
    for theme in (*schema._THEMES, *schema._TEACHERS):
        for orientation in (0, 1):
            evidence, outputs, _ = schema._case(theme, orientation)
            selected = [(int(index), {'reading': float(reading)})
                        for index, reading in re.findall(r'Memory (\d+): position=([+-][\d.]+)', evidence)]
            assessed = assess_schema_definition(outputs['compose'], selected=selected)
            assert assessed['accepted'], assessed['reason']
            assert assessed['structural_only'] and assessed['semantic_truth_checked'] is False


@pytest.mark.parametrize('corruption', ['part', 'layer', 'tensor', 'prefix', 'shift_claim'])
def test_invalid_cache_is_reminted_once_with_exact_profile_coverage(tmp_path, corruption):
    model = wrapped()
    path = tmp_path / 'fixed.pt'
    _, metadata = schema.get_or_mint_definition_schema(model, (0, 2), path)
    payload = torch.load(path, weights_only=True, map_location='cpu')
    if corruption == 'part':
        del payload['bank']['compose']
    elif corruption == 'layer':
        del payload['bank']['core'][2]
    elif corruption == 'tensor':
        payload['bank']['genus'][0]['center_change'][0] = float('nan')
    elif corruption == 'prefix':
        payload['identity']['assistant_prefix'] = 'Other prefix'
    else:
        payload['provenance']['online_intervention_source_shift'] = 'validated'
    torch.save(payload, path)
    model.tok.calls.clear()
    bank, reminted = schema.get_or_mint_definition_schema(model, (0, 2), path)
    assert len(model.tok.calls) == 192 and reminted['cache_status'] == 'minted'
    assert 'cache_rejection' in reminted and reminted['path'] == metadata['path']
    assert set(bank) == set(schema.SCHEMA_ORDER)


@pytest.mark.parametrize('failure', ['forward', 'nan'])
def test_capture_failure_removes_only_temporary_hooks_and_preserves_old_file(tmp_path, failure):
    model = wrapped()
    model.model.failure = failure
    existing = model.model.model.layers[0].register_forward_hook(lambda *_: None)
    path = tmp_path / 'keep.pt'
    path.write_bytes(b'old data')
    try:
        with pytest.raises((ValueError, RuntimeError)):
            schema.get_or_mint_definition_schema(model, (0, 1, 2), path)
        assert list(model.model.model.layers[0]._forward_hooks) == [existing.id]
        assert not model.model.model.layers[1]._forward_hooks and not model.model.model.layers[2]._forward_hooks
        assert path.read_bytes() == b'old data' and not list(tmp_path.glob('*.tmp'))
    finally:
        existing.remove()


def test_registration_failure_cleans_already_registered_capture_hooks(monkeypatch):
    model = wrapped()
    def fail(*args, **kwargs):
        raise RuntimeError('registration failed')
    monkeypatch.setattr(model.model.model.layers[1], 'register_forward_hook', fail)
    evidence = schema._calibration_rows('genus')[0]['evidence']
    with pytest.raises(RuntimeError, match='registration failed'):
        schema._decision_states(model, [dict(role='user', content=evidence)], (0, 1), 'Prefix ')
    assert not model.model.model.layers[0]._forward_hooks


@pytest.mark.parametrize('prefix', ['  ', None, 4])
def test_invalid_prefix_fails_before_capture(tmp_path, prefix):
    model = wrapped()
    with pytest.raises(ValueError, match='exact assistant prefix'):
        schema.get_or_mint_definition_schema(model, (0,), tmp_path, assistant_prefix=prefix)
    assert model.tok.calls == [] and not list(tmp_path.glob('*.pt'))


def test_real_tiny_native_block_capture_excludes_final_rms_norm_and_mints_all_parts(tmp_path):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from invariants.engine import HF, _inputs

    rows = [row for role in schema.SCHEMA_ORDER for row in schema._calibration_rows(role)]
    texts = [message['content'] for row in rows for key in ('positive', 'negative') for message in row[key]]
    special = ['[UNK]', '[PAD]', '[BOS]', '[EOS]', '<user>', '<assistant>', '<eot>']
    words = set((' '.join(texts) + ' ' + schema.DEFAULT_ASSISTANT_PREFIX).split())
    vocabulary = {word: index for index, word in enumerate(special + sorted(words - set(special)))}
    backend = BackendTokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
                                 bos_token='[BOS]', eos_token='[EOS]', name_or_path='test/native-schema')
    tok.model_input_names = ['input_ids', 'attention_mask']
    tok.chat_template = ("{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' <eot> ' }}{% endfor %}"
                         "{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}")
    config = LlamaConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         max_position_embeddings=2048, attention_dropout=0.)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    handles = []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            raw = LlamaForCausalLM(config).cpu().eval()
        raw.config._name_or_path = 'test/native-schema-llama'
        raw.config._commit_hash = 'c' * 40
        model = HF(raw, tok)
        count = []
        expected_prefix_token = tok(schema.DEFAULT_ASSISTANT_PREFIX,
                                    add_special_tokens=False)['input_ids'][-1]
        def capture_root(_module, args, kwargs):
            assert not torch.is_grad_enabled() and not kwargs['output_hidden_states']
            ids = kwargs['input_ids']
            assert int(ids[0, -1]) == expected_prefix_token
            count.append(ids.shape[-1])
        handles.append(raw.register_forward_pre_hook(capture_root, with_kwargs=True))
        bank, metadata = schema.get_or_mint_definition_schema(model, (0, 1), tmp_path)
        assert len(count) == 192 and metadata['provenance']['semantic_truth_checked'] is False
        for profiles in bank.values():
            assert set(profiles) == {0, 1} and profiles[1]['diagnostics']['adaptive']
        _, cached = schema.get_or_mint_definition_schema(model, (0, 1), tmp_path)
        assert len(count) == 192 and cached['cache_status'] == 'hit'
        handles[0].remove()
        expected = {}
        handles.append(raw.model.layers[1].register_forward_hook(lambda _m, _a, output:
            expected.update(last=(output[0] if isinstance(output, tuple) else output)[0, -1].detach().clone())))
        messages = [dict(role='user', content=rows[0]['evidence'])]
        observed = schema._decision_states(model, messages, (1,), schema.DEFAULT_ASSISTANT_PREFIX)
        torch.testing.assert_close(observed[1], expected['last'])
        clean_header = schema._decision_states(model, messages, (1,), '')
        torch.testing.assert_close(clean_header[1], expected['last'])
        assert not torch.allclose(clean_header[1], observed[1])
        with torch.no_grad():
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + schema.DEFAULT_ASSISTANT_PREFIX
            normalized = raw(**_inputs(model, prompt, pre_formatted=True), output_hidden_states=True, use_cache=False).hidden_states[-1][0, -1]
        assert not torch.allclose(observed[1], normalized)
    finally:
        for handle in handles:
            handle.remove()
        torch.set_num_threads(previous_threads)
