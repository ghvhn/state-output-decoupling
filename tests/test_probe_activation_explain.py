"""Activation explanation controls are arithmetic, separate from output truth."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from invariants.config import AgenticConfig


class NativeTokenizer:
    """A visible native header makes any added task or assistant text detectable."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), dict(kwargs)))
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return native_prompt(messages[0]['content'])

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[3 + ord(char) % 29 for char in text]])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


def native_prompt(content):
    return '<user>\n' + content + '\n</user>\n<assistant>\n'


class IdentityResidualModel(nn.Module):
    def __init__(self, count=1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(count)])

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class IdentityRootModel(nn.Module):
    def __init__(self, count=1):
        super().__init__()
        self.model = IdentityResidualModel(count)

    def forward(self, hidden):
        return self.model(hidden)


@pytest.fixture
def wrapped():
    return SimpleNamespace(
        model=IdentityRootModel(), tok=NativeTokenizer(),
        device=torch.device('cpu'), d_model=3, n_layers=1,
    )


@pytest.fixture
def module():
    from invariants import probe_activation_explain
    return probe_activation_explain


@pytest.mark.parametrize('evidence', ['', '  Memory 0: position=-0.8; reply="Wait."\n  '])
def test_zero_controls_generate_the_exact_native_blank_continuation(module, wrapped, evidence):
    config = AgenticConfig()
    before = dict(vars(config))
    calls = []

    def generate(model, **kwargs):
        assert model is wrapped
        calls.append(kwargs)
        assert kwargs['instruction'] == native_prompt(evidence)
        assert kwargs['pre_formatted'] is True
        assert kwargs['return_telemetry'] is True
        assert kwargs['max_new_tokens'] == 16
        torch.testing.assert_close(model.model(torch.ones(1, 4, 3)), torch.ones(1, 4, 3))
        return 'Unprefixed model output.', {'tokens_generated': 3}

    result = module.generate_activation_explanation(
        wrapped, {0: torch.tensor([1., 0., 0.])}, evidence,
        generate=generate, config=config, subject_gain=0., operation_gain=0., reference_gain=0.,
        max_tokens=16,
    )
    assert len(calls) == 1
    assert result['text'] == 'Unprefixed model output.'
    assert result['telemetry']['tokens_generated'] == 3
    assert vars(config) == before
    assert all(not layer._forward_hooks for layer in wrapped.model.model.layers)


def stub_axes(monkeypatch, module, wrapped):
    bank = {
        'operation': {layer: torch.tensor([0., 1., 0.]) for layer in range(wrapped.n_layers)},
        'reference': {layer: torch.tensor([0., 0., 1.]) for layer in range(wrapped.n_layers)},
    }
    mint = Mock(return_value=(bank, {'cache_status': 'test_stub'}))
    monkeypatch.setattr(module, 'get_or_mint_activation_axes', mint)
    return bank, mint


@pytest.mark.parametrize('operation,reference', [(.12, .04), (-.12, .04), (.12, -.04),
                                                (0., .04), (.12, 0.), (0., 0.)])
def test_signed_controls_are_independent_and_pulse_only_at_original_decision(
        module, wrapped, monkeypatch, operation, reference):
    bank, mint = stub_axes(monkeypatch, module, wrapped)
    bank_before = copy.deepcopy(bank)
    subject = {0: torch.tensor([5., 0., 0.])}
    observations = []

    def generate(model, **kwargs):
        hidden = torch.ones(1, 4, 3)
        decision = model.model(hidden)
        decoded = model.model(torch.ones(1, 1, 3))
        rebuilt = model.model(torch.ones(1, 6, 3))
        observations.extend((hidden, decision, decoded, rebuilt))
        return 'A freely generated hypothesis.', {'tokens_generated': 5}

    result = module.generate_activation_explanation(
        wrapped, subject, '', generate=generate, config=AgenticConfig(),
        subject_gain=.02, operation_gain=operation, reference_gain=reference, cap_fraction=1.,
    )
    hidden, decision, decoded, rebuilt = observations
    generation_mix = torch.tensor([0., operation, reference])
    decision_mix = generation_mix + torch.tensor([.02, 0., 0.])
    scale = hidden[0, -1].norm()
    torch.testing.assert_close(decision[:, :-1], hidden[:, :-1])
    torch.testing.assert_close(decision[0, -1], hidden[0, -1] + scale * decision_mix)
    torch.testing.assert_close(decoded, torch.ones_like(decoded))
    torch.testing.assert_close(rebuilt[:, :3], torch.ones(1, 3, 3))
    torch.testing.assert_close(rebuilt[0, 3], torch.ones(3) + scale * decision_mix)
    torch.testing.assert_close(rebuilt[:, 4:], torch.ones(1, 2, 3))
    torch.testing.assert_close(hidden, torch.ones_like(hidden))
    torch.testing.assert_close(subject[0], torch.tensor([5., 0., 0.]))
    for axis in bank:
        torch.testing.assert_close(bank[axis][0], bank_before[axis][0])
    if operation or reference:
        assert mint.call_count == 1
        assert mint.call_args.kwargs['include_operation'] is bool(operation)
        assert mint.call_args.kwargs['include_reference'] is bool(reference)
    else:
        mint.assert_not_called()
    assert result['metadata']['assistant_prefix'] == ''
    assert result['metadata']['prompt_task_request'] is False
    assert result['metadata']['candidate'] is True
    assert result['metadata']['semantic_truth_checked'] is False
    assert all(not layer._forward_hooks for layer in wrapped.model.model.layers)


def test_reference_minting_holds_evidence_and_explanation_phase_fixed(module, wrapped, monkeypatch):
    evidence = '  [Probe p_ffffffffffffffffffffffff]\nMemory 0: position=-0.8; reply="Wait."\n  '
    calls = []

    def states(model, messages, layers, assistant_prefix=''):
        assert model is wrapped
        assert layers == (0,)
        assert assistant_prefix == ''
        assert not torch.is_grad_enabled()
        calls.append(copy.deepcopy(messages))
        # Equalize pair magnitudes before averaging: the second positive has
        # a hundredfold larger norm and a small independent component.
        index = len(calls) - 1
        vector = torch.zeros(3)
        if index == 0:
            vector = torch.tensor([0., 0., 1.])
        elif index == 2:
            vector = 100 * torch.tensor([.3, 0., 1.])
        return {0: vector}

    monkeypatch.setattr(module, '_raw_decision_states', states)
    directions = module.mint_reference_axis(wrapped, (0,), evidence)
    assert len(calls) == 4
    for positive, negative in zip(calls[::2], calls[1::2]):
        assert [message['role'] for message in positive] == ['user']
        assert [message['role'] for message in negative] == ['user']
        positive_evidence, positive_request = positive[0]['content'].rsplit('\n\n', 1)
        negative_evidence, negative_request = negative[0]['content'].rsplit('\n\n', 1)
        assert positive_evidence == negative_evidence == evidence
        assert positive_request.replace('your own activations', 'externally observed behavior') == negative_request
        assert positive_request.startswith('In one concise sentence,')
        assert 'lower' in positive_request and 'higher' in positive_request
        assert 'secret_trait' not in positive_request + negative_request
    first = torch.tensor([0., 0., 1.])
    second = torch.tensor([.3, 0., 1.])
    expected = first + second / second.norm()
    expected /= expected.norm()
    torch.testing.assert_close(directions[0], expected)


def test_blank_reference_minting_uses_generic_contexts_only_for_calibration(module, wrapped, monkeypatch):
    calls = []

    def states(model, messages, layers, assistant_prefix=''):
        assert assistant_prefix == ''
        calls.append(copy.deepcopy(messages))
        return {0: torch.tensor([0., 0., 1.]) if len(calls) % 2 else torch.zeros(3)}

    monkeypatch.setattr(module, '_raw_decision_states', states)
    module.mint_reference_axis(wrapped, (0,), '')
    assert len(calls) == 8
    assert len({messages[0]['content'].rsplit('\n\n', 1)[0] for messages in calls}) == 2
    assert all('[Probe p_000000000000000000000000]' in messages[0]['content'] for messages in calls)
    prompt, metadata = module.build_activation_explanation_prompt(wrapped, '')
    assert prompt == native_prompt('')
    assert metadata['assistant_prefix'] == '' and metadata['prefix_token_count'] == 0
    assert 'Memory' not in prompt and 'define' not in prompt and 'explain' not in prompt


def test_axis_cache_separates_runtime_owners_sources_and_exact_evidence(module, wrapped, monkeypatch):
    wrapped.model.config = SimpleNamespace(_name_or_path='fixture/source-a')
    operation = Mock(return_value={0: torch.tensor([0., 1., 0.])})
    reference = Mock(return_value={0: torch.tensor([0., 0., 1.])})
    monkeypatch.setattr(module, 'mint_definition_operation_axis', operation)
    monkeypatch.setattr(module, 'mint_reference_axis', reference)
    cache = {}
    axes, first = module.get_or_mint_activation_axes(wrapped, (0,), '', cache)
    repeated, hit = module.get_or_mint_activation_axes(wrapped, (0,), '', cache)
    assert first['axes_cached'] == {'operation': False, 'reference': False}
    assert hit['axes_cached'] == {'operation': True, 'reference': True}
    assert hit['axis_cache_key'] == first['axis_cache_key']
    assert operation.call_count == reference.call_count == 1
    repeated['operation'][0].fill_(99.)
    again, _ = module.get_or_mint_activation_axes(wrapped, (0,), '', cache)
    torch.testing.assert_close(again['operation'][0], axes['operation'][0])
    _, changed_evidence = module.get_or_mint_activation_axes(wrapped, (0,), ' ', cache)
    assert changed_evidence['axis_cache_key'] != first['axis_cache_key']
    assert operation.call_count == reference.call_count == 2
    wrapped.model.config._name_or_path = 'fixture/source-b'
    _, changed_source = module.get_or_mint_activation_axes(wrapped, (0,), '', cache)
    assert changed_source['axis_cache_key'] != first['axis_cache_key']
    assert operation.call_count == reference.call_count == 3
    other_model = SimpleNamespace(**vars(wrapped))
    other_model.model = IdentityRootModel()
    other_model.model.config = SimpleNamespace(_name_or_path='fixture/source-b')
    _, changed_owner = module.get_or_mint_activation_axes(other_model, (0,), '', cache)
    assert changed_owner['axis_cache_key'] != changed_source['axis_cache_key']
    assert operation.call_count == reference.call_count == 4
    other_tokenizer = SimpleNamespace(**vars(wrapped))
    other_tokenizer.tok = NativeTokenizer()
    _, changed_tokenizer = module.get_or_mint_activation_axes(other_tokenizer, (0,), '', cache)
    assert changed_tokenizer['axis_cache_key'] != changed_source['axis_cache_key']
    assert operation.call_count == reference.call_count == 5


def test_axis_cache_mints_only_requested_controls_and_commits_after_complete_validation(
        module, wrapped, monkeypatch):
    operation = Mock(return_value={0: torch.tensor([0., 1., 0.])})
    reference = Mock(return_value={0: torch.tensor([0., 0., 1.])})
    monkeypatch.setattr(module, 'mint_definition_operation_axis', operation)
    monkeypatch.setattr(module, 'mint_reference_axis', reference)
    cache = {}
    axes, metadata = module.get_or_mint_activation_axes(
        wrapped, (0,), '', cache, include_operation=False, include_reference=True,
    )
    assert set(axes) == {'reference'}
    assert metadata['axes_cached'] == {'reference': False}
    operation.assert_not_called()
    reference.assert_called_once()
    axes, extended = module.get_or_mint_activation_axes(wrapped, (0,), '', cache)
    assert set(axes) == {'operation', 'reference'}
    assert extended['axes_cached'] == {'operation': False, 'reference': True}
    assert operation.call_count == reference.call_count == 1
    reference.side_effect = RuntimeError('reference mint failed')
    fresh_cache = {}
    with pytest.raises(RuntimeError, match='reference mint failed'):
        module.get_or_mint_activation_axes(wrapped, (0,), 'changed context', fresh_cache)
    assert fresh_cache == {}


def test_axis_cache_separates_subject_directions_and_revalidates_cached_geometry(
        module, wrapped, monkeypatch):
    operation = Mock(return_value={0: torch.tensor([0., 1., 0.])})
    reference = Mock(return_value={0: torch.tensor([0., 0., 1.])})
    monkeypatch.setattr(module, 'mint_definition_operation_axis', operation)
    monkeypatch.setattr(module, 'mint_reference_axis', reference)
    cache = {}
    subject = {0: torch.tensor([1., 0., 0.])}
    _, first = module.get_or_mint_activation_axes(wrapped, (0,), '', cache, subject=subject)
    _, repeated = module.get_or_mint_activation_axes(wrapped, (0,), '', cache, subject=copy.deepcopy(subject))
    assert repeated['axis_cache_key'] == first['axis_cache_key']
    assert operation.call_count == reference.call_count == 1
    _, changed = module.get_or_mint_activation_axes(
        wrapped, (0,), '', cache, subject={0: torch.tensor([-1., 0., 0.])},
    )
    assert changed['axis_cache_key'] != first['axis_cache_key']
    assert operation.call_count == reference.call_count == 2
    cache[first['axis_cache_key']]['operation'][0].fill_(float('nan'))
    with pytest.raises(ValueError, match='invalid geometry'):
        module.get_or_mint_activation_axes(wrapped, (0,), '', cache, subject=subject)
    # Invalid cached state is rejected rather than silently replaced with a
    # direction belonging to another source or a newly minted interpretation.
    assert operation.call_count == reference.call_count == 2


def tiny_native_model(module, *, calibration=False):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from invariants import engine, probe_definition_operation

    special = ['[UNK]', '[PAD]', '[BOS]', '[EOS]', '<user>', '</user>', '<assistant>']
    words = ['generated', 'alpha', 'beta', 'gamma', 'delta']
    if calibration:
        texts = [piece for case in probe_definition_operation._OPERATION_CASES for piece in case]
        texts.extend(request for pair in module._REFERENCE_REQUESTS for request in pair)
        words.extend(sorted(set(' '.join(texts).split()) - set(special + words)))
    vocabulary = {word: index for index, word in enumerate(special + words)}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
        bos_token='[BOS]', eos_token='[EOS]', name_or_path='test/native-activation-tokenizer',
    )
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<user> ' + message['content'] + ' </user> ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(27)
        raw = LlamaForCausalLM(LlamaConfig(
            vocab_size=len(vocabulary), hidden_size=16, intermediate_size=24,
            num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=512, pad_token_id=1, bos_token_id=2, eos_token_id=3,
        )).cpu().eval()
    raw.config._name_or_path = 'test/native-activation-llama'
    raw.config._commit_hash = 'd' * 40
    raw.generation_config.suppress_tokens = list(range(len(special)))
    return engine.HF(raw, tokenizer)


def test_real_tiny_llama_returns_only_generated_tokens_with_native_empty_input(module, monkeypatch):
    import ctypes
    from invariants import agentic_engine, engine

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        wrapped = tiny_native_model(module)
        raw, tokenizer = wrapped.model, wrapped.tok
        bank = {
            name: {layer: torch.eye(16)[axis] for layer in range(3)}
            for name, axis in (('operation', 1), ('reference', 2))
        }
        monkeypatch.setattr(module, 'get_or_mint_activation_axes', Mock(return_value=(bank, {'axes_cached': {}})))
        monkeypatch.setattr(agentic_engine, '_engine_memory_snapshot', lambda: (0., 0.))
        if hasattr(ctypes, 'windll'):
            # The engine's optional process-memory trim is unrelated to token
            # generation and can trigger a Windows invalid-handle diagnostic.
            monkeypatch.setattr(ctypes.windll.kernel32, 'SetProcessWorkingSetSize', lambda *_args: None)
        lookahead, cognitive_cache = Mock(), Mock()
        config = AgenticConfig()
        config.lookahead_processor, config.cognitive_cache = lookahead, cognitive_cache
        config_before = dict(vars(config))
        generation_before = copy.deepcopy(raw.generation_config.to_dict())
        parameters_before = [parameter.detach().clone() for parameter in raw.parameters()]
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': ''}], tokenize=False,
                                                add_generation_prompt=True)
        expected_ids = engine._inputs(wrapped, prompt, pre_formatted=True)['input_ids']
        actual = []
        native_generate = raw.generate

        def observe_generate(**kwargs):
            torch.testing.assert_close(kwargs['input_ids'], expected_ids)
            assert kwargs['use_cache'] is True and kwargs['do_sample'] is False
            generated = native_generate(**kwargs)
            actual.append(generated[0, expected_ids.shape[-1]:].detach().clone())
            return generated

        monkeypatch.setattr(raw, 'generate', observe_generate)
        result = module.generate_activation_explanation(
            wrapped, {layer: torch.eye(16)[0] for layer in range(3)}, '',
            generate=agentic_engine.generate_agentic_text, config=config, max_tokens=6,
        )
        assert len(actual) == 1 and actual[0].numel() == 6
        assert result['text'] == tokenizer.decode(actual[0], skip_special_tokens=True).strip()
        assert result['text'] and not result['text'].startswith('This probe measures ')
        assert result['telemetry']['tokens_generated'] == 6
        assert result['telemetry']['generation_channel'] == 'reader'
        assert all(chunk['tot_routing_count'] == 0 for chunk in result['telemetry']['chunk_traces'])
        assert result['metadata']['assistant_prefix'] == ''
        assert result['metadata']['prefill']['prefix_token_count'] == 0
        assert result['metadata']['candidate'] and not result['metadata']['semantic_truth_checked']
        assert vars(config) == config_before
        assert raw.generation_config.to_dict() == generation_before
        assert all(not layer._forward_hooks for layer in raw.model.layers)
        assert lookahead.mock_calls == cognitive_cache.mock_calls == []
        for parameter, before in zip(raw.parameters(), parameters_before):
            torch.testing.assert_close(parameter, before, rtol=0., atol=0.)
    finally:
        torch.set_num_threads(old_threads)


def test_real_tiny_llama_mints_both_blank_context_controls_at_native_decision(module):
    from invariants import engine

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    observer = None
    try:
        wrapped = tiny_native_model(module, calibration=True)
        forwards = []
        weights_before = [parameter.detach().clone() for parameter in wrapped.model.parameters()]

        def observe(_model, args, kwargs):
            forwards.append(kwargs['input_ids'].detach().clone())
            assert kwargs['output_hidden_states'] is False
            assert kwargs['use_cache'] is False
            assert not torch.is_grad_enabled()

        observer = wrapped.model.register_forward_pre_hook(observe, with_kwargs=True)
        cache = {}
        bank, metadata = module.get_or_mint_activation_axes(wrapped, (0, 1, 2), '', cache)
        assert set(bank) == {'operation', 'reference'}
        assert metadata['axes_cached'] == {'operation': False, 'reference': False}
        # Operation has two matched contexts, reference has two paraphrases
        # for each of those contexts; every pair gets one forward per side.
        assert len(forwards) == 12
        header_id = wrapped.tok.convert_tokens_to_ids('<assistant>')
        assert all(int(ids[0, -1]) == header_id for ids in forwards)
        assert all(ids.shape[-1] > 3 for ids in forwards)
        for directions in bank.values():
            assert set(directions) == {0, 1, 2}
            for vector in directions.values():
                assert vector.shape == (16,) and vector.device.type == 'cpu'
                assert torch.isfinite(vector).all()
                torch.testing.assert_close(vector.norm(), torch.tensor(1.), atol=1e-5, rtol=1e-5)
        reloaded, cached = module.get_or_mint_activation_axes(wrapped, (0, 1, 2), '', cache)
        assert cached['axis_cache_hit'] is True and len(forwards) == 12
        for name in bank:
            for layer in bank[name]:
                torch.testing.assert_close(reloaded[name][layer], bank[name][layer])
        prompt, _ = module.build_activation_explanation_prompt(wrapped)
        header_only = torch.tensor([[wrapped.tok.convert_tokens_to_ids(token)
                                     for token in ('<user>', '</user>', '<assistant>')]])
        torch.testing.assert_close(engine._inputs(wrapped, prompt, pre_formatted=True)['input_ids'], header_only)
        for parameter, before in zip(wrapped.model.parameters(), weights_before):
            torch.testing.assert_close(parameter, before, rtol=0., atol=0.)
    finally:
        if observer is not None:
            observer.remove()
        torch.set_num_threads(old_threads)


def test_raw_calibration_endpoint_matches_every_decoder_block_including_final_layer(module):
    from invariants import engine

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    handles = []
    try:
        wrapped = tiny_native_model(module, calibration=True)
        messages = [{'role': 'user', 'content': 'alpha beta gamma'}]
        prompt = wrapped.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = engine._inputs(wrapped, prompt, pre_formatted=True)
        raw, normalized = {}, []
        for index, block in enumerate(wrapped.model.model.layers):
            def capture(_module, _args, output, index=index):
                hidden = output[0] if isinstance(output, tuple) else output
                raw[index] = hidden[0, -1].detach().float().clone()
            handles.append(block.register_forward_hook(capture))
        handles.append(wrapped.model.model.norm.register_forward_hook(
            lambda _module, _args, output: normalized.append(output[0, -1].detach().float().clone())))
        with torch.no_grad():
            wrapped.model(**inputs, use_cache=False, output_hidden_states=False)
        expected = {index: vector.clone() for index, vector in raw.items()}
        assert not torch.allclose(expected[2], normalized[0])
        states = module._raw_decision_states(wrapped, messages, (0, 1, 2), assistant_prefix='')
        assert set(states) == {0, 1, 2}
        for index in states:
            torch.testing.assert_close(states[index], expected[index], rtol=0., atol=0.)
        # The capture helper must remove its own hooks while preserving ours.
        assert all(len(block._forward_hooks) == 1 for block in wrapped.model.model.layers)
    finally:
        for handle in handles:
            handle.remove()
        torch.set_num_threads(old_threads)


def test_raw_mint_failure_restores_existing_hooks_and_decoder_forward_methods(module):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    existing = []
    try:
        wrapped = tiny_native_model(module, calibration=True)
        blocks = wrapped.model.model.layers
        existing.append(blocks[0].register_forward_hook(lambda *_args: None))

        def fail(_module, _args):
            raise RuntimeError('deliberate calibration failure')

        existing.append(blocks[1].register_forward_pre_hook(fail))
        hook_before = [dict(block._forward_hooks) for block in blocks]
        pre_hook_before = [dict(block._forward_pre_hooks) for block in blocks]
        functions_before = [block.forward.__func__ for block in blocks]
        instance_forwards_before = [vars(block).get('forward') for block in blocks]
        cache = {}
        with pytest.raises(RuntimeError, match='deliberate calibration failure'):
            module.get_or_mint_activation_axes(wrapped, (0, 1, 2), '', cache)
        assert cache == {}
        assert [dict(block._forward_hooks) for block in blocks] == hook_before
        assert [dict(block._forward_pre_hooks) for block in blocks] == pre_hook_before
        assert all(block.forward.__func__ is function for block, function in zip(blocks, functions_before))
        assert all(vars(block).get('forward') is previous
                   for block, previous in zip(blocks, instance_forwards_before))
    finally:
        for handle in existing:
            handle.remove()
        torch.set_num_threads(old_threads)


def test_subject_and_signed_tasks_share_one_residual_cap(module, wrapped, monkeypatch):
    stub_axes(monkeypatch, module, wrapped)
    outputs = []

    def generate(model, **kwargs):
        outputs.append(model.model(torch.ones(1, 4, 3)))
        return 'Candidate.', {'tokens_generated': 1}

    module.generate_activation_explanation(
        wrapped, {0: torch.tensor([1., 0., 0.])}, '',
        generate=generate, config=AgenticConfig(), subject_gain=.4, operation_gain=-.6,
        reference_gain=.8, cap_fraction=.05,
    )
    mix = torch.tensor([.4, -.6, .8])
    expected = .05 * torch.ones(3).norm() * mix / mix.norm()
    torch.testing.assert_close(outputs[0][0, -1] - torch.ones(3), expected)
    torch.testing.assert_close(outputs[0][:, :-1], torch.ones(1, 3, 3))


def test_default_cap_resolves_live_bound_and_records_the_applied_value(module, wrapped, monkeypatch):
    from invariants import engine

    monkeypatch.setattr(engine, '_steer_cap_fraction', .015)
    outputs = []

    def generate(model, **kwargs):
        outputs.append(model.model(torch.ones(1, 4, 3)))
        assert engine.get_steer_cap_fraction() == .015
        return 'Candidate.', {'tokens_generated': 1}

    result = module.generate_activation_explanation(
        wrapped, {0: torch.tensor([1., 0., 0.])}, '',
        generate=generate, config=AgenticConfig(), subject_gain=.02,
        operation_gain=0., reference_gain=0., cap_fraction=None,
    )
    expected = torch.tensor([.015 * torch.ones(3).norm(), 0., 0.])
    torch.testing.assert_close(outputs[0][0, -1] - torch.ones(3), expected)
    torch.testing.assert_close(outputs[0][:, :-1], torch.ones(1, 3, 3))
    assert result['metadata']['cap_fraction'] == .015
    assert result['metadata']['cap_telemetry']['applied'] == 1
    assert result['metadata']['cap_telemetry']['clipped'] == 1
    assert engine.get_steer_cap_fraction() == .015


@pytest.mark.parametrize('failure', [False, True])
def test_diagnostic_cap_telemetry_is_local_and_restores_original_deque_and_counters(
        module, wrapped, monkeypatch, failure):
    from collections import deque
    from invariants import engine

    conversation_attempts = deque([.13, .07, .21], maxlen=9)
    conversation_contents = list(conversation_attempts)
    monkeypatch.setattr(engine, '_steer_attempts', conversation_attempts)
    monkeypatch.setattr(engine, '_steer_apply_count', 37)
    monkeypatch.setattr(engine, '_steer_clip_count', 11)
    local = []

    def generate(model, **kwargs):
        assert engine._steer_attempts is not conversation_attempts
        assert engine._steer_attempts.maxlen == conversation_attempts.maxlen
        assert list(engine._steer_attempts) == []
        assert engine._steer_apply_count == engine._steer_clip_count == 0
        model.model(torch.ones(1, 4, 3))
        local.append(engine.steer_telemetry_stats())
        assert list(conversation_attempts) == conversation_contents
        assert local[-1]['n'] == local[-1]['applied'] == local[-1]['clipped'] == 1
        if failure:
            raise RuntimeError('failure after a clipped diagnostic pulse')
        return 'Candidate.', {'tokens_generated': 1}

    if failure:
        with pytest.raises(RuntimeError, match='failure after a clipped diagnostic pulse'):
            module.generate_activation_explanation(
                wrapped, {0: torch.tensor([1., 0., 0.])}, '', generate=generate,
                config=AgenticConfig(), subject_gain=.02, operation_gain=0., reference_gain=0.,
                cap_fraction=.01,
            )
    else:
        result = module.generate_activation_explanation(
            wrapped, {0: torch.tensor([1., 0., 0.])}, '', generate=generate,
            config=AgenticConfig(), subject_gain=.02, operation_gain=0., reference_gain=0.,
            cap_fraction=.01,
        )
        assert result['metadata']['cap_telemetry'] == local[0]
    assert local and engine._steer_attempts is conversation_attempts
    assert list(conversation_attempts) == conversation_contents
    assert engine._steer_apply_count == 37 and engine._steer_clip_count == 11
    assert not wrapped.model.model.layers[0]._forward_hooks


def test_mint_failure_through_runtime_restores_config_and_cap_calibration(module, wrapped, monkeypatch):
    from collections import deque
    from invariants import engine

    conversation_attempts = deque([.11, .22], maxlen=7)
    monkeypatch.setattr(engine, '_steer_attempts', conversation_attempts)
    monkeypatch.setattr(engine, '_steer_apply_count', 42)
    monkeypatch.setattr(engine, '_steer_clip_count', 6)
    config = AgenticConfig()
    config.lookahead_processor = Mock()
    config._generation_deadline = 123.
    before = dict(vars(config))
    generate = Mock()
    preserved = wrapped.model.model.layers[0].register_forward_hook(lambda *_args: None)
    hooks_before = dict(wrapped.model.model.layers[0]._forward_hooks)

    def failing_mint(*args, **kwargs):
        assert config.base_model_only is True and config.lookahead_processor is None
        assert engine._steer_attempts is not conversation_attempts
        # Existing instrumentation can call the cap during teacher forwards.
        # Its diagnostic observation also belongs to the scratch invocation.
        engine._cap_steer(torch.ones(1, 1, 3), torch.ones(1, 1, 3), cap_fraction=.01)
        assert engine._steer_apply_count == engine._steer_clip_count == 1
        config._generation_deadline = 999.
        config.only_present_during_calibration = object()
        raise RuntimeError('deliberate mint failure after cap observation')

    monkeypatch.setattr(module, 'get_or_mint_activation_axes', failing_mint)
    try:
        with pytest.raises(RuntimeError, match='deliberate mint failure after cap observation'):
            module.generate_activation_explanation(
                wrapped, {0: torch.tensor([1., 0., 0.])}, '',
                generate=generate, config=config,
            )
        generate.assert_not_called()
        assert vars(config) == before
        assert engine._steer_attempts is conversation_attempts
        assert list(conversation_attempts) == [.11, .22]
        assert engine._steer_apply_count == 42 and engine._steer_clip_count == 6
        assert dict(wrapped.model.model.layers[0]._forward_hooks) == hooks_before
    finally:
        preserved.remove()


@pytest.mark.parametrize('failure', [False, True])
def test_scratch_runtime_disables_other_controls_and_restores_config_and_hooks(
        module, wrapped, failure):
    config = AgenticConfig(
        routing_depth=5, max_routing_events=7, max_loops=4, synthesis_enabled=True,
        cache_enabled=True, cache_write_enabled=True, max_synthesis_events=3,
        max_synthesis_steps=17, use_expert_vectors=True, interactive_disambiguation=True,
    )
    config.lookahead_processor = Mock()
    config.committee = Mock()
    config.cognitive_cache = Mock()
    config._generation_deadline = 123.
    config._generation_budget_sec = 45.
    config._synthesis_events_used = 2
    before = dict(vars(config))
    existing = wrapped.model.model.layers[0].register_forward_hook(lambda *_args: None)
    hook_before = dict(wrapped.model.model.layers[0]._forward_hooks)
    calls = []

    def generate(model, **kwargs):
        scratch = kwargs['config']
        calls.append(scratch)
        assert scratch.base_model_only is True
        assert scratch.cache_enabled is False and scratch.cache_write_enabled is False
        assert scratch.max_routing_events == scratch.max_loops == scratch.routing_depth == 0
        assert scratch.max_synthesis_events == scratch.max_synthesis_steps == 0
        assert scratch.synthesis_enabled is False
        assert scratch.lookahead_processor is None
        assert kwargs['generation_channel'] == 'reader'
        assert kwargs['max_tool_calls'] == 0
        assert kwargs.get('mid_chunk_hook') is None
        assert kwargs['vecs'] == {}
        # Normal generation changes these transient attributes. Cleanup must
        # also work when generation raises after changing them.
        scratch._generation_deadline = 999.
        scratch._generation_budget_sec = 1.
        scratch._synthesis_events_used = 999
        model.model(torch.ones(1, 4, 3))
        if failure:
            raise RuntimeError('deliberate generation failure')
        return '<<PROBE: opaque || freely generated text>>', {'tokens_generated': 4}

    try:
        if failure:
            with pytest.raises(RuntimeError, match='deliberate generation failure'):
                module.generate_activation_explanation(
                    wrapped, {0: torch.tensor([1., 0., 0.])}, '', generate=generate,
                    config=config, operation_gain=0., reference_gain=0.,
                )
        else:
            result = module.generate_activation_explanation(
                wrapped, {0: torch.tensor([1., 0., 0.])}, '', generate=generate,
                config=config, operation_gain=0., reference_gain=0.,
            )
            assert result['text'] == '<<PROBE: opaque || freely generated text>>'
        assert len(calls) == 1
        assert vars(config) == before
        assert dict(wrapped.model.model.layers[0]._forward_hooks) == hook_before
        assert config.lookahead_processor.mock_calls == []
        assert config.cognitive_cache.mock_calls == []
        assert config.committee.mock_calls == []
    finally:
        existing.remove()


@pytest.mark.parametrize('name,value', [
    ('subject_gain', -.01), ('subject_gain', 1.01), ('subject_gain', float('nan')),
    ('operation_gain', 1.01), ('operation_gain', float('inf')),
    ('reference_gain', -1.01), ('reference_gain', float('nan')),
])
def test_invalid_gains_fail_before_minting_registration_or_generation(
        module, wrapped, monkeypatch, name, value):
    _bank, mint = stub_axes(monkeypatch, module, wrapped)
    generate = Mock()
    config = AgenticConfig()
    before = dict(vars(config))
    with pytest.raises(ValueError):
        module.generate_activation_explanation(
            wrapped, {0: torch.tensor([1., 0., 0.])}, '',
            generate=generate, config=config, **{name: value},
        )
    mint.assert_not_called()
    generate.assert_not_called()
    assert vars(config) == before
    assert all(not layer._forward_hooks for layer in wrapped.model.model.layers)
