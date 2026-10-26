"""Exercise stored step vectors and real native chunk generation on CPU.

The random tiny model tests integration and phase positions, not semantic skill.
No pretrained weights, downloaded tokenizers, or external assets are required.
"""
import copy

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from invariants import engine, probe_definition_steps
from invariants.probe_definition_process import (
    DefinitionStepProcess, build_definition_process_handles,
)


def _native_model(evidence):
    text = '\n'.join((probe_definition_steps._ROUTES, probe_definition_steps._ASSEMBLY,
                      evidence, *(request for pair in probe_definition_steps._REQUESTS.values() for request in pair)))
    special = ['[UNK]', '[PAD]', '[BOS]', '[EOS]', '<user>', '</user>', '<assistant>']
    vocabulary = {word: index for index, word in enumerate(special + sorted(set(text.split()) - set(special)))}
    backend = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
        bos_token='[BOS]', eos_token='[EOS]', name_or_path='test/native-definition-tokenizer')
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<user> ' + message['content'] + ' </user> ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"
    )
    config = LlamaConfig(vocab_size=len(vocabulary), hidden_size=16, intermediate_size=32,
                         num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=2048, attention_dropout=0.,
                         bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(7)
        model = LlamaForCausalLM(config).cpu().eval()
    model.config._name_or_path = 'test/native-definition-llama'
    model.config._commit_hash = 'c' * 40
    # The production engine explicitly passes its EOS IDs. Suppressing this
    # tokenizer's EOS guarantees all eight test positions are reached.
    model.generation_config.suppress_tokens = [tokenizer.eos_token_id]
    return engine.HF(model, tokenizer)


def test_stored_steps_drive_native_cpu_chunk_prefills_with_evidence_only_input(tmp_path, monkeypatch):
    evidence = (
        '[Probe p_ffffffffffffffffffffffff]\n'
        'Memory 0: position=-0.8; input="Which route?"; reply="Take the first corridor."\n'
        'Memory 1: position=-0.3; input="Which route?"; reply="Use the first available route."\n'
        'Memory 2: position=+0.4; input="Which route?"; reply="Compare the routes before choosing."\n'
        'Memory 3: position=+0.9; input="Which route?"; reply="Check the alternatives before choosing."'
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    all_handles = []
    try:
        wrapped = _native_model(evidence)
        bank, metadata = probe_definition_steps.get_or_mint_definition_steps(wrapped, (0, 1, 2, 3), tmp_path / 'banks')
        for directions in bank.values():
            for vector in directions.values():
                assert vector.shape == (16,) and vector.device.type == 'cpu' and torch.isfinite(vector).all()
                torch.testing.assert_close(vector.norm(), torch.tensor(1.))
        def forbidden_mint(*args, **kwargs):
            raise AssertionError('a persisted bank must reload without mint forwards')
        monkeypatch.setattr(probe_definition_steps, '_native_states', forbidden_mint)
        reloaded, loaded = probe_definition_steps.get_or_mint_definition_steps(wrapped, (0, 1, 2, 3), tmp_path / 'banks')
        assert loaded['path'] == metadata['path'] and loaded['cache_status'] == 'hit'
        for step in bank:
            for layer in bank[step]:
                torch.testing.assert_close(reloaded[step][layer], bank[step][layer])

        original_config = copy.deepcopy(wrapped.model.generation_config.to_dict())
        expected = engine._inputs(wrapped, evidence)['input_ids']
        prefix_length = expected.shape[-1]
        forwards, generations, changes = [], [], []
        before = {}

        def observe_root(_module, args, kwargs):
            input_ids = kwargs.get('input_ids', args[0] if args else None)
            forwards.append(input_ids.detach().clone())
        all_handles.append(wrapped.model.register_forward_pre_hook(observe_root, with_kwargs=True))
        for layer in range(4):
            def capture_before(_module, _args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                before[layer] = hidden.detach().clone()
            all_handles.append(wrapped.model.model.layers[layer].register_forward_hook(capture_before))

        process = DefinitionStepProcess(step_tokens=2)
        subject = {layer: reloaded['compare'][layer].clone() for layer in range(4)}
        subject_before = {layer: vector.clone() for layer, vector in subject.items()}
        all_handles.extend(build_definition_process_handles(
            wrapped, subject, reloaded, process=process, cap_fraction=.08))
        for layer in range(4):
            def capture_after(_module, _args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                changes.append((layer, hidden.shape[-2], (hidden - before[layer]).detach().clone()))
            all_handles.append(wrapped.model.model.layers[layer].register_forward_hook(capture_after))

        native_generate = wrapped.model.generate
        def observe_generate(*args, **kwargs):
            # Every chunk starts from the same evidence-only native prefix.
            torch.testing.assert_close(kwargs['input_ids'][:, :prefix_length], expected)
            result = native_generate(*args, **kwargs)
            generations.append(result.shape[-1] - kwargs['input_ids'].shape[-1])
            return result
        monkeypatch.setattr(wrapped.model, 'generate', observe_generate)
        monkeypatch.setenv('TDA_GENERATION_CHUNK_TOKENS', '2')
        result = engine.generate_text(wrapped, evidence, max_new_tokens=8)
        assert isinstance(result, str) and generations == [2, 2, 2, 2]
        assert [tensor.shape[-1] for tensor in forwards] == [
            prefix_length, 1, prefix_length + 2, 1, prefix_length + 4, 1, prefix_length + 6, 1]
        for tensor in forwards:
            if tensor.shape[-1] > 1:
                torch.testing.assert_close(tensor[:, :prefix_length], expected)
        for _layer, sequence_length, difference in changes:
            if sequence_length > 1:
                assert torch.count_nonzero(difference[:, :prefix_length - 1]) == 0
                assert torch.count_nonzero(difference[:, prefix_length - 1:]) > 0
            else:
                assert torch.count_nonzero(difference) > 0
        trace = process.trace()
        assert trace['diagnostic_only'] is True and trace['decision_position'] == prefix_length - 1
        assert trace['forward_count'] == 8 and trace['furthest_output_position'] == 7
        assert [window['step'] for window in trace['windows']] == list(probe_definition_steps.STEP_ORDER)
        assert all(window['entered'] and window['injected_positions'] == 2 for window in trace['windows'])
        assert wrapped.model.generation_config.to_dict() == original_config
        for layer in subject:
            torch.testing.assert_close(subject[layer], subject_before[layer])
    finally:
        for handle in reversed(all_handles):
            handle.remove()
        torch.set_num_threads(previous_threads)
    assert not wrapped.model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in wrapped.model.model.layers)
