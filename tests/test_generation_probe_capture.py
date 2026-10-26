"""Live probe readings describe prediction states, without replaying the answer.

The native CPU model is random and tiny. These tests establish observation and
generation boundaries, not the semantic validity of any named direction.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from invariants import agentic_engine, engine
from invariants.config import AgenticConfig


@pytest.fixture
def native():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    special = ['[UNK]', '[PAD]', '[BOS]', '[EOS]', '<user>', '</user>', '<assistant>']
    words = special + ['Explain', 'the', 'route', 'first', 'second', 'answer', 'now']
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), unk_token='[UNK]'))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
        bos_token='[BOS]', eos_token='[EOS]', name_or_path='test/live-probe-tokenizer')
    tokenizer.model_input_names = ['input_ids', 'attention_mask']
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<user> ' + message['content'] + ' </user> ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}")
    config = LlamaConfig(
        vocab_size=len(words), hidden_size=16, intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, attention_dropout=0.,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(17)
        model = LlamaForCausalLM(config).cpu().eval()
    # An anisotropic final norm makes pre-norm versus post-norm cosine readings
    # observably different. Default all-one RMSNorm weights would hide the bug.
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(.2, 2.4, 16))
    model.generation_config.suppress_tokens = list(range(len(special)))
    try:
        yield engine.HF(model, tokenizer)
    finally:
        torch.set_num_threads(previous_threads)


def _cosine(hidden, direction):
    return torch.nn.functional.cosine_similarity(hidden.float(), direction.float(), dim=-1)


@pytest.mark.parametrize('token_count', [1, 4, 26])
def test_native_readings_match_actual_prefill_decode_and_rebuilt_chunk_states(native, token_count):
    wrapped = native
    direction = torch.linspace(-1., 1., wrapped.d_model)
    probes = {
        'early': {'direction': {0: direction}, 'token_mode': 'last'},
        'final': {'direction': {2: direction}, 'token_mode': 'last'},
        'mean': {'direction': {0: direction, 2: direction}, 'token_mode': 'mean'},
    }
    probes_before = copy.deepcopy(probes)
    config = AgenticConfig(base_model_only=True, repetition_penalty=1.)
    generation_before = copy.deepcopy(wrapped.model.generation_config.to_dict())
    handles, sequence_lengths, states, pre_norm = [], [], {0: [], 2: []}, []

    def steer(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        edited = hidden.clone()
        edited[..., 0] += .03
        return (edited,) + output[1:] if isinstance(output, tuple) else edited

    # The capture must observe an already installed intervention, including the
    # prompt-tail decision which produces the first output token.
    handles.append(wrapped.model.model.layers[0].register_forward_hook(steer))

    def root(_module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        sequence_lengths.append(ids.shape[-1])

    def observe(layer):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            states[layer].append(hidden[0, -1].detach().clone())
        return hook

    handles.append(wrapped.model.register_forward_pre_hook(root, with_kwargs=True))
    handles.append(wrapped.model.model.layers[0].register_forward_hook(observe(0)))
    handles.append(wrapped.model.model.norm.register_forward_hook(observe(2)))
    handles.append(wrapped.model.model.layers[2].register_forward_hook(
        lambda _module, _args, out: pre_norm.append(
            (out[0] if isinstance(out, tuple) else out)[0, -1].detach().clone())))
    try:
        baseline, baseline_telemetry = agentic_engine.generate_agentic_text(
            wrapped, instruction='Explain the route now', config=config,
            max_new_tokens=token_count, return_telemetry=True)
        baseline_lengths = list(sequence_lengths)
        sequence_lengths.clear()
        pre_norm.clear()
        for values in states.values():
            values.clear()
        output, telemetry = agentic_engine.generate_agentic_text(
            wrapped, instruction='Explain the route now', config=config,
            max_new_tokens=token_count, return_telemetry=True, generation_probes=probes)

        assert output == baseline
        assert telemetry['tokens_generated'] == baseline_telemetry['tokens_generated'] == token_count
        assert sequence_lengths == baseline_lengths  # No answer re-encoding/forward.
        assert sequence_lengths[0] > 1
        assert sequence_lengths[1:24] == [1] * (min(token_count, 24) - 1)
        if token_count > 24:
            assert sequence_lengths[24] == sequence_lengths[0] + 24
            assert sequence_lengths[25:] == [1]

        readings = telemetry['generation_probe_readings']
        for name, layer in [('early', 0), ('final', 2)]:
            expected = _cosine(torch.stack(states[layer]), direction)
            assert readings[name]['token_count'] == token_count
            torch.testing.assert_close(torch.tensor(readings[name]['by_token']), expected)
            assert readings[name]['raw_score'] == pytest.approx(expected[-1].item())
            assert readings[name]['last_token_score'] == pytest.approx(expected[-1].item())
        expected_tokens = torch.stack([
            _cosine(torch.stack(states[layer]), direction) for layer in (0, 2)]).mean(0)
        expected_mean = torch.stack([
            _cosine(torch.stack(states[layer]).mean(0), direction) for layer in (0, 2)]).mean()
        torch.testing.assert_close(torch.tensor(readings['mean']['by_token']), expected_tokens)
        assert readings['mean']['raw_score'] == pytest.approx(expected_mean.item(), abs=1e-7)
        # This separately calibrated endpoint must not be a raw last block read.
        assert not torch.allclose(_cosine(torch.stack(pre_norm), direction),
                                  _cosine(torch.stack(states[2]), direction))
        for name in probes:
            assert probes[name].keys() == probes_before[name].keys()
            for layer in probes[name]['direction']:
                torch.testing.assert_close(probes[name]['direction'][layer],
                                           probes_before[name]['direction'][layer])
        assert wrapped.model.generation_config.to_dict() == generation_before
        assert len(wrapped.model._forward_pre_hooks) == 1
        assert len(wrapped.model.model.layers[0]._forward_hooks) == 2
        assert len(wrapped.model.model.layers[2]._forward_hooks) == 1
        assert len(wrapped.model.model.norm._forward_hooks) == 1
    finally:
        for handle in reversed(handles):
            handle.remove()


def test_generation_failure_removes_capture_hooks_and_restores_call_config(native, monkeypatch):
    wrapped = native
    config = AgenticConfig(base_model_only=True, repetition_penalty=1., chatty_log=False)
    config._generation_deadline = config._generation_budget_sec = None
    before = copy.deepcopy(vars(config))

    def broken_generate(**kwargs):
        # Fail after a real forward, with a pending uncommitted sample.
        wrapped.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask'])
        raise RuntimeError('native generation failed after prefill')

    monkeypatch.setattr(wrapped.model, 'generate', broken_generate)
    with pytest.raises(RuntimeError, match='native generation failed after prefill'):
        agentic_engine.generate_agentic_text(
            wrapped, instruction='Explain the route now', config=config,
            max_new_tokens=2, chatty_log=True, max_time=60, return_telemetry=True,
            generation_probes={'early': {0: torch.ones(wrapped.d_model)}})
    assert vars(config) == before
    assert not wrapped.model._forward_pre_hooks and not wrapped.model._forward_hooks
    assert all(not layer._forward_hooks for layer in wrapped.model.model.layers)
    assert not wrapped.model.model.norm._forward_hooks


class _ResidualStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        self.norm = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.norm.weight.copy_(torch.diag(torch.tensor([1., 2., 3.])))

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class _Root(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _ResidualStack()

    def forward(self, hidden):
        return self.model(hidden)


def test_collector_excludes_inner_replays_and_nested_forwards_and_rolls_back_chunks():
    from invariants.generation_probe_capture import GenerationProbeCapture

    wrapped = SimpleNamespace(model=_Root(), n_layers=3, d_model=3, device=torch.device('cpu'))
    state = {'replay_active': False}
    inside = False

    def inner_work(_module, _args, output):
        nonlocal inside
        if inside:
            return
        inside = True
        try:
            # A recursive full-model diagnostic forward and layer-only replay
            # must contribute neither a token nor the final endpoint state.
            wrapped.model(torch.tensor([[[0., 0., 50.]]]))
            state['replay_active'] = True
            for layer in wrapped.model.model.layers[1:]:
                layer(torch.tensor([[[0., 50., 0.]]]))
            wrapped.model.model.norm(torch.tensor([[[0., 50., 0.]]]))
        finally:
            state['replay_active'] = False
            inside = False

    unrelated = wrapped.model.model.layers[0].register_forward_hook(inner_work)
    direction = torch.tensor([1., 0., 0.])
    capture = GenerationProbeCapture(
        wrapped, {'probe': {'direction': {0: direction, 2: direction}}},
        replay_state=state, capture_states=True).attach()
    first = torch.tensor([[[1., 2., 0.], [2., 1., 0.]]])
    second = torch.tensor([[[0., 1., 0.]]])
    try:
        capture.start_chunk()
        actual = wrapped.model(first)
        torch.testing.assert_close(actual, wrapped.model.model.norm(first))
        capture.commit_prediction(torch.zeros(1, 4, dtype=torch.long))
        capture.commit_prediction(torch.zeros(1, 4, dtype=torch.long))
        capture.commit_chunk(1)
        initial = capture.summary()['probe']
        expected_first = torch.stack([
            _cosine(first[0, -1], direction),
            _cosine(actual[0, -1], direction)]).mean().item()
        assert initial['token_count'] == 1
        assert initial['by_token'] == pytest.approx([expected_first])

        capture.start_chunk()
        wrapped.model(second)
        capture.commit_prediction(torch.zeros(1, 5, dtype=torch.long))
        wrapped.model(torch.tensor([[[8., 0., 0.]]]))
        capture.commit_prediction(torch.zeros(1, 6, dtype=torch.long))
        capture.commit_chunk(1)  # Only the first prediction actually emitted.
        accepted = capture.summary()['probe']
        assert accepted['token_count'] == 2
        assert accepted['by_token'] == pytest.approx([expected_first, 0.])

        capture.start_chunk()
        wrapped.model(torch.tensor([[[9., 0., 0.]]]))
        capture.commit_prediction(torch.zeros(1, 7, dtype=torch.long))
        capture.abort_chunk()
        assert capture.summary() == {'probe': accepted}
        last_states = capture.last_hidden_states()
        assert last_states.shape == (3, 1, 3) and last_states.device.type == 'cpu'
        torch.testing.assert_close(last_states[0, 0], second[0, -1])
        torch.testing.assert_close(last_states[2, 0], torch.tensor([0., 2., 0.]))
    finally:
        capture.close()
        unrelated.remove()
    assert not wrapped.model._forward_pre_hooks and not wrapped.model._forward_hooks
    assert all(not layer._forward_hooks for layer in wrapped.model.model.layers)
    assert not wrapped.model.model.norm._forward_hooks


def test_state_sink_runs_after_capture_cleanup_and_does_not_put_tensors_in_telemetry(native):
    snapshots = []

    def receive(states):
        assert not native.model._forward_pre_hooks and not native.model._forward_hooks
        assert all(not layer._forward_hooks for layer in native.model.model.layers)
        assert not native.model.model.norm._forward_hooks
        snapshots.append(states)

    _, telemetry = agentic_engine.generate_agentic_text(
        native, instruction='Explain the route now', config=AgenticConfig(base_model_only=True),
        max_new_tokens=3, return_telemetry=True, generation_state_sink=receive)
    assert len(snapshots) == 1
    assert snapshots[0].shape == (native.n_layers, 1, native.d_model)
    assert snapshots[0].device.type == 'cpu' and not snapshots[0].requires_grad

    def contains_tensor(value):
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, dict):
            return any(contains_tensor(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(telemetry)


def test_next_chunk_observes_mutating_hooks_added_between_chunks(native, monkeypatch):
    factory = agentic_engine.get_agentic_handles
    monkeypatch.setattr(agentic_engine, 'get_agentic_handles',
                        lambda model, **_kwargs: factory(model, config=AgenticConfig(base_model_only=True)))
    direction = torch.eye(native.d_model)[0]
    changed_states = []

    class MidChunk:
        handle = None
        calls = 0

        def __call__(self, _text, _state):
            self.calls += 1
            if self.handle is not None:
                return

            def steer(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                edited = hidden.clone()
                edited[..., 0] += 1.
                changed_states.append(edited[0, -1].detach().clone())
                return (edited,) + output[1:] if isinstance(output, tuple) else edited

            self.handle = native.model.model.layers[0].register_forward_hook(steer)

        def cleanup(self):
            if self.handle is not None:
                self.handle.remove()

    mid = MidChunk()
    _, telemetry = agentic_engine.generate_agentic_text(
        native, instruction='Explain the route now', config=AgenticConfig(repetition_penalty=1.),
        max_new_tokens=26, return_telemetry=True, mid_chunk_hook=mid,
        generation_probes={'early': {0: direction}})
    assert mid.calls >= 1 and len(changed_states) == 2
    expected = _cosine(torch.stack(changed_states), direction)
    reading = telemetry['generation_probe_readings']['early']
    assert reading['token_count'] == 26
    torch.testing.assert_close(torch.tensor(reading['by_token'][-2:]), expected)
    assert not native.model._forward_pre_hooks and not native.model._forward_hooks
    assert all(not layer._forward_hooks for layer in native.model.model.layers)


def test_failed_chunk_completion_discards_prediction_readings_with_partial_output(native, monkeypatch):
    factory = agentic_engine.get_agentic_handles
    monkeypatch.setattr(agentic_engine, 'get_agentic_handles',
                        lambda model, **_kwargs: factory(model, config=AgenticConfig(base_model_only=True)))

    class Lookahead:
        finished = False

        def __call__(self, input_ids, scores):
            return scores

        def finish_chunk(self, output):
            raise MemoryError('chunk postprocessing failed')

        def finish_generation(self):
            self.finished = True

    lookahead = Lookahead()
    config = AgenticConfig(repetition_penalty=1.)
    config.lookahead_processor = lookahead
    output, telemetry = agentic_engine.generate_agentic_text(
        native, instruction='Explain the route now', config=config,
        max_new_tokens=3, return_telemetry=True,
        generation_probes={'early': {0: torch.ones(native.d_model)}})
    assert output == '' and telemetry['tokens_generated'] == 0
    assert telemetry['generation_probe_readings'] == {}
    assert telemetry['generation_probe_capture']['token_count'] == 0
    assert lookahead.finished
    assert not native.model._forward_pre_hooks and not native.model._forward_hooks
    assert all(not layer._forward_hooks for layer in native.model.model.layers)


def test_whole_long_native_reply_keeps_every_prediction_and_late_steering_trace(native, monkeypatch):
    from transformers import LogitsProcessor

    factory = agentic_engine.get_agentic_handles
    monkeypatch.setattr(agentic_engine, 'get_agentic_handles',
                        lambda model, **_kwargs: factory(model, config=AgenticConfig(base_model_only=True)))
    prompt = 'Explain the route now'
    prefix_length = engine._inputs(native, prompt)['input_ids'].shape[-1]
    words = ['first', 'second', 'answer', 'now', 'Explain', 'route', 'the']
    word_ids = native.tok.convert_tokens_to_ids(words)

    class ContinueInSequence(LogitsProcessor):
        def __call__(self, input_ids, scores):
            # A varied deterministic continuation avoids EOS/repetition stopping
            # while retaining a genuine native cached forward for every token.
            index = (input_ids.shape[-1] - prefix_length) % len(word_ids)
            result = torch.full_like(scores, -float('inf'))
            result[:, word_ids[index]] = 0.
            return result

    generate = native.model.generate

    def force_continuation(**kwargs):
        kwargs['logits_processor'].append(ContinueInSequence())
        return generate(**kwargs)

    monkeypatch.setattr(native.model, 'generate', force_continuation)
    observed, late_states, forward_lengths = [], [], []
    direction = torch.eye(native.d_model)[0]

    def observe(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        observed.append(hidden[0, -1].detach().clone())

    def observe_root(_module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        forward_lengths.append(ids.shape[-1])

    class LateSteer:
        handle = None
        chunks = 0

        def __call__(self, _text, _state):
            self.chunks += 1
            if self.chunks != 5:
                return

            def steer(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                edited = hidden.clone()
                edited[..., 0] += .8
                late_states.append(edited[0, -1].detach().clone())
                return (edited,) + output[1:] if isinstance(output, tuple) else edited

            self.handle = native.model.model.layers[0].register_forward_hook(steer)

        def cleanup(self):
            if self.handle is not None:
                self.handle.remove()

    observer = native.model.model.layers[0].register_forward_hook(observe)
    root_observer = native.model.register_forward_pre_hook(observe_root, with_kwargs=True)
    late = LateSteer()
    try:
        reply, telemetry = agentic_engine.generate_agentic_text(
            native, instruction=prompt, config=AgenticConfig(repetition_penalty=1.),
            max_new_tokens=160, return_telemetry=True, mid_chunk_hook=late,
            generation_probes={'whole_reply': {0: direction}})
        expected_text = ' '.join(words[index % len(words)] for index in range(160))
        assert reply == expected_text and len(reply) > 600
        assert len(' '.join(words[index % len(words)] for index in range(120))) > 600
        assert telemetry['tokens_generated'] == len(observed) == len(forward_lengths) == 160
        assert len(late_states) == 40 and late.chunks == 7
        assert [length for length in forward_lengths if length > 1] == [
            prefix_length + offset for offset in range(0, 160, 24)]
        reading = telemetry['generation_probe_readings']['whole_reply']
        expected_states = torch.stack(observed[:120] + late_states)
        expected = _cosine(expected_states, direction)
        assert reading['token_count'] == len(reading['by_token']) == 160
        assert reading['prediction_positions'] == list(range(prefix_length, prefix_length + 160))
        torch.testing.assert_close(torch.tensor(reading['by_token']), expected)
        assert min(reading['by_token'][120:]) > .98
        assert max(reading['by_token'][:120]) < .98
        assert reading['raw_score'] == pytest.approx(expected[-1].item())
    finally:
        late.cleanup()
        observer.remove()
        root_observer.remove()
    assert not native.model._forward_pre_hooks and not native.model._forward_hooks
    assert all(not layer._forward_hooks for layer in native.model.model.layers)
