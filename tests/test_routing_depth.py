"""Sequential ToT depth and replay span, exercised without a model checkpoint."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from invariants import agentic_engine
from invariants.config import AgenticConfig, _global_registry
from invariants.mesa import Proposal


class AdditiveLayer(nn.Module):
    """Collinear states form a known plateau; additions make replay observable."""

    def __init__(self, amount):
        super().__init__()
        self.amount = amount
        self.calls = []

    def forward(self, hidden, **kwargs):
        self.calls.append((hidden.detach().clone(), dict(kwargs)))
        return hidden + hidden.new_tensor([self.amount, 0.])


class SmallLogits(nn.Module):
    def forward(self, hidden):
        # A larger positive x has lower entropy, without saturating after replay.
        return torch.cat((hidden[..., :1] * .01, torch.zeros_like(hidden[..., :1])), dim=-1)


class RecordingCommittee:
    def __init__(self, empty=False):
        self.calls = 0
        self.empty = empty

    def proposals(self, state):
        self.calls += 1
        if self.empty:
            return []
        return [
            Proposal("plus", torch.tensor([1., 0.]), 1.),
            Proposal("minus", torch.tensor([-1., 0.]), 1.),
        ]


@contextmanager
def routing_harness(**overrides):
    """Install the real routing hooks around four tiny deterministic CPU layers."""
    config = AgenticConfig()
    config.synthesis_enabled = False
    config.cache_enabled = False
    config.cache_write_enabled = False
    config.entropy_threshold = -1.
    config.epsilon = .05
    config.steer_fraction = .1
    config.disabled_steer_channels = frozenset()
    config.routing_entropy_weight = 1.
    config.routing_collapse_penalty = 0.
    config.routing_probe_terms = []
    config.expert_proof_weight = 0.
    config.committee = RecordingCommittee()
    for name, value in overrides.items():
        setattr(config, name, value)

    layers = nn.ModuleList([AdditiveLayer(amount) for amount in (0., 1., 2., 3.)])
    wrapped = SimpleNamespace(
        model=SimpleNamespace(
            model=SimpleNamespace(layers=layers, norm=nn.Identity()),
            dtype=torch.float32,
            lm_head=SmallLogits(),
        ),
        device=torch.device("cpu"), d_model=2, n_layers=len(layers),
    )
    trace = []
    handles = []
    with patch.object(_global_registry, "get_special_vec", return_value=None), \
            patch.object(agentic_engine, "get_steer_cap_fraction", return_value=.25), \
            patch.object(agentic_engine.torch.cuda, "is_available", return_value=False):
        try:
            handles, state = agentic_engine.get_agentic_handles(
                wrapped, vecs={}, config=config, synthesis_recorder=trace,
            )

            def forward(seq_length=1, **kwargs):
                hidden = torch.tensor([[[1., 0.]]]).repeat(1, seq_length, 1)
                with torch.no_grad():
                    for layer in layers:
                        hidden = layer(hidden, **kwargs)
                return hidden

            yield SimpleNamespace(
                config=config, layers=layers, state=state, trace=trace, forward=forward,
            )
        finally:
            for handle in handles:
                handle.remove()


def expected_state(stages, replay_addition=6.):
    # Ordinary pass reaches 7; each selected plus branch gets a 10% injection
    # followed by the chosen layer additions, then becomes the next stage input.
    value = 7.
    for _ in range(stages):
        value = value * 1.1 + replay_addition
    return torch.tensor([[[value, 0.]]])


def test_defaults_preserve_one_stage_and_the_entire_detected_region():
    config = AgenticConfig()
    assert config.routing_depth == 1
    assert config.routing_layer_span == 0


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_depth_chains_selected_states_and_counts_one_routing_event(depth):
    with routing_harness(routing_depth=depth) as harness:
        result = harness.forward()
        torch.testing.assert_close(result, expected_state(depth))
        assert [len(layer.calls) for layer in harness.layers] == [1] + [1 + depth] * 3
        assert harness.config.committee.calls == depth
        assert harness.state["routing_events"] == int(depth > 0)
        assert harness.state["chunk_routing_events"] == int(depth > 0)
        assert harness.state["total_loops_this_token"] == int(depth > 0)
        assert len(harness.trace) == depth
        assert [item["stage_index"] for item in harness.trace] == list(range(1, depth + 1))
        assert all(item["winner"] == "plus" for item in harness.trace)
        if depth:
            assert harness.trace[-1]["stop_reason"] == "depth_limit"


@pytest.mark.parametrize("span,start,addition", [(0, 1, 6.), (1, 3, 3.), (2, 2, 5.), (99, 1, 6.)])
def test_layer_span_replays_only_the_tail_of_the_detected_region(span, start, addition):
    with routing_harness(routing_depth=2, routing_layer_span=span) as harness:
        torch.testing.assert_close(harness.forward(), expected_state(2, addition))
        assert [len(layer.calls) for layer in harness.layers] == [
            1 + (2 if index >= start else 0) for index in range(4)
        ]
        assert all(item["replay_start"] == start for item in harness.trace)
        assert all(item["replay_end"] == 3 for item in harness.trace)


def test_satisfied_winner_stops_before_the_depth_limit():
    head = SmallLogits()
    baseline_entropy = agentic_engine._entropy_from_logits(head(expected_state(0))).item()
    winner_entropy = agentic_engine._entropy_from_logits(head(expected_state(1))).item()
    threshold = (baseline_entropy + winner_entropy) / 2
    assert winner_entropy < threshold < baseline_entropy
    with routing_harness(routing_depth=5, entropy_threshold=threshold) as harness:
        torch.testing.assert_close(harness.forward(), expected_state(1))
        assert harness.config.committee.calls == 1
        assert len(harness.trace) == 1
        assert harness.trace[-1]["stop_reason"] == "entropy_threshold"


def test_an_empty_roster_does_not_repeat_the_baseline():
    with routing_harness(routing_depth=5, committee=RecordingCommittee(empty=True)) as harness:
        torch.testing.assert_close(harness.forward(), expected_state(0))
        assert [len(layer.calls) for layer in harness.layers] == [1] * 4
        assert harness.config.committee.calls == 1
        assert harness.state["routing_events"] == 1
        assert len(harness.trace) == 1
        assert harness.trace[-1]["stop_reason"] == "no_proposals"


def test_existing_event_cap_applies_across_tokens_without_capping_stages():
    with routing_harness(routing_depth=3, max_loops=1, max_routing_events=1) as harness:
        torch.testing.assert_close(harness.forward(), expected_state(3))
        torch.testing.assert_close(harness.forward(), expected_state(0))
        assert harness.state["routing_events"] == 1
        assert harness.config.committee.calls == 3
        assert len(harness.trace) == 3
        assert [len(layer.calls) for layer in harness.layers] == [2, 5, 5, 5]


@pytest.mark.parametrize("budget", ["max_loops", "max_routing_events"])
def test_zero_existing_event_budgets_still_disable_routing(budget):
    with routing_harness(routing_depth=3, **{budget: 0}) as harness:
        torch.testing.assert_close(harness.forward(), expected_state(0))
        assert harness.state["routing_events"] == 0
        assert harness.config.committee.calls == 0
        assert harness.trace == []


def test_prefill_skips_replay_and_preserves_the_budget_for_decode():
    with routing_harness(routing_depth=3, max_routing_events=1) as harness:
        torch.testing.assert_close(harness.forward(seq_length=4), expected_state(0).repeat(1, 4, 1))
        assert harness.state["routing_events"] == 0
        assert harness.trace == []
        torch.testing.assert_close(harness.forward(), expected_state(3))
        assert harness.state["routing_events"] == 1


def test_manual_replay_does_not_update_detection_state_or_recurse():
    with routing_harness(routing_depth=3) as harness:
        detector_batch_sizes = []
        observers = []

        def observe(_module, args, _output):
            if args[0].shape[0] == 2:
                detector_batch_sizes.append(harness.state["prev_h"].shape[0])

        try:
            # Observers run after the routing hooks and expose accidental updates
            # by those hooks during the manual parallel layer replay.
            observers = [layer.register_forward_hook(observe) for layer in harness.layers]
            torch.testing.assert_close(harness.forward(), expected_state(3))
        finally:
            for observer in observers:
                observer.remove()
        assert detector_batch_sizes == [1] * 9
        assert harness.config.committee.calls == 3
        assert harness.state["routing_events"] == 1
        torch.testing.assert_close(harness.state["prev_h"], expected_state(3))


def test_replay_uses_no_cache_and_repeats_single_token_position_ids():
    kwargs = {
        "use_cache": True,
        "past_key_value": object(),
        "attention_mask": torch.ones(1, 9),
        "position_ids": torch.tensor([[8]]),
    }
    with routing_harness(routing_depth=2) as harness:
        torch.testing.assert_close(harness.forward(**kwargs), expected_state(2))
        replay_calls = [
            call for layer in harness.layers for call in layer.calls if call[0].shape[0] == 2
        ]
        assert len(replay_calls) == 6
        for _, replay_kwargs in replay_calls:
            assert replay_kwargs["use_cache"] is False
            assert "past_key_value" not in replay_kwargs
            assert "attention_mask" not in replay_kwargs
            torch.testing.assert_close(replay_kwargs["position_ids"], torch.tensor([[8], [8]]))
        assert kwargs["use_cache"] is True


def test_tiny_llama_cached_generation_keeps_one_cache_entry_per_real_token():
    from transformers import LlamaConfig, LlamaForCausalLM

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    handles = []
    observers = []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(122)
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=32, hidden_size=16, intermediate_size=24,
                num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=32, pad_token_id=0, bos_token_id=1, eos_token_id=2,
            )).eval()
        model.generation_config.suppress_tokens = [2]
        wrapped = SimpleNamespace(model=model, device=torch.device("cpu"), d_model=16, n_layers=4)
        config = AgenticConfig(
            routing_depth=3, routing_layer_span=2,
            force_synthesis=True, entropy_threshold=-1.,
            synthesis_enabled=False, cache_enabled=False, cache_write_enabled=False,
            max_routing_events=1, disabled_steer_channels=frozenset(),
        )
        direction = torch.ones(16)
        replay_kwargs = []

        def observe(_module, args, kwargs):
            if args[0].shape[0] == 2:
                replay_kwargs.append(dict(kwargs))

        with patch.object(_global_registry, "get_special_vec", return_value=None), \
                patch.object(agentic_engine.torch.cuda, "is_available", return_value=False):
            trace = []
            handles, state = agentic_engine.get_agentic_handles(
                wrapped, vecs={"plus": direction, "minus": -direction},
                config=config, synthesis_recorder=trace,
            )
            observers = [
                layer.register_forward_pre_hook(observe, with_kwargs=True)
                for layer in model.model.layers
            ]
            with torch.no_grad():
                generated = model.generate(
                    input_ids=torch.tensor([[1, 3, 4]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                    max_new_tokens=3, do_sample=False, use_cache=True, return_dict_in_generate=True,
                )
        assert generated.sequences.shape == (1, 6)
        assert torch.isfinite(generated.past_key_values.layers[0].keys).all()
        # The final generated token has not been fed back yet: 3 prompt tokens
        # plus 2 decode calls are cached, regardless of the six branch replays.
        assert all(generated.past_key_values.get_seq_length(index) == 5 for index in range(4))
        assert state["routing_events"] == 1
        assert len(trace) == 3
        assert len(replay_kwargs) == 6
        assert all(item["use_cache"] is False for item in replay_kwargs)
        assert all("past_key_values" not in item and "past_key_value" not in item for item in replay_kwargs)
    finally:
        for handle in observers + handles:
            handle.remove()
        torch.set_num_threads(old_threads)
