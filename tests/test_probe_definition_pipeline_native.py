"""Run the gated callback with native chunked CPU generation and real hooks.

This checks the rejection boundary and activation placement, not semantic skill.
"""
import copy

import pytest
import torch

from invariants import engine, probe_definition_steps
from invariants.probe_behavior import build_definition_handles
from invariants.probe_definition_pipeline import run_definition_pipeline
from test_probe_definition_process_native import _native_model


@pytest.mark.parametrize('gain', [.04, 0., -.04])
def test_native_stage_budget_stops_before_next_step_and_preserves_evidence(tmp_path, monkeypatch, gain):
    selected = [(index, {'reading': reading}) for index, reading in enumerate((-.8, -.3, .4, .9))]
    evidence = (
        '[Probe p_ffffffffffffffffffffffff]\n'
        'Memory 0: position=-0.8; reply="Take the first corridor."\n'
        'Memory 1: position=-0.3; reply="Use the first available route."\n'
        'Memory 2: position=+0.4; reply="Compare the routes before choosing."\n'
        'Memory 3: position=+0.9; reply="Check the alternatives before choosing."'
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    observers = []
    try:
        wrapped = _native_model(evidence)
        bank, _ = probe_definition_steps.get_or_mint_definition_steps(wrapped, (0, 1, 2, 3), tmp_path / 'banks')
        bank_before = copy.deepcopy(bank)
        generation_config = copy.deepcopy(wrapped.model.generation_config.to_dict())
        subject = {layer: torch.eye(16)[0] for layer in range(4)}
        expected_input = engine._inputs(wrapped, evidence)['input_ids']
        prefix_length = expected_input.shape[-1]
        forwards, changes, calls, before = [], [], [], {}

        def observe_root(_module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            forwards.append(ids.detach().clone())
        observers.append(wrapped.model.register_forward_pre_hook(observe_root, with_kwargs=True))
        for layer in range(4):
            def capture_before(_module, _args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                before[layer] = hidden.detach().clone()
            observers.append(wrapped.model.model.layers[layer].register_forward_hook(capture_before))

        def generate_stage(step, stage_evidence, budget):
            calls.append(step)
            assert step == 'compare' and stage_evidence == evidence and budget == 8
            assert len(wrapped.model._forward_pre_hooks) == 1  # Our unrelated observer.
            handles = build_definition_handles(wrapped, subject, {step: bank[step]},
                                               task_gains={step: gain}, cap_fraction=.08)
            local_observers = []
            try:
                for layer in range(4):
                    def capture_after(_module, _args, output, layer=layer):
                        hidden = output[0] if isinstance(output, tuple) else output
                        changes.append((layer, before[layer], (hidden - before[layer]).detach().clone()))
                    local_observers.append(wrapped.model.model.layers[layer].register_forward_hook(capture_after))
                inputs = engine._inputs(wrapped, stage_evidence)
                full = engine._generate_ids(wrapped, inputs, budget)
                tokens = full[prefix_length:]
                return wrapped.tok.decode(tokens, skip_special_tokens=True), {'tokens_generated': len(tokens)}
            finally:
                for handle in handles + local_observers:
                    handle.remove()

        monkeypatch.setenv('TDA_GENERATION_CHUNK_TOKENS', '2')
        result = run_definition_pipeline(evidence, selected, generate_stage,
            step_gains={**probe_definition_steps.STEP_GAINS, 'compare': gain}, stage_max_tokens=8)
        assert calls == ['compare'] and result['failed_step'] == 'compare'
        assert result['status'] == 'incomplete' and result['completed_steps'] == []
        assert result['tokens_generated'] == 8 and len(result['stages']) == 1
        assert result['structural_completion_only'] is True
        assert [ids.shape[-1] for ids in forwards] == [
            prefix_length, 1, prefix_length + 2, 1, prefix_length + 4, 1, prefix_length + 6, 1]
        for ids in forwards:
            if ids.shape[-1] > 1:
                torch.testing.assert_close(ids[:, :prefix_length], expected_input)
        for layer, original, change in changes:
            expected = torch.zeros_like(change)
            scale = original.norm(dim=-1, keepdim=True)
            if original.shape[-2] == 1:
                expected += scale * gain * bank['compare'][layer]
            else:
                expected[:, prefix_length - 1:] += scale[:, prefix_length - 1:] * gain * bank['compare'][layer]
                expected[:, prefix_length - 1] += scale[:, prefix_length - 1] * .02 * subject[layer]
            torch.testing.assert_close(change, expected, atol=1e-7, rtol=1e-4)
        assert len(wrapped.model._forward_pre_hooks) == 1
        assert all(len(layer._forward_hooks) == 1 for layer in wrapped.model.model.layers)
        assert wrapped.model.generation_config.to_dict() == generation_config
        for step in bank:
            for layer in bank[step]:
                torch.testing.assert_close(bank[step][layer], bank_before[step][layer])
    finally:
        for handle in observers:
            handle.remove()
        torch.set_num_threads(previous_threads)
    assert not wrapped.model._forward_pre_hooks
    assert all(not layer._forward_hooks for layer in wrapped.model.model.layers)
