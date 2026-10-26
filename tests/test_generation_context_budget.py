from types import SimpleNamespace
import torch
import pytest
from invariants.generation_budget import check_context_budget
from scripts.interactive_phenomenality import compact_spawn_command_reference


def test_oversized_context_rejected_before_forward_without_truncation(monkeypatch):
    monkeypatch.setenv('TDA_MAX_CONTEXT_TOKENS', '128')
    inputs = {'input_ids': torch.zeros((1, 120), dtype=torch.long)}
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=4096))
    assert check_context_budget(model, inputs, 8) == 120
    with pytest.raises(ValueError, match='No context was discarded'):
        check_context_budget(model, inputs, 24)
    assert inputs['input_ids'].shape[-1] == 120


def test_model_position_limit_is_respected(monkeypatch):
    monkeypatch.setenv('TDA_MAX_CONTEXT_TOKENS', '8192')
    with pytest.raises(ValueError, match='limit is 64'):
        check_context_budget(SimpleNamespace(config=SimpleNamespace(max_position_embeddings=64)),
                             {'input_ids': torch.zeros((1, 65))})


def test_spawn_reference_is_bounded_and_honors_visibility():
    reference = compact_spawn_command_reference({'spawn'})
    assert len(reference) <= 12000
    assert ':spawn' not in reference
    assert ':probe' in reference and ':tune' in reference
    assert ':help :command' in reference
