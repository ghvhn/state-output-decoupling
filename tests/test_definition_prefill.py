"""Native context and supplied-signal boundaries, without a semantic detector."""
import copy
import hashlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from invariants.definition_prefill import (
    DEFAULT_DEFINITION_PREFIX, HybridBoundaryGate,
    finish_assistant_continuation, prepare_assistant_continuation,
)


EVIDENCE = '  Memory 0: position=-0.2; reply="I qualify my claim."\nMemory 3: position=+0.8.  '


class Tokenizer:
    all_special_tokens = ['<bos>', '<eos>', '[assistant]']
    bos_token, eos_token = '<bos>', '<eos>'

    def __init__(self):
        self.templates = []
        self.encodings = []
        self.native = None

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append((copy.deepcopy(messages), dict(kwargs)))
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        self.native = '<bos>[user]' + messages[0]['content'] + '<eos>[assistant]'
        return self.native

    def __call__(self, text, **kwargs):
        self.encodings.append((text, dict(kwargs)))
        assert kwargs == {'add_special_tokens': False, 'return_tensors': 'pt'}
        # Appending the prefill retokenizes two header-tail tokens. Counting
        # the prefix in isolation (seven tokens here) would be incorrect.
        ids = [1, 2, 30, 31] if text == self.native else [1, 2, 99, 100, 101]
        if text == DEFAULT_DEFINITION_PREFIX:
            ids = [7] * 7
        return {'input_ids': torch.tensor([ids])}


def model():
    raw = torch.nn.Linear(3, 3)
    raw.generation_config = {'use_cache': True, 'max_new_tokens': 77}
    return SimpleNamespace(tok=Tokenizer(), model=raw,
                           history=[{'role': 'user', 'content': 'Persistent conversation'}])


def test_native_prefill_preserves_exact_evidence_and_measures_merged_boundary_tokens():
    wrapped = model()
    history_before, config_before = copy.deepcopy(wrapped.history), copy.deepcopy(wrapped.model.generation_config)
    observer = wrapped.model.register_forward_pre_hook(Mock())
    hook_ids = set(wrapped.model._forward_pre_hooks)
    try:
        prompt, metadata = prepare_assistant_continuation(wrapped, EVIDENCE, DEFAULT_DEFINITION_PREFIX)
        assert wrapped.tok.templates == [([{'role': 'user', 'content': EVIDENCE}],
                                         {'tokenize': False, 'add_generation_prompt': True})]
        assert prompt == wrapped.tok.native + DEFAULT_DEFINITION_PREFIX
        assert prompt.count('[assistant]') == 1
        assert prompt.endswith(DEFAULT_DEFINITION_PREFIX) and not prompt.endswith('<eos>')
        assert [row[0] for row in wrapped.tok.encodings] == [wrapped.tok.native, prompt]
        assert metadata['native_header_token_count'] == 4
        assert metadata['prompt_token_count'] == 5
        assert metadata['prefix_token_count'] == 3
        assert metadata['prefix_token_delta'] == 1
        assert metadata['retokenized_header_tokens'] == 2
        assert metadata['evidence_sha256'] == hashlib.sha256(EVIDENCE.encode('utf-8')).hexdigest()
        assert metadata['assistant_prefix'] == DEFAULT_DEFINITION_PREFIX
        assert metadata['pre_formatted'] and metadata['unclosed_assistant']
        assert metadata['candidate'] and not metadata['semantic_truth_checked']
        assert wrapped.history == history_before
        assert wrapped.model.generation_config == config_before
        assert set(wrapped.model._forward_pre_hooks) == hook_ids
        next(iter(wrapped.model._forward_pre_hooks.values())).assert_not_called()
    finally:
        observer.remove()


@pytest.mark.parametrize('prefix', ['', None])
def test_no_prefix_still_leaves_one_native_assistant_header(prefix):
    wrapped = model()
    prompt, metadata = prepare_assistant_continuation(wrapped, EVIDENCE, prefix)
    assert prompt == wrapped.tok.native
    assert len(wrapped.tok.templates) == len(wrapped.tok.encodings) == 1
    assert metadata['assistant_prefix'] == ''
    assert metadata['prefix_token_count'] == metadata['prefix_token_delta'] == metadata['retokenized_header_tokens'] == 0
    assert finish_assistant_continuation(prefix, 'A generated answer.') == 'A generated answer.'


@pytest.mark.parametrize('prefix', [
    '<eos>', 'Words [assistant]', 'Words [INST]', '</s>',
    '<|eot_id|>', '<|unregistered_header|>', '<<SYS>>system instruction',
])
def test_chat_control_prefix_is_rejected_before_rendering_or_tokenization(prefix):
    wrapped = model()
    with pytest.raises(ValueError, match='chat control tokens'):
        prepare_assistant_continuation(wrapped, EVIDENCE, prefix)
    assert not wrapped.tok.templates and not wrapped.tok.encodings


def test_ordinary_comparison_text_is_allowed_and_suffix_is_reconstructed_verbatim():
    wrapped = model()
    prefix = 'At x < y, '
    prompt, _ = prepare_assistant_continuation(wrapped, EVIDENCE, prefix)
    assert prompt.endswith(prefix)
    suffix = 'qualification increases.\n'
    assert finish_assistant_continuation(prefix, suffix) == prefix + suffix
    assert finish_assistant_continuation(DEFAULT_DEFINITION_PREFIX, 'a change in qualification.') == (
        DEFAULT_DEFINITION_PREFIX + 'a change in qualification.')
    with pytest.raises(ValueError, match='already includes'):
        finish_assistant_continuation(DEFAULT_DEFINITION_PREFIX, DEFAULT_DEFINITION_PREFIX + suffix)


@pytest.mark.parametrize('evidence', ['', '   ', None, 3])
def test_invalid_evidence_fails_without_touching_tokenizer(evidence):
    wrapped = model()
    with pytest.raises(ValueError, match='nonempty evidence'):
        prepare_assistant_continuation(wrapped, evidence, DEFAULT_DEFINITION_PREFIX)
    assert not wrapped.tok.templates and not wrapped.tok.encodings


def test_native_rendering_failure_leaves_persistent_history_and_hooks_unchanged():
    wrapped = model()
    before = copy.deepcopy(wrapped.history)
    wrapped.tok.apply_chat_template = Mock(return_value='')
    with pytest.raises(ValueError, match='native text chat template'):
        prepare_assistant_continuation(wrapped, EVIDENCE, DEFAULT_DEFINITION_PREFIX)
    assert wrapped.history == before and not wrapped.model._forward_pre_hooks
    assert not wrapped.tok.encodings


def evaluate(gate, phase='boundary', activation=.9, probability=.9, text='A qualified claim. ', remaining=0):
    return gate.evaluate(phase, activation_score=activation, next_connector_probability=probability,
                         generated_text=text, connector_queue_remaining=remaining)


@pytest.mark.parametrize('activation,probability', [(.99, .2), (.2, .99), (.79, 1.), (1., .79)])
def test_each_signal_is_independently_insufficient(activation, probability):
    gate = HybridBoundaryGate(['boundary', 'relationship'])
    result = evaluate(gate, activation=activation, probability=probability)
    assert result['status'] == 'waiting' and not result['trigger'] and not result['stop']
    assert gate.current_phase == 'boundary' and gate.trace()['events'] == []
    assert evaluate(gate, activation=.8, probability=.8)['trigger'] is True


def test_ordered_phases_emit_once_even_if_repeated_scores_stay_high():
    gate = HybridBoundaryGate(['boundary', 'relationship', 'rule'])
    with pytest.raises(ValueError, match='Expected boundary phase boundary'):
        evaluate(gate, 'rule')
    assert evaluate(gate, 'boundary')['trigger'] is True
    for _ in range(4):
        assert evaluate(gate, 'boundary')['status'] == 'already_triggered'
    assert gate.trace()['phase_evaluations']['boundary'] == 1
    assert evaluate(gate, 'relationship')['trigger'] is True
    final = evaluate(gate, 'rule')
    assert final['trigger'] and final['all_boundaries_triggered']
    assert final['semantic_completion'] is False
    assert [event['phase'] for event in gate.trace()['events']] == ['boundary', 'relationship', 'rule']
    assert evaluate(gate, 'rule')['trigger'] is False


def test_unfinished_literal_queue_blocks_each_phase_without_duplicate_events():
    gate = HybridBoundaryGate(['boundary', 'relationship'])
    assert not evaluate(gate, remaining=4)['trigger']
    assert not evaluate(gate, remaining=1)['trigger']
    assert evaluate(gate, remaining=0)['trigger']
    assert not evaluate(gate, 'relationship', remaining=3)['trigger']
    assert evaluate(gate, 'relationship', remaining=0)['trigger']
    assert len(gate.trace()['events']) == 2


@pytest.mark.parametrize('partial', ['The claim is qualifi', 'The relation is unfinished-', "The claim isn't'"])
def test_high_scores_cannot_cut_an_unfinished_word(partial):
    gate = HybridBoundaryGate(['boundary'])
    result = evaluate(gate, text=partial)
    assert not result['trigger'] and 'lexical boundary' in result['reason']
    assert evaluate(gate, text='The claim is qualified. ')['trigger'] is True


@pytest.mark.parametrize('empty_content', ['', '... ', '\n\t', 'word '])
def test_punctuation_or_scant_content_cannot_trigger_a_boundary(empty_content):
    gate = HybridBoundaryGate(['boundary'])
    assert not evaluate(gate, text=empty_content)['trigger']


def test_evaluation_limit_stops_unqualified_without_advancing_or_assuming_completion():
    gate = HybridBoundaryGate(['boundary', 'relationship'], max_evaluations=2)
    assert evaluate(gate, activation=.2)['status'] == 'waiting'
    result = evaluate(gate, probability=.2)
    assert result['status'] == 'unqualified' and result['stop'] and not result['trigger']
    assert result['current_phase'] == 'boundary' and result['semantic_completion'] is False
    late = evaluate(gate)
    assert late['status'] == 'unqualified' and not late['trigger']
    assert gate.trace()['events'] == []
    assert gate.trace()['phase_evaluations'] == {'boundary': 2, 'relationship': 0}


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf'), None, True])
def test_invalid_signal_is_rejected_before_any_boundary_state_changes(value):
    gate = HybridBoundaryGate(['boundary'])
    with pytest.raises(ValueError, match='Activation score'):
        evaluate(gate, activation=value)
    with pytest.raises(ValueError, match='Next connector probability'):
        evaluate(gate, probability=value)
    assert gate.trace()['phase_evaluations'] == {'boundary': 0}
    assert gate.trace()['events'] == []


@pytest.mark.parametrize('phases', [[], ['boundary', 'boundary'], [' '], 'boundary'])
def test_phase_order_requires_distinct_nonempty_names(phases):
    with pytest.raises(ValueError, match='ordered unique nonempty names'):
        HybridBoundaryGate(phases)


def test_provenance_identity_does_not_certify_a_trained_semantic_detector():
    gate = HybridBoundaryGate(['boundary'], calibration_id='operator-test-calibration')
    result = evaluate(gate)
    assert result['trigger'] and result['candidate']
    assert result['semantic_completion'] is False and result['detector_training_pending'] is True
    assert gate.trace()['score_calibration_id'] == 'operator-test-calibration'
    assert gate.trace()['detector_training_pending'] is True
