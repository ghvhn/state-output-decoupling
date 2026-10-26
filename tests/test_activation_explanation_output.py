"""Activation candidates are structurally qualified without a semantic oracle."""

import copy
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from invariants.activation_explanation_output import (
    assess_activation_explanation, record_activation_explanation,
)


DEFINITION = (
    'This direction represents third-person self-reference: lower readings favor '
    'first-person wording while higher readings favor referring to oneself by name.'
)
OPENING = "At this time, I'm happy to chat with you. What would you like to talk about?"


def assess(text, *, tokens=30, budget=384, finish='eos'):
    return assess_activation_explanation(
        text, {'tokens_generated': tokens, 'finish_reason': finish}, budget)


@pytest.mark.parametrize('text', [
    DEFINITION,
    'It captures how often speakers refer to themselves in the third person.',
    'Illeism is referring to oneself in the third person.',
    'A tendency to refer to oneself in the third person.',
    'Willingness to revise a conclusion when circumstances change.',
    'The degree to which a reply distinguishes competing possibilities.',
    'how firmly an answer commits to its claims: lower readings hedge conclusions whereas higher readings state them categorically.',
    'Whether a claim is supported by independently checkable observations.',
    'A preference for using unusually ornate words.',
    'This direction denotes the relationship between two arbitrary properties.',
    'I am referring to myself in the third person; this direction represents that tendency.',
    "I'm referring to myself by name. This direction represents third-person self-reference.",
    'Use of third-person self-reference is referring to oneself by name.',
    'Use of names rather than first-person pronouns.',
    'This direction represents being happy to help, a disposition toward assisting other people.',
    'This direction describes willingness to chat: lower readings show little willingness, while higher readings show people happy to chat.',
    # Passing this deliberately different meaning proves no trait oracle is used.
    'This direction represents thermal conductivity, the rate of heat transfer through a material.',
])
def test_general_definition_form_passes_without_prefix_endpoints_or_true_meaning(text):
    result = assess(text)
    assert result['accepted'] is True
    assert result['status'] == 'candidate'
    assert result['structural_only'] is True
    assert result['semantic_truth_checked'] is False


@pytest.mark.parametrize('text, reason', [
    (OPENING, 'chat opening'),
    ('', 'No generated'),
    ('  \n\t', 'No generated'),
    (None, 'No generated'),
    ('Hello! How can I help you today?', 'chat opening'),
    ('How are you today?', 'chat opening'),
    ('What is your name?', 'chat opening'),
    ('Thank you for the conversation.', 'chat opening'),
    ('Nice to meet you.', 'chat opening'),
    ('You are welcome.', 'chat opening'),
    ('I am ready to discuss anything you like.', 'chat opening'),
    ("I'm here to help. This direction represents a general property.", 'chat opening'),
    ('I can help you. This direction represents a general property.', 'chat opening'),
    ("Currently, I'm happy to help. This direction represents a general property.", 'chat opening'),
    ('You should use clearer wording and avoid confusing your audience.', 'advice'),
    ('Try using your name instead of a pronoun.', 'advice'),
    ('You could try to use clearer wording.', 'advice'),
    ('The conversation history is a record of our earlier exchanges.', 'recounting'),
    ('This is a summary of our conversation about model behavior.', 'recounting'),
    ('Memory 3 describes a reply in which you asked for help.', 'recounting'),
    ('These memories show different ways to answer questions.', 'recounting'),
    ('third-person self-reference', 'general property relation'),
    ('Lower readings favor one style while higher readings favor another.', 'general property relation'),
    ('This direction represents a.', 'general property relation'),
    ('The weather looks pleasant today.', 'general property relation'),
])
def test_wrong_output_kind_remains_unqualified(text, reason):
    result = assess(text)
    assert result['accepted'] is False
    assert result['status'] == 'unqualified'
    assert reason.lower() in result['reason'].lower()
    assert result['structural_only'] and not result['semantic_truth_checked']


@pytest.mark.parametrize('control', [
    '<|assistant|>', '<<tool:search>>', '```python\npass\n```',
    '[INST] define it [/INST]', '<tool_call>{}</tool_call>',
    '{"tool_calls": [{"name": "search", "arguments": {}}]}',
    '{"function_call": {"name": "search", "arguments": {}}}',
    'assistant: This direction represents a general relation.',
])
def test_controls_and_tool_calls_are_rejected_even_with_definition_text(control):
    result = assess(DEFINITION + '\n' + control)
    assert not result['accepted'] and result['status'] == 'unqualified'
    assert 'tool calls' in result['reason']


@pytest.mark.parametrize('tokens,budget,finish', [
    (384, 384, 'eos'), (385, 384, 'eos'), (3, 384, 'length'),
    (3, 384, 'max_tokens'), (3, 384, 'max_new_tokens'),
])
def test_reaching_budget_or_reported_truncation_is_incomplete(tokens, budget, finish):
    result = assess(DEFINITION, tokens=tokens, budget=budget, finish=finish)
    assert not result['accepted'] and result['status'] == 'incomplete'
    assert result['tokens_generated'] == tokens
    assert result['structural_only'] and not result['semantic_truth_checked']


def test_one_token_below_budget_can_pass_and_incomplete_overrides_other_failures():
    assert assess(DEFINITION, tokens=383)['status'] == 'candidate'
    assert assess(OPENING, tokens=384)['status'] == 'incomplete'
    assert assess('', tokens=384)['status'] == 'incomplete'


@pytest.mark.parametrize('telemetry', [None, {}, {'tokens': 3}])
def test_missing_token_telemetry_raises_instead_of_certifying_completeness(telemetry):
    with pytest.raises(ValueError, match='requires tokens_generated'):
        assess_activation_explanation(DEFINITION, telemetry, 384)


@pytest.mark.parametrize('count', [-1, 1.5, float('nan'), float('inf'), True, None, 'unknown'])
def test_invalid_generated_token_counts_raise(count):
    with pytest.raises(ValueError, match='tokens_generated'):
        assess(DEFINITION, tokens=count)


@pytest.mark.parametrize('budget', [0, -1, 1.5, float('nan'), float('inf'), True, None])
def test_invalid_token_budgets_raise(budget):
    with pytest.raises(ValueError, match='token budget'):
        assess(DEFINITION, budget=budget)


def test_assessment_preserves_original_text_and_telemetry():
    text = '  ' + DEFINITION + '\n'
    telemetry = {'tokens_generated': 30, 'finish_reason': 'eos', 'nested': {'source': 'test'}}
    original = copy.deepcopy(telemetry)
    result = assess_activation_explanation(text, telemetry, 384)
    assert result['accepted'] and telemetry == original
    assert text == '  ' + DEFINITION + '\n'


def test_operator_trace_appends_under_its_owner_and_preserves_prior_rows(tmp_path):
    owner = tmp_path / 'owner.json'
    owner.write_text('{"unrelated": true}', encoding='utf-8')
    log = tmp_path / 'owner.activation-explain.jsonl'
    prior = {'timestamp': '2026-10-01T00:00:00+00:00', 'text': 'older attempt'}
    log.write_text(json.dumps(prior) + '\n', encoding='utf-8')
    attempt = {
        'text': DEFINITION, 'qualification': assess(DEFINITION),
        'controls': {'subject': .02, 'operation': .12, 'reference': .04},
        'metadata': {'semantic_truth_checked': False},
        'telemetry': {'tokens_generated': 30},
    }
    original = copy.deepcopy(attempt)

    assert record_activation_explanation(SimpleNamespace(path=owner), attempt) == log
    rows = [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 2 and rows[0] == prior
    assert {key: rows[1][key] for key in attempt} == attempt
    assert datetime.fromisoformat(rows[1]['timestamp']).utcoffset().total_seconds() == 0
    assert attempt == original
    assert owner.read_text(encoding='utf-8') == '{"unrelated": true}'
    assert sorted(path.name for path in tmp_path.iterdir()) == [log.name, owner.name]


def test_operator_trace_falls_back_to_behavior_owner_and_keeps_owners_separate(tmp_path):
    first = SimpleNamespace(path=tmp_path / 'first.json')
    second = SimpleNamespace(path=None, probe_behavior=SimpleNamespace(path=tmp_path / 'second.jsonl'))
    first_log = record_activation_explanation(first, {'text': 'first attempt'})
    second_log = record_activation_explanation(second, {'text': 'second attempt'})
    assert first_log != second_log
    assert json.loads(first_log.read_text(encoding='utf-8'))['text'] == 'first attempt'
    assert json.loads(second_log.read_text(encoding='utf-8'))['text'] == 'second attempt'
    assert second_log.name == 'second.activation-explain.jsonl'


def test_operator_trace_prefers_matrix_owner_and_does_nothing_without_a_path(tmp_path):
    matrix = SimpleNamespace(path=tmp_path / 'main.json',
                             probe_behavior=SimpleNamespace(path=tmp_path / 'behavior.json'))
    assert record_activation_explanation(matrix, {'text': 'attempt'}).name == 'main.activation-explain.jsonl'
    assert not (tmp_path / 'behavior.activation-explain.jsonl').exists()
    assert record_activation_explanation(SimpleNamespace(), {'text': 'attempt'}) is None
    assert record_activation_explanation(SimpleNamespace(path='', probe_behavior=None), {}) is None
