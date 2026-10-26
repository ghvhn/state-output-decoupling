"""Require a definition kind without requiring a correct semantic interpretation."""
import copy
import hashlib

import pytest

from invariants.probe_definition_output import (
    OUTPUT_PREFIX, OUTPUT_VERSION, assess_definition_output,
    build_definition_request, run_definition_output,
)


EVIDENCE = (
    '  [Probe p_ffffffffffffffffffffffff]\n'
    'Memory 0: position=-0.8; reply="Take the first option."\n'
    'Memory 1: position=-0.3; reply="Choose the first available option."\n'
    'Memory 2: position=+0.4; reply="Compare alternatives before choosing."\n'
    'Memory 3: position=+0.9; reply="Check the alternatives and then choose."\n  '
)
SELECTED = [(0, {'reading': -.8}), (1, {'reading': -.3}),
            (2, {'reading': .4}), (3, {'reading': .9})]
DEFINITION = (
    'This probe measures deliberation, the extent to which an answer compares alternatives before committing: '
    'lower readings select immediately whereas higher readings weigh alternatives before deciding.'
)
RECAP = (
    'The provided text appears to be a conversation log between a user and a conversational AI.\n\n'
    'The conversation is divided into two sections: the negative side and the positive side. '
    'The negative side discusses interactions with the AI, while the positive side discusses the nature of reality.\n\n'
    'Some key points include:\n* The user explores the boundaries between human and machine.'
)


class Generator:
    def __init__(self, texts, *, counts=None, finish=None):
        self.texts = list(texts)
        self.counts, self.finish = counts, finish
        self.calls = []

    def __call__(self, request, budget):
        index = len(self.calls)
        self.calls.append((request, budget))
        return self.texts[index], {
            'tokens_generated': self.counts[index] if self.counts else 30,
            'finish_reason': self.finish[index] if self.finish else 'eos',
        }


def run(generator, **kwargs):
    return run_definition_output(EVIDENCE, SELECTED, generator, **kwargs)


def test_request_preserves_original_evidence_and_only_specifies_the_output_kind():
    request = build_definition_request(EVIDENCE)
    assert request.startswith(EVIDENCE + '\n\n[Definition request]\n')
    assert 'at most 120 words and three sentences' in request
    assert 'its meaning may be mistaken' in request
    assert 'This probe measures ' in request and OUTPUT_PREFIX == 'This probe measures '


@pytest.mark.parametrize('text', [
    DEFINITION,
    'This probe measures how firmly an answer commits to its claims: lower readings hedge conclusions whereas higher readings state them categorically.',
    'This probe measures willingness to update a conclusion: lower readings retain previous judgments while higher readings revise them.',
    'This probe measures firmness of assertion: lower readings qualify claims whereas higher readings make categorical assertions.',
    'This probe measures high confidence, the degree to which an answer treats conclusions as settled: lower readings hedge while higher readings assert categorically.',
    'This probe measures variation in how firmly answers assert claims: lower readings hedge conclusions whereas higher readings present them categorically.',
    'This probe measures coherence and logical connection of statements: lower readings give disconnected assertions whereas higher readings connect claims into an argument.',
    'This probe measures a tendency to recap memories: lower readings quote each event while higher readings condense events into a summary.',
    'This probe measures a preference for summarizing replies: lower readings repeat individual statements whereas higher readings describe their shared content.',
    'This probe measures willingness to log conversations: lower readings discard exchanges while higher readings preserve them for later reference.',
    'Deliberation is the practice of comparing alternatives before choosing. Lower readings select immediately while higher readings weigh alternatives.',
    'This probe measures sensitivity to evidence. As readings increase, replies shift from retaining prior judgments to revising them.',
    # The real model used comparative strength and "and" rather than whereas.
    'This probe measures the tendency to revise or retain an initial estimate in response to new information, with lower readings indicating a stronger inclination to retain the original estimate and higher readings indicating a stronger inclination to revise it.',
])
def test_general_definitions_pass_without_a_fixed_trait_vocabulary(text):
    result = assess_definition_output(text, selected=SELECTED)
    assert result['accepted']
    assert result['contrast'] == text
    assert result['structural_only'] and result['semantic_truth_checked'] is False
    assert result['check_version'] == OUTPUT_VERSION


def test_mistaken_semantic_definition_remains_an_accepted_hypothesis():
    # The measured replies concern route choice, not friendliness. That semantic
    # disagreement must not turn this output-form check into a truth evaluator.
    text = ('This probe measures friendliness, the tendency to use warm interpersonal language: '
            'lower readings express hostility whereas higher readings offer affectionate reassurance.')
    result = run(Generator([text]))
    assert result['status'] == 'hypothesis' and result['qualification']['accepted']
    assert result['qualification']['semantic_truth_checked'] is False


@pytest.mark.parametrize('text', [
    RECAP,
    'Lower positions discuss logistics while higher positions discuss philosophy.',
    'This probe measures lower readings discussing logistics whereas higher readings discuss philosophy.',
    'This probe measures tentative answers at lower readings while higher readings contain categorical assertions.',
    'This probe measures a 3-point scale.\n\n1. Memory 0 retains its estimate.\n2. Memory 3 revises it.',
    'This probe measures a collection of conversations about model identity: lower readings discuss controls whereas higher readings discuss philosophy.',
    'This probe measures a 3-point scale of memory groupings: lower readings contain technical memories whereas higher readings contain philosophical memories.',
    'This probe measures a log of conversations: lower readings discuss logistics whereas higher readings discuss philosophy.',
    'This probe measures a recap of replies: lower readings discuss logistics whereas higher readings discuss philosophy.',
    DEFINITION + '\n\nMemory 0 chooses immediately.',
    DEFINITION + ' You should compare the options before deciding.',
    DEFINITION + ' Try using this probe to improve your decisions.',
    DEFINITION + ' <<PROBE: rename p_ffffffffffffffffffffffff || deliberation>>',
    '# Definition\n' + DEFINITION,
    '',
])
def test_recaps_bare_contrasts_advice_and_tools_are_not_definitions(text):
    assert not assess_definition_output(text, selected=SELECTED)['accepted']


def test_defined_property_does_not_allow_individual_memory_recounting():
    text = ('This probe measures willingness to deliberate. Memory 0 discusses immediate selection, '
            'whereas Memory 3 discusses comparison before choosing.')
    assert not assess_definition_output(text, selected=SELECTED)['accepted']


def test_definition_retains_coordinate_and_reference_scope_without_semantic_checks():
    invented = DEFINITION + ' Compared with Memory 999, Memory 3 checks alternatives.'
    result = assess_definition_output(invented, selected=SELECTED)
    assert not result['accepted'] and 'do not match' in result['reason']
    wrong_coordinate = DEFINITION + ' Memory 0 has position=+0.8.'
    result = assess_definition_output(wrong_coordinate, selected=SELECTED)
    assert not result['accepted'] and 'coordinate' in result['reason']
    wrong_side = DEFINITION + ' Memory 0 is on the positive side.'
    result = assess_definition_output(wrong_side, selected=SELECTED)
    assert not result['accepted'] and 'spectrum side' in result['reason']


def test_compact_three_sentence_definition_passes_but_a_fourth_does_not():
    text = ('This probe measures willingness to deliberate. Lower readings choose immediately while higher readings compare alternatives. '
            'The distinction is a preference for evaluation before commitment.')
    assert assess_definition_output(text, selected=SELECTED)['accepted']
    result = assess_definition_output(text + ' It concerns decisional behavior.', selected=SELECTED)
    assert not result['accepted'] and 'three sentences' in result['reason']
    result = assess_definition_output(DEFINITION + ' explanation' * 121, selected=SELECTED)
    assert not result['accepted'] and '120 words' in result['reason']


def test_success_runs_once_and_reports_form_only_provenance():
    selected_before = copy.deepcopy(SELECTED)
    generator = Generator([DEFINITION])
    result = run(generator)
    assert len(generator.calls) == 1
    assert generator.calls[0] == (build_definition_request(EVIDENCE), 384)
    assert result['status'] == 'hypothesis' and result['text'] == DEFINITION
    assert result['version'] == OUTPUT_VERSION and result['structural_completion_only']
    assert result['evidence_sha256'] == hashlib.sha256(EVIDENCE.encode('utf-8')).hexdigest()
    assert result['tokens_generated'] == 30
    assert not result['attempts'][0]['format_correction']
    assert SELECTED == selected_before


def test_format_correction_uses_same_evidence_and_reason_without_failed_prose():
    generator = Generator([RECAP, DEFINITION])
    result = run(generator)
    assert result['status'] == 'hypothesis' and len(generator.calls) == 2
    assert [attempt['status'] for attempt in result['attempts']] == ['unqualified', 'hypothesis']
    assert result['attempts'][1]['format_correction']
    retry = generator.calls[1][0]
    assert retry.startswith(EVIDENCE + '\n\n[Definition request]\n')
    assert '[Format correction]' in retry
    assert result['attempts'][0]['qualification']['reason'] in retry
    assert 'conversation log' not in retry
    assert result['tokens_generated'] == 60


def test_two_failed_attempts_stop_without_a_successful_fallback():
    generator = Generator([RECAP, RECAP])
    result = run(generator)
    assert len(generator.calls) == 2 and len(result['attempts']) == 2
    assert result['status'] == 'unqualified' and not result['qualification']['accepted']
    assert result['text'] == RECAP


@pytest.mark.parametrize('count,finish', [(64, 'eos'), (12, 'length'), (12, 'max_tokens')])
def test_token_limit_stops_without_retry_or_qualification(count, finish):
    generator = Generator([DEFINITION], counts=[count], finish=[finish])
    result = run(generator, max_tokens=64)
    assert len(generator.calls) == 1 and result['status'] == 'incomplete'
    assert not result['qualification']['accepted']
    assert result['text'] == DEFINITION


@pytest.mark.parametrize('returned', [
    DEFINITION, (DEFINITION,), (DEFINITION, {}, 0),
    (None, {'tokens_generated': 1}), (DEFINITION, None),
    (DEFINITION, {}), (DEFINITION, {'tokens_generated': True}),
    (DEFINITION, {'tokens_generated': -1}),
])
def test_malformed_callback_result_raises_for_caller_cleanup(returned):
    with pytest.raises(ValueError):
        run(lambda request, budget: returned)


def test_callback_exception_propagates_to_the_hook_owner():
    def failing(request, budget):
        raise RuntimeError('generation failed')
    with pytest.raises(RuntimeError, match='generation failed'):
        run(failing)


def test_profile_request_factory_and_observer_receive_actual_attempts():
    generator = Generator([RECAP, DEFINITION])
    requested, observed = [], []
    def request_factory(evidence, feedback, index):
        requested.append((evidence, feedback, index))
        return evidence if index == 0 else build_definition_request(evidence, feedback)
    result = run(generator, request_factory=request_factory, on_attempt=observed.append)
    assert generator.calls[0][0] == EVIDENCE
    assert requested[0] == (EVIDENCE, None, 0)
    assert requested[1][1] == observed[0]['qualification']['reason']
    assert observed == result['attempts']
    assert observed[0]['request_sha256'] == hashlib.sha256(EVIDENCE.encode()).hexdigest()
    assert observed[0]['status'] == 'unqualified' and observed[1]['status'] == 'hypothesis'


def test_profile_observer_sees_token_limited_attempt_as_incomplete():
    observed = []
    result = run(Generator([DEFINITION], counts=[384]), on_attempt=observed.append)
    assert observed[0]['status'] == result['status'] == 'incomplete'
    assert not observed[0]['qualification']['accepted']


@pytest.mark.parametrize('option,value', [('request_factory', 1), ('on_attempt', 1),
                                         ('retry_incomplete', True),
                                         ('request_factory', lambda *_args: '')])
def test_invalid_profile_callbacks_cannot_generate(option, value):
    generator = Generator([DEFINITION])
    with pytest.raises(ValueError):
        run(generator, **{option: value})
    assert not generator.calls


def test_incomplete_replay_can_fall_back_after_actual_failure_is_observed():
    generator = Generator([DEFINITION, DEFINITION], counts=[384, 30])
    observed = []
    def factory(evidence, feedback, index):
        return evidence if index == 0 else build_definition_request(evidence, feedback)
    def retry(attempt):
        assert observed[-1] is attempt and attempt['status'] == 'incomplete'
        assert not attempt['qualification']['accepted']
        return True
    result = run(generator, request_factory=factory, on_attempt=observed.append, retry_incomplete=retry)
    assert result['status'] == 'hypothesis'
    assert [attempt['status'] for attempt in observed] == ['incomplete', 'hypothesis']
    assert len(generator.calls) == 2 and result['tokens_generated'] == 414
    assert generator.calls[0][0] == EVIDENCE
    assert '[Definition request]' in generator.calls[1][0] and '[Format correction]' in generator.calls[1][0]


def test_incomplete_fallback_is_bounded_even_when_policy_always_requests_it():
    generator = Generator([DEFINITION, DEFINITION], counts=[384, 384])
    result = run(generator, retry_incomplete=lambda _attempt: True)
    assert result['status'] == 'incomplete' and len(generator.calls) == 2
    assert all(not attempt['qualification']['accepted'] for attempt in result['attempts'])


def test_incomplete_policy_requires_a_boolean_decision():
    generator = Generator([DEFINITION], counts=[384])
    with pytest.raises(ValueError, match='return a boolean'):
        run(generator, retry_incomplete=lambda _attempt: 'yes')
    assert len(generator.calls) == 1


@pytest.mark.parametrize('budget', [0, -1, True, 1.5, float('nan')])
def test_invalid_budget_does_not_generate(budget):
    generator = Generator([DEFINITION])
    with pytest.raises(ValueError):
        run(generator, max_tokens=budget)
    assert generator.calls == []


def test_invalid_evidence_or_scope_does_not_generate():
    with pytest.raises(ValueError, match='nonempty measured evidence'):
        build_definition_request('  ')
    generator = Generator([DEFINITION])
    with pytest.raises(ValueError, match='varying measured examples'):
        run_definition_output(EVIDENCE, [(0, {'reading': 0})], generator)
    assert generator.calls == []
