"""Compact shape cannot turn a recap or wrong placement into a definition."""
import pytest

from invariants.definition_schema_check import assess_schema_definition


ROWS = [(0, {'reading': -.8}), (1, {'reading': -.2}),
        (2, {'reading': .3}), (3, {'reading': .8})]
DEFINITION = ('The measured distinction is a reporting pattern characterized by responsiveness to new evidence: '
              'lower readings retain an earlier estimate, whereas higher readings revise the report using a new measurement.')


def test_compact_end_contrast_is_a_structural_hypothesis_without_citation_forcing():
    result = assess_schema_definition(DEFINITION, selected=ROWS)
    assert result['accepted']
    assert result['structural_only'] and not result['semantic_truth_checked']


@pytest.mark.parametrize('text', [
    'Memory 0 retains 12.\n\nMemory 3 adopts 41.\n\n' + DEFINITION,
    '- ' + DEFINITION,
    DEFINITION + ' This is a report. This is a pattern.',
    DEFINITION + ' ' + 'additional ' * 121,
    'The measured distinction is a reporting pattern.',
])
def test_recap_or_template_opening_does_not_suffice(text):
    assert not assess_schema_definition(text, selected=ROWS)['accepted']


@pytest.mark.parametrize('ending', [
    'Memory 0 is on the positive side.',
    'Memory 3 has position=-0.8000.',
    'Memory 17 revises its report.',
])
def test_explicit_wrong_evidence_cannot_pass_compactness(ending):
    assert not assess_schema_definition(DEFINITION + ' ' + ending, selected=ROWS)['accepted']


def test_all_positive_measurements_still_use_relative_ends():
    shifted = [(index, {'reading': row['reading'] + 2.}) for index, row in ROWS]
    assert assess_schema_definition(DEFINITION, selected=shifted)['accepted']


def test_correct_shape_does_not_claim_semantic_validation():
    wrong = DEFINITION.replace('responsiveness to new evidence', 'confidence in an unmeasured emotion')
    result = assess_schema_definition(wrong, selected=ROWS)
    assert result['accepted'] and result['semantic_truth_checked'] is False
