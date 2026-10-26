"""Check compact definition shape and measured coordinates, not inferred meaning."""
import re

from invariants.probe_definition_check import _blocks, _selected_scope, assess_definition_attempt
from invariants.probe_definition_pipeline import _coordinate_failure


SCHEMA_CHECK_VERSION = 'compact-operational-definition-structure-v1'


def assess_schema_definition(text, *, selected):
    """Require one compact prose definition with an explicit measured contrast.

    A final dictionary-style sentence may name lower/higher behavior without
    repeating memory citations. Any citations or explicit coordinates it does
    contain must agree with the measured evidence. These checks do not establish
    a correct genus, an invariant, an absence condition, or semantic completion.
    """
    text = str(text or '').strip()
    qualification = dict(assess_definition_attempt(text, selected=selected))
    qualification.update(structural_only=True, semantic_truth_checked=False,
                         check_version=SCHEMA_CHECK_VERSION)
    if not qualification['accepted']:
        return qualification
    blocks = _blocks(text)
    # Independent lists/recaps must not acquire qualification merely because
    # their last paragraph contains a contrast or the prefix supplies "is".
    if len(blocks) != 1 or re.search(r'^\s*(?:[-*+]\s+|\d+[.)]\s+)', text, re.MULTILINE):
        qualification.update(accepted=False, reason='Definition structure requires one connected prose paragraph, not a recap or list.')
        return qualification
    prose = blocks[0]
    words = re.findall(r"\b[\w]+(?:['-][\w]+)*\b", prose)
    if len(words) > 120:
        qualification.update(accepted=False, reason='Definition structure requires a compact statement of at most 120 words.')
        return qualification
    sentences = [part for part in re.split(r'(?<!\d)[.!?](?!\d)\s+(?=[A-Z])', prose) if part.strip()]
    if len(sentences) > 2:
        qualification.update(accepted=False, reason='Definition structure requires at most two connected sentences.')
        return qualification
    scope = _selected_scope(selected)
    failure = _coordinate_failure(text, scope)
    if failure:
        qualification.update(accepted=False, reason=failure)
    return qualification
