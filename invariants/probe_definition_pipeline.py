"""Bounded definition stages that advance only after structural gates pass.

Measured evidence is kept unchanged. A previous accepted output is a candidate
artifact, never an additional measured memory. Gates check stated structure and
reliably parsed coordinates; they do not establish semantic or causal truth.
"""
import hashlib
import math
import re

from invariants.probe_definition_check import (
    _blocks, _memory_references, _selected_scope, assess_definition_attempt,
)
from invariants.probe_definition_steps import STEP_ORDER


PIPELINE_VERSION = 'definition-gated-candidate-stages-v1'
_REFERENCE = re.compile(r'\bmemory\s*#?\s*(\d+)\b', re.IGNORECASE)
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_COORDINATE = re.compile(
    rf'\b(?:position|(?:centered\s+)?reading)\s*(?:=|:|is|of|at)\s*({_NUMBER})', re.IGNORECASE)
_SIDE = re.compile(
    r'^\s*(?:[\[(,]\s*)?(?:(?:is|lies|sits|belongs|represents|has|with|on|at|from)'
    r'\s+(?:(?:on|at|to)\s+)?)?(?:(?:the|a)\s+)?'
    r'(negative|positive|lower|low|higher|high)[-\s]+'
    r'(?:sides?|ends?|readings?|positions?|scores?|values?|poles?)\b', re.IGNORECASE)
_PREFIX_SIDE = re.compile(
    r'\b(negative|positive|lower|low|higher|high)[-\s]+'
    r'(?:sides?|ends?|readings?|positions?|scores?|values?|poles?)\s*[:\-]?\s*$', re.IGNORECASE)
_CLAUSE_STOP = re.compile(r'[;!?]|(?<!\d)\.(?!\d)')


def _positive_integer(value, label):
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a positive finite integer') from exc
    if isinstance(value, bool) or not math.isfinite(numeric) or numeric != result or result <= 0:
        raise ValueError(f'{label} must be a positive finite integer')
    return result


def _validated_gains(values):
    if not isinstance(values, dict) or set(values) != set(STEP_ORDER):
        raise ValueError('Definition pipeline gains must contain all four named steps')
    result = {}
    for step in STEP_ORDER:
        try:
            gain = float(values[step])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'Definition pipeline gain for {step} must be finite and between -1 and 1') from exc
        if not math.isfinite(gain) or abs(gain) > 1:
            raise ValueError(f'Definition pipeline gain for {step} must be finite and between -1 and 1')
        result[step] = gain
    return result


def _coordinate_failure(text, scope):
    """Check narrow explicit per-memory placements without guessing valence.

    Direct side modifiers and named position/reading numbers are checked.
    Descriptions such as "negative feelings" are not coordinate assignments.
    Ambiguous prose, grouped assignments, and hypothetical placements remain
    unverified instead of receiving an invented interpretation.
    """
    for block in _blocks(text):
        matches = list(_REFERENCE.finditer(block))
        for index, match in enumerate(matches):
            memory = int(match.group(1))
            if memory not in scope['readings']:
                return f'Memory {memory} is absent from the supplied measured examples.'
            next_start = matches[index + 1].start() if index + 1 < len(matches) else len(block)
            suffix = block[match.end():next_start]
            stop = _CLAUSE_STOP.search(suffix)
            if stop:
                suffix = suffix[:stop.start()]
            # "Memory 0 and Memory 3 are..." is a grouped claim, not a
            # reliably assigned coordinate for either individual reference.
            if re.match(r'^\s*(?:,|and|&)\s*(?:memory\s*)?\d*\s*$', suffix, re.IGNORECASE):
                continue
            for coordinate in _COORDINATE.finditer(suffix):
                stated = float(coordinate.group(1))
                measured = scope['readings'][memory]
                tolerance = max(5e-5, abs(measured) * 1e-6)
                if not math.isfinite(stated) or abs(stated - measured) > tolerance:
                    return f'Memory {memory} has a stated coordinate that disagrees with its measured reading.'
            side_match = _SIDE.search(suffix)
            if side_match is None:
                prefix_start = matches[index - 1].end() if index else 0
                prefix = block[prefix_start:match.start()]
                # Only a direct prefix modifier is unambiguous here.
                side_match = _PREFIX_SIDE.search(prefix[-80:])
            if side_match:
                side = side_match.group(1).lower()
                value = scope['readings'][memory]
                agrees = ((side == 'negative' and value < 0) or
                          (side == 'positive' and value > 0) or
                          (side in ('lower', 'low') and memory in scope['lower']) or
                          (side in ('higher', 'high') and memory in scope['upper']))
                if not agrees:
                    return f'Memory {memory} has a stated spectrum side that disagrees with its measured reading.'
    return None


def _gate(step, text, selected, scope):
    qualification = dict(assess_definition_attempt(text, selected=selected))
    qualification['structural_only'] = True
    qualification['semantic_truth_checked'] = False
    if not qualification['accepted']:
        return qualification
    failure = _coordinate_failure(text, scope)
    if failure:
        qualification.update(accepted=False, reason=failure)
        return qualification
    if step in ('compare', 'verify'):
        cited = _memory_references(qualification.get('contrast', ''))
        if not (cited & scope['lower'] and cited & scope['upper']):
            qualification.update(accepted=False,
                                 reason=f'{step.capitalize()} requires a connected contrast citing measured examples from both ends.')
    return qualification


def _generated_tokens(telemetry):
    if 'tokens_generated' not in telemetry:
        raise ValueError('Definition stage token telemetry requires tokens_generated')
    value = telemetry['tokens_generated']
    try:
        count = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Definition stage token telemetry must be a finite nonnegative integer') from exc
    if isinstance(value, bool) or not math.isfinite(numeric) or numeric != count or count < 0:
        raise ValueError('Definition stage token telemetry must be a finite nonnegative integer')
    return count


def run_definition_pipeline(evidence, selected, generate_step, *, step_gains, stage_max_tokens=384):
    """Run up to four separate calls, stopping immediately at a failed gate.

    ``generate_step(step, stage_evidence, budget)`` owns activation hooks and
    returns ``(text, telemetry)``. Its input contains the exact original evidence
    plus only the most recent accepted contrast as an opaque candidate. No stage requests or
    operation names are inserted into that input. Callback exceptions propagate
    so the caller can preserve its normal hook/configuration cleanup.
    """
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Definition pipeline requires nonempty measured evidence text')
    budget = _positive_integer(stage_max_tokens, 'Definition stage token budget')
    gains = _validated_gains(step_gains)
    selected = list(selected)
    scope = _selected_scope(selected)
    if not scope or not scope['varying'] or not scope['lower'] or not scope['upper']:
        raise ValueError('Definition pipeline requires varying measured examples at both ordered ends')
    if not callable(generate_step):
        raise ValueError('Definition pipeline requires a stage generation callback')
    result = {'text': '', 'status': 'unqualified', 'qualification': {}, 'stages': [],
              'failed_step': None, 'completed_steps': [], 'tokens_generated': 0,
              'version': PIPELINE_VERSION, 'structural_completion_only': True,
              'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest()}
    candidate = None
    candidate_id = None
    for step in STEP_ORDER:
        stage_evidence = evidence if candidate is None else evidence + f'\n\n[Candidate {candidate_id}]\n' + candidate
        generated = generate_step(step, stage_evidence, budget)
        if not isinstance(generated, tuple) or len(generated) != 2:
            raise ValueError('Definition stage callback must return text and telemetry')
        text, telemetry = generated
        if not isinstance(text, str) or not isinstance(telemetry, dict):
            raise ValueError('Definition stage callback must return string text and telemetry mapping')
        text = text.strip()
        count = _generated_tokens(telemetry)
        qualification = _gate(step, text, selected, scope)
        reached_limit = count >= budget or telemetry.get('finish_reason') in ('length', 'max_tokens')
        status = 'incomplete' if reached_limit else 'accepted' if qualification['accepted'] else 'unqualified'
        candidate_text = qualification.get('contrast') if status == 'accepted' else None
        next_id = 'c_' + hashlib.sha256(candidate_text.encode('utf-8')).hexdigest()[:24] if candidate_text else None
        record = {'step': step, 'gain': gains[step], 'budget': budget, 'text': text,
                  'status': status, 'tokens_generated': count, 'qualification': qualification,
                  'previous_candidate_id': candidate_id, 'candidate_id': next_id,
                  'candidate_text': candidate_text, 'structural_only': True}
        result['stages'].append(record)
        result.update(text=text, qualification=qualification)
        result['tokens_generated'] += count
        if status != 'accepted':
            result.update(status=status, failed_step=step)
            return result
        result['completed_steps'].append(step)
        candidate, candidate_id = candidate_text, next_id
    result['status'] = 'hypothesis'
    return result
