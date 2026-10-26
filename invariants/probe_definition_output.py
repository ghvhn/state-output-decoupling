"""Request definition-shaped output without certifying the proposed meaning.

The operator's request supplies an output kind, not a probe's semantic seed.
Only original measured observations are evidence; a format correction never
promotes the rejected response into another observation.
"""
import hashlib
import re

from invariants.probe_definition_check import (
    _blocks, _selected_scope, assess_definition_attempt,
)
from invariants.probe_definition_pipeline import (
    _coordinate_failure, _generated_tokens, _positive_integer,
)


OUTPUT_VERSION = 'definition-scoped-output-kind-v1'
OUTPUT_PREFIX = 'This probe measures '

_REQUEST = (
    'Define the general behavioral property or rule that this probe may measure. '
    'Infer your own interpretation from the measured examples; its meaning may be mistaken. '
    'Give a definition, not a summary of the examples: state what the property means '
    'and how lower versus higher readings express it. '
    'Write one compact prose paragraph, at most 120 words and three sentences, '
    'continuing the assistant prefix "This probe measures ". '
    'Describe a general tendency, relationship, or rule that could apply to other replies. '
    'Do not list or recount individual memories, summarize the conversation, '
    'offer advice, describe the measuring apparatus, emit tool calls, or add a heading. '
    'Lower and higher refer to measured order; their signs do not establish emotional valence. '
    'The request specifies only the kind of output; no true meaning or seed definition is supplied.'
)
_LIST_OR_HEADING = re.compile(r'^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s*)', re.MULTILINE)
_END = re.compile(
    r'\b(?:lower|higher)(?:[-\s]+(?:readings?|positions?|scores?|values?|ends?|sides?|poles?))?\b|'
    r'\b(?:low|high)[-\s]+(?:readings?|positions?|scores?|values?|ends?|sides?|poles?)\b|'
    r'\b(?:negative|positive)[-\s]+(?:readings?|positions?|scores?|values?|ends?|sides?|poles?)\b',
    re.IGNORECASE)
_MEASURES = re.compile(
    r'^\s*(?:(?:this|the)\s+)?probe\s+(?:measures?|detects?|tracks?|reflects?|captures?|indexes?|estimates?)\s+',
    re.IGNORECASE)
_PREDICATE = re.compile(
    r'\b(?:is|means|refers\s+to|is\s+defined\s+as|defined\s+as|characterized\s+by|consists\s+of)\s+\S',
    re.IGNORECASE)
_GENERAL_RELATION = re.compile(
    # These are grammatical relations, not a vocabulary of permitted traits.
    # "how firmly ...", "willingness to ...", and "firmness of ..." can all
    # define a property without agreeing with its real meaning.
    r'^(?:how|whether)\s+\S|'
    r'\b(?:to\s+which|in\s+which|with\s+which|by\s+which|that|whereby)\s+\S|'
    r'\b(?:to|of|for|toward|towards|between|in|with)\s+\S|'
    r'[,;:]\s*(?:a|an|the)\s+\S',
    re.IGNORECASE)
_RECAP = re.compile(
    r'\b(?:conversation\s+(?:log|history)|(?:provided|given)\s+text|'
    r'(?:these|the|individual)\s+memories\s+(?:are|contain|show|describe|discuss)|'
    r'(?:these|the)\s+examples\s+(?:are|contain|show|describe|discuss)|'
    r'memory\s*#?\s*\d+\s+(?:contains?|shows?|describes?|discusses?|records?|says?))\b',
    re.IGNORECASE)
_ARTIFACT_OBJECT = re.compile(
    r'^\s*(?:(?:a|an|the)\s+)?(?:'
    r'(?:collection|log|recap|summary|archive|transcript|record|list|set)\s+of\s+'
    r'(?:(?:individual|measured|recorded|these|the)\s+)*(?:memories|replies|conversations|examples)\b|'
    r'(?:\d+[-\s]+point[-\s]+)?scale\s+of\s+'
    r'(?:memory|conversation|reply|example)[-\s]+(?:groupings?|groups?|categories)\b)',
    re.IGNORECASE)
_ADVICE = re.compile(
    r'\b(?:you|we)\s+(?:should|must|need\s+to|ought\s+to)\b|'
    r'\b(?:recommend|suggest)\s+(?:that\s+)?(?:you|we)\b|'
    r'(?:^|[.!?]\s+)(?:use|try|consider|avoid|ensure|remember|make\s+sure|do\s+not)\b',
    re.IGNORECASE)


def build_definition_request(evidence, feedback=None):
    """Keep the exact evidence and append a scoped output-kind instruction."""
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Definition output requires nonempty measured evidence text')
    if feedback is not None and (not isinstance(feedback, str) or not feedback.strip()):
        raise ValueError('Definition format feedback must be a nonempty string')
    request = evidence + '\n\n[Definition request]\n' + _REQUEST
    if feedback is not None:
        # A short checker reason supplies format feedback; the prior generated
        # prose is deliberately absent, including any invented interpretation.
        reason = ' '.join(feedback.split())[:240]
        request += ('\n\n[Format correction]\nThe previous attempt did not meet the output contract: '
                    + reason + '\nWrite a fresh definition using the original evidence above.')
    return request


def _has_definition_relation(text):
    """Recognize a general predicate before a description of the spectrum ends."""
    head = _END.split(text, maxsplit=1)[0].strip()
    match = _MEASURES.match(head)
    if match:
        content = head[match.end():].strip()
        # The fixed assistant prefix alone cannot supply a definition.
        return bool(content and (_GENERAL_RELATION.search(content) or _PREDICATE.search(content)))
    return bool(_PREDICATE.search(head))


def assess_definition_output(text, *, selected):
    """Check definition form and measured coordinates, never semantic accuracy."""
    text = str(text or '').strip()
    qualification = {
        'accepted': False, 'reason': 'No generated definition.', 'contrast': '',
        'structural_only': True, 'semantic_truth_checked': False,
        'check_version': OUTPUT_VERSION,
    }

    def reject(reason):
        qualification['reason'] = reason
        return qualification

    if not text:
        return qualification
    if '<<' in text or '```' in text or '<|' in text:
        return reject('A definition must not contain tool calls, code blocks, or chat controls.')
    if _LIST_OR_HEADING.search(text) or len(_blocks(text)) != 1:
        return reject('A definition must be one connected prose paragraph, without a list or heading.')
    words = re.findall(r"\b[\w]+(?:['-][\w]+)*\b", text)
    if len(words) > 120:
        return reject('A definition must contain at most 120 words.')
    sentences = [part for part in re.split(r'(?<!\d)[.!?](?!\d)\s+(?=[A-Z])', text) if part.strip()]
    if len(sentences) > 3:
        return reject('A definition must contain at most three sentences.')
    if _RECAP.search(text):
        return reject('Define a general behavioral property instead of recounting memories or the conversation.')
    measured = _MEASURES.match(text)
    if measured and _ARTIFACT_OBJECT.match(text[measured.end():]):
        return reject('Define a behavioral property or rule, not a collection, recap, or grouping of the evidence.')
    if _ADVICE.search(text):
        return reject('A definition must describe a property or rule rather than offer advice.')
    if not _has_definition_relation(text):
        return reject('State what a general behavioral property or rule means, not only a contrast between readings.')
    structure = assess_definition_attempt(text, selected=selected)
    if not structure['accepted']:
        return reject(structure['reason'])
    scope = _selected_scope(selected)
    if not scope:
        return reject('A definition requires supplied measured examples.')
    failure = _coordinate_failure(text, scope)
    if failure:
        return reject(failure)
    qualification.update(
        accepted=True,
        reason='Definition form and measured coordinate scope pass; the proposed meaning remains a hypothesis.',
        contrast=text,
    )
    return qualification


def run_definition_output(evidence, selected, generate_attempt, *, max_tokens=384,
                          request_factory=None, on_attempt=None, retry_incomplete=None):
    """Generate a definition with at most one correction or explicit fallback.

    The callback owns native formatting, the assistant prefix, activation hooks,
    and cleanup. It receives ``(request, budget)`` and returns full prefixed text
    and token telemetry. Callback failures propagate for the shell's cleanup.
    An optional request factory selects a saved recipe or learned replay context;
    the attempt observer receives the actual output-kind assessment before any
    retry. An optional incomplete-output policy can request explicit fallback
    after a truncated speculative replay. Explicit truncation stops by default.
    Their histories are not part of the generation request.
    """
    build_definition_request(evidence)
    budget = _positive_integer(max_tokens, 'Definition output token budget')
    selected = list(selected)
    scope = _selected_scope(selected)
    if not scope or not scope['varying'] or not scope['lower'] or not scope['upper']:
        raise ValueError('Definition output requires varying measured examples at both ordered ends')
    if not callable(generate_attempt):
        raise ValueError('Definition output requires a generation callback')
    if request_factory is not None and not callable(request_factory):
        raise ValueError('Definition request factory must be callable')
    if on_attempt is not None and not callable(on_attempt):
        raise ValueError('Definition attempt observer must be callable')
    if retry_incomplete is not None and not callable(retry_incomplete):
        raise ValueError('Definition incomplete-output policy must be callable')
    result = {
        'text': '', 'status': 'unqualified', 'qualification': {}, 'attempts': [],
        'tokens_generated': 0, 'version': OUTPUT_VERSION,
        'structural_completion_only': True,
        'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest(),
    }
    feedback = None
    for index in range(2):
        request = (build_definition_request(evidence, feedback) if request_factory is None else
                   request_factory(evidence, feedback, index))
        if not isinstance(request, str) or not request.strip():
            raise ValueError('Definition request factory must return nonempty text')
        generated = generate_attempt(request, budget)
        if not isinstance(generated, tuple) or len(generated) != 2:
            raise ValueError('Definition output callback must return text and telemetry')
        text, telemetry = generated
        if not isinstance(text, str) or not isinstance(telemetry, dict):
            raise ValueError('Definition output callback must return string text and telemetry mapping')
        text = text.strip()
        count = _generated_tokens(telemetry)
        qualification = assess_definition_output(text, selected=selected)
        truncated = count >= budget or telemetry.get('finish_reason') in ('length', 'max_tokens')
        if truncated:
            qualification = dict(qualification, accepted=False,
                                 reason='Definition reached its token limit; output is incomplete.')
        status = 'incomplete' if truncated else 'hypothesis' if qualification['accepted'] else 'unqualified'
        attempt = {
            'attempt': index + 1, 'text': text, 'status': status,
            'qualification': qualification, 'tokens_generated': count, 'budget': budget,
            'format_correction': feedback is not None,
            'request_sha256': hashlib.sha256(request.encode('utf-8')).hexdigest(),
        }
        result['attempts'].append(attempt)
        result.update(text=text, status=status, qualification=qualification)
        result['tokens_generated'] += count
        if on_attempt is not None:
            on_attempt(attempt)
        retry_truncated = False
        if status == 'incomplete' and index == 0 and retry_incomplete is not None:
            retry_truncated = retry_incomplete(attempt)
            if type(retry_truncated) is not bool:
                raise ValueError('Definition incomplete-output policy must return a boolean')
        if status != 'unqualified' and not retry_truncated:
            return result
        feedback = qualification['reason']
    return result
