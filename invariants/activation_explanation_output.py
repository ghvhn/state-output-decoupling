"""Qualify activation-only prose by its form, never by a probe's true meaning.

This checker receives no probe identifier, seed definition, or measured examples.
It neither generates another attempt nor puts its assessment into model context.
A passing explanation remains an unverified semantic candidate.
"""

import json
import math
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_VERSION = 'activation-explanation-structural-output-v1'

_CONTROLS = re.compile(
    r'<<|<\||```|\[/?INST\]|</?(?:tool_call|function_call)\b|<function=|'
    r'"(?:tool_calls?|function_call)"\s*:|'
    r'^\s*(?:system|user|assistant|tool)\s*:', re.IGNORECASE | re.MULTILINE)
_CHAT = re.compile(
    r'^(?:hello|hi|hey|greetings|good\s+(?:morning|afternoon|evening))\b|'
    r'^(?:thanks|thank\s+you|welcome)\b|'
    r'^(?:nice|pleased)\s+to\s+meet\b|'
    r'(?:^|[.!?]\s+)(?:(?:at\s+this\s+time|right\s+now|currently)[,]?\s+)?'
    r'I(?:\s+am|[\'’]m)\s+(?:happy|glad|pleased)\s+to\s+(?:chat|help|talk)\b|'
    r'\bwhat\s+would\s+you\s+like\s+to\s+(?:talk|discuss|chat)\b|'
    r'\bhow\s+can\s+I\s+(?:help|assist)\b|'
    r'\bhow\s+(?:are|have|do)\s+you\b|'
    r'^(?:what|who|where|when|why|how)\b[^\n]*\?\s*$|'
    r'\byou(?:\s+are|[\'’]re)\s+welcome\b|'
    r'^(?:I(?:\s+am|[\'’]m)\s+(?:here|ready|available)\s+to\s+'
    r'(?:help|assist|chat|talk|discuss)|'
    r'I\s+(?:can|will|would|want\s+to)\s+(?:help|assist|chat|talk|discuss)|'
    r'let[\'’]s\s+(?:chat|talk|discuss))\b', re.IGNORECASE)
_ADVICE = re.compile(
    r'\b(?:you|we)\s+(?:should|must|need\s+to|ought\s+to)\b|'
    r'\byou\s+(?:can|could|might)\s+(?:try|use|consider|avoid|remember)\b|'
    r'\b(?:recommend|suggest)\s+(?:that\s+)?(?:you|we)\b|'
    r'(?:^|[.!?]\s+)(?:use(?!\s+of\b)|try|consider|avoid|ensure|remember|make\s+sure|do\s+not)\b',
    re.IGNORECASE)
_RECAP = re.compile(
    r'\b(?:conversation\s+(?:log|history|summary)|(?:provided|given)\s+text|'
    r'(?:summary|recap|transcript|collection|archive|record|list)\s+of\s+'
    r'(?:(?:our|the|these|previous|your|individual|recorded)\s+)*'
    r'(?:conversations?|chats?|memories|examples|replies)|'
    r'(?:earlier|previously)\s+(?:you|we|I)\s+(?:said|discussed|talked)|'
    r'(?:these|the|individual)\s+(?:memories|examples)\s+'
    r'(?:are|contain|show|describe|discuss)|'
    r'memory\s*#?\s*\d+\s+(?:contains?|shows?|describes?|discusses?|records?|says?))\b',
    re.IGNORECASE)
_END = re.compile(
    r'\b(?:lower|higher|low|high|negative|positive)[-\s]+'
    r'(?:readings?|positions?|scores?|values?|ends?|sides?|poles?)\b', re.IGNORECASE)
_DEFINITION_RELATION = re.compile(
    r'\b(?:is|are|means?|refers?\s+to|represents?|describes?|denotes?|signifies?|'
    r'consists?\s+of|(?:is\s+)?defined\s+as|(?:is\s+)?characterized\s+by|'
    r'defines?|measures?|captures?|reflects?|indicates?|expresses?|tracks?|'
    r'indexes?|estimates?|encodes?)\s+', re.IGNORECASE)
_NOUN_RELATION = re.compile(
    # A noun phrase can define a general property without a copula or a
    # particular opening: "willingness to revise" or "degree to which ...".
    r'^(?:(?:a|an|the)\s+)?(?:[\w\'’-]+\s+){1,6}'
    r'(?:to\s+which|in\s+which|with\s+which|by\s+which|'
    r'to|of|for|toward|towards|between)\s+', re.IGNORECASE)
_HOW_RELATION = re.compile(r'^(?:how|whether)\s+', re.IGNORECASE)
_WORDS = re.compile(r"\b[\w]+(?:['’-][\w]+)*\b")
_FUNCTION_WORDS = frozenset((
    'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with', 'this',
    'that', 'it', 'is', 'are', 'be', 'being',
))


def _integer(value, *, positive, label):
    try:
        number, numeric = int(value), float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a finite {"positive" if positive else "nonnegative"} integer') from exc
    if (isinstance(value, bool) or not math.isfinite(numeric) or numeric != number
            or number < (1 if positive else 0)):
        raise ValueError(f'{label} must be a finite {"positive" if positive else "nonnegative"} integer')
    return number


def _has_definition_relation(text):
    # Endpoint contrast alone does not say which general property is proposed.
    head = _END.split(text, maxsplit=1)[0].strip()
    for match in _DEFINITION_RELATION.finditer(head):
        before = _WORDS.findall(head[:match.start()])
        after = [word for word in _WORDS.findall(head[match.end():])
                 if word.lower() not in _FUNCTION_WORDS]
        if before and after:
            return True
    match = _NOUN_RELATION.match(head) or _HOW_RELATION.match(head)
    return bool(match and any(word.lower() not in _FUNCTION_WORDS
                              for word in _WORDS.findall(head[match.end():])))


def assess_activation_explanation(text, telemetry, max_tokens):
    """Check one existing output's kind, without retrying or checking meaning.

    Token telemetry is mandatory so a truncated response cannot silently pass.
    A complete, definition-shaped response can pass even when its proposed
    meaning is wrong; the checker has no semantic oracle or measured evidence.
    """
    budget = _integer(max_tokens, positive=True, label='Activation explanation token budget')
    if not isinstance(telemetry, Mapping) or 'tokens_generated' not in telemetry:
        raise ValueError('Activation explanation token telemetry requires tokens_generated')
    count = _integer(telemetry['tokens_generated'], positive=False,
                     label='Activation explanation tokens_generated')
    if text is None:
        text = ''
    if not isinstance(text, str):
        raise ValueError('Activation explanation text must be a string')
    text = text.strip()
    result = {
        'accepted': False, 'reason': 'No generated activation explanation.',
        'structural_only': True, 'semantic_truth_checked': False,
        'status': 'unqualified', 'tokens_generated': count,
        'check_version': OUTPUT_VERSION,
    }
    if count >= budget or telemetry.get('finish_reason') in ('length', 'max_tokens', 'max_new_tokens'):
        result.update(status='incomplete', reason='Activation explanation reached its generation limit.')
        return result
    if not text:
        return result
    if _CONTROLS.search(text):
        result['reason'] = 'An activation explanation must not contain tool calls, code blocks, or chat controls.'
    elif _CHAT.search(text):
        result['reason'] = 'A chat opening or greeting does not define a general property.'
    elif _RECAP.search(text):
        result['reason'] = 'Define a general property instead of recounting memories or the conversation.'
    elif _ADVICE.search(text):
        result['reason'] = 'Describe a general property or rule rather than offer advice.'
    elif not _has_definition_relation(text):
        result['reason'] = 'State a definition or general property relation, not only a label or endpoint contrast.'
    else:
        result.update(accepted=True, status='candidate',
                      reason='Definition-shaped prose passes; its proposed meaning remains unverified.')
    return result


def record_activation_explanation(matrix, attempt):
    """Append a dedicated operator trace for this owner, never model context.

    The caller owns any write-error diagnostic. Owners without a persistence
    path receive no log. This helper does not touch behavior training records,
    conversation records, or prompt construction.
    """
    owner_path = (getattr(matrix, 'path', None)
                  or getattr(getattr(matrix, 'probe_behavior', None), 'path', None))
    if not owner_path:
        return None
    if not isinstance(attempt, dict):
        raise ValueError('An activation explanation trace requires an attempt dict')
    path = Path(owner_path).with_suffix('.activation-explain.jsonl')
    row = dict(attempt)
    row['timestamp'] = datetime.now(timezone.utc).isoformat()
    serialized = json.dumps(row, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(serialized + '\n')
    return path
