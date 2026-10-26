"""Native assistant prefills and a candidate interface for boundary signals.

Prefills change only a temporary generation context. Boundary scores must be
supplied by a caller; this module contains no trained completion detector.
Lexical guards and score thresholds do not establish semantic completion.
"""
import hashlib
import math
import re


DEFAULT_DEFINITION_PREFIX = 'Within these examples, the behavioral distinction is a '
PREFILL_VERSION = 'definition-native-assistant-prefill-candidate-v1'
BOUNDARY_GATE_VERSION = 'definition-hybrid-boundary-interface-candidate-v1'
_CHAT_CONTROL = re.compile(r'<\|[^>\r\n]*\|>|\[/?INST\]|</?s>|<<\/?SYS>>')


def _prefix(prefix):
    if prefix is None:
        return ''
    if not isinstance(prefix, str):
        raise ValueError('Assistant continuation prefix must be a string or None')
    return prefix


def _special_tokens(tokenizer):
    tokens = set()
    for value in getattr(tokenizer, 'all_special_tokens', ()) or ():
        if value:
            tokens.add(str(value))
    for name in ('bos_token', 'eos_token', 'sep_token', 'pad_token'):
        value = getattr(tokenizer, name, None)
        if value:
            tokens.add(str(value))
    return tokens


def _token_ids(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False, return_tensors='pt')
    try:
        values = encoded['input_ids']
        if hasattr(values, 'tolist'):
            values = values.tolist()
        if values and isinstance(values[0], (list, tuple)):
            if len(values) != 1:
                raise ValueError('Expected exactly one tokenized assistant context')
            values = values[0]
        values = list(values)
        if not values or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError('Expected nonempty integer token IDs')
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Assistant continuation requires one valid tokenized context') from exc
    return values


def prepare_assistant_continuation(model, evidence, prefix=DEFAULT_DEFINITION_PREFIX):
    """Render one native user turn followed by an unclosed assistant prefill.

    The returned string is already formatted and must be tokenized without a
    second chat wrapper. ``prefix_token_count`` counts the changed token suffix
    in the full context, including any retokenized tail of the native header.
    ``prefix_token_delta`` is the difference between full and header lengths.
    No standalone prefix tokenization is used to infer these boundary counts.
    """
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Assistant continuation requires nonempty evidence text')
    prefix = _prefix(prefix)
    tokenizer = model.tok
    if _CHAT_CONTROL.search(prefix) or any(token in prefix for token in _special_tokens(tokenizer)):
        raise ValueError('Assistant continuation prefix must not contain chat control tokens')
    native = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': evidence}], tokenize=False, add_generation_prompt=True)
    if not isinstance(native, str) or not native:
        raise ValueError('Assistant continuation requires a native text chat template')
    prompt = native + prefix
    header_ids = _token_ids(tokenizer, native)
    prompt_ids = header_ids if not prefix else _token_ids(tokenizer, prompt)
    common = 0
    for header_token, prompt_token in zip(header_ids, prompt_ids):
        if header_token != prompt_token:
            break
        common += 1
    metadata = {
        'version': PREFILL_VERSION, 'assistant_prefix': prefix,
        'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest(),
        'prompt_token_count': len(prompt_ids), 'native_header_token_count': len(header_ids),
        'prefix_token_count': len(prompt_ids) - common,
        'prefix_token_delta': len(prompt_ids) - len(header_ids),
        'retokenized_header_tokens': len(header_ids) - common,
        'pre_formatted': True, 'unclosed_assistant': True,
        'candidate': True, 'semantic_truth_checked': False,
    }
    return prompt, metadata


def finish_assistant_continuation(prefix, continuation):
    """Restore the literal prefill once to a decoded suffix that excludes it."""
    prefix = _prefix(prefix)
    if not isinstance(continuation, str):
        raise ValueError('Assistant continuation must be a string')
    if prefix and continuation.startswith(prefix):
        raise ValueError('Decoded continuation already includes the assistant prefix')
    return prefix + continuation


def _probability(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be finite and between zero and one') from exc
    if isinstance(value, bool) or not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f'{label} must be finite and between zero and one')
    return result


def _integer(value, label, minimum):
    try:
        number = float(value)
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a finite integer >= {minimum}') from exc
    if isinstance(value, bool) or not math.isfinite(number) or result != number or result < minimum:
        raise ValueError(f'{label} must be a finite integer >= {minimum}')
    return result


class HybridBoundaryGate:
    """A supplied-score gate, not a semantic detector or generation scheduler.

    ``generated_text`` must describe only the current generated gap, excluding
    fixed scaffold text. The conservative lexical guard requires an observed
    terminal delimiter; an unfinished word cannot trigger a literal connector.
    The evaluation limit bounds waiting and never supplies success on its own.
    """
    def __init__(self, phases, *, activation_threshold=.8, connector_threshold=.8,
                 max_evaluations=128, min_content_words=2, calibration_id=None):
        if isinstance(phases, (str, bytes)):
            raise ValueError('Boundary phases must be ordered unique nonempty names')
        try:
            phases = tuple(phases)
        except TypeError as exc:
            raise ValueError('Boundary phases must be ordered unique nonempty names') from exc
        if (not phases or any(not isinstance(phase, str) or not phase.strip() for phase in phases) or
                len(set(phases)) != len(phases)):
            raise ValueError('Boundary phases must be ordered unique nonempty names')
        if calibration_id is not None and (not isinstance(calibration_id, str) or not calibration_id.strip()):
            raise ValueError('Boundary calibration identity must be a nonempty string or None')
        self.phases = phases
        self.activation_threshold = _probability(activation_threshold, 'Activation threshold')
        self.connector_threshold = _probability(connector_threshold, 'Connector threshold')
        self.max_evaluations = _integer(max_evaluations, 'Boundary evaluation limit', 1)
        self.min_content_words = _integer(min_content_words, 'Boundary content word minimum', 1)
        self.calibration_id = calibration_id
        self._index = 0
        self._evaluations = {phase: 0 for phase in phases}
        self._events = []
        self._failed_phase = None

    @property
    def current_phase(self):
        return self.phases[self._index] if self._index < len(self.phases) else None

    def _result(self, phase, status, reason, *, trigger=False):
        return {
            'phase': phase, 'status': status, 'reason': reason, 'trigger': trigger,
            'stop': self._failed_phase is not None,
            'evaluations': self._evaluations[phase], 'current_phase': self.current_phase,
            'all_boundaries_triggered': self._index == len(self.phases),
            'candidate': True, 'semantic_completion': False,
            # A caller's provenance label is not evidence that a detector was
            # trained or validated. This candidate interface certifies neither.
            'detector_training_pending': True,
        }

    def evaluate(self, phase, *, activation_score, next_connector_probability,
                 generated_text, connector_queue_remaining=0):
        """Return a one-shot boundary event after both supplied scores pass."""
        if phase not in self._evaluations:
            raise ValueError('Unknown boundary phase')
        activation = _probability(activation_score, 'Activation score')
        probability = _probability(next_connector_probability, 'Next connector probability')
        remaining = _integer(connector_queue_remaining, 'Literal connector queue remainder', 0)
        if not isinstance(generated_text, str):
            raise ValueError('Boundary generated gap text must be a string')
        if self._failed_phase is not None:
            return self._result(phase, 'unqualified', 'Boundary signal timeout; generation must stop.')
        index = self.phases.index(phase)
        if index < self._index:
            return self._result(phase, 'already_triggered', 'This phase already emitted its boundary.')
        if index != self._index:
            raise ValueError(f'Expected boundary phase {self.current_phase}')
        self._evaluations[phase] += 1
        lexical_boundary = bool(generated_text) and (
            generated_text[-1].isspace() or generated_text[-1] in '.!?;:,')
        content = len(re.findall(r'[^\W\d_]+', generated_text, flags=re.UNICODE)) >= self.min_content_words
        ready = (not remaining and lexical_boundary and content and
                 activation >= self.activation_threshold and probability >= self.connector_threshold)
        if ready:
            self._events.append({'phase': phase, 'evaluation': self._evaluations[phase],
                                 'activation_score': activation,
                                 'next_connector_probability': probability})
            self._index += 1
            return self._result(phase, 'triggered', 'Both supplied signals and lexical guards passed.', trigger=True)
        if self._evaluations[phase] >= self.max_evaluations:
            self._failed_phase = phase
            return self._result(phase, 'unqualified', 'Boundary signal timeout; generation must stop.')
        reason = ('Literal connector queue is unfinished.' if remaining else
                  'Generated gap has no observed lexical boundary.' if not lexical_boundary else
                  'Generated gap has insufficient content.' if not content else
                  'Both supplied boundary signals must reach their thresholds.')
        return self._result(phase, 'waiting', reason)

    def trace(self):
        return {
            'version': BOUNDARY_GATE_VERSION, 'phase_order': list(self.phases),
            'current_phase': self.current_phase, 'failed_phase': self._failed_phase,
            'phase_evaluations': dict(self._evaluations),
            'events': [dict(event) for event in self._events],
            'activation_threshold': self.activation_threshold,
            'connector_threshold': self.connector_threshold,
            'max_evaluations': self.max_evaluations,
            'score_calibration_id': self.calibration_id,
            'candidate': True, 'semantic_completion': False,
            'detector_training_pending': True,
            'stop': self._failed_phase is not None,
        }
