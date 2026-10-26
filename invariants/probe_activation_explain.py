"""Candidate explanations requested through residual activation arithmetic.

Actual generation receives an exact optional evidence body and a native empty
assistant header. Teacher requests are used only to mint candidate directions;
neither a task instruction nor an explanation prefix enters the runtime input.
The reference contrast is a contextual candidate, not a proven self binding.
"""

from collections.abc import Mapping, MutableMapping
from collections import deque
from contextlib import contextmanager
import hashlib
import math

import torch

from invariants.definition_prefill import PREFILL_VERSION, _token_ids
from invariants.probe_behavior import build_definition_handles
from invariants.probe_definition_operation import (
    OPERATION_VERSION, _CONTEXTUAL_OPERATION_REQUESTS, _OPERATION_CASES,
)
from invariants.probe_definition_schema import _decision_states as _raw_decision_states
from invariants.probe_definition_steps import _digest, _identity
from invariants.probe_definition_task import _ASSEMBLY, _ROUTES, _requested_layers, _unit


ACTIVATION_EXPLAIN_VERSION = 'probe-activation-explain-raw-decision-empty-continuation-v2'
REFERENCE_VERSION = 'probe-reference-own-activation-raw-decision-candidate-v2'
CAPTURE_VERSION = 'raw-decoder-block-output-native-assistant-decision-v1'
EXPLAIN_GAINS = {'subject': .02, 'operation': .12, 'reference': .04}
_REFERENCE_REQUESTS = (
    (
        "In one concise sentence, define this probe's lower-versus-higher distinction as a property of your own activations.",
        "In one concise sentence, define this probe's lower-versus-higher distinction as a property of externally observed behavior.",
    ),
    (
        'In one concise sentence, explain the general distinction between lower and higher readings in your own activations.',
        'In one concise sentence, explain the general distinction between lower and higher readings in externally observed behavior.',
    ),
)
_SCRATCH_OVERRIDES = {
    'base_model_only': True,
    'cache_enabled': False,
    'cache_write_enabled': False,
    'cognitive_cache': None,
    'use_expert_vectors': False,
    'emergent_experts_enabled': False,
    'synthesis_enabled': False,
    'force_synthesis': False,
    'max_routing_events': 0,
    'routing_depth': 0,
    'max_synthesis_events': 0,
    'max_synthesis_steps': 0,
    'max_loops': 0,
    'max_tool_calls': 0,
    'lookahead_processor': None,
    'learned_concept_context': None,
    'deterministic_scaffolds_enabled': False,
    'model_scaffold_tool_enabled': False,
    'provide_time_context': False,
    'continuous_urgency_injection': False,
    'committee': None,
    '_generation_deadline': None,
    '_generation_budget_sec': None,
    '_synthesis_events_used': 0,
}


def _evidence(value):
    if not isinstance(value, str):
        raise ValueError('Activation explanation evidence must be a string')
    return value


def _gain(value, name, *, signed):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'Activation explanation {name} must be a finite bounded gain') from exc
    lower = -1 if signed else 0
    if isinstance(value, bool) or not math.isfinite(result) or not lower <= result <= 1:
        raise ValueError(f'Activation explanation {name} must be finite and between {lower} and 1')
    return result


def _budget(value):
    try:
        result = int(value)
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Activation explanation token budget must be a positive finite integer') from exc
    if isinstance(value, bool) or not math.isfinite(number) or number != result or result <= 0:
        raise ValueError('Activation explanation token budget must be a positive finite integer')
    return result


def _validated_direction(direction, layers, name, width=None):
    if (not isinstance(direction, Mapping) or set(direction) != set(layers)
            or any(type(layer) is not int for layer in direction)):
        raise ValueError(f'Activation explanation {name} must cover exactly the requested layers')
    result = {}
    for layer in layers:
        vector = direction[layer]
        if (not isinstance(vector, torch.Tensor) or vector.ndim != 1
                or not vector.is_floating_point() or not vector.numel()):
            raise ValueError(f'Activation explanation {name} needs a floating vector at layer {layer}')
        vector = vector.detach().to(device='cpu', dtype=torch.float32).clone()
        norm = vector.norm()
        if (not torch.isfinite(vector).all() or not torch.isfinite(norm) or norm.item() <= 1e-8
                or (width is not None and vector.numel() != width)):
            raise ValueError(f'Activation explanation {name} has invalid geometry at layer {layer}')
        if width is None:
            width = vector.numel()
        result[layer] = vector
    return result


def build_activation_explanation_prompt(model, evidence=''):
    """Render the exact optional body with no task or assistant-text prefix."""
    evidence = _evidence(evidence)
    prompt = model.tok.apply_chat_template(
        [{'role': 'user', 'content': evidence}], tokenize=False, add_generation_prompt=True)
    if not isinstance(prompt, str) or not prompt:
        raise ValueError('Activation explanation requires a native text chat template')
    count = len(_token_ids(model.tok, prompt))
    return prompt, {
        'version': PREFILL_VERSION, 'assistant_prefix': '',
        'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest(),
        'prompt_token_count': count, 'native_header_token_count': count,
        'prefix_token_count': 0, 'prefix_token_delta': 0, 'retokenized_header_tokens': 0,
        'pre_formatted': True, 'unclosed_assistant': True,
        'candidate': True, 'semantic_truth_checked': False,
    }


@torch.no_grad()
def _mint_decision_contrast(model, layers, name, cases):
    """Average matched raw-block differences at the native assistant header.

    The raw recorder bypasses hidden-state collection and removes capture hooks
    on failure. In particular, the final layer is read before its terminal
    normalization, matching the actual injection site.
    """
    differences = {layer: [] for layer in layers}
    for positive_evidence, positive_request, negative_evidence, negative_request in cases:
        positive = _raw_decision_states(
            model, [{'role': 'user', 'content': positive_evidence + '\n\n' + positive_request}],
            layers, assistant_prefix='')
        negative = _raw_decision_states(
            model, [{'role': 'user', 'content': negative_evidence + '\n\n' + negative_request}],
            layers, assistant_prefix='')
        for layer in layers:
            if (positive[layer].shape != negative[layer].shape or
                    (differences[layer] and positive[layer].shape != differences[layer][0].shape)):
                raise ValueError(f'Activation explanation {name} has incompatible widths at layer {layer}')
            differences[layer].append(_unit(positive[layer] - negative[layer], name, layer))
    return {layer: _unit(torch.stack(values).mean(dim=0), name, layer)
            for layer, values in differences.items()}


@torch.no_grad()
def mint_definition_operation_axis(model, layers, evidence=None):
    """Reuse the define/recap cases while measuring raw decoder-block outputs."""
    requested = _requested_layers(layers)
    if evidence is None:
        cases = _OPERATION_CASES
    else:
        evidence = _evidence(evidence)
        cases = (tuple((evidence, positive, evidence, negative)
                       for positive, negative in _CONTEXTUAL_OPERATION_REQUESTS)
                 if evidence.strip() else _OPERATION_CASES)
    return _mint_decision_contrast(model, requested, 'operation', cases)


@torch.no_grad()
def mint_reference_axis(model, layers, evidence=''):
    """Hold evidence and explain operation fixed while varying own/external scope.

    Both requests end at the native assistant decision header. Two paraphrases
    are averaged per layer. Blank runtime evidence uses generic calibration
    observations only for minting; these never become the runtime user body.
    """
    requested = _requested_layers(layers)
    evidence = _evidence(evidence)
    contexts = (evidence,) if evidence.strip() else (_ROUTES, _ASSEMBLY)
    cases = tuple((context, positive, context, negative)
                  for context in contexts for positive, negative in _REFERENCE_REQUESTS)
    return _mint_decision_contrast(model, requested, 'reference', cases)


def _axis_cache_key(model, layers, evidence, subject=None):
    # This is an invocation-owner memory cache, not a persisted axis bundle.
    # Object identities also prevent sharing axes across separately loaded
    # models that happen to report the same source and tokenizer metadata.
    try:
        identity = _identity(model, layers)
        identity_scope = 'identified_model'
    except (AttributeError, TypeError, ValueError):
        identity = {
            'model_class': type(model.model).__module__ + '.' + type(model.model).__qualname__,
            'model_source': str(getattr(getattr(model.model, 'config', None), '_name_or_path', '')),
            'tokenizer_class': type(model.tok).__module__ + '.' + type(model.tok).__qualname__,
            'chat_template': str(getattr(model.tok, 'chat_template', '')),
            'width': str(getattr(model, 'd_model', '')),
            'layers': list(layers),
        }
        identity_scope = 'runtime_objects'
    subject_fingerprint = None
    if subject is not None:
        validated = _validated_direction(subject, layers, 'cache subject', width=getattr(model, 'd_model', None))
        subject_fingerprint = {
            str(layer): hashlib.sha256(vector.numpy().tobytes()).hexdigest()
            for layer, vector in validated.items()
        }
    key = _digest({
        'version': ACTIVATION_EXPLAIN_VERSION, 'operation_version': OPERATION_VERSION,
        'capture_version': CAPTURE_VERSION,
        'reference_version': REFERENCE_VERSION, 'reference_requests': _REFERENCE_REQUESTS,
        'model_object': id(model.model), 'tokenizer_object': id(model.tok),
        'identity': identity, 'layers': list(layers),
        'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest(),
        'subject_fingerprint': subject_fingerprint,
    })
    return key, identity_scope


def get_or_mint_activation_axes(model, layers, evidence='', axis_cache=None,
                               *, include_operation=True, include_reference=True, subject=None):
    """Return only requested controls, caching exact context and runtime identity."""
    requested = _requested_layers(layers)
    evidence = _evidence(evidence)
    if axis_cache is not None and not isinstance(axis_cache, MutableMapping):
        raise ValueError('Activation explanation axis cache must be a mutable mapping or None')
    if type(include_operation) is not bool or type(include_reference) is not bool:
        raise ValueError('Activation explanation axis selections must be booleans')
    key, identity_scope = _axis_cache_key(model, requested, evidence, subject)
    cached = axis_cache.get(key, {}) if axis_cache is not None else {}
    if not isinstance(cached, Mapping):
        raise ValueError('Activation explanation cached axes must be a mapping')
    selected = []
    if include_operation:
        selected.append('operation')
    if include_reference:
        selected.append('reference')
    axes, hits, additions = {}, {}, {}
    width = getattr(model, 'd_model', None)
    for name in selected:
        hits[name] = name in cached
        if hits[name]:
            directions = cached[name]
        elif name == 'operation':
            directions = mint_definition_operation_axis(
                model, requested, evidence=evidence if evidence.strip() else None)
        else:
            directions = mint_reference_axis(model, requested, evidence=evidence)
        axes[name] = _validated_direction(directions, requested, name, width=width)
        if not hits[name]:
            additions[name] = {layer: vector.clone() for layer, vector in axes[name].items()}
    if axis_cache is not None and additions:
        # Commit only a complete validated requested set after every mint works.
        axis_cache[key] = {**cached, **additions}
    return axes, {'axis_cache_key': key, 'axis_cache_identity_scope': identity_scope,
                  'axes_cached': hits, 'axis_cache_hit': bool(selected) and all(hits.values())}


@contextmanager
def _scratch_config(config):
    """Restore every top-level field, including formerly absent engine scratch fields."""
    try:
        original = dict(vars(config))
    except TypeError as exc:
        raise ValueError('Activation explanation requires a mutable configuration object') from exc
    try:
        for name, value in _SCRATCH_OVERRIDES.items():
            setattr(config, name, value)
        yield
    finally:
        state = vars(config)
        state.clear()
        state.update(original)


@contextmanager
def _scratch_cap_telemetry():
    """Keep diagnostic pushes out of the conversational cap-calibration pool."""
    from invariants import engine

    original = (engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count)
    diagnostics = {}
    engine._steer_attempts = deque(maxlen=original[0].maxlen)
    engine._steer_apply_count = engine._steer_clip_count = 0
    try:
        yield diagnostics
    finally:
        try:
            diagnostics.update(engine.steer_telemetry_stats())
        finally:
            engine._steer_attempts, engine._steer_apply_count, engine._steer_clip_count = original


def generate_activation_explanation(model, subject, evidence='', *, generate, config,
                                    subject_gain=.02, operation_gain=.12, reference_gain=.04,
                                    max_tokens=384, cap_fraction=None, axis_cache=None,
                                    map_enabled=False, map_tokens=24):
    """Generate an unprefixed candidate with three independently bounded controls.

    The default controls pulse at the original assistant decision. The optional
    sustained operation is a learned map of the CURRENT block state, evaluated
    at each assistant prediction position. Both paths share one residual cap
    with subject/reference decision pulses and restore on prefix rebuilding.
    Ordinary agentic mechanisms and tool interpretation are isolated for this
    invocation. No structural or semantic success is inferred from its text.
    """
    evidence = _evidence(evidence)
    gains = {
        'subject': _gain(subject_gain, 'subject gain', signed=False),
        'operation': _gain(operation_gain, 'operation gain', signed=True),
        'reference': _gain(reference_gain, 'reference gain', signed=True),
    }
    budget = _budget(max_tokens)
    if type(map_enabled) is not bool:
        raise ValueError('Activation explanation map selection must be boolean')
    map_tokens = _budget(map_tokens)
    if not 3 <= map_tokens <= 64:
        raise ValueError('Activation map calibration budget must be between 3 and 64')
    if not callable(generate):
        raise ValueError('Activation explanation requires a generation callback')
    if not isinstance(subject, Mapping) or not subject:
        raise ValueError('Activation explanation requires a selected subject direction')
    layers = _requested_layers(subject)
    subject = _validated_direction(subject, layers, 'subject', width=getattr(model, 'd_model', None))
    if cap_fraction is not None:
        if isinstance(cap_fraction, bool):
            raise ValueError('Activation explanation cap must be finite and nonnegative')
        try:
            cap_fraction = float(cap_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Activation explanation cap must be finite and nonnegative') from exc
        if not math.isfinite(cap_fraction) or cap_fraction < 0:
            raise ValueError('Activation explanation cap must be finite and nonnegative')
    else:
        from invariants.engine import get_steer_cap_fraction
        cap_fraction = get_steer_cap_fraction()
    prompt, prefill = build_activation_explanation_prompt(model, evidence)
    axes, cache = {}, {'axes_cached': {}, 'axis_cache_hit': False}
    handles, map_metadata, map_telemetry = [], {}, {}
    mapped_operation = map_enabled and bool(gains['operation'])
    with _scratch_config(config), _scratch_cap_telemetry() as cap_telemetry:
        try:
            if gains['operation'] or gains['reference']:
                axes, cache = get_or_mint_activation_axes(
                    model, layers, evidence, axis_cache,
                    include_operation=bool(gains['operation']) and not map_enabled,
                    include_reference=bool(gains['reference']),
                    subject=subject)
            if mapped_operation:
                from invariants.activation_explanation_map import (
                    get_or_mint_activation_map, build_mapped_explanation_handles,
                )
                profiles, map_metadata = get_or_mint_activation_map(
                    model, subject, evidence, axis_cache, max_tokens=map_tokens)
                handles, map_telemetry = build_mapped_explanation_handles(
                    model, subject, profiles, operation_gain=gains['operation'],
                    subject_gain=gains['subject'], reference=axes.get('reference'),
                    reference_gain=gains['reference'], cap_fraction=cap_fraction)
            elif any(gains.values()):
                handles = build_definition_handles(
                    model, subject, axes, cap_fraction=cap_fraction,
                    subject_fraction=gains['subject'],
                    task_gains={name: gains[name] for name in axes},
                    task_positions={name: 'decision' for name in axes})
            output = generate(
                model, instruction=prompt, config=config, vecs={}, pre_formatted=True,
                max_new_tokens=budget, max_tool_calls=0, chatty_log=False,
                return_telemetry=True, generation_channel='reader')
            if (not isinstance(output, tuple) or len(output) != 2
                    or not isinstance(output[0], str) or not isinstance(output[1], Mapping)):
                raise ValueError('Activation explanation generation requires text and token telemetry')
            text, telemetry = output
        finally:
            for handle in handles:
                handle.remove()
    return {
        'text': text, 'telemetry': telemetry,
        'metadata': {
            'version': ACTIVATION_EXPLAIN_VERSION, 'mode': 'activation',
            'status': 'candidate', 'candidate': True,
            'semantic_truth_checked': False, 'causal_effect_checked': False,
            'reference_candidate': True, 'prompt_task_request': False,
            'assistant_prefix': '', 'evidence_sha256': prefill['evidence_sha256'],
            'evidence_present': bool(evidence.strip()), 'layers': list(layers),
            'gains': gains, 'placement': ('current_state_map_assistant_positions' if mapped_operation
                                         else 'original_assistant_decision'),
            'map_enabled': map_enabled, 'map_tokens': map_tokens,
            'map_telemetry': map_telemetry, **map_metadata,
            'max_tokens': budget, 'cap_fraction': cap_fraction,
            'operation_version': OPERATION_VERSION, 'reference_version': REFERENCE_VERSION,
            'capture_version': CAPTURE_VERSION,
            'calibration_endpoint': 'raw_decoder_block_output_at_assistant_decision',
            'calibration_output_hidden_states': False,
            'prefill': prefill, 'generation_channel': 'reader',
            'cap_telemetry': cap_telemetry,
            'config_overrides': dict(_SCRATCH_OVERRIDES), **cache,
        },
    }
