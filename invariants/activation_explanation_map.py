"""Candidate state-to-correction maps for an unprompted assistant continuation.

The map is evaluated on the current raw block output, not on a token counter or
an initial direction. Calibration text and teacher-forced tokens never become
the runtime user body. Neither fitting nor an adaptive map certifies meaning.
"""

import hashlib
import math
from collections.abc import Mapping, MutableMapping

import torch

from invariants.relational_conversion import (
    fit_relational_conversion,
    validate_relational_conversion,
)


MAP_VERSION = 'activation-explanation-current-state-map-candidate-v1'
MAP_REQUESTS = (
    'Explain the distinction currently active in your own activations. '
    'Describe what this distinction means in one concise sentence.',
    'In one concise sentence, describe the behavioral distinction currently '
    'active in your own activations and what moving in either direction would express.',
)
MAP_HELD_OUT_REQUEST = (
    'Describe the active behavioral distinction and explain how movement along it changes a reply.'
)
DEFAULT_MAP_TOKENS = 24


def _fraction(value, label, *, signed=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a finite bounded number') from exc
    if isinstance(value, bool) or not math.isfinite(result) or not (-1 if signed else 0) <= result <= 1:
        raise ValueError(f'{label} must be finite and between {-1 if signed else 0} and 1')
    return result


def _compiled(profile, width, device):
    clean = validate_relational_conversion(profile, expected_width=width)
    if not clean['diagnostics']['adaptive']:
        raise ValueError('A sustained activation map requires observed state-dependent changes')
    tensors = {name: clean[name].to(device=device, dtype=torch.float32) for name in (
        'center_source', 'center_change', 'source_relations', 'coefficients')}
    return tensors, clean['training_radius'], clean['extrapolation'], clean['diagnostics']


def _execution_device(block, fallback):
    """Accelerate offloading can leave stored parameters on the meta device."""
    pending = [getattr(block, '_hf_hook', None)]
    visited = set()
    while pending:
        hook = pending.pop(0)
        if hook is None or id(hook) in visited:
            continue
        visited.add(id(hook))
        candidate = getattr(hook, 'execution_device', None)
        if candidate is not None:
            device = torch.device(candidate)
            if device.type != 'meta':
                return device
        pending.extend(getattr(hook, 'hooks', ()) or ())
    for parameter in block.parameters():
        if parameter.device.type != 'meta':
            return parameter.device
    device = torch.device(fallback)
    if device.type == 'meta':
        raise ValueError('Activation map needs a concrete block execution device')
    return device


def evaluate_compiled_map(compiled, hidden):
    """Evaluate the low-rank raw-state map without validation/SVD per token."""
    tensors, radius, extrapolation, _diagnostics = compiled
    relation = hidden.float() - tensors['center_source']
    distance = relation.norm(dim=-1, keepdim=True)
    limit = radius * extrapolation
    source_scale = (limit / distance.clamp_min(1e-30)).clamp(max=1.)
    bounded = relation * source_scale
    delta = tensors['center_change'] + (bounded @ tensors['source_relations'].T) @ tensors['coefficients']
    if not torch.isfinite(delta).all() or not torch.isfinite(distance).all():
        raise ValueError('Activation map prediction must be finite')
    return delta, distance, distance > limit, distance > radius


def build_mapped_explanation_handles(model, subject, profiles, *, operation_gain,
                                     subject_gain, reference=None, reference_gain=0.,
                                     cap_fraction=None):
    """Combine current-state corrections and decision pulses under one cap.

    The native prefill is followed by cached single-token decoding or complete
    prefix rebuilding. Rebuilds reevaluate the same causal map on historical
    assistant positions; no map reaches preceding user/header tokens.
    """
    from invariants.engine import _cap_steer, get_steer_cap_fraction
    from invariants.probe_activation_explain import _validated_direction

    operation_gain = _fraction(operation_gain, 'Activation map gain', signed=True)
    subject_gain = _fraction(subject_gain, 'Activation map subject gain')
    reference_gain = _fraction(reference_gain, 'Activation map reference gain', signed=True)
    if cap_fraction is None:
        cap_fraction = get_steer_cap_fraction()
    else:
        original_cap = cap_fraction
        try:
            cap_fraction = float(cap_fraction)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Activation map cap must be finite and nonnegative') from exc
        if isinstance(original_cap, bool) or not math.isfinite(cap_fraction) or cap_fraction < 0:
            raise ValueError('Activation map cap must be finite and nonnegative')
    if not isinstance(subject, Mapping) or not subject or any(type(layer) is not int for layer in subject):
        raise ValueError('Activation maps require integer subject layers')
    layers = tuple(sorted(subject))
    blocks = model.model.model.layers
    if layers[0] < 0 or layers[-1] >= len(blocks):
        raise ValueError('Activation map layer is absent from the model')
    width = int(model.d_model)
    subject = _validated_direction(subject, layers, 'map subject', width=width)
    if not isinstance(profiles, Mapping) or not profiles or any(type(layer) is not int for layer in profiles):
        raise ValueError('Activation maps require an integer-layer profile mapping')
    if not set(profiles).issubset(subject):
        raise ValueError('Activation map profiles must be on selected subject layers')
    reference = {} if reference is None else reference
    if reference_gain:
        reference = _validated_direction(reference, layers, 'map reference', width=width)
    prepared = {}
    for layer in layers:
        # Prefer the block's actual device for sharded models.
        device = _execution_device(blocks[layer], model.device)
        marker = subject[layer].to(device=device, dtype=torch.float32)
        pulse = subject_gain * marker / marker.norm()
        if reference_gain:
            vector = reference[layer].to(device=device, dtype=torch.float32)
            pulse = pulse + reference_gain * vector / vector.norm()
        profile = _compiled(profiles[layer], width, device) if layer in profiles else None
        prepared[layer] = (pulse, profile)
    telemetry = {
        'version': MAP_VERSION, 'state_conditioned': True, 'static_model': False,
        'candidate': True, 'semantic_truth_checked': False,
        'layers': {}, 'raw_delta_units': True,
    }
    handles = []
    try:
        for layer in layers:
            pulse, compiled = prepared[layer]
            state = {'decision': None, 'last_delta': None}
            stats = {'evaluations': 0, 'source_clamped': 0, 'outside_support': 0,
                     'delta_norm_min': None, 'delta_norm_max': None,
                     'delta_change_max': 0., 'rebuild_evaluations': 0,
                     'adaptive': compiled is not None}
            telemetry['layers'][str(layer)] = stats

            def hook(_module, _args, output, pulse=pulse, compiled=compiled,
                     state=state, stats=stats):
                hidden = output[0] if isinstance(output, tuple) else output
                if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or
                        hidden.shape[0] != 1 or hidden.shape[-1] != width or not torch.isfinite(hidden).all()):
                    raise ValueError('Activation maps require finite native [1, tokens, width] block outputs')
                first = state['decision'] is None
                if first:
                    state['decision'] = hidden.shape[1] - 1
                decision = state['decision']
                cached = not first and hidden.shape[1] == 1
                start = 0 if cached else decision
                if start >= hidden.shape[1]:
                    raise ValueError('Activation map requires full-prefix rebuilds or cached single-token decoding')
                with torch.no_grad():
                    add = torch.zeros_like(hidden)
                    current = hidden[:, start:, :]
                    if compiled is not None and operation_gain:
                        delta, _distance, clamped, outside = evaluate_compiled_map(compiled, current)
                        add[:, start:, :] = (operation_gain * delta).to(hidden.dtype)
                        norms = delta.norm(dim=-1)
                        low, high = float(norms.min()), float(norms.max())
                        stats['delta_norm_min'] = low if stats['delta_norm_min'] is None else min(low, stats['delta_norm_min'])
                        stats['delta_norm_max'] = high if stats['delta_norm_max'] is None else max(high, stats['delta_norm_max'])
                        stats['evaluations'] += delta.shape[1]
                        stats['source_clamped'] += int(clamped.sum())
                        stats['outside_support'] += int(outside.sum())
                        if not first and not cached:
                            stats['rebuild_evaluations'] += delta.shape[1]
                        last = delta[0, -1].detach().clone()
                        if state['last_delta'] is not None:
                            stats['delta_change_max'] = max(stats['delta_change_max'], float((last - state['last_delta']).norm()))
                        state['last_delta'] = last
                    if not cached:
                        add[:, decision, :] += (hidden[:, decision, :].float().norm(dim=-1, keepdim=True) * pulse).to(hidden.dtype)
                    result = hidden + _cap_steer(add, hidden, cap_fraction=cap_fraction)
                return (result,) + tuple(output[1:]) if isinstance(output, tuple) else result

            handles.append(blocks[layer].register_forward_hook(hook))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles, telemetry


@torch.no_grad()
def capture_prediction_trajectory(model, header_ids, continuation_ids, layers):
    """Capture prediction states after exactly shared assistant prefix IDs.

    Row zero is the native header decision; row i>0 follows continuation[:i].
    Concatenating IDs avoids retokenizing a decoded prefix or closing its turn.
    """
    if (not isinstance(header_ids, torch.Tensor) or header_ids.ndim != 2 or header_ids.shape[0] != 1
            or not header_ids.shape[1] or header_ids.dtype != torch.long):
        raise ValueError('Trajectory header requires nonempty [1, tokens] integer IDs')
    if (not isinstance(continuation_ids, torch.Tensor) or continuation_ids.ndim != 1
            or continuation_ids.numel() < 3 or continuation_ids.dtype != torch.long):
        raise ValueError('Trajectory continuation requires at least three exact integer IDs')
    if not layers or any(type(layer) is not int for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError('Trajectory capture requires distinct integer layers')
    blocks = model.model.model.layers
    if min(layers) < 0 or max(layers) >= len(blocks):
        raise ValueError('Trajectory capture layer is absent')
    ids = torch.cat((header_ids.to(model.device), continuation_ids[:-1].reshape(1, -1).to(model.device)), dim=1)
    start, count = header_ids.shape[1] - 1, continuation_ids.numel()
    states, handles = {}, []
    try:
        for layer in layers:
            def capture(_module, _args, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                if (not isinstance(hidden, torch.Tensor) or hidden.shape != (1, ids.shape[1], model.d_model)
                        or not torch.isfinite(hidden).all()):
                    raise ValueError('Trajectory capture requires finite raw native block outputs')
                states[layer] = hidden[0, start:start + count].detach().float().cpu().clone()
            handles.append(blocks[layer].register_forward_hook(capture))
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(states) != set(layers):
        raise ValueError('Trajectory capture missed a requested layer')
    return states


def _teacher_ids(model, body, request, budget):
    from invariants.engine import _inputs
    from invariants.probe_activation_explain import build_activation_explanation_prompt
    teacher_body = body + ('\n\n' if body else '') + request
    prompt, _ = build_activation_explanation_prompt(model, teacher_body)
    inputs = _inputs(model, prompt, pre_formatted=True)
    header = inputs['input_ids']
    # Generation is a bounded calibration pass, without runtime steering hooks.
    with torch.no_grad():
        generated = model.model.generate(**inputs, max_new_tokens=budget, do_sample=False, use_cache=True)
    if not isinstance(generated, torch.Tensor) or generated.ndim != 2 or generated.shape[0] != 1:
        raise ValueError('Activation map calibration requires one teacher token trajectory')
    continuation = generated[0, header.shape[1]:].detach().clone()
    special = set(getattr(model.tok, 'all_special_ids', ()) or ())
    for index, token in enumerate(continuation.tolist()):
        if token in special:
            continuation = continuation[:index]
            break
    if continuation.numel() < 3:
        raise ValueError('Activation map teacher produced fewer than three content tokens')
    return header, continuation


@torch.no_grad()
def calibrate_activation_map(model, layers, evidence='', *, max_tokens=DEFAULT_MAP_TOKENS,
                             evaluate_held_out=False):
    """Fit generic task changes along teacher-forced shared-prefix trajectories.

    Teacher outputs are calibration artifacts, not examples of the probe's true
    meaning. The final selected layer avoids stacking independently fit maps.
    """
    from invariants.engine import _inputs
    from invariants.probe_activation_explain import build_activation_explanation_prompt, _budget
    budget = _budget(max_tokens)
    if not 3 <= budget <= 64:
        raise ValueError('Activation map calibration budget must be between 3 and 64')
    if type(evaluate_held_out) is not bool:
        raise ValueError('Held-out activation map selection must be boolean')
    prompt, _ = build_activation_explanation_prompt(model, evidence)
    source_header = _inputs(model, prompt, pre_formatted=True)['input_ids']
    sources, changes = {layer: [] for layer in layers}, {layer: [] for layer in layers}
    metadata = {'version': MAP_VERSION, 'candidate': True, 'semantic_truth_checked': False,
                'capture_endpoint': 'raw_decoder_block_output_before_map',
                'prefix_alignment': 'exact_shared_assistant_token_ids', 'max_tokens': budget,
                'calibration_requests': list(MAP_REQUESTS), 'trajectories': []}
    for request in MAP_REQUESTS:
        teacher_header, continuation = _teacher_ids(model, evidence, request, budget)
        source = capture_prediction_trajectory(model, source_header, continuation, layers)
        target = capture_prediction_trajectory(model, teacher_header, continuation, layers)
        for layer in layers:
            sources[layer].append(source[layer])
            changes[layer].append(target[layer] - source[layer])
        metadata['trajectories'].append({'request_sha256': hashlib.sha256(request.encode()).hexdigest(),
            'continuation_ids': continuation.cpu().tolist(),
            'teacher_text': model.tok.decode(continuation.cpu().tolist(), skip_special_tokens=True),
            'source_header_tokens': source_header.shape[1], 'target_header_tokens': teacher_header.shape[1],
            'paired_rows': continuation.numel()})
    profiles = {layer: fit_relational_conversion(torch.cat(sources[layer]), torch.cat(changes[layer])) for layer in layers}
    if any(not profile['diagnostics']['adaptive'] for profile in profiles.values()):
        raise ValueError('Calibration did not identify state-dependent activation changes')
    metadata['fits'] = {str(layer): dict(profile['diagnostics'], n_samples=profile['n_samples']) for layer, profile in profiles.items()}
    if evaluate_held_out:
        teacher_header, continuation = _teacher_ids(model, evidence, MAP_HELD_OUT_REQUEST, budget)
        source = capture_prediction_trajectory(model, source_header, continuation, layers)
        target = capture_prediction_trajectory(model, teacher_header, continuation, layers)
        held_out = {}
        for layer, profile in profiles.items():
            predicted = evaluate_compiled_map(_compiled(profile, model.d_model, 'cpu'), source[layer])[0]
            observed = target[layer] - source[layer]
            denominator = float(observed.norm().clamp_min(1e-12))
            held_out[str(layer)] = {
                'map_relative_error': float((predicted - observed).norm()) / denominator,
                'mean_relative_error': float((profile['center_change'] - observed).norm()) / denominator,
                'paired_rows': continuation.numel(), 'semantic_truth_checked': False}
        metadata['held_out'] = {'request': MAP_HELD_OUT_REQUEST, 'layers': held_out,
            'teacher_text': model.tok.decode(continuation.cpu().tolist(), skip_special_tokens=True),
            'continuation_ids': continuation.cpu().tolist(), 'whole_trajectory_held_out': True}
    return profiles, metadata


def get_or_mint_activation_map(model, subject, evidence='', cache=None, *, max_tokens=DEFAULT_MAP_TOKENS):
    """Keep the trajectory map in the same owner-local runtime cache as axes."""
    from invariants.probe_activation_explain import _axis_cache_key, _budget, _validated_direction
    if cache is not None and not isinstance(cache, MutableMapping):
        raise ValueError('Activation map cache must be mutable or None')
    layers = tuple(sorted(subject))
    subject = _validated_direction(subject, layers, 'map cache subject', width=model.d_model)
    budget = _budget(max_tokens)
    base, scope = _axis_cache_key(model, layers, evidence, subject)
    key = hashlib.sha256((MAP_VERSION + base + str(budget)).encode()).hexdigest()
    if cache is not None and key in cache:
        profiles, calibration = cache[key]
        profiles = {layer: validate_relational_conversion(profile, expected_width=model.d_model) for layer, profile in profiles.items()}
        if set(profiles) != {layers[-1]} or any(not profile['diagnostics']['adaptive'] for profile in profiles.values()):
            raise ValueError('Cached activation map has invalid layers or is not state conditioned')
        hit = True
    else:
        profiles, calibration = calibrate_activation_map(model, (layers[-1],), evidence, max_tokens=budget)
        if cache is not None:
            cache[key] = (profiles, calibration)
        hit = False
    return profiles, {'map_cache_key': key, 'map_cache_hit': hit, 'map_cache_identity_scope': scope,
                      'map_layers': list(profiles), 'map_calibration': calibration}
