"""Directional predictive distance through the model's actual output logits.

This is an output-side observation, not a source-to-teacher transformation or
a semantic truth check. Counterfactuals share exact token IDs and an unsteered
history. The selected probe, never its name or seed text, supplies the direction.
"""

from collections.abc import Mapping
import hashlib
import math
from numbers import Integral

import torch


LOGIT_DISTANCE_VERSION = 'probe-directional-logit-distance-candidate-v1'


def _epsilon(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Logit distance epsilon must be finite and between 0 (exclusive) and 0.1') from exc
    if isinstance(value, bool) or not math.isfinite(result) or not 0 < result <= .1:
        raise ValueError('Logit distance epsilon must be finite and between 0 (exclusive) and 0.1')
    return result


def _count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 64:
        raise ValueError('Logit distance top_k must be an integer between 1 and 64')
    return int(value)


def _logits(value, width=None):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or value.ndim != 1 or value.numel() < 2):
        raise ValueError('Directional logits require a floating vocabulary vector')
    result = value.detach().to(device='cpu', dtype=torch.float64).clone()
    if not torch.isfinite(result).all() or (width is not None and result.numel() != width):
        raise ValueError('Directional logits must be finite with matching vocabulary widths')
    return result


def _finite(value, label):
    if not torch.isfinite(value).all():
        raise ValueError(f'Directional logit {label} exceeded finite range')
    return value


def _rms(value, probability):
    return float(_finite((probability * value.square()).sum(), 'weighted variance').clamp_min(0).sqrt())


def _fisher_rao(first, second):
    # Equivalent to 2*acos(sum(sqrt(p*q))), but the root-vector chord avoids
    # subtractive loss when a small displacement rounds that coefficient to 1.
    chord = ((first / 2).exp() - (second / 2).exp()).norm()
    return float(4 * (chord / 2).clamp(0, 1).asin())


def analyze_directional_logits(baseline, positive, negative, epsilon, *, top_k=12,
                               positive_half=None, negative_half=None):
    """Measure local log-prob sensitivity along one declared residual direction.

    At temperature one, u=(z+ - z-)/(2*epsilon), s=u-E_p0[u]. This removes
    uniform logit shifts. sqrt(E_p0[s^2]) is local Fisher length per unit step.
    Distances here concern output distributions, not verified word meanings.
    """
    epsilon, count = _epsilon(epsilon), _count(top_k)
    baseline = _logits(baseline)
    width = baseline.numel()
    positive, negative = _logits(positive, width), _logits(negative, width)
    logp, logplus, logminus = [
        _finite(value.log_softmax(-1), 'log probabilities') for value in (baseline, positive, negative)]
    probability = logp.exp()
    raw = _finite((positive - negative) / (2 * epsilon), 'directional derivative')
    mean = _finite((probability * raw).sum(), 'softmax gauge')
    sensitivity = raw - mean
    direct = (logplus - logminus) / (2 * epsilon)
    fisher = _rms(sensitivity, probability)
    midpoint = (positive + negative) / 2 - baseline
    midpoint = midpoint - (probability * midpoint).sum()
    influence = _finite(probability * sensitivity.square(), 'token influence')
    kl_plus = max(0., float((probability * (logp - logplus)).sum()))
    kl_minus = max(0., float((probability * (logp - logminus)).sum()))
    result = {
        'version': LOGIT_DISTANCE_VERSION, 'candidate': True,
        'semantic_truth_checked': False, 'definition_quality_checked': False,
        'measurement': 'directional_predictive_distance', 'temperature': 1.,
        'epsilon': epsilon, 'vocab_size': width,
        'fisher_rms': fisher, 'weighted_sensitivity_mean': float((probability * sensitivity).sum()),
        'fisher_rao_baseline_positive': _fisher_rao(logp, logplus),
        'fisher_rao_baseline_negative': _fisher_rao(logp, logminus),
        'fisher_rao_negative_positive': _fisher_rao(logminus, logplus),
        'kl_baseline_positive': kl_plus, 'kl_baseline_negative': kl_minus,
        'local_kl_prediction': .5 * epsilon ** 2 * fisher ** 2,
        'direct_logprob_relative_error': _rms(direct - sensitivity, probability) / max(fisher, 1e-12),
        'midpoint_relative_error': _rms(midpoint, probability) / max(epsilon * fisher, 1e-12),
        'uniform_logit_shift_removed': True,
        'token_ranking': 'baseline_probability_times_squared_logprob_sensitivity',
    }
    if (positive_half is None) != (negative_half is None):
        raise ValueError('Half-step logit measurement requires both signs')
    if positive_half is not None:
        half_raw = (_logits(positive_half, width) - _logits(negative_half, width)) / epsilon
        half = half_raw - (probability * half_raw).sum()
        result['half_step_fisher_rms'] = _rms(half, probability)
        result['half_step_relative_error'] = _rms(half - sensitivity, probability) / max(fisher, 1e-12)
    indices = torch.argsort(influence, descending=True, stable=True)

    def rows(sign=None):
        selected = indices if sign is None else indices[sensitivity[indices] * sign > 0]
        return [{'id': int(index), 'baseline_probability': float(probability[index]),
                 'positive_probability': float(logplus[index].exp()),
                 'negative_probability': float(logminus[index].exp()),
                 'baseline_logit': float(baseline[index]),
                 'positive_logit': float(positive[index]), 'negative_logit': float(negative[index]),
                 'logit_sensitivity': float(raw[index]),
                 'logprob_sensitivity': float(sensitivity[index]),
                 'direct_logprob_sensitivity': float(direct[index]),
                 'influence': float(influence[index])} for index in selected[:count]]

    result.update(influential=rows(), positive=rows(1), negative=rows(-1))
    return result


@torch.no_grad()
def measure_probe_logit_distance(model, subject, evidence='', *, epsilon=.01,
                                 top_k=12, cap_fraction=None, check_half=True,
                                 assistant_token_ids=None):
    """Observe a probe tangent at an exact unsteered assistant prediction point.

    Perturb only the final token of the last selected block. The baseline raw
    residual norm is frozen across signs; downstream blocks, terminal norm and
    output head all run normally. This does not replay a steered conversation
    or reuse KV state. Optional assistant prefixes are exact token IDs.
    """
    from invariants.engine import _inputs, _cap_steer, get_steer_cap_fraction
    from invariants.probe_activation_explain import (
        build_activation_explanation_prompt, _validated_direction, _scratch_cap_telemetry,
    )

    epsilon, top_k = _epsilon(epsilon), _count(top_k)
    if type(check_half) is not bool:
        raise ValueError('Logit distance half-step selection must be boolean')
    if not isinstance(subject, Mapping) or not subject or any(type(layer) is not int for layer in subject):
        raise ValueError('Logit distance requires selected integer probe layers')
    layers = tuple(sorted(subject))
    direction = _validated_direction(subject, layers, 'logit-distance subject', width=model.d_model)
    layer = layers[-1]
    blocks = model.model.model.layers
    if layers[0] < 0 or layer >= len(blocks):
        raise ValueError('Logit distance layer is absent from the model')
    vector = direction[layer] / direction[layer].norm()
    original_cap = cap_fraction
    try:
        cap_fraction = get_steer_cap_fraction() if cap_fraction is None else float(cap_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Logit distance cap must be finite and nonnegative') from exc
    if isinstance(original_cap, bool) or not math.isfinite(cap_fraction) or cap_fraction < 0:
        raise ValueError('Logit distance cap must be finite and nonnegative')
    if epsilon > cap_fraction:
        raise ValueError('Logit distance epsilon must not exceed the steering cap; clipped differences cannot estimate the requested direction')
    prompt, prefill = build_activation_explanation_prompt(model, evidence)
    inputs = dict(_inputs(model, prompt, pre_formatted=True))
    if assistant_token_ids is not None:
        if (not isinstance(assistant_token_ids, torch.Tensor) or assistant_token_ids.dtype != torch.long
                or assistant_token_ids.ndim != 1 or not assistant_token_ids.numel()
                or (assistant_token_ids < 0).any()):
            raise ValueError('Logit distance assistant prefix requires exact nonnegative integer token IDs')
        if set(inputs) - {'input_ids', 'attention_mask'}:
            raise ValueError('Exact logit prefixes require an input_ids/attention_mask native encoding')
        inputs['input_ids'] = torch.cat((inputs['input_ids'], assistant_token_ids.reshape(1, -1).to(model.device)), dim=1)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
    if inputs['input_ids'].ndim != 2 or inputs['input_ids'].shape[0] != 1:
        raise ValueError('Logit distance requires one exact native context')
    handles, states = [], {}
    raw = model.model
    # A preexisting PyTorch hook may steer or mutate state differently on each
    # counterfactual. Preserve it and refuse the measurement rather than claim
    # an identical unsteered history. Accelerate's native offload wrappers are
    # part of the model execution, not these user-installed hooks.
    if any(module._forward_hooks or module._forward_pre_hooks for module in raw.modules()):
        raise ValueError('Logit distance requires a model without active forward hooks; existing hooks were preserved')
    training = [(module, module.training) for module in raw.modules()]
    perturbations = []

    def forward():
        output = raw(**inputs, use_cache=False, output_hidden_states=False)
        logits = getattr(output, 'logits', None)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError('Logit distance requires native vocabulary logits')
        states['logits_dtype'] = str(logits.dtype)
        return _logits(logits[0, -1])

    def capture(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1
                or hidden.shape[-1] != model.d_model or not torch.isfinite(hidden).all()):
            raise ValueError('Logit distance requires finite raw native block output')
        states['baseline'] = hidden[0, -1].detach().float().clone()
        states['residual_dtype'] = str(hidden.dtype)

    def counterfactual(step):
        def perturb(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1
                    or hidden.shape[-1] != model.d_model or not torch.isfinite(hidden).all()):
                raise ValueError('Logit distance counterfactual requires finite raw native block output')
            add = torch.zeros_like(hidden)
            requested = step * states['norm'] * vector.to(hidden.device)
            add[0, -1] = requested.to(hidden.dtype)
            bounded = _cap_steer(add, hidden, cap_fraction=cap_fraction)
            actual = (hidden + bounded)[0, -1].float() - hidden[0, -1].float()
            applied_norm = float(actual.norm())
            signed_axis = float(actual @ vector.to(hidden.device))
            if applied_norm == 0 or signed_axis * step <= 0:
                raise ValueError('Logit distance step is unresolved at the model residual precision; increase epsilon')
            perturbations.append({'step': step, 'requested_fraction': abs(step),
                'applied_fraction': applied_norm / states['norm'],
                'signed_axis_fraction': signed_axis / states['norm'],
                'realized_step_relative_error': float((actual - requested).norm()) / float(requested.norm())})
            new = hidden + bounded
            return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
        handle = blocks[layer].register_forward_hook(perturb)
        try:
            return forward()
        finally:
            handle.remove()

    with _scratch_cap_telemetry() as telemetry:
        try:
            raw.eval()
            handle = blocks[layer].register_forward_hook(capture)
            handles.append(handle)
            baseline = forward()
            handle.remove()
            handles.clear()
            if 'baseline' not in states:
                raise ValueError('Logit distance missed the selected block')
            states['norm'] = float(states['baseline'].norm())
            if not math.isfinite(states['norm']) or states['norm'] <= 1e-8:
                raise ValueError('Logit distance needs a finite nonzero baseline residual')
            positive, negative = counterfactual(epsilon), counterfactual(-epsilon)
            if check_half:
                positive_half, negative_half = counterfactual(epsilon / 2), counterfactual(-epsilon / 2)
            else:
                positive_half = negative_half = None
        finally:
            for handle in handles:
                handle.remove()
            for module, original_training in training:
                module.training = original_training
    if telemetry['clipped']:
        raise ValueError('Logit distance perturbation was clipped; directional measurement is unavailable')
    report = analyze_directional_logits(baseline, positive, negative, epsilon, top_k=top_k,
        positive_half=positive_half, negative_half=negative_half)
    special = set(getattr(model.tok, 'all_special_ids', ()) or ())
    for channel in ('influential', 'positive', 'negative'):
        for row in report[channel]:
            try:
                row['token'] = model.tok.decode([row['id']])
            except (TypeError, ValueError, RuntimeError):
                row['token'] = ''
            row['special'] = row['id'] in special
    report['metadata'] = {
        'mode': 'logits', 'prompt_task_request': False, 'assistant_prefix': '',
        'prefill': prefill, 'layers': [layer], 'selected_probe_layers': list(layers),
        'placement': 'last_native_assistant_prediction_token',
        'capture_endpoint': 'raw_block_output_before_terminal_normalization_and_downstream_blocks',
        'baseline_residual_norm': states['norm'], 'step_units': 'fraction_of_frozen_baseline_residual_norm',
        'residual_dtype': states['residual_dtype'], 'logits_dtype': states['logits_dtype'],
        'cap_fraction': cap_fraction, 'cap_telemetry': telemetry, 'perturbations': perturbations,
        'counterfactual_context': 'identical_token_ids_unsteered_history',
        'preexisting_forward_hooks': 0, 'residual_step_resolved': True,
        'input_ids_sha256': hashlib.sha256(inputs['input_ids'].detach().cpu().numpy().tobytes()).hexdigest(),
        'direction_sha256': hashlib.sha256(vector.detach().cpu().numpy().tobytes()).hexdigest(),
        'model_identity': str(getattr(getattr(raw, 'config', None), '_name_or_path', type(raw).__name__)),
        'assistant_prefix_token_count': 0 if assistant_token_ids is None else assistant_token_ids.numel(),
        'forward_count': 5 if check_half else 3, 'use_cache': False,
        'semantic_truth_checked': False, 'definition_quality_checked': False,
    }
    return report


def render_probe_logit_distance(report):
    """Keep observations operator-facing, with explicit predictive units."""
    layer = report['metadata']['layers'][-1]
    lines = [f"[Probe Logits] Directional output distance at layer {layer}, epsilon {report['epsilon']:g}.",
             f"Fisher sensitivity {report['fisher_rms']:.6g}; minus-to-plus distance {report['fisher_rao_negative_positive']:.6g}."]
    if 'half_step_relative_error' in report:
        lines.append(f"Smaller-step disagreement: {report['half_step_relative_error']:.3g}.")
    lines.append(f"Midpoint departure from baseline: {report['midpoint_relative_error']:.3g} of the signed step; finite sensitivity includes numerical precision and curvature.")
    for channel, label in (('positive', 'More likely along the direction'), ('negative', 'Less likely along the direction')):
        tokens = [row for row in report[channel] if not row.get('special')][:6]
        lines.append(label + ': ' + (', '.join(
            f"{row.get('token', '')!r} (s={row['logprob_sensitivity']:+.3g}, p={row['baseline_probability']:.3g})"
            for row in tokens) or 'no non-control tokens'))
    lines.append('Predictive distance only; no definition or probe meaning was verified.')
    return '\n'.join(lines)
