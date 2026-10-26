"""Experimental depth composition of four definition-schema conversions.

One native prefill receives genus, core, boundary, then composition pulses in
successive groups of selected blocks. Later conversions observe the actual
block output, including earlier interventions. Calibration used clean evidence,
so that online source shift remains experimental and explicitly unverified.
No decoded intermediate candidates or task instructions are produced here.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Integral

import torch

from invariants.probe_definition_schema import SCHEMA_ORDER, SCHEMA_GAINS, validate_schema_bank
from invariants.relational_conversion import predict_relational_conversion


SEQUENCE_VERSION = 'definition-schema-ordered-block-pulses-candidate-v1'


def _order(value):
    try:
        result = tuple(SCHEMA_ORDER if value is None else value)
        valid = len(result) == len(SCHEMA_ORDER) and set(result) == set(SCHEMA_ORDER)
    except (TypeError, ValueError) as exc:
        raise ValueError('Definition schema order must contain each of the four roles exactly once') from exc
    if not valid:
        raise ValueError('Definition schema order must contain each of the four roles exactly once')
    return result


def _gains(value, defaults=None):
    result = dict(SCHEMA_GAINS if defaults is None else defaults)
    if value is not None:
        if not isinstance(value, Mapping) or not set(value).issubset(SCHEMA_ORDER):
            raise ValueError('Definition schema gains must map declared roles to signed gains')
        for role, supplied in value.items():
            try:
                gain = float(supplied)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'Definition schema gain for {role} must be finite and between -1 and 1') from exc
            if isinstance(supplied, bool) or not math.isfinite(gain) or abs(gain) > 1:
                raise ValueError(f'Definition schema gain for {role} must be finite and between -1 and 1')
            result[role] = gain
    return result


def _fraction(value, label, maximum=None):
    try:
        fraction = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be a finite nonnegative fraction') from exc
    if isinstance(value, bool) or not math.isfinite(fraction) or fraction < 0 or (maximum is not None and fraction > maximum):
        raise ValueError(f'{label} must be a finite nonnegative fraction' + (f' <= {maximum}' if maximum is not None else ''))
    return fraction


class DefinitionSchemaSequence:
    """One invocation's prefill position, immutable pulse memos and operator trace."""

    def __init__(self, *, role_gains=None, order=None):
        self._gains = _gains(role_gains)
        self._order = _order(order)
        self._claimed = False
        self._epoch = 0
        self._current_length = None
        self._current_kind = None
        self._decision_position = None
        self._layers = ()
        self._roles = {}
        self._pulses = {}
        self._predictions = {}
        self._replay_counts = {}
        self._kinds = {'initial_prefill': 0, 'prefix_rebuild': 0, 'cached_decode': 0}
        self._signature = {}

    @property
    def role_gains(self):
        return dict(self._gains)

    @property
    def order(self):
        return tuple(self._order)

    def _bind(self, layers, gains, order, cap_fraction, subject_fraction):
        if self._claimed or self._epoch:
            raise ValueError('Definition schema sequence must be fresh for each invocation')
        self._layers = tuple(layers)
        self._gains = dict(gains)
        self._order = tuple(order)
        for index, role in enumerate(order):
            # Consecutive slices of sorted selected layers. With an uneven
            # count, the final group retains the remaining composition blocks.
            group = layers[index * len(layers) // 4:(index + 1) * len(layers) // 4]
            self._roles.update({layer: role for layer in group})
        from invariants.engine import get_steer_cap_fraction
        self._signature = {
            'layers': list(layers), 'order': list(order), 'role_gains': dict(gains),
            'role_layers': {role: [layer for layer in layers if self._roles[layer] == role] for role in order},
            'cap_fraction': cap_fraction,
            'cap_mode': 'live_global_at_original_prediction' if cap_fraction is None else 'explicit',
            'cap_at_binding': get_steer_cap_fraction() if cap_fraction is None else cap_fraction,
            'subject_fraction': subject_fraction,
            'subject_layer': layers[0],
        }

    def begin_forward(self):
        self._epoch += 1
        self._current_length = None
        self._current_kind = None

    def decision_position(self, sequence_length):
        if not self._epoch:
            raise RuntimeError('Definition schema sequence requires its model-root prehook')
        if sequence_length <= 0:
            raise ValueError('Definition schema sequence requires a nonempty residual sequence')
        if self._current_length is not None:
            if sequence_length != self._current_length:
                raise ValueError('Definition schema blocks must share one sequence length within a forward')
        else:
            self._current_length = sequence_length
            if self._decision_position is None:
                self._decision_position = sequence_length - 1
                self._current_kind = 'initial_prefill'
            elif sequence_length == 1:
                self._current_kind = 'cached_decode'
            elif sequence_length <= self._decision_position:
                raise ValueError('Definition schema prefix rebuild is shorter than its original prefix')
            else:
                self._current_kind = 'prefix_rebuild'
            self._kinds[self._current_kind] += 1
        return None if self._current_kind == 'cached_decode' else self._decision_position

    def trace(self):
        """Describe applied residual operations, without certifying semantic slots."""
        return copy.deepcopy({
            'version': SEQUENCE_VERSION, 'diagnostic_only': True,
            'semantic_truth_checked': False, 'slot_completion_checked': False,
            'calibration_source': 'clean_original_evidence_block_output',
            'runtime_source': 'current_decision_block_output_after_earlier_pulses',
            'online_intervention_source_shift': 'experimental_unverified',
            'replay': 'saved_capped_original_addition_without_new_prediction',
            'cached_decode': 'untouched',
            'decision_position': self._decision_position,
            'forward_count': self._epoch, 'forward_kinds': dict(self._kinds),
            'signature': {**self._signature,
                          **({'role_gains': dict(self._gains)} if self._signature else {})},
            'prediction_count': len(self._predictions),
            'layers': [{**self._predictions[layer], 'replay_count': self._replay_counts.get(layer, 0)}
                       for layer in self._layers if layer in self._predictions],
        })


def build_schema_sequence_handles(
    model, subject, bank, *, role_gains=None, cap_fraction=None,
    subject_fraction=.02, order=None, sequence=None,
):
    """Install one root tracker and one conditional pulse hook per selected block.

    Every bank profile and subject is validated before registration. The fixed
    subject is added only at the earliest selected block. At each block its
    assigned task pulse shares one cap with that subject contribution. The
    first capped addition is memoized in the residual dtype and replayed at the
    same original prefix position on rebuilds; it is not rescaled or predicted
    again from a decode token or a rebuilt prefix. Start a fresh sequence for a
    different cap, gain, input, model or order.
    """
    from invariants.engine import _cap_steer, get_steer_cap_fraction

    if sequence is None:
        sequence = DefinitionSchemaSequence()
    if not isinstance(sequence, DefinitionSchemaSequence) or sequence._claimed or sequence._epoch:
        raise ValueError('Definition schema sequence must be fresh for each invocation')
    gains = _gains(role_gains, sequence.role_gains)
    ordering = sequence.order if order is None else _order(order)
    subject_gain = _fraction(subject_fraction, 'Definition schema subject fraction', 1)
    cap = None if cap_fraction is None else _fraction(cap_fraction, 'Definition schema cap fraction')
    if not isinstance(subject, Mapping) or len(subject) < 4:
        raise ValueError('Definition schema sequence requires at least four selected layers')
    if any(isinstance(layer, bool) or not isinstance(layer, Integral) or layer < 0 for layer in subject):
        raise ValueError('Definition schema subject layers must be nonnegative integer indices')
    layers = tuple(sorted(int(layer) for layer in subject))
    width = getattr(model, 'd_model', None)
    if isinstance(width, bool) or not isinstance(width, Integral) or width < 1:
        raise ValueError('Definition schema sequence requires a known positive residual width')
    markers = {}
    for layer in layers:
        marker = subject[layer]
        if not torch.is_tensor(marker) or not marker.is_floating_point() or marker.layout != torch.strided or marker.shape != (width,):
            raise ValueError(f'Definition schema subject at layer {layer} must match residual width')
        marker = marker.detach().to(device='cpu', dtype=torch.float32).clone()
        norm = float(marker.norm())
        if not torch.isfinite(marker).all() or not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError(f'Definition schema subject at layer {layer} must be finite and nonzero')
        markers[layer] = marker / norm
    validated_bank = validate_schema_bank(bank, layers, int(width))
    root = model.model
    if not hasattr(root, 'register_forward_pre_hook') or not hasattr(root, 'model') or not hasattr(root.model, 'layers'):
        raise ValueError('Definition schema sequence requires a native model root and decoder blocks')
    blocks = root.model.layers
    if layers[-1] >= len(blocks) or any(not hasattr(blocks[layer], 'register_forward_hook') for layer in layers):
        raise ValueError('Definition schema sequence is missing a requested decoder block')
    sequence._bind(layers, gains, ordering, cap, subject_gain)
    handles = []
    try:
        handles.append(root.register_forward_pre_hook(lambda _module, _inputs: sequence.begin_forward()))
        for layer in layers:
            role = sequence._roles[layer]
            profile = validated_bank[role][layer]
            marker = markers[layer]

            def hook(_module, _inputs, output, layer=layer, role=role, profile=profile, marker=marker):
                h = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(h) or h.ndim != 3 or h.shape[0] != 1 or h.shape[-1] != width:
                    raise ValueError('Definition schema sequence requires batch-one [1, tokens, width] residuals')
                position = sequence.decision_position(h.shape[1])
                if position is None:
                    return output
                with torch.no_grad():
                    if layer in sequence._pulses:
                        pulse = sequence._pulses[layer].to(device=h.device, dtype=h.dtype)
                        sequence._replay_counts[layer] = sequence._replay_counts.get(layer, 0) + 1
                    else:
                        if sequence._current_kind != 'initial_prefill':
                            raise ValueError('Definition schema replay requires an original prediction at every selected layer')
                        state = h[0, position].detach().float()
                        state_norm = float(state.norm())
                        if not torch.isfinite(state).all() or not math.isfinite(state_norm):
                            raise ValueError('Definition schema decision residual must be finite with a finite norm')
                        delta, prediction = predict_relational_conversion(profile, state)
                        delta_norm = float(delta.double().norm())
                        reference = profile['changes_norm_mean']
                        layer_scale = min(2., delta_norm / reference)
                        direction = (delta.double() / delta_norm).float() if delta_norm else torch.zeros_like(delta)
                        gain = gains[role]
                        contribution = gain * layer_scale * direction.to(h.device)
                        marks_subject = layer == layers[0]
                        if marks_subject:
                            contribution += subject_gain * marker.to(h.device)
                        attempted = (state_norm * contribution).to(h.dtype)
                        if not torch.isfinite(attempted).all():
                            raise ValueError('Definition schema attempted pulse exceeded the finite residual range')
                        pulse = _cap_steer(attempted, h[0, position], cap_fraction=cap).detach().clone()
                        if not torch.isfinite(pulse).all():
                            raise ValueError('Definition schema capped pulse must be finite')
                        sequence._pulses[layer] = pulse
                        sequence._predictions[layer] = {
                            'layer': layer, 'role': role, 'gain': gain,
                            'prediction': dict(prediction),
                            'source_norm': state_norm, 'raw_delta_norm': delta_norm,
                            'mean_training_change_norm': reference,
                            'layer_scale': layer_scale, 'effective_task_gain': gain * layer_scale,
                            'subject_pulse': marks_subject,
                            'subject_fraction': subject_gain if marks_subject else 0.,
                            'attempted_addition_norm': float(attempted.float().norm()),
                            'applied_addition_norm': float(pulse.float().norm()),
                            'cap_at_prediction': get_steer_cap_fraction() if cap is None else cap,
                            'first_prediction_forward': sequence._epoch,
                            'diagnostic_only': True,
                        }
                    changed = h.clone()
                    changed[0, position] += pulse
                return (changed,) + tuple(output[1:]) if isinstance(output, tuple) else changed

            handles.append(blocks[layer].register_forward_hook(hook))
        sequence._claimed = True
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles
