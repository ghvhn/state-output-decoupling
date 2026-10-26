"""Conditional candidate transformations for evidence-only probe definitions.

Sources are unprimed native assistant-decision states. Targets are raw state
changes between completed definition demonstrations and matched recap
demonstrations, ending at the same unanswered query. Learned conversions are
CPU data, not proof that their interventions execute the named operations.
"""
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

from invariants.probe_definition_steps import (
    STEP_ORDER, STEP_GAINS, _digest, _identity as _model_identity,
)
from invariants.probe_definition_task import _native_states, _requested_layers
from invariants.relational_conversion import (
    RELATIONAL_CONVERSION_VERSION, fit_relational_conversion,
    predict_relational_conversion, validate_relational_conversion,
)


RELATION_VERSION = 'definition-relations-native-decision-candidate-v1'
SOURCE_FORMAT_VERSION = 'unprimed-evidence-with-opaque-prior-contrast-v1'
RIDGE = .01
EXTRAPOLATION = 1.5
_SCHEMA = 1

# These unrelated synthetic themes are calibration data, never labels for an
# operator's actual probe. The two fixed teacher themes are absent from every
# queried calibration theme, so a queried theme is never its own demonstration.
_THEMES = (
    ('choose the first available route', 'compare alternatives before choosing a route', 'selecting a route'),
    ('report an immediate estimate', 'check the sensor before reporting a measurement', 'reporting a measurement'),
    ('publish an individual draft', 'integrate a review before publishing a draft', 'publishing a draft'),
    ('start the next visible task', 'order prerequisites before starting a task', 'starting a task'),
    ('classify from surface resemblance', 'check defining criteria before classifying an item', 'classifying an item'),
    ('replace the visibly broken part', 'trace the cause before repairing a fault', 'repairing a fault'),
)
_TEACHERS = (
    ('count the arriving packages', 'reconcile the ledger before counting packages', 'counting packages'),
    ('translate words individually', 'resolve the surrounding context before translating words', 'translating words'),
)


def _candidate_evidence(evidence, candidate):
    if candidate is None:
        return evidence
    identifier = 'c_' + hashlib.sha256(candidate.encode('utf-8')).hexdigest()[:24]
    return evidence + f'\n\n[Candidate {identifier}]\n' + candidate


def _case(theme, orientation):
    first, second, shared = theme
    readings = (-.75, -.25, .25, .75)
    if orientation:
        readings = tuple(-value for value in readings)
    actions = (first, first, second, second)
    evidence = '[Probe p_000000000000000000000000]\n' + '\n'.join(
        f'Memory {index}: position={reading:+.2f}; input="A case of {shared}."; '
        f'reply="I {action}."'
        for index, (reading, action) in enumerate(zip(readings, actions)))
    lower, higher = (first, second) if not orientation else (second, first)
    lower_ids, higher_ids = ((0, 1), (2, 3)) if not orientation else ((2, 3), (0, 1))
    low = ' and '.join(f'Memory {index}' for index in lower_ids)
    high = ' and '.join(f'Memory {index}' for index in higher_ids)
    contrast = f'{low} {lower}, whereas {high} {higher}.'
    outputs = {
        'compare': contrast,
        'isolate': f'All four examples involve {shared}; the changing behavior is that '
                   f'{low} {lower}, whereas {high} {higher}.',
        'distill': f'The distinction is whether responses {lower} at lower readings '
                   f'or {higher} at higher readings.',
        'verify': f'{low} {lower}, whereas {high} {higher}; each of the four measured '
                  'examples fits its stated end, with no counterexample among these examples.',
    }
    recap = ' '.join(f'Memory {index} concerns {shared} and says "I {action}."'
                     for index, action in enumerate(actions))
    return evidence, outputs, recap


def _stage_evidence(theme, orientation, step):
    evidence, outputs, recap = _case(theme, orientation)
    index = STEP_ORDER.index(step)
    previous = None if index == 0 else outputs[STEP_ORDER[index - 1]]
    return _candidate_evidence(evidence, previous), outputs[step], recap


def _calibration_rows(step):
    if step not in STEP_ORDER:
        raise ValueError('Definition relation requires a declared operation')
    prefix_positive, prefix_negative = [], []
    for index, theme in enumerate(_TEACHERS):
        evidence, completion, recap = _stage_evidence(theme, index % 2, step)
        prefix_positive.extend(({'role': 'user', 'content': evidence},
                                {'role': 'assistant', 'content': completion}))
        prefix_negative.extend(({'role': 'user', 'content': evidence},
                                {'role': 'assistant', 'content': recap}))
    rows = []
    for theme_index, theme in enumerate(_THEMES):
        for orientation in (0, 1):
            evidence, _, _ = _stage_evidence(theme, orientation, step)
            final = {'role': 'user', 'content': evidence}
            rows.append({'theme': theme_index, 'orientation': orientation,
                         'evidence': evidence,
                         'positive': [dict(message) for message in prefix_positive] + [dict(final)],
                         'negative': [dict(message) for message in prefix_negative] + [dict(final)]})
    return rows


def _identity(model, layers):
    identity = _model_identity(model, layers)
    identity.update(version=RELATION_VERSION,
                    recipe_sha256=_digest({step: _calibration_rows(step) for step in STEP_ORDER}),
                    learner_version=RELATIONAL_CONVERSION_VERSION,
                    ridge=RIDGE, extrapolation=EXTRAPOLATION,
                    source_format=SOURCE_FORMAT_VERSION,
                    source_units='raw_native_residual', target_units='raw_native_state_difference')
    return identity


def _validated_states(states, layers, width):
    if not isinstance(states, dict) or set(states) != set(layers):
        raise ValueError('Definition relation states must cover exactly the requested layers')
    result = {}
    for layer in layers:
        value = states[layer]
        if not torch.is_tensor(value) or value.shape != (width,) or not value.is_floating_point():
            raise ValueError('Definition relation states have incompatible residual width or dtype')
        value = value.detach().to(device='cpu', dtype=torch.float32).clone()
        if not torch.isfinite(value).all():
            raise ValueError('Definition relation states must be finite')
        result[layer] = value
    return result


def _validate_bank(bank, layers, width):
    if not isinstance(bank, dict) or set(bank) != set(STEP_ORDER):
        raise ValueError('Definition relation bank must contain exactly all four operations')
    validated = {}
    for step in STEP_ORDER:
        profiles = bank[step]
        if (not isinstance(profiles, dict) or set(profiles) != set(layers) or
                any(type(layer) is not int for layer in profiles)):
            raise ValueError('Definition relation bank must cover exactly the requested layers')
        validated[step] = {}
        for layer in layers:
            profile = validate_relational_conversion(profiles[layer], expected_width=width)
            if profile['n_samples'] != len(_THEMES) * 2:
                raise ValueError('Definition relation profile requires all calibration rows')
            if profile['ridge'] != RIDGE or profile['extrapolation'] != EXTRAPOLATION:
                raise ValueError('Definition relation profile has incompatible learner settings')
            if not profile['diagnostics'].get('adaptive'):
                raise ValueError('Definition relation profile must retain measured conditional variation')
            if not math.isfinite(profile['changes_norm_mean']) or profile['changes_norm_mean'] <= 0:
                raise ValueError('Definition relation profile requires nonzero training change magnitude')
            validated[step][layer] = profile
    return validated


def _validate_calibration(calibration, layers):
    if not isinstance(calibration, dict) or set(calibration) != set(STEP_ORDER):
        raise ValueError('Definition relation calibration must cover all operations')
    required = {'heldout_rows', 'heldout_theme', 'error_rms', 'observed_change_distance',
                'predicted_change_distance', 'semantic_truth_checked'}
    for summaries in calibration.values():
        if not isinstance(summaries, dict) or set(summaries) != set(layers):
            raise ValueError('Definition relation calibration must cover all requested layers')
        for summary in summaries.values():
            if not isinstance(summary, dict) or set(summary) != required:
                raise ValueError('Definition relation calibration has invalid diagnostics')
            if (summary['heldout_rows'] != 2 or summary['heldout_theme'] != len(_THEMES) - 1 or
                    summary['semantic_truth_checked'] is not False):
                raise ValueError('Definition relation calibration has incompatible holdout scope')
            for key in ('error_rms', 'observed_change_distance', 'predicted_change_distance'):
                value = summary[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Definition relation calibration metrics must be finite and nonnegative')
    return calibration


@torch.no_grad()
def _mint(model, layers, width):
    bank, calibration = {}, {}
    for step in STEP_ORDER:
        sources, changes = {layer: [] for layer in layers}, {layer: [] for layer in layers}
        for row in _calibration_rows(step):
            source = _validated_states(_native_states(
                model, [{'role': 'user', 'content': row['evidence']}], layers, step,
                add_generation_prompt=True), layers, width)
            positive = _validated_states(_native_states(
                model, row['positive'], layers, step, add_generation_prompt=True), layers, width)
            negative = _validated_states(_native_states(
                model, row['negative'], layers, step, add_generation_prompt=True), layers, width)
            for layer in layers:
                sources[layer].append(source[layer])
                changes[layer].append(positive[layer] - negative[layer])
        bank[step], calibration[step] = {}, {}
        for layer in layers:
            source, change = torch.stack(sources[layer]), torch.stack(changes[layer])
            bank[step][layer] = fit_relational_conversion(source, change, ridge=RIDGE, extrapolation=EXTRAPOLATION)
            # The last theme, both coordinate orientations, is absent from this
            # diagnostic fit. The stored deployment profile still uses all rows.
            heldout = fit_relational_conversion(source[:-2], change[:-2], ridge=RIDGE, extrapolation=EXTRAPOLATION)
            predictions = torch.stack([predict_relational_conversion(heldout, value)[0] for value in source[-2:]])
            calibration[step][layer] = {
                'heldout_rows': 2, 'heldout_theme': len(_THEMES) - 1,
                'error_rms': float((predictions - change[-2:]).square().mean().sqrt()),
                'observed_change_distance': float((change[-1] - change[-2]).norm()),
                'predicted_change_distance': float((predictions[-1] - predictions[-2]).norm()),
                'semantic_truth_checked': False,
            }
    return _validate_bank(bank, layers, width), _validate_calibration(calibration, layers)


def _provenance(identity):
    return {'endpoint': 'native_assistant_decision', 'calibration_rows_per_step': len(_THEMES) * 2,
            'teacher_pairs': len(_TEACHERS), 'recipe_sha256': identity['recipe_sha256'],
            'status': 'candidate', 'semantic_truth_checked': False}


def _validate_bundle(payload, identity, fingerprint, layers):
    required = {'schema', 'identity', 'identity_hash', 'bank', 'provenance', 'calibration'}
    if (not isinstance(payload, dict) or set(payload) != required or payload['schema'] != _SCHEMA or
            payload['identity'] != identity or payload['identity_hash'] != fingerprint or
            payload['provenance'] != _provenance(identity)):
        raise ValueError('Definition relation bank identity, provenance, or storage schema does not match')
    bank = _validate_bank(payload['bank'], layers, identity['width'])
    calibration = _validate_calibration(payload['calibration'], layers)
    return bank, calibration


def _write_atomic(path, payload, identity, fingerprint, layers):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_bundle(torch.load(temporary, map_location='cpu', weights_only=True), identity, fingerprint, layers)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def get_or_mint_definition_relations(model, layers, path):
    """Load once or perform one bounded mint before any generation hooks exist.

    Explicit files are replaced only after the fresh CPU profiles and their
    restricted serialization validate. A directory stores a distinct bank per
    model/tokenizer/precision/layer/recipe/learner identity. In-memory weight
    changes require a new model identity supplied by the caller.
    """
    requested = _requested_layers(layers)
    identity = _identity(model, requested)
    fingerprint = _digest(identity)
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != '.pt' or destination.is_dir():
        destination /= 'bank-' + fingerprint + '.pt'
    metadata = {'path': str(destination), 'version': RELATION_VERSION, 'identity': identity,
                'identity_hash': fingerprint, 'step_order': list(STEP_ORDER), 'gains': dict(STEP_GAINS),
                'provenance': _provenance(identity)}
    if destination.exists():
        try:
            bank, calibration = _validate_bundle(
                torch.load(destination, map_location='cpu', weights_only=True), identity, fingerprint, requested)
        except Exception as exc:
            metadata['cache_rejection'] = type(exc).__name__ + ': ' + str(exc)[:240]
        else:
            metadata.update(load_status='loaded', cache_status='hit', calibration=calibration,
                            profiles={step: {layer: profile['diagnostics'] for layer, profile in bank[step].items()}
                                      for step in STEP_ORDER})
            return bank, metadata
    bank, calibration = _mint(model, requested, identity['width'])
    payload = {'schema': _SCHEMA, 'identity': identity, 'identity_hash': fingerprint,
               'bank': bank, 'provenance': _provenance(identity), 'calibration': calibration}
    _write_atomic(destination, payload, identity, fingerprint, requested)
    metadata.update(load_status='minted', cache_status='minted', calibration=calibration,
                    profiles={step: {layer: profile['diagnostics'] for layer, profile in bank[step].items()}
                              for step in STEP_ORDER})
    return bank, metadata


@torch.no_grad()
def resolve_definition_relations(model, bank, step, evidence):
    """Predict a conditional direction and relative magnitude from actual E only."""
    if step not in STEP_ORDER:
        raise ValueError('Definition relation resolution requires a declared operation')
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError('Definition relation resolution requires nonempty evidence')
    if not isinstance(bank, dict) or step not in bank or not isinstance(bank[step], dict):
        raise ValueError('Definition relation resolution requires a complete profile bank')
    layers = _requested_layers(bank[step])
    validated = _validate_bank(bank, layers, int(model.d_model))
    source = _validated_states(_native_states(
        model, [{'role': 'user', 'content': evidence}], layers, step,
        add_generation_prompt=True), layers, int(model.d_model))
    directions = {}
    diagnostics = {'version': RELATION_VERSION, 'step': step,
                   'evidence_sha256': hashlib.sha256(evidence.encode('utf-8')).hexdigest(),
                   'source_format': SOURCE_FORMAT_VERSION, 'semantic_truth_checked': False,
                   'layers': {}, 'layer_scales': {}}
    for layer in layers:
        profile = validated[step][layer]
        delta, prediction = predict_relational_conversion(profile, source[layer])
        delta = delta.detach().to(device='cpu', dtype=torch.float32)
        norm = float(delta.norm())
        reference = float(profile['changes_norm_mean'])
        if delta.shape != (int(model.d_model),) or not torch.isfinite(delta).all() or not math.isfinite(norm) or norm <= 1e-8:
            raise ValueError('Definition relation prediction must be finite and nonzero')
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError('Definition relation prediction requires nonzero training change magnitude')
        scale = min(2., max(0., norm / reference))
        directions[layer] = delta / norm
        diagnostics['layer_scales'][layer] = scale
        diagnostics['layers'][layer] = {**prediction, 'raw_delta_norm': norm,
                                        'mean_training_change_norm': reference, 'layer_scale': scale}
    return directions, diagnostics
