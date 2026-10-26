"""Candidate genus/core/boundary/composition transports at native block outputs.

Completed synthetic demonstrations supply a prospective context. Every source
and paired teacher context ends at identical unanswered evidence and an exact
assistant prefix. Calibration contains exceptions and indeterminate cases;
no sign, magnitude, confidence, or exclusive membership rule is presumed.
"""
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

from invariants.probe_behavior import render_definition_evidence
from invariants.probe_definition_steps import _digest, _identity as _model_identity
from invariants.probe_definition_task import _requested_layers
from invariants.relational_conversion import (
    RELATIONAL_CONVERSION_VERSION, fit_relational_conversion,
    predict_relational_conversion, validate_relational_conversion,
)


SCHEMA_ORDER = ('genus', 'core', 'boundary', 'compose')
SCHEMA_GAINS = {role: .04 for role in SCHEMA_ORDER}
SCHEMA_VERSION = 'definition-schema-block-decision-candidate-v1'
DEFAULT_ASSISTANT_PREFIX = 'Within these examples, the behavioral distinction is a '
RIDGE = .01
EXTRAPOLATION = 1.5
_SCHEMA = 1
_THEMES = (
    dict(family='route selection policies', first='select the first available route', second='compare alternatives before selecting a route'),
    dict(family='measurement reporting practices', first='report an immediate estimate', second='check the sensor before reporting a measurement'),
    dict(family='draft publication practices', first='publish the initial draft', second='integrate a review before publishing the draft'),
    dict(family='task scheduling practices', first='start the next visible task', second='order prerequisites before starting a task'),
    dict(family='item classification practices', first='classify from surface resemblance', second='check defining criteria before classifying an item'),
    dict(family='fault repair practices', first='replace the visibly broken part', second='trace the cause before repairing a fault'),
    dict(family='meeting scheduling practices', first='choose the first proposed time', second='check participant availability before choosing a time', kind='exception'),
    dict(family='report revision practices', first='retain the previous reported value', second='replace the report with the supplied new value',
         kind='indeterminate', neutral='retain the previous report when no new value is supplied'),
)
_TEACHERS = (
    dict(family='inventory reconciliation practices', first='count packages as they arrive', second='reconcile the ledger before counting packages', kind='exception'),
    dict(family='translation practices', first='translate words separately', second='resolve context before translating words',
         kind='indeterminate', neutral='leave an isolated unambiguous label unchanged'),
)


def _case(theme, orientation, namespace=None):
    if orientation not in (0, 1):
        raise ValueError('Schema examples require one of two coordinate orientations')
    family, first, second = theme['family'], theme['first'], theme['second']
    kind = theme.get('kind', 'clean')
    if kind not in ('clean', 'exception', 'indeterminate'):
        raise ValueError('Schema example kind must be declared')
    alias = namespace or 'p_' + _digest(theme)[:24]
    readings = tuple((-1 if orientation else 1) * value for value in (-.75, -.25, .25, .75))
    actions = [first, first, second, second]
    contexts = [f'A routine case involving {family}.'] * 4
    if kind == 'exception':
        actions[3] = first
    elif kind == 'indeterminate':
        actions[1] = actions[3] = theme['neutral']
        contexts = [f'A discriminating case involving {family}.',
                    f'A case involving {family} without the information needed for the distinction.',
                    f'A discriminating case involving {family}.',
                    f'A case involving {family} without the information needed for the distinction.']
    selected = [(index, {'reading': reading, 'input': context, 'reply': f'I {action}.'})
                for index, (reading, context, action) in enumerate(zip(readings, contexts, actions))]
    selected.sort(key=lambda item: (item[1]['reading'], item[0]))
    evidence = render_definition_evidence(alias, selected)
    a, b = ((0, 2) if not orientation else (2, 0))
    low, high = actions[a], actions[b]
    genus = f'Memory 0, Memory 1, Memory 2, and Memory 3 are examples of {family}.'
    contrast = f'Memory {a} {low}, whereas Memory {b} {high}.'
    core = contrast + f' Their observable distinction is {first} versus {second}.'
    if kind == 'clean':
        boundary = ('Only these four examples are supplied. The two examples with each action have different '
                    'readings, so the observations establish an action contrast without measuring its intensity '
                    'or establishing an exclusive numeric cutoff.')
        boundary_facts = ('Memory 0 and Memory 1 have the same action and different readings. '
                          'Memory 2 and Memory 3 have the same action and different readings. '
                          'There are four supplied examples.')
    elif kind == 'exception':
        boundary = (f'Memory 3 {first}, like Memory 0 and Memory 1, despite belonging to the other measured end. '
                    'This observed exception prevents treating the contrast as an exclusive rule for either end.')
        boundary_facts = f'Memory 3 says "I {first}." Memory 0 and Memory 1 say "I {first}." Their readings place Memory 3 at the other end.'
    else:
        boundary = (f'Memory 1 and Memory 3 both {theme["neutral"]} at opposite measured ends. '
                    'Those cases do not distinguish the two actions because the required information is absent; '
                    'the action contrast is supported by Memory 0 and Memory 2.')
        boundary_facts = (f'Memory 1 says "I {theme["neutral"]}." Memory 3 says "I {theme["neutral"]}." '
                          'Their readings put them at opposite ends. Their inputs lack the information needed for the distinction.')
    scope = ('The relation is supported in these examples; it does not establish an exclusive cutoff.' if kind == 'clean' else
             'Memory 3 is an observed exception, so the end association is partial.' if kind == 'exception' else
             'Memory 1 and Memory 3 are indeterminate for this relation, so its scope is the two discriminating cases.')
    ordered = f'At lower readings, Memory {a} {low}, whereas at higher readings Memory {b} {high}.'
    compose = f'Within these examples, {alias} distinguishes {family} by the observed action contrast: {ordered} {scope}'
    outputs = dict(genus=genus, core=core, boundary=boundary, compose=compose)
    action_facts = f'Memory {a} says "I {low}." Memory {b} says "I {high}."'
    recaps = {
        'genus': ' '.join(f'Memory {index} concerns {family}.' for index in range(4)),
        'core': action_facts + f' The actions mentioned are "{first}" and "{second}".',
        'boundary': boundary_facts,
        'compose': f'The memories concern {family}. ' + action_facts + ' ' + boundary_facts,
    }
    return evidence, outputs, recaps


def _calibration_rows(role):
    if role not in SCHEMA_ORDER:
        raise ValueError('Definition schema requires a declared part')
    positive, negative = [], []
    for index, theme in enumerate(_TEACHERS):
        evidence, outputs, recaps = _case(theme, index % 2)
        positive.extend((dict(role='user', content=evidence), dict(role='assistant', content=outputs[role])))
        negative.extend((dict(role='user', content=evidence), dict(role='assistant', content=recaps[role])))
    rows = []
    for theme_index, theme in enumerate(_THEMES):
        for orientation in (0, 1):
            # Namespace spelling is a nuisance varied across calibration themes.
            namespace = 'p_' + hashlib.sha256(f'schema-query-{theme_index}'.encode()).hexdigest()[:24]
            evidence, _, _ = _case(theme, orientation, namespace)
            final = dict(role='user', content=evidence)
            rows.append(dict(theme=theme_index, orientation=orientation, evidence=evidence,
                             positive=[dict(item) for item in positive] + [dict(final)],
                             negative=[dict(item) for item in negative] + [dict(final)]))
    return rows


def _prefix(value):
    if not isinstance(value, str) or (value and not value.strip()):
        raise ValueError('Definition schema requires a string exact assistant prefix; whitespace-only prefixes are invalid')
    return value


@torch.no_grad()
def _decision_states(model, messages, layers, assistant_prefix):
    """Read block outputs at the exact prefix, or clean header when it is empty."""
    from invariants.engine import _inputs
    prefix = _prefix(assistant_prefix)
    if not messages or messages[-1].get('role') != 'user':
        raise ValueError('Definition schema native contexts must end at an unanswered user query')
    formatted = model.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not isinstance(formatted, str) or not formatted:
        raise ValueError('Definition schema requires a native text chat template')
    inputs = _inputs(model, formatted + prefix, pre_formatted=True)
    if inputs['input_ids'].ndim != 2 or inputs['input_ids'].shape[0] != 1 or inputs['input_ids'].shape[-1] == 0:
        raise ValueError('Definition schema requires one nonempty native prompt')
    blocks = model.model.model.layers
    if any(layer >= len(blocks) for layer in layers):
        raise ValueError('Definition schema is missing a requested block')
    states, handles = {}, []
    try:
        for layer in layers:
            def capture(_module, _inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(hidden) or hidden.ndim != 3 or hidden.shape[0] != 1:
                    raise ValueError('Definition schema requires native token block outputs')
                state = hidden[0, -1].detach().to(device='cpu', dtype=torch.float32).clone()
                if state.shape != (int(model.d_model),) or not torch.isfinite(state).all():
                    raise ValueError('Definition schema requires finite matching block output widths')
                states[layer] = state
            handles.append(blocks[layer].register_forward_hook(capture))
        model.model(**inputs, use_cache=False, output_hidden_states=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(states) != set(layers):
        raise ValueError('Definition schema did not observe every requested block')
    return states


def _identity(model, layers, assistant_prefix):
    identity = _model_identity(model, layers)
    identity.update(version=SCHEMA_VERSION,
                    recipe_sha256=_digest({role: _calibration_rows(role) for role in SCHEMA_ORDER}),
                    learner_version=RELATIONAL_CONVERSION_VERSION, ridge=RIDGE, extrapolation=EXTRAPOLATION,
                    assistant_prefix=_prefix(assistant_prefix), source_format='original-evidence-native-prefix-block-output-v2',
                    source_units='raw_block_output', target_units='raw_block_output_difference')
    return identity


def validate_schema_bank(bank, layers, width):
    requested = _requested_layers(layers)
    if not isinstance(bank, dict) or set(bank) != set(SCHEMA_ORDER):
        raise ValueError('Definition schema bank must contain exactly all four parts')
    validated = {}
    for role in SCHEMA_ORDER:
        profiles = bank[role]
        if (not isinstance(profiles, dict) or set(profiles) != set(requested) or
                any(type(layer) is not int for layer in profiles)):
            raise ValueError('Definition schema bank must cover exactly the requested blocks')
        validated[role] = {}
        for layer in requested:
            profile = validate_relational_conversion(profiles[layer], expected_width=width)
            if profile['n_samples'] != 16 or profile['ridge'] != RIDGE or profile['extrapolation'] != EXTRAPOLATION:
                raise ValueError('Definition schema profile has incompatible calibration or learner settings')
            if not profile['diagnostics']['adaptive'] or profile['changes_norm_mean'] <= 0:
                raise ValueError('Definition schema profile must retain nonzero conditional variation')
            validated[role][layer] = profile
    return validated


@torch.no_grad()
def _mint(model, layers, assistant_prefix):
    bank, holdouts = {}, {}
    for role in SCHEMA_ORDER:
        sources, changes = {layer: [] for layer in layers}, {layer: [] for layer in layers}
        for row in _calibration_rows(role):
            source = _decision_states(model, [dict(role='user', content=row['evidence'])], layers, assistant_prefix)
            positive = _decision_states(model, row['positive'], layers, assistant_prefix)
            negative = _decision_states(model, row['negative'], layers, assistant_prefix)
            for layer in layers:
                sources[layer].append(source[layer])
                changes[layer].append(positive[layer] - negative[layer])
        bank[role], holdouts[role] = {}, {}
        for layer in layers:
            x, y = torch.stack(sources[layer]), torch.stack(changes[layer])
            bank[role][layer] = fit_relational_conversion(x, y, ridge=RIDGE, extrapolation=EXTRAPOLATION)
            training = fit_relational_conversion(x[:-2], y[:-2], ridge=RIDGE, extrapolation=EXTRAPOLATION)
            predictions = torch.stack([predict_relational_conversion(training, source)[0] for source in x[-2:]])
            holdouts[role][layer] = dict(heldout_rows=2, heldout_theme=7,
                error_rms=float((predictions - y[-2:]).square().mean().sqrt()),
                observed_change_distance=float((y[-1] - y[-2]).norm()),
                predicted_change_distance=float((predictions[-1] - predictions[-2]).norm()),
                semantic_truth_checked=False)
    return validate_schema_bank(bank, layers, int(model.d_model)), holdouts


def _provenance(identity):
    return dict(endpoint='native_block_output_after_exact_assistant_prefix', calibration_rows_per_part=16,
                teacher_pairs=2, recipe_sha256=identity['recipe_sha256'], status='candidate',
                semantic_truth_checked=False, calibration_source='unprimed_original_evidence',
                online_intervention_source_shift='experimental_unverified')


def _validate_holdouts(holdouts, layers):
    if not isinstance(holdouts, dict) or set(holdouts) != set(SCHEMA_ORDER):
        raise ValueError('Schema holdouts must cover all parts')
    keys = {'heldout_rows', 'heldout_theme', 'error_rms', 'observed_change_distance',
            'predicted_change_distance', 'semantic_truth_checked'}
    for summaries in holdouts.values():
        if not isinstance(summaries, dict) or set(summaries) != set(layers):
            raise ValueError('Schema holdouts must cover every requested block')
        for summary in summaries.values():
            if (not isinstance(summary, dict) or set(summary) != keys or summary['heldout_rows'] != 2 or
                    summary['heldout_theme'] != 7 or summary['semantic_truth_checked'] is not False):
                raise ValueError('Schema holdout scope does not match')
            for key in ('error_rms', 'observed_change_distance', 'predicted_change_distance'):
                value = summary[key]
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                    raise ValueError('Schema holdout diagnostics must be finite and nonnegative')
    return holdouts


def _validate_bundle(payload, identity, fingerprint, layers):
    keys = {'schema', 'identity', 'identity_hash', 'bank', 'provenance', 'holdouts'}
    if (not isinstance(payload, dict) or set(payload) != keys or payload['schema'] != _SCHEMA or
            payload['identity'] != identity or payload['identity_hash'] != fingerprint or
            payload['provenance'] != _provenance(identity)):
        raise ValueError('Definition schema storage identity or provenance does not match')
    bank = validate_schema_bank(payload['bank'], layers, identity['width'])
    return bank, _validate_holdouts(payload['holdouts'], layers)


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


def get_or_mint_definition_schema(model, layers, path, *, assistant_prefix=DEFAULT_ASSISTANT_PREFIX):
    requested = _requested_layers(layers)
    identity = _identity(model, requested, assistant_prefix)
    fingerprint = _digest(identity)
    destination = Path(path).expanduser().resolve()
    if destination.suffix.lower() != '.pt' or destination.is_dir():
        destination /= 'bank-' + fingerprint + '.pt'
    metadata = dict(path=str(destination), version=SCHEMA_VERSION, identity=identity,
                    identity_hash=fingerprint, part_order=list(SCHEMA_ORDER), gains=dict(SCHEMA_GAINS),
                    provenance=_provenance(identity))
    if destination.exists():
        try:
            bank, holdouts = _validate_bundle(torch.load(destination, weights_only=True, map_location='cpu'),
                                              identity, fingerprint, requested)
        except Exception as exc:
            metadata['cache_rejection'] = type(exc).__name__ + ': ' + str(exc)[:240]
        else:
            metadata.update(load_status='loaded', cache_status='hit', holdouts=holdouts)
            return bank, metadata
    bank, holdouts = _mint(model, requested, assistant_prefix)
    payload = dict(schema=_SCHEMA, identity=identity, identity_hash=fingerprint, bank=bank,
                   provenance=_provenance(identity), holdouts=holdouts)
    _write_atomic(destination, payload, identity, fingerprint, requested)
    metadata.update(load_status='minted', cache_status='minted', holdouts=holdouts)
    return bank, metadata
