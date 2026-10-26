"""Remember settings and prospective state changes associated with definitions.

Output qualification concerns definition form only. A successful conditioned
decision state is evidence for a candidate intervention, not proof of its
causal contribution or of the inferred meaning. Completed answers are never
encoded as prospective targets, and replay outcomes do not train themselves.
"""
import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

from invariants.definition_prefill import prepare_assistant_continuation
from invariants.probe_definition_output import (
    OUTPUT_PREFIX, OUTPUT_VERSION, build_definition_request,
)
from invariants.probe_definition_steps import _digest, _identity, _plain
from invariants.probe_definition_task import _requested_layers
from invariants.relational_conversion import (
    fit_relational_conversion, predict_relational_conversion,
)


PROFILE_VERSION = 'defining-profile-output-associated-decision-v1'
CAPTURE_PHASE = 'prospective-last-prefill-layer-block-output'
MAX_REPLAY_GAIN = .08
MAX_FAMILIES = 8
MAX_HISTORY = 64
MAX_SAMPLES = 32
MAX_RECIPES = 8
_SCHEMA = 1
_EPS = 1e-8
_SENTINEL = '__defining_profile_evidence__'
_BASE_SUFFIX = build_definition_request(_SENTINEL)[len(_SENTINEL):]
_CORRECTION = '\n\n[Format correction]\n'
_HASH = set('0123456789abcdef')
_IDENTITY_KEYS = {
    'version', 'recipe_sha256', 'model', 'tokenizer', 'effective_load', 'width',
    'model_layers', 'layers', 'libraries', 'output_version', 'output_prefix',
    'capture_phase',
}


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _hash(value):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HASH:
        raise ValueError('Defining profile requires SHA-256 identifiers')
    return value


def _integer(value, minimum=0, maximum=1_000_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('Defining profile has an invalid bounded integer')
    return value


def _text(value, maximum=4096, *, nonempty=False):
    if not isinstance(value, str) or len(value) > maximum or (nonempty and not value.strip()):
        raise ValueError('Defining profile has invalid text metadata')
    return value


def _feedback(value):
    if value is None:
        return None
    _text(value, 4096, nonempty=True)
    return ' '.join(value.split())[:240]


def _request_suffix(feedback=None):
    request = build_definition_request(_SENTINEL, _feedback(feedback))
    return request[len(_SENTINEL):]


def _validate_suffix(suffix, reason):
    if suffix != _request_suffix(reason):
        raise ValueError('Defining profile recipe differs from the scoped output request')
    return suffix


def _settings(settings):
    if not isinstance(settings, dict):
        raise ValueError('Defining profile settings must be a mapping')
    value = _plain(settings)
    if len(str(value)) > 16000:
        raise ValueError('Defining profile settings metadata is too large')
    # Canonical serialization also rejects unsupported or nonfinite values.
    _digest(value)
    return value


def _profile_identity(model, layers):
    value = _identity(model, layers)
    value.update(version=PROFILE_VERSION, recipe_sha256=_digest(_BASE_SUFFIX),
                 output_version=OUTPUT_VERSION, output_prefix=OUTPUT_PREFIX,
                 capture_phase=CAPTURE_PHASE)
    return _validate_identity(value)


def _validate_identity(identity):
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise ValueError('Defining profile model identity has an incompatible schema')
    identity = _plain(identity)
    if (identity['version'] != PROFILE_VERSION or
            identity['recipe_sha256'] != _digest(_BASE_SUFFIX) or
            identity['output_version'] != OUTPUT_VERSION or
            identity['output_prefix'] != OUTPUT_PREFIX or
            identity['capture_phase'] != CAPTURE_PHASE):
        raise ValueError('Defining profile identity uses an incompatible request or capture phase')
    width = _integer(identity['width'], 1, 131072)
    count = _integer(identity['model_layers'], 1, 4096)
    layers = _requested_layers(identity['layers'])
    if list(layers) != identity['layers'] or len(layers) > 256 or layers[-1] >= count:
        raise ValueError('Defining profile identity has incompatible layers')
    for name in ('model', 'tokenizer', 'effective_load', 'libraries'):
        if not isinstance(identity[name], dict):
            raise ValueError('Defining profile identity metadata must be mappings')
    if len(str(identity)) > 160000:
        raise ValueError('Defining profile identity metadata is too large')
    return identity


def _vector(value, width):
    if (not torch.is_tensor(value) or value.dtype != torch.float32 or
            value.device.type != 'cpu' or value.shape != (width,) or
            not torch.isfinite(value).all()):
        raise ValueError('Defining profile states must be finite CPU float32 residual vectors')
    return value.detach().contiguous().clone()


def _states(states, layers, width):
    if (not isinstance(states, dict) or set(states) != set(layers) or
            any(type(layer) is not int for layer in states)):
        raise ValueError('Defining profile states must cover exactly the identified layers')
    return {layer: _vector(states[layer], width) for layer in layers}


def _tensor_hash(vector):
    return hashlib.sha256(vector.detach().contiguous().numpy().tobytes()).hexdigest()


def _norm(vector):
    # Finite float32 elements can overflow a float32 norm during diagnostics.
    return float(vector.detach().double().norm())


def _sample_id(sample, layers):
    return _digest({
        'namespace': sample['namespace'], 'evidence_sha256': sample['evidence_sha256'],
        'recipe_id': sample['recipe_id'],
        'sources': {layer: _tensor_hash(sample['sources'][layer]) for layer in layers},
        'changes': {layer: _tensor_hash(sample['changes'][layer]) for layer in layers},
    })


def _validate_recipe(recipe):
    required = {'id', 'suffix', 'correction_reason', 'settings', 'settings_sha256',
                'successes', 'failures'}
    if not isinstance(recipe, dict) or set(recipe) != required:
        raise ValueError('Defining profile recipe has an incompatible schema')
    reason = _feedback(recipe['correction_reason'])
    if reason != recipe['correction_reason']:
        raise ValueError('Defining profile correction reason is not canonical')
    suffix = _validate_suffix(recipe['suffix'], reason)
    settings = _settings(recipe['settings'])
    settings_hash = _digest(settings)
    recipe_id = _digest({'suffix': suffix, 'settings_sha256': settings_hash})
    if recipe['id'] != recipe_id or recipe['settings_sha256'] != settings_hash:
        raise ValueError('Defining profile recipe identity does not match its settings')
    return dict(recipe, settings=settings,
                successes=_integer(recipe['successes']), failures=_integer(recipe['failures']))


def _validate_history(history):
    required = {'namespace', 'evidence_sha256', 'request_sha256', 'output_sha256',
                'settings', 'settings_sha256', 'suffix', 'correction_reason',
                'mode', 'attempt', 'status', 'accepted', 'reason', 'tokens_generated',
                'budget', 'capture', 'structural_only', 'semantic_truth_checked',
                'causal_effect_checked'}
    if not isinstance(history, list) or len(history) > MAX_HISTORY:
        raise ValueError('Defining profile attempt history is not bounded')
    result = []
    for item in history:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError('Defining profile attempt history has an incompatible schema')
        item = _plain(item)
        for name in ('evidence_sha256', 'request_sha256', 'output_sha256', 'settings_sha256'):
            _hash(item[name])
        _text(item['namespace'], 512, nonempty=True)
        _text(item['reason'], 512)
        if (_settings(item['settings']) != item['settings'] or
                _digest(item['settings']) != item['settings_sha256']):
            raise ValueError('Defining profile attempt settings identity does not match')
        if item['mode'] not in ('explicit', 'replay') or item['status'] not in ('hypothesis', 'unqualified', 'incomplete'):
            raise ValueError('Defining profile attempt outcome is invalid')
        if type(item['accepted']) is not bool or item['accepted'] != (item['status'] == 'hypothesis'):
            raise ValueError('Defining profile qualification and status disagree')
        _integer(item['attempt'], 1, 8)
        _integer(item['tokens_generated'], 0, 1_000_000)
        _integer(item['budget'], 1, 1_000_000)
        if (item['structural_only'] is not True or item['semantic_truth_checked'] is not False or
                item['causal_effect_checked'] is not False):
            raise ValueError('Defining profile must preserve qualification limits')
        if not isinstance(item['capture'], dict) or len(str(item['capture'])) > 16000:
            raise ValueError('Defining profile capture diagnostics are invalid')
        if item['mode'] == 'explicit':
            _validate_suffix(item['suffix'], item['correction_reason'])
        elif item['suffix'] != '' or item['correction_reason'] is not None:
            raise ValueError('Defining profile replay must use evidence without a repeated request')
        result.append(item)
    return result


def _validate_sample(sample, identity, recipes):
    required = {'sample_id', 'namespace', 'evidence_sha256', 'recipe_id', 'settings_sha256',
                'sources', 'changes', 'structural_only', 'semantic_truth_checked', 'causal_effect_checked'}
    if not isinstance(sample, dict) or set(sample) != required:
        raise ValueError('Defining profile sample has an incompatible schema')
    layers, width = identity['layers'], identity['width']
    sources = _states(sample['sources'], layers, width)
    changes = _states(sample['changes'], layers, width)
    if (any(_norm(value) <= _EPS for value in sources.values()) or
            _norm(changes[layers[-1]]) <= _EPS):
        raise ValueError('Defining profile samples require nonzero source and late-layer change')
    _text(sample['namespace'], 512, nonempty=True)
    _hash(sample['evidence_sha256'])
    recipe = recipes.get(sample['recipe_id'])
    if (recipe is None or recipe['successes'] < 1 or
            recipe['settings_sha256'] != sample['settings_sha256']):
        raise ValueError('Defining profile sample lacks a qualified explicit recipe')
    if (sample['structural_only'] is not True or sample['semantic_truth_checked'] is not False or
            sample['causal_effect_checked'] is not False):
        raise ValueError('Defining profile sample cannot certify meaning or causality')
    value = dict(sample, sources=sources, changes=changes)
    if sample['sample_id'] != _sample_id(value, layers):
        raise ValueError('Defining profile sample identity does not match its measured states')
    return value


def _validate_bundle(payload):
    if (not isinstance(payload, dict) or
            set(payload) != {'schema', 'version', 'enabled', 'families', 'unidentified_history'} or
            type(payload['schema']) is not int or payload['schema'] != _SCHEMA or
            payload['version'] != PROFILE_VERSION or type(payload['enabled']) is not bool):
        raise ValueError('Defining profile storage schema does not match')
    families = payload['families']
    if not isinstance(families, dict) or len(families) > MAX_FAMILIES:
        raise ValueError('Defining profile model families are not bounded')
    validated = {}
    for fingerprint, family in families.items():
        _hash(fingerprint)
        if (not isinstance(family, dict) or
                set(family) != {'identity', 'identity_hash', 'recipes', 'samples', 'history', 'replay_stats'}):
            raise ValueError('Defining profile family has an incompatible schema')
        identity = _validate_identity(family['identity'])
        if fingerprint != _digest(identity) or family['identity_hash'] != fingerprint:
            raise ValueError('Defining profile model identity does not match')
        if not isinstance(family['recipes'], dict) or len(family['recipes']) > MAX_RECIPES:
            raise ValueError('Defining profile recipes are not bounded')
        recipes = {}
        for recipe_id, recipe in family['recipes'].items():
            value = _validate_recipe(recipe)
            if recipe_id != value['id']:
                raise ValueError('Defining profile recipe key does not match')
            recipes[recipe_id] = value
        if not isinstance(family['samples'], list) or len(family['samples']) > MAX_SAMPLES:
            raise ValueError('Defining profile successful samples are not bounded')
        samples = [_validate_sample(sample, identity, recipes) for sample in family['samples']]
        if len({sample['sample_id'] for sample in samples}) != len(samples):
            raise ValueError('Defining profile samples must be distinct')
        stats = family['replay_stats']
        if not isinstance(stats, dict) or set(stats) != {'successes', 'failures'}:
            raise ValueError('Defining profile replay outcomes are invalid')
        validated[fingerprint] = dict(
            identity=identity, identity_hash=fingerprint, recipes=recipes, samples=samples,
            history=_validate_history(family['history']),
            replay_stats={key: _integer(value) for key, value in stats.items()})
    return dict(schema=_SCHEMA, version=PROFILE_VERSION, enabled=payload['enabled'],
                families=validated, unidentified_history=_validate_history(payload['unidentified_history']))


@torch.no_grad()
def _capture_states(model, prompt, layers, width):
    """Collect raw block outputs at the last prefill token; keep no token cloud."""
    from invariants.engine import _inputs

    states, handles = {}, []
    try:
        blocks = model.model.model.layers
        for layer in layers:
            def collect(_module, _inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                if (not torch.is_tensor(hidden) or hidden.ndim != 3 or
                        hidden.shape[0] != 1 or hidden.shape[1] == 0 or hidden.shape[-1] != width):
                    raise ValueError('Defining profile capture requires one native residual sequence')
                states[layer] = hidden[0, -1].detach().to(device='cpu', dtype=torch.float32).clone()
            handles.append(blocks[layer].register_forward_hook(collect))
        inputs = _inputs(model, prompt, pre_formatted=True)
        model.model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return _states(states, layers, width)


class DefiningProfile:
    """Tuner-owned, bounded settings outcomes and successful prospective pairs."""

    def __init__(self, path=None):
        self.path = Path(path).expanduser().resolve() if path is not None else None
        self.enabled = True
        self.families = {}
        self.unidentified_history = []
        self.persistence_error = None
        if self.path is not None and self.path.exists():
            try:
                payload = _validate_bundle(torch.load(self.path, map_location='cpu', weights_only=True))
            except Exception as exc:
                # The caller can use an in-memory fallback. The corrupt file is
                # retained intact; construction never overwrites saved evidence.
                raise ValueError(f'Defining profile could not be loaded safely: {type(exc).__name__}: {exc}') from exc
            self.enabled = payload['enabled']
            self.families = payload['families']
            self.unidentified_history = payload['unidentified_history']

    def _payload(self):
        return dict(schema=_SCHEMA, version=PROFILE_VERSION, enabled=self.enabled,
                    families=self.families, unidentified_history=self.unidentified_history)

    def save(self):
        payload = _validate_bundle(self._payload())
        if self.path is None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.' + self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            _validate_bundle(torch.load(temporary, map_location='cpu', weights_only=True))
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return True

    def session(self, model, layers, *, evidence, namespace, settings):
        return DefiningSession(self, model, layers, evidence=evidence,
                               namespace=namespace, settings=settings)

    def summary(self):
        successes = sum(recipe['successes'] for family in self.families.values()
                        for recipe in family['recipes'].values())
        failures_explicit = sum(recipe['failures'] for family in self.families.values()
                                for recipe in family['recipes'].values())
        successes += sum(item['accepted'] for item in self.unidentified_history)
        failures_explicit += sum(not item['accepted'] for item in self.unidentified_history)
        samples = sum(len(family['samples']) for family in self.families.values())
        replayed = sum(family['replay_stats']['successes'] for family in self.families.values())
        failures = sum(family['replay_stats']['failures'] for family in self.families.values())
        persistence = f' Save unavailable: {self.persistence_error}' if self.persistence_error else ''
        return (f'[Defining profile] {"on" if self.enabled else "off"}; '
                f'{successes} qualified / {failures_explicit} failed explicit attempts, {samples} prospective samples, '
                f'{replayed} qualified replays / {failures} failed replays across {len(self.families)} model families. '
                'Associations concern definition form; meaning and causal contribution remain unchecked.' + persistence)


class DefiningSession:
    def __init__(self, profile, model, layers, *, evidence, namespace, settings):
        build_definition_request(evidence)
        self.profile, self.model = profile, model
        self.evidence = evidence
        self.namespace = _text(namespace, 512, nonempty=True)
        self.settings = _settings(settings)
        self.settings_hash = _digest(self.settings)
        self.evidence_hash = _sha(evidence)
        self.layers = ()
        self.family = None
        self.identity_hash = None
        self.family_status = 'unavailable'
        self.capture_error = None
        self._baseline = None
        self._pending = None
        self._steering = None
        self._attempts = []
        self._replay_diagnostics = {'ready': False, 'reason': 'Fewer than three distinct successful source pairs.'}
        try:
            self.layers = _requested_layers(layers)
            identity = _profile_identity(model, self.layers)
            self.width = identity['width']
            self.identity_hash = _digest(identity)
            self.family = profile.families.get(self.identity_hash)
            if self.family is None:
                self.family = dict(identity=identity, identity_hash=self.identity_hash,
                                   recipes={}, samples=[], history=[], replay_stats={'successes': 0, 'failures': 0})
                profile.families[self.identity_hash] = self.family
                while len(profile.families) > MAX_FAMILIES:
                    del profile.families[next(iter(profile.families))]
                self.family_status = 'new'
            else:
                self.family_status = 'compatible'
        except Exception as exc:
            self.capture_error = f'{type(exc).__name__}: {exc}'[:512]
            self._replay_diagnostics = {'ready': False, 'reason': 'Model identity is unavailable.'}

    def _baseline_states(self):
        if self._baseline is None:
            prompt, _ = prepare_assistant_continuation(self.model, self.evidence, OUTPUT_PREFIX)
            self._baseline = _capture_states(self.model, prompt, self.layers, self.width)
        return self._baseline

    def _preferred_recipe(self):
        if self.family is None or not self.profile.enabled:
            return None
        compatible = [recipe for recipe in self.family['recipes'].values()
                      if recipe['successes'] > 0 and recipe['settings_sha256'] == self.settings_hash]
        return max(compatible, key=lambda recipe: (recipe['successes'], -recipe['failures']), default=None)

    def _candidate_steering(self):
        if not self.profile.enabled or self.family is None or self.capture_error:
            return None
        late = self.layers[-1]
        # Repeated sources do not supply new source relationships. Keep the
        # most recent accepted explicit change for each exact raw late source.
        distinct = {}
        for sample in self.family['samples']:
            if sample['settings_sha256'] == self.settings_hash:
                distinct[_tensor_hash(sample['sources'][late])] = sample
        values = list(distinct.values())
        if len(values) < 3:
            self._replay_diagnostics = {'ready': False, 'reason': 'Fewer than three distinct successful source pairs.',
                                        'distinct_sources': len(values)}
            return None
        try:
            sources = torch.stack([sample['sources'][late] for sample in values])
            changes = torch.stack([sample['changes'][late] for sample in values])
            conversion = fit_relational_conversion(sources, changes, ridge=.01, extrapolation=1.5)
            if not conversion['diagnostics']['adaptive']:
                self._replay_diagnostics = {'ready': False, 'reason': 'Successful pairs do not identify an adaptive change.',
                                            'distinct_sources': len(values)}
                return None
            source = self._baseline_states()[late]
            change, diagnostics = predict_relational_conversion(conversion, source)
            source_norm, norm = _norm(source), _norm(change)
            if (not math.isfinite(source_norm) or source_norm <= _EPS or
                    not math.isfinite(norm) or norm <= _EPS):
                raise ValueError('Defining profile predicted source and change must be nonzero and finite')
            raw_gain = norm / source_norm
            gain = min(MAX_REPLAY_GAIN, raw_gain)
            if not math.isfinite(gain) or gain <= 0:
                raise ValueError('Defining profile predicted gain must be finite and positive')
            self._replay_diagnostics = dict(
                diagnostics, ready=True, distinct_sources=len(values), layer=late,
                source_norm=source_norm, raw_change_norm=norm, raw_gain=raw_gain,
                gain=gain, gain_cap=MAX_REPLAY_GAIN, gain_capped=raw_gain > MAX_REPLAY_GAIN,
                capture_phase=CAPTURE_PHASE, structural_only=True,
                semantic_truth_checked=False, causal_effect_checked=False)
            return {'layer': late, 'direction': (change / norm).detach().cpu().float(),
                    'gain': gain, 'diagnostics': dict(self._replay_diagnostics)}
        except Exception as exc:
            self.capture_error = f'{type(exc).__name__}: {exc}'[:512]
            self._replay_diagnostics = {'ready': False, 'reason': 'Candidate capture or conversion was unavailable.',
                                        'error': self.capture_error}
            return None

    def request(self, evidence, feedback, attempt_index):
        if evidence != self.evidence:
            raise ValueError('Defining profile session cannot replace its measured evidence')
        _integer(attempt_index, 0, 7)
        reason = _feedback(feedback)
        self._steering = self._candidate_steering() if attempt_index == 0 and reason is None else None
        if self._steering is not None:
            request, mode, suffix, reason = evidence, 'replay', '', None
        else:
            recipe = self._preferred_recipe()
            if reason is None and recipe is not None:
                suffix, reason = recipe['suffix'], recipe['correction_reason']
            else:
                # A new checker reason replaces an old format correction. No
                # rejected generated text becomes the next request or evidence.
                suffix = _request_suffix(reason)
            request, mode = evidence + suffix, 'explicit'
        self._pending = {'request': request, 'suffix': suffix, 'correction_reason': reason,
                         'mode': mode, 'attempt': attempt_index + 1, 'sample': None,
                         'capture': {'phase': CAPTURE_PHASE, 'status': 'not_captured'}}
        return request

    def capture(self, request):
        if self._pending is None or request != self._pending['request']:
            raise ValueError('Defining profile capture must match the current prospective request')
        pending = self._pending
        if self.family is None or self.capture_error:
            pending['capture'] = {'phase': CAPTURE_PHASE, 'status': 'unavailable',
                                  'reason': self.capture_error or 'Model identity is unavailable.'}
            return
        try:
            source = self._baseline_states()
            if pending['mode'] == 'replay':
                pending['capture'] = {'phase': CAPTURE_PHASE, 'status': 'replay_candidate',
                                      'training_excluded': True, 'replay': dict(self._replay_diagnostics)}
                return
            prompt, _ = prepare_assistant_continuation(self.model, request, OUTPUT_PREFIX)
            target = _capture_states(self.model, prompt, self.layers, self.width)
            changes = {layer: (target[layer] - source[layer]).float() for layer in self.layers}
            changes = _states(changes, self.layers, self.width)
            details = {str(layer): {'source_norm': _norm(source[layer]),
                                   'target_norm': _norm(target[layer]),
                                   'change_norm': _norm(changes[layer])} for layer in self.layers}
            valid = (all(math.isfinite(item['source_norm']) and item['source_norm'] > _EPS
                         and math.isfinite(item['target_norm']) and math.isfinite(item['change_norm'])
                         for item in details.values()) and
                     details[str(self.layers[-1])]['change_norm'] > _EPS)
            pending['capture'] = {'phase': CAPTURE_PHASE, 'status': 'captured' if valid else 'degenerate',
                                  'layers': details, 'structural_only': True,
                                  'semantic_truth_checked': False, 'causal_effect_checked': False}
            if valid:
                pending['sample'] = {'sources': source, 'changes': changes}
        except Exception as exc:
            self.capture_error = f'{type(exc).__name__}: {exc}'[:512]
            self._steering = None
            pending['capture'] = {'phase': CAPTURE_PHASE, 'status': 'unavailable', 'reason': self.capture_error}

    def steering(self):
        if self._pending is None or self._pending['mode'] != 'replay' or self._steering is None:
            return None
        return dict(self._steering, direction=self._steering['direction'].clone(),
                    diagnostics=dict(self._steering['diagnostics']))

    def observe(self, record):
        if self._pending is None:
            raise ValueError('Defining profile outcome has no prospective request')
        if not isinstance(record, dict) or not isinstance(record.get('qualification'), dict):
            raise ValueError('Defining profile requires the actual output qualification record')
        pending = self._pending
        status = record.get('status')
        if status not in ('hypothesis', 'unqualified', 'incomplete'):
            raise ValueError('Defining profile requires an actual definition output status')
        accepted = status == 'hypothesis' and record['qualification'].get('accepted') is True
        if status == 'hypothesis' and not accepted:
            raise ValueError('Defining profile output status disagrees with qualification')
        request_hash = _sha(pending['request'])
        if record.get('request_sha256', request_hash) != request_hash:
            raise ValueError('Defining profile outcome request does not match its captured request')
        item = {
            'namespace': self.namespace, 'evidence_sha256': self.evidence_hash,
            'request_sha256': request_hash, 'output_sha256': _sha(str(record.get('text', ''))),
            'settings': self.settings, 'settings_sha256': self.settings_hash,
            'suffix': pending['suffix'], 'correction_reason': pending['correction_reason'],
            'mode': pending['mode'], 'attempt': pending['attempt'], 'status': status,
            'accepted': accepted, 'reason': str(record['qualification'].get('reason', ''))[:512],
            'tokens_generated': _integer(record.get('tokens_generated', 0), 0, 1_000_000),
            'budget': _integer(record.get('budget', self.settings.get('max_tokens', 384)), 1, 1_000_000),
            'capture': pending['capture'], 'structural_only': True,
            'semantic_truth_checked': False, 'causal_effect_checked': False,
        }
        _validate_history([item])
        self._attempts.append(item)
        if self.family is None:
            self.profile.unidentified_history.append(item)
            self.profile.unidentified_history = self.profile.unidentified_history[-MAX_HISTORY:]
        else:
            family = self.family
            family['history'].append(item)
            family['history'] = family['history'][-MAX_HISTORY:]
            if pending['mode'] == 'replay':
                family['replay_stats']['successes' if accepted else 'failures'] += 1
            else:
                recipe_id = _digest({'suffix': pending['suffix'], 'settings_sha256': self.settings_hash})
                recipe = family['recipes'].setdefault(recipe_id, {
                    'id': recipe_id, 'suffix': pending['suffix'],
                    'correction_reason': pending['correction_reason'], 'settings': self.settings,
                    'settings_sha256': self.settings_hash, 'successes': 0, 'failures': 0})
                recipe['successes' if accepted else 'failures'] += 1
                if accepted and pending['sample'] is not None:
                    sample = dict(pending['sample'], namespace=self.namespace,
                                  evidence_sha256=self.evidence_hash, recipe_id=recipe_id,
                                  settings_sha256=self.settings_hash, structural_only=True,
                                  semantic_truth_checked=False, causal_effect_checked=False)
                    sample['sample_id'] = _sample_id(sample, self.layers)
                    # Identical observations can increase recipe success counts
                    # without pretending to provide independent state pairs.
                    if all(old['sample_id'] != sample['sample_id'] for old in family['samples']):
                        family['samples'].append(sample)
                    family['samples'] = family['samples'][-MAX_SAMPLES:]
                if len(family['recipes']) > MAX_RECIPES:
                    keep = sorted(family['recipes'].values(),
                                  key=lambda value: (value['successes'], -value['failures']), reverse=True)[:MAX_RECIPES]
                    family['recipes'] = {value['id']: value for value in keep}
                    family['samples'] = [sample for sample in family['samples']
                                         if sample['recipe_id'] in family['recipes']]
        self._pending = None
        self._steering = None
        try:
            self.profile.save()
            self.profile.persistence_error = None
        except Exception as exc:
            # An unavailable archive must not discard an already qualified
            # definition. Atomic save leaves the preceding file intact and the
            # current observations remain available in this owner/session.
            self.profile.persistence_error = f'{type(exc).__name__}: {exc}'[:512]

    def needs_explicit_fallback(self, attempt):
        """Allow one explicit retry after an observed truncated learned replay.

        This predicate runs after the actual output gate and observation. A
        truncated explicit request still stops; pending, stale, or mismatched
        records cannot authorize a retry based on another attempt's outcome.
        """
        if self._pending is not None or not self._attempts or not isinstance(attempt, dict):
            return False
        latest = self._attempts[-1]
        return (latest['mode'] == 'replay' and latest['status'] == 'incomplete' and
                attempt.get('status') == 'incomplete' and
                attempt.get('request_sha256') == latest['request_sha256'])

    def metadata(self):
        return _plain({
            'version': PROFILE_VERSION, 'enabled': self.profile.enabled,
            'identity_hash': self.identity_hash, 'family_status': self.family_status,
            'profile_path': str(self.profile.path) if self.profile.path is not None else None,
            'capture_phase': CAPTURE_PHASE, 'capture_error': self.capture_error,
            'persistence_error': self.profile.persistence_error,
            'successful_samples': len(self.family['samples']) if self.family is not None else 0,
            'recipes': len(self.family['recipes']) if self.family is not None else 0,
            'attempts': self._attempts, 'replay': self._replay_diagnostics,
            'structural_only': True, 'semantic_truth_checked': False, 'causal_effect_checked': False,
        })
