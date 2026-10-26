"""Separate, descriptive probe diagnostics and an opt-in measured-record watch.

The ledger revises lexical candidates, never a probe direction or its name.
Vocabulary projections and measured text stay separate channels: agreement,
drift, and persistence cannot establish a meaning or a model belief. Watches
are in-memory registrations and consume explicit fresh measurement identifiers.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


FORMAT_VERSION = 1
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_LAYERS = 128


def _json_copy(value):
    """Reject unsafe/unbounded JSON trees before copying or writing them."""
    visited = 0

    def check(item, depth=0):
        nonlocal visited
        visited += 1
        if depth > 24 or visited > 500_000:
            raise ValueError('diagnostic JSON exceeds structural bounds')
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, str):
            if len(item) > 4096:
                raise ValueError('diagnostic string exceeds its bound')
            return
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            if not math.isfinite(item):
                raise ValueError('diagnostic number must be finite')
            return
        if isinstance(item, list):
            if len(item) > 4096:
                raise ValueError('diagnostic list exceeds its bound')
            for child in item:
                check(child, depth + 1)
            return
        if isinstance(item, dict):
            if len(item) > 4096:
                raise ValueError('diagnostic mapping exceeds its bound')
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 512:
                    raise ValueError('diagnostic keys must be bounded nonempty strings')
                check(child, depth + 1)
            return
        raise ValueError('diagnostics require plain JSON values')

    check(value)
    return copy.deepcopy(value)


def _digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _namespace(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError('namespace must be a bounded nonempty string')
    return value


def _integer(value, name, lower, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f'{name} must be an integer from {lower} through {upper}')
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _signature(report, metadata):
    model = metadata.get('model_identity', report.get('model_identity'))
    tokenizer = metadata.get('tokenizer_identity',
                             model.get('tokenizer_identity') if isinstance(model, dict) else None)
    direction = report.get('direction_sha256')
    scope = metadata.get('source_scope')
    version = report.get('version')
    if not isinstance(model, dict) or not model:
        raise ValueError('a native model identity is required for diagnostic comparisons')
    if not isinstance(tokenizer, dict) or not tokenizer:
        raise ValueError('a tokenizer identity is required for diagnostic comparisons')
    if not isinstance(direction, str) or len(direction) != 64 or any(c not in '0123456789abcdef' for c in direction):
        raise ValueError('direction_sha256 must be a lowercase SHA256 digest')
    if not isinstance(scope, (str, dict)) or not scope:
        raise ValueError('a stable source_scope identity is required')
    if not isinstance(version, str) or not version:
        raise ValueError('diagnostic report version is required')
    return _json_copy({'model': model, 'tokenizer': tokenizer, 'direction': direction,
                       'source_scope': scope, 'diagnostic_version': version})


def _distribution(entries, term_key, score_key, max_terms):
    if not isinstance(entries, list) or len(entries) > max_terms:
        raise ValueError('diagnostic term list exceeds its configured bound')
    weights = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('diagnostic terms must be objects')
        term = entry.get(term_key)
        if not isinstance(term, str) or len(term) > 256:
            raise ValueError('diagnostic terms must be bounded strings')
        score = abs(_finite(entry.get(score_key), score_key))
        if term.strip() and score:
            weights[term] = weights.get(term, 0.0) + score
    total = sum(weights.values())
    if not math.isfinite(total):
        raise ValueError('diagnostic term mass must be finite')
    return {term: weight / total for term, weight in weights.items()} if total else {}


def _channels(report, max_terms):
    channels = {}
    vocabulary = report.get('vocabulary', {})
    if not isinstance(vocabulary, dict):
        raise ValueError('vocabulary channel must be an object')
    layers = vocabulary.get('layers', {})
    if not isinstance(layers, dict) or len(layers) > _MAX_LAYERS:
        raise ValueError('vocabulary layers exceed their bound')
    if vocabulary.get('status') == 'candidate':
        for layer, projection in layers.items():
            if not isinstance(projection, dict):
                raise ValueError('a vocabulary layer must be an object')
            for pole in ('positive', 'negative'):
                channels[f'vocabulary.layer.{layer}.{pole}'] = _distribution(
                    projection.get(pole, []), 'token', 'score', max_terms)
    behavior = report.get('behavior', {})
    if not isinstance(behavior, dict):
        raise ValueError('behavior channel must be an object')
    for source in ('reply', 'input'):
        contrast = behavior.get(source, {})
        if not isinstance(contrast, dict):
            raise ValueError('a behavior contrast must be an object')
        if contrast.get('status') == 'supported':
            for pole in ('high', 'low'):
                channels[f'behavior.{source}.{pole}'] = _distribution(
                    contrast.get(pole, []), 'term', 'log_odds', max_terms)
    return {key: value for key, value in channels.items() if value}


def _mix(previous, observed, smoothing, max_terms):
    weights = {term: (1.0 - smoothing) * weight for term, weight in previous.items()}
    for term, weight in observed.items():
        weights[term] = weights.get(term, 0.0) + smoothing * weight
    retained = sorted(weights.items(), key=lambda item: (-item[1], item[0]))[:max_terms]
    total = sum(weight for _, weight in retained)
    return {term: weight / total for term, weight in retained} if total else {}


def _channel_drift(previous, current):
    # Normalized term-distribution distance, not a semantic discrepancy.
    return {channel: 0.5 * sum(abs(previous[channel].get(term, 0.0) - current[channel].get(term, 0.0))
                              for term in set(previous[channel]) | set(current[channel]))
            for channel in sorted(set(previous) & set(current))}


class ProbeConceptionLedger:
    """Atomic bounded JSON history, kept apart from probes and reply memories."""

    def __init__(self, path=None, *, smoothing=0.2, max_history=32,
                 max_namespaces=64, max_terms=64):
        smoothing = _finite(smoothing, 'smoothing')
        if not 0 < smoothing <= 1:
            raise ValueError('smoothing must be greater than zero and at most one')
        self.path = Path(path) if path is not None else None
        self.smoothing = smoothing
        self.max_history = _integer(max_history, 'max_history', 1, 256)
        self.max_namespaces = _integer(max_namespaces, 'max_namespaces', 1, 256)
        self.max_terms = _integer(max_terms, 'max_terms', 1, 256)
        self._records = {}
        if self.path and self.path.exists():
            if self.path.stat().st_size > _MAX_FILE_BYTES:
                raise ValueError('diagnostic ledger exceeds its file bound')
            self._records = self._validate_payload(json.loads(self.path.read_text(encoding='utf-8')))

    def _validate_payload(self, payload):
        payload = _json_copy(payload)
        if not isinstance(payload, dict) or payload.get('format_version') != FORMAT_VERSION:
            raise ValueError('unsupported diagnostic ledger format')
        records = payload.get('records')
        if not isinstance(records, dict) or len(records) > self.max_namespaces:
            raise ValueError('diagnostic namespace count exceeds its bound')
        for namespace, state in records.items():
            _namespace(namespace)
            if not isinstance(state, dict) or not isinstance(state.get('history'), list):
                raise ValueError('invalid diagnostic namespace state')
            if not state['history'] or len(state['history']) > self.max_history:
                raise ValueError('diagnostic history exceeds its bound')
            signature = state.get('signature')
            if not isinstance(signature, dict) or state.get('signature_sha256') != _digest(signature):
                raise ValueError('invalid diagnostic comparison signature')
            for snapshot in state['history']:
                if (not isinstance(snapshot, dict) or snapshot.get('kind') != 'diagnostic_observation'
                        or snapshot.get('semantic_validated') is not False
                        or snapshot.get('rename_authority') is not False):
                    raise ValueError('history must contain unvalidated diagnostic observations')
                report, metadata = snapshot.get('report'), snapshot.get('metadata')
                if (not isinstance(report, dict) or not isinstance(metadata, dict)
                        or report.get('semantic_validated') is not False
                        or report.get('rename_authority') is not False):
                    raise ValueError('invalid diagnostic report authority')
                snapshot_signature = _signature(report, metadata)
                if snapshot.get('signature_sha256') != _digest(snapshot_signature):
                    raise ValueError('invalid historical diagnostic signature')
                if snapshot.get('evidence_sha256') != _digest({'report': report, 'metadata': metadata}):
                    raise ValueError('invalid historical diagnostic evidence hash')
                _channels(report, self.max_terms)
            candidate = state.get('candidate')
            if (not isinstance(candidate, dict) or candidate.get('kind') != 'lexical_candidate'
                    or candidate.get('semantic_validated') is not False
                    or candidate.get('rename_authority') is not False):
                raise ValueError('invalid candidate authority flags')
            _integer(state.get('observations'), 'observations', 1, 2**53 - 1)
            if not isinstance(candidate.get('channels'), dict):
                raise ValueError('candidate channels must be an object')
            for weights in candidate['channels'].values():
                if not isinstance(weights, dict) or len(weights) > self.max_terms:
                    raise ValueError('candidate terms exceed their bound')
                for weight in weights.values():
                    if _finite(weight, 'candidate weight') < 0:
                        raise ValueError('candidate weight must be nonnegative')
                if weights and not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
                    raise ValueError('candidate weights must be normalized')
            if (state['history'][-1].get('signature_sha256') != state['signature_sha256']
                    or state['history'][-1].get('candidate') != candidate):
                raise ValueError('current candidate must match the latest observation')
        return records

    def get(self, namespace):
        """Return an isolated diagnostic state, or None for an unseen namespace."""
        return copy.deepcopy(self._records.get(_namespace(namespace)))

    def observe(self, namespace, report, metadata=None):
        namespace = _namespace(namespace)
        report = _json_copy(report)
        metadata = _json_copy(metadata or {})
        if not isinstance(report, dict) or not isinstance(metadata, dict):
            raise ValueError('report and metadata must be objects')
        if report.get('semantic_validated') is not False or report.get('rename_authority') is not False:
            raise ValueError('diagnostic reports must explicitly deny semantic and rename authority')
        signature = _signature(report, metadata)
        fingerprint = _digest(signature)
        evidence_fingerprint = _digest({'report': report, 'metadata': metadata})
        observed = _channels(report, self.max_terms)
        previous = self._records.get(namespace)
        comparable = bool(previous and previous['signature_sha256'] == fingerprint)
        if comparable and previous['history'][-1]['evidence_sha256'] == evidence_fingerprint:
            unchanged = copy.deepcopy(previous['history'][-1])
            unchanged['status'] = 'unchanged'
            unchanged['comparison'] = {'comparable': True, 'reason': 'identical_observation',
                                       'channel_drift': {}, 'meaning_validated': False}
            return unchanged
        old_channels = previous['candidate']['channels'] if comparable else {}
        revised = copy.deepcopy(old_channels)
        for channel, distribution in observed.items():
            revised[channel] = (_mix(old_channels[channel], distribution, self.smoothing, self.max_terms)
                                if channel in old_channels else distribution)
        candidate = {'kind': 'lexical_candidate', 'semantic_validated': False,
                     'rename_authority': False, 'channels': revised}
        comparison = {'comparable': comparable, 'reason': 'same_scope' if comparable else
                      'scope_changed' if previous else 'first_observation',
                      'channel_drift': _channel_drift(old_channels, revised) if comparable else {},
                      'meaning_validated': False}
        observation = {'kind': 'diagnostic_observation', 'namespace': namespace,
                       'status': 'revision' if comparable else 'baseline_scope_change' if previous else 'baseline',
                       'observed_at': datetime.now(timezone.utc).isoformat(),
                       'semantic_validated': False, 'rename_authority': False,
                       'signature_sha256': fingerprint, 'evidence_sha256': evidence_fingerprint, 'report': report,
                       'metadata': metadata, 'smoothing': self.smoothing,
                       'observed_channels': sorted(observed), 'comparison': comparison,
                       'candidate': copy.deepcopy(candidate)}
        history = list(previous['history']) if previous else []
        history.append(observation)
        staged = copy.deepcopy(self._records)
        staged.pop(namespace, None)  # Insertion order tracks most recently observed namespaces.
        staged[namespace] = {'signature': signature, 'signature_sha256': fingerprint,
                             'observations': (previous['observations'] if previous else 0) + 1,
                             'candidate': candidate, 'history': history[-self.max_history:]}
        while len(staged) > self.max_namespaces:
            staged.pop(next(iter(staged)))
        self._commit(staged, namespace)
        return copy.deepcopy(observation)

    def _commit(self, staged, current_namespace):
        payload = {'format_version': FORMAT_VERSION, 'records': staged}
        self._validate_payload(payload)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        # Keep each namespace's newest snapshot; drop older snapshots first.
        while len(encoded) > _MAX_FILE_BYTES:
            older = next((state for state in staged.values() if len(state['history']) > 1), None)
            if older:
                older['history'].pop(0)
            elif len(staged) > 1:
                oldest = next(key for key in staged if key != current_namespace)
                staged.pop(oldest)
            else:
                raise ValueError('one diagnostic observation exceeds the ledger file bound')
            encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=self.path.name + '.',
                                                 suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._validate_payload(json.loads(temporary.read_text(encoding='utf-8')))
                os.replace(temporary, self.path)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
        self._records = staged


def measurement_record_ids(records):
    """Identify measured rows without treating a command or missing reading as data.

    Explicit durable record IDs preserve repeated equal samples. Timestamp/content
    fallbacks are conservative for older rows and can collapse ambiguous duplicates.
    Archived backfill rows are not fresh live measurements for watch scheduling.
    """
    if not isinstance(records, (list, tuple)) or len(records) > 4096:
        raise ValueError('measured records must be a bounded sequence')
    identifiers = []
    for row in records:
        if not isinstance(row, dict) or row.get('basis') == 'backfill':
            continue
        value = row.get('reading')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            continue
        text = row.get('reply', row.get('text'))
        if not isinstance(text, str) or not text.strip():
            continue
        identifier = next((row[key] for key in ('record_id', 'source_ts', 'ts', 'id')
                           if isinstance(row.get(key), (str, int)) and not isinstance(row.get(key), bool)
                           and str(row[key]).strip()), None)
        if identifier is not None:
            identifiers.append('record:' + str(identifier))
        else:
            identifiers.append('legacy:' + _digest({'reading': float(value), 'reply': text,
                                                   'input': row.get('input', ''),
                                                   'participant': row.get('participant', '')}))
    return list(dict.fromkeys(identifiers))


class ProbeDiagnosticWatch:
    """Opt-in, ephemeral watches driven by new measured record IDs only."""

    def __init__(self, *, max_seen=4096):
        self.max_seen = _integer(max_seen, 'max_seen', 64, 16_384)
        self._registrations = {}

    @property
    def registrations(self):
        return copy.deepcopy(self._registrations)

    def _ids(self, record_ids):
        if not isinstance(record_ids, (list, tuple)) or len(record_ids) > self.max_seen:
            raise ValueError('watch record IDs must be a bounded sequence')
        return list(dict.fromkeys(_namespace(identifier) for identifier in record_ids))

    def register(self, namespace, *, every=10, record_ids=()):
        namespace = _namespace(namespace)
        every = _integer(every, 'every', 1, self.max_seen)
        self._registrations[namespace] = {'every': every, 'pending': 0, 'seen': self._ids(record_ids)}

    def unregister(self, namespace):
        return self._registrations.pop(_namespace(namespace), None) is not None

    def due(self, namespace, record_ids):
        namespace = _namespace(namespace)
        watch = self._registrations.get(namespace)
        if watch is None:
            return False
        identifiers = self._ids(record_ids)
        seen = set(watch['seen'])
        fresh = [identifier for identifier in identifiers if identifier not in seen]
        if not fresh:
            return False
        watch['seen'] = (watch['seen'] + fresh)[-self.max_seen:]
        watch['pending'] += len(fresh)
        ready = watch['pending'] >= watch['every']
        if ready:
            watch['pending'] %= watch['every']
        return ready
