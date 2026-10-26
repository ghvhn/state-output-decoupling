"""Keep probe evidence from different measurement sources in separate scopes."""
from collections import deque
from copy import deepcopy
import math
import re


LEGACY_PROBE_SOURCE = 'reply_text_replay'


def measurement_source(probe):
    """Unmarked persisted probes were measured by re-encoding reply text."""
    return str(probe.get('measurement_source') or LEGACY_PROBE_SOURCE)


def measurement_namespace_content(content, probe):
    """Leave legacy namespace keys intact; scope explicitly marked sources."""
    source = probe.get('measurement_source')
    return content + '\0measurement_source:' + str(source) if source else content


def _history(values, limit=40):
    result = []
    for value in list(values or ())[-limit:]:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            result.append(number)
    return result


def _snapshot(probe):
    return {'history': _history(probe.get('history')),
            'other_histories': {str(name): _history(values) for name, values in
                                (probe.get('other_histories') or {}).items()}}


def ensure_probe_measurement_source(probe, source):
    """Switch source, preserving independent raw and referent baselines.

    Returns whether the active source changed. The shell consumes
    ``_measurement_seed_pending`` to seed a fresh frozen baseline once.
    Merely marking the legacy source does not change existing legacy keys.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError('Probe measurement source must be a nonempty string')
    source = source.strip()
    previous = measurement_source(probe)
    if previous == source:
        if not isinstance(probe.get('history'), deque):
            maxlen = getattr(probe.get('history'), 'maxlen', None) or 40
            probe['history'] = deque(_history(probe.get('history'), maxlen), maxlen=maxlen)
        return False
    banks = probe.setdefault('measurement_histories', {})
    banks[previous] = _snapshot(probe)
    target = banks.get(source) or {}
    maxlen = getattr(probe.get('history'), 'maxlen', None) or 40
    probe['history'] = deque(_history(target.get('history'), maxlen), maxlen=maxlen)
    probe['other_histories'] = {str(name): _history(values) for name, values in
                                (target.get('other_histories') or {}).items()}
    if source == LEGACY_PROBE_SOURCE:
        probe.pop('measurement_source', None)
    else:
        probe['measurement_source'] = source
    probe['_measurement_seed_pending'] = not bool(probe['history'])
    probe.pop('last_profile', None)
    return True


def prepare_probe_measurements(probes, tuner, source):
    """Switch probes and clear stale displayed/credited tuner measurements.

    Old trigger evidence is archived on the probe. It is not restored as a
    current reading when switching back; the next observation must supply it.
    Operator-set trigger values and unrelated triggers remain unchanged.
    """
    changed = []
    for name, probe in probes.items():
        previous = measurement_source(probe)
        if not ensure_probe_measurement_source(probe, source):
            continue
        changed.append(name)
        trigger = getattr(tuner, 'triggers', {}).get('probe_' + name)
        if trigger is None:
            continue
        snapshot = {field: deepcopy(list(getattr(trigger, field, ())))
                    for field in ('signals', 'outcomes')}
        snapshot.update({field: getattr(trigger, field, 0)
                         for field in ('observed', 'fired', 'calibrations')})
        probe.setdefault('measurement_trigger_histories', {})[previous] = snapshot
        for field in ('signals', 'outcomes'):
            values = getattr(trigger, field, None)
            if values is not None:
                values.clear()
        for field in ('observed', 'fired', 'calibrations'):
            if hasattr(trigger, field):
                setattr(trigger, field, 0)
    if changed and callable(getattr(tuner, 'save', None)):
        tuner.save()
    return changed


def bounded_probe_measurement(value):
    """Retain small scalar provenance, never prompts or captured tensors."""
    if not isinstance(value, dict):
        return {}
    result = {}
    for field, limit in (('source', 80), ('endpoint', 120), ('generation_id', 128)):
        item = value.get(field)
        if isinstance(item, str) and item:
            result[field] = item[:limit]
    for field in ('token_count', 'decision_position'):
        item = value.get(field)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[field] = item
    digest = value.get('prompt_sha256')
    if isinstance(digest, str) and re.fullmatch(r'[0-9a-fA-F]{64}', digest):
        result['prompt_sha256'] = digest.lower()
    return result
