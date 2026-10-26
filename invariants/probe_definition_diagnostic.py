"""Operator-only geometric diagnostics for the active probe owner.

Reports and lexical conception candidates never enter model context, learned
namespace summaries, steering, or the probe's actual direction. Watches count
fresh paired measurements, not commands or the bounded buffer length.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from invariants.exposure_language import probe_alias
from invariants.probe_behavior import definition_key


USAGE = ':probe define2 <active name or opaque alias> [align|watch [N]|off]'


def _runtime_identity(model):
    """Replacing the loaded model or tokenizer invalidates a live watch."""
    return (id(model), id(getattr(model, 'model', model)),
            id(getattr(model, 'tok', getattr(model, 'tokenizer', None))))


def _active_name(identifier, probes):
    if identifier in probes:
        return identifier
    if identifier.startswith('probe_') and identifier[6:] in probes:
        return identifier[6:]
    return next((name for name in probes if probe_alias(name) == identifier), None)


def _direction_identity(probe):
    digest = hashlib.sha256()
    for layer, value in sorted(probe['direction'].items()):
        tensor = value.detach().cpu().float().contiguous()
        digest.update(str((layer, tuple(tensor.shape))).encode('ascii'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _revision_text(observation):
    comparison = observation.get('comparison', {})
    drift = comparison.get('channel_drift', {})
    if not comparison.get('comparable') or not drift:
        return ''
    changed = sorted(((channel, value) for channel, value in drift.items() if value > 1e-8),
                     key=lambda item: (-item[1], item[0]))[:3]
    if not changed:
        return '\n[Conception] Comparable lexical channels show no candidate drift.'
    channels = observation.get('candidate', {}).get('channels', {})
    parts = []
    for channel, value in changed:
        terms = sorted(channels.get(channel, {}).items(), key=lambda item: (-item[1], item[0]))[:3]
        parts.append(f'{channel}: drift={value:.3f}; ' + ', '.join(repr(term) for term, _ in terms))
    return '\n[Conception] Revised lexical channels: ' + ' | '.join(parts)


def _live_records(records):
    return [record for record in records if record.get('basis') != 'backfill']


class ProbeDefinitionDiagnostics:
    """One manager per ``PlasticityOwners`` matrix, with isolated persistence."""

    def __init__(self, behavior):
        from invariants.probe_conception import ProbeConceptionLedger, ProbeDiagnosticWatch
        self.behavior = behavior
        path = Path(behavior.path).with_suffix('.conception.json') if behavior.path else None
        self.ledger = ProbeConceptionLedger(path)
        self.watch = ProbeDiagnosticWatch()
        self._bindings = {}

    def _record_ids(self, namespace):
        from invariants.probe_conception import measurement_record_ids
        return measurement_record_ids(_live_records(self.behavior.records.get(namespace, [])))

    def _diagnose(self, name, probe, model, mode='run'):
        from invariants.probe_definition_geometric import analyze_probe_geometry, render_geometric_report
        namespace = definition_key(name, probe)
        report = analyze_probe_geometry(
            name, probe, model, records=self.behavior.records.get(namespace, []), namespace=namespace)
        scope = {'behavior_path': str(Path(self.behavior.path).resolve()) if self.behavior.path else 'in_memory',
                 'namespace': namespace, 'window': 'paired_probe_behavior_last_64',
                 'projection': 'per_layer_output_head', 'top_k': 20, 'bottom_k': 20}
        observation = self.ledger.observe(namespace, report, metadata={'source_scope': scope})
        text = render_geometric_report(report, mode=mode)
        from invariants.probe_display import console_probe_name
        plain_label = f"[Define2: {report['probe_name']}]"
        text = text.replace(plain_label, f'[Define2: {console_probe_name(name)}]', 1)
        return (text + '\n[Conception] ' + observation['status'] +
                '; lexical candidate only; no namespace or direction changed.' + _revision_text(observation))

    def command(self, args, probes, model):
        args = list(args)
        if not args or len(args) > 3:
            raise ValueError(USAGE)
        mode = args[1].lower() if len(args) > 1 else 'run'
        if mode not in ('run', 'align', 'watch', 'off') or (len(args) > 2 and mode != 'watch'):
            raise ValueError(USAGE)
        every = 10
        if mode == 'watch' and len(args) > 2:
            try:
                every = int(args[2])
            except (TypeError, ValueError) as exc:
                raise ValueError('Watch interval must be a positive integer.') from exc
            if not 1 <= every <= 4096:
                raise ValueError('Watch interval must be between 1 and 4096 measured samples.')
        name = _active_name(args[0], probes)
        if name is None:
            raise ValueError(f'No exact active probe or opaque alias {args[0]!r}; diagnostic commands do not select or load probes.')
        probe = probes[name]
        namespace = definition_key(name, probe)
        if mode == 'off':
            # The selected Other can change before the next scored poll.
            # An explicit stop still applies to this exact active probe.
            for watched in [namespace, *(key for key, binding in self._bindings.items()
                                         if binding['name'] == name and key != namespace)]:
                self.watch.unregister(watched)
                self._bindings.pop(watched, None)
            return f'[Probe] Geometric watch off for {probe_alias(name)}.'
        result = self._diagnose(name, probe, model, 'align' if mode == 'align' else 'run')
        if mode == 'watch':
            self.watch.register(namespace, every=every, record_ids=self._record_ids(namespace))
            self._bindings[namespace] = {'name': name, 'runtime_identity': _runtime_identity(model),
                                        'direction_identity': _direction_identity(probe)}
            result += f'\n[Probe] Watching {probe_alias(name)} every {every} fresh measured samples; archive imports do not advance it.'
        return result

    def poll(self, probes, model):
        """Called only after a scored conversation's paired observations save."""
        messages = []
        for namespace, binding in list(self._bindings.items()):
            name = binding['name']
            probe = probes.get(name)
            if (probe is None or definition_key(name, probe) != namespace or
                    binding['runtime_identity'] != _runtime_identity(model) or
                    binding['direction_identity'] != _direction_identity(probe)):
                self.watch.unregister(namespace)
                self._bindings.pop(namespace, None)
                messages.append(f'[Probe] Geometric watch disabled for {probe_alias(name)}: active probe, referent binding, direction, or loaded model changed.')
                continue
            try:
                if self.watch.due(namespace, self._record_ids(namespace)):
                    messages.append(self._diagnose(name, probe, model, 'align'))
            except Exception as exc:
                # Diagnostics must not abort a scored reply or its learning.
                messages.append(f'[Probe] Geometric watch could not update: {type(exc).__name__}: {exc}')
        return messages


def diagnostic_manager(matrix):
    manager = getattr(matrix, 'probe_diagnostics', None)
    if manager is None or manager.behavior is not matrix.probe_behavior:
        manager = ProbeDefinitionDiagnostics(matrix.probe_behavior)
        matrix.probe_diagnostics = manager
    return manager


def poll_probe_diagnostics(matrix, probes, model):
    """Do not instantiate a ledger on ordinary turns without a diagnostic."""
    manager = getattr(matrix, 'probe_diagnostics', None)
    if manager is None:
        return []
    try:
        return manager.poll(probes, model)
    except Exception as exc:
        return [f'[Probe] Geometric watches could not update: {type(exc).__name__}: {exc}']
