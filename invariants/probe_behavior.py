"""Behavior-grounded probe interpretations with a bounded activation subject."""
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
import torch
from invariants.exposure_language import framing_pair, probe_alias, obscure_probe_identifiers
from invariants.probe_measurement import (
    LEGACY_PROBE_SOURCE, bounded_probe_measurement, measurement_namespace_content,
)

DEFINITION_MAX_TOKENS = 384


def spectrum_side(position):
    return 'negative side' if position < 0 else 'positive side' if position > 0 else 'baseline'


def render_definition_evidence(namespace, selected):
    """One measured-evidence format shared by live probes and calibration.

    Callers supply ordered observations and perform any explicit identifier
    obscuring afterward. No task instruction or seed meaning is added here.
    """
    lines = [f'[Probe {namespace}]',
             'These memories are positioned along this probe\'s spectrum by their measured centered readings; zero is the rolling baseline.']
    side = None
    for index, row in selected:
        memory_side = spectrum_side(row['reading'])
        if memory_side != side:
            lines.append(f'[{memory_side.capitalize()}]')
            side = memory_side
        lines.append(f'Memory {index}: position={row["reading"]:+.4f}; side={memory_side}; '
                     f'input={row["input"]!r}; reply={row["reply"]!r}')
    return '\n'.join(lines)


def definition_key(name, probe):
    content = '\0'.join(framing_pair(probe)) + ':' + str(probe.get('other_binding', ''))
    content = measurement_namespace_content(content, probe)
    return probe_alias(name) + ':' + hashlib.sha256(content.encode()).hexdigest()[:16]


class ProbeBehavior:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.records = {}
        self.interpretations = {}
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            self.records = saved.get('records', {})
            self.interpretations = saved.get('interpretations', {})

    def observe(self, response, user_input, row, probes):
        if not response:
            return
        changed = False
        # A stable observation identity lets diagnostic watches distinguish
        # repeated identical replies even after the bounded window fills.
        source_ts = row.get('source_ts') or row.get('ts')
        provenance = {'source_ts': source_ts} if isinstance(source_ts, str) and source_ts else {}
        measurement = bounded_probe_measurement(row.get('probe_measurement'))
        if measurement:
            provenance['probe_measurement'] = measurement
        for name, probe in probes.items():
            # Parallel Other projections are paired with this same reply;
            # each referent learns even when it is not currently selected.
            parallel = row.get('other_probe_readings', {}).get(name, {})
            seen_bindings = set()
            for participant, measurement in parallel.items():
                reading = measurement.get('centered')
                binding = measurement.get('binding')
                if reading is None or not binding or not math.isfinite(float(reading)):
                    continue
                if binding in seen_bindings:
                    continue
                seen_bindings.add(binding)
                scoped = dict(probe, other_binding=binding)
                key = definition_key(name, scoped)
                outcome = row.get('sense')
                record = {'reply': response[:600], 'input': user_input[:200], 'reading': float(reading),
                          'outcome': float(outcome) if outcome is not None and math.isfinite(float(outcome)) else None,
                          'participant': participant, 'record_id': uuid.uuid4().hex, **provenance}
                examples = self.records.setdefault(key, [])
                examples.append(record)
                self.records[key] = examples[-64:]
                changed = True
            if parallel:
                continue  # do not double-count the selected Other projection
            reading = row.get('probe_' + name)
            if reading is None or not math.isfinite(float(reading)):
                continue
            key = definition_key(name, probe)
            examples = self.records.setdefault(key, [])
            outcome = row.get('sense')
            examples.append({'reply': response[:600], 'input': user_input[:200], 'reading': float(reading),
                             'outcome': float(outcome) if outcome is not None and math.isfinite(float(outcome)) else None,
                             'record_id': uuid.uuid4().hex, **provenance})
            self.records[key] = examples[-64:]
            changed = True
        if changed:
            self.save()

    def backfill(self, observations, probes):
        """Import measured archived replies in chronological order.

        Each observation supplies ``ts``, ``reply``, optional ``input`` and
        ``outcome``, and ``readings`` keyed by probe name. Optional ``bindings``
        must identify the archived Other namespace; the current selection is
        never used to guess an old referent. Missing/invalid readings are
        omitted, so the caller can leave the unbaselined first sample unset.

        Timestamped archive records are replaceable when a backfill changes
        their baseline. Live records take precedence and remain intact. The
        latest archive records fill remaining capacity up to 64 per namespace;
        staging the whole batch keeps repeated long imports idempotent.
        """
        from invariants.other_probes import is_other_probe

        def finite(value):
            if value is None or isinstance(value, bool):
                return None
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return number if math.isfinite(number) else None

        counts = {'imported': 0, 'replaced': 0, 'skipped': 0}
        staged = {}
        for observation in observations:
            if not isinstance(observation, dict):
                counts['skipped'] += 1
                continue
            readings = observation.get('readings')
            if not isinstance(readings, dict):
                counts['skipped'] += 1
                continue
            timestamp = observation.get('ts')
            reply = observation.get('reply')
            if not isinstance(timestamp, str) or not timestamp or not isinstance(reply, str) or not reply.strip():
                counts['skipped'] += len(readings)
                continue
            user_input = observation.get('input')
            user_input = user_input if isinstance(user_input, str) else ''
            outcome = finite(observation.get('outcome'))
            measurement = bounded_probe_measurement(observation.get('probe_measurement'))
            source = measurement.get('source') or observation.get('measurement_source')
            source = source if isinstance(source, str) and source else LEGACY_PROBE_SOURCE
            bindings = observation.get('bindings')
            bindings = bindings if isinstance(bindings, dict) else {}
            for name, value in readings.items():
                reading = finite(value)
                probe = probes.get(name)
                if probe is None or reading is None:
                    counts['skipped'] += 1
                    continue
                # Missing archive provenance denotes the old reply-text replay
                # basis, never the source currently active in the live shell.
                scoped = dict(probe)
                if source == LEGACY_PROBE_SOURCE:
                    scoped.pop('measurement_source', None)
                else:
                    scoped['measurement_source'] = source
                if is_other_probe(probe):
                    binding = bindings.get(name)
                    if not isinstance(binding, str) or not binding:
                        counts['skipped'] += 1
                        continue
                    scoped['other_binding'] = binding
                key = definition_key(name, scoped)
                records = staged.setdefault(key, {})
                if timestamp in records:
                    counts['skipped'] += 1
                records[timestamp] = {'reply': reply[:600], 'input': user_input[:200],
                                      'reading': reading, 'outcome': outcome,
                                      'basis': 'backfill', 'source_ts': timestamp,
                                      **({'probe_measurement': measurement} if measurement else {})}

        changed = False
        for key, imported in staged.items():
            previous = self.records.get(key, [])
            live = [record for record in previous if record.get('basis') != 'backfill']
            live_timestamps = {record.get('source_ts') for record in live if record.get('source_ts')}
            archives = {record['source_ts']: record for record in previous
                        if record.get('basis') == 'backfill' and record.get('source_ts')}
            original_archives = dict(archives)
            for timestamp, record in imported.items():
                if timestamp in live_timestamps:
                    counts['skipped'] += 1
                    continue
                # Move revisited entries into the supplied chronological order.
                archives.pop(timestamp, None)
                archives[timestamp] = record
            capacity = max(0, 64 - len(live))
            retained = list(archives.values())[-capacity:] if capacity else []
            retained_timestamps = {record['source_ts'] for record in retained}
            for timestamp, record in imported.items():
                if timestamp in live_timestamps:
                    continue
                if timestamp not in retained_timestamps or record == original_archives.get(timestamp):
                    counts['skipped'] += 1
                elif timestamp in original_archives:
                    counts['replaced'] += 1
                else:
                    counts['imported'] += 1
            merged = (retained + live)[-64:]
            if merged != previous:
                self.records[key] = merged
                changed = True
        if changed:
            self.save()
        return counts

    def exposure_examples(self, name, probe, probes):
        """Current contrasting observations, without supplying the seed definition."""
        rows = self.records.get(definition_key(name, probe), [])
        if not rows:
            return ''
        ordered = sorted(enumerate(rows), key=lambda pair: (pair[1]['reading'], pair[0]))
        selected = [ordered[0]]
        if ordered[-1][0] != ordered[0][0]:
            selected.append(ordered[-1])
        text = ' | '.join(f'reading={row["reading"]:+.3f}, input={row["input"][:80]!r}, '
                          f'reply={row["reply"][:140]!r}, outcome={row["outcome"]}'
                          for _, row in selected)
        return obscure_probe_identifiers(text, probes)

    def definition_memories(self, name, probe):
        rows = self.records.get(definition_key(name, probe), [])
        if len(rows) < 4 or max(r['reading'] for r in rows) - min(r['reading'] for r in rows) < 1e-6:
            return None
        ordered = sorted(enumerate(rows), key=lambda pair: (pair[1]['reading'], pair[0]))
        return ordered[:2] + ordered[-2:]

    def evidence(self, name, probe, probes):
        selected = self.definition_memories(name, probe)
        if selected is None:
            return None
        return obscure_probe_identifiers(render_definition_evidence(probe_alias(name), selected), probes)

    def annotate_definition_memories(self, name, probe, text):
        """Operator labels use measured coordinates, independent of generated claims."""
        selected = dict(self.definition_memories(name, probe) or [])
        def annotate(match):
            row = selected.get(int(match.group(1)))
            if row is None:
                return match.group(0) + ' [not in supplied probe evidence]'
            position = row['reading']
            return match.group(0) + f' [{spectrum_side(position)}; position={position:+.4f}]'
        return re.sub(r'\bMemory\s+#?(\d+)\b', annotate, text, flags=re.IGNORECASE)

    def record_interpretation(self, name, probe, text, *, status='hypothesis', controls=None,
                              qualification=None):
        from invariants.probe_definition_check import assess_definition_attempt
        structure = assess_definition_attempt(text, selected=self.definition_memories(name, probe))
        qualification = qualification or structure
        if qualification['accepted'] and not structure['accepted']:
            qualification = structure
        if status == 'hypothesis' and not qualification['accepted']:
            status = 'unqualified'
        key = definition_key(name, probe)
        values = self.interpretations.setdefault(key, [])
        values.append({'text': text[:16000], 'observed_replies': len(self.records.get(key, [])),
                       'status': status, 'qualified': status == 'hypothesis' and qualification['accepted'],
                       'qualification_reason': qualification['reason'],
                       'contrast': qualification.get('contrast', ''),
                       **({'controls': controls} if controls is not None else {})})
        self.interpretations[key] = values[-16:]
        self.save()

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({'records': self.records, 'interpretations': self.interpretations}, allow_nan=False), encoding='utf-8')
            for _ in range(10):
                try:
                    temp.replace(self.path)
                    break
                except PermissionError:
                    import time
                    time.sleep(0.02)
            else:
                temp.replace(self.path)


def build_definition_handles(model, subject, inquiry, cap_fraction=None,
                             *, inquiry_fraction=.04, subject_fraction=.02, task_gains=None,
                             task_positions=None, task_layer_scales=None):
    """Subject marks the original decision position; task controls steer output.

    Named task directions have independent signed gains, including negative
    verbosity. Their sum and the subject share one residual cap. Memory tokens
    remain unchanged: generation controls apply at the original assistant
    decision position and subsequent generated positions. Decision controls
    apply only at that original position, restored on full-prefix rebuilds.
    Optional per-layer scales preserve bounded learned conversion magnitude;
    they multiply each task's signed gain and share the same steering cap.
    A single direction remains supported for existing callers.
    """
    from invariants.engine import _cap_steer
    for fraction in (inquiry_fraction, subject_fraction):
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Definition injection fractions must be finite and between zero and one')
    if all(isinstance(value, torch.Tensor) for value in inquiry.values()) and (inquiry or task_gains is None):
        tasks = {'inquiry': inquiry}
        gains = {'inquiry': inquiry_fraction}
    else:
        from invariants.probe_definition_task import DEFINITION_TASK_GAINS
        tasks = inquiry
        gains = DEFINITION_TASK_GAINS if task_gains is None else task_gains
    positions = {} if task_positions is None else task_positions
    if not hasattr(positions, 'items'):
        raise ValueError('Definition task positions must map controls to decision or generation')
    for name, placement in positions.items():
        if placement not in ('decision', 'generation'):
            raise ValueError(f'Definition task position for {name} must be decision or generation')
        if name not in tasks and name not in gains:
            raise ValueError(f'Definition task position for {name} names an undeclared control')
    layer_scales = {} if task_layer_scales is None else task_layer_scales
    if not hasattr(layer_scales, 'items'):
        raise ValueError('Definition task layer scales must map declared controls to layer scales')
    validated_scales = {}
    for name, scales in layer_scales.items():
        if name not in tasks and name not in gains:
            raise ValueError(f'Definition task layer scale for {name} names an undeclared control')
        if not hasattr(scales, 'items') or set(scales) != set(subject):
            raise ValueError(f'Definition task layer scales for {name} must cover exactly the subject layers')
        validated_scales[name] = {}
        for layer, value in scales.items():
            if type(layer) is not int:
                raise ValueError('Definition task layer scale keys must be integer layer indices')
            try:
                scale = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError('Definition task layer scales must be finite and between zero and two') from exc
            if isinstance(value, bool) or not math.isfinite(scale) or not 0 <= scale <= 2:
                raise ValueError('Definition task layer scales must be finite and between zero and two')
            validated_scales[name][layer] = scale
    active = []
    # Validate declared gains as well as supplied axes. Iterating axes alone
    # silently drops an enabled control whose direction was never minted.
    for name in dict.fromkeys((*tasks, *gains)):
        try:
            gain = float(gains.get(name, 0.))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'Definition task gain for {name} must be finite and between -1 and 1') from exc
        if not math.isfinite(gain) or abs(gain) > 1:
            raise ValueError(f'Definition task gain for {name} must be finite and between -1 and 1')
        if gain:
            if name not in tasks:
                raise ValueError(f'Definition task {name} requires a matching axis')
            direction = tasks[name]
            if not set(subject).issubset(direction):
                raise ValueError(f'Definition task {name} must cover every subject layer')
            active.append((name, direction, gain, positions.get(name, 'generation')))
    handles = []
    try:
        for layer in sorted(subject):
            marker = subject[layer].detach().to(model.device).float()
            marker_norm = marker.norm()
            if not torch.isfinite(marker).all() or not torch.isfinite(marker_norm) or marker_norm.item() <= 1e-8:
                raise ValueError(f'Definition subject must be finite and nonzero at layer {layer}')
            marker = marker / marker_norm
            task_mix = torch.zeros_like(marker)
            decision_mix = torch.zeros_like(marker)
            for name, direction, gain, placement in active:
                vector = direction[layer].detach().to(model.device).float()
                norm = vector.norm()
                if (vector.shape != marker.shape or not torch.isfinite(vector).all() or
                        not torch.isfinite(norm) or norm.item() <= 1e-8):
                    raise ValueError(f'Definition task direction must match the subject and be finite and nonzero at layer {layer}')
                contribution = gain * validated_scales.get(name, {}).get(layer, 1.) * vector / norm
                if placement == 'decision':
                    decision_mix += contribution
                else:
                    task_mix += contribution
            if (not torch.isfinite(marker).all() or not torch.isfinite(task_mix).all() or
                    not torch.isfinite(decision_mix).all()):
                raise ValueError('Definition directions must be finite')
            subject_state = {'position': None}
            def hook(module, inp, out, marker=marker, task_mix=task_mix,
                     decision_mix=decision_mix, subject_state=subject_state):
                h = out[0] if isinstance(out, tuple) else out
                with torch.no_grad():
                    add = torch.zeros_like(h)
                    scale = h.float().norm(dim=-1, keepdim=True)
                    first = subject_state['position'] is None
                    if first:
                        subject_state['position'] = h.shape[-2] - 1 if h.ndim >= 3 else 0
                    position = subject_state['position']
                    if h.ndim >= 3:
                        if not first and h.shape[-2] == 1:
                            add += (scale * task_mix.to(h.device)).to(h.dtype)
                        elif position < h.shape[-2]:
                            add[..., position:, :] += (scale[..., position:, :] * task_mix.to(h.device)).to(h.dtype)
                        # Re-prefills restore the same subject token; cached
                        # decode tokens only receive generation controls.
                        if (first or h.shape[-2] > 1) and position < h.shape[-2]:
                            pulse = subject_fraction * marker + decision_mix
                            add[..., position, :] += (scale[..., position, :] * pulse.to(h.device)).to(h.dtype)
                    else:
                        add += (scale * task_mix.to(h.device)).to(h.dtype)
                        if first:
                            pulse = subject_fraction * marker + decision_mix
                            add += (scale * pulse.to(h.device)).to(h.dtype)
                    newh = h + _cap_steer(add, h, cap_fraction=cap_fraction)
                return (newh,) + tuple(out[1:]) if isinstance(out, tuple) else newh
            handles.append(model.model.model.layers[int(layer)].register_forward_hook(hook))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return handles
