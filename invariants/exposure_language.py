"""Exposure-scoped lexical plasticity; associations describe language, not feelings."""
import hashlib
import json
import math
import re
from pathlib import Path
from invariants.probe_measurement import measurement_namespace_content

NAMING_CONFIDENCE = .65
DEFINED_NAMING_CONFIDENCE = .35


def qualified_namespace_summary(summary):
    from invariants.probe_definition_check import assess_definition_attempt
    if (summary.get('status', 'hypothesis') != 'hypothesis' or
            summary.get('observed_replies', 0) < 4 or summary.get('qualified') is False):
        return False
    # Legacy summaries are kept on disk, but recaps no longer count as definitions.
    # This recognizes structure only; neither old nor new meanings are certified.
    return assess_definition_attempt(summary.get('text', ''))['accepted']


def naming_threshold(mapping):
    grounded = any(qualified_namespace_summary(s) for s in mapping.get('summary_hypotheses', []))
    return DEFINED_NAMING_CONFIDENCE if grounded else NAMING_CONFIDENCE


def framing_pair(probe):
    value = probe.get('framings') or ('', '')
    if isinstance(value, dict):
        return str(value.get('a', '')), str(value.get('b', ''))
    return tuple(str(v) for v in (list(value) + ['', ''])[:2])


def _namespace_key(name, probe):
    content = measurement_namespace_content('\0'.join(framing_pair(probe)), probe)
    identity = hashlib.sha256(content.encode()).hexdigest()[:16]
    return name + ':' + identity + (':' + probe['other_binding'] if probe.get('other_binding') else '')


def vocabulary(text):
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())[:250]
    stop = {'the', 'and', 'that', 'this', 'with', 'from', 'have', 'what', 'your', 'they', 'for', 'not', 'are', 'but', 'was'}
    terms = {w for w in words if len(w) > 2 and w not in stop}
    # Preserve self-reference and negation within short local phrases.
    for size in (2, 3):
        terms.update(' '.join(words[i:i+size]) for i in range(len(words)-size+1)
                     if words[i:i+2] in (['i', 'am'], ['i', 'feel'], ['i', 'do'], ['i', "don't"]))
    return sorted(terms)[:96]


def probe_alias(name):
    return 'p_' + hashlib.sha256(('exposed-probe-v1:' + name).encode()).hexdigest()[:24]


def resolve_probe_alias(value, probes):
    return next((name for name, probe in probes.items()
                 if probe.get('exposed') and probe_alias(name) == value), None)


def obscure_probe_identifiers(text, probes):
    """Hash explicit probe references while preserving ordinary speech.

    ``probe_name`` and ``probe:name`` (including ``[probe:name]``) mark
    identifiers. Tool targets are also marked by their protocol syntax; their
    candidate words, proposed labels and evidence remain ordinary text.
    """
    def identifier(name, *, known_only=False):
        if re.fullmatch(r'p_[0-9a-f]{24}', name):
            return name
        if known_only and name not in probes:
            return name
        return probe_alias(name)

    def explicit_references(value):
        # Retired cached identifiers still resolve without a loaded probe.
        value = re.sub(r'(?<![a-zA-Z0-9_])probe_([a-zA-Z0-9_]+)(?![a-zA-Z0-9_])',
                       lambda match: identifier(match.group(1)), value)
        return re.sub(r'(?<![a-zA-Z0-9_])(probe:\s*)([a-zA-Z0-9_]+)(?![a-zA-Z0-9_])',
                      lambda match: match.group(1) + identifier(match.group(2)),
                      value, flags=re.IGNORECASE)

    def tool_body(body):
        head, separator, payload = body.partition('||')
        head = explicit_references(head)
        rename = re.match(r'^(\s*rename\s+)([a-zA-Z0-9_]+)(\s*)$', head, re.IGNORECASE)
        if rename:
            head = rename.group(1) + identifier(rename.group(2), known_only=True) + rename.group(3)
        elif not re.match(r'^\s*(?:other|all)\b', head, re.IGNORECASE):
            head = re.sub(r'[a-zA-Z0-9_]+',
                          lambda match: identifier(match.group(0), known_only=True), head)
        return head + separator + explicit_references(payload)

    # Standalone candidate/naming queries have an explicitly delimited target.
    if re.match(r'^\s*(?:rename\s+)?[a-zA-Z0-9_]+\s*\|\|', text, re.IGNORECASE):
        return tool_body(text)
    parts = []
    cursor = 0
    for match in re.finditer(r'(<<\s*PROBE\s*:\s*)(.*?)(>>)', text, re.IGNORECASE | re.DOTALL):
        parts.append(explicit_references(text[cursor:match.start()]))
        parts.append(match.group(1) + tool_body(match.group(2)) + match.group(3))
        cursor = match.end()
    parts.append(explicit_references(text[cursor:]))
    return ''.join(parts)


class ExposureLanguage:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.maps = {}
        self.active = {}
        if self.path and self.path.exists():
            self.maps = json.loads(self.path.read_text(encoding='utf-8')).get('maps', {})

    def sync(self, probes, tuner):
        active = {}
        for name, probe in sorted(probes.items()):
            if not probe.get('exposed'):
                continue
            poles = framing_pair(probe)
            key = _namespace_key(name, probe)
            mapping = self.maps.setdefault(key, {'n': 0, 'terms': {}})
            trigger = tuner.triggers.get('probe_' + name)
            samples = list(trigger.signals) if trigger is not None else []
            reading = float(samples[-1]) if samples else None
            finite = [float(x) for x in samples[:-1] if math.isfinite(float(x))]
            scale = math.sqrt(sum(x*x for x in finite)/len(finite)) if finite else 0.
            # Centered streams use zero as their pole boundary; scale requires history.
            strength = max(-3., min(3., reading / scale)) if reading is not None and math.isfinite(reading) and len(finite) >= 5 and scale > 1e-6 else None
            active[name] = {'key': key, 'poles': poles, 'strength': strength, 'mapping': mapping,
                            'measurement_source': probe.get('measurement_source'),
                            'reading': reading, 'history': len(finite),
                            'delta': reading - finite[-1] if reading is not None and math.isfinite(reading) and finite else None}
        self.active = active

    def record_namespace_summary(self, name, probe, text, observed_replies, *, status='hypothesis', qualification=None):
        from invariants.probe_definition_check import assess_definition_attempt
        structure = assess_definition_attempt(text)
        qualification = qualification or structure
        accepted = structure['accepted'] and qualification['accepted']
        if status == 'hypothesis' and not accepted:
            status = 'unqualified'
        key = _namespace_key(name, probe)
        mapping = self.maps.setdefault(key, {'n': 0, 'terms': {}})
        values = mapping.setdefault('summary_hypotheses', [])
        values.append({'text': text[:16000], 'observed_replies': observed_replies, 'status': status,
                       'qualified': status == 'hypothesis' and accepted,
                       'qualification_reason': qualification['reason'],
                       'contrast': qualification.get('contrast', '') if accepted else ''})
        mapping['summary_hypotheses'] = values[-16:]
        self.save()

    def observe(self, response, row, probes):
        terms = vocabulary(response)
        changed = False
        for name, snapshot in self.active.items():
            probe = probes.get(name, {})
            if not probe.get('exposed') or _namespace_key(name, probe) != snapshot['key']:
                continue
            reading = row.get('probe_' + name)
            if reading is None or not math.isfinite(float(reading)):
                continue
            # Reply readings associate diction with poles; they do not certify
            # the self-descriptions that generated those readings.
            mapping = snapshot['mapping']
            mapping['n'] += 1
            signal = max(-1., min(1., float(reading)))
            for stats in mapping['terms'].values():
                stats['weight'] *= .99
            for term in terms:
                stats = mapping['terms'].setdefault(term, {'n': 0, 'weight': 0.})
                stats['n'] += 1
                stats['weight'] = .9 * stats['weight'] + .1 * signal
            mapping['terms'] = dict(sorted(mapping['terms'].items(),
                key=lambda item: (-item[1]['n'], -abs(item[1]['weight']), item[0]))[:256])
            changed = True
        if changed:
            self.save()

    def context(self, max_chars=3000, *, behavior=None, probes=None):
        if not self.active:
            return ''
        lines = ['[Exposed sensors: measured behavior and learned language associations]',
                 'Infer distinctions by comparing observations. Positive and negative denote sides of an axis, not good and bad. Intensity and changes are relative measurements.',
                 'Name or revise a sensor when the observations support a useful term at its naming threshold: <<PROBE: rename HASH || local label || confidence 0..1 || supporting observations>>. Confidence is self-reported; names remain revisable. Keep using the hash to consult it.',
                 'With an exposed relational sensor, <<PROBE: other self>> may bind @ to the owning agent.']
        for name, snapshot in self.active.items():
            strength = snapshot['strength']
            pole = 'unresolved' if strength is None or abs(strength) < .5 else ('WITH' if strength > 0 else 'WITHOUT')
            intensity = 'uncalibrated' if strength is None else 'weak' if abs(strength) < .5 else 'moderate' if abs(strength) < 1.5 else 'strong'
            relational = len(snapshot['key'].split(':')) > 2
            line = f'{probe_alias(name)}: prior pole={pole}, intensity={intensity}; measured replies={snapshot["mapping"]["n"]}.'
            if snapshot['reading'] is not None and math.isfinite(snapshot['reading']):
                line += f' Reading={snapshot["reading"]:+.3f}.'
            if snapshot['delta'] is not None:
                line += f' Change from preceding reading={snapshot["delta"]:+.3f}.'
            threshold = naming_threshold(snapshot['mapping'])
            line += f' Naming confidence threshold={threshold:.2f}.'
            if threshold == DEFINED_NAMING_CONFIDENCE:
                line += ' A behavioral definition supports an earlier provisional name; propose or revise one if its pattern fits.'
            if relational:
                line += ' Relational observation: do not translate this directly into an I-am identity.'
            summaries = [s for s in snapshot['mapping'].get('summary_hypotheses', [])
                         if qualified_namespace_summary(s)]
            if summaries:
                selected_summary = summaries[-1]
                from invariants.probe_definition_check import assess_definition_attempt
                contrast = selected_summary.get('contrast') or assess_definition_attempt(selected_summary['text']).get('contrast', '')
                summary = obscure_probe_identifiers(contrast, probes or {})
                summary = re.sub(r'<<.*?>>', '', summary, flags=re.DOTALL).strip()
                line += f' Namespace interpretation hypothesis={summary[:280]!r}.'
            alias = snapshot['mapping'].get('alias')
            if alias:
                line += f' Previous naming hypothesis={alias["label"]!r}, self-reported confidence={alias["confidence"]:.2f}, observed turns={alias["observed_turns"]}.'
            for sign, side in ((1, 'positive'), (-1, 'negative')):
                candidates = [(term, stats) for term, stats in snapshot['mapping']['terms'].items()
                              if stats['n'] >= 3 and sign*stats['weight'] > .03]
                candidates.sort(key=lambda pair: (-sign*pair[1]['weight'], pair[0]))
                if candidates:
                    line += f' Associated diction ({side}): ' + ', '.join(term for term, _ in candidates[:5]) + '.'
            if behavior is not None and probes is not None and name in probes:
                examples = behavior.exposure_examples(name, probes[name], probes)
                if examples:
                    line += ' Contrasting observed replies: ' + examples
            if sum(len(p)+1 for p in lines) + len(line) > max_chars:
                break
            lines.append(line)
        return '\n'.join(lines)

    def rename(self, identifier, label, confidence, evidence, probes):
        name = resolve_probe_alias(identifier, probes)
        snapshot = self.active.get(name)
        if snapshot is None or _namespace_key(name, probes[name]) != snapshot['key']:
            raise ValueError('Only a currently exposed sensor may be named')
        label = ' '.join(label.split())
        if not 1 <= len(label) <= 80 or not re.fullmatch(r"[\w '.,()-]+", label):
            raise ValueError('Use a plain label of 1 to 80 characters')
        confidence = float(confidence)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Confidence must be finite and between 0 and 1')
        threshold = naming_threshold(snapshot['mapping'])
        if confidence < threshold:
            raise ValueError(f'Naming confidence must reach {threshold:.2f} for this namespace')
        evidence = ' '.join(evidence.split())
        if not evidence or len(evidence) > 400 or any(c in evidence for c in '<>'):
            raise ValueError('Include plain supporting observations, at most 400 characters')
        for key, mapping in self.maps.items():
            alias = mapping.get('alias')
            if key != snapshot['key'] and alias and alias['label'].casefold() == label.casefold():
                raise ValueError('That label already belongs to another sensor definition')
        mapping = snapshot['mapping']
        previous = mapping.get('alias')
        if previous:
            mapping.setdefault('alias_history', []).append(dict(previous))
            mapping['alias_history'] = mapping['alias_history'][-16:]
        mapping['alias'] = {'label': label, 'confidence': confidence, 'evidence': evidence,
                            'observed_turns': mapping['n']}
        self.save()
        return f'{identifier}: naming hypothesis {label!r} saved; confidence {confidence:.2f} is self-reported.'

    def naming_query(self, query, probes):
        fields = [part.strip() for part in query.split('||')]
        head = fields[0].split()
        if len(head) != 2 or head[0].lower() != 'rename' or len(fields) != 4:
            raise ValueError('Use rename HASH || label || confidence || observations')
        return head[1], self.rename(head[1], fields[1], fields[2], fields[3], probes)

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({'maps': self.maps}, allow_nan=False), encoding='utf-8')
            temp.replace(self.path)
