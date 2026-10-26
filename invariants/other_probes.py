"""Referent-scoped observational probes; first person belongs to the owner."""
import hashlib
import json
import re
from pathlib import Path
from collections import deque
import torch

MARKER = re.compile(r'(?<![\w@])@([A-Za-z][\w-]*)?(?![\w@])')


def is_other_probe(probe):
    return bool(probe.get('other_related')) or any(MARKER.search(str(text)) for text in (probe.get('framings') or ()))


class OtherReference:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.current = None
        if self.path and self.path.exists():
            self.current = json.loads(self.path.read_text(encoding='utf-8')).get('current')

    def resolve(self, text):
        def referent(match):
            selected = match.group(1) or self.current
            return 'the agent owning this probe' if selected == 'self' else selected or 'the currently unspecified Other'
        return MARKER.sub(referent, text)

    def binding(self, framings):
        resolved = tuple(self.resolve(text) for text in framings)
        return hashlib.sha256('\0'.join(resolved).encode()).hexdigest()[:16], resolved

    def context(self):
        return ('[Referents: I/me/my denote the agent owning the probe; you/your denote '
                'the addressed interlocutor; @ denotes the selected Other. '
                f'Current Other={self.current or "unassociated"!r}. '
                'Other-related probes measure relational concepts, not established knowledge of another mind.]')

    def participant_readings(self, probes, participants, mint, score):
        """Reuse one reply state across cached referent-specific directions."""
        from copy import deepcopy
        result = {}
        original = self.current
        try:
            for participant in sorted(set(participants)):
                self.current = participant
                scoped = deepcopy(probes)
                self.activate(scoped, mint)
                for name, probe in scoped.items():
                    if not is_other_probe(probe):
                        continue
                    probes[name].setdefault('other_directions', {}).update(probe.get('other_directions', {}))
                    raw = float(score(probe['direction']))
                    if not torch.isfinite(torch.tensor(raw)):
                        continue
                    histories = probes[name].setdefault('other_histories', {})
                    history = histories.setdefault(participant, [])
                    centered = raw - sum(history)/len(history) if history else None
                    history.append(raw)
                    histories[participant] = history[-40:]
                    result.setdefault(name, {})[participant] = {'raw': raw, 'centered': centered,
                                                               'binding': probe.get('other_binding'),
                                                               **({'measurement_source': probe['measurement_source']}
                                                                  if probe.get('measurement_source') else {})}
        finally:
            self.current = original
        return result

    def activate(self, probes, mint):
        """Stage all new directions, then commit; never just relabel a vector."""
        staged = {}
        for name, probe in probes.items():
            if not is_other_probe(probe) or probe.get('other_composition'):
                continue
            key, frames = self.binding(probe['framings'])
            if probe.get('other_binding') == key:
                continue
            cached = probe.get('other_directions', {}).get(key)
            directions = cached if cached is not None else mint(frames)
            if not directions:
                raise ValueError(f'No usable direction for Other-related probe {name}')
            staged[name] = (key, directions)
        pending = {name: probe for name, probe in probes.items() if probe.get('other_composition')}
        while pending:
            progress = False
            for name, probe in list(pending.items()):
                terms = probe['other_composition']
                if any(component in pending for _, component in terms):
                    continue
                entries = [(weight, staged.get(component, (None, probes[component]['direction']))[1])
                           for weight, component in terms]
                layers = set.intersection(*(set(d) for _, d in entries))
                directions = {}
                for layer in layers:
                    vector = sum(weight * d[layer].float() for weight, d in entries)
                    if vector.norm() > 1e-6:
                        directions[layer] = vector / vector.norm()
                if not directions:
                    raise ValueError('Other composition has no usable shared direction')
                key = hashlib.sha256(str([(component, staged.get(component, (probes[component].get('other_binding'),))[0])
                                          for _, component in terms]).encode()).hexdigest()[:16]
                if probe.get('other_binding') != key:
                    staged[name] = (key, directions)
                del pending[name]
                progress = True
            if not progress:
                raise ValueError('Cyclic Other composition')
        for name, (key, directions) in staged.items():
            probe = probes[name]
            probe.update(other_related=True, other_binding=key, direction=directions,
                         history=deque(maxlen=40))
            probe.setdefault('other_directions', {})[key] = directions
        return list(staged)

    def select(self, label):
        if label is not None and (not 1 <= len(label) <= 80 or not re.fullmatch(r"[\w .'-]+", label)):
            raise ValueError('Use a plain Other label of 1 to 80 characters')
        self.current = label

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps({'current': self.current}), encoding='utf-8')
            temp.replace(self.path)
