"""Owner-local, contextual relationship trials for a selected target stream.

Online crossover evidence is suggestive, not causal identification: prompts
and hidden state change between turns. Ordinary turns provide a drifting
baseline; bounded signed trials measure subsequent target change against it.
"""
import itertools
import json
import math
from pathlib import Path


class TargetRelations:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.target = None
        self.controls = []
        self.direction = 1.
        self.enabled = False
        self.budget = .15
        self.last_target = None
        self.baselines = {}
        self.records = {}
        self.turn = 0
        self.pending = None
        self.last_result = None
        self.mode = 'observe'
        self.observations = {}
        self.previous_signals = None
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if data.get('target'):
                self.configure(data['target'], data['controls'], data['direction'])
            self.enabled = bool(data.get('enabled', False))
            self.records = data.get('records', {})
            self.baselines = data.get('baselines', {})
            self.turn = int(data.get('turn', 0))
            self.mode = data.get('mode', 'observe')
            self.observations = data.get('observations', {})
            # Deliberately discard stale in-flight actions and target baselines.

    def configure(self, target, controls, direction=1.):
        controls = sorted(set(controls))
        if not target or len(controls) < 2:
            raise ValueError('Select a target stream and at least two controllable probes')
        if len(controls) > 8:
            raise ValueError('Use at most eight controls per target')
        if target.startswith('probe_') and target[6:] in controls:
            raise ValueError('Keep the target probe separate from its candidate controls')
        if direction not in (-1, 1):
            raise ValueError('Direction must be maximize or minimize')
        self.target, self.controls, self.direction = target, controls, float(direction)
        self.enabled = True
        self.last_target = self.pending = self.last_result = None
        self.records, self.baselines, self.turn = {}, {}, 0
        self.observations, self.previous_signals = {}, None
        self.mode = 'observe'

    def _context(self, signals):
        # Coarse signed context keeps trials comparable and inspectable.
        return ','.join(f'{name}:{1 if signals.get(name, 0.) > .05 else -1 if signals.get(name, 0.) < -.05 else 0}'
                        for name in self.controls)

    def prepare(self, signals, available):
        if not self.enabled or self.last_target is None or self.mode != 'trial':
            return {}
        if self.pending is not None:
            return self.pending['overlay']
        names = sorted(set(self.controls) & set(available))
        pairs = list(itertools.combinations(names, 2))
        context = self._context(signals)
        ready = {row['relation'] for row in self.evidence() if row['ready_for_trial']}
        pairs = [pair for pair in pairs if context + '|' + '|'.join(pair) in ready]
        overlay = {}
        choice = None
        # Every other eligible turn is an ordinary-field comparison.
        if pairs and self.turn % 2:
            def score(pair):
                record = self.records.get(context + '|' + '|'.join(pair), {})
                return abs(record.get('effect', 0.)) if record.get('n', 0) >= 4 else -1.
            # Reserve one in four trials for balanced exploration.
            trial = self.turn // 2
            pair = max(pairs, key=score) if trial % 4 and max(map(score, pairs)) >= 0 else pairs[(trial // 2) % len(pairs)]
            key = context + '|' + '|'.join(pair)
            record = self.records.get(key, {})
            sign = (1. if record.get('effect', 0.) >= 0 else -1.) if record.get('n', 0) >= 4 and trial % 4 else (1. if trial % 2 == 0 else -1.)
            choice = {'pair': pair, 'key': key, 'sign': sign}
            overlay['|'.join(pair)] = sign * self.budget
        self.pending = {'context': context, 'before': self.last_target,
                        'overlay': overlay, 'choice': choice}
        return overlay

    def adjustment(self, a, b):
        if not self.enabled or self.pending is None:
            return 0.
        return self.pending['overlay'].get('|'.join(sorted((a, b))), 0.)

    def observe(self, row):
        if not self.enabled:
            return
        value = row.get(self.target)
        if value is None:
            self.pending = None
            self.last_target = None
            self.previous_signals = None
            return
        try:
            if isinstance(value, (dict, list, tuple, set, bool)):
                self.pending = None
                self.last_target = None
                self.previous_signals = None
                return
            f_value = float(value)
            if not math.isfinite(f_value):
                self.pending = None
                self.last_target = None
                self.previous_signals = None
                return
            value = f_value
        except (TypeError, ValueError):
            self.pending = None
            self.last_target = None
            self.previous_signals = None
            return

        signals = {}
        for name in self.controls:
            k = 'probe_' + name
            if k in row:
                v = row[k]
                if not isinstance(v, (dict, list, tuple, set, bool)):
                    try:
                        fv = float(v)
                        if math.isfinite(fv):
                            signals[name] = fv
                    except (TypeError, ValueError):
                        pass
        if self.previous_signals is not None and self.last_target is not None:
            context = self._context(self.previous_signals)
            reward = max(-1., min(1., self.direction * (value - self.last_target)))
            for a, b in itertools.combinations(self.controls, 2):
                if any(name not in signals or name not in self.previous_signals for name in (a, b)):
                    continue
                # Change in joint expression, paired with change in target.
                x = max(-1., min(1., signals[a] * signals[b] - self.previous_signals[a] * self.previous_signals[b]))
                key = context + '|' + a + '|' + b
                stats = self.observations.setdefault(key, {'n': 0, 'x': 0., 'y': 0., 'xx': 0., 'yy': 0., 'xy': 0.})
                stats['n'] += 1
                for field, amount in (('x', x), ('y', reward), ('xx', x*x), ('yy', reward*reward), ('xy', x*reward)):
                    stats[field] += amount
        pending, self.pending = self.pending, None
        if pending is not None:
            reward = max(-1., min(1., self.direction * (value - pending['before'])))
            context = pending['context']
            baseline = self.baselines.get(context, {'mean': 0., 'n': 0})
            choice = pending['choice']
            if choice is None:
                baseline['mean'] += .2 * (reward - baseline['mean'])
                baseline['n'] += 1
                self.baselines[context] = baseline
            elif baseline['n']:
                record = self.records.setdefault(choice['key'], {'n': 0, 'effect': 0.})
                estimate = max(-1., min(1., (reward - baseline['mean']) * choice['sign']))
                record['effect'] = .8 * record['effect'] + .2 * estimate
                record['n'] += 1
            self.last_result = {'target_delta': reward, 'trial': choice,
                                'baseline': baseline['mean']}
            self.turn += 1
        self.last_target = value
        self.previous_signals = signals
        self.save()

    def evidence(self):
        result = []
        for key, s in self.observations.items():
            n = s['n']
            vx, vy = s['xx'] - s['x']**2/n, s['yy'] - s['y']**2/n
            covariance = s['xy'] - s['x']*s['y']/n
            varied = vx > 1e-5 and vy > 1e-5
            correlation = covariance / math.sqrt(vx*vy) if varied else 0.
            result.append({'relation': key, 'n': n, 'correlation': max(-1., min(1., correlation)),
                           'sufficient_variation': varied,
                           'ready_for_trial': n >= 8 and varied})
        return sorted(result, key=lambda r: (-abs(r['correlation']), r['relation']))[:10]

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {key: getattr(self, key) for key in
                    ('target', 'controls', 'direction', 'enabled', 'records', 'baselines', 'turn', 'mode', 'observations')}
            temp = self.path.with_suffix('.tmp')
            temp.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
            temp.replace(self.path)

    def status(self):
        return (f'Target learning {"on" if self.enabled else "off"} ({self.mode}): {self.target or "unset"}; '
                f'controls={",".join(self.controls)}; trials={sum(r["n"] for r in self.records.values())}; '
                f'relationship budget={self.budget:g}; last={self.last_result}; evidence={self.evidence()}')


def handle_target_command(args, controller, probes, tuner):
    if not args or args == ['status']:
        return controller.status()
    if args in (['observe'], ['trial']):
        if args == ['trial'] and not any(e['ready_for_trial'] for e in controller.evidence()):
            raise ValueError('Trials need at least eight comparable observations with target and relationship variation')
        controller.mode = args[0]
        controller.pending = None
    elif args[0] in ('off', 'on') and len(args) == 1:
        if args[0] == 'on' and controller.target is None:
            raise ValueError('Configure a target first')
        controller.enabled = args[0] == 'on'
        controller.pending = None
        controller.last_target = None
        controller.previous_signals = None
    elif args[0] == 'target' and len(args) >= 4:
        target = args[1]
        if target in probes:
            target = 'probe_' + target
        if target not in tuner.triggers:
            raise ValueError('Target must be an observed tuner stream or probe')
        if args[2] not in ('maximize', 'minimize'):
            raise ValueError('Use maximize or minimize')
        controls = args[3:]
        if any(name not in probes for name in controls):
            raise ValueError('Controls must name existing probes')
        controller.configure(target, controls, 1 if args[2] == 'maximize' else -1)
    else:
        raise ValueError(':relate target <metric|probe> maximize|minimize <probeA> <probeB> [...] | status | off | on')
    controller.save()
    return controller.status()
