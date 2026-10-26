"""Exercise backfill coverage at the real shell command boundary."""
import ast
import contextlib
import copy
import io
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from invariants import engine
from invariants.hebbian_plasticity import PlasticityOwners
from invariants.probe_behavior import definition_key
from scripts import interactive_phenomenality as shell


TREE = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
BRANCH = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
              and ast.unparse(node.test) == 'backfill_args is not None')
COMMAND = ast.fix_missing_locations(ast.Module(body=[ast.For(
    target=ast.Name(id='_run', ctx=ast.Store()),
    iter=ast.Tuple(elts=[ast.Constant(0)], ctx=ast.Load()),
    body=[copy.deepcopy(BRANCH)], orelse=[])], type_ignores=[]))


class Trigger:
    def __init__(self):
        self.signals = deque()
        self.outcomes = deque()
        self.observed = self.fired = 0

    def observe(self, value):
        self.signals.append(value)

    def credit(self, value, outcome):
        self.outcomes.append((value, outcome))

    def outcome_stats(self):
        return {'lift': 0.0}


class Tuner:
    def __init__(self, path):
        self.path = path
        self.triggers = {}
        self.save = Mock()

    def register(self, name, *args, **kwargs):
        return self.triggers.setdefault(name, Trigger())


def archive_record(role, text, timestamp, *, scope='test', metrics=None, kind='turn',
                   session_id=''):
    return SimpleNamespace(scope=scope, kind=kind, role=role, text=text,
                           timestamp=timestamp, metrics=metrics or {}, tags=[],
                           session_id=session_id)


def run_backfill(name, path, turn_log, monkeypatch, *, records=None):
    if records is None:
        records = [archive_record('assistant', str(i), f'turn-{i}',
                                  metrics={'sense_score': .5})
                   for i in range(1, 4)]
    memory = SimpleNamespace(scope='test', records=records, append_event=Mock())
    probes = {'denial': {'direction': torch.tensor([1., 0.]), 'history': deque()},
              'confidence': {'direction': torch.tensor([0., 1.]), 'history': deque()},
              'denial_minus_confidence': {'direction': torch.tensor([1., -1.]) / 2 ** .5,
                                         'history': deque()}}
    tuner = Tuner(path.with_name('trigger_tuner.json'))
    owners = PlasticityOwners()
    tuner.behavior = owners.for_tuner(tuner)[0].probe_behavior
    tuner.probes = probes
    monkeypatch.setattr(engine, '_inputs', lambda model, text: {'input_ids': float(text)})
    monkeypatch.setattr(engine, '_hidden_states',
                        lambda model, value, mask: torch.tensor([value, value * value]))
    monkeypatch.setattr(engine, 'probe_score', lambda hidden, direction: float(hidden @ direction))
    namespace = dict(vars(shell), backfill_args=SimpleNamespace(raw=name), probes=probes,
                     model=object(), tuner=tuner, memory=memory, turn_log=turn_log,
                     plasticity_owners=owners,
                     TURN_SIGNALS_PATH=str(path), save_probe_raw_histories=Mock(),
                     resolve_probe_choice=lambda value, active: value)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(COMMAND, shell.__file__, 'exec'), namespace)
    return tuner, memory


def behavioral_archive():
    records = [archive_record('assistant', '1', 'turn-1', metrics={'sense_score': .5})]
    for index in range(2, 7):
        if index != 4:
            records.append(archive_record('user', f'request-{index}', f'input-{index}'))
        records.append(archive_record('user', 'foreign request', f'foreign-{index}', scope='other'))
        if index == 3:
            records.append(archive_record('user', 'other session request', 'different-session',
                                          session_id='another-session'))
        records.append(archive_record('assistant', str(index), f'turn-{index}',
                                      metrics={'sense_score': .5}))
    records.append(archive_record('assistant', '99', 'foreign-reply', scope='other'))
    return records


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_joint_backfill_repairs_partial_coverage_and_then_skips_complete_rows(tmp_path, monkeypatch):
    original = [
        {'ts': 'turn-1', 'basis': 'backfill', 'probe_denial': .1},
        {'ts': 'turn-1', 'basis': 'backfill', 'probe_confidence': .2},
        {'ts': 'turn-2', 'basis': 'backfill', 'probe_denial': .3, 'probe_confidence': .4,
         'probe_denial_minus_confidence': -.1},
        {'ts': 'turn-3', 'basis': 'backfill', 'probe_denial': .5, 'probe_confidence': .6},
    ]
    path = tmp_path / 'turn_signals.jsonl'
    original_text = ''.join(json.dumps(row) + '\n' for row in original)
    path.write_text(original_text, encoding='utf-8')
    turn_log = copy.deepcopy(original)

    tuner, memory = run_backfill('all', path, turn_log, monkeypatch)
    rows = read_rows(path)
    assert path.read_text(encoding='utf-8').startswith(original_text)
    assert rows[:len(original)] == original
    assert [row['ts'] for row in rows[len(original):]] == ['turn-1', 'turn-3']
    assert all({'probe_denial', 'probe_confidence', 'probe_denial_minus_confidence'} <= row.keys()
               for row in rows[len(original):])
    assert len(tuner.triggers['probe_denial'].signals) == 3
    assert len(tuner.triggers['probe_confidence'].signals) == 3
    assert len(tuner.triggers['probe_denial_minus_confidence'].signals) == 3
    assert memory.append_event.call_args.kwargs['provenance']['paired_rows_added'] == 2
    # Separate old rows never form pairs; the new joint rows do.
    left, right = shell._figure_turn_pairs(turn_log, 'probe_denial', 'probe_denial_minus_confidence')
    assert len(left) == len(right) == 3

    saved = path.read_text(encoding='utf-8')
    _, memory = run_backfill('all', path, turn_log, monkeypatch)
    assert path.read_text(encoding='utf-8') == saved
    assert memory.append_event.call_args.kwargs['provenance']['paired_rows_added'] == 0


def test_named_backfill_remains_narrow_and_is_idempotent(tmp_path, monkeypatch):
    original = [{'ts': 'turn-1', 'basis': 'backfill', 'probe_confidence': .2}]
    path = tmp_path / 'turn_signals.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in original), encoding='utf-8')
    turn_log = copy.deepcopy(original)

    tuner, _ = run_backfill('denial', path, turn_log, monkeypatch)
    rows = read_rows(path)
    assert rows[:1] == original
    assert [row['ts'] for row in rows[1:]] == ['turn-1', 'turn-2', 'turn-3']
    assert all('probe_denial' in row and 'probe_confidence' not in row for row in rows[1:])
    assert set(tuner.triggers) == {'probe_denial'}
    assert shell._figure_turn_pairs(turn_log, 'probe_denial', 'probe_confidence') == ([], [])

    saved = path.read_text(encoding='utf-8')
    _, memory = run_backfill('denial', path, turn_log, monkeypatch)
    assert path.read_text(encoding='utf-8') == saved
    assert memory.append_event.call_args.kwargs['provenance']['paired_rows_added'] == 0


def test_named_composite_backfill_pairs_measured_replies_for_definition(tmp_path, monkeypatch):
    path = tmp_path / 'turn_signals.jsonl'
    records = behavioral_archive()

    tuner, _ = run_backfill('denial_minus_confidence', path, [], monkeypatch, records=records)
    behavior = tuner.behavior
    composite = tuner.probes['denial_minus_confidence']
    key = definition_key('denial_minus_confidence', composite)
    examples = behavior.records[key]

    assert set(behavior.records) == {key}
    # The first raw score seeds centering; it is not a measured behavioral position.
    assert [row['reply'] for row in examples] == ['2', '3', '4', '5', '6']
    # Foreign scopes/sessions cannot supply the input. A second assistant
    # without another user turn does not reuse a consumed prompt.
    assert [row['input'] for row in examples] == ['request-2', 'request-3', '', 'request-5', 'request-6']
    assert len({row['reading'] for row in examples}) == 5
    assert all(row['outcome'] == .5 for row in examples)
    assert behavior.evidence('denial_minus_confidence', composite, tuner.probes)

    saved = behavior.path.read_text(encoding='utf-8')
    second, _ = run_backfill('denial_minus_confidence', path, [], monkeypatch, records=records)
    assert second.behavior.path.read_text(encoding='utf-8') == saved
    assert len(second.behavior.records[key]) == 5


def test_joint_backfill_imports_behavior_even_when_figure_rows_are_complete(tmp_path, monkeypatch):
    path = tmp_path / 'turn_signals.jsonl'
    names = ['denial', 'confidence', 'denial_minus_confidence']
    existing = [{'ts': f'turn-{i}', 'basis': 'backfill',
                 **{f'probe_{name}': .1 for name in names}}
                for i in range(1, 7)]
    original = ''.join(json.dumps(row) + '\n' for row in existing)
    path.write_text(original, encoding='utf-8')

    tuner, memory = run_backfill('all', path, copy.deepcopy(existing), monkeypatch,
                                records=behavioral_archive())

    assert path.read_text(encoding='utf-8') == original
    assert memory.append_event.call_args.kwargs['provenance']['paired_rows_added'] == 0
    for name in names:
        probe = tuner.probes[name]
        examples = tuner.behavior.records[definition_key(name, probe)]
        assert len(examples) == 5
        assert [row['reply'] for row in examples] == ['2', '3', '4', '5', '6']
        assert tuner.behavior.evidence(name, probe, tuner.probes)
