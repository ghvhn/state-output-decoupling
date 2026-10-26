import copy
import torch
from types import SimpleNamespace
from invariants.probe_behavior import ProbeBehavior, definition_key, build_definition_handles
from invariants.exposure_language import ExposureLanguage


def probe():
    return {'framings': ('WITH secret_label', 'WITHOUT secret_label'), 'direction': {0: torch.tensor([0., 1., 0.])}}


def test_definition_uses_paired_behavior_not_operator_framings(tmp_path):
    behavior = ProbeBehavior(tmp_path/'behavior.json')
    probes = {'secret_label': probe()}
    assert behavior.evidence('secret_label', probes['secret_label'], probes) is None
    for i, value in enumerate((-.7, -.2, .4, .8)):
        behavior.observe(f'Actual behavior example {i}', f'User input {i}', {'probe_secret_label': value, 'sense': .1*i}, probes)
    evidence = behavior.evidence('secret_label', probes['secret_label'], probes)
    assert 'Actual behavior example 0' in evidence and 'Actual behavior example 3' in evidence
    assert 'WITH secret_label' not in evidence and 'secret_label' not in evidence
    assert "probe's spectrum" in evidence
    assert 'Memory 0: position=-0.7000' in evidence
    assert 'Memory 3: position=+0.8000' in evidence
    assert '[Negative side]' in evidence and '[Positive side]' in evidence
    assert 'side=negative side' in evidence and 'side=positive side' in evidence
    assert evidence.index('position=-0.7000') < evidence.index('position=+0.8000')
    assert 'outcome=' not in evidence and 'rename' not in evidence and 'confidence' not in evidence
    assert behavior.records[definition_key('secret_label', probes['secret_label'])][0]['outcome'] == 0.0
    resumed = ProbeBehavior(tmp_path/'behavior.json')
    assert resumed.evidence('secret_label', probes['secret_label'], probes) == evidence
    probes['secret_label']['other_binding'] = 'different_referent'
    assert resumed.evidence('secret_label', probes['secret_label'], probes) is None


def test_memory_references_show_measured_sides_without_relabeling_their_meanings():
    behavior = ProbeBehavior()
    p = probe()
    for value in (-.7, 0., .4, .8):
        behavior.observe('Observed reply', 'input', {'probe_p': value}, {'p': p})
    text = behavior.annotate_definition_memories('p', p, '**Memory 0:** text; Memory 1; Memory #3; Memory 99')
    assert '**Memory 0 [negative side; position=-0.7000]:**' in text
    assert 'Memory 1 [baseline; position=+0.0000]' in text
    assert 'Memory #3 [positive side; position=+0.8000]' in text
    assert 'Memory 99 [not in supplied probe evidence]' in text
    # Both extreme groups can lie on the positive side of the rolling baseline.
    behavior = ProbeBehavior()
    for value in (.1, .2, .3, .4):
        behavior.observe('Observed reply', 'input', {'probe_p': value}, {'p': p})
    evidence = behavior.evidence('p', p, {'p': p})
    assert '[Negative side]' not in evidence and 'side=negative' not in evidence


def test_constant_or_unmeasured_readings_do_not_support_definition():
    behavior = ProbeBehavior()
    probes = {'p': probe()}
    for value in (1., 1., 1., 1., float('nan')):
        behavior.observe('A reply', 'input', {'probe_p': value}, probes)
    assert len(behavior.records[definition_key('p', probes['p'])]) == 4
    assert behavior.evidence('p', probes['p'], probes) is None


def test_subject_marker_is_once_inquiry_persists_and_joint_cap_holds():
    layer = torch.nn.Identity()
    model = SimpleNamespace(device='cpu', model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
    handles = build_definition_handles(model, {0: torch.tensor([1., 0., 0.])}, {0: torch.tensor([0., 1., 0.])}, cap_fraction=.05)
    state = torch.ones(1, 3, 3)
    try:
        first = layer(state)
        decoded_state = state[:, -1:, :]
        decoded = layer(decoded_state)
        assert first[0, -1, 0] > decoded[0, -1, 0]
        assert decoded[0, -1, 1] > decoded_state[0, -1, 1]
        rebuilt_state = torch.ones(1, 7, 3)
        rebuilt = layer(rebuilt_state)
        torch.testing.assert_close(first, rebuilt[:, :3, :])
        assert rebuilt[0, 2, 0] > rebuilt_state[0, 2, 0]
        torch.testing.assert_close(rebuilt[0, -1, 0], rebuilt_state[0, -1, 0])
        assert torch.all((first-state).norm(dim=-1) <= .05*state.norm(dim=-1) + 1e-6)
    finally:
        for handle in handles:
            handle.remove()
    torch.testing.assert_close(layer(state), state)


def test_definition_injection_scales_with_residual_and_obeys_smaller_cap():
    def injected(scale, cap):
        layer = torch.nn.Identity()
        model = SimpleNamespace(device='cpu', model=SimpleNamespace(model=SimpleNamespace(layers=[layer])))
        handles = build_definition_handles(model, {0: torch.tensor([1., 0., 0.])},
                                           {0: torch.tensor([0., 1., 0.])}, cap_fraction=cap)
        state = scale * torch.ones(1, 3, 3)
        try:
            return state, layer(state), layer(state[:, -1:, :])
        finally:
            for handle in handles:
                handle.remove()
    low, first_low, next_low = injected(1., .1)
    high, first_high, next_high = injected(100., .1)
    torch.testing.assert_close(first_high - high, 100 * (first_low - low))
    torch.testing.assert_close(next_high - high[:, -1:, :], 100 * (next_low - low[:, -1:, :]))
    ratio = (next_high - high[:, -1:, :]).norm(dim=-1) / high[:, -1:, :].norm(dim=-1)
    torch.testing.assert_close(ratio, torch.full_like(ratio, .04))
    state, first, later = injected(100., .01)
    assert torch.all((first-state).norm(dim=-1) <= .01*state.norm(dim=-1) + 1e-5)
    assert torch.all((later-state[:, -1:, :]).norm(dim=-1) <= .01*state[:, -1:, :].norm(dim=-1) + 1e-5)


def test_namespace_summary_is_revisable_and_exposure_scoped(tmp_path):
    probes = {'p': probe()}
    language = ExposureLanguage(tmp_path/'language.json')
    language.record_namespace_summary('p', probes['p'],
                                     'Lower positions settle quickly, whereas higher positions check alternatives before choosing.', 4)
    assert language.context() == ''
    probes['p']['exposed'] = True
    language.sync(probes, SimpleNamespace(triggers={}))
    assert 'check alternatives' in language.context()
    language.record_namespace_summary('p', probes['p'],
                                     'More specifically, lower positions guess, whereas higher positions compare evidence before deciding.', 8)
    assert 'More specifically' in language.context()
    probes['p']['framings'] = ('new', 'other')
    language.sync(probes, SimpleNamespace(triggers={}))
    assert 'More specifically' not in language.context()


def test_parallel_referents_learn_without_selection_and_without_duplicates():
    behavior = ProbeBehavior()
    probes = {'social': dict(probe(), other_binding='user-binding', other_related=True)}
    for value in (-.8, -.2, .3, .9):
        behavior.observe('A shared behavioral reply', 'input', {
            'probe_social': value,
            'other_probe_readings': {'social': {
                'user': {'centered': value, 'binding': 'user-binding'},
                'self': {'centered': -value, 'binding': 'self-binding'}}}}, probes)
    user_key = definition_key('social', probes['social'])
    self_probe = dict(probes['social'], other_binding='self-binding')
    self_key = definition_key('social', self_probe)
    assert len(behavior.records[user_key]) == len(behavior.records[self_key]) == 4
    assert behavior.evidence('social', self_probe, probes) is not None


def archived_observations():
    return [{'ts': f'archive-{i}', 'reply': f'Archived reply {i}', 'input': f'Actual user input {i}',
             'readings': {'p': value}, 'outcome': i / 10}
            for i, value in enumerate((-.7, -.2, .4, .8))]


def test_archived_measured_replies_support_definition_and_repeat_idempotently(tmp_path):
    path = tmp_path / 'behavior.json'
    behavior = ProbeBehavior(path)
    probes = {'p': probe()}
    observations = [{'ts': 'first-unbaselined', 'reply': 'Seed reply', 'readings': {}}]
    observations += archived_observations()
    counts = behavior.backfill(observations, probes)
    assert counts['imported'] == 4 and counts['replaced'] == 0
    key = definition_key('p', probes['p'])
    rows = behavior.records[key]
    assert len(rows) == 4
    assert all(row['basis'] == 'backfill' for row in rows)
    assert [row['source_ts'] for row in rows] == [f'archive-{i}' for i in range(4)]
    assert [row['reading'] for row in rows] == [-.7, -.2, .4, .8]
    assert rows[3]['input'] == 'Actual user input 3' and rows[3]['outcome'] == .3
    evidence = behavior.evidence('p', probes['p'], probes)
    assert 'Archived reply 0' in evidence and 'Archived reply 3' in evidence
    assert 'Seed reply' not in evidence and 'Actual user input 0' in evidence

    saved = path.read_text(encoding='utf-8')
    resumed = ProbeBehavior(path)
    repeated = resumed.backfill(observations, probes)
    assert repeated['imported'] == repeated['replaced'] == 0
    assert path.read_text(encoding='utf-8') == saved
    assert resumed.records == behavior.records


def test_backfill_replaces_archival_baselines_but_preserves_matching_live_reply():
    behavior = ProbeBehavior()
    probes = {'p': probe()}
    behavior.observe('Actual live reply', 'Actual live question',
                     {'probe_p': .9, 'source_ts': 'live-source', 'ts': 'later-metric-time'}, probes)
    key = definition_key('p', probes['p'])
    live = copy.deepcopy(behavior.records[key][0])
    assert live['source_ts'] == 'live-source'
    observations = archived_observations()
    observations.append({'ts': 'live-source', 'reply': 'Actual live reply', 'input': 'Actual live question',
                         'readings': {'p': -.9}})
    counts = behavior.backfill(observations, probes)
    assert counts['imported'] == 4 and counts['replaced'] == 0
    assert len(behavior.records[key]) == 5
    assert behavior.records[key][-1] == live

    changed = copy.deepcopy(observations)
    changed[0]['readings']['p'] = -.6
    counts = behavior.backfill(changed, probes)
    assert counts['imported'] == 0 and counts['replaced'] == 1
    updated = next(row for row in behavior.records[key] if row['source_ts'] == 'archive-0')
    assert updated['reading'] == -.6 and updated['basis'] == 'backfill'
    assert behavior.records[key][-1] == live


def test_backfill_keeps_changed_definitions_in_separate_namespaces():
    behavior = ProbeBehavior()
    probes = {'p': probe()}
    observations = archived_observations()
    behavior.backfill(observations, probes)
    original_key = definition_key('p', probes['p'])
    original = copy.deepcopy(behavior.records[original_key])
    probes['p']['framings'] = ('Changed first pole', 'Changed second pole')
    changed_key = definition_key('p', probes['p'])
    counts = behavior.backfill(observations, probes)
    assert counts['imported'] == 4 and counts['replaced'] == 0
    assert original_key != changed_key
    assert behavior.records[original_key] == original
    assert len(behavior.records[changed_key]) == 4


def test_backfill_rejects_invalid_readings_and_requires_archived_other_binding():
    behavior = ProbeBehavior()
    probes = {'p': probe(), 'social': dict(probe(), other_related=True, other_binding='current-user')}
    observations = [{'ts': f'invalid-{i}', 'reply': 'A reply', 'readings': {'p': value}}
                    for i, value in enumerate((None, float('nan'), float('inf'), 'not numeric', False))]
    observations += [{'ts': 'regular', 'reply': 'Measured reply', 'input': None,
                      'readings': {'p': .2, 'social': .7}, 'outcome': float('nan')},
                     {'ts': 'bound', 'reply': 'Archived relational reply', 'readings': {'social': -.3},
                      'bindings': {'social': 'archived-self'}}]
    counts = behavior.backfill(observations, probes)
    assert counts['imported'] == 2 and counts['replaced'] == 0
    regular = behavior.records[definition_key('p', probes['p'])]
    assert len(regular) == 1 and regular[0]['reading'] == .2
    assert regular[0]['input'] == '' and regular[0]['outcome'] is None
    assert definition_key('social', probes['social']) not in behavior.records
    archived_probe = dict(probes['social'], other_binding='archived-self')
    archived = behavior.records[definition_key('social', archived_probe)]
    assert len(archived) == 1 and archived[0]['reading'] == -.3


def test_long_archival_batches_keep_live_records_and_are_idempotent_at_capacity():
    behavior = ProbeBehavior()
    probes = {'p': probe()}
    behavior.observe('Live reply', 'Live question', {'probe_p': .5, 'ts': 'live'}, probes)
    key = definition_key('p', probes['p'])
    live = copy.deepcopy(behavior.records[key][0])
    observations = [{'ts': f'archive-{i:03}', 'reply': f'Reply {i}', 'readings': {'p': i / 100}}
                    for i in range(120)]
    counts = behavior.backfill(observations, probes)
    assert counts['imported'] == 63 and counts['replaced'] == 0
    assert len(behavior.records[key]) == 64
    assert behavior.records[key][0]['source_ts'] == 'archive-057'
    assert behavior.records[key][-1] == live
    saved = copy.deepcopy(behavior.records)
    repeated = behavior.backfill(observations, probes)
    assert repeated['imported'] == repeated['replaced'] == 0
    assert behavior.records == saved


def test_identical_live_observations_keep_distinct_stable_ids_at_capacity(tmp_path):
    behavior = ProbeBehavior(tmp_path / 'behavior.json')
    probes = {'p': probe()}
    key = definition_key('p', probes['p'])
    for _ in range(64):
        behavior.observe('Identical measured reply', 'Identical question', {'probe_p': .5}, probes)
    previous = {row['record_id'] for row in behavior.records[key]}
    assert len(previous) == 64
    behavior.observe('Identical measured reply', 'Identical question', {'probe_p': .5}, probes)
    current = {row['record_id'] for row in behavior.records[key]}
    assert len(current) == 64 and len(current - previous) == 1
    resumed = ProbeBehavior(behavior.path)
    assert resumed.records == behavior.records
    assert all(row['reply'] == 'Identical measured reply' for row in resumed.records[key])


def test_backfill_keeps_existing_live_ids_and_does_not_mint_archive_ids():
    behavior = ProbeBehavior()
    probes = {'p': probe()}
    key = definition_key('p', probes['p'])
    behavior.observe('Live measured reply', 'Question', {'probe_p': .5}, probes)
    live_id = behavior.records[key][0]['record_id']
    behavior.backfill(archived_observations(), probes)
    assert behavior.records[key][-1]['record_id'] == live_id
    assert all('record_id' not in row for row in behavior.records[key] if row.get('basis') == 'backfill')


def test_parallel_other_observations_have_distinct_ids_without_changing_evidence():
    behavior = ProbeBehavior()
    p = dict(probe(), other_related=True, other_binding='selected')
    probes = {'social': p}
    row = {'other_probe_readings': {'social': {
        'user': {'centered': -.3, 'binding': 'user-binding'},
        'self': {'centered': .3, 'binding': 'self-binding'},
    }}}
    behavior.observe('A shared measured reply', 'Question', row, probes)
    records = [record for rows in behavior.records.values() for record in rows]
    assert len(records) == 2 and len({record['record_id'] for record in records}) == 2
    assert all(record['reply'] == 'A shared measured reply' for record in records)
