"""Persistent diagnostics cannot authorize probe meanings or steering changes."""

import copy
import json

import pytest

from invariants.probe_conception import (
    ProbeConceptionLedger, ProbeDiagnosticWatch, measurement_record_ids,
)


def report(*, token='revise', behavior='change', score=2.0):
    return {
        'version': 'geometric-probe-diagnostic-v1', 'namespace': 'p_test',
        'semantic_validated': False, 'rename_authority': False,
        'direction_sha256': 'a' * 64,
        'model_identity': {'model': 'native-test', 'tokenizer_identity': {'class': 'native-tokenizer'}},
        'vocabulary': {'status': 'candidate', 'layers': {
            '2': {'positive': [{'id': 3, 'token': token, 'score': score}],
                  'negative': [{'id': 4, 'token': 'retain', 'score': -1.0}]},
            '7': {'positive': [{'id': 8, 'token': 'respond', 'score': 0.5}],
                  'negative': [{'id': 9, 'token': 'repeat', 'score': -0.25}]},
        }},
        'behavior': {
            'reply': {'status': 'supported',
                      'support': {'records': 6, 'low_ids': [0, 1], 'high_ids': [4, 5]},
                      'high': [{'term': behavior, 'log_odds': 1.5, 'high_count': 2, 'low_count': 0}],
                      'low': [{'term': 'same', 'log_odds': -2.0, 'high_count': 0, 'low_count': 2}]},
            'input': {'status': 'insufficient', 'high': [], 'low': []},
        },
        'alignment': {'status': 'descriptive', 'semantic_validated': False},
    }


def metadata():
    return {'source_scope': {'schema': 'paired-64-v1', 'binding': 'speaker-1', 'tail_fraction': 0.33}}


def test_separate_channels_and_no_authoritative_meaning(tmp_path):
    ledger = ProbeConceptionLedger(tmp_path / 'diagnostics.json')
    original = report()
    before = copy.deepcopy(original)
    first = ledger.observe('p_test', original, metadata())
    assert original == before
    assert first['kind'] == 'diagnostic_observation'
    assert first['status'] == 'baseline'
    assert first['semantic_validated'] is False
    assert first['rename_authority'] is False
    assert first['comparison']['channel_drift'] == {}
    channels = first['candidate']['channels']
    assert channels['vocabulary.layer.2.positive'] == {'revise': 1.0}
    assert channels['vocabulary.layer.2.negative'] == {'retain': 1.0}
    assert channels['behavior.reply.high'] == {'change': 1.0}
    assert 'behavior.input.high' not in channels
    assert first['candidate']['semantic_validated'] is False
    assert first['report']['vocabulary']['layers']['2']['positive'][0]['score'] == 2.0
    assert first['report']['vocabulary']['layers']['2']['negative'][0]['score'] == -1.0
    assert ProbeConceptionLedger(tmp_path / 'diagnostics.json').get('p_test') == ledger.get('p_test')


def test_low_smoothing_revises_lexical_candidates_not_probe_vector():
    ledger = ProbeConceptionLedger()
    ledger.observe('p_test', report(), metadata())
    second = ledger.observe('p_test', report(token='switch', behavior='adopt'), metadata())
    assert second['status'] == 'revision'
    assert second['comparison']['comparable'] is True
    assert second['candidate']['channels']['vocabulary.layer.2.positive'] == {'revise': 0.8, 'switch': 0.2}
    assert second['candidate']['channels']['behavior.reply.high'] == {'change': 0.8, 'adopt': 0.2}
    assert second['comparison']['channel_drift']['vocabulary.layer.2.positive'] == pytest.approx(0.2)
    assert ledger.get('p_test')['signature']['direction'] == 'a' * 64
    assert second['comparison']['meaning_validated'] is False


def test_zero_drift_cannot_validate_meaning():
    ledger = ProbeConceptionLedger()
    ledger.observe('p_test', report(), metadata())
    second = ledger.observe('p_test', report(), metadata())
    assert all(value == 0 for value in second['comparison']['channel_drift'].values())
    assert second['comparison']['meaning_validated'] is False
    assert second['semantic_validated'] is False
    assert second['candidate']['rename_authority'] is False
    assert 'stable' not in json.dumps(second)


def test_identical_diagnostics_do_not_resmooth_write_or_grow_history(tmp_path, monkeypatch):
    destination = tmp_path / 'ledger.json'
    ledger = ProbeConceptionLedger(destination)
    ledger.observe('p_test', report(), metadata())
    first_revision = ledger.observe('p_test', report(token='new'), metadata())
    before, before_file = ledger.get('p_test'), destination.read_bytes()
    def reject_write(*args):
        raise AssertionError('identical evidence must not attempt a write')
    monkeypatch.setattr('invariants.probe_conception.os.replace', reject_write)
    for _ in range(10):
        unchanged = ledger.observe('p_test', report(token='new'), metadata())
        assert unchanged['status'] == 'unchanged'
        assert unchanged['comparison']['reason'] == 'identical_observation'
        assert unchanged['candidate'] == first_revision['candidate']
    assert ledger.get('p_test') == before
    assert destination.read_bytes() == before_file
    restored = ProbeConceptionLedger(destination)
    assert restored.observe('p_test', report(token='new'), metadata())['status'] == 'unchanged'


@pytest.mark.parametrize('changed', ['model', 'tokenizer', 'direction', 'scope', 'version'])
def test_scope_changes_reset_baseline_and_disallow_drift(changed):
    ledger = ProbeConceptionLedger()
    ledger.observe('p_test', report(), metadata())
    new_report, new_metadata = report(token='new'), metadata()
    if changed == 'model':
        new_report['model_identity']['model'] = 'different-model'
    elif changed == 'tokenizer':
        new_report['model_identity']['tokenizer_identity']['class'] = 'different-tokenizer'
    elif changed == 'direction':
        new_report['direction_sha256'] = 'b' * 64
    elif changed == 'scope':
        new_metadata['source_scope']['binding'] = 'speaker-2'
    else:
        new_report['version'] = 'geometric-probe-diagnostic-v2'
    revised = ledger.observe('p_test', new_report, new_metadata)
    assert revised['status'] == 'baseline_scope_change'
    assert revised['comparison']['comparable'] is False
    assert revised['comparison']['channel_drift'] == {}
    assert revised['candidate']['channels']['vocabulary.layer.2.positive'] == {'new': 1.0}
    assert len(ledger.get('p_test')['history']) == 2


def test_namespaces_cannot_share_candidates():
    ledger = ProbeConceptionLedger()
    ledger.observe('p_one', report(token='first'), metadata())
    ledger.observe('p_two', report(token='second'), metadata())
    assert ledger.get('p_one')['candidate']['channels']['vocabulary.layer.2.positive'] == {'first': 1.0}
    assert ledger.get('p_two')['candidate']['channels']['vocabulary.layer.2.positive'] == {'second': 1.0}


def test_insufficient_channels_preserve_prior_as_unvalidated_history():
    ledger = ProbeConceptionLedger()
    initial = ledger.observe('p_test', report(), metadata())
    incomplete = report()
    incomplete['vocabulary'] = {'status': 'insufficient', 'layers': {}}
    incomplete['behavior']['reply']['status'] = 'insufficient'
    current = ledger.observe('p_test', incomplete, metadata())
    assert current['observed_channels'] == []
    assert current['candidate'] == initial['candidate']
    assert current['comparison']['meaning_validated'] is False


def test_history_terms_and_namespace_bounds(tmp_path):
    ledger = ProbeConceptionLedger(tmp_path / 'ledger.json', max_history=2, max_namespaces=2, max_terms=1)
    for token in ('a', 'b', 'c'):
        ledger.observe('first', report(token=token), metadata())
    state = ledger.get('first')
    assert state['observations'] == 3 and len(state['history']) == 2
    assert len(state['candidate']['channels']['vocabulary.layer.2.positive']) == 1
    ledger.observe('second', report(), metadata())
    ledger.observe('third', report(), metadata())
    assert ledger.get('first') is None
    assert ProbeConceptionLedger(tmp_path / 'ledger.json', max_history=2, max_namespaces=2, max_terms=1).get('third')


def test_read_return_and_input_metadata_cannot_mutate_internal_state():
    ledger = ProbeConceptionLedger()
    context = metadata()
    result = ledger.observe('p_test', report(), context)
    context['source_scope']['binding'] = 'overwrite'
    result['candidate']['channels'].clear()
    retrieved = ledger.get('p_test')
    retrieved['candidate']['channels'].clear()
    assert ledger.get('p_test')['candidate']['channels']
    assert ledger.get('p_test')['signature']['source_scope']['binding'] == 'speaker-1'


@pytest.mark.parametrize('malformation', ['nan', 'authority', 'tokenizer', 'direction', 'scope', 'object'])
def test_invalid_observations_are_transactional(tmp_path, malformation):
    destination = tmp_path / 'ledger.json'
    ledger = ProbeConceptionLedger(destination)
    ledger.observe('p_test', report(), metadata())
    before, before_file = ledger.get('p_test'), destination.read_bytes()
    invalid, context = report(), metadata()
    if malformation == 'nan':
        invalid['vocabulary']['layers']['2']['positive'][0]['score'] = float('nan')
    elif malformation == 'authority':
        invalid['semantic_validated'] = True
    elif malformation == 'tokenizer':
        invalid['model_identity'].pop('tokenizer_identity')
    elif malformation == 'direction':
        invalid['direction_sha256'] = 'same-vector'
    elif malformation == 'scope':
        context.clear()
    else:
        invalid['arbitrary'] = object()
    with pytest.raises(ValueError):
        ledger.observe('p_test', invalid, context)
    assert ledger.get('p_test') == before
    assert destination.read_bytes() == before_file


def test_atomic_replace_failure_preserves_state_file_and_cleans_temporary(tmp_path, monkeypatch):
    destination = tmp_path / 'ledger.json'
    ledger = ProbeConceptionLedger(destination)
    ledger.observe('p_test', report(), metadata())
    before, before_file = ledger.get('p_test'), destination.read_bytes()
    def reject(*args):
        raise OSError('simulated interrupted replace')
    monkeypatch.setattr('invariants.probe_conception.os.replace', reject)
    with pytest.raises(OSError):
        ledger.observe('p_test', report(token='different'), metadata())
    assert ledger.get('p_test') == before and destination.read_bytes() == before_file
    assert list(tmp_path.glob('*.tmp')) == []


@pytest.mark.parametrize('corruption', ['version', 'signature', 'candidate', 'history_authority'])
def test_serialized_corruption_rejected(tmp_path, corruption):
    destination = tmp_path / 'ledger.json'
    ProbeConceptionLedger(destination).observe('p_test', report(), metadata())
    payload = json.loads(destination.read_text(encoding='utf-8'))
    if corruption == 'version':
        payload['format_version'] = 2
    elif corruption == 'signature':
        payload['records']['p_test']['signature']['direction'] = 'b' * 64
    elif corruption == 'candidate':
        payload['records']['p_test']['candidate']['channels']['vocabulary.layer.2.positive']['revise'] = 9
    else:
        payload['records']['p_test']['history'][0]['report']['rename_authority'] = True
    destination.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(ValueError):
        ProbeConceptionLedger(destination)


def test_measurement_ids_preserve_equal_live_rows_and_exclude_archive_or_missing_data():
    rows = [{'record_id': 'one', 'reading': 0.1, 'reply': 'same'},
            {'record_id': 'two', 'reading': 0.1, 'reply': 'same'},
            {'record_id': 'archive', 'basis': 'backfill', 'reading': 0.2, 'reply': 'different'},
            {'record_id': 'command', 'reply': ':probe define2'},
            {'record_id': 'bad', 'reading': float('nan'), 'reply': 'invalid'}]
    assert measurement_record_ids(rows) == ['record:one', 'record:two']
    assert measurement_record_ids([{'reading': 0.1, 'reply': 'legacy'}] * 2) == measurement_record_ids([{'reading': 0.1, 'reply': 'legacy'}])


def test_watch_opt_in_namespace_scoped_and_not_command_or_turn_count():
    watches = ProbeDiagnosticWatch()
    assert watches.due('p_test', ['a']) is False
    watches.register('p_test', every=2, record_ids=['old'])
    for _ in range(40):
        assert watches.due('p_test', ['old']) is False
    assert watches.due('p_test', ['old', 'new1']) is False
    assert watches.due('p_other', ['new1', 'new2']) is False
    assert watches.due('p_test', ['old', 'new1', 'new2']) is True
    assert watches.due('p_test', ['old', 'new1', 'new2']) is False
    assert watches.due('p_test', ['new1', 'new2', 'new3', 'new4', 'new5', 'new6']) is True
    assert watches.due('p_test', ['new1', 'new2', 'new3', 'new4', 'new5', 'new6']) is False
    assert watches.unregister('p_test') is True
    assert watches.due('p_test', ['new7', 'new8']) is False


def test_watch_saturated_window_counts_new_unique_measurements_not_buffer_length():
    watches = ProbeDiagnosticWatch()
    initial = [str(i) for i in range(64)]
    watches.register('p_test', every=3, record_ids=initial)
    assert watches.due('p_test', [str(i) for i in range(1, 65)]) is False
    assert watches.due('p_test', [str(i) for i in range(2, 66)]) is False
    assert watches.due('p_test', [str(i) for i in range(3, 67)]) is True
    assert watches.due('p_test', [str(i) for i in range(3, 67)]) is False


def test_watch_registration_is_ephemeral_and_returns_isolated_views():
    first = ProbeDiagnosticWatch()
    first.register('p_test', every=1)
    first.registrations['p_test']['seen'].append('fake')
    assert first.due('p_test', ['fake']) is True
    assert ProbeDiagnosticWatch().registrations == {}


@pytest.mark.parametrize('kwargs', [{'smoothing': 0}, {'smoothing': True}, {'smoothing': float('nan')},
                                    {'max_history': 0}, {'max_namespaces': 0}, {'max_terms': 0}])
def test_bad_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        ProbeConceptionLedger(**kwargs)


@pytest.mark.parametrize('every', [0, -1, 2.5, True])
def test_bad_watch_interval_rejected(every):
    with pytest.raises(ValueError):
        ProbeDiagnosticWatch().register('p_test', every=every)
