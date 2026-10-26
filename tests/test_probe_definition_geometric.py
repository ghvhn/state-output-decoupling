"""Meaningful mechanics and false-certification guards for geometric diagnostics."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from invariants import probe_definition_geometric as geometric


class Tokenizer:
    name_or_path = 'diagnostic-test-tokenizer'
    all_special_ids = [4]
    bos_token_id = 4
    eos_token_id = 4
    pad_token_id = None
    unk_token_id = None

    def __init__(self, tokens=None):
        self.tokens = tokens or [' revise', 'retain', 'update', 'refuse', '<control>', '']
        self.vocab_size = len(self.tokens)

    def decode(self, token_ids):
        return ''.join(self.tokens[index] for index in token_ids)

    def get_vocab(self):
        return {token: index for index, token in enumerate(self.tokens)}


class NativeModel:
    def __init__(self, weight=None):
        weight = weight if weight is not None else torch.tensor(
            [[3., 0.], [-3., 0.], [2., 1.], [-2., -1.], [5., 0.], [0., 0.]])
        self.head = SimpleNamespace(weight=weight)
        self.config = SimpleNamespace(_name_or_path='diagnostic-test-model', model_type='test',
                                      _commit_hash='revision-a')

    def get_output_embeddings(self):
        return self.head

    def forward(self, *args, **kwargs):
        raise AssertionError('Diagnostics must never execute a model forward')

    def generate(self, *args, **kwargs):
        raise AssertionError('Diagnostics must never generate')


def records():
    return [
        {'reading': -.4, 'reply': 'retain refuse', 'input': 'oldtopic question', 'source_ts': 't0'},
        {'reading': -.2, 'reply': 'retain refuse', 'input': 'oldtopic question', 'source_ts': 't1'},
        {'reading': .2, 'reply': 'revise update', 'input': 'newtopic question', 'source_ts': 't2'},
        {'reading': .4, 'reply': 'revise update', 'input': 'newtopic question', 'source_ts': 't3'},
    ]


def report(probe=None, rows=None, native=None, tokenizer=None, **kwargs):
    wrapper = SimpleNamespace(model=native or NativeModel(), tok=tokenizer or Tokenizer())
    return geometric.analyze_probe_geometry('denial', probe if probe is not None else {'direction': {22: torch.tensor([1., 0.])}},
        wrapper, records=records() if rows is None else rows, namespace='opaque:binding', **kwargs)


def test_streamed_projection_matches_dense_signed_ranking_and_tiny_vocab(monkeypatch):
    weight = torch.tensor([[1., 2.], [-4., 1.], [2., 3.], [-1., -2.]], dtype=torch.float16)
    tokens = Tokenizer(['first', 'negative', 'second', 'other'])
    chunks = []
    original = geometric._cpu_head_chunk

    def track(matrix, start, stop):
        chunks.append((start, stop))
        assert stop - start <= 2
        return original(matrix, start, stop)

    monkeypatch.setattr(geometric, '_cpu_head_chunk', track)
    result = geometric.vocab_fingerprint(torch.tensor([1., 0.]), weight, tokens,
                                        top_k=40, bottom_k=20, chunk_size=2)
    assert chunks == [(0, 2), (2, 4)]
    assert [(e['id'], e['score']) for e in result['positive']] == [(2, 2.), (0, 1.)]
    assert [(e['id'], e['score']) for e in result['negative']] == [(1, -4.), (3, -1.)]
    assert not result['semantic_validated']


def test_projection_preserves_raw_decoded_token_ids_and_marks_specials():
    result = geometric.vocab_fingerprint(torch.tensor([1., 0.]), NativeModel().head.weight, Tokenizer())
    assert result['positive'][0] == {'id': 4, 'token': '<control>', 'score': 5., 'special': True}
    assert result['positive'][1]['token'] == ' revise'
    assert {e['id'] for e in result['negative']} == {1, 3}


def test_rank_ties_are_deterministic_across_chunk_sizes():
    weight = torch.tensor([[1.], [1.], [1.], [-1.], [-1.]])
    tokens = Tokenizer(['a', 'b', 'c', 'd', 'e'])
    first = geometric.vocab_fingerprint(torch.ones(1), weight, tokens, 2, 1, chunk_size=1)
    second = geometric.vocab_fingerprint(torch.ones(1), weight, tokens, 2, 1, chunk_size=5)
    assert first == second
    assert [entry['id'] for entry in first['positive']] == [0, 1]
    assert first['negative'][0]['id'] == 3


@pytest.mark.parametrize('direction', [torch.zeros(2), torch.tensor([float('nan'), 1.]),
    torch.tensor([float('inf'), 1.]), torch.ones(3), torch.ones(1, 2), torch.ones(2, dtype=torch.int64)])
def test_invalid_directions_fail_before_projection(direction):
    with pytest.raises(ValueError):
        geometric.vocab_fingerprint(direction, NativeModel().head.weight, Tokenizer())


@pytest.mark.parametrize('weight', [torch.ones(2), torch.ones(2, 2, dtype=torch.int64),
    torch.empty(0, 2), torch.tensor([[float('nan'), 0.], [1., 0.]]),
    torch.tensor([[float('inf'), 0.], [1., 0.]])])
def test_invalid_heads_rejected(weight):
    with pytest.raises(ValueError):
        geometric.vocab_fingerprint(torch.ones(2), weight, Tokenizer())


@pytest.mark.parametrize('option,value', [('top_k', 0), ('top_k', True), ('top_k', 1.5),
    ('bottom_k', -1), ('chunk_size', 0)])
def test_invalid_projection_options_rejected(option, value):
    with pytest.raises(ValueError):
        geometric.vocab_fingerprint(torch.ones(2), NativeModel().head.weight, Tokenizer(), **{option: value})


def test_each_layer_has_independent_fingerprint_and_no_average_or_model_execution():
    probe = {'direction': {29: torch.tensor([-1., 0.]), 22: torch.tensor([1., 0.])}}
    original = {key: value.clone() for key, value in probe['direction'].items()}
    result = report(probe)
    assert list(result['vocabulary']['layers']) == ['22', '29']
    assert result['vocabulary']['layers']['22']['positive'][1]['id'] == 0
    assert result['vocabulary']['layers']['29']['negative'][1]['id'] == 0
    assert all(torch.equal(probe['direction'][key], value) for key, value in original.items())
    assert not result['semantic_validated'] and not result['rename_authority']
    json.dumps(result, allow_nan=False)


def test_native_output_accessor_preferred_over_inner_decoder():
    native = NativeModel()
    native.model = SimpleNamespace()
    result = geometric.analyze_probe_geometry('x', {3: torch.ones(2)}, native, Tokenizer(), records())
    assert result['model_identity']['name_or_path'] == 'diagnostic-test-model'


@pytest.mark.parametrize('layer', [True, -1, '22', 2.5])
def test_invalid_layer_identity_rejected(layer):
    with pytest.raises(ValueError, match='layer identifiers'):
        report({'direction': {layer: torch.ones(2)}})


def test_single_legacy_direction_has_unspecified_layer_not_fake_identity():
    result = report(torch.tensor([1., 0.]))
    assert list(result['vocabulary']['layers']) == ['unspecified']


def test_missing_output_head_and_tokenizer_fail_clearly():
    with pytest.raises(ValueError, match='output embedding'):
        geometric.analyze_probe_geometry('x', torch.ones(2), SimpleNamespace(), Tokenizer(), records())
    with pytest.raises(ValueError, match='tokenizer'):
        geometric.analyze_probe_geometry('x', torch.ones(2), NativeModel(), None, records())


def test_support_exact_original_ids_and_reply_input_separation():
    rows = records()
    rows.insert(0, {'reading': float('nan'), 'reply': 'contamination', 'input': 'contamination'})
    result = report(rows=rows)
    reply, inputs = result['behavior']['reply'], result['behavior']['input']
    assert reply['support']['low_ids'] == [1, 2]
    assert reply['support']['high_ids'] == [3, 4]
    assert reply['support']['skipped_records'] == 1
    assert {entry['term'] for entry in reply['high']} == {'revise', 'update'}
    assert {entry['term'] for entry in inputs['high']} == {'newtopic'}
    assert not ({'oldtopic', 'newtopic'} & {entry['term'] for entry in reply['high'] + reply['low']})


def test_document_frequency_ignores_repetition_and_smoothing_bounds_absence():
    rows = records()
    once = geometric.behavioral_contrast(rows)
    rows[2]['reply'] = 'revise ' * 1000 + 'update '
    repeated = geometric.behavioral_contrast(rows)
    assert once == repeated
    revise = next(entry for entry in repeated['high'] if entry['term'] == 'revise')
    assert revise['high_count'] == 2 and revise['low_count'] == 0
    assert 0 < revise['log_odds'] < 4


@pytest.mark.parametrize('rows,reason', [
    (records()[:3], 'at least'),
    ([dict(row, reading=0.) for row in records()], 'tied/constant'),
    ([dict(row, reading=value) for row, value in zip(records(), [0., 1., 1., 2.])], 'overlap'),
    ([dict(row, reply='same language') for row in records()], 'no distinguishing'),
    ([dict(row, reply='') for row in records()], 'at least'),
])
def test_insufficient_support_cannot_certify_alignment(rows, reason):
    result = report(rows=rows)
    assert result['status'] == 'insufficient'
    assert reason in result['behavior']['reply']['reason']
    assert result['alignment']['status'] == 'unknown'
    assert all(layer['status'] == 'unknown' for layer in result['alignment']['layers'].values())
    assert not result['semantic_validated']


def test_boundary_ties_are_included_as_complete_groups():
    rows = [{'reading': score, 'reply': 'retain' if score == 0 else 'revise'}
            for score in [0., 0., 0., 1., 2., 3., 4.]]
    result = geometric.behavioral_contrast(rows)
    assert result['support']['low_ids'] == [0, 1, 2]
    assert result['support']['high_ids'] == [5, 6]
    assert result['status'] == 'supported'


def test_boolean_and_nonfinite_readings_skipped_and_blank_reply_not_replaced_by_input():
    rows = records() + [{'reading': True, 'reply': 'bad', 'input': 'bad'},
                        {'reading': float('inf'), 'reply': 'bad', 'input': 'bad'},
                        {'reading': 5., 'reply': '', 'text': 'bad fallback', 'input': 'usable input'}]
    result = geometric.behavioral_contrast(rows)
    assert result['support']['records'] == 4
    assert result['support']['skipped_records'] == 3
    assert not any(entry['term'] == 'bad' for entry in result['high'])


def test_legacy_flat_rows_supported_but_explicitly_unseparated():
    rows = [{'probe_regret': row['reading'], 'text': row['reply']} for row in records()]
    result = geometric.behavioral_contrast(rows, 'probe_regret')
    assert result['status'] == 'supported'
    assert result['support']['legacy_unseparated_text_records'] == 4
    assert geometric.behavioral_contrast(rows, 'probe_regret', field='input')['status'] == 'insufficient'


def test_explicit_example_ids_and_generator_retained_for_both_channels():
    rows = [dict(row, example_id=f'memory-{index}') for index, row in enumerate(records())]
    result = report(rows=iter(rows))
    assert result['behavior']['reply']['support']['low_ids'] == ['memory-0', 'memory-1']
    assert result['behavior']['input']['support']['high_ids'] == ['memory-2', 'memory-3']
    pairs = [(index + 10, row) for index, row in enumerate(records())]
    assert geometric.behavioral_contrast(pairs)['support']['low_ids'] == [10, 11]


@pytest.mark.parametrize('option,value', [('threshold_pct', 0), ('threshold_pct', .6),
    ('threshold_pct', True), ('threshold_pct', float('nan')), ('min_records', 3),
    ('ngram_n', 0), ('field', 'text')])
def test_invalid_behavior_options_rejected(option, value):
    with pytest.raises(ValueError):
        geometric.behavioral_contrast(records(), **{option: value})


def test_alignment_compares_matching_lexical_granularity_and_both_polarities():
    fingerprint = {'positive': [{'token': ' confidence', 'score': 1.}],
                   'negative': [{'token': ' uncertainty', 'score': -1.}]}
    contrast = {'status': 'supported', 'high': [{'term': 'confidence grows', 'log_odds': 1.}],
                'low': [{'term': 'uncertainty remains', 'log_odds': -1.}]}
    result = geometric.channel_divergence(fingerprint, contrast)
    assert result['same_pole']['positive_high']['shared'] == ['confidence']
    assert result['same_pole']['negative_low']['shared'] == ['uncertainty']
    assert result['same_pole']['positive_high']['overlap_score'] == .5
    assert 'verdict' not in result
    reversed_contrast = dict(contrast, high=contrast['low'], low=contrast['high'])
    reversed_result = geometric.channel_divergence(fingerprint, reversed_contrast)
    assert reversed_result['opposite_pole']['positive_low']['shared'] == ['confidence']
    assert reversed_result['opposite_pole']['negative_high']['shared'] == ['uncertainty']


def test_special_tokens_do_not_become_semantic_lexical_matches():
    result = geometric.channel_divergence(
        {'positive': [{'token': '<control>', 'score': 1., 'special': True}], 'negative': []},
        {'status': 'supported', 'high': [{'term': 'control', 'log_odds': 1.}], 'low': []})
    assert result['status'] == 'unknown'
    assert result['same_pole']['positive_high']['shared'] == []


def test_model_and_tokenizer_identity_changes_are_visible_with_sample_limit():
    native, tokenizer = NativeModel(), Tokenizer()
    first = report(native=native, tokenizer=tokenizer)
    native.config._commit_hash = 'revision-b'
    native.head.weight[0, 0] += .5
    second = report(native=native, tokenizer=tokenizer)
    assert first['model_identity']['output_head_sample_sha256'] != second['model_identity']['output_head_sample_sha256']
    assert second['model_identity']['config_revision'] == 'revision-b'
    tokenizer.tokens[0] = 'different spelling'
    third = report(native=native, tokenizer=tokenizer)
    assert first['model_identity']['tokenizer_identity']['vocabulary_sha256'] != third['model_identity']['tokenizer_identity']['vocabulary_sha256']
    assert third['model_identity']['tokenizer_identity']['vocabulary_identity_complete']
    assert 'outside sampled rows' in third['model_identity']['output_head_identity_limit']


def test_special_registration_changes_comparison_identity_even_with_same_vocabulary():
    tokenizer = Tokenizer()
    first = report(tokenizer=tokenizer)
    tokenizer.all_special_ids = [4, 0]
    second = report(tokenizer=tokenizer)
    assert first['model_identity']['tokenizer_identity']['vocabulary_sha256'] == second['model_identity']['tokenizer_identity']['vocabulary_sha256']
    assert first['model_identity']['tokenizer_identity'] != second['model_identity']['tokenizer_identity']
    assert second['vocabulary']['layers']['22']['positive'][1]['special']


def _untrusted_payload_function():
    raise AssertionError('Unsafe probe payload executed')


class UntrustedPayload:
    def __reduce__(self):
        return (_untrusted_payload_function, ())


def test_probe_loading_is_restricted_and_corrupt_payloads_are_not_executed(tmp_path):
    unsafe = tmp_path / 'unsafe.pt'
    torch.save(UntrustedPayload(), unsafe)
    assert geometric._direction_from_probe_file(unsafe) is None
    corrupt = tmp_path / 'corrupt.pt'
    corrupt.write_bytes(b'not a tensor archive')
    assert geometric._direction_from_probe_file(corrupt) is None


def test_direction_digest_changes_by_direction_and_layer_identity_but_not_mapping_order():
    first = report({'direction': {22: torch.tensor([1., 0.]), 29: torch.tensor([-1., 0.])}})
    second = report({'direction': {29: torch.tensor([-1., 0.]), 22: torch.tensor([1., 0.])}})
    third = report({'direction': {23: torch.tensor([1., 0.]), 29: torch.tensor([-1., 0.])}})
    fourth = report({'direction': {22: torch.tensor([0., 1.]), 29: torch.tensor([-1., 0.])}})
    assert first['direction_sha256'] == second['direction_sha256']
    assert first['direction_sha256'] != third['direction_sha256'] != fourth['direction_sha256']


@pytest.mark.parametrize('name', ['regret', 'probe_regret', 'reality', 'probe_reality'])
def test_exact_prefix_removal_preserves_probe_name(name):
    clean = name.removeprefix('probe_')
    probes = {clean: {'direction': {22: torch.tensor([1., 0.])}}}
    result = geometric.handle_define2_command([name], probes, NativeModel(), Tokenizer(), records(), return_report=True)
    assert result['probe_name'] == clean


def test_current_saved_nested_layer_format_loads_without_average(tmp_path):
    path = tmp_path / 'denial.pt'
    torch.save({'direction': {22: torch.tensor([1., 0.]), 29: torch.tensor([-1., 0.])},
                'framings': ('operator seed must not enter report', 'other seed')}, path)
    loaded = geometric._direction_from_probe_file(path)
    assert set(loaded) == {22, 29}
    result = geometric.handle_define2_command(['denial'], {}, NativeModel(), Tokenizer(), records(),
        tmp_path, return_report=True)
    assert list(result['vocabulary']['layers']) == ['22', '29']
    assert 'operator seed' not in json.dumps(result)


def test_no_silent_d4096_fallback_or_path_traversal(tmp_path):
    torch.save({'direction': {22: torch.ones(2)}}, tmp_path / 'regret_d4096.pt')
    with pytest.raises(ValueError, match='No exact'):
        geometric.handle_define2_command(['regret'], {}, NativeModel(), Tokenizer(), records(), tmp_path)
    with pytest.raises(ValueError, match='path components'):
        geometric.handle_define2_command(['../regret_d4096'], {}, NativeModel(), Tokenizer(), records(), tmp_path)


def test_behavior_namespace_scoping_does_not_mutate_records_or_interpretations():
    from invariants.probe_behavior import ProbeBehavior, definition_key
    probe = {'direction': {22: torch.tensor([1., 0.])}, 'framings': ('seed A', 'seed B'),
             'other_binding': 'binding-a'}
    behavior = ProbeBehavior()
    namespace = definition_key('regret', probe)
    behavior.records[namespace] = records()
    behavior.records['irrelevant-scope'] = [dict(row, reply='wrongnamespace') for row in records()]
    before = copy.deepcopy(behavior.records)
    result = geometric.handle_define2_command(['probe_regret'], {'regret': probe}, NativeModel(), Tokenizer(),
        turn_records=[], behavior=behavior, return_report=True)
    assert result['namespace'] == namespace
    assert result['behavior']['reply']['status'] == 'supported'
    assert behavior.records == before and behavior.interpretations == {}
    assert 'seed A' not in json.dumps(result) and 'wrongnamespace' not in json.dumps(result)


def test_views_are_operator_diagnostics_without_authority_or_fake_watch():
    result = report()
    text = geometric.render_geometric_report(result)
    assert 'special]' in text and '-3.000' in text
    assert 'heuristic' in text and 'do not authorize' in text
    assert 'authoritative' not in text and 'story was wrong' not in text
    align = geometric.render_geometric_report(result, 'align')
    assert 'positive_low' in align
    with pytest.raises(ValueError, match='watch scheduling'):
        geometric.handle_define2_command(['denial', 'watch'], {'denial': {'direction': {22: torch.ones(2)}}},
                                        NativeModel(), Tokenizer(), records())
    with pytest.raises(ValueError):
        geometric.render_geometric_report(result, 'rename')
