"""Matched replay tests use CPU hooks and synthetic directions, not semantic claims."""

from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from invariants import engine
from invariants import replay_investigation as replay
from invariants.memory_engine import MemoryRecord


class TinyConfig:
    _name_or_path = 'test/controlled-replay'
    hidden_size = 3
    num_hidden_layers = 2
    max_position_embeddings = 4096

    def to_dict(self):
        return {'_name_or_path': self._name_or_path, 'hidden_size': self.hidden_size,
                'num_hidden_layers': self.num_hidden_layers,
                'max_position_embeddings': self.max_position_embeddings}


class TinyTokenizer:
    name_or_path = 'test/controlled-replay-tokenizer'
    model_max_length = 4096

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
        text = ''.join(f"<{message['role']}>{message['content']}\n" for message in messages)
        return text + ('<assistant>' if add_generation_prompt else '')

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[ord(character) + 1 for character in text]], dtype=torch.long)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


class TinyStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mixing = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            # Increasing the helper coordinate causes a lower target coordinate
            # one block later. This is a known synthetic mechanism.
            mixing.weight.copy_(torch.tensor([[1., -1., 0.], [0., 1., 0.], [0., 0., 1.]]))
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), mixing])
        self.norm = torch.nn.Identity()


class TinyNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyStack()
        self.config = TinyConfig()
        self.is_loaded_in_4bit = False


class SyntheticGeneration:
    def __init__(self):
        self.calls = []
        self.fail_once_at = None
        self.drop_positive_target_for = None

    def __call__(self, wrapped, *, instruction, generation_probes, config, **kwargs):
        number = len(self.calls) + 1
        self.calls.append({
            'prompt': instruction, 'python_draw': random.random(), 'torch_draw': float(torch.rand(())),
            'numpy_draw': float(np.random.random()), 'kwargs': kwargs,
            'settings': {key: getattr(config, key) for key in (
                'base_model_only', 'activation_memory_enabled', 'cache_enabled',
                'cache_write_enabled', 'synthesis_enabled', 'repetition_penalty')},
        })
        if self.fail_once_at == number:
            # A failing generator can itself leave a hook; replay must clean it.
            wrapped.model.model.norm.register_forward_hook(lambda _module, _args, output: output)
            raise RuntimeError('synthetic generation interrupted')
        traces = {name: [] for name in generation_probes}
        positive_helper = False
        for step in range(3):
            state = torch.tensor([[[step / 10, 0., 1.]]])
            states = {}
            for layer_index, layer in enumerate(wrapped.model.model.layers):
                state = layer(state)
                states[layer_index] = state
            states[len(wrapped.model.model.layers) - 1] = wrapped.model.model.norm(state)
            positive_helper = positive_helper or bool(states[0][..., 1] > 0)
            for name, data in generation_probes.items():
                scores = [float(torch.nn.functional.cosine_similarity(
                    states[int(layer)][0, -1], vector, dim=0).detach())
                    for layer, vector in data['direction'].items()]
                traces[name].append(sum(scores) / len(scores))
        readings = {
            name: {'raw_score': values[-1], 'last_token_score': values[-1],
                   'mean_token_score': sum(values) / len(values), 'by_token': values,
                   'token_count': len(values), 'prediction_positions': [3, 4, 5]}
            for name, values in traces.items()
        }
        if (self.drop_positive_target_for and self.drop_positive_target_for in instruction
                and positive_helper):
            readings.pop('target', None)
            readings.pop('__replay_final__target', None)
        return f'fresh synthetic reply {number}', {
            'tokens_generated': 3, 'generation_probe_readings': readings,
            'generation_probe_capture': {'source': 'live_assistant_prediction',
                                         'generation_id': f'fresh-{number}', 'token_count': 3},
        }


@pytest.fixture
def synthetic(monkeypatch):
    model = SimpleNamespace(model=TinyNative().eval(), tok=TinyTokenizer(),
                            device='cpu', n_layers=2, d_model=3)
    probes = {
        'target': {'direction': {1: torch.tensor([1., 0., 0.])},
                   'framings': ('Synthetic target is present.', 'Synthetic target is absent.'),
                   'history': [.1, .2], 'measurement_source': 'live_assistant_prediction'},
        'helper': {'direction': {0: torch.tensor([0., 1., 0.])},
                   'framings': ('Synthetic helper is present.', 'Synthetic helper is absent.'),
                   'history': [.3, .4], 'measurement_source': 'live_assistant_prediction'},
    }
    generation = SyntheticGeneration()
    monkeypatch.setattr(replay, 'generate_agentic_text', generation)
    return model, probes, generation


def case(name='case-a', user='Original user question'):
    return {'case_id': name, 'source_record_id': name + '-assistant',
            'user_record_id': name + '-user', 'session_id': 'session-a', 'scope': 'scope-a',
            'prompt_messages': [{'role': 'user', 'content': user}],
            'archived_reply': 'ARCHIVED ANSWER MUST NOT BE A PROMPT',
            'context_quality': 'partial_reconstruction'}


def run(synthetic, output_dir, **overrides):
    model, probes, _generation = synthetic
    options = {'cases': [case()], 'output_dir': output_dir, 'layers': [0],
               'max_new_tokens': 3, 'random_control': False, 'progress': None}
    options.update(overrides)
    return replay.run_investigation(model, probes, {'target': -1}, ['helper'], **options)


def trial_rows(output):
    return [json.loads(line) for line in (output / 'trials.jsonl').read_text().splitlines() if line]


def archive_record(identifier, role, text, *, session='session-a', scope='scope-a'):
    return {'record_id': identifier, 'kind': 'turn', 'session_id': session, 'scope': scope,
            'role': role, 'text': text, 'timestamp': identifier}


@pytest.mark.parametrize('as_records', [False, True])
def test_archive_context_is_same_session_and_scope_without_future_reply_leakage(as_records):
    records = [
        archive_record('u1', 'user', 'OLD USER'),
        archive_record('a1', 'assistant', 'OLD ASSISTANT'),
        archive_record('foreign-u', 'user', 'FOREIGN SESSION', session='session-b'),
        archive_record('foreign-a', 'assistant', 'FOREIGN ANSWER', session='session-b'),
        archive_record('scope-u', 'user', 'FOREIGN SCOPE', scope='scope-b'),
        archive_record('scope-a', 'assistant', 'FOREIGN SCOPE ANSWER', scope='scope-b'),
        archive_record('u2', 'user', 'CURRENT USER'),
        archive_record('a2', 'assistant', 'FUTURE ARCHIVED ANSWER'),
    ]
    original = deepcopy(records)
    inputs = [MemoryRecord.from_dict(item) for item in records] if as_records else records
    selected = replay.select_archive_cases(inputs, limit=1, context_turns=8, scope='scope-a')
    assert len(selected) == 1
    chosen = selected[0]
    assert chosen['source_record_id'] == 'a2' and chosen['user_record_id'] == 'u2'
    assert chosen['prompt_messages'] == [
        {'role': 'user', 'content': 'OLD USER'},
        {'role': 'assistant', 'content': 'OLD ASSISTANT'},
        {'role': 'user', 'content': 'CURRENT USER'},
    ]
    assert chosen['archived_reply'] == 'FUTURE ARCHIVED ANSWER'
    assert chosen['context_quality'] == 'partial_reconstruction'
    assert records == original


def test_context_truncation_is_explicit_and_does_not_start_with_orphan_assistant():
    records = [archive_record('u1', 'user', 'FIRST'), archive_record('a1', 'assistant', 'FIRST ANSWER'),
               archive_record('u2', 'user', 'SECOND'), archive_record('a2', 'assistant', 'SECOND ANSWER')]
    chosen = replay.select_archive_cases(records, limit=1, context_turns=2)[0]
    assert chosen['context_truncated'] is True
    assert chosen['original_messages_omitted'] > 0
    assert chosen['prompt_messages'] == [{'role': 'user', 'content': 'SECOND'}]


def test_missing_session_identity_does_not_create_cross_session_cases():
    records = [archive_record('u1', 'user', 'UNSCOPED USER', session=''),
               archive_record('a1', 'assistant', 'UNSCOPED ANSWER', session='')]
    assert replay.select_archive_cases(records) == []


def test_signed_helper_effects_are_matched_fresh_generation_means(synthetic, tmp_path):
    summary = run(synthetic, tmp_path)
    _model, _probes, generation = synthetic
    rows = trial_rows(tmp_path)
    assert {item['arm'] for item in rows} == {'baseline', 'helper:+', 'helper:-'}
    assert len({item['prompt_sha256'] for item in rows}) == 1
    assert len({item['prompt'] for item in generation.calls}) == 1
    assert all('ARCHIVED ANSWER' not in item['prompt'] for item in generation.calls)
    assert all(item['source'] == replay.SOURCE for item in rows)
    assert all(item['reply'].startswith('fresh synthetic reply') for item in rows)
    effects = {item['sign']: item for item in summary['effects']}
    assert effects[1]['mean_delta'] < 0 < effects[-1]['mean_delta']
    assert effects[1]['desired_mean_delta'] > 0
    assert all(item['metric'] == 'final_layer_mean_token_score' and item['readout_layer'] == 1
               for item in effects.values())
    baseline = next(item for item in rows if item['sign'] == 0)
    baseline_mean = baseline['telemetry']['generation_probe_readings']['target']['mean_token_score']
    for item in rows:
        profile = item['telemetry']['generation_probe_readings']['target']
        assert len(profile['by_token']) == 3
        if item['sign']:
            assert effects[item['sign']]['mean_delta'] == pytest.approx(profile['mean_token_score'] - baseline_mean)
            assert effects[item['sign']]['n_cases'] == 1
    assert summary['source'] == 'controlled_replay_investigation'
    assert summary['completed_trials'] == summary['expected_trials'] == 3


def test_replay_retains_trajectories_and_uses_mean_rather_than_last_token_only(synthetic, tmp_path):
    summary = run(synthetic, tmp_path)
    rows = trial_rows(tmp_path)
    baseline = next(item for item in rows if item['sign'] == 0)['telemetry']['generation_probe_readings']['target']
    positive = next(item for item in rows if item['sign'] == 1)['telemetry']['generation_probe_readings']['target']
    effect = next(item for item in summary['effects'] if item['sign'] == 1)
    assert len(set(baseline['by_token'])) > 1
    assert effect['mean_delta'] == pytest.approx(positive['mean_token_score'] - baseline['mean_token_score'])
    assert effect['mean_delta'] != pytest.approx(positive['raw_score'] - baseline['raw_score'], abs=1e-7)


def test_incomplete_target_arm_is_excluded_from_its_paired_effect(synthetic, tmp_path):
    _model, _probes, generation = synthetic
    generation.drop_positive_target_for = 'DROP TARGET'
    summary = run(synthetic, tmp_path, cases=[case('complete'), case('incomplete', 'DROP TARGET')])
    effects = {item['sign']: item for item in summary['effects']}
    assert effects[1]['n_cases'] == 1 and effects[-1]['n_cases'] == 2
    assert [item['case_id'] for item in effects[1]['case_deltas']] == ['complete']


def test_random_direction_controls_are_reported_separately_from_helper_effects(synthetic, tmp_path):
    summary = run(synthetic, tmp_path, random_control=True)
    assert {item['helper'] for item in summary['effects']} == {'helper'}
    assert {item['helper'] for item in summary['negative_controls']} == {'__random_control__'}
    assert summary['expected_trials'] == 5
    vectors = torch.load(tmp_path / 'intervention_vectors.pt', weights_only=True)
    for arm in ('helper:+', 'helper:-', 'random_control:+', 'random_control:-'):
        assert vectors[arm][0].norm() == pytest.approx(1)
    torch.testing.assert_close(vectors['random_control:+'][0], vectors['random_control:-'][0])


def test_missing_final_readout_is_explicit_and_never_falls_back_to_a_steered_layer(synthetic, tmp_path):
    synthetic[1]['target']['direction'] = {0: torch.tensor([1., 0., 0.])}
    summary = run(synthetic, tmp_path)
    assert summary['effects'] == []
    assert summary['unavailable_final_readouts'] == ['target']
    assert summary['settings']['final_readout_available']['target'] is False


def test_final_readout_layer_cannot_be_selected_for_intervention(synthetic, tmp_path):
    with pytest.raises(ValueError, match='below the final'):
        run(synthetic, tmp_path, layers=[1])
    assert not synthetic[2].calls


def test_repeats_do_not_turn_one_archive_case_into_independent_evidence(synthetic, tmp_path):
    summary = run(synthetic, tmp_path, repeats=3)
    assert summary['completed_trials'] == 9
    for effect in summary['effects']:
        assert effect['n_cases'] == 1 and effect['n'] == 3
        assert effect['standard_error'] is None
        assert effect['case_deltas'][0]['n_repeats'] == 3


def test_generation_configuration_is_identical_isolated_and_rng_is_preserved(synthetic, tmp_path):
    _model, probes, generation = synthetic
    before_probes = deepcopy(probes)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone()
    telemetry_before = deepcopy(engine.steer_telemetry_stats())
    cap_before, band_before = engine.get_steer_cap_fraction(), engine.get_steer_band()
    run(synthetic, tmp_path)
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    assert engine.steer_telemetry_stats() == telemetry_before
    assert engine.get_steer_cap_fraction() == cap_before and engine.get_steer_band() == band_before
    assert len({item['python_draw'] for item in generation.calls}) == 1
    assert len({item['torch_draw'] for item in generation.calls}) == 1
    assert all(item['settings'] == generation.calls[0]['settings'] for item in generation.calls)
    assert generation.calls[0]['settings']['activation_memory_enabled'] is False
    assert generation.calls[0]['settings']['cache_enabled'] is False
    for name, data in probes.items():
        assert data['history'] == before_probes[name]['history']
        torch.testing.assert_close(data['direction'][next(iter(data['direction']))],
                                   before_probes[name]['direction'][next(iter(data['direction']))])


def test_completed_run_resumes_without_repeating_a_generation(synthetic, tmp_path):
    first = run(synthetic, tmp_path)
    count = len(synthetic[2].calls)
    trial_bytes = (tmp_path / 'trials.jsonl').read_bytes()
    second = run(synthetic, tmp_path)
    assert len(synthetic[2].calls) == count
    assert (tmp_path / 'trials.jsonl').read_bytes() == trial_bytes
    assert second == first


@pytest.mark.parametrize('change', ['prompt', 'alpha', 'probe', 'model'])
def test_resume_rejects_changed_trial_identity_without_appending_data(synthetic, tmp_path, change):
    run(synthetic, tmp_path)
    before = (tmp_path / 'trials.jsonl').read_bytes()
    options = {}
    if change == 'prompt':
        options['cases'] = [case(user='Changed archived user prompt')]
    elif change == 'alpha':
        options['alpha'] = .3
    elif change == 'probe':
        synthetic[1]['helper']['direction'][0][0] = .1
    else:
        with torch.no_grad():
            synthetic[0].model.model.layers[1].weight[0, 0] += .1
    with pytest.raises(ValueError, match='manifest|differs|directory'):
        run(synthetic, tmp_path, **options)
    assert (tmp_path / 'trials.jsonl').read_bytes() == before


def test_partial_trailing_json_is_rejected_without_erasing_valid_trials(synthetic, tmp_path):
    run(synthetic, tmp_path)
    path = tmp_path / 'trials.jsonl'
    with path.open('a', encoding='utf-8') as stream:
        stream.write('{"case_id":')
    damaged = path.read_bytes()
    count = len(synthetic[2].calls)
    with pytest.raises(ValueError, match='Incomplete trial JSONL'):
        run(synthetic, tmp_path)
    assert path.read_bytes() == damaged and len(synthetic[2].calls) == count


@pytest.mark.parametrize('change', [
    'case', 'arm', 'repeat', 'boolean_repeat', 'source', 'prompt',
    'helper', 'baseline_sign', 'boolean_sign', 'control', 'nonboolean_control',
    'inner_prompt', 'inner_source',
])
def test_resume_rejects_valid_json_with_unmatched_trial_metadata(synthetic, tmp_path, change):
    run(synthetic, tmp_path)
    path = tmp_path / 'trials.jsonl'
    rows = trial_rows(tmp_path)
    baseline = next(row for row in rows if row['arm'] == 'baseline')
    if change == 'case':
        baseline['case_id'] = 'unplanned-case'
    elif change == 'arm':
        baseline['arm'] = 'unplanned-arm'
    elif change == 'repeat':
        baseline['repeat'] = 99
    elif change == 'boolean_repeat':
        baseline['repeat'] = False
    elif change == 'source':
        baseline['source'] = 'reply_text_replay'
    elif change == 'prompt':
        baseline['prompt_sha256'] = 'different-prompt'
    elif change == 'helper':
        baseline['helper'] = 'helper'
    elif change == 'baseline_sign':
        baseline['sign'] = 1
    elif change == 'boolean_sign':
        baseline['sign'] = False
    elif change == 'control':
        baseline['is_control'] = True
    elif change == 'nonboolean_control':
        baseline['is_control'] = 0
    elif change == 'inner_prompt':
        baseline['telemetry']['generation_probe_capture']['prompt_sha256'] = 'different-prompt'
    else:
        baseline['telemetry']['generation_probe_capture']['source'] = 'reply_text_replay'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    damaged = path.read_bytes()
    calls = len(synthetic[2].calls)
    with pytest.raises(ValueError, match='Incompatible.*trial'):
        run(synthetic, tmp_path)
    assert path.read_bytes() == damaged
    assert len(synthetic[2].calls) == calls


def test_failed_generation_removes_hooks_preserves_rng_and_can_resume(synthetic, tmp_path):
    model, _probes, generation = synthetic
    generation.fail_once_at = 2
    python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
    telemetry_before = deepcopy(engine.steer_telemetry_stats())
    with pytest.raises(RuntimeError, match='synthetic generation interrupted'):
        run(synthetic, tmp_path)
    assert random.getstate() == python_state
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    assert engine.steer_telemetry_stats() == telemetry_before
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.model.modules())
    assert len(trial_rows(tmp_path)) == 1
    summary = run(synthetic, tmp_path)
    assert summary['completed_trials'] == 3
    assert len(trial_rows(tmp_path)) == 3
    assert len(generation.calls) == 4  # Three successful arms and one failed attempt.


def test_preexisting_hook_is_rejected_without_removing_it(synthetic, tmp_path):
    model, _probes, generation = synthetic
    handle = model.model.model.layers[0].register_forward_hook(lambda _module, _args, output: output)
    try:
        with pytest.raises(ValueError, match='active.*hooks|without active'):
            run(synthetic, tmp_path)
        assert not generation.calls
        assert model.model.model.layers[0]._forward_hooks
    finally:
        handle.remove()


def test_manifest_records_exact_token_prompt_and_partial_archive_provenance(synthetic, tmp_path):
    run(synthetic, tmp_path)
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    prepared = manifest['prepared_prompts']['case-a']
    token_hash = hashlib.sha256(json.dumps(prepared['token_ids'], sort_keys=True,
                                           separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    assert prepared['prompt_sha256'] == token_hash
    assert manifest['cases'][0]['context_quality'] == 'partial_reconstruction'
    assert manifest['model_identity']['model'] == 'test/controlled-replay'
    assert manifest['definition_fingerprints'].keys() == {'target', 'helper'}
    assert manifest['version'] == 2
    assert {'indirect_steering.py', 'other_probes.py', 'config.py', 'generation_budget.py'} <= manifest['source_sha256'].keys()
    assert manifest['arm_specs']['baseline'] == {'helper': None, 'sign': 0, 'is_control': False}
    assert (tmp_path / 'intervention_vectors.pt').exists()


def test_cli_parses_multiple_objectives_helpers_and_exact_case_selection(tmp_path):
    from scripts.investigate_indirect_backfill import parser
    arguments = parser().parse_args([
        '--minimize', 'target', 'second', '--maximize', 'third',
        '--via', 'helper', '--layers', '0', '1', '--case-id', 'record-a',
        '--case-id', 'record-b', '--no-random-control', '--output', str(tmp_path),
    ])
    assert arguments.minimize == ['target', 'second'] and arguments.maximize == ['third']
    assert arguments.via == ['helper'] and arguments.layers == [0, 1]
    assert arguments.case_id == ['record-a', 'record-b']
    assert arguments.no_random_control is True and arguments.output == tmp_path


def test_cli_repeated_via_flags_preserve_explicit_negative_helper_sign(tmp_path):
    from scripts.investigate_indirect_backfill import parser
    arguments = parser().parse_args([
        '--minimize', 'target', '--via=-helper', '--via=second_helper',
        '--output', str(tmp_path),
    ])
    assert arguments.via == ['-helper', 'second_helper']


def test_archive_loader_accepts_only_valid_json_after_leading_nul_padding(tmp_path, capsys):
    from scripts.investigate_indirect_backfill import load_records
    path = tmp_path / 'archive.jsonl'
    payload = {'kind': 'turn', 'role': 'user', 'text': 'unaltered content',
               'session_id': 'session-a', 'scope': 'scope-a'}
    original = b'\0\0' + json.dumps(payload).encode() + b'\n'
    path.write_bytes(original)
    audit = []
    records = load_records(path, audit=audit)
    assert len(records) == 1
    assert {key: records[0][key] for key in payload} == payload
    output = capsys.readouterr().out.lower()
    assert 'nul' in output and 'original file preserved' in output
    assert audit and 'nul' in audit[0].lower()
    assert path.read_bytes() == original


@pytest.mark.parametrize('malformed', [
    b'\0\0{"text":"valid JSON followed by junk"}garbage\n',
    b'{"text":"embedded\0NUL"}\n',
    b'garbage{"text":"not a JSON record"}\n',
])
def test_archive_loader_rejects_other_malformed_records_without_rewriting(tmp_path, malformed):
    from scripts.investigate_indirect_backfill import load_records
    path = tmp_path / 'archive.jsonl'
    path.write_bytes(malformed)
    with pytest.raises(ValueError, match='archive JSON|JSON'):
        load_records(path)
    assert path.read_bytes() == malformed


def test_cli_carries_archive_byte_hash_and_read_normalization_into_case_provenance(
        synthetic, tmp_path, monkeypatch):
    from scripts.investigate_indirect_backfill import main
    archive = tmp_path / 'archive.jsonl'
    user = archive_record('u1', 'user', 'SAVED USER')
    assistant = archive_record('a1', 'assistant', 'SAVED ANSWER')
    original = b'\0\0' + json.dumps(user).encode() + b'\n' + json.dumps(assistant).encode() + b'\n'
    archive.write_bytes(original)
    probe_dir = tmp_path / 'probes'
    probe_dir.mkdir()
    for name, payload in synthetic[1].items():
        torch.save(payload, probe_dir / (name + '.pt'))
    captured = {}

    def capture_run(model, probes, objectives, **kwargs):
        captured.update(kwargs)
        return {'source': replay.SOURCE, 'effects': [], 'completed_trials': 0}

    monkeypatch.setattr(engine, 'load_model', lambda *args, **kwargs: synthetic[0])
    monkeypatch.setattr(replay, 'run_investigation', capture_run)
    main(['--archive', str(archive), '--probe-dir', str(probe_dir),
          '--controller', str(tmp_path / 'owner.json'), '--minimize', 'target',
          '--via', 'helper', '--limit', '1', '--layers', '0', '0',
          '--output', str(tmp_path / 'investigation')])
    case_provenance = captured['cases'][0]['archive_input']
    assert case_provenance['path'] == str(archive.resolve())
    assert case_provenance['sha256'] == hashlib.sha256(original).hexdigest()
    assert any('nul' in note.lower() for note in case_provenance['read_notes'])
    assert archive.read_bytes() == original


def importable_summary(model, probes, case_ids):
    from invariants.indirect_steering import definition_fingerprints
    return {
        'source': replay.SOURCE, 'fingerprint': 'shell-replay-fixture',
        'model_identity': replay.model_identity(model),
        'definition_fingerprints': definition_fingerprints(probes),
        'settings': {'alpha': .25, 'layers': [0]}, 'case_ids': list(case_ids),
        'completed_trials': len(case_ids) * 3, 'expected_trials': len(case_ids) * 3,
        'effects': [{
            'target': 'target', 'helper': 'helper', 'sign': 1, 'n_cases': len(case_ids),
            'mean_delta': -.01, 'standard_error': 0 if len(case_ids) > 1 else None,
            'metric': 'final_layer_mean_token_score', 'readout_layer': 1,
            'case_deltas': [{'case_id': identifier, 'mean_delta': -.01, 'n_repeats': 1}
                            for identifier in case_ids],
        }],
    }


def shell_memory():
    records = [archive_record(f'u{index}', 'user', f'USER {index}') if role == 'user'
               else archive_record(f'a{index}', 'assistant', f'ANSWER {index}')
               for index in range(3) for role in ('user', 'assistant')]
    records.extend([archive_record('foreign-u', 'user', 'FOREIGN USER', scope='scope-b'),
                    archive_record('foreign-a', 'assistant', 'FOREIGN ANSWER', scope='scope-b')])
    return SimpleNamespace(scope='scope-a', records=records,
                           append_turn=Mock(side_effect=AssertionError('No conversation turn writes')),
                           append_event=Mock(side_effect=AssertionError('No archive writes')))


def test_shell_wrapper_passes_options_scope_and_imports_only_into_separate_evidence(
        synthetic, tmp_path, monkeypatch):
    from invariants.indirect_steering import IndirectSteering
    from scripts.interactive_phenomenality import handle_replay_investigation_command
    model, probes, generation = synthetic
    controller = IndirectSteering(tmp_path / 'owner.json')
    controller.configure('target', -1, ['+helper'])
    controller.set_role('helper', 'thinking')
    controller.observe({'probe_measurement': {'source': 'live_assistant_prediction', 'generation_id': 'live'},
                        'probe_target': .2, 'probe_helper': -.2}, probes)
    memory = shell_memory()
    before_rows, before_records = deepcopy(controller.observations), deepcopy(memory.records)
    before_probes = deepcopy(probes)
    captured = {}

    def capture_run(current_model, specifications, objectives, **kwargs):
        captured.update(model=current_model, specifications=specifications, objectives=objectives, **kwargs)
        return importable_summary(current_model, specifications, [item['case_id'] for item in kwargs['cases']])

    monkeypatch.setattr(replay, 'run_investigation', capture_run)
    output = tmp_path / 'run with spaces'
    result = handle_replay_investigation_command([
        'investigate', 'limit', '3', 'helpers', '1', 'tokens', '12', 'alpha', '.25',
        'cap', '.02', 'layers', '0', '0', 'seed', '17', 'context', '2',
        'resume', 'off', 'random', 'off', 'output', str(output),
    ], controller, probes, model, memory)
    assert captured['model'] is model and captured['objectives'] == {'target': -1}
    assert captured['controls'] == [{'name': 'helper', 'sign': 1}]
    assert captured['max_helpers'] == 1 and captured['max_new_tokens'] == 12
    assert captured['alpha'] == .25 and captured['cap_fraction'] == .02
    assert captured['layers'] == [0] and captured['seed'] == 17
    assert captured['resume'] is captured['random_control'] is False
    assert captured['output_dir'] == output
    assert len(captured['cases']) == 3
    assert all(item['scope'] == memory.scope for item in captured['cases'])
    assert captured['specifications']['helper']['role'] == 'thinking'
    assert result['import']['imported_effects'] == 1
    assert controller.observations == before_rows and memory.records == before_records
    assert controller.replay_evidence[0]['source'] == replay.SOURCE
    assert probes.keys() == before_probes.keys() and 'role' not in probes['helper']
    assert probes['target']['history'] == before_probes['target']['history']
    assert not generation.calls
    memory.append_turn.assert_not_called()
    memory.append_event.assert_not_called()


@pytest.mark.parametrize('automatic', [False, True])
def test_shell_wrapper_unions_helper_sign_constraints_and_ignores_disabled_routes(
        synthetic, tmp_path, monkeypatch, automatic):
    from scripts.interactive_phenomenality import handle_replay_investigation_command
    routes = {
        'target': {'enabled': True, 'direction': -1, 'controls': [
            {'name': 'helper', 'sign': 1}, {'name': 'other', 'sign': None}]},
        'second': {'enabled': True, 'direction': 1,
                   'controls': None if automatic else [{'name': 'helper', 'sign': -1}]},
        'disabled': {'enabled': False, 'direction': -1, 'controls': [{'name': 'hidden', 'sign': 1}]},
    }
    controller = SimpleNamespace(routes=routes, roles={}, import_investigation=Mock(return_value={'imported': True}))
    captured = {}

    def capture_run(model, probes, objectives, **kwargs):
        captured.update(objectives=objectives, **kwargs)
        return {'source': replay.SOURCE, 'completed_trials': 1, 'expected_trials': 1, 'effects': []}

    monkeypatch.setattr(replay, 'run_investigation', capture_run)
    handle_replay_investigation_command(['investigate', 'output', str(tmp_path)],
                                        controller, synthetic[1], synthetic[0], shell_memory())
    assert captured['objectives'] == {'target': -1, 'second': 1}
    if automatic:
        assert captured['controls'] is None
    else:
        assert captured['controls'] == [
            {'name': 'helper', 'sign': None}, {'name': 'other', 'sign': None}]
    assert 'disabled' not in captured['objectives']
    controller.import_investigation.assert_called_once()


@pytest.mark.parametrize('arguments', [
    ['investigate', 'unknown', '1'], ['investigate', 'tokens'],
    ['investigate', 'layers', '2', '1'], ['investigate', 'layers', '0'],
    ['investigate', 'random', 'true'], ['investigate', 'resume', 'maybe'],
    ['investigate', 'alpha', 'not-a-number'], ['import'], ['import', 'one', 'two'],
])
def test_shell_wrapper_rejects_bad_options_before_starting_an_experiment(
        synthetic, monkeypatch, arguments):
    from scripts.interactive_phenomenality import handle_replay_investigation_command
    generator = Mock(side_effect=AssertionError('Invalid options must not start an experiment'))
    monkeypatch.setattr(replay, 'run_investigation', generator)
    controller = SimpleNamespace(routes={'target': {'direction': -1, 'controls': None}}, roles={})
    with pytest.raises(ValueError):
        handle_replay_investigation_command(arguments, controller, synthetic[1], synthetic[0], shell_memory())
    generator.assert_not_called()


def test_shell_wrapper_requires_active_objectives(synthetic, monkeypatch):
    from scripts.interactive_phenomenality import handle_replay_investigation_command
    generator = Mock(side_effect=AssertionError('No configured objectives'))
    monkeypatch.setattr(replay, 'run_investigation', generator)
    controller = SimpleNamespace(routes={'target': {'enabled': False}}, roles={})
    with pytest.raises(ValueError, match='Configure indirect'):
        handle_replay_investigation_command(['investigate'], controller,
                                            synthetic[1], synthetic[0], shell_memory())
    generator.assert_not_called()


def test_shell_import_accepts_summary_file_and_directory_without_live_observation_writes(
        synthetic, tmp_path):
    from invariants.indirect_steering import IndirectSteering
    from scripts.interactive_phenomenality import handle_replay_investigation_command
    model, probes, generation = synthetic
    controller = IndirectSteering(tmp_path / 'owner.json')
    controller.observe({'probe_measurement': {'source': 'live_assistant_prediction', 'generation_id': 'live'},
                        'probe_target': .2, 'probe_helper': -.2}, probes)
    earlier = deepcopy(controller.observations)
    run_dir = tmp_path / 'saved run'
    run_dir.mkdir()
    summary_path = run_dir / 'summary.json'
    summary_path.write_text(json.dumps(importable_summary(model, probes, ['a', 'b', 'c'])))
    first = handle_replay_investigation_command(['import', str(run_dir)],
                                                controller, probes, model, shell_memory())
    second = handle_replay_investigation_command(['import', str(summary_path)],
                                                 controller, probes, model, shell_memory())
    assert first['imported_effects'] == 1
    assert second['imported_effects'] == 0 and second['duplicate_effects'] == 1
    assert controller.observations == earlier
    assert len(controller.replay_evidence) == 1 and not generation.calls


def test_imported_replay_layer_band_reaches_only_the_tested_field_hook(
        synthetic, tmp_path, monkeypatch):
    from invariants.indirect_steering import IndirectSteering
    from invariants.trigger_tuner import TriggerTuner
    from scripts import interactive_phenomenality as shell
    model, probes, _generation = synthetic
    bodies = {'loaded': True, 'anchors': {}, 'probe_poles': {}, 'laws': {},
              'fields': {}, 'g_families': {}, 'qualities': {}, 'sets': {},
              'bands': {}, 'spaces': {}, 'rotations': {}, 'law_gates': {},
              'trigger_bodies': {}, 'impulses': {}}
    monkeypatch.setattr(shell, '_STEER_BODIES', bodies)
    monkeypatch.setattr(shell, '_GRAVITY_STATE', deepcopy(shell._GRAVITY_STATE))
    monkeypatch.setattr(shell, 'save_steer_bodies', lambda: None)
    tuner = TriggerTuner(tmp_path / 'tuner.json')
    tuner.register('prioritize_alpha', .1, kind='coefficient')
    tuner.register('prioritize_gravity', 1, kind='coefficient')
    controller = tuner.indirect_controller = IndirectSteering()
    controller.configure('target', -1, ['+helper'])
    controller.observe({'probe_measurement': {'source': 'live_assistant_prediction', 'generation_id': 'live'},
                        'probe_target': .2, 'probe_helper': -.2}, probes)
    controller.import_investigation(importable_summary(model, probes, ['a', 'b', 'c']),
                                    probes, model_identity=replay.model_identity(model))
    assert controller.plan(probes)[0]['layers'] == [0]
    handles, description = shell.build_gravity_field_handles(model, probes, tuner)
    try:
        assert len(handles) == 1, description
        assert model.model.model.layers[0]._forward_hooks
        assert not model.model.model.layers[1]._forward_hooks
        state = torch.tensor([[[0., 0., 1.]]])
        for layer in model.model.model.layers:
            state = layer(state)
        assert state[0, 0, 1] > 0
    finally:
        for handle in handles:
            handle.remove()
