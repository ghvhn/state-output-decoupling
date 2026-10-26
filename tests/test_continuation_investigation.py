"""Short branch selection uses real CPU hooks without semantic probe claims."""

from copy import deepcopy
import json
import random
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from invariants import engine
from invariants import continuation_investigation as continuation


class TinyConfig:
    _name_or_path = 'test/continuation-investigation'
    hidden_size = 3
    num_hidden_layers = 3
    max_position_embeddings = 4096
    eos_token_id = 90
    pad_token_id = 0

    def to_dict(self):
        return {key: getattr(self, key) for key in (
            '_name_or_path', 'hidden_size', 'num_hidden_layers',
            'max_position_embeddings', 'eos_token_id', 'pad_token_id')}


class TinyTokenizer:
    name_or_path = 'test/non-roundtrip-tokenizer'
    model_max_length = 4096
    eos_token_id = 90
    pad_token_id = 0

    def __init__(self):
        self.encoded = []

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True, **kwargs):
        text = ''.join(f"<{item['role']}>{item['content']}\n" for item in messages)
        return text + ('<assistant>' if add_generation_prompt else '')

    def __call__(self, text, **kwargs):
        self.encoded.append(text)
        # Decoded tokens intentionally cannot round-trip through this encoder.
        ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

    def decode(self, ids, *, skip_special_tokens=True, **kwargs):
        values = ids.tolist() if isinstance(ids, torch.Tensor) else list(ids)
        return ''.join(f'<{value}>' for value in values
                       if not skip_special_tokens or value not in (0, 90))


class TinyStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mixing = torch.nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            # Positive helper at block zero lowers coordinate zero downstream.
            mixing.weight.copy_(torch.tensor([[1., -1., 0.], [0., 1., 0.], [0., 1., 1.]]))
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), mixing, torch.nn.Identity()])
        self.norm = torch.nn.Identity()

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class TinyNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyStack()
        self.config = TinyConfig()
        self.generation_config = SimpleNamespace(eos_token_id=90, pad_token_id=0)
        self.is_loaded_in_4bit = False
        self.calls = []
        self.fail_at = None
        self.eos_at = None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, input_ids, **kwargs):
        length = input_ids.shape[-1]
        hidden = torch.tensor([.4, 0., 1.]).reshape(1, 1, 3).repeat(1, length, 1)
        state = self.model(hidden)
        return SimpleNamespace(logits=torch.zeros(1, length, 128), hidden=state)

    def generate(self, input_ids, *, max_new_tokens, logits_processor=None, **kwargs):
        self.calls.append({'input_ids': input_ids.detach().clone(),
                           'max_new_tokens': max_new_tokens, 'kwargs': deepcopy(kwargs),
                           'python_draw': random.random(), 'numpy_draw': float(np.random.random()),
                           'torch_draw': float(torch.rand(())), 'states': []})
        if self.fail_at == len(self.calls):
            self.model.norm.register_forward_hook(lambda _module, _args, output: output)
            raise RuntimeError('synthetic branch failure')
        current = input_ids.clone()
        for index in range(max_new_tokens):
            output = self(input_ids=current)
            self.calls[-1]['states'].append(output.hidden.detach().clone())
            scores = output.logits[:, -1]
            for processor in logits_processor or ():
                scores = processor(current, scores)
            helper = float(output.hidden[0, -1, 1])
            base = 40 if helper > 0 else 50 if helper < 0 else 30
            token = 90 if self.eos_at is not None and index == self.eos_at else base + index
            current = torch.cat((current, torch.tensor([[token]], dtype=torch.long)), dim=1)
            if token == 90:
                break
        return current


@pytest.fixture
def synthetic():
    model = SimpleNamespace(model=TinyNative().eval(), tok=TinyTokenizer(),
                            device=torch.device('cpu'), n_layers=3, d_model=3)
    probes = {
        'target': {'direction': {2: torch.tensor([1., 0., 0.])},
                   'framings': ('Synthetic target present.', 'Synthetic target absent.'),
                   'history': [.1, .2], 'measurement_source': 'live_assistant_prediction'},
        'helper': {'direction': {0: torch.tensor([0., 1., 0.])},
                   'framings': ('Synthetic helper present.', 'Synthetic helper absent.'),
                   'history': [.3, .4], 'measurement_source': 'live_assistant_prediction'},
    }
    return model, probes


def run(synthetic, output_dir, **overrides):
    model, probes = synthetic
    options = {'prompt_messages': [{'role': 'user', 'content': 'Finish this sentence'}],
               'output_dir': output_dir, 'max_new_tokens': 3, 'stride': 1, 'horizon': 3,
               'layers': [0], 'alpha': .25, 'cap_fraction': .02, 'progress': None}
    options.update(overrides)
    return continuation.run_continuation_investigation(
        model, probes, {'target': -1}, ['helper'], **options)


def rows(output_dir, name):
    return [json.loads(line) for line in (output_dir / name).read_text().splitlines() if line]


def test_branches_share_exact_prefix_ids_and_only_chosen_stride_becomes_next_prefix(synthetic, tmp_path):
    model, _probes = synthetic
    commits = []
    result = run(synthetic, tmp_path, on_commit=commits.append)
    calls = model.model.calls
    assert len(calls) == 9  # Three decisions, each baseline and both helper signs.
    for offset in range(0, 9, 3):
        assert all(torch.equal(call['input_ids'], calls[offset]['input_ids'])
                   for call in calls[offset:offset + 3])
    assert calls[0]['input_ids'].tolist() == [[7, 8, 9]]
    assert calls[3]['input_ids'].tolist() == [[7, 8, 9, 40]]
    assert calls[6]['input_ids'].tolist() == [[7, 8, 9, 40, 40]]
    assert len(model.tok.encoded) == 1
    assert all(41 not in call['input_ids'] and 42 not in call['input_ids']
               for call in calls)
    assert commits == ['<40>', '<40><40>', '<40><40><40>']
    assert result['response'] == commits[-1]
    assert result['telemetry']['tokens_generated'] == 3
    readings = result['telemetry']['generation_probe_readings']['target']
    assert readings['token_count'] == len(readings['by_token']) == 3


def test_baseline_wins_when_all_probe_effects_tie(synthetic, tmp_path):
    model, _probes = synthetic
    with torch.no_grad():
        # A helper can be measured at block zero while having no downstream effect.
        model.model.model.layers[1].weight.copy_(torch.diag(torch.tensor([1., 0., 1.])))
    result = run(synthetic, tmp_path, max_new_tokens=1)
    assert result['response'] == '<30>'
    assert result['telemetry']['tokens_generated'] == 1


def test_both_signed_helpers_are_separate_experimental_trials_and_do_not_change_live_histories(synthetic, tmp_path):
    _model, probes = synthetic
    original = deepcopy(probes)
    result = run(synthetic, tmp_path, max_new_tokens=1)
    trials = rows(tmp_path, 'trials.jsonl')
    assert {row['arm'] for row in trials} == {'baseline', 'helper:+', 'helper:-'}
    assert all(row['source'] == 'controlled_continuation_investigation' for row in trials)
    assert result['source'] == result['telemetry']['source'] == 'controlled_continuation_investigation'
    assert all(row['telemetry']['generation_probe_capture']['source'] == continuation.SOURCE for row in trials)
    assert all(row['telemetry']['generation_probe_capture']['capture_protocol_source'] == 'live_assistant_prediction'
               for row in trials)
    assert all(profile['measurement'] == continuation.SOURCE for row in trials
               for profile in row['telemetry']['generation_probe_readings'].values())
    for name, payload in probes.items():
        assert payload['history'] == original[name]['history']
        for layer, vector in payload['direction'].items():
            torch.testing.assert_close(vector, original[name]['direction'][layer])


def test_eos_is_committed_once_and_stops_future_branching(synthetic, tmp_path):
    model, _probes = synthetic
    model.model.eos_at = 0
    commits = []
    result = run(synthetic, tmp_path, max_new_tokens=10, stride=3, horizon=4,
                 on_commit=commits.append)
    assert len(model.model.calls) == 3
    assert result['telemetry']['tokens_generated'] == 1
    assert result['response'] == ''
    assert len(commits) <= 1


def test_eos_in_discarded_lookahead_does_not_stop_selected_output(synthetic, tmp_path):
    model, _probes = synthetic
    model.model.eos_at = 2
    result = run(synthetic, tmp_path, max_new_tokens=4, stride=1, horizon=3)
    assert result['selected_token_ids'] == [40, 40, 40, 40]
    assert result['telemetry']['stop_reason'] == 'max_new_tokens'
    assert len(model.model.calls) == 12


def test_output_token_limit_does_not_commit_lookahead_tail(synthetic, tmp_path):
    model, _probes = synthetic
    result = run(synthetic, tmp_path, max_new_tokens=5, stride=2, horizon=4)
    assert result['response'] == '<40><41><40><41><40>'
    assert result['telemetry']['tokens_generated'] == 5
    assert model.model.calls[-1]['max_new_tokens'] <= 4


def test_selected_probe_trajectory_excludes_discarded_continuations(synthetic, tmp_path):
    result = run(synthetic, tmp_path, max_new_tokens=2, stride=1, horizon=4)
    readings = result['telemetry']['generation_probe_readings']
    assert readings['target']['token_count'] == 2
    assert len(readings['target']['by_token']) == 2
    assert result['telemetry']['branch_tokens'] > result['telemetry']['tokens_generated']


def test_final_readout_cannot_be_directly_steered(synthetic, tmp_path):
    with pytest.raises(ValueError, match='below the final'):
        run(synthetic, tmp_path, layers=[2])
    assert not synthetic[0].model.calls


def test_injection_obeys_residual_cap_and_does_not_change_final_layer_hook_state(synthetic, tmp_path):
    model, _probes = synthetic
    before_norm = model.model.model.norm
    run(synthetic, tmp_path, max_new_tokens=1, alpha=1000, cap_fraction=.01)
    baseline = next(call['states'][0] for call in model.model.calls
                    if float(call['states'][0][0, -1, 1]) == 0)
    positive = next(call['states'][0] for call in model.model.calls
                    if float(call['states'][0][0, -1, 1]) > 0)
    # The helper coordinate passes unchanged through downstream blocks.
    assert float(positive[0, -1, 1]) <= .01 * torch.tensor([.4, 0., 1.]).norm().item() + 1e-7
    assert model.model.model.norm is before_norm
    assert not model.model.model.layers[2]._forward_hooks
    assert not model.model.model.norm._forward_hooks
    assert not torch.equal(baseline, positive)


def test_rng_operator_telemetry_and_temporary_hooks_are_restored(synthetic, tmp_path):
    model, _probes = synthetic
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    telemetry_before = deepcopy(engine.steer_telemetry_stats())
    cap_before, band_before = engine.get_steer_cap_fraction(), engine.get_steer_band()
    run(synthetic, tmp_path)
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    assert engine.steer_telemetry_stats() == telemetry_before
    assert engine.get_steer_cap_fraction() == cap_before
    assert engine.get_steer_band() == band_before
    assert not any(module._forward_hooks or module._forward_pre_hooks
                   for module in model.model.modules())


def test_generation_error_cleans_new_hooks_and_preserves_rng(synthetic, tmp_path):
    model, _probes = synthetic
    model.model.fail_at = 2
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    telemetry_before = deepcopy(engine.steer_telemetry_stats())
    with pytest.raises(RuntimeError, match='synthetic branch failure'):
        run(synthetic, tmp_path)
    assert random.getstate() == python_state
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state)
    assert engine.steer_telemetry_stats() == telemetry_before
    assert not any(module._forward_hooks or module._forward_pre_hooks
                   for module in model.model.modules())


def test_preexisting_hooks_are_rejected_and_preserved(synthetic, tmp_path):
    model, _probes = synthetic
    handle = model.model.model.layers[0].register_forward_hook(lambda _module, _args, output: output)
    try:
        with pytest.raises(ValueError, match='active.*hooks|without active'):
            run(synthetic, tmp_path)
        assert not model.model.calls
        assert model.model.model.layers[0]._forward_hooks
    finally:
        handle.remove()


def test_nonempty_output_is_preserved_and_refused(synthetic, tmp_path):
    existing = tmp_path / 'keep.txt'
    existing.write_text('Existing investigation data')
    with pytest.raises(ValueError, match='empty|nonempty|non-empty|exists'):
        run(synthetic, tmp_path)
    assert existing.read_text() == 'Existing investigation data'
    assert not synthetic[0].model.calls


def stub_branches(monkeypatch, means_by_arm):
    """Give the selector controlled trajectories independent of neural fixtures."""
    calls = []

    def generate(_model, _capture_probes, prefix_ids, arm, *, horizon, **kwargs):
        name = arm['arm'] if isinstance(arm, dict) and 'arm' in arm else arm.get('name') if isinstance(arm, dict) else arm
        calls.append((name, list(prefix_ids), horizon))
        means = means_by_arm[name]
        readings = {}
        for probe, value in means.items():
            for key in (probe, continuation.FINAL_PREFIX + probe):
                readings[key] = {'mean_token_score': value, 'raw_score': value,
                                 'last_token_score': value, 'by_token': [value] * horizon,
                                 'token_count': horizon,
                                 'prediction_positions': list(range(len(prefix_ids), len(prefix_ids) + horizon))}
        token = {'baseline': 30, 'helper:+': 40, 'helper:-': 50}[name]
        return {'token_ids': [token] * horizon, 'readings': readings,
                'capture': {'source': 'live_assistant_prediction', 'token_count': horizon}}

    monkeypatch.setattr(continuation, '_generate_branch', generate)
    return calls


def test_all_objectives_contribute_equally_to_selection(synthetic, tmp_path, monkeypatch):
    model, probes = synthetic
    probes['second'] = {'direction': {2: torch.tensor([0., 0., 1.])},
                        'framings': ('Second synthetic target present.', 'Second absent.')}
    stub_branches(monkeypatch, {
        'baseline': {'target': 0., 'second': 0.},
        'helper:+': {'target': -.4, 'second': -.1},
        'helper:-': {'target': -.2, 'second': .2},
    })
    result = continuation.run_continuation_investigation(
        model, probes, {'target': -1, 'second': 1}, ['helper'],
        prompt_messages=[{'role': 'user', 'content': 'Continue'}], output_dir=tmp_path,
        layers=[0], max_new_tokens=1, stride=1, horizon=1)
    assert result['response'] == '<50>'
    decision = result['telemetry']['decisions'][0]
    arms = {row['arm']: row for row in decision['arms']}
    assert arms['helper:+']['utility'] == pytest.approx(.15)
    assert arms['helper:-']['utility'] == pytest.approx(.2)
    assert arms['baseline']['utility'] == 0


@pytest.mark.parametrize('missing_from', ['baseline', 'helper:+'])
def test_missing_runtime_target_uses_explicit_baseline_fallback(synthetic, tmp_path, monkeypatch, missing_from):
    values = {'baseline': {'target': 0.}, 'helper:+': {'target': -.5},
              'helper:-': {'target': .5}}
    values[missing_from] = {}
    stub_branches(monkeypatch, values)
    result = run(synthetic, tmp_path, max_new_tokens=1)
    assert result['response'] == '<30>'
    decision = result['telemetry']['decisions'][0]
    assert decision['selected_arm'] == 'baseline'
    assert next(row for row in decision['arms'] if row['arm'] == 'helper:+')['eligible'] is False


def test_max_decisions_stops_without_an_untested_continuation(synthetic, tmp_path):
    result = run(synthetic, tmp_path, max_new_tokens=10, stride=2, horizon=4, max_decisions=1)
    assert result['telemetry']['tokens_generated'] == 2
    assert result['response'] == '<40><41>'
    assert result['telemetry']['stop_reason'] == 'max_decisions'
    assert len(synthetic[0].model.calls) == 3


def test_candidate_budget_requires_room_for_every_arm(synthetic, tmp_path):
    result = run(synthetic, tmp_path, max_new_tokens=10, candidate_token_budget=2)
    assert result['response'] == ''
    assert result['telemetry']['tokens_generated'] == 0
    assert result['telemetry']['branch_tokens'] == 0
    assert result['telemetry']['stop_reason'] == 'candidate_token_budget'
    assert not synthetic[0].model.calls


def test_candidate_budget_shrinks_all_arm_horizons_equally(synthetic, tmp_path):
    model, _probes = synthetic
    result = run(synthetic, tmp_path, max_new_tokens=10, horizon=4, candidate_token_budget=6)
    assert result['telemetry']['branch_tokens'] == 6
    assert result['telemetry']['tokens_generated'] == 1
    assert result['telemetry']['stop_reason'] == 'candidate_token_budget'
    assert [call['max_new_tokens'] for call in model.model.calls] == [2, 2, 2]


def test_missing_final_target_direction_refuses_before_generation(synthetic, tmp_path):
    synthetic[1]['target']['direction'] = {0: torch.tensor([1., 0., 0.])}
    with pytest.raises(ValueError, match='final.*readout|final.*direction|final.*layer'):
        run(synthetic, tmp_path)
    assert not synthetic[0].model.calls


def shell_memory():
    def record(identifier, role, text, **kwargs):
        return {'record_id': identifier, 'role': role, 'text': text, 'kind': 'turn',
                'session_id': 'session-a', 'scope': 'scope-a', **kwargs}

    return SimpleNamespace(
        session_id='session-a', scope='scope-a', records=[
            record('u1', 'user', 'PREVIOUS USER'),
            record('a1', 'assistant', 'PREVIOUS ANSWER'),
            record('foreign-session', 'user', 'FOREIGN SESSION', session_id='session-b'),
            record('foreign-scope', 'user', 'FOREIGN SCOPE', scope='scope-b'),
            record('event', 'assistant', 'OPERATOR EVENT', kind='event')],
        append_turn=Mock(), append_event=Mock())


def selected_result(output):
    return {'source': continuation.SOURCE, 'fingerprint': 'selected-fixture',
            'response': 'SELECTED ANSWER', 'output_dir': str(output),
            'telemetry': {'tokens_generated': 2, 'generation_probe_readings': {}}}


def test_shell_passes_current_session_context_and_saves_only_selected_turns(
        synthetic, tmp_path, monkeypatch):
    from scripts.interactive_phenomenality import handle_continuation_investigation_command
    model, probes = synthetic
    controller = SimpleNamespace(
        routes={'target': {'direction': -1, 'controls': [{'name': 'helper', 'sign': 1}]}},
        roles={'helper': 'thinking'}, observe=Mock(), import_investigation=Mock())
    memory = shell_memory()
    original_records = deepcopy(memory.records)
    captured = {}

    def capture_run(current_model, specifications, objectives, **kwargs):
        captured.update(model=current_model, specs=specifications, objectives=objectives, **kwargs)
        return selected_result(kwargs['output_dir'])

    monkeypatch.setattr(continuation, 'run_continuation_investigation', capture_run)
    output = tmp_path / 'run with spaces'
    result = handle_continuation_investigation_command([
        'continue', 'tokens', '9', 'stride', '2', 'horizon', '4', 'helpers', '1',
        'alpha', '.3', 'cap', '.01', 'layers', '0', '1', 'seed', '17',
        'context', '8', 'decisions', '5', 'candidate_tokens', '45',
        'output', str(output), '--', 'CURRENT', 'PROMPT'], controller, probes, model, memory)
    assert captured['model'] is model
    assert captured['objectives'] == {'target': -1}
    assert captured['controls'] == [{'name': 'helper', 'sign': 1}]
    assert captured['prompt_messages'] == [
        {'role': 'user', 'content': 'PREVIOUS USER'},
        {'role': 'assistant', 'content': 'PREVIOUS ANSWER'},
        {'role': 'user', 'content': 'CURRENT PROMPT'}]
    assert captured['output_dir'] == output
    assert captured['max_new_tokens'] == 9 and captured['stride'] == 2
    assert captured['horizon'] == 4 and captured['max_helpers'] == 1
    assert captured['alpha'] == .3 and captured['cap_fraction'] == .01
    assert captured['layers'] == [0, 1] and captured['seed'] == 17
    assert captured['max_decisions'] == 5 and captured['candidate_token_budget'] == 45
    assert captured['specs']['helper']['role'] == 'thinking'
    assert 'role' not in probes['helper']
    assert result['response'] == 'SELECTED ANSWER'
    writes = memory.append_turn.call_args_list
    assert [call.args for call in writes] == [('user', 'CURRENT PROMPT'), ('assistant', 'SELECTED ANSWER')]
    assert all(call.kwargs['provenance']['source'] == continuation.SOURCE for call in writes)
    assert all(call.kwargs['tags'] == ['indirect_continuation'] for call in writes)
    assert memory.records == original_records
    memory.append_event.assert_not_called()
    controller.observe.assert_not_called()
    controller.import_investigation.assert_not_called()
    assert not model.model.calls


@pytest.mark.parametrize('automatic', [False, True])
def test_shell_unions_helper_sign_constraints_without_disabled_objectives(
        synthetic, tmp_path, monkeypatch, automatic):
    from scripts.interactive_phenomenality import handle_continuation_investigation_command
    controller = SimpleNamespace(routes={
        'target': {'direction': -1, 'controls': [{'name': 'helper', 'sign': 1}]},
        'second': {'direction': 1, 'controls': None if automatic else [{'name': 'helper', 'sign': -1}]},
        'disabled': {'enabled': False, 'direction': -1, 'controls': []}}, roles={})
    captured = {}

    def capture_run(_model, _probes, objectives, **kwargs):
        captured.update(objectives=objectives, **kwargs)
        return selected_result(kwargs['output_dir'])

    monkeypatch.setattr(continuation, 'run_continuation_investigation', capture_run)
    handle_continuation_investigation_command([
        'continue', 'output', str(tmp_path), '--', 'PROMPT'], controller,
        synthetic[1], synthetic[0], shell_memory())
    assert captured['objectives'] == {'target': -1, 'second': 1}
    assert captured['controls'] is None if automatic else captured['controls'] == [{'name': 'helper', 'sign': None}]


@pytest.mark.parametrize('arguments', [
    ['continue'], ['continue', '--'], ['continue', 'tokens', '--', 'PROMPT'],
    ['continue', 'unknown', '1', '--', 'PROMPT'],
    ['continue', 'layers', '1', '0', '--', 'PROMPT'],
    ['continue', 'context', '0', '--', 'PROMPT'],
    ['continue', 'alpha', 'not-number', '--', 'PROMPT'],
])
def test_shell_bad_options_do_not_generate_or_write_turns(synthetic, monkeypatch, arguments):
    from scripts.interactive_phenomenality import handle_continuation_investigation_command
    generator = Mock(side_effect=AssertionError('Invalid request must not generate'))
    monkeypatch.setattr(continuation, 'run_continuation_investigation', generator)
    controller = SimpleNamespace(routes={'target': {'direction': -1, 'controls': None}}, roles={})
    memory = shell_memory()
    with pytest.raises(ValueError):
        handle_continuation_investigation_command(arguments, controller, synthetic[1], synthetic[0], memory)
    generator.assert_not_called()
    memory.append_turn.assert_not_called()


@pytest.mark.parametrize('routes', [
    {'target': {'enabled': False, 'direction': -1, 'controls': None}},
    {'target': {'direction': -1, 'controls': None, 'when': {'probe': 'helper', 'op': 'above', 'threshold': .2}}},
])
def test_shell_requires_active_unguarded_objectives(synthetic, monkeypatch, routes):
    from scripts.interactive_phenomenality import handle_continuation_investigation_command
    generator = Mock(side_effect=AssertionError('Unsupported routes must not generate'))
    monkeypatch.setattr(continuation, 'run_continuation_investigation', generator)
    memory = shell_memory()
    with pytest.raises(ValueError, match='Configure|unguarded'):
        handle_continuation_investigation_command(['continue', '--', 'PROMPT'],
            SimpleNamespace(routes=routes, roles={}), synthetic[1], synthetic[0], memory)
    generator.assert_not_called()
    memory.append_turn.assert_not_called()


def test_shell_generation_error_does_not_save_prompt_or_draft(synthetic, monkeypatch):
    from scripts.interactive_phenomenality import handle_continuation_investigation_command
    generator = Mock(side_effect=RuntimeError('BRANCH FAILURE'))
    monkeypatch.setattr(continuation, 'run_continuation_investigation', generator)
    memory = shell_memory()
    controller = SimpleNamespace(routes={'target': {'direction': -1, 'controls': None}}, roles={})
    with pytest.raises(RuntimeError, match='BRANCH FAILURE'):
        handle_continuation_investigation_command(['continue', '--', 'PROMPT'],
            controller, synthetic[1], synthetic[0], memory)
    memory.append_turn.assert_not_called()


def test_cli_parses_multiple_objectives_signed_helpers_and_branch_budget(tmp_path):
    from scripts.investigate_indirect_continuation import parser
    arguments = parser().parse_args([
        '--prompt', 'USER PROMPT', '--minimize', 'target', 'second', '--maximize', 'third',
        '--via=+helper', '--via=-other', '--tokens', '9', '--stride', '2', '--horizon', '4',
        '--decisions', '5', '--candidate-tokens', '45', '--layers', '0', '1', '--output', str(tmp_path)])
    assert arguments.prompt == 'USER PROMPT' and arguments.messages is None
    assert arguments.minimize == ['target', 'second'] and arguments.maximize == ['third']
    assert arguments.via == ['+helper', '-other']
    assert arguments.tokens == 9 and arguments.stride == 2 and arguments.horizon == 4
    assert arguments.decisions == 5 and arguments.candidate_tokens == 45
    assert arguments.layers == [0, 1] and arguments.output == tmp_path


@pytest.mark.parametrize('learnable', [False, True])
@pytest.mark.parametrize('failure', [False, True])
def test_known_synaptic_owner_hooks_are_withheld_without_bank_changes_and_restored(
        synthetic, monkeypatch, learnable, failure):
    from invariants.synaptic_memory import SynapticMemoryEngine
    from scripts.interactive_phenomenality import suspend_owned_investigation_memory_hooks
    model, _probes = synthetic
    layer = model.model.model.layers[0]
    hook_call = Mock(side_effect=lambda _module, _args, output: output)
    handle = layer.register_forward_hook(hook_call)
    owner = SynapticMemoryEngine(model=None, learnable=learnable)
    owner.model = model
    bank = torch.tensor([.1, .2, .3])
    owner.hooks = {0: SimpleNamespace(module=layer, hook_handle=handle, bank=bank)}
    owner.last_activation_stats = {'preserved': 'owner-state'}
    monkeypatch.setattr(SynapticMemoryEngine, '_INSTANCE', owner)
    original = list(layer._forward_hooks.items())
    try:
        with owner.activate():
            try:
                with suspend_owned_investigation_memory_hooks(model):
                    assert not layer._forward_hooks
                    model.model(input_ids=torch.tensor([[7, 8, 9]]))
                    if failure:
                        raise RuntimeError('inside investigation failed')
            except RuntimeError as error:
                assert failure and str(error) == 'inside investigation failed'
        assert list(layer._forward_hooks.items()) == original
        assert owner.hooks[0].hook_handle is handle
        torch.testing.assert_close(owner.hooks[0].bank, torch.tensor([.1, .2, .3]))
        assert owner.last_activation_stats == {'preserved': 'owner-state'}
        hook_call.assert_not_called()
    finally:
        handle.remove()


def test_shell_withholds_known_owner_but_unknown_hook_still_rejects_and_restores_order(
        synthetic, tmp_path, monkeypatch):
    from invariants.synaptic_memory import SynapticMemoryEngine
    from scripts.interactive_phenomenality import handle_continuation_investigation_command
    model, probes = synthetic
    layer = model.model.model.layers[0]
    unknown_call, owned_call = Mock(), Mock()
    unknown_handle = layer.register_forward_hook(unknown_call)
    owned_handle = layer.register_forward_hook(owned_call)
    owner = SynapticMemoryEngine(model=None)
    owner.model = model
    owner.hooks = {0: SimpleNamespace(module=layer, hook_handle=owned_handle)}
    monkeypatch.setattr(SynapticMemoryEngine, '_INSTANCE', owner)
    before = list(layer._forward_hooks.items())
    memory = shell_memory()
    controller = SimpleNamespace(routes={'target': {'direction': -1, 'controls': None}}, roles={})
    try:
        with owner.activate(), pytest.raises(ValueError, match='active.*hooks|without active'):
            handle_continuation_investigation_command([
                'continue', 'layers', '0', '0', 'output', str(tmp_path), '--', 'PROMPT'],
                controller, probes, model, memory)
        assert list(layer._forward_hooks.items()) == before
        unknown_call.assert_not_called()
        owned_call.assert_not_called()
        assert not model.model.calls
        memory.append_turn.assert_not_called()
    finally:
        unknown_handle.remove()
        owned_handle.remove()


@pytest.mark.parametrize('selection', ['inherited', 'explicit-objectives', 'explicit-via'])
def test_cli_inherits_route_helpers_only_with_inherited_objectives(
        synthetic, tmp_path, monkeypatch, selection):
    from invariants.indirect_steering import IndirectSteering
    from scripts import investigate_indirect_backfill as backfill_cli
    from scripts.investigate_indirect_continuation import main
    model, probes = synthetic
    controller_path = tmp_path / 'owner.json'
    owner = IndirectSteering(controller_path)
    owner.configure('target', -1, ['+helper'])
    monkeypatch.setattr(engine, 'load_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(backfill_cli, 'load_probe_bank', lambda *args, **kwargs: probes)
    captured = {}

    def capture_run(current_model, specifications, objectives, **kwargs):
        captured.update(model=current_model, specs=specifications, objectives=objectives, **kwargs)
        return selected_result(kwargs['output_dir'])

    monkeypatch.setattr(continuation, 'run_continuation_investigation', capture_run)
    arguments = ['--prompt', 'CURRENT PROMPT', '--controller', str(controller_path),
                 '--layers', '0', '0', '--output', str(tmp_path / 'run')]
    if selection == 'explicit-objectives':
        arguments.extend(['--minimize', 'target'])
    elif selection == 'explicit-via':
        arguments.append('--via=-helper')
    result = main(arguments)
    assert captured['model'] is model and captured['specs'] is probes
    assert captured['objectives'] == {'target': -1}
    expected = {'inherited': [{'name': 'helper', 'sign': 1}],
                'explicit-objectives': None, 'explicit-via': ['-helper']}[selection]
    assert captured['controls'] == expected
    assert captured['prompt_messages'] == [{'role': 'user', 'content': 'CURRENT PROMPT'}]
    assert result['response'] == 'SELECTED ANSWER'


def test_cli_passes_explicit_json_messages_and_budget_without_conversation_writes(
        synthetic, tmp_path, monkeypatch):
    from scripts import investigate_indirect_backfill as backfill_cli
    from scripts.investigate_indirect_continuation import main
    model, probes = synthetic
    messages = [{'role': 'system', 'content': 'SYSTEM TEXT'},
                {'role': 'user', 'content': 'PREVIOUS USER'},
                {'role': 'assistant', 'content': 'PREVIOUS ANSWER'},
                {'role': 'user', 'content': 'CURRENT USER'}]
    messages_path = tmp_path / 'messages.json'
    messages_path.write_text(json.dumps(messages), encoding='utf-8')
    monkeypatch.setattr(engine, 'load_model', lambda *args, **kwargs: model)
    monkeypatch.setattr(backfill_cli, 'load_probe_bank', lambda *args, **kwargs: probes)
    captured = {}

    def capture_run(_model, _probes, _objectives, **kwargs):
        captured.update(kwargs)
        return selected_result(kwargs['output_dir'])

    monkeypatch.setattr(continuation, 'run_continuation_investigation', capture_run)
    main(['--messages', str(messages_path), '--controller', str(tmp_path / 'missing-owner.json'),
          '--minimize', 'target', '--via', 'helper', '--tokens', '9', '--stride', '2',
          '--horizon', '4', '--decisions', '5', '--candidate-tokens', '45',
          '--output', str(tmp_path / 'run')])
    assert captured['prompt_messages'] == messages
    assert captured['max_new_tokens'] == 9 and captured['stride'] == 2 and captured['horizon'] == 4
    assert captured['max_decisions'] == 5 and captured['candidate_token_budget'] == 45
    assert messages_path.read_text(encoding='utf-8') == json.dumps(messages)
