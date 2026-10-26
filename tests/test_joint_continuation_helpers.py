"""Joint helper candidates use one bounded physical vector and keep provenance."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from invariants import continuation_investigation as continuation
from invariants.continuation_mode import ContinuationMode


def helpers(count=2, signs=None, layers=(0,)):
    signs = signs or [None] * count
    names = ['left', 'right', 'third', 'fourth', 'fifth'][:count]
    return [{'name': name, 'sign': sign,
             'vectors': {layer: torch.eye(count + 1)[index] for layer in layers}}
            for index, (name, sign) in enumerate(zip(names, signs))]


def components(arm):
    return tuple((part['helper'], part['sign'], part['coefficient'])
                 for part in arm['components'])


def test_two_unsigned_helpers_include_all_singles_and_simultaneous_pairs():
    arms, plan = continuation._build_candidate_arms(helpers(), [0], mixture_resolution=1)
    assert len(arms) == plan['selected'] == plan['total_possible'] == 9
    assert not plan['truncated']
    assert plan['required_single_candidates'] == 5
    assert arms[0]['arm'] == 'baseline' and not arms[0]['components']
    assert [components(arm) for arm in arms[1:5]] == [
        (('left', 1, 1.),), (('left', -1, -1.),),
        (('right', 1, 1.),), (('right', -1, -1.),)]
    assert {components(arm) for arm in arms[5:]} == {
        (('left', left, left / 2), ('right', right, right / 2))
        for left in (1, -1) for right in (1, -1)}


def test_signed_helpers_constrain_singles_and_every_joint_component():
    arms, plan = continuation._build_candidate_arms(helpers(signs=[1, -1]), [0], mixture_resolution=1)
    assert plan['selected'] == plan['total_possible'] == 4
    assert components(arms[-1]) == (('left', 1, .5), ('right', -1, -.5))
    assert all(part['sign'] == {'left': 1, 'right': -1}[part['helper']]
               for arm in arms for part in arm['components'])


def test_joint_one_retains_individual_axis_comparisons():
    arms, plan = continuation._build_candidate_arms(helpers(), [0], max_joint_helpers=1)
    assert len(arms) == plan['total_possible'] == 5
    assert all(arm['joint_size'] <= 1 for arm in arms)


def test_candidate_cap_truncates_joints_without_removing_any_single():
    arms, plan = continuation._build_candidate_arms(
        helpers(3), [0], max_joint_helpers=2, max_candidates=9, mixture_resolution=1)
    assert plan['total_possible'] == 19
    assert plan['selected'] == len(arms) == 9 and plan['truncated']
    assert plan['required_single_candidates'] == 7
    assert [arm['joint_size'] for arm in arms] == [0, 1, 1, 1, 1, 1, 1, 2, 2]


def test_triples_are_opt_in_and_use_shared_l1_injection_budget():
    arms, plan = continuation._build_candidate_arms(
        helpers(3, layers=(0, 1)), [0, 1], max_joint_helpers=3, mixture_resolution=1)
    assert plan['total_possible'] == plan['selected'] == 27
    triples = [arm for arm in arms if arm['joint_size'] == 3]
    assert len(triples) == 8
    for arm in arms[1:]:
        assert sum(abs(part['coefficient']) for part in arm['components']) == pytest.approx(1.)
        for vector in arm['vectors'].values():
            assert float(vector.norm()) <= 1. + 1e-6
    for arm in triples:
        expected = torch.tensor([part['coefficient'] for part in arm['components']] + [0.])
        for vector in arm['vectors'].values():
            torch.testing.assert_close(vector, expected)


def test_joint_order_is_deterministic_and_preserves_declared_helper_order():
    declared = list(reversed(helpers(3)))
    first, first_plan = continuation._build_candidate_arms(declared, [0], max_candidates=9, mixture_resolution=1)
    second, second_plan = continuation._build_candidate_arms(deepcopy(declared), [0], max_candidates=9, mixture_resolution=1)
    assert first_plan == second_plan
    assert [components(arm) for arm in first] == [components(arm) for arm in second]
    assert [part['helper'] for part in first[-1]['components']] == ['third', 'right']
    assert first[1]['components'][0]['helper'] == 'third'


class JointConfig:
    _name_or_path = 'test/nonlinear-joint-continuation'
    hidden_size = 4
    num_hidden_layers = 3
    max_position_embeddings = 4096
    eos_token_id = 90
    pad_token_id = 0

    def to_dict(self):
        return dict(model=self._name_or_path, hidden_size=self.hidden_size,
                    num_hidden_layers=self.num_hidden_layers,
                    max_position_embeddings=self.max_position_embeddings,
                    eos_token_id=self.eos_token_id, pad_token_id=self.pad_token_id)


class JointTokenizer:
    name_or_path = 'test/joint-tokenizer'
    model_max_length = 4096
    eos_token_id = 90
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return '\n'.join(item['content'] for item in messages) + '\nAssistant:'

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

    def decode(self, ids, **kwargs):
        return ''.join(f'<{item}>' for item in ids if item not in (0, 90))


class ProductLayer(torch.nn.Module):
    def forward(self, hidden):
        output = hidden.clone()
        # Neither single axis changes the target. Applying both does.
        output[..., 2] = hidden[..., 0] * hidden[..., 1]
        return output


class JointStack(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Identity(), ProductLayer(), torch.nn.Identity()])
        self.norm = torch.nn.Identity()

    def forward(self, hidden):
        for layer in self.layers:
            hidden = layer(hidden)
        return self.norm(hidden)


class JointNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = JointStack()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = JointConfig()
        self.generation_config = SimpleNamespace(eos_token_id=90, pad_token_id=0)
        self.is_loaded_in_4bit = False
        self.calls = []

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, input_ids, **kwargs):
        hidden = torch.tensor([0., 0., 0., 1.]).reshape(1, 1, 4)
        hidden = hidden.repeat(1, input_ids.shape[-1], 1)
        state = self.model(hidden)
        return SimpleNamespace(hidden=state, logits=torch.zeros(1, input_ids.shape[-1], 128))

    def generate(self, input_ids, *, max_new_tokens, logits_processor=None, **kwargs):
        current = input_ids.clone()
        states = []
        self.calls.append({'input_ids': current.clone(), 'states': states})
        for _ in range(max_new_tokens):
            output = self(input_ids=current)
            states.append(output.hidden.detach().clone())
            scores = output.logits[:, -1]
            for processor in logits_processor or ():
                scores = processor(current, scores)
            token = 60 if float(output.hidden[0, -1, 2]) > 0 else 30
            current = torch.cat([current, torch.tensor([[token]])], dim=1)
        return current


@pytest.fixture
def nonlinear():
    model = SimpleNamespace(model=JointNative().eval(), tok=JointTokenizer(),
                            device=torch.device('cpu'), n_layers=3, d_model=4)
    probes = {
        'target': {'direction': {2: torch.tensor([0., 0., 1., 0.])},
                   'framings': ('Target present', 'Target absent'), 'history': [.1, .2],
                   'measurement_source': 'live_assistant_prediction'},
        'left': {'direction': {0: torch.tensor([1., 0., 0., 0.])},
                 'framings': ('Left present', 'Left absent'), 'history': [.3],
                 'measurement_source': 'live_assistant_prediction'},
        'right': {'direction': {0: torch.tensor([0., 1., 0., 0.])},
                  'framings': ('Right present', 'Right absent'), 'history': [.4],
                  'measurement_source': 'live_assistant_prediction'},
    }
    return model, probes


def run(nonlinear, output_dir, controls=None, **overrides):
    model, probes = nonlinear
    options = dict(prompt_messages=[{'role': 'user', 'content': 'Finish this'}],
                   output_dir=output_dir, max_new_tokens=1, stride=1, horizon=1,
                   alpha=.25, cap_fraction=.02, layers=[0], max_helpers=2,
                   progress=None, mixture_resolution=1)
    options.update(overrides)
    return continuation.run_continuation_investigation(
        model, probes, {'target': 1}, ['left', 'right'] if controls is None else controls,
        **options)


def read_rows(directory, name):
    return [json.loads(line) for line in (directory / name).read_text().splitlines() if line]


def test_real_nonlinear_joint_wins_with_one_combined_physical_hook_and_no_live_pollution(
        nonlinear, tmp_path, monkeypatch):
    model, probes = nonlinear
    before = deepcopy(probes)
    original = continuation._steer_handles
    physical = []

    def record(_model, vectors, layers, alpha, **kwargs):
        physical.append({'vectors': deepcopy(vectors), 'layers': layers,
                         'alpha': alpha, **kwargs})
        return original(_model, vectors, layers, alpha, **kwargs)

    monkeypatch.setattr(continuation, '_steer_handles', record)
    result = run(nonlinear, tmp_path)
    trials = read_rows(tmp_path, 'trials.jsonl')
    decision = read_rows(tmp_path, 'decisions.jsonl')[0]
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert len(trials) == len(model.model.calls) == 9
    assert len(physical) == 8  # One hook envelope per nonbaseline arm.
    assert result['response'] == '<60>'
    assert decision['selected_joint_size'] == 2
    assert decision['selected_components'] == [
        {'helper': 'left', 'sign': 1, 'coefficient': .5},
        {'helper': 'right', 'sign': 1, 'coefficient': .5}]
    assert decision['selected_utility'] > 0
    assert all(arm['utility'] == 0 for arm in decision['arms'][:5])
    torch.testing.assert_close(physical[4]['vectors'][0], torch.tensor([.5, .5, 0., 0.]))
    assert physical[4]['alpha'] == .25 and physical[4]['cap_fraction'] == .02
    assert float(model.model.calls[5]['states'][0][0, -1, 0]) > 0
    assert float(model.model.calls[5]['states'][0][0, -1, 1]) > 0
    # The common physical cap is applied to the combined residual, once.
    state = model.model.calls[5]['states'][0][0, -1]
    assert float(state[:2].norm()) == pytest.approx(.02, rel=1e-5)
    assert all(torch.equal(call['input_ids'], model.model.calls[0]['input_ids'])
               for call in model.model.calls)
    assert manifest['settings']['candidate_plan']['total_possible'] == 9
    assert manifest['settings']['candidate_token_budget'] == 9
    selected_spec = next(arm for arm in manifest['arm_specs']
                         if arm['arm'] == decision['selected_arm'])
    assert selected_spec['components'] == decision['selected_components']
    assert all(row['source'] == continuation.SOURCE for row in trials)
    selected = result['telemetry']['generation_probe_readings']['target']
    assert selected['measurement'] == continuation.SOURCE
    assert selected['token_count'] == len(selected['by_token']) == 1
    assert selected['mean_token_score'] > 0
    for name in probes:
        assert probes[name]['history'] == before[name]['history']
        for layer in probes[name]['direction']:
            torch.testing.assert_close(probes[name]['direction'][layer], before[name]['direction'][layer])


def test_signed_via_is_respected_in_end_to_end_trial_components(nonlinear, tmp_path):
    result = run(nonlinear, tmp_path, controls=['+left', '-right'])
    rows = read_rows(tmp_path, 'trials.jsonl')
    assert len(rows) == 4
    assert all(part['sign'] == {'left': 1, 'right': -1}[part['helper']]
               for row in rows for part in row['components'])
    assert result['response'] == '<30>'
    assert result['telemetry']['decisions'][0]['selected_arm'] == 'baseline'


def test_cap_below_all_singles_fails_before_generation(nonlinear, tmp_path):
    model, _ = nonlinear
    with pytest.raises(ValueError, match='single|candidate|baseline'):
        run(nonlinear, tmp_path, max_candidates=4)
    assert not model.model.calls
    assert not list(tmp_path.iterdir())


def test_auto_pairs_exclude_target_proxy_and_released_helpers_but_keep_readouts(nonlinear, tmp_path):
    model, probes = nonlinear
    probes['a_proxy'] = deepcopy(probes['target'])
    probes['a_released'] = {
        'direction': {0: torch.tensor([0., 0., 0., 1.])}, 'released': True,
        'framings': ('Released present', 'Released absent'),
        'measurement_source': 'live_assistant_prediction'}
    result = continuation.run_continuation_investigation(
        model, probes, {'target': 1}, controls=None,
        prompt_messages=[{'role': 'user', 'content': 'Finish this'}],
        output_dir=tmp_path, max_new_tokens=1, stride=1, horizon=1,
        layers=[0], max_helpers=5, progress=None, mixture_resolution=1)
    rows = read_rows(tmp_path, 'trials.jsonl')
    assert len(rows) == 9
    assert {part['helper'] for row in rows for part in row['components']} == {'left', 'right'}
    assert {'a_proxy', 'a_released'} <= result['telemetry']['generation_probe_readings'].keys()


def test_joint_sum_cannot_reconstruct_target_axis_or_its_inverse(nonlinear, tmp_path, monkeypatch):
    model, probes = nonlinear
    target_axis = torch.tensor([1., 1., 0., 0.]) / 2 ** .5
    # The two individually independent helpers sum to the intervention target.
    probes['target']['direction'][0] = target_axis
    physical = []
    original = continuation._steer_handles

    def record(_model, vectors, layers, alpha, **kwargs):
        physical.append(deepcopy(vectors))
        return original(_model, vectors, layers, alpha, **kwargs)

    monkeypatch.setattr(continuation, '_steer_handles', record)
    result = run(nonlinear, tmp_path)
    rows = read_rows(tmp_path, 'trials.jsonl')
    assert len(rows) == len(model.model.calls) == 7
    assert len(physical) == 6
    joints = [row for row in rows if row['joint_size'] == 2]
    assert len(joints) == 2
    assert all(row['components'][0]['sign'] != row['components'][1]['sign']
               for row in joints)
    for vectors in physical:
        cosine = torch.dot(vectors[0], target_axis) / vectors[0].norm()
        assert float(abs(cosine)) < .99
    assert result['telemetry']['decisions'][0]['selected_arm'] == 'baseline'
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    plan = manifest['settings']['candidate_plan']
    assert plan['total_possible'] == plan['constructed_candidates'] == 9
    assert plan['selected'] == plan['effective_candidates'] == 7
    assert plan['blocked_joint_count'] == 2
    assert plan['unexamined_candidates'] == 0 and not plan['truncated']
    assert all(item['objectives'] == ['target'] for item in plan['blocked_joint_arms'])
    blocked = {item['arm'] for item in plan['blocked_joint_arms']}
    assert not blocked.intersection(row['arm'] for row in rows)


def test_candidate_budget_stops_without_skipping_joint_comparisons(nonlinear, tmp_path):
    model, _ = nonlinear
    result = run(nonlinear, tmp_path, candidate_token_budget=8)
    assert not model.model.calls
    assert result['telemetry']['stop_reason'] == 'candidate_token_budget'
    assert result['telemetry']['branch_tokens'] == 0
    assert result['selected_token_ids'] == []


@pytest.mark.parametrize(('key', 'value'), [
    ('max_joint_helpers', 0), ('max_joint_helpers', True),
    ('max_joint_helpers', 1.5), ('max_candidates', 0),
    ('max_candidates', False), ('max_candidates', 3.5),
])
def test_invalid_joint_bounds_fail_before_generation(nonlinear, tmp_path, key, value):
    model, _ = nonlinear
    with pytest.raises(ValueError):
        run(nonlinear, tmp_path, **{key: value})
    assert not model.model.calls


def test_mode_keeps_joint_controls_across_turn_arguments_and_on_off():
    mode = ContinuationMode()
    mode.configure(['on', 'joint', '3', 'candidates', '27'])
    assert mode.options['joint'] == 3 and mode.options['candidates'] == 27
    for prompt in ['First prompt', 'Second prompt']:
        arguments = mode.arguments(prompt)
        assert arguments[arguments.index('joint') + 1] == '3'
        assert arguments[arguments.index('candidates') + 1] == '27'
    mode.configure(['off'])
    mode.configure(['on', 'tokens', '128'])
    assert mode.options['joint'] == 3 and mode.options['candidates'] == 27


@pytest.mark.parametrize('arguments', [
    ['on', 'joint', '0'], ['on', 'joint', '1.2'],
    ['on', 'candidates', '0'], ['on', 'candidates', '-1'],
])
def test_invalid_joint_mode_settings_are_atomic(arguments):
    mode = ContinuationMode()
    before = mode.status()
    with pytest.raises(ValueError):
        mode.configure(arguments)
    assert mode.status() == before


def test_shell_maps_joint_controls_to_runner_and_mode_turns(nonlinear, tmp_path, monkeypatch):
    from unittest.mock import Mock
    from scripts import interactive_phenomenality as shell

    model, probes = nonlinear
    controller = SimpleNamespace(routes={'target': {'direction': 1, 'controls': None}}, roles={})
    memory = SimpleNamespace(records=[], scope='joint-test', session_id='joint-session',
                             append_turn=Mock())
    calls = []

    def fake_run(_model, _probes, objectives, **options):
        calls.append(options)
        return {'source': continuation.SOURCE, 'fingerprint': 'joint-test',
                'response': 'SELECTED', 'output_dir': str(options['output_dir']),
                'telemetry': {'tokens_generated': 1, 'generation_probe_readings': {}}}

    monkeypatch.setattr(continuation, 'run_continuation_investigation', fake_run)
    shell.handle_continuation_investigation_command(
        ['continue', 'joint', '3', 'candidates', '27', 'output', str(tmp_path / 'one'),
         '--', 'One prompt'], controller, probes, model, memory)
    assert calls[0]['max_joint_helpers'] == 3 and calls[0]['max_candidates'] == 27
    shell.handle_continuation_mode_command(
        ['on', 'joint', '1', 'candidates', '9', 'output', str(tmp_path / 'turns')],
        controller, probes, model)
    for prompt in ['First turn', 'Second turn']:
        result = shell.run_enabled_continuation_turn(prompt, controller, probes, model, memory)
        assert result['response'] == 'SELECTED'
    assert all(call['max_joint_helpers'] == 1 and call['max_candidates'] == 9
               for call in calls[1:])


def test_cli_exposes_joint_controls_with_default_pairs():
    from scripts.investigate_indirect_continuation import parser

    default = parser().parse_args(['--prompt', 'One prompt', '--output', 'unused'])
    assert default.joint == 2 and default.candidates == 33
    options = parser().parse_args(['--prompt', 'One prompt', '--output', 'unused',
                                  '--joint', '3', '--candidates', '27'])
    assert options.joint == 3 and options.candidates == 27
