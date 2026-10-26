"""Successful output-kind associations are prospective, scoped, and reusable."""
import copy
import hashlib
import pickle
import re
from types import SimpleNamespace

import pytest
import torch

from invariants.defining_profile import (
    CAPTURE_PHASE, DefiningProfile, MAX_HISTORY, MAX_REPLAY_GAIN,
)
from invariants import defining_profile
from invariants.probe_definition_output import OUTPUT_PREFIX, build_definition_request


class Config:
    def __init__(self):
        self._name_or_path = 'toy/defining-model'
        self._commit_hash = 'a' * 40
        self._attn_implementation = 'eager'
        self.quantization_config = None

    def to_dict(self):
        return {'name': self._name_or_path, 'revision': self._commit_hash,
                'hidden_size': 4, 'num_hidden_layers': 3}


class Tokenizer:
    def __init__(self):
        self.chat_template = 'toy-native-user-assistant-v1'
        self.name_or_path = 'toy/defining-tokenizer'
        self.init_kwargs = {}
        self.special_tokens_map = {'eos_token': '<eos>'}
        self.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"toy":"v1"}')
        self.prompts = []
        self.messages = []

    def get_vocab(self):
        return {'toy': 0, '<eos>': 1}

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
        assert len(messages) == 1 and messages[0]['role'] == 'user'
        return '<native:user>\n' + messages[0]['content'] + '\n<native:assistant>\n'

    def __call__(self, prompt, **kwargs):
        assert kwargs == {'return_tensors': 'pt', 'add_special_tokens': False}
        self.prompts.append(prompt)
        index = len(self.prompts) - 1
        return {'input_ids': torch.tensor([[index, index, index]]),
                'attention_mask': torch.ones(1, 3, dtype=torch.long)}


class Block(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, hidden):
        return hidden + self.increment


class NativeModel(torch.nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tok = tokenizer
        self.config = Config()
        self.parameter = torch.nn.Parameter(torch.ones(1))
        self.dtype = torch.float32
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([Block(1), Block(2), Block(3)])
        self.calls = []
        self.delta_scale = 1.
        self.constant_change = None
        self.failure = None

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        prompt = self.tok.prompts[int(input_ids[0, 0])]
        self.calls.append({'prompt': prompt, 'grad': torch.is_grad_enabled(), 'use_cache': use_cache})
        if self.failure == 'forward':
            raise RuntimeError('prospective capture unavailable')
        x = float(re.search(r'x=([0-9.-]+)', prompt)[1])
        y = float(re.search(r'y=([0-9.-]+)', prompt)[1])
        source = torch.tensor([10 + x, 20 + y, 30 + 2 * x, 40 + 3 * y])
        change = torch.tensor([.2 + .1 * x, .2 + .15 * y, .1 + .05 * x, .2 + .12 * y])
        if self.constant_change is not None:
            change = self.constant_change.clone()
        if '[Definition request]' in prompt:
            source += self.delta_scale * change
            if '[Format correction]' in prompt:
                source += self.delta_scale * .03
        if self.failure == 'nonfinite':
            source[0] = float('nan')
        hidden = torch.stack((source + 1000, source + 2000, source)).unsqueeze(0)
        for layer in self.model.layers:
            hidden = layer(hidden)
        # Deliberately changed final state: profiles must capture block outputs,
        # rather than an exposed final-normalized hidden-state endpoint.
        return SimpleNamespace(last_hidden_state=hidden * 100)


def wrapped_model():
    tokenizer = Tokenizer()
    native = NativeModel(tokenizer)
    return SimpleNamespace(model=native, tok=tokenizer, device='cpu', d_model=4, n_layers=3)


SETTINGS = {
    'assistant_prefix': OUTPUT_PREFIX, 'subject_fraction': .02, 'max_tokens': 384,
    'decoding': 'greedy', 'config_overrides': {'cache_enabled': False, 'cache_write_enabled': False,
                                            'max_routing_events': 0, 'max_synthesis_events': 0,
                                            'max_synthesis_steps': 0, 'max_loops': 0},
}


def evidence(x=0., y=0.):
    return f'[Probe p_000000000000000000000000]\nx={x} y={y}\nOriginal measured observations.'


def session(profile, model=None, x=0., y=0., settings=None, layers=(0, 2), namespace='probe:one'):
    return profile.session(model or wrapped_model(), layers, evidence=evidence(x, y),
                           namespace=namespace, settings=copy.deepcopy(settings or SETTINGS))


def observe(current, request, *, accepted=True, status=None, reason='Definition form passed.', text='A mistaken definition.'):
    status = status or ('hypothesis' if accepted else 'unqualified')
    record = {'text': text, 'status': status,
              'qualification': {'accepted': accepted, 'reason': reason,
                                'structural_only': True, 'semantic_truth_checked': False},
              'tokens_generated': 42, 'budget': 384,
              'request_sha256': hashlib.sha256(request.encode('utf-8')).hexdigest()}
    current.observe(record)
    return record


def explicit_success(profile, model, x=0., y=0., *, feedback=None, settings=None, namespace='probe:one'):
    current = session(profile, model, x, y, settings=settings, namespace=namespace)
    request = current.request(current.evidence, feedback, 1 if feedback else 0)
    assert request != current.evidence
    current.capture(request)
    observe(current, request)
    return current


def train(profile, model):
    # Independent sources and varying conditioning changes identify transport.
    for x, y in ((0., 0.), (1., 0.), (0., 1.)):
        explicit_success(profile, model, x, y)


def test_exact_native_prefill_raw_block_capture_and_recipe_outcome(tmp_path):
    model = wrapped_model()
    profile = DefiningProfile(tmp_path / 'owner.defining.pt')
    current = explicit_success(profile, model, 2., 3.)
    assert len(model.model.calls) == 2
    assert all(call['grad'] is False and call['use_cache'] is False for call in model.model.calls)
    assert model.model.calls[0]['prompt'] == '<native:user>\n' + evidence(2., 3.) + '\n<native:assistant>\n' + OUTPUT_PREFIX
    assert model.model.calls[1]['prompt'] == '<native:user>\n' + build_definition_request(evidence(2., 3.)) + '\n<native:assistant>\n' + OUTPUT_PREFIX
    assert all('mistaken definition' not in call['prompt'] for call in model.model.calls)
    family = profile.families[current.identity_hash]
    sample = family['samples'][0]
    torch.testing.assert_close(sample['sources'][2], torch.tensor([18., 29., 40., 55.]))
    torch.testing.assert_close(sample['changes'][2], torch.tensor([.4, .65, .2, .56]), atol=3e-6, rtol=1e-5)
    assert set(sample['sources']) == {0, 2}
    assert family['history'][0]['capture']['phase'] == CAPTURE_PHASE
    assert family['history'][0]['accepted'] and len(family['recipes']) == 1
    assert all(not layer._forward_hooks for layer in model.model.model.layers)
    assert current.metadata()['semantic_truth_checked'] is False


def test_restart_reuses_successful_correction_with_fresh_evidence_and_no_failed_prose(tmp_path, monkeypatch):
    path = tmp_path / 'owner.defining.pt'
    load = torch.load
    seen = []
    def safe_load(*args, **kwargs):
        seen.append(kwargs.get('weights_only'))
        return load(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', safe_load)
    model = wrapped_model()
    profile = DefiningProfile(path)
    reason = 'State the general rule before its lower and higher ends.'
    previous = explicit_success(profile, model, 0., 0., feedback=reason)
    profile.enabled = False
    profile.save()
    assert DefiningProfile(path).enabled is False
    profile.enabled = True
    profile.save()
    restored = DefiningProfile(path)
    current = session(restored, model, 9., 8.)
    request = current.request(current.evidence, None, 0)
    assert request == build_definition_request(evidence(9., 8.), reason)
    assert evidence(0., 0.) not in request
    assert previous.metadata()['attempts'][0]['settings'] == SETTINGS
    assert seen and all(value is True for value in seen)


def test_failure_history_never_promotes_failure_recipe_or_change(tmp_path):
    model = wrapped_model()
    profile = DefiningProfile(tmp_path / 'owner.pt')
    accepted = explicit_success(profile, model, feedback='Successful correction.')
    current = session(profile, model, 1., 2.)
    request = current.request(current.evidence, 'Reject this recap.', 1)
    current.capture(request)
    observe(current, request, accepted=False, text='Rejected prose with invented facts.')
    family = profile.families[current.identity_hash]
    assert len(family['samples']) == 1
    assert family['history'][-1]['status'] == 'unqualified'
    assert sorted(recipe['successes'] for recipe in family['recipes'].values()) == [0, 1]
    fresh = session(profile, model, 3., 4.)
    working = fresh.request(fresh.evidence, None, 0)
    assert 'Successful correction.' in working and 'Reject this recap.' not in working
    assert 'Rejected prose with invented facts.' not in str(fresh.metadata())


@pytest.mark.parametrize('change', ['revision', 'tokenizer', 'layers', 'settings'])
def test_identity_layer_and_settings_mismatch_do_not_reuse(change, tmp_path):
    profile, model = DefiningProfile(tmp_path / 'owner.pt'), wrapped_model()
    train(profile, model)
    changed_settings, layers = copy.deepcopy(SETTINGS), (0, 2)
    if change == 'revision':
        model.model.config._commit_hash = 'b' * 40
    elif change == 'tokenizer':
        model.tok.chat_template = 'toy-native-user-assistant-v2'
    elif change == 'layers':
        layers = (1, 2)
    else:
        changed_settings['subject_fraction'] = .03
    current = session(profile, model, 2., 2., settings=changed_settings, layers=layers)
    request = current.request(current.evidence, None, 0)
    assert request == build_definition_request(current.evidence)
    assert current.steering() is None
    if change != 'settings':
        assert current.family_status == 'new' and len(profile.families) == 2


def test_owners_do_not_share_success_samples(tmp_path):
    first = DefiningProfile(tmp_path / 'owner-a.pt')
    model = wrapped_model()
    train(first, model)
    second = DefiningProfile(tmp_path / 'owner-b.pt')
    current = session(second, model, 2., 2.)
    assert current.request(current.evidence, None, 0) != current.evidence
    assert current.metadata()['successful_samples'] == 0


def test_adaptive_replay_preserves_raw_magnitude_and_excludes_replay_training(tmp_path):
    model = wrapped_model()
    profile = DefiningProfile(tmp_path / 'owner.pt')
    train(profile, model)
    current = session(profile, model, .3, .4)
    before = len(model.model.calls)
    request = current.request(current.evidence, None, 0)
    assert request == current.evidence
    current.capture(request)
    assert len(model.model.calls) == before + 1  # Source only; no conditioned target on replay.
    steering = current.steering()
    assert steering['layer'] == 2 and steering['direction'].shape == (4,)
    assert steering['direction'].dtype == torch.float32 and steering['direction'].device.type == 'cpu'
    torch.testing.assert_close(steering['direction'].norm(), torch.tensor(1.))
    diagnostics = steering['diagnostics']
    assert 0 < steering['gain'] < MAX_REPLAY_GAIN
    assert steering['gain'] == pytest.approx(diagnostics['raw_change_norm'] / diagnostics['source_norm'])
    reconstructed = steering['direction'] * steering['gain'] * diagnostics['source_norm']
    assert reconstructed.norm().item() == pytest.approx(diagnostics['raw_change_norm'], rel=1e-5)
    observe(current, request)
    family = profile.families[current.identity_hash]
    assert len(family['samples']) == 3
    assert sum(recipe['successes'] for recipe in family['recipes'].values()) == 3
    assert family['replay_stats'] == {'successes': 1, 'failures': 0}
    assert family['history'][-1]['capture']['training_excluded']


def test_replay_gain_cap_and_rejected_replay_falls_back_to_fresh_explicit_request(tmp_path):
    model = wrapped_model()
    model.model.delta_scale = 100.
    profile = DefiningProfile(tmp_path / 'owner.pt')
    train(profile, model)
    current = session(profile, model, .3, .4)
    request = current.request(current.evidence, None, 0)
    current.capture(request)
    steering = current.steering()
    assert steering['gain'] == MAX_REPLAY_GAIN and steering['diagnostics']['gain_capped']
    observe(current, request, accepted=False, text='Never put this rejected prose in a request.')
    fallback = current.request(current.evidence, 'A definition needs a general property.', 1)
    assert fallback == build_definition_request(current.evidence, 'A definition needs a general property.')
    assert current.steering() is None and 'Never put this' not in fallback
    current.capture(fallback)
    observe(current, fallback)
    family = profile.families[current.identity_hash]
    assert family['replay_stats'] == {'successes': 0, 'failures': 1}
    assert len(family['samples']) == 4  # Only the subsequent explicit success joins training.


def test_duplicate_or_constant_changes_do_not_claim_adaptive_readiness(tmp_path):
    model = wrapped_model()
    profile = DefiningProfile(tmp_path / 'owner.pt')
    for _ in range(3):
        explicit_success(profile, model)
    current = session(profile, model, 2., 2.)
    assert current.request(current.evidence, None, 0) != current.evidence
    assert current.metadata()['replay']['distinct_sources'] == 1
    assert len(current.family['samples']) == 1


def test_distinct_sources_with_constant_conditioning_change_remain_explicit(tmp_path):
    model = wrapped_model()
    model.model.constant_change = torch.full((4,), .25)
    profile = DefiningProfile(tmp_path / 'owner.pt')
    train(profile, model)
    current = session(profile, model, 2., 2.)
    assert current.request(current.evidence, None, 0) != current.evidence
    assert current.steering() is None
    assert current.metadata()['replay']['distinct_sources'] == 3
    assert 'do not identify an adaptive change' in current.metadata()['replay']['reason']


def test_incomplete_attempt_cannot_train_even_if_its_form_would_pass():
    model, profile = wrapped_model(), DefiningProfile()
    current = session(profile, model)
    request = current.request(current.evidence, None, 0)
    current.capture(request)
    observe(current, request, accepted=True, status='incomplete', reason='Token limit reached.')
    assert current.family['samples'] == []
    assert current.family['history'][-1]['accepted'] is False
    assert next(iter(current.family['recipes'].values()))['successes'] == 0


@pytest.mark.parametrize('failure', ['forward', 'nonfinite', 'identity', 'degenerate'])
def test_capture_failure_or_degeneracy_retains_settings_outcome_and_cleans_hooks(failure, tmp_path):
    model = wrapped_model()
    if failure == 'identity':
        model.model.config._commit_hash = None
    elif failure == 'degenerate':
        model.model.delta_scale = 0.
    else:
        model.model.failure = failure
    profile = DefiningProfile(tmp_path / 'owner.pt')
    current = session(profile, model)
    request = current.request(current.evidence, None, 0)
    assert '[Definition request]' in request
    current.capture(request)
    observe(current, request)
    assert current.steering() is None
    assert all(not layer._forward_hooks for layer in model.model.model.layers)
    history = current.family['history'] if current.family is not None else profile.unidentified_history
    assert history[-1]['accepted'] and history[-1]['settings'] == SETTINGS
    if current.family is not None:
        assert current.family['samples'] == []
    restored = DefiningProfile(profile.path)
    assert 'qualified' in restored.summary()


def test_existing_unrelated_hooks_survive_success_and_capture_error(tmp_path):
    model = wrapped_model()
    calls = []
    handle = model.model.model.layers[2].register_forward_hook(lambda *_args: calls.append('existing'))
    before = set(model.model.model.layers[2]._forward_hooks)
    try:
        explicit_success(DefiningProfile(), model)
        assert calls == ['existing', 'existing']
        model.model.failure = 'forward'
        current = session(DefiningProfile(), model)
        current.capture(current.request(current.evidence, None, 0))
        assert set(model.model.model.layers[2]._forward_hooks) == before
    finally:
        handle.remove()


@pytest.mark.parametrize('corruption', ['nan', 'dtype', 'shape', 'phase', 'suffix', 'schema'])
def test_corrupt_or_malformed_saved_profile_is_rejected_without_replacement(corruption, tmp_path):
    path = tmp_path / 'owner.pt'
    model, profile = wrapped_model(), DefiningProfile(path)
    accepted = explicit_success(profile, model)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    family = payload['families'][accepted.identity_hash]
    if corruption == 'nan':
        family['samples'][0]['changes'][2][0] = float('nan')
    elif corruption == 'dtype':
        family['samples'][0]['sources'][2] = family['samples'][0]['sources'][2].double()
    elif corruption == 'shape':
        family['samples'][0]['sources'][2] = torch.ones(3)
    elif corruption == 'phase':
        family['identity']['capture_phase'] = 'completed-answer-final-state'
    elif corruption == 'suffix':
        next(iter(family['recipes'].values()))['suffix'] = '\nThe probe means a fixed invented meaning.'
    else:
        payload['schema'] = 999
    torch.save(payload, path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='could not be loaded safely'):
        DefiningProfile(path)
    assert path.read_bytes() == original


def test_nonserialized_file_remains_intact_on_safe_load_rejection(tmp_path):
    path = tmp_path / 'owner.pt'
    path.write_bytes(b'not a restricted tensor profile')
    with pytest.raises(ValueError, match='could not be loaded safely'):
        DefiningProfile(path)
    assert path.read_bytes() == b'not a restricted tensor profile'


def test_persistence_failure_keeps_prior_file_and_completed_definition_in_session(tmp_path, monkeypatch):
    path = tmp_path / 'owner.pt'
    model, profile = wrapped_model(), DefiningProfile(path)
    explicit_success(profile, model)
    original = path.read_bytes()
    def cannot_replace(*_args):
        raise OSError('Archive is temporarily unavailable')
    monkeypatch.setattr(defining_profile.os, 'replace', cannot_replace)
    current = explicit_success(profile, model, 1., 1.)
    assert current.metadata()['attempts'][-1]['accepted'] is True
    assert len(current.family['samples']) == 2
    assert 'Archive is temporarily unavailable' in current.metadata()['persistence_error']
    assert path.read_bytes() == original
    assert len(DefiningProfile(path).families[current.identity_hash]['samples']) == 1
    assert list(tmp_path.glob('*.tmp')) == []
    assert 'Save unavailable' in profile.summary()


@pytest.mark.parametrize('failure', [pickle.UnpicklingError, EOFError])
def test_temporary_safe_validation_failure_cannot_hide_a_completed_definition(failure, tmp_path, monkeypatch):
    path = tmp_path / 'owner.pt'
    model, profile = wrapped_model(), DefiningProfile(path)
    explicit_success(profile, model)
    original = path.read_bytes()
    restricted_load = torch.load
    def rejected_temporary_load(*args, **kwargs):
        if str(args[0]).endswith('.tmp'):
            raise failure('Temporary archive validation rejected')
        return restricted_load(*args, **kwargs)
    monkeypatch.setattr(defining_profile.torch, 'load', rejected_temporary_load)
    current = explicit_success(profile, model, 1., 1.)
    assert current.metadata()['attempts'][-1]['accepted'] is True
    assert len(current.family['samples']) == 2
    assert failure.__name__ in current.metadata()['persistence_error']
    assert path.read_bytes() == original
    assert list(tmp_path.glob('*.tmp')) == []


def test_only_observed_truncated_replay_requests_explicit_fallback(tmp_path):
    model, profile = wrapped_model(), DefiningProfile(tmp_path / 'owner.pt')
    train(profile, model)
    current = session(profile, model, .3, .4)
    request = current.request(current.evidence, None, 0)
    assert request == current.evidence
    current.capture(request)
    record = {'text': 'Unfinished prospective replay', 'status': 'incomplete',
              'qualification': {'accepted': False, 'reason': 'Token limit reached.'},
              'tokens_generated': 384, 'budget': 384,
              'request_sha256': hashlib.sha256(request.encode('utf-8')).hexdigest()}
    assert current.needs_explicit_fallback(record) is False  # Still pending actual observation.
    current.observe(record)
    assert current.needs_explicit_fallback(record) is True
    assert current.needs_explicit_fallback(dict(record, request_sha256='f' * 64)) is False
    assert current.needs_explicit_fallback(dict(record, status='unqualified')) is False
    fallback = current.request(current.evidence, 'Token limit reached.', 1)
    assert fallback != current.evidence
    assert current.needs_explicit_fallback(record) is False  # Another request is now pending.
    current.capture(fallback)
    explicit_record = observe(current, fallback, accepted=False, status='incomplete', reason='Token limit reached.')
    assert current.needs_explicit_fallback(explicit_record) is False
    assert current.needs_explicit_fallback(record) is False
    assert len(current.family['samples']) == 3
    assert current.family['replay_stats'] == {'successes': 0, 'failures': 1}


def test_actual_gate_hash_must_match_the_captured_request():
    current = session(DefiningProfile())
    request = current.request(current.evidence, None, 0)
    current.capture(request)
    with pytest.raises(ValueError, match='does not match'):
        current.observe({'status': 'hypothesis', 'qualification': {'accepted': True},
                         'request_sha256': 'f' * 64})


def test_unidentified_attempt_history_remains_bounded(tmp_path):
    model = wrapped_model()
    model.model.config._commit_hash = None
    profile = DefiningProfile(tmp_path / 'owner.pt')
    for index in range(MAX_HISTORY + 3):
        current = session(profile, model, float(index), 0.)
        request = current.request(current.evidence, None, 0)
        current.capture(request)
        observe(current, request, accepted=False)
    assert len(profile.unidentified_history) == MAX_HISTORY
    assert len(DefiningProfile(profile.path).unidentified_history) == MAX_HISTORY
