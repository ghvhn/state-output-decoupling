from collections import deque
from types import SimpleNamespace
import torch
import pytest
from invariants.other_probes import OtherReference, is_other_probe
from invariants.exposure_language import ExposureLanguage
from scripts.interactive_phenomenality import probe_is_released


def probe():
    return {'framings': ('I enjoy conversing with @.', 'Conversing with @ is monotonous.'),
            'direction': {0: torch.tensor([1., 0.])}, 'history': deque([1.]), 'exposed': True}


def test_marker_semantics_and_self():
    ref = OtherReference()
    assert 'unspecified Other' in ref.resolve('I talk with @.')
    ref.select('Gavin')
    assert ref.resolve('I talk with @. @Alex listens.') == 'I talk with Gavin. Alex listens.'
    assert ref.resolve('a@example.com') == 'a@example.com'
    ref.select('self')
    assert 'agent owning this probe' in ref.resolve('I talk with @.')
    assert '@Alex' not in ref.resolve('@Alex')
    assert is_other_probe(probe()) and probe_is_released(probe())


def test_binding_mints_caches_resets_and_is_transactional():
    ref = OtherReference()
    probes = {'social': probe()}
    calls = []
    def mint(frames):
        calls.append(frames)
        return {0: torch.tensor([0., 1.])}
    ref.activate(probes, mint)
    first = probes['social']['other_binding']
    assert not probes['social']['history']
    ref.select('self')
    ref.activate(probes, mint)
    assert first != probes['social']['other_binding']
    ref.select(None)
    ref.activate(probes, mint)
    assert len(calls) == 2
    assert probes['social']['other_binding'] == first
    ref.select('new person')
    with pytest.raises(ValueError):
        ref.activate(probes, lambda frames: {})
    assert probes['social']['other_binding'] == first


def test_composition_inherits_observation_and_rebinds():
    ref = OtherReference()
    probes = {'a': probe(), 'b': probe(), 'social': {'direction': {}, 'framings': ('composed', ''),
              'other_related': True, 'other_composition': [(1., 'a'), (-.5, 'b')]}}
    ref.activate(probes, lambda frames: {0: torch.tensor([1., 0.])})
    assert probe_is_released(probes['social'])
    ref.select('self')
    ref.activate(probes, lambda frames: {0: torch.tensor([0., 1.])})
    torch.testing.assert_close(probes['social']['direction'][0], torch.tensor([0., 1.]))


def test_other_identity_partitions_names_and_language(tmp_path):
    ref = OtherReference(tmp_path/'others.json')
    probes = {'social': probe()}
    tuner = SimpleNamespace(triggers={})
    language = ExposureLanguage()
    ref.activate(probes, lambda frames: {0: torch.tensor([1., 0.])})
    language.sync(probes, tuner)
    first_key = language.active['social']['key']
    ref.select('self')
    ref.activate(probes, lambda frames: {0: torch.tensor([0., 1.])})
    language.sync(probes, tuner)
    assert language.active['social']['key'] != first_key
    assert 'Relational observation' in language.context()
    ref.save()
    assert OtherReference(tmp_path/'others.json').current == 'self'


def test_participants_share_reply_scoring_but_keep_baselines_separate():
    reference = OtherReference()
    probes = {'social': probe()}
    calls = []
    def mint(frames):
        calls.append(frames)
        return {0: torch.tensor([1., 0.])}
    first = reference.participant_readings(probes, ['self', 'user'], mint, lambda direction: 2.)
    assert first['social']['self']['centered'] is None
    second = reference.participant_readings(probes, ['self', 'user'], mint, lambda direction: 3.)
    assert second['social']['self']['centered'] == 1.
    assert second['social']['user']['centered'] == 1.
    assert len(calls) == 2
    assert reference.current is None
