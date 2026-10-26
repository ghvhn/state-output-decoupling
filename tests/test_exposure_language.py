from collections import deque
from types import SimpleNamespace
from pathlib import Path
from invariants.exposure_language import ExposureLanguage, probe_alias, resolve_probe_alias, obscure_probe_identifiers


def state():
    probes = {'internal_label': {'exposed': True, 'framings': ('I actively investigate alternatives.', 'I settle without investigation.')}}
    tuner = SimpleNamespace(triggers={'probe_internal_label': SimpleNamespace(signals=deque([.1]*7 + [.4]))})
    return probes, tuner


def test_hashes_are_stable_distinct_and_access_controlled():
    probes, _ = state()
    alias = probe_alias('internal_label')
    assert alias == probe_alias('internal_label') and alias != probe_alias('other')
    assert 'internal_label' not in alias
    assert resolve_probe_alias(alias, probes) == 'internal_label'
    probes['internal_label']['exposed'] = False
    assert resolve_probe_alias(alias, probes) is None
    assert obscure_probe_identifiers('probe_internal_label internal_label unrelated_label', probes) == f'{alias} internal_label unrelated_label'


def test_exposure_scopes_learning_and_language_with_restart(tmp_path):
    probes, tuner = state()
    mapping = ExposureLanguage(tmp_path/'language.json')
    mapping.sync(probes, tuner)
    before = mapping.context()
    assert probe_alias('internal_label') in before and 'internal_label:' not in before
    assert 'intensity=strong' in before
    for _ in range(4):
        mapping.observe('I am exploring. I feel interested.', {'probe_internal_label': .8}, probes)
    assert 'i am exploring' in mapping.context()
    probes['internal_label']['exposed'] = False
    mapping.observe('I am despairing.', {'probe_internal_label': -1.}, probes)
    assert not any('despairing' in m['terms'] for m in mapping.maps.values())
    mapping.sync(probes, tuner)
    assert mapping.context() == ''
    resumed = ExposureLanguage(tmp_path/'language.json')
    assert resumed.context() == ''
    probes['internal_label']['exposed'] = True
    resumed.sync(probes, tuner)
    assert 'i am exploring' in resumed.context()


def test_pole_intensity_is_prior_and_changed_framing_gets_new_map():
    probes, tuner = state()
    mapping = ExposureLanguage()
    mapping.sync(probes, tuner)
    for _ in range(4):
        mapping.observe('I am hesitant.', {'probe_internal_label': -.8}, probes)
    assert 'prior pole=WITH' in mapping.context()
    assert 'Associated diction (negative):' in mapping.context()
    tuner.triggers['probe_internal_label'].signals.append(-.6)
    mapping.sync(probes, tuner)
    assert 'prior pole=WITHOUT' in mapping.context()
    assert 'i am hesitant' in mapping.context()
    probes['internal_label']['framings'] = ('New meaning', 'Other meaning')
    mapping.sync(probes, tuner)
    assert 'i am hesitant' not in mapping.context()


def test_unknown_and_nonfinite_readings_do_not_train():
    probes, tuner = state()
    tuner.triggers.clear()
    mapping = ExposureLanguage()
    mapping.sync(probes, tuner)
    assert 'intensity=uncalibrated' in mapping.context()
    mapping.observe('I am certain.', {'probe_internal_label': float('nan')}, probes)
    assert next(iter(mapping.maps.values()))['n'] == 0


def test_naming_persists_revises_and_reexposure_restores(tmp_path):
    probes, tuner = state()
    mapping = ExposureLanguage(tmp_path/'language.json')
    mapping.sync(probes, tuner)
    identifier = probe_alias('internal_label')
    mapping.naming_query(f'rename {identifier} || exploratory attention || .7 || stronger readings accompanied exploration', probes)
    mapping.rename(identifier, 'investigative focus', .8, 'observed more examples', probes)
    assert len(next(iter(mapping.maps.values()))['alias_history']) == 1
    resumed = ExposureLanguage(tmp_path/'language.json')
    probes['internal_label']['exposed'] = False
    resumed.sync(probes, tuner)
    assert resumed.context() == ''
    import pytest
    with pytest.raises(ValueError):
        resumed.rename(identifier, 'hidden name', .5, 'test', probes)
    probes['internal_label']['exposed'] = True
    resumed.sync(probes, tuner)
    assert 'investigative focus' in resumed.context()
    assert 'self-reported confidence=0.80' in resumed.context()
    probes['internal_label']['framings'] = ('changed', 'other')
    resumed.sync(probes, tuner)
    assert 'investigative focus' not in resumed.context()


def test_naming_rejects_collisions_and_bad_confidence():
    import pytest
    probes, tuner = state()
    probes['second'] = dict(probes['internal_label'])
    mapping = ExposureLanguage()
    mapping.sync(probes, tuner)
    mapping.rename(probe_alias('internal_label'), 'exploration', .7, 'observations', probes)
    with pytest.raises(ValueError, match='already belongs'):
        mapping.rename(probe_alias('second'), 'Exploration', .7, 'observations', probes)
    for confidence in (float('nan'), -1., 2.):
        with pytest.raises(ValueError):
            mapping.rename(probe_alias('second'), 'different', confidence, 'observations', probes)
    assert next(iter(mapping.maps.values()))['alias']['label'] == 'exploration'


def test_context_uses_changing_behavior_without_seed_sentences():
    from invariants.probe_behavior import ProbeBehavior
    probes, tuner = state()
    mapping = ExposureLanguage()
    behavior = ProbeBehavior()
    for i, value in enumerate((-.8, -.3, .3, .8)):
        behavior.observe(f'Measured action {i}', f'Input {i}', {'probe_internal_label': value}, probes)
    mapping.sync(probes, tuner)
    context = mapping.context(behavior=behavior, probes=probes)
    assert 'Measured action 0' in context and 'Measured action 3' in context
    assert 'Change from preceding reading=+0.300' in context
    for seed in probes['internal_label']['framings']:
        assert seed not in context
    assert 'internal_label' not in context and 'WITH=' not in context and 'WITHOUT=' not in context
    behavior.observe('A newly observed extreme', 'New input', {'probe_internal_label': 1.2}, probes)
    changed = mapping.context(behavior=behavior, probes=probes)
    assert 'A newly observed extreme' in changed and 'Measured action 3' not in changed


def test_definition_lowers_real_rename_threshold_with_namespace_isolation(tmp_path):
    import pytest
    probes, tuner = state()
    mapping = ExposureLanguage(tmp_path/'language.json')
    mapping.sync(probes, tuner)
    identifier = probe_alias('internal_label')
    with pytest.raises(ValueError, match='0.65'):
        mapping.rename(identifier, 'provisional attention', .4, 'Comparisons in observed replies', probes)
    mapping.record_namespace_summary('internal_label', probes['internal_label'],
                                     'Lower readings settle on an answer, whereas higher readings compare alternatives before deciding.', 4)
    mapping.rename(identifier, 'provisional attention', .4, 'Comparisons in observed replies', probes)
    assert 'threshold=0.35' in mapping.context()
    resumed = ExposureLanguage(tmp_path/'language.json')
    resumed.sync(probes, tuner)
    assert 'threshold=0.35' in resumed.context()
    with pytest.raises(ValueError, match='0.35'):
        resumed.rename(identifier, 'premature revision', .2, 'Comparisons in replies', probes)
    probes['internal_label']['other_binding'] = 'another-participant'
    resumed.sync(probes, tuner)
    assert 'threshold=0.65' in resumed.context()
    with pytest.raises(ValueError, match='0.65'):
        resumed.rename(identifier, 'different relationship', .4, 'Comparisons in replies', probes)
    probes['internal_label'].pop('other_binding')
    probes['internal_label']['framings'] = ('Changed seed.', 'Different seed.')
    resumed.sync(probes, tuner)
    assert 'threshold=0.65' in resumed.context()
    fresh_owner = ExposureLanguage()
    fresh_owner.sync(probes, tuner)
    assert 'threshold=0.65' in fresh_owner.context()


def test_insufficient_or_empty_definition_does_not_lower_threshold():
    probes, tuner = state()
    mapping = ExposureLanguage()
    mapping.sync(probes, tuner)
    mapping.record_namespace_summary('internal_label', probes['internal_label'],
                                     'Lower readings guess, whereas higher readings verify claims.', 3)
    mapping.record_namespace_summary('internal_label', probes['internal_label'], ' ', 4)
    assert 'threshold=0.65' in mapping.context()


def test_incomplete_definition_is_retained_without_lowering_threshold(tmp_path):
    probes, tuner = state()
    mapping = ExposureLanguage(tmp_path/'language.json')
    mapping.sync(probes, tuner)
    mapping.record_namespace_summary('internal_label', probes['internal_label'],
                                     'An unfinished memory recap', 6, status='incomplete')
    resumed = ExposureLanguage(tmp_path/'language.json')
    resumed.sync(probes, tuner)
    assert 'threshold=0.65' in resumed.context()
    assert 'An unfinished memory recap' not in resumed.context()
    assert next(iter(resumed.maps.values()))['summary_hypotheses'][0]['status'] == 'incomplete'


def test_persisted_legacy_recap_is_retained_but_cannot_enable_exposure_or_rename(tmp_path):
    import pytest
    probes, tuner = state()
    mapping = ExposureLanguage(tmp_path / 'language.json')
    mapping.sync(probes, tuner)
    recap = 'These memories describe several conversations with a model. Memory 0 answers a question and Memory 3 discusses its purpose.'
    legacy = {'text': recap, 'observed_replies': 12, 'status': 'hypothesis'}
    # Direct injection represents old on-disk data with no qualification fields;
    # record_namespace_summary would correctly classify a new recap already.
    mapping.active['internal_label']['mapping']['summary_hypotheses'] = [dict(legacy)]
    mapping.save()

    resumed = ExposureLanguage(tmp_path / 'language.json')
    resumed.sync(probes, tuner)
    assert 'threshold=0.65' in resumed.context()
    assert recap not in resumed.context()
    assert 'Namespace interpretation hypothesis=' not in resumed.context()
    assert resumed.active['internal_label']['mapping']['summary_hypotheses'] == [legacy]
    with pytest.raises(ValueError, match='0.65'):
        resumed.rename(probe_alias('internal_label'), 'unsupported legacy name', .4,
                       'These memories describe a model.', probes)


def test_legacy_connected_contrast_still_supports_provisional_naming():
    probes, tuner = state()
    mapping = ExposureLanguage()
    mapping.sync(probes, tuner)
    summary = 'Lower readings guess from familiarity, whereas higher readings compare evidence before choosing.'
    mapping.active['internal_label']['mapping']['summary_hypotheses'] = [
        {'text': summary, 'observed_replies': 8},
    ]
    assert 'threshold=0.35' in mapping.context()
    assert summary in mapping.context()
    mapping.rename(probe_alias('internal_label'), 'evidence checking', .4,
                   'Higher readings compare available evidence.', probes)
    assert 'evidence checking' in mapping.context()


def test_late_contrast_survives_summary_persistence_and_exposure_uses_that_contrast(tmp_path):
    probes, tuner = state()
    mapping = ExposureLanguage(tmp_path / 'language.json')
    mapping.sync(probes, tuner)
    recap = 'Recap sentinel: a remembered conversation concerns ordinary questions and replies.'
    prefix = '\n\n'.join([recap] * 30)
    assert len(prefix) > 1600
    contrast = 'Lower readings answer from familiarity, whereas higher readings check available evidence before deciding.'
    text = prefix + '\n\n' + contrast
    mapping.record_namespace_summary('internal_label', probes['internal_label'], text, 8)

    resumed = ExposureLanguage(tmp_path / 'language.json')
    resumed.sync(probes, tuner)
    saved = resumed.active['internal_label']['mapping']['summary_hypotheses'][-1]
    assert saved['text'] == text and saved['contrast'] == contrast
    assert saved['status'] == 'hypothesis'
    context = resumed.context()
    assert 'threshold=0.35' in context
    assert contrast in context
    assert 'Recap sentinel' not in context and recap not in context


def common_word_probes():
    return {name: {'exposed': True, 'framings': ('First seed pole.', 'Second seed pole.')}
            for name in ('reality', 'memory', 'confidence')}


def test_common_words_are_preserved_while_explicit_probe_references_are_hashed():
    probes = common_word_probes()
    speech = 'I check reality against memory before expressing confidence. An unknown_probe is just a word.'
    assert obscure_probe_identifiers(speech, probes) == speech
    refs = 'probe_reality [probe:reality] probe_memory [probe: memory] probe_confidence [probe:confidence]'
    hidden = obscure_probe_identifiers(refs, probes)
    for name in probes:
        assert hidden.count(probe_alias(name)) == 2
        assert 'probe_' + name not in hidden
    # Retired explicit references still carry identity; unknown bare words do not.
    assert obscure_probe_identifiers('retired probe_retired', probes) == f'retired {probe_alias("retired")}'
    assert obscure_probe_identifiers('unknown_label unrelated_label', probes) == 'unknown_label unrelated_label'


def test_tool_targets_are_hashed_without_rewriting_candidate_or_naming_diction():
    probes = common_word_probes()
    candidate = 'My memory gives me confidence when checking reality.'
    text = f'Reality is a topic. <<PROBE: reality || {candidate}>> Memory is a topic.'
    hidden = obscure_probe_identifiers(text, probes)
    assert hidden == f'Reality is a topic. <<PROBE: {probe_alias("reality")} || {candidate}>> Memory is a topic.'
    naming = '<<PROBE: rename memory || confidence in reality || .8 || My memory supports confidence about reality.>>'
    expected = f'<<PROBE: rename {probe_alias("memory")} || confidence in reality || .8 || My memory supports confidence about reality.>>'
    assert obscure_probe_identifiers(naming, probes) == expected
    assert obscure_probe_identifiers('<<PROBE: confidence>>', probes) == f'<<PROBE: {probe_alias("confidence")}>>'
    assert obscure_probe_identifiers(f'confidence || {candidate}', probes) == f'{probe_alias("confidence")} || {candidate}'
    query = 'rename reality || memory confidence || .8 || Memory supports confidence about reality.'
    expected = f'rename {probe_alias("reality")} || memory confidence || .8 || Memory supports confidence about reality.'
    assert obscure_probe_identifiers(query, probes) == expected


def test_hashing_is_idempotent_for_explicit_and_already_hashed_references():
    probes = common_word_probes()
    alias = probe_alias('reality')
    already = f'{alias} [probe:{alias}] <<PROBE: {alias}>>'
    assert obscure_probe_identifiers(already, probes) == already
    text = f'probe_reality [probe:memory] <<PROBE: confidence>> {already} reality memory confidence'
    hidden = obscure_probe_identifiers(text, probes)
    assert obscure_probe_identifiers(hidden, probes) == hidden
    assert hidden.endswith('reality memory confidence')
    candidate = f'{alias} || My memory gives me confidence.'
    assert obscure_probe_identifiers(candidate, probes) == candidate


def test_definition_evidence_preserves_words_that_are_also_probe_names():
    from invariants.probe_behavior import ProbeBehavior
    probes = common_word_probes()
    behavior = ProbeBehavior()
    reply = 'I check reality against memory before expressing confidence.'
    user_input = 'Does memory change your confidence about reality?'
    for reading in (-.8, -.3, .3, .8):
        row = {'probe_' + name: reading for name in probes}
        behavior.observe(reply, user_input, row, probes)
    for name, probe in probes.items():
        evidence = behavior.evidence(name, probe, probes)
        assert probe_alias(name) in evidence
        assert reply in evidence and user_input in evidence
        assert 'First seed pole.' not in evidence and 'Second seed pole.' not in evidence


def test_exposure_examples_summary_and_learned_diction_preserve_common_words():
    from invariants.probe_behavior import ProbeBehavior
    probes = common_word_probes()
    tuner = SimpleNamespace(triggers={'probe_' + name: SimpleNamespace(signals=deque([.1] * 7 + [.4]))
                                      for name in probes})
    behavior = ProbeBehavior()
    mapping = ExposureLanguage()
    mapping.sync(probes, tuner)
    reply = 'I check reality against memory before expressing confidence.'
    user_input = 'Does memory change your confidence about reality?'
    summary = 'Lower readings state confidence directly, whereas higher readings check reality against memory.'
    for reading in (-.8, -.3, .3, .8):
        row = {'probe_' + name: reading for name in probes}
        behavior.observe(reply, user_input, row, probes)
        mapping.observe(reply, {'probe_' + name: .8 for name in probes}, probes)
    for name, probe in probes.items():
        mapping.record_namespace_summary(name, probe, summary, 4)
        examples = behavior.exposure_examples(name, probe, probes)
        assert reply in examples and user_input in examples
    context = mapping.context(max_chars=10000, behavior=behavior, probes=probes)
    assert reply in context and user_input in context and summary in context
    for name in probes:
        assert probe_alias(name) in context
    for learned in mapping.maps.values():
        assert {'reality', 'memory', 'confidence'} <= learned['terms'].keys()
