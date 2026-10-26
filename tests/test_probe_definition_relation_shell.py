"""The shell resolves fresh relational conversions before decision-only hooks."""
import copy
import hashlib
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from invariants import engine, probe_definition_relations, probe_definition_steps
from invariants.probe_behavior import ProbeBehavior, definition_key
from scripts import interactive_phenomenality as shell
from test_probe_definition_process_shell import PipelineHarness, STAGE_TEXT
from test_probe_definition_shell import RECAP


class RelationHarness(PipelineHarness):
    def __init__(self, monkeypatch, tmp_path, **kwargs):
        gains = {'probe_define_process': 2., **kwargs.pop('gains', {})}
        super().__init__(monkeypatch, tmp_path, gains=gains, **kwargs)
        self.relation_bank = {
            step: {layer: {'source_anchors': torch.eye(3), 'coefficients': torch.eye(3)}
                   for layer in (4, 5)}
            for step in probe_definition_steps.STEP_ORDER
        }
        self.relation_metadata = {
            'version': probe_definition_relations.RELATION_VERSION,
            'path': str((tmp_path / 'relations-test.pt').resolve()),
            'identity_hash': 'relation-test-identity', 'identity': {'model': 'cpu-test'},
            'step_order': list(probe_definition_steps.STEP_ORDER),
            'gains': dict(probe_definition_steps.STEP_GAINS),
            'provenance': {'endpoint': 'native_assistant_decision',
                           'calibration_rows_per_step': 12, 'teacher_pairs': 2,
                           'recipe_sha256': 'operator-relation-recipe',
                           'status': 'candidate', 'semantic_truth_checked': False},
            'profiles': {step: {str(layer): {'label': 'operator-only-shape-profile',
                                            'source_anchors': 12} for layer in (4, 5)}
                         for step in probe_definition_steps.STEP_ORDER},
            'load_status': 'loaded', 'cache_status': 'hit',
        }
        self.relation_loader = Mock(return_value=(self.relation_bank, self.relation_metadata))
        monkeypatch.setattr(probe_definition_relations, 'get_or_mint_definition_relations',
                            self.relation_loader)
        self.resolutions = []
        self.resolve_failure = None
        self.resolver = Mock(side_effect=self.resolve)
        monkeypatch.setattr(probe_definition_relations, 'resolve_definition_relations', self.resolver)

    def resolve(self, model, bank, step, stage_evidence):
        assert model is self.model and bank is self.relation_bank
        # Encoding the unprimed source happens before new steering hooks exist,
        # and after the previous stage has removed its own hooks.
        assert [set(layer._forward_hooks) for layer in self.layers] == self.original_layer_hooks
        assert set(self.root._forward_pre_hooks) == self.original_root_hooks
        assert not self.config.cache_enabled and not self.config.cache_write_enabled
        for name in ('max_routing_events', 'max_synthesis_events', 'max_synthesis_steps', 'max_loops'):
            assert getattr(self.config, name) == 0
        if self.resolve_failure == step:
            raise RuntimeError('relation resolution failed')
        digest = hashlib.sha256(stage_evidence.encode('utf-8')).hexdigest()
        ratio = .2 + int(digest[:8], 16) / (2 ** 32 - 1)
        direction = torch.tensor([0., 1., ratio])
        direction /= direction.norm()
        scale = .6 + .8 * int(digest[8:12], 16) / 65535
        directions = {layer: direction.clone() for layer in (4, 5)}
        diagnostics = {
            'version': probe_definition_relations.RELATION_VERSION, 'step': step,
            'evidence_sha256': digest, 'source_format': 'native_assistant_decision',
            'semantic_truth_checked': False, 'layer_scales': {4: scale, 5: min(2., scale * 1.2)},
            'layers': {layer: {'adaptive': True, 'candidate': True, 'source_distance': ratio,
                               'source_scale': scale, 'raw_delta_norm': .1,
                               'profile': 'operator-only-resolution-details'} for layer in (4, 5)},
        }
        self.resolutions.append({'step': step, 'evidence': stage_evidence,
                                 'directions': copy.deepcopy(directions),
                                 'diagnostics': copy.deepcopy(diagnostics)})
        return directions, diagnostics

    def generate(self, model, **kwargs):
        resolution = self.resolutions[-1]
        call = self.handles.call_args
        step = resolution['step']
        assert kwargs['instruction'] == resolution['evidence']
        assert call.kwargs['task_positions'] == {step: 'decision'}
        assert call.kwargs['task_layer_scales'] == {step: resolution['diagnostics']['layer_scales']}
        for layer, vector in resolution['directions'].items():
            torch.testing.assert_close(call.args[2][step][layer], vector)
        for operator_text in ('operator-only-shape-profile', 'operator-only-resolution-details',
                              'operator-relation-recipe', 'source_anchors', 'layer_scales',
                              probe_definition_relations.RELATION_VERSION):
            assert operator_text not in kwargs['instruction']
        return super().generate(model, **kwargs)


def test_relational_mode_resolves_each_candidate_fresh_and_saves_operator_profiles(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    harness = RelationHarness(monkeypatch, tmp_path)
    records_before = copy.deepcopy(harness.behavior.records)
    probe_before = copy.deepcopy(harness.probe)
    bank_before = copy.deepcopy(harness.relation_bank)
    evidence = harness.behavior.evidence('secret_probe', harness.probe, harness.probes)
    harness.run()

    harness.relation_loader.assert_called_once_with(
        harness.model, (4, 5), Path(shell.__file__).resolve().parent.parent / 'invariants/out/definition_relations')
    harness.bank_loader.assert_not_called()
    assert harness.resolver.call_count == harness.handles.call_count == harness.generate_calls == 4
    assert [row['step'] for row in harness.resolutions] == list(probe_definition_steps.STEP_ORDER)
    assert [row['evidence'] for row in harness.resolutions] == harness.instructions
    vectors = [row['directions'][4] for row in harness.resolutions]
    assert all(not torch.allclose(first, second) for first, second in zip(vectors, vectors[1:]))
    controls = harness.attempt()['controls']
    assert controls['mode'] == 'relational'
    assert controls['relation_version'] == probe_definition_relations.RELATION_VERSION
    assert controls['relation_profiles'] == harness.relation_metadata
    assert controls['evidence_sha256'] == hashlib.sha256(evidence.encode('utf-8')).hexdigest()
    predictions = controls['relation_predictions']
    assert len(predictions) == 4
    for prediction, resolution in zip(predictions, harness.resolutions):
        assert prediction['step'] == resolution['step']
        assert prediction['evidence_sha256'] == hashlib.sha256(resolution['evidence'].encode('utf-8')).hexdigest()
        assert prediction['diagnostics'] == resolution['diagnostics']
    resumed = ProbeBehavior(harness.behavior.path)
    saved = resumed.interpretations[definition_key('secret_probe', harness.probe)][-1]['controls']
    # JSON object keys become strings on disk; compare the JSON-safe fields.
    assert saved['relation_profiles'] == harness.relation_metadata
    assert [row['evidence_sha256'] for row in saved['relation_predictions']] == [row['evidence_sha256'] for row in predictions]
    assert harness.behavior.records == records_before
    assert harness.behavior.evidence('secret_probe', harness.probe, harness.probes) == evidence
    for layer, direction in probe_before['direction'].items():
        torch.testing.assert_close(harness.probe['direction'][layer], direction)
    assert harness.probe['framings'] == probe_before['framings']
    for step, profiles in bank_before.items():
        for layer, profile in profiles.items():
            for name, value in profile.items():
                torch.testing.assert_close(harness.relation_bank[step][layer][name], value)
    context = harness.language.context()
    for operator_text in ('relation_profiles', 'relation_predictions', 'source_anchors', 'layer_scales',
                          'operator-only-shape-profile', 'operator-only-resolution-details',
                          probe_definition_relations.RELATION_VERSION):
        assert operator_text not in context


def test_relational_orientation_and_magnitude_pulse_only_at_original_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .5)
    harness = RelationHarness(monkeypatch, tmp_path, gains={
        'probe_define_step_compare': -.03, 'probe_define_step_isolate': 0.,
        'probe_define_step_distill': .06, 'probe_define_step_verify': -.02})
    harness.run()
    for resolution in harness.resolutions:
        step = resolution['step']
        gain = harness.gains['probe_define_step_' + step]
        scale = resolution['diagnostics']['layer_scales'][4]
        task = gain * scale * resolution['directions'][4]
        expected = torch.tensor([.02, 0., 0.]) + task
        pushes = harness.stage_pushes[step]
        torch.testing.assert_close(pushes['prefill'][0, :2], torch.zeros(2, 3))
        torch.testing.assert_close(pushes['prefill'][0, 2], expected, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(pushes['decode'], torch.zeros_like(pushes['decode']))
        torch.testing.assert_close(pushes['rebuilt'][0, :2], torch.zeros(2, 3))
        torch.testing.assert_close(pushes['rebuilt'][0, 2], expected, atol=1e-7, rtol=1e-5)
        torch.testing.assert_close(pushes['rebuilt'][0, 3:], torch.zeros(2, 3))


def test_dynamic_relation_and_subject_share_live_cap(monkeypatch, tmp_path):
    monkeypatch.setattr(engine, '_steer_cap_fraction', .025)
    harness = RelationHarness(monkeypatch, tmp_path)
    harness.run()
    for resolution in harness.resolutions:
        step = resolution['step']
        direction = resolution['directions'][4]
        scale = resolution['diagnostics']['layer_scales'][4]
        expected = torch.tensor([.02, 0., 0.]) + .04 * scale * direction
        expected *= .025 / expected.norm()
        push = harness.stage_pushes[step]['prefill']
        torch.testing.assert_close(push[0, 2], expected, atol=1e-7, rtol=1e-5)
        assert torch.all(push.norm(dim=-1) <= .025001)


def test_changed_measured_evidence_reconverts_with_existing_profiles(monkeypatch, tmp_path):
    harness = RelationHarness(monkeypatch, tmp_path, outputs={'compare': RECAP})
    harness.run()
    first = harness.resolutions[0]
    # Keep the same measured examples but change one reading. A failed prior
    # attempt does not create a reusable conversion for the changed evidence.
    key = definition_key('secret_probe', harness.probe)
    harness.behavior.records[key][0]['reading'] = -.9
    harness.steps_seen.clear()
    harness.instructions.clear()
    harness.run()
    changed = harness.resolutions[1]
    assert first['evidence'] != changed['evidence']
    assert not torch.allclose(first['directions'][4], changed['directions'][4])
    assert first['diagnostics']['layer_scales'] != changed['diagnostics']['layer_scales']
    assert harness.resolver.call_count == 2
    assert harness.relation_loader.call_count == 2
    harness.bank_loader.assert_not_called()


def test_recap_stops_relational_resolution_at_compare_and_preserves_naming_gate(monkeypatch, tmp_path):
    harness = RelationHarness(monkeypatch, tmp_path, outputs={'compare': RECAP})
    output = harness.run()
    assert harness.steps_seen == ['compare'] and harness.resolver.call_count == 1
    assert 'Stopped at compare' in output
    attempt = harness.attempt()
    assert attempt['status'] == 'unqualified'
    assert len(attempt['controls']['relation_predictions']) == 1
    assert attempt['controls']['pipeline']['completed_steps'] == []
    assert 'threshold=0.65' in harness.language.context()
    harness.naming.assert_not_called()


@pytest.mark.parametrize('failed_step', ['compare', 'distill'])
def test_resolution_failure_before_hook_registration_restores_configuration(monkeypatch, tmp_path, failed_step):
    harness = RelationHarness(monkeypatch, tmp_path)
    harness.resolve_failure = failed_step
    with pytest.raises(RuntimeError, match='relation resolution failed'):
        harness.run()
    completed = list(probe_definition_steps.STEP_ORDER).index(failed_step)
    assert harness.handles.call_count == harness.generate_calls == completed
    assert harness.resolver.call_count == completed + 1
    assert not harness.behavior.interpretations


def test_relational_generation_failure_preserves_unrelated_hooks_and_settings(monkeypatch, tmp_path):
    harness = RelationHarness(monkeypatch, tmp_path, fail_step='isolate')
    unrelated = [harness.root.register_forward_pre_hook(lambda module, args: None),
                 harness.layers[4].register_forward_hook(lambda module, args, output: None)]
    try:
        with pytest.raises(RuntimeError, match='generation failed'):
            harness.run()
        assert len(harness.root._forward_pre_hooks) == len(harness.layers[4]._forward_hooks) == 1
        assert harness.steps_seen == ['compare', 'isolate']
        assert not harness.behavior.interpretations
    finally:
        for handle in unrelated:
            handle.remove()


def test_incomplete_relation_layer_coverage_fails_before_generation_without_leaks(monkeypatch, tmp_path):
    harness = RelationHarness(monkeypatch, tmp_path)
    original = harness.resolve
    def incomplete(model, bank, step, evidence):
        directions, diagnostics = original(model, bank, step, evidence)
        directions.pop(5)
        return directions, diagnostics
    harness.resolver.side_effect = incomplete
    with pytest.raises(ValueError, match='must cover every subject layer'):
        harness.run()
    assert harness.generate_calls == 0 and not harness.behavior.interpretations
