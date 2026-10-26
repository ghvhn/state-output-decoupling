import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM

from invariants.activation_memory import ActivationMemory, ActivationSession, activation_memory_command
from invariants.config import AgenticConfig
from invariants.synaptic_memory import SynapticMemoryEngine


class ActivationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c = torch.eye(3)

    def test_individual_strength_and_disable(self):
        bank = ActivationMemory()
        name = bank.write(self.a, self.b, strength=0.02, origin='thought')
        weak, _ = bank.read(self.a, hops=1, max_norm_ratio=1)
        bank.set_strength(name, 0.04)
        strong, _ = bank.read(self.a, hops=1, max_norm_ratio=1)
        torch.testing.assert_close(strong - self.a, 2 * (weak - self.a))
        bank.set_strength(name, 0)
        silent, stats = bank.read(self.a)
        torch.testing.assert_close(silent, self.a)
        self.assertEqual(stats['matched_memories'], 0)
        for invalid in (-1, float('nan'), float('inf'), 5):
            with self.assertRaises(ValueError):
                bank.set_strength(name, invalid)

    def test_recalled_state_activates_next_association_with_total_budget(self):
        bank = ActivationMemory()
        bank.write(self.a, self.b)
        bank.write(self.b, self.c)
        one, first = bank.read(self.a, hops=1, threshold=0.6)
        two, second = bank.read(self.a, hops=2, threshold=0.6)
        self.assertEqual(float(one[2]), 0)
        self.assertGreater(float(two[2]), 0)
        self.assertEqual(first['hops'], 1)
        self.assertEqual(second['hops'], 2)
        self.assertLessEqual(float((two-self.a).norm()), 0.050001)
        opposite, stats = bank.read(-self.a)
        torch.testing.assert_close(opposite, -self.a)
        self.assertEqual(stats['matched_memories'], 0)
        zero, _ = bank.read(torch.zeros(3))
        torch.testing.assert_close(zero, torch.zeros(3))

    def test_capacity_reload_and_origins_do_not_imply_truth_weights(self):
        bank = ActivationMemory(capacity=2)
        old = bank.write(self.a, self.b, strength=0.1)
        thought = bank.write(self.a, self.c, strength=2, origin='thought', metadata={'correct': False})
        context = bank.write(self.a, self.b, strength=2, origin='context', metadata={'correct': True})
        self.assertNotIn(old, bank.records)
        restored = ActivationMemory.from_state_dict(bank.state_dict())
        self.assertEqual(restored.records[thought]['strength'], restored.records[context]['strength'])
        torch.testing.assert_close(restored.read(self.a)[0], bank.read(self.a)[0], rtol=0, atol=0)
        dump = bank.state_dict()
        dump['records'][thought]['value'].zero_()
        with self.assertRaises(ValueError):
            ActivationMemory.from_state_dict(dump)

    def test_recording_skips_reprefill_and_keeps_thought_origin(self):
        session = ActivationSession(max_writes_per_forward=8, strength=0.3)
        bank = ActivationMemory()
        session.begin_segment(2)
        session.observe(0, bank, torch.stack([self.a, self.b]).unsqueeze(0))
        session.observe(0, bank, self.c.view(1, 1, 3))
        session.mark_generated(2, 4)
        session.begin_segment(4)
        session.observe(0, bank, torch.stack([self.a, self.b, self.c, self.a]).unsqueeze(0))
        records = list(bank.records.values())
        self.assertEqual(len(records), 4)
        self.assertEqual([r['origin'] for r in records], ['context', 'context', 'thought', 'thought'])
        self.assertEqual([r['metadata']['token_position'] for r in records], [0, 1, 2, 3])
        self.assertTrue(all(r['strength'] == 0.3 for r in records))
        torch.testing.assert_close(records[2]['key'], self.b)
        torch.testing.assert_close(records[2]['value'], self.c)
        self.assertEqual(records[2]['metadata']['predecessor_id'], list(bank.records)[1])

    def test_episode_binding_avoids_mixing_competing_experiences(self):
        bank = ActivationMemory()
        bank.write(self.a, self.b, strength=1, metadata={'session_id': 'episode-a'})
        bank.write(self.a, self.c, strength=0.9, metadata={'session_id': 'episode-b'})
        bound, stats = bank.read(self.a, hops=1, episode_binding=True)
        mixed, _ = bank.read(self.a, hops=1, episode_binding=False)
        self.assertEqual(float(bound[2]), 0)
        self.assertGreater(float(mixed[2]), 0)
        self.assertEqual(stats['episode_locked_positions'], 1)
        self.assertEqual(stats['matched_memories'], 1)

    def test_links_reactivate_a_later_state_beyond_geometric_match_and_reload(self):
        bank = ActivationMemory()
        first = bank.write(self.a, self.b, metadata={'session_id': 'episode-a'})
        later = bank.write(-self.a, self.c, metadata={'session_id': 'episode-a', 'predecessor_id': first})
        geometric, _ = bank.read(self.a, hops=2, threshold=0.6, link_gain=0)
        linked, stats = bank.read(self.a, hops=2, threshold=0.6, link_gain=0.5)
        self.assertEqual(float(geometric[2]), 0)
        self.assertGreater(float(linked[2]), 0)
        self.assertEqual(stats['linked_matches'], 1)
        self.assertLessEqual(float((linked-self.a).norm()), 0.050001)
        restored = ActivationMemory.from_state_dict(bank.state_dict())
        torch.testing.assert_close(restored.read(self.a, hops=2, threshold=0.6)[0], linked, rtol=0, atol=0)
        bank.set_strength(later, 0)
        self.assertEqual(float(bank.read(self.a, hops=2, threshold=0.6)[0][2]), 0)
        # Dangling links after capacity eviction must also remain harmless.
        bank.set_capacity(1)
        bank.read(self.a, hops=2)

    def test_cross_episode_link_is_not_followed(self):
        bank = ActivationMemory()
        first = bank.write(self.a, self.b, metadata={'session_id': 'episode-a'})
        bank.write(-self.a, self.c, metadata={'session_id': 'episode-b', 'predecessor_id': first})
        recalled, stats = bank.read(self.a, hops=2, threshold=0.6, episode_binding=False)
        self.assertEqual(float(recalled[2]), 0)
        self.assertEqual(stats['linked_matches'], 0)

    def test_old_checkpoint_recovers_ordered_episode_links(self):
        bank = ActivationMemory()
        first = bank.write(self.a, self.b, metadata={'session_id': 'old', 'token_position': 10})
        later = bank.write(-self.a, self.c, metadata={'session_id': 'old', 'token_position': 11})
        restored = ActivationMemory.from_state_dict(bank.state_dict())
        self.assertEqual(restored.records[later]['metadata']['predecessor_id'], first)
        self.assertGreater(float(restored.read(self.a, hops=2, threshold=0.6)[0][2]), 0)


class ActivationIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(24)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        previous = SynapticMemoryEngine._INSTANCE
        self.addCleanup(setattr, SynapticMemoryEngine, '_INSTANCE', previous)
        raw = LlamaForCausalLM(LlamaConfig(
            vocab_size=24, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )).eval()
        raw.generation_config.suppress_tokens = [0, 1, 2]
        self.model = SimpleNamespace(model=raw, tok=SimpleNamespace(
            eos_token_id=2, decode=lambda tokens, **kw: ''.join(chr(97 + int(t)) for t in tokens)),
            device=torch.device('cpu'), d_model=16, n_layers=2)
        self.owner = SynapticMemoryEngine(self.model, target_layers=[0, 1])
        SynapticMemoryEngine._INSTANCE = self.owner
        self.addCleanup(self.owner.detach)
        self.ids = torch.tensor([[1, 4, 5]])

    def test_real_generation_records_and_reuses_states_without_extra_tokens(self):
        from invariants.agentic_engine import generate_agentic_text
        cfg = AgenticConfig(activation_memory_autosave=False, activation_memory_threshold=0)
        parameters = {k: v.clone() for k, v in self.model.model.state_dict().items()}
        prompts = []
        snapshots = []
        original = self.model.model.generate

        def tracked(**kwargs):
            prompts.append(kwargs['input_ids'].clone())
            return original(**kwargs)

        with patch('invariants.agentic_engine._inputs', return_value={
            'input_ids': self.ids, 'attention_mask': torch.ones_like(self.ids)}), patch(
                'invariants.agentic_engine.get_agentic_handles', return_value=([], {'steer_channels': {}})), patch.object(
                self.model.model, 'generate', side_effect=tracked):
            _, report = generate_agentic_text(self.model, instruction='cue', config=cfg,
                                              max_new_tokens=5, return_telemetry=True,
                                              generation_channel='speech',
                                              generation_probes={'test': {0: torch.ones(16)}},
                                              generation_state_sink=snapshots.append)
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].shape, (2, 1, 16))
            self.assertEqual(report['generation_probe_readings']['test']['token_count'], 5)
            self.assertGreater(report['activation_memory']['writes'], 0)
            origins = {r['origin'] for h in self.owner.hooks.values() for r in h.activation_memory.records.values()}
            self.assertEqual(origins, {'thought', 'context'})
            cfg.activation_memory_write_enabled = False
            _, read_report = generate_agentic_text(self.model, instruction='cue', config=cfg,
                                                   max_new_tokens=5, return_telemetry=True)
        self.assertGreater(read_report['activation_memory']['matched_memories'], 0)
        self.assertEqual(read_report['activation_memory']['writes'], 0)
        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            torch.testing.assert_close(prompt, self.ids)
        for key, value in self.model.model.state_dict().items():
            self.assertTrue(torch.equal(value, parameters[key]), key)
        self.assertIsNone(self.owner._activation_session())
        status = activation_memory_command('status', self.owner, cfg)
        self.assertIn('last_active_session=', status)
        self.assertGreater(self.owner.last_activation_stats['matched_memories'], 0)

    def test_ownership_auxiliary_exclusion_checkpoint_and_strength_command(self):
        from invariants.memory_attention import auxiliary_encoding
        other = SynapticMemoryEngine(self.model, target_layers=[0, 1])
        self.addCleanup(other.detach)
        with self.owner.activation_generation() as session:
            session.begin_segment(3)
            with auxiliary_encoding():
                self.model.model(self.ids)
            self.assertEqual(session.stats['writes'], 0)
            self.model.model(self.ids)
        self.assertGreater(session.stats['writes'], 0)
        self.assertFalse(any(h.activation_memory.records for h in other.hooks.values()))
        with tempfile.TemporaryDirectory() as directory:
            self.owner.persistence_path = Path(directory) / 'memory.pt'
            cfg = AgenticConfig()
            name = next(iter(self.owner.hooks[0].activation_memory.records))
            activation_memory_command(f'strength {name} 0.75', self.owner, cfg)
            self.assertTrue(other.load_synaptic_weights(self.owner.persistence_path))
            self.assertEqual(other.hooks[0].activation_memory.records[name]['strength'], 0.75)
            with self.owner.activation_generation(write_enabled=False) as s:
                s.begin_segment(3)
                expected = self.model.model(self.ids).logits
            with other.activate(), other.activation_generation(write_enabled=False) as s:
                s.begin_segment(3)
                actual = self.model.model(self.ids).logits
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        activation_memory_command('set hops 3', self.owner, cfg)
        self.assertEqual(cfg.activation_memory_hops, 3)
        with self.assertRaises(ValueError):
            activation_memory_command('set threshold 1', self.owner, cfg)
        self.assertEqual(cfg.activation_memory_threshold, 0.5)

    def test_chunk_reprefill_and_generation_failure_cleanup(self):
        from invariants.agentic_engine import generate_agentic_text
        cfg = AgenticConfig(activation_memory_autosave=False)
        with patch('invariants.agentic_engine._inputs', return_value={
            'input_ids': self.ids, 'attention_mask': torch.ones_like(self.ids)}), patch(
                'invariants.agentic_engine.get_agentic_handles', return_value=([], {'steer_channels': {}})):
            generate_agentic_text(self.model, config=cfg, max_new_tokens=27)
            for hook in self.owner.hooks.values():
                records = list(hook.activation_memory.records.values())
                positions = [r['metadata']['token_position'] for r in records]
                self.assertEqual(len(positions), len(set(positions)))
                self.assertEqual(sorted(positions), list(range(29)))
                self.assertTrue(all(r['origin'] == ('context' if r['metadata']['token_position'] < 3 else 'thought') for r in records))
            with patch.object(self.model.model, 'generate', side_effect=RuntimeError('failed generation')):
                with self.assertRaisesRegex(RuntimeError, 'failed generation'):
                    generate_agentic_text(self.model, config=cfg, max_new_tokens=2)
            self.assertIsNone(self.owner._activation_session())
            counts = self.owner.telemetry()['activation_memories']
            cfg.activation_memory_enabled = False
            _, report = generate_agentic_text(self.model, config=cfg, max_new_tokens=2, return_telemetry=True)
            self.assertEqual(report['activation_memory'], {})
            self.assertEqual(counts, self.owner.telemetry()['activation_memories'])


if __name__ == '__main__':
    unittest.main()
