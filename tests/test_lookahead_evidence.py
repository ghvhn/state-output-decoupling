"""Behavioral tests for causal, equally weighted, context-local learning."""
import sys
import tempfile
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invariants.lookahead_predictor import LookaheadApparatus
from invariants.trigger_tuner import TriggerTuner


class TestEvidence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.tuner = TriggerTuner(Path(self.tmp.name)/"tuner.json")
        self.app = LookaheadApparatus(tuner=self.tuner, d_model=8, hidden_dim=16, embedding_matrix=torch.eye(8))
        self.h = 10*torch.eye(8)[0]
        self.d = torch.eye(8)[1]

    def forecast(self, h=None, token=1, k=1):
        h = self.h if h is None else h
        self.app._streams[self.app.current_mode].update(context=h.clone(), last_token=token)
        return self.app.predict_waypoint(h, k)

    def observe(self, h, token=2):
        self.app._streams[self.app.current_mode]["context"] = h.clone()
        return self.app.step_arrival(token, h)

    def row(self, ticket):
        return self.app.evidence.stats(ticket.environment_id, ticket.horizon_k, ticket.metadata["source_token_id"])

    def test_arbitrary_horizons_only_learn_on_actual_arrival(self):
        tickets = self.app.map_future_waypoints(self.h, [5, 2, 1])
        self.assertEqual([t.arrival_step for t in tickets], [5, 2, 1])
        self.assertEqual(self.app.learning_updates, 0)
        for expected in ([1], [2], [], [], [5]):
            self.assertEqual([r.ticket.horizon_k for r in self.observe(self.h)], expected)
        self.assertEqual(self.app.learning_updates, 3)

    def test_hit_and_miss_are_equal_full_weight_observations(self):
        hit = self.forecast()
        hit.predicted_token_id = 2
        self.observe(self.h+self.d)
        miss = self.forecast()
        miss.predicted_token_id = 7
        self.observe(self.h+3*self.d)
        row = self.row(hit)
        self.assertEqual((row["count"], row["matches"], row["misses"]), (2, 1, 1))
        self.assertEqual(row["token_counts"], {2: 2})
        self.assertTrue(torch.equal(row["mean_delta"], 2*self.d))
        ticket = self.forecast()
        self.assertTrue(torch.equal(ticket.predicted_state, self.h+2*self.d))
        self.assertEqual(ticket.predicted_token_id, 2)
        self.assertEqual(ticket.metadata["evidence_weight"], 1.0)
        self.assertLess(float(ticket.dimension_certainty[1]), float(ticket.dimension_certainty[0]))

    def test_local_learning_does_not_leak(self):
        ticket = self.forecast()
        self.observe(self.h+self.d)
        self.assertEqual(self.forecast(self.h+.1*torch.eye(8)[2]).metadata["local_samples"], 1)
        others = [self.forecast(10*torch.eye(8)[3]), self.forecast(token=4), self.forecast(k=2)]
        self.app.set_mode(self.app.MODE_EXTERNAL)
        others.append(self.forecast())
        self.app.set_environment("another task")
        others.append(self.forecast())
        self.assertTrue(all(t.metadata["local_samples"] == 0 for t in others))
        self.assertEqual(self.row(ticket)["count"], 1)

    def test_origin_owns_data_after_context_shift(self):
        ticket = self.forecast()
        self.observe(10*torch.eye(8)[3], 5)
        self.assertEqual(self.row(ticket)["count"], 1)
        self.assertEqual(self.row(ticket)["token_counts"], {5: 1})
        self.assertIsNone(self.app.active_post_mature_delta)

    def test_stream_timelines_are_independent(self):
        ticket = self.forecast()
        self.app.set_mode(self.app.MODE_EXTERNAL)
        self.observe(self.h, 3)
        self.assertEqual(self.app.learning_updates, 0)
        self.app.set_mode(self.app.MODE_SELF)
        self.assertEqual(self.app.current_step, 0)
        self.assertIs(self.observe(self.h)[0].ticket, ticket)

    def test_correction_is_uneven_and_direction_conditional(self):
        for _ in range(3):
            self.forecast()
            self.observe(self.h+self.d)
        self.forecast()
        report = self.observe(self.h+2*self.d)[0]
        self.assertLess(float(report.post_mature_delta[1]), 0)
        self.assertEqual(float(report.post_mature_delta[2]), 0)
        self.assertGreater(float(self.app.get_post_mature_adjustment(16, self.h+3*self.d).norm()), 0)
        self.assertEqual(float(self.app.get_post_mature_adjustment(16, self.h+self.d).norm()), 0)
        self.assertIsNone(self.app.get_post_mature_adjustment(16, 10*torch.eye(8)[5]))
        self.tuner.set("lookahead_alpha_self", 0)
        self.assertIsNone(self.app.get_post_mature_adjustment(16))
        self.assertEqual(self.app.learning_updates, 4)

    def test_claimmap_does_not_override_environment_or_direction(self):
        self.app.attach_claimmap(self.d)
        ticket = self.forecast()
        report = self.observe(self.h+2*self.d)[0]
        self.assertAlmostEqual(report.claimmap_drift, 2.0)
        self.assertEqual(float(report.post_mature_delta.norm()), 0)
        self.assertEqual(self.row(ticket)["count"], 1)

    def test_degree_controls_leave_evidence_weight_unchanged(self):
        self.forecast()
        self.observe(self.h+self.d)
        self.tuner.set("degree_scale_diff", 2)
        ticket = self.forecast()
        self.assertTrue(torch.equal(ticket.predicted_state, self.h+2*self.d))
        self.tuner.set("degree_scale_horizon", 2)
        self.tuner.set("lookahead_default_k", 3)
        self.assertEqual(self.app._horizons(), [12, 6, 3])
        self.tuner.set("degree_scale_factor", 0)
        self.assertEqual(float(self.observe(self.h+3*self.d)[0].post_mature_delta.norm()), 0)
        self.assertEqual(self.row(ticket)["count"], 2)

    def test_terminal_token_counts_without_fabricated_state(self):
        ticket = self.forecast()
        report = self.app.step_arrival(2, None)[0]
        self.assertIsNone(report.actual_state)
        self.assertEqual((self.row(ticket)["count"], self.row(ticket)["state_count"]), (1, 0))

    def test_checkpoint_preserves_learned_forecasts(self):
        self.forecast()
        self.observe(self.h+self.d)
        path = Path(self.tmp.name)/"lookahead.pt"
        self.app.save_checkpoint(path)
        self.app = LookaheadApparatus(tuner=self.tuner, d_model=8, hidden_dim=16, embedding_matrix=torch.eye(8))
        self.app.load_checkpoint(path)
        ticket = self.forecast()
        self.assertEqual(ticket.metadata["local_samples"], 1)
        self.assertEqual(ticket.predicted_token_id, 2)
        self.assertTrue(torch.equal(ticket.predicted_state, self.h+self.d))

    def test_bounded_pending_and_history_keep_summary_evidence(self):
        self.tuner.set("lookahead_max_ledger", 2)
        self.tuner.set("lookahead_max_arrivals", 2)
        self.app.map_future_waypoints(self.h, [5, 4, 3, 2, 1])
        self.assertEqual(sum(map(len, self.app.ledger.values())), 2)
        with self.assertRaises(ValueError):
            self.forecast()
        for _ in range(5):
            self.observe(self.h+self.d)
        for _ in range(5):
            self.forecast()
            self.observe(self.h+self.d)
        self.assertLessEqual(len(self.app.arrivals), 2)
        self.assertGreater(self.app.learning_updates, 2)

    def test_external_ingestion_and_input_validation(self):
        self.assertEqual(len(self.app.ingest_external_sequence([1, 1, 1, 1], horizons=[2, 1])), 5)
        for k in (0, -1, 65, 1.5):
            with self.assertRaises(ValueError):
                self.app.predict_waypoint(self.h, k)
        with self.assertRaises(ValueError):
            self.app.ingest_external_sequence([-1])
        with self.assertRaises(ValueError):
            self.app.ingest_external_sequence([1, 2], hidden_states=[self.h])

    def test_sinusoidal_composition_identity_cancellation_and_interference(self):
        self.app.set_composition("sinusoidal")
        a = .6*self.d
        self.assertTrue(torch.allclose(self.app.combine_corrections([a]), a))
        self.assertTrue(torch.allclose(self.app.combine_corrections([a, -a]), torch.zeros(8)))
        wave = self.app.combine_corrections([a, a])
        expected = torch.sin(2*torch.asin(a))
        self.assertTrue(torch.allclose(wave, expected))
        self.assertLessEqual(float(wave.abs().max()), 1)
        self.assertFalse(torch.allclose(wave, 2*a))
        self.app.set_composition("linear")
        self.assertTrue(torch.equal(self.app.combine_corrections([a, a]), 2*a))
        with self.assertRaises(ValueError):
            self.app.set_composition("unknown")

    def test_large_environment_count_and_safe_observation(self):
        # Verify scaling past 256 environments maintains stable ticket-environment associations
        for i in range(400):
            self.app.evidence.locate(torch.randn(8), mode=self.app.current_mode, namespace="default", create=True)
        self.assertGreater(len(self.app.evidence.environments), 256)
        # Schedule waypoint in existing environment
        ticket = self.forecast()
        env_id = ticket.environment_id
        self.assertIsNotNone(env_id)
        # Add more environments
        for i in range(50):
            self.app.evidence.locate(torch.randn(8), mode=self.app.current_mode, namespace="default", create=True)
        # Arrival must observe into the original environment without IndexError
        arrivals = self.observe(self.h + self.d)
        self.assertEqual(len(arrivals), 1)
        self.assertEqual(self.row(ticket)["count"], 1)
        # Out-of-bounds observation should not raise IndexError
        safe_obs = self.app.evidence.observe(99999, 1, self.h, self.h, 1, True)
        self.assertIsNone(safe_obs)
        self.assertFalse(self.app.evidence.applies(99999, self.h, self.app.current_mode, "default"))
        self.assertIsNone(self.app.evidence.stats(99999, 1))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
