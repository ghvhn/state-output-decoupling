"""Tests for Hebbian plasticity field geometry.

Covers:
  - Core update: co-activation builds LTP/LTD correctly
  - Outcome gating: sign of outcome determines LTP vs LTD
  - Signal product: anti-correlated probes on good turn → LTD
  - Symmetry: key order independent
  - Clamping: potentiation never exceeds configured bounds
  - Decay: exponential per-turn decay
  - Min-signal filter: weak signals don't participate
  - Distance scale: LTP reduces d, LTD increases d
  - Mass multiplier: LTP boosts, LTD reduces
  - Enable/disable: disabled matrix returns neutral values
  - Reset: clears pairs correctly
  - Persistence: save/load round-trip
  - Shell handler: subcommands dispatch correctly
  - Field integration: distance_scale * mass_multiplier compose
"""

import json
import math
import os
import tempfile
import unittest

from invariants.hebbian_plasticity import HebbianPlasticityMatrix, handle_plasticity_command


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh(path=None, **kwargs) -> HebbianPlasticityMatrix:
    return HebbianPlasticityMatrix(path=path, **kwargs)


def one_update(mat, sig_a, sig_b, outcome, a="alpha", b="beta"):
    mat.update({a: sig_a, b: sig_b}, outcome)


# ---------------------------------------------------------------------------
# Update mechanics
# ---------------------------------------------------------------------------

class TestUpdate(unittest.TestCase):

    def test_co_firing_same_direction_positive_outcome_ltp(self):
        mat = fresh()
        one_update(mat, 0.5, 0.4, outcome=1.0)
        p = mat.potentiation("alpha", "beta")
        self.assertGreater(p, 0.0, "same-direction co-fire + good outcome → LTP")

    def test_co_firing_opposite_direction_positive_outcome_ltd(self):
        mat = fresh()
        one_update(mat, 0.5, -0.4, outcome=1.0)
        p = mat.potentiation("alpha", "beta")
        self.assertLess(p, 0.0, "opposed probes on good turn → LTD (should diverge)")

    def test_positive_outcome_negative_signals_still_ltp(self):
        # Both negative signals → product positive → LTP on good outcome
        mat = fresh()
        one_update(mat, -0.5, -0.4, outcome=1.0)
        p = mat.potentiation("alpha", "beta")
        self.assertGreater(p, 0.0)

    def test_negative_outcome_co_firing_same_direction_ltd(self):
        mat = fresh()
        one_update(mat, 0.5, 0.4, outcome=-1.0)
        p = mat.potentiation("alpha", "beta")
        self.assertLess(p, 0.0, "co-fire on bad turn → LTD")

    def test_magnitude_proportional_to_signal_product(self):
        mat_weak = fresh(learning_rate=0.1)
        mat_strong = fresh(learning_rate=0.1)
        mat_weak.update({"a": 0.1, "b": 0.1}, outcome=1.0)
        mat_strong.update({"a": 0.9, "b": 0.9}, outcome=1.0)
        self.assertGreater(
            mat_strong.potentiation("a", "b"),
            mat_weak.potentiation("a", "b"),
            "stronger signals build faster bonds",
        )

    def test_zero_signal_skipped(self):
        mat = fresh()
        mat.update({"a": 0.0, "b": 0.5}, outcome=1.0)
        self.assertEqual(mat.potentiation("a", "b"), 0.0)

    def test_below_min_signal_skipped(self):
        mat = fresh(min_signal=0.05)
        mat.update({"a": 0.04, "b": 0.5}, outcome=1.0)
        self.assertEqual(mat.potentiation("a", "b"), 0.0)

    def test_accumulation_across_turns(self):
        mat = fresh(learning_rate=0.1)
        for _ in range(5):
            one_update(mat, 0.5, 0.5, outcome=1.0)
        p5 = mat.potentiation("alpha", "beta")
        mat2 = fresh(learning_rate=0.1)
        one_update(mat2, 0.5, 0.5, outcome=1.0)
        p1 = mat2.potentiation("alpha", "beta")
        self.assertGreater(p5, p1, "repeated co-activation accumulates")

    def test_update_returns_pair_count(self):
        mat = fresh()
        n = mat.update({"a": 0.5, "b": 0.3, "c": 0.4}, outcome=1.0)
        # 3 probes → 3 pairs
        self.assertEqual(n, 3)

    def test_three_probe_all_pairs_updated(self):
        mat = fresh(learning_rate=0.2)
        mat.update({"a": 0.5, "b": 0.5, "c": 0.5}, outcome=1.0)
        self.assertGreater(mat.potentiation("a", "b"), 0.0)
        self.assertGreater(mat.potentiation("a", "c"), 0.0)
        self.assertGreater(mat.potentiation("b", "c"), 0.0)


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

class TestSymmetry(unittest.TestCase):

    def test_key_order_independent(self):
        mat = fresh()
        one_update(mat, 0.5, 0.4, outcome=1.0, a="alpha", b="beta")
        self.assertEqual(
            mat.potentiation("alpha", "beta"),
            mat.potentiation("beta", "alpha"),
        )

    def test_reversed_update_same_result(self):
        mat = fresh()
        mat.update({"b": 0.4, "a": 0.5}, outcome=1.0)
        self.assertGreater(mat.potentiation("a", "b"), 0.0)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------

class TestClamping(unittest.TestCase):

    def test_ltp_capped_at_max_potentiation(self):
        mat = fresh(learning_rate=1.0, max_potentiation=0.5, decay_rate=0.0)
        for _ in range(100):
            one_update(mat, 1.0, 1.0, outcome=1.0)
        p = mat.potentiation("alpha", "beta")
        self.assertLessEqual(p, 0.5 + 1e-9)

    def test_ltd_capped_at_max_depression(self):
        mat = fresh(learning_rate=1.0, max_depression=0.3, decay_rate=0.0)
        for _ in range(100):
            one_update(mat, 1.0, 1.0, outcome=-1.0)
        p = mat.potentiation("alpha", "beta")
        self.assertGreaterEqual(p, -0.3 - 1e-9)


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

class TestDecay(unittest.TestCase):

    def test_decay_reduces_potentiation(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.1)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        p_before = mat.potentiation("alpha", "beta")
        mat.decay()
        p_after = mat.potentiation("alpha", "beta")
        self.assertLess(p_after, p_before)

    def test_decay_factor_correct(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.1)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        p = mat.potentiation("alpha", "beta")
        mat.decay()
        p2 = mat.potentiation("alpha", "beta")
        self.assertAlmostEqual(p2, p * 0.9, places=9)

    def test_decay_prunes_near_zero(self):
        mat = fresh(learning_rate=0.01, decay_rate=0.5)
        one_update(mat, 0.1, 0.1, outcome=1.0)
        for _ in range(100):
            mat.decay()
        self.assertEqual(len(mat._cofire), 0, "near-zero entries pruned after many decays")

    def test_zero_decay_rate_no_change(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        p = mat.potentiation("alpha", "beta")
        mat.decay()
        self.assertEqual(mat.potentiation("alpha", "beta"), p)


# ---------------------------------------------------------------------------
# Distance scale
# ---------------------------------------------------------------------------

class TestDistanceScale(unittest.TestCase):

    def test_no_history_returns_one(self):
        mat = fresh()
        scale = mat.distance_scale("alpha", {"beta": 0.5})
        self.assertAlmostEqual(scale, 1.0)

    def test_ltp_reduces_distance(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        # alpha and beta have strong LTP; when beta is firing, alpha feels closer
        scale = mat.distance_scale("alpha", {"beta": 1.0})
        self.assertLess(scale, 1.0, "LTP reduces effective distance")

    def test_ltd_increases_distance(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=-1.0)
        scale = mat.distance_scale("alpha", {"beta": 1.0})
        self.assertGreater(scale, 1.0, "LTD increases effective distance")

    def test_scale_bounded(self):
        mat = fresh(learning_rate=2.0, decay_rate=0.0, max_potentiation=0.7)
        for _ in range(50):
            one_update(mat, 1.0, 1.0, outcome=1.0)
        scale = mat.distance_scale("alpha", {"beta": 1.0})
        self.assertGreaterEqual(scale, 0.2)
        self.assertLessEqual(scale, 2.0)

    def test_self_excluded(self):
        # The probe being scored should not gate off its own signal
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        scale_with_self = mat.distance_scale("alpha", {"alpha": 1.0, "beta": 1.0})
        scale_without_self = mat.distance_scale("alpha", {"beta": 1.0})
        # alpha gating itself is excluded; beta gating is the same
        self.assertAlmostEqual(scale_with_self, scale_without_self, places=9)

    def test_only_positive_signals_gate(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        scale_pos = mat.distance_scale("alpha", {"beta": 1.0})
        scale_neg = mat.distance_scale("alpha", {"beta": -1.0})
        # Negative signal from beta doesn't gate alpha's distance
        self.assertAlmostEqual(scale_neg, 1.0, places=9)
        self.assertLess(scale_pos, 1.0)


# ---------------------------------------------------------------------------
# Mass multiplier
# ---------------------------------------------------------------------------

class TestMassMultiplier(unittest.TestCase):

    def test_no_history_returns_one(self):
        mat = fresh()
        mult = mat.mass_multiplier("alpha", {"beta": 0.5})
        self.assertAlmostEqual(mult, 1.0)

    def test_ltp_boosts_mass(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        mult = mat.mass_multiplier("alpha", {"beta": 1.0})
        self.assertGreater(mult, 1.0, "LTP boosts effective mass")

    def test_ltd_reduces_mass(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=-1.0)
        mult = mat.mass_multiplier("alpha", {"beta": 1.0})
        self.assertLess(mult, 1.0, "LTD reduces effective mass")

    def test_multiplier_bounded(self):
        mat = fresh(learning_rate=2.0, decay_rate=0.0)
        for _ in range(50):
            one_update(mat, 1.0, 1.0, outcome=1.0)
        mult = mat.mass_multiplier("alpha", {"beta": 1.0})
        self.assertGreaterEqual(mult, 0.1)
        self.assertLessEqual(mult, 3.0)


# ---------------------------------------------------------------------------
# Disabled matrix
# ---------------------------------------------------------------------------

class TestDisabled(unittest.TestCase):

    def test_disabled_distance_scale_neutral(self):
        mat = fresh(enabled=False)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        self.assertAlmostEqual(mat.distance_scale("alpha", {"beta": 1.0}), 1.0)

    def test_disabled_mass_multiplier_neutral(self):
        mat = fresh(enabled=False)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        self.assertAlmostEqual(mat.mass_multiplier("alpha", {"beta": 1.0}), 1.0)

    def test_disabled_update_no_op(self):
        mat = fresh(enabled=False)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        # No pairs should have been recorded while disabled
        self.assertEqual(len(mat._cofire), 0)

    def test_toggle_on_off(self):
        mat = fresh(enabled=True)
        one_update(mat, 0.5, 0.5, outcome=1.0)
        p_on = mat.potentiation("alpha", "beta")
        self.assertGreater(p_on, 0.0)
        mat.enabled = False
        mat.reset()
        one_update(mat, 1.0, 1.0, outcome=1.0)
        self.assertEqual(mat.potentiation("alpha", "beta"), 0.0)
        mat.enabled = True
        one_update(mat, 0.5, 0.5, outcome=1.0)
        self.assertGreater(mat.potentiation("alpha", "beta"), 0.0)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset(unittest.TestCase):

    def test_reset_all(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        mat.update({"a": 0.5, "b": 0.5, "c": 0.5}, outcome=1.0)
        removed = mat.reset()
        self.assertEqual(removed, 3)
        self.assertEqual(len(mat._cofire), 0)

    def test_reset_specific_pair(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        mat.update({"a": 0.5, "b": 0.5, "c": 0.5}, outcome=1.0)
        mat.reset("a", "b")
        self.assertEqual(mat.potentiation("a", "b"), 0.0)
        self.assertGreater(mat.potentiation("a", "c"), 0.0)
        self.assertGreater(mat.potentiation("b", "c"), 0.0)

    def test_reset_nonexistent_pair_safe(self):
        mat = fresh()
        removed = mat.reset("x", "y")
        self.assertEqual(removed, 0)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence(unittest.TestCase):

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "plasticity.json")
            mat = fresh(path=path, learning_rate=0.3, decay_rate=0.01)
            one_update(mat, 0.8, 0.7, outcome=0.9)
            one_update(mat, 0.6, -0.5, outcome=0.8)
            p_ab = mat.potentiation("alpha", "beta")
            mat.save()

            mat2 = fresh(path=path)
            self.assertAlmostEqual(mat2.potentiation("alpha", "beta"), p_ab, places=9)
            self.assertAlmostEqual(mat2.learning_rate, 0.3, places=9)
            self.assertAlmostEqual(mat2.decay_rate, 0.01, places=9)

    def test_load_missing_file_starts_fresh(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nonexistent.json")
            mat = fresh(path=path)
            self.assertEqual(len(mat._cofire), 0)
            self.assertEqual(mat._turn, 0)

    def test_save_fails_gracefully_without_path(self):
        mat = fresh(path=None)
        ok = mat.save()
        self.assertFalse(ok)

    def test_atomic_write_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "plasticity.json")
            mat = fresh(path=path)
            one_update(mat, 0.5, 0.5, outcome=1.0)
            mat.save()
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertIn("cofire", payload)
            self.assertIn("knobs", payload)
            self.assertIn("version", payload)


# ---------------------------------------------------------------------------
# Knob setting
# ---------------------------------------------------------------------------

class TestKnobs(unittest.TestCase):

    def test_set_known_knob(self):
        mat = fresh()
        ok = mat.set_knob("learning_rate", 0.99)
        self.assertTrue(ok)
        self.assertAlmostEqual(mat.learning_rate, 0.99)

    def test_set_unknown_knob_returns_false(self):
        mat = fresh()
        ok = mat.set_knob("nonexistent_knob", 1.0)
        self.assertFalse(ok)


# ---------------------------------------------------------------------------
# Status and top_pairs
# ---------------------------------------------------------------------------

class TestStatus(unittest.TestCase):

    def test_status_empty(self):
        mat = fresh()
        s = mat.status()
        self.assertEqual(s["pairs"], 0)
        self.assertEqual(s["ltp_pairs"], 0)
        self.assertEqual(s["ltd_pairs"], 0)

    def test_status_counts_ltp_ltd(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        mat.update({"a": 0.5, "b": 0.5, "c": -0.5}, outcome=1.0)
        s = mat.status()
        # a↔b: positive product + good outcome → LTP
        # a↔c: negative product + good outcome → LTD
        # b↔c: negative product + good outcome → LTD
        self.assertGreaterEqual(s["ltp_pairs"], 1)
        self.assertGreaterEqual(s["ltd_pairs"], 1)

    def test_top_pairs_sorted_by_abs(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        mat.update({"a": 1.0, "b": 1.0, "c": 0.1}, outcome=1.0)
        top = mat.top_pairs(10)
        abs_vals = [abs(p["potentiation"]) for p in top]
        self.assertEqual(abs_vals, sorted(abs_vals, reverse=True))

    def test_pair_info_neutral(self):
        mat = fresh()
        info = mat.pair_info("x", "y")
        self.assertEqual(info["kind"], "neutral")
        self.assertEqual(info["potentiation"], 0.0)

    def test_format_status_string(self):
        mat = fresh()
        s = mat.format_status()
        self.assertIn("Plasticity", s)
        self.assertIn("lr=", s)

    def test_history_recorded(self):
        mat = fresh()
        mat.update({"a": 0.5, "b": 0.5}, outcome=0.3)
        h = mat.recent_history(1)
        self.assertEqual(len(h), 1)
        self.assertAlmostEqual(h[0]["outcome"], 0.3)


# ---------------------------------------------------------------------------
# Shell command handler
# ---------------------------------------------------------------------------

class TestShellHandler(unittest.TestCase):

    def _mat_with_bond(self):
        mat = fresh(learning_rate=0.5, decay_rate=0.0)
        one_update(mat, 1.0, 1.0, outcome=1.0)
        return mat

    def test_status_command(self):
        mat = self._mat_with_bond()
        result = handle_plasticity_command([], mat, {})
        self.assertIn("Plasticity", result)

    def test_top_command(self):
        mat = self._mat_with_bond()
        result = handle_plasticity_command(["top", "5"], mat, {})
        self.assertIn("alpha", result)
        self.assertIn("beta", result)

    def test_pair_command(self):
        mat = self._mat_with_bond()
        result = handle_plasticity_command(["pair", "alpha", "beta"], mat, {})
        self.assertIn("potentiation", result)

    def test_pair_command_missing_args(self):
        mat = fresh()
        with self.assertRaises(ValueError):
            handle_plasticity_command(["pair", "alpha"], mat, {})

    def test_history_command(self):
        mat = self._mat_with_bond()
        result = handle_plasticity_command(["history", "5"], mat, {})
        self.assertIn("turn=", result)

    def test_reset_all(self):
        mat = self._mat_with_bond()
        result = handle_plasticity_command(["reset"], mat, {})
        self.assertIn("Cleared", result)
        self.assertEqual(len(mat._cofire), 0)

    def test_reset_pair(self):
        mat = self._mat_with_bond()
        result = handle_plasticity_command(["reset", "alpha", "beta"], mat, {})
        self.assertIn("Cleared", result)
        self.assertEqual(mat.potentiation("alpha", "beta"), 0.0)

    def test_on_off(self):
        mat = fresh(enabled=True)
        handle_plasticity_command(["off"], mat, {})
        self.assertFalse(mat.enabled)
        handle_plasticity_command(["on"], mat, {})
        self.assertTrue(mat.enabled)

    def test_set_command(self):
        mat = fresh()
        handle_plasticity_command(["set", "learning_rate", "0.77"], mat, {})
        self.assertAlmostEqual(mat.learning_rate, 0.77)

    def test_set_bad_knob(self):
        mat = fresh()
        with self.assertRaises(ValueError):
            handle_plasticity_command(["set", "bad_knob", "1.0"], mat, {})

    def test_set_bad_value(self):
        mat = fresh()
        with self.assertRaises(ValueError):
            handle_plasticity_command(["set", "learning_rate", "notanumber"], mat, {})

    def test_unknown_subcommand(self):
        mat = fresh()
        with self.assertRaises(ValueError):
            handle_plasticity_command(["frobnicate"], mat, {})

    def test_save_command_no_path(self):
        mat = fresh(path=None)
        result = handle_plasticity_command(["save"], mat, {})
        self.assertIn("failed", result.lower())

    def test_save_command_with_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "p.json")
            mat = fresh(path=path)
            one_update(mat, 0.5, 0.5, outcome=1.0)
            result = handle_plasticity_command(["save"], mat, {})
            self.assertIn("Saved", result)
            self.assertTrue(os.path.isfile(path))


# ---------------------------------------------------------------------------
# Field integration: composed distance and mass effects
# ---------------------------------------------------------------------------

class TestFieldIntegration(unittest.TestCase):
    """Verify that distance_scale and mass_multiplier compose as expected
    and that the combined effect on the force law is monotone in potentiation."""

    def _force(self, distance, mass, scale, mult, eps=0.05):
        """Simulate _gravity_kernel force for a point mass."""
        d_eff = max(distance * scale, 0.0)
        m_eff = mass * mult
        return m_eff / (d_eff + eps) ** 2

    def test_ltp_increases_force_at_medium_distance(self):
        """High potentiation → smaller d, higher mass → stronger pull."""
        mat_ltp = fresh(learning_rate=0.5, decay_rate=0.0)
        mat_none = fresh(learning_rate=0.0, decay_rate=0.0)

        one_update(mat_ltp, 1.0, 1.0, outcome=1.0)

        signals = {"beta": 1.0}
        d_base = 0.3
        mass = 0.5

        scale_none = mat_none.distance_scale("alpha", signals)
        mult_none = mat_none.mass_multiplier("alpha", signals)
        scale_ltp = mat_ltp.distance_scale("alpha", signals)
        mult_ltp = mat_ltp.mass_multiplier("alpha", signals)

        f_none = self._force(d_base, mass, scale_none, mult_none)
        f_ltp = self._force(d_base, mass, scale_ltp, mult_ltp)

        self.assertGreater(f_ltp, f_none, "LTP increases total force")

    def test_ltd_decreases_force_at_medium_distance(self):
        """LTD → larger d, smaller mass → weaker pull."""
        mat_ltd = fresh(learning_rate=0.5, decay_rate=0.0)
        mat_none = fresh(learning_rate=0.0, decay_rate=0.0)

        one_update(mat_ltd, 1.0, 1.0, outcome=-1.0)

        signals = {"beta": 1.0}
        d_base = 0.3
        mass = 0.5

        scale_none = mat_none.distance_scale("alpha", signals)
        mult_none = mat_none.mass_multiplier("alpha", signals)
        scale_ltd = mat_ltd.distance_scale("alpha", signals)
        mult_ltd = mat_ltd.mass_multiplier("alpha", signals)

        f_none = self._force(d_base, mass, scale_none, mult_none)
        f_ltd = self._force(d_base, mass, scale_ltd, mult_ltd)

        self.assertLess(f_ltd, f_none, "LTD decreases total force")

    def test_force_monotone_in_potentiation_strength(self):
        """More positive potentiation → more force. Monotone across 5 steps."""
        forces = []
        for n_updates in range(0, 6):
            mat = fresh(learning_rate=0.15, decay_rate=0.0)
            for _ in range(n_updates):
                one_update(mat, 1.0, 1.0, outcome=1.0)
            scale = mat.distance_scale("alpha", {"beta": 1.0})
            mult = mat.mass_multiplier("alpha", {"beta": 1.0})
            forces.append(self._force(0.3, 0.5, scale, mult))
        for i in range(len(forces) - 1):
            self.assertLessEqual(forces[i], forces[i + 1])


if __name__ == "__main__":
    unittest.main()
