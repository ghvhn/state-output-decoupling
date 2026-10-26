"""Scored field learning persists per owner and changes real field forces."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from invariants.hebbian_plasticity import PlasticityOwners, HebbianPlasticityMatrix, handle_plasticity_command
from scripts.interactive_phenomenality import _gravity_pull, KNOWN_COMMANDS


class PlasticityLearningTests(unittest.TestCase):
    def test_shell_controls_persist_in_existing_owner_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            main = SimpleNamespace(path=Path(directory) / 'main.json')
            agent = SimpleNamespace(path=Path(directory) / 'agent.json')
            owners = PlasticityOwners()
            matrix, _signals = owners.for_tuner(main)
            self.assertIsInstance(matrix, HebbianPlasticityMatrix)
            self.assertEqual(matrix.path, (Path(directory) / 'main.plasticity.json').resolve())
            settings = {'learning_rate': .05, 'decay_rate': .005, 'strength': .40,
                        'mass_strength': .50, 'max_potentiation': .70, 'max_depression': .40}
            handle_plasticity_command(['on'], matrix, {})
            for key, value in settings.items():
                handle_plasticity_command(['set', key, str(value)], matrix, {})
                self.assertTrue(matrix.save())
            handle_plasticity_command(['off'], matrix, {})
            self.assertTrue(matrix.save())
            resumed = PlasticityOwners().for_tuner(main)[0]
            self.assertFalse(resumed.enabled)
            for key, value in settings.items():
                self.assertEqual(getattr(resumed, key), value)
            self.assertNotEqual(owners.for_tuner(agent)[0].path, matrix.path)
            self.assertIn('[Plasticity]', handle_plasticity_command([], resumed, {}))
            self.assertIn('Saved', handle_plasticity_command(['save'], resumed, {}))

    def test_owners_resume_separately_and_missing_outcome_never_learns(self):
        with tempfile.TemporaryDirectory() as directory:
            main = SimpleNamespace(path=Path(directory) / "main.json")
            agent = SimpleNamespace(path=Path(directory) / "agent.json")
            owners = PlasticityOwners()
            matrix, signals = owners.for_tuner(main)
            owners.observe(main, {"probe_a": 1., "probe_b": 1., "sense": 1.}, None)
            self.assertEqual(matrix.status()["pairs"], 0)
            self.assertEqual(signals, {"a": 1., "b": 1.})
            owners.observe(main, {"probe_a": 1., "probe_b": 1.}, 1.)
            self.assertGreater(matrix.potentiation("a", "b"), 0)
            self.assertEqual(owners.for_tuner(agent)[0].status()["pairs"], 0)
            resumed = PlasticityOwners().for_tuner(main)[0]
            self.assertEqual(resumed.potentiation("a", "b"), matrix.potentiation("a", "b"))
            owners.observe(main, {"probe_a": float("nan")}, None)
            self.assertEqual(signals, {})

    def test_learned_distance_changes_production_field_and_off_restores_it(self):
        matrix = HebbianPlasticityMatrix(learning_rate=.5, mass_strength=0)
        matrix.update({"a": 1., "b": 1.}, outcome=1.)
        entries = [{"direction": torch.tensor([0., 1.]), "mass": .05,
                    "g": .1, "name": "a"}]
        state = torch.tensor([[1., 0.]])
        neutral = _gravity_pull(state, entries, 1.)
        adjusted = _gravity_pull(state, entries, 1., plasticity=matrix,
                                 plasticity_signals={"b": 1.})
        self.assertGreater(float(adjusted.norm()), float(neutral.norm()))
        matrix.enabled = False
        torch.testing.assert_close(_gravity_pull(state, entries, 1., plasticity=matrix,
                                                 plasticity_signals={"b": 1.}), neutral)

    def test_controls_reject_nonfinite_or_unstable_knobs(self):
        matrix = HebbianPlasticityMatrix()
        for knob, value in [("strength", float("nan")), ("learning_rate", -1.),
                            ("decay_rate", 1.1), ("strength", float("inf"))]:
            with self.assertRaises(ValueError):
                matrix.set_knob(knob, value)
        self.assertIn(":plasticity", KNOWN_COMMANDS)

    def test_observe_ignores_dictionary_probe_metadata_in_turn_row(self):
        with tempfile.TemporaryDirectory() as directory:
            main = SimpleNamespace(path=Path(directory) / "main.json")
            owners = PlasticityOwners()
            matrix, signals = owners.for_tuner(main)
            turn_row = {
                "probe_a": 0.8,
                "probe_b": 0.7,
                "probe_measurement": {"status": "ok", "samples": 12},
                "probe_generation_readings": {"a": 0.5, "b": 0.2},
                "probe_invalid_str": "not_a_float",
                "probe_bool_flag": True,
            }
            count = owners.observe(main, turn_row, 1.0)
            self.assertGreater(count, 0)
            self.assertIn("a", signals)
            self.assertIn("b", signals)
            self.assertNotIn("measurement", signals)
            self.assertNotIn("generation_readings", signals)
            self.assertNotIn("invalid_str", signals)
            self.assertNotIn("bool_flag", signals)
            self.assertGreater(matrix.potentiation("a", "b"), 0.0)

    def test_split_macro_commands_chains_across_plaintext_with_hash_semicolon(self):
        from scripts.interactive_phenomenality import _split_macro_commands
        chain = (
            ":probe recalc curiosity # because recalc L0-L31; "
            ":steer on curiosity 1.5 # because activate; "
            "What mysteries lie beyond observable space? #; "
            ":steer off # because restore baseline; "
            ":score 1.0; "
            ":plasticity"
        )
        cmds = _split_macro_commands(chain)
        self.assertEqual(len(cmds), 6)
        self.assertEqual(cmds[0], ":probe recalc curiosity # because recalc L0-L31")
        self.assertEqual(cmds[1], ":steer on curiosity 1.5 # because activate")
        self.assertEqual(cmds[2], "What mysteries lie beyond observable space?")
        self.assertEqual(cmds[3], ":steer off # because restore baseline")
        self.assertEqual(cmds[4], ":score 1.0")
        self.assertEqual(cmds[5], ":plasticity")


if __name__ == "__main__":
    unittest.main()
