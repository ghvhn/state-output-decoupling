"""Unit tests for single-pass execution, tag scrubbing, and synaptic weight integration."""

import os
import unittest
import torch
import torch.nn as nn
from scripts.interactive_phenomenality import (
    build_prompt,
    scrub_unstaged_memory_status,
    remove_tool_calls,
)
from invariants.synaptic_memory import SynapticMemoryEngine, PlasticWeightHook
from invariants.central_executive import DefaultModeNetwork, SharedSubstrateMemory


class TestSinglePassLatent(unittest.TestCase):
    def test_scrub_all_tool_tags_single_pass(self):
        raw_model_response = (
            "<claimmap>valid=True; claim_a=The sky is blue.; claim_b=The sky is green.; "
            "tension=0.35; steer_layers=[14,16,18]</claimmap>\n"
            "<<MEMORY: user preferences on coffee>>\n"
            "<<PROBE: epistemic_humility>>\n"
            "<<METHOD: causal inference>>\n"
            "I understand you prefer black coffee without sugar. Is there anything else you need?"
        )
        scrubbed = scrub_unstaged_memory_status(raw_model_response)
        self.assertNotIn("<claimmap>", scrubbed)
        self.assertNotIn("</claimmap>", scrubbed)
        self.assertNotIn("<<MEMORY:", scrubbed)
        self.assertNotIn("<<PROBE:", scrubbed)
        self.assertNotIn("<<METHOD:", scrubbed)
        self.assertIn("I understand you prefer black coffee without sugar.", scrubbed)

    def test_remove_tool_calls_regex(self):
        text = "Hello! <<MEMORY: test>> That was a memory tag."
        cleaned = remove_tool_calls(text)
        self.assertEqual(cleaned, "Hello!  That was a memory tag.")

    def test_default_tool_reprompting_disabled(self):
        allow_tool_reprompting = (
            os.environ.get("ENABLE_TOOL_REPROMPTING", "0").strip() in {"1", "true", "yes"}
        )
        self.assertFalse(allow_tool_reprompting)

    def test_synaptic_memory_text_integration(self):
        # Create a mock transformer layer
        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.down_proj = nn.Linear(32, 32, bias=False)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = 32
                self.device = torch.device("cpu")
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([MockLayer() for _ in range(22)])

        model = MockModel()
        engine = SynapticMemoryEngine(model=model, target_layers=[14, 16, 18, 20])
        self.assertEqual(len(engine.hooks), 4)

        # Directly consolidate an associative memory trace
        k = torch.randn(32)
        v = torch.randn(32)
        tid = engine.consolidate_associative_memory(k, v, layer_idx=16, learning_rate=0.05)
        self.assertIn(tid, engine.hooks[16].traces)

        # Telemetry
        tel = engine.telemetry()
        self.assertGreater(tel["total_synaptic_traces"], 0)
        self.assertIn(16, tel["active_plastic_layers"])

        engine.detach()

    def test_dmn_synaptic_integration(self):
        substrate = SharedSubstrateMemory(buffer_size=100)
        
        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = nn.Module()
                self.mlp.down_proj = nn.Linear(16, 16, bias=False)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.d_model = 16
                self.device = torch.device("cpu")
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([MockLayer() for _ in range(22)])

        model = MockModel()
        engine = SynapticMemoryEngine(model=model, target_layers=[14, 16])
        dmn = DefaultModeNetwork(substrate=substrate, consolidation_window=2, synaptic_engine=engine)
        self.assertEqual(dmn.synaptic_engine, engine)

        engine.detach()

    def test_steer_band_registered_and_configured(self):
        from scripts.interactive_phenomenality import (
            COMMAND_SPECS,
            steer_bodies,
            save_steer_bodies,
            _band_weight,
            build_band_field_entries,
        )
        # 1. Check spec registration
        self.assertIn(":steer band", COMMAND_SPECS)
        spec = COMMAND_SPECS[":steer band"]
        self.assertIsNotNone(spec)

        # 2. Hooke band weight mechanics
        pull_term = {"mode": "pull", "rest": 0.4, "give": 0.2, "strength_val": 1.0}
        self.assertEqual(float(_band_weight(pull_term, torch.tensor(0.2))), 0.0)
        self.assertGreater(float(_band_weight(pull_term, torch.tensor(0.8))), 0.0)

        push_term = {"mode": "push", "rest": 0.4, "give": 0.2, "strength_val": 1.0}
        self.assertEqual(float(_band_weight(push_term, torch.tensor(0.8))), 0.0)
        self.assertLess(float(_band_weight(push_term, torch.tensor(0.2))), 0.0)

        # 3. Store and retrieve band in steer_bodies
        bodies = steer_bodies()
        bodies.setdefault("bands", {})["test_band"] = {
            "node": "probe:care",
            "mode": "pull",
            "rest": 0.5,
            "give": 0.2,
            "strength": 0.25,
            "when": [],
        }
        del bodies["bands"]["test_band"]

    def test_suggest_suggests_actions_and_separates_from_command_catalog(self):
        from scripts.interactive_phenomenality import (
            suggest_actions,
            suggest_command_catalog,
            automatic_command_help_entries,
            COMMAND_HELP_LINES,
        )
        from invariants.trigger_tuner import TriggerTuner
        import tempfile
        from pathlib import Path

        # 1. suggest_actions provides concrete runnable commands
        with tempfile.TemporaryDirectory() as tmpdir:
            tuner = TriggerTuner(path=Path(tmpdir) / "tuner.json")
            sugg = suggest_actions(tuner, [], probes={"accuracy": {}}, archive_size=0)
            self.assertTrue(len(sugg) > 0)
            for cat, desc, cmd in sugg:
                self.assertTrue(cmd.startswith(":"), f"Expected runnable command starting with ':', got: {cmd}")

        # 2. COMMAND_HELP_LINES and docs for :suggest describe suggesting actions, not listing the catalog
        suggest_help = [line for line in COMMAND_HELP_LINES if ":suggest" in line]
        self.assertTrue(any("scan the accrued state and suggest" in line for line in suggest_help))
        self.assertFalse(any("launcher of ALL visible user-facing commands" in line for line in suggest_help))

        commands_md = Path(__file__).parent.parent / "COMMANDS.md"
        if commands_md.exists():
            content = commands_md.read_text(encoding="utf-8")
            self.assertIn("- `:suggest [domain|entity|goal] [...]` -- scan the accrued state", content)
            self.assertNotIn("then show a one-line launcher of ALL visible user-facing commands", content)

        docs_commands_md = Path(__file__).parent.parent / "docs" / "COMMANDS.md"
        if docs_commands_md.exists():
            content = docs_commands_md.read_text(encoding="utf-8")
            self.assertIn("- `:suggest [domain|entity|goal] [...]` -- scan the accrued state", content)
            self.assertNotIn("then show a one-line launcher of ALL visible user-facing commands", content)

        # 3. automatic_command_help_entries works both with and without query
        all_specs = automatic_command_help_entries()
        self.assertTrue(len(all_specs) > 0)
        steer_specs = automatic_command_help_entries(":steer band")
        self.assertEqual(len(steer_specs), 1)

        # 4. suggest_command_catalog is available for :suggest commands
        catalog = suggest_command_catalog()
        self.assertTrue(len(catalog) >= 50)

        # 5. suggest_targeted_actions gets more specific as arguments are provided
        from scripts.interactive_phenomenality import suggest_targeted_actions
        with tempfile.TemporaryDirectory() as tmpdir:
            tuner = TriggerTuner(path=Path(tmpdir) / "tuner.json")
            probes = {"accuracy": {}}

            # Steer domain suggestions
            steer_sugg = suggest_targeted_actions(["steer"], tuner, [], probes=probes)
            self.assertIsNotNone(steer_sugg)
            self.assertTrue(any(":steer" in cmd for _, _, cmd in steer_sugg))

            # Narrowing to band
            band_sugg = suggest_targeted_actions(["steer", "band"], tuner, [], probes=probes)
            self.assertIsNotNone(band_sugg)
            self.assertTrue(all(":steer band" in cmd for _, _, cmd in band_sugg))

            # Narrowing to band + specific probe
            target_band = suggest_targeted_actions(["steer", "band", "accuracy"], tuner, [], probes=probes)
            self.assertIsNotNone(target_band)
            self.assertTrue(any("accuracy_pull pull accuracy" in cmd for _, _, cmd in target_band))
            self.assertTrue(any("accuracy_spring spring accuracy" in cmd for _, _, cmd in target_band))

            # Probe domain + specific probe
            probe_sugg = suggest_targeted_actions(["probe", "accuracy"], tuner, [], probes=probes)
            self.assertIsNotNone(probe_sugg)
            self.assertTrue(any(cmd == ":probe accuracy" for _, _, cmd in probe_sugg))
            self.assertTrue(any(cmd == ":probe backfill accuracy" for _, _, cmd in probe_sugg))

            # Direct entity query (accuracy) suggests across subsystems
            entity_sugg = suggest_targeted_actions(["accuracy"], tuner, [], probes=probes)
            self.assertIsNotNone(entity_sugg)
            self.assertTrue(any(":probe" in cmd for _, _, cmd in entity_sugg))
            self.assertTrue(any(":steer band" in cmd for _, _, cmd in entity_sugg))

            # Free-form goal returns None (signaling model goal synthesis)
            free_goal = suggest_targeted_actions(["make", "answers", "concise"], tuner, [], probes=probes)
            self.assertIsNone(free_goal)

            # 6. Suggesting 'remember' recommends :remember thoughts and fact recording
            remember_sugg = suggest_targeted_actions(["remember"], tuner, [], probes=probes)
            self.assertIsNotNone(remember_sugg)
            self.assertTrue(any(":remember thoughts" in cmd for _, _, cmd in remember_sugg))
            self.assertTrue(any(":remember <fact or rule>" in cmd for _, _, cmd in remember_sugg))
            self.assertTrue(any(":remember episodes" in cmd for _, _, cmd in remember_sugg))

    def test_remember_facts_and_thoughts_via_cen(self):
        from invariants.memory_engine import MemoryEngine
        from invariants.central_executive import (
            CentralExecutiveNetwork,
            DefaultModeNetwork,
            SharedSubstrateMemory,
            TensorActivationSnapshot,
        )
        from scripts.interactive_phenomenality import handle_remember_command, COMMAND_HELP_LINES
        import tempfile
        from pathlib import Path

        # 1. COMMAND_HELP_LINES documents :remember
        rem_help = [line for line in COMMAND_HELP_LINES if ":remember" in line]
        self.assertTrue(len(rem_help) >= 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            mem_path = Path(tmpdir) / "test_memory.jsonl"
            mem = MemoryEngine(path=mem_path)

            # Test help
            ok, rec, msg = handle_remember_command("help", memory=mem)
            self.assertTrue(ok)
            self.assertIn("Usage:", msg)

            # Test remember fact
            staged_result = []
            def _stage_cb(res):
                staged_result.append(res)

            ok, rec, msg = handle_remember_command("User prefers black coffee", memory=mem, stage_callback=_stage_cb)
            self.assertTrue(ok)
            self.assertIsNotNone(rec)
            self.assertEqual(rec.kind, "fact")
            self.assertIn("remember", rec.tags)
            self.assertEqual(rec.text, "User prefers black coffee")
            self.assertEqual(len(staged_result), 0)

            # Test remember stage fact
            ok2, rec2, msg2 = handle_remember_command("stage State-output decoupling is verified", memory=mem, stage_callback=_stage_cb)
            self.assertTrue(ok2)
            self.assertIsNotNone(rec2)
            self.assertEqual(rec2.kind, "fact")
            self.assertEqual(len(staged_result), 1)
            self.assertIn("State-output decoupling is verified", staged_result[0])

            # Test list
            ok_l, _, msg_l = handle_remember_command("list", memory=mem)
            self.assertTrue(ok_l)
            self.assertIn("User prefers black coffee", msg_l)

            # Test search
            ok_s, _, msg_s = handle_remember_command("search coffee", memory=mem)
            self.assertTrue(ok_s)
            self.assertIn("User prefers black coffee", msg_s)

            # 2. Test remembering thoughts via CEN and DMN
            substrate = SharedSubstrateMemory(buffer_size=100)
            cen = CentralExecutiveNetwork(substrate=substrate, gamma=0.08)
            dmn = DefaultModeNetwork(substrate=substrate, consolidation_window=2)

            # Push thought snapshots into substrate
            snap1 = TensorActivationSnapshot(
                timestamp=1000.0,
                token_string="thought_step_1",
                friction_gradient=0.015,
                j_space_variance=[0.1, 0.05],
                step=1,
                derivative_magnitude=0.18,
                raw_tensors=[0.5, 0.2, 0.8],
            )
            snap2 = TensorActivationSnapshot(
                timestamp=1001.0,
                token_string="thought_step_2",
                friction_gradient=0.042,
                j_space_variance=[0.12, 0.06],
                step=2,
                derivative_magnitude=0.25,
                raw_tensors=[0.6, 0.3, 0.9],
            )
            substrate.put_snapshot(snap1)
            substrate.put_snapshot(snap2)

            # Consolidate thought stream via :remember thoughts
            ok_t, rec_t, msg_t = handle_remember_command(
                "thoughts",
                memory=mem,
                cen_substrate=substrate,
                cen_network=cen,
                dmn_network=dmn,
            )
            self.assertTrue(ok_t)
            self.assertIsNotNone(rec_t)
            self.assertEqual(rec_t.kind, "thought_episode")
            self.assertIn("thought_episode", rec_t.tags)
            self.assertIn("EP_0001", rec_t.text)
            self.assertIn("thought_step_1 thought_step_2", rec_t.text)
            self.assertIn("Dynamic Friction", msg_t)
            self.assertIn("Trajectory Coherence", msg_t)

            # Verify DMN episode repository is updated
            self.assertEqual(len(substrate.consolidated_repository), 1)
            ep = substrate.consolidated_repository[0]
            self.assertEqual(ep["episode_id"], "EP_0001")
            self.assertTrue(ep["trajectory_coherence"] > 0)

            # Test :remember episodes
            ok_ep, _, msg_ep = handle_remember_command(
                "episodes",
                memory=mem,
                cen_substrate=substrate,
                cen_network=cen,
                dmn_network=dmn,
            )
            self.assertTrue(ok_ep)
            self.assertIn("EP_0001", msg_ep)


if __name__ == "__main__":
    unittest.main()


