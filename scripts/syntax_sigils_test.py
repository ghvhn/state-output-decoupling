#!/usr/bin/env python3
"""
Unit tests for semicolon command splitting and punctuation sigils:
  ! -> Knob (e.g. !claimmap_alpha)
  % -> Status (e.g. %cpu_pct, %ram_pct)
  ? -> Probe (e.g. ?care, ?accuracy)
  + / - -> Positive / Negative polarity / correlation
  ; -> Strict command splitting (touching word to the right)
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.interactive_phenomenality import (
    _split_macro_commands,
    split_because,
    parse_field_source,
    field_source_error,
    resolve_field_source,
    canonical_field_node,
    normalize_watch_item,
    resolve_stream,
    resolve_probe_choice,
    startup_macro_files,
    startup_macro_path,
    declare_startup_macro,
    startup_input_queue,
)
from invariants.trigger_tuner import TriggerTuner


class TestSemicolonCommandSplitting(unittest.TestCase):
    def test_strict_semicolon_splitting_touching_right(self):
        # Touching word to the right (no extra space)
        cmd = ":clock;travel"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, [":clock", ":travel"])

    def test_strict_semicolon_splitting_explicit_colons(self):
        cmd = ":clock;:travel"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, [":clock", ":travel"])

        cmd2 = ":clock; :travel"
        res2 = _split_macro_commands(cmd2)
        self.assertEqual(res2, [":clock", ":travel"])

        cmd3 = ":clock ; :travel"
        res3 = _split_macro_commands(cmd3)
        self.assertEqual(res3, [":clock", ":travel"])

    def test_strict_semicolon_chain_multiple(self):
        cmd = ":clock;travel;timestamps on"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, [":clock", ":travel", ":timestamps on"])

    def test_strict_semicolon_splitting_space_before_no_space_after(self):
        # Space BEFORE semicolon with NO space AFTER chains commands
        cmd = ":clock ;travel ;timestamps on"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, [":clock", ":travel", ":timestamps on"])

    def test_semicolon_with_space_after_indicates_prompt(self):
        # Semicolon followed by space indicates an actual prompt (sending text)
        cmd = ":tune !claimmap_alpha 0.05 ; What are your thoughts on this?"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, [":tune !claimmap_alpha 0.05", "What are your thoughts on this?"])

        cmd2 = ":clock ; travel"
        res2 = _split_macro_commands(cmd2)
        self.assertEqual(res2, [":clock", "travel"])

        cmd3 = ":clock; travel"
        res3 = _split_macro_commands(cmd3)
        self.assertEqual(res3, [":clock", "travel"])

        cmd4 = ":clock ;travel ; Explain your thoughts"
        res4 = _split_macro_commands(cmd4)
        self.assertEqual(res4, [":clock", ":travel", "Explain your thoughts"])

        cmd5 = "; Explain your thoughts"
        res5 = _split_macro_commands(cmd5)
        self.assertEqual(res5, ["Explain your thoughts"])

    def test_space_colon_does_not_split(self):
        # Space-colon must NEVER split commands now
        cmd = ":suggest :clock"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, [":suggest :clock"])

        cmd2 = ":steer field init 0.1 :steer field shape"
        res2 = _split_macro_commands(cmd2)
        self.assertEqual(res2, [":steer field init 0.1 :steer field shape"])

    def test_escaped_semicolons(self):
        cmd = r":say hello\; world;say second"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, [r":say hello; world", ":say second"])

    def test_non_colon_chain(self):
        cmd = "first question;second question"
        res = _split_macro_commands(cmd)
        self.assertEqual(res, ["first question", "second question"])


class TestPunctuationSigils(unittest.TestCase):
    def setUp(self):
        self.tuner = TriggerTuner()
        self.tuner.register("claimmap_alpha", 0.02, kind="real")
        self.tuner.register("lookahead_alpha", 0.10, kind="real")
        self.probes = {
            "care": {"direction": {}, "framings": ("I care", "I don't care")},
            "accuracy": {"direction": {}, "framings": ("I am accurate", "I hallucinate")},
            "hallucination": {"direction": {}, "framings": ("I hallucinate", "I am accurate")},
        }

    def test_parse_field_source_constants(self):
        s1 = parse_field_source("1.5")
        self.assertEqual(s1["kind"], "constant")
        self.assertEqual(s1["value"], 1.5)

        s2 = parse_field_source("-0.25")
        self.assertEqual(s2["kind"], "constant")
        self.assertEqual(s2["value"], -0.25)

    def test_parse_field_source_sigils(self):
        # ! knob
        knob = parse_field_source("!claimmap_alpha")
        self.assertEqual(knob["kind"], "knob")
        self.assertEqual(knob["name"], "claimmap_alpha")
        self.assertEqual(knob["scale"], 1.0)

        # % status
        status = parse_field_source("%cpu_pct")
        self.assertEqual(status["kind"], "status")
        self.assertEqual(status["name"], "cpu_pct")
        self.assertEqual(status["scale"], 1.0)

        # ? probe
        probe = parse_field_source("?care")
        self.assertEqual(probe["kind"], "probe")
        self.assertEqual(probe["name"], "care")
        self.assertEqual(probe["scale"], 1.0)

        # $ anchor (absolute reference)
        anchor = parse_field_source("$origin")
        self.assertEqual(anchor["kind"], "anchor")
        self.assertEqual(anchor["name"], "origin")
        self.assertEqual(anchor["scale"], 1.0)

        # @ anchor (backwards-compat alias)
        anchor_at = parse_field_source("@origin")
        self.assertEqual(anchor_at["kind"], "anchor")
        self.assertEqual(anchor_at["name"], "origin")
        self.assertEqual(anchor_at["scale"], 1.0)

    def test_parse_field_source_polarity(self):
        # +?care
        pos_probe = parse_field_source("+?care")
        self.assertEqual(pos_probe["kind"], "probe")
        self.assertEqual(pos_probe["name"], "care")
        self.assertEqual(pos_probe["scale"], 1.0)

        # -?hallucination
        neg_probe = parse_field_source("-?hallucination")
        self.assertEqual(neg_probe["kind"], "probe")
        self.assertEqual(neg_probe["name"], "hallucination")
        self.assertEqual(neg_probe["scale"], -1.0)

        # +!claimmap_alpha
        pos_knob = parse_field_source("+!claimmap_alpha")
        self.assertEqual(pos_knob["kind"], "knob")
        self.assertEqual(pos_knob["name"], "claimmap_alpha")
        self.assertEqual(pos_knob["scale"], 1.0)

        # -!lookahead_alpha
        neg_knob = parse_field_source("-!lookahead_alpha")
        self.assertEqual(neg_knob["kind"], "knob")
        self.assertEqual(neg_knob["name"], "lookahead_alpha")
        self.assertEqual(neg_knob["scale"], -1.0)

        # -%ram_pct
        neg_status = parse_field_source("-%ram_pct")
        self.assertEqual(neg_status["kind"], "status")
        self.assertEqual(neg_status["name"], "ram_pct")
        self.assertEqual(neg_status["scale"], -1.0)

        # -probe:hallucination
        neg_colon = parse_field_source("-probe:hallucination")
        self.assertEqual(neg_colon["kind"], "probe")
        self.assertEqual(neg_colon["name"], "hallucination")
        self.assertEqual(neg_colon["scale"], -1.0)

    def test_status_sigil_cannot_resolve_to_a_knob(self):
        source = parse_field_source("%claimmap_alpha")
        self.assertEqual(source["kind"], "status")
        self.assertEqual(field_source_error(source, tuner=self.tuner), "unknown status 'claimmap_alpha'")
        self.assertIsNone(canonical_field_node("%claimmap_alpha", tuner=self.tuner))
        with patch("scripts.interactive_phenomenality.field_status_value", return_value=0.0):
            self.assertEqual(resolve_field_source(source, tuner=self.tuner), 0.0)

    def test_explicit_anchor_namespace(self):
        self.assertEqual(canonical_field_node("anchor:truth", anchors={"truth": {}}), "anchor:truth")
        self.assertIsNone(canonical_field_node("anchor:missing", anchors={"truth": {}}))

    def test_legacy_probe_alias_does_not_change_canonical_sigils(self):
        self.assertEqual(parse_field_source(".care")["kind"], "probe")
        self.assertEqual(canonical_field_node(".care", probes=self.probes), "probe:care")

    def test_canonical_field_node(self):
        # !knob
        self.assertEqual(
            canonical_field_node("!claimmap_alpha", tuner=self.tuner),
            "knob:claimmap_alpha"
        )
        # ?probe
        self.assertEqual(
            canonical_field_node("?care", probes=self.probes),
            "probe:care"
        )
        # With sign (+?care, -!claimmap_alpha)
        self.assertEqual(
            canonical_field_node("+?care", probes=self.probes),
            "probe:care"
        )
        self.assertEqual(
            canonical_field_node("-!claimmap_alpha", tuner=self.tuner),
            "knob:claimmap_alpha"
        )
        # ~family
        self.assertEqual(
            canonical_field_node("~affect", families={"affect": {}}),
            "family:affect"
        )
        # #16 and L16
        self.assertEqual(
            canonical_field_node("#16"),
            "layer:16"
        )
        self.assertEqual(
            canonical_field_node("L16"),
            "layer:16"
        )
        # $anchor (absolute reference) and @anchor (backwards-compat)
        self.assertEqual(
            canonical_field_node("$truth", anchors={"truth": {}}),
            "anchor:truth"
        )
        self.assertEqual(
            canonical_field_node("@truth", anchors={"truth": {}}),
            "anchor:truth"
        )

    def test_normalize_watch_item(self):
        # !knob
        item, err = normalize_watch_item("!claimmap_alpha", tuner=self.tuner)
        self.assertIsNone(err)
        self.assertEqual(item, {"kind": "knob", "name": "claimmap_alpha"})

        # %status
        item, err = normalize_watch_item("%cpu", tuner=self.tuner)
        self.assertIsNone(err)
        self.assertEqual(item, {"kind": "status", "name": "cpu_pct"})

        # ?probe
        item, err = normalize_watch_item("?care", probes=self.probes, tuner=self.tuner)
        self.assertIsNone(err)
        self.assertEqual(item, {"kind": "probe", "name": "care"})

        # Signed watch item (+?care, -%ram)
        item, err = normalize_watch_item("+?care", probes=self.probes, tuner=self.tuner)
        self.assertIsNone(err)
        self.assertEqual(item, {"kind": "probe", "name": "care"})

        item, err = normalize_watch_item("-%ram", tuner=self.tuner)
        self.assertIsNone(err)
        self.assertEqual(item, {"kind": "status", "name": "ram_pct"})

    def test_resolve_stream(self):
        self.assertEqual(resolve_stream("!claimmap_alpha", self.tuner), "claimmap_alpha")
        self.assertEqual(resolve_stream("+!claimmap_alpha", self.tuner), "claimmap_alpha")
        self.assertEqual(resolve_stream("-!claimmap_alpha", self.tuner), "claimmap_alpha")

    def test_resolve_probe_choice(self):
        self.assertEqual(resolve_probe_choice("?care", self.probes), "care")


class TestSplitBecause(unittest.TestCase):
    def test_split_because_with_hash_and_space(self):
        cmd = ":tune !claimmap_alpha 0.05 # stabilizing intent"
        c, r = split_because(cmd)
        self.assertEqual(c, ":tune !claimmap_alpha 0.05")
        self.assertEqual(r, "stabilizing intent")

    def test_split_because_no_space_after_hash(self):
        # Space BEFORE #, but NO space AFTER #
        cmd = ":tune !claimmap_alpha 0.05 #stabilizing intent"
        c, r = split_because(cmd)
        self.assertEqual(c, ":tune !claimmap_alpha 0.05")
        self.assertEqual(r, "stabilizing intent")

    def test_split_because_space_before_with_hash(self):
        cmd = ":clock #my reason"
        c, r = split_because(cmd)
        self.assertEqual(c, ":clock")
        self.assertEqual(r, "my reason")

        cmd2 = ":clock #myreason"
        c2, r2 = split_because(cmd2)
        self.assertEqual(c2, ":clock")
        self.assertEqual(r2, "myreason")

    def test_split_because_no_space_before_hash_does_not_split(self):
        # No space BEFORE # does NOT split (per requirement: space BEFORE the #)
        cmd = ":tune !claimmap_alpha 0.05#stabilizing intent"
        c, r = split_because(cmd)
        self.assertEqual(c, ":tune !claimmap_alpha 0.05#stabilizing intent")
        self.assertIsNone(r)

        cmd2 = ":clock#my reason"
        c2, r2 = split_because(cmd2)
        self.assertEqual(c2, ":clock#my reason")
        self.assertIsNone(r2)

        cmd3 = ":clock#myreason"
        c3, r3 = split_because(cmd3)
        self.assertEqual(c3, ":clock#myreason")
        self.assertIsNone(r3)

    def test_split_because_with_word(self):
        cmd = ":doc myfile.md because reading background"
        c, r = split_because(cmd)
        self.assertEqual(c, ":doc myfile.md")
        self.assertEqual(r, "reading background")

    def test_split_because_subagent_no_space_after_hash(self):
        cmd = "@subagent :travel #walking latent vectors"
        c, r = split_because(cmd)
        self.assertEqual(c, "@subagent :travel")
        self.assertEqual(r, "walking latent vectors")

        # Without space before #, does not peel
        cmd2 = "@subagent :travel#walking latent vectors"
        c2, r2 = split_because(cmd2)
        self.assertEqual(c2, "@subagent :travel#walking latent vectors")
        self.assertIsNone(r2)

    def test_split_because_multiple_hashes(self):
        cmd = ":clock #item #1 in queue"
        c, r = split_because(cmd)
        self.assertEqual(c, ":clock")
        self.assertEqual(r, "item #1 in queue")

    def test_split_because_escaped_hash(self):
        cmd = r":say hello\#world #real comment"
        c, r = split_because(cmd)
        self.assertEqual(c, r":say hello\#world")
        self.assertEqual(r, "real comment")

    def test_split_because_empty_hash(self):
        cmd = ":clock #"
        c, r = split_because(cmd)
        self.assertEqual(c, ":clock")
        self.assertIsNone(r)

        cmd2 = ":clock#"
        c2, r2 = split_because(cmd2)
        self.assertEqual(c2, ":clock#")
        self.assertIsNone(r2)

    def test_split_because_none(self):
        cmd = ":clock"
        c, r = split_because(cmd)
        self.assertEqual(c, ":clock")
        self.assertIsNone(r)


class TestStartupMacros(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        (self.root / "startup").mkdir()

    def test_only_declared_text_macros_run_in_filename_order(self):
        (self.root / "startup" / "02_setup.TXT").write_text(":clock\n", encoding="utf-8")
        (self.root / "startup" / "01_setup.txt").write_text("# setup\n:cen off\n", encoding="utf-8")
        (self.root / "startup" / "README.md").write_text("not runnable", encoding="utf-8")
        self.assertEqual(startup_macro_files(self.root), ["01_setup.txt", "02_setup.TXT"])
        self.assertEqual(
            startup_input_queue({}, "Who are you?", root=self.root),
            [(":cen off", True), (":clock", True), "Who are you?"],
        )

    def test_disabled_startup_keeps_explicit_prompt(self):
        (self.root / "startup" / "setup.txt").write_text(":clock\n", encoding="utf-8")
        self.assertEqual(startup_input_queue({}, "Hello", enabled=False, root=self.root), ["Hello"])

    def test_empty_startup_does_not_invent_a_macro(self):
        (self.root / "init_lookahead.txt").write_text(":lookahead on\n", encoding="utf-8")
        self.assertEqual(startup_input_queue({}, root=self.root), [])

    def test_add_preserves_spaces_and_is_idempotent(self):
        source = self.root / "my setup.txt"
        source.write_text(":cen off\n", encoding="utf-8")
        target = declare_startup_macro(f'"{source}"', self.root)
        self.assertEqual(Path(target).read_text(encoding="utf-8"), ":cen off\n")
        self.assertEqual(declare_startup_macro(target, self.root), target)

    def test_add_rejects_files_that_would_not_run(self):
        source = self.root / "notes.md"
        source.write_text(":clock", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "txt extension"):
            declare_startup_macro(source, self.root)

    def test_remove_target_stays_inside_startup(self):
        for name in ("../important", "..\\important", str(self.root / "important.txt")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                startup_macro_path(name, self.root)
        self.assertEqual(Path(startup_macro_path('"my setup"', self.root)).resolve(),
                         (self.root / "startup" / "my setup.txt").resolve())


if __name__ == "__main__":
    unittest.main()
