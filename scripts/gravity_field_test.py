"""Model-free checks for personal G, families, clocks, and shaped fields."""

from __future__ import annotations

import tempfile
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import scripts.interactive_phenomenality as shell
from invariants.trigger_tuner import TriggerTuner


def fresh_tuner(tmp):
    tuner = TriggerTuner(Path(tmp) / "tuner.json")
    tuner.register("prioritize_alpha", 0.4, kind="coefficient")
    tuner.register("gain", 2.0, kind="coefficient")
    tuner.register("probe_care", 0.0)
    tuner.triggers["probe_care"].signals.append(0.75)
    return tuner


def test_live_sources():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        probes = {"care": {"history": []}}
        families = {
            "fast": {
                "g": shell.parse_field_source("knob:gain", 0.5, 0.25),
                "time": shell.parse_field_source("0.5"),
            }
        }
        assert shell.resolve_field_source(shell.parse_field_source("3"), probes, tuner, families) == 3
        assert shell.resolve_field_source(shell.parse_field_source("global"), probes, tuner, families) == 0.4
        assert shell.resolve_field_source(shell.parse_field_source("probe:care"), probes, tuner, families) == 0.75
        assert shell.resolve_field_source(shell.parse_field_source("family:fast"), probes, tuner, families) == 1.25
        assert shell.resolve_field_source(
            shell.parse_field_source("family:fast"), probes, tuner, families, channel="time"
        ) == 0.5
        assert shell.field_source_error(shell.parse_field_source("probe:missing"), probes, tuner, families) == "no probe named 'missing'"


def test_family_layer_and_personal_precedence():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {},
            "probe_poles": {},
            "laws": {},
            "fields": {
                "probe:care": {"family": "felt"},
                "layer:7": {"g": shell.parse_field_source("2"), "time": shell.parse_field_source("0.5")},
            },
            "g_families": {
                "felt": {
                    "g": shell.parse_field_source("probe:care", 2.0, 0.5),
                    "time": shell.parse_field_source("knob:gain"),
                    "shape": {"kind": "shell", "radius": 0.4, "width": 0.1},
                }
            },
        }
        try:
            cfg = shell.field_entry_config("care", 7, {"care": {"history": []}}, tuner)
            assert abs(cfg["g"] - 4.0) < 1e-9  # (0.75*2 + .5) personal-family G, then layer x2
            assert cfg["time"] == 1.0          # family rate 2, then layer x0.5
            assert cfg["shape"]["kind"] == "shell"
            linked = shell.resolve_field_source(
                shell.parse_field_source("layer:7"),
                {"care": {"history": []}},
                tuner,
                shell._STEER_BODIES["g_families"],
                shell._STEER_BODIES["fields"],
            )
            assert linked == 0.5
        finally:
            shell._STEER_BODIES = old


def test_hollow_shell_has_a_surface_not_a_point():
    shape = {"kind": "shell", "radius": 0.5, "width": 0.1}
    inside = shell._gravity_kernel(torch.tensor(0.25), shape)
    outside = shell._gravity_kernel(torch.tensor(0.75), shape)
    surface = shell._gravity_kernel(torch.tensor(0.5), shape)
    assert inside < 0
    assert outside > 0
    assert abs(float(surface)) < 1e-6


def test_quality_formula_can_spike_nearby_but_is_safe():
    near = shell.quality_formula_value("1/(d+0.02)**4", torch.tensor(0.01))
    far = shell.quality_formula_value("1/(d+0.02)**4", torch.tensor(1.0))
    assert near > far * 1000
    assert near <= 1e6
    assert shell.validate_quality_formula("__import__('os')") is not None


def test_quality_set_compliance_time_and_exact_exclusion():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {},
            "qualities": {
                "smelliness": {
                    "formula": "1/(d+0.02)**4",
                    "strength": shell.parse_field_source("knob:gain"),
                }
            },
            "sets": {
                "odor": {
                    "quality": "smelliness",
                    "members": {
                        "probe:care": {
                            "compliance": shell.parse_field_source("probe:care"),
                            "time_vector": [0.0, 1.0, 0.5],
                        }
                    },
                    "excluded": [],
                }
            },
        }
        try:
            cfg = shell.field_entry_config("care", 0, {"care": {"history": []}}, tuner)
            assert len(cfg["qualities"]) == 1
            assert cfg["qualities"][0]["amplitude"] == 1.5
            assert cfg["qualities"][0]["time_vector"] == [0.0, 1.0, 0.5]
            shell._STEER_BODIES["sets"]["odor"]["excluded"] = ["probe:care"]
            assert shell.quality_terms_for_body(
                "probe:care", {"care": {"history": []}}, tuner
            ) == []
            shell._STEER_BODIES["fields"]["probe:care"] = {"excluded": True}
            assert shell.gravity_field_masses(
                {"care": {"direction": {0: torch.tensor([1.0, 0.0])}, "history": []}},
                tuner,
            ) == []
        finally:
            shell._STEER_BODIES = old


def test_quality_time_vector_drives_without_g():
    h = torch.tensor([[[1.0, 0.0]]])
    entry = {
        "direction": torch.tensor([0.0, 1.0]),
        "mass": 1.0,
        "g": 0.0,
        "time": 1.0,
        "shape": {"kind": "point"},
        "name": "odor",
        "qualities": [{
            "formula": "1/(d+0.02)**4",
            "amplitude": 1.0,
            "time_vector": [0.0, 1.0],
        }],
    }
    dormant = shell._gravity_pull(h, [entry], G=0.0, time_index=0)
    active = shell._gravity_pull(h, [entry], G=0.0, time_index=1)
    assert dormant.abs().max() == 0
    assert active.norm() > 0


def test_personal_g_and_time_change_effect_without_global_g():
    h = torch.tensor([[[1.0, 0.0]]])
    toward_y = torch.tensor([0.0, 1.0])
    base = {
        "direction": toward_y,
        "mass": 1.0,
        "g": 0.5,
        "time": 1.0,
        "shape": {"kind": "gaussian", "width": 2.0},
        "name": "care",
    }
    weak = shell._gravity_pull(h, [base], G=0.0)
    strong = shell._gravity_pull(h, [{**base, "g": 2.0, "time": 2.0}], G=0.0)
    assert weak.norm() > 0  # personal G is independent of global prioritize_alpha
    assert strong.norm() > weak.norm()


def test_explicit_personal_g_activates_before_lift_exists():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {"probe:care": {"g": shell.parse_field_source("1")}},
            "g_families": {},
        }
        try:
            masses = shell.gravity_field_masses(
                {"care": {"direction": {0: torch.tensor([1.0, 0.0])}, "history": []}},
                tuner,
            )
            assert len(masses) == 1
            assert masses[0][0] == "care"
            assert masses[0][1] == 1.0
        finally:
            shell._STEER_BODIES = old


def test_field_hook_runs_with_personal_g_and_global_zero():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.set("prioritize_alpha", 0.0)
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {"probe:care": {"g": shell.parse_field_source("1")}},
            "g_families": {},
        }
        model = SimpleNamespace(
            device="cpu",
            model=SimpleNamespace(
                model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
            ),
        )
        probes = {
            "care": {
                "direction": {0: torch.tensor([0.0, 1.0])},
                "history": [],
            }
        }
        handles = []
        try:
            handles, desc = shell.build_gravity_field_handles(model, probes, tuner)
            assert len(handles) == 1, desc
            x = torch.tensor([[[1.0, 0.0]]])
            y = model.model.model.layers[0](x)
            assert y[0, 0, 1] > 0
        finally:
            for handle in handles:
                handle.remove()
            shell._STEER_BODIES = old


def test_quality_hook_runs_with_every_g_zero():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.set("prioritize_alpha", 0.0)
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {},
            "qualities": {
                "smelliness": {
                    "formula": "1/(d+0.02)**4",
                    "strength": shell.parse_field_source("1"),
                }
            },
            "sets": {
                "odor": {
                    "quality": "smelliness",
                    "members": {
                        "probe:care": {
                            "compliance": shell.parse_field_source("1"),
                            "time_vector": [1.0],
                        }
                    },
                    "excluded": [],
                }
            },
        }
        model = SimpleNamespace(
            device="cpu",
            model=SimpleNamespace(
                model=SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Identity()]))
            ),
        )
        probes = {
            "care": {
                "direction": {0: torch.tensor([0.0, 1.0])},
                "history": [],
            }
        }
        handles = []
        try:
            handles, desc = shell.build_gravity_field_handles(model, probes, tuner)
            assert len(handles) == 1, desc
            x = torch.tensor([[[1.0, 0.0]]])
            y = model.model.model.layers[0](x)
            assert y[0, 0, 1] > 0
        finally:
            for handle in handles:
                handle.remove()
            shell._STEER_BODIES = old


def test_field_state_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        old_path = shell.STEER_BODIES_PATH
        old = shell._STEER_BODIES
        shell.STEER_BODIES_PATH = str(Path(tmp) / "field.pt")
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {},
            "probe_poles": {},
            "laws": {},
            "fields": {"layer:3": {"time": shell.parse_field_source("status:ram", 0.2, 1.0)}},
            "g_families": {"holo": {"g": shell.parse_field_source("1.5"), "shape": {"kind": "shell", "radius": 0.3, "width": 0.08}}},
            "qualities": {
                "smelliness": {
                    "formula": "1/(d+0.02)**4",
                    "strength": shell.parse_field_source("1"),
                }
            },
            "sets": {
                "odor": {
                    "quality": "smelliness",
                    "members": {
                        "layer:3": {
                            "compliance": shell.parse_field_source("0.5"),
                            "time_vector": [0.0, 1.0],
                        }
                    },
                    "excluded": [],
                }
            },
        }
        try:
            shell.save_steer_bodies()
            shell._STEER_BODIES = {
                "loaded": False, "anchors": {}, "probe_poles": {}, "laws": {},
                "fields": {}, "g_families": {},
            }
            loaded = shell.steer_bodies()
            assert loaded["fields"]["layer:3"]["time"]["name"] == "ram"
            assert loaded["g_families"]["holo"]["shape"]["kind"] == "shell"
            assert loaded["qualities"]["smelliness"]["formula"] == "1/(d+0.02)**4"
            assert loaded["sets"]["odor"]["members"]["layer:3"]["time_vector"] == [0.0, 1.0]
        finally:
            shell.STEER_BODIES_PATH = old_path
            shell._STEER_BODIES = old


def test_init_activates_full_schema_without_inventing_laws():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old_path = shell.STEER_BODIES_PATH
        old = shell._STEER_BODIES
        shell.STEER_BODIES_PATH = str(Path(tmp) / "field.pt")
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {"fixed": {"dirs": {}, "mass": 1.0, "frozen": True}},
            "probe_poles": {},
            "laws": {},
            "fields": {},
            "g_families": {},
        }
        try:
            summary = shell.initialize_field_system(tuner, 0.1)
            assert tuner.get("prioritize_gravity") == 1.0
            assert tuner.get("prioritize_alpha") == 0.1
            assert summary["anchors"] == 1
            assert shell._STEER_BODIES["anchors"]["fixed"]["frozen"] is True
            assert shell._STEER_BODIES["laws"] == {}
            assert set(("qualities", "sets", "fields", "g_families")) <= set(shell._STEER_BODIES)
            payload = torch.load(shell.STEER_BODIES_PATH, weights_only=True)
            assert "qualities" in payload and "sets" in payload
        finally:
            shell.STEER_BODIES_PATH = old_path
            shell._STEER_BODIES = old


def test_repo_init_profile_enables_and_reports_full_field():
    root = Path(__file__).parent.parent
    assert (root / "init").read_text(encoding="utf-8").strip() == "startup.txt"
    lines = [
        line.strip()
        for line in (root / "startup.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert ":steer field init 0.1" in lines
    assert ":steer bodies" in lines
    assert ":relations fields" in lines


def test_field_prompt_context_contains_full_live_mechanics():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.set("prioritize_gravity", 1.0)
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {},
            "probe_poles": {},
            "laws": {("odor", "care"): 0.4},
            "fields": {"probe:sealed": {"excluded": True}},
            "g_families": {
                "felt": {
                    "g": shell.parse_field_source("knob:gain"),
                    "time": shell.parse_field_source("0.5"),
                    "shape": {"kind": "shell", "radius": 0.3, "width": 0.1},
                }
            },
            "qualities": {
                "smelliness": {
                    "formula": "1/(d+0.02)**4",
                    "strength": shell.parse_field_source("1"),
                }
            },
            "sets": {
                "odor": {
                    "quality": "smelliness",
                    "members": {
                        "probe:care": {
                            "compliance": shell.parse_field_source("probe:care"),
                            "time_vector": [0.0, 1.0, 0.5],
                        }
                    },
                    "excluded": [],
                }
            },
        }
        try:
            context = shell.format_field_prompt_context(
                tuner, {"care": {"history": []}, "sealed": {"history": []}}
            )
            assert "[Field Context" in context
            assert "quality:smelliness formula=1/(d+0.02)**4" in context
            assert "set:odor quality:smelliness -> probe:care" in context
            assert "time=[0.0, 1.0, 0.5]" in context
            assert "probe:sealed HARD-EXCLUDED from entire field" in context
            assert "pole-law odor<->care" in context
            assert "does not grant command access" in context
            prompt = shell.build_prompt("hello", field_context=context)
            assert context in prompt
            assert prompt.index(context) < prompt.index("hello")
        finally:
            shell._STEER_BODIES = old


def test_suggest_includes_gravity_field_moves():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.register("prioritize_gravity", 0.0, kind="coefficient")
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {},
            "probe_poles": {},
            "laws": {},
            "fields": {},
            "g_families": {},
            "qualities": {},
            "sets": {},
        }
        try:
            suggestions = shell.suggest_actions(
                tuner,
                [],
                probes={"care": {"direction": {"0": torch.ones(4)}, "history": []}},
            )
            field_cmds = [cmd for cat, _line, cmd in suggestions if cat == "field"]
            assert ":steer field init" in field_cmds
            assert any(cmd.startswith(":steer g care ") for cmd in field_cmds)
            assert ":steer family field_core global" in field_cmds
            assert ":steer quality smelliness formula 1/(d+0.02)**4" in field_cmds
        finally:
            shell._STEER_BODIES = old


def test_persisted_prompt_version_refreshes_when_context_contract_changes():
    with tempfile.TemporaryDirectory() as tmp:
        old_root = shell.ROOT
        shell.ROOT = tmp
        prompt_dir = Path(tmp) / "invariants" / "out" / "prompts"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "spawn.txt").write_text("old prompt", encoding="utf-8")
        try:
            rendered = shell.load_prompt(
                "spawn",
                "# SPAWN_PROFILE_V6_FIELD_CONTEXT\n$field_context",
                required_substring="# SPAWN_PROFILE_V6_FIELD_CONTEXT",
                field_context="CURRENT",
            )
            assert rendered == "# SPAWN_PROFILE_V6_FIELD_CONTEXT\nCURRENT"
            assert "SPAWN_PROFILE_V6_FIELD_CONTEXT" in (
                prompt_dir / "spawn.txt"
            ).read_text(encoding="utf-8")
        finally:
            shell.ROOT = old_root


def test_main_prompt_sites_use_live_context_wrapper():
    source = Path(shell.__file__).read_text(encoding="utf-8")
    # Only the function definition and the wrapper's delegated call remain.
    assert len(re.findall(r"(?<!_)build_prompt\(", source)) == 2
    assert source.count("_build_live_prompt(") >= 10


TESTS = [
    test_live_sources,
    test_family_layer_and_personal_precedence,
    test_hollow_shell_has_a_surface_not_a_point,
    test_quality_formula_can_spike_nearby_but_is_safe,
    test_quality_set_compliance_time_and_exact_exclusion,
    test_quality_time_vector_drives_without_g,
    test_personal_g_and_time_change_effect_without_global_g,
    test_explicit_personal_g_activates_before_lift_exists,
    test_field_hook_runs_with_personal_g_and_global_zero,
    test_quality_hook_runs_with_every_g_zero,
    test_field_state_round_trips,
    test_init_activates_full_schema_without_inventing_laws,
    test_repo_init_profile_enables_and_reports_full_field,
    test_field_prompt_context_contains_full_live_mechanics,
    test_suggest_includes_gravity_field_moves,
    test_persisted_prompt_version_refreshes_when_context_contract_changes,
    test_main_prompt_sites_use_live_context_wrapper,
]


if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
