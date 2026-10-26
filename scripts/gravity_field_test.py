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


def test_position_sources_resolve_from_field_telemetry():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        probes = {"care": {"history": []}}
        old_bodies = shell._STEER_BODIES
        old_gravity = shell._GRAVITY_STATE
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {"pin": {}},
            "probe_poles": {},
            "laws": {},
            "fields": {},
            "g_families": {},
            "qualities": {},
            "sets": {},
            "impulses": {"kick": {}},
            "spaces": {"void": {}},
            "rotations": {},
            "bands": {"tether": {}},
        }
        shell._GRAVITY_STATE = {
            "accum": None,
            "position": {"care": 0.25, "@pin": -0.1, "space:void": 0.4, "band:tether": 0.2},
            "distance": {"care": 0.75, "@pin": 1.1, "space:void": 0.6, "band:tether": 0.8},
            "prev_cos": {},
            "velocity": {"care": -0.05, "@pin": 0.2, "space:void": 0.07, "band:tether": -0.03},
            "acceleration": {"care": 0.01, "@pin": -0.04, "space:void": 0.02, "band:tether": -0.01},
            "phi": None,
            "time": None,
            "live_handles": [],
        }
        try:
            def value(source):
                spec = shell.parse_field_source(source)
                assert shell.field_source_error(spec, probes, tuner, {}) is None
                return shell.resolve_field_source(spec, probes, tuner, {})

            assert value("pos:care") == 0.25
            assert value("position:probe:care") == 0.25
            assert value("dist:care") == 0.75
            assert value("vel:care") == -0.05
            assert value("dpos:care") == -0.05
            assert value("speed:care") == 0.05
            assert value("accel:care") == 0.01
            assert value("position:@pin") == -0.1
            assert value("velocity:anchor:pin") == 0.2
            assert value("vel:space:void") == 0.07
            assert value("speed:band:tether") == 0.03
            missing = shell.field_source_error(shell.parse_field_source("pos:missing"), probes, tuner, {})
            assert missing.startswith("no field object or recent position telemetry named")
        finally:
            shell._STEER_BODIES = old_bodies
            shell._GRAVITY_STATE = old_gravity


def test_flush_gravity_telemetry_tracks_position_velocity_acceleration():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old_gravity = shell._GRAVITY_STATE
        shell._GRAVITY_STATE = {
            "accum": None,
            "position": {},
            "distance": {},
            "prev_cos": {},
            "velocity": {},
            "acceleration": {},
            "phi": None,
            "time": None,
            "live_handles": [],
        }
        try:
            def flush(cosine):
                shell._GRAVITY_STATE["accum"] = {
                    "n": 1,
                    "phi": 0.2,
                    "time": 1.5,
                    "cos": {"care": cosine},
                    "cnt": {"care": 1},
                }
                shell.flush_gravity_telemetry(tuner)

            flush(0.2)
            assert shell._GRAVITY_STATE["position"]["care"] == 0.2
            assert shell._GRAVITY_STATE["distance"]["care"] == 0.8
            assert shell._GRAVITY_STATE["velocity"] == {}
            assert shell._GRAVITY_STATE["acceleration"] == {}

            flush(0.5)
            assert abs(shell._GRAVITY_STATE["velocity"]["care"] - 0.3) < 1e-9
            assert shell._GRAVITY_STATE["acceleration"] == {}

            flush(0.4)
            assert abs(shell._GRAVITY_STATE["velocity"]["care"] + 0.1) < 1e-9
            assert abs(shell._GRAVITY_STATE["acceleration"]["care"] + 0.4) < 1e-9
        finally:
            shell._GRAVITY_STATE = old_gravity


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


def test_knob_node_can_govern_claimmap_alpha_with_real_time_source():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.register("claimmap_alpha", 0.02, kind="coefficient")
        old = shell._STEER_BODIES
        old_clock = dict(shell._LAST_CLOCK)
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {
                "knob:claimmap_alpha": {
                    "g": shell.parse_field_source("knob:gain", 0.01, 0.0),
                    "time": shell.parse_field_source("status:tokens_per_sec", 0.1, 1.0),
                }
            },
            "g_families": {}, "qualities": {}, "sets": {},
        }
        shell._LAST_CLOCK["tokens_per_sec"] = 10.0
        try:
            assert shell.canonical_field_node(
                "claimmap_alpha", tuner=tuner
            ) == "knob:claimmap_alpha"
            node, cfg = shell.field_target_config(
                "knob:claimmap_alpha", create=True, tuner=tuner
            )
            assert node == "knob:claimmap_alpha"
            assert cfg is shell._STEER_BODIES["fields"]["knob:claimmap_alpha"]
            # G = gain*0.01 = 0.02; time = tokens_per_sec*0.1+1 = 2.
            assert abs(shell.field_governed_knob_value(
                "claimmap_alpha", tuner, {}
            ) - 0.04) < 1e-9
        finally:
            shell._STEER_BODIES = old
            shell._LAST_CLOCK.clear()
            shell._LAST_CLOCK.update(old_clock)


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
            assert f"set:odor quality:smelliness -> probe:{shell.probe_alias('care')}" in context
            assert "time=[0.0, 1.0, 0.5]" in context
            assert f"probe:{shell.probe_alias('sealed')} HARD-EXCLUDED from entire field" in context
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


def test_live_source_masses_resolve_through_g_families():
    # Regression: gravity_field_masses resolved dict masses against
    # steer_bodies().get("families") -- the key is g_families -- so a mass
    # bound to ~family silently fell back to global G instead of the
    # family's chain.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {"pin": {"dirs": {12: torch.ones(4)}, "mass": shell.parse_field_source("!gain")}},
            "probe_poles": {},
            "laws": {},
            "fields": {},
            "g_families": {"fast": {"g": shell.parse_field_source("knob:gain", 0.5, 0.25)}},
            "qualities": {},
            "sets": {},
            "impulses": {},
        }
        try:
            probes = {"care": {"history": [], "direction": {12: torch.ones(4)},
                               "mass": shell.parse_field_source("~fast")}}
            masses = {name: m for name, m, _d in shell.gravity_field_masses(probes, tuner)}
            # care mass = family fast G = gain(2.0)*0.5+0.25 = 1.25;
            # @pin mass = knob gain = 2.0; normalized |sum| = 1.
            assert abs(masses["care"] - 1.25 / 3.25) < 1e-9, masses
            assert abs(masses["@pin"] - 2.0 / 3.25) < 1e-9, masses
        finally:
            shell._STEER_BODIES = old


def test_mass_details_keep_raw_source_separate_from_effective_normalized_mass():
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        e_x = torch.zeros(4); e_x[0] = 1.0
        e_y = torch.zeros(4); e_y[1] = 1.0
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {},
            "probe_poles": {},
            "laws": {},
            "fields": {},
            "g_families": {},
            "qualities": {},
            "sets": {},
            "impulses": {},
            "spaces": {},
            "rotations": {},
            "bands": {},
        }
        try:
            probes = {
                "care": {"history": [], "direction": {5: e_x}, "mass": shell.parse_field_source("2")},
                "regret": {"history": [], "direction": {5: e_y}, "mass": -1.0},
            }
            details = {row["name"]: row for row in shell.gravity_field_mass_details(probes, tuner)}
            assert abs(details["care"]["raw"] - 2.0) < 1e-9
            assert abs(details["care"]["mass"] - (2.0 / 3.0)) < 1e-9
            assert details["care"]["source"] == "2"
            assert abs(details["regret"]["raw"] + 1.0) < 1e-9
            assert abs(details["regret"]["mass"] + (1.0 / 3.0)) < 1e-9
            assert details["regret"]["source"] == "constant"

            compact = {name: m for name, m, _dirs in shell.gravity_field_masses(probes, tuner)}
            assert compact == {"care": details["care"]["mass"], "regret": details["regret"]["mass"]}
            assert "raw +2" in shell.format_mass_detail(details["care"])
        finally:
            shell._STEER_BODIES = old


def test_space_regions_warp_metric_and_drift_force_free():
    # :steer space -- selective metric steering. Warp rescales state<->member
    # distances (contract a<1 grips, expand a>1 thins); rate is a permanent
    # force-free comoving drift riding the impulse term machinery.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        care_dir = torch.zeros(8); care_dir[0] = 1.0
        regret_dir = torch.zeros(8); regret_dir[1] = 1.0
        probes = {
            "care": {"history": [], "direction": {5: care_dir}},
            "regret": {"history": [], "direction": {5: regret_dir}},
        }
        shell._STEER_BODIES = {
            "loaded": True,
            "anchors": {},
            "probe_poles": {},
            "laws": {},
            "fields": {},
            "g_families": {},
            "qualities": {},
            "sets": {},
            "impulses": {},
            "spaces": {
                "care_well": {"node": "probe:care", "sign": -1.0,
                              "warp": shell.parse_field_source("0.5"), "rate": None},
                "regret_void": {"node": "probe:regret", "sign": 1.0,
                                "rate": shell.parse_field_source("0.3"), "warp": None},
            },
        }
        try:
            warp_node, warp_family, warp_layer, drift_layers, summaries = (
                shell.build_space_effects(probes, tuner)
            )
            # contract with w=0.5 -> a = 1/1.5; expand-only region has no warp
            assert abs(warp_node["probe:care"] - 1.0 / 1.5) < 1e-9
            assert "probe:regret" not in warp_node and not warp_family and not warp_layer
            # drift: permanent non-decaying velocity term, expand -> away
            entries = drift_layers[5]
            assert len(entries) == 1 and entries[0]["name"] == "space:regret_void"
            term = entries[0]["impulses"][0]
            assert term["sign"] == -1.0 and abs(term["speed"] - 0.3) < 1e-9
            assert term["decay"] == 1.0 and term["duration"] == 0.0
            assert shell._impulse_decay_factor(term, 500) == 1.0  # never expires
            assert {s[0]: s[4] for s in summaries} == {"care_well": "active", "regret_void": "active"}

            # metric in the pull: contraction makes the same body pull harder
            h = torch.ones(1, 8) * 0.3
            entry = {"direction": care_dir, "mass": 1.0, "g": 1.0, "time": 1.0,
                     "shape": {"kind": "point"}, "name": "care", "qualities": [],
                     "impulses": [], "warp": 1.0}
            plain = shell._gravity_pull(h.clone(), [dict(entry)], 1.0)
            gripped = shell._gravity_pull(h.clone(), [dict(entry, warp=1.0 / 1.5)], 1.0)
            thinned = shell._gravity_pull(h.clone(), [dict(entry, warp=1.5)], 1.0)
            assert gripped.norm() > plain.norm() > thinned.norm()

            # persistence: spaces round-trip through save/load
            old_path = shell.STEER_BODIES_PATH
            shell.STEER_BODIES_PATH = str(Path(tmp) / "field.pt")
            try:
                shell.save_steer_bodies()
                shell._STEER_BODIES = {"loaded": False}
                loaded = shell.steer_bodies()
                assert set(loaded["spaces"]) == {"care_well", "regret_void"}
                assert loaded["spaces"]["care_well"]["sign"] == -1.0
            finally:
                shell.STEER_BODIES_PATH = old_path
        finally:
            shell._STEER_BODIES = old


def test_rotation_transports_state_norm_preserved_orthogonal_untouched():
    # :steer rotate -- rigid transport. The from-component rotates onto the
    # to-direction; the norm is exactly preserved (isometry) and everything
    # orthogonal to the plane is untouched. This is shifting-as-remap, not a
    # pull fighting the source basin.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        u = torch.zeros(8); u[0] = 1.0   # regret direction
        v = torch.zeros(8); v[1] = 1.0   # care direction (orthogonal: plane angle pi/2)
        probes = {
            "regret": {"history": [], "direction": {5: u}},
            "care": {"history": [], "direction": {5: v}},
        }
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
            "impulses": {}, "spaces": {},
            "rotations": {
                "shift_regret": {"from": "probe:regret", "to": "probe:care",
                                 "angle": shell.parse_field_source("0.3")},
            },
        }
        try:
            per_layer, summaries = shell.build_rotation_field_entries(probes, tuner)
            assert summaries[0][3] == "active"
            rots = per_layer[5]
            assert len(rots) == 1
            e1, e2, theta = rots[0]
            assert abs(theta - 0.3) < 1e-9
            assert torch.allclose(e2, v)  # orthonormalized to-direction

            # state: mostly regret, some orthogonal content in dim 2
            h = torch.zeros(1, 8); h[0, 0] = 0.8; h[0, 2] = 0.6
            h2 = h + shell._rotation_delta(h, rots)
            assert abs(float(h2.norm() - h.norm())) < 1e-6          # isometry
            assert float(h2[0, 1]) > float(h[0, 1])                  # gained care
            assert float(h2[0, 0]) < float(h[0, 0])                  # lost regret
            assert abs(float(h2[0, 2] - h[0, 2])) < 1e-7             # orthogonal untouched

            # a full quarter turn maps regret exactly onto care
            full = shell._rotation_delta(u.reshape(1, -1), [(e1, e2, 3.141592653589793 / 2)])
            assert torch.allclose(u.reshape(1, -1) + full, v.reshape(1, -1), atol=1e-6)

            # parallel bodies span no plane: layer skipped, rotation inert
            probes["care2"] = {"history": [], "direction": {5: u.clone()}}
            shell._STEER_BODIES["rotations"]["degenerate"] = {
                "from": "probe:regret", "to": "probe:care2", "angle": shell.parse_field_source("0.3"),
            }
            per_layer2, summaries2 = shell.build_rotation_field_entries(probes, tuner)
            states = {s[0]: s[3] for s in summaries2}
            assert states["degenerate"] == "inert"
            assert len(per_layer2[5]) == 1  # only the real plane remains
        finally:
            shell._STEER_BODIES = old


def test_bands_are_one_sided_and_gate_on_live_thresholds():
    # :steer band -- one-sided Hooke tethers. pull exerts force only when
    # stretched past rest (slack inside costs exactly zero); push only when
    # compressed inside rest; the when-gate engages a band only while a live
    # source is below/above its threshold.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)  # probe_care last signal = 0.75
        old = shell._STEER_BODIES
        u = torch.zeros(8); u[0] = 1.0
        probes = {"care": {"history": [], "direction": {5: u}}}
        base = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
            "impulses": {}, "spaces": {}, "rotations": {},
        }
        shell._STEER_BODIES = dict(base, bands={
            "tether": {"node": "probe:care", "mode": "pull", "rest": 0.4, "give": 0.2,
                       "strength": shell.parse_field_source("1"),
                       "when": {"source": shell.parse_field_source(".care"), "op": "below",
                                "value": 0.5, "tempo": "turn"}},
        })
        try:
            # gate: care reads 0.75, not below 0.5 -> slack (no entries)
            per_layer, summaries = shell.build_band_field_entries(probes, tuner)
            assert summaries[0][4] == "slack" and not per_layer
            # flip the comparator -> engaged
            shell._STEER_BODIES["bands"]["tether"]["when"]["op"] = "above"
            per_layer, summaries = shell.build_band_field_entries(probes, tuner)
            assert summaries[0][4] == "active" and 5 in per_layer
            term = per_layer[5][0]["bands"][0]

            # one-sidedness of the kernel itself
            slack = shell._band_weight(term, torch.tensor(0.2))     # d < rest
            taut = shell._band_weight(term, torch.tensor(0.8))      # d > rest
            assert float(slack) == 0.0
            assert float(taut) > 0.0                                 # pulls toward
            strut = dict(term, mode="push")
            assert float(shell._band_weight(strut, torch.tensor(0.8))) == 0.0
            assert float(shell._band_weight(strut, torch.tensor(0.2))) < 0.0  # pushes away
            spring = dict(term, mode="spring")
            assert float(shell._band_weight(spring, torch.tensor(0.2))) < 0.0
            assert float(shell._band_weight(spring, torch.tensor(0.8))) > 0.0

            # through the real pull: slack band moves nothing, taut band pulls
            entry = per_layer[5][0]
            entry = dict(entry)
            near = torch.zeros(1, 8); near[0, 0] = 1.0; near[0, 2] = 0.2   # d ~ 0.02
            far = torch.zeros(1, 8); far[0, 0] = 0.1; far[0, 2] = 1.0     # d ~ 0.9
            pull_near = shell._gravity_pull(near, [entry], 1.0)
            pull_far = shell._gravity_pull(far, [entry], 1.0)
            assert float(pull_near.norm()) < 1e-9
            assert float(pull_far.norm()) > 0.0

            # the NET: same taut geometry, but the band exists only in one
            # hemisphere of a side axis, tested per token on the live state
            axis = torch.zeros(8); axis[2] = 1.0
            net_term = dict(entry["bands"][0], side=axis, side_sign=-1.0, side_boundary=0.0)
            net_entry = dict(entry, bands=[net_term])
            above = far.clone()                       # h.axis = +1.0 -> positive side
            below = far.clone(); below[0, 2] = -1.0   # negative side, same distance to body
            pull_above = shell._gravity_pull(above, [net_entry], 1.0)
            pull_below = shell._gravity_pull(below, [net_entry], 1.0)
            assert float(pull_above.norm()) < 1e-9    # roams free above the net
            assert float(pull_below.norm()) > 0.0     # caught below it
        finally:
            shell._STEER_BODIES = old


def test_conditional_laws_gate_on_live_sources_and_mass_source_reads():
    # Slingshot: unlike poles + k>0 attract, force scales with the OTHER
    # body's mass -- but a when-gated law exerts force only while every
    # live condition holds. Also: mass:<body> reads the last build's
    # resolved signed mass as a geometry source.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)  # probe_care last signal = 0.75
        tuner.register("gravity_law_step", 0.0, kind="coefficient")
        tuner.set("gravity_law_step", 0.05)
        old = shell._STEER_BODIES
        old_path = shell.STEER_BODIES_PATH
        shell.STEER_BODIES_PATH = str(Path(tmp) / "field.pt")
        e_x = torch.zeros(8); e_x[0] = 1.0
        e_y = torch.zeros(8); e_y[1] = 1.0
        probes = {
            "illeism": {"history": [], "direction": {5: e_y}, "mass": 2.0},
            "care": {"history": []},
        }
        base = {
            "loaded": True,
            "anchors": {"acc": {"dirs": {5: e_x.clone()}, "mass": 1.0,
                                "frozen": False, "poles": {"slingshot": 1.0}}},
            "probe_poles": {"illeism": {"slingshot": -1.0}},
            "laws": {("slingshot", "slingshot"): 0.5},
            "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
            "impulses": {}, "spaces": {}, "rotations": {}, "bands": {},
            "law_gates": {("slingshot", "slingshot"): [
                {"source": shell.parse_field_source(".care"), "op": "above", "value": 0.5},
            ]},
        }
        shell._STEER_BODIES = base
        try:
            # gate met (care reads 0.75, above 0.5): the anchor slingshots
            before = base["anchors"]["acc"]["dirs"][5].clone()
            moved = shell.apply_body_laws(tuner, probes)
            assert moved > 0
            after = base["anchors"]["acc"]["dirs"][5]
            assert float((after * e_y).sum()) > float((before * e_y).sum())

            # gate unmet: statics -- no force at all
            base["law_gates"][("slingshot", "slingshot")][0]["op"] = "below"
            assert shell.apply_body_laws(tuner, probes) == 0

            # LIVE threshold: the bar itself is a source. care (0.75) above
            # knob gain: gain=2.0 -> unmet; retune gain to 0.5 -> met.
            base["law_gates"][("slingshot", "slingshot")] = [{
                "source": shell.parse_field_source(".care"),
                "op": "above",
                "threshold_source": shell.parse_field_source("!gain"),
            }]
            assert shell.apply_body_laws(tuner, probes) == 0
            tuner.set("gain", 0.5)
            assert shell.apply_body_laws(tuner, probes) > 0
            tuner.set("gain", 2.0)

            # the strict law grammar accepts (and strips) the when-clause
            ns = shell.STEER_LAW_SPEC.parse("integrity chaos -0.3 when .care above 0.5")
            assert ns.k == "-0.3"

            # mass:<body> geometry source
            shell._GRAVITY_STATE["mass"] = {"illeism": -0.4, "@acc": 0.6}
            spec = shell.parse_field_source("mass:illeism", -1.0)
            assert spec is not None and spec["kind"] == "mass"
            assert abs(shell.resolve_field_source(spec, probes, tuner, {}) - 0.4) < 1e-9
            spec2 = shell.parse_field_source("mass:@acc")
            assert abs(shell.resolve_field_source(spec2, probes, tuner, {}) - 0.6) < 1e-9
        finally:
            shell._STEER_BODIES = old
            shell.STEER_BODIES_PATH = old_path
            shell._GRAVITY_STATE["mass"] = {}


def test_per_token_gates_integrate_laws_by_duty_cycle():
    # Token-tempo gates (the default) integrate a law by the fraction of
    # last generation's tokens where the LIVE state satisfied every
    # condition; with no telemetry yet they fall back to a turn evaluation.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.register("gravity_law_step", 0.0, kind="coefficient")
        tuner.set("gravity_law_step", 0.05)
        old = shell._STEER_BODIES
        old_path = shell.STEER_BODIES_PATH
        old_duty = dict(shell._GRAVITY_STATE.get("gate_duty") or {})
        shell.STEER_BODIES_PATH = str(Path(tmp) / "field.pt")
        e_x = torch.zeros(8); e_x[0] = 1.0
        e_y = torch.zeros(8); e_y[1] = 1.0
        probes = {
            "illeism": {"history": [], "direction": {5: e_y}, "mass": 2.0},
        }

        def fresh_bodies():
            return {
                "loaded": True,
                "anchors": {"acc": {"dirs": {5: e_x.clone()}, "mass": 1.0,
                                    "frozen": False, "poles": {"slingshot": 1.0}}},
                "probe_poles": {"illeism": {"slingshot": -1.0}},
                "laws": {("slingshot", "slingshot"): 0.5},
                "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
                "impulses": {}, "spaces": {}, "rotations": {}, "bands": {},
                "law_gates": {("slingshot", "slingshot"): [
                    {"source": shell.parse_field_source(".illeism"), "op": "above", "value": 0.0},
                ]},
            }

        def gain_with_duty(duty):
            shell._STEER_BODIES = fresh_bodies()
            if duty is None:
                shell._GRAVITY_STATE["gate_duty"] = {}
            else:
                shell._GRAVITY_STATE["gate_duty"] = {"slingshot|slingshot": duty}
            before = shell._STEER_BODIES["anchors"]["acc"]["dirs"][5].clone()
            shell.apply_body_laws(tuner, probes)
            after = shell._STEER_BODIES["anchors"]["acc"]["dirs"][5]
            return float((after * e_y).sum()) - float((before * e_y).sum())

        try:
            full = gain_with_duty(1.0)
            half = gain_with_duty(0.5)
            zero = gain_with_duty(0.0)
            assert full > half > 0.0, (full, half)
            assert zero == 0.0
            # no telemetry yet: falls back to a turn evaluation of the live
            # source (.illeism reading via its trigger/history -> 0 here, not
            # above 0 -> statics)
            assert gain_with_duty(None) == 0.0
            # the compile step: probe source -> per-layer cosine test
            compiled = shell.gate_token_test(
                {"source": shell.parse_field_source(".illeism"), "op": "above", "value": 0.0},
                probes, tuner,
            )
            assert compiled is not None
            dirs, op, bound = compiled
            assert op == "above" and bound == 0.0 and 5 in dirs
            # dist: flips into cosine space
            compiled2 = shell.gate_token_test(
                {"source": shell.parse_field_source("dist:illeism"), "op": "above", "value": 0.7},
                probes, tuner,
            )
            d2, op2, bound2 = compiled2
            assert op2 == "below" and abs(bound2 - 0.3) < 1e-9
            # knob sources cannot vary per token -> not compilable
            assert shell.gate_token_test(
                {"source": shell.parse_field_source("!gain"), "op": "above", "value": 0.0},
                probes, tuner,
            ) is None
        finally:
            shell._STEER_BODIES = old
            shell.STEER_BODIES_PATH = old_path
            shell._GRAVITY_STATE["gate_duty"] = old_duty


def test_band_token_when_gates_ride_the_pull_as_live_tests():
    # A token-tempo when-gate on a band compiles into the same live-state
    # test machinery as the side clause: engaged per token, not per turn.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        old = shell._STEER_BODIES
        u = torch.zeros(8); u[0] = 1.0
        gate_axis = torch.zeros(8); gate_axis[2] = 1.0
        probes = {
            "care": {"history": [], "direction": {5: u}},
            "illeism": {"history": [], "direction": {5: gate_axis}},
        }
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
            "impulses": {}, "spaces": {}, "rotations": {}, "law_gates": {},
            "bands": {
                "tether": {"node": "probe:care", "mode": "pull", "rest": 0.4, "give": 0.2,
                           "strength": shell.parse_field_source("1"),
                           "when": {"source": shell.parse_field_source(".illeism"),
                                    "op": "above", "value": 0.0}},
            },
        }
        try:
            per_layer, summaries = shell.build_band_field_entries(probes, tuner)
            assert summaries[0][4] == "active"   # not decided at build
            entry = per_layer[5][0]
            tests = shell._band_side_tests(entry["bands"][0])
            assert len(tests) == 1 and tests[0][1] == "above"
            # far from care in both cases; gate axis decides engagement
            hot = torch.zeros(1, 8); hot[0, 1] = 1.0; hot[0, 2] = 0.5   # illeism-positive
            cold = torch.zeros(1, 8); cold[0, 1] = 1.0; cold[0, 2] = -0.5
            pull_hot = shell._gravity_pull(hot, [dict(entry)], 1.0)
            pull_cold = shell._gravity_pull(cold, [dict(entry)], 1.0)
            assert float(pull_hot.norm()) > 0.0
            assert float(pull_cold.norm()) < 1e-9
        finally:
            shell._STEER_BODIES = old


def test_trigger_bodies_move_by_evidence_and_join_the_mass_table():
    # :steer trigger -- a trigger becomes a field body: position = EMA
    # centroid of the states at turns it FIRED, mass = its signed lift by
    # default. Below min evidence it exerts nothing.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.register("cmd_steer", 0.0)
        trig = tuner.triggers["cmd_steer"]
        trig.signals.append(1.0)            # fires (threshold 0.0, >=)
        trig.credit(1.0, 0.8)               # fired outcome
        trig.credit(-1.0, 0.2)              # unfired outcome -> lift +0.6
        old = shell._STEER_BODIES
        old_path = shell.STEER_BODIES_PATH
        shell.STEER_BODIES_PATH = str(Path(tmp) / "field.pt")
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
            "impulses": {}, "spaces": {}, "rotations": {}, "bands": {},
            "law_gates": {},
            "trigger_bodies": {"cmd_steer": {"dirs": {}, "n": 0, "mass": None,
                                             "ema": 0.5, "min_evidence": 2}},
        }
        probes = {"care": {"history": [], "direction": {2: torch.ones(4)}}}
        try:
            hs = torch.zeros(4, 3, 4)       # layers x tokens x dim
            hs[2, :, 0] = 1.0               # layer-2 states point at e_x
            assert shell.update_trigger_bodies(tuner, hs, probes) == 1
            body = shell._STEER_BODIES["trigger_bodies"]["cmd_steer"]
            assert body["n"] == 1 and 2 in body["dirs"]
            # below min evidence: no dirs served, no mass row
            assert shell.impulse_body_dirs("trigger:cmd_steer", probes) == {}
            names = [n for n, _m, _d in shell.gravity_field_masses(probes, tuner)]
            assert "trigger:cmd_steer" not in names

            hs2 = torch.zeros(4, 3, 4)
            hs2[2, :, 1] = 1.0              # second fired turn points at e_y
            assert shell.update_trigger_bodies(tuner, hs2, probes) == 1
            assert body["n"] == 2
            d = body["dirs"][2]
            assert float(d[0]) > 0 and float(d[1]) > 0   # EMA blended both turns
            # now evidence suffices: direction served, mass = normalized lift
            assert 2 in shell.impulse_body_dirs("trigger:cmd_steer", probes)
            rows = {n: m for n, m, _d in shell.gravity_field_masses(probes, tuner)}
            assert "trigger:cmd_steer" in rows and rows["trigger:cmd_steer"] > 0

            # a non-fired turn moves nothing
            trig.signals.append(-5.0)       # does not fire
            assert shell.update_trigger_bodies(tuner, hs, probes) == 0
        finally:
            shell._STEER_BODIES = old
            shell.STEER_BODIES_PATH = old_path


def test_band_when_gates_chain_with_and():
    # Regression: the band parser rejected 'and'-chained conditions that the
    # law parser accepted. Multiple when-gates now AND together; token-tempo
    # ones all ride the pull as live tests.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        tuner.register("probe_illeism", 0.0)
        tuner.triggers["probe_illeism"].signals.append(0.1)
        u = torch.zeros(8); u[0] = 1.0
        g1 = torch.zeros(8); g1[1] = 1.0
        probes = {
            "accuracy": {"history": [], "direction": {5: u}},
            "illeism": {"history": [], "direction": {5: g1}},
        }
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
            "impulses": {}, "spaces": {}, "rotations": {}, "law_gates": {},
            "trigger_bodies": {},
            "bands": {
                "sling": {"node": "probe:accuracy", "mode": "pull", "rest": 0.0, "give": 0.25,
                          "strength": shell.parse_field_source("probe:illeism", 20.0),
                          "when": [
                              {"source": shell.parse_field_source("probe:accuracy"), "op": "below",
                               "threshold_source": shell.parse_field_source("probe:illeism")},
                              {"source": shell.parse_field_source(".illeism"), "op": "above", "value": 0.0},
                          ]},
            },
        }
        try:
            per_layer, summaries = shell.build_band_field_entries(probes, tuner)
            assert summaries[0][4] == "active", summaries
            term = per_layer[5][0]["bands"][0]
            assert len(shell._band_side_tests(term)) == 2  # both gates ride per token
            assert len(shell.band_when_list({"when": {"op": "above"}})) == 1  # legacy dict

            completion, error = shell.validate_command_autocomplete(
                ":steer band accuracy_slingshot pull accuracy rest 0 give 0.25 "
                "strength probe:illeism 20 when probe:accuracy below probe:illeism "
                "and .illeism above 0",
                known_commands={"steer"},
                probe_names=probes,
                knob_names=tuner.triggers,
            )
            assert error is None, error
        finally:
            shell._STEER_BODIES = old


def test_gate_class_is_the_single_grammar():
    # The Gate class owns the when-grammar: parse, evaluate, compile,
    # display. Pins both bug shapes from the hand-rolled era: a parser
    # missing and-chaining entirely, and the off-by-one after 'and'
    # (the keyword slot rule: source/op/threshold at +1..+3).
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        u = torch.zeros(8); u[0] = 1.0
        probes = {
            "accuracy": {"history": [], "direction": {5: u}},
            "illeism": {"history": [], "direction": {5: torch.ones(8)}},
        }
        old = shell._STEER_BODIES
        shell._STEER_BODIES = {
            "loaded": True, "anchors": {}, "probe_poles": {}, "laws": {},
            "fields": {}, "g_families": {}, "qualities": {}, "sets": {},
            "impulses": {}, "spaces": {}, "rotations": {}, "bands": {},
            "law_gates": {}, "trigger_bodies": {},
        }
        try:
            builder = shell.Gate.live_source_builder(probes, tuner, {})
            tokens = ("when probe:accuracy below probe:illeism "
                      "and .illeism above 0 each turn").split()
            gates, nxt, err = shell.Gate.parse_clause(tokens, 0, source_builder=builder)
            assert err is None, err
            assert nxt == len(tokens)
            assert len(gates) == 2
            assert "threshold_source" in gates[0] and gates[0]["op"] == "below"
            assert gates[1]["value"] == 0.0 and gates[1]["tempo"] == "turn"
            assert shell.Gate(gates[1]).tempo() == "turn"
            assert shell.Gate(gates[0]).tempo() == "token"  # default

            # errors carry position and stay errors
            bad, _i, err2 = shell.Gate.parse_clause(
                "when probe:accuracy sideways 3".split(), 0, source_builder=builder)
            assert bad is None and "comparator" in err2
            bad, _i, err3 = shell.Gate.parse_clause(
                "when probe:accuracy below".split(), 0, source_builder=builder)
            assert bad is None and "each condition needs" in err3

            # evaluation + token compilation flow through the same object
            g = shell.Gate({"source": shell.parse_field_source(".accuracy"),
                            "op": "above", "value": 0.0})
            compiled = g.token_test(probes, tuner)
            assert compiled is not None and 5 in compiled[0]
            passes, live = g.evaluate(probes, tuner)
            assert isinstance(passes, bool)

            # name-only builder (the autocomplete validator's mode)
            def name_builder(token):
                return ({"kind": "name", "name": token}, None)
            gates2, nxt2, err4 = shell.Gate.parse_clause(
                "when x above 1 and y below 2".split(), 0, source_builder=name_builder)
            assert err4 is None and len(gates2) == 2 and nxt2 == 8

            # display renders every condition
            txt = shell.Gate.describe_all(gates, probes, tuner)
            assert txt.startswith(" when ") and " and " in txt
        finally:
            shell._STEER_BODIES = old


def test_space_sigils_resolve_in_node_slots():
    # The settled sigil map works in node positions too: ~family #layer
    # .probe %knob translate before canonical resolution.
    with tempfile.TemporaryDirectory() as tmp:
        tuner = fresh_tuner(tmp)
        probes = {"care": {"history": []}}
        families = {"felt": {}}
        assert shell.canonical_field_node("~felt", probes, {}, families, tuner) == "family:felt"
        assert shell.canonical_field_node("#12", probes, {}, families, tuner) == "layer:12"
        assert shell.canonical_field_node(".care", probes, {}, families, tuner) == "probe:care"
        assert shell.canonical_field_node("!gain", probes, {}, families, tuner) == "knob:gain"
        assert shell.canonical_field_node("=ram", probes, {}, families, tuner) is None


TESTS = [
    test_live_sources,
    test_position_sources_resolve_from_field_telemetry,
    test_flush_gravity_telemetry_tracks_position_velocity_acceleration,
    test_family_layer_and_personal_precedence,
    test_live_source_masses_resolve_through_g_families,
    test_mass_details_keep_raw_source_separate_from_effective_normalized_mass,
    test_space_regions_warp_metric_and_drift_force_free,
    test_rotation_transports_state_norm_preserved_orthogonal_untouched,
    test_bands_are_one_sided_and_gate_on_live_thresholds,
    test_conditional_laws_gate_on_live_sources_and_mass_source_reads,
    test_per_token_gates_integrate_laws_by_duty_cycle,
    test_band_token_when_gates_ride_the_pull_as_live_tests,
    test_trigger_bodies_move_by_evidence_and_join_the_mass_table,
    test_band_when_gates_chain_with_and,
    test_gate_class_is_the_single_grammar,
    test_space_sigils_resolve_in_node_slots,
    test_hollow_shell_has_a_surface_not_a_point,
    test_quality_formula_can_spike_nearby_but_is_safe,
    test_quality_set_compliance_time_and_exact_exclusion,
    test_quality_time_vector_drives_without_g,
    test_personal_g_and_time_change_effect_without_global_g,
    test_knob_node_can_govern_claimmap_alpha_with_real_time_source,
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
