"""Frameless novel direction composition and flexible distributions for :relate indirect."""

from copy import deepcopy
from types import SimpleNamespace
import math
import pytest
import torch

from invariants import continuation_investigation as continuation
from invariants.continuation_mode import ContinuationMode
from invariants.indirect_steering import (
    IndirectSteering,
    compose_direction,
    definition_fingerprints,
    handle_indirect_command,
)
from invariants.other_probes import is_other_probe
from invariants.replay_investigation import _prepare_helpers
from test_joint_continuation_helpers import helpers as make_helpers, nonlinear, read_rows


def _unit(values):
    vec = torch.tensor(values, dtype=torch.float32)
    return vec / vec.norm()


def _probe_bank():
    return {
        "target": {
            "direction": {
                0: _unit([0.0, 0.0, 1.0, 0.0]),
                1: _unit([0.0, 0.0, 1.0, 0.0]),
                2: _unit([0.0, 0.0, 1.0, 0.0]),
            },
            "framings": ("Target present", "Target absent"),
            "history": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "measurement_source": "live_assistant_prediction",
        },
        "left": {
            "direction": {
                0: _unit([1.0, 0.0, 0.0, 0.0]),
                1: _unit([1.0, 0.2, 0.0, 0.0]),
            },
            "framings": ("Left present", "Left absent"),
            "history": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
            "measurement_source": "live_assistant_prediction",
        },
        "right": {
            "direction": {
                0: _unit([0.6, 0.8, 0.0, 0.0]),
                1: _unit([0.0, 1.0, 0.0, 0.0]),
            },
            "framings": ("Right present", "Right absent"),
            "history": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
            "measurement_source": "live_assistant_prediction",
        },
        "third": {
            "direction": {
                0: _unit([0.0, 0.0, 0.0, 1.0]),
                1: _unit([0.0, 0.0, 0.0, 1.0]),
            },
            "framings": ("Third present", "Third absent"),
            "history": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55],
            "measurement_source": "live_assistant_prediction",
        },
    }


def test_frameless_compose_direction_modes_and_target_orthogonalization():
    probes = _probe_bank()
    # Weighted composition without framings
    weighted = compose_direction(probes, [("left", 0.7), ("right", 0.3)], mode="weighted")
    assert weighted["framings"] is None
    assert weighted["composition_mode"] == "weighted"
    assert weighted["composed_from"] == ["left", "right"]
    assert not is_other_probe(weighted)
    for layer, vec in weighted["direction"].items():
        assert float(vec.norm()) == pytest.approx(1.0, rel=1e-5)
        weights = weighted["projection_weights"][layer]
        recon = sum(probes[name]["direction"][layer] * w for name, w in weights.items())
        torch.testing.assert_close(vec, recon, atol=1e-5, rtol=1e-5)

    # Residual composition projects out secondary components from primary on each layer
    residual = compose_direction(probes, [("left", 1.0), ("right", 1.0)], mode="residual")
    assert residual["framings"] is None
    dot_l0 = float(torch.dot(residual["direction"][0], probes["right"]["direction"][0]))
    assert abs(dot_l0) < 1e-5

    # Orthogonalization against target removes any shared-layer target component
    probes["skew"] = {
        "direction": {0: _unit([1.0, 0.0, 2.0, 0.0])},
        "framings": None,
    }
    safe = compose_direction(
        probes, [("skew", 1.0), ("right", 0.5)],
        mode="orthogonal", orthogonal_to=["target"],
    )
    dot_target = float(torch.dot(safe["direction"][0], probes["target"]["direction"][0]))
    assert abs(dot_target) < 1e-5


def test_handle_indirect_compose_backfills_observations_and_plans_without_framings(tmp_path):
    probes = _probe_bank()
    controller = IndirectSteering(tmp_path / "indirect.json")
    tuner = SimpleNamespace(
        _last_probe_readout_source="live_assistant_prediction",
        indirect_controller=controller,
    )
    # Record paired live observations on base probes first
    base_series = [
        {"target": 0.10, "left": 0.20, "right": -0.10, "third": 0.05},
        {"target": 0.25, "left": 0.45, "right": -0.25, "third": 0.10},
        {"target": 0.40, "left": 0.70, "right": -0.35, "third": 0.15},
        {"target": 0.55, "left": 0.90, "right": -0.50, "third": 0.20},
        {"target": 0.75, "left": 1.15, "right": -0.65, "third": 0.25},
        {"target": 0.95, "left": 1.40, "right": -0.80, "third": 0.30},
        {"target": 1.15, "left": 1.65, "right": -0.95, "third": 0.35},
        {"target": 1.35, "left": 1.90, "right": -1.10, "third": 0.40},
    ]
    for step, values in enumerate(base_series, 1):
        for name, val in values.items():
            probes[name]["last_profile"] = {
                "raw_score": val,
                "by_layer": {layer: val for layer in probes[name]["direction"]},
            }
            probes[name]["_last_generation_observation"] = {"generation_id": f"gen-{step}"}
        controller.observe(probes)

    # Compose a novel frameless direction via :relate indirect compose
    result = handle_indirect_command(
        ["compose", "novel_axis", "from", "0.75*left", "-0.25*right", "mode", "orthogonal"],
        controller, probes, tuner,
    )
    assert "novel_axis" in probes
    assert probes["novel_axis"]["framings"] is None
    assert result["composed"]["name"] == "novel_axis"

    # Configure target route using the frameless composed probe; backfill provides immediate evidence
    handle_indirect_command(
        ["maximize", "target", "via", "novel_axis"],
        controller, probes, tuner,
    )
    actions = controller.plan(probes)
    assert any(a["name"] == "novel_axis" and a["coefficient"] > 0 for a in actions)

    # Compose joint plan directions into a single frameless vector per layer
    synthesized = controller.compose_plan_directions(probes)
    assert set(synthesized["direction"].keys()) == {0, 1}
    for vec in synthesized["direction"].values():
        assert float(vec.norm()) == pytest.approx(1.0, rel=1e-5)


def test_inline_composition_in_route_and_flexible_live_distributions(tmp_path):
    probes = _probe_bank()
    controller = IndirectSteering(tmp_path / "indirect.json")
    tuner = SimpleNamespace(
        _last_probe_readout_source="live_assistant_prediction",
        indirect_controller=controller,
    )
    for step in range(1, 10):
        for name, scale in (("target", 0.2), ("left", 0.3), ("right", -0.15), ("third", 0.1)):
            val = scale * step + (0.02 * (step % 2) if name == "third" else 0.0)
            probes[name]["last_profile"] = {
                "raw_score": val,
                "by_layer": {0: val * (1.2 if name == "left" else 0.8), 1: val * (0.8 if name == "left" else 1.2)},
            }
            probes[name]["_last_generation_observation"] = {"generation_id": f"gen-{step}"}
        controller.observe(probes)

    # Inline helper expression left+right and prior weights
    handle_indirect_command(
        ["maximize", "target", "via", "left+right@0.7", "third@0.3"],
        controller, probes, tuner,
    )
    assert "left+right" in probes
    assert probes["left+right"]["framings"] is None

    # Compare rigid power distribution vs softened softmax / temperature distribution
    handle_indirect_command(["set", "distribution", "power", "temperature", "1.0"], controller, probes, tuner)
    plan_power = controller.propose_many(probes)
    w_power = [abs(h["coefficient"]) for h in plan_power["helpers"]]

    handle_indirect_command(["set", "distribution", "softmax", "temperature", "2.5", "layer_tilt", "0.5"], controller, probes, tuner)
    plan_soft = controller.propose_many(probes)
    w_soft = [abs(h["coefficient"]) for h in plan_soft["helpers"]]
    assert len(w_power) == len(w_soft) == 2
    # Higher temperature softens the ratio between the stronger and weaker helper
    assert max(w_soft) / min(w_soft) < max(w_power) / min(w_power)
    # Layer-specific weights vary across layers 0 and 1
    for helper in plan_soft["helpers"]:
        assert "layer_weights" in helper and "layer_coefficients" in helper
        assert helper["layer_coefficients"][0] != pytest.approx(helper["layer_coefficients"][1])


def test_continuation_flexible_distributions_novel_arms_and_layer_tilt():
    # Even when size > mixture_resolution (e.g. 3 helpers with resolution=2),
    # flexible/continuous distributions do not collapse to a single uniform profile
    profiles = list(continuation._mixture_profiles(
        3, 2, distribution="flexible", temperature=0.8, priors=(0.6, 0.3, 0.1),
    ))
    assert len(profiles) > 1
    # First profile reflects the prior ordering (w0 > w1 > w2)
    assert profiles[0][0] > profiles[0][1] > profiles[0][2]

    # Candidate arms with compose_novel=True and layer_tilt across 2 layers
    arms, plan = continuation._build_candidate_arms(
        make_helpers(2, layers=(0, 1)), [0, 1],
        max_joint_helpers=2, max_candidates=25, mixture_resolution=4,
        distribution="flexible", temperature=0.9, compose_novel=True, layer_tilt=0.4,
    )
    assert plan["distribution"] == "flexible"
    assert plan["compose_novel"] is True
    assert plan["novel_arms_selected"] > 0
    novel_arms = [arm for arm in arms if arm.get("novel")]
    assert novel_arms
    for arm in novel_arms:
        assert arm["framings"] is None
        assert arm["composition_mode"] == "orthogonal"
        for vec in arm["vectors"].values():
            assert float(vec.norm()) == pytest.approx(1.0, rel=1e-5)
    tilted_joints = [arm for arm in arms if arm.get("layer_tilted")]
    assert tilted_joints
    for arm in tilted_joints:
        lw0 = arm["components"][0]["layer_weights"][0]
        lw1 = arm["components"][0]["layer_weights"][1]
        assert lw0 != pytest.approx(lw1)


def test_continuation_and_replay_accept_inline_frameless_helpers_without_mutating_caller(
        nonlinear, tmp_path):
    model, probes = nonlinear
    before = deepcopy(probes)
    # Prepare helpers with an inline orthogonal composition token
    local_probes = dict(probes)
    helpers, rejected = _prepare_helpers(
        local_probes, {"target": 1}, ["ortho(left,right)"], [0], model.d_model, max_helpers=2,
    )
    assert len(helpers) == 1
    assert helpers[0]["name"] == "ortho(left,right)"
    assert local_probes["ortho(left,right)"]["framings"] is None

    # End-to-end continuation investigation with inline frameless helper and flexible distribution
    result = continuation.run_continuation_investigation(
        model, probes, {"target": 1}, controls=["0.75*left+0.25*right", "right"],
        prompt_messages=[{"role": "user", "content": "Finish this"}],
        output_dir=tmp_path / "cont", max_new_tokens=1, stride=1, horizon=1,
        layers=[0], max_helpers=2, distribution="flexible", temperature=0.8,
        compose_novel=True, progress=None,
    )
    assert result["telemetry"]["tokens_generated"] == 1
    # Caller's probes dictionary was not mutated
    assert set(probes.keys()) == set(before.keys())
    for name in probes:
        assert probes[name]["history"] == before[name]["history"]


def test_continuation_mode_supports_distribution_temperature_novel_and_layer_tilt():
    mode = ContinuationMode()
    mode.configure([
        "on", "distribution", "flexible", "temperature", "0.75",
        "novel", "on", "layer_tilt", "0.4",
    ])
    assert mode.options["distribution"] == "flexible"
    assert mode.options["temperature"] == pytest.approx(0.75)
    assert mode.options["novel"] is True
    assert mode.options["layer_tilt"] == pytest.approx(0.4)
    args = mode.arguments("Hello world")
    assert args[args.index("distribution") + 1] == "flexible"
    assert args[args.index("temperature") + 1] == "0.75"
    assert args[args.index("novel") + 1] == "on"
    assert args[args.index("layer_tilt") + 1] == "0.4"
