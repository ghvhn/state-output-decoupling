"""Model-facing identifiers do not replace ordinary conversational concepts."""

import ast
import copy
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from invariants.exposure_language import probe_alias
from invariants.hebbian_plasticity import PlasticityOwners
from scripts import interactive_phenomenality as shell


def shell_tree():
    return ast.parse(Path(shell.__file__).read_text(encoding="utf-8"))


def actual_nested_function(name, namespace):
    function = next(node for node in ast.walk(shell_tree())
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    module = ast.Module(body=[copy.deepcopy(function)], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), shell.__file__, "exec"), namespace)
    return namespace[name]


def probes():
    return {
        "reality": {"exposed": True, "framings": ("I accept facts.", "I reject facts."),
                    "history": deque()},
        "confidence": {"exposed": True, "framings": ("I am certain.", "I am uncertain."),
                       "history": deque()},
    }


def test_actual_live_prompt_preserves_words_in_speech_memories_and_reader(tmp_path):
    active = probes()
    tuner = SimpleNamespace(path=tmp_path / "tuner.json", triggers={})
    namespace = dict(vars(shell), tuner=tuner, probes=active,
                     plasticity_owners=PlasticityOwners(), _other_mint=lambda _pair: {},
                     _reader_context=lambda: [("SOURCE: reality", "Confidence comes from checking reality.")],
                     format_field_prompt_context=lambda *_args: None)
    build = actual_nested_function("_build_live_prompt", namespace)
    prompt = build(
        "You are a part of reality. I mean [probe:reality] specifically.",
        memory_tool_result="Memory: I spoke with confidence about reality; probe_confidence was measured.",
        session_context=[("assistant", "main_assistant", "I accept reality with confidence.", None)],
    )
    assert "You are a part of reality." in prompt
    assert "I spoke with confidence about reality;" in prompt
    assert "I accept reality with confidence." in prompt
    assert "Confidence comes from checking reality." in prompt
    assert "SOURCE: reality" in prompt
    assert probe_alias("reality") in prompt and probe_alias("confidence") in prompt
    assert "[probe:reality]" not in prompt and "probe_confidence" not in prompt


def test_actual_field_context_emits_ids_at_typed_sources(monkeypatch):
    state = copy.deepcopy(shell._STEER_BODIES)
    for key, value in state.items():
        if isinstance(value, dict):
            state[key] = {}
    state["fields"] = {"probe:reality": {"g": {"kind": "probe", "name": "confidence"}}}
    monkeypatch.setattr(shell, "steer_bodies", lambda: state)
    tuner = SimpleNamespace(get=lambda name, default=0.: 1. if name == "prioritize_gravity" else default,
                            triggers={})
    context = shell.format_field_prompt_context(tuner, probes())
    assert f'active_probes={probe_alias("confidence")},{probe_alias("reality")}' in context
    assert "probe:reality" not in context and "probe:confidence" not in context
    assert probe_alias("reality") in context and probe_alias("confidence") in context


def test_actual_spawn_report_emits_probe_ids_and_preserves_knob_names():
    active = probes()
    tuner = SimpleNamespace(triggers={})
    namespace = dict(vars(shell), tuner=tuner, probes=active, doc_library=[],
                     queued_calibrations=[], queued_commands=[], hidden_commands=set(),
                     exposed_commands={}, exposed_knobs={"confidence"}, macro_aliases={},
                     _all_shell_commands=lambda: set(), rank_probes=lambda *_args: [],
                     format_field_prompt_context=lambda *_args: None)
    report = actual_nested_function("_spawn_state_report", namespace)
    for compact in (False, True):
        text = report("worker", tuner, active, compact=compact)
        assert f'- {probe_alias("reality")}: ' in text
        assert f'- {probe_alias("confidence")}: ' in text
        assert "- reality:" not in text and "- confidence:" not in text
        sensor_line = next(line for line in text.splitlines()
                           if line.startswith("Model-readable probes/knobs:"))
        assert probe_alias("reality") in sensor_line and probe_alias("confidence") in sensor_line
        assert sensor_line.endswith(", confidence")


def actual_sensor_result(**overrides):
    """Execute real result formatting without running UI or generation."""
    branch = next(node for node in ast.walk(shell_tree()) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "model_probe_query and (probes or exposed_knobs)")
    start = next(i for i, node in enumerate(branch.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "probe_lines"
                         for target in node.targets))
    stop = next(i for i, node in enumerate(branch.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "model_probe_tool_result"
                        for target in node.targets))
    module = ast.Module(body=copy.deepcopy(branch.body[start:stop + 1]), type_ignores=[])
    namespace = dict(vars(shell), naming_result=None, cand_text="", readable=["reality"],
                     readable_knobs=[], blocked=[], probes=probes(),
                     tuner=SimpleNamespace(triggers={}), turn_log=[{"probe_reality": .25}],
                     exposed_probe_all=["confidence", "reality"], exposed_knob_all=[],
                     impact_note=lambda cause: f"[Effect: {cause}]")
    namespace.update(overrides)
    exec(compile(ast.fix_missing_locations(module), shell.__file__, "exec"), namespace)
    return namespace["model_probe_tool_result"]


def test_actual_sensor_result_and_consultation_cause_emit_ids():
    result = actual_sensor_result()
    identifier = probe_alias("reality")
    assert f"- {identifier}: +0.250 on my last turn" in result
    assert f"consulted my {identifier} probe(s)" in result
    assert "- reality:" not in result and "consulted my reality" not in result


def test_actual_sensor_menu_emits_probe_ids():
    result = actual_sensor_result(readable=[])
    assert f'probes: {probe_alias("confidence")}, {probe_alias("reality")}' in result
    assert "probes: confidence, reality" not in result


def test_actual_blocked_existing_sensor_emits_id():
    result = actual_sensor_result(readable=[], blocked=["confidence"])
    assert f'- {probe_alias("confidence")}: not a sensor I can consult.' in result
    assert "- confidence:" not in result
