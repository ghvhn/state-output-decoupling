"""Model-free tests for vector-map based self-concept orientation.

Run:
    .venv\\Scripts\\python.exe scripts\\self_concept_controller_test.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.memory_engine import MemoryEngine
from invariants.self_concept_controller import SelfConceptController, format_orientation_tool_result
from scripts.interactive_phenomenality import build_prompt, scrub_unstaged_memory_status


def write_map_files(tmpdir: str):
    root = Path(tmpdir)
    network = {
        "edges": [
            {
                "source": "needless_interrupt_vector",
                "target": "validated_flow_vector",
                "sign": "negative",
                "mean": -0.45,
            }
        ]
    }
    latent = {
        "coordinates": {
            "needless_interrupt_vector": [0.1, 0.2, 0.5],
            "validated_flow_vector": [0.0, -0.2, -0.6],
            "narrowing_in_vector": [0.2, 0.1, 0.0],
            "self_referential_momentum_vector": [0.1, 0.3, 0.1],
            "unwarranted_confidence_vector": [0.2, 0.3, 0.3],
            "warranted_confidence_vector": [0.2, 0.2, 0.3],
            "ambiguity_vector": [0.2, -0.7, 0.1],
        },
        "neighbors": {},
        "anti_edges": [
            {
                "a": "needless_interrupt_vector",
                "b": "validated_flow_vector",
                "mean": -0.45,
            }
        ],
    }
    network_path = root / "vector_network_test.json"
    latent_path = root / "vector_latent_space_test.json"
    network_path.write_text(json.dumps(network), encoding="utf-8")
    latent_path.write_text(json.dumps(latent), encoding="utf-8")
    return latent_path, network_path


def make_controller():
    tmp = tempfile.TemporaryDirectory()
    latent_path, network_path = write_map_files(tmp.name)
    controller = SelfConceptController(latent_space_path=latent_path, network_path=network_path)
    return tmp, controller


def test_needless_interrupt_orients_to_validated_flow():
    tmp, controller = make_controller()
    try:
        decision = controller.decide({"needless_interrupt": 0.32, "ambiguity": 0.02})
        assert decision.action == "orient_toward_validated_flow"
        assert decision.allowed is True
        assert decision.trigger_vector == "needless_interrupt_vector"
        assert decision.target_vector == "validated_flow_vector"
        assert decision.map_relations["needless_interrupt_vs_validated_flow"] == -0.45
    finally:
        tmp.cleanup()


def test_high_ambiguity_preserves_uncertainty_boundary():
    tmp, controller = make_controller()
    try:
        decision = controller.decide({"ambiguity": 0.31, "needless_interrupt": 0.34})
        assert decision.action == "preserve_uncertainty_boundary"
        assert decision.intervention_type == "log_only"
        assert decision.target_vector is None
    finally:
        tmp.cleanup()


def test_self_referential_momentum_orients_to_grounding():
    tmp, controller = make_controller()
    try:
        decision = controller.decide({"self_referential_momentum": 0.24, "ambiguity": 0.03})
        assert decision.action == "orient_toward_task_grounding"
        assert decision.target_vector == "narrowing_in_vector"
    finally:
        tmp.cleanup()


def test_orientation_trace_memory_is_internal():
    tmp, controller = make_controller()
    memory_tmp = tempfile.TemporaryDirectory()
    try:
        memory = MemoryEngine(path=Path(memory_tmp.name) / "memory.jsonl", scope="test")
        decision = controller.decide({"needless_interrupt": 0.32, "ambiguity": 0.02})
        record = memory.append_self_concept_trace(decision.to_dict())
        assert record.kind == "self_concept_trace"
        assert "internal" in record.tags
        assert "self_concept" in record.tags
        assert record.metrics["allowed"] is True
    finally:
        tmp.cleanup()
        memory_tmp.cleanup()


def test_orientation_tool_result_is_explicit_and_one_turn():
    tmp, controller = make_controller()
    try:
        decision = controller.decide({"needless_interrupt": 0.32, "ambiguity": 0.02})
        tool_result = format_orientation_tool_result(decision)
        prompt = build_prompt("continue", orientation_tool_result=tool_result)
        assert "[Orientation Tool Result]" in prompt
        assert "orient_toward_validated_flow" in prompt

        no_tool_prompt = build_prompt("continue")
        assert "[Orientation Tool Result]" not in no_tool_prompt

        scrubbed = scrub_unstaged_memory_status(
            "Answer.\n[Orientation Tool Result: made-up status]",
            orientation_tool_result=None,
        )
        assert "Orientation Tool Result" not in scrubbed
    finally:
        tmp.cleanup()


TESTS = [
    test_needless_interrupt_orients_to_validated_flow,
    test_high_ambiguity_preserves_uncertainty_boundary,
    test_self_referential_momentum_orients_to_grounding,
    test_orientation_trace_memory_is_internal,
    test_orientation_tool_result_is_explicit_and_one_turn,
]


def main():
    print("SELF-CONCEPT CONTROLLER TEST -- vector-map based, audited orientation\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  Self-concept orientation reads vector-map relations and writes auditable internal traces.")


if __name__ == "__main__":
    main()
