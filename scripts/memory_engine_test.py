"""Model-free tests for the explicit memory tool.

Run:
    .venv\\Scripts\\python.exe scripts\\memory_engine_test.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.memory_engine import MemoryEngine, sanitize_methodology_payload
from scripts.interactive_phenomenality import (
    build_prompt,
    extract_memory_query,
    scrub_unstaged_memory_status,
)


def make_memory():
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "memory.jsonl"
    return tmp, MemoryEngine(path=path, scope="test_scope")


def test_memory_engine_is_tool_not_prompt_builder():
    assert not hasattr(MemoryEngine, "build_llama3_prompt")


def test_turns_are_logged_with_provenance_and_reloaded():
    tmp, memory = make_memory()
    try:
        memory.append_turn(
            "user",
            "The mesa-objective citation was wrong.",
            tags=["correction"],
            provenance={"source": "operator"},
        )
        memory.append_turn("assistant", "It came from Hubinger et al.", tags=["answer"])

        raw = [json.loads(line) for line in memory.path.read_text(encoding="utf-8").splitlines()]
        assert raw[0]["kind"] == "turn"
        assert raw[0]["role"] == "user"
        assert raw[0]["provenance"]["source"] == "operator"
        assert "conversation_trace" in raw[0]["tags"]
        assert "external_io" in raw[0]["tags"]

        reloaded = MemoryEngine(path=memory.path, scope="test_scope", include_existing_in_session_view=True)
        assert len(reloaded.recent_turns(max_turns=1, scope="test_scope")) == 2
    finally:
        tmp.cleanup()


def test_internal_trace_memory_is_separate_from_external_io():
    tmp, memory = make_memory()
    try:
        memory.append_turn("user", "What happened internally?")
        memory.append_internal_trace(
            "synthesis_trace",
            text="synthesis reason=optimizer; expert=Analytical; layers=12->18; steps=21",
            tags=["synthesis", "phenomenality"],
            provenance={"phenomenality": {"ambiguity": 0.2}},
            metrics={"steps": 21},
        )
        external = memory.search("internally", kinds=["turn"])
        internal = memory.search("optimizer analytical ambiguity", kinds=["internal_trace"])
        assert len(external) == 1
        assert "external_io" in external[0].tags
        assert len(internal) == 1
        assert internal[0].kind == "internal_trace"
        assert "internal" in internal[0].tags
        assert internal[0].metrics["steps"] == 21
    finally:
        tmp.cleanup()


def test_search_returns_explicit_tool_result():
    tmp, memory = make_memory()
    try:
        memory.append_turn("user", "Nick Bostrom did not coin mesa-objective.")
        memory.append_turn("assistant", "Search says Hubinger et al. introduced the term.")

        records = memory.search("mesa objective Hubinger", scope="test_scope")
        result = memory.format_tool_result(records)
        assert result.startswith("[Memory Tool Result]")
        assert "Hubinger" in result
    finally:
        tmp.cleanup()


def test_session_boundary_does_not_delete_persistent_memory():
    tmp, memory = make_memory()
    try:
        memory.append_turn("user", "Keep this persistent.")
        before = memory.status()["total_records"]
        memory.mark_session_boundary("test")
        memory.append_turn("user", "Only this is in the current session view.")

        assert memory.status()["total_records"] == before + 2
        recent = memory.recent_turns(max_turns=4, scope="test_scope")
        assert [r.text for r in recent] == ["Only this is in the current session view."]
        all_hits = memory.search("persistent", scope="test_scope")
        assert all_hits and all_hits[0].text == "Keep this persistent."
    finally:
        tmp.cleanup()


def test_prompt_only_contains_memory_when_tool_result_is_staged():
    base = build_prompt("Elaborate, please.")
    assert "The first topic was next-token prediction." not in base
    assert "[Memory Tool Result]" not in base

    tool_result = (
        "[Memory Tool Result]\n"
        "- turn/user scope=test_scope tags=conversation_trace: The first topic was next-token prediction."
    )
    with_tool = build_prompt("Elaborate, please.", memory_tool_result=tool_result)
    assert "The first topic was next-token prediction." in with_tool
    assert "[Current User Message]" in with_tool
    assert with_tool.count("[Memory Tool Result]") == 1


def test_current_session_context_is_not_long_term_memory():
    prompt = build_prompt(
        "right. so where's the difference",
        session_context=[
            ("user", "Are you conscious?"),
            ("assistant", "I do not have subjective experience, but I can discuss the distinction."),
        ],
    )
    assert "Are you conscious?" in prompt
    assert "subjective experience" in prompt
    assert "[Memory Tool Result]" not in prompt


def test_model_memory_tool_call_is_parseable_and_removed():
    response = "<<MEMORY: periodic discount methodology>>"
    assert extract_memory_query(response) == "periodic discount methodology"
    assert scrub_unstaged_memory_status(response, memory_tool_result="[Memory Tool Result]\n- real") == ""


def test_fake_memory_status_is_scrubbed_when_unstaged():
    response = (
        "I can answer the current question.\n\n"
        "[Memory Tool Result: No prior conversation or context is available.]"
    )
    scrubbed = scrub_unstaged_memory_status(response, memory_tool_result=None)
    assert "Memory Tool Result" not in scrubbed
    assert "I can answer the current question." in scrubbed

    staged = scrub_unstaged_memory_status(response, memory_tool_result="[Memory Tool Result]\n- real")
    assert "Memory Tool Result" in staged


def test_activation_trace_records_artifact_reference_not_tensor_blob():
    tmp, memory = make_memory()
    try:
        artifact = Path(tmp.name) / "trace.pt"
        artifact.write_text("placeholder", encoding="utf-8")
        memory.append_activation_trace(
            artifact,
            provenance={"probe": "confidence"},
            metrics={"records": 1},
        )
        raw = json.loads(memory.path.read_text(encoding="utf-8").splitlines()[0])
        assert raw["kind"] == "activation_trace"
        assert raw["artifact_path"].endswith("trace.pt")
        assert raw["provenance"]["probe"] == "confidence"
    finally:
        tmp.cleanup()


def test_methodology_import_keeps_sanitized_maps_only():
    tmp, memory = make_memory()
    try:
        payload = {
            "question": "A private word problem with 19 exact things.",
            "answer": "secret-answer",
            "metadata": {
                "clause_methodology": {
                    "kind": "periodic_discount_partition",
                    "methodology": "Partition every-nth discounts before summing.",
                    "structural_features": ["every_nth_item_rule", "discounted_group"],
                    "clause_map_status": "complete",
                    "roles_declared": ["asked", "givens", "rules"],
                    "privacy": {
                        "tier": "reusable_sanitized",
                        "raw_clauses_saved": False,
                        "source_numbers_saved": False,
                        "entity_names_saved": False,
                    },
                }
            },
        }
        assert memory.import_methodologies([payload], source="test_json", source_path="fake.json") == 1
        result = memory.search("periodic discount partition", kinds=["methodology"])
        assert len(result) == 1
        record = result[0]
        assert record.kind == "methodology"
        assert "methodology" in record.tags
        assert "sanitized" in record.tags
        assert "clause_map" in record.tags
        blob = json.dumps(record.to_dict(), sort_keys=True)
        assert "secret-answer" not in blob
        assert "private word problem" not in blob.lower()
        assert "19 exact" not in blob
    finally:
        tmp.cleanup()


def test_methodology_import_rejects_raw_clause_payloads():
    unsafe = {
        "kind": "general_clause_role_binding",
        "methodology": "Bind roles.",
        "privacy": {"raw_clauses_saved": True},
    }
    assert sanitize_methodology_payload(unsafe) is None


TESTS = [
    test_memory_engine_is_tool_not_prompt_builder,
    test_turns_are_logged_with_provenance_and_reloaded,
    test_internal_trace_memory_is_separate_from_external_io,
    test_search_returns_explicit_tool_result,
    test_session_boundary_does_not_delete_persistent_memory,
    test_prompt_only_contains_memory_when_tool_result_is_staged,
    test_current_session_context_is_not_long_term_memory,
    test_model_memory_tool_call_is_parseable_and_removed,
    test_fake_memory_status_is_scrubbed_when_unstaged,
    test_activation_trace_records_artifact_reference_not_tensor_blob,
    test_methodology_import_keeps_sanitized_maps_only,
    test_methodology_import_rejects_raw_clause_payloads,
]


def main():
    print("MEMORY ENGINE TEST -- explicit tool, not hidden prompt context\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  Memory records persist, retrieval is explicit, and prompt use is one-turn only.")


if __name__ == "__main__":
    main()
