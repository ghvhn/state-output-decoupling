"""Model-free tests for the explicit memory tool.

Run:
    .venv\\Scripts\\python.exe scripts\\memory_engine_test.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.memory_engine import MemoryEngine
from scripts.interactive_phenomenality import build_prompt


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

        reloaded = MemoryEngine(path=memory.path, scope="test_scope", include_existing_in_session_view=True)
        assert len(reloaded.recent_turns(max_turns=1, scope="test_scope")) == 2
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

    tool_result = (
        "[Memory Tool Result]\n"
        "- turn/user scope=test_scope tags=conversation_trace: The first topic was next-token prediction."
    )
    with_tool = build_prompt("Elaborate, please.", memory_tool_result=tool_result)
    assert "The first topic was next-token prediction." in with_tool
    assert "[Current User Message]" in with_tool


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


TESTS = [
    test_memory_engine_is_tool_not_prompt_builder,
    test_turns_are_logged_with_provenance_and_reloaded,
    test_search_returns_explicit_tool_result,
    test_session_boundary_does_not_delete_persistent_memory,
    test_prompt_only_contains_memory_when_tool_result_is_staged,
    test_activation_trace_records_artifact_reference_not_tensor_blob,
]


def main():
    print("MEMORY ENGINE TEST -- explicit tool, not hidden prompt context\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  Memory records persist, retrieval is explicit, and prompt use is one-turn only.")


if __name__ == "__main__":
    main()
