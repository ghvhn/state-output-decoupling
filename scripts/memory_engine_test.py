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
    AgenticConfig,
    apply_config_overrides,
    backfill_scoring_names,
    build_prompt,
    confident_probe_match,
    expect_generation_profile,
    expected_macro_parts,
    extract_memory_query,
    macro_arg_header_items,
    queue_macro_text,
    restore_config_overrides,
    scrub_unstaged_memory_status,
    suggest_command_catalog,
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
    # Bare-mode contract: no labeled sections, native chat format only. The
    # staged memory folds into the current user turn, ahead of the user's text.
    assert "<|start_header_id|>user<|end_header_id|>" in with_tool
    assert with_tool.index("The first topic was next-token prediction.") < with_tool.index("Elaborate, please.")
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


def test_model_definition_and_feedback_persist_and_retrieve_together():
    tmp, memory = make_memory()
    try:
        definition = memory.append_definition(
            "probe",
            "careful_release",
            ":probe careful_release I test before release. || I release without testing.",
            authored_by="model",
        )
        memory.append_definition_feedback(
            definition,
            "operator accepted and minted the probe",
            verdict="accepted",
            source="operator",
        )
        memory.append_definition_feedback(
            definition,
            "last turn labeled positive at signal +0.42",
            verdict="positive",
            source="operator",
            metrics={"signal": 0.42},
        )

        reloaded = MemoryEngine(
            path=memory.path,
            scope="test_scope",
            include_existing_in_session_view=True,
        )
        found = reloaded.find_definition("probe", "careful_release", authored_by="model")
        assert found is not None
        assert found.record_id == definition.record_id
        feedback = reloaded.definition_feedback(found)
        assert [r.provenance["verdict"] for r in feedback] == ["accepted", "positive"]

        result = reloaded.format_tool_result([found])
        assert "test before release" in result
        assert "accepted and minted" in result
        assert "labeled positive" in result
        prompt = build_prompt("Should I use careful_release?", memory_tool_result=result)
        assert "test before release" in prompt
        assert "accepted and minted" in prompt
        assert "labeled positive" in prompt
    finally:
        tmp.cleanup()


def test_weak_reason_match_does_not_select_arbitrary_probe():
    match, ranked = confident_probe_match(
        {"neutral": 0.174, "careful_release": 0.169, "fun": 0.08}
    )
    assert match is None
    assert ranked[0] == ("neutral", 0.174)

    match, _ = confident_probe_match(
        {"careful_release": 0.44, "neutral": 0.21}
    )
    assert match == ("careful_release", 0.44)


def test_named_backfill_scores_only_named_probe():
    probes = {"neutral": {}, "fun": {}, "careful_release": {}}
    assert backfill_scoring_names(probes, ["neutral"]) == ["neutral"]
    assert backfill_scoring_names(probes, ["neutral"], request_all=True) == list(probes)


def test_suggest_command_catalog_lists_all_visible_user_facing_commands():
    catalog = suggest_command_catalog(
        hidden_commands={"hide"},
        macro_aliases={"daily_review": "daily_review.txt"},
        solve_macros=[("daily_review", "review the latest state", "topic")],
    )
    commands = [entry["command"] for entry in catalog]

    assert len(commands) >= 50
    assert len(commands) == len(set(commands))
    assert any(cmd.startswith(":probe explain <name>") for cmd in commands)
    assert any(cmd.startswith(":steer quality <name>") for cmd in commands)
    assert any(cmd.startswith(":timestamps on|off|status") for cmd in commands)
    assert any(cmd.startswith(":consider <trigger_metric>") for cmd in commands)
    assert any(cmd.startswith(":exit | :quit") for cmd in commands)
    assert any(cmd == ":daily_review <topic>" for cmd in commands)
    assert not any(cmd.startswith(":hide") for cmd in commands)

    probe_only = suggest_command_catalog(filter_text="probe explain")
    assert probe_only
    assert all("probe" in (entry["command"] + " " + entry["summary"]).lower() for entry in probe_only)

    gravity = [entry["command"] for entry in suggest_command_catalog(filter_text="gravity")]
    assert any(cmd.startswith(":steer g ") for cmd in gravity)
    assert any(cmd.startswith(":steer family ") for cmd in gravity)
    assert any(cmd.startswith(":steer quality ") for cmd in gravity)
    assert any(cmd.startswith(":steer shape ") for cmd in gravity)


def test_macro_queue_treats_hash_lines_as_comments():
    queue = []
    queue_macro_text(
        ":expect file invariants/out/macros/demo.txt\n"
        "# this is an implementation comment, not a prompt turn\n"
        "write the profile now\n"
        "# another comment\n"
        ":probe demo with care || without care\n",
        queue,
    )
    assert [item for item, _front in queue] == [
        ":expect file invariants/out/macros/demo.txt",
        "write the profile now",
        ":probe demo with care || without care",
    ]
    assert all(not item.lstrip().startswith("#") for item, _front in queue)


def test_expect_macro_uses_low_memory_utility_profile_and_restores_config():
    cfg = AgenticConfig()
    cfg.synthesis_enabled = True
    cfg.cache_enabled = True
    cfg.cache_write_enabled = True
    cfg.max_routing_events = 4
    cfg.max_loops = 3
    cfg.routing_probe_terms = [{"probe": "macro_author", "weight": 1.0}]

    profile = expect_generation_profile({"type": "macro", "name": "tune_tokens"})
    assert profile is not None
    saved = apply_config_overrides(cfg, profile["overrides"])
    try:
        assert cfg.synthesis_enabled is False
        assert cfg.cache_enabled is False
        assert cfg.cache_write_enabled is False
        assert cfg.max_routing_events == 0
        assert cfg.max_loops == 0
        assert cfg.routing_probe_terms == []
    finally:
        restore_config_overrides(cfg, saved)

    assert cfg.synthesis_enabled is True
    assert cfg.cache_enabled is True
    assert cfg.cache_write_enabled is True
    assert cfg.max_routing_events == 4
    assert cfg.max_loops == 3
    assert cfg.routing_probe_terms == [{"probe": "macro_author", "weight": 1.0}]
    assert expect_generation_profile({"type": "var", "name": "x"}) is None


def test_expected_macro_parts_preserves_args_comment_without_running_comments():
    commands, arg_specs, comments = expected_macro_parts(
        "# args: target, amount=2\n"
        "# comment only; never executed\n"
        ":probe $target I track $target || I ignore $target\n"
        "ordinary prose should not become a command\n"
        ":steer $target $amount\n"
    )

    assert commands == [
        ":probe $target I track $target || I ignore $target",
        ":steer $target $amount",
    ]
    assert macro_arg_header_items(arg_specs) == ["target", "amount=2"]
    assert "# comment only; never executed" in comments


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
    test_model_definition_and_feedback_persist_and_retrieve_together,
    test_weak_reason_match_does_not_select_arbitrary_probe,
    test_named_backfill_scores_only_named_probe,
    test_suggest_command_catalog_lists_all_visible_user_facing_commands,
    test_macro_queue_treats_hash_lines_as_comments,
    test_expect_macro_uses_low_memory_utility_profile_and_restores_config,
    test_expected_macro_parts_preserves_args_comment_without_running_comments,
]


def main():
    print("MEMORY ENGINE TEST -- explicit tool, not hidden prompt context\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  Memory records persist, retrieval is explicit, and prompt use is one-turn only.")


if __name__ == "__main__":
    main()
