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
from invariants.trigger_tuner import TriggerTuner
from scripts.interactive_phenomenality import (
    AgenticConfig,
    add_watch_items,
    apply_config_overrides,
    backfill_scoring_names,
    build_command_autocomplete_reference,
    build_goal_command_reference,
    build_prompt,
    command_takes_colon_command_arg,
    _split_macro_commands,
    macro_create_lines,
    macro_safe_block,
    EXPOSE_TARGET_SPEC,
    expect_generation_profile,
    expected_macro_parts,
    extract_memory_query,
    field_source_error,
    load_watch_groups,
    macro_arg_header_items,
    normalize_watch_item,
    parse_field_source,
    parse_suggest_goal_request,
    queue_macro_text,
    remove_watch_items,
    resolve_expose_choice_target,
    resolve_field_source,
    resolve_probe_choice,
    restore_config_overrides,
    save_watch_groups,
    scrub_unstaged_memory_status,
    STEER_LAW_SPEC,
    suggest_command_catalog,
    validate_command_autocomplete,
    watch_choose_pool,
    watch_group_lines,
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


def test_macro_safe_block_quotes_columnar_grammar():
    quoted = macro_safe_block(
        ":steer mass <probe|choose> <m|auto>\n"
        "# A markdown heading\n"
        "plain description line\n"
    )
    # No quoted line can trigger the macro grammar (column-0 ':' or '#').
    assert all(not line.startswith((":", "#")) for line in quoted.splitlines())
    queue = []
    queue_macro_text(":expect autocomplete :steer\nintro\n" + quoted + "\ntail", queue)
    # Exactly one command + one text turn: the quoted block never splits.
    assert len(queue) == 2
    assert queue[0][0] == ":expect autocomplete :steer"
    text_turn = queue[1][0]
    assert ":steer mass" in text_turn          # usage line kept as content
    assert "# A markdown heading" in text_turn  # heading kept, not dropped


def test_suggest_autocomplete_prompt_queues_one_command_one_text_turn():
    # Regression: CommandSpec usage lines (':steer mass ...') sit at column 0
    # in help entries; embedded raw in the :suggest autocomplete prompt they
    # were split out and EXECUTED -- a cascade of usage errors interleaved
    # with stray text-turn generations instead of one completion.
    for prefix in (":steer", ":probe backfill"):
        reference = build_command_autocomplete_reference(prefix, hidden_commands=set())
        assert reference.strip()
        assert all(not line.startswith((":", "#")) for line in reference.splitlines())
        prompt = (
            f":expect autocomplete {prefix}\n"
            "You are a command autocomplete assistant.\n"
            "--- live command reference ---\n"
            f"{reference}\n"
            "--- end live command reference ---\n"
            "Output only the full completed command string.\n"
        )
        queue = []
        queue_macro_text(prompt, queue)
        assert [item for item, _front in queue][:1] == [f":expect autocomplete {prefix}"]
        assert len(queue) == 2, f"{prefix}: prompt shredded into {len(queue)} queue items"
        assert "command reference" in queue[1][0]


def test_suggest_command_goal_prompt_queues_one_command_one_text_turn():
    reference = build_goal_command_reference(
        hidden_commands={"hide"},
        macro_aliases={"daily_review": "daily_review.txt"},
        solve_macros=[("daily_review", "review the latest state", "topic")],
    )
    assert reference.strip()
    assert ":daily_review <topic>" in reference
    assert not any(line.startswith((":", "#")) for line in reference.splitlines())

    prompt = (
        ":expect autocomplete\n"
        "Choose one command toward this goal: make a compact watch group for ram and time\n"
        "--- live command catalog ---\n"
        f"{reference}\n"
        "--- end live command catalog ---\n"
        "Output exactly one runnable command.\n"
    )
    queue = []
    queue_macro_text(prompt, queue)
    assert [item for item, _front in queue][:1] == [":expect autocomplete"]
    assert len(queue) == 2
    assert "watch group" in queue[1][0]
    assert ":group" in queue[1][0]


def test_suggest_goal_request_accepts_optional_prefix_after_goal_words():
    goal, prefix = parse_suggest_goal_request("make a watch group for ram :group")
    assert goal == "make a watch group for ram"
    assert prefix == ":group"

    goal, prefix = parse_suggest_goal_request(":group make a watch group")
    assert goal == "make a watch group"
    assert prefix == ":group"

    goal, prefix = parse_suggest_goal_request("show all commands")
    assert goal == "show all commands"
    assert prefix == ""


def test_macro_create_at_source_loads_existing_profile_without_text_turn():
    tmp = tempfile.TemporaryDirectory()
    try:
        src = Path(tmp.name) / "startup.txt"
        src.write_text(
            "# comment\n"
            ":timestamps on\n"
            ":steer field init 0.1\n"
            "plain text turn\n",
            encoding="utf-8",
        )
        lines, source = macro_create_lines("@" + str(src), {})
        assert source == str(src)
        assert lines == [":timestamps on", ":steer field init 0.1", "plain text turn"]

        literal, source = macro_create_lines(str(src), {})
        assert source is None
        assert literal == [str(src)]
    finally:
        tmp.cleanup()


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


def test_expose_choose_resolves_through_command_spec_class():
    ns = EXPOSE_TARGET_SPEC.parse(
        "choose steer 0.1",
        choice_resolver=lambda raw: ":suggest",
    )
    assert ns.target == ":suggest"
    assert ns.tail == "steer 0.1"
    assert resolve_expose_choice_target("suggest", ["suggest"], model=None, config=None) == ":suggest"


def test_validate_autocomplete_accepts_expose_choose():
    completion, error = validate_command_autocomplete(
        ":expose choose",
        known_commands={"suggest", "probe", "steer"},
        probe_names=set(),
        knob_names=set(),
    )
    assert completion == ":expose choose"
    assert error is None


def test_watch_group_items_resolve_probes_knobs_and_status_aliases():
    tmp = tempfile.TemporaryDirectory()
    try:
        tuner = TriggerTuner(path=Path(tmp.name) / "tuner.json")
        tuner.register("prioritize_alpha", 0.1, kind="coefficient")
        probes = {"self_model": {"history": [0.1, 0.4]}}

        assert normalize_watch_item("self_model", probes, tuner)[0] == {
            "kind": "probe",
            "name": "self_model",
        }
        assert normalize_watch_item("knob:prioritize_alpha", probes, tuner)[0] == {
            "kind": "knob",
            "name": "prioritize_alpha",
        }
        assert normalize_watch_item("ram", probes, tuner)[0] == {
            "kind": "status",
            "name": "ram_pct",
        }
        assert normalize_watch_item("status:util", probes, tuner)[0] == {
            "kind": "status",
            "name": "cpu_pct",
        }
        assert normalize_watch_item("status:time", probes, tuner)[0] == {
            "kind": "status",
            "name": "time",
        }
    finally:
        tmp.cleanup()


def test_watch_groups_round_trip_and_render_small_groups():
    tmp = tempfile.TemporaryDirectory()
    try:
        tuner = TriggerTuner(path=Path(tmp.name) / "tuner.json")
        tuner.register("prioritize_alpha", 0.1, kind="coefficient")
        tuner.observe("probe_self_model", 0.25)
        probes = {"self_model": {"history": [0.1, 0.4]}}
        group_path = Path(tmp.name) / "watch_groups.json"
        items = [
            normalize_watch_item("self_model", probes, tuner)[0],
            normalize_watch_item("prioritize_alpha", probes, tuner)[0],
            normalize_watch_item("status:time", probes, tuner)[0],
        ]
        groups = {"core": add_watch_items([], items + [items[0]])}
        assert len(groups["core"]) == 3

        save_watch_groups(groups, path=group_path)
        loaded = load_watch_groups(path=group_path)
        assert loaded == groups

        lines = watch_group_lines(loaded, probes, tuner, [{"prioritize_alpha": 0.1}], names=["core"])
        joined = "\n".join(lines)
        assert "[Group] core" in joined
        assert "probe:self_model" in joined
        assert "knob:prioritize_alpha" in joined
        assert "status:time" in joined

        loaded["core"] = remove_watch_items(loaded["core"], [items[1]])
        assert all(item["name"] != "prioritize_alpha" for item in loaded["core"])
    finally:
        tmp.cleanup()


def test_validate_autocomplete_rejects_unfilled_placeholders():
    # Observed live: the model ran ':steer law [famA] [famB] [k|off]' -- a
    # copied usage line. Placeholder tokens can only ever end in a usage
    # error, so the validator must refuse to suggest them.
    for bad in (
        ":steer law [famA] [famB] [k|off]",
        ":steer mass <probe|choose> <m|auto>",
        ":probe adopt <dim> ...",
    ):
        candidate, error = validate_command_autocomplete(bad)
        assert candidate is None, bad
        assert "placeholder" in error
    good, error = validate_command_autocomplete(":steer law integrity chaos -0.3")
    assert error is None
    assert good == ":steer law integrity chaos -0.3"


def test_field_sources_bare_names_resolve_and_collisions_demand_qualifiers():
    # Grammar rule: symbols only when necessary. A bare source name resolves
    # to the single namespace that owns it (probe/knob/status/family); a
    # real collision errors and demands the qualified form -- sources exert
    # force, so a collision is never guessed. The structurally different
    # kinds carry sigils: #N layer, ^trigger lift, *trigger outcome, ~family.
    tmp = tempfile.TemporaryDirectory()
    try:
        tuner = TriggerTuner(path=Path(tmp.name) / "tuner.json")
        tuner.register("steer_cap_fraction", 0.2, kind="coefficient")
        tuner.register("memory_alpha", 0.3, kind="coefficient")
        tuner.register("cmd_steer", 0.0)
        probes = {
            "self_model": {"history": [0.1, 0.4]},
            "memory_alpha": {"history": [0.2, 0.3]},
        }
        families = {"self_axis": {"g": {"kind": "constant", "value": 0.2, "scale": 1.0, "offset": 0.0}}}

        def parsed(raw):
            spec = parse_field_source(raw)
            assert spec is not None, raw
            return spec, field_source_error(spec, probes, tuner, families)

        for raw, kind in (
            ("self_model", "probe"),
            ("steer_cap_fraction", "knob"),
            ("ram", "status"),
            ("self_axis", "family"),
        ):
            spec, err = parsed(raw)
            assert err is None, f"{raw}: {err}"
            assert spec["kind"] == kind, (raw, spec)

        spec, err = parsed("memory_alpha")  # probe AND knob
        assert err and "ambiguous" in err
        assert "probe:memory_alpha" in err and "knob:memory_alpha" in err
        for raw, kind in (("probe:memory_alpha", "probe"), ("knob:memory_alpha", "knob")):
            spec, err = parsed(raw)
            assert err is None and spec["kind"] == kind

        for raw, kind, name in (
            ("#16", "layer", "16"),
            ("~self_axis", "family", "self_axis"),
            ("^cmd_steer", "lift", "cmd_steer"),
            ("*cmd_steer", "outcome", "cmd_steer"),
        ):
            spec, err = parsed(raw)
            assert err is None, f"{raw}: {err}"
            assert (spec["kind"], spec["name"]) == (kind, name)

        _, err = parsed("wobble")
        assert "no probe, knob, status, or family named 'wobble'" in err

        knob_spec = parse_field_source("steer_cap_fraction", 2.0, 0.1)
        assert field_source_error(knob_spec, probes, tuner, families) is None
        assert abs(resolve_field_source(knob_spec, probes, tuner, families, {}) - 0.5) < 1e-9
        fam_spec = parse_field_source("~self_axis", 0.5)
        assert field_source_error(fam_spec, probes, tuner, families) is None
        assert abs(resolve_field_source(fam_spec, probes, tuner, families, {}) - 0.1) < 1e-9
    finally:
        tmp.cleanup()


def test_expect_autocomplete_prefix_survives_colon_splitter():
    # Regression: ':expect autocomplete :steer pole' (queued by ':suggest
    # :<prefix>') was split on the mid-line ' :' into ':expect autocomplete'
    # -- the expectation registered with an empty prefix ('') -- plus a stray
    # ':steer pole' appended to the input queue, which ran bare after the
    # model turn and printed '[Shell] missing body'.
    line = ":expect autocomplete :steer pole"
    assert command_takes_colon_command_arg(line)
    kept = _split_macro_commands(
        line, split_colon_commands=not command_takes_colon_command_arg(line)
    )
    assert kept == [line], f"the :expect line was shredded: {kept}"
    eargs = line.split()
    assert " ".join(eargs[2:]) == ":steer pole"


def test_validate_autocomplete_rejects_completion_failing_its_own_spec():
    # Generation awareness: a suggested completion is parsed by the SAME
    # CommandSpec that will execute it on :accept. Observed live: the model
    # completed ':steer pole accuracy_d4096+' (sign on the probe, family
    # missing) and the shell suggested it anyway; the usage error surfaced
    # only when the command ran.
    completion, error = validate_command_autocomplete(
        ":steer pole accuracy_d4096+",
        ":steer pole",
        known_commands={"steer", "probe", "tune"},
        probe_names={"accuracy_d4096"},
        knob_names=set(),
    )
    assert completion is None
    assert "usage: :steer pole" in error

    completion, error = validate_command_autocomplete(
        ":steer pole accuracy_d4096 warmth+",
        ":steer pole",
        known_commands={"steer", "probe", "tune"},
        probe_names={"accuracy_d4096"},
        knob_names=set(),
    )
    assert error is None
    assert completion == ":steer pole accuracy_d4096 warmth+"

    # Spec-level shorthand (the pre hook) must not be rejected: a bare
    # number on :steer field means 'on <G>'.
    completion, error = validate_command_autocomplete(
        ":steer field 0.1",
        ":steer field",
        known_commands={"steer"},
        probe_names=set(),
        knob_names=set(),
    )
    assert error is None
    assert completion == ":steer field 0.1"


def test_steer_law_families_accept_choose_through_spec():
    picks = iter(["integrity", "chaos"])
    ns = STEER_LAW_SPEC.parse(
        "choose choose -0.3",
        choice_resolver=lambda tok: next(picks),
    )
    assert (ns.famA, ns.famB, ns.k) == ("integrity", "chaos", "-0.3")
    # Concrete family names pass through without touching the resolver.
    ns = STEER_LAW_SPEC.parse(
        "integrity chaos off",
        choice_resolver=lambda tok: (_ for _ in ()).throw(AssertionError("resolver called")),
    )
    assert (ns.famA, ns.famB, ns.k) == ("integrity", "chaos", "off")


def test_watch_choose_pool_lists_probes_and_knobs_not_bars():
    tmp = tempfile.TemporaryDirectory()
    try:
        tuner = TriggerTuner(path=Path(tmp.name) / "tuner.json")
        tuner.register("prioritize_alpha", 0.1, kind="coefficient")
        tuner.observe("probe_self_model", 0.25)  # a per-probe bar
        probes = {"self_model": {"history": [0.1, 0.4]}}

        pool = watch_choose_pool(probes, tuner)
        assert "self_model" in pool          # the probe itself is pickable
        assert "prioritize_alpha" in pool     # a knob/stream is pickable
        assert all(not p.startswith("probe_") for p in pool)  # bars excluded
        assert len(pool) == len(set(pool))    # order-preserving + deduped
        # Empty state is safe (choose then simply has nothing to offer).
        assert watch_choose_pool({}, None) == []
    finally:
        tmp.cleanup()


def test_resolve_probe_choice_passthrough_leaves_concrete_names_without_model():
    # The four manual handlers (:probe expose, :place, :label, :group items) now
    # pass model/config so choose/auto/choice can reach the picker. That is only
    # safe because a CONCRETE token returns unchanged WITHOUT touching the model
    # -- otherwise wiring it in would regress the common (named-target) path.
    assert resolve_probe_choice("self_model", {"self_model": {}}) == "self_model"
    assert resolve_probe_choice("probe_self_model", {}) == "self_model"
    assert resolve_probe_choice("", {}) == ""
    # A choose token with no model/options resolves to nothing (never raises).
    assert resolve_probe_choice("choose", {}) is None


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
    test_named_backfill_scores_only_named_probe,
    test_suggest_command_catalog_lists_all_visible_user_facing_commands,
    test_macro_queue_treats_hash_lines_as_comments,
    test_macro_safe_block_quotes_columnar_grammar,
    test_suggest_autocomplete_prompt_queues_one_command_one_text_turn,
    test_suggest_command_goal_prompt_queues_one_command_one_text_turn,
    test_suggest_goal_request_accepts_optional_prefix_after_goal_words,
    test_macro_create_at_source_loads_existing_profile_without_text_turn,
    test_expect_macro_uses_low_memory_utility_profile_and_restores_config,
    test_expected_macro_parts_preserves_args_comment_without_running_comments,
    test_expose_choose_resolves_through_command_spec_class,
    test_validate_autocomplete_accepts_expose_choose,
    test_validate_autocomplete_rejects_unfilled_placeholders,
    test_field_sources_bare_names_resolve_and_collisions_demand_qualifiers,
    test_expect_autocomplete_prefix_survives_colon_splitter,
    test_validate_autocomplete_rejects_completion_failing_its_own_spec,
    test_steer_law_families_accept_choose_through_spec,
    test_watch_group_items_resolve_probes_knobs_and_status_aliases,
    test_watch_groups_round_trip_and_render_small_groups,
    test_watch_choose_pool_lists_probes_and_knobs_not_bars,
    test_resolve_probe_choice_passthrough_leaves_concrete_names_without_model,
]


def main():
    print("MEMORY ENGINE TEST -- explicit tool, not hidden prompt context\n")
    for test in TESTS:
        test()
        print(f"  PASS {test.__name__}")
    print("\n  Memory records persist, retrieval is explicit, and prompt use is one-turn only.")


if __name__ == "__main__":
    main()
