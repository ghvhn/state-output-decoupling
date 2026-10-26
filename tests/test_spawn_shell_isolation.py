"""Spawn boundaries exercise the real shell code without loading its checkpoint."""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import interactive_phenomenality as shell


def isolated_functions(functions, values, namespace):
    """Give nested production helpers real shared lexical bindings."""
    names = sorted(values)
    initialized = [ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.Subscript(value=ast.Name(id="values", ctx=ast.Load()),
                            slice=ast.Constant(value=name), ctx=ast.Load()),
    ) for name in names]
    snapshot = ast.FunctionDef(name="snapshot", decorator_list=[],
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[ast.Return(value=ast.Dict(
            keys=[ast.Constant(value=name) for name in names],
            values=[ast.Name(id=name, ctx=ast.Load()) for name in names]))])
    factory = ast.FunctionDef(name="factory", decorator_list=[],
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="values")],
                           kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=initialized + [copy.deepcopy(fn) for fn in functions] + [snapshot, ast.Return(
            value=ast.Tuple(elts=[*[ast.Name(id=fn.name, ctx=ast.Load()) for fn in functions],
                                  ast.Name(id="snapshot", ctx=ast.Load())], ctx=ast.Load()))])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[])),
                 shell.__file__, "exec"), namespace)
    return namespace["factory"](values)


class SpawnShellIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(Path(shell.__file__).read_text(encoding="utf-8"))

    def nested(self, name):
        return next(node for node in ast.walk(self.tree)
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    def branch(self, condition):
        return next(node for node in ast.walk(self.tree)
                    if isinstance(node, ast.If) and ast.unparse(node.test) == condition)

    def run_statements(self, body, namespace):
        # Preserve command-level continue statements without entering main().
        module = ast.Module(body=[ast.For(
            target=ast.Name(id="_test_iteration", ctx=ast.Store()),
            iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
            body=copy.deepcopy(body), orelse=[])], type_ignores=[])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(module), shell.__file__, "exec"), namespace)

    def test_identifiers_do_not_implicitly_request_traits(self):
        for command in (":spawn careful_reader join", ":spawn base replace 10"):
            request = shell.parse_spawn_request(command)
            self.assertFalse(request["profile_requested"])
            self.assertIsNone(request["profile_name"])
            self.assertIsNone(request["traits"])
        self.assertEqual(shell.parse_spawn_request(":spawn base replace 10")["turns"], 10)

    def test_name_and_traits_clauses_are_explicit_profile_requests(self):
        request = shell.parse_spawn_request(":spawn researcher replace 3 name careful_reader")
        self.assertEqual((request["name"], request["turns"], request["profile_name"]),
                         ("researcher", 3, "careful_reader"))
        self.assertTrue(request["profile_requested"])
        request = shell.parse_spawn_request(":spawn researcher join traits concise and skeptical")
        self.assertEqual(request["traits"], "concise and skeptical")
        self.assertTrue(request["profile_requested"])

    def test_bad_counts_and_incomplete_clauses_are_rejected_before_creation(self):
        for command in (":spawn x replace 0", ":spawn x replace -2", ":spawn x join traits",
                        ":spawn x join name", ":spawn x join unknown", ":spawn x invalid"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                shell.parse_spawn_request(command)

    def test_base_cannot_be_given_a_starting_profile(self):
        for command in (":spawn base join name careful", ":spawn BASE replace traits kind",
                        ":spawn base create"):
            with self.subTest(command=command), self.assertRaises(ValueError):
                shell.parse_spawn_request(command)

    def tokenizer_bundle(self):
        tokenizer = SimpleNamespace(apply_chat_template=Mock(return_value="native conversation"))
        return SimpleNamespace(tok=tokenizer), tokenizer

    def test_base_prompt_uses_native_actual_messages_without_reports_or_attribution(self):
        model, tokenizer = self.tokenizer_bundle()
        context = [("user", "operator", "What is quartz?", 0.1),
                   ("assistant", "persona_name", "A mineral.", 0.2)]
        self.assertEqual(shell.build_base_agent_prompt(model, context), "native conversation")
        tokenizer.apply_chat_template.assert_called_once_with([
            {"role": "user", "content": "What is quartz?"},
            {"role": "assistant", "content": "A mineral."},
        ], tokenize=False, add_generation_prompt=True)

    def test_replacement_base_prompt_swaps_sides_and_preserves_system_role(self):
        model, tokenizer = self.tokenizer_bundle()
        shell.build_base_agent_prompt(model, [("system", "system", "Source: quartz"),
            ("user", "operator", "Question"), ("assistant", "assistant", "Answer")],
            replacing_user=True)
        self.assertEqual(tokenizer.apply_chat_template.call_args.args[0], [
            {"role": "system", "content": "Source: quartz"},
            {"role": "assistant", "content": "Question"},
            {"role": "user", "content": "Answer"}])

    def test_base_reader_source_is_only_the_actual_staged_passage(self):
        model, tokenizer = self.tokenizer_bundle()
        shell.build_base_agent_prompt(model, [], source="Source: actual document passage")
        self.assertEqual(tokenizer.apply_chat_template.call_args.args[0],
                         [{"role": "user", "content": "Source: actual document passage"}])

    def spawn_namespace(self, root, command, because=None):
        agent = shell.AgentState(name=shell.parse_spawn_request(command)["name"],
                                 tuner=SimpleNamespace(triggers={}))
        namespace = dict(vars(shell), ROOT=str(root), user_input=command, command_because=because,
            tuner=SimpleNamespace(triggers={}), joined_agents=[], replace_agent=None,
            replace_turns_remaining=0, input_queue=[], macro_aliases={},
            hidden_commands=set(), SPAWN_FAKE_KNOBS=set(), _new_spawn_agent=Mock(return_value=agent),
            _spawn_profile_runnable_lines=Mock(return_value=([":tune response_tokens 100"], [])),
            _spawn_state_report=Mock(return_value="profile report"),
            render_visible_commands_md=Mock(return_value="command reference"),
            load_prompt=Mock(side_effect=lambda _key, prompt, **_kwargs: prompt),
            queue_macro_text=Mock())
        return namespace, agent

    def test_plain_spawn_bypasses_even_an_existing_same_name_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            macro = root / "invariants" / "out" / "macros" / "careful_reader.txt"
            macro.parent.mkdir(parents=True)
            macro.write_text(":tune response_tokens 100", encoding="utf-8")
            ns, agent = self.spawn_namespace(root, ":spawn careful_reader join")
            self.run_statements(self.branch("user_input.startswith(':spawn')").body, ns)
            self.assertEqual(ns["joined_agents"], [agent])
            self.assertFalse(agent.profile_requested)
            self.assertEqual(ns["input_queue"], [])
            ns["_spawn_profile_runnable_lines"].assert_not_called()
            ns["load_prompt"].assert_not_called()
            ns["queue_macro_text"].assert_not_called()

    def test_base_replacement_starts_immediately_without_profile_file(self):
        with tempfile.TemporaryDirectory() as directory:
            ns, agent = self.spawn_namespace(Path(directory), ":spawn base replace 10")
            self.run_statements(self.branch("user_input.startswith(':spawn')").body, ns)
            self.assertIs(ns["replace_agent"], agent)
            self.assertEqual(ns["replace_turns_remaining"], 10)
            ns["load_prompt"].assert_not_called()
            ns["queue_macro_text"].assert_not_called()

    def test_explicit_saved_name_queues_commands_for_the_new_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            macro = root / "invariants" / "out" / "macros" / "careful.txt"
            macro.parent.mkdir(parents=True)
            macro.write_text(":tune response_tokens 100", encoding="utf-8")
            ns, agent = self.spawn_namespace(root, ":spawn worker join name careful")
            self.run_statements(self.branch("user_input.startswith(':spawn')").body, ns)
            self.assertTrue(agent.profile_requested)
            self.assertEqual(ns["input_queue"], [("@worker :tune response_tokens 100", True)])
            self.assertEqual(ns["joined_agents"], [agent])
            ns["load_prompt"].assert_not_called()

    def test_explicit_traits_and_because_clauses_opt_into_profile_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            for command, because, declaration in (
                (":spawn worker join traits precise and curious", None, "precise and curious"),
                (":spawn worker join", "careful with uncertainty", "careful with uncertainty"),
            ):
                ns, _agent = self.spawn_namespace(Path(directory), command, because)
                self.run_statements(self.branch("user_input.startswith(':spawn')").body, ns)
                self.assertIn(declaration, ns["queue_macro_text"].call_args.args[0])
                self.assertEqual(ns["joined_agents"], [])

    def test_because_clause_cannot_override_reserved_base(self):
        with tempfile.TemporaryDirectory() as directory:
            ns, _agent = self.spawn_namespace(Path(directory), ":spawn base join", "be kind")
            self.run_statements(self.branch("user_input.startswith(':spawn')").body, ns)
            ns["_new_spawn_agent"].assert_not_called()
            ns["queue_macro_text"].assert_not_called()

    def generation_namespace(self):
        model, tokenizer = self.tokenizer_bundle()
        primary = shell.AgentState("main", tuner={"response_tokens": 128})
        primary.probes = {"main_only": {"direction": {0: "main vector"}}}
        primary.memory = Mock()
        agent = shell.AgentState("base", tuner={"response_tokens": 64})
        agent.overlay = SimpleNamespace(config=SimpleNamespace(base_model_only=True,
                                                               lookahead_processor=None))
        namespace = dict(vars(shell), model=model, config=SimpleNamespace(base_model_only=False),
            memory=primary.memory, lookahead_apparatus=None,
            _agent_generation_restore=None, replace_agent=agent, replace_turns_remaining=2,
            joined_agents=[agent], session_context_enabled=True,
            session_context=[("user", "user", "Question", 0.0),
                             ("assistant", "assistant", "Answer", 0.0)],
            last_assistant_response="Answer", response="Answer", user_input="Question",
            _spawn_state_report=Mock(side_effect=AssertionError("unexpected persona report")),
            _build_live_prompt=Mock(return_value="unused shell prompt"),
            _priority_steer_handles=Mock(return_value=([], None)), _document_reader=Mock(return_value=None),
            generate_agentic_text=Mock(return_value="Next conversation turn"))
        fields = ("tuner", "probes", "tuner_bindings", "prioritize_pin", "queued_calibrations",
                  "queued_commands", "doc_session", "doc_library", "pending_document_tool_result", "doc_autoread")
        namespace.update({name: getattr(primary, name) for name in fields})
        calls = []
        def activate(target, restore):
            calls.append(("activate", target.name))
            self.assertIs(namespace["probes"], target.probes)
            restore["test_config"] = namespace["config"]
            namespace["config"] = target.overlay.config
        def restore(saved):
            calls.append(("restore", saved["agent"].name))
            self.assertIs(namespace["probes"], primary.probes)
            namespace["config"] = saved.pop("test_config")
        namespace.update(_activate_agent_overlay=activate, _restore_agent_overlay=restore)
        return namespace, primary, agent, tokenizer, calls

    def test_real_replace_branch_activates_base_then_restores_primary(self):
        ns, primary, agent, tokenizer, calls = self.generation_namespace()
        self.run_statements(self.branch("replace_turns_remaining > 0 and replace_agent").body, ns)
        self.assertEqual(calls, [("activate", "base"), ("restore", "base")])
        self.assertIs(ns["probes"], primary.probes)
        self.assertEqual(ns["replace_turns_remaining"], 1)
        self.assertIsNone(ns["_agent_generation_restore"])
        self.assertEqual(ns["generate_agentic_text"].call_args.kwargs["instruction"], "native conversation")
        self.assertIs(ns["generate_agentic_text"].call_args.kwargs["config"], agent.overlay.config)
        self.assertEqual(tokenizer.apply_chat_template.call_args.args[0][-1]["role"], "user")
        ns["_spawn_state_report"].assert_not_called()

    def test_real_join_branch_activates_base_then_restores_primary(self):
        ns, primary, agent, tokenizer, calls = self.generation_namespace()
        loop = next(node for node in ast.walk(self.tree) if isinstance(node, ast.For)
                    and ast.unparse(node.target) == "ja" and ast.unparse(node.iter) == "joined_agents")
        ns["ja"] = agent
        self.run_statements(loop.body, ns)
        self.assertEqual(calls, [("activate", "base"), ("restore", "base")])
        self.assertIs(ns["probes"], primary.probes)
        self.assertIsNone(ns["_agent_generation_restore"])
        self.assertIs(ns["generate_agentic_text"].call_args.kwargs["config"], agent.overlay.config)
        self.assertEqual(tokenizer.apply_chat_template.call_args.args[0][-1]["role"], "assistant")
        ns["_spawn_state_report"].assert_not_called()

    def test_real_target_command_and_next_loop_restore_use_owner_boundaries(self):
        ns, primary, agent, _tokenizer, calls = self.generation_namespace()
        ns.update(target_agent=agent, _target_restore=None)
        self.run_statements(self.branch("target_agent").body, ns)
        self.assertIs(ns["probes"], agent.probes)
        agent_change = {"learned_probe": {}}
        ns["probes"] = agent_change
        self.run_statements(self.branch("_target_restore is not None").body, ns)
        self.assertIs(agent.probes, agent_change)
        self.assertIs(ns["probes"], primary.probes)
        self.assertIsNone(ns["_target_restore"])
        self.assertEqual(calls, [("activate", "base"), ("restore", "base")])

    def test_real_failure_recovery_restores_overlay_and_local_bindings(self):
        ns, primary, agent, _tokenizer, calls = self.generation_namespace()
        ns["generate_agentic_text"].side_effect = RuntimeError("generation failed")
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            self.run_statements(self.branch("replace_turns_remaining > 0 and replace_agent").body, ns)
        saved = ns["_agent_generation_restore"]
        # Recovery must invoke its real overlay restoration after the real
        # generation branch has left the agent active.
        saved["overlay_restore"] = {"active": True}
        helper = self.nested("_pause_swapped_reader")
        names = {name for node in ast.walk(helper) if isinstance(node, ast.Nonlocal)
                 for name in node.names}
        def restored_overlay(restore):
            calls.append(("restore", restore["agent"].name))
        recover, snapshot = isolated_functions([helper], {name: ns[name] for name in names},
                                              dict(vars(shell), _restore_agent_overlay=restored_overlay))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(recover(saved))
        self.assertIs(snapshot()["probes"], primary.probes)
        self.assertIs(snapshot()["tuner"], primary.tuner)
        self.assertIsNone(agent.doc_autoread)
        self.assertEqual(calls, [("activate", "base"), ("restore", "base")])

    def test_wrapper_reads_current_agent_probes_instead_of_main_probes(self):
        agent_direction = {2: "agent only direction"}
        with patch.object(shell, "_ACTIVE_PROBES", {"agent": {"direction": agent_direction}}), \
             patch.object(shell, "_engine_generate", return_value="output") as generate:
            shell.generate_agentic_text(None)
        self.assertEqual(generate.call_args.kwargs["vecs"], {"agent": agent_direction})

    def overlay_roundtrip(self, save_error=None):
        activate = self.nested("_activate_agent_overlay")
        restore = self.nested("_restore_agent_overlay")
        names = {name for fn in (activate, restore) for node in ast.walk(fn)
                 if isinstance(node, ast.Nonlocal) for name in node.names}
        primary = SimpleNamespace(probes={"primary": {}}, tuner=object())
        active_probes, active_tuner = {"agent": {}}, object()
        initial = {name: object() for name in names}
        initial.update(tool_sense=SimpleNamespace(tools=[object()]), cen_enabled=True,
                       tuner=active_tuner, probes=active_probes)
        events = []
        @contextlib.contextmanager
        def owned_context():
            events.append("enter")
            try:
                yield
            finally:
                events.append("exit")
        overlay = SimpleNamespace(
            directory=Path("isolated_agent"), config=object(), synaptic=object(),
            lookahead=object(), lookahead_path=Path("isolated_agent/lookahead.pt"),
            memory=object(), steer_map=object(), self_concept=object(),
            executive_owner=object(), shell_state={},
            activate=owned_context, save=Mock(side_effect=save_error))
        agent = SimpleNamespace(overlay=overlay)
        namespace = dict(vars(shell), model=object(), primary=primary,
                         PROBE_RAW_HISTORY_PATH="main/history.json", _sync_steer_tunables=Mock(),
                         _ACTIVE_PROBE_HISTORY_PATH="main/custom-history.json",
                         ToolSense=lambda _model, tuner: SimpleNamespace(tuner=tuner, tools=[]))
        switch_locals = ast.parse(
            "def switch_locals():\n"
            "    nonlocal probes, tuner\n"
            "    probes, tuner = primary.probes, primary.tuner\n").body[0]
        activate_fn, restore_fn, switch_fn, snapshot = isolated_functions(
            [activate, restore, switch_locals], initial, namespace)
        saved = {"agent": agent}
        activate_fn(agent, saved)
        activated = snapshot()
        for name in ("config", "synaptic_engine", "lookahead_apparatus", "memory", "steer_map", "self_concept"):
            attribute = {"synaptic_engine": "synaptic", "lookahead_apparatus": "lookahead"}.get(name, name)
            self.assertIs(activated[name], getattr(overlay, attribute))
        self.assertEqual(activated["PROBE_DIR"], str(overlay.directory / "probes"))
        self.assertFalse(activated["cen_enabled"])
        self.assertIs(namespace["_ACTIVE_PROBES"], active_probes)
        self.assertIs(namespace["_ACTIVE_TUNER"], active_tuner)
        self.assertEqual(namespace["_ACTIVE_PROBE_HISTORY_PATH"],
                         str(overlay.directory / "probe_raw_histories.json"))
        self.assertIs(activated["tool_sense"].tuner, active_tuner)
        self.assertEqual(activated["tool_sense"].tools, initial["tool_sense"].tools)
        # Normal production branches restore lexical tuner/probes before the
        # overlay helper restores the corresponding global routing bindings.
        switch_fn()
        with contextlib.redirect_stdout(io.StringIO()):
            restore_fn(saved)
        restored = snapshot()
        for name in names:
            self.assertIs(restored[name], initial[name], name)
        self.assertIs(namespace["_ACTIVE_PROBES"], primary.probes)
        self.assertIs(namespace["_ACTIVE_TUNER"], primary.tuner)
        self.assertEqual(namespace["_ACTIVE_PROBE_HISTORY_PATH"], "main/custom-history.json")
        self.assertNotIn("overlay_restore", saved)
        self.assertEqual(events, ["enter", "exit"])
        overlay.save.assert_called_once_with()
        namespace["_sync_steer_tunables"].assert_any_call(active_tuner, overlay.config)
        namespace["_sync_steer_tunables"].assert_any_call(primary.tuner, initial["config"])

    def test_actual_overlay_helpers_switch_every_owner_and_global_probe_path(self):
        self.overlay_roundtrip()

    def test_actual_overlay_helpers_restore_primary_even_when_agent_save_fails(self):
        self.overlay_roundtrip(OSError("disk temporarily unavailable"))

    def postgeneration_request_statements(self):
        for node in ast.walk(self.tree):
            for _field, siblings in ast.iter_fields(node):
                if not isinstance(siblings, list):
                    continue
                for start, statement in enumerate(siblings):
                    if (isinstance(statement, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "_generation_tools_disabled"
                        for target in statement.targets
                    )):
                        end = next(index for index in range(start + 1, len(siblings))
                                   if isinstance(siblings[index], ast.If)
                                   and ast.unparse(siblings[index].test) ==
                                   'not _generation_tools_disabled and help_query is not None')
                        return siblings[start:end + 1]
        self.fail("Could not find the production postgeneration request boundary")

    def request_namespace(self, *, base=False, utility=False):
        output = ("<<MEMORY: stored claim>> <<CLAIMMAP: fact>> <<METHOD: solve>> "
                  "<<DOC: source>> <<PROBE: careful>> <<CMD: tune response_tokens 100>> "
                  "<<GAME_PROPOSE: chess>> <<GAME_ACCEPT: chess>> <<GAME_DECLINE: chess>> "
                  "<<GAME_END>> <<GAME_EXPOSE: careful>> <<GAME_HIDE: fast>> "
                  "<<HELP: :spawn>>\n```python\nprint('literal code')\n```")
        extractor_names = ("extract_memory_query", "extract_claimmap_payload", "extract_methodmap_query",
                           "extract_doc_query", "extract_probe_query", "extract_cmd_requests",
                           "extract_game_propose", "extract_game_accept", "extract_game_decline",
                           "extract_game_end", "extract_game_expose_hide", "extract_help_request")
        namespace = dict(vars(shell), response=output, config=SimpleNamespace(base_model_only=base),
            utility_expect_turn=utility, pending_help_tool_result="existing help",
            exposed_commands={}, exposed_knobs={}, hidden_commands=set(), memory_tool_exposed=False,
            _enabled_memory_lanes=Mock(return_value=[]), list_solve_macros=Mock(return_value=[]),
            command_help_tool_result=Mock(return_value="requested help"))
        extractors = {name: Mock(wraps=getattr(shell, name)) for name in extractor_names}
        namespace.update(extractors)
        return namespace, extractors

    def test_base_and_utility_outputs_never_become_requests_even_with_literal_tool_tags(self):
        for base, utility in ((True, False), (False, True), (True, True)):
            with self.subTest(base=base, utility=utility):
                ns, extractors = self.request_namespace(base=base, utility=utility)
                self.run_statements(self.postgeneration_request_statements(), ns)
                for name, extractor in extractors.items():
                    extractor.assert_not_called()
                for name in ("model_memory_query", "model_claimmap_payload", "model_methodmap_query",
                             "model_doc_query", "model_probe_query", "model_game_propose",
                             "model_game_accept", "model_game_decline", "model_game_end", "help_query"):
                    self.assertIsNone(ns[name], name)
                self.assertEqual(ns["model_cmd_requests"], [])
                self.assertEqual(ns["model_game_exposed"], [])
                self.assertEqual(ns["model_game_hidden"], [])
                self.assertEqual(ns["pending_help_tool_result"], "existing help")
                ns["command_help_tool_result"].assert_not_called()

    def test_ordinary_speech_still_parses_and_serves_explicit_tool_requests(self):
        ns, extractors = self.request_namespace()
        self.run_statements(self.postgeneration_request_statements(), ns)
        for extractor in extractors.values():
            extractor.assert_called_once_with(ns["response"])
        self.assertEqual(ns["model_memory_query"], "stored claim")
        self.assertEqual(ns["model_claimmap_payload"], "fact")
        self.assertEqual(ns["model_doc_query"], "source")
        self.assertEqual(ns["model_cmd_requests"], [":tune response_tokens 100"])
        self.assertEqual(ns["model_game_exposed"], ["careful"])
        self.assertEqual(ns["model_game_hidden"], ["fast"])
        self.assertTrue(ns["model_game_end"])
        self.assertEqual(ns["pending_help_tool_result"], "requested help")
        ns["command_help_tool_result"].assert_called_once()

    def test_sandbox_predicate_skips_base_and_utility_outputs(self):
        branch = self.branch("sandbox_enabled and (not utility_expect_turn) and (not getattr(config, 'base_model_only', False))")
        # The predicate and extraction statement come from production. Leave
        # execution out of this parser boundary test, so no generated code runs.
        boundary = copy.deepcopy(branch)
        boundary.body = boundary.body[:1]
        for base, utility, enabled, expected in (
            (True, False, True, False), (False, True, True, False),
            (False, False, False, False), (False, False, True, True),
        ):
            with self.subTest(base=base, utility=utility, enabled=enabled):
                ns, _extractors = self.request_namespace(base=base, utility=utility)
                ns.update(sandbox_enabled=enabled,
                          extract_python_block=Mock(wraps=shell.extract_python_block))
                self.run_statements([boundary], ns)
                if expected:
                    ns["extract_python_block"].assert_called_once_with(ns["response"])
                    self.assertIn("print('literal code')", ns["sandbox_code"])
                else:
                    ns["extract_python_block"].assert_not_called()
                    self.assertNotIn("sandbox_code", ns)

    def test_targeted_base_generation_removes_outer_steering_and_uses_native_prompt(self):
        branch = next(node for node in ast.walk(self.tree)
                      if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "getattr(config, 'base_model_only', False)"
                      and any(isinstance(child, ast.Name) and child.id == "steer_handles"
                              for child in ast.walk(node)))
        model, tokenizer = self.tokenizer_bundle()
        handles = [SimpleNamespace(remove=Mock()), SimpleNamespace(remove=Mock())]
        ns = dict(vars(shell), config=SimpleNamespace(base_model_only=True), model=model,
                  steer_handles=handles.copy(), prompt="shell prompt with internal field context",
                  session_context_enabled=True, session_context=[("user", "user", "Earlier question")],
                  user_input="Current question")
        self.run_statements([branch], ns)
        for handle in handles:
            handle.remove.assert_called_once_with()
        self.assertEqual(ns["steer_handles"], [])
        self.assertEqual(ns["prompt"], "native conversation")
        self.assertEqual(tokenizer.apply_chat_template.call_args.args[0], [
            {"role": "user", "content": "Earlier question"},
            {"role": "user", "content": "Current question"}])


if __name__ == "__main__":
    unittest.main()
