"""Reader working memory and interrupted agent phases survive the shell seam."""

import ast
import copy
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import interactive_phenomenality as shell


def nested_function(tree, name):
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def isolated_closure(function, values, namespace, extra_functions=()):
    """Run the real nested function in an explicit lexical state, without UI/model."""
    functions = [function, *extra_functions]
    names = sorted(set(name for fn in functions for node in ast.walk(fn)
                       if isinstance(node, ast.Nonlocal) for name in node.names))
    initialized = [ast.Assign(
        targets=[ast.Name(id=name, ctx=ast.Store())],
        value=ast.Subscript(value=ast.Name(id="values", ctx=ast.Load()),
                            slice=ast.Constant(value=name), ctx=ast.Load()),
    ) for name in names]
    snapshot = ast.FunctionDef(
        name="snapshot", args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]), decorator_list=[], body=[ast.Return(
            value=ast.Dict(keys=[ast.Constant(value=name) for name in names],
                           values=[ast.Name(id=name, ctx=ast.Load()) for name in names]))],
    )
    factory = ast.FunctionDef(
        name="factory", args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg="values")], kwonlyargs=[], kw_defaults=[], defaults=[]),
        decorator_list=[], body=initialized + [copy.deepcopy(fn) for fn in functions] + [snapshot,
        ast.Return(value=ast.Tuple(elts=[ast.Name(id=function.name, ctx=ast.Load()),
                                       ast.Name(id="snapshot", ctx=ast.Load()),
                                       *[ast.Name(id=fn.name, ctx=ast.Load()) for fn in extra_functions]], ctx=ast.Load()))],
    )
    module = ast.Module(body=[factory], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), shell.__file__, "exec"), namespace)
    return namespace["factory"](values)


class ReaderRuntimeRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(Path(shell.__file__).read_text(encoding="utf-8"))

    def prompt_builder(self):
        function = nested_function(self.tree, "_build_live_prompt")
        from invariants.hebbian_plasticity import PlasticityOwners
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        tuner = SimpleNamespace(path=Path(directory.name) / 'tuner.json', triggers={})
        namespace = dict(vars(shell), tuner=tuner, probes={},
                         plasticity_owners=PlasticityOwners(), _other_mint=lambda _pair: {},
                         _reader_context=lambda: [("SOURCE: granite", "Granite contains quartz.")],
                         format_field_prompt_context=lambda *_args: "")
        module = ast.Module(body=[copy.deepcopy(function)], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), shell.__file__, "exec"), namespace)
        return namespace[function.name]

    def test_reader_evidence_reaches_speech_when_transcript_context_is_disabled(self):
        prompt = self.prompt_builder()("Which mineral?", session_context=None)
        self.assertIn("SOURCE: granite", prompt)
        self.assertIn("[reader] Granite contains quartz.", prompt)
        self.assertIn("Which mineral?", prompt)
        self.assertEqual(prompt.count(shell.LLAMA3_START + "user" + shell.LLAMA3_END), 1)

    def test_isolated_utility_can_explicitly_omit_reader_context(self):
        prompt = self.prompt_builder()("Complete this command", reader_context=None)
        self.assertNotIn("SOURCE: granite", prompt)
        self.assertNotIn("[reader]", prompt)
        self.assertIn("Complete this command", prompt)

    def test_current_document_still_receives_reader_memory_without_fake_operator_turn(self):
        prompt = self.prompt_builder()("", document_tool_result="SOURCE: mica", session_context=None)
        self.assertIn("SOURCE: granite", prompt)
        self.assertIn("SOURCE: mica", prompt)
        self.assertNotIn(shell.LLAMA3_START + "user" + shell.LLAMA3_END, prompt)

    def recovery_state(self):
        helper = nested_function(self.tree, "_pause_swapped_reader")
        names = {name for node in ast.walk(helper) if isinstance(node, ast.Nonlocal)
                 for name in node.names}
        main = {name: object() for name in names}
        main["doc_session"] = {"source_name": "main.md", "read": {0}}
        main["doc_library"] = [main["doc_session"]]
        main["doc_autoread"] = {"remaining": 3}
        main["pending_document_tool_result"] = "main source waiting"
        active = {name: object() for name in names}
        active["doc_session"] = {"source_name": "agent.md", "read": {0}, "_staged_read": {"indices": [1]}}
        active["doc_library"] = [active["doc_session"]]
        active["doc_autoread"] = {"remaining": 2}
        active["pending_document_tool_result"] = "agent source interrupted"
        agent = SimpleNamespace(name="reader-agent")
        return helper, main, active, agent

    def test_swapped_agent_failure_preserves_coverage_and_restores_every_main_binding(self):
        helper, main, active, agent = self.recovery_state()
        recover, snapshot = isolated_closure(helper, active, dict(vars(shell)))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(recover(dict(main, agent=agent)))
        for name, value in main.items():
            self.assertIs(snapshot()[name], value, name)
        for name in active.keys() - {"pending_document_tool_result", "doc_autoread"}:
            self.assertIs(getattr(agent, name), active[name], name)
        self.assertIsNone(agent.doc_autoread)
        self.assertIsNone(agent.pending_document_tool_result)
        self.assertEqual(agent.doc_session["read"], {0})
        self.assertNotIn(1, agent.doc_session["read"])
        self.assertEqual(agent.doc_session["_staged_read"]["indices"], [1])

    def test_no_swapped_agent_leaves_bindings_untouched(self):
        helper, _main, active, _agent = self.recovery_state()
        recover, snapshot = isolated_closure(helper, active, dict(vars(shell)))
        self.assertFalse(recover(None))
        for name, value in active.items():
            self.assertIs(snapshot()[name], value, name)

    def actual_error_dispatch(self, restore):
        helper, main, active, agent = self.recovery_state()
        # Execute the recovery portion of the real outer exception handler;
        # traceback printing and unrelated shell logging are outside this seam.
        handler = next(node for node in ast.walk(self.tree)
                       if isinstance(node, ast.ExceptHandler) and node.name == "_turn_exc")
        names = sorted({name for node in ast.walk(helper) if isinstance(node, ast.Nonlocal)
                        for name in node.names} | {"_agent_generation_restore"})
        dispatch = ast.FunctionDef(name="dispatch", decorator_list=[],
            args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=[ast.Nonlocal(names=names)] + copy.deepcopy(handler.body[:3]))
        active["_agent_generation_restore"] = dict(main, agent=agent) if restore else None
        _recover, snapshot, dispatch_error = isolated_closure(
            helper, active, dict(vars(shell)), extra_functions=[dispatch])
        with contextlib.redirect_stdout(io.StringIO()):
            dispatch_error()
        return main, active, agent, snapshot()

    def test_actual_outer_agent_error_keeps_main_reading_run_available(self):
        main, _active, agent, restored = self.actual_error_dispatch(restore=True)
        self.assertIs(restored["doc_autoread"], main["doc_autoread"])
        self.assertIs(restored["pending_document_tool_result"], main["pending_document_tool_result"])
        self.assertIsNone(restored["_agent_generation_restore"])
        self.assertIsNone(agent.doc_autoread)
        self.assertEqual(agent.doc_session["read"], {0})

    def test_actual_outer_main_error_pauses_without_advancing_unread_source(self):
        _main, active, _agent, restored = self.actual_error_dispatch(restore=False)
        self.assertIsNone(restored["doc_autoread"])
        self.assertIsNone(restored["pending_document_tool_result"])
        self.assertIs(restored["doc_session"], active["doc_session"])
        self.assertEqual(restored["doc_session"]["read"], {0})


if __name__ == "__main__":
    unittest.main()
