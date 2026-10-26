"""Document command regressions using real temporary reading history, no model."""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import sys
import tempfile
import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import interactive_phenomenality as shell
from invariants.document_engine import ingest_document, record_chunk_read, select_next_chunk
from invariants.memory_engine import MemoryEngine


class TestDocumentShellFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(Path(shell.__file__).read_text(encoding="utf-8"))
        cls.branch_cache = {}

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "reading.md"
        paragraphs = []
        for index in range(25):
            topic = "photosynthesis chlorophyll sunlight" if index == 17 else "geology minerals rocks"
            paragraphs.append(f"Section {index}: {topic}. " + "evidence " * 15)
        self.path.write_text("\n\n".join(paragraphs), encoding="utf-8")
        self.memory = MemoryEngine(path=self.root / "memory.jsonl", scope="reading_test")
        self.session = self.reload_document()
        self.library = [self.session]
        self.assertGreater(self.session["chunk_count"], 20)

    def reload_document(self):
        return ingest_document(self.memory, self.path, why="review the evidence", max_chars=200)

    def request(self, text="read"):
        request, error = shell.parse_doc_read_request(text, [])
        self.assertIsNone(error)
        return request

    def configure(self, text="read", context="", restart=False):
        return shell.configure_document_reading(
            self.memory, self.library, self.request(text), context=context, restart=restart,
        )

    def test_default_state_read_covers_more_than_twenty_chunks(self):
        autoread = self.configure(context="How does photosynthesis use chlorophyll?")
        self.assertEqual(autoread["mode"], "state")
        self.assertEqual(autoread["remaining"], self.session["chunk_count"])
        scores = {(0, index): (1.0 if index == 17 else 0.1) for index in range(self.session["chunk_count"])}
        seen = set()
        while True:
            pick = select_next_chunk(self.library, "geology", autoread["mode"], state_scores=scores)
            if pick is None:
                break
            index = pick["chunk_index"]
            if not seen:
                self.assertIn("photosynthesis", self.session["chunks"][index])
            self.assertNotIn(index, seen)
            record_chunk_read(self.memory, self.session, index)
            seen.add(index)
        self.assertEqual(seen, set(range(self.session["chunk_count"])))

    def test_explicit_reply_mode_retains_lexical_selection(self):
        autoread = self.configure("read reply")
        self.assertEqual(autoread["mode"], "reply")
        pick = select_next_chunk(self.library, "photosynthesis chlorophyll", autoread["mode"])
        self.assertEqual(pick["chunk_index"], 17)

    def test_default_state_without_valid_scores_fills_in_document_order(self):
        autoread = self.configure()
        record_chunk_read(self.memory, self.session, 0)
        pick = select_next_chunk(self.library, "photosynthesis chlorophyll", autoread["mode"])
        self.assertEqual(pick["chunk_index"], 1)
        self.assertEqual(pick["mode"], "state_fallback")

    def test_explicit_count_and_order_remain_deliberate(self):
        autoread = self.configure("read 3 order")
        self.assertEqual(autoread["remaining"], 3)
        self.assertEqual(autoread["mode"], "order")
        for expected in range(3):
            pick = select_next_chunk(self.library, "photosynthesis chlorophyll", autoread["mode"])
            self.assertEqual(pick["chunk_index"], expected)
            record_chunk_read(self.memory, self.session, expected)
        self.assertEqual(len(self.session["read"]), 3)
        self.assertEqual(select_next_chunk(self.library, mode="order")["chunk_index"], 3)

    def test_completed_read_starts_fresh_pass_without_reingesting_document_records(self):
        self.configure()
        first_pass = self.session["reading_pass_id"]
        for index in range(self.session["chunk_count"]):
            record_chunk_read(self.memory, self.session, index)
        document_records = [r.record_id for r in self.memory.records if "document" in r.tags]
        autoread = self.configure(context="Now reassess this in light of the new question.")
        self.assertNotEqual(self.session["reading_pass_id"], first_pass)
        self.assertEqual(self.session["read"], set())
        self.assertEqual(autoread["remaining"], self.session["chunk_count"])
        self.assertEqual(document_records, [r.record_id for r in self.memory.records if "document" in r.tags])
        historical = [r for r in self.memory.records if "document_chunk_read" in r.tags]
        self.assertEqual(len(historical), self.session["chunk_count"])
        self.assertIn("new question", self.session["reading_context"])

    def test_explicit_reread_replaces_partial_pass_and_resume_restores_only_active_pass(self):
        self.configure()
        record_chunk_read(self.memory, self.session, 0)
        first_pass = self.session["reading_pass_id"]
        self.configure(context="Revisit the argument after the discussion.", restart=True)
        second_pass = self.session["reading_pass_id"]
        self.assertNotEqual(first_pass, second_pass)
        self.assertEqual(self.session["read"], set())
        record_chunk_read(self.memory, self.session, 17)
        self.memory = MemoryEngine(path=self.root / "memory.jsonl", scope="reading_test")
        self.session = self.reload_document()
        self.library = [self.session]
        autoread = self.configure()
        self.assertEqual(self.session["reading_pass_id"], second_pass)
        self.assertEqual(self.session["read"], {17})
        self.assertEqual(autoread["remaining"], self.session["chunk_count"] - 1)
        self.assertEqual(select_next_chunk(self.library, mode="order")["chunk_index"], 0)

    def test_next_stages_earliest_unread_hole_without_marking_it_processed(self):
        self.configure()
        last = self.session["chunk_count"] - 1
        record_chunk_read(self.memory, self.session, 2)
        record_chunk_read(self.memory, self.session, last)
        source = Path(shell.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        branch = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "dargs.lower() == 'next'"
        )
        # Execute the actual command branch once; a surrounding loop preserves
        # command-level `continue` statements without launching the interactive UI.
        module = ast.Module(body=[ast.For(
            target=ast.Name(id="_test_iteration", ctx=ast.Store()),
            iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
            body=copy.deepcopy(branch.body), orelse=[],
        )], type_ignores=[])
        namespace = dict(vars(shell), doc_session=self.session, doc_library=self.library,
                         memory=self.memory, pending_document_tool_result=None,
                         probes={}, last_assistant_response="new question", doc_autoread=None)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.fix_missing_locations(module), shell.__file__, "exec"), namespace)
        self.assertEqual(self.session["read"], {2, last})
        self.assertIsNone(namespace["pending_document_tool_result"])
        self.assertEqual(namespace["doc_autoread"]["remaining"], 1)
        pick = select_next_chunk(self.library, mode=namespace["doc_autoread"]["mode"])
        self.assertEqual(pick["chunk_index"], 0)

    def execute_branch(self, condition, namespace):
        if condition not in self.branch_cache:
            branch = next(node for node in ast.walk(self.tree)
                          if isinstance(node, ast.If) and ast.unparse(node.test) == condition)
            module = ast.Module(body=[ast.For(
                target=ast.Name(id="_test_iteration", ctx=ast.Store()),
                iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
                body=copy.deepcopy(branch.body), orelse=[],
            )], type_ignores=[])
            self.branch_cache[condition] = compile(ast.fix_missing_locations(module), shell.__file__, "exec")
        with contextlib.redirect_stdout(io.StringIO()):
            exec(self.branch_cache[condition], namespace)

    def test_actual_autoread_branch_covers_full_pass_then_rereads_new_context(self):
        control = self.configure("read reply", context="photosynthesis chlorophyll")
        listener = SimpleNamespace(start=Mock())
        namespace = dict(vars(shell), doc_session=self.session, doc_library=self.library,
                         memory=self.memory, pending_document_tool_result=None,
                         doc_autoread=control, listener=listener,
                         impact_state={"pending": []},
                         recent_responses=deque(), last_assistant_response="")
        namespace["_select_autoread_chunk"] = lambda library, cfg, thought, **kw: select_next_chunk(
            library, thought, cfg["mode"], **kw)
        generated = []
        reader_context = []
        def generate(prompt):
            self.assertNotIn("answers and evidence", prompt)
            self.assertNotIn("No prior question", prompt)
            self.assertNotIn(shell.LLAMA3_START + "user" + shell.LLAMA3_END, prompt)
            self.assertIn("[Document Tool Result]", prompt)
            generated.append(prompt)
            return "The evidence supports geology. What do the minerals imply?"
        for _ in range(self.session["chunk_count"]):
            before = set(self.session["read"])
            self.execute_branch("doc_autoread and doc_autoread['remaining'] > 0", namespace)
            self.assertEqual(self.session["read"], before)
            self.assertEqual(namespace["user_input"], "")
            tool_result = namespace["pending_document_tool_result"]
            prompt = shell.build_prompt("", document_tool_result=tool_result, reader_context=reader_context)
            response = generate(prompt)
            namespace["doc_autoread"] = shell.complete_document_reading(
                self.memory, self.library, tool_result, response, namespace["doc_autoread"])
            shell.retain_document_thought(self.memory, reader_context, self.session, tool_result, response)
            namespace["pending_document_tool_result"] = None
            namespace["last_assistant_response"] = response
            namespace["recent_responses"].append(response)
        self.assertIsNone(namespace["doc_autoread"])
        self.assertEqual(len(self.session["read"]), self.session["chunk_count"])
        self.assertIn("photosynthesis", generated[0])
        self.assertIn("What do the minerals imply?", generated[1])
        first_pass = self.session["reading_pass_id"]
        control = self.configure(context="geology", restart=True)
        self.assertNotEqual(first_pass, self.session["reading_pass_id"])
        self.assertEqual(control["remaining"], self.session["chunk_count"])

    def test_empty_response_and_mismatched_pass_never_credit_staged_text(self):
        control = self.configure()
        text = shell.stage_document_read(self.session, 3)
        self.assertIsNone(shell.complete_document_reading(self.memory, self.library, text, "", control))
        self.assertEqual(self.session["read"], set())
        staged = self.session["_staged_read"]
        self.configure(restart=True)
        self.session["_staged_read"] = staged
        shell.complete_document_reading(self.memory, self.library, text, "A response", control)
        self.assertEqual(self.session["read"], set())

    def test_actual_load_reloads_without_duplicate_library_or_ingestion(self):
        namespace = dict(vars(shell), doc_session=None, doc_library=[], memory=self.memory,
                         user_input=f':doc "{self.path}" because changing questions',
                         command_because="", probes={}, last_assistant_response="geology",
                         pending_document_tool_result=None, doc_autoread=None)
        self.execute_branch("user_input.startswith(':doc')", namespace)
        doc = namespace["doc_library"][0]
        self.assertTrue(namespace["doc_autoread"]["until_complete"])
        self.assertEqual(doc["read"], set())
        first_pass = doc["reading_pass_id"]
        for index in range(doc["chunk_count"]):
            record_chunk_read(self.memory, doc, index)
        records = [r.record_id for r in self.memory.records if "document_chunk" in r.tags]
        self.execute_branch("user_input.startswith(':doc')", namespace)
        self.assertEqual(len(namespace["doc_library"]), 1)
        self.assertNotEqual(namespace["doc_library"][0]["reading_pass_id"], first_pass)
        self.assertEqual(records, [r.record_id for r in self.memory.records if "document_chunk" in r.tags])

    def test_inject_stages_then_credits_only_successful_response(self):
        namespace = dict(vars(shell), doc_session=self.session, doc_library=self.library,
                         memory=self.memory, pending_document_tool_result=None,
                         last_assistant_response="geology", doc_autoread=None)
        self.execute_branch("dargs.lower() == 'inject'", namespace)
        self.assertEqual(self.session["read"], set())
        shell.complete_document_reading(self.memory, self.library,
                                       namespace["pending_document_tool_result"], "Summary")
        self.assertEqual(len(self.session["read"]), self.session["chunk_count"])

    def test_model_request_reopens_completed_pass_without_claiming_unseen_reads(self):
        self.configure()
        for index in range(self.session["chunk_count"]):
            record_chunk_read(self.memory, self.session, index)
        first_pass = self.session["reading_pass_id"]
        namespace = dict(vars(shell), memory=self.memory, doc_library=self.library,
                         model_doc_query="Recheck chlorophyll in the new context", probes={})
        self.execute_branch("model_doc_query and doc_library and (document_tool_result is None) and (not reading_turn_source)", namespace)
        self.assertNotEqual(self.session["reading_pass_id"], first_pass)
        self.assertEqual(self.session["read"], set())
        self.assertEqual(namespace["doc_autoread"]["focus"], namespace["model_doc_query"])

    def test_actual_primary_generation_commits_only_after_success(self):
        # Execute the real generation try/finally and immediately following
        # completion statements, avoiding model startup and unrelated shell IO.
        statements = None
        for node in ast.walk(self.tree):
            for _, value in ast.iter_fields(node):
                if not isinstance(value, list):
                    continue
                for index, item in enumerate(value):
                    if (isinstance(item, ast.Try) and any(
                        isinstance(child, ast.Assign) and isinstance(child.value, ast.Call)
                        and ast.unparse(child.value.func) == "generate_agentic_text"
                        and any(keyword.arg == "mid_chunk_hook" for keyword in child.value.keywords)
                        for child in item.body
                    )):
                        end = next(
                            pos for pos in range(index + 1, len(value))
                            if isinstance(value[pos], ast.If)
                            and ast.unparse(value[pos].test) == "reading_turn_source"
                        )
                        statements = copy.deepcopy(value[index:end + 1])
        self.assertIsNotNone(statements)
        statements.append(ast.Expr(value=ast.Call(func=ast.Name(id="speech_or_tools", ctx=ast.Load()), args=[], keywords=[])))
        module = ast.Module(body=[ast.For(
            target=ast.Name(id="_test_iteration", ctx=ast.Store()),
            iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
            body=statements, orelse=[],
        )], type_ignores=[])
        code = compile(ast.fix_missing_locations(module), shell.__file__, "exec")
        control = self.configure()
        tool_result = shell.stage_document_read(self.session, 0)
        reader_context = []
        namespace = dict(vars(shell), memory=self.memory, doc_library=self.library, doc_session=self.session,
                         doc_autoread=control, document_tool_result=tool_result,
                         reading_turn_source=self.session["source_name"],
                         lookahead_apparatus=None, model=None, prompt="document test",
                         autocomplete_turn=False, utility_expect_turn=False,
                         config=SimpleNamespace(cache_enabled=False, cache_write_enabled=False),
                         tuner={"response_tokens": 64}, synthesis_records=[], tool_sense=None,
                         _save_lookahead=lambda: None, _reader_should_stop=lambda *_args: False, _generation_config_restore=None,
                         _cache_enabled_before=False, _cache_write_before=False, steer_handles=[],
                         _document_reader=lambda: None, _reader_context=lambda: reader_context,
                         speech_or_tools=Mock())
        namespace["generate_agentic_text"] = Mock(side_effect=RuntimeError("generation interrupted"))
        with self.assertRaisesRegex(RuntimeError, "generation interrupted"):
            exec(code, namespace)
        self.assertEqual(self.session["read"], set())
        self.assertEqual(control["remaining"], self.session["chunk_count"])
        thought = "Evidence from this passage. <<MEMORY: internal draft only>>"
        namespace["generate_agentic_text"] = Mock(return_value=(thought, {}))
        exec(code, namespace)
        self.assertEqual(self.session["read"], {0})
        self.assertEqual(control["remaining"], self.session["chunk_count"] - 1)
        namespace["speech_or_tools"].assert_not_called()
        generation = namespace["generate_agentic_text"].call_args.kwargs
        self.assertEqual(generation["generation_channel"], "reader")
        self.assertFalse(generation["chatty_log"])
        self.assertIsNone(generation["mid_chunk_hook"])
        events = [row for row in self.memory.records if "document_reader_thought" in row.tags]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].text, thought)
        self.assertEqual(events[0].provenance["channel"], "reader")
        self.assertFalse(events[0].provenance["spoken"])
        self.assertEqual(events[0].provenance["reading_pass_id"], self.session["reading_pass_id"])
        self.assertFalse(any(row.kind == "turn" for row in self.memory.records))
        self.assertEqual(reader_context, [(tool_result, thought)])

    def test_bounded_reader_history_feeds_later_speech_without_invented_user_text(self):
        self.configure()
        reader_context = []
        for index in range(4):
            self.session["cursor"] = index
            source = shell.stage_document_read(self.session, index)
            thought = f"Reader continuation {index}: " + "finding " * 360
            shell.retain_document_thought(self.memory, reader_context, self.session, source, thought)
        self.assertLessEqual(len(reader_context), 2)
        self.assertLessEqual(sum(len(source) + len(thought) for source, thought in reader_context), 8000)
        events = [row for row in self.memory.records if "document_reader_thought" in row.tags]
        self.assertEqual(len(events), 4)
        self.assertFalse(any(row.kind == "turn" for row in self.memory.records))
        question = "What did the sources establish?"
        prompt = shell.build_prompt(question, reader_context=reader_context)
        user_header = shell.LLAMA3_START + "user" + shell.LLAMA3_END
        self.assertEqual(prompt.count(user_header), 1)
        self.assertIn(user_header + "\n\n" + question + shell.LLAMA3_EOT, prompt)
        for source, thought in reader_context:
            self.assertIn(source, prompt)
            self.assertIn(thought, prompt)
        self.assertNotIn("Reader continuation 0:", prompt)
        self.assertNotIn("answers and evidence", prompt)
        self.assertNotIn("No prior question", prompt)
        self.assertLess(prompt.index(reader_context[-1][1]), prompt.index(user_header))

    def test_empty_reader_thought_adds_no_event_or_context(self):
        self.configure()
        before = len(self.memory.records)
        context = []
        shell.retain_document_thought(self.memory, context, self.session, "Source", "  ")
        self.assertEqual(len(self.memory.records), before)
        self.assertEqual(context, [])

    def test_single_oversized_reader_pair_is_bounded_but_full_thought_is_durable(self):
        self.configure()
        context = []
        source = "Source start: " + "source evidence " * 1000
        thought = "Thought start: " + "internal continuation " * 1000 + " final observation."
        shell.retain_document_thought(self.memory, context, self.session, source, thought)
        self.assertEqual(len(context), 1)
        self.assertLessEqual(sum(len(part) for part in context[0]), 8000)
        retained_source, retained_thought = context[0]
        self.assertTrue(retained_source.startswith("Source start:"))
        self.assertTrue(retained_thought.endswith("final observation."))
        self.assertLess(len(retained_source), len(source))
        self.assertLess(len(retained_thought), len(thought))
        restored = MemoryEngine(path=self.root / "memory.jsonl", scope="reading_test")
        events = [row for row in restored.records if "document_reader_thought" in row.tags]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].text, thought)
        self.assertFalse(events[0].provenance["spoken"])

    def test_scoped_selector_uses_original_library_indices_and_probe_rank(self):
        other_path = self.root / "other.txt"
        other_path.write_text("Another document", encoding="utf-8")
        other = ingest_document(self.memory, other_path)
        function = next(node for node in ast.walk(self.tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "_select_autoread_chunk")
        namespace = dict(vars(shell), model=None, probes={}, doc_probe_score_cache={})
        namespace["_document_reader"] = lambda: SimpleNamespace(rank=Mock(return_value=({}, 0, 25)))
        namespace["score_unread_document_chunks_by_probes"] = Mock(return_value=({(0, 17): 10}, 0, 25))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(function)], type_ignores=[])), shell.__file__, "exec"), namespace)
        control = self.configure()
        control["probe_specs"] = [(1, "green")]
        pick = namespace["_select_autoread_chunk"]([other, self.session], control, "geology")
        self.assertEqual((pick["session_index"], pick["chunk_index"]), (1, 17))

    def test_scoped_state_selector_uses_captured_rank_not_query_words(self):
        other_path = self.root / "other.txt"
        other_path.write_text("Another document", encoding="utf-8")
        other = ingest_document(self.memory, other_path)
        function = next(node for node in ast.walk(self.tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "_select_autoread_chunk")
        rank = Mock(return_value=({(0, 4): 0.9, (0, 17): 0.1}, 0, 25))
        namespace = dict(vars(shell), model=None, probes={}, doc_probe_score_cache={},
                         _document_reader=lambda: SimpleNamespace(rank=rank))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(function)], type_ignores=[])), shell.__file__, "exec"), namespace)
        control = self.configure()
        pick = namespace["_select_autoread_chunk"]([other, self.session], control, "photosynthesis chlorophyll")
        self.assertEqual((pick["session_index"], pick["chunk_index"]), (1, 4))
        rank.assert_called_once_with([self.session])


if __name__ == "__main__":
    unittest.main()
