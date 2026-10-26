"""Exercise the actual shell statements involved in context-off chain failures."""
import ast
from pathlib import Path
from unittest.mock import Mock


SOURCE = Path(__file__).resolve().parents[1] / 'scripts' / 'interactive_phenomenality.py'
TREE = ast.parse(SOURCE.read_text(encoding='utf-8'))


def extract(name, namespace):
    node = next(n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]


def test_disabled_lanes_block_detection_and_stale_action_payloads():
    memory = Mock(scope='test')
    env = {'_enabled_memory_lanes': lambda: [], 'memory': memory}
    detect = extract('_memory_detect', env)
    act = extract('_memory_act', env)
    assert detect('need a memory', {'last_phenomenality': {'ambiguity': 10}}) == (0.0, None)
    assert act('previously detected query', object()) == []
    memory.search.assert_not_called()
    memory.append_event.assert_not_called()


def test_enabled_lanes_keep_detection_and_search_available():
    memory = Mock(scope='test')
    memory.search.return_value = []
    env = {'_enabled_memory_lanes': lambda: ['talk'], 'memory': memory}
    detect = extract('_memory_detect', env)
    act = extract('_memory_act', env)
    score, query = detect('recall the observatory', {'last_phenomenality': {'ambiguity': 2, 'disagreement': 1}})
    assert score == 3 and query == 'recall the observatory'
    assert act(query, object()) == []
    memory.search.assert_called_once_with(query, max_records=3, scope='test')


def test_turn_identity_is_bound_before_prompt_independently_of_context_flag():
    # Find the actual prompt-building statement and its same-block binding.
    siblings = None
    prompt_index = None
    for parent in ast.walk(TREE):
        for _, children in ast.iter_fields(parent):
            if not isinstance(children, list):
                continue
            for index, node in enumerate(children):
                if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                        and isinstance(node.value.func, ast.Name) and node.value.func.id == '_build_live_prompt'
                        and any(k.arg == 'active_u_name' and isinstance(k.value, ast.Name)
                                and k.value.id == 'active_u_name' for k in node.value.keywords)):
                    siblings, prompt_index = children, index
                    break
    assert siblings is not None
    binding = next(node for node in reversed(siblings[:prompt_index])
                   if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                       and target.id == 'active_u_name' for target in node.targets))
    code = compile(ast.Module(body=[binding], type_ignores=[]), str(SOURCE), 'exec')
    for enabled in (False, True):
        for replacement, expected in ((None, 'user'), ('Merith', 'Merith')):
            env = {'session_context_enabled': enabled, '_last_replace_name': replacement}
            exec(code, env)
            assert env['active_u_name'] == expected
    # The next turn must replace a previous participant rather than reuse it.
    env['_last_replace_name'] = None
    exec(code, env)
    assert env['active_u_name'] == 'user'


def test_isolated_prompt_has_no_probe_or_history_text():
    from invariants.config import AgenticConfig
    from scripts.interactive_phenomenality import build_prompt
    env = {'config': AgenticConfig(activation_memory_isolate=True), 'build_prompt': build_prompt}
    build = extract('_build_live_prompt', env)
    prompt = build('Which island?', field_context='SECRET PROBE REPLY',
                   probe_tool_result='SECRET SENSOR', memory_tool_result='SECRET MEMORY',
                   document_tool_result='SECRET DOCUMENT',
                   session_context=[('assistant', 'assistant', 'SECRET HISTORY')],
                   reader_context=[('SECRET SOURCE', 'SECRET THOUGHT')])
    assert 'SECRET' not in prompt
    assert prompt == build_prompt('Which island?')


def test_isolated_generation_pauses_other_engine_paths_and_restores_on_failure():
    from invariants.config import AgenticConfig
    from scripts import interactive_phenomenality as shell
    from unittest.mock import patch
    cfg = AgenticConfig(activation_memory_isolate=True)
    before = vars(cfg).copy()

    def fail(*args, **kwargs):
        assert not cfg.synthesis_enabled and not cfg.cache_enabled
        assert not cfg.cache_write_enabled and cfg.max_routing_events == 0
        assert not cfg.use_expert_vectors and kwargs['mid_chunk_hook'] is None
        raise RuntimeError('isolated generation failure')

    with patch.object(shell, '_engine_generate', side_effect=fail):
        try:
            shell.generate_agentic_text(object(), config=cfg, mid_chunk_hook=Mock())
        except RuntimeError as exc:
            assert str(exc) == 'isolated generation failure'
        else:
            raise AssertionError('Expected generation failure')
    assert vars(cfg) == before
