"""Definition commands address the probe shown in the operator console."""
import ast
from pathlib import Path

import pytest

from invariants.exposure_language import probe_alias
from scripts import interactive_phenomenality as shell


@pytest.mark.parametrize('raw, expected', [
    ('confidence define', 'define confidence'),
    ('confidence DEFINE', 'define confidence'),
    ('[confidence] define', 'define [confidence]'),
    ('define confidence', 'define confidence'),
    ('DEFINE\tconfidence', 'define confidence'),
    ('define   choose based on replies', 'define choose based on replies'),
    ('define', 'define '),
    ('confidence define-experiment', 'define-experiment confidence'),
    ('confidence explain', 'explain confidence'),
    ('confidence EXPLAIN activation', 'explain confidence activation'),
    ('[confidence] explain evidence', 'explain [confidence] evidence'),
    ('explain confidence evidence', 'explain confidence evidence'),
    ('confidence I define words || I avoid definitions',
     'confidence I define words || I avoid definitions'),
])
def test_definition_ordering_does_not_consume_framings(raw, expected):
    assert shell.normalize_probe_definition_args(raw) == expected


@pytest.mark.parametrize('exposed', [False, True])
@pytest.mark.parametrize('target', ['confidence', '[confidence]', 'probe_confidence',
                                     '[probe:confidence]', 'hash', '[hash]'])
def test_operator_definition_resolves_names_and_displayed_ids(target, exposed):
    active = {'confidence': {'exposed': exposed}}
    target = target.replace('hash', probe_alias('confidence'))
    assert shell.resolve_probe_definition_target(target, active) == 'confidence'


def test_unknown_definition_identifier_is_not_minted():
    assert shell.resolve_probe_definition_target('[p_missing]', {'confidence': {}}) == 'p_missing'


def test_definition_preserves_active_name_starting_with_probe_prefix():
    active = {'probe_confidence': {}}
    assert shell.resolve_probe_definition_target('probe_confidence', active) == 'probe_confidence'


def test_actual_probe_dispatch_normalizes_before_any_subcommand():
    tree = ast.parse(Path(shell.__file__).read_text(encoding='utf-8'))
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "user_input.startswith(':probe')")
    assignment = branch.body[0]
    assert isinstance(assignment, ast.Assign)
    namespace = dict(vars(shell), user_input=f':probe [{probe_alias("confidence")}] define')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                 shell.__file__, 'exec'), namespace)
    assert namespace['pargs'] == f'define [{probe_alias("confidence")}]'


@pytest.mark.parametrize('tokens, expected', [
    ([], ('activation', None)),
    (['activation'], ('activation', None)),
    (['ACTIVATION'], ('activation', None)),
    (['evidence'], ('evidence', None)),
    (['prompt'], ('prompt', None)),
    (['words'], ('prompt', None)),
    (['verbal'], ('prompt', None)),
    (['ping'], ('ping', None)),
    (['ping', '-2.5'], ('ping', '-2.5')),
    (['2.5'], ('ping', '2.5')),
    (['choose'], ('ping', 'choose')),
    (['ping', 'AUTO'], ('ping', 'choose')),
])
def test_explain_mode_preserves_explicit_modes_and_legacy_ping_suffixes(tokens, expected):
    assert shell.parse_probe_explain_mode(tokens) == expected


@pytest.mark.parametrize('tokens', [
    ['activation', 'evidence'], ['activation', 'ping'], ['prompt', 'evidence'],
    ['evidence', '2'], ['prompt', 'choose'], ['ping', '1', '2'],
    ['choose', 'auto'], ['unknown'], ['nan'], ['inf'], ['-inf'],
])
def test_explain_mode_rejects_conflicting_unknown_or_nonfinite_options(tokens):
    with pytest.raises(ValueError):
        shell.parse_probe_explain_mode(tokens)
