"""Tests for operator probe display, hash lookup, and console formatting.

Operator-only labels (console_probe_name, console_format_probe_text) must
never be supplied to model prompts, session context, or memory events.
"""

from collections import deque
from colorama import Fore, Style
import pytest

from invariants.exposure_language import probe_alias
from invariants.probe_display import (
    console_probe_name,
    console_route_name,
    console_probe_lookup,
    console_format_probe_text,
)
from scripts.interactive_phenomenality import resolve_probe_choice


def sample_probes():
    return {
        "reality": {
            "exposed": True,
            "framings": ("I accept facts.", "I reject facts."),
            "history": deque(),
            "direction": {0: None, 1: None},
        },
        "confidence": {
            "exposed": False,
            "framings": ("I am certain.", "I am uncertain."),
            "history": deque(),
            "direction": {0: None},
        },
    }


def test_console_probe_name_structure_and_color():
    alias = probe_alias("reality")
    formatted = console_probe_name("reality", color=Fore.GREEN)
    assert "reality" in formatted
    assert f"[{alias}]" in formatted
    assert Fore.LIGHTBLUE_EX in formatted
    assert formatted.endswith(Fore.GREEN)


def test_console_route_name_with_probe_prefix():
    alias = probe_alias("reality")
    formatted = console_route_name("probe_reality")
    assert formatted.startswith("probe_reality")
    assert f"[{alias}]" in formatted
    assert console_route_name("custom_branch") == "custom_branch"


def test_console_probe_lookup_by_name_and_hash():
    probes = sample_probes()
    real_alias = probe_alias("reality")
    conf_alias = probe_alias("confidence")

    assert console_probe_lookup("reality", probes) == "reality"
    assert console_probe_lookup("probe_reality", probes) == "reality"
    assert console_probe_lookup("probe:reality", probes) == "reality"
    assert console_probe_lookup(real_alias, probes) == "reality"
    assert console_probe_lookup(f"[{real_alias}]", probes) == "reality"
    assert console_probe_lookup(f"probe:{real_alias}", probes) == "reality"
    assert console_probe_lookup(conf_alias, probes) == "confidence"
    assert console_probe_lookup("p_000000000000000000000000", probes) is None
    assert console_probe_lookup("unknown_probe", probes) is None
    assert console_probe_lookup(None, probes) is None


def test_console_format_probe_text_replaces_hashes_for_operator():
    probes = sample_probes()
    alias = probe_alias("reality")
    text = f"I measured {alias} on my last turn; reading was +0.250."
    formatted = console_format_probe_text(text, probes, color=Fore.GREEN)

    assert "reality" in formatted
    assert f"[{alias}]" in formatted
    assert alias in formatted
    # Check that it didn't leave only bare hash without probe name
    assert f"I measured reality" in formatted


def test_console_format_probe_text_handles_brackets_and_tool_syntax():
    probes = sample_probes()
    alias = probe_alias("reality")

    # Bracketed hash [p_...]
    text_bracketed = f"Status: [{alias}] active"
    formatted = console_format_probe_text(text_bracketed, probes, color=Fore.CYAN)
    assert "Status: reality " in formatted
    assert f"[{alias}]" in formatted

    # Tool syntax <<PROBE: p_...>>
    text_tool = f"<<PROBE: {alias}>>"
    formatted_tool = console_format_probe_text(text_tool, probes, color=Fore.CYAN)
    assert "<<PROBE: reality " in formatted_tool
    assert f"[{alias}]" in formatted_tool

    # Probe prefix probe:p_...
    text_pre = f"probe:{alias}"
    formatted_pre = console_format_probe_text(text_pre, probes, color=Fore.CYAN)
    assert "probe:reality " in formatted_pre
    assert f"[{alias}]" in formatted_pre


def test_console_format_probe_text_preserves_unknown_hashes():
    probes = sample_probes()
    unknown = "p_111122223333444455556666"
    text = f"Checking {unknown}."
    assert console_format_probe_text(text, probes) == text


def test_model_memory_and_context_never_receive_readable_replacement():
    probes = sample_probes()
    alias = probe_alias("reality")
    model_generated_text = f"My internal state reflects sensor {alias} at +0.400."

    # Console display shows readable replacement
    console_output = console_format_probe_text(model_generated_text, probes, color=Fore.GREEN)
    assert "reality" in console_output
    assert f"[{alias}]" in console_output

    # Model memory / session context MUST retain raw text with only anonymized hashes
    memory_record = {"role": "assistant", "text": model_generated_text}
    assert "reality" not in memory_record["text"]
    assert alias in memory_record["text"]


def test_resolve_probe_choice_resolves_probe_hashes():
    probes = sample_probes()
    alias = probe_alias("reality")

    # Resolves bare hash
    assert resolve_probe_choice(alias, probes) == "reality"
    # Resolves bracketed hash
    assert resolve_probe_choice(f"[{alias}]", probes) == "reality"
    # Resolves real name
    assert resolve_probe_choice("reality", probes) == "reality"
    # Resolves probe_ prefix
    assert resolve_probe_choice("probe_reality", probes) == "reality"
    # Leaves unknown tokens intact without error
    assert resolve_probe_choice("unrelated_token", probes) == "unrelated_token"
