"""Operator-only probe labels. Never use these strings in model context."""

import re
from colorama import Fore

from invariants.exposure_language import probe_alias


def console_probe_name(name, *, color=Fore.CYAN):
    """Show the operator name and the exact model identifier, then restore color."""
    return f"{name} {Fore.LIGHTBLUE_EX}[{probe_alias(name)}]{color or Fore.RESET}"


def console_route_name(name):
    """Routing branches use a probe_ prefix; other branch names stay unchanged."""
    if name.startswith("probe_"):
        return "probe_" + console_probe_name(name[len("probe_"):], color="")
    return name


def console_probe_lookup(target, probes):
    """Find the real probe name for a target string (name, probe_name, or p_ hash)."""
    if not isinstance(target, str):
        return None
    raw = target.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    if raw.startswith("probe:"):
        raw = raw[len("probe:"):].strip()
    if raw.startswith("probe_"):
        raw = raw[len("probe_"):].strip()
    if raw in probes:
        return raw
    for name in probes:
        if probe_alias(name) == raw:
            return name
    return None


def console_format_probe_text(text, probes, *, color=None):
    """Replace bare or bracketed p_ hashes with operator labels for terminal display.

    Never pass the output of this function to model prompts, cognitive memory,
    or transcripts; model context must remain strictly anonymized.
    """
    if not text or not probes:
        return text
    alias_map = {probe_alias(name): name for name in probes}

    def _repl(match):
        prefix = match.group(1) or ""
        bracketed = match.group(3)
        bare = match.group(4)
        alias = bracketed or bare
        name = alias_map.get(alias)
        if name:
            return f"{prefix}{console_probe_name(name, color=color)}"
        return match.group(0)

    return re.sub(r'(probe[:_]\s*)?(\[(p_[0-9a-f]{24})\]|(p_[0-9a-f]{24}))', _repl, text)
