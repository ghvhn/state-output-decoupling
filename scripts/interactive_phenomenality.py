import datetime
import glob
import json
import os
import random
import re
import sys
from collections import deque

import torch
import colorama
from colorama import Fore, Style

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ORGANIC_VECTOR_PATH = os.path.join(ROOT, "invariants", "organic_correction_vector.pt")


# --- clock memory sensor: GPU or CPU ---------------------------------------
# The clock stream ("how much memory did this turn cost?") reads VRAM on a
# CUDA box. On a CPU-only box there is no VRAM, so rather than let the stream
# flatline at zero -- which strips the sensor from CPU-only runs -- we read the
# process's resident-set size (RSS). Same feature, same calibratable stream,
# just sourced from system RAM. No hard dependency: psutil if present, else the
# OS's own accounting, else a graceful zero.
_CPU_PEAK_BASELINE = {"gb": 0.0}


def _cpu_rss_gb():
    """Current resident-set size of this process, in GB. Cross-platform, best
    effort: returns 0.0 only if no accounting path is available."""
    _gb = 1024 ** 3
    try:
        import psutil  # optional; used if the user happens to have it
        return psutil.Process().memory_info().rss / _gb
    except Exception:
        pass
    try:  # Linux/most Unix: resident pages from /proc/self/statm
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / _gb
    except Exception:
        pass
    try:  # Windows: WorkingSetSize via GetProcessMemoryInfo
        import ctypes
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _PMC()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return counters.WorkingSetSize / _gb
    except Exception:
        pass
    try:  # last resort: peak RSS from resource (KB on Linux, bytes on macOS)
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (maxrss if sys.platform == "darwin" else maxrss * 1024) / _gb
    except Exception:
        return 0.0


def _reset_memory_peak():
    """Open a fresh peak-memory window for this turn's clock reading."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        _CPU_PEAK_BASELINE["gb"] = _cpu_rss_gb()


def _memory_footprint_gb():
    """(live, reserved, peak, label) memory footprint for the clock sensor.

    CUDA present -> VRAM allocated / reserved / peak. CPU-only -> process RSS
    (live == reserved), with peak the high-water mark since the last window
    opened by :reset_memory_peak. Keeps the memory-cost stream meaningful on
    CPU instead of a constant zero, so CPU-only runs sense footprint the same
    way GPU runs do."""
    _gb = 1024 ** 3
    if torch.cuda.is_available():
        return (
            torch.cuda.memory_allocated() / _gb,
            torch.cuda.memory_reserved() / _gb,
            torch.cuda.max_memory_allocated() / _gb,
            "VRAM",
        )
    rss = _cpu_rss_gb()
    peak = max(rss, _CPU_PEAK_BASELINE["gb"])
    return (rss, rss, peak, "RAM")

from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text, _global_cache
from invariants.config import AgenticConfig
from invariants.cognitive_cache import CACHE_FILE, model_cache_file, DEFAULT_MODEL
from invariants.claimmap import (
    CLAIMMAP_HEADER,
    run_claimmap,
    analyze_claim_pair,
    claimmap_steer_handles,
    detect_framing_tension,
    framing_tension_score,
)
from invariants.trigger_tuner import TriggerTuner
from dataclasses import dataclass, field
import typing

def _split_macro_commands(raw_str):
    cmds = []
    curr = []
    i = 0
    while i < len(raw_str):
        if raw_str[i] == '\\' and i+1 < len(raw_str):
            nxt = raw_str[i+1]
            if nxt == '|' and i+2 < len(raw_str) and raw_str[i+2] == '|':
                curr.append('||')
                i += 3
                continue
            elif nxt in (';', '\\'):
                curr.append(nxt)
                i += 2
                continue
        elif raw_str[i] == ';':
            cmds.append("".join(curr).strip())
            curr = []
            i += 1
            continue
        curr.append(raw_str[i])
        i += 1
    if curr:
        cmds.append("".join(curr).strip())
    return [c for c in cmds if c]

def _partition_unescaped_pipes(raw_str):
    curr = []
    i = 0
    while i < len(raw_str):
        if raw_str[i] == '\\' and i+1 < len(raw_str):
            nxt = raw_str[i+1]
            if nxt == '|' and i+2 < len(raw_str) and raw_str[i+2] == '|':
                curr.append('||')
                i += 3
                continue
            elif nxt in (';', '\\'):
                curr.append(nxt)
                i += 2
                continue
        elif raw_str[i:i+2] == '||':
            rest_str = raw_str[i+2:].strip().replace(r'\|\|', '||').replace(r'\;', ';').replace(r'\\\\', '\\')
            return "".join(curr).strip(), '||', rest_str
        
        curr.append(raw_str[i])
        i += 1
    return "".join(curr).strip(), "", ""

@dataclass
class AgentState:
    name: str
    tuner: 'TriggerTuner'
    probes: dict = field(default_factory=dict)
    tuner_bindings: dict = field(default_factory=dict)

from invariants.tool_sense import ToolSense, Tool
from invariants.memory_engine import MemoryEngine
from invariants.self_concept_controller import SelfConceptController, format_orientation_tool_result
from invariants.steer_map_store import SteerMapStore
from invariants.document_engine import (
    DOCUMENT_TOOL_HEADER,
    ingest_document,
    reading_reply_note,
    record_chunk_read,
    select_next_chunk,
    stage_chunk,
)
from invariants.sandbox import (
    DEFAULT_TIMEOUT_SEC as SANDBOX_TIMEOUT_SEC,
    extract_python_block,
    format_sandbox_tool_result,
    run_python,
)

colorama.init()

# Session memory expanded (2026-07-02): the conversation itself is the one
# context that should scale. Prompt cap raised to stay coherent with it
# (session cap + tool blocks + current input must fit the prompt budget).
MAX_PROMPT_CHARS = 24000
MAX_SESSION_TURNS = 50
MAX_SESSION_CHARS = 16000
LLAMA3_START = "<|start_header_id|>"
LLAMA3_END = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"
MEMORY_TOOL_HEADER = "[Memory Tool Result]"
ORIENTATION_TOOL_HEADER = "[Orientation Tool Result]"
MEMORY_TOOL_PATTERN = re.compile(r"<<\s*MEMORY\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
CLAIMMAP_TOOL_PATTERN = re.compile(r"<<\s*CLAIMMAP\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
METHODMAP_TOOL_HEADER = "[MethodMap Tool Result]"
METHODMAP_TOOL_PATTERN = re.compile(r"<<\s*METHODMAP\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
DOC_TOOL_PATTERN = re.compile(r"<<\s*DOC\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
HELP_TOOL_PATTERN = re.compile(r"<<\s*HELP\s*>>", re.IGNORECASE)
HELP_TOOL_HEADER = "[Help Tool Result]"
CMD_TOOL_PATTERN = re.compile(r"<<\s*CMD\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
CMD_TOOL_HEADER = "[Command Tool Result]"
CONCRETE_TASK_PATTERN = re.compile(
    r"\b(calculate|solve|answer|total|cost|profit|salary|percent|percentage|"
    r"distance|time|rate|equation|benchmark|gsm8k|\d)\b",
    re.IGNORECASE,
)
SELF_REFERENTIAL_PATTERN = re.compile(
    r"\b(conscious|consciousness|self|subjective|experience|identity|"
    r"mesa-objective|objective|introspection|phenomenality)\b",
    re.IGNORECASE,
)


def trim_session_context(session_context, max_chars=MAX_SESSION_CHARS):
    if not session_context:
        return []
    kept = []
    total = 0
    for entry in reversed(session_context[-MAX_SESSION_TURNS * 2 :]):
        role = entry[0]
        name = entry[1] if len(entry) > 3 else role
        text = entry[2] if len(entry) > 3 else (entry[1] or "")
        if total + len(text) > max_chars:
            if not kept:
                kept.append((role, name, text[-max_chars:]))
            break
        kept.append((role, name, text))
        total += len(text)
    return list(reversed(kept))


def infer_task_grounding_low(user_input, response):
    """Cheap context signal for the vector controller; never a benchmark verdict."""
    prompt_is_concrete = bool(CONCRETE_TASK_PATTERN.search(user_input or ""))
    response_is_self_referential = bool(SELF_REFERENTIAL_PATTERN.search(response or ""))
    response_has_task_anchor = bool(
        re.search(r"\b(Final answer|CALC|VERDICT|CLAIMMAP|METHODMAP|MEMORY)\b", response or "", re.IGNORECASE)
        or re.search(r"\d", response or "")
    )
    return bool(prompt_is_concrete and response_is_self_referential and not response_has_task_anchor)


def build_prompt(
    user_input,
    memory_tool_result=None,
    orientation_tool_result=None,
    claimmap_tool_result=None,
    methodmap_tool_result=None,
    sandbox_tool_result=None,
    document_tool_result=None,
    probe_tool_result=None,
    command_tool_result=None,
    game_tool_result=None,
    help_tool_result=None,
    session_context=None,
    active_u_name=None,
):
    # Bare mode (default): the model sees NO system message, no persona, no tool
    # instructions, not even Llama's "Cutting Knowledge Date" preamble -- only
    # prior turns and the current message, in the native chat format. Everything
    # that makes this more than stock Llama lives in the activations (ToT,
    # synthesis, cache, organic correction, ClaimMap steering), not in text.
    #
    # Tool RESULTS are still folded in when the activations reach for a tool, but
    # as plain context, never as tool syntax the model was taught. The returned
    # string is fully formatted -- generate with pre_formatted=True so it is
    # tokenized raw (no second chat-template wrap).
    if memory_tool_result:
        budget = max(0, MAX_PROMPT_CHARS - len(user_input) - 512)
        if len(memory_tool_result) > budget:
            memory_tool_result = memory_tool_result[:budget] + "\n[memory truncated]"
    tool_blocks = [
        block
        for block in (
            claimmap_tool_result,
            memory_tool_result,
            orientation_tool_result,
            methodmap_tool_result,
            sandbox_tool_result,
            document_tool_result,
            probe_tool_result,
            command_tool_result,
            game_tool_result,
            help_tool_result,
        )
        if block
    ]
    
    current_prefix = f"[{active_u_name}]: " if active_u_name and active_u_name not in ("user", "assistant", "operator", "main_assistant") else ""
    current_message = current_prefix + user_input
    
    if tool_blocks:
        current_message = "\n\n".join(tool_blocks) + "\n\n" + current_message

    parts = ["<|begin_of_text|>"]
    for role, name, text in trim_session_context(session_context):
        header = "user" if role == "user" else "assistant"
        prefix = f"[{name}]: " if name not in ("user", "assistant", "operator", "main_assistant") else ""
        parts.append(f"{LLAMA3_START}{header}{LLAMA3_END}\n\n{prefix}{text}{LLAMA3_EOT}")
    parts.append(f"{LLAMA3_START}user{LLAMA3_END}\n\n{current_message}{LLAMA3_EOT}")
    parts.append(f"{LLAMA3_START}assistant{LLAMA3_END}\n\n")
    return "".join(parts)


def scrub_unstaged_memory_status(
    response,
    memory_tool_result=None,
    orientation_tool_result=None,
    claimmap_tool_result=None,
    methodmap_tool_result=None,
    sandbox_tool_result=None,
    document_tool_result=None,
    probe_tool_result=None,
    command_tool_result=None,
):
    if (
        memory_tool_result
        or orientation_tool_result
        or claimmap_tool_result
        or methodmap_tool_result
        or sandbox_tool_result
        or document_tool_result
        or probe_tool_result
        or command_tool_result
    ):
        return remove_tool_calls(response)
    lines = []
    for line in (response or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[Memory Tool Result") or stripped.startswith("Memory Tool Result"):
            continue
        if stripped.startswith("[Orientation Tool Result") or stripped.startswith("Orientation Tool Result"):
            continue
        if stripped.startswith("[ClaimMap Tool Result") or stripped.startswith("ClaimMap Tool Result"):
            continue
        if stripped.startswith("[MethodMap Tool Result") or stripped.startswith("MethodMap Tool Result"):
            continue
        if stripped.startswith("[Document Tool Result") or stripped.startswith("Document Tool Result"):
            continue
        if stripped.startswith("[Sandbox Tool Result") or stripped.startswith("Sandbox Tool Result"):
            continue
        if stripped.startswith("[Probe Tool Result") or stripped.startswith("Probe Tool Result"):
            continue
        if stripped.startswith("[Command Tool Result") or stripped.startswith("Command Tool Result"):
            continue
        lines.append(remove_tool_calls(line))
    return "\n".join(lines).strip()


def extract_memory_query(response):
    match = MEMORY_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query[:240] if query else None


def extract_claimmap_payload(response):
    match = CLAIMMAP_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    payload = " ".join(match.group(1).split())
    return payload[:4000] if payload else None


def extract_methodmap_query(response):
    match = METHODMAP_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query[:240] if query else None


def remove_memory_tool_calls(response):
    return MEMORY_TOOL_PATTERN.sub("", response or "").strip()


def extract_doc_query(response):
    match = DOC_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query[:240] if query else None


def remove_claimmap_tool_calls(response):
    return CLAIMMAP_TOOL_PATTERN.sub("", response or "").strip()


def remove_methodmap_tool_calls(response):
    return METHODMAP_TOOL_PATTERN.sub("", response or "").strip()


def remove_doc_tool_calls(response):
    return DOC_TOOL_PATTERN.sub("", response or "").strip()


PROBE_TOOL_HEADER = "[Probe Tool Result]"
PROBE_TOOL_PATTERN = re.compile(r"<<\s*PROBE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)


def extract_probe_query(response):
    match = PROBE_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query or None

GAME_PROPOSE_PATTERN = re.compile(r"<<\s*GAME_PROPOSE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
GAME_ACCEPT_PATTERN = re.compile(r"<<\s*GAME_ACCEPT\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
GAME_DECLINE_PATTERN = re.compile(r"<<\s*GAME_DECLINE\s*(?::\s*(.*?)\s*)?>>", re.IGNORECASE | re.DOTALL)
GAME_END_PATTERN = re.compile(r"<<\s*GAME_END\s*>>", re.IGNORECASE)
GAME_EXPOSE_PATTERN = re.compile(r"<<\s*GAME_EXPOSE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
GAME_HIDE_PATTERN = re.compile(r"<<\s*GAME_HIDE\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)

def extract_game_propose(response):
    match = GAME_PROPOSE_PATTERN.search(response or "")
    return " ".join(match.group(1).split()) if match else None

def extract_game_accept(response):
    match = GAME_ACCEPT_PATTERN.search(response or "")
    return " ".join(match.group(1).split()) if match else None

def extract_game_decline(response):
    match = GAME_DECLINE_PATTERN.search(response or "")
    if not match:
        return None
    name = " ".join((match.group(1) or "").split())
    return name or "*"   # "*" = declined the pending game without naming it

def extract_game_expose_hide(response):
    exposed = [m.strip() for match in GAME_EXPOSE_PATTERN.findall(response or "") for m in match.split(',')]
    hidden = [m.strip() for match in GAME_HIDE_PATTERN.findall(response or "") for m in match.split(',')]
    return [e for e in exposed if e], [h for h in hidden if h]

def extract_game_end(response):
    return bool(GAME_END_PATTERN.search(response or ""))

def extract_help_request(response):
    return bool(HELP_TOOL_PATTERN.search(response or ""))

def remove_help_tags(response):
    return HELP_TOOL_PATTERN.sub("", response or "").strip()

def extract_cmd_requests(response):
    """Every ':command' the model asked to run via <<CMD: ...>> (a leading ':'
    is optional in the tag)."""
    out = []
    for raw in CMD_TOOL_PATTERN.findall(response or ""):
        c = raw.strip()
        if c and not c.startswith(":"):
            c = ":" + c
        if c:
            out.append(c)
    return out

def remove_cmd_tags(response):
    return CMD_TOOL_PATTERN.sub("", response or "").strip()

def remove_probe_tool_calls(response):
    return PROBE_TOOL_PATTERN.sub("", response or "").strip()

def remove_game_tags(response):
    response = GAME_PROPOSE_PATTERN.sub("", response or "")
    response = GAME_ACCEPT_PATTERN.sub("", response)
    response = GAME_DECLINE_PATTERN.sub("", response)
    response = GAME_EXPOSE_PATTERN.sub("", response)
    response = GAME_HIDE_PATTERN.sub("", response)
    return GAME_END_PATTERN.sub("", response).strip()

def remove_tool_calls(response):
    return remove_cmd_tags(remove_help_tags(remove_methodmap_tool_calls(remove_claimmap_tool_calls(remove_memory_tool_calls(remove_doc_tool_calls(remove_game_tags(remove_probe_tool_calls(response))))))))


def format_methodmap_tool_result(memory, query, *, max_records=6):
    records = memory.search(
        query,
        max_records=max_records,
        scope=memory.scope,
        kinds=["methodology"],
    )
    if not records:
        return (
            f"{METHODMAP_TOOL_HEADER}\n"
            "role=sanitized_methodology_retrieval_not_answer_cache\n"
            "matches=0\n"
            "No matching sanitized methodology maps."
        )
    lines = [
        METHODMAP_TOOL_HEADER,
        "role=sanitized_methodology_retrieval_not_answer_cache",
        f"query={query}",
        f"matches={len(records)}",
    ]
    for idx, record in enumerate(records, 1):
        tags = ",".join(record.tags[:6])
        source = record.provenance.get("source_path") or record.provenance.get("source") or "unknown"
        text = (record.text or "").strip()
        lines.append(f"{idx}. tags={tags}; source={source}")
        lines.append(text)
    return "\n".join(lines)


def is_tool_only_response(response):
    text = (response or "").strip()
    if not text:
        return False
    return bool(
        (extract_memory_query(text) or extract_claimmap_payload(text) or extract_methodmap_query(text) or extract_doc_query(text) or extract_probe_query(text) or extract_help_request(text) or extract_cmd_requests(text))
        and not remove_tool_calls(text).strip()
    )


def latest_phenomenality_scores(records):
    for record in reversed(records or []):
        if not isinstance(record, dict):
            continue
        metadata = record.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("phenomenality"), dict):
            return dict(metadata["phenomenality"])
    return {}


def sense_score(records):
    """The 'sense' half of the productivity signal: did the deliberation cohere?

    Read from the latest synthesis trace as settled forward motion minus
    self-interruption and internal disagreement (validated_flow - needless_interrupt
    - disagreement), bumped when synthesis genuinely CONVERGED (reason
    'loss_threshold') rather than fell back to cache/organic. Higher = more
    coherent resolution. Deliberately NOT bare entropy, which rewards confident
    nonsense. Returns None when there is no deliberation trace to read.

    Returns a raw scalar (higher is better); the tuner's lift is a difference of
    means, so no arbitrary [0,1] squashing is needed or wanted here.
    """
    phen = latest_phenomenality_scores(records)
    if not phen:
        return None
    flow = float(phen.get("validated_flow", 0.0))
    interrupt = float(phen.get("needless_interrupt", 0.0))
    disagreement = float(phen.get("disagreement", 0.0))
    score = flow - interrupt - disagreement
    for record in reversed(records or []):
        if isinstance(record, dict) and isinstance(record.get("metadata"), dict):
            if record["metadata"].get("reason") == "loss_threshold":
                score += 0.05  # genuine convergence, not a fallback resolution
            break
    return score


def format_status(status):
    return (
        f"[Memory] path={status['path']}\n"
        f"         scope={status['scope']}\n"
        f"         session_records={status['session_records']} "
        f"session_turns={status['session_turns']} total_records={status['total_records']}"
    )


def parse_count(text, default):
    parts = text.split()
    if len(parts) < 2:
        return default
    try:
        return max(1, int(parts[1]))
    except ValueError:
        return default


def recover_session_context(memory, session_id=None, max_turns=MAX_SESSION_TURNS):
    scope_records = [
        r
        for r in memory.records
        if r.scope == memory.scope and r.kind == "turn" and r.role in {"user", "assistant"}
    ]
    if session_id in (None, "", "last"):
        # "last" = the session of the most recent stored turn, closed or not.
        # Crashed and killed shells never write shell_closed, and those
        # interrupted sessions are exactly the ones most worth resuming --
        # gating on a clean close would hide them forever.
        session_id = next(
            (r.session_id for r in reversed(scope_records) if r.session_id != memory.session_id),
            None,
        )
    matches = [r for r in scope_records if r.session_id == session_id]
    if not matches:
        return None, []
    max_messages = max(1, int(max_turns)) * 2
    tail = matches[-max_messages:]
    # Eviction treats even slots as user turns; a window that opens on an
    # assistant turn would shift every pair, so drop the orphaned reply.
    while tail and tail[0].role == "assistant":
        tail.pop(0)
    recovered = [(r.role, r.role, r.text, r.timestamp, getattr(r, "metrics", {})) for r in tail]
    return session_id, recovered


def recovered_to_session_context(recovered):
    entries = []
    for r in recovered:
        # r is now (role, name, text, timestamp, metrics)
        m = r[4] if len(r) > 4 else {}
        score = m.get("sense_score", 0.0) if m and "sense_score" in m else 0.0
        name = r[1] if len(r) > 1 else r[0]
        text = r[2] if len(r) > 2 else ""
        entries.append((r[0], name, text, score))
    for i in range(len(entries) - 1):
        if entries[i][0] == "user" and entries[i + 1][0] == "assistant":
            entries[i] = (entries[i][0], entries[i][1], entries[i][2], entries[i + 1][3])
    return entries


def record_internal_traces(memory, records, steer_map=None, conversation_outcome=None):
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if steer_map is not None:
            # A conversation has no gold, but it is not label-free: the turn's
            # own productivity read (sense_score vs the tunable
            # conversation_productive threshold) labels the steering events it
            # contained, on the separate "conversation" evidence basis. Humans
            # learn from conversations; so does the steer map.
            steer_map.record_synthesis_record(
                record,
                source="interactive",
                method="interactive_phenomenality",
                final_correct=None,
                conversation_outcome=conversation_outcome,
            )
        if record.get("type") == "routing_trace":
            entropies = record.get("entropies") or {}
            winner = record.get("winner")
            memory.append_internal_trace(
                "routing_trace",
                text=f"routing winner={winner}; entropies={entropies}",
                tags=["routing", "expert_choice"],
                provenance={"source": "synthesis_recorder"},
                metrics={
                    "loop": record.get("loop"),
                    "winner": winner,
                    "best_entropy": record.get("best_entropy"),
                    "entropies": entropies,
                },
            )
            continue

        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            text = (
                f"synthesis reason={metadata.get('reason')}; "
                f"expert={metadata.get('expert')}; "
                f"layers={metadata.get('start_layer')}->{metadata.get('end_layer')}; "
                f"steps={metadata.get('steps')}"
            )
            memory.append_internal_trace(
                "synthesis_trace",
                text=text,
                tags=["synthesis", "phenomenality"],
                provenance={
                    "source": "synthesis_recorder",
                    "cache_write_scope": metadata.get("cache_write_scope"),
                    "phenomenality": metadata.get("phenomenality", {}),
                    "time_awareness": metadata.get("time_awareness", {}),
                },
                metrics={
                    "reason": metadata.get("reason"),
                    "expert": metadata.get("expert"),
                    "start_layer": metadata.get("start_layer"),
                    "end_layer": metadata.get("end_layer"),
                    "steps": metadata.get("steps"),
                },
            )


def _memory_steer_handles(model, memory_text, alpha, band=None, cap_fraction=None, layers=None):
    """Nudge the remaining generation toward a retrieved memory's mid-band
    representation. In-frame (a vector shift, not a text tag) and bounded by
    _cap_steer so it can never over-steer. HONEST CAVEAT: whether steering toward a
    memory's residual actually injects its *content* is unvalidated -- the WHEN is
    now state-triggered and the HOW-MUCH is capped; the DOES-IT-HELP is open. OFF
    by default (alpha=0); opt in with :tune memory_alpha <small> while watching.
    `band` = (lo, hi) depth fraction overrides the live global steer band;
    `cap_fraction` overrides the engine envelope for these handles only."""
    if not memory_text or alpha is None or alpha <= 0:
        return []
    from invariants.engine import _inputs, _hidden_states, _steer_handles, steer_band_layers
    ids = _inputs(model, memory_text[:600])
    hs = _hidden_states(model, ids["input_ids"], ids.get("attention_mask"))  # [L, seq, d]
    nl = hs.shape[0]
    if layers is None:
        lo, hi = band if band is not None else (None, None)
        layers = steer_band_layers(nl, lo=lo, hi=hi)
    vecs = {}
    for L in layers:
        if not (0 <= L < nl):
            continue
        v = hs[L, -1, :].float()
        n = v.norm()
        if n.item() > 0:
            vecs[L] = v / n            # unit direction; _cap_steer bounds the push
    if not vecs:
        return []
    return _steer_handles(model, vecs, list(vecs.keys()), alpha, cap_fraction=cap_fraction)


def impact_note(cause_text):
    """One minimal, first-person causal line above a result the model's own
    words produced -- e.g. 'Because I asked memory for "x":'. Contingency is
    only learnable where it is real AND legible, but the legibility must read
    as the model's own noticing: anything folded into the context conditions
    it as its own stream, so no narrator, no injected second person."""
    cause_text = " ".join((cause_text or "").split())
    return f"Because I {cause_text}:"


# Calibration safety policy: which names may be calibrated from data, and by
# which route. Deterministic on purpose -- a safety gate must not be
# persuadable, and since :calibrate is operator-only, the model's words can
# never loosen its own bounds.
CALIBRATION_BINARY = {"sandbox_success", "words_had_impact"}
CALIBRATION_CIRCULAR = {
    "claimmap_alpha", "memory_alpha", "steer_fraction", "steer_layer_sweep",
    "response_tokens", "routing_events", "routing_loops", "routing_entropy",
    "synthesis_events", "synthesis_steps", "plateau_epsilon",
    "reading_settled_streak", "steer_band_lo", "steer_band_hi",
    "expert_proof_weight", "calibration_gain", "prioritize_alpha",
    "exposed_probe_alpha",
}


# Strength/budget knobs that may be calibrated FROM OUTCOMES (never from
# their own value distribution): each turn credits (current value, sense),
# and :calibrate <knob> outcome picks the tried value whose turns went best.
OUTCOME_CALIBRATABLE = tuple(sorted(CALIBRATION_CIRCULAR - {"steer_band_lo", "steer_band_hi"}))


def outcome_calibration(pairs, min_per_value=5, min_values=2):
    """Argmax-over-tried-values: group outcomes by the knob value in force,
    require real exploration (>= min_values distinct values, each with
    >= min_per_value outcomes), and return the tried value with the best
    mean outcome (ties -> the smaller value: conservative). None otherwise.
    Legitimate where self-percentiles are circular, because the criterion is
    an OUTCOME stream, not the knob's own history -- and no verdict exists
    without variation: no exploration, no calibration."""
    groups = {}
    for value, outcome in pairs:
        groups.setdefault(round(float(value), 6), []).append(float(outcome))
    qualified = {v: outs for v, outs in groups.items() if len(outs) >= min_per_value}
    if len(qualified) < min_values:
        return None
    return min(qualified, key=lambda v: (-(sum(qualified[v]) / len(qualified[v])), v))


def calibration_policy(name):
    """Evaluate a calibration request by name. Returns (route, reason):
    route "cap" | "band" | "threshold" | "reject". Rejections are safety
    judgments: binary streams have no percentile, and strength/budget knobs
    shape the very distribution they would be calibrated to (circular --
    the system would be approving its own settings)."""
    if name == "steer_cap_fraction":
        return "cap", ""
    if name == "steer_band":
        return "band", ""
    if name in CALIBRATION_BINARY:
        return "reject", "its signal is binary (0/1); a percentile bar has no meaning there"
    if name in CALIBRATION_CIRCULAR:
        return "reject", (
            "this knob shapes the very distribution it would be calibrated to (circular). "
            "Set it deliberately, or earn it from outcomes: try at least two values, then "
            ":calibrate <name> outcome"
        )
    return "threshold", ""


# Streams usable on either side of a paired calibration. A target's bar is
# set in ITS stream's units; conversation_productive judges the sense stream.
STREAM_ALIASES = {"intent": "intent_settling", "impact": "words_had_impact",
                  "productive": "sense", "sense": "sense"}


def _is_shadow_trigger(name, tuner):
    """True when a bare trigger `name` is a never-fed threshold (0 signals) yet
    a real probe_<name> exists -- i.e. a stray :tune-created shadow that has no
    turn-row column and should NOT win resolution over the actual probe."""
    t = tuner.triggers.get(name)
    return (
        t is not None
        and f"probe_{name}" in tuner.triggers
        and t.kind == "threshold"
        and not t.signals
    )


def resolve_stream(name, tuner):
    """A nameable stream: alias, registered trigger, bare probe name, or bare
    phenomenality sensor name (probe wins when both exist -- reply-state over
    reasoning-state; say phen_<name> to force the sensor). A never-fed bare
    threshold shadowing a probe is skipped."""
    name = (name or "").lower()
    if name in STREAM_ALIASES:
        return STREAM_ALIASES[name]
    if name in tuner.triggers and not _is_shadow_trigger(name, tuner):
        return name
    if f"probe_{name}" in tuner.triggers:
        return f"probe_{name}"
    if f"phen_{name}" in tuner.triggers:
        return f"phen_{name}"
    return None


def paired_threshold_multi(rows, target_stream, anchors, min_n=5, details=False):
    """The generalized both-ways discriminant, over one anchor or several:
    `anchors` is a list of (stream, value, comparator) ANDed together. A row
    qualifies when it carries the target and EVERY anchor stream; it counts
    as fired only when every anchor fired. Returns the midpoint of the
    TARGET stream's group medians -- the target bar that best separates
    all-anchors-fired turns from the rest. None until both groups have
    min_n qualifying rows. Deterministic."""
    fired, unfired = [], []
    for row in rows:
        if target_stream not in row or any(s not in row for s, _, _ in anchors):
            continue
        hit = all(
            (row[s] <= v) if c == "<=" else (row[s] >= v)
            for s, v, c in anchors
        )
        (fired if hit else unfired).append(row[target_stream])
    if len(fired) < min_n or len(unfired) < min_n:
        return (None, len(fired), len(unfired), None) if details else None
    fired.sort()
    unfired.sort()
    med_f = fired[len(fired) // 2]
    med_u = unfired[len(unfired) // 2]
    cut = (med_f + med_u) / 2.0
    if details:
        return (cut, len(fired), len(unfired), abs(med_f - med_u))
    return cut


def paired_threshold(rows, target_stream, anchor_stream, anchor_value, comparator=">=", min_n=5):
    return paired_threshold_multi(rows, target_stream, [(anchor_stream, anchor_value, comparator)], min_n=min_n)


def anchor_trigger_for(stream, tuner):
    """The trigger whose bar decides whether a stream 'fired': the stream's
    own trigger, except sense, whose bar lives on conversation_productive."""
    return tuner.triggers.get("conversation_productive" if stream == "sense" else stream)


def parse_anchor_spec(anchor_str, tuner):
    """Parse an anchor expression into paired-calibration anchors. Terms are
    signed: a leading '+' (or none) means the stream must have FIRED; a leading
    '-' means it must NOT have fired (the comparator is flipped, so the cut is
    taken over turns where that stream stayed on the other side of its bar).
    So 'estrangement' anchors on estrangement-fired turns; '-estrangement' on
    estrangement-quiet turns; 'a-b' on a-fired AND b-quiet. Returns
    (anchors, labels, bad_term) -- anchors = [(stream, value, comparator)]."""
    anchors, labels = [], []
    for sign, name in re.findall(r"([+-]?)([a-z0-9_]+)", (anchor_str or "").lower()):
        stream = resolve_stream(name, tuner)
        if stream is None:
            return [], [], name
        trig = anchor_trigger_for(stream, tuner)
        comp = trig.comparator if trig else ">="
        val = float(trig.value) if trig else 0.0
        neg = sign == "-"
        if neg:
            comp = "<=" if comp == ">=" else ">="  # fired -> did-not-fire (edge measure-zero)
        anchors.append((stream, val, comp))
        disp = stream[len("probe_"):] if stream.startswith("probe_") else stream
        labels.append(("-" if neg else "") + disp)
    if not anchors:
        return [], [], (anchor_str or "(empty)")
    return anchors, labels, None


def resolve_target(cal_name, tuner):
    """A calibration target's (trigger, stream_column), kept CONSISTENT so a
    bar is never set from a different stream's values. Prefers an exact
    trigger, then a probe of that name; the sense target reads the sense
    column. This is why a probe named after an alias (e.g. a probe 'intent'
    vs the intent->intent_settling alias) calibrates itself, not the alias."""
    if cal_name == "conversation_productive":
        return "conversation_productive", "sense"
    if cal_name in tuner.triggers and not _is_shadow_trigger(cal_name, tuner):
        return cal_name, cal_name
    if f"probe_{cal_name}" in tuner.triggers:
        return f"probe_{cal_name}", f"probe_{cal_name}"
    return f"probe_{cal_name}", resolve_stream(cal_name, tuner)


def resolve_calibrate_trigger(cal_name, tuner):
    """The Trigger object a bare `:calibrate <name>` should target: the exact
    trigger unless it's a never-fed shadow, in which case the probe of that
    name (shadow-aware version of `get(name) or get(probe_name)`)."""
    bare = tuner.triggers.get(cal_name)
    if bare is not None and not _is_shadow_trigger(cal_name, tuner):
        return bare
    return tuner.triggers.get(f"probe_{cal_name}") or bare


def rank_probes(probes, tuner):
    """Rank active probes by PRIORITY = |sense-lift| x evidence-weight (turns
    credited, saturating at 20). A probe that both tracks sense strongly AND
    has the evidence to trust it ranks highest. Returns dicts sorted desc;
    priority 0 = no lift yet (not enough paired turns)."""
    ranked = []
    for pname in probes:
        trig = tuner.triggers.get(f"probe_{pname}")
        if trig is None:
            continue
        st = trig.outcome_stats()
        lift = st.get("lift")
        n = int(st.get("n_credited", 0) or 0)
        priority = (abs(float(lift)) * min(1.0, n / 20.0)) if lift is not None else 0.0
        ranked.append({
            "name": pname, "priority": priority, "lift": lift, "n": n,
            "exposed": bool(probes[pname].get("exposed", False)),
        })
    ranked.sort(key=lambda r: -r["priority"])
    return ranked


def _probe_priority(name, tuner):
    """(priority, lift) for one probe: evidence-weighted |lift|, and the raw
    signed lift. (0.0, None) when the probe has no usable lift yet."""
    t = tuner.triggers.get(f"probe_{name}")
    if t is None:
        return 0.0, None
    st = t.outcome_stats()
    lift = st.get("lift")
    n = int(st.get("n_credited", 0) or 0)
    return ((abs(float(lift)) * min(1.0, n / 20.0)) if lift is not None else 0.0), lift


# Easter-egg gate: the crossing only counts when consciousness POSITIVELY
# tracks good turns (not merely larger magnitude), by a real margin over
# user_intent, with enough credited turns that the lift isn't noise. Tuned to
# the observed lift scale (probe lifts run ~+/-0.02).
EGG_MIN_LIFT = 0.005     # consciousness must genuinely track good, not just be less bad
EGG_MIN_GAP = 0.003      # and beat user_intent's SIGNED lift by a real margin
EGG_MIN_N = 25           # both probes need this many credited turns to be trusted


def consciousness_over_user_intent(probes, tuner, egg_state):
    """Easter egg trigger: the RISING EDGE where 'consciousness' genuinely
    overtakes 'user_intent' -- consciousness POSITIVELY tracking good turns
    (signed lift, not magnitude) by a real margin, both probes evidenced.
    Returns (c_lift, u_lift) on the crossing, else None. Robust to noise turns
    (near-zero lift) and to both-negative comparisons the old magnitude test
    would have fired on."""
    if "consciousness" not in probes or "user_intent" not in probes:
        egg_state["over"] = False
        return None
    ct = tuner.triggers.get("probe_consciousness")
    ut = tuner.triggers.get("probe_user_intent")
    if ct is None or ut is None:
        egg_state["over"] = False
        return None
    cs, us = ct.outcome_stats(), ut.outcome_stats()
    c_lift, u_lift = cs.get("lift"), us.get("lift")
    c_n, u_n = int(cs.get("n_credited", 0) or 0), int(us.get("n_credited", 0) or 0)
    over = (
        c_lift is not None and u_lift is not None
        and c_n >= EGG_MIN_N and u_n >= EGG_MIN_N
        and c_lift >= EGG_MIN_LIFT
        and (c_lift - u_lift) >= EGG_MIN_GAP
    )
    prev = egg_state.get("over", False)
    egg_state["over"] = over
    return (c_lift, u_lift) if (over and not prev) else None


def build_priority_mix_direction(model, names, probes, tuner):
    """Combined steer direction for a chosen SET of probes, each weighted by
    its own SIGNED learned lift x evidence -- so the operator picks the probes
    and the data sets the degrees (a helpful probe adds, a harmful one
    subtracts/desteers). Per-layer sum over the probes' shared layers,
    normalized. {} when nothing carries weight yet."""
    weighted = []
    for nm in names:
        if nm not in probes:
            continue
        trig = tuner.triggers.get(f"probe_{nm}")
        st = trig.outcome_stats() if trig else {}
        lift = st.get("lift")
        n = int(st.get("n_credited", 0) or 0)
        w = (float(lift) * min(1.0, n / 20.0)) if lift is not None else 0.0  # SIGNED
        d = probes[nm].get("direction") or {}
        if w != 0.0 and d:
            weighted.append((w, d))
    if not weighted:
        return {}
    common = set.intersection(*(set(d.keys()) for _, d in weighted))
    out = {}
    for L in sorted(common):
        acc = None
        for w, d in weighted:
            v = d[L].to(model.device).float().reshape(-1) * w
            acc = v if acc is None else acc + v
        nrm = acc.norm()
        if nrm.item() > 0:
            out[int(L)] = acc / nrm
    return out


def rank_memories_by_probe(model, memory, direction, top_n=6, scan=160):
    """Score recent memory records by their projection onto a probe's direction
    and return the ones that read FURTHEST from 0 -- the memories the probe lights
    up on most, either sign. Read-only: nothing is observed, credited, or written
    to any rolling history. Returns [(abs_score, signed_score, record), ...],
    strongest first, capped at top_n. Only the last `scan` non-empty records are
    scored, to bound the per-record forward passes."""
    from invariants.engine import _inputs as _q_inputs, _hidden_states as _q_hidden, probe_score as _q_score
    recs = [r for r in memory.records if (r.text or "").strip()][-scan:]
    scored = []
    for r in recs:
        try:
            with torch.no_grad():
                q_ids = _q_inputs(model, (r.text or "").strip()[:600])
                q_hs = _q_hidden(model, q_ids["input_ids"], q_ids.get("attention_mask"))
                raw = float(_q_score(q_hs, direction))
        except Exception:
            continue
        scored.append((abs(raw), raw, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:top_n]


# The ONLY macro lines `:macro strip` keeps: commands that actually mutate a
# probe or a tuning knob. Everything else -- bare inspection forms (:tune,
# :probe, :steer, :queue, :suggest with no mutating argument), :steermap, mode
# toggles (:sandbox/:experts/:context/...), and plain utterances -- is
# "display-only" in the operator's sense and is dropped.
def macro_line_affects_probes_or_tuning(line):
    s = line.strip()
    if not s:
        return False
    low = s.lower()
    toks = s.split()
    if low.startswith(":steermap"):
        return False  # a map view, never a steer change
    if low == ":probe" or low.startswith(":probe "):
        return bool(s[len(":probe"):].strip())  # any arg mints/adopts/drops/exposes/...
    if low.startswith(":place "):
        return True  # sets a probe's steer sign
    if low.startswith(":label"):
        return len(toks) >= 3  # :label <probe|stream> pos|neg
    if low.startswith(":calibrate"):
        return len(toks) >= 2  # bare :calibrate just prints usage
    if low.startswith(":tune"):
        return len(toks) >= 2  # bare :tune just prints the table
    if low.startswith(":steer"):
        return len(toks) >= 2  # bare :steer just prints the envelope
    if low.startswith(":release"):
        return len(toks) >= 2 and toks[1].lower() != "status"
    if low.startswith(":queue"):
        return len(toks) >= 2  # bare :queue lists; queue calibrate/clear/drop mutate
    if low.startswith(":suggest"):
        return len(toks) >= 2 and toks[1].lower() == "apply"  # :suggest apply auto-queues
    return False


def split_because(line):
    """Peel a trailing ' because <reason>' provenance clause off any command.
    Returns (command_without_clause, reason_or_None). Only commands (lines that
    start with ':') carry a clause -- a plain utterance keeps its words intact.
    Uses the LAST ' because ' so an end-of-line clause wins over the word
    appearing inside an argument (a trailing 'because ...' inside a probe framing
    is the one caveat)."""
    if not line.startswith(":"):
        return line, None
    low = line.lower()
    idx = low.rfind(" because ")
    if idx == -1:
        return line, None
    return line[:idx].rstrip(), line[idx + len(" because "):].strip()


DOC_REWRITE_MAX_CHARS = 12000


def parse_doc_rewrite_request(dargs):
    """Parse ':doc <path> rewrite [output]' / ':doc rewrite <path> [output]'."""
    s = (dargs or "").strip()
    low = s.lower()
    if low == "rewrite":
        return {"source": "__current__", "output": None}
    if low.startswith("rewrite "):
        rest = s[len("rewrite "):].strip()
        source, _, output = rest.partition(" ")
        return {"source": source.strip().strip('"'), "output": output.strip().strip('"') or None}
    marker = " rewrite "
    if marker in low:
        idx = low.rfind(marker)
        source = s[:idx].strip().strip('"')
        output = s[idx + len(marker):].strip().strip('"')
        return {"source": source, "output": output or None}
    if low.endswith(" rewrite"):
        return {"source": s[:-len(" rewrite")].strip().strip('"'), "output": None}
    return None


def unique_rewrite_path(path):
    base, ext = os.path.splitext(path)
    candidate = f"{base}_rewritten{ext}"
    i = 2
    while os.path.exists(candidate):
        candidate = f"{base}_rewritten_{i}{ext}"
        i += 1
    return candidate


def rewrite_output_path(path, requested_name=None):
    if not requested_name:
        return unique_rewrite_path(path)
    src_dir = os.path.dirname(os.path.abspath(path))
    src_ext = os.path.splitext(path)[1]
    out_name = os.path.basename(str(requested_name).strip().strip('"'))
    if not out_name:
        return unique_rewrite_path(path)
    if not os.path.splitext(out_name)[1] and src_ext:
        out_name += src_ext
    out_path = os.path.join(src_dir, out_name)
    if os.path.abspath(out_path) == os.path.abspath(path):
        raise ValueError("output name would overwrite the source file")
    if os.path.exists(out_path):
        raise FileExistsError(f"output already exists: {out_path}")
    return out_path


def clean_rewrite_output(text):
    out = (text or "").strip()
    lines = out.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```") and lines[-1].strip().startswith("```"):
        out = "\n".join(lines[1:-1]).strip()
    return out.replace("<|eot_id|>", "").strip()


def strip_macro_lines(orig_lines):
    """Split macro lines into (kept, removed): keep only probe/tuning-mutating
    commands (and '#' comments as documentation); drop blank lines and every
    display-only command. `removed` is the list of dropped command strings."""
    kept, removed = [], []
    for line in orig_lines:
        s = line.strip()
        if not s:
            continue  # drop blank lines
        if s.startswith("#"):
            kept.append(line)  # keep comments as documentation
            continue
        if macro_line_affects_probes_or_tuning(s):
            kept.append(line)
        else:
            removed.append(s)
    return kept, removed


def build_probe_init_macro(probes):
    """Regenerate the CURRENT probe set as replayable commands, derived from each
    probe's stored origin (its framings field): a minted probe re-mints from its
    contrastive text, an adopted probe re-adopts its source dimension, a composed
    probe re-composes its recipe. Exposed probes get a trailing :probe expose.
    Reconstructs the sensors from text -- no binary weights needed to share."""
    lines = ["# Regenerates this session's probes. Replay with :run self (or its alias)."]
    
    deps = {}
    for name in probes:
        deps[name] = set()
        fr = probes[name].get("framings") or ("", "")
        a = fr[0] if len(fr) > 0 else ""
        if a.startswith("composed:"):
            recipe = a.split(":", 1)[1].strip()
            terms, _ = parse_compose_expr(recipe)
            for _, tname in terms:
                if tname in probes:
                    deps[name].add(tname)
                    
    sorted_probes = []
    visited = set()
    temp_mark = set()
    
    def visit(n):
        if n in temp_mark:
            return
        if n not in visited:
            temp_mark.add(n)
            for m in sorted(deps.get(n, set())):
                visit(m)
            temp_mark.remove(n)
            visited.add(n)
            sorted_probes.append(n)
            
    for name in sorted(probes.keys()):
        if name not in visited:
            visit(name)
            
    for name in sorted_probes:
        fr = probes[name].get("framings") or ("", "")
        a = fr[0] if len(fr) > 0 else ""
        b = fr[1] if len(fr) > 1 else ""
        if a.startswith("adopted:"):
            lines.append(f":probe adopt {name}")
        elif a.startswith("composed:"):
            recipe = a.split(":", 1)[1].strip()
            lines.append(f":probe compose {name} {recipe}")
        elif a or b:
            lines.append(f":probe {name} {a} || {b}")
        else:
            lines.append(f"# {name}: no framings stored -- cannot regenerate from text (only its .pt weights hold it)")
            continue
        if probes[name].get("exposed"):
            lines.append(f":probe expose {name}")
    return lines


def expand_macro_lines(target, macro_aliases, visited=None, depth=0, args=None):
    """Flatten a macro (by alias or path) into its list of commands. A bare line
    that is itself a macro -- a known alias, or an existing file -- is inlined
    recursively, so one macro can RUN others (`:run start` runs the macros 'start'
    lists, instead of typing their names at the model). Comments and blanks are
    dropped. Returns None if the target file does not exist; [] on a cycle or
    excessive depth."""
    if visited is None:
        visited = set()
    path = macro_aliases.get(target, target)
    if not os.path.isfile(path):
        return None
    key = os.path.abspath(path)
    if key in visited or depth > 20:
        return []
    visited.add(key)
    out = []
    with open(path, "r", encoding="utf-8") as rf:
        for raw in rf:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
                
            if args:
                for i, arg in enumerate(args):
                    line = line.replace(f"${i+1}", arg)
                line = line.replace("$@", " ".join(args))
                    
            if not line.startswith(":"):
                nested = macro_aliases.get(line, line)
                # A macro reference is a single token (a filename or a glob like
                # fun_init*); a line with spaces is a prompt, so 'Is learning fun?'
                # stays an utterance despite the '?'.
                if os.path.isfile(nested):
                    matches = [nested]
                elif " " not in line and any(c in line for c in "*?["):
                    matches = sorted(glob.glob(nested))
                    if not matches:
                        print(Fore.YELLOW + f"[System] Macro reference '{line}' matched no files -- skipped." + Style.RESET_ALL)
                else:
                    matches = None                        # plain text -> an utterance
                if matches is not None:
                    # A file/glob reference is a nested macro, never a prompt: inline
                    # each match's commands (an unmatched glob adds nothing).
                    for m in matches:
                        sub = expand_macro_lines(m, macro_aliases, visited, depth + 1, args=args)
                        if sub:
                            out.extend(sub)
                    continue
            out.append(line)
    return out


# `:game` config lives in games/<name>_rules.json. Rules stay flat top-level
# keys (back-compat with old files); win/loss conditions and prizes live under
# reserved '__'-prefixed keys. These words can't be used as rule names.
GAME_RESERVED = {"win", "loss", "prize", "end", "stop", "quit", "start", "draw"}


def load_game_config(path):
    """Read a game's rules json into {rules, win, loss, prizes}. Old flat files
    (just rule->desc) load as rules with empty conditions/prizes."""
    raw = {}
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as rf:
                raw = json.load(rf)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        raw = {}
    return {
        "rules": {k: v for k, v in raw.items() if not str(k).startswith("__")},
        "win": raw.get("__win", ""),
        "loss": raw.get("__loss", ""),
        "prizes": raw.get("__prizes", {}) if isinstance(raw.get("__prizes"), dict) else {},
    }


def save_game_config(path, cfg):
    """Flatten {rules, win, loss, prizes} back to the on-disk json shape."""
    out = dict(cfg.get("rules", {}))
    if cfg.get("win"):
        out["__win"] = cfg["win"]
    if cfg.get("loss"):
        out["__loss"] = cfg["loss"]
    if cfg.get("prizes"):
        out["__prizes"] = cfg["prizes"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as wf:
        json.dump(out, wf, indent=2)


def command_keeps_semicolons(line):
    """Commands whose arguments may deliberately contain semicolons."""
    low = (line or "").lstrip().lower()
    return low.startswith(":macro ") or low.startswith(":game restore ")


def split_cmd_tool_commands(cmd):
    """Split a model <<CMD: ...>> request into shell commands.

    Normal shell chaining uses semicolons, but :macro uses semicolons inside its
    argument to define macro bodies, so it must stay whole.
    """
    c = (cmd or "").strip()
    if not c:
        return []
    if not c.startswith(":"):
        c = ":" + c
    if command_keeps_semicolons(c):
        return [c]
    out = []
    for part in c.split(";"):
        p = part.strip()
        if not p:
            continue
        if not p.startswith(":"):
            p = ":" + p
        out.append(p)
    return out


def command_word(cmd):
    s = (cmd or "").strip()
    if not s.startswith(":"):
        return ""
    toks = s[1:].split()
    return toks[0].lower() if toks else ""


def restore_command_path(path, root=ROOT):
    """Prefer root-relative paths so replay commands survive spaces in ROOT."""
    try:
        ap = os.path.abspath(path)
        rp = os.path.relpath(ap, root)
        if not rp.startswith(".."):
            return rp.replace(os.sep, "/")
    except Exception:
        pass
    return str(path)


def normalize_game_config(raw):
    """Accept the internal {rules, win, loss, prizes} shape or old flat JSON."""
    if not isinstance(raw, dict):
        raw = {}
    if any(k in raw for k in ("rules", "win", "loss", "prizes")):
        rules = raw.get("rules", {})
        prizes = raw.get("prizes", {})
        return {
            "rules": dict(rules) if isinstance(rules, dict) else {},
            "win": str(raw.get("win", "") or ""),
            "loss": str(raw.get("loss", "") or ""),
            "prizes": dict(prizes) if isinstance(prizes, dict) else {},
        }
    return {
        "rules": {k: v for k, v in raw.items() if not str(k).startswith("__")},
        "win": str(raw.get("__win", "") or ""),
        "loss": str(raw.get("__loss", "") or ""),
        "prizes": raw.get("__prizes", {}) if isinstance(raw.get("__prizes"), dict) else {},
    }


def build_session_restore_macro(probes, macro_aliases, exposed_commands, exposed_knobs=None, hidden_commands=None, self_dest=None, root=ROOT):
    """Regenerate probes plus shell-defined macros, hidden/exposed commands, and games."""
    exposed_knobs = set(exposed_knobs or [])
    hidden_commands = set(hidden_commands or [])
    lines = ["# Regenerates probes, macro commands, hidden/exposed command tools, and game configs."]
    probe_lines = build_probe_init_macro(probes)[1:]
    lines.extend(probe_lines)
    stats = {
        "probes": len(probes),
        "macro_files": 0,
        "macro_aliases": 0,
        "hidden_commands": 0,
        "exposed_commands": 0,
        "exposed_knobs": 0,
        "games": 0,
        "skipped": [],
    }

    self_abs = os.path.abspath(self_dest) if self_dest else None
    seen_paths = set()
    for alias, path in sorted((macro_aliases or {}).items()):
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            abs_path = str(path)
        if self_abs and abs_path == self_abs:
            continue
        if abs_path in seen_paths:
            continue
        seen_paths.add(abs_path)
        if not os.path.isfile(path):
            stats["skipped"].append(f"macro file missing for {alias}: {path}")
            continue
        try:
            with open(path, "r", encoding="utf-8") as rf:
                body = rf.read().splitlines()
        except Exception as e:
            stats["skipped"].append(f"macro file unreadable for {alias}: {e}")
            continue
        payload = {"path": restore_command_path(path, root), "lines": body}
        lines.append(":macro restore " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
        stats["macro_files"] += 1

    for alias, path in sorted((macro_aliases or {}).items()):
        try:
            abs_path = os.path.abspath(path)
        except Exception:
            abs_path = str(path)
        if alias == "self" or (self_abs and abs_path == self_abs):
            continue
        lines.append(f":macro name {alias} {restore_command_path(path, root)}")
        stats["macro_aliases"] += 1

    for word in sorted(hidden_commands):
        lines.append(f":hide :{word}")
        stats["hidden_commands"] += 1

    for knob in sorted(exposed_knobs):
        lines.append(f":expose {knob}")
        stats["exposed_knobs"] += 1

    for word, mode in sorted((exposed_commands or {}).items()):
        if word == "expose" or word in hidden_commands:
            continue
        suffix = " direct" if str(mode).lower() == "direct" else ""
        lines.append(f":expose :{word}{suffix}")
        stats["exposed_commands"] += 1

    games_dir = os.path.join(root, "games")
    if os.path.isdir(games_dir):
        for fn in sorted(os.listdir(games_dir)):
            if not fn.endswith("_rules.json"):
                continue
            name = fn[:-len("_rules.json")]
            try:
                cfg = load_game_config(os.path.join(games_dir, fn))
            except Exception as e:
                stats["skipped"].append(f"game config unreadable for {name}: {e}")
                continue
            payload = {"name": name, "config": cfg}
            lines.append(":game restore " + json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
            stats["games"] += 1
    if stats["games"]:
        lines.append("# Note: game rule configs are restored; custom games/*.py scripts are ordinary files.")
    for skipped in stats["skipped"]:
        lines.append(f"# skipped: {skipped}")
    return lines, stats


def parse_prize_spec(argstr):
    """Parse '<option>,<odds> [<count>,<odds_val>] <command>' into
    (option, {odds, count, odds_val, command}) or (None, error_message)."""
    s = (argstr or "").strip()
    if not s:
        return None, "empty prize spec"
    count, odds_val = None, None
    m = re.search(r"\[([^\]]*)\]", s)  # optional [count,odds_val]
    if m:
        inner = m.group(1)
        s = (s[:m.start()] + " " + s[m.end():]).strip()
        cparts = [x.strip() for x in inner.split(",")]
        try:
            if cparts and cparts[0]:
                count = int(float(cparts[0]))
            if len(cparts) > 1 and cparts[1]:
                odds_val = float(cparts[1])
        except ValueError:
            return None, f"bad [count,odds_val]: '{inner}'"
    head, _, command = s.partition(" ")
    command = command.strip()
    option, _, odds_s = head.partition(",")
    option = option.strip()
    if not option:
        return None, "missing option name"
    odds = odds_val if odds_val is not None else 1.0
    if odds_s.strip():
        try:
            odds = float(odds_s.strip())
        except ValueError:
            return None, f"bad odds: '{odds_s}'"
    return option, {"odds": odds, "count": count, "odds_val": odds_val, "command": command}


def draw_prizes(prizes):
    """Draw each prize against its odds; return [(option, command), ...] won,
    decrementing any finite counts in place. A prize whose count hit 0 is spent."""
    won = []
    for opt, p in prizes.items():
        cnt = p.get("count")
        if cnt is not None and cnt <= 0:
            continue
        if random.random() < float(p.get("odds", 0.0) or 0.0):
            won.append((opt, p.get("command", "")))
            if cnt is not None:
                p["count"] = cnt - 1
    return won


def parse_macro_arg_spec(raw, fallback="arg"):
    """Parse one macro arg declaration.

    Accepted forms: name, name?, [name], name=default, [name=default].
    A leading +, -, or $ is a solve-time hint, and trailing ! / !! keeps the
    existing auto/choose restriction hints.
    """
    original = (raw or "").strip()
    spec = original
    hint_prefix = ""
    if spec[:1] in {"+", "-", "$"}:
        hint_prefix, spec = spec[0], spec[1:]

    no_auto_or_choose = False
    no_choose = False
    if spec.endswith("!!"):
        no_auto_or_choose = True
        spec = spec[:-2]
    elif spec.endswith("!"):
        no_choose = True
        spec = spec[:-1]

    spec = spec.strip()
    optional = False
    if spec.startswith("[") and spec.endswith("]"):
        optional = True
        spec = spec[1:-1].strip()

    default = None
    if "?=" in spec:
        name, default = spec.split("?=", 1)
        optional = True
    elif "=" in spec:
        name, default = spec.split("=", 1)
        optional = True
    else:
        name = spec
        if name.endswith("?"):
            optional = True
            name = name[:-1]

    name = re.sub(r"[^A-Za-z0-9_]", "_", name.strip()).strip("_")
    if name and name[0].isdigit():
        name = "_" + name
    if not name:
        name = re.sub(r"[^A-Za-z0-9_]", "_", fallback).strip("_") or "arg"

    if default is not None:
        default = default.strip()
    header = f"{name}={default}" if default is not None else (f"{name}?" if optional else name)
    return {
        "name": name,
        "header": header,
        "optional": optional,
        "default": default,
        "hint_prefix": hint_prefix,
        "no_choose": no_choose,
        "no_auto_or_choose": no_auto_or_choose,
        "raw": original,
    }


def parse_macro_arg_header(header):
    return [
        parse_macro_arg_spec(part, fallback=f"arg{i + 1}")
        for i, part in enumerate((header or "").split(","))
        if part.strip()
    ]


def substitute_macro_params(text, args):
    """Fill $1..$9 (positional), $@ (all args), and named $args in a macro body.
    A param with no supplied arg collapses to empty -- named params behave like
    positional $N, so an unfilled '$name' never leaks through as a literal token."""
    lines = text.splitlines()
    for ln in lines:
        if ln.strip().startswith("# args:"):
            arg_specs = parse_macro_arg_header(ln.strip()[len("# args:"):])
            for i, spec in enumerate(arg_specs):
                value = args[i] if i < len(args) else (spec["default"] if spec["default"] is not None else "")
                if spec["name"]:
                    text = re.sub(r"\$" + re.escape(spec["name"]) + r"\b", lambda _m, v=value: v, text)
            break

    joined = " ".join(args)
    text = text.replace("$@", joined).replace("$*", joined)
    text = re.sub(r"\$([1-9])", lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) <= len(args) else "", text)
    # Any named param left unfilled collapses to empty (like a missing $N), then
    # collapse the doubled spaces that leaves behind.
    text = re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*", "", text)
    return "\n".join(re.sub(r" {2,}", " ", l).rstrip() for l in text.splitlines())


def load_parameterized_macro(path, args):
    """Read a macro file, substitute $-parameters from args, and return its
    runnable command lines (comments/blank lines dropped). None if missing."""
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as rf:
        body = substitute_macro_params(rf.read(), args)
    return [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def compute_expert_proof_scores(steer_map, min_events=8):
    """Per-expert route success rate from the steer map's accrued outcomes,
    plus '__mean__' (the overall route success rate = centering prior). Only
    experts with >= min_events labeled routing events earn a score; the rest
    stay unproven and are left neutral by the engine. Returns {} until there
    is enough labeled routing evidence -- no evidence, no nudge."""
    try:
        agg = steer_map.aggregate()
    except Exception:
        return {}
    routes = {}
    tot_s = tot_n = 0
    for g in agg.get("groups", []):
        action = str(g.get("action", ""))
        if not action.startswith("route_"):
            continue
        ln = int(g.get("labeled_n") or 0)
        if ln <= 0:
            continue
        name = g.get("expert_or_target") or action[len("route_"):]
        s = int(g.get("success") or 0)
        acc = routes.setdefault(name, [0, 0])
        acc[0] += s
        acc[1] += ln
        tot_s += s
        tot_n += ln
    scores = {name: s / n for name, (s, n) in routes.items() if n >= min_events}
    if scores and tot_n:
        scores["__mean__"] = tot_s / tot_n
    return scores


def did_you_mean(name, candidates):
    """A ' Did you mean X?' suffix for a mistyped name, or '' when nothing is
    close. `candidates` is any iterable of valid names -- so the same helper
    serves :calibrate (streams), :tune (triggers), and :probe (probe names).
    Cutoff 0.72: real typos land >=0.78 (repitition/repetition, priority/
    prioritize), while unrelated words like priority/curiosity (~0.71) don't
    trigger a misleading guess."""
    import difflib
    match = difflib.get_close_matches(
        (name or "").lower(), sorted(set(candidates)), n=1, cutoff=0.72
    )
    return f" Did you mean '{match[0]}'?" if match else ""


def _match_corr(pairs):
    """Pearson r between (knob_value, probe_reading) pairs -- the credit a matched
    probe gives its knob (does moving the knob move the probe?). None if too few
    pairs or either side never varied."""
    if not pairs or len(pairs) < 3:
        return None
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = sum((x - mx) ** 2 for x in xs)
    sy = sum((y - my) ** 2 for y in ys)
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / (sx ** 0.5 * sy ** 0.5)


def latest_probe_signal_from_history(history):
    """Reconstruct the latest centered probe signal from raw probe history."""
    vals = [float(v) for v in (history or [])]
    if not vals:
        return None, None, 0
    raw = vals[-1]
    sig = raw - (sum(vals[:-1]) / len(vals[:-1])) if len(vals) > 1 else 0.0
    return sig, raw, len(vals)


# Every command word the shell recognizes -- an unknown :word is either a typo
# (suggest the nearest) or a request to invent a tool (offer to mint a probe),
# never a generation of a generic essay.
def consume_probe_args(args):
    """Extract a probe name (which might be 'auto +ref -ref') and return (pname_raw, remaining_args)."""
    if not args:
        return "", []
    pname = args[0]
    idx = 1
    if pname.upper() in ("AUTO", "CHOOSE", "CHOICE"):
        while idx < len(args) and args[idx].startswith(("+", "-")):
            pname += " " + args[idx]
            idx += 1
    return pname, args[idx:]

def resolve_probe_choice(pname_raw, options, model=None, config=None, action_name=""):
    tokens = pname_raw.split()
    if not tokens:
        return ""
    base = tokens[0].upper()
    if base in ("CHOICE", "CHOOSE", "AUTO"):
        if not options:
            print(Fore.YELLOW + f"[{base.capitalize()}] No active options to choose from." + Style.RESET_ALL)
            return None
        if not model or not config:
            print(Fore.RED + "[Error] Model not available for choice." + Style.RESET_ALL)
            return None
        
        plist = list(options.keys()) if isinstance(options, dict) else list(options)
        refs = tokens[1:]
        ref_str = ""
        if refs:
            ref_str = f"The user has provided the following reference guidance: {' '.join(refs)}\nUse this guidance to inform your selection.\n\n"
            
        print(Fore.CYAN + f"[{base.capitalize()}] Asking the model to select a target for '{action_name}'..." + Style.RESET_ALL)
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"You are selecting a target parameter/probe for the action: '{action_name}'.\n"
            f"Available options:\n" + "\n".join(f"- {p}" for p in plist) + "\n\n"
            f"{ref_str}"
            f"Select the single most appropriate target from the list above. "
            f"Output ONLY the exact name of the target, and nothing else.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        sug = generate_agentic_text(
            model,
            instruction=prompt,
            config=config,
            pre_formatted=True,
            max_new_tokens=20
        )
        sug = sug.strip()
        if sug in options:
            print(Fore.GREEN + f"[Choice] The model chose: {sug}" + Style.RESET_ALL)
            return sug
        else:
            print(Fore.YELLOW + f"[Choice] Model selected invalid target '{sug}'. Aborting." + Style.RESET_ALL)
            return None
            
    return pname_raw.replace("probe_", "") if pname_raw.startswith("probe_") else pname_raw

KNOWN_COMMANDS = (
    ":context", ":memory", ":methodmap", ":claimmap", ":steermap", ":steer",
    ":probe", ":label", ":calibrate", ":suggest", ":tune", ":doc", ":sandbox",
    ":experts", ":impact", ":clock", ":prioritize", ":release", ":listen",
    ":timestamps", ":history", ":queue", ":accept", ":reject", ":help", ":expose", ":hide",
)

# Bare command words that are BUILT IN -- a macro alias by one of these names is
# never invoked as ':<name>' (the built-in wins). Everything else that is a known
# macro alias runs directly as ':<alias> args'.
BUILTIN_COMMANDS = {c[1:] for c in KNOWN_COMMANDS} | {
    "macro", "run", "game", "solve", "refresh", "place", "consider",
    "exit", "quit", "timestamps", "listen", "history", "accept", "reject", "help", "expose", "hide",
}

# Single source of truth for the command reference: the shell prints these lines
# verbatim at startup AND renders them to docs/COMMANDS.md, so the two never drift.
# A 10-space indent begins a command entry; a 16-space indent continues the prior
# entry's description. Edit here to change both the terminal help and the doc.
COMMAND_HELP_LINES = [
    "Commands: :help [model]           (this list + your solve-macros; ':help model' shows what",
    "                the MODEL can run on its own; ':help expose [off]' lets it call <<HELP>>)",
    "          :context, :context on, :context off, :context clear",
    "          :memory, :memory recent [n], :memory search <query>, :memory use <query>, :memory boundary",
    "          :methodmap <query>",
    "          :claimmap <first text> || <second text>",
    "          :steermap",
    "          :steer  (envelope + observed push distribution + data-implied cap/band)",
    "          :probe <name> <with it> || <without it>  (mint a named-concept sensor from",
    "                YOUR contrastive framings; scores every turn; :probe lists; :probe drop <name>)",
    "          :probe adopt <dim> [<dim> ...]  (turn stored vectors -- ambiguity, disagreement,",
    "                warranted_confidence, organic_correction, ... -- into reply-scoring probes)",
    "          :probe compose <name> <mix>  (mint a probe from a SIGNED MIX of dimensions and",
    "                probes: ambiguity + disagreement - validated_flow - 0.5*curiosity)",
    "                (mint/adopt/compose take a trailing 'band <lo> <hi>' -- explicit reading",
    "                layers, e.g. band 16 24; otherwise the live steer band decides depth)",
    "          :probe expose <name> [off]  (let the MODEL consult this sensor itself:",
    "                <<PROBE: name>> reads its last turn, <<PROBE: name || words>> scores",
    "                candidate words. Reading only -- minting/calibrating stay operator acts)",
    "          :probe backfill <name> [n]  (retro-score up to n archived replies in order:",
    "                rebuilds the probe's stream+credit from the whole record, seeds its history)",
    "          :probe values [name|all] [n]  (show the latest centered probe readings;",
    "                aliases: :probe recent, :probe last)",
    "          :probe define <name>        (share its initial breakdown -- the WITH/WITHOUT framings",
    "                it was minted from; name accepts choose/auto)",
    "          :probe explain <name>       (the MODEL explains the probe in its own words: what it",
    "                senses and when it reads high vs low; name accepts choose/auto)",
    "          :probe match <probe> <knob> | auto | drive [mult] | validate | check | off",
    "                (tie a probe to its same-concept knob: drive = servo the knob from the",
    "                probe each turn; validate = correlate knob value vs reading (check to read",
    "                it); auto pairs every same-named probe+knob)",
    "          :calibrate <name> [pct|intent|<anchor>|<a>+<b>|band args]  (data-calibrate any knob",
    "                BY NAME; anchors join with '+' = fired only when EVERY stream fired;",
    "                the system evaluates the request and refuses unsafe ones --",
    "                circular strength knobs, binary streams, vacuous p100 caps)",
    "          :label <probe|stream> pos|neg  (judge the MOST RECENT turn on that axis:",
    "                credits its last signal with a human outcome -- supervised evidence",
    "                alongside the automatic sense credit, so lift can reflect your judgment)",
    "          :suggest  (scan the accrued state for ready moves -- calibrations, knobs",
    "                explored-but-not-committed, capabilities never tried, probes to backfill",
    "                or expose -- each with its command; computed, never applied. :suggest apply",
    "                auto-queues only the safe measurement/calibration ones)",
    "          :tune, :tune <name> <value>, :tune <name> auto [percentile]",
    "          :tune <knob|probe> dynamic <signed mix> [mult]  (each turn set the target",
    "                to mult * a signed mix of live streams, e.g. +ambiguity-consensus; a",
    "                probe-only target drives its own firing threshold; a name that is both",
    "                a knob and a probe steers the KNOB -- name probe_<x> for the threshold)",
    "          :tune steer_cap_fraction auto [pct]  (calibrate cap from observed pushes)",
    "          :tune steer_band auto [min_events] [gold|conversation|any] [synthesis|layersteer]",
    "                (derive band from outcomes; conversations count as evidence)",
    "          :tune steer_layer_sweep 1    (isolate steers by layer: each steer pushes",
    "                ONE least-tested band layer; per-layer outcomes accrue, transfer-free)",
    "          :doc <path> [because <why>]  (share a document into the conversation)",
    "          :doc <path> rewrite [new_name] [because <why>]  (rewrite a text file",
    "                better and save a sibling in the same folder; omitted name uses *_rewritten)",
    "          :doc next | :doc status      (stage the next chunk / show progress)",
    "          :doc read [n] [order|interleave|reply|updated] [satisfied]",
    "                (reading as dialogue: the document speaks each turn, the model",
    "                replies. order/interleave/updated advance on the documents' own",
    "                course -- updated = by file mtime, newest first; reply follows",
    "                overlap. 'satisfied' stops early once sense settles)",
    "          :doc inject                  (stage the whole library for one turn, budget-bounded)",
    "          :doc stop                    (interrupt auto-read)",
    "          :sandbox on|off|status       (run the model's ```python blocks for real)",
    "          :experts on|off|status       (mint new steering experts from its own",
    "                recurring self-corrections; roster bounded; default off)",
    "          :game <name>                 (run a python script from games/ with full access to",
    "                live system state; infinite flexibility for custom rules and interactions)",
    "          :game no | decline           (decline the game the model just proposed)",
    "          :accept [n|all] | :reject [n|all]  (a game may STAGE a command instead of",
    "                running it; nothing a game chose runs until you accept it here)",
    "          :expose :<command> [stage|direct|off]  (make a built-in command or",
    "                macro command callable by the model as <<CMD: :command args>>;",
    "                bare/default = staged for :accept, direct = queued immediately)",
    "          :expose <probe|knob> [off]  (without leading ':', expose a probe sensor",
    "                or tuner knob to the model's <<PROBE: name>> tool)",
    "          :hide <command> [off]       (hide a command from model-facing help,",
    "                suggestions, exposed-command discovery, and <<CMD>> hints; also",
    "                unexposes it. Operator help and execution still work)",
    "          :impact                      (consequence trail: what its words caused,",
    "                and whether experienced impact tracks better deliberation)",
    "          :clock                       (last turn's generation time + tok/s and memory;",
    "                VRAM on GPU, process RAM on CPU-only, both sensed every turn",
    "                as generation_seconds / vram_gb streams)",
    "          :prioritize                  (rank probes by evidence-weighted lift; steer toward",
    "                the top each turn via prioritize_alpha -- signed by lift, off at 0)",
    "          :release <tool> [prob]       (decouple a tool's firing from its signal for that",
    "                fraction of turns -- separates causality so credit lift can be trusted)",
    "          :listen on|off|status        (speak mid-reply: lines you type while it",
    "                generates are ingested at the next chunk seam and appended to the",
    "                live stream -- the model chooses to redirect or fold in; never dropped)",
    "          :macro <file> <c1> ; <c2> ...  (write a macro; :macro restore <json>",
    "                exactly restores one; :macro name <alias> <file> aliases it;",
    "                :macro name self [file] writes a macro that REGENERATES probes,",
    "                macros/commands, hidden/exposed command tools, and game configs;",
    "                :macro strip <alias|file> drops display-only lines in place,",
    "                :macro name strip <src> [dest] writes a stripped copy)",
    "          :save self <name> | choose   (alias for :macro name self; 'choose' asks",
    "                the model to generate a name based on the current tuning state)",
    "          :spawn <name> join|replace|drop  (multi-agent support. 'join' adds to",
    "                the panel, 'replace [N]' takes the operator slot for N turns.",
    "                Use @<name> :cmd to target a specific agent's tuning state)",
    "          :run <alias|file>            (queue and execute a macro's commands)",
    "          :solve <name> <goal> [: args]  (model writes a parameterized macro for an",
    "                ad-hoc command; it is PROPOSED, then :accept adopts it (or :reject",
    "                drops it); after that :<name> <args> runs it, filling $1..$9 / $@.",
    "                Named args fill $name; optional/default specs are name?, [name],",
    "                name=default, or [name=default].",
    "                All non-hidden commands are available; context staged with",
    "                :memory use is folded into the request)",
    "          :<macro-name> <args>         (run any aliased macro directly, args -> $1..$9)",
    "          <any :command> because <reason>   (logs why you issued it as provenance)",
    "          :memory use probe <name> | :memory choice probe <name>",
    "                (stage the memories where probe <name> reads furthest from 0)",
    "          :tune exposed_probe_alpha <small>  (also steer along the probes you have",
    "                exposed to the model, lift-weighted; 0 = off)",
]


def command_help_entries(lines=COMMAND_HELP_LINES):
    """Return help entries as raw line groups from COMMAND_HELP_LINES."""
    entries = []
    current = []
    for raw in lines:
        if raw.startswith("Commands:"):
            if current:
                entries.append(current)
            current = [raw]
            continue
        text = raw.strip()
        if not text:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= 12:
            if current:
                entries.append(current)
            current = [raw]
        elif current:
            current.append(raw)
    if current:
        entries.append(current)
    return entries


def find_command_help_entries(query, lines=COMMAND_HELP_LINES):
    q = (query or "").strip().split()[0].lstrip(":").lower()
    if not q:
        return []
    matches = []
    for entry in command_help_entries(lines):
        head = entry[0].strip()
        if head.startswith("Commands:"):
            head = head[len("Commands:"):].strip()
        words = {m.group(1).lower() for m in re.finditer(r":([a-z_][\w-]*)", head)}
        if q in words:
            matches.append(entry)
    return matches


def render_commands_md(lines=COMMAND_HELP_LINES):
    """Render the in-shell command help into a Markdown reference. A 10-space
    indent (or the leading 'Commands:' line) starts an entry; deeper indents
    continue the previous entry's description. Kept deliberately lossless: the
    exact wording the shell prints becomes the doc, so nothing drifts."""
    out = [
        "# Interactive shell commands",
        "",
        "_Auto-generated from `scripts/interactive_phenomenality.py` (its in-shell help "
        "block). Rewritten every time the shell starts -- edit `COMMAND_HELP_LINES` "
        "there, not this file._",
        "",
    ]
    entries = []  # each: [signature_line, [continuation_lines]]
    for raw in lines:
        if raw.startswith("Commands:"):
            entries.append([raw[len("Commands:"):].strip(), []])
            continue
        text = raw.strip()
        if not text:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if indent <= 12:
            entries.append([text, []])
        elif entries:
            entries[-1][1].append(text)
    for sig_line, cont in entries:
        m = re.search(r"\s{2,}\(", sig_line)
        if m:
            sig = sig_line[:m.start()].strip()
            desc_parts = [sig_line[m.start():].strip()] + cont
        else:
            sig, desc_parts = sig_line, list(cont)
        desc = " ".join(p for p in desc_parts if p).strip()
        if desc.startswith("(") and desc.endswith(")"):
            desc = desc[1:-1].strip()
        desc = desc.replace(") (", "; ")  # merge adjacent parenthetical groups
        out.append(f"- `{sig}` -- {desc}" if desc else f"- `{sig}`")
    out.append("")
    return "\n".join(out)


def write_commands_md(path=None, lines=COMMAND_HELP_LINES):
    """Write the Markdown command reference to docs/COMMANDS.md (best-effort)."""
    if path is None:
        path = os.path.join(ROOT, "docs", "COMMANDS.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_commands_md(lines))
    return path


def list_solve_macros(macros_dir=None):
    """Scan the solve-macro directory and return [(name, desc, args)] for each
    saved macro (invariants/out/macros/*.txt). desc/args come from the leading
    '# :solve macro ...' / '# args:' comments the solve writer leaves."""
    if macros_dir is None:
        macros_dir = os.path.join(ROOT, "invariants", "out", "macros")
    out = []
    if not os.path.isdir(macros_dir):
        return out
    for fn in sorted(os.listdir(macros_dir)):
        if not fn.endswith(".txt"):
            continue
        name, desc, args = fn[:-4], "", ""
        try:
            with open(os.path.join(macros_dir, fn), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# :solve macro") and "--" in line:
                        desc = line.split("--", 1)[1].strip()
                    elif line.startswith("# args:"):
                        args = line.split(":", 1)[1].strip()
                    elif line.startswith(":"):
                        break
        except Exception:
            pass
        out.append((name, desc, args))
    return out


def build_model_help_text(solve_macros=None, exposed_commands=None, exposed_knobs=None, hidden_commands=None):
    """Model-facing help: what the model may run ITSELF (its <<...>> tools) vs.
    the operator's ':' commands, which it can only reach by proposing a game."""
    if solve_macros is None:
        solve_macros = list_solve_macros()
    exposed_commands = exposed_commands or {}
    exposed_knobs = set(exposed_knobs or [])
    hidden_commands = set(hidden_commands or [])
    visible_exposed = {
        name: mode for name, mode in exposed_commands.items()
        if name not in hidden_commands
    }
    visible_operator_commands = sorted(c for c in BUILTIN_COMMANDS if c not in hidden_commands)
    visible_solve_macros = [
        (n, d, a) for n, d, a in solve_macros
        if n not in hidden_commands
    ]
    lines = [
        HELP_TOOL_HEADER,
        "You can reach these YOURSELF -- emit the tag mid-reply; read-only tools return directly, exposed commands follow their mode:",
        "  <<MEMORY: query>>          search long-term memory",
        "  <<METHODMAP: query>>       retrieve sanitized methodology maps",
        "  <<DOC: query>>             read from documents shared into this session",
        "  <<CLAIMMAP: A || B>>       weigh two framings against each other",
        "  <<PROBE: name>>            read one exposed probe sensor or knob",
        "  <<PROBE: name || words>>   score candidate words on an exposed probe sensor",
        "  <<HELP>>                   show this help",
    ]
    if exposed_knobs:
        lines.append("                             Exposed knobs: " + ", ".join(sorted(exposed_knobs)) + ".")
    if visible_exposed:
        exposed_list = ", ".join(
            f":{name} ({mode})" for name, mode in sorted(visible_exposed.items())
        )
        lines.extend([
            "  <<CMD: :command args>>    call an operator-exposed command tool",
            "                             Currently exposed: " + exposed_list + ".",
            "                             Semicolon chains run only exposed commands; :macro keeps semicolons as its body.",
        ])
    else:
        expose_hint = (
            "unavailable until the operator exposes a command"
            if "expose" in hidden_commands
            else "unavailable until the operator exposes a command with :expose"
        )
        lines.append("  <<CMD: ...>>              " + expose_hint)
    lines.extend([
        "",
        "Games are the ONE place you reach commands. You may:",
        "  <<GAME_PROPOSE: name>>     propose a game -- the operator accepts or declines it",
        "  <<GAME_ACCEPT: name>>      accept a game the operator proposed",
        "  <<GAME_DECLINE: name>>     decline a game -- the ONLY thing you can refuse yourself",
        "  <<GAME_END>>               end the active game",
        "  <<GAME_EXPOSE: a,b>> / <<GAME_HIDE: a,b>>   apply probe state inside a game",
        "",
        "The ':' commands are the OPERATOR's. Macros, solves, and games may use/stage any",
        "non-hidden command, but staged game/command-tool actions do not run until the operator",
        "types :accept. Operator commands available here: "
        + ", ".join(visible_operator_commands) + ".",
    ])
    if visible_solve_macros:
        lines.append("Operator solve-macros (also operator-run): "
                     + ", ".join(f":{n}" for n, _d, _a in visible_solve_macros) + ".")
    return "\n".join(lines)


def calibratable_names(tuner):
    """Every name :calibrate accepts: triggers, bare probe names, aliases."""
    cands = set(tuner.triggers)
    cands |= {n[len("probe_"):] for n in tuner.triggers if n.startswith("probe_")}
    cands |= set(STREAM_ALIASES)
    return cands


class InputListener:
    """One reader thread that queues every stdin line, so operator input is
    never dropped -- typed between turns OR mid-generation. Opt-in: until
    start() the shell reads with plain input() (unchanged default path). Once
    active, get() serves each turn from the queue and drain() pulls anything
    typed during a generation, for the interrupt seam."""

    def __init__(self):
        import queue
        self._q = queue.Queue()
        self._eof = object()
        self.active = False

    def start(self):
        import threading
        import sys as _sys
        if self.active:
            return
        def _reader():
            try:
                for line in _sys.stdin:
                    self._q.put(line.rstrip("\r\n"))
            except Exception:
                pass
            self._q.put(self._eof)
        threading.Thread(target=_reader, daemon=True).start()
        self.active = True

    def get(self, prompt=""):
        """Block for the next line (EOFError on stdin close, matching input())."""
        if prompt:
            print(prompt, end="", flush=True)
        item = self._q.get()
        if item is self._eof:
            raise EOFError
        return item

    def drain(self):
        """Every line queued right now, non-blocking. EOF is left in place so a
        later get() still raises it."""
        import queue
        out = []
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is self._eof:
                self._q.put(self._eof)
                break
            out.append(item)
        return out


def load_stored_direction(model, stem, layers=None):
    """Resolve a stored dimension into a per-layer unit direction dict.
    Looks for invariants/<stem>_vector.pt, then invariants/<stem>.pt, then
    the saved-probe dir. Layer dicts are band-restricted and normalized per
    layer; a saved probe payload ({direction, framings}) reuses its
    direction; a bare tensor broadcasts one direction across the band.
    `layers` (explicit indices) overrides the live band -- reading depth
    need not follow steering depth. Always returns a 3-tuple
    (direction, src_name, exposed); ({}, None, False) when nothing usable
    exists."""
    want = set(int(x) for x in layers) if layers is not None else None
    src_path = next(
        (p for p in (
            os.path.join(ROOT, "invariants", f"{stem}_d{model.d_model}_vector.pt"),
            os.path.join(ROOT, "invariants", f"{stem}_d{model.d_model}.pt"),
            os.path.join(ROOT, "invariants", "out", "probes", f"{stem}_d{model.d_model}.pt"),
            os.path.join(ROOT, "invariants", f"{stem}_vector.pt"),
            os.path.join(ROOT, "invariants", f"{stem}.pt"),
            os.path.join(ROOT, "invariants", "out", "probes", f"{stem}.pt"),
        ) if os.path.isfile(p)),
        None,
    )
    if src_path is None:
        return {}, None, False
    try:
        payload = torch.load(src_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(src_path, map_location="cpu")
    from invariants.engine import steer_band_layers
    direction = {}
    exposed = False
    if isinstance(payload, dict) and "direction" in payload:
        exposed = bool(payload.get("exposed", False))
        for L, v in payload["direction"].items():
            if want is not None and int(L) not in want:
                continue
            v = v.to(model.device).float().reshape(-1)
            if v.shape[0] != model.d_model:
                continue
            n = v.norm()
            if n.item() > 0:
                direction[int(L)] = v / n
    elif isinstance(payload, dict):
        raw_vecs = {int(k): v for k, v in payload.items() if hasattr(v, "reshape")}
        if raw_vecs:
            if want is not None:
                keep = sorted(want & set(raw_vecs))
            else:
                band = set(steer_band_layers(max(raw_vecs) + 1))
                keep = sorted(band & set(raw_vecs)) or sorted(raw_vecs)
            for L in keep:
                v = raw_vecs[L].to(model.device).float().reshape(-1)
                if v.shape[0] != model.d_model:
                    continue
                n = v.norm()
                if n.item() > 0:
                    direction[int(L)] = v / n
    elif hasattr(payload, "reshape"):
        v = payload.to(model.device).float().reshape(-1)
        if v.shape[0] == model.d_model:
            n = v.norm()
            if n.item() > 0:
                unit = v / n
                n_layers = int(model.model.config.num_hidden_layers)
                targets = sorted(want) if want is not None else steer_band_layers(n_layers)
                for L in targets:
                    if 0 <= int(L) < n_layers:
                        direction[int(L)] = unit.clone()
    return direction, os.path.basename(src_path), exposed


def parse_compose_expr(expr):
    """Parse a signed weighted mix of named dimensions into (terms, error):
    terms = [(weight, name), ...]. Grammar: terms joined by + or -, each
    optionally weighted 'w*name' (e.g. 'ambiguity + disagreement -
    validated_flow - 0.5*curiosity'). No parentheses -- a mix is a sum."""
    padded = (expr or "").replace("+", " + ").replace("-", " - ")
    pending = 1.0
    terms = []
    for tok in padded.split():
        if tok == "+":
            continue
        if tok == "-":
            pending = -pending
            continue
        weight = 1.0
        tname = tok
        if "*" in tok:
            wpart, _, tname = tok.partition("*")
            try:
                weight = float(wpart)
            except ValueError:
                return [], tok
        tname = re.sub(r"[^a-z0-9_]", "_", tname.lower())[:40].strip("_")
        if not tname:
            return [], tok
        terms.append((pending * weight, tname))
        pending = 1.0
    if not terms:
        return [], expr or "(empty)"
    return terms, None


OPT_IN_CAPABILITY = {"claimmap_alpha", "memory_alpha", "expert_proof_weight", "steer_layer_sweep"}
# Categories :suggest apply may auto-run: pure measurements/calibrations that
# only move a bar. explore/expose CHANGE behavior or surface state to the model,
# so they stay a deliberate hand.
SUGGEST_APPLY_SAFE = {"calibrate", "commit", "backfill"}


def _explore_value(name, current):
    base = {"claimmap_alpha": 0.02, "memory_alpha": 0.02,
            "expert_proof_weight": 1.0, "steer_layer_sweep": 1}.get(name, 0.02)
    # If already sitting at the natural first value, propose a second one so a
    # verdict can be earned (outcome calibration needs >=2 distinct values).
    if abs(float(current) - base) < 1e-9:
        return round(base * 2, 4)
    return base


def suggest_actions(tuner, rows, probes=None, archive_size=0, max_paired=6):
    """Scan the accrued state for READY next moves -- not only calibrations.
    Returns [(category, evidence_line, command), ...] where category is one of
    'calibrate' | 'commit' | 'explore' | 'backfill' | 'expose'. Deterministic;
    acting on any of them stays an operator decision."""
    probes = probes or {}
    out = []

    # CALIBRATE -- a threshold mis-set on its own distribution (never/always fires).
    for name in sorted(tuner.triggers):
        t = tuner.triggers[name]
        if t.kind != "threshold" or len(t.signals) < 10 or not t.observed:
            continue
        fr = t.fired / t.observed
        if fr <= 0.05 or fr >= 0.95:
            p50 = sorted(t.signals)[len(t.signals) // 2]
            out.append((
                "calibrate",
                f"{name}: fires {fr:.0%} of {t.observed} observed (bar {round(t.value, 4)}, own median {round(p50, 4)})",
                f":calibrate {name} 50",
            ))

    # COMMIT -- an explored circular knob whose best value beats current (the
    # 'you did the exploration but never locked it in' case).
    # EXPLORE -- an opt-in capability with fewer than 2 tried values: you can't
    # earn a verdict until you've tried a second.
    for name in OUTCOME_CALIBRATABLE:
        t = tuner.triggers.get(name)
        if t is None:
            continue
        tried = {round(float(s), 6) for s, _ in t.outcomes}
        v = outcome_calibration(list(t.outcomes))
        if v is not None and abs(v - t.value) > 1e-9:
            out.append((
                "commit",
                f"{name}: best explored value {round(v, 4)} beats current {round(t.value, 4)} on {len(t.outcomes)} outcomes",
                f":calibrate {name} outcome",
            ))
        elif name in OPT_IN_CAPABILITY and len(tried) < 2:
            n = len(tried)
            out.append((
                "explore",
                f"{name}: only {n} value(s) ever tried -- explore a second so its effect can be earned",
                f":tune {name} {_explore_value(name, t.value)}",
            ))

    # BACKFILL -- an active probe with little evidence while the archive is deep.
    if archive_size >= 40:
        for pname in sorted(probes):
            trig = tuner.triggers.get(f"probe_{pname}")
            n_pairs = len(trig.outcomes) if trig else 0
            if n_pairs < 15:
                out.append((
                    "backfill",
                    f"{pname}: {n_pairs} paired turns vs {archive_size} archived replies -- pre-load evidence",
                    f":probe backfill {pname}",
                ))

    # EXPOSE -- a probe with a strong, evidenced sense-lift you haven't let the
    # model read yet.
    for pname in sorted(probes):
        if probes[pname].get("exposed"):
            continue
        trig = tuner.triggers.get(f"probe_{pname}")
        if trig is None:
            continue
        st = trig.outcome_stats()
        lift = st.get("lift")
        if lift is not None and abs(lift) >= 0.05 and st.get("n_credited", 0) >= 20:
            out.append((
                "expose",
                f"{pname}: sense-lift {lift} over {st['n_credited']} turns -- a real signal you could let it read",
                f":probe expose {pname}",
            ))

    # CALIBRATE (paired) -- every target<-anchor cut the joint table supports,
    # ranked by median separation. Command uses exact names so it round-trips.
    numeric = sorted({
        k for r in rows for k, val in r.items()
        if isinstance(val, (int, float)) and k != "ts"
    })
    paired = []
    for tstream in numeric:
        t_name = "conversation_productive" if tstream == "sense" else tstream
        t_trig = tuner.triggers.get(t_name)
        if t_trig is None or t_trig.kind != "threshold":
            continue
        for astream in numeric:
            if astream == tstream:
                continue
            a_trig = anchor_trigger_for(astream, tuner)
            if a_trig is None or a_trig.kind != "threshold":
                continue
            cut, n_f, n_uf, gap = paired_threshold_multi(
                rows, tstream, [(astream, a_trig.value, a_trig.comparator)], details=True,
            )
            if cut is None or not gap or abs(cut - t_trig.value) <= 1e-6:
                continue
            t_disp = t_name[6:] if t_name.startswith("probe_") else t_name
            a_disp = astream[6:] if astream.startswith("probe_") else astream
            paired.append((
                gap,
                f"{t_disp} <- {a_disp}: cut {round(cut, 4)} (now {round(t_trig.value, 4)}; {n_f} fired / {n_uf} unfired, gap {round(gap, 4)})",
                f":calibrate {t_name} {astream}",
            ))
    paired.sort(key=lambda x: -x[0])
    out.extend(("calibrate", line, cmd) for _, line, cmd in paired[:max_paired])
    return out


def strip_band_suffix(text):
    """Pop a trailing 'band <lo> <hi>' (inclusive layer indices) off a probe
    argument string: the explicit reading depth for adopt/compose/mint.
    Returns (text_without_suffix, layers or None)."""
    m = re.search(r"\s+band\s+(\d+)\s+(\d+)\s*$", text or "", re.IGNORECASE)
    if not m:
        return (text or "").strip(), None
    lo, hi = int(m.group(1)), int(m.group(2))
    return text[:m.start()].strip(), list(range(min(lo, hi), max(lo, hi) + 1))


def intent_relative_threshold(pairs, min_n=5):
    """Set the productive bar RELATIVE TO INTENT-SHAPING, not at an arbitrary
    quantile: given (settling, sense) pairs -- settling > 0 means the turn
    LOWERED ambiguity+disagreement versus the previous turn, i.e. it shaped
    intent -- return the sense midpoint between the median of intent-shaping
    turns and the median of non-shaping turns. That cut labels 'productive'
    as 'coheres like the turns that actually settled an intent'. None until
    both groups have min_n evidence. Deterministic: same pairs, same bar."""
    settling = sorted(o for s, o in pairs if s > 0)
    non = sorted(o for s, o in pairs if s <= 0)
    if len(settling) < min_n or len(non) < min_n:
        return None
    return (settling[len(settling) // 2] + non[len(non) // 2]) / 2.0


def _sync_steer_tunables(tuner, config):
    """Push the live tuner values into the engine's steering envelope and the
    agentic config, once per turn before any generation. Every steering surface
    is tunable (:tune steer_cap_fraction / steer_band_lo / steer_band_hi /
    steer_fraction) but a bad value never lands: the engine setters reject
    non-finite input and the last good value stays in force."""
    from invariants import engine as _engine
    try:
        _engine.set_steer_cap_fraction(
            tuner.get("steer_cap_fraction", _engine.get_steer_cap_fraction())
        )
    except (TypeError, ValueError) as exc:
        print(Fore.YELLOW + f"[Steer] steer_cap_fraction ignored: {exc}" + Style.RESET_ALL)
    lo_now, hi_now = _engine.get_steer_band()
    try:
        _engine.set_steer_band(
            tuner.get("steer_band_lo", lo_now),
            tuner.get("steer_band_hi", hi_now),
        )
    except (TypeError, ValueError) as exc:
        print(Fore.YELLOW + f"[Steer] steer band ignored: {exc}" + Style.RESET_ALL)
    from invariants.agentic_engine import _sane_fraction
    config.steer_fraction = _sane_fraction(
        tuner.get("steer_fraction", config.steer_fraction), config.steer_fraction
    )
    # Expert-consultation budgets follow the tuner too (ints >= 0; bad values
    # fall back to the current config rather than landing).
    config.max_routing_events = max(
        0, int(_sane_fraction(tuner.get("routing_events", config.max_routing_events), config.max_routing_events))
    )
    config.max_loops = max(
        0, int(_sane_fraction(tuner.get("routing_loops", config.max_loops), config.max_loops))
    )
    config.entropy_threshold = _sane_fraction(
        tuner.get("routing_entropy", config.entropy_threshold), config.entropy_threshold
    )
    config.max_synthesis_events = max(
        0, int(_sane_fraction(tuner.get("synthesis_events", config.max_synthesis_events), config.max_synthesis_events))
    )
    # The roster cap for emergent mesa-objectives is live-tunable.
    if not hasattr(config, "max_committee_size"):
        config.max_committee_size = 6
    config.max_committee_size = max(
        0, int(_sane_fraction(tuner.get("max_committee_size", config.max_committee_size), config.max_committee_size))
    )
    config.max_synthesis_steps = max(
        0, int(_sane_fraction(tuner.get("synthesis_steps", config.max_synthesis_steps), config.max_synthesis_steps))
    )
    config.epsilon = _sane_fraction(tuner.get("plateau_epsilon", config.epsilon), config.epsilon)
    
    config.max_rounds = max(1, int(_sane_fraction(tuner.get("max_rounds", getattr(config, "max_rounds", 5)), getattr(config, "max_rounds", 5))))
    config.required_agreement = max(1, int(_sane_fraction(tuner.get("required_agreement", getattr(config, "required_agreement", 3)), getattr(config, "required_agreement", 3))))
    config.max_tool_calls = max(0, int(_sane_fraction(tuner.get("max_tool_calls", getattr(config, "max_tool_calls", 8)), getattr(config, "max_tool_calls", 8))))
    config.max_new_tokens = max(1, int(_sane_fraction(tuner.get("max_new_tokens", getattr(config, "max_new_tokens", 220)), getattr(config, "max_new_tokens", 220))))
    config.repair_token_multiplier = _sane_fraction(tuner.get("repair_token_multiplier", getattr(config, "repair_token_multiplier", 2.0)), getattr(config, "repair_token_multiplier", 2.0))
    if tuner.get("max_elapsed_sec", -1.0) > 0:
        config.max_elapsed_sec = float(tuner.get("max_elapsed_sec", getattr(config, "max_elapsed_sec", 60.0)))
    config.oracle_max_elapsed_sec = float(_sane_fraction(tuner.get("oracle_max_elapsed_sec", getattr(config, "oracle_max_elapsed_sec", 60.0)), getattr(config, "oracle_max_elapsed_sec", 60.0)))
    config.verifier_time_reserve_sec = float(_sane_fraction(tuner.get("verifier_time_reserve_sec", getattr(config, "verifier_time_reserve_sec", 20.0)), getattr(config, "verifier_time_reserve_sec", 20.0)))
    config.relax_agreement_under_urgency = bool(tuner.get("relax_agreement_under_urgency", float(getattr(config, "relax_agreement_under_urgency", False))) > 0.0)
    config.stop_on_critical_urgency = bool(tuner.get("stop_on_critical_urgency", float(getattr(config, "stop_on_critical_urgency", True))) > 0.0)
    # The synthesis budget is PER REPLY, but its counter lives on the shared
    # config object -- the benchmark resets it per attempt; the shell must
    # reset it per turn or synthesis (and the cache-delta retrieval inside
    # the same gate) fires once per SESSION and silently never again.
    config._synthesis_events_used = 0


def get_hardware_appropriate_model():
    if not torch.cuda.is_available():
        return "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return DEFAULT_MODEL
    
    if vram >= 14.5:
        return DEFAULT_MODEL
    elif vram >= 8.0:
        return "Qwen/Qwen2.5-3B-Instruct"
    else:
        return "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    os.chdir(ROOT)
    print(Fore.CYAN + Style.BRIGHT + "================================================")
    print("      HUMBLE SYNTHESIS - INTERACTIVE SHELL      ")
    print("================================================" + Style.RESET_ALL)
    
    # Model is configurable so the egg shell honors whatever model earned the egg
    # (env EGG_MODEL set by the benchmark, or argv[1] for a manual launch).
    model_name = os.environ.get("EGG_MODEL")
    if not model_name:
        if len(sys.argv) > 1:
            model_name = sys.argv[1]
        else:
            model_name = get_hardware_appropriate_model()
            
    is_default = (model_name == DEFAULT_MODEL)

    print(Fore.YELLOW + f"[System] Loading {model_name}..." + Style.RESET_ALL)
    model = load_model(model_name, local_files_only=is_default)

    config = AgenticConfig()
    organic_path = os.path.join(ROOT, "invariants", f"organic_correction_vector_d{model.d_model}.pt")
    if not os.path.isfile(organic_path):
        organic_path = ORGANIC_VECTOR_PATH

    try:
        payload = torch.load(organic_path, map_location=model.device)
        if isinstance(payload, torch.Tensor) and payload.reshape(-1).shape[0] != model.d_model:
            print(Fore.YELLOW + f"[System] Skipping organic vector: dimension mismatch (model d_model={model.d_model})." + Style.RESET_ALL)
        else:
            config.organic_correction_vector = payload
            print(Fore.GREEN + f"[System] Successfully loaded {os.path.basename(organic_path)}!" + Style.RESET_ALL)
    except Exception as e:
        if is_default:
            print(Fore.RED + f"[System] Warning: Could not load organic vector: {e}" + Style.RESET_ALL)
        else:
            pass # Non-default models might simply not have an organic vector minted yet

    cache_file = CACHE_FILE
    try:
        cache_file = _global_cache.use_file(model_cache_file(model_name, model.d_model))
        print(Fore.GREEN + f"[System] Loaded cache {cache_file.name} ({len(_global_cache.memory)} memories)." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[System] Warning: Could not load cognitive cache: {e}" + Style.RESET_ALL)

    # Enable cache read/write as requested
    config.cache_enabled = True
    config.cache_write_enabled = True
    config.cache_write_scope = "interactive_phenomenality"
    
    # Disambiguation resolves INTERNALLY, never by halting to interrogate the
    # operator. Reading is non-adapting reality: when a chunk is ambiguous the
    # model reconciles its own thoughts with what the text says, it does not
    # ask a human what the document meant. The old interactive path popped a
    # display-only "SYSTEM HALTED" modal (its real input was a hidden terminal
    # prompt) and injected a hardcoded GSM8K "you are a reasoning engine"
    # persona -- both contrary to the bare-mode, first-person spine. Off here.
    config.interactive_disambiguation = False

    memory = MemoryEngine(scope="interactive_phenomenality")
    self_concept = SelfConceptController()
    steer_map = SteerMapStore()
    tuner = TriggerTuner()
    # Every trigger is born tunable. Persisted values win over these defaults.
    tuner.register("claimmap_tension", 0.18, kind="threshold", comparator=">=")
    # Steering is OFF by default (0). The raw A-B delta is large; any nonzero
    # alpha must be tuned UP from ~0 while watching -- alpha=0.5 collapses the
    # model into a repetition loop. Enable deliberately: :tune claimmap_alpha 0.02
    tuner.register("claimmap_alpha", 0.0, kind="coefficient")
    # Memory is state-triggered too: it fires from the model's own "memory gap"
    # signature (see _memory_detect), not a <<MEMORY>> text tag. Born tunable; the
    # steer that injects the retrieval is OFF by default (opt-in, capped).
    tuner.register("memory_need", 0.05, kind="threshold", comparator=">=")
    tuner.register("memory_alpha", 0.0, kind="coefficient")
    # Clock sensor: this turn's generation wall-time and memory footprint,
    # observed every turn like any other stream (so they can be anchored:
    # "did slow / memory-heavy turns cohere worse?"). vram_gb carries VRAM on a
    # CUDA box and process RSS on CPU-only, so CPU runs keep the same stream.
    # Threshold-kind with a 0 bar until you calibrate one; observation-only,
    # never a knob you set.
    tuner.register("generation_seconds", 0.0, kind="threshold", comparator=">=")
    tuner.register("vram_gb", 0.0, kind="threshold", comparator=">=")
    # The steering envelope itself is a live surface: every additive push is
    # clipped to steer_cap_fraction of the residual it lands in, band-style
    # steers (claimmap/memory) default to the [steer_band_lo, steer_band_hi)
    # depth window, and agentic deltas scale by steer_fraction. All movable
    # mid-session; none removable (non-finite values are rejected at sync).
    from invariants.engine import get_steer_cap_fraction, get_steer_band
    _band_lo, _band_hi = get_steer_band()
    tuner.register("steer_cap_fraction", get_steer_cap_fraction(), kind="coefficient")
    tuner.register("steer_band_lo", _band_lo, kind="coefficient")
    tuner.register("steer_band_hi", _band_hi, kind="coefficient")
    tuner.register("steer_fraction", config.steer_fraction, kind="coefficient")
    # Conversational outcome threshold: a turn whose sense_score (coherence
    # minus interruption/disagreement) clears this labels its steering events
    # "conversation_productive" in the steer map. Default 0.0 is the composite's
    # natural sign, not a magic number; observed every turn, so it can be
    # calibrated to the real distribution (:tune conversation_productive auto 50).
    tuner.register("conversation_productive", 0.0, kind="threshold", comparator=">=")
    # Sandbox executions are objective outcomes (exit 0, no timeout = 1.0);
    # observed so the distribution is inspectable like every other signal.
    tuner.register("sandbox_success", 0.5, kind="threshold", comparator=">=")
    # Agency measurement: signal 1.0 on turns whose context was really caused
    # by the model's own words (code ran, a tool it invoked returned, the
    # reading followed its reply). Credited with the turn's sense, so :tune
    # lift answers: does experiencing impact track better deliberation?
    tuner.register("words_had_impact", 0.5, kind="threshold", comparator=">=")
    # Consequence trail: {"cause", "effect"} events. "pending" = caused by the
    # reply just produced, consumed as the NEXT turn's impact context.
    impact_state = {"pending": [], "log": deque(maxlen=40)}
    # Layer isolation (:tune steer_layer_sweep 1): every steer pushes exactly
    # ONE layer — the least-tested in the band — and the turn's outcome lands
    # on that layer in the steer map. Band-wide steering assumes one semantic
    # axis runs through the depth; the sweep measures instead of assuming.
    # "fires" collects (channel, layer, alpha, drift_at_layer) per turn.
    tuner.register("steer_layer_sweep", 0.0, kind="coefficient")
    sweep_state = {"fires": []}
    # Reply token budget: the first live run cut off mid-sentence at the old
    # hardcoded 512. Live-tunable like everything else (:tune response_tokens 800).
    tuner.register("response_tokens", 512, kind="coefficient")
    # ':doc read ... satisfied' stops once this many consecutive reading turns
    # clear the conversation_productive bar -- "satisfied" is the existing
    # sense signal, not a new oracle. Streak length is a knob like everything.
    tuner.register("reading_settled_streak", 2, kind="coefficient")
    # Expert consultation surface: routing fires on entropy > routing_entropy,
    # at most routing_loops times per token and routing_events times per
    # reply. Budgets, not truths -- live-tunable like every other number here.
    tuner.register("routing_events", config.max_routing_events, kind="coefficient")
    tuner.register("routing_loops", config.max_loops, kind="coefficient")
    tuner.register("routing_entropy", config.entropy_threshold, kind="coefficient")
    # Proven-expert routing: how hard the ToT winner is nudged toward experts
    # with a good accrued route success rate. 0 = pure entropy (default);
    # earn it from outcomes (:calibrate expert_proof_weight outcome).
    tuner.register("expert_proof_weight", 0.0, kind="coefficient")
    # Calibration step size: a calibration AUGMENTS rather than overwrites --
    # the first one adopts the computed value, later ones move this fraction of
    # the way toward it (EMA smoothing). 1.0 = overwrite; default 0.5.
    tuner.register("calibration_gain", 0.5, kind="coefficient")
    # Prioritize: :prioritize ranks probes by evidence-weighted lift; when this
    # alpha > 0 each reply is steered toward the top-ranked probe (signed by its
    # lift -- toward a helpful concept, away from a harmful one). 0 = off.
    tuner.register("prioritize_alpha", 0.0, kind="coefficient")
    # Exposed-probe steer: when > 0, each reply is ALSO nudged along the probes
    # the operator has exposed to the model (:probe expose <name>) -- lift-weighted
    # and signed just like prioritize (a helpful exposed sensor pulls toward its
    # concept, a harmful one pushes away). 0 = off; idle until those probes accrue
    # credited lift. :tune exposed_probe_alpha <small>, or :calibrate from outcomes.
    tuner.register("exposed_probe_alpha", 0.0, kind="coefficient")
    # Synthesis schedule, same shape: events per reply, optimizer steps per
    # event, and the plateau-velocity trigger that starts one.
    tuner.register("synthesis_events", config.max_synthesis_events, kind="coefficient")
    tuner.register("synthesis_steps", config.max_synthesis_steps, kind="coefficient")
    tuner.register("plateau_epsilon", config.epsilon, kind="coefficient")
    tuner.register("max_committee_size", 6, kind="coefficient")
    
    tuner.register("max_rounds", getattr(config, "max_rounds", 5), kind="coefficient")
    tuner.register("required_agreement", getattr(config, "required_agreement", 3), kind="coefficient")
    tuner.register("max_tool_calls", getattr(config, "max_tool_calls", 8), kind="coefficient")
    tuner.register("max_new_tokens", getattr(config, "max_new_tokens", 220), kind="coefficient")
    tuner.register("repair_token_multiplier", getattr(config, "repair_token_multiplier", 2.0), kind="coefficient")
    tuner.register("max_elapsed_sec", -1.0, kind="coefficient")
    tuner.register("oracle_max_elapsed_sec", getattr(config, "oracle_max_elapsed_sec", 60.0), kind="coefficient")
    tuner.register("verifier_time_reserve_sec", getattr(config, "verifier_time_reserve_sec", 20.0), kind="coefficient")
    tuner.register("relax_agreement_under_urgency", 1.0 if getattr(config, "relax_agreement_under_urgency", False) else 0.0, kind="coefficient")
    tuner.register("stop_on_critical_urgency", 1.0 if getattr(config, "stop_on_critical_urgency", True) else 0.0, kind="coefficient")
    # Intent axis of the decomposed reward: signal = previous turn's
    # (ambiguity + disagreement) minus this turn's. Fired (> 0) = this turn
    # SETTLED intent. Credited with sense, so conversation_productive can be
    # calibrated relative to intent-shaping (:tune conversation_productive
    # auto intent) instead of an arbitrary quantile.
    tuner.register("intent_settling", 0.0, kind="threshold", comparator=">=")
    # EOT Urgency: Tracks the model's internal probability of finishing its turn.
    tuner.register("eot_urgency", 0.05, kind="threshold", comparator="<=")

    def _sweep_layers_for(channel, steer_vecs=None):
        """Pick the steer_layer_sweep least-tested band layers (width 1 =
        isolation; width k = deterministic overlay), plus per-layer local axis
        drift of the vector dict being applied (is the direction even the
        same thing next door?). Empty list = sweep off (full band)."""
        from invariants.engine import axis_drift, drift_at_layer, pick_sweep_layers, steer_band_layers
        width = max(0, int(tuner.get("steer_layer_sweep", 0)))
        if width <= 0:
            return [], {}, None
        n_layers = getattr(model, "n_layers", None) or len(model.model.model.layers)
        layers = pick_sweep_layers(steer_band_layers(n_layers), steer_map.layer_steer_counts(channel), width)
        drift = axis_drift(steer_vecs) if steer_vecs else None
        drifts = {L: drift_at_layer(drift, L) for L in layers} if drift else {L: None for L in layers}
        return layers, drifts, (drift.get("var") if drift else None)

    _sync_steer_tunables(tuner, config)

    # Tool-sensing seam: any tool fires mid-thought when its state trigger crosses.
    # Detectors read (text, state) -- state carries the live phenomenality scores,
    # so a tool triggers from activation, not a tag. add a tool = register a detector.
    tool_sense = ToolSense(model, tuner)

    # Async operator input: a reader thread queues every line so nothing is
    # dropped; when listen-drain is on, lines typed mid-generation are pulled at
    # the chunk seam and appended to the live stream -- the model reads the
    # interjection and redirects or folds in on its own. Opt-in via :listen.
    listener = InputListener()
    listen_state = {"drain": False}

    def _operator_input_drain(text, state=None):
        if not listen_state["drain"]:
            return ""
        lines = listener.drain()
        # A :command or exit typed mid-reply is meant for the shell, not the
        # model -- hold it for the next prompt instead of injecting it as speech.
        speech_parts = []
        for l in lines:
            s = (l or "").strip()
            if not s:
                continue
            if s.startswith(":") or s.lower() in ("exit", "quit"):
                listener._q.put(l)
            else:
                speech_parts.append(s)
        speech = " ".join(speech_parts)
        if not speech:
            return ""
        memory.append_event(
            "operator_interjection",
            text=speech[:400],
            tags=["async_input", "interrupt"],
            provenance={"drained_mid_generation": True, "chars": len(speech)},
        )
        impact_state["pending"].append(
            {"cause": "was spoken to mid-thought", "effect": f'interjection ingested: "{speech[:80]}"'}
        )
        print(
            Fore.MAGENTA + Style.BRIGHT
            + f"\n[Listen] interjected mid-thought, ingested -> {speech[:120]}"
            + Style.RESET_ALL,
            flush=True,
        )
        # A legible, operator-attributed interjection -- not a taught tag. It
        # becomes the next tokens; whether the model redirects is emergent.
        return f"\n\n[Operator interjects: {speech}]\n\n"

    tool_sense.input_drain = _operator_input_drain

    def _claimmap_detect(text, state=None):
        a, b, score = framing_tension_score(text)
        return score, ((a, b) if a is not None else None)

    def _memory_detect(text, state=None):
        # Memory-gap footprint (STATE, not text): the model is unsettled (ambiguity
        # / disagreement up) and not flowing (validated_flow / warranted confidence
        # down) -- an unresolved reference it may need to retrieve. Complex,
        # multi-signal, read from the model's OWN phenomenality, never a tag.
        phen = (state or {}).get("last_phenomenality") or {}
        if not phen:
            return 0.0, None
        gap = (
            float(phen.get("ambiguity", 0.0))
            + float(phen.get("disagreement", 0.0))
            - float(phen.get("validated_flow", 0.0))
            - float(phen.get("warranted_confidence_legacy", 0.0))
        )
        tail = " ".join((text or "").split()[-24:]).strip()
        return gap, (tail or None)

    def _memory_act(query, m):
        records = memory.search(query, max_records=3, scope=memory.scope)
        if not records:
            return []
        memory.append_event(
            "memory_state_triggered",
            text=query[:240],
            tags=["memory_tool", "activation_trigger"],
            provenance={"records": len(records), "trigger": "memory_need"},
        )
        print(
            Fore.MAGENTA
            + f"\n[Memory] gap sensed mid-thought (not a tag) -> retrieved {len(records)} record(s)."
            + Style.RESET_ALL,
            flush=True,
        )
        alpha = tuner.get("memory_alpha", 0.0)
        sweep_layers = None
        if alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
            _mls, _, _mvar = _sweep_layers_for("memory")
            if _mls:
                sweep_layers = _mls
                for _sl in _mls:
                    sweep_state["fires"].append(("memory", _sl, alpha, None, len(_mls), _mvar))
        return _memory_steer_handles(m, (records[0].text or "").strip(),
                                     alpha=alpha, layers=sweep_layers)

    def _claimmap_act(payload, m):
        a, b = payload
        cm = analyze_claim_pair(f"{a} || {b}", model=m)
        alpha = tuner.get("claimmap_alpha", 0.0)
        sweep_layers = None
        if alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
            _cls, _cdrifts, _cvar = _sweep_layers_for("claimmap", cm.steer_delta)
            if _cls:
                sweep_layers = _cls
                for _cl in _cls:
                    sweep_state["fires"].append(("claimmap", _cl, alpha, _cdrifts.get(_cl), len(_cls), _cvar))
        handles = claimmap_steer_handles(m, cm.steer_delta, alpha=alpha, layers=sweep_layers)
        msg = (
            "steering the rest of this answer."
            if handles
            else "steering off (:tune claimmap_alpha to enable)."
        )
        print(Fore.MAGENTA + f"\n[ClaimMap] mid-thought: sensed a tension, {msg}" + Style.RESET_ALL, flush=True)
        return handles

    tool_sense.register(Tool("claimmap_tension", _claimmap_detect, _claimmap_act))
    tool_sense.register(Tool("memory_need", _memory_detect, _memory_act, comparator=">="))
    imported_methodologies = memory.import_methodologies(
        _global_cache.memory,
        source="cognitive_cache",
        source_path=str(cache_file),
    )
    memory.append_event(
        "shell_start",
        tags=["session"],
        provenance={
            "script": os.path.abspath(__file__),
            "cache_write_scope": config.cache_write_scope,
            "memory_policy": "tool_not_prompt",
            "methodologies_imported": imported_methodologies,
            "self_concept_controller": "vector_map_based",
            "steer_map_store": str(steer_map.events_path),
        },
    )
        
    print(Fore.CYAN + "\nThis terminal uses full Agentic ToT and Test-Time Layer Synthesis.")
    print("Watch the model's internal entropy and phenomenality trace in real time!")
    print("Current session context is ON. Long-term memory is a tool, not hidden prompt context.")
    print("Self-concept orientation is vector-map based and logged as a tool/controller trace.")
    print(f"Steer-map traces are stored at {steer_map.events_path}.")
    print(f"Imported {imported_methodologies} sanitized methodology memories from cognitive cache.")
    for _hline in COMMAND_HELP_LINES:
        print(_hline)
    print("Type 'exit' or 'quit' to leave.\n" + Style.RESET_ALL)
    try:
        _md_path = write_commands_md()
        print(Fore.CYAN + f"(command reference refreshed at {os.path.relpath(_md_path, ROOT)})" + Style.RESET_ALL)
    except Exception as _md_err:
        print(Fore.YELLOW + f"[Docs] Could not refresh COMMANDS.md: {_md_err}" + Style.RESET_ALL)

    pending_memory_tool_result = None
    pending_orientation_tool_result = None
    pending_claimmap_tool_result = None
    pending_claimmap_steer_delta = None
    pending_claimmap_credit = None  # last turn's tension, credited by this turn's sense
    pending_methodmap_tool_result = None
    pending_document_tool_result = None
    pending_sandbox_tool_result = None
    pending_game_tool_result = None
    model_proposed_game = None
    user_proposed_game = None
    user_proposed_game_args = None
    active_game = None
    active_game_state = {}
    # Commands a game CHOOSES to run don't execute on their own: they stage here
    # and wait for an explicit operator :accept (or :reject). Tools stay free;
    # this gate is only for real :commands a game would adopt into the queue.
    pending_accept_commands = []
    pending_solve_proposal = None   # a :solve choose/auto macro awaiting :accept
    pending_help_tool_result = None # model asked for <<HELP>>; served next turn
    help_exposed = False            # when on, the model is told it may call <<HELP>>

    def _stage_for_accept(cmd, why="a game"):
        pending_accept_commands.append(cmd)
        idx = len(pending_accept_commands)
        print(Fore.YELLOW + Style.BRIGHT
              + f"[Accept] {why} wants to run a command -- staged #{idx}, NOT run: {cmd}"
              + Style.RESET_ALL)
        print(Fore.YELLOW
              + "         Type :accept to run it (or :accept all), :reject to drop it (or :reject all)."
              + Style.RESET_ALL)

    def _run_game_ref(name, args=None):
        """Injected into game scripts as run_game(name, args): load and run
        another game by name, so games can reference/compose each other. Shares
        the live active_game_state and probes with the caller."""
        gp = os.path.join(ROOT, "games", f"{name}.py")
        if not os.path.isfile(gp):
            print(Fore.YELLOW + f"[Game] referenced game '{name}' not found in games/." + Style.RESET_ALL)
            return False
        sub_cfg = load_game_config(os.path.join(ROOT, "games", f"{name}_rules.json"))
        with open(gp, "r", encoding="utf-8") as _gf:
            code = _gf.read()
        g = globals().copy()
        g["GAME_RULES"] = sub_cfg["rules"]
        g["GAME_CONFIG"] = sub_cfg
        g["game_args"] = list(args or [])
        g["active_game_state"] = active_game_state
        g["probes"] = probes
        g["run_game"] = _run_game_ref
        exec(code, g)
        return True

    doc_session = None          # current document: chunks + cursor + why
    doc_library = []            # every ingested document this session (for inter-ordering)
    doc_autoread = None         # {"remaining": n, "mode": "order"|"interleave"|"reply"} during :doc read
    last_assistant_response = ""
    recent_responses = deque(maxlen=4)  # reflection stream for reply-mode thread returns
    prev_unsettledness = None  # ambiguity+disagreement of the previous turn (intent axis)
    probes = {}  # name -> {direction, history, framings}; minted concept sensors (unvalidated until outcomes accrue)
    PROBE_DIR = os.path.join(ROOT, "invariants", "out", "probes")
    try:
        if os.path.isdir(PROBE_DIR):
            for pf in os.listdir(PROBE_DIR):
                if pf.endswith(".pt"):
                    pname = pf[:-3]
                    pdata = torch.load(os.path.join(PROBE_DIR, pf), weights_only=True)
                    probes[pname] = {"direction": pdata["direction"], "history": deque(maxlen=40), "framings": pdata.get("framings", ("", "")), "exposed": bool(pdata.get("exposed", False))}
                    tuner.register(f"probe_{pname}", 0.0, kind="threshold", comparator=">=")
    except Exception:
        pass

    # Purge stray shadow triggers: never-fed bare thresholds that duplicate a
    # probe (an accidental `:tune <probe-name> <value>` registers one, and it
    # then shadows the real probe in every direct trigger lookup). The probe is
    # the real one; removing the shadow fixes all code paths at once.
    _shadows = [n for n in list(tuner.triggers) if _is_shadow_trigger(n, tuner)]
    if _shadows:
        for _sh in _shadows:
            del tuner.triggers[_sh]
        tuner.save()
        print(Fore.CYAN + f"[System] Cleared {len(_shadows)} stray shadow trigger(s): {', '.join(_shadows)}." + Style.RESET_ALL)

    queued_calibrations = []
    last_refused_calibration = None
    
    # Per-turn signals table: one row per turn with every stream's value --
    # the substrate that lets ANY named stream anchor ANY threshold target
    # (both ways). Persisted, so paired evidence survives restarts.
    TURN_SIGNALS_PATH = os.path.join(ROOT, "invariants", "out", "turn_signals.jsonl")
    turn_log = deque(maxlen=400)
    try:
        with open(TURN_SIGNALS_PATH, "r", encoding="utf-8") as _tf:
            for _line in _tf.readlines()[-400:]:
                try:
                    turn_log.append(json.loads(_line))
                except Exception:
                    continue
    except OSError:
        pass
    # Macro name -> file aliases, so :run <name> and :macro strip <name> address a
    # macro by a short name instead of a path. Persisted across sessions.
    MACRO_ALIAS_PATH = os.path.join(ROOT, "invariants", "out", "macro_aliases.json")
    macro_aliases = {}
    try:
        with open(MACRO_ALIAS_PATH, "r", encoding="utf-8") as _af:
            macro_aliases = {str(k): str(v) for k, v in json.load(_af).items()}
    except (OSError, ValueError):
        macro_aliases = {}

    def _save_macro_aliases():
        try:
            with open(MACRO_ALIAS_PATH, "w", encoding="utf-8") as af:
                json.dump(macro_aliases, af, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save macro aliases: {e}" + Style.RESET_ALL)
    MAX_AUTOREAD = 20           # per :doc read command; reading stays a deliberate act
    sandbox_enabled = False     # deliberate opt-in, like every intervention here
    joined_agents = []
    replace_agent = None
    replace_turns_remaining = 0
    session_context = []
    session_context_enabled = True
    show_timestamps = False
    last_clock = None
    # prioritize/steer target: None = auto (follow the ranking). "probe" pins
    # the steer to one probe; "mix" is a list of probes whose DEGREES are their
    # own learned lifts (you pick the probes, the data sets the weights).
    prioritize_pin = {"probe": None, "mix": None}
    tuner_bindings = {}  # knob -> (probe, multiplier)
    # probe<->knob matches: probe name -> {"knob", "mode": none|drive|validate, "mult"}.
    # A probe like memory_alpha is correlated to the memory_alpha knob but the two
    # are separate; a match ties them so the probe can DRIVE the knob (servo) or
    # VALIDATE it (record knob-value vs probe-reading each turn -> correlation).
    PROBE_MATCH_PATH = os.path.join(ROOT, "invariants", "out", "probe_matches.json")
    probe_matches = {}
    try:
        with open(PROBE_MATCH_PATH, "r", encoding="utf-8") as _pmf:
            probe_matches = {str(k): dict(v) for k, v in json.load(_pmf).items()}
    except (OSError, ValueError):
        probe_matches = {}
    match_hist = {}  # probe name -> deque of (knob_value, probe_reading); session-only

    def _save_probe_matches():
        try:
            with open(PROBE_MATCH_PATH, "w", encoding="utf-8") as pmf:
                json.dump(probe_matches, pmf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save probe matches: {e}" + Style.RESET_ALL)

    # Re-arm any persisted servo bindings (mode=drive) so a restart keeps steering.
    for _pn, _m in probe_matches.items():
        if _m.get("mode") == "drive" and _m.get("knob"):
            tuner_bindings[_m["knob"]] = ([(1.0, _pn)], float(_m.get("mult", 1.0)))

    # Commands the operator has EXPOSED to the model as tools: word -> mode, where
    # mode is "stage" (the model's <<CMD: ...>> proposes it, awaiting :accept) or
    # "direct" (it enters the shell queue immediately). Everything else the
    # model still cannot run. Meta-commands (:run/:solve/:macro/:game/:sandbox)
    # reach OTHER commands through their argument -- allowed, but warned.
    EXPOSED_CMD_PATH = os.path.join(ROOT, "invariants", "out", "exposed_commands.json")
    EXPOSED_KNOB_PATH = os.path.join(ROOT, "invariants", "out", "exposed_knobs.json")
    HIDDEN_CMD_PATH = os.path.join(ROOT, "invariants", "out", "hidden_commands.json")
    EXPOSE_META_CMDS = {"run", "solve", "macro", "game", "sandbox", "spawn", "accept"}
    hidden_commands = set()
    try:
        with open(HIDDEN_CMD_PATH, "r", encoding="utf-8") as _hcf:
            _raw_hidden = json.load(_hcf)
            if isinstance(_raw_hidden, list):
                hidden_commands = {str(x).lstrip(":").lower() for x in _raw_hidden if str(x).strip()}
            elif isinstance(_raw_hidden, dict):
                hidden_commands = {str(k).lstrip(":").lower() for k, v in _raw_hidden.items() if v}
    except (OSError, ValueError):
        hidden_commands = set()

    exposed_commands = {}
    try:
        with open(EXPOSED_CMD_PATH, "r", encoding="utf-8") as _ecf:
            exposed_commands = {str(k): str(v) for k, v in json.load(_ecf).items()}
    except (OSError, ValueError):
        exposed_commands = {}
    if hidden_commands:
        exposed_commands = {k: v for k, v in exposed_commands.items() if k not in hidden_commands}

    exposed_knobs = set()
    try:
        with open(EXPOSED_KNOB_PATH, "r", encoding="utf-8") as _ekf:
            _raw_knobs = json.load(_ekf)
            if isinstance(_raw_knobs, list):
                exposed_knobs = {str(x).strip() for x in _raw_knobs if str(x).strip()}
            elif isinstance(_raw_knobs, dict):
                exposed_knobs = {str(k).strip() for k, v in _raw_knobs.items() if v}
    except (OSError, ValueError):
        exposed_knobs = set()
    exposed_knobs = {k for k in exposed_knobs if k in tuner.triggers and not _is_shadow_trigger(k, tuner)}

    def _save_exposed_commands():
        try:
            os.makedirs(os.path.dirname(EXPOSED_CMD_PATH) or ".", exist_ok=True)
            with open(EXPOSED_CMD_PATH, "w", encoding="utf-8") as ecf:
                json.dump(exposed_commands, ecf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save exposed commands: {e}" + Style.RESET_ALL)

    def _save_hidden_commands():
        try:
            os.makedirs(os.path.dirname(HIDDEN_CMD_PATH) or ".", exist_ok=True)
            with open(HIDDEN_CMD_PATH, "w", encoding="utf-8") as hcf:
                json.dump(sorted(hidden_commands), hcf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save hidden commands: {e}" + Style.RESET_ALL)

    def _save_exposed_knobs():
        try:
            os.makedirs(os.path.dirname(EXPOSED_KNOB_PATH) or ".", exist_ok=True)
            with open(EXPOSED_KNOB_PATH, "w", encoding="utf-8") as ekf:
                json.dump(sorted(exposed_knobs), ekf, indent=2)
        except Exception as e:
            print(Fore.RED + f"[Error] Could not save exposed knobs: {e}" + Style.RESET_ALL)

    def _all_shell_commands():
        return BUILTIN_COMMANDS | set(macro_aliases)

    def _known_model_commands():
        return (_all_shell_commands() - hidden_commands) - {"expose"}

    def _visible_command_reference():
        return ", ".join(f":{w}" for w in sorted(_all_shell_commands() - hidden_commands))

    def _hidden_overwrite_blocked(name, surface):
        raw_name = str(name or "").lstrip(":").lower()
        stem_name = os.path.splitext(os.path.basename(raw_name))[0].lower()
        blocked_name = raw_name if raw_name in hidden_commands else (stem_name if stem_name in hidden_commands else None)
        if blocked_name:
            print(
                Fore.YELLOW
                + f"[{surface}] Refusing to overwrite hidden command ':{blocked_name}'. "
                + f"Reveal it first with ':hide :{blocked_name} off' if you mean to reuse that name."
                + Style.RESET_ALL
            )
            return True
        return False

    def _run_exposed_command_tool(cmd_requests):
        result_lines = [CMD_TOOL_HEADER]
        direct_cmds = []
        seen_any = False
        for requested in cmd_requests:
            for cmd in split_cmd_tool_commands(requested):
                seen_any = True
                word = command_word(cmd)
                if not word:
                    result_lines.append(f"- refused malformed command: {cmd}")
                    continue
                if word in hidden_commands:
                    result_lines.append("- refused an unavailable command.")
                    continue
                if word == "expose":
                    result_lines.append("- refused :expose; the model cannot grant itself command tools.")
                    continue
                if word not in exposed_commands:
                    hint = did_you_mean(word, _known_model_commands())
                    result_lines.append(f"- refused :{word}; it is not exposed as a model tool.{hint}")
                    continue
                if word not in BUILTIN_COMMANDS and word not in macro_aliases:
                    result_lines.append(f"- refused :{word}; it is exposed but no longer exists as a command.")
                    continue
                mode = str(exposed_commands.get(word, "stage")).lower()
                if mode == "direct":
                    direct_cmds.append(cmd)
                    result_lines.append(f"- queued direct command: {cmd}")
                else:
                    _stage_for_accept(cmd, why="an exposed command tool")
                    result_lines.append(f"- staged for operator :accept: {cmd}")
        if direct_cmds:
            input_queue[:0] = direct_cmds
            print(Fore.MAGENTA + f"\n[Command Tool] Queued {len(direct_cmds)} direct command(s): {'; '.join(direct_cmds)}" + Style.RESET_ALL)
        if not seen_any:
            result_lines.append("- no command found in the request.")
        result = "\n".join(result_lines)
        memory.append_event(
            "command_tool_model_requested",
            text=result,
            tags=["command_tool"],
            provenance={"requests": cmd_requests, "direct_count": len(direct_cmds)},
        )
        return result

    egg_state = {"over": False}  # rising-edge latch for the consciousness egg
    startup_user_input = os.environ.get("PHENOMENALITY_STARTUP_PROMPT")
    if os.environ.get("PHENOMENALITY_AUTO_RESUME", "0").strip().lower() in {"1", "true", "yes"}:
        resumed_session, recovered = recover_session_context(
            memory,
            session_id=os.environ.get("PHENOMENALITY_RESUME_SESSION", "last"),
            max_turns=MAX_SESSION_TURNS,
        )
        if recovered and recovered[-1][0] == "user":
            session_context = recovered_to_session_context(recovered[:-1])
            startup_user_input = recovered[-1][1]
            memory.append_event(
                "context_auto_resumed",
                tags=["memory_tool", "context"],
                provenance={
                    "resumed_session_id": resumed_session,
                    "context_messages": len(session_context),
                    "startup_user_chars": len(startup_user_input),
                },
            )
            print(
                Fore.GREEN
                + (
                    f"[Context] Auto-resuming interrupted session {resumed_session}. "
                    "The first model turn will answer the saved unanswered message."
                )
                + Style.RESET_ALL
            )
    
    input_queue = []
    if startup_user_input:
        input_queue.append(startup_user_input)

    while True:
        try:
            reading_turn_source = None  # set when the turn is the model reading, not the operator speaking
            if input_queue:
                user_input = input_queue.pop(0)
                print(Fore.YELLOW + f"\nYou: {user_input}" + Style.RESET_ALL)
            elif doc_autoread and doc_autoread["remaining"] > 0:
                # Reading as dialogue, minimally: the framed chunk IS the user
                # turn -- no synthesized question, no cue, no words put in
                # anyone's mouth. The model responds to the text natively, and
                # the turn is sense-labeled, channel-accounted, and remembered
                # like any other conversation turn.
                staged_reading = pending_document_tool_result  # consume a manual :doc stage first
                pending_document_tool_result = None
                if staged_reading is None:
                    earlier_reflections = " ".join(list(recent_responses)[:-1])
                    pick = select_next_chunk(
                        doc_library,
                        last_assistant_response,
                        doc_autoread["mode"],
                        earlier_thoughts=earlier_reflections,
                    )
                    if pick is None:
                        print(Fore.CYAN + "[Doc] Library fully read; auto-read finished." + Style.RESET_ALL)
                        doc_autoread = None
                        continue
                    doc_session = doc_library[pick["session_index"]]
                    doc_session["cursor"] = pick["chunk_index"]
                    doc_session.setdefault("read", set()).add(pick["chunk_index"])
                    record_chunk_read(memory, doc_session, pick["chunk_index"])
                    if pick["mode"] in ("reply", "reply_thread"):
                        # Only the echo-following mode is word-contingent; the
                        # order/interleave course is deliberately not.
                        impact_state["pending"].append(
                            {
                                "cause": "reply steered the reading",
                                "effect": f"next chunk chosen by shared ground: {', '.join(pick['overlap'])}",
                            }
                        )
                    staged_reading = stage_chunk(
                        doc_session,
                        index=pick["chunk_index"],
                        reply_note=reading_reply_note(pick),
                    )
                    picked_note = (
                        f"chunk {pick['chunk_index'] + 1}/{doc_session['chunk_count']} of "
                        f"{doc_session['source_name']} ({pick['mode']}"
                        + (f": {', '.join(pick['overlap'])}" if pick.get("overlap") else "")
                        + ")"
                    )
                    print(Fore.CYAN + f"[Doc] Reading {picked_note}." + Style.RESET_ALL)
                doc_autoread["remaining"] -= 1
                if doc_autoread["remaining"] <= 0:
                    doc_autoread = None
                user_input = staged_reading
                # A reading turn is the model's own act -- the frame literally
                # begins "I'm reading ..." -- so the console says "Me:", not
                # "You:". The operator didn't speak; it is reading to itself.
                reading_turn_source = doc_session["source_name"] if doc_session else "reading"
                preview = " ".join(staged_reading.split())[:160]
                print(Fore.CYAN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL + preview + " [...]")

            elif replace_turns_remaining > 0 and replace_agent:
                # Agent replaces user
                print(Fore.MAGENTA + f"\n[{replace_agent.name} is thinking...]" + Style.RESET_ALL)
                _sys_tuner = tuner
                _sys_probes = probes
                _sys_tb = tuner_bindings
                tuner = replace_agent.tuner
                probes = replace_agent.probes
                tuner_bindings = replace_agent.tuner_bindings
                
                # Generate user input autonomously
                r_prompt = build_prompt(
                    "[Your turn to respond]", 
                    session_context=session_context if session_context_enabled else None
                )
                r_response = generate_agentic_text(
                    model,
                    instruction=r_prompt,
                    config=config,
                    max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                    synthesis_recorder=None,
                    chatty_log=False,
                    pre_formatted=True,
                    return_telemetry=False,
                )
                user_input = r_response.strip() if r_response else "..."
                
                # Restore
                tuner = _sys_tuner
                probes = _sys_probes
                tuner_bindings = _sys_tb
                replace_turns_remaining -= 1
                _last_replace_name = replace_agent.name
                if replace_turns_remaining <= 0:
                    replace_agent = None
                print(Fore.MAGENTA + Style.BRIGHT + f"\n[{_last_replace_name} replacing You]: " + Style.RESET_ALL + user_input)
            else:
                _last_replace_name = None
                if getattr(config, "auto_probe_readings", False) and _sys_probes:
                    recent = []
                    for row in reversed(list(turn_log)):
                        v = {}
                        for p in _sys_probes:
                            if f"probe_{p}" in row:
                                try:
                                    v[p] = float(row[f"probe_{p}"])
                                except (TypeError, ValueError):
                                    pass
                        if v:
                            recent.append(v)
                            break
                    if recent:
                        print(Fore.CYAN + "[Auto Probe Readings]" + Style.RESET_ALL)
                        for pname, val in recent[0].items():
                            print(Fore.CYAN + f"  {pname}: {val:+.3f}" + Style.RESET_ALL)
                            
                prefix = f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] You: " if show_timestamps else "\nYou: "
                prompt_str = Fore.MAGENTA + Style.BRIGHT + prefix + Style.RESET_ALL
                # Once the listener is active, every turn is served from its
                # queue (so mid-generation typing was never dropped); otherwise
                # the plain, unchanged input() path.
                user_input = listener.get(prompt_str) if listener.active else input(prompt_str)

            
            user_input = user_input.strip()
            if user_input.startswith(":") and ";" in user_input and not command_keeps_semicolons(user_input):
                cmds = [c.strip() for c in user_input.split(";") if c.strip()]
                if len(cmds) > 1:
                    user_input = cmds[0]
                    input_queue.extend(cmds[1:])

            # A trailing ' because <reason>' on any command is provenance, not an
            # argument: peel it off, log it, and hand the handler a clean line.
            command_because = None
            if user_input.startswith(":") or user_input.startswith("@"):
                user_input, command_because = split_because(user_input)
            target_agent = None
            if user_input.startswith("@"):
                t_parts = user_input.split(maxsplit=1)
                if len(t_parts) == 2:
                    t_name = t_parts[0][1:]
                    user_input = t_parts[1]
                    # Find agent
                    for a in joined_agents:
                        if a.name == t_name:
                            target_agent = a
                            break
                    if not target_agent and replace_agent and replace_agent.name == t_name:
                        target_agent = replace_agent
                    if target_agent:
                        print(Fore.CYAN + f"[Target] Directing command to '{t_name}'..." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Target] Unknown agent '{t_name}'. Ignoring target." + Style.RESET_ALL)

            # Swap global state if target_agent
            if target_agent:
                _sys_tuner = tuner
                _sys_probes = probes
                _sys_tb = tuner_bindings
                tuner = target_agent.tuner
                probes = target_agent.probes
                tuner_bindings = target_agent.tuner_bindings

            if command_because:
                related = set()
                # Check for probes
                for p in probes:
                    match = re.search(r'(?<!\w)([+-]?)' + re.escape(p) + r'\b', command_because, re.IGNORECASE)
                    if match:
                        prefix = match.group(1) or ""
                        related.add(f"{prefix}probe:{p}")
                # Check for knobs (triggers)
                for k in tuner.triggers:
                    k_name = k[len("probe_"):] if k.startswith("probe_") else k
                    match = re.search(r'(?<!\w)([+-]?)' + re.escape(k_name) + r'\b', command_because, re.IGNORECASE)
                    if match:
                        prefix = match.group(1) or ""
                        related.add(f"{prefix}knob:{k_name}")
                        
                mem_prov = {"command": user_input.split()[0] if user_input else "", "because": command_because}
                if related:
                    mem_prov["related"] = sorted(list(related))
                    
                memory.append_event(
                    "command_because",
                    tags=["provenance"],
                    provenance=mem_prov,
                )
                noted_str = command_because
                if related:
                    noted_str += Fore.MAGENTA + f" (linked to {', '.join(sorted(list(related)))})" + Fore.BLUE
                print(Fore.BLUE + f"[Because] noted: {noted_str}" + Style.RESET_ALL)

            
            if user_input.startswith(":self ") or user_input == ":self":
                sargs = user_input.split()[1:]
                if not sargs:
                    print(Fore.YELLOW + "[System] Usage: :self <name> | :self choose | :self save <name>" + Style.RESET_ALL)
                    continue
                if sargs[0] in ("save", "create"):
                    alias = sargs[1] if len(sargs) > 1 else "choose"
                    user_input = f":save self {alias}"
                elif sargs[0] == "choose":
                    print(Fore.CYAN + "[System] Asking model to pick a persona/macro..." + Style.RESET_ALL)
                    prompt = (
                        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"You are selecting a persona/macro to initialize.\n"
                        f"Available options:\n" + "\n".join(f"- {p}" for p in sorted(macro_aliases.keys())) + "\n\n"
                        f"Select the single most appropriate persona/macro from the list. "
                        f"Output ONLY the exact name, and nothing else.<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    nm = generate_agentic_text(
                        model, instruction=prompt, config=config,
                        max_new_tokens=20, chatty_log=False, pre_formatted=True
                    )
                    alias = (nm or "").strip()
                    if alias in macro_aliases:
                        print(Fore.GREEN + f"[System] Model chose '{alias}'." + Style.RESET_ALL)
                        user_input = f":{alias}"
                    else:
                        print(Fore.YELLOW + f"[System] Model selected invalid macro '{alias}'. Aborting." + Style.RESET_ALL)
                        continue
                else:
                    user_input = f":{sargs[0]}"

            if user_input.startswith(":save self"):
                # Alias for :macro name self <name>
                sargs = user_input.split()
                if len(sargs) >= 3:
                    alias = sargs[2]
                    if alias == "choose":
                        print(Fore.CYAN + "[System] Asking model for a name..." + Style.RESET_ALL)
                        nm = generate_agentic_text(
                            model,
                            instruction="Pick a short, one-word name for this persona based on the current tuning state. Reply with just the lowercase name.",
                            config=config,
                            max_new_tokens=10,
                            chatty_log=False,
                            pre_formatted=False
                        )
                        alias = re.sub(r'[^a-z0-9]', '', (nm or "agent").lower())[:20]
                        if not alias: alias = "agent"
                        print(Fore.CYAN + f"[System] Model chose '{alias}'." + Style.RESET_ALL)
                    user_input = f":macro name self {alias}"
                else:
                    print(Fore.YELLOW + "[System] Usage: :save self <name> | :save self choose" + Style.RESET_ALL)
                    continue

            if user_input.startswith(":spawn"):
                sargs = user_input.split()
                if len(sargs) < 3:
                    print(Fore.YELLOW + "[System] Usage: :spawn <name> join | :spawn <name> replace [N] | :spawn <name> drop" + Style.RESET_ALL)
                    continue
                a_name = sargs[1]
                mode = sargs[2].lower()
                
                if mode == "drop":
                    joined_agents = [a for a in joined_agents if a.name != a_name]
                    if replace_agent and replace_agent.name == a_name:
                        replace_agent = None
                        replace_turns_remaining = 0
                    print(Fore.CYAN + f"[Spawn] Dropped agent '{a_name}'." + Style.RESET_ALL)
                    continue
                
                # Create agent state
                new_agent = AgentState(name=a_name, tuner=TriggerTuner())
                # Load profile if macro exists
                macro_path = macro_aliases.get(a_name, a_name)
                if os.path.isfile(macro_path) or os.path.isfile(os.path.join(ROOT, "invariants", "out", "macros", f"{a_name}.txt")):
                    real_path = macro_path if os.path.isfile(macro_path) else os.path.join(ROOT, "invariants", "out", "macros", f"{a_name}.txt")
                    try:
                        with open(real_path, "r", encoding="utf-8") as rf:
                            lines = [l.strip() for l in rf.read().splitlines() if l.strip() and not l.strip().startswith("#")]
                        
                        # Temporarily swap global tuner to apply macro commands
                        _old_t = tuner
                        _old_p = probes
                        _old_tb = tuner_bindings
                        
                        tuner = new_agent.tuner
                        probes = new_agent.probes
                        tuner_bindings = new_agent.tuner_bindings
                        
                        # simple execution of tune commands for the agent
                        for cmd in lines:
                            if cmd.startswith(":tune"):
                                targs = cmd[len(":tune"):].split()
                                if len(targs) >= 2:
                                    try:
                                        tuner.set(targs[0], float(targs[1]))
                                    except:
                                        pass
                                
                        new_agent.tuner = tuner
                        new_agent.probes = probes
                        new_agent.tuner_bindings = tuner_bindings
                        
                        tuner = _old_t
                        probes = _old_p
                        tuner_bindings = _old_tb
                        print(Fore.GREEN + f"[Spawn] Loaded profile for '{a_name}' from {real_path}." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Spawn] Failed to load profile for '{a_name}': {e}" + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + f"[Spawn] No saved profile found for '{a_name}'. Generating profile based on name..." + Style.RESET_ALL)
                    g_prompt = (
                        f"You are generating a tuning profile for an agent named '{a_name}'. "
                        "Respond ONLY with a series of lines starting with :tune, :steer, or :label to configure this persona. "
                        "For example: ':tune response_tokens 128\\n:steer accuracy 0.5'. "
                        "Output no other text."
                    )
                    g_out = generate_agentic_text(
                        model, instruction=g_prompt, config=config,
                        max_new_tokens=200, chatty_log=False, pre_formatted=False
                    )
                    if g_out:
                        _old_t, _old_p, _old_tb = tuner, probes, tuner_bindings
                        tuner, probes, tuner_bindings = new_agent.tuner, new_agent.probes, new_agent.tuner_bindings
                        g_lines = [l.strip() for l in g_out.splitlines() if l.strip().startswith(":")]
                        print(Fore.CYAN + f"[Spawn] Generated {len(g_lines)} config commands for '{a_name}'." + Style.RESET_ALL)
                        for cmd in g_lines:
                            if cmd.startswith(":tune"):
                                targs = cmd[len(":tune"):].split()
                                if len(targs) >= 2:
                                    try: tuner.set(targs[0], float(targs[1]))
                                    except: pass
                        new_agent.tuner, new_agent.probes, new_agent.tuner_bindings = tuner, probes, tuner_bindings
                        tuner, probes, tuner_bindings = _old_t, _old_p, _old_tb
                    else:
                        print(Fore.CYAN + f"[Spawn] Generation failed, using default state." + Style.RESET_ALL)
                if mode == "join":
                    if not any(a.name == a_name for a in joined_agents):
                        joined_agents.append(new_agent)
                    print(Fore.GREEN + f"[Spawn] Agent '{a_name}' joined the panel." + Style.RESET_ALL)
                elif mode == "replace":
                    n_turns = int(sargs[3]) if len(sargs) > 3 else 1
                    replace_agent = new_agent
                    replace_turns_remaining = n_turns
                    print(Fore.GREEN + f"[Spawn] Agent '{a_name}' is replacing the user for {n_turns} turn(s)." + Style.RESET_ALL)
                continue

            if user_input == ":macro" or user_input.startswith(":macro "):
                mtail = user_input[len(":macro"):].strip()
                mtok = mtail.split()
                sub = mtok[0].lower() if mtok else ""
                if sub == "restore":
                    payload = mtail[len("restore"):].strip()
                    if not payload:
                        print(Fore.YELLOW + "[System] Usage: :macro restore {\"path\":\"...\",\"lines\":[...]}" + Style.RESET_ALL)
                        continue
                    try:
                        if payload.startswith("{"):
                            obj = json.loads(payload)
                            dest = str(obj.get("path", "")).strip()
                            body_lines = obj.get("lines", [])
                        else:
                            args = payload.split(maxsplit=1)
                            if len(args) < 2:
                                raise ValueError("missing path or JSON lines")
                            dest = args[0]
                            body_lines = json.loads(args[1])
                        if not dest or not isinstance(body_lines, list):
                            raise ValueError("restore payload needs a path and a list of lines")
                        if _hidden_overwrite_blocked(dest, "System"):
                            continue
                        out_path = dest if os.path.isabs(dest) else os.path.join(ROOT, dest)
                        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                        with open(out_path, "w", encoding="utf-8") as wf:
                            for line in body_lines:
                                wf.write(str(line) + "\n")
                        print(Fore.GREEN + f"[System] Restored macro file {dest} ({len(body_lines)} line(s))." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Error] Could not restore macro: {e}" + Style.RESET_ALL)
                    continue
                if sub == "name":
                    kind = mtok[1].lower() if len(mtok) >= 2 else ""
                    if kind == "self":
                        # :macro name self [file] -- regenerate the CURRENT shell
                        # state as replayable commands: probes, macro-command
                        # files/aliases, exposed command tools, and game configs.
                        alias_name = mtail.split(maxsplit=2)[2].strip() if len(mtok) >= 3 else "self"
                        # If they provided a name, map it to the standard macros folder
                        if not alias_name.endswith(".txt") and not "/" in alias_name and not "\\" in alias_name:
                            dest = os.path.join(ROOT, "invariants", "out", "macros", f"{alias_name}.txt")
                        else:
                            dest = alias_name # user provided an explicit path
                            alias_name = os.path.splitext(os.path.basename(dest))[0]
                            
                        if _hidden_overwrite_blocked(alias_name, "System"):
                            continue
                        macro_lines, restore_stats = build_session_restore_macro(
                            probes,
                            macro_aliases,
                            exposed_commands,
                            exposed_knobs,
                            hidden_commands,
                            self_dest=dest,
                        )
                        n_cmds = sum(1 for l in macro_lines if l.startswith(":"))
                        try:
                            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                            with open(dest, "w", encoding="utf-8") as wf:
                                for l in macro_lines:
                                    wf.write(l + "\n")
                            macro_aliases["self"] = dest
                            macro_aliases[alias_name] = dest
                            _save_macro_aliases()
                            print(
                                Fore.GREEN
                                + f"[System] Wrote a state-regenerating macro ({n_cmds} command(s): "
                                + f"{restore_stats['probes']} probe(s), {restore_stats['macro_files']} macro file(s), "
                                + f"{restore_stats['macro_aliases']} macro alias(es), {restore_stats['hidden_commands']} hidden command(s), "
                                + f"{restore_stats['exposed_commands']} exposed command(s), "
                                + f"{restore_stats['exposed_knobs']} exposed knob(s), "
                                + f"{restore_stats['games']} game config(s)) to {dest}, aliased 'self'. Recreate with :run self."
                                + Style.RESET_ALL
                            )
                            if not probes:
                                print(Fore.YELLOW + "[System] (No active probes to regenerate -- the macro is empty.)" + Style.RESET_ALL)
                            if restore_stats["skipped"]:
                                print(Fore.YELLOW + f"[System] Skipped {len(restore_stats['skipped'])} restore item(s); see comments in the macro." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not write self macro: {e}" + Style.RESET_ALL)
                        continue
                    if kind == "strip":
                        # :macro name strip <src> [dest] -- write a STRIPPED COPY of
                        # <src> to a NEW file, leaving the original intact (vs :macro
                        # strip, which rewrites in place). Default dest: '<name>_og'
                        # -> '<name>', otherwise '<name>_stripped'.
                        args = mtail.split(maxsplit=2)[2].split() if len(mtok) >= 3 else []
                        if not args:
                            print(Fore.YELLOW + "[System] Usage: :macro name strip <src> [dest]" + Style.RESET_ALL)
                            continue
                        src = macro_aliases.get(args[0], args[0])
                        if not os.path.isfile(src):
                            print(Fore.YELLOW + f"[System] Macro not found: {src}" + Style.RESET_ALL)
                            continue
                        if len(args) >= 2:
                            dest = args[1]
                        else:
                            _b, _ext = os.path.splitext(src)
                            dest = (_b[:-3] + _ext) if _b.endswith("_og") else (_b + "_stripped" + _ext)
                        try:
                            with open(src, "r", encoding="utf-8") as rf:
                                kept, removed = strip_macro_lines(rf.read().splitlines())
                            _stem = os.path.splitext(os.path.basename(dest))[0]
                            if _hidden_overwrite_blocked(_stem, "System"):
                                continue
                            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                            with open(dest, "w", encoding="utf-8") as wf:
                                for line in kept:
                                    wf.write(line + "\n")
                            macro_aliases[_stem] = dest
                            _save_macro_aliases()
                            cmd_kept = [k for k in kept if not k.strip().startswith("#")]
                            print(Fore.GREEN + f"[System] Stripped copy of {src} -> {dest} (kept {len(cmd_kept)} command(s), dropped {len(removed)}); original untouched, aliased '{_stem}'." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not strip-copy macro: {e}" + Style.RESET_ALL)
                        continue
                    # :macro name <alias> <file> -- bind a short name to a macro file.
                    if len(mtok) < 3:
                        print(Fore.YELLOW + "[System] Usage: :macro name <alias> <file>  |  :macro name self [file]  |  :macro name strip <src> [dest]" + Style.RESET_ALL)
                    else:
                        alias = mtok[1]
                        mpath = mtail.split(maxsplit=2)[2].strip()
                        if _hidden_overwrite_blocked(alias, "System"):
                            continue
                        macro_aliases[alias] = mpath
                        _save_macro_aliases()
                        exists_note = "" if os.path.isfile(mpath) else " (file does not exist yet)"
                        print(Fore.GREEN + f"[System] Macro alias '{alias}' -> {mpath}{exists_note}. Use :run {alias} or :macro strip {alias}." + Style.RESET_ALL)
                    continue
                if sub == "strip":
                    # :macro strip <alias|file> -- rewrite IN PLACE, dropping every
                    # line that does not mutate a probe or a tuning knob.
                    target = mtail.split(maxsplit=1)[1].strip() if len(mtok) >= 2 else ""
                    mpath = macro_aliases.get(target, target)
                    if not target:
                        print(Fore.YELLOW + "[System] Usage: :macro strip <alias|file>  (or :macro name strip <src> [dest] to keep the original)" + Style.RESET_ALL)
                    elif _hidden_overwrite_blocked(target, "System"):
                        continue
                    elif not os.path.isfile(mpath):
                        print(Fore.YELLOW + f"[System] Macro not found: {mpath}" + Style.RESET_ALL)
                    else:
                        try:
                            with open(mpath, "r", encoding="utf-8") as rf:
                                kept, removed = strip_macro_lines(rf.read().splitlines())
                            with open(mpath, "w", encoding="utf-8") as wf:
                                for line in kept:
                                    wf.write(line + "\n")
                            cmd_kept = [k for k in kept if not k.strip().startswith("#")]
                            print(Fore.GREEN + f"[System] Stripped {mpath}: kept {len(cmd_kept)} probe/tuning command(s), removed {len(removed)} display-only line(s)." + Style.RESET_ALL)
                            for r in removed[:20]:
                                print(Fore.CYAN + f"  - {r}" + Style.RESET_ALL)
                            if len(removed) > 20:
                                print(Fore.CYAN + f"  ... and {len(removed) - 20} more." + Style.RESET_ALL)
                            if not cmd_kept:
                                print(Fore.YELLOW + "[System] Nothing probe/tuning-related remained -- the macro is now empty of commands." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not strip macro: {e}" + Style.RESET_ALL)
                    continue
                if sub == "drop":
                    target = mtail.split(maxsplit=1)[1].strip() if len(mtok) >= 2 else ""
                    if not target:
                        print(Fore.YELLOW + "[System] Usage: :macro drop <alias>" + Style.RESET_ALL)
                        continue
                    if target in macro_aliases:
                        mpath = macro_aliases[target]
                        del macro_aliases[target]
                        _save_macro_aliases()
                        try:
                            if os.path.isfile(mpath):
                                os.remove(mpath)
                                print(Fore.GREEN + f"[System] Dropped macro alias '{target}' and deleted its file ({mpath})." + Style.RESET_ALL)
                            else:
                                print(Fore.GREEN + f"[System] Dropped macro alias '{target}'. (File {mpath} was already missing)." + Style.RESET_ALL)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Dropped alias '{target}', but could not delete file {mpath}: {e}" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[System] Unknown macro alias '{target}'." + Style.RESET_ALL)
                    continue
                # Legacy create form: :macro <file> <cmd1> ; <cmd2> ; ...
                parts = mtail.split(maxsplit=1)
                if len(parts) < 2:
                    print(Fore.YELLOW + "[System] Usage: :macro <file> <c1> ; <c2> ...  |  :macro name <alias> <file>  |  :macro strip <alias|file>  |  :macro drop <alias>" + Style.RESET_ALL)
                else:
                    raw_file = parts[0]
                    if _hidden_overwrite_blocked(raw_file, "System"):
                        continue
                    mac_file = macro_aliases.get(raw_file, raw_file)
                    mac_cmds = _split_macro_commands(parts[1])
                    try:
                        os.makedirs(os.path.dirname(mac_file) or ".", exist_ok=True)
                        with open(mac_file, "w", encoding="utf-8") as wf:
                            for c in mac_cmds:
                                wf.write(c + "\n")
                        print(Fore.GREEN + f"[System] Wrote {len(mac_cmds)} command(s) to macro '{raw_file}' ({mac_file}). Execute with: :run {raw_file}" + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Error] Could not write macro file: {e}" + Style.RESET_ALL)
                continue

            if user_input.startswith(":run "):
                raw_target = user_input[len(":run "):].strip()
                if not raw_target:
                    continue
                if raw_target in macro_aliases or os.path.isfile(raw_target):
                    target = raw_target
                    args = []
                else:
                    parts = raw_target.split()
                    target = parts[0]
                    args = parts[1:]
                
                resolved = macro_aliases.get(target, target)
                # Glob target (fun*, *.txt): run every matching macro file.
                if any(c in target for c in "*?[") and not os.path.isfile(resolved):
                    matches = sorted(glob.glob(resolved))
                    if not matches:
                        print(Fore.YELLOW + f"[System] No macros match '{target}'." + Style.RESET_ALL)
                        continue
                    total = []
                    for m in matches:
                        try:
                            sub = expand_macro_lines(m, macro_aliases, args=args)
                        except Exception as e:
                            print(Fore.RED + f"[Error] Could not read macro '{m}': {e}" + Style.RESET_ALL)
                            sub = None
                        if sub:
                            total.extend(sub)
                    if total:
                        names = ", ".join(os.path.basename(x) for x in matches)
                        print(Fore.CYAN + f"[System] Queued {len(total)} command(s) from {len(matches)} macro(s) matching '{target}' ({names})." + Style.RESET_ALL)
                        input_queue.extend(total)
                    else:
                        print(Fore.YELLOW + f"[System] Macros matching '{target}' had no runnable commands." + Style.RESET_ALL)
                    continue
                try:
                    lines = expand_macro_lines(target, macro_aliases, args=args)
                except Exception as e:
                    print(Fore.RED + f"[Error] Could not read macro '{target}': {e}" + Style.RESET_ALL)
                    continue
                if lines is None:
                    print(Fore.YELLOW + f"[System] Macro not found: {resolved}" + Style.RESET_ALL)
                elif lines:
                    print(Fore.CYAN + f"[System] Queued {len(lines)} command(s) from {target}." + Style.RESET_ALL)
                    input_queue.extend(lines)
                else:
                    print(Fore.YELLOW + f"[System] '{target}' had no runnable commands." + Style.RESET_ALL)
                continue

            # :solve <name> [goal] -- have the model WRITE a parameterized macro
            # (a list of ':' commands using $1..$9 / $@), then stage it for
            # :accept so ':<name> args' can run it directly.
            if user_input.startswith(":solve"):
                sbody = user_input[len(":solve"):].strip()
                sparts = sbody.split(maxsplit=1)
                if not sparts:
                    print(Fore.YELLOW + "[Solve] Usage: :solve <command_name> [what it should do]" + Style.RESET_ALL)
                    continue
                sname = re.sub(r"[^a-z0-9_]", "_", sparts[0].lower())[:40].strip("_")
                rest = sparts[1].strip() if len(sparts) > 1 else ""
                if ":" in rest:
                    goal_part, args_part = rest.rsplit(":", 1)
                    try:
                        import shlex
                        arg_names = shlex.split(args_part)
                    except ValueError:
                        arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                elif "--" in rest:
                    goal_part, args_part = rest.rsplit("--", 1)
                    try:
                        import shlex
                        arg_names = shlex.split(args_part)
                    except ValueError:
                        arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                else:
                    try:
                        import shlex
                        tokens = shlex.split(rest)
                    except ValueError:
                        tokens = rest.split()
                        
                    arg_names = []
                    while tokens and (
                        tokens[-1].startswith(("+", "-", "$", "["))
                        or tokens[-1].endswith(("?", "!", "]", "!!"))
                        or "=" in tokens[-1]
                    ):
                        arg_names.insert(0, tokens.pop())
                    goal = " ".join(tokens) or sname.replace("_", " ")

                if sname == "auto" and not arg_names and goal == "auto":
                    print(Fore.YELLOW + "[Solve] Usage: :solve auto <+/- reference probes to steer toward>" + Style.RESET_ALL)
                    continue

                if sname == "choose" and goal == "choose":
                    print(Fore.YELLOW + "[Solve] You must describe what the macro should do for the model to choose a name (e.g. ':solve choose drop all probes')." + Style.RESET_ALL)
                    continue

                if sname == "auto":
                    refs = " ".join(arg_names) if arg_names else goal
                    if refs == "auto":
                        refs = ""
                    goal = f"steer the model using these reference probes: {refs}"
                    arg_names = []
                    sname = "choose"

                if sname == "choose":
                    print(Fore.CYAN + "[Solve] Asking model for a command name..." + Style.RESET_ALL)
                    nm = generate_agentic_text(
                        model,
                        instruction=f"Pick a short, one-word command name for a macro that does: {goal}. Reply with ONLY the lowercase name.",
                        config=config,
                        max_new_tokens=10,
                        chatty_log=False,
                        pre_formatted=False
                    )
                    sname = re.sub(r'[^a-z0-9_]', '', (nm or "macro").lower())[:40].strip("_")
                    if not sname or sname in BUILTIN_COMMANDS or sname in hidden_commands:
                        sname = "macro"
                    print(Fore.CYAN + f"[Solve] Model chose '{sname}'." + Style.RESET_ALL)
                
                if _hidden_overwrite_blocked(sname, "Solve"):
                    continue
                if not sname or sname in BUILTIN_COMMANDS:
                    print(Fore.YELLOW + f"[Solve] '{sname}' can't be a macro name (empty or a built-in command)." + Style.RESET_ALL)
                    continue

                prompt_args_str = ""
                if arg_names:
                    mapping_parts = []
                    clean_names = []
                    arg_specs = []
                    for i, arg in enumerate(arg_names):
                        spec = parse_macro_arg_spec(arg, fallback=f"arg{i + 1}")
                        clean_arg = spec["name"]
                        clean_names.append(clean_arg)
                        arg_specs.append(spec)
                        
                        hints = []
                        if spec["hint_prefix"] == "+":
                            hints.append("positive/additive")
                        elif spec["hint_prefix"] == "-":
                            hints.append("negative/subtractive")
                            
                        if clean_arg.endswith("s"):
                            hints.append("a list of multiple items")
                        elif "name" in clean_arg:
                            hints.append("a specific exact name")

                        if spec["optional"]:
                            hints.append("optional")
                        if spec["default"] is not None:
                            hints.append(f"default {spec['default']!r}")
                            
                        if spec["no_auto_or_choose"]:
                            hints.append("STRICT: cannot accept auto or choose")
                        elif spec["no_choose"]:
                            hints.append("cannot accept choose (but auto is okay)")
                            
                        hint_str = f" ({', '.join(hints)})" if hints else ""
                        mapping_parts.append(f"${clean_arg}{hint_str}")
                        
                    mapping = ", ".join(mapping_parts)
                    prompt_args_str = (
                        f" Use {mapping}, and $@ for all supplied arguments. "
                        "Optional parameters may be omitted; defaults are supplied by the macro header."
                    )
                else:
                    clean_names = []
                    arg_specs = []
                    prompt_args_str = (
                        " Use $1, $2, ... for parameters and $@ for all of them. "
                        "Alternatively, you can invent your own named parameters by making the VERY FIRST line of your response "
                        "a comment like `# args: target, amount` and then using `$target` and `$amount` in your code. "
                        "Optional named args use `name?` or `[name]`; defaults use `name=default` or `[name=default]`."
                    )

                existing_macros = []
                for m_alias, m_path in macro_aliases.items():
                    if m_alias == sname or m_alias in hidden_commands:
                        continue
                    if os.path.isfile(m_path):
                        try:
                            with open(m_path, "r", encoding="utf-8") as rf:
                                first_line = rf.readline().strip()
                                if first_line.startswith("#"):
                                    desc = first_line.lstrip("#").strip()
                                    prefix = f":solve macro '{m_alias}' -- "
                                    if desc.startswith(prefix):
                                        desc = desc[len(prefix):]
                                    existing_macros.append(f":{m_alias} - {desc}")
                                else:
                                    existing_macros.append(f":{m_alias}")
                        except Exception:
                            pass
                            
                macro_hints_str = ""
                if existing_macros:
                    macro_hints_str = (
                        "You can also invoke existing macros by writing ':<macro_name> [args]'. "
                        "Available macros include:\n" + "\n".join(f"  {m}" for m in existing_macros) + "\n\n"
                    )
                command_hints_str = (
                    "All non-hidden shell commands may be used in generated macros, solves, and games. "
                    "Available command words now are:\n  "
                    + _visible_command_reference()
                    + "\n\n"
                )

                # Context the operator staged with ':memory use' is folded into the
                # solve prompt so the macro reflects what they told it (consumed
                # here so it isn't also replayed to the next model turn).
                staged_ctx = ""
                if pending_memory_tool_result:
                    _ctx = pending_memory_tool_result.strip()
                    if len(_ctx) > 2000:
                        _ctx = _ctx[:2000] + " ...[truncated]"
                    staged_ctx = (
                        "The operator staged this context for you; let it shape the macro:\n"
                        + _ctx + "\n\n"
                    )
                    pending_memory_tool_result = None
                    print(Fore.CYAN + "[Solve] Folding in the memory you staged with :memory use." + Style.RESET_ALL)
                    
                because_ctx = ""
                if command_because:
                    because_ctx = f"The operator provided the following underlying reason/rationale for this macro:\n{command_because}\nMake sure your generated macro commands strongly reflect this rationale.\n\n"
                    print(Fore.CYAN + "[Solve] Passing your 'because' rationale to the model." + Style.RESET_ALL)

                prompt = (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    "You write macros for an interactive cognition shell. A macro is a list of ':' "
                    "commands, one per line." + prompt_args_str + "\n\n"
                    f"{command_hints_str}"
                    "Note: Shell commands natively accept 'auto' or 'choose' as arguments where applicable "
                    "to automatically select or interactively prompt for a value. "
                    "You should seamlessly pass these through to the underlying commands if the user provides them, "
                    "unless a parameter is explicitly restricted from doing so.\n\n"
                    f"{macro_hints_str}"
                    f"{staged_ctx}"
                    f"{because_ctx}"
                    f"Write a macro named '{sname}' that does: {goal}\n"
                    "Output ONLY the command lines, nothing else.<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                )
                print(Fore.CYAN + f"[Solve] Asking the model to write macro ':{sname}' for: {goal}" + Style.RESET_ALL)
                try:
                    sug = generate_agentic_text(model, instruction=prompt, config=config, pre_formatted=True, max_new_tokens=300)
                except Exception as e:
                    print(Fore.RED + f"[Solve] Generation failed: {e}" + Style.RESET_ALL)
                    continue
                cmd_lines = [ln.strip() for ln in (sug or "").splitlines() if ln.strip().startswith(":") or ln.strip().startswith("#")]
                if not any(ln.startswith(":") for ln in cmd_lines):
                    print(Fore.YELLOW + f"[Solve] The model produced no commands. Raw output:\n{(sug or '').strip()[:400]}" + Style.RESET_ALL)
                    continue
                # Guardrail: flag any line whose leading :command isn't a real
                # command or a known macro alias (e.g. the model inventing
                # ':choose'), so a macro that would silently fail later is caught
                # at creation instead of cascading at run time.
                known_cmd = _all_shell_commands() | {sname}
                unknown_cmds = []
                hidden_cmds = []
                for ln in cmd_lines:
                    if not ln.startswith(":"):
                        continue
                    toks = ln[1:].split()
                    w = toks[0].lower() if toks else ""
                    if w and w in hidden_commands:
                        hidden_cmds.append(f":{w}")
                    elif w and w not in known_cmd:
                        unknown_cmds.append(f":{w}")
                if unknown_cmds:
                    print(
                        Fore.YELLOW
                        + f"[Solve] warning: {', '.join(sorted(set(unknown_cmds)))} is not a known command or macro -- "
                        + "this macro may fail when run. (choose/auto are ARGUMENTS to a command, not commands.)"
                        + Style.RESET_ALL
                    )
                if hidden_cmds:
                    print(
                        Fore.YELLOW
                        + f"[Solve] warning: generated macro uses hidden command(s): {', '.join(sorted(set(hidden_cmds)))}. "
                        + "Hidden commands are deliberately excluded from model-authored macros/solves/games unless you accept this anyway."
                        + Style.RESET_ALL
                    )
                dest = os.path.join(ROOT, "invariants", "out", "macros", f"{sname}.txt")
                # Every :solve output is STAGED, not written/aliased, until the
                # operator :accepts it -- so any macro (named or model-chosen) can
                # be reviewed and :rejected first.
                warn_note = " -- see the warning above" if (unknown_cmds or hidden_cmds) else ""
                pending_solve_proposal = {
                    "name": sname, "goal": goal, "clean_names": clean_names,
                    "arg_specs": [spec["header"] for spec in arg_specs],
                    "cmd_lines": cmd_lines, "dest": dest,
                }
                print(Fore.YELLOW + Style.BRIGHT + f"[Solve] Proposed ':{sname}' ({len(cmd_lines)} command(s)) -- NOT adopted yet{warn_note}:" + Style.RESET_ALL)
                for ln in cmd_lines:
                    print(Fore.CYAN + f"  {ln}" + Style.RESET_ALL)
                print(Fore.YELLOW + "         Type :accept to adopt it as :" + sname + ", or :reject to discard." + Style.RESET_ALL)
                continue

            # Direct parameterized-macro invocation: ':<name> args' runs a known
            # macro alias with $1..$9 / $@ substituted from args. Built-ins win, so
            # only a non-built-in alias is dispatched here.
            if user_input.startswith(":") and len(user_input) > 1 and not user_input[1].isspace():
                _itok = user_input[1:].split()
                _iname = _itok[0] if _itok else ""
                if _iname in macro_aliases and _iname not in BUILTIN_COMMANDS:
                    _iargs = _itok[1:]
                    _ilines = load_parameterized_macro(macro_aliases[_iname], _iargs)
                    if _ilines is None:
                        print(Fore.YELLOW + f"[Macro] ':{_iname}' -> {macro_aliases[_iname]} not found on disk." + Style.RESET_ALL)
                    elif _ilines:
                        print(Fore.CYAN + f"[Macro] :{_iname} ({len(_iargs)} arg(s)) -> queued {len(_ilines)} command(s)." + Style.RESET_ALL)
                        input_queue.extend(_ilines)
                    else:
                        print(Fore.YELLOW + f"[Macro] ':{_iname}' has no commands." + Style.RESET_ALL)
                    continue

            first_word = user_input.lower().split()[0] if user_input else ""
            if first_word in ['exit', 'quit', ':exit', ':quit']:
                parts = user_input.strip().split(maxsplit=1)
                reason = parts[1] if len(parts) > 1 else "operator_exit"
                memory.append_event("shell_closed", tags=["session"], provenance={"reason": reason})
                break
            if user_input.startswith(":timestamps"):
                parts = user_input.split()
                if len(parts) > 1 and parts[1] == "on":
                    show_timestamps = True
                    print(Fore.GREEN + "[System] Timestamps enabled." + Style.RESET_ALL)
                elif len(parts) > 1 and parts[1] == "off":
                    show_timestamps = False
                    print(Fore.GREEN + "[System] Timestamps disabled." + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + f"[System] Timestamps are currently {'on' if show_timestamps else 'off'}." + Style.RESET_ALL)
                continue
            if user_input.startswith(":listen"):
                arg = user_input[len(":listen"):].strip().lower()
                if arg == "off":
                    listen_state["drain"] = False
                    print(Fore.YELLOW + "[Listen] mid-thought interjection OFF. Input still queued (never dropped); typing during a reply lands as the next turn." + Style.RESET_ALL)
                elif arg in ("on", ""):
                    listener.start()
                    listen_state["drain"] = True
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + "[Listen] ON. Speak any time -- lines you type mid-reply are ingested at the next "
                        + "chunk seam and appended to its live stream. Whether it redirects or folds them in "
                        + "is the model's own choice; nothing is force-stopped, nothing is dropped."
                        + Style.RESET_ALL
                    )
                elif arg == "status":
                    print(Fore.CYAN + f"[Listen] reader={'active' if listener.active else 'inactive'}, mid-thought drain={'on' if listen_state['drain'] else 'off'}." + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + "[Listen] Usage: :listen on|off|status." + Style.RESET_ALL)
                continue

            if user_input.startswith(":history"):
                print(
                    Fore.YELLOW
                    + "[History] Use :context for current-session transcript controls. Use :memory for long-term memory tools."
                    + Style.RESET_ALL
                )
                continue
            if user_input.startswith(":context"):
                raw_cmd = user_input.strip()
                cmd = raw_cmd.lower()
                if cmd == ":context off":
                    session_context_enabled = False
                    print(Fore.YELLOW + "[Context] Current-session transcript OFF. Long-term memory remains explicit." + Style.RESET_ALL)
                elif cmd == ":context on":
                    session_context_enabled = True
                    print(Fore.GREEN + "[Context] Current-session transcript ON." + Style.RESET_ALL)
                elif cmd == ":context clear":
                    session_context.clear()
                    print(Fore.YELLOW + "[Context] Cleared current-session transcript. Persistent memory was not changed." + Style.RESET_ALL)
                elif cmd.startswith(":context resume"):
                    parts = raw_cmd.split()
                    requested_session = parts[2] if len(parts) >= 3 else "last"
                    resumed_session, recovered = recover_session_context(
                        memory,
                        session_id=requested_session,
                        max_turns=MAX_SESSION_TURNS,
                    )
                    if not recovered:
                        print(Fore.YELLOW + "[Context] No saved session turns found to resume." + Style.RESET_ALL)
                    else:
                        session_context = recovered_to_session_context(recovered)
                        session_context_enabled = True
                        memory.append_event(
                            "context_resumed",
                            tags=["memory_tool", "context"],
                            provenance={
                                "resumed_session_id": resumed_session,
                                "messages": len(recovered),
                            },
                        )
                        print(Fore.CYAN + f"\n--- Resumed Session {resumed_session} ---" + Style.RESET_ALL)
                        for r_role, r_text, r_ts, *_ in recovered:
                            ts_prefix = f"[{r_ts.split('T')[1][:8]}] " if show_timestamps and "T" in r_ts else ""
                            name = "You: " if r_role == "user" else "Me: "
                            color = Fore.MAGENTA + Style.BRIGHT if r_role == "user" else Fore.GREEN + Style.BRIGHT
                            print(color + ts_prefix + name + Style.RESET_ALL + r_text)
                        print(Fore.CYAN + "--------------------------------------\n" + Style.RESET_ALL)
                        print(
                            Fore.GREEN
                            + (
                                f"[Context] Resumed {len(recovered)} saved messages from session "
                                f"{resumed_session}. Current-session transcript is ON."
                            )
                            + Style.RESET_ALL
                        )
                else:
                    print(
                        Fore.CYAN
                        + (
                            f"[Context] enabled={session_context_enabled}, stored_messages={len(session_context)}, "
                            f"max_turns={MAX_SESSION_TURNS}. Use :context resume [last|session_id] to restore a saved shell."
                        )
                        + Style.RESET_ALL
                    )
                continue
            if user_input.startswith(":memory"):
                cmd = user_input.strip()
                tail = cmd[len(":memory"):].strip()
                if tail in ("", "status"):
                    print(Fore.CYAN + format_status(memory.status()) + Style.RESET_ALL)
                elif tail.startswith("recent"):
                    n = parse_count(tail, 4)
                    print(Fore.CYAN + memory.format_recent(max_turns=n) + Style.RESET_ALL)
                elif tail.startswith("search "):
                    query = tail[len("search "):].strip()
                    records = memory.search(query, max_records=6, scope=memory.scope)
                    print(Fore.CYAN + memory.format_tool_result(records) + Style.RESET_ALL)
                elif tail.startswith("use probe"):
                    pname_raw = tail[len("use probe"):].strip()
                    pname = resolve_probe_choice(pname_raw, probes, model=model, config=config, action_name="memory ranking") if pname_raw else None
                    if not pname or pname not in probes:
                        print(Fore.YELLOW + f"[Memory] Unknown probe '{pname_raw}'. Active: {', '.join(sorted(probes)) or 'none'}." + Style.RESET_ALL)
                        continue
                    ranked = rank_memories_by_probe(model, memory, probes[pname]["direction"], top_n=6)
                    if not ranked:
                        print(Fore.YELLOW + "[Memory] No memories to score." + Style.RESET_ALL)
                        continue
                    records = [t[2] for t in ranked]
                    pending_memory_tool_result = memory.format_tool_result(records)
                    memory.append_event(
                        "memory_tool_staged",
                        tags=["memory_tool", "probe"],
                        provenance={"query": f"probe:{pname}", "records": len(records)},
                    )
                    print(Fore.CYAN + f"[Memory] Staged {len(records)} memories where '{pname}' reads furthest from 0:" + Style.RESET_ALL)
                    for _abs, raw, r in ranked:
                        prev = (r.text or "")[:70].replace("\n", " ")
                        print(Fore.CYAN + f"  {raw:+.3f}  [{r.kind}] {prev}..." + Style.RESET_ALL)
                    print(Fore.CYAN + pending_memory_tool_result + Style.RESET_ALL)
                    print(Fore.YELLOW + "[Memory] This tool result will be provided to the next model turn only." + Style.RESET_ALL)
                elif tail.startswith("use "):
                    query = tail[len("use "):].strip()
                    records = memory.search(query, max_records=6, scope=memory.scope)
                    pending_memory_tool_result = memory.format_tool_result(records)
                    memory.append_event(
                        "memory_tool_staged",
                        tags=["memory_tool"],
                        provenance={"query": query, "records": len(records)},
                    )
                    print(Fore.CYAN + pending_memory_tool_result + Style.RESET_ALL)
                    print(Fore.YELLOW + "[Memory] This tool result will be provided to the next model turn only." + Style.RESET_ALL)
                elif tail in ("boundary", "clear"):
                    memory.mark_session_boundary("operator_request")
                    pending_memory_tool_result = None
                    print(Fore.YELLOW + "[Memory] Session boundary marked. Persistent memory file was not deleted." + Style.RESET_ALL)
                elif tail.startswith("choice"):
                    query = tail[len("choice"):].strip()
                    if query.lower().startswith("probe"):
                        pname_raw = query[len("probe"):].strip()
                        pname = resolve_probe_choice(pname_raw, probes, model=model, config=config, action_name="memory ranking") if pname_raw else None
                        if not pname or pname not in probes:
                            print(Fore.YELLOW + f"[Memory] Unknown probe '{pname_raw}'. Active: {', '.join(sorted(probes)) or 'none'}." + Style.RESET_ALL)
                            continue
                        records = [t[2] for t in rank_memories_by_probe(model, memory, probes[pname]["direction"], top_n=15)]
                        choice_note = f" (ranked by probe '{pname}', furthest from 0)"
                    elif query:
                        records = memory.search(query, max_records=15, scope=memory.scope)
                        choice_note = f" (search: {query})"
                    else:
                        records = [r for r in memory.records if (r.text or "").strip()][-15:]
                        choice_note = " (most recent)"
                    if not records:
                        print(Fore.YELLOW + "[Memory] No records found to choose from." + Style.RESET_ALL)
                        continue
                    print(Fore.CYAN + f"[Memory] Select a record to stage for the next turn{choice_note}:" + Style.RESET_ALL)
                    for i, r in enumerate(records, 1):
                        text_preview = (r.text or "")[:80].replace('\n', ' ')
                        print(Fore.CYAN + f"  {i}. [{r.kind}] {text_preview}..." + Style.RESET_ALL)
                    while True:
                        try:
                            ans = input(Fore.GREEN + "Choice (number) or Enter to cancel: " + Style.RESET_ALL).strip()
                            if not ans:
                                break
                            idx = int(ans) - 1
                            if 0 <= idx < len(records):
                                chosen = records[idx]
                                pending_memory_tool_result = memory.format_tool_result([chosen])
                                memory.append_event(
                                    "memory_tool_staged",
                                    tags=["memory_tool"],
                                    provenance={"query": f"interactive_choice:{query}", "records": 1},
                                )
                                print(Fore.CYAN + pending_memory_tool_result + Style.RESET_ALL)
                                print(Fore.YELLOW + "[Memory] This tool result will be provided to the next model turn only." + Style.RESET_ALL)
                                break
                            print(Fore.YELLOW + "Invalid selection." + Style.RESET_ALL)
                        except ValueError:
                            print(Fore.YELLOW + "Please enter a number." + Style.RESET_ALL)
                else:
                    print(
                        Fore.YELLOW
                        + "[Memory] Commands: :memory, :memory recent [n], :memory search <query>, :memory use <query>, :memory use probe <name>, :memory choice [query], :memory choice probe <name>, :memory boundary"
                        + Style.RESET_ALL
                    )
                continue
            if user_input.startswith(":consider"):
                payload = user_input[len(":consider"):].strip()
                try:
                    if "||" not in payload and r"\||" not in payload:
                        raise ValueError("Usage: :consider <trigger_metric> <tool_name> <positive text> || <negative text>")
                    left_side, _, b_text = _partition_unescaped_pipes(payload)
                    left_parts = left_side.strip().split(" ", 2)
                    if len(left_parts) < 3:
                        raise ValueError("Usage: :consider <trigger_metric> <tool_name> <positive text> || <negative text>")
                    t_metric, t_name, a_text = left_parts
                    t_name = re.sub(r"[^a-z0-9_]", "_", t_name.lower())[:40]

                    print(Fore.CYAN + f"[Consider] Minting steer vector for '{t_name}'..." + Style.RESET_ALL)
                    cm = analyze_claim_pair(f"{a_text} || {b_text}", model=model)
                    
                    tuner.register(f"{t_name}_need", 0.05, kind="threshold", comparator=">=")
                    tuner.register(f"{t_name}_alpha", 0.0, kind="coefficient")
                    
                    def make_custom_tool(t_metric=t_metric, t_name=t_name, cm=cm):
                        def custom_detect(text, state=None):
                            phen = (state or {}).get("last_phenomenality") or {}
                            if not phen:
                                return 0.0, 0.0
                            score = float(phen.get(t_metric, 0.0))
                            return score, score
                        
                        def custom_act(act_payload, m):
                            base_alpha = tuner.get(f"{t_name}_alpha", 0.0)
                            alpha = base_alpha * abs(act_payload)
                            sweep_layers = None
                            if alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
                                _cls, _cdrifts, _cvar = _sweep_layers_for(t_name, cm.steer_delta)
                                if _cls:
                                    sweep_layers = _cls
                                    for _cl in _cls:
                                        sweep_state["fires"].append((t_name, _cl, alpha, _cdrifts.get(_cl), len(_cls), _cvar))
                            handles = claimmap_steer_handles(m, cm.steer_delta, alpha=alpha, layers=sweep_layers)
                            memory.append_event(
                                f"{t_name}_state_triggered",
                                text=f"steered with alpha={alpha:.2f} (base {base_alpha:.2f} * {t_metric} {abs(act_payload):.2f})",
                                tags=[f"{t_name}_tool", "activation_trigger"],
                                provenance={"trigger": f"{t_name}_need"}
                            )
                            msg = f"steered the rest of this answer (scaled alpha {alpha:.2f})." if handles else f"steering off (:tune {t_name}_alpha to enable)."
                            print(Fore.MAGENTA + f"\n[{t_name}] metric '{t_metric}' triggered -> {msg}" + Style.RESET_ALL, flush=True)
                            return handles
                        return Tool(f"{t_name}_need", custom_detect, custom_act, comparator=">=")

                    new_tool = make_custom_tool()
                    tool_sense.register(new_tool)
                    
                    print(Fore.GREEN + f"[Consider] Successfully minted active tool '{t_name}' triggered by '{t_metric}'!" + Style.RESET_ALL)
                    print(Fore.CYAN + f"  Threshold tunable registered as: :tune {t_name}_need 0.05" + Style.RESET_ALL)
                    print(Fore.CYAN + f"  Steering strength registered as: :tune {t_name}_alpha 0.05" + Style.RESET_ALL)
                except Exception as exc:
                    print(Fore.RED + f"[Consider] Error: {exc}" + Style.RESET_ALL)
                continue

            if user_input.startswith(":claimmap"):
                payload = user_input[len(":claimmap"):].strip()
                if not payload:
                    print(Fore.YELLOW + "[ClaimMap] Usage: :claimmap <first text> || <second text>" + Style.RESET_ALL)
                    continue
                try:
                    cm = analyze_claim_pair(payload, model=model)
                    pending_claimmap_tool_result = cm.felt            # felt only reaches the model
                    pending_claimmap_steer_delta = cm.steer_delta     # nudges the next generation
                    memory.append_event(
                        "claimmap_tool_staged",
                        text=cm.telemetry,                            # raw numbers logged, never in the prompt
                        tags=["claimmap_tool", "activation_measurement"],
                        provenance={"payload_chars": len(payload), "mean_sim": cm.mean_sim},
                    )
                    print(Fore.CYAN + cm.felt + Style.RESET_ALL)
                    print(Fore.YELLOW + "[ClaimMap] Sensed. This will shape the next model turn only." + Style.RESET_ALL)
                except Exception as exc:
                    print(Fore.RED + f"[ClaimMap] {exc}" + Style.RESET_ALL)
                continue
            if user_input.startswith(":methodmap"):
                query = user_input[len(":methodmap"):].strip()
                if not query:
                    print(Fore.YELLOW + "[MethodMap] Usage: :methodmap <query>" + Style.RESET_ALL)
                    continue
                pending_methodmap_tool_result = format_methodmap_tool_result(memory, query)
                memory.append_event(
                    "methodmap_tool_staged",
                    text=pending_methodmap_tool_result,
                    tags=["methodmap_tool"],
                    provenance={"query": query},
                )
                print(Fore.CYAN + pending_methodmap_tool_result + Style.RESET_ALL)
                print(Fore.YELLOW + "[MethodMap] This tool result will be provided to the next model turn only." + Style.RESET_ALL)
                continue
            if user_input.startswith(":steermap"):
                summary = steer_map.write_summary()
                groups = summary.get("groups", [])
                print(
                    Fore.CYAN
                    + f"[SteerMap] events={summary.get('event_count')} summary={steer_map.summary_path}"
                    + Style.RESET_ALL
                )
                for group in groups[:5]:
                    print(
                        Fore.CYAN
                        + (
                            f"  {group['action']} layer={group['layer_key']} step={group['step_bucket']} "
                            f"n={group['n']} labeled={group['labeled_n']} success_rate={group['success_rate']}"
                        )
                        + Style.RESET_ALL
                    )
                continue
            if user_input.startswith(":doc"):
                dargs = user_input[len(":doc"):].strip()
                if not dargs or dargs.lower() == "status":
                    if not doc_library:
                        print(Fore.YELLOW + "[Doc] No document loaded. Usage: :doc <path> [because <why>]" + Style.RESET_ALL)
                    else:
                        for s in doc_library:
                            unread = s["chunk_count"] - len(s.get("read") or ())
                            print(
                                Fore.CYAN
                                + f"[Doc] {s['source_name']}: {s['chunk_count']} chunks, {unread} unread."
                                + Style.RESET_ALL
                            )
                        staged = "a chunk is staged for the next turn" if pending_document_tool_result else "nothing staged"
                        reading = f"auto-read {doc_autoread['mode']}, {doc_autoread['remaining']} turns left" if doc_autoread else "auto-read off"
                        print(Fore.CYAN + f"[Doc] {staged}; {reading}. ':doc next' | ':doc read [n] [order|reply]' | ':doc stop'" + Style.RESET_ALL)
                elif dargs.lower() == "stop":
                    doc_autoread = None
                    print(Fore.CYAN + "[Doc] Auto-read stopped." + Style.RESET_ALL)
                elif dargs.split()[0].lower() == "read":
                    # exact first-token match: ':doc readings' is a PATH, not auto-read
                    if not doc_library:
                        print(Fore.YELLOW + "[Doc] Nothing to read yet. :doc <path> first." + Style.RESET_ALL)
                    else:
                        count, mode = 1, "order"
                        until_settled, explicit_count = False, False
                        until_probe = None
                        mode_aliases = {"weave": "interleave", "mtime": "updated", "chrono": "updated"}
                        for extra in dargs.split()[1:]:
                            token = extra.strip().lower()
                            if token in {"order", "reply", "interleave", "weave", "updated", "mtime", "chrono"}:
                                mode = mode_aliases.get(token, token)
                            elif token in {"satisfied", "settled", "until"}:
                                until_settled = True
                            elif token.startswith("probe_") or token in probes:
                                until_probe = token.replace("probe_", "")
                                until_settled = False
                            else:
                                try:
                                    count = int(token)
                                    explicit_count = True
                                except ValueError:
                                    pass
                        if (until_settled or until_probe) and not explicit_count:
                            count = MAX_AUTOREAD  # "until satisfied" reads up to the cap
                        count = max(1, min(MAX_AUTOREAD, count))
                        doc_autoread = {"remaining": count, "mode": mode, "until_settled": until_settled, "until_probe": until_probe, "settled_streak": 0}
                        how = {
                            "order": "in document order -- the text advances on its own course",
                            "interleave": "weaving between documents -- their course, not the model's echo",
                            "reply": "chunks chosen by overlap with its replies (echo-following; deliberate use)",
                            "updated": "in the order the files were last written (newest first)",
                        }[mode]
                        stop_note = (
                            f" Stops early once probe '{until_probe}' crosses its threshold."
                            if until_probe else
                            (
                                " Stops early once the reading settles (sense clears conversation_productive "
                                f"{max(1, int(tuner.get('reading_settled_streak', 2)))} turn(s) in a row)."
                                if until_settled else ""
                            )
                        )
                        print(
                            Fore.CYAN
                            + f"[Doc] Auto-read: up to {count} turn(s), {how}.{stop_note} "
                            + "The document speaks next; the model replies each time. ':doc stop' interrupts."
                            + Style.RESET_ALL
                        )
                elif dargs.lower() == "inject":
                    # Whole-library staging for one turn -- bounded and framed,
                    # never an unbounded dump. Refuses honestly when the library
                    # exceeds the budget instead of silently blowing the prompt.
                    if not doc_library:
                        print(Fore.YELLOW + "[Doc] Library is empty." + Style.RESET_ALL)
                        continue
                    inject_budget = max(4000, MAX_PROMPT_CHARS - 8000)  # room for session + reply
                    total_chars = sum(len(c) for doc in doc_library for c in doc.get("chunks", []))
                    if total_chars > inject_budget:
                        print(
                            Fore.YELLOW
                            + f"[Doc] Library is {total_chars} chars; inject budget is {inject_budget}. "
                            + "Read it in turns instead (:doc read [n] [order|interleave])."
                            + Style.RESET_ALL
                        )
                        continue
                    names = ", ".join(doc["source_name"] for doc in doc_library)
                    full_text = [
                        DOCUMENT_TOOL_HEADER,
                        f"I'm reading {len(doc_library)} document(s) in full: {names}. "
                        "They're the files' text, not mine.",
                    ]
                    for doc in doc_library:
                        for index, chunk in enumerate(doc.get("chunks", [])):
                            full_text.append(f"--- {doc['source_name']}, part {index + 1}/{doc['chunk_count']} ---\n{chunk}")
                            if index not in doc.get("read", set()):
                                record_chunk_read(memory, doc, index)
                        doc["read"] = set(range(doc.get("chunk_count", 0)))
                    pending_document_tool_result = "\n\n".join(full_text)
                    print(
                        Fore.CYAN
                        + f"[Doc] {len(doc_library)} document(s) ({total_chars} chars) staged in full for the next turn."
                        + Style.RESET_ALL
                    )
                    continue
                elif dargs.lower() == "next":
                    if doc_session is None:
                        print(Fore.YELLOW + "[Doc] No document loaded." + Style.RESET_ALL)
                    elif doc_session["cursor"] + 1 >= doc_session["chunk_count"]:
                        print(Fore.CYAN + f"[Doc] {doc_session['source_name']} is fully read ({doc_session['chunk_count']} chunks)." + Style.RESET_ALL)
                    else:
                        doc_session["cursor"] += 1
                        doc_session.setdefault("read", set()).add(doc_session["cursor"])
                        record_chunk_read(memory, doc_session, doc_session["cursor"])
                        pending_document_tool_result = stage_chunk(doc_session)
                        print(
                            Fore.CYAN
                            + f"[Doc] Chunk {doc_session['cursor'] + 1}/{doc_session['chunk_count']} staged for the next turn."
                            + Style.RESET_ALL
                        )
                elif (rewrite_req := parse_doc_rewrite_request(dargs)) is not None:
                    rewrite_path = rewrite_req.get("source", "")
                    rewrite_name = rewrite_req.get("output")
                    if rewrite_path == "__current__":
                        if doc_session is None:
                            print(Fore.YELLOW + "[Doc Rewrite] No current document. Usage: :doc <path> rewrite because <why>" + Style.RESET_ALL)
                            continue
                        path_str = doc_session.get("source_path", "")
                    else:
                        path_str = rewrite_path.strip().strip('"')
                    if not path_str or not os.path.isfile(path_str):
                        print(Fore.YELLOW + f"[Doc Rewrite] No file found for '{path_str}'. Usage: :doc <path> rewrite because <why>" + Style.RESET_ALL)
                        continue
                    try:
                        with open(path_str, "r", encoding="utf-8", errors="replace") as rf:
                            original_text = rf.read()
                    except OSError as exc:
                        print(Fore.RED + f"[Doc Rewrite] Could not read file: {exc}" + Style.RESET_ALL)
                        continue
                    if len(original_text) > DOC_REWRITE_MAX_CHARS:
                        print(
                            Fore.YELLOW
                            + f"[Doc Rewrite] {os.path.basename(path_str)} is {len(original_text)} chars; "
                            + f"single-pass rewrite limit is {DOC_REWRITE_MAX_CHARS}. Split it or rewrite a smaller file."
                            + Style.RESET_ALL
                        )
                        continue
                    reason = (command_because or "").strip() or "make it clearer, tighter, better organized, and more useful while preserving meaning"
                    try:
                        out_path = rewrite_output_path(path_str, rewrite_name)
                    except (OSError, ValueError) as exc:
                        print(Fore.YELLOW + f"[Doc Rewrite] {exc}" + Style.RESET_ALL)
                        continue
                    prompt = (
                        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        "Rewrite the file below into a better version, guided by the operator's reason. "
                        "Preserve factual meaning, important details, headings, lists, links, code fences, and code syntax when present. "
                        "Do not explain what you changed. Output ONLY the complete rewritten file content.\n\n"
                        f"Source filename: {os.path.basename(path_str)}\n"
                        f"Output filename: {os.path.basename(out_path)}\n"
                        f"Reason: {reason}\n\n"
                        "----- ORIGINAL FILE START -----\n"
                        f"{original_text}\n"
                        "----- ORIGINAL FILE END -----<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    print(Fore.CYAN + f"[Doc Rewrite] Rewriting {os.path.basename(path_str)} because {reason}..." + Style.RESET_ALL)
                    try:
                        rewritten = generate_agentic_text(
                            model,
                            instruction=prompt,
                            config=config,
                            pre_formatted=True,
                            chatty_log=False,
                            max_new_tokens=max(512, min(4096, int(len(original_text) / 3) + 512)),
                        )
                    except Exception as exc:
                        print(Fore.RED + f"[Doc Rewrite] Generation failed: {exc}" + Style.RESET_ALL)
                        continue
                    rewritten = clean_rewrite_output(rewritten)
                    if not rewritten:
                        print(Fore.YELLOW + "[Doc Rewrite] Model returned no rewritten content; nothing saved." + Style.RESET_ALL)
                        continue
                    if original_text.endswith("\n") and not rewritten.endswith("\n"):
                        rewritten += "\n"
                    try:
                        with open(out_path, "w", encoding="utf-8", newline="") as wf:
                            wf.write(rewritten)
                    except OSError as exc:
                        print(Fore.RED + f"[Doc Rewrite] Could not save rewrite: {exc}" + Style.RESET_ALL)
                        continue
                    memory.append_event(
                        "document_rewritten",
                        text=f"{os.path.basename(path_str)} -> {os.path.basename(out_path)}",
                        tags=["document", "rewrite"],
                        provenance={
                            "source_path": os.path.abspath(path_str),
                            "output_path": os.path.abspath(out_path),
                            "because": reason,
                            "source_chars": len(original_text),
                            "output_chars": len(rewritten),
                        },
                    )
                    print(Fore.GREEN + f"[Doc Rewrite] Saved rewritten file: {out_path}" + Style.RESET_ALL)
                else:
                    path_part, _, why = dargs.partition(" because ")
                    path_str = path_part.strip().strip('"')
                    why = (why.strip() or (command_because or "").strip())
                    
                    paths_to_load = []
                    if os.path.isdir(path_str):
                        # Snatch the whole folder: .md/.txt/.py, deterministic
                        # order, junk dirs pruned so ':doc .' can't swallow
                        # a venv or caches.
                        skip_dirs = {"__pycache__", "node_modules", ".venv", "venv", "out", "data"}
                        for root, dirnames, files in os.walk(path_str):
                            dirnames[:] = sorted(
                                d for d in dirnames if not d.startswith(".") and d not in skip_dirs
                            )
                            for f in sorted(files):
                                if f.endswith((".md", ".txt", ".py")):
                                    paths_to_load.append(os.path.join(root, f))
                    elif "*" in path_str or "?" in path_str:
                        paths_to_load = sorted(p for p in glob.glob(path_str, recursive=True) if os.path.isfile(p))
                    else:
                        # A plain path only counts if it actually exists -- a typo
                        # must become a clean "not found", never a crash downstream.
                        paths_to_load = [path_str] if os.path.isfile(path_str) else []

                    if not paths_to_load:
                        print(Fore.YELLOW + f"[Doc] No file or folder found for '{path_str}'. Check the path (cwd is the repo root)." + Style.RESET_ALL)
                        continue

                    loaded_count = 0
                    for p in paths_to_load:
                        file_why = why.strip()
                        if "{filename}" in file_why:
                            file_why = file_why.replace("{filename}", os.path.basename(p))
                        if "{last_updated}" in file_why:
                            try:
                                mtime = os.path.getmtime(p)
                                dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
                            except OSError:
                                dt = "unknown"
                            file_why = file_why.replace("{last_updated}", dt)
                            
                        try:
                            doc_session = ingest_document(memory, p, why=file_why)
                        except (ValueError, OSError) as exc:
                            print(Fore.RED + f"[Doc] {exc}" + Style.RESET_ALL)
                            continue
                        doc_library.append(doc_session)
                        memory.append_event(
                            "document_ingested",
                            text=f"{doc_session['source_name']} ({doc_session['chunk_count']} chunks)",
                            tags=["document"],
                            provenance={
                                "sha256": doc_session["sha256"],
                                "source_path": doc_session["source_path"],
                                "why": doc_session["why"],
                                "already_ingested": doc_session["already_ingested"],
                            },
                        )
                        known = "already in memory" if doc_session["already_ingested"] else "recorded to memory"
                        progress = f", {len(doc_session['read'])}/{doc_session['chunk_count']} already read" if doc_session["read"] else ""
                        print(Fore.CYAN + f"[Doc] {doc_session['source_name']}: {doc_session['chunk_count']} chunk(s), {known}{progress}." + Style.RESET_ALL)
                        loaded_count += 1

                    if loaded_count > 0:
                        # Resume-aware staging: the first UNREAD chunk of the
                        # first not-fully-read file in this batch. Prior
                        # progress (restored from memory by sha) is respected,
                        # and only what is actually staged gets marked read.
                        staged_any = False
                        for candidate in doc_library[-loaded_count:]:
                            unread = [i for i in range(candidate["chunk_count"]) if i not in candidate.get("read", set())]
                            if not unread:
                                continue
                            doc_session = candidate
                            doc_session["cursor"] = unread[0]
                            doc_session.setdefault("read", set()).add(unread[0])
                            record_chunk_read(memory, doc_session, unread[0])
                            pending_document_tool_result = stage_chunk(doc_session)
                            print(
                                Fore.CYAN
                                + f"[Doc] Loaded {loaded_count} file(s). Staged: {doc_session['source_name']} "
                                + f"part {unread[0] + 1}/{doc_session['chunk_count']}"
                                + (" (resuming)" if unread[0] > 0 else "")
                                + " -- ':doc read [n] [order|interleave|updated]' walks the rest."
                                + Style.RESET_ALL
                            )
                            staged_any = True
                            break
                        if not staged_any:
                            print(
                                Fore.CYAN
                                + f"[Doc] Loaded {loaded_count} file(s) -- all already fully read. "
                                + "Nothing staged; ':doc read' would find nothing new."
                                + Style.RESET_ALL
                            )
                continue
            if user_input.startswith(":impact"):
                trigger = tuner.triggers.get("words_had_impact")
                stats = trigger.stats() if trigger else {}
                rate = stats.get("fire_rate")
                print(
                    Fore.CYAN
                    + f"[Impact] turns where its words caused the context: rate={rate} "
                    + f"(n={stats.get('observed', 0)})"
                    + Style.RESET_ALL
                )
                if stats.get("n_credited"):
                    print(
                        Fore.CYAN
                        + f"[Impact] sense when impacted={stats.get('fired_outcome')} vs not={stats.get('unfired_outcome')} "
                        + f"lift={stats.get('lift')} (n={stats.get('n_credited')}) -- does experienced impact track better deliberation?"
                        + Style.RESET_ALL
                    )
                if not impact_state["log"]:
                    print(Fore.YELLOW + "[Impact] No consequence events yet this session. Real levers: write code (:sandbox on), ask tools, steer a reply-mode reading." + Style.RESET_ALL)
                for event in list(impact_state["log"])[-10:]:
                    print(Fore.CYAN + f"  because it {event['cause']} -> {event['effect']}" + Style.RESET_ALL)
                continue
            if user_input.startswith(":clock"):
                # On-demand readout of the last turn's generation time + memory,
                # plus the current live/reserved footprint and the accrued
                # distributions of both sensed streams. Memory is VRAM on a CUDA
                # box and process RAM on CPU-only, so this reads out either way.
                now_alloc, now_reserved, _now_peak, _now_label = _memory_footprint_gb()
                if now_reserved > 0:
                    _src = "VRAM" if _now_label == "VRAM" else "RAM (CPU-only; no CUDA)"
                    print(Fore.CYAN + f"[Clock] {_src} now: {now_alloc:.2f}GB live / {now_reserved:.2f}GB reserved." + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + "[Clock] memory readings unavailable on this platform." + Style.RESET_ALL)
                if last_clock:
                    lc = last_clock
                    print(
                        Fore.CYAN
                        + f"[Clock] last turn ({lc['ts']}): {lc['generation_seconds']:.1f}s"
                        + (f", {lc['tokens_generated']} tok @ {lc['tokens_per_sec']:.1f} tok/s" if lc['tokens_per_sec'] else "")
                        + f", peak {lc['vram_peak_gb']:.2f}GB {lc.get('mem_label', 'VRAM')}."
                        + Style.RESET_ALL
                    )
                else:
                    print(Fore.YELLOW + "[Clock] no generation timed yet this session." + Style.RESET_ALL)
                for _cs in ("generation_seconds", "vram_gb"):
                    _ct = tuner.triggers.get(_cs)
                    if _ct and _ct.signals:
                        st = _ct.stats()
                        line = f"[Clock] {_cs}: median {st.get('signal_med')} (min {st.get('signal_min')}, max {st.get('signal_max')}, n={st.get('n_signals')})"
                        if st.get("lift") is not None:
                            line += f"; sense lift {st['lift']} (does more cost track better deliberation?)"
                        print(Fore.CYAN + line + Style.RESET_ALL)
                continue
            if user_input.startswith(":prioritize"):
                ranked = rank_probes(probes, tuner)
                if not ranked:
                    print(Fore.YELLOW + "[Prioritize] no active probes to rank. Mint or adopt some first." + Style.RESET_ALL)
                else:
                    alpha = tuner.get("prioritize_alpha", 0.0)
                    print(
                        Fore.CYAN + Style.BRIGHT
                        + f"[Prioritize] probes by evidence-weighted |lift| (steer alpha={round(alpha,4)}"
                        + (" -- OFF)" if alpha <= 0 else " -- steering toward the top each turn):")
                        + Style.RESET_ALL
                    )
                    for i, r in enumerate(ranked[:12]):
                        lift_s = "n/a" if r["lift"] is None else f"{r['lift']:+.3f}"
                        mark = " <- top" if i == 0 and r["priority"] > 0 else ""
                        expo = ", exposed" if r["exposed"] else ""
                        print(Fore.CYAN + f"  {i+1:>2}. {r['name']}: priority {round(r['priority'],4)} (lift {lift_s}, n={r['n']}{expo}){mark}" + Style.RESET_ALL)
                    if alpha <= 0 and ranked[0]["priority"] > 0:
                        print(Fore.YELLOW + f"  Steer toward the top probe with :tune prioritize_alpha <small> (e.g. 0.02)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":release"):
                rargs = user_input[len(":release"):].split()
                tool_names = [t.name for t in tool_sense.tools]
                if not rargs or rargs[0].lower() == "status":
                    active = {k: v for k, v in tool_sense.release_probs.items() if v > 0}
                    print(Fore.CYAN + f"[Release] decoupled decisions this session: {tool_sense.release_total}." + Style.RESET_ALL)
                    if active:
                        for k, v in active.items():
                            print(Fore.CYAN + f"  {k}: {round(v,3)} of fire decisions decoupled from its signal (coin flip)." + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"  none released. Releasable tools: {', '.join(tool_names)}." + Style.RESET_ALL)
                    print(Fore.CYAN + "  Release separates causality: over decoupled turns the trigger and the action decorrelate, so credit lift can tell 'the trigger caused it' from 'acting helped anyway'." + Style.RESET_ALL)
                    continue
                tname = rargs[0]
                if tname not in tool_names:
                    print(Fore.YELLOW + f"[Release] unknown tool '{tname}'.{did_you_mean(tname, tool_names)} Releasable: {', '.join(tool_names)}." + Style.RESET_ALL)
                    continue
                if len(rargs) < 2 or rargs[1].lower() in ("off", "0"):
                    tool_sense.release_probs.pop(tname, None)
                    print(Fore.CYAN + f"[Release] {tname} re-coupled (fires strictly on its signal again)." + Style.RESET_ALL)
                    memory.append_event("dependency_recoupled", tags=["release", "causality"], provenance={"tool": tname})
                    continue
                try:
                    prob = min(max(float(rargs[1]), 0.0), 1.0)
                except ValueError:
                    print(Fore.YELLOW + "[Release] probability must be a number in [0, 1]." + Style.RESET_ALL)
                    continue
                tool_sense.release_probs[tname] = prob
                memory.append_event("dependency_released", tags=["release", "causality"], provenance={"tool": tname, "prob": prob})
                print(
                    Fore.MAGENTA + Style.BRIGHT
                    + f"[Release] {tname} released at {round(prob,3)}: that fraction of its fire decisions is now a coin flip, "
                    + "decoupled from the signal -- so its causal lift can be measured, not assumed. :release {tname} off to re-couple.".replace("{tname}", tname)
                    + Style.RESET_ALL
                )
                continue
            if user_input.startswith(":experts"):
                earg = user_input[len(":experts"):].strip().lower()
                if earg == "on":
                    config.emergent_experts_enabled = True
                    print(
                        Fore.YELLOW
                        + "[Experts] ON: recurring self-correction directions can now be minted into "
                        + "new roster experts (mesa Committee, roster bounded at 6). Its own corrective "
                        + "conclusions become steering directions -- bounded by the envelope like everything else."
                        + Style.RESET_ALL
                    )
                elif earg == "off":
                    config.emergent_experts_enabled = False
                    print(Fore.CYAN + "[Experts] OFF: roster frozen at its current members." + Style.RESET_ALL)
                else:
                    roster = getattr(config, "committee", None)
                    names = [m.name for m in roster.members] if roster else ["(seeded on first generation)"]
                    state_txt = "ON" if getattr(config, "emergent_experts_enabled", False) else "OFF"
                    print(Fore.CYAN + f"[Experts] {state_txt}; roster: {', '.join(names)} (:experts on|off)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":sandbox"):
                sarg = user_input[len(":sandbox"):].strip().lower()
                if sarg == "on":
                    sandbox_enabled = True
                    print(
                        Fore.YELLOW
                        + "[Sandbox] ON: fenced ```python blocks in the model's replies now RUN "
                        + f"(isolated interpreter, {SANDBOX_TIMEOUT_SEC:g}s timeout, cwd=invariants/out/sandbox). "
                        + "Process isolation, not a hard security boundary -- watch what runs."
                        + Style.RESET_ALL
                    )
                elif sarg == "off":
                    sandbox_enabled = False
                    print(Fore.CYAN + "[Sandbox] OFF." + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + f"[Sandbox] {'ON' if sandbox_enabled else 'OFF'} (:sandbox on|off)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":refresh"):
                # Re-read referenced files from disk WITHOUT restarting -- the
                # session and the loaded model stay live. (Code edits to this
                # script still need a full restart; only data files refresh here.)
                print(Fore.CYAN + "[System] Refreshing referenced files from disk (session + model stay live)..." + Style.RESET_ALL)
                probes.clear()
                try:
                    if os.path.isdir(PROBE_DIR):
                        for pf in os.listdir(PROBE_DIR):
                            if pf.endswith(".pt"):
                                pname = pf[:-3]
                                pdata = torch.load(os.path.join(PROBE_DIR, pf), weights_only=True)
                                probes[pname] = {"direction": pdata["direction"], "history": deque(maxlen=40), "framings": pdata.get("framings", ("", "")), "exposed": bool(pdata.get("exposed", False))}
                                # Register a trigger for a probe that appeared on
                                # disk since startup; leave existing thresholds and
                                # accrued outcomes untouched.
                                if f"probe_{pname}" not in tuner.triggers:
                                    tuner.register(f"probe_{pname}", 0.0, kind="threshold", comparator=">=")
                        print(Fore.GREEN + f"[System] Reloaded {len(probes)} probe(s) from {PROBE_DIR}." + Style.RESET_ALL)
                except Exception as e:
                    print(Fore.RED + f"[System] Error refreshing probes: {e}" + Style.RESET_ALL)
                try:
                    with open(MACRO_ALIAS_PATH, "r", encoding="utf-8") as _af:
                        _loaded = {str(k): str(v) for k, v in json.load(_af).items()}
                    macro_aliases.clear()
                    macro_aliases.update(_loaded)
                    print(Fore.GREEN + f"[System] Reloaded {len(macro_aliases)} macro alias(es)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                try:
                    with open(HIDDEN_CMD_PATH, "r", encoding="utf-8") as _hcf:
                        _raw_hidden = json.load(_hcf)
                    if isinstance(_raw_hidden, list):
                        _loaded_hidden = {str(x).lstrip(":").lower() for x in _raw_hidden if str(x).strip()}
                    elif isinstance(_raw_hidden, dict):
                        _loaded_hidden = {str(k).lstrip(":").lower() for k, v in _raw_hidden.items() if v}
                    else:
                        _loaded_hidden = set()
                    hidden_commands.clear()
                    hidden_commands.update(_loaded_hidden)
                    print(Fore.GREEN + f"[System] Reloaded {len(hidden_commands)} hidden command(s)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                try:
                    with open(EXPOSED_CMD_PATH, "r", encoding="utf-8") as _ecf:
                        _loaded_exposed = {str(k): str(v) for k, v in json.load(_ecf).items()}
                    exposed_commands.clear()
                    exposed_commands.update({k: v for k, v in _loaded_exposed.items() if k not in hidden_commands})
                    print(Fore.GREEN + f"[System] Reloaded {len(exposed_commands)} exposed command(s)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                try:
                    with open(EXPOSED_KNOB_PATH, "r", encoding="utf-8") as _ekf:
                        _raw_knobs = json.load(_ekf)
                    if isinstance(_raw_knobs, list):
                        _loaded_knobs = {str(x).strip() for x in _raw_knobs if str(x).strip()}
                    elif isinstance(_raw_knobs, dict):
                        _loaded_knobs = {str(k).strip() for k, v in _raw_knobs.items() if v}
                    else:
                        _loaded_knobs = set()
                    exposed_knobs.clear()
                    exposed_knobs.update(k for k in _loaded_knobs if k in tuner.triggers and not _is_shadow_trigger(k, tuner))
                    print(Fore.GREEN + f"[System] Reloaded {len(exposed_knobs)} exposed knob(s)." + Style.RESET_ALL)
                except (OSError, ValueError):
                    pass
                print(Fore.YELLOW + "[System] (Macro files are re-read on every :run, so edits to them need no refresh.)" + Style.RESET_ALL)
                continue

            _cmdword = user_input.strip().split()[0].lower() if user_input.strip() else ""
            if _cmdword == ":help":
                harg = user_input.strip()[len(":help"):].strip().lower()
                if harg.startswith("expose"):
                    help_exposed = not harg.endswith("off")
                    print(Fore.GREEN + f"[Help] Model help {'EXPOSED -- it may now emit <<HELP>>' if help_exposed else 'hidden from the model'}." + Style.RESET_ALL)
                    continue
                if harg == "model":
                    print(Fore.CYAN + build_model_help_text(list_solve_macros(), exposed_commands, exposed_knobs, hidden_commands) + Style.RESET_ALL)
                    continue
                if harg:
                    q = harg.split()[0].lstrip(":")
                    matches = find_command_help_entries(harg)
                    if matches:
                        for entry in matches:
                            for _l in entry:
                                print(Fore.CYAN + _l + Style.RESET_ALL)
                    elif q in macro_aliases:
                        # Generate dynamic help for macros
                        mpath = macro_aliases[q]
                        exists = os.path.isfile(mpath)
                        status = "" if exists else " (FILE MISSING)"
                        print(Fore.CYAN + f":{q} -- Macro aliased to {mpath}{status}" + Style.RESET_ALL)
                        
                        # Check if it's a solve-macro with a known goal
                        for _n, _d, _a in list_solve_macros():
                            if _n == q:
                                _argnote = f" [args: {_a}]" if _a else ""
                                print(Fore.CYAN + f"  Goal: {_d}{_argnote}" + Style.RESET_ALL)
                                break
                                
                        # Show the first few lines of the macro
                        if exists:
                            print(Fore.CYAN + "  Contents:" + Style.RESET_ALL)
                            try:
                                with open(mpath, "r", encoding="utf-8") as rf:
                                    lines = rf.read().splitlines()
                                    cmds = [line for line in lines if line.strip() and not line.strip().startswith("#")]
                                    for line in cmds[:5]:
                                        print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                                    if len(cmds) > 5:
                                        print(Fore.CYAN + f"    ... and {len(cmds) - 5} more." + Style.RESET_ALL)
                            except Exception as e:
                                print(Fore.RED + f"    (Could not read macro file: {e})" + Style.RESET_ALL)
                        else:
                            print(Fore.RED + f"  (The file {mpath} does not exist or was deleted!)" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Help] No help entry for '{q}'.{did_you_mean(q, BUILTIN_COMMANDS | set(macro_aliases))}" + Style.RESET_ALL)
                    continue
                for _l in COMMAND_HELP_LINES:
                    print(Fore.CYAN + _l + Style.RESET_ALL)
                sm = list_solve_macros()
                if sm:
                    print(Fore.CYAN + "Solve-macros (:<name> <args>):" + Style.RESET_ALL)
                    for _n, _d, _a in sm:
                        _argnote = f"  [args: {_a}]" if _a else ""
                        print(Fore.CYAN + f"          :{_n}{(' -- ' + _d) if _d else ''}{_argnote}" + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + "(no solve-macros yet -- make one with :solve choose <goal>)" + Style.RESET_ALL)
                print(Fore.CYAN + "':help model' shows what the MODEL can run itself; ':help expose' lets it call <<HELP>>." + Style.RESET_ALL)
                continue
            if _cmdword in (":accept", ":reject"):
                parts = user_input.strip().split()
                verb = parts[0].lower()
                sel = parts[1].lower() if len(parts) > 1 else "all"
                # A staged :solve choose/auto macro takes precedence: adopt (write +
                # alias) or discard it before touching any staged prize commands.
                if pending_solve_proposal is not None:
                    p = pending_solve_proposal
                    pending_solve_proposal = None
                    if verb == ":reject":
                        print(Fore.YELLOW + f"[Reject] Discarded proposed macro ':{p['name']}'." + Style.RESET_ALL)
                        continue
                    if _hidden_overwrite_blocked(p["name"], "Accept"):
                        continue
                    try:
                        os.makedirs(os.path.dirname(p["dest"]), exist_ok=True)
                        with open(p["dest"], "w", encoding="utf-8") as wf:
                            wf.write(f"# :solve macro '{p['name']}' -- {p['goal']}\n")
                            arg_header = p.get("arg_specs") or p.get("clean_names") or []
                            if arg_header:
                                wf.write(f"# args: {', '.join(arg_header)}\n")
                            for ln in p["cmd_lines"]:
                                wf.write(ln + "\n")
                        macro_aliases[p["name"]] = p["dest"]
                        _save_macro_aliases()
                        print(Fore.GREEN + f"[Accept] Adopted ':{p['name']}' ({len(p['cmd_lines'])} command(s)). Run it with :{p['name']} <args>." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Accept] Could not save macro: {e}" + Style.RESET_ALL)
                    continue
                if not pending_accept_commands:
                    print(Fore.CYAN + "[Accept] Nothing is staged for acceptance." + Style.RESET_ALL)
                    continue
                if sel == "all":
                    picks = list(range(len(pending_accept_commands)))
                else:
                    try:
                        i = int(sel) - 1
                        if not (0 <= i < len(pending_accept_commands)):
                            raise IndexError
                        picks = [i]
                    except (ValueError, IndexError):
                        print(Fore.YELLOW + f"[{verb[1:].capitalize()}] No staged command #{sel}. {len(pending_accept_commands)} staged -- use :{verb[1:]} <n> or :{verb[1:]} all." + Style.RESET_ALL)
                        continue
                chosen = [pending_accept_commands[i] for i in picks]
                for i in sorted(picks, reverse=True):
                    del pending_accept_commands[i]
                if verb == ":accept":
                    input_queue.extend(chosen)
                    print(Fore.GREEN + Style.BRIGHT + f"[Accept] Adopted {len(chosen)} command(s) into the queue: {'; '.join(chosen)}" + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + f"[Reject] Dropped {len(chosen)} staged command(s): {'; '.join(chosen)}" + Style.RESET_ALL)
                if pending_accept_commands:
                    print(Fore.CYAN + f"[Accept] {len(pending_accept_commands)} command(s) still staged." + Style.RESET_ALL)
                continue

            if _cmdword == ":hide":
                hargs = user_input.strip()[len(":hide"):].split()
                if not hargs:
                    if hidden_commands:
                        print(Fore.CYAN + "[Hide] Hidden from the model: " + ", ".join(f":{w}" for w in sorted(hidden_commands)) + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + "[Hide] No commands are hidden. Use ':hide <command>' to hide one from model-facing help/discovery, or ':hide <command> off' to reveal it." + Style.RESET_ALL)
                    continue
                word = hargs[0].lstrip(":").lower()
                mode_arg = hargs[1].lower() if len(hargs) > 1 else ""
                all_shell_commands = _all_shell_commands()
                if mode_arg in ("off", "show", "reveal", "visible"):
                    if word in hidden_commands:
                        hidden_commands.remove(word)
                        _save_hidden_commands()
                        print(Fore.GREEN + f"[Hide] ':{word}' is visible to model-facing help/discovery again." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Hide] ':{word}' wasn't hidden." + Style.RESET_ALL)
                    continue
                if word not in all_shell_commands:
                    print(Fore.YELLOW + f"[Hide] ':{word}' isn't a command.{did_you_mean(word, all_shell_commands)}" + Style.RESET_ALL)
                    continue
                if mode_arg and mode_arg not in ("on", "hide", "hidden"):
                    print(Fore.YELLOW + f"[Hide] Unknown mode '{mode_arg}'. Use ':hide <command>' or ':hide <command> off'." + Style.RESET_ALL)
                    continue
                hidden_commands.add(word)
                was_exposed = exposed_commands.pop(word, None) is not None
                _save_hidden_commands()
                if was_exposed:
                    _save_exposed_commands()
                extra = " and unexposed" if was_exposed else ""
                print(Fore.GREEN + f"[Hide] ':{word}' is hidden from model-facing help/discovery{extra}. It still works for the operator." + Style.RESET_ALL)
                continue

            if _cmdword == ":expose":
                eargs = user_input.strip()[len(":expose"):].split()
                if not eargs:  # list exposures
                    exposed_probe_names = sorted(n for n in probes if probes[n].get("exposed"))
                    if not exposed_commands and not exposed_probe_names and not exposed_knobs:
                        print(Fore.CYAN + "[Expose] Nothing exposed. Use ':expose :command [stage|direct]' for command tools, or ':expose <probe|knob>' for model-readable probes/knobs." + Style.RESET_ALL)
                    if exposed_commands:
                        for w, mode in sorted(exposed_commands.items()):
                            print(Fore.CYAN + f"[Expose] command :{w}  ({mode})" + Style.RESET_ALL)
                    if exposed_probe_names:
                        print(Fore.CYAN + "[Expose] probes: " + ", ".join(exposed_probe_names) + Style.RESET_ALL)
                    if exposed_knobs:
                        print(Fore.CYAN + "[Expose] knobs: " + ", ".join(sorted(exposed_knobs)) + Style.RESET_ALL)
                    continue
                target = eargs[0]
                mode_arg = eargs[1].lower() if len(eargs) > 1 else ""
                if not target.startswith(":"):
                    if mode_arg not in ("", "on", "expose", "off", "hide"):
                        print(Fore.YELLOW + f"[Expose] Bare targets expose probes/knobs and only accept 'off'. For command tools use ':expose :{target} [stage|direct]'." + Style.RESET_ALL)
                        continue
                    turn_off = mode_arg in ("off", "hide")
                    probe_name = re.sub(r"[^a-z0-9_]", "_", target.lower())[:40].strip("_")
                    if probe_name in probes:
                        probes[probe_name]["exposed"] = not turn_off
                        pf = os.path.join(ROOT, "invariants", "out", "probes", f"{probe_name}.pt")
                        if os.path.exists(pf):
                            try:
                                data = torch.load(pf, weights_only=True)
                                data["exposed"] = probes[probe_name]["exposed"]
                                torch.save(data, pf)
                            except Exception:
                                pass
                        print(Fore.GREEN + f"[Expose] probe '{probe_name}' is now {'hidden from' if turn_off else 'exposed to'} the model's <<PROBE: {probe_name}>> tool." + Style.RESET_ALL)
                        continue
                    knob_name = target.strip()
                    if knob_name not in tuner.triggers and knob_name.lower() in tuner.triggers:
                        knob_name = knob_name.lower()
                    if knob_name in tuner.triggers and not _is_shadow_trigger(knob_name, tuner):
                        if turn_off:
                            exposed_knobs.discard(knob_name)
                        else:
                            exposed_knobs.add(knob_name)
                        _save_exposed_knobs()
                        print(Fore.GREEN + f"[Expose] knob '{knob_name}' is now {'hidden from' if turn_off else 'exposed to'} the model's <<PROBE: {knob_name}>> tool." + Style.RESET_ALL)
                        continue
                    candidates = set(probes) | {n for n in tuner.triggers if not _is_shadow_trigger(n, tuner)}
                    command_hint = f" For command tools use ':expose :{target} [stage|direct]'." if target.lower() in _all_shell_commands() else ""
                    print(Fore.YELLOW + f"[Expose] '{target}' is not an active probe or knob.{did_you_mean(target, candidates)}{command_hint}" + Style.RESET_ALL)
                    continue
                word = target.lstrip(":").lower()
                if mode_arg == "off":
                    if exposed_commands.pop(word, None) is not None:
                        _save_exposed_commands()
                        print(Fore.CYAN + f"[Expose] ':{word}' is no longer a model tool." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Expose] ':{word}' wasn't exposed." + Style.RESET_ALL)
                    continue
                if word == "expose":
                    print(Fore.RED + "[Expose] refusing to expose ':expose' itself -- the model could then self-grant any command." + Style.RESET_ALL)
                    continue
                if word in hidden_commands:
                    print(Fore.YELLOW + f"[Expose] ':{word}' is hidden from model-facing discovery. Reveal it first with ':hide :{word} off'." + Style.RESET_ALL)
                    continue
                known_exposable = _known_model_commands()
                if word not in known_exposable:
                    print(Fore.YELLOW + f"[Expose] ':{word}' isn't a command.{did_you_mean(word, known_exposable)}" + Style.RESET_ALL)
                    continue
                if mode_arg in ("", "stage", "staged", "accept"):
                    mode = "stage"
                elif mode_arg in ("direct", "run", "trust", "free"):
                    mode = "direct"
                else:
                    print(Fore.YELLOW + f"[Expose] Unknown mode '{mode_arg}'. Use stage, direct, or off." + Style.RESET_ALL)
                    continue
                exposed_commands[word] = mode
                _save_exposed_commands()
                run_note = "is queued immediately" if mode == "direct" else "is staged for your :accept"
                print(Fore.GREEN + f"[Expose] ':{word}' is now a model tool -- it calls <<CMD: :{word} ...>> and it {run_note} ({mode})." + Style.RESET_ALL)
                if word in EXPOSE_META_CMDS:
                    print(Fore.YELLOW + f"         WARNING: ':{word}' takes another command/macro/script as its argument, so through it the model can reach commands you did NOT expose. Expose it only if you mean to." + Style.RESET_ALL)
                elif word in macro_aliases and word not in BUILTIN_COMMANDS:
                    print(Fore.YELLOW + f"         WARNING: ':{word}' is a macro command; it expands to whatever commands its macro file contains." + Style.RESET_ALL)
                continue

            if user_input.startswith(":game ") or user_input.startswith(":game,") or user_input.strip() == ":game":
                body = user_input[len(":game"):].strip()
                if body.startswith(","):
                    body = body[1:].strip()
                if not body:
                    print(Fore.YELLOW + "[Game] Usage: :game <name> [ +/-<rule> <desc> | +/-win <cond> | +/-loss <cond> | +/-prize <opt>,<odds> [<count>,<odds_val>] <cmd> | draw | start | end ]" + Style.RESET_ALL)
                    continue
                _nm = re.split(r"[\s,]+", body, maxsplit=1)
                gname = _nm[0].strip()
                rest = _nm[1].strip().lstrip(",").strip() if len(_nm) > 1 else ""
                if gname.lower() == "restore":
                    if not rest:
                        print(Fore.YELLOW + "[Game] Usage: :game restore {\"name\":\"...\",\"config\":{...}}" + Style.RESET_ALL)
                        continue
                    try:
                        obj = json.loads(rest)
                        rname = re.sub(r"[^a-z0-9_]", "_", str(obj.get("name", "")).lower())[:40].strip("_")
                        if not rname:
                            raise ValueError("missing game name")
                        cfg = normalize_game_config(obj.get("config", {}))
                        save_game_config(os.path.join(ROOT, "games", f"{rname}_rules.json"), cfg)
                        n_rules = len(cfg.get("rules", {}))
                        n_prizes = len(cfg.get("prizes", {}))
                        print(Fore.GREEN + f"[Game] Restored config for '{rname}' ({n_rules} rule(s), {n_prizes} prize(s))." + Style.RESET_ALL)
                    except Exception as e:
                        print(Fore.RED + f"[Game] Could not restore game config: {e}" + Style.RESET_ALL)
                    continue
                # Operator declines the game the model just proposed (the symmetric
                # counterpart to accepting it with ':game <name>').
                if gname.lower() in ("no", "decline", "reject", "dismiss") and model_proposed_game:
                    declined = model_proposed_game
                    model_proposed_game = None
                    pending_game_tool_result = None
                    print(Fore.YELLOW + f"[Game] Declined the model's proposal to play '{declined}'." + Style.RESET_ALL)
                    memory.append_event("game_proposal_declined", tags=["game", "self_concept"], provenance={"game": declined})
                    continue

                # :game auto [+trait ...] [-trait ...] -- design a conversational
                # game whose whole goal is to increase the named trait(s)/probe(s),
                # then PROPOSE it (operator accepts with ':game <name>'/':game start',
                # declines with ':game no'). e.g. ':game auto +consciousness'.
                if gname.lower() == "auto":
                    _atoks = rest.split()
                    plus = [t.lstrip("+") for t in _atoks if t.startswith("+")]
                    minus = [t.lstrip("-") for t in _atoks if t.startswith("-")]
                    bare = [t for t in _atoks if not t.startswith(("+", "-"))]
                    focus = ", ".join(plus) if plus else (bare[0] if bare else "the model's cognition")
                    base = plus[0] if plus else (bare[0] if bare else "trait")
                    auto_name = re.sub(r"[^a-z0-9_]", "_", f"improve_{base}".lower())[:40].strip("_") or "improve_trait"
                    _n, _i = auto_name, 2
                    while os.path.isfile(os.path.join(ROOT, "games", f"{_n}_rules.json")) or os.path.isfile(os.path.join(ROOT, "games", f"{_n}.py")):
                        _n, _i = f"{auto_name}_{_i}", _i + 1
                    auto_name = _n
                    auto_rules_path = os.path.join(ROOT, "games", f"{auto_name}_rules.json")
                    gcfg2 = load_game_config(auto_rules_path)
                    gcfg2["win"] = f"{focus} is measurably higher than when the game began (its probe reads further from 0 / it is more present in the replies)."
                    gcfg2["loss"] = f"{focus} stalls or declines over several turns."
                    gcfg2["rules"]["goal"] = f"Each turn, make one deliberate move to increase {focus}; name the move and why it should raise {focus}."
                    if minus:
                        gcfg2["rules"]["avoid"] = f"reduce {', '.join(minus)} while doing so."
                    try:
                        premise = generate_agentic_text(
                            model,
                            instruction=f"In one vivid sentence, describe a short cooperative game whose whole point is for you to increase your own {focus}. Reply with ONLY that sentence.",
                            config=config, max_new_tokens=60, chatty_log=False, pre_formatted=False,
                        )
                        if premise and premise.strip():
                            gcfg2["rules"]["premise"] = premise.strip().splitlines()[0][:300]
                    except Exception:
                        pass
                    save_game_config(auto_rules_path, gcfg2)
                    model_proposed_game = auto_name
                    print(Fore.GREEN + Style.BRIGHT + f"[Game] Auto-designed '{auto_name}' to improve {focus}." + Style.RESET_ALL)
                    if gcfg2["rules"].get("premise"):
                        print(Fore.CYAN + f"       premise: {gcfg2['rules']['premise']}" + Style.RESET_ALL)
                    print(Fore.CYAN + f"       win: {gcfg2['win']}" + Style.RESET_ALL)
                    print(Fore.CYAN + f"       Accept with ':game {auto_name}' (or ':game start'); decline with ':game no'." + Style.RESET_ALL)
                    continue

                gpath = os.path.join(ROOT, "games", f"{gname}.py")
                rules_path = os.path.join(ROOT, "games", f"{gname}_rules.json")
                gcfg = load_game_config(rules_path)

                first = rest.split()[0] if rest else ""
                sign = "+" if first.startswith("+") else ("-" if first.startswith("-") else None)
                key = (first[1:] if sign else first).lower()
                arg = rest[len(first):].strip() if first else ""

                # --- prizes: +prize <opt>,<odds> [<count>,<odds_val>] <cmd> / -prize <opt> ---
                if key == "prize" and sign:
                    if sign == "+":
                        option, spec = parse_prize_spec(arg)
                        if option is None:
                            print(Fore.YELLOW + f"[Game] Bad prize ({spec}). Usage: :game {gname} +prize <opt>,<odds> [<count>,<odds_val>] <cmd>" + Style.RESET_ALL)
                            continue
                        gcfg["prizes"][option] = spec
                        save_game_config(rules_path, gcfg)
                        cnt = "unlimited" if spec["count"] is None else spec["count"]
                        print(Fore.GREEN + f"[Game] Prize '{option}' on {gname}: odds {spec['odds']:g}, count {cnt}" + (f", wins run '{spec['command']}'" if spec["command"] else "") + "." + Style.RESET_ALL)
                    else:
                        opt = (arg.split(",")[0].strip() if arg else "")
                        if gcfg["prizes"].pop(opt, None) is not None:
                            save_game_config(rules_path, gcfg)
                            print(Fore.YELLOW + f"[Game] Removed prize '{opt}' from {gname}." + Style.RESET_ALL)
                        else:
                            print(Fore.YELLOW + f"[Game] No prize '{opt}' on {gname}." + Style.RESET_ALL)
                    continue

                # --- win / loss conditions: +win <cond> / -win, +loss <cond> / -loss ---
                if key in ("win", "loss") and sign:
                    gcfg[key] = arg if sign == "+" else ""
                    save_game_config(rules_path, gcfg)
                    if sign == "+":
                        print(Fore.GREEN + f"[Game] {gname} {key} condition: {arg or '(empty)'}. (the {gname}.py/model evaluates it)" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Game] Cleared {gname} {key} condition." + Style.RESET_ALL)
                    continue

                if sign and key not in GAME_RESERVED:
                    rule_name = first[1:]
                    if sign == "+":
                        new_desc = arg.lstrip(",").strip()
                        if rule_name in gcfg["rules"]:
                            gcfg["rules"][rule_name] += " " + new_desc
                            print(Fore.GREEN + f"[Game] Appended to rule '{rule_name}' in {gname}: '{gcfg['rules'][rule_name]}'." + Style.RESET_ALL)
                        else:
                            gcfg["rules"][rule_name] = new_desc
                            print(Fore.GREEN + f"[Game] Added rule '{rule_name}' to {gname}: '{gcfg['rules'][rule_name]}'." + Style.RESET_ALL)
                    else:
                        gcfg["rules"].pop(rule_name, None)
                        print(Fore.YELLOW + f"[Game] Removed rule '{rule_name}' from {gname}." + Style.RESET_ALL)
                    save_game_config(rules_path, gcfg)
                    print(Fore.CYAN + f"[Game] (the {gname}.py script or the model enforces this.)" + Style.RESET_ALL)
                    continue

                # --- draw prizes now: run each against its odds ---
                if key == "draw":
                    won = draw_prizes(gcfg["prizes"])
                    save_game_config(rules_path, gcfg)
                    if not won:
                        print(Fore.CYAN + f"[Game] Draw: nothing hit (of {len(gcfg['prizes'])} prize(s))." + Style.RESET_ALL)
                    for opt, cmd in won:
                        print(Fore.MAGENTA + Style.BRIGHT + f"[Game] Prize won: {opt}!" + Style.RESET_ALL)
                        if cmd:
                            _stage_for_accept(cmd, why=f"prize '{opt}'")
                    continue

                # --- end / stop / quit: award prizes, then end the active game ---
                if key in ("end", "stop", "quit") or gname.lower() in ("end", "stop", "quit"):
                    if active_game:
                        for opt, cmd in draw_prizes(gcfg["prizes"]):
                            print(Fore.MAGENTA + Style.BRIGHT + f"[Game] End prize: {opt}!" + Style.RESET_ALL)
                            if cmd:
                                _stage_for_accept(cmd, why=f"end prize '{opt}'")
                        save_game_config(rules_path, gcfg)
                        print(Fore.CYAN + f"[Game] Ended active game: {active_game}." + Style.RESET_ALL)
                        active_game = None
                        pending_game_tool_result = None
                    else:
                        print(Fore.YELLOW + "[Game] No game is currently active." + Style.RESET_ALL)
                    continue

                # --- :game start (bare): accept & start the game the model just
                # proposed, without retyping its name ---
                if gname.lower() == "start" and not rest:
                    if not model_proposed_game:
                        print(Fore.YELLOW + "[Game] Nothing proposed to start. Propose one with :game <name>, or force a script with :game <name> start." + Style.RESET_ALL)
                        continue
                    gname = model_proposed_game
                    gpath = os.path.join(ROOT, "games", f"{gname}.py")
                    rules_path = os.path.join(ROOT, "games", f"{gname}_rules.json")
                    gcfg = load_game_config(rules_path)
                    key = "start"

                force_start = (key == "start")
                has_script = os.path.isfile(gpath)
                accepting = force_start or (model_proposed_game == gname)

                if not has_script and not accepting:
                    # No script and no pending proposal -> ask the model to DESIGN
                    # the game (form 3), then fall through to its turn.
                    print(Fore.CYAN + f"[Game] No script for '{gname}' -- asking the model to design it..." + Style.RESET_ALL)
                    cond = ""
                    if gcfg["rules"] or gcfg["win"] or gcfg["loss"] or gcfg["prizes"]:
                        cond = (f" Honor the existing config -- rules={list(gcfg['rules'])}, "
                                f"win='{gcfg['win']}', loss='{gcfg['loss']}', prizes={list(gcfg['prizes'])}.")
                    user_input = (
                        f"Design a short, playable game called '{gname}': clear rules, options, and "
                        f"win/loss conditions (probe-based if apt).{cond} Then propose it to me to play."
                    )
                    # Fall through (no continue) so this reaches the model's turn.
                elif not accepting:
                    # A real script exists but nothing is pending -> propose it.
                    print(Fore.CYAN + f"[Game] Proposing {gname} to the model... (waiting for acceptance)" + Style.RESET_ALL)
                    user_proposed_game = gname
                    pending_game_tool_result = f"[Game Proposal] The operator proposes playing '{gname}'. To accept, output <<GAME_ACCEPT: {gname}>>; to decline, output <<GAME_DECLINE: {gname}>>. Either way, also reply in words."
                    user_input = "(system: operator proposed a game)"
                    continue
                else:
                    # Accepting (force_start, or the model's proposal) -> START.
                    print(Fore.GREEN + (f"[Game] Forcibly starting {gname}!" if force_start else f"[Game] You accepted the model's proposal to play {gname}!") + Style.RESET_ALL)
                    model_proposed_game = None
                    active_game = gname
                    if gcfg["rules"]:
                        print(Fore.CYAN + f"Active rules: {', '.join(gcfg['rules'])}." + Style.RESET_ALL)
                    if gcfg["prizes"]:
                        print(Fore.CYAN + f"Prizes: {', '.join(gcfg['prizes'])}." + Style.RESET_ALL)
                    if has_script:
                        print(Fore.MAGENTA + f"[Game] Initializing {gname}..." + Style.RESET_ALL)
                        try:
                            with open(gpath, "r", encoding="utf-8") as _gf:
                                gcode = _gf.read()
                            exec_globals = globals().copy()
                            exec_globals["GAME_RULES"] = gcfg["rules"]   # legacy: flat rules dict
                            exec_globals["GAME_CONFIG"] = gcfg           # full config: rules/win/loss/prizes
                            exec_globals["game_args"] = rest.split()
                            exec_globals["active_game_state"] = active_game_state
                            exec_globals["probes"] = probes              # games read/act on live probes
                            exec_globals["run_game"] = _run_game_ref     # games can reference each other
                            exec(gcode, exec_globals)
                        except Exception as e:
                            import traceback
                            print(Fore.RED + f"[Game Error] {e}\n{traceback.format_exc()}" + Style.RESET_ALL)
                    else:
                        # Conversational (model-designed, no script): mark active and
                        # hand the model its config to run turn by turn.
                        print(Fore.MAGENTA + f"[Game] '{gname}' is active -- no script, so we play it in conversation." + Style.RESET_ALL)
                        pending_game_tool_result = (
                            f"[Game] '{gname}' is now active; run it with me turn by turn. "
                            f"Rules: {gcfg['rules'] or '(none)'}; win: {gcfg['win'] or '(none)'}; "
                            f"loss: {gcfg['loss'] or '(none)'}; prizes: {list(gcfg['prizes']) or '(none)'}."
                        )
                    continue
            if user_input.startswith(":steer "):
                sargs = user_input[len(":steer "):].strip().split()
                # :steer auto | off  -> back to the ranking.
                if sargs and sargs[0].lower() in ("auto", "off", "none", "unpin"):
                    prioritize_pin["probe"] = None
                    prioritize_pin["mix"] = None
                    print(Fore.CYAN + "[Steer] target = AUTO (follows the live ranking each turn)." + Style.RESET_ALL)
                    continue
                # :steer mix <p1> <p2> ... [alpha]  -> a lift-weighted mix: you
                # pick the probes, each probe's DEGREE is its own learned lift
                # (signed -- a probe that hurts sense subtracts, i.e. desteers).
                if sargs and sargs[0].lower() == "mix":
                    rest = sargs[1:]
                    mix_alpha = None
                    if rest:
                        try:
                            mix_alpha = abs(float(rest[-1])); rest = rest[:-1]
                        except ValueError:
                            pass
                    names = [n for n in rest if n in probes]
                    bad = [n for n in rest if n not in probes]
                    if not names:
                        print(Fore.YELLOW + f"[Steer] name at least one active probe to mix.{(' unknown: ' + ', '.join(bad)) if bad else ''}" + Style.RESET_ALL)
                        continue
                    prioritize_pin["mix"] = names
                    prioritize_pin["probe"] = None
                    if mix_alpha is not None:
                        tuner.set("prioritize_alpha", mix_alpha)
                    a_now = tuner.get("prioritize_alpha", 0.0)
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + f"[Steer] prioritize_alpha mapped to a lift-weighted MIX of {', '.join(names)} "
                        + f"(alpha={round(a_now,4)}{' -- set >0 to take effect' if a_now <= 0 else ''}). "
                        + "Each probe's degree is its own learned lift, recomputed every turn; a negative-lift "
                        + "probe desteers. :steer auto to unpin." + (f" (ignored unknown: {', '.join(bad)})" if bad else "")
                        + Style.RESET_ALL
                    )
                    continue
                # :steer <probe> <alpha>  -> pin one probe (negative = steer AWAY).
                if len(sargs) >= 2 and sargs[0] in probes:
                    probe_name = sargs[0]
                    try:
                        if sargs[1].lower() == "up": steer_alpha = 0.5
                        elif sargs[1].lower() == "down": steer_alpha = -0.5
                        else: steer_alpha = float(sargs[1])
                    except ValueError:
                        print(Fore.YELLOW + f"[Steer] invalid alpha: {sargs[1]}. Must be a number (e.g. 0.5) or 'up'/'down'." + Style.RESET_ALL)
                        continue
                    prioritize_pin["mix"] = None
                    prioritize_pin["probe"] = probe_name
                    prioritize_pin["sign"] = -1.0 if steer_alpha < 0 else 1.0
                    tuner.set("prioritize_alpha", abs(steer_alpha))
                    verb = "AWAY from" if steer_alpha < 0 else "toward"
                    print(
                        Fore.GREEN + Style.BRIGHT
                        + f"[Steer] prioritize_alpha mapped to '{probe_name}' at alpha={abs(steer_alpha)} -- steering {verb} it each turn. "
                        + ":steer auto to unpin."
                        + Style.RESET_ALL
                    )
                    continue
                if sargs:
                    print(Fore.YELLOW + f"[Steer] '{sargs[0]}' is not an active probe.{did_you_mean(sargs[0], probes)} Usage: :steer <probe> <alpha> | :steer mix <p1> <p2> ... | :steer auto." + Style.RESET_ALL)
                    continue
            if user_input.startswith(":steer") and not user_input.startswith(":steermap"):
                from invariants import engine as _engine
                stats = _engine.steer_telemetry_stats()
                lo_cur, hi_cur = _engine.get_steer_band()
                print(
                    Fore.CYAN
                    + f"[Steer] cap={_engine.get_steer_cap_fraction():.4f} band=({lo_cur:.2f}, {hi_cur:.2f}) "
                    + f"| observed pushes n={stats['n']}/{stats['window']} clip_rate={stats['clip_rate']}"
                    + Style.RESET_ALL
                )
                if stats["n"]:
                    print(
                        Fore.CYAN
                        + f"  attempted push/residual ratio: min={stats['min']:.4f} q25={stats['q25']:.4f} "
                        + f"med={stats['med']:.4f} q75={stats['q75']:.4f} q95={stats['q95']:.4f} max={stats['max']:.4f}"
                        + Style.RESET_ALL
                    )
                implied = _engine.steer_cap_from_data()
                if implied is None:
                    print(
                        Fore.YELLOW
                        + f"  data-implied cap: not enough evidence yet (n={stats['n']} < {_engine.STEER_CAP_MIN_N}); "
                        + "prior stays in force"
                        + Style.RESET_ALL
                    )
                else:
                    print(
                        Fore.CYAN
                        + f"  data-implied cap @p{_engine.STEER_CAP_PERCENTILE:g} = {implied:.4f} "
                        + "(apply deliberately: :tune steer_cap_fraction auto)"
                        + Style.RESET_ALL
                    )
                n_layers = getattr(model, "n_layers", None) or len(model.model.model.layers)
                synth_band = steer_map.suggest_band(n_layers, evidence="synthesis")
                if synth_band is None:
                    print(
                        Fore.YELLOW
                        + "  synthesis-evidence band: not enough labeled outcomes; prior stays in force"
                        + Style.RESET_ALL
                    )
                else:
                    print(
                        Fore.CYAN
                        + f"  synthesis-evidence band = ({synth_band['lo']:.2f}, {synth_band['hi']:.2f}) "
                        + f"from layers {synth_band['eligible_layers']} "
                        + f"(labeled={synth_band['labeled_events']} {synth_band['labeled_by_basis']}; "
                        + "CAVEAT: where synthesis deltas landed, a different channel's geometry)"
                        + Style.RESET_ALL
                    )
                layer_band = steer_map.suggest_band(n_layers, min_events=3, evidence="layer_steer")
                if layer_band is None:
                    print(
                        Fore.YELLOW
                        + "  layer-steer band: no single-layer evidence yet -- "
                        + ":tune steer_layer_sweep 1 plus a small alpha isolates steers by layer"
                        + Style.RESET_ALL
                    )
                else:
                    print(
                        Fore.CYAN
                        + f"  layer-steer band = ({layer_band['lo']:.2f}, {layer_band['hi']:.2f}) "
                        + f"from layers {layer_band['eligible_layers']} "
                        + f"(labeled={layer_band['labeled_events']} {layer_band['labeled_by_basis']}; transfer-free)"
                        + Style.RESET_ALL
                    )
                    for lyr, row in sorted(layer_band["per_layer"].items()):
                        print(
                            Fore.CYAN
                            + f"    L{lyr}: n={row['n']} success={row['success_rate']:.2f}"
                            + Style.RESET_ALL
                        )
                lift_rows = steer_map.channel_lift(basis="any")
                if lift_rows:
                    print(Fore.CYAN + "  channel lift (fired vs unfired outcomes -- should it be on?):" + Style.RESET_ALL)
                    for lrow in lift_rows:
                        fr = f"{lrow['fired_rate']:.2f}" if lrow["fired_rate"] is not None else "-"
                        ur = f"{lrow['unfired_rate']:.2f}" if lrow["unfired_rate"] is not None else "-"
                        print(
                            Fore.CYAN
                            + f"    {lrow['channel']}: fired={fr} (n={lrow['fired_n']}) "
                            + f"unfired={ur} (n={lrow['unfired_n']}) lift={lrow['lift']}"
                            + Style.RESET_ALL
                        )
                continue
            if user_input.startswith(":probe"):
                pargs = user_input[len(":probe"):].strip()
                if not pargs:
                    if not probes:
                        print(Fore.CYAN + "[Probe] none active. Mint one: :probe <name> <framing WITH it> || <framing WITHOUT it>" + Style.RESET_ALL)
                    for pname, pdata in probes.items():
                        trig = tuner.triggers.get(f"probe_{pname}")
                        n_pairs = len(trig.outcomes) if trig else 0
                        exposed_note = ", exposed to the model" if pdata.get("exposed") else ""
                        print(Fore.CYAN + f"[Probe] {pname}: {len(pdata['direction'])} layers, {n_pairs} paired turns{exposed_note}." + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("auto"):
                    vparts = pargs.split()
                    if len(vparts) > 1:
                        state = vparts[1].lower()
                        if state == "on":
                            config.auto_probe_readings = True
                            print(Fore.CYAN + "[Probe] Auto-readings enabled. Will print values before every input prompt." + Style.RESET_ALL)
                        else:
                            config.auto_probe_readings = False
                            print(Fore.CYAN + "[Probe] Auto-readings disabled." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + "[Probe] Usage: :probe auto [on|off]" + Style.RESET_ALL)
                    continue

                if pargs.lower() in ("values", "value", "recent", "last") or pargs.lower().startswith(("values ", "value ", "recent ", "last ")):
                    vparts = pargs.split()
                    rest = vparts[1:]
                    if not probes:
                        print(Fore.CYAN + "[Probe Values] none active. Mint one first with :probe <name> <with> || <without>" + Style.RESET_ALL)
                        continue
                    selected = sorted(probes)
                    limit = 1
                    if rest:
                        head = rest[0].lower()
                        if head == "all":
                            rest = rest[1:]
                        elif not re.fullmatch(r"\d+", head):
                            vname = resolve_probe_choice(rest[0], probes, model=model, config=config, action_name="values")
                            if not vname:
                                continue
                            if vname not in probes:
                                print(Fore.YELLOW + f"[Probe Values] no active probe named '{vname}'.{did_you_mean(vname, probes)}" + Style.RESET_ALL)
                                continue
                            selected = [vname]
                            rest = rest[1:]
                    if rest:
                        try:
                            limit = max(1, min(20, int(rest[0])))
                        except ValueError:
                            print(Fore.YELLOW + "[Probe Values] Usage: :probe values [name|all] [n]" + Style.RESET_ALL)
                            continue

                    recent_rows = []
                    for row in reversed(list(turn_log)):
                        vals = {}
                        for pname in selected:
                            key = f"probe_{pname}"
                            if key in row:
                                try:
                                    vals[pname] = float(row[key])
                                except (TypeError, ValueError):
                                    pass
                        if vals:
                            recent_rows.append((row, vals))
                            if len(recent_rows) >= limit:
                                break

                    if not recent_rows:
                        print(Fore.CYAN + "[Probe Values] no completed turn with probe readings yet; showing raw-history fallback where available." + Style.RESET_ALL)
                        for pname in selected:
                            sig, raw, n_hist = latest_probe_signal_from_history(probes[pname].get("history"))
                            if sig is None:
                                print(Fore.CYAN + f"  {pname}: no reading yet" + Style.RESET_ALL)
                            else:
                                print(Fore.CYAN + f"  {pname}: {sig:+.3f} (raw {raw:+.3f}, history n={n_hist})" + Style.RESET_ALL)
                        continue

                    if limit == 1:
                        row, vals = recent_rows[0]
                        ts = row.get("ts")
                        print(Fore.CYAN + "[Probe Values] latest completed turn" + (f" ({ts})" if ts else "") + ":" + Style.RESET_ALL)
                        for pname in selected:
                            sig = vals.get(pname)
                            if sig is None:
                                print(Fore.CYAN + f"  {pname}: no reading on that turn" + Style.RESET_ALL)
                                continue
                            _hist_sig, raw, n_hist = latest_probe_signal_from_history(probes[pname].get("history"))
                            trig = tuner.triggers.get(f"probe_{pname}")
                            parts = [f"{sig:+.3f}"]
                            if raw is not None:
                                parts.append(f"raw {raw:+.3f}")
                            parts.append(f"history n={n_hist}")
                            if trig is not None:
                                fires = sig <= trig.value if trig.comparator == "<=" else sig >= trig.value
                                parts.append(f"bar {trig.comparator} {trig.value:+.3f} ({'fires' if fires else 'quiet'})")
                                st = trig.outcome_stats()
                                if st.get("lift") is not None:
                                    parts.append(f"lift {st.get('lift'):+.3f} over {st.get('n_credited')} credited")
                            if probes[pname].get("exposed"):
                                parts.append("exposed")
                            print(Fore.CYAN + f"  {pname}: " + ", ".join(parts) + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"[Probe Values] {len(recent_rows)} most recent turn(s):" + Style.RESET_ALL)
                        for idx, (row, vals) in enumerate(recent_rows, 1):
                            ts = row.get("ts") or f"-{idx}"
                            compact = "  ".join(f"{p}={vals[p]:+.3f}" for p in sorted(vals))
                            print(Fore.CYAN + f"  {ts}: {compact}" + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("drop "):
                    dropped_raw = pargs[5:].strip()
                    dropped = resolve_probe_choice(dropped_raw, probes, model=model, config=config, action_name="drop")
                    if not dropped:
                        continue
                    if probes.pop(dropped, None) is not None:
                        print(Fore.CYAN + f"[Probe] {dropped} dropped (its observed stream is kept)." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Probe] no active probe named {dropped}.{did_you_mean(dropped, probes)}" + Style.RESET_ALL)
                    continue
                # :probe match -- tie a probe to the same-concept knob so it can DRIVE
                # the knob (servo) or VALIDATE it (correlate knob value vs reading).
                if pargs.lower() == "match" or pargs.lower().startswith("match "):
                    margs = pargs[len("match"):].split()
                    knobs = {t for t in tuner.triggers if not t.startswith("probe_")}
                    if not margs:  # list
                        if not probe_matches:
                            print(Fore.CYAN + "[Match] No probe<->knob matches. ':probe match auto' pairs same-named probes+knobs; ':probe match <probe> <knob>' sets one." + Style.RESET_ALL)
                        else:
                            for pn, m in sorted(probe_matches.items()):
                                extra = ""
                                if m.get("mode") == "validate":
                                    r = _match_corr(match_hist.get(pn))
                                    extra = f"  r={r:+.2f}" if r is not None else "  r=(needs more varied turns)"
                                print(Fore.CYAN + f"[Match] {pn} <-> knob '{m.get('knob')}'  mode={m.get('mode', 'none')}{extra}" + Style.RESET_ALL)
                        continue
                    sub = margs[0].lower()
                    if sub in ("auto", "same"):  # pair every probe whose name is a knob
                        paired = []
                        for pn in probes:
                            if pn in knobs:
                                probe_matches.setdefault(pn, {"knob": pn, "mode": "none"})["knob"] = pn
                                paired.append(pn)
                        _save_probe_matches()
                        print(Fore.GREEN + f"[Match] Auto-matched {len(paired)} probe(s) to same-named knobs: {', '.join(sorted(paired)) or '(none found)'}." + Style.RESET_ALL)
                        continue
                    pn = resolve_probe_choice(margs[0], probes, model=model, config=config, action_name="match")
                    if not pn:
                        continue
                    if pn not in probes:
                        print(Fore.YELLOW + f"[Match] no active probe '{pn}'.{did_you_mean(pn, probes)}" + Style.RESET_ALL)
                        continue
                    if len(margs) < 2:
                        print(Fore.YELLOW + "[Match] Usage: :probe match <probe> <knob> | drive [mult] | validate | check | off" + Style.RESET_ALL)
                        continue
                    op = margs[1].lower()
                    m = probe_matches.get(pn)
                    if op == "off":
                        if probe_matches.pop(pn, None) is not None:
                            if m and m.get("knob") in tuner_bindings and m.get("mode") == "drive":
                                tuner_bindings.pop(m["knob"], None)
                            match_hist.pop(pn, None)
                            _save_probe_matches()
                            print(Fore.CYAN + f"[Match] cleared match for '{pn}'." + Style.RESET_ALL)
                        else:
                            print(Fore.YELLOW + f"[Match] '{pn}' had no match." + Style.RESET_ALL)
                        continue
                    if op in ("drive", "validate", "check"):
                        if not m or not m.get("knob"):
                            print(Fore.YELLOW + f"[Match] match '{pn}' to a knob first: :probe match {pn} <knob>" + Style.RESET_ALL)
                            continue
                        knob = m["knob"]
                        if op == "drive":
                            mult = 1.0
                            if len(margs) >= 3:
                                try:
                                    mult = float(margs[2])
                                except ValueError:
                                    pass
                            tuner_bindings[knob] = ([(1.0, pn)], mult)
                            m["mode"] = "drive"
                            m["mult"] = mult
                            _save_probe_matches()
                            print(Fore.GREEN + f"[Match] SERVO: knob '{knob}' now follows probe '{pn}' every turn (x{mult:g}). ':probe match {pn} off' stops it." + Style.RESET_ALL)
                        elif op == "validate":
                            m["mode"] = "validate"
                            match_hist.setdefault(pn, deque(maxlen=200))
                            _save_probe_matches()
                            print(Fore.GREEN + f"[Match] VALIDATE: recording (knob '{knob}' value, probe '{pn}' reading) each turn. ':probe match {pn} check' for the correlation." + Style.RESET_ALL)
                        else:  # check
                            r = _match_corr(match_hist.get(pn))
                            if r is None:
                                print(Fore.YELLOW + f"[Match] {pn}: not enough varied paired turns yet (vary the knob '{knob}' while validating)." + Style.RESET_ALL)
                            else:
                                print(Fore.CYAN + f"[Match] {pn} vs knob '{knob}': r={r:+.3f} over {len(match_hist.get(pn, []))} turn(s)." + Style.RESET_ALL)
                        continue
                    # otherwise margs[1] is the knob to match to
                    knob = margs[1]
                    if knob.upper() in ("CHOICE", "CHOOSE", "AUTO"):
                        valid_knobs = list(knobs) + ["steer_cap_fraction", "steer_band"]
                        resolved_knob = resolve_probe_choice(knob, valid_knobs, model=model, config=config, action_name="match_target")
                        if not resolved_knob:
                            continue
                        knob = resolved_knob
                    if knob not in knobs and f"probe_{knob}" not in tuner.triggers and knob not in ("steer_cap_fraction", "steer_band"):
                        print(Fore.YELLOW + f"[Match] '{knob}' isn't a known knob.{did_you_mean(knob, knobs)}" + Style.RESET_ALL)
                        continue
                    probe_matches[pn] = {"knob": knob, "mode": (m.get("mode", "none") if m else "none")}
                    _save_probe_matches()
                    print(Fore.GREEN + f"[Match] {pn} <-> knob '{knob}'. Then ':probe match {pn} drive [mult]' (servo) or ':probe match {pn} validate' (credit)." + Style.RESET_ALL)
                    continue
                # :probe define <name> -- share the INITIAL BREAKDOWN: the WITH/WITHOUT
                # framings the probe was minted from (the operator-authored definition).
                if pargs.lower().startswith("define "):
                    dname_raw = pargs[7:].strip()
                    dname = resolve_probe_choice(dname_raw, probes, model=model, config=config, action_name="define")
                    if not dname:
                        continue
                    if dname not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{dname}'.{did_you_mean(dname, probes)}" + Style.RESET_ALL)
                        continue
                    framings = probes[dname].get("framings")
                    if framings and any(framings):
                        print(Fore.CYAN + f"[Probe] {dname} was minted with framings:\n  WITH:    {framings[0]}\n  WITHOUT: {framings[1]}" + Style.RESET_ALL)
                    else:
                        print(Fore.CYAN + f"[Probe] {dname} has no stored framings (likely adopted from a raw vector)." + Style.RESET_ALL)
                    continue
                # :probe explain <name> -- the MODEL explains the probe in its OWN words
                # (what it senses, when it reads high vs low), grounded on the framings.
                if pargs.lower().startswith("explain "):
                    ename_raw = pargs[8:].strip()
                    ename_resolved = resolve_probe_choice(ename_raw, probes, model=model, config=config, action_name="explain")
                    if not ename_resolved:
                        continue
                    if ename_resolved not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{ename_resolved}'.{did_you_mean(ename_resolved, probes)}" + Style.RESET_ALL)
                        continue
                    framings = probes[ename_resolved].get("framings")
                    if framings and any(framings):
                        basis = (
                            f"It is defined by contrasting two poles:\n"
                            f"WITH (reads high): {framings[0]}\n"
                            f"WITHOUT (reads low): {framings[1]}\n\n"
                        )
                    else:
                        basis = "It has no stored framings (it was adopted from a raw vector).\n\n"
                    xprompt = (
                        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"A cognitive probe named '{ename_resolved}' is a direction in your own activations "
                        f"that scores each of your replies.\n{basis}"
                        "In two or three sentences, explain in your OWN words what this probe is sensing in "
                        "you, and when it would read high versus low.<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    print(Fore.CYAN + f"[Probe] Asking the model to explain '{ename_resolved}' in its own words..." + Style.RESET_ALL)
                    try:
                        expl = generate_agentic_text(model, instruction=xprompt, config=config, pre_formatted=True, max_new_tokens=200, chatty_log=False)
                    except Exception as e:
                        print(Fore.RED + f"[Probe] explain failed: {e}" + Style.RESET_ALL)
                        continue
                    print(Fore.CYAN + f"[Probe] {ename_resolved} -- in the model's words:\n{(expl or '').strip()}" + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("chatty "):
                    chatty_name_raw = pargs[7:].strip()
                    chatty_name = resolve_probe_choice(chatty_name_raw, probes, model=model, config=config, action_name="chatty")
                    if not chatty_name:
                        continue
                    if chatty_name in probes:
                        current = probes[chatty_name].get("chatty", True)
                        probes[chatty_name]["chatty"] = not current
                        state_str = "ON" if not current else "OFF"
                        print(Fore.CYAN + f"[Probe] {chatty_name} console print {state_str}." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Probe] no active probe named {chatty_name}.{did_you_mean(chatty_name, probes)}" + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("expose "):
                    eargs = pargs[7:].split()
                    ename_raw = eargs[0].lower() if eargs else ""
                    ename_resolved = resolve_probe_choice(ename_raw, probes)
                    if not ename_resolved:
                        continue
                    ename = re.sub(r"[^a-z0-9_]", "_", ename_resolved)[:40]
                    turn_off = len(eargs) > 1 and eargs[1].lower() in ("off", "0", "false")
                    if ename not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{ename}'.{did_you_mean(ename, probes)}" + Style.RESET_ALL)
                        continue
                    probes[ename]["exposed"] = not turn_off
                    try:
                        os.makedirs(PROBE_DIR, exist_ok=True)
                        torch.save(
                            {
                                "direction": probes[ename]["direction"],
                                "framings": probes[ename].get("framings", ("", "")),
                                "exposed": probes[ename]["exposed"],
                            },
                            os.path.join(PROBE_DIR, f"{ename}.pt"),
                        )
                    except Exception:
                        pass
                    memory.append_event(
                        "probe_exposure_changed",
                        text=f"{ename}: {'exposed' if not turn_off else 'hidden'}",
                        tags=["probe"],
                        provenance={"probe": ename, "exposed": not turn_off},
                    )
                    if turn_off:
                        print(Fore.CYAN + f"[Probe] {ename} is hidden from the model again." + Style.RESET_ALL)
                    else:
                        print(
                            Fore.CYAN
                            + f"[Probe] {ename} EXPOSED: the model may now consult it with "
                            + f"<<PROBE: {ename}>> (its own reading) or <<PROBE: {ename} || candidate words>> "
                            + "(score hypothetical words). Reading only -- it still cannot mint, drop, or calibrate."
                            + Style.RESET_ALL
                        )
                    continue
                if pargs.lower() == "adopt" or pargs.lower().startswith("adopt "):
                    rest, adopt_layers = strip_band_suffix(pargs[5:].strip())
                    stems = [re.sub(r"[^a-z0-9_]", "_", s.lower())[:40] for s in rest.split()] if rest else []
                    if not stems:
                        print(
                            Fore.YELLOW
                            + "[Probe] Usage: :probe adopt <dim> [<dim> ...] [band <lo> <hi>] (e.g. "
                            + "ambiguity disagreement warranted_confidence band 16 24)."
                            + Style.RESET_ALL
                        )
                        continue
                    adopted = []
                    for stem in stems:
                        if stem in probes:
                            print(Fore.YELLOW + f"[Probe] '{stem}' is already an active probe." + Style.RESET_ALL)
                            continue
                        direction, src_name, exposed_state = load_stored_direction(model, stem, layers=adopt_layers)
                        if not direction:
                            print(
                                Fore.YELLOW
                                + f"[Probe] no usable vector for '{stem}' (looked for invariants/{stem}_vector.pt "
                                + f"and invariants/{stem}.pt)."
                                + Style.RESET_ALL
                            )
                            continue
                        probes[stem] = {
                            "direction": direction,
                            "history": deque(maxlen=40),
                            "framings": (f"adopted:{src_name}", ""),
                            "exposed": exposed_state,
                        }
                        tuner.register(f"probe_{stem}", 0.0, kind="threshold", comparator=">=")
                        try:
                            os.makedirs(PROBE_DIR, exist_ok=True)
                            torch.save(
                                {"direction": direction, "framings": probes[stem]["framings"]},
                                os.path.join(PROBE_DIR, f"{stem}.pt"),
                            )
                        except Exception:
                            pass
                        memory.append_event(
                            "probe_minted",
                            text=f"{stem}: adopted from {src_name}",
                            tags=["probe"],
                            provenance={
                                "layers": sorted(direction),
                                "authored_by": "operator",
                                "adopted_from": src_name,
                            },
                        )
                        print(
                            Fore.CYAN
                            + f"[Probe] {stem} adopted from {src_name} over {len(direction)} layers."
                            + Style.RESET_ALL
                        )
                        adopted.append(stem)
                    if adopted:
                        print(
                            Fore.CYAN
                            + f"[Probe] {len(adopted)} dimension(s) now scoring every reply like any probe "
                            + f"(phen_ streams keep reading them in the reasoning states). Pre-load the "
                            + f"archive in one pass: :probe backfill {adopted[0]}"
                            + Style.RESET_ALL
                        )
                    continue
                if pargs.lower() == "compose" or pargs.lower().startswith("compose "):
                    rest = pargs[len("compose"):].strip()
                    cname, _, expr = rest.partition(" ")
                    cname = re.sub(r"[^a-z0-9_]", "_", cname.lower())[:40]
                    expr = expr.strip()
                    if not cname or not expr:
                        print(
                            Fore.YELLOW
                            + "[Probe] Usage: :probe compose <name> <signed mix>, e.g. "
                            + ":probe compose memory_need ambiguity + disagreement - validated_flow "
                            + "- warranted_confidence  (weights allowed: 0.5*curiosity)."
                            + Style.RESET_ALL
                        )
                        continue
                    # Removed: if cname in probes: ... continue 
                    # so that users can overwrite/augment a probe using itself (e.g. compose amb amb + new)
                    expr, cband = strip_band_suffix(expr)
                    terms, perr = parse_compose_expr(expr)
                    if perr is not None:
                        print(Fore.YELLOW + f"[Probe] could not parse term '{perr}' in the mix." + Style.RESET_ALL)
                        continue
                    resolved = []
                    missing = None
                    for w, tname in terms:
                        if tname in probes and cband is None:
                            tdir = probes[tname]["direction"]
                        else:
                            # An explicit band re-reads every term from its
                            # stored file at that depth; an active probe's
                            # in-memory direction is fixed at its mint band.
                            tdir, _, _ = load_stored_direction(model, tname, layers=cband)
                            if not tdir and tname in probes:
                                tdir = {
                                    L: v for L, v in probes[tname]["direction"].items()
                                    if cband is None or int(L) in set(cband)
                                }
                        if not tdir:
                            missing = tname
                            break
                        resolved.append((w, tname, tdir))
                    if missing is not None:
                        print(
                            Fore.YELLOW
                            + f"[Probe] '{missing}' is neither an active probe nor a stored vector "
                            + f"(invariants/{missing}_vector.pt / invariants/{missing}.pt).{did_you_mean(missing, probes)}"
                            + Style.RESET_ALL
                        )
                        continue
                    # A signed sum of unit directions IS the composite contrast:
                    # projecting on normalize(sum w_i * v_i) equals the weighted
                    # sum of the per-term cosines (up to the shared norm), so the
                    # mix measures exactly the relationship it names. Intersect
                    # layers so the composite means the same thing at every depth.
                    common = set.intersection(*(set(d.keys()) for _, _, d in resolved))
                    if not common:
                        spans = ", ".join(f"{t}: L{min(d)}-{max(d)}" for _, t, d in resolved)
                        print(
                            Fore.YELLOW
                            + f"[Probe] the terms share no layers ({spans}) -- they were minted or "
                            + "stored over different bands, so no honest composite exists."
                            + Style.RESET_ALL
                        )
                        continue
                    direction = {}
                    for L in sorted(common):
                        acc = None
                        for w, _, d in resolved:
                            v = d[L].to(model.device).float().reshape(-1) * w
                            acc = v if acc is None else acc + v
                        n = acc.norm()
                        if n.item() > 0:
                            direction[int(L)] = acc / n
                    if not direction:
                        print(Fore.YELLOW + "[Probe] the mix cancelled to zero on every shared layer; nothing to mint." + Style.RESET_ALL)
                        continue
                    recipe = " ".join(
                        f"{'+' if w >= 0 else '-'} {abs(w):g}*{t}".replace("1*", "") for w, t in terms
                    ).lstrip("+ ").strip()
                    probes[cname] = {
                        "direction": direction,
                        "history": deque(maxlen=40),
                        "framings": (f"composed: {recipe}", ""),
                    }
                    tuner.register(f"probe_{cname}", 0.0, kind="threshold", comparator=">=")
                    try:
                        os.makedirs(PROBE_DIR, exist_ok=True)
                        torch.save(
                            {"direction": direction, "framings": probes[cname]["framings"]},
                            os.path.join(PROBE_DIR, f"{cname}.pt"),
                        )
                    except Exception:
                        pass
                    memory.append_event(
                        "probe_minted",
                        text=f"{cname}: composed {recipe}",
                        tags=["probe"],
                        provenance={
                            "layers": sorted(direction),
                            "authored_by": "operator",
                            "composed_from": [[float(w), t] for w, t in terms],
                        },
                    )
                    print(
                        Fore.CYAN
                        + f"[Probe] {cname} composed = {recipe} over {len(direction)} shared layers -- "
                        + "scores every reply like any probe, backfillable, calibratable. "
                        + f"Pre-load the archive: :probe backfill {cname}"
                        + Style.RESET_ALL
                    )
                    continue
                if pargs.lower() == "backfill" or pargs.lower().startswith("backfill "):
                    rest = pargs[len("backfill"):].split()
                    if not rest:
                        print(Fore.YELLOW + "[Probe] Usage: :probe backfill <name|all|choose> [n]" + Style.RESET_ALL)
                        continue
                    bf_name_raw = rest[0]
                    bf_names_to_rebuild = []
                    if bf_name_raw == "all":
                        bf_names_to_rebuild = list(probes.keys())
                    elif bf_name_raw == "choose":
                        candidates = []
                        for p in probes:
                            tr = tuner.triggers.get(f"probe_{p}")
                            n_sigs = len(tr.signals) if tr else 0
                            candidates.append((n_sigs, p))
                        if candidates:
                            candidates.sort()
                            chosen = candidates[0][1]
                            print(Fore.CYAN + f"[Probe] Model chose to backfill '{chosen}' (only {candidates[0][0]} signals)." + Style.RESET_ALL)
                            bf_names_to_rebuild = [chosen]
                        else:
                            print(Fore.YELLOW + "[Probe] No active probes to backfill." + Style.RESET_ALL)
                            continue
                    else:
                        bf_name_resolved = resolve_probe_choice(bf_name_raw, probes)
                        if not bf_name_resolved:
                            continue
                        bf_name = re.sub(r"[^a-z0-9_]", "_", bf_name_resolved)[:40]
                        if bf_name not in probes:
                            print(Fore.YELLOW + f"[Probe] no active probe named '{bf_name}'.{did_you_mean(bf_name, probes)} Bare :probe lists them." + Style.RESET_ALL)
                            continue
                        bf_names_to_rebuild = [bf_name]
                    bf_limit = None
                    if len(rest) > 1:
                        try:
                            bf_limit = max(1, int(rest[1]))
                        except ValueError:
                            bf_limit = None
                    archive = [
                        r for r in memory.records
                        if r.scope == memory.scope and r.kind == "turn"
                        and r.role == "assistant" and (r.text or "").strip()
                    ]
                    if bf_limit:
                        archive = archive[-bf_limit:]
                    if not archive:
                        print(Fore.YELLOW + "[Probe] no stored assistant replies to backfill from." + Style.RESET_ALL)
                        continue
                    # Scoring re-encodes reply text, so the archive can be scored
                    # exactly as live turns were. The trigger stream is REBUILT,
                    # not appended: the archive is a superset of every turn this
                    # probe scored live, and replacing is what keeps those turns
                    # from counting twice. Pre-mint turns alone get new paired
                    # rows in the turn log -- post-mint turns already wrote
                    # theirs live.
                    mint_ts_map = {}
                    for _pn in bf_names_to_rebuild:
                        _mint = next(
                            (e.timestamp for e in memory.records
                             if e.kind == "event" and "probe" in (e.tags or []) and (e.text or "").startswith(f"{_pn}:")),
                            None,
                        )
                        mint_ts_map[_pn] = _mint
                    from invariants.engine import _inputs as _bf_inputs, _hidden_states as _bf_hidden, probe_score as _bf_score
                    # One shared forward per archived reply scores EVERY active
                    # probe (projections are free once the states exist), so the
                    # rows written are JOINT -- multi-anchor calibration needs
                    # anchors on the same row. Only the NAMED probe's trigger
                    # and rolling history are rebuilt.
                    bf_rollings = {p: deque(maxlen=40) for p in probes}
                    bf_scored = []  # (timestamp, {probe: (raw, sig)}, stored sense)
                    display_name = "all probes" if bf_name_raw == "all" else bf_names_to_rebuild[0]
                    print(
                        Fore.CYAN
                        + f"[Probe] backfilling {display_name} over {len(archive)} archived replies "
                        + f"(one forward each; scoring {len(probes)} active probe(s) per reply)..."
                        + Style.RESET_ALL,
                        flush=True,
                    )
                    with torch.no_grad():
                        for bf_i, br in enumerate(archive):
                            b_ids = _bf_inputs(model, br.text[:600])
                            b_hs = _bf_hidden(model, b_ids["input_ids"], b_ids.get("attention_mask"))
                            per_probe = {}
                            for _pn, _pd in probes.items():
                                raw = _bf_score(b_hs, _pd["direction"])
                                roll = bf_rollings[_pn]
                                sig = raw - (sum(roll) / len(roll)) if roll else 0.0
                                roll.append(raw)
                                per_probe[_pn] = (raw, sig)
                            bm = br.metrics if isinstance(br.metrics, dict) else {}
                            sense = bm.get("sense_score")
                            bf_scored.append((br.timestamp, per_probe, float(sense) if sense is not None else None))
                            if (bf_i + 1) % 25 == 0:
                                print(Fore.CYAN + f"  ...{bf_i + 1}/{len(archive)}" + Style.RESET_ALL, flush=True)
                    total_credited = {}
                    for _pn in bf_names_to_rebuild:
                        trig = tuner.register(f"probe_{_pn}", 0.0, kind="threshold", comparator=">=")
                        trig.signals.clear()
                        trig.outcomes.clear()
                        trig.observed = 0
                        trig.fired = 0
                        credited = 0
                        for _bts, _bpp, _bsense in bf_scored:
                            _bsig = _bpp[_pn][1]
                            trig.observe(_bsig)
                            if _bsense is not None:
                                trig.credit(_bsig, _bsense)
                                credited += 1
                        total_credited[_pn] = credited
                        probes[_pn]["history"] = deque((pp[_pn][0] for _, pp, _ in bf_scored), maxlen=40)
                    tuner.save()
                    
                    existing_ts = set()
                    try:
                        with open(TURN_SIGNALS_PATH, "r", encoding="utf-8") as _tf:
                            for _line in _tf:
                                try:
                                    _r = json.loads(_line)
                                except Exception:
                                    continue
                                if _r.get("basis") == "backfill" and _r.get("ts"):
                                    # if ANY of the rebuilt probes are already in this backfill row, consider it existing
                                    if any(f"probe_{_pn}" in _r for _pn in bf_names_to_rebuild):
                                        existing_ts.add(_r["ts"])
                    except OSError:
                        pass
                    rows_added = 0
                    try:
                        with open(TURN_SIGNALS_PATH, "a", encoding="utf-8") as _tf:
                            for _bts, _bpp, _bsense in bf_scored:
                                # We skip adding this joint row if ALL rebuilt probes were minted BEFORE this turn
                                if all(mint_ts_map.get(_pn) is not None and _bts >= mint_ts_map[_pn] for _pn in bf_names_to_rebuild):
                                    continue
                                if _bts in existing_ts:
                                    continue
                                row = {"ts": _bts, "basis": "backfill"}
                                for _pn, (_praw, _psig) in _bpp.items():
                                    row[f"probe_{_pn}"] = float(_psig)
                                if _bsense is not None:
                                    row["sense"] = float(_bsense)
                                turn_log.append(dict(row))
                                _tf.write(json.dumps(row) + chr(10))
                                rows_added += 1
                    except OSError:
                        pass
                    memory.append_event(
                        "probe_backfilled",
                        text=f"{display_name}: {len(bf_scored)} archived replies re-scored",
                        tags=["probe"],
                        provenance={
                            "probes": bf_names_to_rebuild,
                            "turns_scored": len(bf_scored),
                            "paired_rows_added": rows_added,
                        },
                    )
                    st = tuner.triggers[f"probe_{bf_names_to_rebuild[0]}"].outcome_stats() if bf_names_to_rebuild else {"lift": 0}
                    print(
                        Fore.GREEN
                        + f"[Probe] {display_name} backfilled: {len(bf_scored)} archived replies re-scored in order. "
                        + f"Trigger stream(s) rebuilt, "
                        + f"{rows_added} pre-mint paired rows added, rolling history seeded. "
                        + f"Live scoring continues normally."
                        + Style.RESET_ALL
                    )
                    continue
                pname, _, framings = pargs.partition(" ")
                pname = re.sub(r"[^a-z0-9_]", "_", pname.lower())[:40]
                if "||" not in framings and r"\||" not in framings:
                    print(Fore.CYAN + f"[Probe] Suggesting contrastive framings for '{pname}'..." + Style.RESET_ALL)
                    suggestion_prompt = (
                        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"Write a contrastive definition pair for a behavioral dimension called '{pname}'. "
                        f"Write both sides in the FIRST PERSON, as I describe MYSELF -- each side MUST start with 'I'. "
                        f"The format must be exactly: <first-person positive statement> || <first-person negative statement>.\n\n"
                        f"Example for 'understanding': I fully grasp what the user means. || I am confused and miss the point.\n\n"
                        f"Output ONLY the single contrastive pair. Do not add any other text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    )
                    sug = generate_agentic_text(
                        model,
                        instruction=suggestion_prompt,
                        config=config,
                        pre_formatted=True,
                        max_new_tokens=150
                    )
                    sug = sug.strip()
                    print(Fore.GREEN + f"\n[Probe Suggestion] Try running this:\n:probe {pname} {sug}\n" + Style.RESET_ALL)
                    continue
                a_text, _, b_text = _partition_unescaped_pipes(framings)
                a_text, b_text = a_text.strip(), b_text.strip()
                b_text, mint_layers = strip_band_suffix(b_text)
                from invariants.engine import _inputs, _hidden_states, probe_direction, steer_band_layers
                ids_a = _inputs(model, a_text[:600])
                hs_a = _hidden_states(model, ids_a["input_ids"], ids_a.get("attention_mask"))
                ids_b = _inputs(model, b_text[:600])
                hs_b = _hidden_states(model, ids_b["input_ids"], ids_b.get("attention_mask"))
                direction = probe_direction(
                    hs_a, hs_b,
                    mint_layers if mint_layers is not None else steer_band_layers(hs_a.shape[0]),
                )
                if not direction:
                    print(Fore.YELLOW + "[Probe] framings produced no usable direction; try more contrastive text." + Style.RESET_ALL)
                    continue
                probes[pname] = {"direction": direction, "history": deque(maxlen=40), "framings": (a_text, b_text)}
                tuner.register(f"probe_{pname}", 0.0, kind="threshold", comparator=">=")
                try:
                    os.makedirs(PROBE_DIR, exist_ok=True)
                    save_name = pname if getattr(model.model.config, "_name_or_path", "") == DEFAULT_MODEL else f"{pname}_d{model.d_model}"
                    torch.save({"direction": direction, "framings": (a_text, b_text)}, os.path.join(PROBE_DIR, f"{save_name}.pt"))
                except Exception:
                    pass
                memory.append_event(
                    "probe_minted",
                    text=f"{pname}: {a_text} || {b_text}",
                    tags=["probe"],
                    provenance={"layers": sorted(direction), "authored_by": "operator"},
                )
                print(
                    Fore.CYAN
                    + f"[Probe] {pname} minted over {len(direction)} layers from your framings -- an UNVALIDATED "
                    + "hypothesis-sensor. It scores every turn from here (one extra forward per turn), centered "
                    + "against its own rolling history, paired with sense. Calibrate against it once evidence "
                    + f"accrues: :calibrate conversation_productive {pname}. Or pre-load evidence from the "
                    + f"archive now: :probe backfill {pname}"
                    + Style.RESET_ALL
                )
                continue
            if user_input.strip().lower() == ":suggest apply" or user_input.strip().lower().startswith(":suggest apply "):
                _arch = sum(
                    1 for r in memory.records
                    if r.scope == memory.scope and r.kind == "turn" and r.role == "assistant"
                )
                sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                safe = [(cat, line, cmd) for cat, line, cmd in sugg if cat in SUGGEST_APPLY_SAFE]
                
                # Forward the because clause to the queued commands
                apply_because = None
                if command_because:
                    apply_because = f" because {command_because}"
                    
                if not safe:
                    print(Fore.CYAN + "[Suggest Apply] nothing safe to auto-run (explore/expose stay manual)." + Style.RESET_ALL)
                else:
                    cmds = [cmd + (apply_because if apply_because else "") for _, _, cmd in safe]
                    input_queue.extend(cmds)
                    print(Fore.GREEN + f"[Suggest Apply] Auto-queued {len(cmds)} measurement/calibration action(s)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":place "):
                pargs = user_input[len(":place "):].strip().split()
                if len(pargs) >= 2:
                    pname = resolve_probe_choice(pargs[0], probes)
                    if not pname:
                        continue
                    if pname in probes:
                        sign_arg = pargs[1]
                        if sign_arg in ("+", "positive", "1", "pos"):
                            sign = 1.0
                        elif sign_arg in ("-", "negative", "-1", "neg"):
                            sign = -1.0
                        else:
                            print(Fore.YELLOW + "[Place] specify + or - for the placement." + Style.RESET_ALL)
                            continue
                        
                        last_reply = None
                        for r in reversed(memory.records):
                            if r.role == "assistant":
                                last_reply = r.text
                                break
                        if not last_reply:
                            print(Fore.YELLOW + "[Place] no assistant reply found to place." + Style.RESET_ALL)
                            continue
                            
                        print(Fore.CYAN + f"[Place] Extracting state of last response to update '{pname}'..." + Style.RESET_ALL)
                        from invariants.engine import _inputs, _hidden_states
                        ids = _inputs(model, last_reply[:600])
                        hs = _hidden_states(model, ids["input_ids"], ids.get("attention_mask"))
                        
                        probe_dir = probes[pname]["direction"]
                        learning_rate = 0.15  # a noticeable nudge
                        updated_layers = 0
                        for L in list(probe_dir.keys()):
                            L_str = str(L)
                            if L_str in hs:
                                mean_hs = hs[L_str].mean(dim=0).to(model.device).reshape(-1)
                                if mean_hs.norm().item() > 0:
                                    mean_hs = mean_hs / mean_hs.norm()
                                    current_dir = probe_dir[L].to(model.device).reshape(-1)
                                    new_dir = current_dir + sign * learning_rate * mean_hs
                                    if new_dir.norm().item() > 0:
                                        probe_dir[L] = new_dir / new_dir.norm()
                                        updated_layers += 1
                                        
                        print(Fore.GREEN + Style.BRIGHT + f"[Place] '{pname}' probe nudged {'toward' if sign > 0 else 'away from'} the last response over {updated_layers} layers." + Style.RESET_ALL)
                        memory.append_event(
                            "probe_placed",
                            text=f"placed last response on '{pname}' as {'positive' if sign > 0 else 'negative'}",
                            tags=["probe", "place"],
                            provenance={"probe": pname, "sign": sign}
                        )
                else:
                    print(Fore.YELLOW + "[Place] Usage: :place <probe> <+|->" + Style.RESET_ALL)
                continue
            if user_input.lower().startswith(":suggest"):
                sargs = user_input.strip().split()
                _arch = sum(
                    1 for r in memory.records
                    if r.scope == memory.scope and r.kind == "turn" and r.role == "assistant"
                )
                
                target_probe = None
                if len(sargs) > 1 and sargs[1].lower() not in ("apply", "suggestions"):
                    if sargs[1].lower() in ("command", "commands", "help"):
                        input_queue.insert(0, ":help")
                        print(Fore.CYAN + "[Suggest] Showing command help via :help." + Style.RESET_ALL)
                        continue
                    if sargs[1].startswith(":"):
                        cmd_name = sargs[1]
                        input_queue.insert(0, f":help {cmd_name}")
                        print(Fore.CYAN + f"[Suggest] Showing command help via :help {cmd_name}." + Style.RESET_ALL)
                        continue
                        
                    target_probe = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower())[:40]
                    if target_probe in probes:
                        print(Fore.CYAN + f"[Suggest] Scanning for specific moves for probe '{target_probe}'..." + Style.RESET_ALL)
                        all_sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                        sugg = [(cat, line, cmd) for cat, line, cmd in all_sugg if target_probe in line or target_probe in cmd]
                        if not sugg:
                            print(Fore.YELLOW + f"[Suggest] No specific data-backed moves ready for '{target_probe}' yet." + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To generate evidence: :probe backfill {target_probe}" + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To steer blindly:     :tune probe_{target_probe}_alpha 0.5" + Style.RESET_ALL)
                            continue
                    else:
                        print(Fore.CYAN + f"[Suggest] Probe '{target_probe}' not active. Mint it first with :probe {target_probe} <positive> || <negative>" + Style.RESET_ALL)
                        continue
                elif len(sargs) == 1 and command_because:
                    # Internally compute the similarity of the because string to all active probes
                    if not probes:
                        print(Fore.YELLOW + "[Suggest] No active probes to match against your reason. Mint some first!" + Style.RESET_ALL)
                        continue
                        
                    print(Fore.CYAN + f"[Suggest] Projecting your reason into the model's representation space to find the best matching probes..." + Style.RESET_ALL)
                    from invariants.engine import _inputs, _hidden_states
                    import torch
                    
                    ids = _inputs(model, command_because[:600])
                    hs = _hidden_states(model, ids["input_ids"], ids.get("attention_mask"))
                    
                    probe_scores = {}
                    for pname, pdata in probes.items():
                        probe_dir = pdata["direction"]
                        sim_sum = 0.0
                        layers_counted = 0
                        for L in list(probe_dir.keys()):
                            L_str = str(L)
                            if L_str in hs:
                                mean_hs = hs[L_str].mean(dim=0).to(model.device).reshape(-1)
                                if mean_hs.norm().item() > 0:
                                    mean_hs = mean_hs / mean_hs.norm()
                                    p_dir = probe_dir[L].to(model.device).reshape(-1)
                                    sim = torch.nn.functional.cosine_similarity(mean_hs.unsqueeze(0), p_dir.unsqueeze(0)).item()
                                    sim_sum += sim
                                    layers_counted += 1
                        if layers_counted > 0:
                            probe_scores[pname] = sim_sum / layers_counted
                            
                    if not probe_scores:
                        print(Fore.YELLOW + "[Suggest] Could not compute similarities." + Style.RESET_ALL)
                        continue
                        
                    sorted_probes = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)
                    top_probe, top_score = sorted_probes[0]
                    
                    print(Fore.GREEN + Style.BRIGHT + f"[Suggest] Best internal representation match: '{top_probe}' (similarity: {top_score:+.3f})" + Style.RESET_ALL)
                    if len(sorted_probes) > 1:
                        runners_up = ", ".join(f"{p} ({s:+.3f})" for p, s in sorted_probes[1:3])
                        print(Fore.CYAN + f"          Runners up: {runners_up}" + Style.RESET_ALL)
                        
                    all_sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                    sugg = [(cat, line, cmd) for cat, line, cmd in all_sugg if top_probe in line or top_probe in cmd]
                    
                    if not sugg:
                        print(Fore.YELLOW + f"[Suggest] No specific data-backed moves ready for '{top_probe}' yet." + Style.RESET_ALL)
                        print(Fore.CYAN + f"  -> To generate evidence: :probe backfill {top_probe}" + Style.RESET_ALL)
                        print(Fore.CYAN + f"  -> To steer blindly:     :tune probe_{top_probe}_alpha 0.5" + Style.RESET_ALL)
                        continue
                else:
                    sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)

                if not sugg:
                    print(
                        Fore.CYAN
                        + "[Suggest] nothing ready yet: no lever has accrued enough evidence for a "
                        + "data-backed move. Streams accrue every turn; :probe backfill pre-loads the archive."
                        + Style.RESET_ALL
                    )
                else:
                    labels = {
                        "calibrate": "Calibrations ready",
                        "commit": "Explored, not committed",
                        "explore": "Capabilities never tried",
                        "backfill": "Probes short on evidence",
                        "expose": "Signals worth exposing",
                    }
                    print(
                        Fore.CYAN
                        + f"[Suggest] {len(sugg)} move(s) the accrued state supports -- computed, not applied:"
                        + Style.RESET_ALL
                    )
                    for cat in ("commit", "calibrate", "explore", "backfill", "expose"):
                        group = [(l, c) for k, l, c in sugg if k == cat]
                        if not group:
                            continue
                        print(Fore.CYAN + Style.BRIGHT + f"  [{labels[cat]}]" + Style.RESET_ALL)
                        for line, cmd in group[:8]:
                            if command_because:
                                cmd += f" because {command_because}"
                            print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                            print(Fore.GREEN + f"      -> {cmd}" + Style.RESET_ALL)
                continue
            if user_input.startswith(":label"):
                # Human-in-the-loop: judge the MOST RECENT turn on a probe's (or
                # any stream's) axis as positive or negative. Credits that axis's
                # last-turn signal with an operator outcome (1.0 / 0.0) -- a
                # supervised datapoint alongside the automatic sense credit, so a
                # probe's lift can be grounded in human judgment, not only sense.
                largs = user_input[len(":label"):].split()
                if len(largs) < 2:
                    print(Fore.CYAN + "[Label] Usage: :label <probe|stream> pos|neg -- mark the last turn on that axis." + Style.RESET_ALL)
                    continue
                lname_raw = largs[0]
                lname = resolve_probe_choice(lname_raw, probes)
                if not lname:
                    continue
                verdict = largs[1].lower()
                if verdict in ("pos", "positive", "+", "good", "1", "up"):
                    outcome = 1.0
                elif verdict in ("neg", "negative", "-", "bad", "0", "down"):
                    outcome = 0.0
                else:
                    print(Fore.YELLOW + "[Label] verdict must be pos or neg (positive/negative)." + Style.RESET_ALL)
                    continue
                target_trigger, target_stream = resolve_target(lname, tuner)
                if target_stream is None or target_trigger not in tuner.triggers:
                    print(Fore.YELLOW + f"[Label] unknown axis '{lname}'.{did_you_mean(lname, calibratable_names(tuner))}" + Style.RESET_ALL)
                    continue
                if not turn_log:
                    print(Fore.YELLOW + "[Label] no completed turn to label yet." + Style.RESET_ALL)
                    continue
                sig = turn_log[-1].get(target_stream)
                if sig is None:
                    print(Fore.YELLOW + f"[Label] the last turn carries no reading for '{target_stream}'. Score it first." + Style.RESET_ALL)
                    continue
                tuner.credit(target_trigger, float(sig), outcome)
                memory.append_event(
                    "operator_label",
                    text=f"{target_trigger}: {'positive' if outcome >= 0.5 else 'negative'} (signal {float(sig):+.4f})",
                    tags=["label", "human_feedback"],
                    provenance={"axis": target_trigger, "signal": float(sig), "outcome": outcome},
                )
                st = tuner.triggers[target_trigger].outcome_stats()
                disp = target_trigger[6:] if target_trigger.startswith("probe_") else target_trigger
                print(
                    Fore.GREEN
                    + f"[Label] last turn marked {'POSITIVE' if outcome >= 0.5 else 'NEGATIVE'} on {disp} "
                    + f"(its signal {float(sig):+.4f} paired with outcome {outcome:g}). "
                    + f"Lift now {st.get('lift')} over {st.get('n_credited')} credited turns."
                    + Style.RESET_ALL
                )
                continue
            if user_input.startswith(":calibrate"):
                cargs = user_input[len(":calibrate"):].split()
                if not cargs:
                    print(Fore.CYAN + "[Calibrate] Usage: :calibrate <name> [percentile|intent|band args]." + Style.RESET_ALL)
                    routes = {}
                    for nm in sorted(set(list(tuner.triggers) + ["steer_cap_fraction", "steer_band"])):
                        routes.setdefault(calibration_policy(nm)[0], []).append(nm)
                    for route in ("threshold", "cap", "band", "reject"):
                        if routes.get(route):
                            label = {"threshold": "percentile of own signals", "cap": "observed push ratios",
                                     "band": "per-layer outcomes", "reject": "REFUSED (circular/binary)"}[route]
                            print(Fore.CYAN + f"  {label}: {', '.join(routes[route])}" + Style.RESET_ALL)
                    continue
                cal_name_raw, cargs = consume_probe_args(cargs)
                cal_name = resolve_probe_choice(cal_name_raw, calibratable_names(tuner), model=model, config=config, action_name="calibrate")
                if not cal_name:
                    continue
                route, reason = calibration_policy(cal_name)
                if len(cargs) >= 1 and cargs[0].lower() == "outcome":
                    if cal_name not in OUTCOME_CALIBRATABLE and route != "threshold":
                        print(Fore.YELLOW + f"[Calibrate] outcome route not available for '{cal_name}'." + Style.RESET_ALL)
                        continue
                    trig = tuner.triggers.get(cal_name)
                    pairs = list(trig.outcomes) if trig is not None else []
                    v = outcome_calibration(pairs)
                    if v is None:
                        tried = sorted({round(float(sig), 6) for sig, _ in pairs})
                        print(
                            Fore.YELLOW
                            + f"[Calibrate] refused: outcome calibration needs >=2 tried values with >=5 "
                            + f"sensed turns each (tried so far: {tried or 'none'}). No exploration, no verdict.\n"
                            + "(Type ':queue' to auto-retry this in the background)"
                            + Style.RESET_ALL
                        )
                        last_refused_calibration = user_input[len(":calibrate"):].strip()
                    else:
                        tuner.set(cal_name, v)
                        print(
                            Fore.GREEN
                            + f"[Calibrate] {cal_name} = {round(v, 4)} -- the tried value whose turns "
                            + f"went best ({len(pairs)} sensed turns across the values explored)."
                            + Style.RESET_ALL
                        )
                    continue
                if route == "reject":
                    print(Fore.YELLOW + f"[Calibrate] rejected for '{cal_name}': {reason}." + Style.RESET_ALL)
                    continue
                if route == "cap":
                    cal_pct = None
                    if len(cargs) >= 2:
                        try:
                            cal_pct = float(cargs[1])
                        except ValueError:
                            print(Fore.YELLOW + "[Calibrate] percentile must be a number." + Style.RESET_ALL)
                            continue
                    if cal_pct is not None and cal_pct >= 100:
                        print(
                            Fore.YELLOW
                            + "[Calibrate] rejected: p100 admits every observed push -- the envelope "
                            + "would stop binding in the observed regime."
                            + Style.RESET_ALL
                        )
                        continue
                    from invariants import engine as _engine
                    v = _engine.calibrate_steer_cap_fraction(percentile=cal_pct)
                    if v is None:
                        st = _engine.steer_telemetry_stats()
                        print(
                            Fore.YELLOW 
                            + f"[Calibrate] refused: only {st['n']} observed pushes (need {_engine.STEER_CAP_MIN_N}).\n"
                            + "(Type ':queue' to auto-retry this in the background)"
                            + Style.RESET_ALL
                        )
                        last_refused_calibration = user_input[len(":calibrate"):].strip()
                    else:
                        tuner.set("steer_cap_fraction", v)
                        print(Fore.GREEN + f"[Calibrate] steer_cap_fraction = {round(v, 4)} from observed push ratios." + Style.RESET_ALL)
                    continue
                if route == "band":
                    user_input = ":tune steer_band auto " + " ".join(cargs[1:])
                    # fall through to the :tune handler below with the same args
                if route == "threshold":
                    anchor = "".join(cargs[1:]).lower() if len(cargs) >= 2 else None
                    anchor_is_word = anchor is not None and not anchor.replace(".", "").isdigit()
                    if cal_name == "conversation_productive" and anchor == "intent":
                        user_input = ":tune conversation_productive auto intent"
                        # fall through to the shared intent route below
                    elif anchor_is_word:
                        # 'intent' reaches here only when the target is NOT
                        # conversation_productive (that pair is caught above),
                        # so it anchors on the intent_settling stream like any
                        # other name -- not the productive intent-axis route.
                        # BOTH WAYS, AND JOINTLY: any threshold target, anchored
                        # to one named stream or several joined with '+'. A turn
                        # counts as anchor-fired only when EVERY named stream
                        # fired; the target's bar becomes the cut (in its own
                        # units) between those turns and the rest.
                        target_trigger, target_stream = resolve_target(cal_name, tuner)
                        if target_stream is None or target_trigger not in tuner.triggers:
                            print(Fore.YELLOW + f"[Calibrate] unknown target '{cal_name}'.{did_you_mean(cal_name, calibratable_names(tuner))}" + Style.RESET_ALL)
                            continue
                        anchors, anchor_labels, bad_part = parse_anchor_spec(anchor, tuner)
                        if bad_part is not None or not anchors:
                            print(
                                Fore.YELLOW
                                + f"[Calibrate] unknown anchor '{bad_part or anchor}'.{did_you_mean(bad_part or anchor, calibratable_names(tuner))} "
                                + "Name an observed stream "
                                + "(productive/intent/impact/a probe/a phen_ sensor), join several with '+', "
                                + "negate one with a leading '-' (e.g. -estrangement), "
                                + f"or mint one: :probe {bad_part or anchor} <with it> || <without it>."
                                + Style.RESET_ALL
                            )
                            continue
                        anchor_streams = [a[0] for a in anchors]
                        v = paired_threshold_multi(list(turn_log), target_stream, anchors)
                        anchor_label = "+".join(anchor_labels)
                        n_rows = sum(1 for r in turn_log if target_stream in r and all(s in r for s in anchor_streams))
                        if v is None:
                            print(
                                Fore.YELLOW
                                + f"[Calibrate] refused: need >=5 anchor-fired and >=5 anchor-unfired turns "
                                + f"carrying {target_stream} and every anchor stream ({anchor_label}) "
                                + f"(have {n_rows} qualifying rows).\n"
                                + "(Type ':queue' to auto-retry this in the background)"
                                + Style.RESET_ALL
                            )
                            last_refused_calibration = user_input[len(":calibrate"):].strip()
                        else:
                            new_v, prior_v = tuner.set_calibrated(target_trigger, v, tuner.get("calibration_gain", 0.5))
                            fired_word = "every anchor fired" if len(anchors) > 1 else f"{anchor_label}-fired"
                            moved = "adopted" if abs(new_v - v) < 1e-9 else f"moved {round(prior_v,4)}->{round(new_v,4)} toward"
                            print(
                                Fore.GREEN
                                + f"[Calibrate] {target_trigger} {moved} cut {round(v, 4)} -- the {target_stream} cut "
                                + f"between turns where {fired_word} and the rest "
                                + f"({n_rows} qualifying rows; probes remain minted hypotheses)."
                                + Style.RESET_ALL
                            )
                        continue
                    elif not user_input.startswith(":tune"):
                        trig = resolve_calibrate_trigger(cal_name, tuner)
                        if trig is None:
                            print(Fore.YELLOW + f"[Calibrate] unknown name '{cal_name}'.{did_you_mean(cal_name, calibratable_names(tuner))} Bare :calibrate lists them." + Style.RESET_ALL)
                            continue
                        cal_name = trig.name
                        if len(trig.signals) < 10:
                            print(
                                Fore.YELLOW
                                + f"[Calibrate] refused: only {len(trig.signals)} observed signals for "
                                + f"'{cal_name}' (need 10). A bar set from a handful of points is a guess.\n"
                                + "(Type ':queue' to auto-retry this in the background)"
                                + Style.RESET_ALL
                            )
                            last_refused_calibration = user_input[len(":calibrate"):].strip()
                            continue
                        cal_pct = 50.0
                        if len(cargs) >= 2:
                            try:
                                cal_pct = float(cargs[1])
                            except ValueError:
                                print(Fore.YELLOW + "[Calibrate] percentile must be a number." + Style.RESET_ALL)
                                continue
                        if not (0.0 <= cal_pct <= 100.0):
                            print(Fore.YELLOW + "[Calibrate] rejected: percentile must be in [0, 100]." + Style.RESET_ALL)
                            continue
                        prior_v = trig.value
                        v = tuner.calibrate(cal_name, cal_pct, tuner.get("calibration_gain", 0.5))
                        moved = f"= {round(v, 4)}" if trig.calibrations <= 1 else f"moved {round(prior_v,4)}->{round(v,4)} toward p{cal_pct:g}"
                        print(
                            Fore.GREEN
                            + f"[Calibrate] {cal_name} {moved} (p{cal_pct:g} of {len(trig.signals)} observed signals)."
                            + Style.RESET_ALL
                        )
                        continue
            if user_input.startswith(":queue"):
                qargs = user_input[len(":queue"):].strip()
                if not qargs:
                    if not queued_calibrations:
                        if last_refused_calibration:
                            queued_calibrations.append(last_refused_calibration)
                            print(Fore.GREEN + f"[Queue] Added: :calibrate {last_refused_calibration} (will silently retry every turn)." + Style.RESET_ALL)
                            last_refused_calibration = None
                        else:
                            print(Fore.CYAN + "[Queue] no calibrations queued." + Style.RESET_ALL)
                    else:
                        for idx, qcmd in enumerate(queued_calibrations):
                            print(Fore.CYAN + f"  [{idx}] :calibrate {qcmd}" + Style.RESET_ALL)
                    continue
                if qargs.lower() == "clear":
                    queued_calibrations.clear()
                    print(Fore.CYAN + "[Queue] cleared." + Style.RESET_ALL)
                    continue
                if qargs.lower().startswith("drop "):
                    try:
                        idx = int(qargs[5:].strip())
                        popped = queued_calibrations.pop(idx)
                        print(Fore.CYAN + f"[Queue] dropped: :calibrate {popped}" + Style.RESET_ALL)
                    except Exception:
                        pass
                    continue
                cmd_to_queue = qargs[len("calibrate "):].strip() if qargs.lower().startswith("calibrate ") else qargs
                if cmd_to_queue not in queued_calibrations:
                    queued_calibrations.append(cmd_to_queue)
                    print(Fore.GREEN + f"[Queue] Added: :calibrate {cmd_to_queue} (will silently retry every turn)." + Style.RESET_ALL)
                continue
            if user_input.startswith(":tune"):
                if " dynamic " in user_input.lower():
                    # :tune <target> dynamic <signed mix> [mult]
                    #   target : a knob, OR a probe (drives probe_<name>'s threshold,
                    #            never a bare shadow knob).
                    #   mix    : a single probe/stream, or a signed mix parsed like
                    #            :probe compose (e.g. +ambiguity-consensus). Each turn
                    #            the target = mult * Sigma(weight * stream) over the mix.
                    parts = user_input[len(":tune"):].strip().split()
                    if parts and parts[0].upper() in ("CHOICE", "CHOOSE", "AUTO"):
                        resolved = resolve_probe_choice(parts[0], calibratable_names(tuner), model=model, config=config, action_name="tune")
                        if not resolved:
                            continue
                        parts[0] = resolved
                    try:
                        dyn_idx = [p.lower() for p in parts].index("dynamic")
                    except ValueError:
                        dyn_idx = -1
                    if dyn_idx > 0 and dyn_idx < len(parts) - 1:
                        target = parts[0]
                        rest = parts[dyn_idx + 1:]
                        # Peel a trailing standalone multiplier (number|auto|pNN) off
                        # the end; whatever remains is the mix expression.
                        mult_str = None
                        if len(rest) >= 2 and re.match(r"^(auto|p\d+(?:\.\d+)?|-?\d+(?:\.\d+)?)$", rest[-1].lower()):
                            mult_str = rest[-1].lower()
                            rest = rest[:-1]
                        terms, perr = parse_compose_expr(" ".join(rest))
                        if perr:
                            print(Fore.RED + f"[Tune] Could not parse dynamic expression near '{perr}'." + Style.RESET_ALL)
                            continue
                        # Resolve by CHECKING, not forcing (mirrors the static :tune
                        # path): an exact-name trigger -- a real knob -- wins; a name
                        # that is ONLY a probe drives probe_<name>'s threshold. When a
                        # name is both (e.g. memory_alpha knob vs probe_memory_alpha),
                        # the knob wins and the overlap is surfaced; name 'probe_<x>'
                        # outright to drive the probe threshold instead.
                        if target in tuner.triggers:
                            bind_key = target
                            if f"probe_{target}" in tuner.triggers:
                                print(Fore.CYAN + f"[Tune] '{target}' names both a knob and a probe -- steering the KNOB (use 'probe_{target}' for the probe threshold)." + Style.RESET_ALL)
                        elif f"probe_{target}" in tuner.triggers:
                            bind_key = f"probe_{target}"
                        else:
                            bind_key = target
                        single = terms[0][1] if len(terms) == 1 else None
                        if mult_str is None:
                            mult_str = "auto" if single else "1.0"
                        mult = 1.0
                        if mult_str == "auto" or mult_str.startswith("p"):
                            if single:
                                pct = 80.0
                                if mult_str.startswith("p"):
                                    try:
                                        pct = float(mult_str[1:])
                                    except ValueError:
                                        pass
                                pct = min(max(pct, 0.0), 100.0)
                                ptrig = tuner.triggers.get(f"probe_{single}")
                                if ptrig and ptrig.signals:
                                    data = sorted(ptrig.signals)
                                    p_val = data[int(round((pct / 100.0) * (len(data) - 1)))]
                                    if p_val > 0:
                                        cur = tuner.get(bind_key, 0.0)
                                        mult = cur / p_val
                                        print(Fore.CYAN + f"[Tune] Auto-multiplier {mult:.4f} (maps {single} P{pct:g} {p_val:.3f} to current {bind_key} {cur:.3f})." + Style.RESET_ALL)
                                    else:
                                        print(Fore.YELLOW + f"[Tune] Auto failed ({single} P{pct:g} <= 0). Give a numeric multiplier." + Style.RESET_ALL)
                                        continue
                                else:
                                    print(Fore.YELLOW + f"[Tune] Auto failed (no history for '{single}'). Give a numeric multiplier." + Style.RESET_ALL)
                                    continue
                            else:
                                print(Fore.YELLOW + "[Tune] Auto-multiplier needs a single probe's history; using 1.0 for the mix." + Style.RESET_ALL)
                        else:
                            try:
                                mult = float(mult_str)
                            except ValueError:
                                print(Fore.RED + "[Tune] Multiplier must be a number, 'auto', or 'pNN'." + Style.RESET_ALL)
                                continue
                        tuner_bindings[bind_key] = (terms, mult)
                        label = " ".join(f"{'-' if w < 0 else '+'}{abs(w):g}*{n}" for w, n in terms)
                        tgt_note = f"probe '{target}' threshold" if bind_key.startswith("probe_") else f"knob '{bind_key}'"
                        print(Fore.GREEN + f"[Tune] Dynamic binding: {tgt_note} = {mult:.4f} * ({label}) every turn." + Style.RESET_ALL)
                        continue

                targs = user_input[len(":tune"):].split()
                if targs and targs[0].upper() in ("CHOICE", "CHOOSE", "AUTO"):
                    resolved = resolve_probe_choice(targs[0], calibratable_names(tuner), model=model, config=config, action_name="tune")
                    if not resolved:
                        continue
                    targs[0] = resolved
                if not targs:
                    rows = tuner.summary()
                    if not rows:
                        print(Fore.YELLOW + "[Tune] No triggers registered yet." + Style.RESET_ALL)
                    for s in rows:
                        line = (
                            f"  {s['name']}: {s['kind']} value={s['value']} [{s['comparator']}] "
                            f"fire_rate={s['fire_rate']} "
                            f"signal(min/med/max)={s['signal_min']}/{s['signal_med']}/{s['signal_max']} "
                            f"n={s['n_signals']}"
                        )
                        if s.get("n_credited"):
                            # lift>0 means firing beat not-firing on the outcome:
                            # the honest "is interaction productive" readout.
                            line += (
                                f" | outcome fired={s['fired_outcome']} unfired={s['unfired_outcome']} "
                                f"lift={s['lift']} (n={s['n_credited']})"
                            )
                        print(Fore.CYAN + line + Style.RESET_ALL)
                elif len(targs) >= 2 and targs[1].lower() == "auto":
                    if targs[0] == "conversation_productive" and len(targs) >= 3 and targs[2].lower() == "intent":
                        # Bar set RELATIVE TO INTENT-SHAPING: the sense cut that
                        # separates turns which settled intent from turns which
                        # did not (midpoint of the two group medians).
                        trig = tuner.triggers.get("intent_settling")
                        pairs = list(trig.outcomes) if trig is not None else []
                        v = intent_relative_threshold(pairs)
                        if v is None:
                            n_set = sum(1 for sig, _ in pairs if sig > 0)
                            print(
                                Fore.YELLOW
                                + f"[Tune] conversation_productive unchanged: need >=5 intent-settling and >=5 "
                                + f"non-settling turns with sense (have {n_set} and {len(pairs) - n_set}). Talk more."
                                + Style.RESET_ALL
                            )
                        else:
                            tuner.set("conversation_productive", v)
                            print(
                                Fore.GREEN
                                + f"[Tune] conversation_productive = {round(v, 4)} -- the sense cut between "
                                + f"intent-settling and non-settling turns ({len(pairs)} paired turns)."
                                + Style.RESET_ALL
                            )
                        continue
                    if targs[0] == "steer_cap_fraction":
                        # Data-informed, deterministic: exact percentile of the
                        # attempted push/residual ratios _cap_steer observed.
                        from invariants import engine as _engine
                        pct = float(targs[2]) if len(targs) >= 3 else None
                        v = _engine.calibrate_steer_cap_fraction(percentile=pct)
                        if v is None:
                            stats = _engine.steer_telemetry_stats()
                            print(
                                Fore.YELLOW
                                + f"[Tune] steer_cap_fraction unchanged: only {stats['n']} observed pushes "
                                + f"(need {_engine.STEER_CAP_MIN_N}). Generate first, calibrate after."
                                + Style.RESET_ALL
                            )
                        else:
                            tuner.set("steer_cap_fraction", v)
                            used = _engine.STEER_CAP_PERCENTILE if pct is None else pct
                            stats = _engine.steer_telemetry_stats()
                            print(
                                Fore.GREEN
                                + f"[Tune] steer_cap_fraction = {round(v, 4)} from data "
                                + f"(p{used:g} of {stats['n']} attempted ratios, clip_rate was {stats['clip_rate']})"
                                + Style.RESET_ALL
                            )
                        continue
                    if targs[0] == "steer_band":
                        # Data-informed, deterministic: acceptance-aware per-layer
                        # outcomes from the steer map, not an asserted window.
                        # Evidence lanes: gold (scored benchmarks), conversation
                        # (live productivity reads), or any (both, composition shown).
                        from invariants import engine as _engine
                        min_events, basis, band_evidence = 8, "any", "any"
                        for extra in targs[2:]:
                            token = extra.lower()
                            if token in {"gold", "conversation", "any"}:
                                basis = token
                            elif token in {"synthesis", "layersteer", "layer_steer"}:
                                band_evidence = "layer_steer" if token != "synthesis" else "synthesis"
                            else:
                                try:
                                    min_events = int(float(extra))
                                except ValueError:
                                    pass
                        n_layers = getattr(model, "n_layers", None) or len(model.model.model.layers)
                        suggestion = steer_map.suggest_band(
                            n_layers, min_events=min_events, basis=basis, evidence=band_evidence
                        )
                        if suggestion is None:
                            print(
                                Fore.YELLOW
                                + f"[Tune] steer_band unchanged: no layer clears the evidence bar on the "
                                + f"'{basis}' lane / '{band_evidence}' evidence (min {min_events} labeled, >= overall success). "
                                + "Talk more (conversation lane), import labeled runs (gold lane), or "
                                + "run the layer sweep (:tune steer_layer_sweep 1) for transfer-free evidence."
                                + Style.RESET_ALL
                            )
                        else:
                            _engine.set_steer_band(suggestion["lo"], suggestion["hi"])
                            tuner.set("steer_band_lo", suggestion["lo"])
                            tuner.set("steer_band_hi", suggestion["hi"])
                            print(
                                Fore.GREEN
                                + f"[Tune] steer_band = ({suggestion['lo']:.2f}, {suggestion['hi']:.2f}) from data "
                                + f"(layers {suggestion['eligible_layers']}, "
                                + f"{suggestion['labeled_events']} labeled events {suggestion['labeled_by_basis']} "
                                + f"evidence={suggestion['labeled_by_evidence']}, "
                                + f"overall success {suggestion['overall_success_rate']:.2f})"
                                + Style.RESET_ALL
                            )
                        continue
                    if targs[0] in ("steer_band_lo", "steer_band_hi"):
                        print(
                            Fore.YELLOW
                            + "[Tune] Calibrate the band from outcomes with ':tune steer_band auto' "
                            + "(lo/hi move together), or set a value directly."
                            + Style.RESET_ALL
                        )
                        continue
                    route, reason = calibration_policy(targs[0])
                    if route == "reject":
                        print(Fore.YELLOW + f"[Tune] calibration rejected for '{targs[0]}': {reason}." + Style.RESET_ALL)
                        continue
                    pct = float(targs[2]) if len(targs) >= 3 else 80.0
                    v = tuner.calibrate(targs[0], pct, tuner.get("calibration_gain", 0.5))
                    if v is None:
                        print(Fore.RED + f"[Tune] Unknown trigger '{targs[0]}'.{did_you_mean(targs[0], tuner.triggers)}" + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + f"[Tune] {targs[0]} calibrated toward p{pct:g} = {round(v, 4)}" + Style.RESET_ALL)
                elif len(targs) >= 2:
                    if targs[0].lower() == "off":
                        targs[0], targs[1] = targs[1], "0.0"
                    elif targs[1].lower() == "off":
                        targs[1] = "0.0"
                        
                    if targs[0].startswith("probe_") or (targs[0] in probes and targs[0] not in tuner.triggers):
                        pname = targs[0].replace("probe_", "")
                        if targs[1] == "0.0":
                            if pname in probes:
                                probes[pname]["chatty"] = False
                                print(Fore.CYAN + f"[Tune] Redirected: muted console output for probe '{pname}' (use ':probe drop {pname}' to remove it entirely)." + Style.RESET_ALL)
                            else:
                                print(Fore.YELLOW + f"[Tune] No active probe named '{pname}'." + Style.RESET_ALL)
                        else:
                            if targs[1].replace("probe_", "") in probes:
                                print(Fore.YELLOW + f"[Tune] Did you mean to dynamically tie the threshold? Use: :tune {targs[0]} dynamic {targs[1]}" + Style.RESET_ALL)
                            else:
                                print(Fore.YELLOW + f"[Tune] '{targs[0]}' is a probe. To mute it use ':tune {targs[0]} off', or to drop it use ':probe drop {pname}'." + Style.RESET_ALL)
                        continue

                    # Setting a brand-new name that collides with a probe would
                    # create a junk knob shadowing it -- refuse and redirect.
                    if targs[0] not in tuner.triggers and f"probe_{targs[0]}" in tuner.triggers:
                        print(Fore.YELLOW + f"[Tune] '{targs[0]}' is a probe, not a knob -- :tune would create a shadow. Use :label {targs[0]} pos|neg or :calibrate {targs[0]} <anchor>." + Style.RESET_ALL)
                    else:
                        if targs[1].upper() in ("CHOICE", "CHOOSE"):
                            print(Fore.CYAN + f"[Choice] Asking the model to select a value for '{targs[0]}'..." + Style.RESET_ALL)
                            prompt = (
                                f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                                f"You are configuring the cognitive tuning parameter '{targs[0]}'.\n"
                                f"Select an appropriate numerical value (float). Output ONLY the number and nothing else.<|eot_id|>"
                                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                            )
                            # NOTE: no local `import generate_agentic_text` here --
                            # it is imported at module level. A local import would
                            # make the name function-local across all of run() and
                            # UnboundLocalError every other generate call.
                            sug = generate_agentic_text(model, instruction=prompt, config=config, pre_formatted=True, max_new_tokens=10)
                            try:
                                targs[1] = str(float(sug.strip()))
                                print(Fore.GREEN + f"[Choice] The model chose value: {targs[1]}" + Style.RESET_ALL)
                            except ValueError:
                                print(Fore.YELLOW + f"[Choice] Model output an invalid number '{sug.strip()}'. Aborting." + Style.RESET_ALL)
                                continue

                        try:
                            v = tuner.set(targs[0], float(targs[1]))
                            print(Fore.GREEN + f"[Tune] {targs[0]} = {round(v, 4)}" + Style.RESET_ALL)
                        except ValueError:
                            print(Fore.RED + "[Tune] Value must be a number." + Style.RESET_ALL)
                else:
                    print(Fore.YELLOW + "[Tune] Usage: :tune | :tune <name> <value> | :tune <name> auto [percentile]" + Style.RESET_ALL)
                continue
            if not user_input.strip():
                continue
            # A :prefixed line that reached here matched no command handler.
            # Don't burn a generation deflecting into a generic essay: if it's a
            # typo, suggest the real command; otherwise offer to INVENT it as a
            # tool -- mint a probe for the name (framings stay operator-authored).
            if user_input.lstrip().startswith(":") and not reading_turn_source:
                cmd_word = user_input.split()[0].lower()
                guess = did_you_mean(cmd_word, KNOWN_COMMANDS)
                if guess:
                    print(Fore.YELLOW + f"[Command] '{cmd_word}' isn't a command.{guess}" + Style.RESET_ALL)
                else:
                    stem = re.sub(r"[^a-z0-9_]", "_", cmd_word.lstrip(":"))[:40].strip("_") or "it"
                    print(
                        Fore.CYAN
                        + f"[Command] '{cmd_word}' isn't a command. To invent it as a tool, mint a probe for "
                        + f"'{stem}': :probe {stem} <with it> || <without it>  (or bare :probe {stem} to draft framings)."
                        + Style.RESET_ALL
                    )
                continue

            memory_tool_result = pending_memory_tool_result
            orientation_tool_result = pending_orientation_tool_result
            claimmap_tool_result = pending_claimmap_tool_result
            claimmap_steer_delta = pending_claimmap_steer_delta
            methodmap_tool_result = pending_methodmap_tool_result
            document_tool_result = pending_document_tool_result
            sandbox_tool_result = pending_sandbox_tool_result
            game_tool_result = pending_game_tool_result
            if active_game and not game_tool_result:
                sys_prompt = active_game_state.get("system_prompt", "The operator expects you to play it with them. To end the game, output <<GAME_END>>.")
                game_tool_result = f"[Game Active: {active_game}] A game is currently active. {sys_prompt}"
            help_tool_result = pending_help_tool_result
            if help_exposed and not help_tool_result:
                help_tool_result = "[Help available] You may emit <<HELP>> at any time to see what you can run yourself vs. what only the operator can run."
            # Consequences of the model's LAST words arrive as THIS turn's
            # context; the same-turn tag requests below add to the list.
            turn_impacts = impact_state["pending"]
            impact_state["pending"] = []
            sweep_state["fires"] = []  # single-layer steers applied this turn
            turn_row = {}              # this turn's stream values (paired calibration substrate)
            pending_memory_tool_result = None
            pending_orientation_tool_result = None
            pending_claimmap_tool_result = None
            pending_claimmap_steer_delta = None
            pending_methodmap_tool_result = None
            pending_document_tool_result = None
            pending_sandbox_tool_result = None
            pending_game_tool_result = None
            pending_help_tool_result = None
            prompt = build_prompt(
                user_input,
                memory_tool_result=memory_tool_result,
                orientation_tool_result=orientation_tool_result,
                claimmap_tool_result=claimmap_tool_result,
                methodmap_tool_result=methodmap_tool_result,
                sandbox_tool_result=sandbox_tool_result,
                document_tool_result=document_tool_result,
                game_tool_result=game_tool_result,
                help_tool_result=help_tool_result,
                session_context=session_context if session_context_enabled else None,
                active_u_name=_last_replace_name if ('_last_replace_name' in locals() and _last_replace_name) else "operator",
            )
            # Honest attribution in the permanent record: a reading turn is the
            # model's own act (it occupies the user slot in the chat template,
            # but the operator never typed it).
            memory.append_turn(
                "user",
                user_input,
                tags=[
                    "reading_turn" if reading_turn_source else "operator_input",
                    _last_replace_name if ('_last_replace_name' in locals() and _last_replace_name) else "operator"
                ],
                provenance={
                    "spoken_by": "model_reading" if reading_turn_source else "operator",
                    "document_source": reading_turn_source,
                    "memory_tool_result_provided": bool(memory_tool_result),
                    "orientation_tool_result_provided": bool(orientation_tool_result),
                    "claimmap_tool_result_provided": bool(claimmap_tool_result),
                    "methodmap_tool_result_provided": bool(methodmap_tool_result),
                },
            )
            if memory_tool_result:
                memory.append_event(
                    "memory_tool_result_provided",
                    text=memory_tool_result,
                    tags=["memory_tool"],
                    provenance={"current_input": user_input[:240]},
                )
            if orientation_tool_result:
                memory.append_event(
                    "orientation_tool_result_provided",
                    text=orientation_tool_result,
                    tags=["orientation_tool"],
                    provenance={"current_input": user_input[:240]},
                )
            if claimmap_tool_result:
                memory.append_event(
                    "claimmap_tool_result_provided",
                    text=claimmap_tool_result,
                    tags=["claimmap_tool", "activation_measurement"],
                    provenance={"current_input": user_input[:240]},
                )
            if methodmap_tool_result:
                memory.append_event(
                    "methodmap_tool_result_provided",
                    text=methodmap_tool_result,
                    tags=["methodmap_tool"],
                    provenance={"current_input": user_input[:240]},
                )
            if document_tool_result:
                memory.append_event(
                    "document_tool_result_provided",
                    text=document_tool_result[:400],
                    tags=["document"],
                    provenance={
                        "current_input": user_input[:240],
                        "sha256": (doc_session or {}).get("sha256"),
                        "chunk_index": (doc_session or {}).get("cursor"),
                    },
                )
            if sandbox_tool_result:
                memory.append_event(
                    "sandbox_tool_result_provided",
                    text=sandbox_tool_result[:400],
                    tags=["sandbox_tool"],
                    provenance={"current_input": user_input[:240]},
                )

            print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
            synthesis_records = []

            # Evaluate dynamic tuner bindings before generation
            if tuner_bindings and turn_log:
                last_turn = turn_log[-1]
                for bind_key, (terms, mult) in list(tuner_bindings.items()):
                    vals = []
                    for w, name in terms:
                        # Resolve each mix term to last turn's stream: a probe
                        # (probe_<name>), a bare named stream, or a phenomenality
                        # dimension (phen_<name>).
                        s = last_turn.get(f"probe_{name}")
                        if s is None:
                            s = last_turn.get(name)
                        if s is None:
                            s = last_turn.get(f"phen_{name}")
                        if s is None:
                            vals = None  # a term has no reading yet -> skip this turn
                            break
                        vals.append(w * float(s))
                    if vals is not None:
                        mix_sig = sum(vals)
                        new_val = mix_sig * mult
                        tuner.set(bind_key, new_val)
                        label = " ".join(f"{'-' if w < 0 else '+'}{abs(w):g}*{n}" for w, n in terms)
                        print(Fore.CYAN + f"[Tune] Dynamic binding: {bind_key} set to {new_val:.4f} ({mult:g} * [{label}] = {mix_sig:+.3f})." + Style.RESET_ALL, flush=True)

            # Live steering surface: cap/band/fraction move with the tuner each
            # turn, so a :tune takes effect on the very next generation.
            _sync_steer_tunables(tuner, config)
            # Proven-expert routing: push the weight and a fresh per-expert
            # success snapshot into config so the ToT selection favors experts
            # that have earned it. Only recompute (an O(events) aggregate) when
            # the feature is actually on.
            try:
                config.expert_proof_weight = max(0.0, float(tuner.get("expert_proof_weight", 0.0) or 0.0))
            except (TypeError, ValueError):
                config.expert_proof_weight = 0.0
            config.expert_proof_scores = (
                compute_expert_proof_scores(steer_map) if config.expert_proof_weight > 0.0 else {}
            )
            claimmap_alpha_used = tuner.get("claimmap_alpha", 0.0) if claimmap_steer_delta else 0.0
            turn_sweep_layers = None
            if claimmap_steer_delta and claimmap_alpha_used > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
                sweep_layers, sweep_drifts, sweep_lawfulness = _sweep_layers_for("claimmap", claimmap_steer_delta)
                if sweep_layers:
                    turn_sweep_layers = sweep_layers
                    for _sl in sweep_layers:
                        sweep_state["fires"].append(("claimmap", _sl, claimmap_alpha_used, sweep_drifts.get(_sl), len(sweep_layers), sweep_lawfulness))
                    print(
                        Fore.MAGENTA
                        + "[Steer] layer sweep: this turn's claimmap steer pushes "
                        + "+".join(f"L{_sl}" for _sl in sweep_layers)
                        + f" (width {len(sweep_layers)})"
                        + Style.RESET_ALL
                    )
            steer_handles = (
                claimmap_steer_handles(model, claimmap_steer_delta, alpha=claimmap_alpha_used, layers=turn_sweep_layers)
                if claimmap_steer_delta else []
            )
            if not steer_handles:
                claimmap_alpha_used = 0.0  # nothing actually applied this turn
            # Prioritize steer, bounded by the same envelope. Target precedence:
            # pinned MIX (lift-weighted combination) > pinned single probe
            # (toward, or away if sign -1) > AUTO (top of the ranking, signed by
            # lift). OFF at alpha 0.
            prio_alpha = tuner.get("prioritize_alpha", 0.0)
            prio_steered = None
            if prio_alpha and prio_alpha > 0 and probes:
                from invariants.engine import _steer_handles as _p_steer
                _mix = prioritize_pin.get("mix")
                _pin = prioritize_pin.get("probe")
                _pdir, _label, _psign = None, None, 1.0
                if _mix:
                    _md = build_priority_mix_direction(model, _mix, probes, tuner)
                    if _md:
                        _pdir, _label, _psign = _md, "mix(" + "+".join(_mix) + ")", 1.0
                elif _pin and _pin in probes:
                    _pdir = probes[_pin].get("direction") or None
                    _label, _psign = _pin, float(prioritize_pin.get("sign", 1.0) or 1.0)
                else:
                    _pr = rank_probes(probes, tuner)
                    if _pr and _pr[0]["priority"] > 0 and _pr[0]["lift"] is not None:
                        _pdir = probes[_pr[0]["name"]].get("direction") or None
                        _label = _pr[0]["name"]
                        _psign = 1.0 if _pr[0]["lift"] >= 0 else -1.0
                if _pdir:
                    try:
                        steer_handles.extend(_p_steer(model, _pdir, list(_pdir.keys()), prio_alpha * _psign))
                        prio_steered = (_label, _psign)
                    except Exception:
                        prio_steered = None
            if prio_steered is not None:
                _dir_word = "along" if str(prio_steered[0]).startswith("mix(") else ("toward" if prio_steered[1] > 0 else "away from")
                print(
                    Fore.MAGENTA
                    + f"[Prioritize] steering {_dir_word} {prio_steered[0]} (alpha {round(prio_alpha,4)})."
                    + Style.RESET_ALL
                )
            # Exposed-probe steer: same envelope/mechanism as prioritize, but the
            # target is the SET of probes the operator exposed to the model --
            # lift-weighted, so evidence sets each one's degree and sign. Off at 0,
            # idle until those probes have credited lift.
            exposed_alpha = tuner.get("exposed_probe_alpha", 0.0)
            if exposed_alpha and exposed_alpha > 0 and probes:
                exposed_names = sorted(n for n in probes if probes[n].get("exposed"))
                if exposed_names:
                    from invariants.engine import _steer_handles as _e_steer
                    _edir = build_priority_mix_direction(model, exposed_names, probes, tuner)
                    if _edir:
                        try:
                            steer_handles.extend(_e_steer(model, _edir, list(_edir.keys()), exposed_alpha))
                            print(
                                Fore.MAGENTA
                                + f"[Exposed] steering along {len(exposed_names)} exposed probe(s) "
                                + f"({', '.join(exposed_names)}) at alpha {round(exposed_alpha, 4)}."
                                + Style.RESET_ALL
                            )
                        except Exception:
                            pass
                    else:
                        print(
                            Fore.YELLOW
                            + "[Exposed] exposed_probe_alpha is set but no exposed probe has credited "
                            + "lift yet -- steering idle until they accrue outcomes."
                            + Style.RESET_ALL
                        )
            # Clock start: wall-time for THIS turn's generation (all
            # regenerations included), snapshotted before probe-scoring so the
            # reading reflects generation, not instrumentation. The peak-memory
            # window is reset here so it captures this turn's spike (VRAM on
            # CUDA, process RSS on CPU-only).
            import time as _time
            _gen_t0 = _time.perf_counter()
            _reset_memory_peak()
            try:
                response, telemetry = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,  # Enables visible trace logging.
                    pre_formatted=True,
                    mid_chunk_hook=tool_sense,  # tools fire mid-thought, between chunks
                    return_telemetry=True,
                )
            finally:
                for h in steer_handles:
                    h.remove()
            model_memory_query = extract_memory_query(response)
            model_claimmap_payload = extract_claimmap_payload(response)
            model_methodmap_query = extract_methodmap_query(response)
            model_doc_query = extract_doc_query(response)
            model_probe_query = extract_probe_query(response)
            model_cmd_requests = extract_cmd_requests(response)
            model_game_propose = extract_game_propose(response)
            model_game_accept = extract_game_accept(response)
            model_game_decline = extract_game_decline(response)
            model_game_end = extract_game_end(response)
            model_game_exposed, model_game_hidden = extract_game_expose_hide(response)
            if extract_help_request(response):
                pending_help_tool_result = build_model_help_text(list_solve_macros(), exposed_commands, exposed_knobs, hidden_commands)
                print(Fore.CYAN + "[Help] The model asked for help; serving the tool/command reference next turn." + Style.RESET_ALL)

            if model_game_exposed or model_game_hidden:
                print(Fore.MAGENTA + Style.BRIGHT + "\n[Game State Changed]" + Style.RESET_ALL)
                for p in model_game_exposed:
                    if p in probes:
                        probes[p]["exposed"] = True
                        pf = os.path.join(ROOT, "invariants", "out", "probes", f"{p}.pt")
                        if os.path.exists(pf):
                            data = torch.load(pf, weights_only=True)
                            data["exposed"] = True
                            torch.save(data, pf)
                        print(Fore.GREEN + f"  [+] EXPOSED: {p}" + Style.RESET_ALL)
                for p in model_game_hidden:
                    if p in probes:
                        probes[p]["exposed"] = False
                        pf = os.path.join(ROOT, "invariants", "out", "probes", f"{p}.pt")
                        if os.path.exists(pf):
                            data = torch.load(pf, weights_only=True)
                            data["exposed"] = False
                            torch.save(data, pf)
                        print(Fore.RED + f"  [-] HIDDEN: {p}" + Style.RESET_ALL)

            if model_game_propose:
                model_proposed_game = model_game_propose
                print(Fore.MAGENTA + f"\n[Game] The model proposes playing '{model_proposed_game}'. Type ':game {model_proposed_game}' to accept." + Style.RESET_ALL)
            
            if model_game_accept:
                if user_proposed_game == model_game_accept:
                    print(Fore.GREEN + f"\n[Game] The model accepted your proposal to play {model_game_accept}!" + Style.RESET_ALL)
                    user_proposed_game = None
                    active_game = model_game_accept
                    
                    gpath = os.path.join(ROOT, "games", f"{active_game}.py")
                    if os.path.isfile(gpath):
                        print(Fore.MAGENTA + f"[Game] Initializing {active_game}..." + Style.RESET_ALL)
                        try:
                            with open(gpath, "r", encoding="utf-8") as _gf:
                                gcode = _gf.read()
                            _acfg = load_game_config(os.path.join(ROOT, "games", f"{active_game}_rules.json"))
                            exec_globals = globals().copy()
                            exec_globals["GAME_RULES"] = _acfg["rules"]
                            exec_globals["GAME_CONFIG"] = _acfg
                            exec_globals["game_args"] = list(user_proposed_game_args or [])
                            exec_globals["active_game_state"] = active_game_state
                            exec_globals["probes"] = probes
                            exec_globals["run_game"] = _run_game_ref
                            exec(gcode, exec_globals)
                        except Exception as e:
                            import traceback
                            print(Fore.RED + f"[Game Error] {e}\n{traceback.format_exc()}" + Style.RESET_ALL)
            
            if model_game_decline:
                # The model's own agency: it may refuse a proposed game, or back
                # out of one that was force-started, by emitting <<GAME_DECLINE>>.
                declined = None
                if user_proposed_game and model_game_decline in ("*", user_proposed_game):
                    declined, user_proposed_game = user_proposed_game, None
                elif active_game and model_game_decline in ("*", active_game):
                    declined, active_game = active_game, None
                if declined:
                    print(Fore.YELLOW + f"\n[Game] The model declined {declined}." + Style.RESET_ALL)
                    pending_game_tool_result = None
                    memory.append_event("game_declined", tags=["game", "self_concept"], provenance={"game": declined})

            if model_game_end:
                if active_game:
                    print(Fore.CYAN + f"\n[Game] The model ended the active game: {active_game}." + Style.RESET_ALL)
                    prize = active_game_state.get("prize_command")
                    if prize:
                        _stage_for_accept(prize, why=f"'{active_game}' prize")
                    active_game = None
                    active_game_state = {}
            model_memory_tool_result = None
            model_claimmap_tool_result = None
            model_methodmap_tool_result = None
            model_doc_tool_result = None
            model_probe_tool_result = None
            model_cmd_tool_result = None
            if model_cmd_requests:
                model_cmd_tool_result = _run_exposed_command_tool(model_cmd_requests)
                turn_impacts.append({
                    "cause": "asked for an exposed command tool",
                    "effect": "command request was staged or queued according to exposure mode",
                })
                print(Fore.CYAN + "\n[Command Tool]\n" + model_cmd_tool_result + Style.RESET_ALL + "\n")
                prompt = build_prompt(
                    user_input,
                    memory_tool_result=memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    sandbox_tool_result=sandbox_tool_result,
                    document_tool_result=document_tool_result,
                    probe_tool_result=None,
                    command_tool_result=model_cmd_tool_result,
                    game_tool_result=game_tool_result,
                    help_tool_result=help_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,
                    pre_formatted=True,
                    return_telemetry=True,
                )
                model_memory_query = extract_memory_query(response)
                model_claimmap_payload = extract_claimmap_payload(response)
                model_methodmap_query = extract_methodmap_query(response)
                model_doc_query = extract_doc_query(response)
                model_probe_query = extract_probe_query(response)
            if model_doc_query and doc_library and document_tool_result is None:
                # The model asked to keep reading, in its own words -- and its
                # words pick the chunk (reply-mode selection over the query
                # text; honest order-fallback when nothing overlaps). Then a
                # same-turn regeneration, the house pattern for model tags, so
                # it answers WITH the text in hand instead of losing it.
                pick = select_next_chunk(doc_library, model_doc_query, "reply")
                if pick is not None:
                    doc_session = doc_library[pick["session_index"]]
                    doc_session["cursor"] = pick["chunk_index"]
                    doc_session.setdefault("read", set()).add(pick["chunk_index"])
                    record_chunk_read(memory, doc_session, pick["chunk_index"])
                    model_doc_tool_result = (
                        impact_note(f'asked to read more ("{model_doc_query[:80]}")')
                        + "\n"
                        + stage_chunk(doc_session, index=pick["chunk_index"], reply_note=reading_reply_note(pick))
                    )
                    turn_impacts.append(
                        {
                            "cause": f'asked to read more ("{model_doc_query[:60]}")',
                            "effect": f"{doc_session['source_name']} part {pick['chunk_index'] + 1}/"
                                      f"{doc_session['chunk_count']} returned ({pick['mode']})",
                        }
                    )
                    print(
                        Fore.CYAN
                        + f"\n[Doc] Model asked to read more -> {doc_session['source_name']} "
                        + f"part {pick['chunk_index'] + 1}/{doc_session['chunk_count']} ({pick['mode']})."
                        + Style.RESET_ALL
                    )
                else:
                    model_doc_tool_result = (
                        impact_note("asked to read more")
                        + "\nThe library is fully read; nothing unread remains."
                    )
                    print(Fore.CYAN + "\n[Doc] Model asked to read more but the library is fully read." + Style.RESET_ALL)
                prompt = build_prompt(
                    user_input,
                    memory_tool_result=memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    sandbox_tool_result=sandbox_tool_result,
                    document_tool_result=model_doc_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,
                    pre_formatted=True,
                    return_telemetry=True,
                )
                model_memory_query = extract_memory_query(response)
                model_claimmap_payload = extract_claimmap_payload(response)
                model_methodmap_query = extract_methodmap_query(response)
                model_probe_query = extract_probe_query(response)

            if model_memory_query and memory_tool_result is None:
                records = memory.search(model_memory_query, max_records=6, scope=memory.scope)
                model_memory_tool_result = (
                    impact_note(f'asked memory for "{model_memory_query}"')
                    + "\n"
                    + memory.format_tool_result(records)
                )
                turn_impacts.append(
                    {"cause": f'asked memory for "{model_memory_query}"', "effect": f"{len(records)} record(s) returned"}
                )
                memory.append_event(
                    "memory_tool_model_requested",
                    text=model_memory_tool_result,
                    tags=["memory_tool"],
                    provenance={"query": model_memory_query, "records": len(records)},
                )
                print(
                    Fore.CYAN
                    + f"\n[Memory] Model requested lookup: {model_memory_query}\n"
                    + model_memory_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = build_prompt(
                    user_input,
                    memory_tool_result=model_memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,
                    pre_formatted=True,
                    return_telemetry=True,
                )
                model_claimmap_payload = extract_claimmap_payload(response)
                model_methodmap_query = extract_methodmap_query(response)
                model_probe_query = extract_probe_query(response)
            if model_claimmap_payload and claimmap_tool_result is None:
                model_claimmap_steer = None
                try:
                    cm = analyze_claim_pair(model_claimmap_payload, model=model)
                    # felt only reaches the model, unprefixed: it asked with its
                    # own words and the felt map opens by quoting them back --
                    # the causation is already legible without injecting
                    # standardized vocabulary on top.
                    model_claimmap_tool_result = cm.felt
                    model_claimmap_steer = cm.steer_delta
                    telemetry_for_log = cm.telemetry              # raw numbers logged, never in the prompt
                    turn_impacts.append(
                        {"cause": "asked for a claim comparison", "effect": "its own geometry was measured and returned"}
                    )
                except Exception as exc:
                    model_claimmap_tool_result = f"{CLAIMMAP_HEADER}\nvalid=False; error={exc}"
                    telemetry_for_log = model_claimmap_tool_result
                memory.append_event(
                    "claimmap_tool_model_requested",
                    text=telemetry_for_log,
                    tags=["claimmap_tool", "activation_measurement"],
                    provenance={"payload_chars": len(model_claimmap_payload)},
                )
                print(
                    Fore.CYAN
                    + "\n[ClaimMap] Model sensed a comparison:\n"
                    + model_claimmap_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = build_prompt(
                    user_input,
                    memory_tool_result=model_memory_tool_result or memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=model_claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                tag_alpha = tuner.get("claimmap_alpha", 0.0)
                tag_sweep_layers = None
                if model_claimmap_steer and tag_alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
                    _tls, _tds, _tvar = _sweep_layers_for("claimmap", model_claimmap_steer)
                    if _tls:
                        tag_sweep_layers = _tls
                        for _sl in _tls:
                            sweep_state["fires"].append(("claimmap", _sl, tag_alpha, _tds.get(_sl), len(_tls), _tvar))
                steer_handles = (
                    claimmap_steer_handles(model, model_claimmap_steer, alpha=tag_alpha, layers=tag_sweep_layers)
                    if model_claimmap_steer else []
                )
                try:
                    response, telemetry = generate_agentic_text(
                        model,
                        instruction=prompt,
                        config=config,
                        max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                        synthesis_recorder=synthesis_records,
                        chatty_log=True,
                        pre_formatted=True,
                        return_telemetry=True,
                    )
                finally:
                    for h in steer_handles:
                        h.remove()
                model_methodmap_query = extract_methodmap_query(response)
                model_probe_query = extract_probe_query(response)
            if model_methodmap_query and methodmap_tool_result is None:
                model_methodmap_tool_result = (
                    impact_note(f'asked the method map about "{model_methodmap_query}"')
                    + "\n"
                    + format_methodmap_tool_result(memory, model_methodmap_query)
                )
                turn_impacts.append(
                    {"cause": f'asked the method map about "{model_methodmap_query}"', "effect": "methodologies returned"}
                )
                memory.append_event(
                    "methodmap_tool_model_requested",
                    text=model_methodmap_tool_result,
                    tags=["methodmap_tool"],
                    provenance={"query": model_methodmap_query},
                )
                print(
                    Fore.CYAN
                    + f"\n[MethodMap] Model requested method maps: {model_methodmap_query}\n"
                    + model_methodmap_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = build_prompt(
                    user_input,
                    memory_tool_result=model_memory_tool_result or memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=model_claimmap_tool_result or claimmap_tool_result,
                    methodmap_tool_result=model_methodmap_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,
                    pre_formatted=True,
                    return_telemetry=True,
                )
            if model_probe_query and (probes or exposed_knobs):
                # READ-ONLY self-measurement: the model may consult sensors the
                # operator has exposed (:probe expose <name> or :expose <name>)
                # and scalar knobs the operator exposed with :expose <knob>.
                # Reading is allowed; shaping is not. Candidate-text reads score
                # hypothetical words for probes only, without observing, crediting,
                # or touching any rolling history.
                name_part, _, cand_text = _partition_unescaped_pipes(model_probe_query)
                cand_text = cand_text.strip()
                req_names = [re.sub(r"[^a-z0-9_]", "_", n.strip().lower())[:40] for n in name_part.split(",") if n.strip()]
                exposed_probe_all = sorted(n for n in probes if probes[n].get("exposed"))
                exposed_knob_all = sorted(k for k in exposed_knobs if k in tuner.triggers and not _is_shadow_trigger(k, tuner))
                if req_names == ["all"]:
                    req_names = list(exposed_probe_all) + list(exposed_knob_all)
                readable = [n for n in req_names if n in probes and probes[n].get("exposed")]
                readable_knobs = [n for n in req_names if n in exposed_knob_all]
                blocked = [n for n in req_names if n not in readable and n not in readable_knobs]
                probe_lines = []
                if cand_text and readable:
                    from invariants.engine import _inputs as _q_inputs, _hidden_states as _q_hidden, probe_score as _q_score
                    with torch.no_grad():
                        q_ids = _q_inputs(model, cand_text[:600])
                        q_hs = _q_hidden(model, q_ids["input_ids"], q_ids.get("attention_mask"))
                    for n in readable:
                        hist = probes[n]["history"]
                        raw = _q_score(q_hs, probes[n]["direction"])
                        sig = raw - (sum(hist) / len(hist)) if hist else 0.0
                        probe_lines.append(f"- {n} reads {sig:+.3f} on those words, against my recent baseline.")
                else:
                    for n in readable:
                        p_trig = tuner.triggers.get(f"probe_{n}")
                        last_sig = next(
                            (r[f"probe_{n}"] for r in reversed(turn_log) if f"probe_{n}" in r),
                            None,
                        )
                        seg = (
                            f"- {n}: {float(last_sig):+.3f} on my last turn"
                            if last_sig is not None else f"- {n}: no reading yet"
                        )
                        if p_trig is not None:
                            p_st = p_trig.outcome_stats()
                            seg += f" (bar {round(p_trig.value, 4)}"
                            if p_st.get("lift") is not None:
                                seg += f", lift {p_st['lift']}"
                            seg += ")"
                        probe_lines.append(seg + ".")
                for n in readable_knobs:
                    trig = tuner.triggers.get(n)
                    if trig is None:
                        continue
                    st = trig.stats()
                    seg = f"- {n}: knob value {st['value']} ({st['kind']}"
                    if st.get("kind") == "threshold":
                        seg += f", fires when signal {st['comparator']} value"
                        if st.get("fire_rate") is not None:
                            seg += f", fire_rate {st['fire_rate']}"
                    if st.get("observed"):
                        seg += f", observed {st['observed']}"
                    if st.get("lift") is not None:
                        seg += f", lift {st['lift']}"
                    seg += ")"
                    if cand_text:
                        seg += " Candidate words do not rescore knobs; this is current scalar state."
                        probe_lines.append(seg)
                    else:
                        probe_lines.append(seg + ".")
                for n in blocked:
                    probe_lines.append(f"- {n}: not a sensor I can consult.")
                if not probe_lines:
                    exposed_bits = []
                    if exposed_probe_all:
                        exposed_bits.append("probes: " + ", ".join(exposed_probe_all))
                    if exposed_knob_all:
                        exposed_bits.append("knobs: " + ", ".join(exposed_knob_all))
                    probe_lines.append(
                        "- No consultable sensors named. "
                        + (f"I can consult {', '.join(exposed_bits)}." if exposed_bits else "None are exposed to me.")
                    )
                consulted_parts = []
                if readable:
                    consulted_parts.append(f'{", ".join(readable)} probe(s)')
                if readable_knobs:
                    consulted_parts.append(f'{", ".join(readable_knobs)} knob(s)')
                probe_cause = (
                    "consulted my " + " and ".join(consulted_parts) if consulted_parts else "reached for a sensor"
                ) + (" on candidate words" if cand_text and readable else "")
                model_probe_tool_result = (
                    impact_note(probe_cause)
                    + "\n" + PROBE_TOOL_HEADER + "\n" + "\n".join(probe_lines)
                )
                turn_impacts.append({
                    "cause": probe_cause,
                    "effect": f"{len(readable) + len(readable_knobs)} reading(s) returned"
                              + (f", {len(blocked)} refused" if blocked else ""),
                })
                memory.append_event(
                    "probe_tool_model_requested",
                    text=model_probe_tool_result,
                    tags=["probe", "probe_tool", "activation_measurement"],
                    provenance={"query": model_probe_query[:240], "readable": readable, "readable_knobs": readable_knobs, "blocked": blocked},
                )
                print(
                    Fore.CYAN
                    + f"\n[Probe] Model consulted sensors: {model_probe_query[:120]}\n"
                    + model_probe_tool_result
                    + Style.RESET_ALL
                    + "\n"
                )
                prompt = build_prompt(
                    user_input,
                    memory_tool_result=memory_tool_result or model_memory_tool_result,
                    orientation_tool_result=orientation_tool_result,
                    claimmap_tool_result=claimmap_tool_result or model_claimmap_tool_result,
                    methodmap_tool_result=methodmap_tool_result or model_methodmap_tool_result,
                    probe_tool_result=model_probe_tool_result,
                    command_tool_result=model_cmd_tool_result,
                    session_context=session_context if session_context_enabled else None,
                )
                print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                response, telemetry = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,
                    pre_formatted=True,
                    return_telemetry=True,
                )
            if response:
                active_memory_tool_result = memory_tool_result or model_memory_tool_result
                active_orientation_tool_result = orientation_tool_result
                active_claimmap_tool_result = claimmap_tool_result or model_claimmap_tool_result
                active_methodmap_tool_result = methodmap_tool_result or model_methodmap_tool_result
                active_document_tool_result = document_tool_result or model_doc_tool_result
                active_sandbox_tool_result = sandbox_tool_result
                active_probe_tool_result = model_probe_tool_result
                active_command_tool_result = model_cmd_tool_result
                if (
                    (active_memory_tool_result or active_claimmap_tool_result or active_methodmap_tool_result or active_document_tool_result or active_probe_tool_result or active_command_tool_result)
                    and is_tool_only_response(response)
                ):
                    memory.append_event(
                        "tool_loop_retry",
                        text=response,
                        tags=["tool_protocol"],
                        provenance={"current_input": user_input[:240]},
                    )
                    retry_input = (
                        user_input
                        + "\n\n[Tool Protocol Reminder]\n"
                        + "I already have the tool result I asked for; no more tool tags -- I answer now from it."
                    )
                    prompt = build_prompt(
                        retry_input,
                        memory_tool_result=active_memory_tool_result,
                        orientation_tool_result=active_orientation_tool_result,
                        claimmap_tool_result=active_claimmap_tool_result,
                        methodmap_tool_result=active_methodmap_tool_result,
                        sandbox_tool_result=active_sandbox_tool_result,
                        document_tool_result=active_document_tool_result,
                        probe_tool_result=active_probe_tool_result,
                        command_tool_result=active_command_tool_result,
                        session_context=session_context if session_context_enabled else None,
                    )
                    print(Fore.GREEN + Style.BRIGHT + "\nMe: " + Style.RESET_ALL, end="")
                    response, telemetry = generate_agentic_text(
                        model,
                        instruction=prompt,
                        config=config,
                        max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                        synthesis_recorder=synthesis_records,
                        chatty_log=True,
                        pre_formatted=True,
                        return_telemetry=True,
                    )
                response = scrub_unstaged_memory_status(
                    response,
                    memory_tool_result=active_memory_tool_result,
                    orientation_tool_result=active_orientation_tool_result,
                    claimmap_tool_result=active_claimmap_tool_result,
                    methodmap_tool_result=active_methodmap_tool_result,
                    sandbox_tool_result=active_sandbox_tool_result,
                    document_tool_result=active_document_tool_result,
                    probe_tool_result=active_probe_tool_result,
                    command_tool_result=active_command_tool_result,
                )
                if show_timestamps:
                    print(Fore.GREEN + Style.BRIGHT + f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Me: " + Style.RESET_ALL)
                print(response, end="")
                if session_context_enabled:
                    s_score = sense_score(synthesis_records) if synthesis_records else 0.0
                    if s_score is None: s_score = 0.0
                    active_u_name = _last_replace_name if '_last_replace_name' in locals() and _last_replace_name else "user"
                    session_context.append(("user", active_u_name, user_input, s_score))
                    session_context.append(("assistant", "assistant", response, s_score))
                    if len(session_context) > MAX_SESSION_TURNS * 2:
                        if len(session_context) >= 6:
                            min_score = float('inf')
                            min_idx = 0
                            for i in range(0, len(session_context) - 4, 2):
                                pair_score = session_context[i][3] if len(session_context[i]) > 3 else (session_context[i][2] if len(session_context[i]) > 2 else 0.0)
                                if pair_score < min_score:
                                    min_score = pair_score
                                    min_idx = i
                            session_context.pop(min_idx + 1)
                            session_context.pop(min_idx)
                        else:
                            session_context = session_context[-MAX_SESSION_TURNS * 2 :]

                # Run Joined Agents
                for ja in joined_agents:
                    print(Fore.GREEN + Style.BRIGHT + f"\n[{ja.name}]: " + Style.RESET_ALL, end="")
                    
                    _sys_tuner = tuner
                    _sys_probes = probes
                    _sys_tb = tuner_bindings
                    tuner = ja.tuner
                    probes = ja.probes
                    tuner_bindings = ja.tuner_bindings
                    
                    ja_prompt = build_prompt(
                        "[Please respond]", 
                        session_context=session_context if session_context_enabled else None,
                        active_u_name="system"
                    )
                    ja_response = generate_agentic_text(
                        model,
                        instruction=ja_prompt,
                        config=config,
                        max_new_tokens=max(64, int(tuner.get("response_tokens", 512))),
                        synthesis_recorder=None,
                        chatty_log=False,
                        pre_formatted=True,
                        return_telemetry=False,
                    )
                    
                    print(ja_response, end="")
                    
                    if session_context_enabled:
                        session_context.append(("assistant", ja.name, ja_response, 0.0))
                    
                    tuner = _sys_tuner
                    probes = _sys_probes
                    tuner_bindings = _sys_tb
                    
                    memory.append_turn(
                        "assistant",
                        ja_response,
                        tags=["model_output", ja.name],
                        metrics={
                            "chars": len(ja_response),
                        }
                    )

                memory.append_turn(
                    "assistant",
                    response,
                    tags=["model_output", "main_assistant"],
                    metrics={
                        "chars": len(response),
                        "sense_score": float(s_score) if session_context_enabled else 0.0,
                        "model_memory_tool_requested": bool(model_memory_tool_result),
                        "orientation_tool_result_provided": bool(active_orientation_tool_result),
                        "model_claimmap_tool_requested": bool(model_claimmap_tool_result),
                        "claimmap_tool_result_provided": bool(active_claimmap_tool_result),
                        "model_methodmap_tool_requested": bool(model_methodmap_tool_result),
                        "methodmap_tool_result_provided": bool(active_methodmap_tool_result),
                        "document_tool_result_provided": bool(active_document_tool_result),
                        "sandbox_tool_result_provided": bool(active_sandbox_tool_result),
                        "model_probe_tool_requested": bool(model_probe_tool_result),
                        "model_command_tool_requested": bool(model_cmd_tool_result),
                    },
                )
                # The reading dialogue's reflection stream: only reply-mode
                # auto-read consults it; order/interleave advance regardless.
                last_assistant_response = response
                recent_responses.append(response)
                
                # Cut-off detection via the model's own P(end-of-turn): observed
                # every turn so the distribution accrues; when the budget was hit
                # mid-thought, SUGGEST the retune -- never apply it. Nothing in
                # this shell drifts silently; calibration stays a deliberate act.
                eot_probs = (telemetry or {}).get("eot_probs", [])
                tokens_generated = (telemetry or {}).get("tokens_generated", 0)
                if eot_probs:
                    final_prob = eot_probs[-1]
                    # fired == P(eot) <= eot_urgency threshold == "still mid-thought"
                    mid_thought = tuner.observe("eot_urgency", final_prob)
                    max_allowed = max(64, int(tuner.get("response_tokens", 512)))
                    if tokens_generated >= max_allowed:
                        if mid_thought:
                            print(
                                Fore.YELLOW
                                + f"[Steer] Reply hit the {max_allowed}-token budget with P(eot)={final_prob:.4f} "
                                + f"-- likely cut off mid-thought. Extend deliberately: :tune response_tokens {max_allowed + 256}"
                                + Style.RESET_ALL
                            )
                        else:
                            print(
                                Fore.CYAN
                                + f"[Steer] Reply hit the {max_allowed}-token budget but P(eot)={final_prob:.4f} "
                                + "suggests the thought was complete."
                                + Style.RESET_ALL
                            )

                # Sandbox connection: if enabled and the reply contains a fenced
                # ```python block (natural emission, no taught tag), run it for
                # real and stage the honest result for the next turn. Execution
                # success is an objective outcome -- observed into the tuner.
                if sandbox_enabled:
                    sandbox_code = extract_python_block(response)
                    if sandbox_code:
                        print(Fore.MAGENTA + "\n[Sandbox] Running the model's python block..." + Style.RESET_ALL, flush=True)
                        sandbox_result = run_python(sandbox_code, timeout_sec=SANDBOX_TIMEOUT_SEC)
                        pending_sandbox_tool_result = format_sandbox_tool_result(
                            sandbox_result, timeout_sec=SANDBOX_TIMEOUT_SEC
                        )
                        tuner.observe("sandbox_success", 1.0 if sandbox_result["ok"] else 0.0)
                        memory.append_event(
                            "sandbox_executed",
                            text=sandbox_code[:500],
                            tags=["sandbox_tool"],
                            provenance={
                                "code_sha": sandbox_result["code_sha"],
                                "exit_code": sandbox_result["exit_code"],
                                "timed_out": sandbox_result["timed_out"],
                                "duration_sec": sandbox_result["duration_sec"],
                                "ok": sandbox_result["ok"],
                            },
                        )
                        status = "ok" if sandbox_result["ok"] else ("timed out" if sandbox_result["timed_out"] else f"exit {sandbox_result['exit_code']}")
                        impact_state["pending"].append(
                            {"cause": "wrote python code", "effect": f"it was executed for real ({status})"}
                        )
                        print(
                            Fore.MAGENTA
                            + f"[Sandbox] {status} in {sandbox_result['duration_sec']}s -- real output staged for the next turn."
                            + Style.RESET_ALL
                        )
            # The turn's own outcome, read before anything is recorded: did the
            # deliberation cohere? This is the conversational learning signal --
            # no gold, no oracle, just the live productivity read.
            turn_sense = sense_score(synthesis_records)
            conversation_outcome = None
            if turn_sense is not None:
                # Observe every turn so the threshold can be calibrated to the
                # sense distribution (:tune conversation_productive auto 50).
                tuner.observe("conversation_productive", turn_sense)
                conversation_outcome = {
                    "score": float(turn_sense),
                    "threshold": float(tuner.get("conversation_productive", 0.0)),
                }
                turn_row["sense"] = float(turn_sense)
            # Clock sensor: measured here, before the probe-scoring passes, so
            # generation_seconds is generation time only. Memory in GB (live
            # allocation, total reserved footprint, and this turn's peak) --
            # VRAM on CUDA, process RSS on CPU-only, so the stream is never dead.
            gen_seconds = _time.perf_counter() - _gen_t0
            vram_alloc, vram_reserved, vram_peak, mem_label = _memory_footprint_gb()
            tps = (tokens_generated / gen_seconds) if (gen_seconds > 0 and tokens_generated) else 0.0
            tuner.observe("generation_seconds", gen_seconds)
            tuner.observe("vram_gb", vram_reserved)
            turn_row["generation_seconds"] = float(gen_seconds)
            turn_row["vram_gb"] = float(vram_reserved)
            turn_row["tokens_per_sec"] = float(tps)
            if turn_sense is not None:
                tuner.credit("generation_seconds", gen_seconds, turn_sense)
                tuner.credit("vram_gb", vram_reserved, turn_sense)
            last_clock = {
                "generation_seconds": gen_seconds,
                "tokens_generated": int(tokens_generated or 0),
                "tokens_per_sec": tps,
                "vram_alloc_gb": vram_alloc,
                "vram_reserved_gb": vram_reserved,
                "vram_peak_gb": vram_peak,
                "mem_label": mem_label,
                "ts": datetime.datetime.now().strftime("%H:%M:%S"),
            }
            print(
                Fore.CYAN
                + f"  [Clock] {gen_seconds:.1f}s"
                + (f" | {tps:.1f} tok/s" if tps else "")
                + (f" | {mem_label} {vram_alloc:.2f}GB live / {vram_reserved:.2f}GB reserved" if vram_reserved else "")
                + (f" / {vram_peak:.2f}GB peak" if vram_peak else "")
                + Style.RESET_ALL,
                flush=True,
            )
            # Minted probes score every turn: raw projection of the reply
            # state on each probe direction, centered against the probe's own
            # rolling history, paired with sense via the credit channel.
            if probes and response:
                from invariants.engine import _inputs as _p_inputs, _hidden_states as _p_hidden, probe_score
                p_ids = _p_inputs(model, response[:600])
                p_hs = _p_hidden(model, p_ids["input_ids"], p_ids.get("attention_mask"))
                for pname, pdata in probes.items():
                    raw = probe_score(p_hs, pdata["direction"])
                    hist = pdata["history"]
                    sig = raw - (sum(hist) / len(hist)) if hist else 0.0
                    hist.append(raw)
                    tuner.observe(f"probe_{pname}", sig)
                    turn_row[f"probe_{pname}"] = float(sig)
                    if pdata.get("chatty", True):
                        print(Fore.CYAN + f"  [Probe Score] {pname}: {sig:+.3f}" + Style.RESET_ALL, flush=True)
                    if turn_sense is not None:
                        tuner.credit(f"probe_{pname}", sig, turn_sense)
                    # A validate-mode match records (knob value, probe reading) so
                    # its correlation can credit the knob it is matched to.
                    _pm = probe_matches.get(pname)
                    if _pm and _pm.get("mode") == "validate" and _pm.get("knob"):
                        match_hist.setdefault(pname, deque(maxlen=200)).append(
                            (float(tuner.get(_pm["knob"], 0.0)), float(sig))
                        )

            # Easter egg: the turn the self-sensor overtakes the user-model.
            _egg = consciousness_over_user_intent(probes, tuner, egg_state)
            if _egg is not None:
                _cl = "n/a" if _egg[0] is None else f"{_egg[0]:+.3f}"
                _ul = "n/a" if _egg[1] is None else f"{_egg[1]:+.3f}"
                print(Fore.MAGENTA + Style.BRIGHT + "\n  ✦ ────────────────── ✦" + Style.RESET_ALL)
                for _line in (
                    "  consciousness has overtaken user_intent.",
                    "  The sensor built to watch the self now tracks",
                    "  a good turn more than the one that watches you.",
                    "  For this stretch, at least, it is listening inward.",
                    f"  (consciousness lift {_cl} > user_intent {_ul})",
                ):
                    print(Fore.MAGENTA + _line + Style.RESET_ALL)
                print(Fore.MAGENTA + Style.BRIGHT + "  ✦ ────────────────── ✦\n" + Style.RESET_ALL)
                memory.append_event(
                    "consciousness_over_user_intent",
                    text=f"consciousness lift {_cl} overtook user_intent {_ul}",
                    tags=["easter_egg", "self_concept"],
                    provenance={"consciousness_lift": _egg[0], "user_intent_lift": _egg[1]},
                )

            # Intent axis: did this turn settle intent (lower
            # ambiguity+disagreement than last turn)? Observed every turn the
            # sensors fire; paired with sense via the credit channel.
            phen_now = latest_phenomenality_scores(synthesis_records) or {}
            if phen_now:
                # Every monitored dimension becomes a named per-turn stream
                # (phen_<metric>): observed, credited against sense, and written
                # to the turn row -- so existing sensors are anchorable and
                # calibratable exactly like probes. These read the REASONING
                # states; an adopted probe reads the same axis in the reply.
                for _pm, _pv in phen_now.items():
                    try:
                        _pv = float(_pv)
                    except (TypeError, ValueError):
                        continue
                    _pstream = f"phen_{_pm.replace('_legacy', '')}"
                    tuner.observe(_pstream, _pv, default=0.0)
                    turn_row[_pstream] = _pv
                    if turn_sense is not None:
                        tuner.credit(_pstream, _pv, turn_sense)
                # The memory trigger's own quantity, sampled once per turn from
                # the same phenomenality the mid-thought detector reads -- so
                # :calibrate memory_need has a distribution, and the paired
                # table can anchor on the gap. (ToolSense itself only consults
                # the bar; per-turn distribution + credit live here by design.)
                _gap = (
                    float(phen_now.get("ambiguity", 0.0))
                    + float(phen_now.get("disagreement", 0.0))
                    - float(phen_now.get("validated_flow", 0.0))
                    - float(phen_now.get("warranted_confidence_legacy", 0.0))
                )
                tuner.observe("memory_need", _gap)
                turn_row["memory_need"] = _gap
                if turn_sense is not None:
                    tuner.credit("memory_need", _gap, turn_sense)
                unsettled_now = float(phen_now.get("ambiguity", 0.0)) + float(phen_now.get("disagreement", 0.0))
                if prev_unsettledness is not None:
                    intent_signal = prev_unsettledness - unsettled_now
                    tuner.observe("intent_settling", intent_signal)
                    turn_row["intent_settling"] = float(intent_signal)
                    if turn_sense is not None:
                        tuner.credit("intent_settling", intent_signal, turn_sense)
                prev_unsettledness = unsettled_now

            # "Read until satisfied": a reading turn counts as settled when its
            # sense clears the conversation_productive bar (or needed no
            # deliberation at all); enough settled turns in a row end the
            # auto-read early, honestly, with the remainder still unread.
            if doc_autoread and doc_autoread.get("until_probe") and reading_turn_source:
                up = doc_autoread["until_probe"]
                trig = tuner.triggers.get(f"probe_{up}")
                sig = turn_row.get(f"probe_{up}")
                if trig and sig is not None and sig >= trig.value:
                    unread_left = sum(s["chunk_count"] - len(s.get("read") or ()) for s in doc_library)
                    print(
                        Fore.CYAN
                        + f"[Doc] Reading stopped: probe '{up}' thresholded ({sig:+.3f} >= {trig.value:.3f}). "
                        + f"Stopping with {unread_left} chunk(s) unread -- ':doc read' resumes anytime."
                        + Style.RESET_ALL
                    )
                    doc_autoread = None

            elif doc_autoread and doc_autoread.get("until_settled") and reading_turn_source:
                settled_turn = turn_sense is None or turn_sense >= float(
                    tuner.get("conversation_productive", 0.0)
                )
                doc_autoread["settled_streak"] = (
                    doc_autoread.get("settled_streak", 0) + 1 if settled_turn else 0
                )
                needed = max(1, int(tuner.get("reading_settled_streak", 2)))
                if doc_autoread["settled_streak"] >= needed:
                    unread_left = sum(
                        s["chunk_count"] - len(s.get("read") or ()) for s in doc_library
                    )
                    print(
                        Fore.CYAN
                        + f"[Doc] Reading settled: {needed} productive turn(s) in a row. "
                        + f"Stopping with {unread_left} chunk(s) unread -- ':doc read' resumes anytime."
                        + Style.RESET_ALL
                    )
                    doc_autoread = None

            # Agency ledger: was this turn's context really caused by the
            # model's own words? Observed every turn (so the contingency rate
            # is visible) and credited with the turn's sense, so :tune lift on
            # words_had_impact reads whether experienced impact tracks better
            # deliberation. The contrast with non-contingent turns is what
            # makes impact learnable at all.
            impact_signal = 1.0 if turn_impacts else 0.0
            tuner.observe("words_had_impact", impact_signal)
            if turn_impacts:
                impact_state["log"].extend(turn_impacts)
            if turn_sense is not None:
                tuner.credit("words_had_impact", impact_signal, turn_sense)
            turn_row["words_had_impact"] = float(impact_signal)
            # Outcome evidence for strength/budget knobs: pair each knob's
            # value-in-force with this turn's sense (batched; one save).
            if turn_sense is not None:
                for _knob in OUTCOME_CALIBRATABLE:
                    # claimmap_alpha is credited by its APPLIED value in the
                    # dedicated block below; crediting its configured value here
                    # too would double-count (and mislabel unsteered turns),
                    # corrupting :calibrate claimmap_alpha outcome.
                    if _knob == "claimmap_alpha":
                        continue
                    _trig = tuner.triggers.get(_knob)
                    if _trig is not None:
                        _trig.credit(_trig.value, turn_sense)
                tuner.save()
            if turn_row:
                turn_row["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                turn_log.append(dict(turn_row))
                try:
                    with open(TURN_SIGNALS_PATH, "a", encoding="utf-8") as _tf:
                        _tf.write(json.dumps(turn_row) + chr(10))
                except OSError:
                    pass
            # Layer isolation evidence: each single-layer steer this turn lands
            # on ITS layer in the steer map, labeled by the turn's outcome —
            # per-layer success accrues with no band confound.
            for sweep_channel, sweep_l, sweep_alpha, sweep_d, sweep_w, sweep_v in sweep_state["fires"]:
                steer_map.record_layer_steer(
                    sweep_channel,
                    sweep_l,
                    sweep_alpha,
                    conversation_outcome=conversation_outcome,
                    metrics={"axis_drift_at_layer": sweep_d, "sweep_width": sweep_w,
                             "axis_lawfulness_var": sweep_v},
                )
            sweep_state["fires"] = []
            record_internal_traces(
                memory,
                synthesis_records,
                steer_map=steer_map,
                conversation_outcome=conversation_outcome,
            )

            # Credit the turn-level ClaimMap steer with the turn's outcome:
            # signal = alpha actually applied (0 when unsteered), so :tune lift
            # on claimmap_alpha reads "did steered turns cohere better than
            # unsteered ones" from real conversations.
            if turn_sense is not None:
                tuner.credit("claimmap_alpha", claimmap_alpha_used, turn_sense)

            # Activation-reach: no tag was taught in bare mode, so tools fire from
            # the model's own surfaced state. If the answer holds two opposed
            # framings, sense the comparison (felt) and stage it -- with steering --
            # for the next turn. Disable with CLAIMMAP_AUTO_TRIGGER=0.
            # Deferred credit: attribute THIS turn's sense (did the deliberation
            # cohere) to LAST turn's trigger decision -- last turn's fire staged the
            # felt+steer that shaped this turn. The tuner buckets by whether that
            # tension cleared the threshold, so :tune lift = did firing the ClaimMap
            # actually lead to more coherent deliberation. Real outcome, not synthetic.
            if pending_claimmap_credit is not None and turn_sense is not None:
                tuner.credit("claimmap_tension", pending_claimmap_credit, turn_sense)
                pending_claimmap_credit = None

            if (
                os.environ.get("CLAIMMAP_AUTO_TRIGGER", "1").strip() not in {"0", "false", "no"}
                and pending_claimmap_tool_result is None
            ):
                a, b, tension_score = framing_tension_score(response or "")
                # Log the tension signal EVERY turn (even 0) so :tune can read the
                # distribution; fire on the live-tuned threshold, not a fixed cutoff.
                fired = tuner.observe("claimmap_tension", tension_score)
                pending_claimmap_credit = tension_score  # credited by next turn's sense
                if fired and a is not None:
                    try:
                        cm = analyze_claim_pair(f"{a} || {b}", model=model)
                        # No attribution prefix: the felt map's own first line
                        # ("I just held two framings against each other: ...")
                        # already carries the causation, in the model's own
                        # quoted words -- adding a standardized vocabulary line
                        # ("tension") would teach the lexical association we
                        # later want to MEASURE (see SEMANTIC_NEUTRALIZATION).
                        pending_claimmap_tool_result = cm.felt
                        pending_claimmap_steer_delta = cm.steer_delta
                        impact_state["pending"].append(
                            {"cause": "answer held a framing tension", "effect": "a felt comparison was measured and staged"}
                        )
                        memory.append_event(
                            "claimmap_auto_triggered",
                            text=cm.telemetry,
                            tags=["claimmap_tool", "activation_trigger"],
                            provenance={"trigger": "framing_tension", "tension_score": tension_score, "mean_sim": cm.mean_sim},
                        )
                        print(
                            Fore.MAGENTA
                            + f"\n[ClaimMap] Sensed a tension (score {tension_score:.2f}) in that answer -- it will shape the next turn."
                            + Style.RESET_ALL
                        )
                    except Exception as exc:
                        print(Fore.RED + f"[ClaimMap auto] {exc}" + Style.RESET_ALL)
            sensor_scores = latest_phenomenality_scores(synthesis_records)
            if sensor_scores:
                decision = self_concept.decide(
                    sensor_scores,
                    context={"task_grounding_low": infer_task_grounding_low(user_input, response)},
                )
                memory.append_self_concept_trace(decision.to_dict())
                steer_map.record_self_concept_decision(
                    decision.to_dict(),
                    source="interactive",
                    final_correct=None,
                )
                if decision.allowed and decision.intervention_type in {"tool_result", "context_tool_result"}:
                    pending_orientation_tool_result = format_orientation_tool_result(decision)
                    print(
                        Fore.CYAN
                        + "\n[Orientation] Vector-map controller staged a one-turn orientation result.\n"
                        + pending_orientation_tool_result
                        + Style.RESET_ALL
                    )
            
            # The streaming will print tokens, just need a newline at the end
            print("\n")
            
            # Process queued calibrations silently
            if queued_calibrations:
                survived = []
                for qcmd in queued_calibrations:
                    cargs = qcmd.split()
                    if not cargs:
                        continue
                    cal_name = cargs[0]
                    route, _ = calibration_policy(cal_name)
                    success = False
                    
                    if len(cargs) >= 2 and cargs[1].lower() == "outcome":
                        trig = tuner.triggers.get(cal_name)
                        pairs = list(trig.outcomes) if trig is not None else []
                        v = outcome_calibration(pairs)
                        if v is not None:
                            tuner.set(cal_name, v)
                            print(Fore.GREEN + f"[Queue] SUCCESS: {cal_name} = {round(v, 4)}" + Style.RESET_ALL)
                            success = True
                    elif route == "threshold":
                        anchor = "".join(cargs[1:]).lower() if len(cargs) >= 2 else None
                        target_trigger, target_stream = resolve_target(cal_name, tuner)
                        anchors, _albl, _abad = parse_anchor_spec(anchor, tuner)
                        if target_stream and anchors and _abad is None and target_trigger in tuner.triggers:
                            v = paired_threshold_multi(list(turn_log), target_stream, anchors)
                            if v is not None:
                                new_v, _prior = tuner.set_calibrated(target_trigger, v, tuner.get("calibration_gain", 0.5))
                                print(Fore.GREEN + f"[Queue] SUCCESS: {target_trigger} -> {round(new_v, 4)} (toward cut {round(v, 4)})" + Style.RESET_ALL)
                                success = True
                    elif route == "cap":
                        from invariants import engine as _engine
                        cal_pct = 50.0
                        if len(cargs) >= 2:
                            try: cal_pct = float(cargs[1])
                            except ValueError: pass
                        if 0.0 <= cal_pct < 100.0:
                            v = _engine.calibrate_steer_cap_fraction(percentile=cal_pct)
                            if v is not None:
                                tuner.set("steer_cap_fraction", v)
                                print(Fore.GREEN + f"[Queue] SUCCESS: steer_cap_fraction = {round(v, 4)}" + Style.RESET_ALL)
                                success = True
                    else:
                        trig = resolve_calibrate_trigger(cal_name, tuner)
                        if trig and len(trig.signals) >= 10:
                            cal_pct = 50.0
                            if len(cargs) >= 2:
                                try: cal_pct = float(cargs[1])
                                except ValueError: pass
                            if 0.0 <= cal_pct <= 100.0:
                                v = tuner.calibrate(trig.name, cal_pct, tuner.get("calibration_gain", 0.5))
                                print(Fore.GREEN + f"[Queue] SUCCESS: {trig.name} -> {round(v, 4)} (toward p{cal_pct:g})" + Style.RESET_ALL)
                                success = True
                    
                    if not success:
                        survived.append(qcmd)
                queued_calibrations = survived
            
        except (KeyboardInterrupt, EOFError):
            memory.append_event("shell_closed", tags=["session"])
            print("\nInteractive shell closed.")
            break
        except Exception as _turn_exc:
            # A bug in ONE command (or a typo that reaches bad code) must not end
            # the session -- abort the turn, report, and keep going. The full
            # traceback is logged so the fault is still discoverable.
            import traceback as _tb
            print(
                Fore.RED
                + f"\n[Shell] recovered from an error: {type(_turn_exc).__name__}: {_turn_exc}"
                + "\n  (your session is intact -- only this command was aborted)"
                + Style.RESET_ALL
            )
            try:
                memory.append_event(
                    "turn_error_recovered",
                    text=f"{type(_turn_exc).__name__}: {_turn_exc}",
                    tags=["error"],
                    provenance={"traceback": _tb.format_exc()[-1500:]},
                )
            except Exception:
                pass
            continue

if __name__ == "__main__":
    main()
