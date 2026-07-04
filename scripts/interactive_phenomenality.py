import datetime
import glob
import json
import os
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
        text = entry[1] or ""
        if total + len(text) > max_chars:
            if not kept:
                kept.append((role, text[-max_chars:]))
            break
        kept.append((role, text))
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
    session_context=None,
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
            claimmap_tool_result,   # already pure first-person felt language
            memory_tool_result,
            orientation_tool_result,
            methodmap_tool_result,
            sandbox_tool_result,    # real execution output from last turn's code
            document_tool_result,   # operator-shared document chunk, framed what/why
            probe_tool_result,      # sensor readings the model asked for itself
        )
        if block
    ]
    current_message = user_input
    if tool_blocks:
        current_message = "\n\n".join(tool_blocks) + "\n\n" + user_input

    parts = ["<|begin_of_text|>"]
    for role, text in trim_session_context(session_context):
        header = "user" if role == "user" else "assistant"
        parts.append(f"{LLAMA3_START}{header}{LLAMA3_END}\n\n{text}{LLAMA3_EOT}")
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
):
    if (
        memory_tool_result
        or orientation_tool_result
        or claimmap_tool_result
        or methodmap_tool_result
        or sandbox_tool_result
        or document_tool_result
        or probe_tool_result
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


def remove_probe_tool_calls(response):
    return PROBE_TOOL_PATTERN.sub("", response or "").strip()


def remove_tool_calls(response):
    return remove_methodmap_tool_calls(remove_claimmap_tool_calls(remove_memory_tool_calls(remove_doc_tool_calls(remove_probe_tool_calls(response)))))


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
        (extract_memory_query(text) or extract_claimmap_payload(text) or extract_methodmap_query(text) or extract_doc_query(text) or extract_probe_query(text))
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
    recovered = [(r.role, r.text, r.timestamp, getattr(r, "metrics", {})) for r in tail]
    return session_id, recovered


def recovered_to_session_context(recovered):
    entries = []
    for r in recovered:
        m = r[3] if len(r) > 3 else {}
        score = m.get("sense_score", 0.0) if m and "sense_score" in m else 0.0
        entries.append((r[0], r[1], score))
    # Live turns give both halves of a pair the same earned score, but the
    # persistent record only carries it on the assistant turn (the user turn
    # is logged before generation). Propagate it back one slot so sense-based
    # eviction sees resumed pairs the same way it saw them live.
    for i in range(len(entries) - 1):
        if entries[i][0] == "user" and entries[i + 1][0] == "assistant":
            entries[i] = (entries[i][0], entries[i][1], entries[i + 1][2])
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


def resolve_stream(name, tuner):
    """A nameable stream: alias, registered trigger, bare probe name, or bare
    phenomenality sensor name (probe wins when both exist -- reply-state over
    reasoning-state; say phen_<name> to force the sensor)."""
    name = (name or "").lower()
    if name in STREAM_ALIASES:
        return STREAM_ALIASES[name]
    if name in tuner.triggers:
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


def resolve_target(cal_name, tuner):
    """A calibration target's (trigger, stream_column), kept CONSISTENT so a
    bar is never set from a different stream's values. Prefers an exact
    trigger, then a probe of that name; the sense target reads the sense
    column. This is why a probe named after an alias (e.g. a probe 'intent'
    vs the intent->intent_settling alias) calibrates itself, not the alias."""
    if cal_name == "conversation_productive":
        return "conversation_productive", "sense"
    if cal_name in tuner.triggers:
        return cal_name, cal_name
    if f"probe_{cal_name}" in tuner.triggers:
        return f"probe_{cal_name}", f"probe_{cal_name}"
    return f"probe_{cal_name}", resolve_stream(cal_name, tuner)


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
    Kept forgiving (0.6 cutoff) since these are short identifiers."""
    import difflib
    match = difflib.get_close_matches(
        (name or "").lower(), sorted(set(candidates)), n=1, cutoff=0.6
    )
    return f" Did you mean '{match[0]}'?" if match else ""


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
    need not follow steering depth. Returns ({}, None) when nothing usable
    exists."""
    want = set(int(x) for x in layers) if layers is not None else None
    src_path = next(
        (p for p in (
            os.path.join(ROOT, "invariants", f"{stem}_vector.pt"),
            os.path.join(ROOT, "invariants", f"{stem}.pt"),
            os.path.join(ROOT, "invariants", "out", "probes", f"{stem}.pt"),
        ) if os.path.isfile(p)),
        None,
    )
    if src_path is None:
        return {}, None
    try:
        payload = torch.load(src_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(src_path, map_location="cpu")
    from invariants.engine import steer_band_layers
    direction = {}
    if isinstance(payload, dict) and "direction" in payload:
        for L, v in payload["direction"].items():
            if want is not None and int(L) not in want:
                continue
            v = v.to(model.device).float().reshape(-1)
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
                n = v.norm()
                if n.item() > 0:
                    direction[int(L)] = v / n
    elif hasattr(payload, "reshape"):
        v = payload.to(model.device).float().reshape(-1)
        n = v.norm()
        if n.item() > 0:
            unit = v / n
            n_layers = int(model.model.config.num_hidden_layers)
            targets = sorted(want) if want is not None else steer_band_layers(n_layers)
            for L in targets:
                if 0 <= int(L) < n_layers:
                    direction[int(L)] = unit.clone()
    return direction, os.path.basename(src_path)


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


def main():
    os.chdir(ROOT)
    print(Fore.CYAN + Style.BRIGHT + "================================================")
    print("      HUMBLE SYNTHESIS - INTERACTIVE SHELL      ")
    print("================================================" + Style.RESET_ALL)
    
    # Model is configurable so the egg shell honors whatever model earned the egg
    # (env EGG_MODEL set by the benchmark, or argv[1] for a manual launch).
    model_name = os.environ.get("EGG_MODEL") or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    is_default = (model_name == DEFAULT_MODEL)

    print(Fore.YELLOW + f"[System] Loading {model_name}..." + Style.RESET_ALL)
    model = load_model(model_name, local_files_only=is_default)

    config = AgenticConfig()
    # The organic correction vector is calibrated for the default model's geometry;
    # skip it on a swapped model rather than inject a dimension-mismatched steer.
    if is_default:
        try:
            config.organic_correction_vector = torch.load(ORGANIC_VECTOR_PATH, map_location=model.device)
            print(Fore.GREEN + "[System] Successfully loaded organic_correction_vector.pt!" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"[System] Warning: Could not load organic vector: {e}" + Style.RESET_ALL)
    else:
        print(Fore.YELLOW + "[System] Skipping organic vector (calibrated for the default model)." + Style.RESET_ALL)

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
    # Clock sensor: this turn's generation wall-time and VRAM footprint,
    # observed every turn like any other stream (so they can be anchored:
    # "did slow / memory-heavy turns cohere worse?"). Threshold-kind with a
    # 0 bar until you calibrate one; observation-only, never a knob you set.
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
    print("Commands: :context, :context on, :context off, :context clear")
    print("          :memory, :memory recent [n], :memory search <query>, :memory use <query>, :memory boundary")
    print("          :methodmap <query>")
    print("          :claimmap <first text> || <second text>")
    print("          :steermap")
    print("          :steer  (envelope + observed push distribution + data-implied cap/band)")
    print("          :probe <name> <with it> || <without it>  (mint a named-concept sensor from")
    print("                YOUR contrastive framings; scores every turn; :probe lists; :probe drop <name>)")
    print("          :probe adopt <dim> [<dim> ...]  (turn stored vectors -- ambiguity, disagreement,")
    print("                warranted_confidence, organic_correction, ... -- into reply-scoring probes)")
    print("          :probe compose <name> <mix>  (mint a probe from a SIGNED MIX of dimensions and")
    print("                probes: ambiguity + disagreement - validated_flow - 0.5*curiosity)")
    print("                (mint/adopt/compose take a trailing 'band <lo> <hi>' -- explicit reading")
    print("                layers, e.g. band 16 24; otherwise the live steer band decides depth)")
    print("          :probe expose <name> [off]  (let the MODEL consult this sensor itself:")
    print("                <<PROBE: name>> reads its last turn, <<PROBE: name || words>> scores")
    print("                candidate words. Reading only -- minting/calibrating stay operator acts)")
    print("          :probe backfill <name> [n]  (retro-score up to n archived replies in order:")
    print("                rebuilds the probe's stream+credit from the whole record, seeds its history)")
    print("          :calibrate <name> [pct|intent|<anchor>|<a>+<b>|band args]  (data-calibrate any knob")
    print("                BY NAME; anchors join with '+' = fired only when EVERY stream fired;")
    print("                the system evaluates the request and refuses unsafe ones --")
    print("                circular strength knobs, binary streams, vacuous p100 caps)")
    print("          :suggest  (scan the accrued state for ready moves -- calibrations, knobs")
    print("                explored-but-not-committed, capabilities never tried, probes to backfill")
    print("                or expose -- each with its command; computed, never applied. :suggest apply")
    print("                auto-queues only the safe measurement/calibration ones)")
    print("          :tune, :tune <name> <value>, :tune <name> auto [percentile]")
    print("          :tune steer_cap_fraction auto [pct]  (calibrate cap from observed pushes)")
    print("          :tune steer_band auto [min_events] [gold|conversation|any] [synthesis|layersteer]")
    print("                (derive band from outcomes; conversations count as evidence)")
    print("          :tune steer_layer_sweep 1    (isolate steers by layer: each steer pushes")
    print("                ONE least-tested band layer; per-layer outcomes accrue, transfer-free)")
    print("          :doc <path> [because <why>]  (share a document into the conversation)")
    print("          :doc next | :doc status      (stage the next chunk / show progress)")
    print("          :doc read [n] [order|interleave|reply|updated] [satisfied]")
    print("                (reading as dialogue: the document speaks each turn, the model")
    print("                replies. order/interleave/updated advance on the documents' own")
    print("                course -- updated = by file mtime, newest first; reply follows")
    print("                overlap. 'satisfied' stops early once sense settles)")
    print("          :doc inject                  (stage the whole library for one turn, budget-bounded)")
    print("          :doc stop                    (interrupt auto-read)")
    print("          :sandbox on|off|status       (run the model's ```python blocks for real)")
    print("          :experts on|off|status       (mint new steering experts from its own")
    print("                recurring self-corrections; roster bounded; default off)")
    print("          :impact                      (consequence trail: what its words caused,")
    print("                and whether experienced impact tracks better deliberation)")
    print("          :clock                       (last turn's generation time + tok/s and VRAM;")
    print("                sensed every turn as generation_seconds / vram_gb streams)")
    print("          :prioritize                  (rank probes by evidence-weighted lift; steer toward")
    print("                the top each turn via prioritize_alpha -- signed by lift, off at 0)")
    print("          :release <tool> [prob]       (decouple a tool's firing from its signal for that")
    print("                fraction of turns -- separates causality so credit lift can be trusted)")
    print("          :listen on|off|status        (speak mid-reply: lines you type while it")
    print("                generates are ingested at the next chunk seam and appended to the")
    print("                live stream -- the model chooses to redirect or fold in; never dropped)")
    print("Type 'exit' or 'quit' to leave.\n" + Style.RESET_ALL)

    pending_memory_tool_result = None
    pending_orientation_tool_result = None
    pending_claimmap_tool_result = None
    pending_claimmap_steer_delta = None
    pending_claimmap_credit = None  # last turn's tension, credited by this turn's sense
    pending_methodmap_tool_result = None
    pending_document_tool_result = None
    pending_sandbox_tool_result = None
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
    MAX_AUTOREAD = 20           # per :doc read command; reading stays a deliberate act
    sandbox_enabled = False     # deliberate opt-in, like every intervention here
    session_context = []
    session_context_enabled = True
    show_timestamps = False
    last_clock = None
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
            else:
                prefix = f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] You: " if show_timestamps else "\nYou: "
                prompt_str = Fore.MAGENTA + Style.BRIGHT + prefix + Style.RESET_ALL
                # Once the listener is active, every turn is served from its
                # queue (so mid-generation typing was never dropped); otherwise
                # the plain, unchanged input() path.
                user_input = listener.get(prompt_str) if listener.active else input(prompt_str)
                
            if user_input.lower() in ['exit', 'quit']:
                memory.append_event("shell_closed", tags=["session"], provenance={"reason": "operator_exit"})
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
                else:
                    print(
                        Fore.YELLOW
                        + "[Memory] Commands: :memory, :memory recent [n], :memory search <query>, :memory use <query>, :memory boundary"
                        + Style.RESET_ALL
                    )
                continue
            if user_input.startswith(":consider"):
                payload = user_input[len(":consider"):].strip()
                try:
                    if "||" not in payload:
                        raise ValueError("Usage: :consider <trigger_metric> <tool_name> <positive text> || <negative text>")
                    left_side, _, b_text = payload.partition("||")
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
                        mode_aliases = {"weave": "interleave", "mtime": "updated", "chrono": "updated"}
                        for extra in dargs.split()[1:]:
                            token = extra.strip().lower()
                            if token in {"order", "reply", "interleave", "weave", "updated", "mtime", "chrono"}:
                                mode = mode_aliases.get(token, token)
                            elif token in {"satisfied", "settled", "until"}:
                                until_settled = True
                            else:
                                try:
                                    count = int(token)
                                    explicit_count = True
                                except ValueError:
                                    pass
                        if until_settled and not explicit_count:
                            count = MAX_AUTOREAD  # "until satisfied" reads up to the cap
                        count = max(1, min(MAX_AUTOREAD, count))
                        doc_autoread = {"remaining": count, "mode": mode, "until_settled": until_settled, "settled_streak": 0}
                        how = {
                            "order": "in document order -- the text advances on its own course",
                            "interleave": "weaving between documents -- their course, not the model's echo",
                            "reply": "chunks chosen by overlap with its replies (echo-following; deliberate use)",
                            "updated": "in the order the files were last written (newest first)",
                        }[mode]
                        stop_note = (
                            " Stops early once the reading settles (sense clears conversation_productive "
                            f"{max(1, int(tuner.get('reading_settled_streak', 2)))} turn(s) in a row)."
                            if until_settled
                            else ""
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
                else:
                    path_part, _, why = dargs.partition(" because ")
                    path_str = path_part.strip().strip('"')
                    
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
                        paths_to_load = [path_str]

                    if not paths_to_load:
                        print(Fore.YELLOW + f"[Doc] No files found for {path_str}" + Style.RESET_ALL)
                        continue

                    loaded_count = 0
                    for p in paths_to_load:
                        file_why = why.strip()
                        if "{filename}" in file_why:
                            file_why = file_why.replace("{filename}", os.path.basename(p))
                        if "{last_updated}" in file_why:
                            mtime = os.path.getmtime(p)
                            dt = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
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
                # On-demand readout of the last turn's generation time + VRAM,
                # plus the current live/reserved footprint and the accrued
                # distributions of both sensed streams.
                if torch.cuda.is_available():
                    _gb = 1024 ** 3
                    now_alloc = torch.cuda.memory_allocated() / _gb
                    now_reserved = torch.cuda.memory_reserved() / _gb
                    print(Fore.CYAN + f"[Clock] VRAM now: {now_alloc:.2f}GB live / {now_reserved:.2f}GB reserved." + Style.RESET_ALL)
                else:
                    print(Fore.CYAN + "[Clock] CUDA not available; VRAM readings are 0." + Style.RESET_ALL)
                if last_clock:
                    lc = last_clock
                    print(
                        Fore.CYAN
                        + f"[Clock] last turn ({lc['ts']}): {lc['generation_seconds']:.1f}s"
                        + (f", {lc['tokens_generated']} tok @ {lc['tokens_per_sec']:.1f} tok/s" if lc['tokens_per_sec'] else "")
                        + f", peak {lc['vram_peak_gb']:.2f}GB."
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
                if pargs.lower().startswith("drop "):
                    dropped = pargs[5:].strip()
                    if probes.pop(dropped, None) is not None:
                        print(Fore.CYAN + f"[Probe] {dropped} dropped (its observed stream is kept)." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Probe] no active probe named {dropped}.{did_you_mean(dropped, probes)}" + Style.RESET_ALL)
                    continue
                if pargs.lower().startswith("chatty "):
                    chatty_name = pargs[7:].strip()
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
                    ename = re.sub(r"[^a-z0-9_]", "_", eargs[0].lower())[:40] if eargs else ""
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
                        direction, src_name = load_stored_direction(model, stem, layers=adopt_layers)
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
                    if cname in probes:
                        print(Fore.YELLOW + f"[Probe] '{cname}' is already an active probe (:probe drop {cname} first)." + Style.RESET_ALL)
                        continue
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
                            tdir, _ = load_stored_direction(model, tname, layers=cband)
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
                    bf_name = re.sub(r"[^a-z0-9_]", "_", rest[0].lower())[:40] if rest else ""
                    if bf_name not in probes:
                        print(Fore.YELLOW + f"[Probe] no active probe named '{bf_name}'.{did_you_mean(bf_name, probes)} Bare :probe lists them." + Style.RESET_ALL)
                        continue
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
                    mint_ts = next(
                        (e.timestamp for e in memory.records
                         if e.kind == "event" and "probe" in (e.tags or []) and (e.text or "").startswith(f"{bf_name}:")),
                        None,
                    )
                    from invariants.engine import _inputs as _bf_inputs, _hidden_states as _bf_hidden, probe_score as _bf_score
                    # One shared forward per archived reply scores EVERY active
                    # probe (projections are free once the states exist), so the
                    # rows written are JOINT -- multi-anchor calibration needs
                    # anchors on the same row. Only the NAMED probe's trigger
                    # and rolling history are rebuilt.
                    bf_rollings = {p: deque(maxlen=40) for p in probes}
                    bf_scored = []  # (timestamp, {probe: (raw, sig)}, stored sense)
                    print(
                        Fore.CYAN
                        + f"[Probe] backfilling {bf_name} over {len(archive)} archived replies "
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
                    trig = tuner.register(f"probe_{bf_name}", 0.0, kind="threshold", comparator=">=")
                    trig.signals.clear()
                    trig.outcomes.clear()
                    trig.observed = 0
                    trig.fired = 0
                    credited = 0
                    for _bts, _bpp, _bsense in bf_scored:
                        _bsig = _bpp[bf_name][1]
                        trig.observe(_bsig)
                        if _bsense is not None:
                            trig.credit(_bsig, _bsense)
                            credited += 1
                    tuner.save()
                    probes[bf_name]["history"] = deque((pp[bf_name][0] for _, pp, _ in bf_scored), maxlen=40)
                    # Rows already carrying this probe's column (an earlier
                    # backfill) are skipped by timestamp; recomputation is
                    # deterministic, so a re-run adds nothing twice.
                    existing_ts = set()
                    try:
                        with open(TURN_SIGNALS_PATH, "r", encoding="utf-8") as _tf:
                            for _line in _tf:
                                try:
                                    _r = json.loads(_line)
                                except Exception:
                                    continue
                                if _r.get("basis") == "backfill" and f"probe_{bf_name}" in _r and _r.get("ts"):
                                    existing_ts.add(_r["ts"])
                    except OSError:
                        pass
                    rows_added = 0
                    try:
                        with open(TURN_SIGNALS_PATH, "a", encoding="utf-8") as _tf:
                            for _bts, _bpp, _bsense in bf_scored:
                                if mint_ts is not None and _bts >= mint_ts:
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
                        text=f"{bf_name}: {len(bf_scored)} archived replies re-scored",
                        tags=["probe"],
                        provenance={
                            "probe": bf_name,
                            "turns_scored": len(bf_scored),
                            "credited": credited,
                            "paired_rows_added": rows_added,
                            "history_seeded": len(probes[bf_name]["history"]),
                        },
                    )
                    st = trig.outcome_stats()
                    print(
                        Fore.GREEN
                        + f"[Probe] {bf_name} backfilled: {len(bf_scored)} archived replies re-scored in order. "
                        + f"Trigger stream rebuilt ({credited} turns credited with stored sense, lift={st['lift']}), "
                        + f"{rows_added} pre-mint paired rows added, rolling history seeded with the last "
                        + f"{len(probes[bf_name]['history'])} raws -- live scoring continues from it."
                        + Style.RESET_ALL
                    )
                    continue
                pname, _, framings = pargs.partition(" ")
                pname = re.sub(r"[^a-z0-9_]", "_", pname.lower())[:40]
                if "||" not in framings:
                    print(Fore.CYAN + f"[Probe] Suggesting contrastive framings for '{pname}'..." + Style.RESET_ALL)
                    suggestion_prompt = (
                        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        f"Write a contrastive definition pair for a behavioral dimension called '{pname}'. "
                        f"The format must be exactly: <positive statement about the assistant> || <negative statement about the assistant>.\n\n"
                        f"Example for 'understanding': The assistant fully comprehends the user's intent. || The assistant is confused and misses the point.\n\n"
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
                a_text, _, b_text = framings.partition("||")
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
                    torch.save({"direction": direction, "framings": (a_text, b_text)}, os.path.join(PROBE_DIR, f"{pname}.pt"))
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
            if user_input.strip().lower() == ":suggest apply":
                _arch = sum(
                    1 for r in memory.records
                    if r.scope == memory.scope and r.kind == "turn" and r.role == "assistant"
                )
                sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                safe = [(cat, line, cmd) for cat, line, cmd in sugg if cat in SUGGEST_APPLY_SAFE]
                if not safe:
                    print(Fore.CYAN + "[Suggest Apply] nothing safe to auto-run (explore/expose stay manual)." + Style.RESET_ALL)
                else:
                    cmds = [cmd for _, _, cmd in safe]
                    input_queue.extend(cmds)
                    print(Fore.GREEN + f"[Suggest Apply] Auto-queued {len(cmds)} measurement/calibration action(s)." + Style.RESET_ALL)
                continue
            if user_input.strip().lower() in (":suggest", ":suggestions"):
                # The evidence is already accrued every turn; this only reads
                # it back. A suggestion is a computed value plus the command
                # that would enact it -- the choice stays with the operator.
                _arch = sum(
                    1 for r in memory.records
                    if r.scope == memory.scope and r.kind == "turn" and r.role == "assistant"
                )
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
                            print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                            print(Fore.GREEN + f"      -> {cmd}" + Style.RESET_ALL)
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
                cal_name = cargs[0]
                route, reason = calibration_policy(cal_name)
                if len(cargs) >= 2 and cargs[1].lower() == "outcome":
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
                        anchor_parts = [p for p in anchor.split("+") if p]
                        anchor_streams = []
                        bad_part = None
                        for part in anchor_parts:
                            resolved = resolve_stream(part, tuner)
                            if resolved is None:
                                bad_part = part
                                break
                            anchor_streams.append(resolved)
                        if bad_part is not None or not anchor_streams:
                            print(
                                Fore.YELLOW
                                + f"[Calibrate] unknown anchor '{bad_part or anchor}'.{did_you_mean(bad_part or anchor, calibratable_names(tuner))} "
                                + "Name an observed stream "
                                + "(productive/intent/impact/a probe/a phen_ sensor), join several with '+', "
                                + f"or mint one: :probe {bad_part or anchor} <with it> || <without it>."
                                + Style.RESET_ALL
                            )
                            continue
                        anchors = []
                        for a_stream in anchor_streams:
                            a_trig = anchor_trigger_for(a_stream, tuner)
                            anchors.append((
                                a_stream,
                                a_trig.value if a_trig else 0.0,
                                a_trig.comparator if a_trig else ">=",
                            ))
                        v = paired_threshold_multi(list(turn_log), target_stream, anchors)
                        anchor_label = "+".join(anchor_streams)
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
                        trig = tuner.triggers.get(cal_name) or tuner.triggers.get(f"probe_{cal_name}")
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
                targs = user_input[len(":tune"):].split()
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
            
            memory_tool_result = pending_memory_tool_result
            orientation_tool_result = pending_orientation_tool_result
            claimmap_tool_result = pending_claimmap_tool_result
            claimmap_steer_delta = pending_claimmap_steer_delta
            methodmap_tool_result = pending_methodmap_tool_result
            document_tool_result = pending_document_tool_result
            sandbox_tool_result = pending_sandbox_tool_result
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
            prompt = build_prompt(
                user_input,
                memory_tool_result=memory_tool_result,
                orientation_tool_result=orientation_tool_result,
                claimmap_tool_result=claimmap_tool_result,
                methodmap_tool_result=methodmap_tool_result,
                sandbox_tool_result=sandbox_tool_result,
                document_tool_result=document_tool_result,
                session_context=session_context if session_context_enabled else None,
            )
            # Honest attribution in the permanent record: a reading turn is the
            # model's own act (it occupies the user slot in the chat template,
            # but the operator never typed it).
            memory.append_turn(
                "user",
                user_input,
                tags=["reading_turn" if reading_turn_source else "operator_input"],
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
            # Prioritize steer: lean the reply toward the top-ranked probe
            # (signed by its lift -- toward a concept that helps, away from one
            # that hurts), bounded by the same envelope. OFF at alpha 0.
            prio_alpha = tuner.get("prioritize_alpha", 0.0)
            prio_steered = None
            if prio_alpha and prio_alpha > 0 and probes:
                _pr = rank_probes(probes, tuner)
                if _pr and _pr[0]["priority"] > 0 and _pr[0]["lift"] is not None:
                    _pdir = probes[_pr[0]["name"]].get("direction") or {}
                    if _pdir:
                        _psign = 1.0 if _pr[0]["lift"] >= 0 else -1.0
                        try:
                            from invariants.engine import _steer_handles as _p_steer
                            steer_handles.extend(_p_steer(model, _pdir, list(_pdir.keys()), prio_alpha * _psign))
                            prio_steered = (_pr[0]["name"], _psign)
                        except Exception:
                            prio_steered = None
            if prio_steered is not None:
                print(
                    Fore.MAGENTA
                    + f"[Prioritize] steering {'toward' if prio_steered[1] > 0 else 'away from'} top probe '{prio_steered[0]}' (alpha {round(prio_alpha,4)})."
                    + Style.RESET_ALL
                )
            # Clock start: wall-time for THIS turn's generation (all
            # regenerations included), snapshotted before probe-scoring so the
            # reading reflects generation, not instrumentation. Peak VRAM is
            # reset here so max_memory_allocated captures this turn's spike.
            import time as _time
            _gen_t0 = _time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
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
            model_memory_tool_result = None
            model_claimmap_tool_result = None
            model_methodmap_tool_result = None
            model_doc_tool_result = None
            model_probe_tool_result = None
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
            if model_probe_query and probes:
                # READ-ONLY self-measurement: the model may consult sensors the
                # operator has exposed (:probe expose <name>) -- reading the
                # instrument is allowed, shaping it is not (no minting,
                # composing, dropping, or calibrating from its side). A
                # candidate-text read scores hypothetical words without
                # observing, crediting, or touching any rolling history.
                name_part, _, cand_text = model_probe_query.partition("||")
                cand_text = cand_text.strip()
                req_names = [re.sub(r"[^a-z0-9_]", "_", n.strip().lower())[:40] for n in name_part.split(",") if n.strip()]
                exposed_all = sorted(n for n in probes if probes[n].get("exposed"))
                if req_names == ["all"]:
                    req_names = list(exposed_all)
                readable = [n for n in req_names if n in probes and probes[n].get("exposed")]
                blocked = [n for n in req_names if n not in readable]
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
                for n in blocked:
                    probe_lines.append(f"- {n}: not a sensor I can consult.")
                if not probe_lines:
                    probe_lines.append(
                        "- No consultable sensors named. "
                        + (f"I can consult: {', '.join(exposed_all)}." if exposed_all else "None are exposed to me.")
                    )
                probe_cause = (
                    f'consulted my {", ".join(readable)} sensor(s)' if readable else "reached for a sensor"
                ) + (" on candidate words" if cand_text and readable else "")
                model_probe_tool_result = (
                    impact_note(probe_cause)
                    + "\n" + PROBE_TOOL_HEADER + "\n" + "\n".join(probe_lines)
                )
                turn_impacts.append({
                    "cause": probe_cause,
                    "effect": f"{len(readable)} reading(s) returned"
                              + (f", {len(blocked)} refused" if blocked else ""),
                })
                memory.append_event(
                    "probe_tool_model_requested",
                    text=model_probe_tool_result,
                    tags=["probe", "probe_tool", "activation_measurement"],
                    provenance={"query": model_probe_query[:240], "readable": readable, "blocked": blocked},
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
                if (
                    (active_memory_tool_result or active_claimmap_tool_result or active_methodmap_tool_result or active_document_tool_result or active_probe_tool_result)
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
                )
                if show_timestamps:
                    print(Fore.GREEN + Style.BRIGHT + f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Me: " + Style.RESET_ALL)
                print(response, end="")
                if session_context_enabled:
                    s_score = sense_score(synthesis_records) if synthesis_records else 0.0
                    if s_score is None: s_score = 0.0
                    session_context.append(("user", user_input, s_score))
                    session_context.append(("assistant", response, s_score))
                    if len(session_context) > MAX_SESSION_TURNS * 2:
                        if len(session_context) >= 6:
                            min_score = float('inf')
                            min_idx = 0
                            for i in range(0, len(session_context) - 4, 2):
                                pair_score = session_context[i][2] if len(session_context[i]) > 2 else 0.0
                                if pair_score < min_score:
                                    min_score = pair_score
                                    min_idx = i
                            session_context.pop(min_idx + 1)
                            session_context.pop(min_idx)
                        else:
                            session_context = session_context[-MAX_SESSION_TURNS * 2 :]
                memory.append_turn(
                    "assistant",
                    response,
                    tags=["model_output"],
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
            # generation_seconds is generation time only. VRAM in GB (live
            # allocation, total reserved footprint, and this turn's peak).
            gen_seconds = _time.perf_counter() - _gen_t0
            if torch.cuda.is_available():
                _gb = 1024 ** 3
                vram_alloc = torch.cuda.memory_allocated() / _gb
                vram_reserved = torch.cuda.memory_reserved() / _gb
                vram_peak = torch.cuda.max_memory_allocated() / _gb
            else:
                vram_alloc = vram_reserved = vram_peak = 0.0
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
                "ts": datetime.datetime.now().strftime("%H:%M:%S"),
            }
            print(
                Fore.CYAN
                + f"  [Clock] {gen_seconds:.1f}s"
                + (f" | {tps:.1f} tok/s" if tps else "")
                + f" | VRAM {vram_alloc:.2f}GB live / {vram_reserved:.2f}GB reserved"
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
            if doc_autoread and doc_autoread.get("until_settled") and reading_turn_source:
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
                        anchor_streams = [resolve_stream(p, tuner) for p in (anchor or "").split("+") if p]
                        if target_stream and anchor_streams and all(anchor_streams) and target_trigger in tuner.triggers:
                            anchors = []
                            for a_stream in anchor_streams:
                                a_trig = anchor_trigger_for(a_stream, tuner)
                                anchors.append((
                                    a_stream,
                                    a_trig.value if a_trig else 0.0,
                                    a_trig.comparator if a_trig else ">=",
                                ))
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
                        trig = tuner.triggers.get(cal_name) or tuner.triggers.get(f"probe_{cal_name}")
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

if __name__ == "__main__":
    main()
