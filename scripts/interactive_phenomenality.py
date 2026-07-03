import datetime
import glob
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
    for role, text in reversed(session_context[-MAX_SESSION_TURNS * 2 :]):
        text = text or ""
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
):
    if (
        memory_tool_result
        or orientation_tool_result
        or claimmap_tool_result
        or methodmap_tool_result
        or sandbox_tool_result
        or document_tool_result
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


def remove_tool_calls(response):
    return remove_methodmap_tool_calls(remove_claimmap_tool_calls(remove_memory_tool_calls(remove_doc_tool_calls(response))))


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
        (extract_memory_query(text) or extract_claimmap_payload(text) or extract_methodmap_query(text) or extract_doc_query(text))
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
        closed_sessions = {
            r.session_id
            for r in memory.records
            if r.scope == memory.scope and r.kind == "event" and r.text == "shell_closed"
        }
        candidate_sessions = []
        for record in scope_records:
            if record.session_id == memory.session_id or record.session_id not in closed_sessions:
                continue
            if not candidate_sessions or candidate_sessions[-1] != record.session_id:
                candidate_sessions.append(record.session_id)
        session_id = candidate_sessions[-1] if candidate_sessions else None
    matches = [r for r in scope_records if r.session_id == session_id]
    if not matches:
        return None, []
    max_messages = max(1, int(max_turns)) * 2
    recovered = [(r.role, r.text) for r in matches[-max_messages:]]
    return session_id, recovered


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
    config.max_synthesis_steps = max(
        0, int(_sane_fraction(tuner.get("synthesis_steps", config.max_synthesis_steps), config.max_synthesis_steps))
    )
    config.epsilon = _sane_fraction(tuner.get("plateau_epsilon", config.epsilon), config.epsilon)
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
    # Synthesis schedule, same shape: events per reply, optimizer steps per
    # event, and the plateau-velocity trigger that starts one.
    tuner.register("synthesis_events", config.max_synthesis_events, kind="coefficient")
    tuner.register("synthesis_steps", config.max_synthesis_steps, kind="coefficient")
    tuner.register("plateau_epsilon", config.epsilon, kind="coefficient")
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
            return [], {}
        n_layers = getattr(model, "n_layers", None) or len(model.model.model.layers)
        layers = pick_sweep_layers(steer_band_layers(n_layers), steer_map.layer_steer_counts(channel), width)
        drift = axis_drift(steer_vecs) if steer_vecs else None
        drifts = {L: drift_at_layer(drift, L) for L in layers} if drift else {L: None for L in layers}
        return layers, drifts

    _sync_steer_tunables(tuner, config)

    # Tool-sensing seam: any tool fires mid-thought when its state trigger crosses.
    # Detectors read (text, state) -- state carries the live phenomenality scores,
    # so a tool triggers from activation, not a tag. add a tool = register a detector.
    tool_sense = ToolSense(model, tuner)

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
            _mls, _ = _sweep_layers_for("memory")
            if _mls:
                sweep_layers = _mls
                for _sl in _mls:
                    sweep_state["fires"].append(("memory", _sl, alpha, None, len(_mls)))
        return _memory_steer_handles(m, (records[0].text or "").strip(),
                                     alpha=alpha, layers=sweep_layers)

    def _claimmap_act(payload, m):
        a, b = payload
        cm = analyze_claim_pair(f"{a} || {b}", model=m)
        alpha = tuner.get("claimmap_alpha", 0.0)
        sweep_layers = None
        if alpha > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
            layer, drift = _sweep_layer_for("claimmap", cm.steer_delta)
            if layer is not None:
                sweep_layers = [layer]
                sweep_state["fires"].append(("claimmap", layer, alpha, drift))
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
    print("                course -- updated = by file mtime, chronological; reply follows")
    print("                overlap. 'satisfied' stops early once sense settles)")
    print("          :doc inject                  (stage the whole library for one turn, budget-bounded)")
    print("          :doc stop                    (interrupt auto-read)")
    print("          :sandbox on|off|status       (run the model's ```python blocks for real)")
    print("          :experts on|off|status       (mint new steering experts from its own")
    print("                recurring self-corrections; roster bounded; default off)")
    print("          :impact                      (consequence trail: what its words caused,")
    print("                and whether experienced impact tracks better deliberation)")
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
    MAX_AUTOREAD = 20           # per :doc read command; reading stays a deliberate act
    sandbox_enabled = False     # deliberate opt-in, like every intervention here
    session_context = []
    session_context_enabled = True
    startup_user_input = os.environ.get("PHENOMENALITY_STARTUP_PROMPT")
    if os.environ.get("PHENOMENALITY_AUTO_RESUME", "0").strip().lower() in {"1", "true", "yes"}:
        resumed_session, recovered = recover_session_context(
            memory,
            session_id=os.environ.get("PHENOMENALITY_RESUME_SESSION", "last"),
            max_turns=MAX_SESSION_TURNS,
        )
        if recovered and recovered[-1][0] == "user":
            session_context = recovered[:-1]
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
    
    while True:
        try:
            reading_turn_source = None  # set when the turn is the model reading, not the operator speaking
            if startup_user_input:
                user_input = startup_user_input
                print(Fore.MAGENTA + Style.BRIGHT + "\nYou: " + Style.RESET_ALL + user_input)
                startup_user_input = None
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
                user_input = input(Fore.MAGENTA + Style.BRIGHT + "\nYou: " + Style.RESET_ALL)
                
            if user_input.lower() in ['exit', 'quit']:
                memory.append_event("shell_closed", tags=["session"], provenance={"reason": "operator_exit"})
                break
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
                        session_context = recovered
                        session_context_enabled = True
                        memory.append_event(
                            "context_resumed",
                            tags=["memory_tool", "context"],
                            provenance={
                                "resumed_session_id": resumed_session,
                                "messages": len(recovered),
                            },
                        )
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
                            "updated": "in the order the files were last written (chronological)",
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
                    pct = float(targs[2]) if len(targs) >= 3 else 80.0
                    v = tuner.calibrate(targs[0], pct)
                    if v is None:
                        print(Fore.RED + f"[Tune] Unknown trigger '{targs[0]}'." + Style.RESET_ALL)
                    else:
                        print(Fore.GREEN + f"[Tune] {targs[0]} calibrated to p{pct:g} = {round(v, 4)}" + Style.RESET_ALL)
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
            claimmap_alpha_used = tuner.get("claimmap_alpha", 0.0) if claimmap_steer_delta else 0.0
            turn_sweep_layers = None
            if claimmap_steer_delta and claimmap_alpha_used > 0 and tuner.get("steer_layer_sweep", 0.0) > 0:
                sweep_layers, sweep_drifts = _sweep_layers_for("claimmap", claimmap_steer_delta)
                if sweep_layers:
                    turn_sweep_layers = sweep_layers
                    for _sl in sweep_layers:
                        sweep_state["fires"].append(("claimmap", _sl, claimmap_alpha_used, sweep_drifts.get(_sl), len(sweep_layers)))
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
            model_memory_tool_result = None
            model_claimmap_tool_result = None
            model_methodmap_tool_result = None
            model_doc_tool_result = None
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
                    _tls, _tds = _sweep_layers_for("claimmap", model_claimmap_steer)
                    if _tls:
                        tag_sweep_layers = _tls
                        for _sl in _tls:
                            sweep_state["fires"].append(("claimmap", _sl, tag_alpha, _tds.get(_sl), len(_tls)))
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
            if response:
                active_memory_tool_result = memory_tool_result or model_memory_tool_result
                active_orientation_tool_result = orientation_tool_result
                active_claimmap_tool_result = claimmap_tool_result or model_claimmap_tool_result
                active_methodmap_tool_result = methodmap_tool_result or model_methodmap_tool_result
                active_document_tool_result = document_tool_result or model_doc_tool_result
                active_sandbox_tool_result = sandbox_tool_result
                if (
                    (active_memory_tool_result or active_claimmap_tool_result or active_methodmap_tool_result or active_document_tool_result)
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
                )
                print(response, end="")
                if session_context_enabled:
                    session_context.append(("user", user_input))
                    session_context.append(("assistant", response))
                    if len(session_context) > MAX_SESSION_TURNS * 2:
                        session_context = session_context[-MAX_SESSION_TURNS * 2 :]
                memory.append_turn(
                    "assistant",
                    response,
                    tags=["model_output"],
                    metrics={
                        "chars": len(response),
                        "model_memory_tool_requested": bool(model_memory_tool_result),
                        "orientation_tool_result_provided": bool(active_orientation_tool_result),
                        "model_claimmap_tool_requested": bool(model_claimmap_tool_result),
                        "claimmap_tool_result_provided": bool(active_claimmap_tool_result),
                        "model_methodmap_tool_requested": bool(model_methodmap_tool_result),
                        "methodmap_tool_result_provided": bool(active_methodmap_tool_result),
                        "document_tool_result_provided": bool(active_document_tool_result),
                        "sandbox_tool_result_provided": bool(active_sandbox_tool_result),
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
            # Intent axis: did this turn settle intent (lower
            # ambiguity+disagreement than last turn)? Observed every turn the
            # sensors fire; paired with sense via the credit channel.
            phen_now = latest_phenomenality_scores(synthesis_records) or {}
            if phen_now:
                unsettled_now = float(phen_now.get("ambiguity", 0.0)) + float(phen_now.get("disagreement", 0.0))
                if prev_unsettledness is not None:
                    intent_signal = prev_unsettledness - unsettled_now
                    tuner.observe("intent_settling", intent_signal)
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
            # Layer isolation evidence: each single-layer steer this turn lands
            # on ITS layer in the steer map, labeled by the turn's outcome —
            # per-layer success accrues with no band confound.
            for sweep_channel, sweep_l, sweep_alpha, sweep_d, sweep_w in sweep_state["fires"]:
                steer_map.record_layer_steer(
                    sweep_channel,
                    sweep_l,
                    sweep_alpha,
                    conversation_outcome=conversation_outcome,
                    metrics={"axis_drift_at_layer": sweep_d, "sweep_width": sweep_w},
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
            
        except (KeyboardInterrupt, EOFError):
            memory.append_event("shell_closed", tags=["session"])
            print("\nInteractive shell closed.")
            break

if __name__ == "__main__":
    main()
