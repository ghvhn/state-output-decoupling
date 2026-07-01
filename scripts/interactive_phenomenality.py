import os
import re
import sys
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
from invariants.memory_engine import MemoryEngine
from invariants.self_concept_controller import SelfConceptController, format_orientation_tool_result
from invariants.steer_map_store import SteerMapStore

colorama.init()

MAX_PROMPT_CHARS = 6000
MAX_SESSION_TURNS = 6
MAX_SESSION_CHARS = 3500
LLAMA3_START = "<|start_header_id|>"
LLAMA3_END = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"
MEMORY_TOOL_HEADER = "[Memory Tool Result]"
ORIENTATION_TOOL_HEADER = "[Orientation Tool Result]"
MEMORY_TOOL_PATTERN = re.compile(r"<<\s*MEMORY\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
CLAIMMAP_TOOL_PATTERN = re.compile(r"<<\s*CLAIMMAP\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
METHODMAP_TOOL_HEADER = "[MethodMap Tool Result]"
METHODMAP_TOOL_PATTERN = re.compile(r"<<\s*METHODMAP\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)
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
            claimmap_tool_result,   # already pure second-person felt language
            memory_tool_result,
            orientation_tool_result,
            methodmap_tool_result,
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
):
    if memory_tool_result or orientation_tool_result or claimmap_tool_result or methodmap_tool_result:
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


def remove_claimmap_tool_calls(response):
    return CLAIMMAP_TOOL_PATTERN.sub("", response or "").strip()


def remove_methodmap_tool_calls(response):
    return METHODMAP_TOOL_PATTERN.sub("", response or "").strip()


def remove_tool_calls(response):
    return remove_methodmap_tool_calls(remove_claimmap_tool_calls(remove_memory_tool_calls(response)))


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
        (extract_memory_query(text) or extract_claimmap_payload(text) or extract_methodmap_query(text))
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


def record_internal_traces(memory, records, steer_map=None):
    for record in records or []:
        if not isinstance(record, dict):
            continue
        if steer_map is not None:
            steer_map.record_synthesis_record(
                record,
                source="interactive",
                method="interactive_phenomenality",
                final_correct=None,
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
    
    # Enable interactive disambiguation as requested
    config.interactive_disambiguation = True

    memory = MemoryEngine(scope="interactive_phenomenality")
    self_concept = SelfConceptController()
    steer_map = SteerMapStore()
    tuner = TriggerTuner()
    # Every trigger is born tunable. Persisted values win over these defaults.
    tuner.register("claimmap_tension", 0.18, kind="threshold", comparator=">=")
    tuner.register("claimmap_alpha", 0.5, kind="coefficient")
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
    print("          :tune, :tune <name> <value>, :tune <name> auto [percentile]")
    print("Type 'exit' or 'quit' to leave.\n" + Style.RESET_ALL)

    pending_memory_tool_result = None
    pending_orientation_tool_result = None
    pending_claimmap_tool_result = None
    pending_claimmap_steer_delta = None
    pending_methodmap_tool_result = None
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
            if startup_user_input:
                user_input = startup_user_input
                print(Fore.MAGENTA + Style.BRIGHT + "\nYou: " + Style.RESET_ALL + user_input)
                startup_user_input = None
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
            if user_input.startswith(":tune"):
                targs = user_input[len(":tune"):].split()
                if not targs:
                    rows = tuner.summary()
                    if not rows:
                        print(Fore.YELLOW + "[Tune] No triggers registered yet." + Style.RESET_ALL)
                    for s in rows:
                        print(
                            Fore.CYAN
                            + (
                                f"  {s['name']}: {s['kind']} value={s['value']} [{s['comparator']}] "
                                f"fire_rate={s['fire_rate']} "
                                f"signal(min/med/max)={s['signal_min']}/{s['signal_med']}/{s['signal_max']} "
                                f"n={s['n_signals']}"
                            )
                            + Style.RESET_ALL
                        )
                elif len(targs) >= 2 and targs[1].lower() == "auto":
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
            pending_memory_tool_result = None
            pending_orientation_tool_result = None
            pending_claimmap_tool_result = None
            pending_claimmap_steer_delta = None
            pending_methodmap_tool_result = None
            prompt = build_prompt(
                user_input,
                memory_tool_result=memory_tool_result,
                orientation_tool_result=orientation_tool_result,
                claimmap_tool_result=claimmap_tool_result,
                methodmap_tool_result=methodmap_tool_result,
                session_context=session_context if session_context_enabled else None,
            )
            memory.append_turn(
                "user",
                user_input,
                tags=["operator_input"],
                provenance={
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

            print(Fore.GREEN + Style.BRIGHT + "\nAssistant: " + Style.RESET_ALL, end="")
            synthesis_records = []

            steer_handles = (
                claimmap_steer_handles(model, claimmap_steer_delta, alpha=tuner.get("claimmap_alpha", 0.5))
                if claimmap_steer_delta else []
            )
            try:
                response = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=512,
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,  # Enables visible trace logging.
                    pre_formatted=True,
                )
            finally:
                for h in steer_handles:
                    h.remove()
            model_memory_query = extract_memory_query(response)
            model_claimmap_payload = extract_claimmap_payload(response)
            model_methodmap_query = extract_methodmap_query(response)
            model_memory_tool_result = None
            model_claimmap_tool_result = None
            model_methodmap_tool_result = None
            if model_memory_query and memory_tool_result is None:
                records = memory.search(model_memory_query, max_records=6, scope=memory.scope)
                model_memory_tool_result = memory.format_tool_result(records)
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
                print(Fore.GREEN + Style.BRIGHT + "\nAssistant: " + Style.RESET_ALL, end="")
                response = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=512,
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,
                    pre_formatted=True,
                )
                model_claimmap_payload = extract_claimmap_payload(response)
                model_methodmap_query = extract_methodmap_query(response)
            if model_claimmap_payload and claimmap_tool_result is None:
                model_claimmap_steer = None
                try:
                    cm = analyze_claim_pair(model_claimmap_payload, model=model)
                    model_claimmap_tool_result = cm.felt          # felt only reaches the model
                    model_claimmap_steer = cm.steer_delta
                    telemetry_for_log = cm.telemetry              # raw numbers logged, never in the prompt
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
                print(Fore.GREEN + Style.BRIGHT + "\nAssistant: " + Style.RESET_ALL, end="")
                steer_handles = (
                    claimmap_steer_handles(model, model_claimmap_steer, alpha=tuner.get("claimmap_alpha", 0.5))
                    if model_claimmap_steer else []
                )
                try:
                    response = generate_agentic_text(
                        model,
                        instruction=prompt,
                        config=config,
                        max_new_tokens=512,
                        synthesis_recorder=synthesis_records,
                        chatty_log=True,
                        pre_formatted=True,
                    )
                finally:
                    for h in steer_handles:
                        h.remove()
                model_methodmap_query = extract_methodmap_query(response)
            if model_methodmap_query and methodmap_tool_result is None:
                model_methodmap_tool_result = format_methodmap_tool_result(memory, model_methodmap_query)
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
                print(Fore.GREEN + Style.BRIGHT + "\nAssistant: " + Style.RESET_ALL, end="")
                response = generate_agentic_text(
                    model,
                    instruction=prompt,
                    config=config,
                    max_new_tokens=512,
                    synthesis_recorder=synthesis_records,
                    chatty_log=True,
                    pre_formatted=True,
                )
            if response:
                active_memory_tool_result = memory_tool_result or model_memory_tool_result
                active_orientation_tool_result = orientation_tool_result
                active_claimmap_tool_result = claimmap_tool_result or model_claimmap_tool_result
                active_methodmap_tool_result = methodmap_tool_result or model_methodmap_tool_result
                if (
                    (active_memory_tool_result or active_claimmap_tool_result or active_methodmap_tool_result)
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
                        + "You already received the requested tool result. Do not emit another tool-call tag. "
                        + "Answer now using the provided tool result."
                    )
                    prompt = build_prompt(
                        retry_input,
                        memory_tool_result=active_memory_tool_result,
                        orientation_tool_result=active_orientation_tool_result,
                        claimmap_tool_result=active_claimmap_tool_result,
                        methodmap_tool_result=active_methodmap_tool_result,
                        session_context=session_context if session_context_enabled else None,
                    )
                    print(Fore.GREEN + Style.BRIGHT + "\nAssistant: " + Style.RESET_ALL, end="")
                    response = generate_agentic_text(
                        model,
                        instruction=prompt,
                        config=config,
                        max_new_tokens=512,
                        synthesis_recorder=synthesis_records,
                        chatty_log=True,
                        pre_formatted=True,
                    )
                response = scrub_unstaged_memory_status(
                    response,
                    memory_tool_result=active_memory_tool_result,
                    orientation_tool_result=active_orientation_tool_result,
                    claimmap_tool_result=active_claimmap_tool_result,
                    methodmap_tool_result=active_methodmap_tool_result,
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
                    },
                )
            record_internal_traces(memory, synthesis_records, steer_map=steer_map)

            # Activation-reach: no tag was taught in bare mode, so tools fire from
            # the model's own surfaced state. If the answer holds two opposed
            # framings, sense the comparison (felt) and stage it -- with steering --
            # for the next turn. Disable with CLAIMMAP_AUTO_TRIGGER=0.
            if (
                os.environ.get("CLAIMMAP_AUTO_TRIGGER", "1").strip() not in {"0", "false", "no"}
                and pending_claimmap_tool_result is None
            ):
                a, b, tension_score = framing_tension_score(response or "")
                # Log the tension signal EVERY turn (even 0) so :tune can read the
                # distribution; fire on the live-tuned threshold, not a fixed cutoff.
                fired = tuner.observe("claimmap_tension", tension_score)
                if fired and a is not None:
                    try:
                        cm = analyze_claim_pair(f"{a} || {b}", model=model)
                        pending_claimmap_tool_result = cm.felt
                        pending_claimmap_steer_delta = cm.steer_delta
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
