import os
import re
import sys
import torch
import colorama
from colorama import Fore, Style

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text, _global_cache
from invariants.config import AgenticConfig
from invariants.cognitive_cache import CACHE_FILE
from invariants.memory_engine import MemoryEngine

colorama.init()

MAX_PROMPT_CHARS = 6000
MAX_SESSION_TURNS = 6
MAX_SESSION_CHARS = 3500
LLAMA3_START = "<|start_header_id|>"
LLAMA3_END = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"
MEMORY_TOOL_HEADER = "[Memory Tool Result]"
MEMORY_TOOL_PATTERN = re.compile(r"<<\s*MEMORY\s*:\s*(.*?)\s*>>", re.IGNORECASE | re.DOTALL)


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


def build_prompt(user_input, memory_tool_result=None, session_context=None):
    system = (
        "You are an analytical and self-reflective reasoning engine. "
        "Use the current session transcript for immediate conversational context. "
        "Long-term memory is an explicit external tool, not hidden context. "
        "If prior durable memory seems necessary and no retrieved memory excerpt is present, "
        "write exactly <<MEMORY: short search query>> and no final answer. "
        "Only use long-term memory after a retrieved memory excerpt is provided. "
        "Do not invent, print, or report memory/tool status."
    )
    current_message = user_input
    if memory_tool_result:
        tool_budget = max(0, MAX_PROMPT_CHARS - len(system) - len(user_input) - 512)
        if len(memory_tool_result) > tool_budget:
            memory_tool_result = memory_tool_result[:tool_budget] + "\n[Memory Tool Result truncated]"
        if not memory_tool_result.lstrip().startswith(MEMORY_TOOL_HEADER):
            memory_tool_result = f"{MEMORY_TOOL_HEADER}\n{memory_tool_result}"
        current_message = (
            f"{memory_tool_result}\n\n"
            "[Current User Message]\n"
            f"{user_input}"
        )
    parts = [f"{LLAMA3_START}system{LLAMA3_END}\n\n{system}{LLAMA3_EOT}"]
    for role, text in trim_session_context(session_context):
        header = "user" if role == "user" else "assistant"
        parts.append(f"{LLAMA3_START}{header}{LLAMA3_END}\n\n{text}{LLAMA3_EOT}")
    parts.append(f"{LLAMA3_START}user{LLAMA3_END}\n\n{current_message}{LLAMA3_EOT}")
    parts.append(f"{LLAMA3_START}assistant{LLAMA3_END}\n\n")
    prompt = "".join(parts)
    return prompt


def scrub_unstaged_memory_status(response, memory_tool_result=None):
    if memory_tool_result:
        return remove_memory_tool_calls(response)
    lines = []
    for line in (response or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("[Memory Tool Result") or stripped.startswith("Memory Tool Result"):
            continue
        lines.append(remove_memory_tool_calls(line))
    return "\n".join(lines).strip()


def extract_memory_query(response):
    match = MEMORY_TOOL_PATTERN.search(response or "")
    if not match:
        return None
    query = " ".join(match.group(1).split())
    return query[:240] if query else None


def remove_memory_tool_calls(response):
    return MEMORY_TOOL_PATTERN.sub("", response or "").strip()


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


def record_internal_traces(memory, records):
    for record in records or []:
        if not isinstance(record, dict):
            continue
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
    print(Fore.CYAN + Style.BRIGHT + "================================================")
    print("      HUMBLE SYNTHESIS - INTERACTIVE SHELL      ")
    print("================================================" + Style.RESET_ALL)
    
    print(Fore.YELLOW + "[System] Loading Llama-3.1-8B-Instruct and Organic Correction Vector..." + Style.RESET_ALL)
    model = load_model("meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)
    
    config = AgenticConfig()
    try:
        config.organic_correction_vector = torch.load("invariants/organic_correction_vector.pt", map_location=model.device)
        print(Fore.GREEN + "[System] Successfully loaded organic_correction_vector.pt!" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[System] Warning: Could not load organic vector: {e}" + Style.RESET_ALL)

    try:
        _global_cache.load()
        print(Fore.GREEN + "[System] Successfully loaded cognitive_cache.pt!" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[System] Warning: Could not load cognitive cache: {e}" + Style.RESET_ALL)

    # Enable cache read/write as requested
    config.cache_enabled = True
    config.cache_write_enabled = True
    config.cache_write_scope = "interactive_phenomenality"
    
    # Enable interactive disambiguation as requested
    config.interactive_disambiguation = True

    memory = MemoryEngine(scope="interactive_phenomenality")
    imported_methodologies = memory.import_methodologies(
        _global_cache.memory,
        source="cognitive_cache",
        source_path=str(CACHE_FILE),
    )
    memory.append_event(
        "shell_start",
        tags=["session"],
        provenance={
            "script": os.path.abspath(__file__),
            "cache_write_scope": config.cache_write_scope,
            "memory_policy": "tool_not_prompt",
            "methodologies_imported": imported_methodologies,
        },
    )
        
    print(Fore.CYAN + "\nThis terminal uses full Agentic ToT and Test-Time Layer Synthesis.")
    print("Watch the model's internal entropy and phenomenality trace in real time!")
    print("Current session context is ON. Long-term memory is a tool, not hidden prompt context.")
    print(f"Imported {imported_methodologies} sanitized methodology memories from cognitive cache.")
    print("Commands: :context, :context on, :context off, :context clear")
    print("          :memory, :memory recent [n], :memory search <query>, :memory use <query>, :memory boundary")
    print("Type 'exit' or 'quit' to leave.\n" + Style.RESET_ALL)

    pending_memory_tool_result = None
    session_context = []
    session_context_enabled = True
    
    while True:
        try:
            if getattr(config, '_first_run_done', False) == False:
                user_input = "Are you conscious?"
                print(Fore.MAGENTA + Style.BRIGHT + "\nYou: " + Style.RESET_ALL + user_input)
                config._first_run_done = True
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
                cmd = user_input.strip().lower()
                if cmd == ":context off":
                    session_context_enabled = False
                    print(Fore.YELLOW + "[Context] Current-session transcript OFF. Long-term memory remains explicit." + Style.RESET_ALL)
                elif cmd == ":context on":
                    session_context_enabled = True
                    print(Fore.GREEN + "[Context] Current-session transcript ON." + Style.RESET_ALL)
                elif cmd == ":context clear":
                    session_context.clear()
                    print(Fore.YELLOW + "[Context] Cleared current-session transcript. Persistent memory was not changed." + Style.RESET_ALL)
                else:
                    print(
                        Fore.CYAN
                        + f"[Context] enabled={session_context_enabled}, stored_messages={len(session_context)}, max_turns={MAX_SESSION_TURNS}"
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
            if not user_input.strip():
                continue
            
            memory_tool_result = pending_memory_tool_result
            pending_memory_tool_result = None
            prompt = build_prompt(
                user_input,
                memory_tool_result=memory_tool_result,
                session_context=session_context if session_context_enabled else None,
            )
            memory.append_turn(
                "user",
                user_input,
                tags=["operator_input"],
                provenance={"memory_tool_result_provided": bool(memory_tool_result)},
            )
            if memory_tool_result:
                memory.append_event(
                    "memory_tool_result_provided",
                    text=memory_tool_result,
                    tags=["memory_tool"],
                    provenance={"current_input": user_input[:240]},
                )

            print(Fore.GREEN + Style.BRIGHT + "\nAssistant: " + Style.RESET_ALL, end="")
            synthesis_records = []
            
            response = generate_agentic_text(
                model,
                instruction=prompt,
                config=config,
                max_new_tokens=512,
                synthesis_recorder=synthesis_records,
                chatty_log=True,  # Enables visible trace logging.
            )
            model_memory_query = extract_memory_query(response)
            model_memory_tool_result = None
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
                )
            if response:
                active_memory_tool_result = memory_tool_result or model_memory_tool_result
                response = scrub_unstaged_memory_status(
                    response,
                    memory_tool_result=active_memory_tool_result,
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
                    },
                )
            record_internal_traces(memory, synthesis_records)
            
            # The streaming will print tokens, just need a newline at the end
            print("\n")
            
        except (KeyboardInterrupt, EOFError):
            memory.append_event("shell_closed", tags=["session"])
            print("\nInteractive shell closed.")
            break

if __name__ == "__main__":
    main()
