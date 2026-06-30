import os
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
LLAMA3_START = "<|start_header_id|>"
LLAMA3_END = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"


def build_prompt(user_input, memory_tool_result=None):
    system = (
        "You are an analytical and self-reflective reasoning engine. "
        "Memory is an explicit tool, not hidden context. "
        "If a [Memory Tool Result] block is present in the current message, use it only when relevant. "
        "If no memory tool result is present, do not pretend to remember prior turns."
    )
    current_message = user_input
    if memory_tool_result:
        tool_budget = max(0, MAX_PROMPT_CHARS - len(system) - len(user_input) - 512)
        if len(memory_tool_result) > tool_budget:
            memory_tool_result = memory_tool_result[:tool_budget] + "\n[Memory Tool Result truncated]"
        current_message = (
            f"{memory_tool_result}\n\n"
            "[Current User Message]\n"
            f"{user_input}"
        )
    parts = [
        f"{LLAMA3_START}system{LLAMA3_END}\n\n{system}{LLAMA3_EOT}",
        f"{LLAMA3_START}user{LLAMA3_END}\n\n{current_message}{LLAMA3_EOT}",
        f"{LLAMA3_START}assistant{LLAMA3_END}\n\n",
    ]
    prompt = "".join(parts)
    return prompt


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
    print("Memory is a tool, not hidden prompt context; cache writes remain flagged as interactive_phenomenality.")
    print(f"Imported {imported_methodologies} sanitized methodology memories from cognitive cache.")
    print("Commands: :memory, :memory recent [n], :memory search <query>, :memory use <query>, :memory boundary")
    print("Type 'exit' or 'quit' to leave.\n" + Style.RESET_ALL)

    pending_memory_tool_result = None
    
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
                    + "[History] Automatic prompt history is disabled. Use :memory search or :memory use as an explicit tool."
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
            prompt = build_prompt(user_input, memory_tool_result=memory_tool_result)
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
            
            response = generate_agentic_text(
                model,
                instruction=prompt,
                config=config,
                max_new_tokens=512,
                synthesis_recorder=None,
                chatty_log=True,  # Enables visible trace logging.
            )
            if response:
                print(response, end="")
                memory.append_turn(
                    "assistant",
                    response,
                    tags=["model_output"],
                    metrics={"chars": len(response)},
                )
            
            # The streaming will print tokens, just need a newline at the end
            print("\n")
            
        except (KeyboardInterrupt, EOFError):
            memory.append_event("shell_closed", tags=["session"])
            print("\nInteractive shell closed.")
            break

if __name__ == "__main__":
    main()
