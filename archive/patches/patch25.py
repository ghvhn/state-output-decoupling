import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''            if user_input.strip().lower() == ":suggest apply":
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
                continue'''

replacement = '''            if user_input.strip().lower() == ":suggest apply" or user_input.strip().lower().startswith(":suggest apply "):
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
                continue'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched suggest apply to propagate because clauses")
else:
    print("Failed to patch suggest apply")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
