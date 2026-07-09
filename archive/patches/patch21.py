import re
import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                staged_ctx = ""
                if pending_memory_tool_result:
                    _ctx = pending_memory_tool_result.strip()
                    if len(_ctx) > 2000:
                        _ctx = _ctx[:2000] + " ...[truncated]"
                    staged_ctx = (
                        "The operator staged this context for you; let it shape the macro:\\n"
                        + _ctx + "\\n\\n"
                    )
                    pending_memory_tool_result = None
                    print(Fore.CYAN + "[Solve] Folding in the memory you staged with :memory use." + Style.RESET_ALL)

                prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
                    "You write macros for an interactive cognition shell. A macro is a list of ':' "
                    "commands, one per line." + prompt_args_str + "\\n\\n"
                    f"{command_hints_str}"
                    "Note: Shell commands natively accept 'auto' or 'choose' as arguments where applicable "
                    "to automatically select or interactively prompt for a value. "
                    "You should seamlessly pass these through to the underlying commands if the user provides them, "
                    "unless a parameter is explicitly restricted from doing so.\\n\\n"
                    f"{macro_hints_str}"
                    f"{staged_ctx}"
                    f"Write a macro named '{sname}' that does: {goal}\\n"
                    "Output ONLY the command lines, nothing else.<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
                )'''

replacement = '''                staged_ctx = ""
                if pending_memory_tool_result:
                    _ctx = pending_memory_tool_result.strip()
                    if len(_ctx) > 2000:
                        _ctx = _ctx[:2000] + " ...[truncated]"
                    staged_ctx = (
                        "The operator staged this context for you; let it shape the macro:\\n"
                        + _ctx + "\\n\\n"
                    )
                    pending_memory_tool_result = None
                    print(Fore.CYAN + "[Solve] Folding in the memory you staged with :memory use." + Style.RESET_ALL)
                    
                because_ctx = ""
                if command_because:
                    because_ctx = f"The operator provided the following underlying reason/rationale for this macro:\\n{command_because}\\nMake sure your generated macro commands strongly reflect this rationale.\\n\\n"
                    print(Fore.CYAN + "[Solve] Passing your 'because' rationale to the model." + Style.RESET_ALL)

                prompt = (
                    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
                    "You write macros for an interactive cognition shell. A macro is a list of ':' "
                    "commands, one per line." + prompt_args_str + "\\n\\n"
                    f"{command_hints_str}"
                    "Note: Shell commands natively accept 'auto' or 'choose' as arguments where applicable "
                    "to automatically select or interactively prompt for a value. "
                    "You should seamlessly pass these through to the underlying commands if the user provides them, "
                    "unless a parameter is explicitly restricted from doing so.\\n\\n"
                    f"{macro_hints_str}"
                    f"{staged_ctx}"
                    f"{because_ctx}"
                    f"Write a macro named '{sname}' that does: {goal}\\n"
                    "Output ONLY the command lines, nothing else.<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
                )'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched solve prompt with because ctx")
else:
    print("Failed to patch solve prompt")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
