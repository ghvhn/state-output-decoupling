import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''            if user_input.startswith(":save self"):'''
replacement = '''            if user_input.startswith(":self ") or user_input == ":self":
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
                        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
                        f"You are selecting a persona/macro to initialize.\\n"
                        f"Available options:\\n" + "\\n".join(f"- {p}" for p in sorted(macro_aliases.keys())) + "\\n\\n"
                        f"Select the single most appropriate persona/macro from the list. "
                        f"Output ONLY the exact name, and nothing else.<|eot_id|>"
                        f"<|start_header_id|>assistant<|end_header_id|>\\n\\n"
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

            if user_input.startswith(":save self"):'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched :self command")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
