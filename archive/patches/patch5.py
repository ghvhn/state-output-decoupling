import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update resolve_probe_choice to handle general options (knobs and probes)
target_resolve = '''def resolve_probe_choice(pname_raw, probes, model=None, config=None, action_name=""):
    tokens = pname_raw.split()
    if not tokens:
        return ""
    base = tokens[0].upper()
    if base in ("CHOICE", "CHOOSE", "AUTO"):
        if not probes:
            print(Fore.YELLOW + f"[{base.capitalize()}] No active probes to choose from." + Style.RESET_ALL)
            return None
        if not model or not config:
            print(Fore.RED + "[Error] Model not available for choice." + Style.RESET_ALL)
            return None
        
        plist = list(probes.keys())
        refs = tokens[1:]
        ref_str = ""
        if refs:
            ref_str = f"The user has provided the following reference guidance: {' '.join(refs)}\\nUse this guidance to inform your selection.\\n\\n"
            
        print(Fore.CYAN + f"[{base.capitalize()}] Asking the model to select a probe for '{action_name}'..." + Style.RESET_ALL)
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
            f"You are selecting a cognitive probe for the action: '{action_name}'.\\n"
            f"Available probes:\\n" + "\\n".join(f"- {p}" for p in plist) + "\\n\\n"
            f"{ref_str}"
            f"Select the single most appropriate probe from the list above. "
            f"Output ONLY the exact name of the probe, and nothing else.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\\n\\n"
        )

        # generate_agentic_text is imported at module level; no local import (it
        # would make the name function-local and risk UnboundLocalError).
        sug = generate_agentic_text(
            model,
            instruction=prompt,
            config=config,
            pre_formatted=True,
            max_new_tokens=20
        )
        sug = sug.strip()
        if sug in probes:
            print(Fore.GREEN + f"[Choice] The model chose: {sug}" + Style.RESET_ALL)
            return sug
        else:
            print(Fore.YELLOW + f"[Choice] Model selected invalid probe '{sug}'. Aborting." + Style.RESET_ALL)
            return None
            
    return pname_raw.replace("probe_", "") if pname_raw.startswith("probe_") else pname_raw'''

replacement_resolve = '''def resolve_probe_choice(pname_raw, options, model=None, config=None, action_name=""):
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
            ref_str = f"The user has provided the following reference guidance: {' '.join(refs)}\\nUse this guidance to inform your selection.\\n\\n"
            
        print(Fore.CYAN + f"[{base.capitalize()}] Asking the model to select a target for '{action_name}'..." + Style.RESET_ALL)
        
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n"
            f"You are selecting a target parameter/probe for the action: '{action_name}'.\\n"
            f"Available options:\\n" + "\\n".join(f"- {p}" for p in plist) + "\\n\\n"
            f"{ref_str}"
            f"Select the single most appropriate target from the list above. "
            f"Output ONLY the exact name of the target, and nothing else.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\\n\\n"
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
            
    return pname_raw.replace("probe_", "") if pname_raw.startswith("probe_") else pname_raw'''

if target_resolve in content:
    content = content.replace(target_resolve, replacement_resolve)
    print("Replaced resolve_probe_choice")
else:
    print("Could not find resolve_probe_choice chunk!")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
