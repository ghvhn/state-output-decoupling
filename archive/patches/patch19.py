import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''            # Swap global state if target_agent
            if target_agent:
                _sys_tuner = tuner
                _sys_probes = probes
                _sys_tb = tuner_bindings
                tuner = target_agent.tuner
                probes = target_agent.probes
                tuner_bindings = target_agent.tuner_bindings

                if command_because:
                    memory.append_event(
                        "command_because",
                        tags=["provenance"],
                        provenance={"command": user_input.split()[0] if user_input else "", "because": command_because},
                    )
                    print(Fore.BLUE + f"[Because] noted: {command_because}" + Style.RESET_ALL)'''

replacement = '''            # Swap global state if target_agent
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
                    if re.search(r'\\b' + re.escape(p) + r'\\b', command_because, re.IGNORECASE):
                        related.add(f"probe:{p}")
                # Check for knobs (triggers)
                for k in tuner.triggers:
                    k_name = k[len("probe_"):] if k.startswith("probe_") else k
                    if re.search(r'\\b' + re.escape(k_name) + r'\\b', command_because, re.IGNORECASE):
                        related.add(f"knob:{k_name}")
                        
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
                print(Fore.BLUE + f"[Because] noted: {noted_str}" + Style.RESET_ALL)'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched because handler")
else:
    print("Failed to patch because handler")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
