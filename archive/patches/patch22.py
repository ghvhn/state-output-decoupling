import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target1 = '''            else:
                _last_replace_name = None
                prefix = f"\\n[{datetime.datetime.now().strftime('%H:%M:%S')}] You: " if show_timestamps else "\\nYou: "'''

replacement1 = '''            else:
                _last_replace_name = None
                if getattr(config, "auto_probe_readings", False) and _sys_probes:
                    recent = []
                    for row in reversed(list(turn_log)):
                        v = {}
                        for p in _sys_probes:
                            if f"probe_{p}" in row:
                                try:
                                    v[p] = float(row[f"probe_{p}"])
                                except (TypeError, ValueError):
                                    pass
                        if v:
                            recent.append(v)
                            break
                    if recent:
                        print(Fore.CYAN + "[Auto Probe Readings]" + Style.RESET_ALL)
                        for pname, val in recent[0].items():
                            print(Fore.CYAN + f"  {pname}: {val:+.3f}" + Style.RESET_ALL)
                            
                prefix = f"\\n[{datetime.datetime.now().strftime('%H:%M:%S')}] You: " if show_timestamps else "\\nYou: "'''

target2 = '''                if pargs.lower() in ("values", "value", "recent", "last") or pargs.lower().startswith(("values ", "value ", "recent ", "last ")):'''

replacement2 = '''                if pargs.lower().startswith("auto"):
                    vparts = pargs.split()
                    if len(vparts) > 1:
                        state = vparts[1].lower()
                        if state == "on":
                            config.auto_probe_readings = True
                            print(Fore.CYAN + "[Probe] Auto-readings enabled. Will print values before every input prompt." + Style.RESET_ALL)
                        else:
                            config.auto_probe_readings = False
                            print(Fore.CYAN + "[Probe] Auto-readings disabled." + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + "[Probe] Usage: :probe auto [on|off]" + Style.RESET_ALL)
                    continue

                if pargs.lower() in ("values", "value", "recent", "last") or pargs.lower().startswith(("values ", "value ", "recent ", "last ")):'''

patched = False
if target1 in content and target2 in content:
    content = content.replace(target1, replacement1)
    content = content.replace(target2, replacement2)
    patched = True
    print("Patched auto probe readings")
else:
    print("Failed to patch auto probe readings. Target not found.")

if patched:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
