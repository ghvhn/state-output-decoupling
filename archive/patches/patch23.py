import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                    if cname in probes:
                        print(Fore.YELLOW + f"[Probe] '{cname}' is already an active probe (:probe drop {cname} first)." + Style.RESET_ALL)
                        continue'''

replacement = '''                    # Removed: if cname in probes: ... continue 
                    # so that users can overwrite/augment a probe using itself (e.g. compose amb amb + new)'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched compose to allow overwriting")
else:
    print("Failed to patch compose")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
