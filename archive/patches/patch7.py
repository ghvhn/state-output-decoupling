import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix :tune dynamic
target1 = '''                    parts = user_input[len(":tune"):].strip().split()
                    try:
                        dyn_idx = [p.lower() for p in parts].index("dynamic")'''
replacement1 = '''                    parts = user_input[len(":tune"):].strip().split()
                    if parts and parts[0].upper() in ("CHOICE", "CHOOSE", "AUTO"):
                        resolved = resolve_probe_choice(parts[0], calibratable_names(tuner), model=model, config=config, action_name="tune")
                        if not resolved:
                            continue
                        parts[0] = resolved
                    try:
                        dyn_idx = [p.lower() for p in parts].index("dynamic")'''

if target1 in content:
    content = content.replace(target1, replacement1)
    print("Patched :tune dynamic")

# Fix :tune static
target2 = '''                targs = user_input[len(":tune"):].split()
                if not targs:'''
replacement2 = '''                targs = user_input[len(":tune"):].split()
                if targs and targs[0].upper() in ("CHOICE", "CHOOSE", "AUTO"):
                    resolved = resolve_probe_choice(targs[0], calibratable_names(tuner), model=model, config=config, action_name="tune")
                    if not resolved:
                        continue
                    targs[0] = resolved
                if not targs:'''

if target2 in content:
    content = content.replace(target2, replacement2)
    print("Patched :tune static")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
