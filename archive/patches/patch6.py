import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                cal_name = resolve_probe_choice(cal_name_raw, probes, model=model, config=config, action_name="calibrate")'''
replacement = '''                cal_name = resolve_probe_choice(cal_name_raw, calibratable_names(tuner), model=model, config=config, action_name="calibrate")'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched calibrate choice")
else:
    print("Target not found")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
