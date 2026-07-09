import re
import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                related = set()
                # Check for probes
                for p in probes:
                    if re.search(r'\\b' + re.escape(p) + r'\\b', command_because, re.IGNORECASE):
                        related.add(f"probe:{p}")
                # Check for knobs (triggers)
                for k in tuner.triggers:
                    k_name = k[len("probe_"):] if k.startswith("probe_") else k
                    if re.search(r'\\b' + re.escape(k_name) + r'\\b', command_because, re.IGNORECASE):
                        related.add(f"knob:{k_name}")'''

replacement = '''                related = set()
                # Check for probes
                for p in probes:
                    match = re.search(r'(?<!\\w)([+-]?)' + re.escape(p) + r'\\b', command_because, re.IGNORECASE)
                    if match:
                        prefix = match.group(1) or ""
                        related.add(f"{prefix}probe:{p}")
                # Check for knobs (triggers)
                for k in tuner.triggers:
                    k_name = k[len("probe_"):] if k.startswith("probe_") else k
                    match = re.search(r'(?<!\\w)([+-]?)' + re.escape(k_name) + r'\\b', command_because, re.IGNORECASE)
                    if match:
                        prefix = match.group(1) or ""
                        related.add(f"{prefix}knob:{k_name}")'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched because clause regex to support +/- prefix")
else:
    print("Failed to patch because clause")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
