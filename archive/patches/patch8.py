import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Add helper functions at the top, just after imports
target_imports = '''import torch
from transformers import AutoTokenizer, PreTrainedModel

from invariants.trigger_tuner import TriggerTuner
from invariants import engine as _engine'''

replacement_imports = '''import torch
from transformers import AutoTokenizer, PreTrainedModel

from invariants.trigger_tuner import TriggerTuner
from invariants import engine as _engine

def _split_macro_commands(raw_str):
    cmds = []
    curr = []
    i = 0
    while i < len(raw_str):
        if raw_str[i] == '\\\\' and i+1 < len(raw_str):
            nxt = raw_str[i+1]
            if nxt == '|' and i+2 < len(raw_str) and raw_str[i+2] == '|':
                curr.append('||')
                i += 3
                continue
            elif nxt in (';', '\\\\'):
                curr.append(nxt)
                i += 2
                continue
        elif raw_str[i] == ';':
            cmds.append("".join(curr).strip())
            curr = []
            i += 1
            continue
        curr.append(raw_str[i])
        i += 1
    if curr:
        cmds.append("".join(curr).strip())
    return [c for c in cmds if c]

def _partition_unescaped_pipes(raw_str):
    curr = []
    i = 0
    while i < len(raw_str):
        if raw_str[i] == '\\\\' and i+1 < len(raw_str):
            nxt = raw_str[i+1]
            if nxt == '|' and i+2 < len(raw_str) and raw_str[i+2] == '|':
                curr.append('||')
                i += 3
                continue
            elif nxt in (';', '\\\\'):
                curr.append(nxt)
                i += 2
                continue
        elif raw_str[i:i+2] == '||':
            # Note: we unescape the rest of the string too
            rest = _split_macro_commands(raw_str[i+2:].strip() + ";")
            # _split_macro_commands returns a list of commands, we just want the unescaped string
            # Actually, simpler: just unescape the rest using string replacement for the basic ones
            rest_str = raw_str[i+2:].strip().replace(r'\||', '||').replace(r'\;', ';').replace(r'\\\\', '\\\\')
            return "".join(curr).strip(), '||', rest_str
        
        curr.append(raw_str[i])
        i += 1
    return "".join(curr).strip(), "", ""
'''

if target_imports in content:
    content = content.replace(target_imports, replacement_imports)
    print("Injected helper functions")
else:
    print("Failed to find import block")

# 1. Patch :macro creation
target_macro = '''                    mac_cmds = [c.strip() for c in parts[1].split(";") if c.strip()]'''
replacement_macro = '''                    mac_cmds = _split_macro_commands(parts[1])'''
if target_macro in content:
    content = content.replace(target_macro, replacement_macro)
    print("Patched :macro split")

# 2. Patch :consider
target_consider1 = '''                    if "||" not in payload:'''
replacement_consider1 = '''                    if "||" not in payload and r"\||" not in payload:'''
if target_consider1 in content:
    content = content.replace(target_consider1, replacement_consider1)

target_consider2 = '''                    left_side, _, b_text = payload.partition("||")'''
replacement_consider2 = '''                    left_side, _, b_text = _partition_unescaped_pipes(payload)'''
if target_consider2 in content:
    content = content.replace(target_consider2, replacement_consider2)
    print("Patched :consider partition")

# 3. Patch :probe mint
target_probe1 = '''                if "||" not in framings:'''
replacement_probe1 = '''                if "||" not in framings and r"\||" not in framings:'''
if target_probe1 in content:
    content = content.replace(target_probe1, replacement_probe1)

target_probe2 = '''                a_text, _, b_text = framings.partition("||")'''
replacement_probe2 = '''                a_text, _, b_text = _partition_unescaped_pipes(framings)'''
if target_probe2 in content:
    content = content.replace(target_probe2, replacement_probe2)
    print("Patched :probe mint partition")

# 4. Patch model probe query <<PROBE: name || words>>
target_model1 = '''                name_part, _, cand_text = model_probe_query.partition("||")'''
replacement_model1 = '''                name_part, _, cand_text = _partition_unescaped_pipes(model_probe_query)'''
if target_model1 in content:
    content = content.replace(target_model1, replacement_model1)
    print("Patched <<PROBE: >> partition")


with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
