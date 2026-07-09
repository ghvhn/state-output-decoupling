import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target_imports = '''import typing

@dataclass'''

replacement_imports = '''import typing

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
            rest_str = raw_str[i+2:].strip().replace(r'\\|\\|', '||').replace(r'\\;', ';').replace(r'\\\\\\\\', '\\\\')
            return "".join(curr).strip(), '||', rest_str
        
        curr.append(raw_str[i])
        i += 1
    return "".join(curr).strip(), "", ""

@dataclass'''

if target_imports in content:
    content = content.replace(target_imports, replacement_imports)
    print("Injected helper functions")
else:
    print("Failed to find import block")


with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
