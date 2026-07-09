import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''    if vram >= 16.0:
        return DEFAULT_MODEL'''

replacement = '''    if vram >= 14.5:
        return DEFAULT_MODEL'''

if target in content:
    content = content.replace(target, replacement)
    print("Replaced chunk")
else:
    print("Chunk not found")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
