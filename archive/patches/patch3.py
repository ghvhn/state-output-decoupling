import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''    elif vram >= 8.0:
        return "meta-llama/Llama-3.2-3B-Instruct"'''

replacement = '''    elif vram >= 8.0:
        return "Qwen/Qwen2.5-3B-Instruct"'''

if target in content:
    content = content.replace(target, replacement)
    print("Replaced chunk")
else:
    print("Chunk not found")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
