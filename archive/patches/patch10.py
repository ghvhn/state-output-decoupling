import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''            if args:
                for i, arg in enumerate(args):
                    line = line.replace(f"${i+1}", arg)'''

replacement = '''            if args:
                for i, arg in enumerate(args):
                    line = line.replace(f"${i+1}", arg)
                line = line.replace("$@", " ".join(args))'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched $@ in expand_macro_lines")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
