import re
import os

files_to_patch = [
    (r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\docs\COMMANDS.md", "`:solve <name> [args :] <goal>`", "`:solve <name> <goal> [: args]`"),
    (r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py", " :solve <name> [args :] <goal>", " :solve <name> <goal> [: args]")
]

for file_path, target, replacement in files_to_patch:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if target in content:
            content = content.replace(target, replacement)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Patched doc {file_path}")

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                if "--" in rest:
                    args_part, goal_part = rest.split("--", 1)
                    arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                elif ":" in rest:
                    args_part, goal_part = rest.split(":", 1)
                    arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")'''

replacement = '''                if ":" in rest:
                    goal_part, args_part = rest.rsplit(":", 1)
                    arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                elif "--" in rest:
                    goal_part, args_part = rest.rsplit("--", 1)
                    arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched script parsing logic")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
