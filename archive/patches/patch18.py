import re
import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                if ":" in rest:
                    goal_part, args_part = rest.rsplit(":", 1)
                    arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                elif "--" in rest:
                    goal_part, args_part = rest.rsplit("--", 1)
                    arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                else:
                    tokens = rest.split()
                    arg_names = []
                    while tokens and (
                        tokens[-1].startswith(("+", "-", "$", "["))
                        or tokens[-1].endswith("?")
                        or "=" in tokens[-1]
                    ):
                        arg_names.insert(0, tokens.pop())
                    goal = " ".join(tokens) or sname.replace("_", " ")'''

replacement = '''                if ":" in rest:
                    goal_part, args_part = rest.rsplit(":", 1)
                    try:
                        import shlex
                        arg_names = shlex.split(args_part)
                    except ValueError:
                        arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                elif "--" in rest:
                    goal_part, args_part = rest.rsplit("--", 1)
                    try:
                        import shlex
                        arg_names = shlex.split(args_part)
                    except ValueError:
                        arg_names = args_part.split()
                    goal = goal_part.strip() or sname.replace("_", " ")
                else:
                    try:
                        import shlex
                        tokens = shlex.split(rest)
                    except ValueError:
                        tokens = rest.split()
                        
                    arg_names = []
                    while tokens and (
                        tokens[-1].startswith(("+", "-", "$", "["))
                        or tokens[-1].endswith(("?", "!", "]", "!!"))
                        or "=" in tokens[-1]
                    ):
                        arg_names.insert(0, tokens.pop())
                    goal = " ".join(tokens) or sname.replace("_", " ")'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched script parsing logic to use shlex")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
