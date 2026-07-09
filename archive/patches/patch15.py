import re
import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                        dest = mtail.split(maxsplit=2)[2].strip() if len(mtok) >= 3 else os.path.join(ROOT, "invariants", "out", "macros", "self.txt")
                        if _hidden_overwrite_blocked("self", "System"):
                            continue
                        macro_lines, restore_stats = build_session_restore_macro(
                            probes,
                            macro_aliases,
                            exposed_commands,
                            exposed_knobs,
                            hidden_commands,
                            self_dest=dest,
                        )
                        n_cmds = sum(1 for l in macro_lines if l.startswith(":"))
                        try:
                            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                            with open(dest, "w", encoding="utf-8") as wf:
                                for l in macro_lines:
                                    wf.write(l + "\\n")
                            macro_aliases["self"] = dest'''

replacement = '''                        alias_name = mtail.split(maxsplit=2)[2].strip() if len(mtok) >= 3 else "self"
                        # If they provided a name, map it to the standard macros folder
                        if not alias_name.endswith(".txt") and not "/" in alias_name and not "\\\\" in alias_name:
                            dest = os.path.join(ROOT, "invariants", "out", "macros", f"{alias_name}.txt")
                        else:
                            dest = alias_name # user provided an explicit path
                            alias_name = os.path.splitext(os.path.basename(dest))[0]
                            
                        if _hidden_overwrite_blocked(alias_name, "System"):
                            continue
                        macro_lines, restore_stats = build_session_restore_macro(
                            probes,
                            macro_aliases,
                            exposed_commands,
                            exposed_knobs,
                            hidden_commands,
                            self_dest=dest,
                        )
                        n_cmds = sum(1 for l in macro_lines if l.startswith(":"))
                        try:
                            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
                            with open(dest, "w", encoding="utf-8") as wf:
                                for l in macro_lines:
                                    wf.write(l + "\\n")
                            macro_aliases["self"] = dest
                            macro_aliases[alias_name] = dest'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched :macro name self dest logic")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
