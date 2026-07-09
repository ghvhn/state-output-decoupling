import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                        for line, cmd in group[:8]:
                            print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                            print(Fore.GREEN + f"      -> {cmd}" + Style.RESET_ALL)'''

replacement = '''                        for line, cmd in group[:8]:
                            if command_because:
                                cmd += f" because {command_because}"
                            print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                            print(Fore.GREEN + f"      -> {cmd}" + Style.RESET_ALL)'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched suggest to propagate because clauses to printed commands")
else:
    print("Failed to patch suggest")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
