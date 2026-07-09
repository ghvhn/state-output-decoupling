import re
import os

files_to_patch = [
    (r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\docs\COMMANDS.md", "[args --]", "[args :]"),
    (r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py", "[args --]", "[args :]")
]

for file_path, target, replacement in files_to_patch:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if target in content:
            content = content.replace(target, replacement)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Patched {file_path}")
