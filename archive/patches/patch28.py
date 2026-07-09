import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                        reason_str = f" The user wants this dimension because: '{command_because}'. Tailor the definitions to this intent.\\n" if command_because else "\\n"'''

replacement = '''                        reason_str = f" I want this dimension because: '{command_because}'. Tailor the definitions to my intent.\\n" if command_because else "\\n"'''

target2 = '''                        f"The user wants to steer the model's behavior. Their reason is: '{command_because}'.\\n"
                        f"Invent a short, 1-word name for a behavioral dimension that captures this intent, and write 3 different contrastive definition pairs for it.\\n"'''

replacement2 = '''                        f"I want to steer the model's behavior. My reason is: '{command_because}'.\\n"
                        f"Invent a short, 1-word name for a behavioral dimension that captures my intent, and write 3 different contrastive definition pairs for it.\\n"'''

patched = False
if target in content and target2 in content:
    content = content.replace(target, replacement)
    content = content.replace(target2, replacement2)
    patched = True
    print("Fixed prompt phrasing to use first-person (intended for user header)")
else:
    print("Failed to patch prompt phrasing")

if patched:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
