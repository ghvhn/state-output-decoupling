import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

count = content.count("<|start_header_id|>system<|end_header_id|>")

if count > 0:
    content = content.replace("<|start_header_id|>system<|end_header_id|>", "<|start_header_id|>user<|end_header_id|>")
    print(f"Patched {count} instances of system header to user header.")
else:
    print("No system headers found.")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
