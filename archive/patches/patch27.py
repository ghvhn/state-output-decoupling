import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                    else:
                        print(Fore.CYAN + f"[Suggest] Probe '{target_probe}' not active. Generating multiple contrastive framings to mint it..." + Style.RESET_ALL)
                        suggestion_prompt = (
                            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
                            f"Write 3 different contrastive definition pairs for a behavioral dimension called '{target_probe}'. "
                            f"Write every side in the FIRST PERSON, as I describe MYSELF -- each side MUST start with 'I'. "
                            f"Each pair must be exactly: <first-person positive statement> || <first-person negative statement>.\\n\\n"
                            f"Example for 'understanding':\\n"
                            f"I fully grasp what the user means. || I am confused and miss the point.\\n"
                            f"I address the core issue clearly. || I give a surface-level, irrelevant answer.\\n"
                            f"I read between the lines accurately. || I hallucinate details that weren't there.\\n\\n"
                            f"Output ONLY the 3 contrastive pairs separated by newlines. Do not add any other text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
                        )
                        sugg_rec = []
                        sug = generate_agentic_text(model, instruction=suggestion_prompt, config=config, pre_formatted=True, max_new_tokens=200, chatty_log=False, synthesis_recorder=sugg_rec)'''

replacement = '''                    else:
                        print(Fore.CYAN + f"[Suggest] Probe '{target_probe}' not active. Generating multiple contrastive framings to mint it..." + Style.RESET_ALL)
                        reason_str = f" The user wants this dimension because: '{command_because}'. Tailor the definitions to this intent.\\n" if command_because else "\\n"
                        suggestion_prompt = (
                            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
                            f"Write 3 different contrastive definition pairs for a behavioral dimension called '{target_probe}'.{reason_str}"
                            f"Write every side in the FIRST PERSON, as I describe MYSELF -- each side MUST start with 'I'. "
                            f"Each pair must be exactly: <first-person positive statement> || <first-person negative statement>.\\n\\n"
                            f"Example for 'understanding':\\n"
                            f"I fully grasp what the user means. || I am confused and miss the point.\\n"
                            f"I address the core issue clearly. || I give a surface-level, irrelevant answer.\\n"
                            f"I read between the lines accurately. || I hallucinate details that weren't there.\\n\\n"
                            f"Output ONLY the 3 contrastive pairs separated by newlines. Do not add any other text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
                        )
                        sugg_rec = []
                        sug = generate_agentic_text(model, instruction=suggestion_prompt, config=config, pre_formatted=True, max_new_tokens=200, chatty_log=False, synthesis_recorder=sugg_rec)'''

target2 = '''                else:
                    sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)

                if not sugg:'''

replacement2 = '''                elif len(sargs) == 1 and command_because:
                    print(Fore.CYAN + f"[Suggest] Generating a new behavioral dimension based on your reason..." + Style.RESET_ALL)
                    suggestion_prompt = (
                        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n\\n"
                        f"The user wants to steer the model's behavior. Their reason is: '{command_because}'.\\n"
                        f"Invent a short, 1-word name for a behavioral dimension that captures this intent, and write 3 different contrastive definition pairs for it.\\n"
                        f"Write every side in the FIRST PERSON, as I describe MYSELF -- each side MUST start with 'I'.\\n"
                        f"The format must be exactly:\\n"
                        f"Name: <word>\\n"
                        f"<first-person positive statement> || <first-person negative statement>\\n"
                        f"<first-person positive statement> || <first-person negative statement>\\n"
                        f"<first-person positive statement> || <first-person negative statement>\\n\\n"
                        f"Output ONLY the name and the 3 pairs. Do not add any other text.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n"
                    )
                    sugg_rec = []
                    sug = generate_agentic_text(model, instruction=suggestion_prompt, config=config, pre_formatted=True, max_new_tokens=200, chatty_log=False, synthesis_recorder=sugg_rec)
                    print(Fore.GREEN + f"Suggested new dimension:\\n" + sug.strip() + Style.RESET_ALL)
                    print(Fore.CYAN + f"Mint it with: :probe <name> <positive> || <negative>" + Style.RESET_ALL)
                    continue
                else:
                    sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)

                if not sugg:'''

patched = False
if target in content and target2 in content:
    content = content.replace(target, replacement)
    content = content.replace(target2, replacement2)
    patched = True
    print("Patched suggest to generate probes from because clause")
else:
    print("Failed to patch suggest")

if patched:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
