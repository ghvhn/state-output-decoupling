import os

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                    target_probe = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower())[:40]
                    if target_probe in probes:
                        print(Fore.CYAN + f"[Suggest] Scanning for specific moves for probe '{target_probe}'..." + Style.RESET_ALL)
                        all_sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                        sugg = [(cat, line, cmd) for cat, line, cmd in all_sugg if target_probe in line or target_probe in cmd]
                        if not sugg:
                            print(Fore.YELLOW + f"[Suggest] No specific data-backed moves ready for '{target_probe}' yet." + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To generate evidence: :probe backfill {target_probe}" + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To steer blindly:     :tune probe_{target_probe}_alpha 0.5" + Style.RESET_ALL)
                            continue
                    else:
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
                        sug = generate_agentic_text(model, instruction=suggestion_prompt, config=config, pre_formatted=True, max_new_tokens=200, chatty_log=False, synthesis_recorder=sugg_rec)
                        
                        traces = [r for r in sugg_rec if r.get("type") == "routing_trace"]
                        if traces:
                            sums = {}
                            counts = {}
                            for tr in traces:
                                for k, v in tr.get("entropies", {}).items():
                                    sums[k] = sums.get(k, 0) + v
                                    counts[k] = counts.get(k, 0) + 1
                            if sums:
                                avg_str = " | ".join(f"{k[:3]}: {sums[k]/counts[k]:.2f}" for k in sorted(sums))
                                print(Fore.MAGENTA + f"[Agentic ToT Trace Summary] {avg_str}" + Style.RESET_ALL)

                        print(Fore.GREEN + f"Suggested framings for '{target_probe}':\\n" + sug.strip() + Style.RESET_ALL)
                        print(Fore.CYAN + f"Mint one with: :probe compose {target_probe} <positive> || <negative>" + Style.RESET_ALL)
                        continue
                else:
                    sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)'''

replacement = '''                    target_probe = re.sub(r"[^a-z0-9_]", "_", sargs[1].lower())[:40]
                    if target_probe in probes:
                        print(Fore.CYAN + f"[Suggest] Scanning for specific moves for probe '{target_probe}'..." + Style.RESET_ALL)
                        all_sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                        sugg = [(cat, line, cmd) for cat, line, cmd in all_sugg if target_probe in line or target_probe in cmd]
                        if not sugg:
                            print(Fore.YELLOW + f"[Suggest] No specific data-backed moves ready for '{target_probe}' yet." + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To generate evidence: :probe backfill {target_probe}" + Style.RESET_ALL)
                            print(Fore.CYAN + f"  -> To steer blindly:     :tune probe_{target_probe}_alpha 0.5" + Style.RESET_ALL)
                            continue
                    else:
                        print(Fore.CYAN + f"[Suggest] Probe '{target_probe}' not active. Mint it first with :probe {target_probe} <positive> || <negative>" + Style.RESET_ALL)
                        continue
                elif len(sargs) == 1 and command_because:
                    # Internally compute the similarity of the because string to all active probes
                    if not probes:
                        print(Fore.YELLOW + "[Suggest] No active probes to match against your reason. Mint some first!" + Style.RESET_ALL)
                        continue
                        
                    print(Fore.CYAN + f"[Suggest] Projecting your reason into the model's representation space to find the best matching probes..." + Style.RESET_ALL)
                    from invariants.engine import _inputs, _hidden_states
                    import torch
                    
                    ids = _inputs(model, command_because[:600])
                    hs = _hidden_states(model, ids["input_ids"], ids.get("attention_mask"))
                    
                    probe_scores = {}
                    for pname, pdata in probes.items():
                        probe_dir = pdata["direction"]
                        sim_sum = 0.0
                        layers_counted = 0
                        for L in list(probe_dir.keys()):
                            L_str = str(L)
                            if L_str in hs:
                                mean_hs = hs[L_str].mean(dim=0).to(model.device).reshape(-1)
                                if mean_hs.norm().item() > 0:
                                    mean_hs = mean_hs / mean_hs.norm()
                                    p_dir = probe_dir[L].to(model.device).reshape(-1)
                                    sim = torch.nn.functional.cosine_similarity(mean_hs.unsqueeze(0), p_dir.unsqueeze(0)).item()
                                    sim_sum += sim
                                    layers_counted += 1
                        if layers_counted > 0:
                            probe_scores[pname] = sim_sum / layers_counted
                            
                    if not probe_scores:
                        print(Fore.YELLOW + "[Suggest] Could not compute similarities." + Style.RESET_ALL)
                        continue
                        
                    sorted_probes = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)
                    top_probe, top_score = sorted_probes[0]
                    
                    print(Fore.GREEN + Style.BRIGHT + f"[Suggest] Best internal representation match: '{top_probe}' (similarity: {top_score:+.3f})" + Style.RESET_ALL)
                    if len(sorted_probes) > 1:
                        runners_up = ", ".join(f"{p} ({s:+.3f})" for p, s in sorted_probes[1:3])
                        print(Fore.CYAN + f"          Runners up: {runners_up}" + Style.RESET_ALL)
                        
                    all_sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)
                    sugg = [(cat, line, cmd) for cat, line, cmd in all_sugg if top_probe in line or top_probe in cmd]
                    
                    if not sugg:
                        print(Fore.YELLOW + f"[Suggest] No specific data-backed moves ready for '{top_probe}' yet." + Style.RESET_ALL)
                        print(Fore.CYAN + f"  -> To generate evidence: :probe backfill {top_probe}" + Style.RESET_ALL)
                        print(Fore.CYAN + f"  -> To steer blindly:     :tune probe_{top_probe}_alpha 0.5" + Style.RESET_ALL)
                        continue
                else:
                    sugg = suggest_actions(tuner, list(turn_log), probes=probes, archive_size=_arch)'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched suggest to use internal representations (cosine similarity)")
else:
    print("Failed to patch suggest")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
