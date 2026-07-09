import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                if harg:
                    matches = find_command_help_entries(harg)
                    if matches:
                        for entry in matches:
                            for _l in entry:
                                print(Fore.CYAN + _l + Style.RESET_ALL)
                    else:
                        q = harg.split()[0]
                        print(Fore.YELLOW + f"[Help] No help entry for '{q}'.{did_you_mean(q.lstrip(':'), BUILTIN_COMMANDS | set(macro_aliases))}" + Style.RESET_ALL)
                    continue'''

replacement = '''                if harg:
                    q = harg.split()[0].lstrip(":")
                    matches = find_command_help_entries(harg)
                    if matches:
                        for entry in matches:
                            for _l in entry:
                                print(Fore.CYAN + _l + Style.RESET_ALL)
                    elif q in macro_aliases:
                        # Generate dynamic help for macros
                        mpath = macro_aliases[q]
                        print(Fore.CYAN + f":{q} -- Macro aliased to {mpath}" + Style.RESET_ALL)
                        
                        # Check if it's a solve-macro with a known goal
                        for _n, _d, _a in list_solve_macros():
                            if _n == q:
                                _argnote = f" [args: {_a}]" if _a else ""
                                print(Fore.CYAN + f"  Goal: {_d}{_argnote}" + Style.RESET_ALL)
                                break
                                
                        # Show the first few lines of the macro
                        if os.path.isfile(mpath):
                            print(Fore.CYAN + "  Contents:" + Style.RESET_ALL)
                            try:
                                with open(mpath, "r", encoding="utf-8") as rf:
                                    lines = rf.read().splitlines()
                                    cmds = [line for line in lines if line.strip() and not line.strip().startswith("#")]
                                    for line in cmds[:5]:
                                        print(Fore.CYAN + f"    {line}" + Style.RESET_ALL)
                                    if len(cmds) > 5:
                                        print(Fore.CYAN + f"    ... and {len(cmds) - 5} more." + Style.RESET_ALL)
                            except Exception as e:
                                print(Fore.RED + f"    (Could not read macro file: {e})" + Style.RESET_ALL)
                    else:
                        print(Fore.YELLOW + f"[Help] No help entry for '{q}'.{did_you_mean(q, BUILTIN_COMMANDS | set(macro_aliases))}" + Style.RESET_ALL)
                    continue'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched :help macro support")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
