import re

file_path = r"c:\Users\Gavin Powell\Downloads\tda-domain-mapper\scripts\interactive_phenomenality.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

target = '''                        mpath = macro_aliases[q]
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
                    else:'''

replacement = '''                        mpath = macro_aliases[q]
                        exists = os.path.isfile(mpath)
                        status = "" if exists else " (FILE MISSING)"
                        print(Fore.CYAN + f":{q} -- Macro aliased to {mpath}{status}" + Style.RESET_ALL)
                        
                        # Check if it's a solve-macro with a known goal
                        for _n, _d, _a in list_solve_macros():
                            if _n == q:
                                _argnote = f" [args: {_a}]" if _a else ""
                                print(Fore.CYAN + f"  Goal: {_d}{_argnote}" + Style.RESET_ALL)
                                break
                                
                        # Show the first few lines of the macro
                        if exists:
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
                            print(Fore.RED + f"  (The file {mpath} does not exist or was deleted!)" + Style.RESET_ALL)
                    else:'''

if target in content:
    content = content.replace(target, replacement)
    print("Patched :help missing file logic")

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
