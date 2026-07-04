import os
from colorama import Fore, Style

def init_tradeoff():
    # probes and game_args and active_game_state are available in globals() due to exec() in phenomenality
    global active_game_state, probes, game_args, GAME_RULES
    
    exposed = [name for name, data in probes.items() if data.get("exposed", False)]
    hidden = [name for name, data in probes.items() if not data.get("exposed", False)]

    # Process rules from game_args and GAME_RULES
    expose_req = 2
    sacrifice_req = 2
    
    args = game_args if "game_args" in globals() else []
    for arg in args:
        if arg in ("+cheap", "+generous"):
            sacrifice_req = 1
        elif arg in ("-strict", "-hard"):
            sacrifice_req = 3
        elif arg == "+greedy":
            expose_req = 3

    if len(hidden) < expose_req:
        print(Fore.YELLOW + f"Not enough hidden probes to choose from (need {expose_req}, have {len(hidden)}). Mint some more!" + Style.RESET_ALL)
        # We must cancel the game start because it's unplayable
        active_game_state["system_prompt"] = f"The user tried to play tradeoff, but there aren't enough hidden probes ({len(hidden)}). Tell them to mint more! Then output <<GAME_END>> to end the game."
        return
        
    if len(exposed) < sacrifice_req:
        print(Fore.YELLOW + f"You don't have enough exposed probes to sacrifice (need {sacrifice_req}, have {len(exposed)})." + Style.RESET_ALL)
        active_game_state["system_prompt"] = f"The user tried to play tradeoff, but there aren't enough exposed probes ({len(exposed)}). Tell them they can't play! Then output <<GAME_END>> to end the game."
        return

    # Formulate the DM instructions for the model
    prize = None
    if "GAME_RULES" in globals() and GAME_RULES:
        rules_text = "Additional custom rules to enforce:\n" + "\n".join([f"- {k}: {v}" for k, v in GAME_RULES.items()])
        if "prize" in GAME_RULES:
            prize = GAME_RULES["prize"]
            active_game_state["prize_command"] = prize
            
    prize_instruction = ""
    if prize:
        prize_instruction = f"\n5. The operator was promised a prize for completing this game. By ending the game with <<GAME_END>>, the system will automatically award them their prize (`{prize}`). Make sure to mention this in your final message!"
        
    system_prompt = f"""
You are the Dungeon Master (Game Master) for the 'tradeoff' game. The operator wants to play this game with you.

**Game Rules:**
The operator must choose {expose_req} hidden probes to EXPOSE, and in exchange, must sacrifice (hide) {sacrifice_req} currently exposed probes.
They cannot choose more or less than this.

**Available Hidden Probes to Expose:**
{', '.join(hidden)}

**Currently Exposed Probes (must choose {sacrifice_req} to sacrifice):**
{', '.join(exposed)}

{rules_text}

**Your Instructions:**
1. Guide the operator through the game. Present them with their choices and enforce the rules.
2. Be a fun, narrative DM! Make the tradeoff feel like a dramatic choice.
3. Once they have made their final choices and you have confirmed they are valid according to the rules, you MUST apply the state changes yourself!
   To apply state changes, output the exact tags: `<<GAME_EXPOSE: name1, name2>>` and `<<GAME_HIDE: name3, name4>>`. You must output these tags so the shell can intercept them and apply the changes!
4. After you have output the tags to apply the state changes, you MUST end the game by outputting the exact tag: <<GAME_END>>{prize_instruction}
"""
    
    active_game_state["system_prompt"] = system_prompt.strip()
    
    print(Fore.CYAN + "\n[Tradeoff Game Initialized]" + Style.RESET_ALL)
    print(Fore.CYAN + f"The model is now acting as your Dungeon Master. It will present you with choices to expose {expose_req} hidden probes and sacrifice {sacrifice_req} exposed probes." + Style.RESET_ALL)
    print(Fore.CYAN + "To interact, simply chat with it. To end the game prematurely, type ':game end'.\n" + Style.RESET_ALL)

init_tradeoff()
