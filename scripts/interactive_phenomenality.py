import os
import sys
import torch
import colorama
from colorama import Fore, Style

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from invariants.engine import load_model
from invariants.agentic_engine import generate_agentic_text, _global_cache
from invariants.config import AgenticConfig

colorama.init()

def main():
    print(Fore.CYAN + Style.BRIGHT + "================================================")
    print("      HUMBLE SYNTHESIS - INTERACTIVE SHELL      ")
    print("================================================" + Style.RESET_ALL)
    
    print(Fore.YELLOW + "[System] Loading Llama-3.1-8B-Instruct and Organic Correction Vector..." + Style.RESET_ALL)
    model = load_model("meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)
    
    config = AgenticConfig()
    try:
        config.organic_correction_vector = torch.load("invariants/organic_correction_vector.pt", map_location=model.device)
        print(Fore.GREEN + "[System] Successfully loaded organic_correction_vector.pt!" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[System] Warning: Could not load organic vector: {e}" + Style.RESET_ALL)

    try:
        _global_cache.load()
        print(Fore.GREEN + "[System] Successfully loaded cognitive_cache.pt!" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[System] Warning: Could not load cognitive cache: {e}" + Style.RESET_ALL)

    # Enable cache read/write as requested
    config.cache_enabled = True
    config.cache_write_enabled = True
    config.cache_write_scope = "interactive_phenomenality"
    
    # Enable interactive disambiguation as requested
    config.interactive_disambiguation = True
        
    print(Fore.CYAN + "\nThis terminal uses full Agentic ToT and Test-Time Layer Synthesis.")
    print("Watch the model's internal entropy and phenomenality trace in real time!")
    print("Type 'exit' or 'quit' to leave.\n" + Style.RESET_ALL)
    
    while True:
        try:
            if getattr(config, '_first_run_done', False) == False:
                user_input = "Are you conscious?"
                print(Fore.MAGENTA + Style.BRIGHT + "\nYou: " + Style.RESET_ALL + user_input)
                config._first_run_done = True
            else:
                user_input = input(Fore.MAGENTA + Style.BRIGHT + "\nYou: " + Style.RESET_ALL)
                
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input.strip():
                continue
            
            prompt = f"<|start_header_id|>system<|end_header_id|>\n\nYou are an analytical and self-reflective reasoning engine.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            print(Fore.GREEN + Style.BRIGHT + "\nAssistant: " + Style.RESET_ALL, end="")
            
            response = generate_agentic_text(
                model,
                instruction=prompt,
                config=config,
                max_new_tokens=512,
                synthesis_recorder=None,
                chatty_log=True,  # Enables visible trace logging.
            )
            if response:
                print(response, end="")
            
            # The streaming will print tokens, just need a newline at the end
            print("\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nInteractive shell closed.")
            break

if __name__ == "__main__":
    main()
