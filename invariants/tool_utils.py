import ast
import math
import re
import time
import torch
from transformers import StoppingCriteria

class ToolStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, start_length=0, stop_string=">>"):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.stop_string = stop_string
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.start_length:]
        if generated.numel() == 0:
            return False
        decoded = self.tokenizer.decode(generated[-15:], skip_special_tokens=True)
        return self.stop_string in decoded


class FinalAnswerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, start_length=0, min_tokens_after_marker=6):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.min_tokens_after_marker = min_tokens_after_marker

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.start_length:]
        if generated.numel() == 0:
            return False
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        marker_index = decoded.lower().rfind("final answer")
        if marker_index < 0:
            return False
        after_marker = decoded[marker_index:]
        if not re.search(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?", after_marker):
            return False
        token_count = len(self.tokenizer.encode(after_marker, add_special_tokens=False))
        return token_count >= self.min_tokens_after_marker


class VerifierStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, start_length=0, min_tokens_after_marker=5):
        self.tokenizer = tokenizer
        self.start_length = start_length
        self.min_tokens_after_marker = min_tokens_after_marker

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated = input_ids[0, self.start_length:]
        if generated.numel() == 0:
            return False
        decoded = self.tokenizer.decode(generated, skip_special_tokens=True)
        if not re.search(r"^VERDICT\s*:\s*(?:pass|unsettled|uncertain)", decoded, flags=re.IGNORECASE | re.MULTILINE):
            return False
        final_match = re.search(
            r"^INDEPENDENT_FINAL\s*:\s*(?:none|[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            decoded,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not final_match:
            return False
        after_marker = decoded[final_match.start():]
        token_count = len(self.tokenizer.encode(after_marker, add_special_tokens=False))
        return token_count >= self.min_tokens_after_marker


class TimeStoppingCriteria(StoppingCriteria):
    def __init__(self, deadline: float):
        self.deadline = deadline

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return time.time() >= self.deadline


def evaluate_python_expression(expr: str) -> str:
    import sys
    from io import StringIO
    import traceback
    
    expr = expr.strip().replace(",", "")
    safe_globals = {
        "__builtins__": {},
        "math": math,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
    }
    
    def _eval_expression(candidate: str):
        tree = ast.parse(candidate, mode='eval')
        code = compile(tree, filename="<string>", mode="eval")
        result = eval(code, safe_globals, {})
        return str(result)

    # Try to eval as expression first. If the model included a derivation like
    # "2+2 = 4" inside CALC, retry just the expression before the first equals.
    try:
        return _eval_expression(expr)
    except SyntaxError:
        if "=" in expr and "==" not in expr:
            prefix = expr.split("=", 1)[0].strip()
            if prefix:
                try:
                    return _eval_expression(prefix)
                except Exception:
                    pass
        # If it's a statement (like num1 = 123), exec it and capture stdout
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            exec(expr, safe_globals, {})
            sys.stdout = old_stdout
            out = redirected_output.getvalue().strip()
            return out if out else "[Code Executed Successfully - No Output. Use print() to see results]"
        except Exception as e:
            sys.stdout = old_stdout
            return f"[Error: {type(e).__name__} - {str(e)}]"
    except Exception as e:
        return f"[Error: {type(e).__name__} - {str(e)}]"

def intercept_tool_call(decoded_text: str):
    """Checks if text contains calculator brackets and returns the expression if so."""
    import re
    match = re.search(r"<<\s*CALC:\s*(.+?)>>", decoded_text)
    if match:
        return match.group(1)
    bare = re.search(r"<<\s*([^<>]+?)\s*>>", decoded_text)
    if bare:
        expr = bare.group(1).strip()
        if re.fullmatch(r"(?=.*\d)[0-9+\-*/().,\s=]+", expr):
            return expr
    return None

def popup_massive_question(question: str):
    """Pops up a massive Tkinter window to display a question to the user from afar."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.title("Agentic ToT: Disambiguation Required")
        root.attributes('-topmost', True)
        
        window_width = 1200
        window_height = 600
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)
        root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        
        root.configure(bg='#1e1e1e')
        
        lbl_title = tk.Label(root, text="SYSTEM HALTED: COGNITIVE TRAP DETECTED", font=("Helvetica", 24, "bold"), fg="#ff5555", bg="#1e1e1e")
        lbl_title.pack(pady=30)
        
        lbl_question = tk.Label(root, text=question, font=("Helvetica", 36), fg="white", bg="#1e1e1e", wraplength=1100, justify="center")
        lbl_question.pack(expand=True, fill='both', padx=50, pady=20)
        
        btn = tk.Button(root, text="ACKNOWLEDGE", font=("Helvetica", 20), command=root.destroy, bg="#333333", fg="white", padx=20, pady=10)
        btn.pack(pady=40)
        
        root.mainloop()
    except Exception as e:
        print(f"Failed to show popup: {e}")
