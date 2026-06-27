import ast
import math
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

def evaluate_python_expression(expr: str) -> str:
    import sys
    from io import StringIO
    import traceback
    
    expr = expr.strip()
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
    """Checks if text contains <<CALC: ...>> and returns the expression if so."""
    import re
    # Match <<CALC: expression>>
    match = re.search(r"<<CALC:\s*(.+?)>>", decoded_text)
    if match:
        return match.group(1)
    return None
