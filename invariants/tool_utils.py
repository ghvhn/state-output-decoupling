import ast
import math
import re
import time
import torch
from dataclasses import dataclass
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
            r"^REASON\s*:\s*(.+)",
            decoded,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not final_match:
            return False
        
        # Stop 5 tokens after the reason starts generating to make sure it captures a full sentence/clause
        after_marker = decoded[final_match.start():]
        token_count = len(self.tokenizer.encode(after_marker, add_special_tokens=False))
        return token_count >= self.min_tokens_after_marker


class TimeStoppingCriteria(StoppingCriteria):
    def __init__(self, deadline: float):
        self.deadline = deadline

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return time.time() >= self.deadline


def _format_sympy_solution(value) -> str:
    try:
        import sympy as sp

        simplified = sp.simplify(value)
        if simplified.is_integer:
            return str(int(simplified))
        return str(float(simplified.evalf(15))).rstrip("0").rstrip(".")
    except Exception:
        return str(value)


def _normalize_tool_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.12g}"


def _normalize_unit_token(token: str) -> str:
    cleaned = token.strip().lower()
    cleaned = cleaned.replace("$", "dollars")
    cleaned = re.sub(r"[^a-z_]", "", cleaned)
    aliases = {
        "dollar": "dollars",
        "usd": "dollars",
        "egg": "eggs",
        "duckegg": "eggs",
        "glass": "glasses",
        "item": "items",
        "ducks": "ducks",
        "student": "students",
        "hour": "hours",
        "hr": "hours",
        "minute": "minutes",
        "min": "minutes",
        "day": "days",
        "week": "weeks",
        "month": "months",
        "year": "years",
        "percent": "",
        "percentage": "",
        "profit": "dollars",
        "earning": "dollars",
        "earnings": "dollars",
        "revenue": "dollars",
        "money": "dollars",
    }
    return aliases.get(cleaned, cleaned)


def _parse_unit(unit_text: str | None) -> dict[str, int]:
    unit = (unit_text or "").strip().lower()
    if unit in {"", "1", "unitless", "none", "number", "count"}:
        return {}
    unit = unit.replace("$", "dollars")
    unit = unit.replace(" per ", "/")
    unit = unit.replace(" each ", "/")
    unit = unit.replace(" a ", "/")
    unit = unit.replace(" every ", "/")
    unit = unit.replace("-", "_")
    dims: dict[str, int] = {}
    sign = 1
    for part in re.split(r"([*/])", unit):
        part = part.strip()
        if not part:
            continue
        if part == "*":
            sign = 1
            continue
        if part == "/":
            sign = -1
            continue
        token = _normalize_unit_token(part)
        if not token:
            continue
        dims[token] = dims.get(token, 0) + sign
        if dims[token] == 0:
            dims.pop(token)
    return dims


def _format_unit(dims: dict[str, int]) -> str:
    if not dims:
        return "unitless"
    numerator: list[str] = []
    denominator: list[str] = []
    for unit, power in sorted(dims.items()):
        target = numerator if power > 0 else denominator
        for _ in range(abs(power)):
            target.append(unit)
    if not denominator:
        return "*".join(numerator)
    return f"{'*'.join(numerator) or '1'}/{'*'.join(denominator)}"


@dataclass
class UnitValue:
    value: float
    unit: dict[str, int]


def _combine_units(left: dict[str, int], right: dict[str, int], sign: int = 1) -> dict[str, int]:
    combined = dict(left)
    for unit, power in right.items():
        combined[unit] = combined.get(unit, 0) + sign * power
        if combined[unit] == 0:
            combined.pop(unit)
    return combined


def _eval_unit_ast(node: ast.AST, env: dict[str, UnitValue]) -> UnitValue:
    if isinstance(node, ast.Expression):
        return _eval_unit_ast(node.body, env)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return UnitValue(float(node.value), {})
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(f"unknown quantity '{node.id}'")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval_unit_ast(node.operand, env)
        return UnitValue(-value.value if isinstance(node.op, ast.USub) else value.value, value.unit)
    if isinstance(node, ast.BinOp):
        left = _eval_unit_ast(node.left, env)
        right = _eval_unit_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            if left.unit != right.unit:
                raise ValueError(f"unit mismatch for addition: {_format_unit(left.unit)} + {_format_unit(right.unit)}")
            return UnitValue(left.value + right.value, left.unit)
        if isinstance(node.op, ast.Sub):
            if left.unit != right.unit:
                raise ValueError(f"unit mismatch for subtraction: {_format_unit(left.unit)} - {_format_unit(right.unit)}")
            return UnitValue(left.value - right.value, left.unit)
        if isinstance(node.op, ast.Mult):
            return UnitValue(left.value * right.value, _combine_units(left.unit, right.unit))
        if isinstance(node.op, ast.Div):
            return UnitValue(left.value / right.value, _combine_units(left.unit, right.unit, sign=-1))
        if isinstance(node.op, ast.Pow):
            if right.unit:
                raise ValueError("unitful exponent is invalid")
            exponent = right.value
            if abs(exponent - round(exponent)) > 1e-9:
                raise ValueError("fractional unit exponents are not supported")
            power = int(round(exponent))
            return UnitValue(left.value ** power, {unit: exp * power for unit, exp in left.unit.items()})
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [_eval_unit_ast(arg, env) for arg in node.args]
        if name in {"min", "max"} and args:
            first_unit = args[0].unit
            if any(arg.unit != first_unit for arg in args):
                raise ValueError(f"{name} arguments must share units")
            values = [arg.value for arg in args]
            return UnitValue((min if name == "min" else max)(values), first_unit)
        if name in {"floor", "ceil", "round"} and len(args) == 1:
            if args[0].unit:
                raise ValueError(f"{name} requires a unitless input")
            fn = {"floor": math.floor, "ceil": math.ceil, "round": round}[name]
            return UnitValue(float(fn(args[0].value)), {})
    raise ValueError(f"unsupported scaffold expression node: {type(node).__name__}")


def validate_quantity_scaffold(scaffold: str) -> str:
    """Validate a model-authored unit scaffold.

    Expected syntax:
      target=dollars/day; eggs=16 eggs/day; price=2 dollars/egg; expression=eggs * price
    """
    body = scaffold.strip()
    if body.lower().startswith("scaffold:"):
        body = body.split(":", 1)[1].strip()
    fields: dict[str, str] = {}
    for part in body.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()

    target_unit = _parse_unit(fields.get("target", ""))
    expr = fields.get("expression") or fields.get("expr")
    if not expr:
        return (
            "valid=False; error=missing expression. "
            "Use direct numeric assignments plus expression=..., e.g. "
            "full=8 glasses; discounted=8 glasses; price=5 dollars/glass; "
            "discount=0.6; expression=full * price + discounted * price * discount"
        )

    env: dict[str, UnitValue] = {}
    for key, value in fields.items():
        if key in {"target", "expression", "expr"}:
            continue
        value = value.replace("$", " dollars ")
        match = re.match(r"\s*(-?\d[\d,]*(?:\.\d+)?)\s*(.*)\s*$", value)
        if not match:
            return (
                f"valid=False; error=bad quantity '{key}'. "
                "Use direct assignments like jewelry=5000 dollars; jewelry_rate=2.5; "
                "not name/value fields"
            )
        number = float(match.group(1).replace(",", ""))
        env[key] = UnitValue(number, _parse_unit(match.group(2)))

    try:
        tree = ast.parse(expr, mode="eval")
        if any(
            not isinstance(
                node,
                (
                    ast.Expression,
                    ast.BinOp,
                    ast.UnaryOp,
                    ast.Constant,
                    ast.Add,
                    ast.Sub,
                    ast.Mult,
                    ast.Div,
                    ast.Pow,
                    ast.USub,
                    ast.UAdd,
                    ast.Call,
                    ast.Name,
                    ast.Load,
                ),
            )
            for node in ast.walk(tree)
        ):
            return "valid=False; error=unsupported expression syntax"
        result = _eval_unit_ast(tree, env)
    except Exception as e:
        return f"valid=False; error={str(e)}"

    unit_ok = result.unit == target_unit
    return (
        f"valid={str(unit_ok)}; "
        f"value={_normalize_tool_number(result.value)}; "
        f"unit={_format_unit(result.unit)}; "
        f"target={_format_unit(target_unit)}"
        + ("" if unit_ok else "; error=target unit mismatch")
    )


def validate_clause_map(clause_map: str) -> str:
    """Validate a model-authored map from numbered clauses into reasoning roles.

    Expected syntax:
      asked=C4; givens=C1,C3; rules=C2; operations=C2,C3; ignored=none
    """
    body = clause_map.strip()
    if body.lower().startswith("clausemap:"):
        body = body.split(":", 1)[1].strip()

    fields: dict[str, str] = {}
    for part in body.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key.strip().lower()] = value.strip()

    if not fields:
        return "valid=False; error=missing fields"

    role_aliases = {
        "given": "givens",
        "givens": "givens",
        "rule": "rules",
        "rules": "rules",
        "operation": "operations",
        "operations": "operations",
        "method": "operations",
        "methods": "operations",
        "asked": "asked",
        "question": "asked",
        "target": "asked",
        "ignore": "ignored",
        "ignored": "ignored",
        "distractor": "ignored",
        "distractors": "ignored",
    }
    unknown = sorted(key for key in fields if key not in role_aliases)
    if unknown:
        return f"valid=False; error=unknown role(s) {','.join(unknown)}"

    normalized: dict[str, list[str]] = {}
    for key, value in fields.items():
        role = role_aliases[key]
        if value.lower() in {"", "none", "n/a", "na"}:
            normalized.setdefault(role, [])
            continue
        ids = []
        for raw in re.split(r"[\s,]+", value):
            if not raw:
                continue
            match = re.fullmatch(r"C?(\d+)", raw.strip(), flags=re.IGNORECASE)
            if not match:
                return f"valid=False; error=bad clause id '{raw}'"
            ids.append(f"C{int(match.group(1))}")
        normalized.setdefault(role, []).extend(ids)

    if not normalized.get("asked"):
        return "valid=False; error=missing asked clause"
    if len(normalized.get("asked", [])) > 2:
        return "valid=False; error=too many asked clauses"
    if not (normalized.get("givens") or normalized.get("rules") or normalized.get("operations")):
        return "valid=False; error=missing supporting clauses"

    role_bits = []
    covered = set()
    repeated: set[str] = set()
    for role in ("asked", "givens", "rules", "operations", "ignored"):
        ids = normalized.get(role, [])
        for clause_id in ids:
            if clause_id in covered:
                repeated.add(clause_id)
            covered.add(clause_id)
        if ids:
            role_bits.append(f"{role}={','.join(ids)}")

    warning = "" if not repeated else f"; warning=repeated {','.join(sorted(repeated))}"
    return f"valid=True; covered={','.join(sorted(covered, key=lambda x: int(x[1:])))}; " + "; ".join(role_bits) + warning


def solve_one_variable_equation(expr: str) -> str | None:
    """Safely solve a single equation with exactly one symbolic variable."""
    if not expr or "=" not in expr:
        return None
    if any(op in expr for op in ("!=", "<=", ">=")):
        return None

    cleaned = expr.strip().replace(",", "").replace("$", "").replace("×", "*").replace("Ã—", "*")
    cleaned = re.sub(r"(?<=\d)\s+x\s+(?=\d)", "*", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("==", "=")
    if cleaned.count("=") != 1:
        return None

    allowed_functions = {"abs", "ceil", "floor", "round"}
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", cleaned))
    variables = sorted(name for name in names if name not in allowed_functions)
    if len(variables) != 1:
        return None

    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        symbol = sp.Symbol(variables[0], real=True)
        local_dict = {
            variables[0]: symbol,
            "abs": abs,
            "ceil": sp.ceiling,
            "floor": sp.floor,
            "round": round,
        }
        transformations = standard_transformations + (implicit_multiplication_application,)
        left, right = cleaned.split("=", 1)
        lhs = parse_expr(left, local_dict=local_dict, transformations=transformations, evaluate=True)
        rhs = parse_expr(right, local_dict=local_dict, transformations=transformations, evaluate=True)
        solutions = sp.solve(sp.Eq(lhs, rhs), symbol)
    except Exception:
        return None

    real_solutions = []
    for solution in solutions:
        try:
            if solution.is_real is False:
                continue
            if not solution.is_finite:
                continue
            real_solutions.append(solution)
        except Exception:
            continue
    if len(real_solutions) != 1:
        return None
    return _format_sympy_solution(real_solutions[0])


def evaluate_python_expression(expr: str) -> str:
    import sys
    from io import StringIO
    import traceback
    
    expr = expr.strip()
    if expr.lower().startswith("scaffold:"):
        return validate_quantity_scaffold(expr)
    if expr.lower().startswith("clausemap:"):
        return validate_clause_map(expr)
    expr = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", expr)
    safe_globals = {
        "__builtins__": {},
        "math": math,
        "abs": abs,
        "ceil": math.ceil,
        "floor": math.floor,
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

    solved = solve_one_variable_equation(expr)
    if solved is not None:
        return solved

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

def iter_tool_calls(decoded_text: str):
    """Yield complete tool calls as (start, end, expression)."""
    calls = []
    tagged_spans = []
    for match in re.finditer(
        r"<<\s*(CLAUSEMAP|SCAFFOLD|CALC)\s*:\s*(.+?)>>",
        decoded_text,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        name = match.group(1).upper()
        body = match.group(2)
        if name == "CLAUSEMAP":
            expr = "CLAUSEMAP: " + body
        elif name == "SCAFFOLD":
            expr = "SCAFFOLD: " + body
        else:
            expr = body
        calls.append((match.start(), match.end(), expr))
        tagged_spans.append((match.start(), match.end()))

    for bare in re.finditer(r"<<\s*([^<>]+?)\s*>>", decoded_text):
        if any(start <= bare.start() < end for start, end in tagged_spans):
            continue
        expr = bare.group(1).strip()
        if re.fullmatch(r"(?=.*\d)[0-9+\-*/().,\s=]+", expr):
            calls.append((bare.start(), bare.end(), expr))

    yield from sorted(calls, key=lambda call: call[0])


def intercept_tool_call(decoded_text: str):
    """Checks if text contains a complete tool call and returns the expression."""
    for _, _, expr in iter_tool_calls(decoded_text):
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
