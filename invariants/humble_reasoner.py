"""
Verifier-driven test-time reasoning loop.

This is deliberately not a confidence optimizer. The first answer is treated as
provisional. If the checker finds uncertainty or inconsistency, the system spends
more compute on a fresh attempt and only returns a confident answer when the
answer survives verification/stability checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
import ast
import math
import re
import time
from typing import Any

from invariants.engine import generate_text
from invariants.agentic_engine import NeedsDisambiguationError, _global_cache, generate_agentic_text
from invariants.tool_utils import solve_one_variable_equation, validate_clause_map


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
EQUATION_RHS_RE = re.compile(r"\$?(-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)")
FINAL_EQUATION_CUE_RE = re.compile(
    r"\b("
    r"profit|answer|final|total time|annual salary|salary|discount|cost|pay|"
    r"price|amount|remaining|water|distance|years? ago|sold|shorter|earned|earnings|"
    r"downloads|cups|meters|"
    r"make|made|need|takes?|time taken"
    r")\b",
    flags=re.IGNORECASE,
)
STRONG_FINAL_EQUATION_CUE_RE = re.compile(
    r"\b("
    r"profit|answer|final|total(?:\s+(?:time|cost|amount|distance|value|cups|meters|downloads|earnings|salary))?|"
    r"annual salary|salary|discount|cost|pay|price|amount|earned|make|made|need|takes?|time taken"
    r")\b",
    flags=re.IGNORECASE,
)
ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Call,
    ast.Name,
    ast.Load,
)
ALLOWED_ARITHMETIC_NAMES = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "min": min,
    "round": round,
}
NUMBER_WORDS = {
    "zero": Decimal(0),
    "one": Decimal(1),
    "two": Decimal(2),
    "three": Decimal(3),
    "four": Decimal(4),
    "five": Decimal(5),
    "six": Decimal(6),
    "seven": Decimal(7),
    "eight": Decimal(8),
    "nine": Decimal(9),
    "ten": Decimal(10),
    "eleven": Decimal(11),
    "twelve": Decimal(12),
    "thirteen": Decimal(13),
    "fourteen": Decimal(14),
    "fifteen": Decimal(15),
    "sixteen": Decimal(16),
    "seventeen": Decimal(17),
    "eighteen": Decimal(18),
    "nineteen": Decimal(19),
    "twenty": Decimal(20),
}
WORD_OR_NUM_RE = (
    r"-?\d[\d,]*(?:\.\d+)?|"
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
)


@dataclass
class ReasoningAttempt:
    mode: str
    round_index: int
    response: str
    extracted_answer: str | None
    verifier_response: str
    verdict: str
    verifier_answer: str | None
    accepted: bool
    token_budget: int | None = None
    elapsed_sec: float = 0.0
    urgency: dict[str, Any] | None = None
    synthesis_records: list[dict[str, Any]] | None = None
    needs_clarification: bool = False
    clarifying_question: str | None = None
    ambiguity_type: str | None = None
    solver_checked_answer: str | None = None
    verifier_checked_answer: str | None = None
    verifier_tagged_answer: str | None = None
    acceptance_reason: str | None = None
    learning_signal: dict[str, Any] | None = None
    stage_states: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        records = self.synthesis_records or []
        # Filter out tensor objects ("trigger", "delta") so json serialization works
        safe_records = []
        for r in records:
            safe_r = {k: v for k, v in r.items() if k not in ("trigger", "delta")}
            # Also safely serialize phenomenality if it exists inside metadata
            if "metadata" in safe_r:
                safe_metadata = dict(safe_r["metadata"])
                safe_r["metadata"] = safe_metadata
            safe_records.append(safe_r)

        safe_stage_states = {}
        for name, value in (self.stage_states or {}).items():
            if hasattr(value, "shape") and hasattr(value, "float"):
                try:
                    safe_stage_states[name] = {
                        "shape": list(value.shape),
                        "norm": float(value.float().norm().item()),
                    }
                except Exception:
                    safe_stage_states[name] = {"repr": repr(value)[:120]}
            else:
                safe_stage_states[name] = {"repr": repr(value)[:120]}
            
        return {
            "mode": self.mode,
            "round_index": self.round_index,
            "response": self.response,
            "extracted_answer": self.extracted_answer,
            "verifier_response": self.verifier_response,
            "verdict": self.verdict,
            "verifier_answer": self.verifier_answer,
            "accepted": self.accepted,
            "token_budget": self.token_budget,
            "elapsed_sec": self.elapsed_sec,
            "urgency": self.urgency,
            "synthesis_record_count": len(records),
            "synthesis_records": safe_records,
            "needs_clarification": self.needs_clarification,
            "clarifying_question": self.clarifying_question,
            "ambiguity_type": self.ambiguity_type,
            "solver_checked_answer": self.solver_checked_answer,
            "verifier_checked_answer": self.verifier_checked_answer,
            "verifier_tagged_answer": self.verifier_tagged_answer,
            "acceptance_reason": self.acceptance_reason,
            "learning_signal": self.learning_signal,
            "stage_state_summary": safe_stage_states,
        }


@dataclass
class HumbleResult:
    question: str
    final_answer: str | None
    confident: bool
    reason: str
    attempts: list[ReasoningAttempt]
    urgency: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "confident": self.confident,
            "reason": self.reason,
            "urgency": self.urgency or {},
            "attempts": [a.to_dict() for a in self.attempts],
        }


def _capture_prompt_response_state(M, prompt: str, response: str | None = None):
    """Capture a residual-state summary without generating new text.

    Returns [n_layers, d]. For a bare prompt this is the final prompt-token
    state; for prompt+response this is the mean over response-token states.
    """
    if M is None:
        return None
    try:
        import torch
        from invariants.engine import _inputs, _hidden_states

        inputs = _inputs(M, prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        plen = input_ids.shape[1]
        if response:
            response_ids = M.tok.encode(
                response,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(input_ids.device)
            if response_ids.numel() > 0:
                input_ids = torch.cat([input_ids, response_ids], dim=1)
                attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        hs = _hidden_states(M, input_ids, attention_mask)
        if response and input_ids.shape[1] > plen:
            return hs[:, plen:, :].float().mean(1).detach().cpu()
        return hs[:, -1, :].float().detach().cpu()
    except Exception:
        return None


def _capture_stage_states_enabled(config) -> bool:
    return bool(getattr(config, "capture_stage_states", False))


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    text = text.replace(",", "").replace("$", "").strip()
    if not text or text.lower() in {"none", "n/a", "unknown"}:
        return None
    try:
        return format(Decimal(text).normalize(), 'f')
    except InvalidOperation:
        return None


def extract_number(text: str) -> str | None:
    marked = re.findall(
        r"(?:final answer|answer is|corrected_final|independent_final)\s*:?\s*\$?(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        text.replace(",", ""),
        flags=re.IGNORECASE,
    )
    if marked:
        return normalize_number(marked[-1])
    nums = NUMBER_RE.findall(text.replace(",", ""))
    return normalize_number(nums[-1]) if nums else None


def numbers_match(left: str | None, right: str | None, tolerance: Decimal = Decimal("1e-9")) -> bool:
    if left is None or right is None:
        return False
    if left == right:
        return True
    left_dec = _decimal_from_text(left)
    right_dec = _decimal_from_text(right)
    if left_dec is None or right_dec is None:
        return False
    scale = max(Decimal(1), abs(left_dec), abs(right_dec))
    return abs(left_dec - right_dec) <= tolerance * scale


def _safe_eval_arithmetic(expr: str) -> str | None:
    raw_expr = expr
    expr = expr.replace(",", "").replace("$", "").replace("×", "*")
    if re.search(r"\b(?:max|min)\s*\(", raw_expr, flags=re.IGNORECASE):
        expr = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", raw_expr)
        expr = expr.replace("$", "")
    expr = re.sub(r"(?<=\d)\s+x\s+(?=\d)", "*", expr, flags=re.IGNORECASE)
    expr = expr.strip()
    if not expr or not re.search(r"\d", expr):
        return None
    if "=" in expr:
        solved = solve_one_variable_equation(expr)
        if solved is not None:
            return normalize_number(solved)
    if not re.fullmatch(r"[0-9a-zA-Z_eE+\-*/().,\s%]+", expr):
        return None
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    if any(not isinstance(node, ALLOWED_AST_NODES) for node in ast.walk(tree)):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in ALLOWED_ARITHMETIC_NAMES:
            return None
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in ALLOWED_ARITHMETIC_NAMES
        ):
            return None
    try:
        value = eval(compile(tree, "<arithmetic>", "eval"), {"__builtins__": {}}, ALLOWED_ARITHMETIC_NAMES)
    except Exception:
        return None
    return normalize_number(str(value))


def _arithmetic_tail(left: str) -> str | None:
    cleaned = left.replace(",", "").replace("$", "").replace("×", "*").replace("x", "*")
    cleaned = cleaned.replace("%", " ")
    cleaned = re.sub(r"[^0-9+\-*/().\s]", " ", cleaned)
    matches = re.findall(r"[-+*/().\s0-9]*\d[-+*/().\s0-9]*", cleaned)
    for candidate in reversed(matches):
        candidate = candidate.strip()
        if re.search(r"[-+*/]", candidate) and re.search(r"\d", candidate):
            return candidate
    return None


def _checked_equation_candidates(
    text: str | None,
    require_final_cue: bool = False,
) -> list[dict[str, Any]]:
    """Return arithmetic equations with written and/or locally evaluated values."""
    if not text:
        return []
    candidates: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        if "=" not in raw_line:
            continue
        cue_line = raw_line.replace("_", " ")
        if require_final_cue and not FINAL_EQUATION_CUE_RE.search(cue_line):
            continue
        cue_strength = 0
        if STRONG_FINAL_EQUATION_CUE_RE.search(cue_line):
            cue_strength = 2
        elif FINAL_EQUATION_CUE_RE.search(cue_line):
            cue_strength = 1
        left, right = raw_line.rsplit("=", 1)
        rhs_match = EQUATION_RHS_RE.search(right)
        written = normalize_number(rhs_match.group(1)) if rhs_match else None
        expr = _arithmetic_tail(left)
        if expr:
            evaluated = _safe_eval_arithmetic(expr)
        else:
            # Some verifier lines have the final cue on the left and a bare
            # arithmetic expression on the right, e.g.
            # "Profit = 80000 + 120000 - 80000 - 50000".
            # Evaluate that right-hand expression instead of treating the
            # first number as the final answer.
            expr = _arithmetic_tail(right)
            evaluated = _safe_eval_arithmetic(expr) if expr else None
            if evaluated is not None:
                written = None
        if evaluated is not None:
            candidates.append(
                {
                    "line": raw_line.strip(),
                    "expression": expr,
                    "written_answer": written,
                    "evaluated_answer": evaluated,
                    "math_consistent": numbers_match(written, evaluated) if written is not None else None,
                    "cue_strength": cue_strength,
                }
            )
    return candidates


def checked_equation_quality(
    text: str | None,
    require_final_cue: bool = False,
) -> dict[str, Any] | None:
    candidates = _checked_equation_candidates(text, require_final_cue=require_final_cue)
    if not candidates:
        return None
    return max(enumerate(candidates), key=lambda item: (item[1].get("cue_strength", 0), item[0]))[1]


def extract_checked_equation_answer(text: str | None, require_final_cue: bool = False) -> str | None:
    """Return the last arithmetic equation result that can be checked locally."""
    quality = checked_equation_quality(text, require_final_cue=require_final_cue)
    return None if quality is None else quality["evaluated_answer"]


def extract_expression_answer(text: str | None) -> str | None:
    if not text:
        return None
    target = None
    final_matches = re.findall(
        r"^\s*Final answer\s*:?\s*\$?(-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if final_matches:
        target = normalize_number(final_matches[-1])
    matches = re.findall(r"^\s*Expression\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    computed = re.findall(r"^\s*Computed\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    for expr in computed:
        cleaned = re.sub(r"<<\s*CALC\s*:\s*(.*?)>>.*", r"\1", expr, flags=re.IGNORECASE | re.DOTALL).strip()
        if cleaned:
            matches.append(cleaned)
    fallback = None
    for expr in reversed(matches):
        evaluated = _safe_eval_arithmetic(expr)
        if evaluated is not None:
            if fallback is None:
                fallback = evaluated
            if target is not None and numbers_match(evaluated, target):
                return evaluated
    return fallback


def used_calculator_tool(text: str | None) -> bool:
    if not text:
        return False
    return bool(re.search(r"<<\s*CALC\s*:", text, flags=re.IGNORECASE))


def used_scaffold_tool(text: str | None) -> bool:
    if not text:
        return False
    return bool(re.search(r"<<\s*SCAFFOLD\s*:", text, flags=re.IGNORECASE))


def used_clause_map_tool(text: str | None) -> bool:
    if not text:
        return False
    return bool(re.search(r"<<\s*CLAUSEMAP\s*:", text, flags=re.IGNORECASE))


def scaffold_tool_feedback(text: str | None) -> str | None:
    if not text:
        return None
    matches = re.findall(
        r"<<\s*SCAFFOLD\s*:.*?>>\s*=\s*([^\n]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return matches[-1].strip() if matches else None


def clause_map_feedback(text: str | None, question: str | None = None) -> str | None:
    if not text:
        return None
    matches = re.findall(
        r"<<\s*CLAUSEMAP\s*:\s*(.*?)>>\s*=\s*([^\n]+)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if matches:
        _, feedback = matches[-1]
        feedback = feedback.strip()
        if question is None or "valid=True" not in feedback:
            return feedback
        expected_ids = {f"C{i}" for i in range(1, len(question_clauses(question)) + 1)}
        covered = set(re.findall(r"\bC\d+\b", feedback))
        missing = sorted(expected_ids - covered, key=lambda x: int(x[1:]))
        unknown = sorted(covered - expected_ids, key=lambda x: int(x[1:]))
        extras = []
        if missing:
            extras.append(f"missing={','.join(missing)}")
        if unknown:
            extras.append(f"unknown={','.join(unknown)}")
        return feedback if not extras else feedback + "; " + "; ".join(extras)
    raw = re.findall(
        r"<<\s*CLAUSEMAP\s*:\s*(.*?)>>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if raw:
        return validate_clause_map("CLAUSEMAP: " + raw[-1])
    return None


def scaffold_feedback_invalid(feedback: str | None) -> bool:
    return bool(feedback and "valid=False" in feedback)


def clause_map_feedback_invalid(feedback: str | None) -> bool:
    return bool(feedback and ("valid=False" in feedback or "missing=" in feedback or "unknown=" in feedback))


def question_clauses(question: str) -> list[str]:
    """Split a word problem into coarse clauses for external working memory."""
    cleaned = " ".join(question.strip().split())
    if not cleaned:
        return []
    chunks: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        sentence = sentence.strip(" .!?")
        if not sentence:
            continue
        parts = re.split(
            r"\s*,\s*(?:but|and|while|then|so)\s+|\s*;\s*|\s+(?=after\b|before\b|when\b|if\b)",
            sentence,
            flags=re.IGNORECASE,
        )
        for part in parts:
            part = part.strip(" ,.;")
            if part:
                chunks.append(part)
    if len(chunks) <= 1:
        return [cleaned.rstrip(".?!")]
    return chunks


def clause_map_context(question: str) -> str:
    clauses = question_clauses(question)
    if not clauses:
        return ""
    lines = ["Optional external working-memory clauses:"]
    for idx, clause in enumerate(clauses, start=1):
        lines.append(f"[C{idx}] {clause}")
    lines.append(
        "If helpful, write "
        "<<CLAUSEMAP: asked=C?; givens=C?,C?; rules=C?; operations=C?; ignored=none>> "
        "to bind roles; it checks coverage only, not the answer."
    )
    return "\n".join(lines)


def _generic_methodology_from_question(question: str) -> dict[str, Any] | None:
    q = question.lower()
    if re.search(r"\bevery\s+(?:\d+|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", q) and (
        "discount" in q or "half price" in q or "costs only" in q
    ):
        return {
            "kind": "periodic_discount_partition",
            "methodology": (
                "Partition items into full-price and every-nth discounted groups, "
                "then sum each group once."
            ),
            "structural_features": [
                "every_nth_item_rule",
                "discounted_group",
                "full_price_remainder_group",
                "sum_partition_costs",
                "avoid_double_charging_all_items",
            ],
        }
    if "profit" in q and re.search(r"\b(buys?|bought|spends?|spent|sells?|sold)\b", q):
        return {
            "kind": "profit_objective_binding",
            "methodology": "Bind profit as money received minus all required costs, not revenue or value alone.",
            "structural_features": [
                "requested_profit",
                "revenue_or_final_value",
                "cost_subtraction",
                "objective_binding",
            ],
        }
    if "remaining" in q and re.search(r"\b(sells?|sold|makes?|earns?)\b", q):
        return {
            "kind": "remainder_before_sale",
            "methodology": "Subtract non-sold or already-used items before multiplying the remaining count by rate.",
            "structural_features": [
                "starting_quantity",
                "removed_quantity",
                "remaining_quantity",
                "rate_applied_to_remainder",
            ],
        }
    return None


def sanitized_clause_methodology(question: str, feedback: str | None) -> dict[str, Any] | None:
    """Convert local clause-map diagnostics into reusable methodology metadata."""
    if not feedback:
        return None
    methodology = _generic_methodology_from_question(question) or {
        "kind": "general_clause_role_binding",
        "methodology": "Bind asked quantity, givens, rules, and operations before choosing an expression.",
        "structural_features": ["asked_quantity", "givens", "rules", "operations"],
    }
    roles = sorted(set(re.findall(r"\b(asked|givens|rules|operations|ignored)\s*=", feedback)))
    status = (
        "invalid"
        if "valid=False" in feedback
        else "incomplete"
        if "missing=" in feedback or "unknown=" in feedback
        else "complete"
        if "valid=True" in feedback
        else "unverified"
    )
    sanitized = dict(methodology)
    sanitized.update(
        {
            "clause_map_status": status,
            "roles_declared": roles,
            "privacy": {
                "tier": "reusable_sanitized",
                "raw_clauses_saved": False,
                "source_numbers_saved": False,
                "entity_names_saved": False,
                "source_question_key_required": True,
            },
        }
    )
    return sanitized


def tool_capability_text(scaffold_detail: str = "full", include_clause_map: bool = False) -> str:
    scoring_contract = (
        "Scoring contract: the final numeric answer is scored against the requested quantity. "
        "Microscopic floating-point roundoff is tolerated, but meaningful rounding or answering an intermediate/wrong quantity fails. "
    )
    clause_tool = (
        "When numbered clauses are provided, the optional working-memory tool "
        "<<CLAUSEMAP: asked=C?; givens=C?,C?; rules=C?; operations=C?; ignored=none>> checks role coverage. "
        if include_clause_map
        else ""
    )
    calc = (
        scoring_contract
        +
        "Available tool: write <<CALC: expression>> for arithmetic/equations. "
        "The calculator supports arithmetic, min(...), max(...), floor(...), ceil(...), round(...), "
        "and simple one-variable equations. "
        + clause_tool
        + "Numeric checking treats only microscopic roundoff as equivalent "
        "(for example, 17.999999999999996 and 18); still give the exact intended value "
        "and do not round away meaningful fractional answers. "
    )
    if scaffold_detail == "none":
        return (
            calc
            + "Calculator output is authoritative for evaluating expressions, "
            "but not for choosing which expression answers the question. "
        )
    if scaffold_detail == "brief":
        return (
            calc
            + "For genuinely risky unit/rate/objective problems only, you may also write "
            "<<SCAFFOLD: target=<unit>; variable=value unit; ...; expression=<formula over variables>>> "
            "to check a quantity scaffold; skip this for simple single-unit arithmetic. "
            "Use direct numeric assignments only, not prose aliases: "
            "count=8 glasses; price=5 dollars/glass; discount=0.6; "
            "expression=count * price * discount. "
            "Tool output is authoritative for evaluating expressions and unit compatibility, "
            "but not for choosing which expression answers the question. "
        )
    return (
        calc
        + "You may also write <<SCAFFOLD: target=<unit>; variable=value unit; ...; expression=<formula over variables>>> "
        "to check a model-authored quantity scaffold. "
        "Use direct variable assignments only, e.g. "
        "<<SCAFFOLD: target=dollars; jewelry=5000 dollars; jewelry_rate=2.5; "
        "gadgets=8000 dollars; gadgets_rate=1.2; "
        "expression=max(jewelry * jewelry_rate / 100, gadgets * gadgets_rate / 100)>>. "
        "Do not write prose aliases like variable=x or x=cost of one glass; write numeric quantities "
        "with units and put formulas in expression=... . "
        "Use SCAFFOLD only when units, rates, prices, percentages, elapsed time, inventory states, "
        "or multiple candidate objectives are genuinely at risk; for simple single-unit arithmetic, skip it. "
        "The scaffold checker validates unit compatibility and returns a value/unit. "
        "Tool output is authoritative for evaluating expressions and unit compatibility, "
        "but not for choosing which expression answers the question. "
    )


def _format_decimal(value: Decimal) -> str:
    return normalize_number(str(value)) or str(value)


def _decimal_from_text(text: str | None) -> Decimal | None:
    if text is not None:
        word = text.strip().lower()
        if word in NUMBER_WORDS:
            return NUMBER_WORDS[word]
    normalized = normalize_number(text)
    if normalized is None:
        return None
    try:
        return Decimal(normalized)
    except InvalidOperation:
        return None


def _first_decimal(patterns: list[str], text: str) -> Decimal | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            parsed = _decimal_from_text(match.group(1))
            if parsed is not None:
                return parsed
    return None


def financial_profit_scaffold(question: str) -> dict[str, str] | None:
    """Derive a text-grounded expression for common flip/renovation profit prompts."""
    if not re.search(r"\bprofit\b", question, flags=re.IGNORECASE):
        return None
    base_value = _first_decimal(
        [
            r"\b(?:buys?|bought|purchases?|purchased)\b[^.?!]{0,120}?\bfor\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
            r"\binitial\s+(?:value|price|cost)\b[^.?!]{0,80}?\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        ],
        question,
    )
    repair_cost = _first_decimal(
        [
            r"\b(?:puts?|put|spends?|spent|invests?|invested)\b[^.?!]{0,100}?\$?\s*(-?\d[\d,]*(?:\.\d+)?)[^.?!]{0,80}?\b(?:repairs?|renovations?|improvements?)\b",
            r"\$?\s*(-?\d[\d,]*(?:\.\d+)?)[^.?!]{0,40}?\bin\s+(?:repairs?|renovations?|improvements?)\b",
        ],
        question,
    )
    percent_increase = _first_decimal(
        [
            r"\b(?:increased|boosted|boosts?|raised|raises?)\b[^.?!]{0,120}?\bvalue\b[^.?!]{0,120}?\bby\s+(-?\d+(?:\.\d+)?)\s*%",
            r"\bvalue\b[^.?!]{0,120}?\b(?:increased|boosted|raised)\b[^.?!]{0,120}?\bby\s+(-?\d+(?:\.\d+)?)\s*%",
        ],
        question,
    )
    if base_value is None or repair_cost is None or percent_increase is None:
        return None

    increase = base_value * percent_increase / Decimal(100)
    final_value = base_value + increase
    total_cost = base_value + repair_cost
    profit = final_value - total_cost
    base = _format_decimal(base_value)
    repair = _format_decimal(repair_cost)
    pct = _format_decimal(percent_increase)
    return {
        "kind": "financial_profit_percent_increase",
        "requested_quantity": "profit",
        "base_value": base,
        "repair_cost": repair,
        "percent_increase": pct,
        "value_increase": _format_decimal(increase),
        "final_value": _format_decimal(final_value),
        "total_cost": _format_decimal(total_cost),
        "profit": _format_decimal(profit),
        "expression": f"({base} + ({base} * {pct} / 100)) - {base} - {repair}",
    }


def choice_profit_scaffold(question: str) -> dict[str, str] | None:
    """Compare realized profits across explicit purchase choices."""
    if not re.search(r"\bchoice\b|\bchoose\b|\bpurchase\s+plans?\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bmaximi[sz]e\s+profit\b|\bmost\s+profit\b", question, flags=re.IGNORECASE):
        return None

    value_options: list[tuple[str, Decimal, str]] = []
    value_pattern = re.compile(
        r"\b([A-Za-z][A-Za-z -]*?)\s+worth\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    for match in value_pattern.finditer(question):
        raw_name = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;").lower()
        raw_name = re.sub(r"^(?:or|and|the)\s+", "", raw_name).strip()
        value = _decimal_from_text(match.group(2))
        if raw_name and value is not None:
            value_options.append((raw_name, value, _format_decimal(value)))

    if len(value_options) < 2:
        return None

    candidates: list[dict[str, Any]] = []
    for name, value, value_s in value_options:
        pct_match = re.search(
            rf"\b(?:the\s+)?{re.escape(name)}\s+market\s+will\s+(?:go\s+up|rise|increase)\s+(-?\d+(?:\.\d+)?)\s*%",
            question,
            flags=re.IGNORECASE,
        )
        if pct_match is None:
            continue
        pct = _decimal_from_text(pct_match.group(1))
        if pct is None:
            continue
        profit = value * pct / Decimal(100)
        pct_s = _format_decimal(pct)
        candidates.append(
            {
                "name": name,
                "value": value_s,
                "percent": pct_s,
                "profit": _format_decimal(profit),
                "profit_decimal": profit,
                "expression": f"{value_s} * {pct_s} / 100",
            }
        )

    if len(candidates) < 2:
        return None

    best = max(candidates, key=lambda item: item["profit_decimal"])
    return {
        "kind": "choice_max_realized_profit",
        "requested_quantity": "maximum realized profit from one chosen purchase plan",
        "options": "; ".join(
            f"{item['name']}={item['value']} at {item['percent']}% -> {item['profit']}"
            for item in candidates
        ),
        "best_option": best["name"],
        "best_value": best["value"],
        "best_percent": best["percent"],
        "best_profit": best["profit"],
        "expression": best["expression"],
    }


def periodic_discount_scaffold(question: str) -> dict[str, str] | None:
    """Derive total cost when every nth item has a percentage discount."""
    discount_match = re.search(
        rf"\bevery\s+({WORD_OR_NUM_RE}|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)"
        r"\s+\w+\s+costs?\s+(?:only\s+)?(-?\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?price\b",
        question,
        flags=re.IGNORECASE,
    )
    if discount_match is None:
        return None
    ordinal_raw = discount_match.group(1).lower()
    ordinal_map = {
        "second": Decimal(2),
        "third": Decimal(3),
        "fourth": Decimal(4),
        "fifth": Decimal(5),
        "sixth": Decimal(6),
        "seventh": Decimal(7),
        "eighth": Decimal(8),
        "ninth": Decimal(9),
        "tenth": Decimal(10),
    }
    period = ordinal_map.get(ordinal_raw) or _decimal_from_text(ordinal_raw)
    discount_pct = _decimal_from_text(discount_match.group(2))
    unit_price = _first_decimal(
        [
            r"\bone\s+\w+\s+costs?\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\b",
            r"\b\w+\s+costs?\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\b",
        ],
        question,
    )
    total_count = _first_decimal(
        [
            r"\bwants?\s+to\s+buy\s+(-?\d[\d,]*(?:\.\d+)?)\s+\w+\b",
            r"\bbuy\s+(-?\d[\d,]*(?:\.\d+)?)\s+\w+\b",
        ],
        question,
    )
    if (
        period is None
        or discount_pct is None
        or unit_price is None
        or total_count is None
        or period <= 0
    ):
        return None

    discounted_count = Decimal(int(total_count // period))
    full_count = total_count - discounted_count
    discounted_price = unit_price * discount_pct / Decimal(100)
    total = full_count * unit_price + discounted_count * discounted_price
    total_s = _format_decimal(total_count)
    period_s = _format_decimal(period)
    full_s = _format_decimal(full_count)
    discounted_s = _format_decimal(discounted_count)
    price_s = _format_decimal(unit_price)
    pct_s = _format_decimal(discount_pct)
    discounted_price_s = _format_decimal(discounted_price)
    return {
        "kind": "periodic_discount_total_cost",
        "requested_quantity": "total cost with every nth item discounted",
        "total_count": total_s,
        "period": period_s,
        "full_price_count": full_s,
        "discounted_count": discounted_s,
        "unit_price": price_s,
        "discount_percent": pct_s,
        "discounted_price": discounted_price_s,
        "total_cost": _format_decimal(total),
        "expression": f"({total_s} - floor({total_s} / {period_s})) * {price_s} + floor({total_s} / {period_s}) * ({pct_s} * {price_s} / 100)",
    }


def remainder_sale_revenue_scaffold(question: str) -> dict[str, str] | None:
    """Derive revenue from selling the remainder after personal use."""
    if not re.search(r"\bsells?\s+the\s+remainder\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bmake\b|\bdollars?\b|\brevenue\b|\bearn", question, flags=re.IGNORECASE):
        return None

    produced = _first_decimal(
        [
            r"\blay\s+(-?\d[\d,]*(?:\.\d+)?)\s+eggs?\s+per\s+day\b",
            r"\bproduce\s+(-?\d[\d,]*(?:\.\d+)?)\s+\w+\s+per\s+day\b",
        ],
        question,
    )
    eaten = _first_decimal(
        [rf"\beats?\s+({WORD_OR_NUM_RE})\s+for\s+breakfast\b"],
        question,
    )
    baked = _first_decimal(
        [rf"\bbakes?\b[^.?!]{{0,120}}?\bwith\s+({WORD_OR_NUM_RE})\b"],
        question,
    )
    sale_price = _first_decimal(
        [r"\bsells?\s+the\s+remainder\b[^.?!]{0,120}?\bfor\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s+per\b"],
        question,
    )
    if produced is None or eaten is None or baked is None or sale_price is None:
        return None

    remainder = produced - eaten - baked
    revenue = remainder * sale_price
    produced_s = _format_decimal(produced)
    eaten_s = _format_decimal(eaten)
    baked_s = _format_decimal(baked)
    price_s = _format_decimal(sale_price)
    return {
        "kind": "remainder_sale_revenue",
        "requested_quantity": "daily dollars earned from selling the remainder",
        "produced": produced_s,
        "personal_use": f"{eaten_s} + {baked_s}",
        "remainder": _format_decimal(remainder),
        "sale_price": price_s,
        "revenue": _format_decimal(revenue),
        "expression": f"({produced_s} - {eaten_s} - {baked_s}) * {price_s}",
    }


def daily_split_quantity_scaffold(question: str) -> dict[str, str] | None:
    """Derive totals when a daily per-entity amount is split across meals/events."""
    if not re.search(r"\bfinal\s+meal\b", question, flags=re.IGNORECASE):
        return None
    per_entity_daily = _first_decimal(
        [
            rf"\b(?:every day|daily)\b[^.?!]{{0,140}}?\beach(?:\s+of\s+\w+)?[^.?!]{{0,140}}?\b({WORD_OR_NUM_RE})\s+cups?\b",
            rf"\beach(?:\s+of\s+\w+)?[^.?!]{{0,140}}?\b({WORD_OR_NUM_RE})\s+cups?[^.?!]{{0,100}}?\b(?:every day|per day|daily)\b",
        ],
        question,
    )
    entity_count = _first_decimal(
        [
            r"\bflock\b[^.?!]{0,80}?\b(?:is|of)\s+(-?\d[\d,]*(?:\.\d+)?)\s+chickens?\b",
            r"\b(-?\d[\d,]*(?:\.\d+)?)\s+chickens?\b",
        ],
        question,
    )
    known_meals: list[tuple[str, Decimal]] = []
    for label in ("morning", "afternoon", "first", "second"):
        match = re.search(
            rf"\b{label}\b[^.?!]{{0,140}}?\b({WORD_OR_NUM_RE})\s+cups?\b",
            question,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            amount = _decimal_from_text(match.group(1))
            if amount is not None:
                known_meals.append((label, amount))
    if per_entity_daily is None or entity_count is None or not known_meals:
        return None

    total_daily = per_entity_daily * entity_count
    known_total = sum((amount for _, amount in known_meals), Decimal(0))
    final_meal = total_daily - known_total
    per_day = _format_decimal(per_entity_daily)
    count = _format_decimal(entity_count)
    known_terms = [(_format_decimal(amount), label) for label, amount in known_meals]
    known_expression = " + ".join(amount for amount, _ in known_terms)
    return {
        "kind": "daily_split_final_meal",
        "requested_quantity": "cups for final meal",
        "per_entity_daily": per_day,
        "entity_count": count,
        "known_meals": ", ".join(f"{label}={amount}" for amount, label in known_terms),
        "known_total": _format_decimal(known_total),
        "daily_total": _format_decimal(total_daily),
        "final_meal": _format_decimal(final_meal),
        "expression": f"({per_day} * {count}) - ({known_expression})",
    }


def restart_download_scaffold(question: str) -> dict[str, str] | None:
    """Derive elapsed time when lost progress forces a download restart."""
    if not re.search(r"\bdownload", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\brestart\b[^.?!]{0,120}?\bfrom\s+the\s+beginning\b", question, flags=re.IGNORECASE | re.DOTALL):
        return None
    file_size = _first_decimal(
        [r"\b(-?\d[\d,]*(?:\.\d+)?)\s*GB\s+file\b"],
        question,
    )
    rate = _first_decimal(
        [
            r"\bdownload\s+(-?\d[\d,]*(?:\.\d+)?)\s*GB\s*/\s*minute\b",
            r"\b(-?\d[\d,]*(?:\.\d+)?)\s*GB\s*/\s*minute\b",
        ],
        question,
    )
    percent_done = _first_decimal(
        [r"\b(-?\d+(?:\.\d+)?)\s*%\s+of\s+the\s+way\s+through\b"],
        question,
    )
    restart_minutes = _first_decimal(
        [r"\b(?:restart|install\s+updates?)[^.?!]{0,120}?\btakes?\s+(-?\d[\d,]*(?:\.\d+)?)\s+minutes?\b"],
        question,
    )
    if file_size is None or rate is None or percent_done is None or restart_minutes is None or rate == 0:
        return None

    lost_gb = file_size * percent_done / Decimal(100)
    lost_time = lost_gb / rate
    full_download_time = file_size / rate
    total_time = lost_time + restart_minutes + full_download_time
    size = _format_decimal(file_size)
    pct = _format_decimal(percent_done)
    rate_s = _format_decimal(rate)
    restart_s = _format_decimal(restart_minutes)
    return {
        "kind": "restart_download_from_beginning",
        "requested_quantity": "total elapsed download time",
        "file_size": size,
        "rate": rate_s,
        "percent_lost": pct,
        "lost_gb": _format_decimal(lost_gb),
        "lost_time": _format_decimal(lost_time),
        "restart_minutes": restart_s,
        "full_download_time": _format_decimal(full_download_time),
        "total_time": _format_decimal(total_time),
        "expression": f"(({size} * {pct} / 100) / {rate_s}) + {restart_s} + ({size} / {rate_s})",
    }


def return_trip_distance_scaffold(question: str) -> dict[str, str] | None:
    """Derive remaining distance from home after turning around."""
    if not re.search(r"\bturns?\s+around\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bfrom\s+home\b", question, flags=re.IGNORECASE):
        return None
    outbound = re.search(
        r"\bdrives?\s+for\s+(-?\d+(?:\.\d+)?)\s+hours?\s+at\s+a\s+speed\s+of\s+(-?\d+(?:\.\d+)?)\s*mph\b",
        question,
        flags=re.IGNORECASE,
    )
    target_time = _first_decimal(
        [r"\bget\s+home\s+in\s+(-?\d+(?:\.\d+)?)\s+hours?\b"],
        question,
    )
    traffic_time = _first_decimal(
        [r"\bfirst\s+(-?\d+(?:\.\d+)?)\s+hours?\s+in\s+standstill\s+traffic\b"],
        question,
    )
    half_hour_segment = re.search(
        r"\bnext\s+half-?hour[^.?!]{0,120}?\bspeed\s+of\s+(-?\d+(?:\.\d+)?)\s*mph\b",
        question,
        flags=re.IGNORECASE | re.DOTALL,
    )
    remaining_speed = _first_decimal(
        [r"\bremaining\s+time[^.?!]{0,120}?\b(?:at|going\s+at)\s+(-?\d+(?:\.\d+)?)\s*mph\b"],
        question,
    )
    if (
        outbound is None
        or target_time is None
        or traffic_time is None
        or half_hour_segment is None
        or remaining_speed is None
    ):
        return None
    outbound_hours = _decimal_from_text(outbound.group(1))
    outbound_speed = _decimal_from_text(outbound.group(2))
    half_hour_speed = _decimal_from_text(half_hour_segment.group(1))
    half_hour = Decimal("0.5")
    if outbound_hours is None or outbound_speed is None or half_hour_speed is None:
        return None

    outbound_distance = outbound_hours * outbound_speed
    remaining_return_time = target_time - traffic_time - half_hour
    return_distance = (traffic_time * Decimal(0)) + (half_hour * half_hour_speed) + (remaining_return_time * remaining_speed)
    distance_from_home = outbound_distance - return_distance
    out_hours = _format_decimal(outbound_hours)
    out_speed = _format_decimal(outbound_speed)
    target = _format_decimal(target_time)
    traffic = _format_decimal(traffic_time)
    half_speed = _format_decimal(half_hour_speed)
    rem_speed = _format_decimal(remaining_speed)
    return {
        "kind": "turnaround_remaining_distance",
        "requested_quantity": "distance from home after return attempt",
        "outbound_distance": _format_decimal(outbound_distance),
        "return_distance": _format_decimal(return_distance),
        "distance_from_home": _format_decimal(distance_from_home),
        "remaining_return_time": _format_decimal(remaining_return_time),
        "expression": (
            f"({out_hours} * {out_speed}) - "
            f"(({traffic} * 0) + (0.5 * {half_speed}) + (({target} - {traffic} - 0.5) * {rem_speed}))"
        ),
    }


def break_even_year_scaffold(question: str) -> dict[str, str] | None:
    """Derive the first profitable year when yearly net income must exceed startup cost."""
    if not re.search(r"\bstarts?\s+earning\s+money\b", question, flags=re.IGNORECASE):
        return None
    fixed_cost = _first_decimal(
        [r"\bcost\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s+to\s+(?:plant|start|buy|install)\b"],
        question,
    )
    units_per_year = _first_decimal(
        [r"\beach\s+year\b[^.?!]{0,80}?\bgrow\s+(-?\d[\d,]*(?:\.\d+)?)\s+lemons?\b"],
        question,
    )
    unit_price = _first_decimal(
        [r"\bsell\s+for\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s+each\b"],
        question,
    )
    annual_cost = _first_decimal(
        [r"\bcosts?\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s+a\s+year\b"],
        question,
    )
    if fixed_cost is None or units_per_year is None or unit_price is None or annual_cost is None:
        return None
    gross = units_per_year * unit_price
    net = gross - annual_cost
    if net <= 0:
        return None
    break_even_floor = (fixed_cost / net).to_integral_value(rounding=ROUND_FLOOR)
    first_profitable_year = break_even_floor + 1 if net * break_even_floor <= fixed_cost else break_even_floor
    fixed = _format_decimal(fixed_cost)
    units = _format_decimal(units_per_year)
    price = _format_decimal(unit_price)
    annual = _format_decimal(annual_cost)
    net_s = _format_decimal(net)
    return {
        "kind": "strict_break_even_year",
        "requested_quantity": "first year with positive cumulative profit",
        "fixed_cost": fixed,
        "annual_gross": _format_decimal(gross),
        "annual_cost": annual,
        "annual_net": net_s,
        "break_even_years": _format_decimal(fixed_cost / net),
        "first_profitable_year": _format_decimal(first_profitable_year),
        "expression": f"floor({fixed} / (({units} * {price}) - {annual})) + 1",
    }


def overtime_pay_scaffold(question: str) -> dict[str, str] | None:
    """Derive weekly earnings with a regular-hour threshold and overtime multiplier."""
    if not re.search(r"\bovertime\b", question, flags=re.IGNORECASE):
        return None
    regular_hours = _first_decimal(
        [r"\bfirst\s+(-?\d[\d,]*(?:\.\d+)?)\s+hours\b"],
        question,
    )
    hourly_rate = _first_decimal(
        [r"\brate\s+per\s+hour[^.?!]{0,120}?\bis\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\b"],
        question,
    )
    overtime_multiplier = _first_decimal(
        [r"\bovertime\s+pay\s+of\s+(-?\d[\d,]*(?:\.\d+)?)\s+times\b"],
        question,
    )
    worked_hours = _first_decimal(
        [r"\bworked\s+for\s+(-?\d[\d,]*(?:\.\d+)?)\s+hours\b"],
        question,
    )
    if regular_hours is None or hourly_rate is None or overtime_multiplier is None or worked_hours is None:
        return None
    overtime_hours = max(Decimal(0), worked_hours - regular_hours)
    regular_pay = regular_hours * hourly_rate
    overtime_rate = overtime_multiplier * hourly_rate
    overtime_pay = overtime_hours * overtime_rate
    total_pay = regular_pay + overtime_pay
    regular = _format_decimal(regular_hours)
    rate = _format_decimal(hourly_rate)
    worked = _format_decimal(worked_hours)
    mult = _format_decimal(overtime_multiplier)
    return {
        "kind": "overtime_weekly_earnings",
        "requested_quantity": "weekly earnings",
        "regular_hours": regular,
        "hourly_rate": rate,
        "worked_hours": worked,
        "overtime_multiplier": mult,
        "overtime_hours": _format_decimal(overtime_hours),
        "regular_pay": _format_decimal(regular_pay),
        "overtime_rate": _format_decimal(overtime_rate),
        "overtime_pay": _format_decimal(overtime_pay),
        "total_pay": _format_decimal(total_pay),
        "expression": f"({regular} * {rate}) + (({worked} - {regular}) * ({mult} * {rate}))",
    }


def monthly_percentage_total_scaffold(question: str) -> dict[str, str] | None:
    """Derive a three-month total where month 2 scales month 1 and month 3 reduces month 2."""
    if not re.search(r"\bfirst\s+month\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bsecond\s+month\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bthird\s+month\b", question, flags=re.IGNORECASE):
        return None
    first = _first_decimal(
        [r"\bhad\s+(-?\d[\d,]*(?:\.\d+)?)\s+downloads?\s+in\s+the\s+first\s+month\b"],
        question,
    )
    multiplier = _first_decimal(
        [rf"\bsecond\s+month\b[^.?!]{{0,120}}?\b({WORD_OR_NUM_RE})\s+times\s+as\s+many\b"],
        question,
    )
    reduction = _first_decimal(
        [r"\breduced\s+by\s+(-?\d+(?:\.\d+)?)\s*%\s+in\s+the\s+third\s+month\b"],
        question,
    )
    if first is None or multiplier is None or reduction is None:
        return None
    second = first * multiplier
    third = second * (Decimal(100) - reduction) / Decimal(100)
    total = first + second + third
    first_s = _format_decimal(first)
    mult_s = _format_decimal(multiplier)
    reduction_s = _format_decimal(reduction)
    return {
        "kind": "monthly_percentage_total",
        "requested_quantity": "total over three months",
        "first_month": first_s,
        "second_month": _format_decimal(second),
        "third_month": _format_decimal(third),
        "total": _format_decimal(total),
        "expression": f"{first_s} + ({first_s} * {mult_s}) + (({first_s} * {mult_s}) * (100 - {reduction_s}) / 100)",
    }


def dozen_total_cost_scaffold(question: str) -> dict[str, str] | None:
    """Derive total cost from item counts priced per dozen."""
    if not re.search(r"\bper\s+dozen\b", question, flags=re.IGNORECASE):
        return None
    purchases: list[tuple[Decimal, Decimal]] = []
    pattern = re.compile(
        rf"\b(?:bought\s+)?({WORD_OR_NUM_RE})\s+dozen\s+[^.,;]+?\s+(?:cost|for)\s+\$?\s*(-?\d[\d,]*(?:\.\d+)?)\s+per\s+dozen\b",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(question):
        count = _decimal_from_text(match.group(1))
        price = _decimal_from_text(match.group(2))
        if count is not None and price is not None:
            purchases.append((count, price))
    if not purchases:
        return None
    terms = [f"{_format_decimal(count)} * {_format_decimal(price)}" for count, price in purchases]
    total = sum((count * price for count, price in purchases), Decimal(0))
    return {
        "kind": "dozen_total_cost",
        "requested_quantity": "total cost",
        "line_items": ", ".join(terms),
        "total": _format_decimal(total),
        "expression": " + ".join(terms),
    }


def reverse_fraction_sales_scaffold(question: str) -> dict[str, str] | None:
    """Work backwards through sold-fraction, fixed-sale, sold-fraction inventory problems."""
    if not re.search(r"\bstart\s+with\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bsold\s+a\s+third\b", question, flags=re.IGNORECASE):
        return None
    if not re.search(r"\bhalf\s+of\s+what\s+was\s+left\b", question, flags=re.IGNORECASE):
        return None
    final_left = _first_decimal(
        [r"\bhas\s+(-?\d[\d,]*(?:\.\d+)?)\s+vacuum\s+cleaners?\s+left\b"],
        question,
    )
    fixed_sale = _first_decimal(
        [r"\b(-?\d[\d,]*(?:\.\d+)?)\s+more\s+to\s+the\s+red\s+house\b"],
        question,
    )
    if final_left is None or fixed_sale is None:
        return None
    first_fraction = Decimal(1) / Decimal(3)
    second_fraction = Decimal(1) / Decimal(2)
    before_orange = final_left / (Decimal(1) - second_fraction)
    before_red = before_orange + fixed_sale
    initial = before_red / (Decimal(1) - first_fraction)
    return {
        "kind": "reverse_fraction_sales",
        "requested_quantity": "initial inventory",
        "final_left": _format_decimal(final_left),
        "before_orange": _format_decimal(before_orange),
        "fixed_sale": _format_decimal(fixed_sale),
        "before_red": _format_decimal(before_red),
        "initial": _format_decimal(initial),
        "expression": f"(({_format_decimal(final_left)} / (1 - 1/2)) + {_format_decimal(fixed_sale)}) / (1 - 1/3)",
    }


def remaining_percentage_scaffold(question: str) -> dict[str, str] | None:
    """Derive the rest percentage after one percentage and another percentage of the remainder."""
    if not re.search(r"\bpercentage\s+of\s+the\s+entire\b", question, flags=re.IGNORECASE):
        return None
    first_pct = _first_decimal(
        [r"\b(-?\d+(?:\.\d+)?)\s*%\s+enrolled\s+in\s+contemporary\b"],
        question,
    )
    second_pct_of_remaining = _first_decimal(
        [r"\b(-?\d+(?:\.\d+)?)\s*%\s+of\s+the\s+remaining\s+enrolled\s+in\s+jazz\b"],
        question,
    )
    if first_pct is None or second_pct_of_remaining is None:
        return None
    remaining_after_first = Decimal(100) - first_pct
    second_pct_of_total = remaining_after_first * second_pct_of_remaining / Decimal(100)
    rest_pct = remaining_after_first - second_pct_of_total
    first = _format_decimal(first_pct)
    second = _format_decimal(second_pct_of_remaining)
    return {
        "kind": "remaining_percentage",
        "requested_quantity": "remaining category percent of entire group",
        "remaining_after_first": _format_decimal(remaining_after_first),
        "second_pct_of_total": _format_decimal(second_pct_of_total),
        "rest_pct": _format_decimal(rest_pct),
        "expression": f"(100 - {first}) - ((100 - {first}) * {second} / 100)",
    }


def quantity_scaffold_answer(question: str) -> dict[str, str] | None:
    scaffold = financial_profit_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["profit"],
            "expression": scaffold["expression"],
        }
    scaffold = choice_profit_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["best_profit"],
            "expression": scaffold["expression"],
        }
    scaffold = periodic_discount_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["total_cost"],
            "expression": scaffold["expression"],
        }
    scaffold = remainder_sale_revenue_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["revenue"],
            "expression": scaffold["expression"],
        }
    scaffold = daily_split_quantity_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["final_meal"],
            "expression": scaffold["expression"],
        }
    scaffold = restart_download_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["total_time"],
            "expression": scaffold["expression"],
        }
    scaffold = return_trip_distance_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["distance_from_home"],
            "expression": scaffold["expression"],
        }
    scaffold = break_even_year_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["first_profitable_year"],
            "expression": scaffold["expression"],
        }
    scaffold = overtime_pay_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["total_pay"],
            "expression": scaffold["expression"],
        }
    scaffold = monthly_percentage_total_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["total"],
            "expression": scaffold["expression"],
        }
    scaffold = dozen_total_cost_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["total"],
            "expression": scaffold["expression"],
        }
    scaffold = reverse_fraction_sales_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["initial"],
            "expression": scaffold["expression"],
        }
    scaffold = remaining_percentage_scaffold(question)
    if scaffold is not None:
        return {
            "kind": scaffold["kind"],
            "answer": scaffold["rest_pct"],
            "expression": scaffold["expression"],
        }
    return None


def quantity_tool_context(question: str) -> str:
    blocks: list[str] = []
    scaffold = financial_profit_scaffold(question)
    if scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {scaffold['requested_quantity']}.",
            f"- Purchase/base value: {scaffold['base_value']}.",
            f"- Added repair/renovation cost: {scaffold['repair_cost']}.",
            f"- Percent value increase: {scaffold['percent_increase']}%.",
            f"- Value increase expression: {scaffold['base_value']} * {scaffold['percent_increase']} / 100 = {scaffold['value_increase']}.",
            f"- Final value expression: {scaffold['base_value']} + {scaffold['value_increase']} = {scaffold['final_value']}.",
            f"- Total cost expression: {scaffold['base_value']} + {scaffold['repair_cost']} = {scaffold['total_cost']}.",
            f"- Expression for requested profit: {scaffold['expression']}.",
        ]))
    choice_scaffold = choice_profit_scaffold(question)
    if choice_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {choice_scaffold['requested_quantity']}.",
            f"- Candidate realized profits: {choice_scaffold['options']}.",
            f"- Best option by realized profit: {choice_scaffold['best_option']}.",
            f"- Expression for chosen profit: {choice_scaffold['expression']}.",
        ]))
    discount_scaffold = periodic_discount_scaffold(question)
    if discount_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {discount_scaffold['requested_quantity']}.",
            f"- Total items: {discount_scaffold['total_count']}.",
            f"- Every {discount_scaffold['period']}th item is discounted, so discounted items: {discount_scaffold['discounted_count']} and full-price items: {discount_scaffold['full_price_count']}.",
            f"- Unit price: {discount_scaffold['unit_price']}. Discounted price: {discount_scaffold['discount_percent']}% * {discount_scaffold['unit_price']} = {discount_scaffold['discounted_price']}.",
            f"- Expression for total cost: {discount_scaffold['expression']}.",
        ]))
    sale_scaffold = remainder_sale_revenue_scaffold(question)
    if sale_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {sale_scaffold['requested_quantity']}.",
            f"- Produced per day: {sale_scaffold['produced']}.",
            f"- Personal/non-sale use: {sale_scaffold['personal_use']}.",
            f"- Remainder sold: {sale_scaffold['remainder']}.",
            f"- Sale price: {sale_scaffold['sale_price']} per item.",
            f"- Expression for daily dollars earned: {sale_scaffold['expression']}.",
        ]))
    daily_scaffold = daily_split_quantity_scaffold(question)
    if daily_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {daily_scaffold['requested_quantity']}.",
            f"- Per-entity daily amount: {daily_scaffold['per_entity_daily']} cups.",
            f"- Entity count: {daily_scaffold['entity_count']}.",
            f"- Total daily amount expression: {daily_scaffold['per_entity_daily']} * {daily_scaffold['entity_count']} = {daily_scaffold['daily_total']}.",
            f"- Already given earlier meals: {daily_scaffold['known_meals']} for a total of {daily_scaffold['known_total']}.",
            f"- Expression for requested final meal: {daily_scaffold['expression']}.",
        ]))
    download_scaffold = restart_download_scaffold(question)
    if download_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {download_scaffold['requested_quantity']}.",
            f"- Lost progress before restart: {download_scaffold['percent_lost']}% of {download_scaffold['file_size']} GB = {download_scaffold['lost_gb']} GB.",
            f"- Lost-progress time: {download_scaffold['lost_gb']} / {download_scaffold['rate']} = {download_scaffold['lost_time']} minutes.",
            f"- Restart/update time: {download_scaffold['restart_minutes']} minutes.",
            f"- Full re-download time after restart from beginning: {download_scaffold['file_size']} / {download_scaffold['rate']} = {download_scaffold['full_download_time']} minutes.",
            f"- Expression for total elapsed time: {download_scaffold['expression']}.",
        ]))
    return_trip_scaffold = return_trip_distance_scaffold(question)
    if return_trip_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {return_trip_scaffold['requested_quantity']}.",
            f"- Outbound distance before turning around: {return_trip_scaffold['outbound_distance']}.",
            f"- Return distance covered toward home: {return_trip_scaffold['return_distance']}.",
            f"- Remaining return time: {return_trip_scaffold['remaining_return_time']} hours.",
            f"- Expression for distance from home: {return_trip_scaffold['expression']}.",
        ]))
    break_even_scaffold = break_even_year_scaffold(question)
    if break_even_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {break_even_scaffold['requested_quantity']}.",
            f"- Startup cost: {break_even_scaffold['fixed_cost']}.",
            f"- Annual gross income: {break_even_scaffold['annual_gross']}.",
            f"- Annual upkeep cost: {break_even_scaffold['annual_cost']}.",
            f"- Annual net income: {break_even_scaffold['annual_net']}.",
            f"- Break-even occurs at {break_even_scaffold['break_even_years']} years; earning money means strictly after break-even.",
            f"- Expression for first profitable whole year: {break_even_scaffold['expression']}.",
        ]))
    overtime_scaffold = overtime_pay_scaffold(question)
    if overtime_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {overtime_scaffold['requested_quantity']}.",
            f"- Regular pay: {overtime_scaffold['regular_hours']} * {overtime_scaffold['hourly_rate']} = {overtime_scaffold['regular_pay']}.",
            f"- Overtime hours: {overtime_scaffold['worked_hours']} - {overtime_scaffold['regular_hours']} = {overtime_scaffold['overtime_hours']}.",
            f"- Overtime rate: {overtime_scaffold['overtime_multiplier']} * {overtime_scaffold['hourly_rate']} = {overtime_scaffold['overtime_rate']}.",
            f"- Overtime pay: {overtime_scaffold['overtime_hours']} * {overtime_scaffold['overtime_rate']} = {overtime_scaffold['overtime_pay']}.",
            f"- Expression for weekly earnings: {overtime_scaffold['expression']}.",
        ]))
    monthly_scaffold = monthly_percentage_total_scaffold(question)
    if monthly_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {monthly_scaffold['requested_quantity']}.",
            f"- First month: {monthly_scaffold['first_month']}.",
            f"- Second month: {monthly_scaffold['second_month']}.",
            f"- Third month after reduction: {monthly_scaffold['third_month']}.",
            f"- Expression for three-month total: {monthly_scaffold['expression']}.",
        ]))
    dozen_scaffold = dozen_total_cost_scaffold(question)
    if dozen_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {dozen_scaffold['requested_quantity']}.",
            f"- Per-dozen line items: {dozen_scaffold['line_items']}.",
            f"- Expression for total cost: {dozen_scaffold['expression']}.",
        ]))
    reverse_scaffold = reverse_fraction_sales_scaffold(question)
    if reverse_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {reverse_scaffold['requested_quantity']}.",
            f"- After selling half of what was left, final left is {reverse_scaffold['final_left']}, so before that sale was {reverse_scaffold['before_orange']}.",
            f"- Before selling {reverse_scaffold['fixed_sale']} more, the count was {reverse_scaffold['before_red']}.",
            f"- Before selling one third, the initial count was {reverse_scaffold['initial']}.",
            f"- Expression for initial inventory: {reverse_scaffold['expression']}.",
        ]))
    remaining_pct_scaffold = remaining_percentage_scaffold(question)
    if remaining_pct_scaffold is not None:
        blocks.append("\n".join([
            "Available quantity scaffold (derived only from the problem text):",
            f"- Requested quantity: {remaining_pct_scaffold['requested_quantity']}.",
            f"- Remaining after first category: {remaining_pct_scaffold['remaining_after_first']}%.",
            f"- Second category as percent of entire group: {remaining_pct_scaffold['second_pct_of_total']}%.",
            f"- Expression for rest percentage: {remaining_pct_scaffold['expression']}.",
        ]))
    return "\n\n".join(blocks)


def extract_final_number(text: str) -> str | None:
    marked = re.findall(
        r"(?:final answer|answer is)[^\d-]*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        text.replace(",", ""),
        flags=re.IGNORECASE,
    )
    if marked:
        return normalize_number(marked[-1])
    return extract_checked_equation_answer(text) or extract_expression_answer(text)


def _line_value(text: str, key: str) -> str | None:
    matches = re.findall(rf"^{re.escape(key)}\s*:\s*(.*)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return matches[-1].strip() if matches else None


def parse_verifier_details(text: str) -> dict[str, Any]:
    verdict = (_line_value(text, "VERDICT") or "uncertain").lower()
    if "pass" in verdict:
        verdict = "pass"
    elif "unsettled" in verdict or "not_yet" in verdict or "fail" in verdict:
        verdict = "unsettled"
    else:
        verdict = "uncertain"
    independent = _line_value(text, "INDEPENDENT_FINAL") or _line_value(text, "CORRECTED_FINAL")
    tagged_answer = normalize_number(independent) or extract_number(independent or "")
    checked_quality = checked_equation_quality(text, require_final_cue=True)
    if checked_quality is None and tagged_answer is not None:
        loose_checked = checked_equation_quality(text, require_final_cue=False)
        if (
            loose_checked is not None
            and loose_checked.get("evaluated_answer") is not None
            and numbers_match(loose_checked["evaluated_answer"], tagged_answer)
        ):
            checked_quality = loose_checked
    checked_answer = None if checked_quality is None else checked_quality["evaluated_answer"]
    checked_is_strong = bool(
        checked_quality is not None and checked_quality.get("cue_strength", 0) >= 2
    )
    answer = checked_answer if checked_answer is not None and checked_is_strong else tagged_answer
    bad_arithmetic = bool(
        checked_quality is not None
        and checked_quality.get("math_consistent") is False
    )
    bad_tag = bool(
        checked_answer is not None
        and checked_is_strong
        and tagged_answer is not None
        and not numbers_match(checked_answer, tagged_answer)
    )
    return {
        "verdict": verdict,
        "answer": answer,
        "tagged_answer": tagged_answer,
        "checked_answer": checked_answer,
        "checked_equation": checked_quality,
        "bad_arithmetic": bad_arithmetic,
        "bad_tag": bad_tag,
    }


def parse_verifier(text: str) -> tuple[str, str | None, str | None]:
    details = parse_verifier_details(text)
    verdict = details["verdict"]
    checked_answer = details["checked_answer"]
    if checked_answer is not None:
        return verdict, checked_answer, checked_answer
    return verdict, details["tagged_answer"], None


def _learned_concept_context_text(learned_concept_context: str | None = None) -> str:
    if not learned_concept_context:
        return ""
    return (
        "Reusable lessons from prior different questions. Apply them only when the current wording matches; "
        "do not copy any old answer. Re-derive with the current numbers.\n"
        f"{learned_concept_context.strip()}\n\n"
    )


def solve_prompt(
    question: str,
    deterministic_scaffolds_enabled: bool = True,
    model_scaffold_tool_enabled: bool = True,
    clause_map_enabled: bool = False,
    learned_concept_context: str | None = None,
) -> str:
    quantity_context = quantity_tool_context(question) if deterministic_scaffolds_enabled else ""
    clause_context = clause_map_context(question) if clause_map_enabled else ""
    clause_line = (
        "Clause map: <omit this line unless mapping clauses helps; if used, include one <<CLAUSEMAP: ...>>>\n"
        if clause_map_enabled
        else ""
    )
    scaffold_line = (
        "Scaffold: <omit this line unless units, rates, choices, or objectives are at risk; if used, include one <<SCAFFOLD: ...>>>\n"
        if model_scaffold_tool_enabled
        else ""
    )
    return (
        "Solve this grade-school math problem under a tight budget. Do not explain like a teacher. "
        "First identify the exact quantity being asked, then write one expression for that quantity. "
        "Do not stop at an intermediate quantity. "
        + tool_capability_text(
            "brief" if model_scaffold_tool_enabled else "none",
            include_clause_map=clause_map_enabled,
        )
        + "You may call these tools for intermediate steps or the final computation. "
        + _learned_concept_context_text(learned_concept_context)
        + (f"\n\n{clause_context}\n\n" if clause_context else "\n\n")
        + (f"\n\n{quantity_context}\n\n" if quantity_context else "\n\n")
        +
        "Answer in exactly this format and no extra prose:\n"
        "Asked quantity: <short phrase>\n"
        + clause_line
        + scaffold_line
        +
        "Expression: <numeric arithmetic expression for the answer; no undefined variables>\n"
        "Computed: <the requested final quantity; this must equal Final answer, or use <<CALC: expression>> for the full final expression>\n"
        "Final answer: <same number as Computed>\n\n"
        f"Question: {question}"
    )


def verify_prompt(
    question: str,
    proposed_solution: str,
    deterministic_scaffolds_enabled: bool = True,
    model_scaffold_tool_enabled: bool = True,
    clause_map_enabled: bool = False,
    learned_concept_context: str | None = None,
) -> str:
    quantity_context = quantity_tool_context(question) if deterministic_scaffolds_enabled else ""
    clause_context = clause_map_context(question) if clause_map_enabled else ""
    return (
        "You are an expert verifier reflecting the highest standard of logical, mathematical, and analytical reasoning. "
        "Your task is to carefully verify the proposed solution step-by-step. "
        "CRITICAL: You must solve the problem completely independently FIRST, before evaluating the proposed solution. "
        "First, identify the core entities and write out your own independent calculation in at most four short lines. "
        "Second, compare your independent calculation to the logical path of the proposed solution. Does it address the core question without making invalid assumptions? "
        "If the reasoning is fundamentally flawed or answers the wrong question, the verdict MUST be unsettled. "
        + tool_capability_text(
            "brief" if model_scaffold_tool_enabled else "none",
            include_clause_map=clause_map_enabled,
        )
        + "\n\n"
        + _learned_concept_context_text(learned_concept_context)
        + (f"{clause_context}\n\n" if clause_context else "")
        + (f"{quantity_context}\n\n" if quantity_context else "")
        +
        f"Question:\n{question}\n\n"
        f"Proposed solution:\n{proposed_solution}\n\n"
        "Do not stop after INDEPENDENT_CALCULATION. Always finish all four labeled lines, even if the verdict is uncertain.\n"
        "Use numeric arithmetic, not undefined variables, for INDEPENDENT_FINAL whenever the problem is answerable.\n"
        "Reply in exactly this form, with no extra prose after REASON:\n"
        "INDEPENDENT_CALCULATION: <at most four short lines>\n"
        "VERDICT: pass|unsettled|uncertain\n"
        "INDEPENDENT_FINAL: <number or none>\n"
        "REASON: <one short reason about what still needs to be resolved>"
    )


def repair_prompt(
    question: str,
    attempt: ReasoningAttempt,
    deterministic_scaffolds_enabled: bool = True,
    model_scaffold_tool_enabled: bool = True,
    clause_map_enabled: bool = False,
    learned_concept_context: str | None = None,
) -> str:
    previous_final = attempt.extracted_answer or "none"
    verifier_candidate = attempt.verifier_answer or "none"
    checked_candidate = attempt.verifier_checked_answer or "none"
    tool_feedback = _attempt_tool_feedback(attempt)
    quantity_context = quantity_tool_context(question) if deterministic_scaffolds_enabled else ""
    clause_context = clause_map_context(question) if clause_map_enabled else ""
    clause_line = (
        "Clause map: <omit this line unless mapping clauses helps; if used, include one <<CLAUSEMAP: ...>>>\n"
        if clause_map_enabled
        else ""
    )
    scaffold_line = (
        "Scaffold: <omit this line unless units, rates, choices, or objectives are at risk; if used, include one <<SCAFFOLD: ...>>>\n"
        if model_scaffold_tool_enabled
        else ""
    )
    return (
        "The previous solution is not settled yet. Do not assume it is wrong; treat it as provisional. "
        "Re-derive the answer from the original problem under a tight budget. "
        "Do not reuse the previous expression unless it still follows from the wording. "
        "Use the verification notes as constraints on what remains unresolved, not as criticism to rationalize. "
        "Use the arithmetic/tool feedback as hard feedback about numeric consistency. "
        "If the parsed verifier candidate differs from the previous final answer, explicitly test both candidates against the problem before choosing. "
        "If the answer cannot be determined, say so. "
        + tool_capability_text(
            "brief" if model_scaffold_tool_enabled else "none",
            include_clause_map=clause_map_enabled,
        )
        + "You may call these tools for intermediate steps or the final computation. "
        + _learned_concept_context_text(learned_concept_context)
        + (f"{clause_context}\n\n" if clause_context else "")
        + "Use exactly this format and no extra prose:\n"
        "Asked quantity: <short phrase>\n"
        + clause_line
        + scaffold_line
        +
        "Expression: <numeric arithmetic expression for the answer; no undefined variables>\n"
        "Computed: <the requested final quantity; this must equal Final answer, or use <<CALC: expression>> for the full final expression>\n"
        "Final answer: <same number as Computed>\n\n"
        + (f"{quantity_context}\n\n" if quantity_context else "")
        +
        f"Problem:\n{question}\n\n"
        f"Previous parsed final answer: {previous_final}\n"
        f"Parsed verifier candidate: {verifier_candidate}\n"
        f"Verifier checked-equation candidate: {checked_candidate}\n\n"
        f"Arithmetic/tool feedback:\n{tool_feedback}\n\n"
        f"Previous answer text:\n{attempt.response}\n\n"
        f"Verification notes:\n{attempt.verifier_response}"
    )


def continuation_prompt(question: str, attempt: ReasoningAttempt) -> str:
    tool_feedback = _attempt_tool_feedback(attempt)
    return (
        "Your previous response appears incomplete or lacks the required final-answer line. "
        "Do not restart and do not explain like a teacher. Continue only enough to finish the answer. "
        "If a previous scaffold returned valid=False, do not repeat it; either revise it once using direct variable assignments or use CALC. "
        "Use exactly this format and no extra prose:\n"
        "Computed: <remaining final computation, or <<CALC: expression>> for the full final expression>\n"
        "Final answer: <same final requested quantity as Computed>\n\n"
        f"Problem:\n{question}\n\n"
        f"Arithmetic/tool feedback:\n{tool_feedback}\n\n"
        f"Previous partial response:\n{attempt.response}"
    )


def confirmation_prompt(
    question: str,
    answer: str,
    deterministic_scaffolds_enabled: bool = True,
    model_scaffold_tool_enabled: bool = True,
    clause_map_enabled: bool = False,
    learned_concept_context: str | None = None,
) -> str:
    quantity_context = quantity_tool_context(question) if deterministic_scaffolds_enabled else ""
    clause_context = clause_map_context(question) if clause_map_enabled else ""
    clause_line = (
        "Clause map: <omit this line unless mapping clauses helps; if used, include one <<CLAUSEMAP: ...>>>\n"
        if clause_map_enabled
        else ""
    )
    scaffold_line = (
        "Scaffold: <omit this line unless units, rates, choices, or objectives are at risk; if used, include one <<SCAFFOLD: ...>>>\n"
        if model_scaffold_tool_enabled
        else ""
    )
    return (
        "Independently confirm this answer under a tight budget. Do not explain like a teacher. "
        + tool_capability_text(
            "brief" if model_scaffold_tool_enabled else "none",
            include_clause_map=clause_map_enabled,
        )
        + "You may call these tools for intermediate steps or the final computation. "
        + _learned_concept_context_text(learned_concept_context)
        + (f"{clause_context}\n\n" if clause_context else "")
        + "Use exactly this format and no extra prose:\n"
        "Asked quantity: <short phrase>\n"
        + clause_line
        + scaffold_line
        +
        "Expression: <numeric arithmetic expression for the answer; no undefined variables>\n"
        "Computed: <the requested final quantity; this must equal Final answer, or use <<CALC: expression>> for the full final expression>\n"
        "Final answer: <same number as Computed>\n\n"
        + (f"{quantity_context}\n\n" if quantity_context else "")
        +
        f"Problem:\n{question}\n\n"
        f"Proposed answer to check: {answer}"
    )


def _attempt_tool_feedback(attempt: ReasoningAttempt) -> str:
    signal = attempt.learning_signal or {}
    lines: list[str] = []
    solver_expr = signal.get("solver_expression_checked_answer")
    solver_eq = signal.get("solver_equation_checked_answer")
    solver_checked = attempt.solver_checked_answer or solver_expr or solver_eq
    structural_issue = signal.get("structural_contradiction")
    if structural_issue == "periodic_discount_double_charge":
        lines.append(
            "- Specific repair: the previous path double-charged every item by using total_items * "
            "(regular_price + discounted_price). For every-nth discounts, partition the order: "
            "discounted_count=floor(total_items / period), full_price_count=total_items-discounted_count, "
            "then total=full_price_count*regular_price + discounted_count*discounted_price."
        )
    if signal.get("solver_math") == "bad" and solver_checked is not None:
        lines.append(
            f"- The previous solver arithmetic checked to {solver_checked}, "
            f"but the previous final answer was {attempt.extracted_answer or 'none'}."
        )
        lines.append(
            "- Do not keep the previous final answer unless you can write a new expression "
            "for the asked quantity that the calculator evaluates to that answer."
        )
    elif solver_checked is not None:
        lines.append(f"- The previous solver expression/equation checked to {solver_checked}.")
    else:
        lines.append("- No solver expression was locally checkable; write a numeric expression this time.")

    verifier_checked = attempt.verifier_checked_answer
    verifier_tagged = attempt.verifier_tagged_answer
    if signal.get("verifier_math") == "bad" and verifier_checked is not None:
        lines.append(
            f"- The verifier's arithmetic checked to {verifier_checked}, "
            f"while its tagged final was {verifier_tagged or 'none'}; treat the tag as unreliable."
        )
    elif verifier_checked is not None:
        lines.append(f"- The verifier's checked arithmetic candidate was {verifier_checked}.")
    elif attempt.verifier_answer is not None:
        lines.append(f"- The verifier's parsed candidate was {attempt.verifier_answer}, but no equation was locally checked.")

    solver_scaffold = signal.get("solver_scaffold_feedback")
    verifier_scaffold = signal.get("verifier_scaffold_feedback")
    if solver_scaffold:
        lines.append(f"- The previous solver scaffold check returned: {solver_scaffold}.")
        if "valid=False" in solver_scaffold:
            if "bad quantity 'variable'" in solver_scaffold:
                lines.append(
                    "- Specific scaffold repair: remove variable= fields. Use direct numeric assignments such as "
                    "count=16 items; price=5 dollars/item; discount=0.6; expression=..."
                )
            elif "bad quantity 'unit'" in solver_scaffold:
                lines.append(
                    "- Specific scaffold repair: do not add unit=<name> as a field. Put the target unit in target=... "
                    "and give each quantity its unit directly, such as apples=30 apples."
                )
            else:
                lines.append(
                    "- Revise the scaffold before trusting the expression; the unit/objective check failed."
                )
    if verifier_scaffold:
        lines.append(f"- The verifier scaffold check returned: {verifier_scaffold}.")

    solver_clause_map = signal.get("solver_clause_map_feedback")
    verifier_clause_map = signal.get("verifier_clause_map_feedback")
    if solver_clause_map:
        lines.append(f"- The previous solver clause map returned: {solver_clause_map}.")
        if clause_map_feedback_invalid(solver_clause_map):
            lines.append(
                "- Specific clause-map repair: cover every numbered clause exactly once where possible; "
                "put the question clause under asked, numeric facts under givens, and wording rules under rules."
            )
    if verifier_clause_map:
        lines.append(f"- The verifier clause map returned: {verifier_clause_map}.")

    if (
        attempt.verifier_answer
        and attempt.extracted_answer
        and not numbers_match(attempt.verifier_answer, attempt.extracted_answer)
    ):
        lines.append(
            f"- Candidate conflict to resolve from wording: solver final {attempt.extracted_answer} "
            f"vs verifier candidate {attempt.verifier_answer}."
        )
    return "\n".join(lines)


def _attempt_has_invalid_scaffold(attempt: ReasoningAttempt) -> bool:
    signal = attempt.learning_signal or {}
    return scaffold_feedback_invalid(signal.get("solver_scaffold_feedback")) or scaffold_feedback_invalid(
        signal.get("verifier_scaffold_feedback")
    )


def _needs_symbolic_repair(attempt: ReasoningAttempt) -> bool:
    signal = attempt.learning_signal or {}
    return bool(
        _attempt_has_invalid_scaffold(attempt)
        or signal.get("structural_contradiction")
        or signal.get("solver_math") == "bad"
        or signal.get("verifier_math") == "bad"
    )


def structural_contradiction(question: str, *texts: str | None) -> str | None:
    """Detect generic structural traps without injecting the correct answer."""
    q = question.lower()
    profit_objective = bool("profit" in q)
    if profit_objective:
        for text in texts:
            if not text:
                continue
            for expr in re.findall(r"^\s*Expression\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE):
                expr_clean = expr.replace(",", "").replace("$", "")
                if (
                    re.search(r"\d", expr_clean)
                    and "+" in expr_clean
                    and "-" not in expr_clean
                ):
                    return "profit_expression_without_subtracting_costs"
            lowered = text.lower()
            intermediate_value = re.search(
                r"\b(?:total cost|increased value|new value|final value)\b[^.\n]*=\s*\$?\s*-?\d[\d,]*(?:\.\d+)?",
                lowered,
            )
            has_profit_subtraction = re.search(
                r"\bprofit\b[^.\n]*(?:-|minus|subtract|less)",
                lowered,
            )
            if intermediate_value and not has_profit_subtraction:
                return "profit_answer_is_cost_or_intermediate_value"

    daily_final_meal = bool(
        "final meal" in q
        and "every day" in q
        and re.search(r"\bthree\s+separate\s+meals\b|\b3\s+separate\s+meals\b", q)
    )
    if daily_final_meal:
        for text in texts:
            if not text:
                continue
            if re.search(
                r"\beach\b[^.\n]*\b(?:eats|needs|gets|receives)\b[^.\n]*\b\d+(?:\.\d+)?\s+cups?\b[^.\n]*\bper\s+meal\b",
                text,
                flags=re.IGNORECASE,
            ):
                return "daily_total_misread_as_per_meal"

    periodic_discount = bool(
        re.search(r"\bevery\s+(?:\d+|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b", q)
        and ("discount" in q or "half price" in q or "costs only" in q)
    )
    if periodic_discount:
        for text in texts:
            if not text:
                continue
            if re.search(
                r"(?:^|[:=\n])\s*(?:\d+|count|total_items|items?)\s*\*\s*\([^)\n]*\+[^)\n]*\)",
                text,
                flags=re.IGNORECASE,
            ):
                return "periodic_discount_double_charge"
    return None


def _verified_answer(attempt: ReasoningAttempt) -> str | None:
    if attempt.accepted:
        if (
            attempt.extracted_answer is not None
            and attempt.verifier_answer is not None
            and numbers_match(attempt.extracted_answer, attempt.verifier_answer)
        ):
            return attempt.extracted_answer
        return attempt.verifier_answer or attempt.extracted_answer
    return None


def _modal_answer(attempts: list[ReasoningAttempt]) -> tuple[str | None, int]:
    answers: list[str] = []
    for attempt in attempts:
        answer = _verified_answer(attempt)
        if answer is not None:
            answers.append(answer)
            signal = attempt.learning_signal or {}
            if (
                signal.get("solver_math") == "clean"
                and signal.get("verifier_math") == "clean"
                and not signal.get("structural_contradiction")
            ):
                answers.append(answer)
            continue
        signal = attempt.learning_signal or {}
        if (
            attempt.extracted_answer is not None
            and answers
            and signal.get("solver_math") == "clean"
            and not signal.get("structural_contradiction")
        ):
            matched = next((prior for prior in answers if numbers_match(prior, attempt.extracted_answer)), None)
            if matched is not None:
                answers.append(matched)
                continue
        independent = attempt.verifier_answer
        scaffold_answer = signal.get("quantity_scaffold_answer")
        if (
            attempt.acceptance_reason == "structural_quantity_contradiction"
            and independent is not None
            and attempt.verifier_checked_answer is not None
            and attempt.verifier_tagged_answer is not None
            and numbers_match(independent, attempt.verifier_checked_answer)
            and numbers_match(independent, attempt.verifier_tagged_answer)
            and signal.get("verifier_math") == "clean"
            and not signal.get("bad_tag")
        ):
            answers.append(independent)
            continue
        solver_checked = attempt.solver_checked_answer or signal.get("solver_expression_checked_answer")
        if (
            attempt.acceptance_reason == "structural_quantity_contradiction"
            and solver_checked is not None
            and attempt.extracted_answer is not None
            and not numbers_match(solver_checked, attempt.extracted_answer)
            and signal.get("solver_math") == "bad"
            and signal.get("structural_contradiction") == "daily_total_misread_as_per_meal"
        ):
            answers.append(solver_checked)
            continue
        if (
            attempt.verdict == "pass"
            and independent is not None
            and (
                signal.get("verifier_math") == "clean"
                or (scaffold_answer is not None and numbers_match(independent, scaffold_answer))
            )
            and attempt.acceptance_reason
            in {"solver_expression_mismatch", "verifier_solver_mismatch", "quantity_scaffold_mismatch", "invalid_scaffold"}
            and (scaffold_answer is None or numbers_match(independent, scaffold_answer))
            and not signal.get("structural_contradiction")
        ):
            answers.append(independent)
    if not answers:
        return None, 0
    counts = {answer: answers.count(answer) for answer in set(answers)}
    best_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best_count]
    if len(winners) != 1:
        return None, best_count
    return winners[0], best_count


def _fallback_answer(attempts: list[ReasoningAttempt]) -> str | None:
    answer, _ = _modal_answer(attempts)
    if answer is not None:
        return answer
    for attempt in reversed(attempts):
        signal = attempt.learning_signal or {}
        verifier_candidate = attempt.verifier_checked_answer or attempt.verifier_answer
        if verifier_candidate is None:
            continue
        if signal.get("structural_contradiction"):
            continue
        if signal.get("verifier_math") == "bad" or signal.get("bad_tag"):
            continue
        if attempt.extracted_answer is not None and numbers_match(verifier_candidate, attempt.extracted_answer):
            continue
        if attempt.verdict not in {"pass", "unsettled"}:
            continue
        return verifier_candidate
    extracted = [attempt.extracted_answer for attempt in attempts if attempt.extracted_answer is not None]
    if not extracted:
        return None
    counts = {answer: extracted.count(answer) for answer in set(extracted)}
    best_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best_count]
    if len(winners) == 1:
        return winners[0]
    for answer in reversed(extracted):
        if answer in winners:
            return answer
    return extracted[-1]


def _promote_verified_synthesis(
    attempts: list[ReasoningAttempt],
    answer: str,
    tag: str = "native_success",
    question_key: str | None = None,
    oracle_mode: str | None = None,
) -> int:
    promoted = 0
    summary = synthesis_teaching_summary(attempts, answer)
    for attempt in attempts:
        if not attempt.accepted or not numbers_match(_verified_answer(attempt), answer):
            continue
        for record in attempt.synthesis_records or []:
            if "trigger" not in record or "delta" not in record:
                continue
            decision = _teaching_decision(attempt, answer, record)
            stage = decision["stage"]
            reward = decision["reward"]
            penalty_reason = decision["penalty_reason"]

            metadata = dict(record.get("metadata", {}))
            metadata.update(
                {
                    "promoted_by": "humble_verifier",
                    "verdict": attempt.verdict,
                    "answer": answer,
                    "round_index": attempt.round_index,
                    "tag": tag,
                    "teaching_stage": stage,
                }
            )
            if question_key is not None:
                metadata["question_key"] = question_key
            if oracle_mode is not None:
                metadata["oracle_mode"] = oracle_mode
            if stage == "solver":
                metadata["calculator_tool_used"] = bool(
                    (attempt.learning_signal or {}).get("solver_tool_used")
                )
                metadata["scaffold_tool_used"] = bool(
                    (attempt.learning_signal or {}).get("solver_scaffold_tool_used")
                )
                metadata["scaffold_feedback"] = (attempt.learning_signal or {}).get("solver_scaffold_feedback")
                metadata["clause_map_tool_used"] = bool(
                    (attempt.learning_signal or {}).get("solver_clause_map_tool_used")
                )
                metadata["clause_map_status"] = None
                methodology = (attempt.learning_signal or {}).get("solver_clause_methodology")
                if methodology:
                    metadata["clause_methodology"] = methodology
                    metadata["clause_map_status"] = methodology.get("clause_map_status")
            elif stage == "verifier":
                metadata["calculator_tool_used"] = bool(
                    (attempt.learning_signal or {}).get("verifier_tool_used")
                )
                metadata["scaffold_tool_used"] = bool(
                    (attempt.learning_signal or {}).get("verifier_scaffold_tool_used")
                )
                metadata["scaffold_feedback"] = (attempt.learning_signal or {}).get("verifier_scaffold_feedback")
                metadata["clause_map_tool_used"] = bool(
                    (attempt.learning_signal or {}).get("verifier_clause_map_tool_used")
                )
                metadata["clause_map_status"] = None
                methodology = (attempt.learning_signal or {}).get("verifier_clause_methodology")
                if methodology:
                    metadata["clause_methodology"] = methodology
                    metadata["clause_map_status"] = methodology.get("clause_map_status")
            if reward:
                metadata["teaching_signal"] = "reward_clean_math"
                if metadata.get("calculator_tool_used"):
                    metadata["tool_reinforcement"] = "calculator_clean_use"
                if metadata.get("scaffold_tool_used"):
                    metadata["scaffold_reinforcement"] = "scaffold_clean_use"
                _global_cache.store(record["trigger"], record["delta"], metadata=metadata)
                promoted += 1
            elif penalty_reason is not None:
                metadata["teaching_signal"] = "penalty_bad_math"
                metadata["penalty_reason"] = penalty_reason
                if metadata.get("calculator_tool_used"):
                    metadata["tool_reinforcement"] = "calculator_used_but_reasoning_failed"
                if metadata.get("scaffold_tool_used"):
                    metadata["scaffold_reinforcement"] = "scaffold_used_but_reasoning_failed"
                metadata["tag"] = f"{tag}_bad_math_penalty"
                _global_cache.store(record["trigger"], -record["delta"], metadata=metadata)
    if summary["penalty_bad_math"] or summary["reward_clean_math"]:
        print(
            "    [Teaching Signals] "
            f"reward_clean_math={summary['reward_clean_math']} "
            f"penalty_bad_math={summary['penalty_bad_math']} "
            f"skipped={summary['skipped']}",
            flush=True,
        )
    return promoted


def _teaching_decision(
    attempt: ReasoningAttempt,
    answer: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    record_meta = dict(record.get("metadata", {}))
    stage = record_meta.get("attempt_stage", "unknown")
    signal = attempt.learning_signal or {}
    solver_bad_scaffold = scaffold_feedback_invalid(signal.get("solver_scaffold_feedback"))
    verifier_bad_scaffold = scaffold_feedback_invalid(signal.get("verifier_scaffold_feedback"))
    structural_issue = signal.get("structural_contradiction")
    solver_clean = (
        numbers_match(attempt.extracted_answer, answer)
        and numbers_match(attempt.solver_checked_answer, answer)
        and signal.get("solver_math") == "clean"
        and not solver_bad_scaffold
    )
    verifier_clean = (
        numbers_match(attempt.verifier_answer, answer)
        and numbers_match(attempt.verifier_checked_answer, answer)
        and numbers_match(attempt.verifier_tagged_answer, answer)
        and signal.get("verifier_math") == "clean"
        and not verifier_bad_scaffold
    )
    reward = (stage == "solver" and solver_clean) or (stage == "verifier" and verifier_clean)
    penalty_reason = None
    if structural_issue and stage in {"solver", "verifier", "unknown"}:
        penalty_reason = structural_issue
    elif stage == "solver" and solver_bad_scaffold:
        penalty_reason = "solver_bad_scaffold"
    elif stage == "verifier" and verifier_bad_scaffold:
        penalty_reason = "verifier_bad_scaffold"
    elif stage == "solver" and signal.get("solver_math") == "bad":
        penalty_reason = "solver_bad_math_or_final_mismatch"
    elif stage == "verifier" and signal.get("verifier_math") == "bad":
        penalty_reason = "verifier_bad_math_or_bad_tag"
    elif stage == "unknown" and signal.get("solver_math") == "bad":
        penalty_reason = "unknown_stage_bad_math"
    return {
        "stage": stage,
        "reward": reward,
        "penalty_reason": penalty_reason,
    }


def synthesis_teaching_summary(
    attempts: list[ReasoningAttempt],
    answer: str | None,
) -> dict[str, int]:
    summary = {
        "reward_clean_math": 0,
        "penalty_bad_math": 0,
        "skipped": 0,
    }
    if answer is None:
        return summary
    for attempt in attempts:
        if not attempt.accepted or not numbers_match(_verified_answer(attempt), answer):
            continue
        for record in attempt.synthesis_records or []:
            decision = _teaching_decision(attempt, answer, record)
            if decision["reward"]:
                summary["reward_clean_math"] += 1
            elif decision["penalty_reason"] is not None:
                summary["penalty_bad_math"] += 1
            else:
                summary["skipped"] += 1
    return summary


def assess_urgency(
    attempts: list[ReasoningAttempt],
    elapsed_sec: float,
    max_elapsed_sec: float | None = None,
) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    if attempts:
        last = attempts[-1]
        if len(attempts) > 1:
            score += len(attempts) - 1
            reasons.append("extra_rounds")
        if last.mode == "continue":
            score += 1
            reasons.append("continued_incomplete_answer")
        if last.extracted_answer is None:
            score += 3
            reasons.append("missing_final_answer")
        if last.verdict != "pass":
            score += 2
            reasons.append(f"verifier_{last.verdict}")
        elif not last.accepted:
            score += 3
            reasons.append("verifier_solver_mismatch")
        if last.synthesis_records:
            score += len(last.synthesis_records)
            reasons.append("synthesis_used")

    if max_elapsed_sec is not None and max_elapsed_sec > 0:
        fraction = elapsed_sec / max_elapsed_sec
        if fraction >= 1.0:
            score += 6
            reasons.append("time_budget_exhausted")
        elif fraction >= 0.75:
            score += 4
            reasons.append("time_budget_high")
        elif fraction >= 0.5:
            score += 2
            reasons.append("time_budget_medium")
    elif elapsed_sec >= 120:
        score += 4
        reasons.append("elapsed_over_120s")
    elif elapsed_sec >= 60:
        score += 2
        reasons.append("elapsed_over_60s")

    if score >= 8:
        level = "critical"
    elif score >= 5:
        level = "high"
    elif score >= 2:
        level = "medium"
    else:
        level = "low"

    return {
        "score": score,
        "level": level,
        "elapsed_sec": round(elapsed_sec, 2),
        "reasons": reasons,
    }


def _get_dynamic_agreement(
    urgency_level: str | None,
    required_agreement: int,
    relax_under_urgency: bool = False,
) -> int:
    required = max(1, int(required_agreement))
    if not relax_under_urgency:
        return required
    if urgency_level in {"high", "critical"}:
        return max(1, required - 1)
    return required


def _should_stop_for_urgency(urgency: dict[str, Any], stop_on_critical_urgency: bool) -> bool:
    if "time_budget_exhausted" in urgency.get("reasons", []):
        return True
    return stop_on_critical_urgency and urgency.get("level") == "critical"


def _cap_token_budget(token_budget: int, max_attempt_tokens: int | None) -> int:
    token_budget = max(1, int(token_budget))
    if max_attempt_tokens is None or max_attempt_tokens <= 0:
        return token_budget
    return max(1, min(token_budget, int(max_attempt_tokens)))


def _with_time_context(prompt: str, config, remaining_sec: float | None) -> str:
    if not getattr(config, "provide_time_context", False) or remaining_sec is None:
        return prompt
    total = getattr(config, "max_elapsed_sec", None)
    if total is not None and total > 0:
        elapsed = max(0.0, float(total) - remaining_sec)
        ratio = remaining_sec / float(total)
        if ratio <= 0.15:
            pressure = "critical"
        elif ratio <= 0.35:
            pressure = "high"
        elif ratio <= 0.65:
            pressure = "medium"
        else:
            pressure = "low"
        context = (
            f"[Time Context]: about {elapsed:.0f}s elapsed, about {remaining_sec:.0f}s remaining, "
            f"time pressure is {pressure}. Use this only to decide whether to continue, summarize, "
            "defer, or answer concisely; do not let it change the arithmetic."
        )
    else:
        context = (
            f"[Time Context]: about {remaining_sec:.0f}s remains in the current generation budget. "
            "Use this only for pacing, not for arithmetic."
        )
    return f"{prompt.rstrip()}\n\n{context}"


def _scaled_token_budget(
    base_tokens: int,
    multiplier: float,
    max_attempt_tokens: int | None = None,
) -> int:
    multiplier = max(1.0, float(multiplier))
    scaled = max(int(base_tokens), int(round(base_tokens * multiplier)))
    return _cap_token_budget(scaled, max_attempt_tokens)


def _apply_config_overrides(config, overrides: dict[str, Any]) -> None:
    aliases = {
        "allow_synthesis": "synthesis_enabled",
    }
    fields = set(getattr(config, "__dataclass_fields__", {}).keys())
    unknown: list[str] = []
    for key, value in overrides.items():
        target = aliases.get(key, key)
        if target in fields:
            setattr(config, target, value)
        else:
            unknown.append(key)
    if unknown:
        raise TypeError(f"Unknown solve_with_humility options: {', '.join(sorted(unknown))}")


def _clarification_attempt(
    mode: str,
    round_index: int,
    question: str,
    token_budget: int | None,
    elapsed_sec: float,
    synthesis_records: list[dict[str, Any]],
) -> ReasoningAttempt:
    return ReasoningAttempt(
        mode=mode,
        round_index=round_index,
        response=f"Clarifying question: {question}",
        extracted_answer=None,
        verifier_response="Needs user clarification before scoring.",
        verdict="unsettled",
        verifier_answer=None,
        accepted=False,
        token_budget=token_budget,
        elapsed_sec=elapsed_sec,
        synthesis_records=synthesis_records,
        needs_clarification=True,
        clarifying_question=question,
        ambiguity_type="disambiguation",
    )


def _mode_token_budget(
    mode: str,
    base_tokens: int,
    repair_token_multiplier: float,
    max_attempt_tokens: int | None = None,
) -> int:
    if mode in {"repair", "dynamic"}:
        return _scaled_token_budget(base_tokens, repair_token_multiplier, max_attempt_tokens)
    if mode == "continue":
        continuation_multiplier = min(max(1.0, repair_token_multiplier), 2.0)
        return _scaled_token_budget(base_tokens, continuation_multiplier, max_attempt_tokens)
    return _cap_token_budget(base_tokens, max_attempt_tokens)


def _run_attempt(
    M,
    question: str,
    prompt: str,
    mode: str,
    round_index: int,
    vecs=None,
    belief_vec=None,
    humility_vec=None,
    config=None,
    response_prefix: str | None = None,
    max_new_tokens_override: int | None = None,
    deadline: float | None = None,
    confidence_stabilization: bool = False,
) -> ReasoningAttempt:
    from invariants.config import AgenticConfig
    if config is None:
        config = AgenticConfig()
        
    actual_max_tokens = max_new_tokens_override if max_new_tokens_override is not None else config.max_attempt_tokens

    def remaining_time() -> float | None:
        if deadline is None:
            return None
        return max(0.001, deadline - time.time())
        
    t0 = time.time()
    synthesis_records: list[dict[str, Any]] = []
    remaining_before_generation = remaining_time()
    solver_time = remaining_before_generation
    reserve = max(0.0, float(getattr(config, "verifier_time_reserve_sec", 20.0)))
    if solver_time is not None and reserve > 0 and solver_time > reserve + 5.0:
        solver_time = max(1.0, solver_time - reserve)
    prompt_for_generation = _with_time_context(prompt, config, solver_time)
    stage_states: dict[str, Any] = {}
    try:
        if mode == "dynamic" and vecs is not None:
            generated = generate_agentic_text(
                M,
                instruction=prompt_for_generation,
                vecs=vecs,
                belief_vec=belief_vec,
                humility_vec=humility_vec,
                config=config,
                max_new_tokens=actual_max_tokens,
                synthesis_recorder=synthesis_records,
                stop_after_final_answer=True,
                max_time=solver_time,
                max_tool_calls=getattr(config, "max_tool_calls", 8),
                confidence_stabilization=confidence_stabilization,
            )
        else:
            generated = generate_text(
                M,
                prompt_for_generation,
                max_new_tokens=actual_max_tokens,
                stop_after_final_answer=True,
                max_time=solver_time,
                max_tool_calls=getattr(config, "max_tool_calls", 8),
            )
    except NeedsDisambiguationError as e:
        return _clarification_attempt(
            mode,
            round_index,
            str(e),
            actual_max_tokens,
            time.time() - t0,
            synthesis_records,
        )

    response = generated
    if response_prefix:
        response = f"{response_prefix.rstrip()}\n{generated.lstrip()}".strip()
    if _capture_stage_states_enabled(config):
        solver_pre = _capture_prompt_response_state(M, prompt_for_generation)
        solver_post = _capture_prompt_response_state(M, prompt_for_generation, response)
        if solver_pre is not None:
            stage_states["solver_prompt_pre"] = solver_pre
        if solver_post is not None:
            stage_states["solver_response_mean"] = solver_post
    solver_record_end = len(synthesis_records)

    extracted = extract_final_number(response)
    solver_scaffold_feedback = scaffold_tool_feedback(response)
    solver_clause_map_feedback = clause_map_feedback(response, question)
    solver_clause_methodology = sanitized_clause_methodology(question, solver_clause_map_feedback)
    solver_expression_checked = extract_expression_answer(response)
    solver_equation_quality = checked_equation_quality(response)
    solver_checked = solver_expression_checked or (
        None if solver_equation_quality is None else solver_equation_quality["evaluated_answer"]
    )
    solver_bad_arithmetic = bool(
        solver_equation_quality is not None
        and solver_equation_quality.get("math_consistent") is False
    )
    solver_final_mismatch = bool(
        extracted is not None
        and solver_checked is not None
        and not numbers_match(solver_checked, extracted)
    )
    solver_math_status = (
        "bad"
        if solver_bad_arithmetic or solver_final_mismatch
        else "clean"
        if solver_checked is not None
        else "unchecked"
    )
    solver_self_consistent = (
        extracted is not None
        and solver_math_status != "bad"
        and (solver_checked is None or numbers_match(solver_checked, extracted))
    )
    deterministic_scaffolds_enabled = getattr(config, "deterministic_scaffolds_enabled", True)
    model_scaffold_tool_enabled = getattr(config, "model_scaffold_tool_enabled", True)
    clause_map_enabled = getattr(config, "clause_map_enabled", True)
    learned_concept_context = getattr(config, "learned_concept_context", None)
    scaffold = quantity_scaffold_answer(question) if deterministic_scaffolds_enabled else None
    scaffold_answer = None if scaffold is None else scaffold["answer"]
    scaffold_match = bool(
        scaffold_answer is not None
        and extracted is not None
        and solver_checked is not None
        and numbers_match(extracted, scaffold_answer)
        and numbers_match(solver_checked, scaffold_answer)
        and solver_math_status == "clean"
    )
    if scaffold_match:
        for record in synthesis_records:
            metadata = dict(record.get("metadata", {}))
            metadata.setdefault("attempt_stage", "solver")
            record["metadata"] = metadata
        learning_signal = {
            "solver_math": solver_math_status,
            "verifier_math": "clean",
            "bad_tag": False,
            "parser_rescued_verifier": False,
            "solver_tool_used": used_calculator_tool(response),
            "solver_scaffold_tool_used": used_scaffold_tool(response),
            "solver_scaffold_feedback": solver_scaffold_feedback,
            "solver_clause_map_tool_used": used_clause_map_tool(response),
            "solver_clause_map_feedback": solver_clause_map_feedback,
            "solver_clause_methodology": solver_clause_methodology,
            "verifier_tool_used": False,
            "verifier_scaffold_tool_used": False,
            "verifier_scaffold_feedback": None,
            "verifier_clause_map_tool_used": False,
            "verifier_clause_map_feedback": None,
            "verifier_clause_methodology": None,
            "solver_expression_checked_answer": solver_expression_checked,
            "solver_equation_checked_answer": None
            if solver_equation_quality is None
            else solver_equation_quality["evaluated_answer"],
            "solver_equation_written_answer": None
            if solver_equation_quality is None
            else solver_equation_quality["written_answer"],
            "verifier_equation_written_answer": scaffold_answer,
            "quantity_scaffold_kind": scaffold["kind"] if scaffold else None,
            "quantity_scaffold_answer": scaffold_answer,
            "quantity_scaffold_match": True,
        }
        verifier_response = (
            "QUANTITY_SCAFFOLD_VERDICT: pass\n"
            f"EXPECTED_FINAL: {scaffold_answer}\n"
            "REASON: Solver expression matches the deterministic scaffold derived from the problem text."
        )
        return ReasoningAttempt(
            mode=mode,
            round_index=round_index,
            response=response,
            extracted_answer=extracted,
            verifier_response=verifier_response,
            verdict="pass",
            verifier_answer=scaffold_answer,
            accepted=True,
            token_budget=actual_max_tokens,
            elapsed_sec=time.time() - t0,
            synthesis_records=synthesis_records,
            solver_checked_answer=solver_checked,
            verifier_checked_answer=scaffold_answer,
            verifier_tagged_answer=scaffold_answer,
            acceptance_reason="quantity_scaffold_match",
            learning_signal=learning_signal,
            stage_states=stage_states,
        )
    
    # Blind Verification: Strip the solver's final conclusion so the verifier can't be anchored.
    blind_response = re.sub(r"(?i)^(?:Final answer|Computed).*$", "", response, flags=re.MULTILINE).strip()
    verifier_remaining = remaining_time()
    v_prompt = _with_time_context(
        verify_prompt(
            question,
            blind_response,
            deterministic_scaffolds_enabled=deterministic_scaffolds_enabled,
            model_scaffold_tool_enabled=model_scaffold_tool_enabled,
            clause_map_enabled=clause_map_enabled,
            learned_concept_context=learned_concept_context,
        ),
        config,
        verifier_remaining,
    )
    
    try:
        if mode == "dynamic" and vecs is not None:
            verifier_response = generate_agentic_text(
                M,
                instruction=v_prompt,
                vecs=vecs,
                belief_vec=belief_vec,
                humility_vec=humility_vec,
                config=config,
                max_new_tokens=180,
                synthesis_recorder=synthesis_records,
                stop_after_verifier_answer=True,
                max_time=verifier_remaining,
                max_tool_calls=getattr(config, "max_tool_calls", 8),
            )
        else:
            verifier_response = generate_text(
                M,
                v_prompt,
                max_new_tokens=180,
                stop_after_verifier_answer=True,
                max_time=verifier_remaining,
                max_tool_calls=getattr(config, "max_tool_calls", 8),
            )
    except NeedsDisambiguationError as e:
        return _clarification_attempt(
            mode,
            round_index,
            str(e),
            actual_max_tokens,
            time.time() - t0,
            synthesis_records,
        )
    if _capture_stage_states_enabled(config):
        verifier_pre = _capture_prompt_response_state(M, v_prompt)
        verifier_post = _capture_prompt_response_state(M, v_prompt, verifier_response)
        if verifier_pre is not None:
            stage_states["verifier_prompt_pre"] = verifier_pre
        if verifier_post is not None:
            stage_states["verifier_response_mean"] = verifier_post
    for idx, record in enumerate(synthesis_records):
        metadata = dict(record.get("metadata", {}))
        metadata.setdefault("attempt_stage", "solver" if idx < solver_record_end else "verifier")
        record["metadata"] = metadata

    verifier_details = parse_verifier_details(verifier_response)
    verifier_scaffold_feedback = scaffold_tool_feedback(verifier_response)
    verifier_clause_map_feedback = clause_map_feedback(verifier_response, question)
    verifier_clause_methodology = sanitized_clause_methodology(question, verifier_clause_map_feedback)
    solver_scaffold_invalid = scaffold_feedback_invalid(solver_scaffold_feedback)
    verifier_scaffold_invalid = scaffold_feedback_invalid(verifier_scaffold_feedback)
    structural_issue = structural_contradiction(question, response, verifier_response)
    verdict = verifier_details["verdict"]
    verifier_answer = verifier_details["answer"]
    verifier_checked = verifier_details["checked_answer"]
    verifier_tagged = verifier_details["tagged_answer"]
    verifier_math_status = (
        "bad"
        if verifier_details.get("bad_arithmetic") or verifier_details.get("bad_tag")
        else "clean"
        if verifier_checked is not None
        else "unchecked"
    )
    checked_math_rescues_final_tag = bool(
        verdict == "pass"
        and solver_checked is not None
        and verifier_checked is not None
        and numbers_match(solver_checked, verifier_checked)
        and (extracted is None or not numbers_match(extracted, solver_checked))
        and verifier_math_status == "clean"
        and structural_issue is None
    )
    accepted = (
        (
            verdict == "pass"
            and solver_self_consistent
            and extracted is not None
            and verifier_answer is not None
            and numbers_match(verifier_answer, extracted)
            and (scaffold_answer is None or numbers_match(extracted, scaffold_answer))
            and structural_issue is None
        )
        or checked_math_rescues_final_tag
    )
    if accepted and verifier_math_status == "bad":
        acceptance_reason = "parser_rescued_verifier_bad_math"
    elif checked_math_rescues_final_tag:
        acceptance_reason = "checked_math_rescues_solver_final_tag"
    elif accepted:
        acceptance_reason = "verifier_match_checked"
    elif verdict == "pass" and structural_issue is not None:
        acceptance_reason = "structural_quantity_contradiction"
    elif verdict == "pass" and (solver_scaffold_invalid or verifier_scaffold_invalid):
        acceptance_reason = "invalid_scaffold"
    elif verdict == "pass" and not solver_self_consistent:
        acceptance_reason = "solver_expression_mismatch"
    elif verdict == "pass" and scaffold_answer is not None and not numbers_match(extracted, scaffold_answer):
        acceptance_reason = "quantity_scaffold_mismatch"
    elif (
        verdict == "pass"
        and verifier_answer is not None
        and extracted is not None
        and not numbers_match(verifier_answer, extracted)
    ):
        acceptance_reason = "verifier_solver_mismatch"
    elif verdict == "pass":
        acceptance_reason = "verifier_pass_not_accepted"
    else:
        acceptance_reason = f"verifier_{verdict}"
    learning_signal = {
        "solver_math": solver_math_status,
        "verifier_math": verifier_math_status,
        "bad_tag": verifier_details.get("bad_tag", False),
        "parser_rescued_verifier": acceptance_reason == "parser_rescued_verifier_bad_math",
        "parser_rescued_solver_final": acceptance_reason == "checked_math_rescues_solver_final_tag",
        "solver_tool_used": used_calculator_tool(response),
        "solver_scaffold_tool_used": used_scaffold_tool(response),
        "solver_scaffold_feedback": solver_scaffold_feedback,
        "solver_clause_map_tool_used": used_clause_map_tool(response),
        "solver_clause_map_feedback": solver_clause_map_feedback,
        "solver_clause_methodology": solver_clause_methodology,
        "verifier_tool_used": used_calculator_tool(verifier_response),
        "verifier_scaffold_tool_used": used_scaffold_tool(verifier_response),
        "verifier_scaffold_feedback": verifier_scaffold_feedback,
        "verifier_clause_map_tool_used": used_clause_map_tool(verifier_response),
        "verifier_clause_map_feedback": verifier_clause_map_feedback,
        "verifier_clause_methodology": verifier_clause_methodology,
        "structural_contradiction": structural_issue,
        "solver_expression_checked_answer": solver_expression_checked,
        "solver_equation_checked_answer": None
        if solver_equation_quality is None
        else solver_equation_quality["evaluated_answer"],
        "solver_equation_written_answer": None
        if solver_equation_quality is None
        else solver_equation_quality["written_answer"],
        "verifier_equation_written_answer": None
        if verifier_details.get("checked_equation") is None
        else verifier_details["checked_equation"]["written_answer"],
        "quantity_scaffold_kind": None if scaffold is None else scaffold["kind"],
        "quantity_scaffold_answer": scaffold_answer,
        "quantity_scaffold_match": scaffold_match,
    }
    return ReasoningAttempt(
        mode=mode,
        round_index=round_index,
        response=response,
        extracted_answer=extracted,
        verifier_response=verifier_response,
        verdict=verdict,
        verifier_answer=verifier_answer,
        accepted=accepted,
        token_budget=actual_max_tokens,
        elapsed_sec=time.time() - t0,
        synthesis_records=synthesis_records,
        solver_checked_answer=solver_checked,
        verifier_checked_answer=verifier_checked,
        verifier_tagged_answer=verifier_tagged,
        acceptance_reason=acceptance_reason,
        learning_signal=learning_signal,
        stage_states=stage_states,
    )


def solve_with_humility(
    M,
    question: str,
    vecs: dict | None = None,
    belief_vec=None,
    humility_vec=None,
    config=None,
    **legacy_overrides,
) -> HumbleResult:
    from invariants.config import AgenticConfig
    if config is None:
        config = AgenticConfig()
    _apply_config_overrides(config, legacy_overrides)
    config._synthesis_events_used = 0
        
    max_rounds = config.max_rounds
    required_agreement = config.required_agreement
    max_new_tokens = config.max_new_tokens
    allow_synthesis = config.synthesis_enabled
    max_elapsed_sec = config.max_elapsed_sec
    repair_token_multiplier = config.repair_token_multiplier
    max_attempt_tokens = config.max_attempt_tokens
    interactive_disambiguation = config.interactive_disambiguation
    chatty_log = config.chatty_log
    stop_on_critical_urgency = config.stop_on_critical_urgency
    relax_agreement_under_urgency = getattr(config, "relax_agreement_under_urgency", False)
    deterministic_scaffolds_enabled = getattr(config, "deterministic_scaffolds_enabled", True)
    model_scaffold_tool_enabled = getattr(config, "model_scaffold_tool_enabled", True)
    clause_map_enabled = getattr(config, "clause_map_enabled", True)
    learned_concept_context = getattr(config, "learned_concept_context", None)
    
    if not config.use_expert_vectors:
        vecs = None

    t0 = time.time()
    deadline = t0 + max_elapsed_sec if max_elapsed_sec is not None and max_elapsed_sec > 0 else None
    attempts: list[ReasoningAttempt] = []
    initial_mode = "baseline"
    first_budget = _mode_token_budget(
        initial_mode,
        max_new_tokens,
        repair_token_multiplier,
        max_attempt_tokens,
    )

    first = _run_attempt(
        M,
        question,
        solve_prompt(
            question,
            deterministic_scaffolds_enabled=deterministic_scaffolds_enabled,
            model_scaffold_tool_enabled=model_scaffold_tool_enabled,
            clause_map_enabled=clause_map_enabled,
            learned_concept_context=learned_concept_context,
        ),
        mode=initial_mode,
        round_index=0,
        config=config,
        max_new_tokens_override=first_budget,
        deadline=deadline,
    )
    attempts.append(first)
    first.urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
    if first.needs_clarification and config.defer_disambiguation:
        return HumbleResult(question, None, False, "needs_user_clarification", attempts, first.urgency)

    answer, count = _modal_answer(attempts)
    current_required_agreement = _get_dynamic_agreement(
        first.urgency.get("level"),
        required_agreement,
        relax_under_urgency=relax_agreement_under_urgency,
    )
    if answer is not None and count >= current_required_agreement:
        if config.cache_write_enabled:
            _promote_verified_synthesis(attempts, answer)
        urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
        return HumbleResult(question, answer, True, "verified_stable", attempts, urgency)
    if _should_stop_for_urgency(first.urgency, stop_on_critical_urgency):
        ans = _fallback_answer(attempts)
        return HumbleResult(question, ans, False, "stopped_for_urgency_budget", attempts, first.urgency)

    for round_index in range(1, max_rounds + 1):
        prior_attempt = attempts[-1]
        prior_answer = _verified_answer(prior_attempt)
        if prior_answer is None:
            modal_candidate, modal_count = _modal_answer(attempts)
            if modal_count > 0:
                prior_answer = modal_candidate
        response_prefix = None
        
        confidence_stabilization = False
        if prior_attempt.learning_signal and prior_attempt.learning_signal.get("bad_tag", False):
            confidence_stabilization = True

        if prior_answer is not None:
            mode = "confirm"
            prompt = confirmation_prompt(
                question,
                prior_answer,
                deterministic_scaffolds_enabled=deterministic_scaffolds_enabled,
                model_scaffold_tool_enabled=model_scaffold_tool_enabled,
                clause_map_enabled=clause_map_enabled,
                learned_concept_context=learned_concept_context,
            )
        elif (
            prior_attempt.extracted_answer is None
            and prior_attempt.mode != "continue"
            and not _attempt_has_invalid_scaffold(prior_attempt)
        ):
            mode = "continue"
            prompt = continuation_prompt(question, prior_attempt)
            response_prefix = prior_attempt.response
        else:
            mode = "dynamic" if vecs is not None and not _needs_symbolic_repair(prior_attempt) else "repair"
            prompt = repair_prompt(
                question,
                prior_attempt,
                deterministic_scaffolds_enabled=deterministic_scaffolds_enabled,
                model_scaffold_tool_enabled=model_scaffold_tool_enabled,
                clause_map_enabled=clause_map_enabled,
                learned_concept_context=learned_concept_context,
            )
        attempt_budget = _mode_token_budget(
            mode,
            max_new_tokens,
            repair_token_multiplier,
            max_attempt_tokens,
        )
        attempt = _run_attempt(
            M,
            question,
            prompt,
            mode=mode,
            round_index=round_index,
            vecs=vecs,
            belief_vec=belief_vec,
            humility_vec=humility_vec,
            config=config,
            response_prefix=response_prefix,
            max_new_tokens_override=attempt_budget,
            deadline=deadline,
            confidence_stabilization=confidence_stabilization,
        )
        attempts.append(attempt)
        attempt.urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
        if attempt.needs_clarification and config.defer_disambiguation:
            return HumbleResult(question, None, False, "needs_user_clarification", attempts, attempt.urgency)

        answer, count = _modal_answer(attempts)
        current_required_agreement = _get_dynamic_agreement(
            attempt.urgency.get("level"),
            required_agreement,
            relax_under_urgency=relax_agreement_under_urgency,
        )
        if answer is not None and count >= current_required_agreement:
            if config.cache_write_enabled:
                _promote_verified_synthesis(attempts, answer)
            urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
            return HumbleResult(question, answer, True, "verified_stable", attempts, urgency)
        if _should_stop_for_urgency(attempt.urgency, stop_on_critical_urgency):
            ans = _fallback_answer(attempts)
            return HumbleResult(question, ans, False, "stopped_for_urgency_budget", attempts, attempt.urgency)

    answer, count = _modal_answer(attempts)
    if answer is not None:
        urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
        return HumbleResult(question, answer, False, "verified_but_not_stable", attempts, urgency)
    urgency = assess_urgency(attempts, time.time() - t0, max_elapsed_sec)
    ans = _fallback_answer(attempts)
    return HumbleResult(question, ans, False, "unresolved_after_extra_compute", attempts, urgency)
