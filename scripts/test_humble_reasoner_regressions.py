import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from invariants.config import AgenticConfig
from invariants.controller_benchmark import is_correct
from invariants import humble_reasoner as h
from invariants.tool_utils import evaluate_python_expression, intercept_tool_call, validate_quantity_scaffold
from scripts import evaluate_humble_full_suite as suite


def make_attempt(answer="70000", verifier_tag="700", verifier_math="bad"):
    return h.ReasoningAttempt(
        mode="dynamic",
        round_index=0,
        response=(
            "Asked quantity: profit\n"
            "Expression: 200000 - 80000 - 50000\n"
            "Computed: 70000\n"
            f"Final answer: {answer}"
        ),
        extracted_answer=answer,
        verifier_response="",
        verdict="pass",
        verifier_answer=answer,
        accepted=True,
        solver_checked_answer=answer,
        verifier_checked_answer=answer,
        verifier_tagged_answer=verifier_tag,
        acceptance_reason="parser_rescued_verifier_bad_math",
        learning_signal={
            "solver_math": "clean",
            "verifier_math": verifier_math,
            "parser_rescued_verifier": verifier_math == "bad",
            "solver_scaffold_tool_used": False,
            "solver_scaffold_feedback": None,
            "verifier_scaffold_tool_used": False,
            "verifier_scaffold_feedback": None,
        },
        synthesis_records=[
            {
                "trigger": torch.ones(2),
                "delta": torch.ones(2),
                "metadata": {"attempt_stage": "solver"},
            },
            {
                "trigger": torch.ones(2),
                "delta": torch.ones(2) * 2,
                "metadata": {"attempt_stage": "verifier"},
            },
        ],
    )


def capture_cache_stores(fn):
    stores = []
    old_store = h._global_cache.store
    h._global_cache.store = lambda trigger, delta, metadata=None: stores.append(
        (delta.detach().cpu().clone(), dict(metadata or {}))
    )
    try:
        result = fn()
    finally:
        h._global_cache.store = old_store
    return result, stores


def test_controller_scorer_accepts_microscopic_roundoff_only():
    ok, pred, gold = is_correct("Final answer: 17.999999999999996", "#### 18")
    assert ok is True
    assert str(pred) == "17.999999999999996"
    assert str(gold) == "18"

    ok, pred, gold = is_correct("Final answer: 17.9", "#### 18")
    assert ok is False
    assert str(pred) == "17.9"
    assert str(gold) == "18"


def test_parser_rescues_bad_tag_without_hiding_it():
    verifier = (
        "INDEPENDENT_CALCULATION:\n"
        "Profit: $200,000 - $80,000 - $50,000 = $70,000\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 700\n"
        "REASON: none"
    )
    details = h.parse_verifier_details(verifier)
    assert details["answer"] == "70000"
    assert details["checked_answer"] == "70000"
    assert details["tagged_answer"] == "700"
    assert details["bad_tag"] is True
    assert details["bad_arithmetic"] is False


def test_parser_rescues_bad_arithmetic_claim_without_hiding_it():
    verifier = (
        "INDEPENDENT_CALCULATION:\n"
        "Total time taken is 40 + 20 + (200 / 2) = 120\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 160\n"
        "REASON: none"
    )
    details = h.parse_verifier_details(verifier)
    assert details["answer"] == "160"
    assert details["checked_answer"] == "160"
    assert details["tagged_answer"] == "160"
    assert details["bad_arithmetic"] is True


def test_parser_evaluates_final_cue_expression_without_rhs():
    verifier = (
        "INDEPENDENT_CALCULATION:\n"
        "Value increase = 150% of 80000 = 120000\n"
        "Profit = 80000 + 120000 - 80000 - 50000\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 50000\n"
        "REASON: none"
    )
    details = h.parse_verifier_details(verifier)
    assert details["answer"] == "70000"
    assert details["checked_answer"] == "70000"
    assert details["tagged_answer"] == "50000"
    assert details["bad_tag"] is True
    assert details["bad_arithmetic"] is False


def test_parser_prefers_underscore_total_over_intermediate_price():
    verifier = (
        "INDEPENDENT_CALCULATION:\n"
        "price = 5 dollars (regular price)\n"
        "discounted_price = 5 * 0.6 = 3 dollars (price of every second mug)\n"
        "total_cost = 8 * 5 + 8 * 3 = 64 dollars\n"
        "total_cost = 64 dollars\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 64\n"
        "REASON: none"
    )
    details = h.parse_verifier_details(verifier)
    assert details["answer"] == "64"
    assert details["checked_answer"] == "64"
    assert details["tagged_answer"] == "64"
    assert details["bad_tag"] is False
    assert details["bad_arithmetic"] is False


def test_parser_does_not_let_intermediate_remaining_time_override_final_tag():
    verifier = (
        "INDEPENDENT_CALCULATION:\n"
        "Distance covered while driving to the point of return = 60 * 3 = 180\n"
        "Remaining time to cover the distance = 4 - 2 - 0.5 = 1.5\n"
        "Distance covered while driving at 80 mph = 80 * 1.5 = 120\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 315\n"
        "REASON: none"
    )
    details = h.parse_verifier_details(verifier)
    assert details["answer"] == "315"
    assert details["tagged_answer"] == "315"
    assert details["bad_tag"] is False


def test_parser_prefers_total_earnings_over_overtime_rate_intermediate():
    verifier = (
        "INDEPENDENT_CALCULATION:\n"
        "Eliza's earnings for the first 40 hours = 10 * 40 = 400\n"
        "Overtime pay per hour = 1.2 * 10 = 12\n"
        "Total earnings = 400 + 60 = 460\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 460\n"
        "REASON: none"
    )
    details = h.parse_verifier_details(verifier)
    assert details["answer"] == "460"
    assert details["checked_answer"] == "460"
    assert details["bad_tag"] is False


def test_parser_accepts_loose_checked_equation_when_it_matches_verifier_tag():
    verifier = (
        "INDEPENDENT_CALCULATION:\n"
        "F = 16 / 2 = 8\n"
        "D = 16 - 8 = 8\n"
        "C = 8 * 5 + 8 * 3\n"
        "C = 64\n"
        "VERDICT: unsettled\n"
        "INDEPENDENT_FINAL: 64\n"
        "REASON: proposed solution double-counts the discounted glasses"
    )
    details = h.parse_verifier_details(verifier)
    assert details["answer"] == "64"
    assert details["checked_answer"] == "64"
    assert details["tagged_answer"] == "64"
    assert details["bad_arithmetic"] is False
    assert details["bad_tag"] is False


def test_expression_parser_prefers_expression_matching_final_answer():
    response = (
        "Asked quantity: maximum realized profit from one chosen purchase plan.\n"
        "Expression: 5000 * 2.5 / 100\n"
        "Computed: <<CALC: 5000 * 2.5 / 100>> = 125.0\n"
        "Expression: 8000 * 1.2 / 100\n"
        "Computed: <<CALC: 8000 * 1.2 / 100>> = 96.0\n"
        "Final answer: 125"
    )
    assert h.extract_expression_answer(response) == "125"


def test_expression_parser_accepts_leading_whitespace():
    response = (
        " Asked quantity: money made\n"
        " Expression: (40 - 6 - 4) * 2\n"
        " Computed: <<CALC: (40 - 6 - 4) * 2>> = 60\n"
        " Final answer: 60"
    )
    assert h.extract_expression_answer(response) == "60"


def test_expression_parser_accepts_computed_arithmetic_line():
    response = (
        "Asked quantity: total cost\n"
        "Computed: 5 * 8 + 5 * 8 * 0.6\n"
        "Final answer: 64"
    )
    assert h.extract_expression_answer(response) == "64"


def test_financial_profit_scaffold_binds_requested_quantity():
    question = (
        "Josh decides to try flipping a house. He buys a house for $80,000 "
        "and then puts in $50,000 in repairs. This increased the value of "
        "the house by 150%. How much profit did he make?"
    )
    scaffold = h.financial_profit_scaffold(question)
    assert scaffold is not None
    assert scaffold["base_value"] == "80000"
    assert scaffold["repair_cost"] == "50000"
    assert scaffold["percent_increase"] == "150"
    assert scaffold["value_increase"] == "120000"
    assert scaffold["final_value"] == "200000"
    assert scaffold["total_cost"] == "130000"
    assert scaffold["profit"] == "70000"
    assert scaffold["expression"] == "(80000 + (80000 * 150 / 100)) - 80000 - 50000"

    context = h.quantity_tool_context(question)
    assert "Expression for requested profit" in context
    assert "(80000 + (80000 * 150 / 100)) - 80000 - 50000" in context


def test_choice_profit_scaffold_selects_best_realized_profit():
    question = (
        "A merchant wants to make a choice of purchase between 2 purchase plans: "
        "jewelry worth $5,000 or electronic gadgets worth $8,000. His financial "
        "advisor speculates that the jewelry market will go up 2.5% while the "
        "electronic gadgets market will rise 1.2% within the same month. If the "
        "merchant is looking to maximize profit at the end of this month by "
        "making a choice, how much profit would this be?"
    )
    scaffold = h.choice_profit_scaffold(question)
    assert scaffold is not None
    assert scaffold["options"] == "jewelry=5000 at 2.5% -> 125; electronic gadgets=8000 at 1.2% -> 96"
    assert scaffold["best_option"] == "jewelry"
    assert scaffold["best_profit"] == "125"
    assert scaffold["expression"] == "5000 * 2.5 / 100"

    answer = h.quantity_scaffold_answer(question)
    assert answer == {
        "kind": "choice_max_realized_profit",
        "answer": "125",
        "expression": "5000 * 2.5 / 100",
    }

    context = h.quantity_tool_context(question)
    assert "Candidate realized profits" in context
    assert "Expression for chosen profit: 5000 * 2.5 / 100." in context


def test_periodic_discount_scaffold_counts_every_second_items():
    question = (
        "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, "
        "but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. "
        "How much does he need to pay for them?"
    )
    scaffold = h.periodic_discount_scaffold(question)
    assert scaffold is not None
    assert scaffold["period"] == "2"
    assert scaffold["full_price_count"] == "8"
    assert scaffold["discounted_count"] == "8"
    assert scaffold["discounted_price"] == "3"
    assert scaffold["total_cost"] == "64"
    assert scaffold["expression"] == "(16 - floor(16 / 2)) * 5 + floor(16 / 2) * (60 * 5 / 100)"

    answer = h.quantity_scaffold_answer(question)
    assert answer == {
        "kind": "periodic_discount_total_cost",
        "answer": "64",
        "expression": "(16 - floor(16 / 2)) * 5 + floor(16 / 2) * (60 * 5 / 100)",
    }

    context = h.quantity_tool_context(question)
    assert "discounted items: 8 and full-price items: 8" in context


def test_solve_prompt_can_disable_deterministic_scaffold_context():
    question = (
        "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, "
        "but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. "
        "How much does he need to pay for them?"
    )
    enabled_prompt = h.solve_prompt(question, deterministic_scaffolds_enabled=True)
    disabled_prompt = h.solve_prompt(question, deterministic_scaffolds_enabled=False)

    assert "Available quantity scaffold" in enabled_prompt
    assert "Available quantity scaffold" not in disabled_prompt
    assert "<<SCAFFOLD:" in disabled_prompt
    assert "<<CLAUSEMAP:" not in disabled_prompt
    clause_prompt = h.solve_prompt(
        question,
        deterministic_scaffolds_enabled=False,
        clause_map_enabled=True,
    )
    assert "<<CLAUSEMAP:" in clause_prompt
    assert "[C1]" in clause_prompt


def test_remainder_sale_revenue_scaffold_tracks_units():
    question = (
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every "
        "morning and bakes muffins for her friends every day with four. She "
        "sells the remainder at the farmers' market daily for $2 per fresh duck "
        "egg. How much in dollars does she make every day at the farmers' market?"
    )
    scaffold = h.remainder_sale_revenue_scaffold(question)
    assert scaffold is not None
    assert scaffold["produced"] == "16"
    assert scaffold["personal_use"] == "3 + 4"
    assert scaffold["remainder"] == "9"
    assert scaffold["sale_price"] == "2"
    assert scaffold["revenue"] == "18"
    assert scaffold["expression"] == "(16 - 3 - 4) * 2"

    context = h.quantity_tool_context(question)
    assert "Expression for daily dollars earned: (16 - 3 - 4) * 2." in context


def test_daily_split_scaffold_binds_per_day_not_per_meal():
    question = (
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, "
        "containing seeds, mealworms and vegetables to help keep them healthy. She gives "
        "the chickens their feed in three separate meals. In the morning, she gives her "
        "flock of chickens 15 cups of feed. In the afternoon, she gives her chickens "
        "another 25 cups of feed. How many cups of feed does she need to give her chickens "
        "in the final meal of the day if the size of Wendi's flock is 20 chickens?"
    )
    scaffold = h.daily_split_quantity_scaffold(question)
    assert scaffold is not None
    assert scaffold["per_entity_daily"] == "3"
    assert scaffold["entity_count"] == "20"
    assert scaffold["daily_total"] == "60"
    assert scaffold["known_total"] == "40"
    assert scaffold["final_meal"] == "20"
    assert scaffold["expression"] == "(3 * 20) - (15 + 25)"

    context = h.quantity_tool_context(question)
    assert "Per-entity daily amount: 3 cups." in context
    assert "Expression for requested final meal: (3 * 20) - (15 + 25)." in context


def test_restart_download_scaffold_binds_restart_from_beginning():
    question = (
        "Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, "
        "but 40% of the way through the download, Windows forces a restart to install updates, "
        "which takes 20 minutes. Then Carla has to restart the download from the beginning. "
        "How long does it take to download the file?"
    )
    scaffold = h.restart_download_scaffold(question)
    assert scaffold is not None
    assert scaffold["lost_gb"] == "80"
    assert scaffold["lost_time"] == "40"
    assert scaffold["full_download_time"] == "100"
    assert scaffold["total_time"] == "160"
    assert scaffold["expression"] == "((200 * 40 / 100) / 2) + 20 + (200 / 2)"


def test_return_trip_distance_scaffold_subtracts_return_progress():
    question = (
        "John drives for 3 hours at a speed of 60 mph and then turns around because "
        "he realizes he forgot something very important at home. He tries to get home "
        "in 4 hours but spends the first 2 hours in standstill traffic. He spends the "
        "next half-hour driving at a speed of 30mph, before being able to drive the "
        "remaining time of the 4 hours going at 80 mph. How far is he from home at "
        "the end of those 4 hours?"
    )
    scaffold = h.return_trip_distance_scaffold(question)
    assert scaffold is not None
    assert scaffold["outbound_distance"] == "180"
    assert scaffold["return_distance"] == "135"
    assert scaffold["distance_from_home"] == "45"
    assert scaffold["remaining_return_time"] == "1.5"


def test_break_even_year_scaffold_requires_strict_positive_profit():
    question = (
        "Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year "
        "it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year "
        "to water and feed the tree. How many years will it take before he starts "
        "earning money on the lemon tree?"
    )
    scaffold = h.break_even_year_scaffold(question)
    assert scaffold is not None
    assert scaffold["annual_gross"] == "10.5"
    assert scaffold["annual_net"] == "7.5"
    assert scaffold["break_even_years"] == "12"
    assert scaffold["first_profitable_year"] == "13"


def test_overtime_pay_scaffold_binds_regular_and_overtime_pay():
    question = (
        "Eliza's rate per hour for the first 40 hours she works each week is $10. "
        "She also receives an overtime pay of 1.2 times her regular hourly rate. "
        "If Eliza worked for 45 hours this week, how much are her earnings for this week?"
    )
    scaffold = h.overtime_pay_scaffold(question)
    assert scaffold is not None
    assert scaffold["regular_pay"] == "400"
    assert scaffold["overtime_hours"] == "5"
    assert scaffold["overtime_rate"] == "12"
    assert scaffold["overtime_pay"] == "60"
    assert scaffold["total_pay"] == "460"


def test_monthly_percentage_total_scaffold_tracks_reduced_third_month():
    question = (
        "A new program had 60 downloads in the first month. The number of downloads "
        "in the second month was three times as many as the downloads in the first "
        "month, but then reduced by 30% in the third month. How many downloads did "
        "the program have total over the three months?"
    )
    scaffold = h.monthly_percentage_total_scaffold(question)
    assert scaffold is not None
    assert scaffold["first_month"] == "60"
    assert scaffold["second_month"] == "180"
    assert scaffold["third_month"] == "126"
    assert scaffold["total"] == "366"


def test_dozen_total_cost_scaffold_adds_per_dozen_line_items():
    question = (
        "Toula went to the bakery and bought various types of pastries. She bought "
        "3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost "
        "$80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much "
        "was the total cost?"
    )
    scaffold = h.dozen_total_cost_scaffold(question)
    assert scaffold is not None
    assert scaffold["line_items"] == "3 * 68, 2 * 80, 6 * 55"
    assert scaffold["total"] == "694"


def test_reverse_fraction_sales_scaffold_solves_backward_inventory():
    question = (
        "Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners "
        "at the green house, 2 more to the red house, and half of what was left at the "
        "orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?"
    )
    scaffold = h.reverse_fraction_sales_scaffold(question)
    assert scaffold is not None
    assert scaffold["before_orange"] == "10"
    assert scaffold["before_red"] == "12"
    assert scaffold["initial"] == "18"


def test_remaining_percentage_scaffold_tracks_percentage_of_remainder():
    question = (
        "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% "
        "of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop "
        "dance. What percentage of the entire students enrolled in hip-hop dance?"
    )
    scaffold = h.remaining_percentage_scaffold(question)
    assert scaffold is not None
    assert scaffold["remaining_after_first"] == "80"
    assert scaffold["second_pct_of_total"] == "20"
    assert scaffold["rest_pct"] == "60"


def test_floor_is_available_to_parser_and_runtime_calculator():
    assert h._safe_eval_arithmetic("floor(90 / 7.5) + 1") == "13"
    assert evaluate_python_expression("floor(90 / 7.5) + 1") == "13"
    assert h._safe_eval_arithmetic("max(5000 * 2.5 / 100, 8000 * 1.2 / 100)") == "125"
    assert evaluate_python_expression("max(5000 * 2.5 / 100, 8000 * 1.2 / 100)") == "125.0"


def test_model_authored_scaffold_tool_checks_units():
    scaffold = (
        "target=dollars/day; produced=16 eggs/day; eaten=3 eggs/day; "
        "baked=4 eggs/day; price=2 dollars/egg; "
        "expression=(produced - eaten - baked) * price"
    )
    result = validate_quantity_scaffold(scaffold)
    assert "valid=True" in result
    assert "value=18" in result
    assert "unit=dollars/days" in result

    tool_expr = intercept_tool_call(f"Scaffold: <<SCAFFOLD: {scaffold}>>")
    assert tool_expr == f"SCAFFOLD: {scaffold}"
    assert evaluate_python_expression(tool_expr) == result


def test_model_authored_scaffold_tool_rejects_unit_mixing():
    scaffold = (
        "target=dollars/day; produced=16 eggs/day; eaten=3 eggs/day; "
        "baked=4 eggs/day; price=2 dollars/egg; "
        "expression=produced - eaten - baked + price"
    )
    result = validate_quantity_scaffold(scaffold)
    assert "valid=False" in result
    assert "unit mismatch" in result


def test_model_authored_scaffold_handles_singular_plural_unit_aliases():
    scaffold = (
        "target=dollars; full=8 glasses; discounted=8 glasses; "
        "regular_price=5 dollars/glass; discount_rate=0.6; "
        "expression=full * regular_price + discounted * regular_price * discount_rate"
    )
    result = validate_quantity_scaffold(scaffold)
    assert "valid=True" in result
    assert "value=64" in result
    assert "unit=dollars" in result


def test_clause_parser_splits_periodic_discount_problem():
    clauses = h.question_clauses(
        "A mug costs $5, but every second mug costs only 60% of the regular price. "
        "Sam buys 16 mugs. How much does Sam pay in total?"
    )
    assert clauses == [
        "A mug costs $5",
        "every second mug costs only 60% of the regular price",
        "Sam buys 16 mugs",
        "How much does Sam pay in total",
    ]


def test_clause_map_tool_validates_role_binding():
    feedback = h.clause_map_feedback(
        "<<CLAUSEMAP: asked=C4; givens=C1,C3; rules=C2; operations=C2,C3; ignored=none>> "
        "= valid=True; covered=C1,C2,C3,C4; asked=C4; givens=C1,C3; rules=C2; operations=C2,C3"
    )
    assert "valid=True" in feedback
    assert "asked=C4" in feedback


def test_clause_map_feedback_marks_missing_clause():
    question = (
        "A mug costs $5, but every second mug costs only 60% of the regular price. "
        "Sam buys 16 mugs. How much does Sam pay in total?"
    )
    feedback = h.clause_map_feedback(
        "<<CLAUSEMAP: asked=C4; givens=C1; rules=C2; ignored=none>> "
        "= valid=True; covered=C1,C2,C4; asked=C4; givens=C1; rules=C2",
        question,
    )
    assert "missing=C3" in feedback


def test_clause_methodology_is_sanitized_for_reuse():
    question = (
        "A mug costs $5, but every second mug costs only 60% of the regular price. "
        "Sam buys 16 mugs. How much does Sam pay in total?"
    )
    methodology = h.sanitized_clause_methodology(
        question,
        "valid=True; covered=C1,C2,C3,C4; asked=C4; givens=C1,C3; rules=C2; operations=C2,C3",
    )
    payload = repr(methodology).lower()
    assert methodology["kind"] == "periodic_discount_partition"
    assert methodology["privacy"]["raw_clauses_saved"] is False
    assert methodology["privacy"]["source_numbers_saved"] is False
    assert "mug" not in payload
    assert "sam" not in payload
    assert "c1" not in payload
    assert "$5" not in payload
    assert "16" not in payload
    assert "60" not in payload


def test_variable_equations_are_available_to_parser_and_runtime_calculator():
    equation = "(2/3 * x - 2) / 2 = 5"
    assert h._safe_eval_arithmetic(equation) == "18"
    assert evaluate_python_expression(equation) == "18"
    assert evaluate_python_expression("(2/3 * x - 2) / 2 == 5") == "18"
    assert evaluate_python_expression("2 + 2 = 4") == "4"


def test_near_integer_calculator_result_can_match_scaffold_answer():
    question = (
        "Melanie is a door-to-door saleswoman. She sold a third of her vacuum cleaners "
        "at the green house, 2 more to the red house, and half of what was left at the "
        "orange house. If Melanie has 5 vacuum cleaners left, how many did she start with?"
    )
    solver_response = (
        "Asked quantity: initial inventory\n"
        "Expression: ((5 / (1 - 1/2)) + 2) / (1 - 1/3)\n"
        "Computed: <<CALC: ((5 / (1 - 1/2)) + 2) / (1 - 1/3)>> = 17.999999999999996\n"
        "Final answer: 18"
    )
    calls = []
    old_generate_text = h.generate_text
    h.generate_text = lambda *args, **kwargs: calls.append(args) or solver_response
    try:
        attempt = h._run_attempt(
            None,
            question,
            h.solve_prompt(question),
            mode="baseline",
            round_index=0,
            config=AgenticConfig(max_attempt_tokens=100),
        )
    finally:
        h.generate_text = old_generate_text

    assert len(calls) == 1
    assert attempt.accepted is True
    assert attempt.acceptance_reason == "quantity_scaffold_match"
    assert attempt.verifier_answer == "18"


def test_quantity_scaffold_can_verify_checked_solver_expression_without_neural_verifier():
    question = (
        "John drives for 3 hours at a speed of 60 mph and then turns around because "
        "he realizes he forgot something very important at home. He tries to get home "
        "in 4 hours but spends the first 2 hours in standstill traffic. He spends the "
        "next half-hour driving at a speed of 30mph, before being able to drive the "
        "remaining time of the 4 hours going at 80 mph. How far is he from home at "
        "the end of those 4 hours?"
    )
    solver_response = (
        "Asked quantity: Distance from home\n"
        "Expression: (3 * 60) - ((2 * 0) + (0.5 * 30) + ((4 - 2 - 0.5) * 80))\n"
        "Computed: <<CALC: (3 * 60) - ((2 * 0) + (0.5 * 30) + ((4 - 2 - 0.5) * 80))>> = 45\n"
        "Final answer: 45"
    )
    calls = []
    old_generate_text = h.generate_text
    h.generate_text = lambda *args, **kwargs: calls.append(args) or solver_response
    try:
        attempt = h._run_attempt(
            None,
            question,
            h.solve_prompt(question),
            mode="baseline",
            round_index=0,
            config=AgenticConfig(max_attempt_tokens=100),
        )
    finally:
        h.generate_text = old_generate_text

    assert len(calls) == 1
    assert attempt.accepted is True
    assert attempt.acceptance_reason == "quantity_scaffold_match"
    assert attempt.verifier_answer == "45"
    assert attempt.learning_signal["quantity_scaffold_match"] is True


def test_urgency_does_not_relax_agreement_without_explicit_control_flag():
    assert h._get_dynamic_agreement("critical", 2) == 2
    assert h._get_dynamic_agreement("high", 3) == 3
    assert h._get_dynamic_agreement("critical", 2, relax_under_urgency=True) == 1
    assert h._get_dynamic_agreement("high", 3, relax_under_urgency=True) == 2


def test_independent_verifier_support_can_stabilize_later_clean_solver():
    first = make_attempt(answer="4", verifier_tag="3", verifier_math="clean")
    first.accepted = False
    first.verdict = "pass"
    first.verifier_answer = "3"
    first.verifier_checked_answer = "3"
    first.extracted_answer = "4"
    first.solver_checked_answer = "3"
    first.acceptance_reason = "solver_expression_mismatch"
    first.learning_signal["solver_math"] = "bad"
    first.learning_signal["verifier_math"] = "clean"

    second = make_attempt(answer="3", verifier_tag="3", verifier_math="clean")
    second.accepted = True
    second.verdict = "pass"
    second.verifier_answer = "3"
    second.verifier_checked_answer = "3"
    second.extracted_answer = "3"
    second.solver_checked_answer = "3"
    second.acceptance_reason = "verifier_match_checked"
    second.learning_signal["solver_math"] = "clean"
    second.learning_signal["verifier_math"] = "clean"

    assert h._modal_answer([first]) == ("3", 1)
    assert h._modal_answer([first, second]) == ("3", 3)


def test_unchecked_verifier_tags_do_not_create_modal_confidence():
    first = make_attempt(answer="none", verifier_tag="56", verifier_math="unchecked")
    first.accepted = False
    first.extracted_answer = None
    first.verdict = "pass"
    first.verifier_answer = "56"
    first.verifier_checked_answer = None
    first.verifier_tagged_answer = "56"
    first.acceptance_reason = "solver_expression_mismatch"
    first.learning_signal["verifier_math"] = "unchecked"
    first.learning_signal["bad_tag"] = False

    second = make_attempt(answer="none", verifier_tag="56", verifier_math="unchecked")
    second.accepted = False
    second.extracted_answer = None
    second.verdict = "pass"
    second.verifier_answer = "56"
    second.verifier_checked_answer = None
    second.verifier_tagged_answer = "56"
    second.acceptance_reason = "solver_expression_mismatch"
    second.learning_signal["verifier_math"] = "unchecked"
    second.learning_signal["bad_tag"] = False

    assert h._modal_answer([first, second]) == (None, 0)


def test_clean_verifier_after_structural_solver_error_gets_one_support():
    first = make_attempt(answer="none", verifier_tag="64", verifier_math="clean")
    first.accepted = False
    first.extracted_answer = None
    first.verdict = "pass"
    first.verifier_answer = "64"
    first.verifier_checked_answer = "64"
    first.verifier_tagged_answer = "64"
    first.acceptance_reason = "structural_quantity_contradiction"
    first.learning_signal["solver_math"] = "unchecked"
    first.learning_signal["verifier_math"] = "clean"
    first.learning_signal["bad_tag"] = False
    first.learning_signal["structural_contradiction"] = "periodic_discount_double_charge"

    assert h._modal_answer([first]) == ("64", 1)


def test_invalid_scaffold_without_final_routes_to_repair_not_continuation():
    bad = make_attempt(answer="none", verifier_tag="56", verifier_math="unchecked")
    bad.accepted = False
    bad.extracted_answer = None
    bad.verdict = "pass"
    bad.verifier_answer = "56"
    bad.verifier_checked_answer = None
    bad.acceptance_reason = "solver_expression_mismatch"
    bad.learning_signal["solver_scaffold_tool_used"] = True
    bad.learning_signal["solver_scaffold_feedback"] = "valid=False; error=missing expression"
    bad.learning_signal["verifier_math"] = "unchecked"

    fixed = make_attempt(answer="64", verifier_tag="64", verifier_math="clean")
    fixed.mode = "repair"
    fixed.accepted = True
    fixed.extracted_answer = "64"
    fixed.verifier_answer = "64"
    fixed.verifier_checked_answer = "64"
    fixed.verifier_tagged_answer = "64"
    fixed.acceptance_reason = "verifier_match_checked"

    calls = []
    old_run_attempt = h._run_attempt

    def fake_run_attempt(*args, **kwargs):
        calls.append(kwargs.get("mode"))
        return bad if len(calls) == 1 else fixed

    h._run_attempt = fake_run_attempt
    try:
        config = AgenticConfig(
            cache_write_enabled=False,
            required_agreement=1,
            max_rounds=1,
            synthesis_enabled=False,
            use_expert_vectors=False,
            stop_on_critical_urgency=False,
        )
        result = h.solve_with_humility(None, "Question?", config=config)
        assert result.final_answer == "64"
        assert calls == ["baseline", "repair"]
    finally:
        h._run_attempt = old_run_attempt


def test_structural_tool_error_routes_to_repair_before_dynamic_vectors():
    bad = make_attempt(answer="none", verifier_tag="288", verifier_math="unchecked")
    bad.accepted = False
    bad.extracted_answer = None
    bad.verdict = "pass"
    bad.verifier_answer = "288"
    bad.verifier_checked_answer = None
    bad.acceptance_reason = "structural_quantity_contradiction"
    bad.learning_signal["solver_scaffold_tool_used"] = True
    bad.learning_signal["solver_scaffold_feedback"] = "valid=False; error=bad quantity 'variable'"
    bad.learning_signal["structural_contradiction"] = "periodic_discount_double_charge"
    bad.learning_signal["verifier_math"] = "unchecked"

    fixed = make_attempt(answer="64", verifier_tag="64", verifier_math="clean")
    fixed.mode = "repair"
    fixed.accepted = True
    fixed.extracted_answer = "64"
    fixed.verifier_answer = "64"
    fixed.verifier_checked_answer = "64"
    fixed.verifier_tagged_answer = "64"
    fixed.acceptance_reason = "verifier_match_checked"

    calls = []
    old_run_attempt = h._run_attempt

    def fake_run_attempt(*args, **kwargs):
        calls.append(kwargs.get("mode"))
        return bad if len(calls) == 1 else fixed

    h._run_attempt = fake_run_attempt
    try:
        config = AgenticConfig(
            cache_write_enabled=False,
            required_agreement=1,
            max_rounds=1,
            synthesis_enabled=True,
            use_expert_vectors=True,
            stop_on_critical_urgency=False,
        )
        result = h.solve_with_humility(None, "Question?", vecs={"Social": torch.ones(2)}, config=config)
        assert result.final_answer == "64"
        assert calls == ["baseline", "repair"]
    finally:
        h._run_attempt = old_run_attempt


def test_cache_promotion_rewards_clean_stage_and_penalizes_bad_stage():
    attempt = make_attempt()
    summary = h.synthesis_teaching_summary([attempt], "70000")
    assert summary == {
        "reward_clean_math": 1,
        "penalty_bad_math": 1,
        "skipped": 0,
    }

    promoted, stores = capture_cache_stores(
        lambda: h._promote_verified_synthesis([attempt], "70000", question_key="q")
    )

    assert promoted == 1
    assert len(stores) == 2
    assert stores[0][0].tolist() == [1.0, 1.0]
    assert stores[0][1]["teaching_signal"] == "reward_clean_math"
    assert stores[1][0].tolist() == [-2.0, -2.0]
    assert stores[1][1]["teaching_signal"] == "penalty_bad_math"


def test_cache_promotion_skips_metadata_only_synthesis_records():
    attempt = make_attempt(verifier_tag="70000", verifier_math="clean")
    attempt.acceptance_reason = "verifier_match_checked"
    attempt.learning_signal["parser_rescued_verifier"] = False
    attempt.synthesis_records.append(
        {
            "mode": "time_gated_urgency",
            "metadata": {"attempt_stage": "solver"},
        }
    )

    promoted, stores = capture_cache_stores(
        lambda: h._promote_verified_synthesis([attempt], "70000", question_key="q")
    )

    assert promoted == 2
    assert len(stores) == 2


def test_clean_solver_and_verifier_both_get_rewarded():
    attempt = make_attempt(verifier_tag="70000", verifier_math="clean")
    attempt.acceptance_reason = "verifier_match_checked"
    attempt.learning_signal["parser_rescued_verifier"] = False
    summary = h.synthesis_teaching_summary([attempt], "70000")
    assert summary == {
        "reward_clean_math": 2,
        "penalty_bad_math": 0,
        "skipped": 0,
    }

    promoted, stores = capture_cache_stores(
        lambda: h._promote_verified_synthesis([attempt], "70000", question_key="q")
    )

    assert promoted == 2
    assert len(stores) == 2
    assert [metadata["teaching_signal"] for _, metadata in stores] == [
        "reward_clean_math",
        "reward_clean_math",
    ]


def test_calculator_use_is_reinforced_in_cache_metadata():
    attempt = make_attempt(verifier_tag="70000", verifier_math="clean")
    attempt.response = (
        "Asked quantity: profit\n"
        "Expression: 200000 - 80000 - 50000\n"
        "Computed: <<CALC: 200000 - 80000 - 50000>> = 70000\n"
        "Final answer: 70000"
    )
    attempt.learning_signal["solver_tool_used"] = True
    attempt.learning_signal["verifier_tool_used"] = False

    _, stores = capture_cache_stores(
        lambda: h._promote_verified_synthesis([attempt], "70000", question_key="q")
    )

    solver_metadata = stores[0][1]
    verifier_metadata = stores[1][1]
    assert solver_metadata["calculator_tool_used"] is True
    assert solver_metadata["tool_reinforcement"] == "calculator_clean_use"
    assert verifier_metadata["calculator_tool_used"] is False


def test_clause_map_cache_metadata_keeps_methodology_not_clause_map():
    attempt = make_attempt(verifier_tag="70000", verifier_math="clean")
    attempt.acceptance_reason = "verifier_match_checked"
    attempt.learning_signal["parser_rescued_verifier"] = False
    attempt.learning_signal["solver_clause_map_tool_used"] = True
    attempt.learning_signal["solver_clause_map_feedback"] = (
        "valid=True; covered=C1,C2,C3,C4; asked=C4; givens=C1,C3; rules=C2; operations=C2,C3"
    )
    attempt.learning_signal["solver_clause_methodology"] = h.sanitized_clause_methodology(
        "A mug costs $5, but every second mug costs only 60% of the regular price. "
        "Sam buys 16 mugs. How much does Sam pay in total?",
        attempt.learning_signal["solver_clause_map_feedback"],
    )

    promoted, stores = capture_cache_stores(
        lambda: h._promote_verified_synthesis([attempt], "70000", question_key="q")
    )

    assert promoted == 2
    solver_metadata = stores[0][1]
    assert solver_metadata["clause_map_tool_used"] is True
    assert solver_metadata["clause_map_status"] == "complete"
    assert solver_metadata["clause_methodology"]["kind"] == "periodic_discount_partition"
    assert "clause_map_feedback" not in solver_metadata
    assert "C1" not in repr(solver_metadata)


def test_bad_self_authored_scaffold_is_penalized_even_with_correct_answer():
    attempt = make_attempt(verifier_tag="70000", verifier_math="clean")
    attempt.acceptance_reason = "verifier_match_checked"
    attempt.learning_signal["parser_rescued_verifier"] = False
    attempt.learning_signal["solver_scaffold_tool_used"] = True
    attempt.learning_signal["solver_scaffold_feedback"] = (
        "valid=False; value=70000; unit=eggs; target=dollars; error=target unit mismatch"
    )

    summary = h.synthesis_teaching_summary([attempt], "70000")
    assert summary == {
        "reward_clean_math": 1,
        "penalty_bad_math": 1,
        "skipped": 0,
    }

    _, stores = capture_cache_stores(
        lambda: h._promote_verified_synthesis([attempt], "70000", question_key="q")
    )

    solver_metadata = stores[0][1]
    assert solver_metadata["teaching_signal"] == "penalty_bad_math"
    assert solver_metadata["penalty_reason"] == "solver_bad_scaffold"
    assert solver_metadata["scaffold_tool_used"] is True
    assert solver_metadata["scaffold_reinforcement"] == "scaffold_used_but_reasoning_failed"


def test_fallback_prefers_checked_verifier_correction_over_disputed_solver_answer():
    attempt = make_attempt(answer="125", verifier_tag="64", verifier_math="clean")
    attempt.accepted = False
    attempt.verdict = "unsettled"
    attempt.extracted_answer = "125"
    attempt.verifier_answer = "64"
    attempt.verifier_checked_answer = "64"
    attempt.verifier_tagged_answer = "64"
    attempt.acceptance_reason = "verifier_unsettled"
    attempt.learning_signal["bad_tag"] = False
    attempt.learning_signal["verifier_math"] = "clean"

    assert h._modal_answer([attempt]) == (None, 0)
    assert h._fallback_answer([attempt]) == "64"


def test_clean_repetition_supports_prior_verified_answer_only():
    verified = make_attempt(answer="90", verifier_tag="90", verifier_math="clean")
    verified.accepted = True
    verified.acceptance_reason = "verifier_match_checked"
    verified.learning_signal["solver_math"] = "clean"
    verified.learning_signal["verifier_math"] = "clean"

    repeated = make_attempt(answer="90", verifier_tag=None, verifier_math="unchecked")
    repeated.accepted = False
    repeated.verdict = "uncertain"
    repeated.verifier_answer = None
    repeated.verifier_checked_answer = None
    repeated.verifier_tagged_answer = None
    repeated.acceptance_reason = "verifier_uncertain"
    repeated.learning_signal["solver_math"] = "clean"
    repeated.learning_signal["verifier_math"] = "unchecked"

    assert h._modal_answer([repeated]) == (None, 0)
    assert h._modal_answer([verified, repeated]) == ("90", 3)


def test_single_clean_solver_and_verifier_attempt_counts_two_supports():
    attempt = make_attempt(answer="60", verifier_tag="60", verifier_math="clean")
    attempt.accepted = True
    attempt.acceptance_reason = "verifier_match_checked"
    attempt.learning_signal["solver_math"] = "clean"
    attempt.learning_signal["verifier_math"] = "clean"
    attempt.learning_signal["structural_contradiction"] = None
    assert h._modal_answer([attempt]) == ("60", 2)


def test_structural_invalid_scaffold_pass_does_not_create_confidence():
    attempt = make_attempt(answer="128", verifier_tag="128", verifier_math="clean")
    attempt.accepted = False
    attempt.verdict = "pass"
    attempt.verifier_answer = "128"
    attempt.verifier_checked_answer = "128"
    attempt.acceptance_reason = "invalid_scaffold"
    attempt.learning_signal["solver_math"] = "clean"
    attempt.learning_signal["verifier_math"] = "clean"
    attempt.learning_signal["solver_scaffold_tool_used"] = True
    attempt.learning_signal["solver_scaffold_feedback"] = "valid=False; error=bad quantity 'second'"
    attempt.learning_signal["structural_contradiction"] = "periodic_discount_double_charge"

    assert h._modal_answer([attempt]) == (None, 0)


def test_periodic_discount_double_charge_is_structural_contradiction():
    question = (
        "A mug costs $5, but every second mug costs only 60% of the regular price. "
        "Sam buys 16 mugs. How much does Sam pay in total?"
    )
    bad = "Expression: 16 * (5 + 5 * 0.6)\nFinal answer: 128"
    good = "Expression: (16 - floor(16 / 2)) * 5 + floor(16 / 2) * 3\nFinal answer: 64"
    assert h.structural_contradiction(question, bad) == "periodic_discount_double_charge"
    assert h.structural_contradiction(question, good) is None


def test_profit_total_cost_is_structural_contradiction():
    question = (
        "Josh buys a house for $80,000 and puts in $50,000 in repairs. "
        "This increased the value of the house by 150%. How much profit did he make?"
    )
    bad = (
        "Asked quantity: Profit\n"
        "Expression: 80000 + 50000\n"
        "Computed: 130000\n"
        "Final answer: 130000"
    )
    verifier_bad = (
        "INDEPENDENT_CALCULATION:\n"
        "The increased value is I = V + R = $80,000 + $50,000 = $130,000.\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 130000\n"
        "REASON: matched"
    )
    good = (
        "Expression: (80000 + (80000 * 150 / 100)) - 80000 - 50000\n"
        "Final answer: 70000"
    )
    assert h.structural_contradiction(question, bad) == "profit_expression_without_subtracting_costs"
    assert h.structural_contradiction(question, verifier_bad) == "profit_answer_is_cost_or_intermediate_value"
    assert h.structural_contradiction(question, good) is None


def test_daily_final_meal_per_meal_misread_is_structural_contradiction():
    question = (
        "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. "
        "She gives the chickens their feed in three separate meals. In the morning, "
        "she gives 15 cups and in the afternoon 25 cups. How many cups does she need "
        "to give in the final meal if the flock is 20 chickens?"
    )
    verifier_bad = (
        "INDEPENDENT_CALCULATION:\n"
        "Each chicken eats 3 cups per meal.\n"
        "VERDICT: pass\n"
        "INDEPENDENT_FINAL: 40\n"
        "REASON: matched"
    )
    assert h.structural_contradiction(question, verifier_bad) == "daily_total_misread_as_per_meal"


def test_structural_verifier_error_can_promote_checked_solver_expression():
    attempt = make_attempt(answer="35", verifier_tag="40", verifier_math="unchecked")
    attempt.accepted = False
    attempt.verdict = "pass"
    attempt.response = (
        "Asked quantity: Total cups of feed for the final meal\n"
        "Expression: 3 * 20 - 15 - 25\n"
        "Computed: 3 * 20 - 15 - 25\n"
        "Final answer: 35"
    )
    attempt.extracted_answer = "35"
    attempt.solver_checked_answer = "20"
    attempt.verifier_answer = "40"
    attempt.verifier_checked_answer = None
    attempt.acceptance_reason = "structural_quantity_contradiction"
    attempt.learning_signal["solver_math"] = "bad"
    attempt.learning_signal["solver_expression_checked_answer"] = "20"
    attempt.learning_signal["structural_contradiction"] = "daily_total_misread_as_per_meal"

    assert h._modal_answer([attempt]) == ("20", 1)
    assert h._fallback_answer([attempt]) == "20"


def test_repair_feedback_names_periodic_discount_trap():
    attempt = make_attempt(answer="128", verifier_tag="128", verifier_math="clean")
    attempt.learning_signal["structural_contradiction"] = "periodic_discount_double_charge"
    feedback = h._attempt_tool_feedback(attempt)
    assert "double-charged every item" in feedback
    assert "discounted_count=floor(total_items / period)" in feedback


def test_invalid_scaffold_syntax_can_still_count_when_answer_is_verified():
    attempt = make_attempt(answer="54", verifier_tag="54", verifier_math="clean")
    attempt.accepted = True
    attempt.acceptance_reason = "verifier_match_checked"
    attempt.learning_signal["solver_math"] = "clean"
    attempt.learning_signal["verifier_math"] = "clean"
    attempt.learning_signal["solver_scaffold_tool_used"] = True
    attempt.learning_signal["solver_scaffold_feedback"] = "valid=False; error=bad quantity 'unit'"
    attempt.learning_signal["structural_contradiction"] = None

    assert h._modal_answer([attempt]) == ("54", 2)


def test_invalid_scaffold_syntax_does_not_erase_clean_verifier_support():
    first = make_attempt(answer="72", verifier_tag="90", verifier_math="clean")
    first.accepted = False
    first.verdict = "pass"
    first.extracted_answer = "72"
    first.solver_checked_answer = "72"
    first.verifier_answer = "90"
    first.verifier_checked_answer = "90"
    first.verifier_tagged_answer = "90"
    first.acceptance_reason = "invalid_scaffold"
    first.learning_signal["solver_math"] = "clean"
    first.learning_signal["verifier_math"] = "clean"
    first.learning_signal["solver_scaffold_tool_used"] = True
    first.learning_signal["solver_scaffold_feedback"] = "valid=False; error=bad quantity 'variable'"
    first.learning_signal["structural_contradiction"] = None

    second = make_attempt(answer="90", verifier_tag=None, verifier_math="unchecked")
    second.accepted = False
    second.verdict = "uncertain"
    second.extracted_answer = "90"
    second.solver_checked_answer = "90"
    second.verifier_answer = None
    second.verifier_checked_answer = None
    second.acceptance_reason = "verifier_uncertain"
    second.learning_signal["solver_math"] = "clean"
    second.learning_signal["verifier_math"] = "unchecked"
    second.learning_signal["structural_contradiction"] = None

    assert h._modal_answer([first]) == ("90", 1)
    assert h._modal_answer([first, second]) == ("90", 2)


def test_checked_math_can_rescue_solver_final_tag_mismatch():
    old_generate_text = h.generate_text
    calls = []
    question = (
        "A bakery makes 40 muffins in the morning. It sets aside 6 muffins for staff "
        "and 4 muffins for samples. The bakery sells the remaining muffins for $2 each. "
        "How many dollars does it make from those muffins?"
    )

    def fake_generate_text(M, prompt, **kwargs):
        calls.append(prompt)
        if len(calls) == 1:
            return (
                "Asked quantity: money made from muffins\n"
                "Expression: (40 - 6 - 4) * 2\n"
                "Computed: CALC: (40 - 6 - 4) * 2\n"
                "Final answer: 72"
            )
        return (
            "INDEPENDENT_CALCULATION:\n"
            "remaining_muffins = 40 - 6 - 4\n"
            "money_made = remaining_muffins * 2\n"
            "money_made = 30 * 2\n"
            "money_made = 60\n\n"
            "VERDICT: pass\n"
            "INDEPENDENT_FINAL: 60\n"
            "REASON: solver final tag disagrees with its expression"
        )

    h.generate_text = fake_generate_text
    try:
        attempt = h._run_attempt(
            None,
            question,
            h.solve_prompt(question, deterministic_scaffolds_enabled=False),
            mode="baseline",
            round_index=0,
            config=AgenticConfig(deterministic_scaffolds_enabled=False, max_attempt_tokens=100),
        )
    finally:
        h.generate_text = old_generate_text

    assert attempt.accepted is True
    assert attempt.acceptance_reason == "checked_math_rescues_solver_final_tag"
    assert h._verified_answer(attempt) == "60"
    assert attempt.learning_signal["solver_math"] == "bad"
    assert attempt.learning_signal["verifier_math"] == "clean"
    assert attempt.learning_signal["parser_rescued_solver_final"] is True


def test_solve_respects_cache_write_gate_for_internal_promotion():
    attempt = make_attempt(verifier_tag="70000", verifier_math="clean")
    attempt.acceptance_reason = "verifier_match_checked"
    attempt.learning_signal["parser_rescued_verifier"] = False

    old_run_attempt = h._run_attempt
    h._run_attempt = lambda *args, **kwargs: attempt
    try:
        config = AgenticConfig(
            cache_write_enabled=False,
            required_agreement=1,
            max_rounds=0,
        )
        _, stores_disabled = capture_cache_stores(
            lambda: h.solve_with_humility(None, "Question?", config=config)
        )
        assert stores_disabled == []

        config.cache_write_enabled = True
        _, stores_enabled = capture_cache_stores(
            lambda: h.solve_with_humility(None, "Question?", config=config)
        )
        assert len(stores_enabled) == 2
    finally:
        h._run_attempt = old_run_attempt


def test_learned_concept_context_is_prompted_as_prior_different_question_rule():
    prompt = h.solve_prompt(
        "A museum ticket costs $12, but every third ticket costs half price. A club buys 9 tickets. How much?",
        deterministic_scaffolds_enabled=False,
        learned_concept_context="- periodic_discount_partition: partition every-nth discounts before multiplying.",
    )
    assert "Reusable lessons from prior different questions" in prompt
    assert "do not copy any old answer" in prompt
    assert "periodic_discount_partition" in prompt


def test_oracle_concept_lesson_filters_same_exact_question():
    q1 = "A mug costs $5, but every second mug costs only 60% of the regular price. Sam buys 16 mugs."
    key1 = suite.benchmark_question_key(q1)
    lesson = suite.make_concept_lesson(q1, "128", "64", key1, "contrastive_oracle")
    assert lesson is not None
    bank = []
    assert suite.add_concept_lesson(bank, lesson) is True
    assert suite.format_concept_lessons(bank, key1, q1) is None

    q2 = "A ticket costs $12, but every third ticket costs half price. A club buys 9 tickets."
    key2 = suite.benchmark_question_key(q2)
    context = suite.format_concept_lessons(bank, key2, q2)
    assert context is not None
    assert "discounted_count = floor(total_items / period)" in context

    unrelated = "Mina buys a bike for $80, spends $25 fixing it, and sells it for $140. What is the profit?"
    assert suite.format_concept_lessons(bank, suite.benchmark_question_key(unrelated), unrelated) is None


def test_display_number_avoids_scientific_notation_for_oracle_prompt():
    assert suite.display_number("9E+1") == "90"
    assert suite.display_number("17.500") == "17.5"


def test_concept_lesson_can_be_created_from_missing_final_answer():
    q = "A mug costs $5, but every second mug costs only 60% of the regular price. Sam buys 16 mugs."
    lesson = suite.make_concept_lesson(
        q,
        None,
        "64",
        suite.benchmark_question_key(q),
        "contrastive_oracle",
    )
    assert lesson is not None
    assert lesson["source_pred"] is None
    assert lesson["kind"] == "periodic_discount_partition"


def main():
    tests = [
        test_controller_scorer_accepts_microscopic_roundoff_only,
        test_parser_rescues_bad_tag_without_hiding_it,
        test_parser_rescues_bad_arithmetic_claim_without_hiding_it,
        test_parser_evaluates_final_cue_expression_without_rhs,
        test_parser_prefers_underscore_total_over_intermediate_price,
        test_parser_does_not_let_intermediate_remaining_time_override_final_tag,
        test_parser_prefers_total_earnings_over_overtime_rate_intermediate,
        test_parser_accepts_loose_checked_equation_when_it_matches_verifier_tag,
        test_expression_parser_prefers_expression_matching_final_answer,
        test_expression_parser_accepts_leading_whitespace,
        test_expression_parser_accepts_computed_arithmetic_line,
        test_financial_profit_scaffold_binds_requested_quantity,
        test_choice_profit_scaffold_selects_best_realized_profit,
        test_periodic_discount_scaffold_counts_every_second_items,
        test_solve_prompt_can_disable_deterministic_scaffold_context,
        test_remainder_sale_revenue_scaffold_tracks_units,
        test_daily_split_scaffold_binds_per_day_not_per_meal,
        test_restart_download_scaffold_binds_restart_from_beginning,
        test_return_trip_distance_scaffold_subtracts_return_progress,
        test_break_even_year_scaffold_requires_strict_positive_profit,
        test_overtime_pay_scaffold_binds_regular_and_overtime_pay,
        test_monthly_percentage_total_scaffold_tracks_reduced_third_month,
        test_dozen_total_cost_scaffold_adds_per_dozen_line_items,
        test_reverse_fraction_sales_scaffold_solves_backward_inventory,
        test_remaining_percentage_scaffold_tracks_percentage_of_remainder,
        test_floor_is_available_to_parser_and_runtime_calculator,
        test_model_authored_scaffold_tool_checks_units,
        test_model_authored_scaffold_tool_rejects_unit_mixing,
        test_model_authored_scaffold_handles_singular_plural_unit_aliases,
        test_clause_parser_splits_periodic_discount_problem,
        test_clause_map_tool_validates_role_binding,
        test_clause_map_feedback_marks_missing_clause,
        test_clause_methodology_is_sanitized_for_reuse,
        test_variable_equations_are_available_to_parser_and_runtime_calculator,
        test_near_integer_calculator_result_can_match_scaffold_answer,
        test_quantity_scaffold_can_verify_checked_solver_expression_without_neural_verifier,
        test_urgency_does_not_relax_agreement_without_explicit_control_flag,
        test_independent_verifier_support_can_stabilize_later_clean_solver,
        test_unchecked_verifier_tags_do_not_create_modal_confidence,
        test_clean_verifier_after_structural_solver_error_gets_one_support,
        test_invalid_scaffold_without_final_routes_to_repair_not_continuation,
        test_structural_tool_error_routes_to_repair_before_dynamic_vectors,
        test_cache_promotion_rewards_clean_stage_and_penalizes_bad_stage,
        test_cache_promotion_skips_metadata_only_synthesis_records,
        test_clean_solver_and_verifier_both_get_rewarded,
        test_calculator_use_is_reinforced_in_cache_metadata,
        test_clause_map_cache_metadata_keeps_methodology_not_clause_map,
        test_bad_self_authored_scaffold_is_penalized_even_with_correct_answer,
        test_fallback_prefers_checked_verifier_correction_over_disputed_solver_answer,
        test_clean_repetition_supports_prior_verified_answer_only,
        test_single_clean_solver_and_verifier_attempt_counts_two_supports,
        test_structural_invalid_scaffold_pass_does_not_create_confidence,
        test_periodic_discount_double_charge_is_structural_contradiction,
        test_profit_total_cost_is_structural_contradiction,
        test_daily_final_meal_per_meal_misread_is_structural_contradiction,
        test_structural_verifier_error_can_promote_checked_solver_expression,
        test_repair_feedback_names_periodic_discount_trap,
        test_invalid_scaffold_syntax_can_still_count_when_answer_is_verified,
        test_invalid_scaffold_syntax_does_not_erase_clean_verifier_support,
        test_checked_math_can_rescue_solver_final_tag_mismatch,
        test_solve_respects_cache_write_gate_for_internal_promotion,
        test_learned_concept_context_is_prompted_as_prior_different_question_rule,
        test_oracle_concept_lesson_filters_same_exact_question,
        test_display_number_avoids_scientific_notation_for_oracle_prompt,
        test_concept_lesson_can_be_created_from_missing_final_answer,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
