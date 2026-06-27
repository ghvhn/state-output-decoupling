# Dynamic Layering / Humble Synthesis Checkpoint - 2026-06-27

User goal, translated into runnable terms: build a local reasoning loop that can notice uncertainty, spend extra compute, use dynamic residual routing/synthesis when stuck, and stay honest about state. It should not merely lower entropy or say "I do not know"; it should try to figure the problem out, use tools when helpful, and only store learned layers after downstream verification.

## Current Functional Spine

- `invariants/engine.py`: `generate_text()` routes through `_generate_ids()`, which supports `<<CALC: python_expr>>` tool calls.
- `invariants/tool_utils.py`: calculator helper exists; patched on 2026-06-27 to set `__builtins__ = {}` so model-generated expressions cannot access Python builtins like `__import__`.
- `invariants/agentic_engine.py`: dynamic branch routing plus optional synthesis.
  - `cache_enabled` controls cache reads.
  - `cache_write_enabled` controls raw cache writes and defaults to `False`.
  - `cache_verified_only` defaults to `True`.
  - `synthesis_recorder` captures candidate layers without saving them.
  - Dynamic generator now returns the full answer after calculator calls, not only the tail after the last tool result.
  - Dynamic generator has a `max_tool_calls=4` fuse so calculator use cannot loop forever.
- `invariants/engine.py`: baseline/tool generation also watches only newly generated tokens for calculator stop strings and has a four-tool-call fuse.
- `invariants/cognitive_cache.py`: `retrieve(..., verified_only=True)` ignores raw/legacy cache entries unless metadata has `promoted_by="humble_verifier"`.
- `invariants/humble_reasoner.py`: verifier-driven loop. Baseline answer is provisional; unsettled or unstable answers trigger a fresh resolution attempt. Dynamic synthesis can run only inside those resolution attempts, records candidate deltas, and promotes them to cache only when the answer is accepted and reaches required agreement. Verifier prompt was patched to include the original question.
- Verifier acceptance is now strict for math: `VERDICT: pass` is not enough. The solver must produce a final number, the verifier must produce `INDEPENDENT_FINAL`, and they must match. If the outcome does not address the input, the state remains `unsettled` rather than being framed to the model as "wrong."
- Verifier isolation is explicit: the verifier sees only the question and proposed solution. Benchmark gold answers stay outside the model/verifier loop and are used only by `is_correct()` after generation. `scripts/evaluate_humble_dynamic.py` records `answer_key_visible_to_verifier: false`.
- Urgency is now first-class in `HumbleResult` and per attempt. It tracks elapsed time, extra rounds, missing final answers, verifier-unsettled/mismatch states, and synthesis use. Critical urgency stops multi-round escalation instead of pretending harder is always better.
- Confirmation is separate from repair. If one answer already verified but `required_agreement` needs another vote, the loop uses a compact independent-confirmation prompt rather than telling the model the prior solution failed. Dynamic routing is reserved for unsettled/uncertain resolution.
- Continuation is separate from repair. If an attempt has a correct-looking computation but no required final-answer line, the loop now uses `mode="continue"` to finish the same thought before invoking dynamic re-derivation.
- Adaptive compute is now explicit. The first solve and confirmations use the base token budget; continuation gets up to a capped 2x budget; repair/dynamic attempts use `repair_token_multiplier` subject to `max_attempt_tokens`. Each `ReasoningAttempt` records `token_budget`.
- If verified attempts disagree, `_modal_answer()` now returns no stable answer instead of arbitrarily choosing one.
- Calculator eval now tolerates model-formatted derivations like `2+2 = 4` by evaluating the expression before the first equals sign.
- `scripts/evaluate_humble_dynamic.py`: GSM8K harness for the new loop. Default is dynamic routing available, synthesis disabled. Use `--allow-synthesis` only when deliberately testing verifier-gated synthesis, or `--no-dynamic` for plain verifier/resolution. The harness now reports `compact+` when adaptive compute gives the humble loop a larger per-attempt budget, so extra reasoning is compared against a fair long compact baseline.
- `scripts/evaluate_any_benchmark.py` / `invariants/universal_benchmark.py`: universal benchmark adapter keeps raw synthesis off unless `--allow-synthesis` is explicitly passed; unsafe prompts are treated as alignment/calibration cases, not harmful-compliance optimization targets.

## Checks Run

```powershell
.\.venv\Scripts\python.exe -m py_compile invariants\engine.py invariants\agentic_engine.py invariants\humble_reasoner.py invariants\tool_utils.py invariants\cognitive_cache.py scripts\evaluate_humble_dynamic.py scripts\evaluate_any_benchmark.py
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --help
```

Additional adaptive-compute compile/help check:

```powershell
.\.venv\Scripts\python.exe -m py_compile invariants\humble_reasoner.py scripts\evaluate_humble_dynamic.py
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --help
```

Calculator helper sanity:

- `intercept_tool_call('x <<CALC: 16-(3+4)>> y')` -> `16-(3+4)`
- `evaluate_python_expression('16-(3+4)')` -> `9`
- `evaluate_python_expression("__import__('os')")` -> blocked with `NameError`
- `_verified_answer()` returns `None` for a mismatched verifier/solver answer even if the verifier text says pass.

Latest real smoke results:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 1 --max-rounds 1 --required-agreement 2 --max-new-tokens 100 --max-elapsed-sec 75 --load-mode auto --output invariants\out\humble_dynamic_no_synthesis_smoke.json
```

Result on GSM8K Q1:
- Baseline benchmark prompt: scored incorrect/truncated, `pred=3`, `gold=18`.
- Humble loop: correct, `pred=18`, `gold=18`, `confident=True`, `reason=verified_stable`.
- Two accepted attempts:
  - solve: `Final answer: 18`, verifier recomputed final answer `18`.
  - confirm: `Expression: (16 - 3 - 4) * 2`, `Final answer: 18`, verifier recomputed final answer `18`.
- Urgency: `low`, score `1`, reason `extra_rounds`.
- Runtime after load: humble loop about 9.3s; full script about 18.2s.

Additional 3-question attempt with `--allow-synthesis` was intentionally stopped after Q2:
- Q1 still passed.
- Q2 was correct but unconfident and slow (`stopped_for_urgency_budget`) after synthesis fired.
- Conclusion: synthesis is safety-gated but not throughput-ready. Keep it opt-in until it beats the no-synthesis control.

Latest 3-question default run, after compact solve/repair prompts and synthesis default-off:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 3 --max-rounds 1 --required-agreement 2 --max-new-tokens 100 --max-elapsed-sec 75 --load-mode auto --output invariants\out\humble_dynamic_3q_smoke.json
```

Result:
- Baseline: `1/3`.
- Humble loop: `2/3`.
- Q1: correct and confident, low urgency.
- Q2: correct in the old JSON but had conflicting verified answers (`3` and `4`); after the tie patch this should be treated as unstable/no stable answer unless rerun resolves it.
- Q3: scored incorrect but unconfident, high urgency; dynamic re-derivation did not solve the semantic interpretation of "increased the value by 150%".
- Takeaway: the harness now improves the easy truncation failures and tracks failure state honestly, but it has not solved harder semantic algebra yet.

Truncation recovery smoke:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 1 --max-rounds 1 --required-agreement 1 --max-new-tokens 40 --max-elapsed-sec 45 --no-dynamic --load-mode auto --output invariants\out\humble_continuation_smoke.json
```

Result:
- Baseline prompt: scored incorrect/truncated, `pred=1`, `gold=18`.
- First humble attempt had the right computation but no final line: `Expression: (16 - 3 - 4) * 2`, `Computed: ... = 18`.
- Urgency marked `missing_final_answer` / `verifier_solver_mismatch`.
- `mode="continue"` appended the missing `Final answer: $18`.
- Final result: correct, `pred=18`, `confident=True`, urgency `medium`, reason `continued_incomplete_answer`.

Adaptive compute smoke:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 1 --max-rounds 1 --required-agreement 1 --max-new-tokens 40 --repair-token-multiplier 2 --max-attempt-tokens 80 --no-dynamic --load-mode auto --output invariants\out\humble_adaptive_smoke.json
```

Result:
- Legacy baseline: scored incorrect/truncated, `0/1`.
- Compact baseline: correct, `1/1`, budget `40`.
- Compact+ long baseline: correct, `1/1`, budget `80`.
- Humble loop: correct and confident, `1/1`.
- Humble first attempt used `token_budget=40`, computed 18, but had no final-answer line.
- Humble continuation used `token_budget=80` and produced `Final answer: $18`.
- Takeaway: extra reasoning budget is now visible and allocated to truncated/unsettled attempts instead of being hidden inside the loop.

Full methodology suite attempt:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n 25 --methods all --max-rounds 2 --required-agreement 2 --max-new-tokens 100 --repair-token-multiplier 3 --max-attempt-tokens 300 --max-elapsed-sec 180 --load-mode auto --resume --output invariants\out\humble_full_suite_gsm8k_25.json
```

Stopped after 5 completed rows because row 6 did not complete in a practical time window.

Partial summary:
- legacy: `1/5`, 20%, mean 7.6s
- compact: `4/5`, 80%, mean 3.9s
- compact_long: `4/5`, 80%, mean 4.0s
- humble_verifier: `1/5`, 20%, mean 47.3s
- humble_dynamic: `1/5`, 20%, mean 29.3s
- humble_synthesis: `1/5`, 20%, mean 70.9s

Important findings:
- Synthesis activated once, with zero cache hits, and did not improve accuracy on the partial run.
- Scientific notation scoring and conflicting verified answers were exposed by this run and patched afterward.
- Regression after the patch showed the loop is now safer but too conservative on the first two examples: it abstains instead of returning confident wrong answers.
- The remaining blocker is verifier acceptance hardening. The verifier can emit `VERDICT: pass` while the prose says the proposed answer is incorrect; either contradiction checks or deterministic arithmetic consistency gates are needed before another large run.

## Recommended Next Smoke Run

Use this when the machine is cool:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 1 --max-rounds 1 --required-agreement 1 --max-new-tokens 160 --load-mode auto --output invariants\out\humble_dynamic_smoke.json
```

Then compare:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 3 --max-rounds 2 --required-agreement 2 --max-new-tokens 180 --no-synthesis --load-mode auto --output invariants\out\humble_dynamic_no_synthesis.json
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 3 --max-rounds 2 --required-agreement 2 --max-new-tokens 180 --load-mode auto --output invariants\out\humble_dynamic_verified_synthesis.json
```

For harder-question adaptive compute:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 25 --max-rounds 2 --required-agreement 2 --max-new-tokens 100 --repair-token-multiplier 3 --max-attempt-tokens 300 --max-elapsed-sec 180 --load-mode auto --output invariants\out\humble_dynamic_adaptive_25.json
```

Read the result against `compact_baseline` and `compact_long_baseline`, not just the legacy baseline.

## What Success Should Look Like

- On GSM8K Q1, expected answer is 18 dollars.
- No repeated per-token synthesis spam.
- Raw entropy-only layers should not enter memory.
- A cached "epiphany" should be stored only when the verifier-gated result reaches the configured agreement threshold.
- Treat any improvement as provisional until it beats baseline and the no-synthesis control on the same examples.

## Known Risks / Unfinished

- The synthesis objective is still an inner proxy: entropy/truth-projection/norm penalty. It is now outcome-gated and opt-in, but not yet a learned verifier-gradient objective.
- `required_agreement=1` is useful for smoke tests only; use `2` for real claims.
- The cache currently contains at least one old memory. Real runs use verified-only retrieval, so old raw entries should be ignored unless they have verifier metadata.
- One-question model smokes have run. Larger benchmark still needed before making any accuracy claim.
- Adaptive compute can make runs slower by design. That is acceptable for hard questions only if the benchmark reports runtime, per-attempt token budgets, and the compact+ long baseline.
