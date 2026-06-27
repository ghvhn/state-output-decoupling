# Benchmarks

This repo should not claim a benchmark win until the result beats a fair compact-prompt baseline on a meaningful sample.

The useful current benchmark target is narrower:

- Does the loop avoid treating truncated reasoning as a final answer?
- Does it recover by continuing an incomplete thought before invoking a fresh resolution attempt?
- When it reports confidence, is it usually correct?
- Does dynamic re-derivation improve accuracy beyond compact prompting and plain verification?
- If harder items get more reasoning, does the report also give a compact baseline the same larger per-attempt budget?
- Does the verifier stay isolated from benchmark gold answers, using only the question and proposed solution?

## Verifier Isolation

The verifier must not receive the dataset answer key. In `scripts/evaluate_humble_dynamic.py`, gold answers are used only by `is_correct()` after generation for scoring. The verifier prompt receives:

- the original question
- the proposed solution

It does not receive `ex["answer"]`. The verifier returns `INDEPENDENT_FINAL`, meaning its own recomputation from the problem, not a lookup of the benchmark answer.

## Current GSM8K Smoke Results

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 3 --max-rounds 1 --required-agreement 2 --max-new-tokens 100 --max-elapsed-sec 75 --load-mode auto --output invariants\out\humble_dynamic_fair_3q.json
```

Result:

| condition | correct | accuracy |
| --- | ---: | ---: |
| legacy baseline prompt | 1/3 | 33% |
| compact baseline prompt | 2/3 | 67% |
| humble verifier loop | 1/3 | 33% |
| humble confident subset | 1/1 | 100% selective accuracy, 33% coverage |

Interpretation:

- The compact answer contract explains much of the initial improvement.
- The humble loop is not yet a raw-accuracy win on this smoke sample.
- The loop currently has its strongest evidence as a calibration / abstention harness: it can avoid confident incorrect answers and label unresolved cases.
- Dynamic re-derivation still has unresolved harder semantic algebra, e.g. interpreting "increased the value by 150%" correctly.

## Truncation Recovery Smoke

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 1 --max-rounds 1 --required-agreement 1 --max-new-tokens 40 --max-elapsed-sec 45 --no-dynamic --load-mode auto --output invariants\out\humble_continuation_smoke.json
```

Result:

- Baseline prompt truncated and scored incorrect.
- The first humble attempt computed the right value but omitted `Final answer`.
- The loop detected the missing final answer and used `mode="continue"` before any fresh resolution attempt.
- The continued answer verified as correct.

This supports a limited claim: the harness can distinguish "incomplete reasoning due to budget" from "unsettled reasoning" and allocate the next step appropriately.

## Adaptive Compute Smoke

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 1 --max-rounds 1 --required-agreement 1 --max-new-tokens 40 --repair-token-multiplier 2 --max-attempt-tokens 80 --no-dynamic --load-mode auto --output invariants\out\humble_adaptive_smoke.json
```

Result:

| condition | correct | token budget |
| --- | ---: | ---: |
| legacy baseline prompt | 0/1 | 40 |
| compact baseline prompt | 1/1 | 40 |
| compact+ long baseline prompt | 1/1 | 80 |
| humble verifier loop | 1/1 | 40 first attempt, 80 continuation |

Interpretation:

- The first humble attempt computed 18 but omitted the required `Final answer` line.
- Urgency marked the attempt as incomplete instead of treating it as solved.
- The continuation attempt received the larger adaptive budget and finished the same thought.
- This is not a benchmark win by itself. It verifies that extra reasoning budget is now tracked and allocated to unsettled or truncated attempts rather than hidden inside the loop.

## Full Methodology Suite Attempt

Command:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n 25 --methods all --max-rounds 2 --required-agreement 2 --max-new-tokens 100 --repair-token-multiplier 3 --max-attempt-tokens 300 --max-elapsed-sec 180 --load-mode auto --resume --output invariants\out\humble_full_suite_gsm8k_25.json
```

This run was stopped after 5 completed rows because row 6 did not complete in a practical time window. The partial result is still useful:

| method | correct | accuracy | mean time |
| --- | ---: | ---: | ---: |
| legacy | 1/5 | 20% | 7.6s |
| compact | 4/5 | 80% | 3.9s |
| compact_long | 4/5 | 80% | 4.0s |
| humble_verifier | 1/5 | 20% | 47.3s |
| humble_dynamic | 1/5 | 20% | 29.3s |
| humble_synthesis | 1/5 | 20% | 70.9s |

Interpretation:

- The fullest current methodology is not benchmark-ready.
- Synthesis did activate once, but produced no cache hits and did not improve the partial score.
- The run exposed two harness bugs: scientific-notation scoring and conflicting verified answers. These were patched after the run.
- A follow-up regression on the first two examples showed the patched loop is safer but too conservative: it abstained instead of returning a confident wrong answer.
- The remaining blocker is verifier acceptance hardening. The verifier can currently emit `VERDICT: pass` while its prose says the proposed answer is incorrect, so structured verdicts need contradiction checks or a deterministic arithmetic consistency gate.

## What Would Be Publishable

Run at least:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 25 --max-rounds 1 --required-agreement 2 --max-new-tokens 100 --max-elapsed-sec 75 --load-mode auto --output invariants\out\humble_dynamic_fair_25.json
```

For adaptive-compute claims, prefer:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_humble_dynamic.py --n 25 --max-rounds 2 --required-agreement 2 --max-new-tokens 100 --repair-token-multiplier 3 --max-attempt-tokens 300 --max-elapsed-sec 180 --load-mode auto --output invariants\out\humble_dynamic_adaptive_25.json
```

Report:

- legacy baseline accuracy
- compact baseline accuracy
- compact+ long baseline accuracy
- verifier gold-answer access flag
- humble accuracy
- humble coverage
- humble selective accuracy
- confident-incorrect count
- per-attempt token budgets
- mean runtime per item

A credible positive result would be either:

- humble accuracy beats compact baseline, or
- humble accuracy beats compact+ long baseline, if extra compute is part of the claim, or
- humble selective accuracy is high with useful coverage and low confident-incorrect rate.

Until then, the honest repo claim is:

> A verifier-driven reasoning harness with truncation recovery, urgency tracking, and abstention behavior. Early GSM8K smoke tests show calibrated unresolved-state handling and truncation recovery, but not yet a raw benchmark win over compact prompting.
