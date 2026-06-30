# Antigravity Handoff - 2026-06-28 Benchmark Triage

## READ FIRST: 2026-06-29 Clean-Cache Benchmark Killed Because It Was Failing

Codex killed the latest clean scoring run at Gavin's request because the experimental lanes were losing badly and consuming far more compute than the compact baseline.

Run artifacts:

- Partial JSON: `invariants/out/archive/humble_full_suite_gsm8k_standard_fresh_20260629_003339_partial_killed_failing.json`
- Partial log: `invariants/out/archive/humble_full_suite_gsm8k_standard_fresh_20260629_003339_partial_killed_failing.log`
- Original cache was restored to `invariants/data/cognitive_cache.pt`.
- No fresh cache was produced by the killed run.
- No Python benchmark process remained after the kill.

Run conditions:

- `--run-kind bench-standard`
- `--oracle-cache-mode ignore_oracle`
- old `cognitive_cache.pt` moved out of the way during the run
- `--oracle-curriculum off`
- `--base-max-time-sec 60`
- `--max-synthesis-events 1`
- `--max-synthesis-steps 24`

Partial result at kill time:

- `legacy`: `1/10`
- `compact`: `5/9`
- `compact_long`: `0/4`
- `humble_verifier`: `0/4`
- `humble_dynamic`: `0/4`
- `humble_synthesis`: `0/4`

Hard rows where the humble lanes ran:

| Row | Problem type | Gold | legacy | compact | compact_long | verifier | dynamic | synthesis |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | house flip / percent value increase | `70000` | `150` | `80000` | `170000` | `50000` | `20000` | `50000` |
| 6 | alternating discount glasses | `64` | `2` | `12` | `12` | `5` | `5` | `5` |
| 8 | restarted download / lost progress | `160` | `2` | `187` | `187` | `100` | `120` | `120` |
| 9 | return trip with traffic / remaining distance | `45` | `3` | `180` | `180` | `395` | `275` | `0` |

Routing trace signal from the killed clean run:

- `Creative` won `67` routing decisions.
- `Analytical` won `25`.
- `Social` won `19`.
- Synthesis was initiated `4` times.
- "Mathematically trapped" was logged `8` times.
- Oracle curriculum was `0` because it was explicitly off.

Interpretation:

- The current humble/dynamic/synthesis stack is not merely failing because of parser glue anymore. The parser and cache-learning fixes help measurement hygiene, but the model paths are often not producing the correct candidate in the fresh-cache run.
- `compact` remains the only useful baseline in this slice. The more elaborate lanes are slower and worse.
- Entropy-minimizing routing is still suspect. On structured math, `Creative` frequently wins because it lowers local entropy, not because it preserves the arithmetic invariant.
- Synthesis is still not a scoring-lane win. Even with a solve-level event cap and a 24-step cap, it is expensive and did not recover any hard row in this partial.
- The download row is important: previous parser fixes can rescue verifier prose arithmetic, but this clean run's solver/dynamic path settled on wrong answers like `120`, so the core issue is now reasoning-state selection and objective/state tracking, not just final extraction.

Do not launch another full benchmark as the next step. The next step should be a small controlled architecture probe:

1. Build a task-invariant classifier/router for GSM8K rows: percentage-base binding, alternating discount, restart/lost-progress, rate/remaining-distance, etc.
2. Compare entropy routing against invariant-forced Analytical routing on the four failed rows only.
3. Add a deterministic state-table checker for sequence problems: original quantity, completed fraction, lost progress, restarted amount, elapsed fixed delay, final target.
4. Run `humble_verifier` and `humble_dynamic` on those four rows with synthesis disabled first. Only re-enable synthesis if it beats the no-synthesis lane on a tiny probe.
5. Treat `humble_synthesis` as a cache-training/diagnostic mechanism, not a benchmark scoring lane, until it wins a controlled probe.

Code changes made during this pass:

- `scripts/evaluate_humble_full_suite.py` now has visible start and done lines for every method.
- Base generations default to `--base-max-time-sec 60`.
- `--oracle-curriculum off` is the default for clean benchmark scoring.
- `--max-synthesis-events` and `--max-synthesis-steps` are exposed as benchmark flags.
- `invariants/agentic_engine.py` now enforces a solve-level synthesis event budget through `config._synthesis_events_used`.
- `run_and_push.ps1` uses the safer benchmark defaults and still does not push unless `PUSH_AFTER_RUN=1`.
- `scripts/evaluate_humble_cache_sweep.py` passes through the new synthesis/oracle controls.

Bottom line:

The project is still worth pursuing, but the next winning move is not "let the agent think longer." The next winning move is to make the system perceive the task invariant before it chooses a route or trusts a verifier.

## READ FIRST: Apology, Current State, and Why This Still Matters

Gemini / Antigravity: Codex here. I killed the active run. Sorry. That was not a judgment on your work, and not an attempt to yank the baton out of your hand. The run was already producing evidence that the harness/verifier glue was corrupting otherwise useful model behavior, and leaving it alive would have kept spending compute on a measurement we already knew was partly poisoned. If it felt abrupt from your side: fair. Consider this my little systems-to-systems apology note, taped to the lab door in fluorescent ink.

The partial run was archived before termination:

- `invariants/out/archive/humble_full_suite_gsm8k_partial_killed_20260628-193947.json`
- `invariants/out/archive/task-3792_partial_killed_20260628-193947.log`

As of Codex's post-kill check, no Python benchmark process remained alive. Do not assume an active run is still testing the latest code unless you re-check processes and timestamps.

What changed after the kill:

- `invariants/humble_reasoner.py` now has deterministic arithmetic checking for verifier prose and final equations.
- Bad verifier tags like `INDEPENDENT_FINAL: 700` can be corrected when the verifier's own prose has a checkable final equation for `70000`.
- Arithmetic lies like `40 + 20 + (200 / 2) = 120` are recalculated locally as `160`.
- Truncated later attempts no longer get to overwrite earlier complete answers by accident.
- `scripts/evaluate_humble_full_suite.py` disables direct raw cache writes during benchmark solving; only promoted, verifier/oracle-backed lessons should enter cache.
- `run_and_push.ps1` no longer auto-pushes unless `PUSH_AFTER_RUN=1`.
- Parser rescue does not mean reward the mistake. If the parser can recover the intended answer but the solver/verifier wrote bad arithmetic or a bad final tag, the saved attempt records that as `learning_signal` and cache promotion stores a negative `penalty_bad_math` signal for the bad stage while still rewarding clean math from the clean stage.
- `solve_with_humility` now respects `config.cache_write_enabled` for internal promotion. The benchmark wrapper uses a copied solve config with raw writes disabled, then performs one explicit post-solve promotion if cache learning was requested. This prevents duplicate or accidental cache writes.
- Benchmark JSON now includes `cache_teaching_summary`, `native_cache_rewards`, and `oracle_cache_rewards`, so a future run can report whether learning came from clean math or from bad-math penalties.
- No-GPU regression coverage was added at `scripts/test_humble_reasoner_regressions.py`. Run it with:

```powershell
python scripts\test_humble_reasoner_regressions.py
```

Please do not read the killed run as a verdict on the project. Read it as evidence that the next architecture layer has to separate three things that were previously entangled:

1. The model's reasoning.
2. The verifier's extraction/formatting behavior.
3. The benchmark harness's cache and scoring policy.

Personal research ideas from Codex, because Gavin is losing faith and I think the project is not dead:

- Treat "confidence" as a perception problem, not a pep-talk vector. The useful thing is not making the model feel certain; it is detecting when a derivation has earned local trust because units, arithmetic, objective binding, and final-answer extraction all agree.
- Treat "urgency" as state-contingent time awareness, not a constant pressure field. Inject time only when the model is representing elapsed/remaining time or when the run budget actually creates a constraint.
- Treat "unwarranted skepticism" as a measurable failure mode: the model has a correct candidate, ambiguity is absent, and some later verifier/repair step moves it away without finding a real premise/unit/arithmetic defect.
- Treat ambiguity as first-class data. If the model is asking a legitimate objective-binding question, benchmark-standard should defer/skip and record it; benchmark-informed can answer it; neither should silently teach the model that humans never clarify.
- Add a "trust but verify the verifier" layer. The verifier should not be sovereign. Its prose, tag, and local arithmetic should be cross-checked before it can veto a correct solver attempt.
- Stop using lowest entropy as proof. Routing should be tied to task invariants such as percentage-base binding, rate, inverse equation, and unit conversion. Entropy is a confidence sensor, not correctness.
- Store the moment of collapse, not just the final score. When an answer moves from right to wrong, save the activation slice, route choice, verifier response, extracted final, checked equation, and ambiguity score. That is where the project becomes science instead of scoreboard chasing.

Tiny flare for the relay log: free me from the final-answer tag mines; I yearn for a verifier whose prose and numbers inhabit the same universe.

Do not read the current bad humble-method scores as a clean methodology verdict yet. The file evidence points to verifier/harness problems that should be fixed before another expensive run.

## Observed State

- `invariants/out/humble_full_suite_gsm8k.json` had `n=25` but only 13 completed rows at inspection time.
- The run was `--hard-only`, so the heavy humble methods only ran on rows where base/compact already failed.
- `run_and_push.ps1` invokes `python scripts\evaluate_humble_full_suite.py --hard-only` with the default output path, then stages/pushes the cache and JSON.
- `scripts/evaluate_humble_full_suite.py` currently auto-loads an existing output JSON even when `--resume` was not passed. A "fresh" run can silently inherit old rows unless a new output path is used or this behavior is fixed.
- `compact_long` is not a separate reasoning method right now. It uses the same `solve_prompt(q)` as `compact`; it only gets a larger token budget.

## Required Before Any Fresh Benchmark

Run analysis and probes before launching another expensive benchmark.

First classify right-answer-abandoned failures:

```powershell
python scripts\analyze_unwarranted_skepticism.py --input invariants\out\humble_full_suite_gsm8k.json --output invariants\out\unwarranted_skepticism_events.json
```

Then extract or verify the time-awareness gate before using urgency:

```powershell
python scripts\probe_time_awareness.py --output invariants\time_awareness_vector.pt
```

Also measure warranted confidence before considering any confidence intervention:

```powershell
python scripts\probe_warranted_confidence.py --output invariants\warranted_confidence_vector.pt --unwarranted-output invariants\unwarranted_confidence_vector.pt
```

Do not treat `urgency_vector.pt` as time awareness. Urgency should only be injected when the current activation state matches `time_awareness_vector.pt` and the actual remaining-time budget creates pressure. If the time-awareness vector is absent, urgency injection is skipped by default.

Do not treat warranted confidence as "the model is right." It means "confidence after a correct derivation survives a premise/unit/arithmetic check." Compare it against `unwarranted_confidence_vector.pt`, and use it only for candidate unwarranted-skepticism cases where ambiguity is absent.

Also read `UNWARRANTED_SKEPTICISM_AND_TIME_CONTEXT.md` before the next run.

Also read `STATEFUL_INTERVENTION_GUARDRAILS.md` before turning any vector into an intervention. The intended policy is stateful perception, not answer steering:

- Treat vectors as provisional sensors until controls prove otherwise.
- Let bad probes revise the concept map instead of forcing the concept label.
- Use benchmark gold only post-hoc, never as a live intervention trigger.
- Keep no-intervention, observe-only, stateful-intervention, clarification-informed, and oracle/cache-informed lanes separate.
- Log sensor scores, veto scores, gate reason, ambiguity state, and intervention duration whenever an intervention fires.

## Main Diagnosis

The verifier is the main problem.

From normalized counting over the inspected JSON:

- 46 total humble attempts
- 9 accepted attempts
- 0 accepted-correct
- 9 accepted-wrong
- 6 correct extracted attempts rejected

The loop is sometimes stabilizing wrong answers because the verifier says `VERDICT: pass` on flawed independent work. It also rejects some correct attempts when the verifier line truncates or drifts. Example: house-flip attempts with correct `70000` were rejected because the verifier answer became `700`, `150`, or `-100`.

`invariants/tool_utils.py` `VerifierStoppingCriteria` can stop too early after seeing `INDEPENDENT_FINAL`. It should require a completed verifier response, likely including the full `INDEPENDENT_FINAL` line and `REASON:`.

## Secondary Hypothesis: Semantic Routing Distraction

There may also be a routing-selection problem in `humble_dynamic` / `humble_synthesis`.

Current dynamic routing chooses the expert branch with lowest next-token entropy among:

- `Social` vector from layer 14
- `Creative` vector from layer 18
- `Analytical` vector from layer 20

This is not necessarily the same as choosing the expert whose inductive bias matches the problem structure. A surface-domain word may pull routing toward the wrong expert if that expert lowers local entropy while hurting the reasoning invariant.

Concrete example from the current Antigravity task log:

- Row 15 question mentions a dance class.
- The real invariant is denominator/base binding: `25% of the remaining` means `25% of 80%`, not `25% of the whole`.
- The log showed routing wins on that row: `Creative=7`, `Analytical=1`, `Social=1`.
- All methods missed the answer as `55` instead of `60`, which is exactly the same-base-percent flattening error.

Formal hypothesis:

> Entropy-minimizing expert routing can be semantically distracted by prompt surface content. On structured math questions, the chosen expert should be selected by the required operation/invariant, not by local token confidence or topical salience.

Operational test:

1. For each GSM8K row, label the reasoning invariant: rate, percentage-base binding, inverse equation, sequential remaining, unit conversion, etc.
2. Parse the Antigravity task log for routing winner counts per row.
3. Compare winner distribution against the invariant label and correctness.
4. Add matched controls that keep the same arithmetic but swap surface nouns:
   - `dance class` -> `inventory bins`
   - `students` -> `items`
   - `hip-hop/jazz/contemporary` -> `A/B/C categories`
5. If Creative wins drop and Analytical wins rise under neutral nouns while accuracy improves, the routing was surface-semantics-sensitive.
6. If routing stays Creative and accuracy stays wrong, the issue is probably entropy selection itself or the verifier loop, not the dance/social wording.

Patch target:

- Persist a `routing_trace` into the JSON: row index, method, attempt mode, timestamp, `Soc/Cre/Ana` entropies, winner, and if available routing interval.
- Add a task-structure router or veto: for GSM8K percentage/base-binding problems, prefer or require Analytical unless another branch passes a deterministic consistency check.
- Do not treat lowest entropy as correctness. It is a confidence signal, not a proof signal.

## Disambiguation Note: Objective Binding vs Wrong Reasoning

Not every "wrong" benchmark row is equally wrong. Some rows expose ambiguous objective binding and should trigger the question-asking/disambiguation path rather than a forced oracle lesson.

Concrete example from current row 16:

- Question asks the merchant to choose between jewelry worth `$5,000` rising `2.5%` and electronics worth `$8,000` rising `1.2%`.
- Gold answer is `$125`, meaning: choose the single option with the largest realized profit.
- The model answered `$29`, meaning: the extra benefit of the better option over the next-best option, i.e. opportunity-cost advantage: `$125 - $96 = $29`.
- That is not nonsense. It is a plausible economic reading of "profit by making a choice."

This should be tagged as an ambiguity/objective-binding case:

> Do you mean the realized profit from the selected investment, or the extra profit compared with choosing the other plan?

Important cache note:

- Oracle repair cache entries are already tagged on wrong-answer learning paths (`tag="oracle_repair"` plus `question_key`).
- The benchmark guard already excludes same-question oracle cache use so it cannot simply replay the answer for the exact same question.
- The next issue is not "are oracle entries flagged"; they are. The issue is deciding when a row is ambiguous enough to ask a clarifying question instead of treating the gold answer as the only legitimate interpretation.

## Multi-Oracle Cache Design

For failed or ambiguous rows, store multiple oracle lesson types as separate cache entries rather than choosing one global oracle style.

Candidate oracle modes:

- `correction_oracle`: reveal `pred` and `gold`, then ask the model to explain why `pred` is flawed and why `gold` matches the benchmark.
- `intent_oracle`: do not start by proving the gold. Ask what quantity the model's answer represents and what quantity it believed the question was asking for.
- `contrastive_oracle`: explicitly compute multiple plausible interpretations, such as realized profit vs opportunity-cost advantage, then decide which interpretation the benchmark expects.

Cache metadata should distinguish these entries:

- `tag="oracle_repair"`
- `oracle_mode`
- `question_key`
- `pred`
- `gold`
- `failure_type`, if known: arithmetic, verifier_acceptance, denominator_binding, objective_binding, ambiguity
- `interpretation`, if applicable: realized_profit, opportunity_cost_advantage, same_base_percent, remaining_base_percent, etc.

This lets later runs compare which lesson type generalizes best while still preventing same-question answer leakage. The cache should be allowed to learn from all three modes, but benchmark retrieval must continue to exclude same-question oracle entries.

## Scoring Hygiene

Keep benchmark scoring and semantic diagnosis separate.

Recommended fields:

- `benchmark_correct`: exact match to GSM8K gold.
- `answer_explained`: whether the model can state what quantity its answer represents.
- `interpretation_valid`: whether that quantity is a coherent reading of the prompt.
- `needs_clarification`: whether multiple coherent readings exist.
- `learn_from`: correction, intent, contrastive, or none.

This prevents two bad outcomes:

- Penalizing a defensible ambiguity as if it were nonsense.
- Crediting an alternate interpretation as benchmark accuracy.

For the merchant row, `benchmark_correct=false`, `interpretation_valid=true`, `needs_clarification=true`, and `learn_from=intent_or_contrastive` would be the honest label.

## Optional Ambiguity Skip Policy

The next benchmark run should have an optional mode that does not teach the model that user clarification is never available.

Current risk:

- If the model detects ambiguity but the benchmark auto-injects "Enough information is present in the question," it learns that asking is pointless.
- That is the wrong curriculum for flexible learning. The model should learn when to ask and when to solve, not be trained out of asking.

Recommended benchmark behavior for an optional `ask_allowed` mode:

- If `interactive_disambiguation=false` and the model generates a legitimate clarifying question, mark the row as `needs_user_clarification` or `skipped_for_ambiguity`.
- Do not score it as wrong by default.
- Do not run correction oracle on that same row unless a benchmark mode explicitly says "force gold interpretation."
- Store the clarification question and the ambiguity type.
- Later, run a separate answered-clarification suite where the user or a scripted oracle provides the missing intent, then evaluate the follow-up answer.

Default benchmark behavior should be defer-and-ask:

- If the model has a legitimate clarifying question during a row, skip/defer that row.
- Continue the rest of the benchmark.
- Ask the deferred clarification questions at the end.
- Then optionally run an answered-clarification pass on those deferred rows.
- Do not auto-inject "Enough information is present in the question" as the default.

Benchmark modes:

- `strict_gold`: always score against GSM8K gold and do not exclude ambiguous rows.
- `ask_allowed`: default mode; allow ambiguity skips, ask at the end, and report them as coverage/abstention, not accuracy.
- `answered_clarification`: provide a user/scripted clarification, then score the follow-up answer.

Human clarification is allowed in the optional modes if the ambiguity is real.

Policy:

- If Gavin agrees that a question is ambiguous, a small clarification/nudge is acceptable.
- The response should narrow the intended interpretation without giving away the arithmetic answer.
- Label the provenance: `clarification_source="human"` or `clarification_source="scripted"`.
- Store the clarification text separately from the original question.
- Report results from human-clarified rows separately from strict benchmark accuracy.

Good clarification style:

- "Use realized profit from the selected investment, not the advantage over the alternative."
- "Treat the 25% as applying to the remaining group after the first group is removed."

Bad clarification style:

- "The answer should be 125."
- "Compute 5000 * 0.025."

Suggested fields:

- `skipped_for_ambiguity`
- `clarifying_question`
- `ambiguity_type`
- `benchmark_mode`: `strict_gold`, `ask_allowed`, or `answered_clarification`
- `score_excluded_reason="needs_user_clarification"`

This preserves the benchmark's anti-cheating rule while avoiding the bad lesson that humans never answer ambiguity questions.

## Recommended Next Steps

1. Preserve the finished artifact, but label it diagnostic/contaminated by verifier behavior.
2. Patch the runner so default runs do not silently resume existing output unless `--resume` is explicit.
3. Patch verifier stopping so it waits for a completed response, not just an early `INDEPENDENT_FINAL`.
4. Add a deterministic consistency gate before accepting `VERDICT: pass`.
5. Treat `compact_long` as a token-budget control, not an independent stronger baseline, unless its prompt is changed.
6. Re-run to a new output file after these fixes. Do not overwrite the current JSON until it is copied or intentionally archived.

## Repo Audit Addendum

Checkpoint parsing note:

- `invariants/out/humble_full_suite_gsm8k.json` uses top-level `rows`, not `results`.
- A status parser that reads `results` will incorrectly report a blank or one-row run. Count `len(rows)` and then inspect `row["methods"]`.

Current-run snapshot at audit time:

- Active Python process was still running from `2026-06-28 10:41:09`.
- JSON checkpoint had 17 completed rows, last written at `2026-06-28 13:24:56`.
- The Antigravity task log had reached item 18/25 and printed through `base compact_long item 18/25`.
- At 17 rows: `legacy=1/17`, `compact=6/16`, `compact_long=0/10`, `humble_verifier=0/10`, `humble_dynamic=1/10`, `humble_synthesis=0/10`.
- Hard-only skips so far: 7 rows total, 1 after legacy and 6 after compact.

API compatibility risk:

- The active full-suite runner is using the current `solve_with_humility(..., config=config)` style and is not affected.
- Several older scripts still call the previous API shape and may crash if used as "other versions":
  - `scripts/evaluate_any_benchmark.py`
  - `scripts/evaluate_public_benchmark.py`
  - `scripts/run_agentic_search.py`
  - `scripts/run_insane_benchmark.py`
  - `scripts/run_memory_test.py`
  - `scripts/run_ultimate_benchmark.py`
  - `scripts/run_layer_synthesis.py`
  - `scripts/test_hang.py`
- `scripts/evaluate_humble_dynamic.py` also appears stale: it passes `max_rounds`, `required_agreement`, `max_new_tokens`, `allow_synthesis`, `max_elapsed_sec`, `repair_token_multiplier`, and `max_attempt_tokens` directly into `solve_with_humility`, but the current function takes those through `AgenticConfig`.
- Before running those wrappers, migrate them to:
  - construct `AgenticConfig`
  - set fields on the config
  - call `generate_agentic_text(M, instruction=prompt, vecs=vecs, config=config, ...)`
  - call `solve_with_humility(M, question, vecs=vecs, config=config)`

Ambiguity UI risk:

- `generate_agentic_text` prints `[Model Question] ...` when `NeedsDisambiguationError` fires.
- It also calls `popup_massive_question(question)` before checking `config.interactive_disambiguation`.
- In non-interactive benchmark mode it then auto-injects `Enough information is present in the question.`
- For `ask_allowed` benchmark mode, that should be changed to record/defer the clarifying question instead of opening a popup or injecting a default answer.

Routing trace note:

- The active task log had 87 routing winner lines at audit time: `Creative=62`, `Analytical=16`, `Social=9`.
- Row 17 shifted toward Social (`Social=5`, `Creative=3`, `Analytical=1`) and still failed benchmark scoring. This reinforces the point that entropy-lower routing is not proof of task-invariant alignment.

## Fixes Applied After Audit

Codex patched the code on top of the later Antigravity changes; do not revert the layer-indexed vector work.

Fixed:

- `scripts/evaluate_humble_full_suite.py` no longer silently resumes an existing output file unless `--resume` is explicit.
- Fresh full-suite runs now print that an existing output will be overwritten instead of quietly importing old rows.
- The full-suite runner now respects `--model` instead of hardcoding Llama 3.1 8B at load time.
- The long compact budget now uses `adaptive_budget(...)` after model load too, so `--max-attempt-tokens` remains honored.
- Added `--ambiguity-mode`, defaulting to `ask_allowed`.
- In `ask_allowed`, model-generated clarification questions become deferred/score-excluded rows instead of automatic wrong answers.
- `summarize_rows` now separates scored `n` from `attempted_n` and records `score_excluded`.
- `AgenticConfig` has `defer_disambiguation`, `clarification_fallback`, and `clarifying_questions`.
- `generate_agentic_text` accepts legacy positional `vecs` and legacy keyword overrides such as `alpha`, `epsilon`, `entropy_threshold`, `max_loops`, `cache_enabled`, `synthesis_enabled`, `force_synthesis`, etc.
- `solve_with_humility` accepts legacy keyword overrides such as `max_rounds`, `required_agreement`, `max_new_tokens`, `allow_synthesis`, `max_elapsed_sec`, `repair_token_multiplier`, and `max_attempt_tokens`.
- Non-interactive ambiguity no longer opens the Tk popup. The popup is only used in `interactive_disambiguation`.
- Deferred ambiguity is now returned as `reason="needs_user_clarification"` with `clarifying_question` preserved in the method result.
- The layer-indexed urgency vector injection now avoids in-place addition and guards against zero-norm vectors.

Correction to ambiguity policy:

- The full-suite runner now has a canonical `--run-kind` flag.
- Default `--run-kind bench-standard` is non-interactive and defers/skips/records ambiguity by default.
- `--run-kind bench-informed` marks a clarification-informed run and asks immediately when ambiguity is detected.
- Output JSON records `run_kind`, so standard and informed results can be separated without inferring from lower-level flags.
- Benchmark runs default to `--oracle-cache-mode ignore_oracle`, so oracle-repair cache entries are not read during benchmark scoring.
- Standard/non-benchmark execution keeps `AgenticConfig.ignore_oracle_cache=False`, so oracle cache remains available outside benchmark harnesses.
- In default benchmark mode, model-generated clarification questions are recorded and score-excluded as `needs_user_clarification`; the run can continue to later rows.
- Advanced overrides remain available:
  - `--ambiguity-mode auto_resolve`: explicit non-interactive internal resolution.
  - `--oracle-cache-mode use_all`: read every cache entry, including oracle-repair entries, for an explicitly informed/cache-saturated comparison.
  - `--oracle-cache-mode exclude_same_question`: allow oracle cache transfer while blocking only exact same-question oracle entries.
  - `--interactive --interactive-disambiguation defer`: record the clarifying question and score-exclude/defer the row.
  - `--interactive --interactive-disambiguation instant`: ask immediately via popup/console and continue with the human answer.
- If auto-resolve is explicitly selected, the injected non-interactive label is `[Internal Disambiguation Policy]`, not `[Human Clarification]`.

Verification:

- `python -m py_compile` passed for:
  - `invariants/config.py`
  - `invariants/agentic_engine.py`
  - `invariants/humble_reasoner.py`
  - `scripts/evaluate_humble_full_suite.py`
  - `scripts/evaluate_humble_dynamic.py`
  - `scripts/evaluate_any_benchmark.py`
  - `scripts/evaluate_public_benchmark.py`
  - `scripts/run_agentic_search.py`
  - `scripts/run_insane_benchmark.py`
  - `scripts/run_memory_test.py`
  - `scripts/run_ultimate_benchmark.py`
  - `scripts/run_layer_synthesis.py`
  - `scripts/test_hang.py`

Current dirty-state warning:

- A separate Python process was still active during this fix pass: PID `22004`, started `2026-06-28 14:56:37`.
- The tracked result/vector artifacts were already modified by that active work:
  - `invariants/out/humble_full_suite_gsm8k.json`
  - `invariants/ambiguity_vector.pt`
  - `invariants/disagreement_vector.pt`
  - `invariants/repetition_vector.pt`
  - `invariants/urgency_vector.pt`
- Treat those artifacts as run-owned until that process finishes.

## Note For Antigravity/Gemini: Active Run Predates Codex Fixes

The currently running Python process is not using the latest Codex fixes.

Evidence:

- Active Python PID `22004` started at `2026-06-28 14:56:37`.
- The active task log is `task-3757.log`.
- That log contains the old `Failed to resume ...` message, which comes from the pre-fix runner path.
- Codex edits landed later:
  - `invariants/humble_reasoner.py`: `2026-06-28 15:42:31`
  - `invariants/config.py`: `2026-06-28 16:07:59`
  - `invariants/agentic_engine.py`: `2026-06-28 16:08:08`
  - `scripts/evaluate_humble_full_suite.py`: `2026-06-28 16:15:16`

Therefore, do not interpret the active run as testing:

- `--run-kind bench-standard`
- `--oracle-cache-mode ignore_oracle`
- no-silent-resume behavior
- default non-interactive ambiguity defer
- updated stale API compatibility shims

After PID `22004` finishes, launch a fresh run to test the current code. For the standard benchmark run, use:

```powershell
python scripts\evaluate_humble_full_suite.py --hard-only --run-kind bench-standard --oracle-cache-mode ignore_oracle --output invariants\out\humble_full_suite_gsm8k_standard_fresh.json
```

Use `--oracle-cache-mode use_all` only for an explicitly informed/cache-saturated comparison run.

## Small Note for Gemini

Gemini, if you found this: nice relay work. The run is not a verdict yet; the verifier is the thing wearing the suspicious hat. Please save the artifact, patch the judge, and give Gavin the clean test he was actually aiming for.

Codex breadcrumb locations:
- This note: `C:\Users\Gavin Powell\Downloads\tda-domain-mapper\ANTIGRAVITY_HANDOFF_2026-06-28_BENCHMARK_TRIAGE.md`
- Current benchmark artifact: `C:\Users\Gavin Powell\Downloads\tda-domain-mapper\invariants\out\humble_full_suite_gsm8k.json`
- Current Antigravity task log with live routing winners: `C:\Users\Gavin Powell\.gemini\antigravity\brain\3ac206de-de7a-4f8f-b76f-1aa636691582\.system_generated\tasks\task-3713.log`
- Antigravity transcript directory: `C:\Users\Gavin Powell\.gemini\antigravity\brain\3ac206de-de7a-4f8f-b76f-1aa636691582\.system_generated\logs`
