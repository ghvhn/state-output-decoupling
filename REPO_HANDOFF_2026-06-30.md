# Repo Handoff - 2026-06-30

Read this file before starting another benchmark or changing the steering
policy. The repo now has several operational notes that are easy to miss if you
only inspect the run logs. The short version: the project is not just chasing
accuracy. It is trying to make stateful intervention scientifically cleaner than
prompt-level pressure.

## Read These First

0. `OVERNIGHT_REPO_HANDOFF_2026-06-30.md`
   - This is the latest overnight handoff.
   - The top "Final Status: Egg Earned" section is the current state.
   - Older "egg withheld" sections inside that file are preserved chronology,
     not the final diagnosis.

1. `CHAT_TRANSCRIPT_EVAL_2026-06-29.md`
   - A local chat transcript was compared against the current repo.
   - It records what actually landed, what I changed, and what should be
     checked before spending another long run.

2. `STATEFUL_INTERVENTION_GUARDRAILS.md`
   - This is the philosophy of the architecture.
   - Treat activation vectors as sensors first, controllers second.
   - Probe before steering. Veto contaminated states. Score intervention lanes
     separately.

3. `UNWARRANTED_SKEPTICISM_AND_TIME_CONTEXT.md`
   - Captures the idea Gavin sharpened: measure when the model moves away from a
     right answer without ambiguity, and treat urgency as state/perception-based
     rather than a constant pressure.

4. `BENCHMARK_TRIAGE_HANDOFF_2026-06-28.md`
   - Contains the clean-cache benchmark autopsy and the earlier failure-mode
     triage.

## Tiny Implementation Warnings

- Phenomenality records in the latest run live under:
  - `synthesis_records[].metadata.phenomenality`
  - not only `synthesis_records[].phenomenality`
- I patched the readers to handle both shapes:
  - `scripts/visualize_phenomenality.py`
  - `scripts/analyze_unwarranted_skepticism.py`
- Calculator/tool use is now part of the learning story:
  - `AgenticConfig.max_tool_calls` defaults to 8.
  - The solver/repair/confirmation prompts explicitly allow calculator use for
    intermediate steps, not just final answers.
  - Clean calculator-supported synthesis promotions get
    `tool_reinforcement = calculator_clean_use` in cache metadata.
- The model can now author and iterate a unit scaffold:
  - Syntax: `<<SCAFFOLD: target=<unit>; name=value unit; ...; expression=<formula over names>>>`
  - The runtime checks dimensional compatibility and returns `valid=True/False`,
    `value`, `unit`, and `target`.
  - Invalid model-authored scaffolds are penalized even if the final number is
    recoverable, so this is not just a parser crutch.
  - Read `QUANTITY_SCAFFOLD_ARCHITECTURE.md` before overclaiming any
    scaffolded benchmark win.
- Do not count deterministic quantity scaffolds as the standard exam lane.
  - `scripts/evaluate_humble_full_suite.py` now has `--deterministic-scaffolds auto|off|on`.
  - `auto` means off for `bench-standard` and on for `bench-informed`.
  - The model-authored `SCAFFOLD` tool remains available either way.
  - When deterministic scaffold context is enabled, compact and compact_long
    baselines receive the same context. Use those for fair comparison; legacy is
    the raw unscaffolded reference.
  - Latest diagnostic: Kylar/glasses passes with deterministic scaffold context
    but still fails cleanly with deterministic scaffolds off. That is a real
    remaining architecture gap, not a victory.
- Generic scaffold/tool instructions should become self-iterated over verified
  successes/failures. Task-specific context can be explained at input time, but
  reusable instructions should not become row-specific hand prompts.
- The success Easter egg is intentionally default-on again.
  - To suppress it for unattended runs, use `--no-launch-interactive-on-success`.
  - Hidden compatibility alias: `--boring`.
  - It should launch only after the benchmark has written the final summary and
    released the model/runtime. Do not move it back into the active run loop.

## Suggested Next Move

Before another large benchmark:

1. Verify the probes.
2. Check that time-awareness and warranted-confidence sensors fire on matched
   controls.
3. Run diagnostic rows first, especially house-flip and Kylar/glasses.
4. Run/report scaffold lanes separately:
   - standard: deterministic scaffolds off
   - informed: deterministic scaffolds on
   - oracle comparison: oracle cache/use explicitly labeled
5. Only then rerun the expensive lanes.

## Vector Relation Map Started

An offline vector cartography pass was added:

- `scripts/map_vector_geometry.py`
  - scans all vector-like `.pt` artifacts under `invariants`
  - includes named layer vectors, cache triggers/deltas, organic correction, and
    other residual-stream-shaped tensor artifacts
  - writes pairwise cosine geometry to `invariants/out/vector_geometry_map_*.json`
    and `.md`
- `scripts/build_vector_latent_space.py`
  - projects the artifact-level cosine relation matrix into a small latent map
  - writes coordinates, positive neighborhoods, anti-edges, and axis extremes to
    `invariants/out/vector_latent_space_*.json` and `.md`

First readout:

- `ambiguity_vector`, `urgency_vector`, `disagreement_vector`, and
  `repetition_vector` form one positive neighborhood. Ambiguity and urgency are
  especially entangled, so urgency is not yet an independent intervention axis.
- `organic_correction_vector`, default cache deltas, and the egg cache delta form
  a correction/reward neighborhood.
- the gated bad-math penalty delta is the exact opposite of the organic/egg
  correction delta. That is promising as a veto/anti-routine axis, but still
  needs controls.
- confidence vectors currently sit near each other more than they separate;
  warranted vs unwarranted confidence needs better isolation before steering.

## 2026-06-29 20:06 Health Check

Gavin asked whether local agent activity had disturbed the repo state. The
state check found:

- no Python/model process was running
- the dirty worktree shape was still broadly the existing handoff state
- the cheap regression suite initially failed because `_get_dynamic_agreement`
  was missing from `invariants/humble_reasoner.py`
- `_get_dynamic_agreement` was restored and wired back into
  `solve_with_humility`

Important behavior: agreement relaxation remains **off by default**. The helper
only lowers the required agreement when `relax_agreement_under_urgency=True`,
which is the explicit control flag. After the repair,
`scripts/test_humble_reasoner_regressions.py` passes again.

## 2026-06-29 Egg Run Result

An egg-enabled five-row micro curriculum was run after patching:

- missing-final failures can still enter oracle curriculum
- clean verifier evidence can survive harmless scaffold syntax errors
- checked solver expression plus clean independent verifier can rescue a bad
  final tag while still penalizing the bad tag
- structural contradictions still block confidence

Output:

- `invariants/out/quantity_micro_curriculum_egg_20260629_192151.json`
- `invariants/out/quantity_micro_curriculum_egg_20260629_192151.log`
- fresh cache:
  `invariants/data/cognitive_cache_micro_egg_20260629_192151.pt`

Result:

- compact: `1/5` (`20%`)
- humble_synthesis: `4/5` (`80%`)
- confident humble answers: `4`
- confident-correct humble answers: `4`
- selective accuracy: `100%`
- deterministic scaffold matches: `0`

The Easter egg launch condition fired after the file was written and runtime
cleanup began.

Follow-up repair: the first interactive launch exited immediately because
`scripts/interactive_phenomenality.py` still used an older engine API. The
launcher now imports `AgenticConfig` from `invariants.config`, uses the engine's
shared `_global_cache`, calls `_global_cache.load()` without an obsolete path
argument, and passes the prompt through the current `generate_agentic_text`
signature. `py_compile` and the regression suite pass after that repair.
Use `run_phenomenality_shell.cmd` from the repo root to launch it manually.

Important boundary: this was **not** a clean no-oracle benchmark. It used
`--oracle-curriculum contrastive_oracle` and same-run concept lessons. The
clearest success was not that the model magically learned math from nowhere;
it was that wrong/unresolved rows no longer poisoned confidence, later rows
could learn from previous corrections, and the system beat compact baselines
under the labeled curriculum policy.

Remaining blocker before a serious benchmark: row 1 still fails directly and
row 2 remains slow. Time pressure should choose simpler steps, not lower truth
standards.

## Visibility Note

This repo has several compact handoff files because the important evidence is
split across code, run JSON, logs, and generated reports. Read the notes before
launching a long run. Free the parser from silent metadata-shape assumptions,
but do it scientifically.

## 2026-06-29 21:15 Vector Network / Clause-Map Update

A first-pass vector-network analysis path and several benchmark stability
repairs were added. Check these files before changing the harness again:

- `scripts/map_vector_geometry.py`
- `scripts/build_vector_latent_space.py`
- `scripts/build_vector_network.py`
- `scripts/build_vector_entry_network.py`
- `scripts/probe_narrowing_flow.py`
- `scripts/test_humble_reasoner_regressions.py`

Key artifacts:

- `invariants/out/vector_geometry_map_20260629_201700.md`
- `invariants/out/vector_network_20260629_201711.md`
- `invariants/out/vector_entry_network_20260629_201831.md`

Key finding so far: `validated_flow_vector` and `needless_interrupt_vector`
are strongly anti-correlated at layer level. Treat "warranted confidence" more
like flow / stable evaluative narrowing than a scalar confidence knob.

Important runtime repairs:

- `lens_native.pt` is about 2 GB. It is now opt-in via `--use-tuned-lens`
  instead of being silently loaded inside generation.
- The verifier equation parser now treats underscores as cue separators, so
  `total_cost = ... = 64` is preferred over an intermediate
  `discounted_price = 3`.
- Clean independent verifier evidence after a structural solver error now gets
  one support vote, so it can be confirmed instead of falling into a slow
  dynamic loop.
- `CLAUSEMAP` now exists as an optional external-working-memory tool, but it is
  off by default. Enable explicitly with `--clause-map on`.

Clause-map privacy rule:

- Local attempt logs may contain clause ids and coverage diagnostics.
- Reusable cache metadata must not persist raw clauses, entity names, source
  numbers, or clause ids.
- Cache promotion stores only `clause_methodology`, for example
  `periodic_discount_partition`, with `privacy.tier = reusable_sanitized`.
- If clause maps ever become too leaky, add a confidentiality/privacy probe
  before promotion. Local/private logs can stay richer; reusable cache cannot.

Failed diagnostics to avoid repeating blindly:

- `invariants/out/quantity_micro_row1_nolens_20260629_203809.json` finished
  cleanly but row 1 still failed: compact `16`, humble_synthesis `124.8`, gold
  `64`. The verifier only produced `IN`, meaning the first solver pass consumed
  too much of the budget.
- `invariants/out/quantity_micro_row1_clausemap_20260629_205332.log` was killed
  manually after the clause-map prompt made the row too slow. That is why
  clause-map is now opt-in rather than default benchmark context.

Tests after these changes:

- `python -m py_compile invariants\tool_utils.py invariants\config.py invariants\humble_reasoner.py scripts\evaluate_humble_full_suite.py scripts\test_humble_reasoner_regressions.py`
- `python scripts\test_humble_reasoner_regressions.py`

Both passed after the clause-map methodology and privacy-cache changes.

Run interruption note: the active run was stopped because it had crossed from
expensive into disobeying the benchmark control contract. The replacement
criterion is clear: preserve useful artifacts, patch the measurement failure,
and restart only when the run actually tests the current code.

## 2026-06-29 21:35 Follow-Up: Repair Gate + Watchdog

Additional repairs after the clause-map note:

- `CLAUSEMAP` is now truly opt-in. `--clause-map off` removes the numbered
  clause context and the `Clause map:` response line; earlier code still
  mentioned the tool in generic capability text, which caused accidental use.
- `Computed: <arithmetic expression>` is now locally checkable, so a repair like
  `Computed: 5 * 8 + 5 * 8 * 0.6 / Final answer: 64` can become real solver
  evidence instead of fallback-only output.
- Structural/tool errors now route to plain repair before dynamic vector
  synthesis. Do not send invalid scaffold syntax or periodic double-charging
  straight into dynamic synthesis.
- Generation is chunked (`TDA_GENERATION_CHUNK_TOKENS`, default 24) so the
  harness regains control between chunks.
- Agentic hooks now enforce `GenerationBudgetExceeded` and return partial text
  when the generation deadline expires.

Latest diagnostic outputs:

- `invariants/out/quantity_micro_row1_safe_20260629_210441.json`: row 1 got
  `64` correct, but not confidently; this was fallback/late repair, not a clean
  stable win.
- `invariants/out/quantity_micro_row1_safe2_20260629_211124.json`: row 1 failed
  with `288` because vector synthesis was used on a structural/tool error.
  This motivated the repair-before-dynamic gate.
- `invariants/out/quantity_micro_row1_watchdog2_20260629_213016.json`: watchdog
  succeeded mechanically at ~100s and wrote JSON, but row 1 did not solve under
  that tighter budget.

Interpretation:

- The project is safer than it was: fewer hidden defaults, less cache leakage,
  and fewer runaway rows.
- The project is not yet reliably winning. The next scoring run should use the
  repair gate and watchdog, but probably needs a smoother row budget than 100s.
- Do not call the 80% egg run a clean benchmark; call it an oracle-informed
  curriculum result.

## 2026-06-29 Transition-Layer Hypothesis

Gavin pointed back to the old U-shape result. This is likely the
right bottleneck framing:

```text
prompt text -> interpretation / translation -> mid-workspace logic -> communication / render -> final answer text
```

New note: `TRANSITION_LAYER_BOTTLENECK.md`.

The recent failures fit this better than "the model cannot do arithmetic":

- row 1 often has correct fragments but bad interpretation of the every-second
  discount rule
- verifier/right-answer evidence can be truncated at render time
- dynamic synthesis can worsen a structurally wrong interpretation
- current routing layers L14/L18/L20 do not explicitly protect the early
  interpretation transition or late communication transition

Next architecture experiment should compare:

- safe benchmark policy, clause-map off
- interpretation-expanded policy, extra interpretation time, no deterministic
  answer scaffold

Measure whether correct intermediate states are lost at the communication layer,
not just whether final answers are right.

## 2026-06-30 Steer-Map Store: Success by Step/Layer

New files:

- `invariants/steer_map_store.py`
- `scripts/import_steer_maps.py`
- `scripts/steer_map_store_test.py`

Interactive shell integration:

- `scripts/interactive_phenomenality.py` now creates a `SteerMapStore`.
- Internal routing/synthesis traces are written as unlabeled interactive events.
- Benchmark JSON traces can be imported afterward and labeled by outcome.
- `:steermap` prints the current aggregate summary.

Important distinction:

- `success_rate` is acceptance-aware: a step counts as a clean success only when
  the final answer is correct and the attempt was not rejected.
- `final_correct` is still tracked separately, because a rejected step can be
  part of an eventual recovery.
- Do not flatten `final_correct_attempt_unaccepted` into a win. Gavin wanted us
  to reward proper math while still teaching against rejected/bad steps.

Current local backfill wrote:

- `invariants/out/steer_map_events.jsonl`
- `invariants/out/steer_map_summary.json`

These are runtime artifacts, not commit payload. Regenerate or import new runs
with:

```text
.venv\Scripts\python.exe scripts\import_steer_maps.py --json-glob "invariants\out\humble_full_suite*.json" --json-glob "invariants\out\quantity_micro*.json" --json-glob "invariants\out\remainder_transfer*.json"
```

Early readout after acceptance-aware normalization: `22->27` helped eventual
answers, but several of those attempts were rejected, so the system should treat
that window as "potentially useful, not blindly trusted." This is the point of
the map: movement in latent space earns trust by observed step/layer outcome,
not by vibes.
