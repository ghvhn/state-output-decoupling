# Archive

Historical files moved out of the repo root (2026-07-09). Nothing here is
imported by live code; it is kept for provenance.

- **patches/** — one-off find/replace edit scripts (`patch3.py`–`patch30.py`,
  `patch_conflict.py`) that were applied to
  `scripts/interactive_phenomenality.py` on 2026-07-06, plus the original
  `patch_context.txt` from the TDA-mapper era. Already applied; re-running
  them is neither needed nor safe.
- **runners/** — thin launcher stubs for concluded `invariants/` experiments
  (frame shift, arrow/fold, CoT reality, standpoint dialogue, etc.). Their
  results live in `invariants/out/` and the write-ups in `docs/`. To re-run
  one, work from the repo root: `python -m invariants.<experiment>` (the
  stubs assume the root as working directory).
- **scratch/** — one-off debugging/analysis scripts from the 2026-06-27
  benchmark triage. Git-ignored (`scratch_*.py`), kept only locally.
- **tda_pipeline/** — the original TDA domain-mapper core (`pipeline.py`,
  `compute.py`, `orchestrator.py`, `viewer.py`, `check_velocity.py`,
  `evaluate_partial.py`) from before the project became the state-output
  coupling lab. `FINDINGS.json` (which `viewer.py` reads) remains at the
  repo root.

Active launchers stay at the root: `START_HERE.cmd` / `start_here.sh`,
`run_benchmark.cmd` / `.sh`, `run_phenomenality_shell.cmd` / `.sh`,
`run_and_push.ps1`, and the shell config files (`startup.txt`, `fun_init.txt`,
`accuracy.txt`, `lex`, `init`, `start`, `fun`).
