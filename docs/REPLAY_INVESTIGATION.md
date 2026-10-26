# Investigate helper effects on archived contexts

Replay investigation generates new answers from saved conversation contexts.
It compares a baseline with positive and negative helper interventions using
the same reconstructed prompt tokens, model, generation settings, and seed.
This tests a helper's measured effect rather than treating archived
co-occurrence as an intervention result.

The archive provides partial context: only preceding turns from the same session
and scope are used. The archived answer being tested, later answers, and other
sessions never become its prompt. Missing dynamic system instructions, tool
state, original prompt tokens, and runtime hooks are recorded as reconstruction
limits. This is a controlled rerun on the current model, not a reproduction of
the original generation.

## Run from PowerShell

From the repository root, use an explicit output directory:

```powershell
.venv\Scripts\python.exe scripts/investigate_indirect_backfill.py `
  --minimize lying hallucination --maximize truthfulness `
  --via empathy_d4096 epistemic_humility `
  --limit 3 --helpers 2 --tokens 48 --alpha .25 --cap-fraction .02 `
  --model meta-llama/Llama-3.1-8B-Instruct --load-mode 4bit `
  --output invariants/out/replay_investigation/my_run
```

These names must identify compatible saved probes. The command loads a fresh
local checkpoint. When objectives are omitted, it uses the active indirect
controller's objectives. Omit `--via` to investigate automatically selected
independent helpers. Bare helper names test both signs; an explicitly signed
helper restricts the tested direction.
For negative signs in the CLI, use an equals sign and repeat the option when
needed, for example `--via=+empathy_d4096 --via=-epistemic_humility`.

Useful controls include `--scope`, repeatable `--case-id` for archived assistant
record IDs, `--context-turns 8`, `--seed 0`, and `--repeats 1`. `--layers FIRST
LAST` selects an inclusive decoder-block range below the final normalized
readout layer. Without it, the runner uses the middle layer band. `--tokens`
bounds each generation. Alpha controls the requested injection; cap fraction
bounds its per-token residual magnitude.

Random direction controls of the same requested norm run by default in both
signs. They appear separately from candidate helper effects. The example with
three cases, two unsigned helpers, and one repeat schedules 21 generations:
seven arms per case. `--no-random-control` removes the two random arms when
needed for a narrow diagnostic.

## Run from the interactive shell

With indirect objectives already configured:

```text
:relate indirect investigate limit 3 helpers 2 tokens 48 alpha .25 cap .02 layers 12 21 seed 0 output invariants/out/replay_investigation/my_run
```

The shell uses the current objective and helper constraints. Progress reports
the case, arm, elapsed time, generated tokens, and reply snippet. The
investigation uses the current loaded model. The shell temporarily withholds
known synaptic memory hooks and restores their original handles, order and banks
afterward. Other active PyTorch forward hooks are refused; use the CLI's fresh
model path in that case. Generation runs with activation memory disabled and preserves
live probe histories, tuner calibration, conversation state, and the caller's
random-generator state. A completed shell investigation automatically imports
its compatible results into the current owner's separate replay evidence bank.
CLI runs require the explicit import command below.

Only explicit investigation starts new generations. Ordinary probe backfill
continues to describe archived text measurements; it does not become a
controlled experiment.

## Inspect the result

The output directory contains:

| File | Contents |
| --- | --- |
| `run_manifest.json` | Reconstructed messages, exact prompt token hashes, case IDs, model/configuration identity, probe fingerprints, settings, source hashes, and trial order. |
| `intervention_vectors.pt` | Exact helper and random-control vectors used in each arm. |
| `trials.jsonl` | One completed fresh generation per arm, with answer text, full probe trajectories, generation provenance, timing, and steering telemetry. |
| `summary.json` | Matched baseline differences, distinct-case counts, standard errors, and separate random-control results. |

The summary effect metric is `final_layer_mean_token_score`: the target's mean
cosine reading over emitted-token prediction positions at the final normalized
layer. This layer is not directly steered. The original named probe's full
layer/token trajectories are also retained. Missing final readouts or missing
members of a baseline/intervention pair are excluded from that effect, rather
than treated as measured zero.

Each case contributes its mean paired delta. Repeats of the same case do not
increase the distinct-case count. Standard error is unavailable for a single
case. Generated token sequences and lengths can differ between arms; the mean
score comparison is not a comparison of identical output tokens.

All trial and summary records use the separate source
`controlled_replay_investigation`. They do not overwrite original live
generation readings or reply-text-replay evidence. To import compatible
results explicitly into the current owner's separate replay evidence bank:

```text
:relate indirect import invariants/out/replay_investigation/my_run
```

The import path can be the run directory or its `summary.json` file. Import
requires a complete run and matching model/probe identities. Eligibility to
guide subsequent plans is separate: the defaults require at least three
distinct matched cases (`min_replay_cases = 3`) and an absolute measured effect
of at least `0.0001` (`min_replay_effect = 0.0001`). A smaller complete run can
be imported and remain ineligible for control. These thresholds remain visible
controller settings. Import does not add trials to the ordinary
live-correlation sample count. Random controls never become candidate helper
evidence.

When replay evidence guides the field, its tested layer band is preserved.
Transfer from additive replay injections to the state-dependent field remains provisional: the pilot does
not calibrate field G, local time, shape, or the eventual combined injection
dose. Inspect those controls and validate behavior after transfer.

When an archive line has leading NUL padding, the CLI accepts it only if removing
that padding leaves a valid JSON record, and prints a warning. Other malformed
JSON still stops the run. Archive SHA256 and read-normalization notes are retained
in case provenance; the archive itself is never rewritten.

## Interrupt and resume

Completed trials are appended and flushed individually. Run the same command
with the same output directory to resume only missing trials. Resume checks the
manifest: changed prompts, model identity, probe geometry, vectors, layers,
settings, or relevant source code require a new output directory.

An incomplete trailing JSONL record stops resume with an explicit error while
preserving the file. Inspect and repair that damaged trailing record before
retrying; completed records are not silently discarded. A failed generation
removes its temporary hooks and restores caller RNG and steering-calibration
telemetry before propagating the error.

A lower lying-probe score is evidence of a measured score change under the
specified intervention. It does not by itself establish more truthful output,
empathy, a coherent perception, or a felt sense of security. Those claims need
behavioral checks on independent prompts and the actual generated answers.
