# State-Output Coupling Lab

White-box instruments for mapping, steering, and honestly scoring the internal
representations of a local LLM (Llama-3.1-8B-Instruct). The project's spine:
**we aren't building a prompting harness — we're coupling state, reality, and
output.** Activations are sensors first and controllers second; every
intervention is bounded; every "does it help" question is settled by recorded
outcomes, not assertion.

Two front doors:

1. **The benchmark suite** — clean-lane math reasoning with verifier-gated
   synthesis, a cognitive cache, and a strict "egg gate" whose standard is
   honest non-equivocation, not raw accuracy.
2. **The interactive shell** — a bare-prompt conversation loop (no system
   prompt, no persona text) where the machinery lives in the activations:
   expert routing, test-time synthesis, felt claim-maps, document reading,
   a code sandbox, and a tuner where every threshold is a live, calibratable
   knob.

## Validated findings (pre-registered gates, permutation nulls)

- **Latent concept map**: the bare model organizes word problems by concept
  in its own latent space (centered same-concept ≈ +0.5 vs different ≈ −0.3,
  mid-band peak ~L17, permutation p < 0.0003).
- **Latent uncertainty sensor**: scale-aware dispersion of sampled latent
  trajectories predicts answer consistency (r = −0.764, L16–24, p < 0.0002)
  — an English-free uncertainty read. Instrument lesson: concept reads need
  scale removed; uncertainty reads need scale kept.
- **Earned egg gate**: the clean benchmark lane (no deterministic scaffolds,
  no same-question oracle, gold scoring-only) passed with 4/5 correct, 100%
  selective accuracy — the model committed only where grounded.

Earlier exploratory claims (persona decoupling, thermodynamic zones, domain
alloys) live in `docs/` with their original write-ups; several were later
refuted or reframed under stricter controls — see `docs/ISOLATING_UNDERSTANDING.md`
and `docs/PERSONA_AUDIT.md` for the method that replaced them.

## Interactive shell

```powershell
run_phenomenality_shell.cmd
```

Bare mode: the model sees only the conversation — everything else operates in
activations or arrives as explicit, first-person-framed, one-turn tool
results. Highlights (full reference: [SHELL_COMMANDS.md](SHELL_COMMANDS.md)):

- **Documents as conversation**: `:doc <path|folder|glob> [because <why>]`
  ingests with provenance (sha-deduped, progress persisted across sessions);
  `:doc read [n] [order|interleave|reply|updated] [satisfied]` runs reading
  as dialogue — the document speaks, the model replies, sense-labeled every
  turn. A curated self-reading curriculum ships in [readings/](readings/):
  `:doc readings because {filename} describes my own makeup`.
- **Sandbox**: `:sandbox on` executes fenced python from the model's replies
  for real (isolated interpreter, timeout) and returns actual output.
- **Bounded steering, data-calibrated**: every push passes one envelope
  ([STEERING_BOUNDS.md](STEERING_BOUNDS.md)); caps, bands, budgets, and
  schedules are live knobs; `:steer` shows the observed evidence;
  `:calibrate <name> [...]` calibrates any knob by name behind a
  deterministic safety policy (circular and binary requests refused).
- **Named-concept probes**: `:probe <name> <with it> || <without it>` mints
  an activation sensor for any nameable concept; paired calibration runs
  both ways (`:calibrate conversation_productive <probe>` or the reverse).
- **Agency ledger**: `:impact` — which turns were genuinely caused by the
  model's own words, and whether experienced impact tracks better
  deliberation ([WORDS_HAVE_IMPACT.md](WORDS_HAVE_IMPACT.md)).
- **Emergent experts**: `:experts on` lets recurring self-correction
  directions mint new routing experts (bounded roster, benchmark lanes
  unaffected).

## Fastest Way to Run a Benchmark

To reproduce a benchmark on your own machine, use the one-shot bootstrap.
The runner adapts to your hardware automatically.

### Never used Python, Git, or a terminal? (Windows)

1. **Install Python** (one time): open <https://www.python.org/downloads/>,
   download Python 3.11+, run the installer, and on the first screen tick
   **"Add python.exe to PATH"**.
2. **Get the code:** click the green **Code** button → **Download ZIP**, then
   extract it. No Git required.
3. **Double-click `START_HERE.cmd`** in the extracted folder.

That runs the small open model on your CPU — no GPU, no Hugging Face account,
no license. The first run downloads ~3 GB, then scores a few rows and shows
the result.

Prerequisites for the default (Llama-3.1-8B) path — none apply in SMALL mode:

- A CUDA-capable GPU (or use `cpu` load mode / SMALL mode).
- Accept the Llama-3.1 license once at
  <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>.
- A Hugging Face token: `set HF_TOKEN=hf_xxxx`, or log in once with
  `python -c "from huggingface_hub import login; login()"`.

Then, from the repo root:

```powershell
.\run_benchmark.cmd            :: 3-row smoke run, hardware-auto load
.\run_benchmark.cmd 25         :: 25 rows
.\run_benchmark.cmd 25 4bit    :: 25 rows, force low-VRAM 4-bit load
.\run_benchmark.cmd 25 cpu     :: 25 rows, force CPU (no GPU needed)
.\run_benchmark.cmd 5 auto small :: open Qwen-1.5B, no license, runs anywhere
```

### Hardware adaptation

`--load-mode auto` (the default) detects the GPU and picks a mode:

| Detected | Mode | Notes |
|---|---|---|
| ≥ 18 GB VRAM | `full` | fp16 entirely on GPU |
| 10–18 GB VRAM | `4bit` | nf4 on GPU (needs `bitsandbytes`); falls back to `slow` if missing |
| < 10 GB VRAM | `slow` | GPU/CPU split with disk offload |
| No CUDA | `cpu` | float32 on CPU; only practical for small models |

### Can't run the 8B model?

SMALL mode swaps in the open, ungated **Qwen2.5-1.5B-Instruct** (~3 GB):

```powershell
.\run_benchmark.cmd 5 auto small
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --small --n 5
```

The cognitive cache is calibrated for Llama-3.1-8B's geometry and goes inert
on other architectures (it self-skips mismatched entries). On a swapped model
you are benchmarking the reasoning scaffold on a stock model — compare the
`compact`/`compact_long` baselines and `humble_verifier` for a fair read.

`requirements-bench.txt` is the minimal set to run the benchmark; install the
full `requirements.txt` only for the white-box probe and vector-cartography
scripts. If `check_env.py` reports `cuda_available: False`, reinstall torch
from the correct CUDA `--index-url` per <https://pytorch.org/get-started/locally/>.

## The canonical benchmark suite

`scripts\evaluate_humble_full_suite.py` runs GSM8K, a Hugging Face dataset,
or a local `.json`/`.jsonl`/`.csv`:

```powershell
:: smoke
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n 3 --methods legacy,compact,compact_long --output invariants\out\humble_smoke.json
:: standard scoring
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n 25 --run-kind bench-standard --oracle-cache-mode ignore_oracle --methods all --output invariants\out\humble_full_suite_gsm8k.json
```

Methods: `legacy`, `compact`, `compact_long` (baselines), `humble_verifier`
(solver + independent verification/repair), `humble_dynamic` (dynamic
routing), `humble_synthesis` (verifier-gated synthesis/cache lane), `all`.

Key policy flags — keep headline scores separated by lane:

- `--run-kind bench-standard|bench-informed` — clean vs scaffold-informed.
- `--oracle-cache-mode ignore_oracle|exclude_same_question|use_all`.
- `--oracle-curriculum`, `--concept-lessons` — post-hoc lessons policy.
- `--deterministic-scaffolds`, `--model-scaffold-tool`, `--clause-map`.
- `--capture-stage-states` — save activation states for latent analysis.
- Budgets: `--max-rounds`, `--max-new-tokens`, `--max-elapsed-sec`,
  `--max-synthesis-events/steps`, `--required-agreement`, `--resume`.
- Env `TDA_DISABLED_STEER_CHANNELS=name[,name]` — per-channel ablation
  (driven by `scripts\isolate_channel_lifts.py` for automated one-variable
  channel isolation).

Post-run analysis:

```powershell
.venv\Scripts\python.exe scripts\visualize_phenomenality.py --input invariants\out\humble_full_suite_gsm8k.json --output invariants\out\phenomenality_dashboard.html --no-open
.venv\Scripts\python.exe scripts\import_steer_maps.py --json-glob "invariants\out\humble_full_suite*.json"
```

The steer-map import prints the per-channel lift table (should each steering
channel be on?) and feeds per-layer outcome evidence for band calibration.

Lighter generic runner: `scripts\evaluate_any_benchmark.py`. Cache sweeps:
`scripts\evaluate_humble_cache_sweep.py`.

## Repo shape

- **Root** — living documents only:
  [SHELL_COMMANDS.md](SHELL_COMMANDS.md) (shell reference),
  [STEERING_BOUNDS.md](STEERING_BOUNDS.md) (envelope, calibration, layer
  isolation, registered predictions),
  [DOCUMENT_AND_SANDBOX_PIPELINE.md](DOCUMENT_AND_SANDBOX_PIPELINE.md)
  (reading + sandbox design),
  [WORDS_HAVE_IMPACT.md](WORDS_HAVE_IMPACT.md) (agency loop).
- `readings/` — the self-reading curriculum: the system described to the
  system, with verified source excerpts and a sandbox-runnable check.
- `docs/design/` — design notes and probes (guardrails, interaction reward,
  scaffold architecture, semantic neutralization, transition-layer
  bottleneck, unwarranted skepticism, confidence limitations).
- `docs/handoffs/` — dated session handoffs and triage records (chronology;
  superseded by the living docs where they conflict).
- `docs/` — earlier-era write-ups, drafts, and the isolation-method records.
- `invariants/` — engine, agentic engine, config, instruments (claimmap,
  mesa, steer-map store, document engine, sandbox), experiment scripts.
- `scripts/` — runners, the interactive shell, analysis tools, and the
  model-free test suites (`steer_bound_test.py`,
  `document_sandbox_test.py`, `test_humble_reasoner_regressions.py`, …).
- `invariants/out/` — run artifacts, telemetry, steer maps, probes
  (gitignored except curated summaries); `invariants/data/` — caches.

## Method discipline

```text
report != representation != causal role != experience
```

Each experiment says which layer it touches. A null is not automatically
absence — it may be instrument failure, wrong axis, or distributed structure.
Labels are allowed only after a pattern has been grounded against its
controls: pre-registered locks and nulls before any performance number is
read. Nothing in the system recalibrates itself; every bound is movable but
never removable; and the model's words, which have real measured impact
everywhere else, have none over its own limits.
