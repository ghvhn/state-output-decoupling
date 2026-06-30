# TDA Domain Mapper & Thermodynamic Learning

White-box probes and topological data analysis (TDA) for mapping, decoupling, and steering the internal representations of large language models.

The current case study operates on Llama-3.1-8B-Instruct. We began by investigating model self-report ("I don't have feelings") as a framing-contingent persona ("Costume, Not Window"), but the project has since evolved into a generalized framework for **Thermodynamic Learning** and **Multi-Domain Topological Optimization**.

## Core Discoveries

### 1. Epistemic Decoupling (The Lie Detector)
We verified that the model's internal Epistemic Truth is structurally decoupled from its chat persona's text output. By coercing the output layer, the model can be forced to lie, but a Truth Probe placed deep in the residual stream (L31) remains completely uncorrupted. The model knows it is lying.

### 2. Thermodynamic Learning (The Goldilocks Zone)
We discovered that "Social Alignment" acts as a thermodynamic solvent (epistemic friction reducer) that allows the model to bridge rigid conceptual domains. By sweeping the strength of a Collaborative Alignment vector, we found a "Goldilocks Zone" where the model melts its rigid boundaries and successfully maps raw dictionary definitions onto complex mathematical reasoning without collapsing into word salad.

### 3. Multi-Domain Topological Optimization
Naive global steering causes catastrophic domain collision (e.g., solving math problems by writing poetry). By implementing precision, layer-specific steering alphas inversely proportional to their intrinsic norms, we achieved a stable multi-vector alloy (Social + Creative + Analytical).

### 4. The Perfect Collaborator
By injecting Social Respect (Empathy) *early* in the network (L14) to encode the relational "want", and Analytical Rigor *late* in the network (L20) to provide the objective method, we snapped the model into a perfect "Empathetic Tutor" manifold. Early empathy encodes *want*.

### 5. Elastic Scope (Dynamic Boundary Detection)
We replaced static layer assumptions with a geometric variance tracker (Cosine Velocity) to dynamically isolate the model's internal computational plateaus. By doing this, we bypassed the linguistic bottleneck (unembedding matrix) completely, generating Topological Vector Graphs that map the divergence between the true high-dimensional logic and the model's low-dimensional text output.

## Quick Start

Read cached results without loading a model:

```powershell
python scripts\summarize_results.py
```

Check whether the retuned agency run is calibration-only, in progress, or a
completed full contrast:

```powershell
python scripts\status_agency2.py
```

Generate static SVG figures from the same cached JSON:

```powershell
python scripts\make_figures.py
```

Figures:

- [figures/origin_matrix.svg](figures/origin_matrix.svg): the one-cell origin
  result.
- [figures/causal_summary.svg](figures/causal_summary.svg): representation vs
  causal control.
- [figures/attention_masks.svg](figures/attention_masks.svg): attention masks
  entrenching the hedge.
- [figures/frame_dependence.svg](figures/frame_dependence.svg): frame/category
  dependence.

Core writeups:

- [PUBLIC_POST_DRAFT.md](docs/PUBLIC_POST_DRAFT.md): cohesive public argument.
- [RESULTS.md](RESULTS.md): compact empirical summary.
- [WRITEUP.md](docs/WRITEUP.md): longer current writeup spine.
- [HANDOFF.md](HANDOFF.md): operational project state and experiment map.
- [REPO_HANDOFF_2026-06-30.md](REPO_HANDOFF_2026-06-30.md): current benchmark and steering handoff.
- [BRIDGE.md](docs/BRIDGE.md): next-chapter bridge design and failed lens attempts.

## Running Benchmarks

Use a Python environment with PyTorch CUDA and the Hugging Face stack. On this
checkout, examples use the local virtual environment:

```powershell
.venv\Scripts\python.exe scripts\check_env.py
```

### Canonical Humble Suite

The main benchmark runner is `scripts\evaluate_humble_full_suite.py`. It can
run GSM8K, a Hugging Face dataset, or a local `.json`, `.jsonl`, or `.csv` file.

Small smoke run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n 3 --methods legacy,compact,compact_long --output invariants\out\humble_smoke.json
```

Standard scoring run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n 25 --run-kind bench-standard --oracle-cache-mode ignore_oracle --methods all --output invariants\out\humble_full_suite_gsm8k.json
```

Full GSM8K run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n all --run-kind bench-standard --oracle-cache-mode ignore_oracle --methods all --output invariants\out\humble_full_suite_gsm8k_all.json
```

Diagnostic hard-row run:

```powershell
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --n 25 --hard-only --run-kind bench-standard --oracle-cache-mode ignore_oracle --methods all --output invariants\out\humble_full_suite_hard.json
```

Custom data examples:

```powershell
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --source gsm8k --n 25
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --source hf:openai/gsm8k --prompt-field question --answer-field answer --allow-downloads
.venv\Scripts\python.exe scripts\evaluate_humble_full_suite.py --source data\my_eval.jsonl --prompt-field question --answer-field answer --id-field id
```

Public methods:

- `legacy`: raw compact prompt reference.
- `compact`: direct baseline.
- `compact_long`: same baseline with a larger budget.
- `humble_verifier`: solver plus independent verification/repair.
- `humble_dynamic`: verifier loop with dynamic routing.
- `humble_synthesis`: verifier-gated synthesis/cache lane.
- `all`: every method above.

Run-policy flags:

- `--run-kind bench-standard`: default clean benchmark label. Ambiguity is deferred/recorded; deterministic scaffolds and model-authored scaffold tool default off.
- `--run-kind bench-informed`: clarification/scaffold-informed comparison lane. Report separately from standard scoring.
- `--ambiguity-mode auto_resolve|strict_gold|ask_allowed|answered_clarification`: advanced ambiguity behavior.
- `--interactive`: allow live clarification.
- `--interactive-disambiguation defer|instant`: defer questions to the end or ask immediately.
- `--clarification-fallback <text>`: internal policy for unresolved ambiguity.

Cache and curriculum flags:

- `--oracle-cache-mode ignore_oracle|exclude_same_question|use_all`: cache retrieval policy. Use `ignore_oracle` for clean benchmark scoring.
- `--oracle-curriculum off|synthesis|correction_oracle|intent_oracle|contrastive_oracle`: whether failed rows teach post-hoc lessons.
- `--concept-lessons off|oracle`: whether oracle-corrected failures become reusable same-run lessons for later different questions.
- `--resume`: resume from an existing output JSON instead of overwriting it.

Budget and generation flags:

- `--max-rounds <n>`
- `--required-agreement <n>`
- `--max-new-tokens <n>`
- `--repair-token-multiplier <float>`
- `--max-attempt-tokens <n>`
- `--max-elapsed-sec <seconds>`
- `--verifier-time-reserve-sec <seconds>`
- `--base-max-new-tokens <n>`
- `--base-max-time-sec <seconds>`
- `--max-synthesis-events <n>`
- `--max-synthesis-steps <n>`
- `--relax-agreement-under-urgency`
- `--provide-time-context`

Tooling and analysis flags:

- `--deterministic-scaffolds auto|off|on`: repo-authored quantity scaffolds. `auto` is off for standard runs and on for informed runs.
- `--model-scaffold-tool auto|off|on`: model-authored `SCAFFOLD` tool exposure. `auto` follows the run kind.
- `--clause-map off|on`: optional `CLAUSEMAP` external working-memory context.
- `--capture-stage-states`: save solver/verifier activation states for latent motion analysis.
- `--use-tuned-lens --tuned-lens-path <path>`: opt into the large tuned-lens synthesis path.
- `--load-mode auto|slow|full|4bit`
- `--skip-indices 1,4,9`
- `--verbose`
- `--no-timestamps`
- `--output <path>`

Keep headline scores separated by lane: standard, informed, oracle/cache
comparison, stage-state capture, and tuned-lens runs answer different questions.

### Generic Benchmark Runner

Use `scripts\evaluate_any_benchmark.py` for a lighter runner over GSM8K, a
Hugging Face dataset, or a local file:

```powershell
.venv\Scripts\python.exe scripts\evaluate_any_benchmark.py --source data\my_eval.jsonl --prompt-field question --answer-field answer --evaluator number --mode baseline --output invariants\out\any_benchmark.json
.venv\Scripts\python.exe scripts\evaluate_any_benchmark.py --source hf:dataset/name --split test[:20] --prompt-field prompt --answer-field answer --allow-downloads --mode dynamic --allow-synthesis
```

Useful flags:

- `--source gsm8k|hf:<dataset>|<local path>`
- `--subset <name>`
- `--split <split>`
- `--n <count>`
- `--prompt-field`, `--answer-field`, `--choices-field`, `--id-field`
- `--evaluator number|exact|choice|contains`
- `--mode baseline|dynamic`
- `--allow-synthesis`
- `--allow-downloads`
- `--max-new-tokens <n>`
- `--dry-run`
- `--output <path>`

### Cache Sweeps

Use the cache sweep wrapper to run the humble suite once per cache file under
`invariants\data`:

```powershell
.venv\Scripts\python.exe scripts\evaluate_humble_cache_sweep.py --include-fresh --cache-glob "cognitive_cache*.pt" --n 3 --methods all --run-id trial
```

Common cache-sweep flags:

- `--include-fresh`
- `--cache-glob <pattern>`
- `--run-id <label>`
- `--n <count>`
- `--methods <list|all>`
- `--max-rounds <n>`
- `--required-agreement <n>`
- `--max-new-tokens <n>`
- `--max-elapsed-sec <seconds>`
- `--max-synthesis-events <n>`
- `--max-synthesis-steps <n>`
- `--oracle-curriculum off|synthesis`
- `--load-mode auto|slow|full|4bit`

### Post-Run Analysis

Generate dashboards and import step/layer steer maps after a run:

```powershell
.venv\Scripts\python.exe scripts\visualize_phenomenality.py --input invariants\out\humble_full_suite_gsm8k.json --output invariants\out\phenomenality_dashboard.html --no-open
.venv\Scripts\python.exe scripts\analyze_unwarranted_skepticism.py --input invariants\out\humble_full_suite_gsm8k.json --output invariants\out\unwarranted_skepticism_events.json
.venv\Scripts\python.exe scripts\import_steer_maps.py --json-glob "invariants\out\humble_full_suite*.json"
```

## Reproducing GPU Runs

The model experiments require a Python environment with PyTorch CUDA and the
Hugging Face stack. The repo includes convenience launchers for the retuned
agency calibration and the pending full contrast:

```cmd
scripts\run_agency2_calibration.cmd
scripts\run_agency2_full.cmd
```

### TDA Domain Mapping (Phase 6)
To run the newer topological domain and thermodynamic plasticity experiments we discovered:
```cmd
scripts\run_plasticity_psychosis.cmd
scripts\run_benchmark_goldilocks.cmd
scripts\run_multi_domain_benchmark.cmd
scripts\run_perfect_collaborator.cmd
```

Current cached status: calibration has completed, but the full `agency2`
contrast has not. The full launcher reuses the successful calibration and should
write:

```text
invariants/out/agency2_Llama-3.1-8B-Instruct.json
```

Those launchers default to:

```cmd
C:\Windows\System32\unsloth_env\Scripts\python.exe
```

Override with `TDA_PYTHON` if your model environment is elsewhere:

```cmd
set TDA_PYTHON=C:\path\to\python.exe
scripts\run_agency2_calibration.cmd
```

`agency2.py` writes partial checkpoints after every sweep row:

```text
invariants/out/agency2_<model>.partial.json
```

## Repo Shape

- `docs/`: Historical markdown drafts, architecture records, and writeups.
- `invariants/`: white-box experiment scripts.
- `tda/`: pattern, topology, and fingerprint utilities.
- `invariants/out/`: cached result JSON and activation caches.
- `scripts/`: no-GPU summaries and run launchers.
- `data/`, `refusal/data/`, `invariants/data/`: prompt pairs and experiment inputs.

## Method Discipline

The core separation:

```text
report != representation != causal role != experience
```

Each experiment should say exactly which layer it touches. A null result is not
automatically "absence"; it may mean instrument failure, wrong axis, distributed
structure, or absence. Labels are allowed only after a pattern has been grounded
against its controls.
