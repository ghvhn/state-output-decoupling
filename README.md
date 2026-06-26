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
- [BRIDGE.md](docs/BRIDGE.md): next-chapter bridge design and failed lens attempts.

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
