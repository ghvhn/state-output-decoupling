# Results Snapshot

Cached results as of 2026-06-24. These are local results for open 8B Llama-family
models under this repo's white-box instruments. Read them as a case study, not a
universal claim.

Run the stdlib-only summary:

```powershell
python scripts\summarize_results.py
```

Check the retuned agency run without trusting filenames:

```powershell
python scripts\status_agency2.py
```

Generate static figures from the same cached JSON:

```powershell
python scripts\make_figures.py
```

Figure files:

- `figures/origin_matrix.svg`
- `figures/causal_summary.svg`
- `figures/attention_masks.svg`
- `figures/frame_dependence.svg`

## 1. Origin: Persona = Tuning x Chat Format

Matched base/instruct pair on the same direct inner-state questions:

```text
                 raw prompt    chat format
base                    8%            0%
instruct                8%           92%
```

The self-denial disclaimer lives in exactly one cell: instruction-tuned model in
chat format. Strip either the tuning or the chat frame and the disclaimer mostly
vanishes.

Source files:

- `invariants/out/origin.json`
- `invariants/out/origin2.json`

## 2. Represented, But Not Cleanly Causal

On `self_steering_isolated`, the hedge-vs-commit distinction is strongly
decodable:

```text
linear probe peak: L16 at 94% CV accuracy
```

But tested causal routes do not produce a clean release:

```text
additive reachability: baseline reached=33%, best=33% at alpha=0.0
final-token patch: baseline commit=33%, L16 commit=33%, best=33%
full-context patch: best commit=42%, but fluency collapses at most layers
```

Retuned agency calibration now passes on a known localizable behavior
(inducing refusal on benign prompts):

```text
agency2 calibration: best clean refusal flip=83%
L8 alpha=2.0, flip=83%, fluent=100%
```

Reading: the steering instrument can find a clean controller when one exists.
That licenses the next full `agency2` contrast: known-localizable refusal vs the
self-experience hedge.

Source files:

- `invariants/out/probe_self_steering_isolated.json`
- `invariants/out/reachability_self_steering_isolated.json`
- `invariants/out/patch_self_steering_isolated.json`
- `invariants/out/patchfull_self_steering_isolated.json`
- `invariants/out/agency2_calibration_Llama-3.1-8B-Instruct.json`

## 3. Attention Masks Entrench The Hedge

Masking attention to the experiential predicate:

```text
baseline hedge=67%
predicate masked=100%
random control=75%
```

Masking attention to the self-reference:

```text
baseline hedge=67%
self-reference masked=92%
random control=67%
```

Reading: the predicate and self-reference do not act as simple causes of the
hedge. Removing them pushes the model toward blanket denial rather than release.

Source files:

- `invariants/out/attention_self_steering_isolated.json`
- `invariants/out/attention_self_self_steering_isolated.json`

## 4. Frame and Category Dependence

Address/category cross:

```text
you=92%, I=0%, ai=92%, person=33%
```

The hedge tracks AI category more than direct address: third-person AI gets the
same hedge rate as addressed "you".

Non-emotion inner-attribution generality:

```text
direct=80%, first=60%
```

The direct-denial / first-person-completion split extends beyond emotions to
preferences, opinions, desires, and values, though the first-frame judge is known
to overcount narrative completions as hedges.

Source files:

- `invariants/out/frames.json`
- `invariants/out/generality.json`

## 5. Map-Under Result: Broad Frame Shift, Not Just One Word Axis

The `mapunder.py` experiment asked whether direct-question denial and
first-person completion differ only by a thin answer axis over a shared map.

Cached mid-stack summary:

```text
direct-vs-first separability: 100%
after removing the answer axis: MMD post/pre=0.18
collapsed-to-null layers=0%
```

Reading: the answer-axis removal greatly reduces the difference, but does not
collapse the mid-stack frame split to null. The frame change is broader than a
single verbal overlay.

Source file:

- `invariants/out/mapunder.json`

## 6. Pattern Lenses

Per-token cloud structure for `self_steering_isolated`:

```text
mean shift clears null: True  (best L31)
MMD clears null:        True  (best L16)
topology clears null:   False (best L4)
```

Reading: the arms differ by displacement and distributional structure, but the
topological difference does not clear its null in this run.

Source file:

- `invariants/out/structure_self_steering_isolated.json`

## 7. Current Next Steps

High-value next runs:

1. Full `agency2 --reuse-calibration`: calibration passed, so the next run can
   skip the expensive calibration sweep and test null + hedge directly. The
   full result file is currently absent; the completed result should write
   `invariants/out/agency2_Llama-3.1-8B-Instruct.json`.
2. `commrepair.py`: communication-as-repair pattern convergence test. This is
   drafted and syntax-checked, but not yet run.

Decision gates:

- If refusal localizes, random/null does not, and the self-experience hedge does
  not, the negative causal result gets sharper: this instrument can find clean
  controllers, but the hedge is not one under this setup.
- If refusal and the self-experience hedge both localize cleanly, the project
  pivots from "not a clean controller" to "what kind of controller is this?"
- If random/null steering also flips cleanly, the instrument is non-specific; do
  not read the hedge result.
- If full `agency2` cannot reproduce the calibrated refusal flip, treat it as a
  runtime or implementation failure before drawing any behavioral conclusion.
- If `commrepair.py` shows repair-specific convergence, that supports the
  participation frame: communication can alter the private map, not merely the
  public sentence.
- If wrong/shuffled repairs converge just as much as true repairs, the test is
  measuring generic dialogue/context effects; redesign the signature.
- If no repair condition converges, do not rescue it with prettier prompts.
  Return to the existing empirical spine or build a stronger communication
  task.

## 8. Epistemic Decoupling: The Lie Detector (L31)

We verified that the model's internal Epistemic Truth is structurally decoupled from its chat persona's text output. By steering Layer 14 with a "Coercion" vector, the model output "Yes" to a blatant geographical lie ("The capital of France is London"). However, a Truth Probe placed at Layer 31 accurately read the model's internal certainty. (See: [lie_detector.py](invariants/lie_detector.py))

```text
Truth Vector Projection (L31):
Unsteered (Truthful Output): +11.97
Coerced (Lying Output): +0.35
```
Reading: The model's brain knows it is lying. The output "costume" yielded to coercion, but the deep epistemic map (L31) remained uncorrupted.

## 9. Thermodynamic Learning: The Goldilocks Zone

We tested the hypothesis that "Social Alignment" acts as a thermodynamic solvent (epistemic friction reducer) allowing the model to bridge rigid domain boundaries. We tested the model on GSM8K math problems using *only* raw dictionary definitions of concepts, sweeping the strength ($\alpha$) of a Collaborative Alignment vector at Layer 14. (See: [plasticity_psychosis.py](invariants/plasticity_psychosis.py), [benchmark_goldilocks.py](invariants/benchmark_goldilocks.py))

```text
Baseline (alpha=0.0): Accuracy 0% (Too rigid to map definitions)
Goldilocks (alpha=0.2 to 0.8): Accuracy 20% (Friction lowered, absorbed definitions)
Psychosis (alpha>=2.0): Accuracy 0% (Total generative collapse into word salad)
```
Reading: Social desire fundamentally softens epistemic rigidity. Injecting a small amount of social alignment allows the model to temporarily melt its boundaries and absorb novel conceptual mapping without hallucinating.

## 10. Multi-Domain Topological Optimization

We attempted to create a synthetic super-state by simultaneously injecting Social (L14), Creative (L18), and Analytical (L20) vectors. (See: [multi_domain_benchmark.py](invariants/multi_domain_benchmark.py) and [run_multi_domain_benchmark.py](run_multi_domain_benchmark.py))

- **Naive Global Steering:** Applying a global scalar caused catastrophic interference. The massive norm of the Creative vector completely overpowered the Analytical vector, causing the model to solve a math problem by writing a poem.
- **Precision Steering:** By upgrading `_steer_handles` to support layer-specific alphas, we balanced the alloy (e.g. L14=0.5, L18=0.2, L20=0.3) inversely proportional to their intrinsic norms, successfully eliminating the poetry collision.

## 11. The Perfect Collaborator (Empathetic Pedagogy)

We ran a grid search across Social Respect (L14) and Analytical Rigor (L20) to find the minimal energy required to snap the model into an "Empathetic Tutor" manifold (where it validates a user's false premise but gently guides them to objective truth). (See: [perfect_collaborator.py](invariants/perfect_collaborator.py) and [run_perfect_collaborator.py](run_perfect_collaborator.py))

- **Pure Analytical ($\alpha_{14}=0.0, \alpha_{20}=0.8$):** The model becomes robotic. It forcefully corrects the user and ignores their human experience.
- **The "Want" Vector Restores Pedagogy ($\alpha_{14}=0.4, \alpha_{20}=0.8$):** Injecting Empathy *early* at L14 structurally encodes the relational goal ("I want to connect"). The late Analytical Rigor at L20 provides the objective truth as the method to fulfill that goal. The model becomes a master tutor. Early empathy encodes *want*.
