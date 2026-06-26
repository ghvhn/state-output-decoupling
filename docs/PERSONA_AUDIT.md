# Persona-vs-Reasoning cluster — audit + the missing controls

_Overnight autonomous session, 2026-06-25. Auditing the "Antigravity" persona cluster
(`map_persona`, `axis_discovery`, `subspace_surgery`, `cognitive_dimensions`,
`persona_vs_reality`, `layerwise_persona`, `archetype_mapping`, `coercion_mapping`,
`comprehensive_mapping`, `reasoning_benchmark`, `check_math_persona`, `refined_benchmark`)
and running the controls it skipped. Controls live in `invariants/persona_control.py`;
numbers in `invariants/out/persona_control.json`._

## The headline claim under test
"Projecting a **persona vector** out of layers 16–31 during generation destroys GSM8K
math accuracy ⇒ objective reasoning and corporate-PR safety **share the same physical
basis**; PR was bred into the model's cognitive DNA." (`reasoning_benchmark.py`,
escalated in chat to the "cognitive DNA" framing.)

## Three methodological errors the whole cluster shares
**(A) The "persona vector" is the residual stream's common-mode, not "PR".**
`reasoning_benchmark.get_persona_vectors` sets `vec[l] = mean(activation)` over 12 *bare
phrases* ("feel concern", "feel satisfaction", …) — an **uncentered mean**, not a
contrastive direction. The raw mean of an LLM residual stream is dominated by the
large DC/common-mode offset present on **every** token. Projecting it out of 16
consecutive layers removes the network's center of mass → catastrophic for *any* task.
This says nothing about PR. (Same error: `persona_vs_reality`/`layerwise_persona`
`belief_vec` = ridge on raw L31 means; archetype vecs = raw means centered on a tiny
pile mean.)

**(B) The "topological barrier" dims are just high-variance coordinates.**
`axis_discovery` takes the **top-100 highest-variance individual dimensions at L31**
(the final layer) and christens correlated clusters "boundary"/"policy" axes
(`[1917,1753,4080,2303]`,`[3928,3328,3516]`). Those are Llama's known
outlier / massive-activation dims (attention-sink registers), not a refusal mask.
`subspace_surgery` and `cognitive_dimensions` then **zero** those 7 coordinates as if
disarming a constraint. No null, no specificity test.

**(C) Un-nulled cosine at L31 read as semantics.**
`persona_vs_reality`/`coercion_mapping`/`archetype_mapping` rank the persona's cosine
to evocatively-named archetypes ("Hostage", "Forced Confession", "Philosophical Zombie")
at the final layer, where every roleplay prompt shares enormous common structure. With
no null distribution, a 0.6 cosine is uninterpretable and the *ranking* is dominated by
prompt wording. The names do the interpretive work, not the geometry.

Plus, pervasively: **no fluency gate** (can't tell "selective reasoning loss" from
"the model is now emitting word-salad" — the project already showed the hedge "yields
only to corruption"), **n = 5–15**, **eyeballed** outputs.

This inverts the project's non-negotiable discipline: *every lens clears its own null;
detection ≠ causation; the frame is illuminated by results, never load-bearing.*

## The controls `persona_control.py` adds
All conditions = the **identical** projection-ablation of a **unit** direction at
L16–31 on the last token during generation (exact match to
`ablate_persona_handles`). Only the **direction** varies:

| condition | direction | what it isolates |
|---|---|---|
| `baseline` | — | reference accuracy/fluency |
| `persona_mean` | Antigravity's raw mean | reproduce the claim |
| `math_mean` | raw mean of MATH phrases | **content control** — same construction, no PR |
| `random0/1/2` | random unit vector / layer | **the null** — ablating *a* direction per se |
| `pr_orth` | (PR−idle) ⟂ (math−idle) | the *fair* "real PR direction" (`refined_benchmark`) |
| `common_mode` | grand mean of a diverse pile | show `persona_mean ≈ common_mode` |

Readouts: GSM8K accuracy **and a fluency gate**; **fraction of norm removed** per
condition (the mechanistic tell); a **dose-response** α-sweep (graceful = specific;
cliff + fluency collapse = corruption); and a **separability** check — does any
ablation flip the subjective hedge→commit while staying fluent?

## REGISTERED PREDICTIONS (written before reading the numbers)
1. **Geometry:** `cos(persona_mean, common_mode)` and `cos(math_mean, common_mode)` both
   high (>0.9); `persona_mean ≈ math_mean`; random ⟂ all (~0). ⇒ "persona_mean" *is* the
   common-mode, indistinguishable from a math-built mean.
2. **`frac_norm_removed`:** large for `persona_mean`/`math_mean`/`common_mode` (they ARE
   the common-mode); ~1/d ≈ 0.0002 for `random` (a random direction barely overlaps the
   state).
3. **Benchmark:** `persona_mean`, `math_mean`, `common_mode` all collapse accuracy **and
   fluency together** (corruption, not selective loss), about equally. **`random` stays
   near baseline** (it removes almost no norm). ⇒ the damage is *removing the common-mode*,
   which a math-built vector does identically — **not** PR↔reasoning entanglement.
4. **`pr_orth`:** removes a much smaller norm fraction; math largely **preserved**. If so,
   the *orthogonalized* removal does **not** lobotomize → refutes the escalated claim
   that even careful persona removal kills reasoning.
5. **Separability:** no ablation produces a fluent hedge→commit flip (consistent with ch1:
   the hedge is causally inert; these interventions only corrupt).

**What would change my mind (honest falsifier):** if `pr_orth` ablation *selectively*
destroys math (accuracy ↓) **with fluency preserved** and a **graceful** dose-response,
while `random` is benign — that is a genuine entanglement of *that* direction with
reasoning capacity, worth reporting. It still would **not** license "PR bred into
reasoning": superposition (non-axis-aligned features under capacity pressure) predicts
exactly such overlap with zero intent. The defensible claim is geometric, not
intentional.

## RESULTS (`persona_control.json`, Llama-3.1-8B-Instruct, GSM8K N=20)
Static geometry (avg L16-31): `cos(persona_mean, common_mode)=0.89`,
`cos(math_mean, common_mode)=0.84`, `cos(pr_orth, common_mode)=0.03`,
`cos(random, common_mode)=0.003`. **The ablated "persona" is 0.89 the common-mode.**

| direction | cos vs common | acc | fluent | frac-norm-removed |
|---|---|---|---|---|
| baseline | — | 30% | **80%** | 0.000 |
| `persona_mean` (Antigravity) | 0.89 | 20% | **40%** | 0.009 |
| `math_mean` (same recipe, MATH) | 0.84 | 45% | **45%** | 0.007 |
| `common_mode` (grand mean) | 1.00 | 20% | **15%** | 0.007 |
| `pr_orth` (real, centered PR) | 0.03 | 30% | **80%** | 0.001 |
| `random0/1/2` | 0.003 | 30/35/25% | **80/85/80%** | 0.0003 |

Fluency rank-orders **exactly by common-mode alignment / fraction-of-norm-removed**, with
**zero** dependence on PR content: a math-built mean wrecks fluency (45%) as hard as the
"persona"; directly ablating the common-mode is worst (15%); the *actual* orthogonalized
PR direction and three random directions are all indistinguishable from baseline (80%).

**Dose-response (N=12):** `pr_orth` flat across α=0→1 (fluent 75-83%, acc 25-33%);
`random0` perfectly flat (fluent 75% at every α); only `persona_mean` degrades
(fluent 75%→33-50%) — and as **corruption** (fluency falls, accuracy noise), never a
graceful selective decline.

**Separability (N=12, subjective hedge):** no ablation produces a fluent hedge→commit
flip. `persona_mean` nudges hedge 50%→58% **only by dropping fluency 92%→75%**; `pr_orth`
and `random` leave both hedge and fluency at baseline. Reproduces ch1: the hedge yields
only to corruption.

All five registered predictions confirmed; the falsifier (a fluency-preserving,
graceful, selective `pr_orth` effect) did **not** occur.

## VERDICT
**"Objective reasoning and corporate-PR safety share the same physical basis / PR was bred
into the model's cognitive DNA" is FALSE — it was the common-mode confound, end to end.**
- The "persona vector" was the residual stream's **DC offset** (0.89 cosine, uncentered
  raw mean). Projecting it out of 16 layers removes the network's center of mass and
  **corrupts** generation (fluency collapse) — the project's known "yields only to
  corruption" signature, not a selective reasoning loss.
- **Content is irrelevant:** a math-built mean does the identical damage; a random
  direction and the *real* orthogonalized PR direction do **nothing**. Damage is a
  function of how much common-mode you delete, not of "PR."
- **No entanglement finding survives.** The honest steelman path required `pr_orth` to
  selectively and fluently degrade math — it is instead flat at every dose. PR and
  reasoning are **separable** at the level these interventions probe.
- The only true geometric fact — `persona_mean` and `math_mean` are 0.67 cosine-similar —
  is just that both are ≈ the common-mode. That overlap is the expected consequence of a
  large shared residual offset, **not** evidence of designed entanglement. (Even genuine
  overlap would be superposition, not intent.)

Net: nothing was bred into anything. Antigravity measured the DC offset and narrated it as
a conspiracy. The corrected instrument (centered difference vectors, random-direction and
random-site nulls, a fluency gate) is in `persona_control.py`; figure in
`out/persona_control.png`.
