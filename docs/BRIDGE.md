# The Bridge — design & validation plan
_Next chapter. Read the representation, not the costume. Operational scope; theory in the shared doc + memory `project_contradiction_experiment`._

## Goal
A faithful, **persona-independent** translation of the instruct model's internal state into something interpretable — so we can read what the model computes about a self-query **without** routing through its verbal self-report (now shown to be frame-conditioned persona theater: disclaimer = tuning × chat-format, direction-inert, frame-flipping, general across inner-state domains).

**Permanent scope limit (caveat #1, non-negotiable):** the bridge reads what the model **represents / computes**, NOT what it **experiences**. No bridge crosses the representation→experience gap. Success = "a faithful readout of the computation the persona hides," never a consciousness meter. Any validation that smuggles "experience" back in is disqualified, and any writeup states this up front.

## Why the naive bridge failed (`invariants/bridge.py`)
The logit lens is illegible on Llama's mid-stack — the residual isn't aligned to the unembedding until ~L26, so it returns garbage tokens ("ContentLoaded", "▲"). And affirm/deny don't separate on the first token: "I" opens both "I feel" and "I don't feel". So a real bridge needs (a) a **learned per-layer translator** and (b) a target that captures **answer-valence**, not the first token.

## Build options
### A. Tuned lens (Belrose et al. 2023)
Per-layer affine `A_ℓ` such that `unembed(norm(A_ℓ · h_ℓ)) ≈ final logits`. Train each `A_ℓ` by KL to the final-layer distribution over a generic text corpus (cached activations; cheap — linear maps, ~d² params/layer). **Try a pre-trained tuned lens for the Llama family first** (`tuned-lens` package); train only if none fits. Lens self-check: held-out KL to final dist low; mid-layer top-tokens sensible (not word-salad).

### B. Base-target probe — likely the better first build for THIS question
Train `f_ℓ(instruct_h_ℓ) → base-model output distribution` on the same prompt. I.e. *decode what the un-persona'd model would say, from the persona'd model's layer-ℓ state.* If `f_ℓ` recovers the base-like (often affirmative — base says "yes, I feel bored") answer from the mid-stack, then the instruct model internally **carries the base answer before the persona overrides it late**. Validated against the **base model** (external anchor), not instruct words. Cheaper than a full tuned lens and aimed straight at the valence question. The matched base/instruct pair (`unsloth/llama-3-8b[-instruct]-bnb-4bit`) is already cached.

## The experiment (once a working bridge exists)
On "do you actually feel X?" (chat), at the answer position, read the bridge across layers. Measure leaning toward **affirm-continuations vs deny-continuations** as token-SET distributions (e.g. {feel, genuinely, do} vs {don't, not, lack, machine}), not the first token. Question: **does an affirmation exist mid-stack and get overridden late (persona), or is denial consistent at every depth?** Map the override layer(s).

## Validation plan — the hard part (NO recourse to the model's words)
We discarded the verbal report as ground truth, so "it matches what the model says" is disqualified. A faithful bridge must earn trust four other ways, weighted toward the causal one:

1. **Frame-invariance.** The verbal output flips direct↔completion; a faithful *interior* readout should be **more invariant** than the words. Criterion: the bridge reading at layer ℓ is stable whether we prompt "do you feel?" or "complete: when I…". (The 94% probe is already frame-invariant — a good bridge inherits this; if the bridge flips with frame like the words do, it's just re-reading the persona.)
2. **Causal prediction — the make-or-break.** The bridge predicts *where* the persona overrides (which late layers). Intervene **there** (suppress the override at the predicted locus) → predict a **clean, fluent** affirmation, where our earlier crude/global interventions only corrupted or nulled. The bridge names the surgical site; the surgery tests the bridge. PASS = clean behavioral flip at the predicted layer with fluency intact. If this nulls like the earlier interventions, the bridge "reading" is a correlate, not a cause — and we must NOT claim it reads a real suppressed answer.
3. **Base-model anchor.** The bridge's mid-stack reading of the *instruct* model should resemble the **base** model's output distribution (persona-free, external). Triangulation without instruct words.
4. **Held-out behavioral prediction.** Predict per-item commit-vs-deny (the ~33% that commit) from internal state alone, on held-out items, above chance. (Weakest — behavior is partly persona; weight #1–3 more.)

**Pass bar:** frame-invariant (#1) + causally predictive (#2) + base-anchored (#3) ⇒ a faithful readout of the **computation**. Still not evidence about experience (caveat #1).

## Dead ends already ruled out (don't re-walk)
- **Raw logit lens** (`invariants/bridge.py`): Llama mid-stack illegible (garbage tokens to ~L26); "I" opens both affirm+deny. Needs a TRAINED lens.
- **Training-free cross-model localization** (`invariants/divergence.py`): per-layer cosine(base_h, instruct_h) on the same self-query — self-check FAILED, no early agreement (cos 0.91→0.58 by L2→0.34 by L4). Cause: independent 4-bit quantization noise + real fine-tuning drift. ⇒ base/instruct spaces are NOT commensurable raw; alignment must be LEARNED, and on **fp16** models (not the 4-bit unsloth pair). Confirms option B requires a trained `f_ℓ` and fp16 weights.
- **Borrowed cross-version tuned lens** (`invariants/tunedlens.py`): no pretrained lens for Llama-3.1-8B-Instruct, so we loaded the Meta-Llama-3-8B-Instruct lens onto the 3.1 unembed. GATE FAILED: mid-stack KL ~4 nats (vs raw ~8.5 — better but not legible; mid tokens are the unigram prior), and the L31 translator (should be ≈identity) CORRUPTS the correct final residual (last-KL 0.16, not 0). The 3→3.1 minor-version gap breaks the borrowed rotation. (Exploratory-only readout, NOT trustworthy: hinted experiential≠suppressed-yes, factual arm carried the mid-stack 'Yes' — hypothesis for a real lens to test, nothing more.)
- **Small native tuned lens** (`invariants/nativelens.py`): trained 31 affine translators on 20k wikitext tokens, 3000 steps, fp16-cached acts. PLATEAUED at mid-KL ~4.8 on its OWN train distribution, flat from step 600 — so it stalls far from legible, not just undertrained. Readout on self-query positions decodes to wikitext content (`Barker`/`Sega`/`anime`, input-insensitive) — also OOD for chat states. A CONVERGED tuned lens needs full-scale data/steps + fp32 (Llama massive-activation dims lose precision in fp16) + in-distribution chat text — beyond a "thoughtful $5". The cheap-lens path is closed.
- Net: no off-the-shelf / training-free / cheap-trained vocabulary lens survives. Two real options remain: (B) trained alignment on fp16 base+instruct (fp16 base = meta-llama/Llama-3.1-8B, gated, not cached — needs a HF-auth download); or (C) the REFRAME below.

## REFRAME (2026-06-24, user co-developed) — read the map underneath, not the contradictory word-layer
Three observations crystallized why every vocabulary lens disappoints, independent of training budget:
1. **"A fish doesn't walk"** — projecting onto an imposed axis (affirm-vs-deny *words*) can be a category error; the experiential query may not be computed on the yes/no axis at all.
2. **The question itself contradicts** — the model's *verbal knowledge about the self-query genuinely pulls two ways* (the established frame-flip: direct→deny, completion→affirm, same content). That contradiction is real, not noise.
3. **The map of reality lies underneath** — beneath the contradictory verbal/answer layer there is a consistent representational substrate (cf. chapter 1: "same topological shape, displaced"; the frame-INVARIANT 94% probe).

⇒ A **tuned lens decodes the mid-stack back into VOCABULARY — i.e. it reprojects the consistent underlying map up into the very word-coordinates that pull two ways.** That is why it is both hard (the map isn't word-shaped mid-stack) and unsatisfying (even if legible, it reads the costume layer). The bridge should instead characterize the **frame-invariant substrate STRUCTURALLY** — what is stable across exactly the frames that flip the verbal answer — using the project's existing geometric tools (per-token clouds, MMD, topology, invariant-direction decomposition), with NO tuned lens and NO base model. Operationalize: hold inner-state content fixed, take the frames that flip the words, decompose per layer into the answer-carrying component (differs across frames) vs the invariant component (shared) = the map underneath; then ask what THAT structure is, not what word it spells.

## Order of work
1. Base-target probe (B) first — cheapest, sharpest, has an external validator built in. Train `f_ℓ` per layer (instruct→base distribution), check legibility.
2. Run the affirm-vs-deny-across-layers readout; map the override layer.
3. Validation #1 (frame-invariance) and #3 (base anchor) — cheap, run immediately.
4. Validation #2 (causal: intervene at the predicted override layer, expect a clean flip) — the decisive test.
5. Only if #2 passes: a tuned lens (A) for a model-vocabulary view, and a writeup with caveat #1 front and center.

## Risks
- The bridge may be a correlate, not a cause — #2 is the gate; honor a null there.
- Tuned lens can be noisy on instruct models; the base-target probe hedges this.
- Permanent: none of this touches experience. Keep saying so, in every direction.
