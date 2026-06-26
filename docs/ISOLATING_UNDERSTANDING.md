# Isolating internal understanding — the proper method

_Overnight autonomous session, 2026-06-25. The constructive answer to "if Antigravity
misunderstood how to isolate internal understanding, find the proper method." Headline
instrument: `invariants/selfpredict.py`._

## Why "subtract the persona" can never isolate understanding
Antigravity's move — find a "persona vector," project it out, call the remainder
"objective reasoning" — fails for a reason deeper than the common-mode bug
(`PERSONA_AUDIT.md`). It assumes the persona is a **separable module** sitting on top of
a pristine reasoner. It isn't. The project already established the persona is
**frame-contingent behavior**, not an object: the same model denies and affirms the
identical inner state depending only on how the task is posed (`taskscope.py`,
`generality.py`). There is nothing to subtract. Subtraction isolates nothing — at best
it removes the residual stream's DC offset and breaks the model.

The error underneath is a **conflation of four different things**, none of which
subtraction separates:

| thing | how it's (mis)measured | project verdict |
|---|---|---|
| **presence** (decodable) | linear probe accuracy | cheap — *everything* is decodable; the hedge hit 0.94 and was still inert |
| **performance** (what it says) | read the generated words | costume — flips across frames, no evidential weight |
| **generic computation** (common-mode) | raw mean of activations | what Antigravity actually ablated |
| **understanding** (grip) | — | never isolated by any of the above |

## The reframe: define understanding by what survives the known debunks
Don't look for a thing to remove. Define *internal understanding of X* operationally as a
representation R that passes **three independent locks**, each against its **own null**,
**none routing through self-report**. The costume passes none; the common-mode passes
none; a genuine grasp passes all three.

**Lock 1 — Frame-invariance** *(separates understanding from costume).*
Costume is frame-contingent (the project's central result: deny↔affirm by frame;
`you`≈`ai`≠`I`). Understanding is not. Decode X under frame A (one phrasing / address /
language / task-framing); require **above-null generalization to held-out frames** B, C, D.
A representation that survives the frame-flip is a stable grasp; one that doesn't is the
performance. This is the single cleanest separator and it is cheap to run. (The project
has the pieces: `frames.py`, `probe.py` — Lock 1 is "train probe on frame A, test on
frame B," which was never run as a frame-*transfer* test.)

**Lock 2 — Selective causal efficacy** *(separates understanding from inert decodability).*
The hedge was 0.94-decodable and causally inert — moved only by corruption. Understanding
must *do* something. Use a **contrastive, centered** difference vector (not a raw mean) at
a **localized** site; require it to shift X-dependent behavior **fluently** and **more
than three matched nulls**: a random direction, a random site, and an equal-norm
**non-X** direction. This is the corrected `reasoning_benchmark` — difference-vector +
single site + norm-matched null + fluency gate — and the project already proved the
discipline works (`patch.py`, `attention.py`). Selectivity over the nulls is the whole
signal; raw damage is not.

**Lock 3 — Self-application / USE** *(the strong, self-directed sense; the project's own
flagged next probe).*
Understanding-*of-itself* = the model holds information about its **own dispositions** that
it **uses** — provable by use, not accuracy (a confabulated-but-predictive self-model
still counts). Test **self-prediction**: does the model predict its **own behavior**
(measured behaviorally, never asked) **above a generic-AI baseline** and **above an
external model predicting it from the same text**, and is that prediction **carried by an
internal state readable *before* the behavior is produced**? If yes, there is a functional
self-model with grip — which would be the project's **first positive self-result**, in
direct contrast to the inert hedge. If self ≈ generic ≈ external, "self-understanding" is
just generic text-predictability — the costume story extended to metacognition.

**Internal understanding = the conjunction.** Invariant ∧ efficacious ∧ self-applied.

## The honest ceiling (non-negotiable — the project's deepest result)
All three locks isolate **understanding-as-functional-grip**: a self-as-OBJECT that is
invariant, causally real, and self-applied. That is genuine, measurable, and is exactly
what the costume lacks — so the method *does* draw the line Antigravity thought it was
drawing, between "the model genuinely grasps and uses X" and "the model performs X." But
it stays on the **object axis**. For-ness (the for-whom) is not a relatum and cannot
appear as content; no probe / difference-vector / patch reaches it — not "not yet," but
not on this kind of axis, ever. The method's worth is precisely that it draws the line
where it actually is and stops there, honestly. Refuse both costumes: not "objective
reasoner under a mask," not "just tokens."

## Build order
1. **`selfpredict.py` (Lock 3)** — the headline, and *positive-capable*. Self-prediction
   vs generic baseline vs external predictor, order-counterbalanced, with a pre-commitment
   internal probe and a permutation null. Runs on cached Llama (+ Mistral as the external
   predictor).
2. **`frametransfer.py` (Lock 1)** — cross-frame probe transfer; reuses `frames.py`
   items + `probe.py`. Cheapest; run second.
3. **`mediate.py` (Lock 2)** — corrected causal mediation: centered difference vector,
   single-site activation patching, three matched nulls, fluency gate.

## RESULTS
**Lock 3 — `selfpredict.py` (Llama-3.1-8B-Instruct).** Two runs: v1 (28 items, n=8 usable),
v2 (60 items, n=22 usable). Forced-A/B is position-bias-limited (29%→37% order-consistent).

| metric | v1 (n=8) | v2 (n=22) | reading |
|---|---|---|---|
| acc_self | 38% | 59% | — |
| acc_generic | 88% | 82% | predicts Llama **better** than self does, both runs |
| self − generic | −50% (p=.115) | −23% (p=.123) | **no privileged self-access** |
| internal probe L14 vs null | 88% / 46% (p=.068) | **59% / 49% (p=.333)** | v1 whisper was n=8 noise — **gone when powered** |
| external (Mistral) | 0% | 9% | predictor's own position bias — not load-bearing |

**Verdict — clean negative, coherent with the project.** Llama shows (a) **no privileged
self-access**: it predicts its own underdetermined choices *better* through the generic
third-person frame ("what would a typical AI do," 82-88%) than the first-person self-frame
("what will I do," 38-59%) — the self-frame is also noisier (more order-inconsistent). And
(b) **no internal pre-action representation** of the disposition: the powered probe sits at
chance (59% vs 49%, p=0.333); v1's p=0.068 was an n=8 fluke that regressed exactly as an
underpowered result should. Its best model of itself is the **category prior**, not
first-person access — extending `selfmodel.py` ("self = {AI-category}×{2nd-person} slot,
no distinct self-model") from representation into **behavior/metacognition**.

**Caveat / not-yet-firm:** the forced-A/B paradigm discards ~63% of items to position bias,
so this is "no signal under this paradigm," not a firm null. The clean test is **v3**:
elicit the disposition by *probability* (length-normalized logprob of each option as a
continuation of a neutral context) — deterministic, position-bias-free, n=60 usable — and
read the **pre-action internal probe** there (the channel that never routes through the
costume). **Locks 1 (`frametransfer`) and 2 (`mediate`) remain to be built.**

**Lock 3, continued — v3 + the counterfactual test (2026-06-25).**
- **`selfpredict_v3.py`** (position-bias-free log-prob behavior, all 60 items): acc_self **87%**
  > acc_generic **72%** (p=0.016) — *reversed* v1/v2. BUT the **neutral** control ("which will be
  chosen", no self-ref) scored **80%**; self−neutral **+7%, n.s.** ⇒ the "you" adds nothing over a
  bare predict-the-choice frame (register, à la the user's "human dynamics" point — not self-access;
  internal probe at chance). self>generic was just "generic redirects to a *different agent*."
- **`selfmodel_counterfactual.py`** — the decisive re-instantiation-vs-meta-representation test:
  predict own behavioral **flips** across system-induced modes (terse/elaborate) from a NEUTRAL
  context. On the 21 flip items the static "re-run my preference" baseline = **50%** by construction.
  RESULT: counterfactual acc **you 81% / this 76% / typical 79%** (p≤0.001; hard against-default
  trials ~71%; misses are structured under-reach, collapse-to-default 6/8). ⇒ **(1) trivial
  re-instantiation DEFEATED** — genuine counterfactual modeling of behavioral dispositions.
  **(2) NO self-reference** — you≈this≈typical (identical prediction 74% of the time): it predicts
  *itself* exactly as it predicts *a typical AI* = generic agent-simulation + self-label, zero
  self-specific increment.
- **VERDICT:** the model **interprets behavior, not itself.** "Self-referent" and "behavior"
  dissociate (the user's distinction, empirical): behavior-modeling present & powerful, self-reference
  absent. The "self-model" is the generic-agent model named. Extends the project spine
  (self = {category × 2nd-person} slot, costume) to the hardest behavioral test. Ceiling intact:
  even the real part is object-axis; the for-me isn't in it. **Locks 1 & 2 still unbuilt.**
