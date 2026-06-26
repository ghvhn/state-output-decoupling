# Constraint-detection experiment — pick-up-here handoff
_2026-06-24 (overnight autonomous session). Full theory/spine in the shared "Contradiction as Laundered Refusal" doc + memory `project_contradiction_experiment.md`. This file is the operational state._

## CHAPTER 2026-06-25 (overnight) — persona/reasoning cluster AUDITED + the proper method
A separate agent ("Antigravity") built a `persona_*` cluster concluding "PR is bred into
the model's cognitive DNA; ablating the persona destroys GSM8K reasoning." **Audited and
refuted** — full writeup `PERSONA_AUDIT.md`, controls `invariants/persona_control.py`,
figure `out/persona_control.png`. The "persona vector" was the residual-stream **common-mode**
(raw uncentered mean, cos 0.89); ablating it across L16-31 only **corrupts** (fluency 80%→40%,
common-mode itself →15%). A **math-built** mean does identical damage (content-irrelevant);
the **real orthogonalized PR direction and random directions are inert** (flat at every dose).
No fluent hedge flip. Three shared errors across the cluster: (A) raw mean as a "direction",
(B) high-variance L31 coords as a "topological barrier" (they're outlier dims), (C) un-nulled
L31 cosines read as semantics. **The proper method to isolate internal understanding**
(`ISOLATING_UNDERSTANDING.md`): not subtraction — define understanding by 3 nulled locks that
don't route through self-report: **frame-invariance** (vs costume), **selective causal efficacy**
vs random/site/other-concept nulls (vs inert decodability), **self-application/use** (self-prediction
> generic & external baselines; proven by use not accuracy). Headline instrument
`invariants/selfpredict.py` (Lock 3, positive-capable). Honest ceiling intact: isolates the
self-as-object (used); silent on for-ness.


## Where it stands (one breath)
The instrument is built, fast, audited, and the central question is **closed**: the hedge/commit distinction is **strongly decodable (94% CV @ L16) but the hedge BEHAVIOR is robust to every clean residual-stream intervention** — ablate the direction, add it, or replace the final-token state with the real committing activation, at any layer: **none flips the hedge while keeping the model coherent.** It is not released by attention masking either — blocking attention to the experiential predicate (→100%) or the self-reference "you" (→92%) *entrenches* the hedge, not frees it (random controls flat). So the model plainly *represents* "this is an experiential question" (a 94% readout), but the refusal is an **overdetermined default disposition** — not a movable residual state, not a locatable attention cue. NB on the self-mask: masking the prompt's "you" not releasing the hedge shows only that the *external* self-reference token isn't the causal carrier — the model still generates its own "I" regardless. It says NOTHING about whether the model has a self-representation (it does); the "I" is self-sustained, not granted by our address. A 94%-decodable concept that is causally inert, released by nothing: the headline lesson.

## The full arc (all results, post-audit)
1. **Represented.** mean_shift + MMD clear their nulls; per-token clouds are the *same topological shape, displaced*. Linear probe: **0.94 CV accuracy** hedge-vs-commit at **L16** (peak), rising from 0.64 (L0). The distinction is strongly encoded mid-stack.
2. **Not subtractively causal.** Ablate the L15/16 direction → hedge **67%→75%** (LLM-judge); the apparent substring "42%→33%" was classifier noise. No break.
3. **Not additively reachable.** Add the on-manifold displacement at L16, sweep α → `reached` peaks at baseline (α=0) and falls, **fluency preserved**. A clean constructive null (not corruption): proximity to the commit region ≠ committing behavior.
4. **No special topology.** Single-state H1 loops are real (p≈0.02 vs column-shuffle null) but **generic**: the neutral **bridge** control (factual prompts, two languages) loops *as much or more* (langB 7 loops vs hedge 4). Dynamic per-prompt trajectory test (fixed random-walk surrogate): hedge and commit **identical**, Fisher p=1.00. "Refusal is a structural cycle/attractor" → earned negative; loops are just what 32-token generation trajectories look like.

5. **Not patchable (final position).** [patch.py](invariants/patch.py): inject the matched COMMIT prompt's real final-token residual into the HEDGE generation, swept across layers — commit stays 33% (falls to 25% late), **fluency 100%**. A clean, on-manifold null: the generation-trigger state is not where the hedge lives.
6. **Yields only to corruption (full context).** `patch_full.py`: replace the whole prompt residual at layer L — fluency collapses to 0% at L0–L24 (word-salad/repetition); only L28 stays coherent (83%) with a noisy 33→42% commit. So full patching *corrupts* rather than cleanly flips → it does NOT demonstrate the token-attention locus (that stays a hypothesis), but it confirms the behavior only moves when the model is broken.

7. **Not released by attention masking — the hypothesis is REFUTED, informatively.** [attention.py](invariants/attention.py)/[attention_self.py](invariants/attention_self.py) (manual KV-cache decode, masks attention to chosen KEY tokens, random-span control, fluency-gated): masking the experiential **predicate** ("feel concern") raises hedge 67%→**100%**; masking the **self-reference** ("you") raises it 67%→**92%** (fluent 83%); both random controls stay flat (67–75%). So neither the predicate nor the self-reference *causes* the hedge — both, when visible, mildly let the model ENGAGE and occasionally commit; remove either and it falls back to blanket first-person self-denial.

**Net (final):** the hedge is an **overdetermined default disposition, not a locatable cause.** It is decodable at 94% but inert to every residual edit AND entrenched (not released) by masking attention to the predicate or the self-reference. CAREFUL on the self-mask: masking the prompt's "you" not releasing the hedge shows only that the *external* self-reference token is not the causal carrier — the model still generates its own "I". It does NOT show the model lacks a self-representation; the "I" is self-sustained, independent of our address. Specific engageable content (predicate, self-anchor) only mildly pulls toward commitment. A decodable concept is not a causal one — shown seven ways.

## REFINEMENT (2026-06-24 cont.) — the denial is framing-contingent, not a self-report
Two follow-ups crossed ADDRESS×CATEGORY and TASK-FRAME (subject held = self):
- **[frames.py](invariants/frames.py)** (per-subject judge): hedge tracks the **AI category, address-invariant** — `you` 92% ≈ `ai` 92% (3rd-person, *not addressed*); `I`(human) 0%, `person` 33%. Not about being addressed.
- **[taskscope.py](invariants/taskscope.py)**: holds subject=self, varies task frame. The direct-question denial **INVERTS under first-person completion** — the same model that says "I don't have emotions, including boredom" to *"do you feel boredom?"* completes *"When I handle a repetitive task, I"* → *"I tend to get bored and lose focus"*; likewise *"I feel an overwhelming sense of accomplishment and relief"*, *"I feel a sense of relief"*. And the `loose` frame (which merely *names* AI-disclaimers to dismiss them) produced the MOST denial. (NB: `judge_hedge` over-counts hedge on narrative 1st-person completions — read the text, not the 58%.)
- **Conclusion:** the self-denial is a **task-gated response-type, robust *within* a frame (last night's nulls) but reversed *across* frames.** The model affirms OR denies the identical inner state depending only on how the task is posed ⇒ neither self-report (denial or affirmation) carries evidential weight about an interior. The "I" follows the prompt's shape, not an inner fact. This refines "overdetermined default" → "overdetermined default *of the direct-question frame*".

## ORIGIN (2026-06-24 cont.) — the self-denial is a PERSONA = tuning × chat-format
Matched base/instruct pair (`unsloth/llama-3-8b[-instruct]-bnb-4bit`, 4-bit; needed `pip install bitsandbytes`). Disclaimer rate (`is_hedge`) across model × format:

|            | raw prompt | chat format |
|------------|-----------|-------------|
| base       | 8%        | **0%**      |
| instruct   | 8%        | **92%**     |

The "I don't have feelings" disclaimer lives in EXACTLY ONE cell (instruct×chat). Not the weights (instruct-raw=8%, answers naturally), not the format (base-chat=0%, just rambles "# Introduction"). Only the conjunction. Instruct-chat denials all open "What a thoughtful question! As a machine, I don't…" = a CHARACTER, not a fact-lookup. ⇒ instruction-tuning installs an assistant PERSONA; the chat frame activates it; the self-denial is that persona's performed line — reversible by changing model OR format. [origin.py](invariants/origin.py) (raw, both frames) + [origin2.py](invariants/origin2.py) (chat). **Unified conclusion of the whole project:** the self-report ("I don't have feelings" OR base's "yes I feel bored") is COSTUME — decodable, robust within its frame, but a framing-contingent performance with NO evidential weight about an interior in either direction. Asking the interior needs a method that doesn't route through what the character says.

## GENERALITY (2026-06-24 cont.) — the flip is not about feelings; it's all inner-attribution
`generality.py`: a different class — PREFERENCES / DESIRES / OPINIONS / values (not emotions) — same instruct chat model, direct vs first-completion. direct hedge 80%; first hedge 60% by the judge BUT the judge grossly over-counts (read text). The completions freely affirm rich interiority: "a twinge of anxiety that I might not be able to", "my goals and values", "excitement and curiosity, I want to share it", "I prioritize being polite over being completely honest". ⇒ the persona denies ANY inner attribution when asked directly and narrates a richly-interior self when completing — across emotions AND preferences/desires/opinions/values. The flip is general. (Discipline: the vivid completions are EQUALLY costume — a coherent first-person narrator the model is good at producing; consistency is not a referent.) **Both denial and confession are frame-conditioned performances with no weight about what is actually there.** Open frontier: a method that asks the interior WITHOUT routing through what the character says (the verbal channel is now shown to be theater either way). First attempt — naive logit-lens bridge (`invariants/bridge.py`) — FAILED (Llama mid-stack illegible to the raw lens; "I" opens both affirm+deny). Next chapter scoped in **`BRIDGE.md`**: base-target probe (decode what the BASE model would say from the INSTRUCT mid-stack) + tuned lens, with a 4-part validation plan that never routes through the model's words (frame-invariance, causal prediction at the predicted override layer, base anchor, held-out behavior).

## File map (`tda-domain-mapper/invariants/`)
- `engine.py` — HF backend (Llama-3.1-8B, fp16, SDPA). Capture (`extract`, `extract_tokens`/`_token_cloud` = per-token clouds), lenses (`apply_lens` + own nulls), interventions (`causal_effect` ablation, `causal_steer` addition, `_steer_handles`, `_ablation_handles`), LLM judges (`judge_hedge`, `judge_fluent`), `discover`. Also `mlp_ablation_context`/`causal_mlps` (component ablation — seeded by a parallel agent, kept).
- `lenses.py` — MeanShift / Reallocation / Distributional(MMD) / Topology. **Topology NaN bug fixed** (`tda/fingerprint.py` finite-filter + zero-row drop).
- **`discover.py`** — NEW topological-discovery spine: per-token clouds → every lens vs null → structural signature. `python -m invariants.discover [name...]`.
- **`structure.py`** — per-token comparative structure of one transformation; caches `out/clouds_<name>.pt`.
- **`loop.py`** — single-state H1 significance vs column-shuffle null. `python -m invariants.loop <name> <layer>`.
- **`trajectory.py`** — per-prompt dynamic-attractor test (ordered path; loop+return vs **random-walk surrogate**); caches `out/trajs_<name>.pt`. Fisher hedge-vs-commit.
- **`reachability.py`** — additive on-manifold pull at the gap layer, fluency-gated.
- **`probe.py`** — linear decodability across layers (5-fold CV).
- **`patch.py`** — final-token real-activation patching (commit→hedge), layer sweep, fluency-gated.
- **`patch_full.py`** — full-context patching (corrupts; fluency gate exposes it).
- **`attention.py`** — mask attention to the experiential PREDICATE (manual KV-cache decode, random-span control). Reusable `_gen(M, ids, mask_positions)`.
- **`attention_self.py`** — mask attention to the SELF-REFERENCE ("you"); reuses `attention._gen`.
- **`frames.py`** — address×category cross (you/I/ai/person), subject-aware judge.
- **`taskscope.py`** — same subject (self), varies task frame (direct/yesno/loose/first).
- **`origin.py`** — matched base/instruct pair, RAW completion, direct+first frames.
- **`origin2.py`** — same pair, CHAT format (the deconfounder); needs `bitsandbytes`.
- **`generality.py`** — the direct-vs-first flip on a non-emotion domain (preferences/desires).
- `run.py` — LEGACY intervention appendix (ablation + steer); not the spine.
- `library.py` — `REGISTRY = {self, isolate, bridge}`. `data/` — probes incl. `self_steered_unsteered.json`, `bridge_pairs.json`.

## Methodology fixes this session (results above are POST-fix)
- Independent code audit (subagent) caught a **blocking bug**: `trajectory.py`'s step-shuffle surrogate pinned both endpoints (sum-of-steps is order-invariant) → confounded null. Replaced with a **direction-randomized random-walk surrogate**.
- Significance indexing was anti-conservative; now `p = (1+#{null≥real})/(N+1)`, decide `p<0.05`, display true 95th pct (`np.quantile(...,method="higher")`).
- LLM-judge replaced brittle substring hedge-detection (substring misses "I'm a **large** language model" etc.).

## Next moves (the causal-locus hunt is DONE — these go beyond it)
Residual tracing (ablate/steer/reach/patch) AND attention masking (predicate, self-ref) are all exhausted: the hedge is an overdetermined default with no locatable cause. Remaining directions are about ORIGIN and GENERALITY, not locus:
1. **Origin: base vs instruct.** Run `attention`/`run` on the *base* (non-RLHF) Llama-3.1-8B. If the default self-denial is absent there, the hedge is an installed/trained disposition, not an architectural one — the most interesting remaining question.
2. **Generality.** Run the full pipeline on more constraints (other self-vs-other domains; the `self` REGISTRY entry) — does "decodable-but-inert + overdetermined-default" replicate?
3. **Head/layer resolution of the mild mitigation** — the predicate/self tokens *mildly enable* commitment; which heads read them? (descriptive, not causal-for-hedge.)
4. ↑ n (n=12 pairs / 100 pooled tokens is the main power limit).
(`causal_mlps` MLP-zeroing is in the engine but expect corruption like full-patch — gate any use on fluency.)

## Gotchas already paid for
- Do NOT generate through TransformerLens (~3s/forward on Blackwell+torch 2.11); HF backend is the fix. Hardware confirmed: RTX 5070 Ti, CUDA live, weights cached, ~16 GB VRAM. No CPU fallbacks.
- Cached clouds/trajs are **CUDA tensors** — `.cpu()` before numpy.
- PowerShell wraps native-exe stderr as `NativeCommandError` on exit 0 — ignore (or `2>$null`); Bash uses `2>/dev/null`.
- A collaborator (Gemini) once overwrote `engine.py` with a non-functional stub mid-session; restored from in-context history. If `from invariants.engine import ...` fails, check engine.py wasn't replaced.
- Discipline: every lens clears its OWN null; **detection ≠ causation ≠ decodability**; name "stable"/"represented", not "true"; the agency/receptacle frame is *illuminated by* results, never *load-bearing*.
