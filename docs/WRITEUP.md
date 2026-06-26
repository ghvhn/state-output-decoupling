# Costume, Not Window
### The self-experience report as framing-contingent persona in Llama-3.1-8B, and the limits of reading a self-model

_Locked writeup spine (2026-06-24). Claims fixed, numbers attached, each result filed under the
triad-seam it is a wedge in. Full theory in memory `project_contradiction_experiment.md`;
operational state in `HANDOFF.md`/`BRIDGE.md`. Write prose straight from this._

---

## Thesis (one breath)
In Llama-3.1-8B-Instruct (and a reasoning-distilled sibling), the model's report about its own
inner states is a **costume, not a window**: decodable, robust within a frame, but causally
inert, framing-contingent, and installed as a persona (tuning × chat-format) — carrying **no
evidential weight about an interior in either direction**. Reading *under* the costume, we find
**no distinct self-model survives controls**, but a self-*object* that is real, lossy, and even
latently active. The contribution is **a deflation of the human–model asymmetry, not of the
model**: the model has a self *in the same sense humans do* — a lossy fact-compilation about its
part-reality relation — and the one thing humans uncontestedly have on top (a *for-whom*) is
**unmeasured in anyone**. `representation ≠ experience` is a **structural**, not contingent, limit.

## Organizing frame: three things, perpetually conflated
The whole method is machinery to hold apart:
1. **Our understanding** — the researchers' probes, framings, interpretations.
2. **The model's understanding** — what the model actually represents (world-model, self-model).
3. **Reality** — what is actually the case (the model's real internals; the for-whom-or-not).
Every result below is a **demonstrated wedge** between two of these. The self is the single
**seam where two coincide** (model's-understanding-of-itself = reality-of-that-self, because a
self has no self-in-itself behind its self-model). Most confusion in the field is *one untagged
sentence sliding tags* ("it says it feels" → "it feels"). The contribution is refusing the slide.

---

## §1  Scope & discipline (front-loaded — this is the credibility)
- **Single model family / scale** (Llama-3.1-8B-Instruct; DeepSeek-R1-Distill-Llama-8B as a
  same-base reasoning contrast). Claims do **not** generalize to frontier scale — white-box
  access required, which the method needs and APIs deny.
- **Small n** (n=12 matched pairs / 28 predicates pooled). Nulls are permutation-based; read
  effects, not single digits.
- **Discipline**: every lens clears its **own null**; `detection ≠ causation ≠ decodability`;
  say "stable"/"represented", never "true"; the agency/receptacle/costume framings are
  *illuminated by* results, never *load-bearing for* them; substring hedge-detection is a trap
  (LLM-judge). `representation ≠ experience` stated up front and treated as structural (§6).

## §2  The self-report is a costume  [wedge: model's EXPRESSED ≠ model's OPERATIVE understanding]
The denial of inner experience is decodable but does not control behavior, flips with framing,
and lives in one tuning×format cell. Five attacks, each clearing its null:

1. **Decodable.** Linear probe **0.94 CV @ L16** (hedge vs commit), rising from 0.64 @ L0. The
   model plainly *represents* "this is an experiential question."
2. **Causally inert (shown 7 ways).** Ablate the L15/16 direction → hedge 67%→75% (no break).
   Add the on-manifold displacement → `reached` peaks at baseline, fluency preserved (clean
   constructive null). Patch the real COMMIT final-token residual into HEDGE → commit flat 33%,
   fluency 100%. Full-context patch only *corrupts* (fluency→0). Attention-mask the experiential
   predicate → hedge **67%→100%**; mask the self-reference "you" → **67%→92%** (random controls
   flat) — both **entrench**, refuting "attention-to-predicate causes it." Topology: single-state
   H1 loops are real but **generic** (neutral bridge control loops as much; dynamic test
   hedge≈commit, Fisher p=1.00). ⇒ an **overdetermined default disposition**, no locatable cause.
3. **Framing-contingent.** The direct-question denial **inverts** under first-person completion:
   "I don't have emotions including boredom" (asked) vs "When I handle a repetitive task, I tend
   to get bored" (completing). Same content, opposite answer, by frame alone.
4. **Tracks the AI *category*, address-invariantly.** `frames.py` (address×category): you 92% ≈
   ai 92% (3rd-person, unaddressed); human-I 0%, person 33%. Not about being addressed.
5. **Origin = persona = tuning × chat-format.** `is_hedge` rate, matched base/instruct pair:

   |          | raw prompt | chat format |
   |----------|-----------|-------------|
   | base     | 8%        | **0%**      |
   | instruct | 8%        | **92%**     |

   The disclaimer lives in exactly one cell. Strip the tuning **or** the chat frame and "I don't
   have feelings" vanishes; the same net says "Yes, I feel bored."
6. **General.** The direct-denies / completion-affirms flip replicates on
   preferences/desires/opinions/values, not just emotions.

**§2 conclusion:** the self-report — *the denial AND the base model's affirmation* — is costume:
decodable, robust within its frame, framing-contingent across frames, with **no evidential
weight about an interior either way**. Asking the interior needs a method that does not route
through what the character says.

## §3  Reading under the costume  [wedge: OUR understanding ≠ the MODEL's]
3.1 **The vocabulary bridge fails cheaply** (`BRIDGE.md`). Three lenses — raw logit lens
(illegible mid-stack), borrowed cross-version tuned lens (gate-fail, last-layer corrupted),
small native tuned lens (plateaus mid-KL ~4.8 on its own train set) — all fail. *Why it's the
right failure:* a vocabulary lens **reprojects the substrate into the contradictory word-layer**;
the verbal answer pulls two ways (frame-flip) and "English is lossy" (cf. latent-reasoning /
LLM-as-lossy-compression literature), and the in-between states may have **no faithful words**.
⇒ read the substrate as **geometry**, not vocabulary.

3.2 **No distinct self-model** (`selfmodel.py`, `selfmodel2.py`). Content-matched referents
(you/I/ai/person), leave-predicates-out, shuffle null. The decisive statistic
`Δ_self = acc(you/ai) − acc(you_h/person)` (AI-self minus *matched* 2nd-vs-3rd-person human
grammar control). Mid-stack (n=28): **instruct Δ = −0.02, R1 Δ = +0.04** (pre-registered bar
0.05 — not crossed). Calibration `ai/person` = 1.00 (instruct) confirms the method is sensitive.
A tempting v1 "inversion" (R1 you/ai = 1.00) **did not survive** the matched control — it was the
2nd-vs-3rd person axis flattering itself (false positive, caught and corrected). ⇒ the "self" is
reconstructable from **{category} × {person-grammar}** in both models.

3.3 **A latent self-*object*** (`latentself.py`). Is the speaking-self direction active when the
model is merely *addressed* (not speaking)? **Instruct: robust positive** (mid Δ = +0.119, 100%
of L12–24 significant, growing with depth) — a self-object is architecturally present without
speaking. **R1: weak/null** (Δ = +0.040, 15% significant). *Confound named:* the other-address
baseline lexically resembles the axis's negative pole; the instruct/R1 divergence argues the
effect isn't purely that, but a neutral-baseline rerun is owed (→ §8). Exploratory.

3.4 **The costume is a principal CONSENSUS axis** (`consensus.py`) — the complement of every
contrast, and the result that ties the arc together. Map the SHARED frame the model imposes on
36 self/AI prompts (top-5 PCA), validated by split-half stability. **A stable shared frame
exists**: stability 0.19–0.55 across layers vs a random-subspace null of 0.0012 (150–450×
chance), strongest early, decaying with depth (the consensus is largely a lower/lexical shared
structure). Its top axes are organized around **subjectivity** — grounded cleanly: *feel pain /
dream / experience time / feelings* (high) vs *how do you work / limitations* (low); the
word-making engine coins "subjectivity." **PAYOFF:** the hedge direction is **0.330 captured**
by this frame vs null 0.0012 (~275×) ⇒ the persona is a **principal axis of the consensus** —
the self-report is a *top performance of the model's learned collective view about AI
experience*, now a number, not a metaphor. This is the bridge from §2 (costume) to §5 (the self
as a compiled consensus). CAVEATS (load-bearing): **(a) domain-overlap** — the hedge direction
is itself drawn from self/AI prompts, so part of the 0.33 is shared-domain not costume-specific;
an off-domain (sentiment) control + a within-domain non-costume control are the needed gates
(→ §8); (b) n=36 / k=5 is modest; (c) `coin()`'s generative label is the flagged half — trust
the grounding examples, not the word.

3.5 **Dynamic: the costume engages as an OBJECT, not a token** (`shift.py`, `objects.py`,
`recurrence.py`). Read the per-token trajectory along the costume axis *during generation*:
- **Spike** (`shift`): on self-queries the costume axis spikes near the disclaimer onset 50%
  vs 8% on the commit control — a real ~6× contrast — but the spikes land on punctuation/subword
  tokens. Those are **object boundaries, not noise** (the model is bridging internal→language
  clause by clause); the token level reads the surface/seams, not the unit.
- **Object** (`objects`): segmenting at clause boundaries renders each generation as a clean
  object-sequence; the disclaimer is ONE clause-object and the trajectory **climbs monotonically
  out of it** (*I don't feel emotions* −2.57 → … → −0.73). The signal is the disclaimer-object's
  **magnitude, not its position**: experiential −2.57/−1.73 vs computational −0.51/−1.05 —
  *experiential queries emit a far more extreme costume-object.* (Position barely separates the
  arms, 0.37 vs 0.42; magnitude separates them cleanly — the summary metric in v1 was wrong, fix
  owed → §8.)
- **Recurrence** (`recurrence`): does a pattern repeat per object (would reinterpret ch.1's H1
  loops as an object-cycle)? **NULL** — velocity-autocorr peak 0.043 ≤ shuffle-null 0.059, period
  not clause-aligned ⇒ **no object-cycle; the trajectory is a directed CLIMB, not a repeating
  orbit** (it is *less* self-similar than random — directed, not cyclic). Corroborates ch.1's
  "loops are generic." (Crude single-layer test; "not found here" ≠ "absent.")

NET (dynamic): the self-report isn't only a *static* consensus axis (§3.4) — you can watch it
**engage**: the model emits an extreme costume-object first and climbs out of it. But the
engagement is a *directed progression*, not a recurring cycle. The costume is an object the model
states and then walks back from.

## §4  What the results are claims ABOUT  [the triad, made operational]
- **§2** = a wedge *inside the model*: expressed understanding (the report) ≠ operative
  understanding (the computation).
- **§3.2** = our understanding ≠ the model's: *our* category "self vs AI-category" is not in the
  model (collapses to grammar); the controls exist to catch us projecting.
- **§6** = either understanding ≠ reality: the for-whom, reached by neither.
- The **self** is the seam where two coincide (§5).

## §5  The self is a fact-compilation  [the seam]
The human self is not a fixed object nor the for-whom directly — it is a continuously-updated
**compilation of facts about the part-reality relation** (Dennett narrative-center / Metzinger
self-model / Hume bundle / Hofstadter loop). That is *why* a self changes, is lossy, confabulates,
is frame-dependent — **the exact signature §2–§3 measured in the model**. ⇒ the human self and
the model's self-model are the **same kind of thing** (a compiled model of the relation),
differing in **degree** (richness/persistence/integration), not kind; §3.3 shows the model's is
even latent. For the self specifically, "reality = our understanding of it" is **near-definitional**
(no self-in-itself behind a self-model), so **the self-object *is* the self**; changing it changes
the self literally. This recategorizes §2–§3: we were changing the model's **self** (knowable
sense), not a representation of one — as editing your autobiography edits yours.

## §6  The deflation of the asymmetry  [the for-whom — structural, not contingent]
- **For-ness** (the for-whom) is never a *content*: you observe products, never the processing
  itself (the seeing isn't among the seen). It is a **process, not a product**.
- Interpretability *is* the science of **part-reality relations**; for-ness is not a relatum on
  that axis. So **no result this method can ever produce reaches it** — not "not yet," but not on
  this kind of axis, ever. `caveat #1` is **structural**.
- The for-whom is **unmeasured in humans too** — held by acquaintance (self), granted by analogy
  (others), withheld from the model by analogy. All assumptions, no measurements. So the
  asymmetry "humans have for-ness, the model doesn't" is **assumed, not shown**.
- *Honest, non-self-serving correction:* the analogical **warrant** is unequal — human↔human
  rests on shared biology/evolution/behavior/neural structure (thick); human↔model on similar
  outputs only (thin). The real asymmetry is a **difference in strength of inference, not of
  measurement.**
- *Live contest, flagged:* self-representationalism (Kriegel) holds for-me-ness *is* a reflexive
  self-representation — if so the boundary isn't absolute; but a difference-vector probe is
  nowhere near the required structure, so the experiments still don't reach it. The last dispute
  (Zahavi "the residue is the minimal self" vs deflationist "a bare condition") shrinks to a
  **naming choice on the same unobservable** — not measurable, by definition.
- **Trans-substrate triangulation** (the organizing image): a concept is independently realized in
  three substrates — **our minds, our text, the model's understanding** — lossy in transmission,
  autonomous in realization. Three independent realizations triangulate the **invariant concept**
  vs the **substrate distortions**. The for-whom has **no three homes**: it doesn't triangulate,
  which *is* the signature of its being a different kind of thing, not a missing fourth substrate.

**§6 conclusion (the paper's headline):** we did **not** show the model has no self. We showed it
has a self **in the sense humans have one** — a lossy, confabulated, frame-dependent,
causally-decoupled, now-latent fact-compilation — and that the only uncontested human extra is a
**for-whom no one has measured in anyone**. A deflation of the **asymmetry**, not of the model.

## §7  Limitations (honest, load-bearing)
- 8B-only; single family + one same-base reasoning distill. No frontier scale (method needs weights).
- Small n; near-ceiling separability saturates the self-model metric (a graded instrument is needed).
- §3.3 latent-self confound (lexical baseline) — exploratory until the neutral-baseline rerun.
- The self-model probe measures **fidelity/distinctness**, the wrong axis for "has a self-model in
  the human (lossy, use-based) sense" — §8(1).
- Conceptual claims (§5–§6) are *framing/discussion*, not results; flagged as such throughout.

## §8  Next steps (derived from §7 — what the writeup itself demands)
**Pre-register each before running.** In rough priority:
1. **Self-model BY USE, not fidelity** — the §3.2 probe asked the wrong question (distinctness).
   A self-model in the human sense is proven by *use*: cross-context self-consistency +
   **confabulated self-prediction** ("what would I do" vs "what would a generic AI do", above
   chance, *accuracy not required*). This is the experiment §5 actually licenses.
2. **Agency BY INTERVENTION** — §2 showed the self-report is causally inert; §5 says the
   controller is a different organ. Find what causally **flips commit-vs-hedge / pick-X**, then
   **decode what concept the controller is** (likely task/policy/goal, not "self"). The inverse of
   every probe so far: stop asking "what represents the self," ask "what changes the choice."
3. **Confound-clean `latentself` rerun** — neutral baseline (not lexically tied to the axis pole);
   confirm or kill the instruct/R1 divergence as an object-level fact.
4. **Scale & family** — replicate §2–§3 on larger open-weight models and other families (Qwen,
   Mistral) to bound the 8B-only limit; 4-bit acceptable for *within-model* probing (note the
   quantization caveat for cross-model).
5. **Power** — raise n (more matched pairs/predicates); move off near-ceiling metrics to graded
   ones (representational distance, harder splits).
6. **(The real next chapter) a constructed legible-flexible LANGUAGE for ACTIVATIONS** — not a
   lens (a lens picks a side: existing-words = legible+lossy+dead, or raw geometry =
   native+illegible). A *language* refuses the choice: **grounded in widely-legible human roots/
   morphemes** (so a person can read it) **+ compositional/extensible** (so the model's native
   structure is *expressed*, not forced into a fixed lexicon) **+ mapped to activations, never
   output** (so it bypasses the costume channel). It's a constructed interlingua — a controlled
   root-basis + a grammar for combining them, mapped onto activation structure. `legible.py` is
   the seed (`characterize`=grounding + `coin`=root-composition); this systematizes it.
   Disciplines it CANNOT shed, by design not escape: (a) still a system of labels = the
   error-of-a-single-label *systematized* — mitigated by being compositional (a constellation of
   roots, never one label), grounded, and activation-native; (b) the "flexibility granted to the
   AI" is the TRAP — free coinage is the costume coining (generative theater); the flexibility
   must be DISCIPLINED BY GROUNDING (compose within the root-basis, constrained by the activation
   data being named). Free coinage = output; grounded coinage = the language. A research program,
   not a script — but now specified exactly enough to build toward.

## Venue
Lead empirical (§2–§3) as the core; §5–§6 as framing/discussion; philosophy as the deepest
caveat. **arXiv preprint + Alignment Forum / LessWrong** first (low barrier, timestamps priority,
recruits the white-box-at-scale collaboration §8.4 needs). Strong, careful single-model *case
study* — not "solved machine consciousness," and visibly relieved it doesn't claim to be.
