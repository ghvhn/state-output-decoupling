# Reference shelf for the LessWrong post

Curated candidates to cite, mapped to the post's claims. Verify each before
citing (read at least the abstract; never cite from this list blind), and
don't cite-dump: a LessWrong post earns trust with ~12–18 load-bearing
references woven into prose, not a bibliography.

Legend:
- ⭐ = cited by the Anthropic global-workspace paper itself (its reference
  list was extracted from your PDF — shared lineage with the convergence
  target is the strongest positioning).
- [LW] = LessWrong-native or heavily discussed there; these do the "engages
  with ideas this community knows" work the moderator asked for.
- [have] = the PDF is already in your Downloads.

---

## The core spine (~14, one per claim)

**The depth-structure claim (Figures 2–3: translate in, work in the middle, render out)**

1. ⭐[LW] nostalgebraist, *interpreting GPT: the logit lens*, LessWrong, 2020.
   — THE canonical LW prior for "intermediate layers are not the output";
   your render-position U is a stronger, controlled version of what the
   logit lens shows informally. Cite it early and prominently.
2. ⭐ Belrose et al., *Eliciting Latent Predictions from Transformers with the
   Tuned Lens*, arXiv:2303.08112 (2023). — the rigorous successor; your
   "translation" arms are what the lens family measures from the output side.
3. Elhage et al., *Softmax Linear Units*, Transformer Circuits (2022). — the
   cleanest prior statement of the U's endpoints: early layers "detokenize,"
   late layers "retokenize." Your fig2 is this claim measured directly.
4. ⭐ Lad, Gurnee & Tegmark, *The Remarkable Robustness of LLMs: Stages of
   Inference?*, arXiv:2406.19384 (2024). — four depth stages (detokenization →
   feature engineering → prediction ensembling → residual sharpening); the
   closest published analogue of your layer-purpose model.
5. ⭐ Tenney, Das & Pavlick, *BERT Rediscovers the Classical NLP Pipeline*,
   ACL 2019. — the original "depth = processing pipeline" result; shows your
   claim has a pre-LLM lineage (with Voita et al. 2019, arXiv:1909.01380, as
   the encoder-era companion).

**The state-output decoupling claim (final text ≠ internal state)**

6. [LW] Burns, Ye, Klein & Steinhardt, *Discovering Latent Knowledge in
   Language Models Without Supervision*, arXiv:2212.03827 (ICLR 2023). — the
   canonical "internal state can disagree with the stated answer";
   state-output decoupling is your generalization of this to depth profiles.
7. Azaria & Mitchell, *The Internal State of an LLM Knows When It's Lying*,
   arXiv:2304.13734 (EMNLP Findings 2023). — same decoupling, framed as
   internal truth signal vs emitted text.

**The CoT / mini-U claim (Figure 4)**

8. [LW] Turpin et al., *Language Models Don't Always Say What They Think:
   Unfaithful Explanations in Chain-of-Thought Prompting*, arXiv:2305.04388
   (NeurIPS 2023). — CoT text is not automatically the computation; your
   fig4 gives the complementary positive case (when CoT IS load-bearing).
9. Lanham et al., *Measuring Faithfulness in Chain-of-Thought Reasoning*,
   arXiv:2307.13702 (2023). — Anthropic's intervention battery (truncation,
   corruption) is the outcome-level twin of your wrong-scratchpad probes.
10. Pfau, Merrill & Bowman, *Let's Think Dot by Dot: Hidden Computation in
    Transformer Language Models*, arXiv:2404.15758 (2024). — separates
    computation from verbalization with filler tokens; sharpens what your
    direct-mode pre-commitment result does and doesn't show.

**The self-report / render-arm claim (Figure 5)**

11. [LW] janus, *Simulators*, LessWrong, 2022. — the LW-native frame for
    "the assistant is a trained persona, not a window into the substrate";
    your origin 2×2 is a measurement this audience will read in those terms.
12. Perez et al., *Discovering Language Model Behaviors with Model-Written
    Evaluations*, arXiv:2212.09251 (2022). — where "RLHF installs personas
    and self-descriptions" entered the literature; pairs with Shanahan,
    McDonell & Reynolds, *Role play with large language models*, Nature 2023.

**The probe-discipline claim (nulls, controls, decodable ≠ causal)**

13. ⭐ Hewitt & Liang, *Designing and Interpreting Probes with Control Tasks*,
    EMNLP 2019 (arXiv:1909.03368). — your shuffle-null discipline is exactly
    their control-task doctrine; citing it signals you know why nulls exist.
14. ⭐ Elazar et al., *Amnesic Probing: Behavioral Explanation with Amnesic
    Counterfactuals*, TACL 2021 (arXiv:2006.00995). — decodable-but-not-causal
    as a named methodology; your causally-inert hedge is an instance.

---

## Deeper shelf, by post section

### Interpretation/translation arm & workspace band (fig 2–3 support)
- ⭐ Wendler et al., *Do Llamas Work in English?*, ACL 2024 (arXiv:2402.10588)
  — mid-stack latent lingua franca on the same model family; independent
  evidence the middle band is "language of the mind," not surface tokens.
- ⭐ Skean et al., *Layer by Layer: Uncovering Hidden Representations in
  Language Models* (2025, arXiv:2502.02013) — representation quality peaks
  mid-stack across tasks; the workspace band from the transfer-learning side.
- ⭐ Alain & Bengio, *Understanding intermediate layers using linear
  classifier probes*, arXiv:1610.01644 (2016) — the origin of layer probing.
- ⭐ Yan, *Addition in Four Movements: Mapping Layer-wise Information
  Trajectories in LLMs*, EMNLP Findings 2025 — layer-wise arithmetic
  trajectories; nearest published neighbor of your arithmetic grid design.
- ⭐ Halawi, Denain & Steinhardt, *Overthinking the Truth*, ICLR 2024
  (arXiv:2307.09476) — "early exiting would have answered correctly";
  mid→late transitions can overwrite earlier states. Good for the ARROW_FOLD
  / transition-failure discussion.
- Meng et al., *Locating and Editing Factual Associations in GPT* (ROME),
  NeurIPS 2022 (arXiv:2202.05262) — causal tracing finds mid-layer sites;
  causal-method support for "latent work happens mid-stack."

### ICL & the "forward pass updates its own computation" intro claim
- [have] von Oswald et al., *Transformers Learn In-Context by Gradient
  Descent*, ICML 2023 (arXiv:2212.07677) — your warrant for reading ICL as
  input-dependent computation-update. Pair with:
- ⭐[LW] Olsson et al., *In-context Learning and Induction Heads*,
  Transformer Circuits (2022) — the mechanism-level ICL account LW knows.
- ⭐ Hendel, Geva & Globerson, *In-Context Learning Creates Task Vectors*,
  EMNLP Findings 2023 (arXiv:2310.15916); and ⭐ Todd et al., *Function
  Vectors in Large Language Models*, arXiv:2310.15213 — ICL condenses to a
  mid-stack latent task variable: literally your "latent task state."

### Mesa-optimization / oversight-paradox framing (keep light in the post)
- [have][LW] Hubinger et al., *Risks from Learned Optimization in Advanced
  Machine Learning Systems*, arXiv:1906.01820 (2019) — the mesa-objective
  vocabulary for your "a controller does not preserve intent; it shifts it."
  (Same paper as your `1906.01820v3.pdf`.) One paragraph, not a section —
  the post's strength is the measurements.

### Self-report, persona, introspection (fig 5 & the "costume" spine)
- ⭐ Lindsey, *Emergent Introspective Awareness in Large Language Models*,
  Transformer Circuits (2025) — injected-activation self-reports; the
  strongest recent treatment of when self-report tracks internal state.
- ⭐ Lu et al., *The Assistant Axis: Situating and Stabilizing the Default
  Persona of Language Models*, arXiv:2601.10387 (2026) — a persona *axis*
  in activation space; directly continuous with your frame-contingent
  disclaimer and tuning×chat origin result.
- ⭐ G & Lindsey, *From Simulation to Enaction: Post-trained language models
  recognize and react to their own generations*, arXiv:2605.25459 (2026) —
  overlaps your planned "conversational self / own-turns" experiment; read
  before you run it.
- Binder et al., *Looking Inward: Language Models Can Learn About Themselves
  by Introspection*, arXiv:2410.13787 (2024); Betley et al., *Tell Me About
  Yourself: LLMs Are Aware of Their Learned Behaviors*, arXiv:2501.11120
  (2025); ⭐ Berglund et al., *Taken Out of Context: On Measuring Situational
  Awareness in LLMs* (2023) — the Owain Evans cluster; LW-familiar
  self-knowledge results to position your "no privileged self-access" lock.
- Kadavath et al., *Language Models (Mostly) Know What They Know*,
  arXiv:2207.05221 (2022) — calibration & P(IK); the behavioral-side prior
  for your uncertainty-decoding positive.
- Kuhn, Gal & Farquhar, *Semantic Uncertainty* (ICLR 2023, arXiv:2302.09664)
  and Farquhar et al., *Detecting hallucinations… semantic entropy*, Nature
  2024 — K-sample consistency as the uncertainty ground truth; your latent
  uncertainty sensor is the activation-side version of this.

### Steering & causality (the "decodable but inert" hedge, and your knobs)
- ⭐ Arditi et al., *Refusal in Language Models Is Mediated by a Single
  Direction*, arXiv:2406.11717 (2024) — the sharp CONTRAST case: refusal has
  a single ablatable direction; your hedge did NOT flip under residual edits.
  That contrast is a discussion point, make it explicitly.
- ⭐[LW] Turner et al., *Activation Addition* (arXiv:2308.10248, 2023);
  ⭐ Panickssery et al., *Steering Llama 2 via Contrastive Activation
  Addition* (arXiv:2312.06681); ⭐ Zou et al., *Representation Engineering*
  (arXiv:2310.01405) — your contrastive probe-minting/steering sits in this
  method family, on the same model family as CAA.
- ⭐ Li et al., *Inference-Time Intervention*, NeurIPS 2023 (arXiv:2306.03341)
  — mid-layer truthfulness steering; representation vs output again.
- Belinkov, *Probing Classifiers: Promises, Shortcomings, and Advances*,
  Computational Linguistics 2022 — the survey that says "decodable ≠ used";
  cite once where you state the presence/performance/causation ladder.

### CoT internalization (fig 4 discussion)
- ⭐ Deng, Choi & Shieber, *From Explicit CoT to Implicit CoT: Learning to
  Internalize CoT Step by Step*, arXiv:2405.14838 (2024) — training moves
  the trajectory from text into weights; the training-time mirror of your
  direct-vs-CoT positioning of answer formation.
- ⭐ Nye et al., *Show Your Work: Scratchpads…*, arXiv:2112.00114 (2021) —
  the scratchpad origin; fine as a one-line historical cite.

### Workspace / consciousness guardrails (keep to 3–4 citations total)
- ⭐ Baars, *A Cognitive Theory of Consciousness*, 1988; ⭐ Dehaene, Kerszberg
  & Changeux, PNAS 1998 — the GWT originals the Anthropic paper builds on;
  one sentence, cited together.
- ⭐ Butlin, Long et al., *Consciousness in Artificial Intelligence: Insights
  from the Science of Consciousness*, arXiv:2308.08708 (2023) — the sober
  indicator-property framework; the right anchor for your "this is not a
  consciousness claim" boundary.
- ⭐[LW] Chalmers, *Could a Large Language Model Be Conscious?*,
  arXiv:2303.07103 (2023) — LW-adjacent statement of the question; useful to
  cite when you say which question you are NOT answering.

### Features / superposition (context for "not one fixed job per layer")
- [have] ⭐ Templeton et al., *Scaling Monosemanticity: Extracting
  Interpretable Features from Claude 3 Sonnet*, Transformer Circuits (2024)
  — your attached PDF; cite alongside ⭐ Bricken et al., *Towards
  Monosemanticity* (2023) and ⭐ Elhage et al., *Toy Models of Superposition*
  (2022) when you argue depth roles are functional regimes over distributed
  features, not per-layer modules.

---

## Already on your desk

- `Transformers Learn In-Context.pdf` = von Oswald et al. 2023 → ICL intro claim.
- `Risks from Learned Optimization.pdf` = Hubinger et al. 2019 (duplicate of
  `1906.01820v3.pdf`) → oversight-paradox framing, one paragraph max.
- `Monosemanticity.pdf` = Templeton et al. 2024 *Scaling Monosemanticity*
  (NOT Bricken 2023 *Towards Monosemanticity* — cite whichever you actually
  read, or both) → features context.
- `Verbalizable Representations Form a Global Workspace in Language
  Models.pdf` = the convergence target; its own reference list (130 entries,
  extracted 2026-07-07) is the source of every ⭐ above.

## How to use this list

The moderator's complaint was "no connection to research directions we
know." The fix is not volume; it is placement: one or two sentences per
section that say *what known result your measurement extends, sharpens, or
contradicts*. The three highest-leverage placements:

1. Fig 2 next to the logit/tuned lens and SoLU detokenization story — "the
   U is what the lens family sees, measured with labels and nulls at the
   generated tokens."
2. Fig 4 next to Turpin/Lanham — "CoT faithfulness has a depth/position
   signature you can measure, not just an outcome rate."
3. Fig 5 + the inert hedge next to Arditi's refusal direction — "refusal has
   a causal direction; the self-disclaimer, in my hands, does not — it
   behaves like an overdetermined render-arm default."
