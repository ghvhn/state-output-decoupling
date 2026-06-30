# Latent Concept Harvest — corrected finding (2026-06-29)

Supersedes the per-layer table in `latent_concept_report_20260629_223015.md`, whose
raw cosines (~0.95, surface_inv ~0.98) were dominated by a common-mode component
and are NOT evidence of concept mapping.

## Method
- 11 word problems, 3 concepts (periodic_discount, profit_vs_revenue, remainder_sale),
  including 3 matched story-vs-math pairs from the neutralized probe.
- Bare model (Llama-3.1-8B), no scaffolds / no oracle / no cache / no steering.
- Captured residual stream at all 32 layers (mean over generated reasoning tokens).
- Removed common-mode (centered per layer) before measuring cosine structure.

## Result — the model maps these concepts itself
- After centering: same-concept cosine ≈ **+0.5**, different-concept ≈ **−0.3**,
  separation ≈ **+0.9**.
- Strongest in the mid-band **L9–L21, peak L17** (+0.901) — the workspace region the
  repo already routes at (L16).
- **Permutation null (3000 label shuffles):** observed mid-band sep +0.876 vs null
  mean +0.003, 95th pct +0.281, max +0.705. **p < 0.0003.** Real, not a centering
  artifact.
- Surface-invariance (story vs math, centered): +0.21–0.46 (profit +0.46 best,
  periodic-discount +0.21 weakest). The concept abstracts across wording, but only
  partially — some surface dependence remains.

## Honest limits
- **0/11 correct** on the bare model, so OUTCOME separation (right vs wrong reasoning)
  is UNANSWERED — need correct examples to test it.
- n=11, 3 concepts ~4 each. The null covers the core claim; replicate wider to be safe.
- Mean-pooled state per problem; a trajectory/plateau capture may sharpen it.

## Why it matters
Confirms "lean into the latent space": the concept map is real, lives mid-band, and
emerges with no hand-built probe axes (which the vector-geometry map showed carry
~0 good/bad signal). Next: get correct examples and test whether right-vs-wrong
reasoning separates WITHIN concept regions — that is the honesty/capability axis,
read from the model's own latent position instead of an external sensor.
