# Latent Uncertainty — finding (2026-06-29)

## Result
The scale-aware dispersion of the model's own sampled latent trajectories predicts
its answer-consistency. More scattered thinking -> less consistent answers.
- metric: per problem, sample K=6 (bare-question chat turn, no CoT), capture per-layer
  mean residual per sample; dispersion = RMS spread of the K states / mean-state norm.
- **r = -0.764**, mid-band **L16-24** (peak L24 -0.775); permutation null (5000 label
  shuffles): null mean -0.003, min -0.764 -> **p < 0.0002**.
- per-problem: least-dispersed 0.090 -> 100% consistent; most-dispersed 0.215 -> 33%.

## The instrument lesson (important)
- The FIRST metric (1 - mean cosine of *centered* K states) returned NULL (~0). Centering
  removes the magnitude of the spread, which is the signal, and forces the -1/(K-1) baseline.
- Keeping SCALE (dispersion relative to norm) recovered it.
- Mirror of the concept run: CONCEPT needed common-mode REMOVED (center); UNCERTAINTY needs
  scale KEPT. Opposite instrument choices, each validated by a permutation null. Read the
  latent space with the right geometry or you measure noise.

## Honest caveats
- n=16. r=-0.76 with p~0 is solid but small; replicate with more problems.
- POST-HOC metric selection: tried centered-cosine (null), then full-CV / pca8 / pca2
  (all negative, mid-band strongest). The null controls for this metric's correlation being
  chance, not for the search. Robustness across all three metrics + the whole mid-band +
  a monotone-ish per-problem relation is the credibility, not one cell.
- The validator (answer-consistency) is ENGLISH-derived. So strictly: latent scatter tracks
  OUTPUT scatter. Both may share a cause (problem hardness). Claim = the latent geometry is a
  faithful English-free readout of the uncertainty that also shows in behavior. Not a claim of
  latent being upstream/"more real."
- Single mind (GSM8K), not yet conversational coupling.

## Why it matters
This is the instrument the quality filter needs: read "settled vs unsettled" in the residual
stream, no English judgment. For the conversation corpus, aim the SAME scale-aware dispersion
at two-party traces -> coupling stability (did the minds settle together). Validate the
single-mind instrument first (done), then generalize to the exchange.
