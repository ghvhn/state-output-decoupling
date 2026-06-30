# Latent Uncertainty (dispersion of the thinking, no English label)

- problems: 16

## Per-layer latent dispersion (mean across problems / spread)
| layer | mean disp | min | max |
|---:|---:|---:|---:|
| 0 | +1.190 | +1.170 | +1.198 |
| 2 | +1.191 | +1.175 | +1.198 |
| 4 | +1.191 | +1.174 | +1.198 |
| 6 | +1.191 | +1.174 | +1.198 |
| 8 | +1.190 | +1.171 | +1.198 |
| 10 | +1.191 | +1.172 | +1.198 |
| 12 | +1.190 | +1.174 | +1.199 |
| 14 | +1.189 | +1.167 | +1.199 |
| 15 | +1.190 | +1.171 | +1.199 |
| 16 | +1.190 | +1.168 | +1.198 |
| 17 | +1.190 | +1.167 | +1.198 |
| 18 | +1.190 | +1.171 | +1.199 |
| 20 | +1.190 | +1.172 | +1.199 |
| 22 | +1.190 | +1.171 | +1.198 |
| 24 | +1.190 | +1.174 | +1.198 |
| 26 | +1.191 | +1.174 | +1.198 |
| 28 | +1.191 | +1.173 | +1.198 |
| 30 | +1.191 | +1.172 | +1.198 |

## Validation only -- latent dispersion vs ENGLISH answer-consistency
(Spearman-ish: correlation of per-problem latent dispersion with output consistency. Negative = more scattered thinking -> less consistent answers, i.e. the latent sensor tracks the model's own uncertainty.)

| layer | corr(disp, consistency) |
|---:|---:|
| 0 | -0.132 |
| 2 | -0.258 |
| 4 | -0.163 |
| 6 | -0.165 |
| 8 | -0.011 |
| 10 | +0.082 |
| 12 | +0.104 |
| 14 | +0.061 |
| 15 | +0.049 |
| 16 | +0.005 |
| 17 | +0.029 |
| 18 | +0.021 |
| 20 | +0.041 |
| 22 | +0.060 |
| 24 | +0.070 |
| 26 | +0.077 |
| 28 | +0.055 |
| 30 | +0.124 |

- strongest (most negative) at **L1** (corr -0.275).
- if strongly negative: the convergence of the THINKING predicts the consistency of the TALKING -- an English-free uncertainty sensor. If ~0: dispersion as measured is not the right read, or output-consistency is the wrong validator.