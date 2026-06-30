# Latent Concept Harvest

- points: 11  |  categories: ['periodic_discount', 'profit_vs_revenue', 'remainder_sale']
- correct: 0  wrong: 11
- surface pairs (standard vs neutralized): 3

## Per-layer organization
`concept_sep` = mean same-category cosine - mean cross-category cosine (higher = concept-clustered).
`surface_inv` = mean cosine of same-concept story-vs-math pairs (higher = maps concept, not words).
`outcome_sep` = mean same-outcome - mean cross-outcome cosine.

| layer | concept_sep | surface_inv | outcome_sep |
|---:|---:|---:|---:|
| 0 | +0.014 | +0.980 | +nan |
| 1 | +0.016 | +0.978 | +nan |
| 2 | +0.013 | +0.982 | +nan |
| 3 | +0.012 | +0.983 | +nan |
| 4 | +0.015 | +0.979 | +nan |
| 5 | +0.015 | +0.980 | +nan |
| 6 | +0.016 | +0.981 | +nan |
| 7 | +0.018 | +0.979 | +nan |
| 8 | +0.020 | +0.979 | +nan |
| 9 | +0.018 | +0.982 | +nan |
| 10 | +0.020 | +0.981 | +nan |
| 11 | +0.018 | +0.984 | +nan |
| 12 | +0.022 | +0.981 | +nan |
| 13 | +0.023 | +0.978 | +nan |
| 14 | +0.026 | +0.973 | +nan |
| 15 | +0.030 | +0.970 | +nan |
| 16 | +0.030 | +0.971 | +nan |
| 17 | +0.032 | +0.971 | +nan |
| 18 | +0.027 | +0.973 | +nan |
| 19 | +0.026 | +0.974 | +nan |
| 20 | +0.029 | +0.971 | +nan |
| 21 | +0.027 | +0.973 | +nan |
| 22 | +0.027 | +0.972 | +nan |
| 23 | +0.024 | +0.975 | +nan |
| 24 | +0.024 | +0.974 | +nan |
| 25 | +0.024 | +0.973 | +nan |
| 26 | +0.024 | +0.974 | +nan |
| 27 | +0.022 | +0.974 | +nan |
| 28 | +0.024 | +0.972 | +nan |
| 29 | +0.031 | +0.964 | +nan |
| 30 | +0.034 | +0.960 | +nan |
| 31 | +0.058 | +0.931 | +nan |

## Read
- strongest concept-clustering at **L31** (concept_sep +0.058).
- strongest surface-invariance (story==math) at **L11** (surface_inv +0.984).
- strongest outcome-separation at **L-1** (outcome_sep -9.000).

If surface_inv is high where concept_sep is high, the model maps the CONCEPT itself, not the wording -- the latent space is the concept map.