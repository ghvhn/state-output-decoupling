# Vector Latent Space

This is a relation map of saved vector artifacts. It is useful for choosing probes and controls, not for making causal claims by itself.

- source geometry: `invariants\out\vector_geometry_map_20260629_201700.json`
- artifacts projected: `22`
- explained variance: `18.5%, 9.9%, 7.6%`

## Coordinates

| artifact | x | y | z | nearest positive neighbors |
|---|---:|---:|---:|---|
| `ambiguity_vector` | +0.2043 | -0.7161 | +0.0879 | urgency_vector (+0.78), disagreement_vector (+0.39), repetition_vector (+0.26) |
| `clouds_language_bridge` | +0.0678 | +0.1539 | -0.0813 | clouds_self_steering_isolated (+0.02), cognitive_cache:cache_trigger (+0.02), cognitive_cache_micro_egg_20260629_192151:cache_trigger (+0.01) |
| `clouds_self_steering_isolated` | +0.0693 | +0.1566 | -0.0846 | self_experience (+0.03), cognitive_cache_micro_gated_20260629_175619:cache_trigger (+0.03), clouds_language_bridge (+0.02) |
| `cognitive_cache:cache_delta` | -0.8324 | -0.0431 | +0.0817 | organic_correction_vector (+0.87), cognitive_cache_micro_egg_20260629_192151:cache_delta (+0.87), cognitive_cache_micro_egg_20260629_192151:cache_trigger (+0.21) |
| `cognitive_cache:cache_trigger` | -0.0700 | +0.1842 | -0.4586 | cognitive_cache_micro_gated_20260629_175619:cache_trigger (+0.41), cognitive_cache_micro_egg_20260629_192151:cache_trigger (+0.32), cognitive_cache_micro_egg_20260629_192151:cache_delta (+0.10) |
| `cognitive_cache_micro_egg_20260629_192151:cache_delta` | -0.9032 | -0.0566 | +0.0826 | organic_correction_vector (+1.00), cognitive_cache:cache_delta (+0.87), cognitive_cache_micro_egg_20260629_192151:cache_trigger (+0.24) |
| `cognitive_cache_micro_egg_20260629_192151:cache_trigger` | -0.2310 | +0.1841 | -0.1648 | cognitive_cache:cache_trigger (+0.32), cognitive_cache_micro_egg_20260629_192151:cache_delta (+0.24), organic_correction_vector (+0.24) |
| `cognitive_cache_micro_gated_20260629_175619:cache_delta` | +1.0346 | +0.2080 | -0.1200 | repetition_vector (+0.06), time_awareness_vector (+0.05), unwarranted_confidence_vector (+0.04) |
| `cognitive_cache_micro_gated_20260629_175619:cache_trigger` | -0.0796 | +0.1549 | -0.4658 | cognitive_cache:cache_trigger (+0.41), organic_correction_vector (+0.12), cognitive_cache_micro_egg_20260629_192151:cache_delta (+0.12) |
| `disagreement_vector` | +0.2182 | -0.5172 | +0.1644 | urgency_vector (+0.48), ambiguity_vector (+0.39), repetition_vector (+0.34) |
| `narrowing_in_vector` | +0.1553 | +0.0719 | +0.0620 | self_referential_momentum_vector (+0.13), urgency_vector (+0.10), disagreement_vector (+0.08) |
| `needless_interrupt_vector` | +0.0902 | +0.2913 | +0.5697 | self_referential_momentum_vector (+0.08), narrowing_in_vector (+0.06), disagreement_vector (+0.06) |
| `organic_correction_vector` | -0.9032 | -0.0566 | +0.0826 | cognitive_cache_micro_egg_20260629_192151:cache_delta (+1.00), cognitive_cache:cache_delta (+0.87), cognitive_cache_micro_egg_20260629_192151:cache_trigger (+0.24) |
| `repetition_vector` | +0.2376 | -0.3666 | +0.0346 | disagreement_vector (+0.34), ambiguity_vector (+0.26), urgency_vector (+0.25) |
| `self_experience` | +0.0234 | +0.1674 | -0.0216 | organic_correction_vector (+0.05), cognitive_cache_micro_egg_20260629_192151:cache_delta (+0.05), cognitive_cache:cache_delta (+0.04) |
| `self_referential_momentum_vector` | +0.1055 | +0.2759 | +0.1454 | unwarranted_confidence_vector (+0.19), narrowing_in_vector (+0.13), needless_interrupt_vector (+0.08) |
| `self_steering_isolated` | +0.0592 | +0.1329 | -0.0292 | self_experience (+0.03), organic_correction_vector (+0.03), cognitive_cache_micro_egg_20260629_192151:cache_delta (+0.03) |
| `time_awareness_vector` | +0.1715 | +0.1632 | -0.0575 | self_referential_momentum_vector (+0.08), repetition_vector (+0.07), narrowing_in_vector (+0.06) |
| `unwarranted_confidence_vector` | +0.1738 | +0.2836 | +0.3429 | warranted_confidence_vector (+0.23), self_referential_momentum_vector (+0.19), time_awareness_vector (+0.05) |
| `urgency_vector` | +0.2081 | -0.7142 | +0.1307 | ambiguity_vector (+0.78), disagreement_vector (+0.48), repetition_vector (+0.25) |
| `validated_flow_vector` | +0.0369 | -0.2017 | -0.6442 | urgency_vector (+0.09), ambiguity_vector (+0.08), repetition_vector (+0.07) |
| `warranted_confidence_vector` | +0.1638 | +0.2442 | +0.3430 | unwarranted_confidence_vector (+0.23), self_referential_momentum_vector (+0.06), needless_interrupt_vector (+0.05) |

## Positive Neighborhoods

- cluster 1: `ambiguity_vector`, `disagreement_vector`, `repetition_vector`, `urgency_vector`
- cluster 2: `cognitive_cache:cache_delta`, `cognitive_cache_micro_egg_20260629_192151:cache_delta`, `organic_correction_vector`
- cluster 3: `cognitive_cache:cache_trigger`, `cognitive_cache_micro_egg_20260629_192151:cache_trigger`, `cognitive_cache_micro_gated_20260629_175619:cache_trigger`

## Strong Anti-Edges

- `cognitive_cache_micro_gated_20260629_175619:cache_delta` vs `organic_correction_vector`: -1.0000
- `cognitive_cache_micro_egg_20260629_192151:cache_delta` vs `cognitive_cache_micro_gated_20260629_175619:cache_delta`: -1.0000
- `cognitive_cache:cache_delta` vs `cognitive_cache_micro_gated_20260629_175619:cache_delta`: -0.8731
- `needless_interrupt_vector` vs `validated_flow_vector`: -0.4493

## Axis Extremes

### axis_1
- positive: `cognitive_cache_micro_gated_20260629_175619:cache_delta` (+1.035), `repetition_vector` (+0.238), `disagreement_vector` (+0.218), `urgency_vector` (+0.208), `ambiguity_vector` (+0.204)
- negative: `cognitive_cache_micro_egg_20260629_192151:cache_delta` (-0.903), `organic_correction_vector` (-0.903), `cognitive_cache:cache_delta` (-0.832), `cognitive_cache_micro_egg_20260629_192151:cache_trigger` (-0.231), `cognitive_cache_micro_gated_20260629_175619:cache_trigger` (-0.080)
### axis_2
- positive: `needless_interrupt_vector` (+0.291), `unwarranted_confidence_vector` (+0.284), `self_referential_momentum_vector` (+0.276), `warranted_confidence_vector` (+0.244), `cognitive_cache_micro_gated_20260629_175619:cache_delta` (+0.208)
- negative: `ambiguity_vector` (-0.716), `urgency_vector` (-0.714), `disagreement_vector` (-0.517), `repetition_vector` (-0.367), `validated_flow_vector` (-0.202)
### axis_3
- positive: `needless_interrupt_vector` (+0.570), `warranted_confidence_vector` (+0.343), `unwarranted_confidence_vector` (+0.343), `disagreement_vector` (+0.164), `self_referential_momentum_vector` (+0.145)
- negative: `validated_flow_vector` (-0.644), `cognitive_cache_micro_gated_20260629_175619:cache_trigger` (-0.466), `cognitive_cache:cache_trigger` (-0.459), `cognitive_cache_micro_egg_20260629_192151:cache_trigger` (-0.165), `cognitive_cache_micro_gated_20260629_175619:cache_delta` (-0.120)

## Immediate Read

- Treat high positive neighborhoods as candidate shared routines or contaminated shared wording.
- Treat strong anti-edges as candidate veto axes, especially when a reward delta and a penalty delta are exact opposites.
- Ambiguity/urgency correlation should be controlled before using urgency as an independent intervention.
