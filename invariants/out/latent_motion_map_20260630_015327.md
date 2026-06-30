# Latent Motion Map

This artifact stores directed latent displacements. It is a vector map, not a matrix of pairwise similarities.

## Sources

- outcome: `invariants\out\latent_outcome_points_20260629_224925.pt`
- uncertainty: `invariants\out\latent_uncertainty_points_20260629_231453.pt`
- confidence: `invariants\out\latent_confidence_points_20260630_011642.pt`
- concept: `invariants\out\latent_concept_points_20260629_223015.pt`
- stage_states: `['invariants\\out\\humble_stage_states_20260630_014033_humble_synthesis_2b2e3f9639.pt', 'invariants\\out\\humble_stage_states_20260630_014309_humble_synthesis_af61a80349.pt', 'invariants\\out\\humble_stage_states_20260630_014631_humble_synthesis_5380d0e581.pt', 'invariants\\out\\humble_stage_states_20260630_014940_humble_synthesis_08acf5ba3d.pt', 'invariants\\out\\humble_stage_states_20260630_015301_humble_synthesis_dc73b226a1.pt']`

## Edge Families

- `concept_wording_category_motion`: 96 layerwise directed vectors
- `confidence_centroid_to_sample`: 64 layerwise directed vectors
- `confidence_sample_contrast`: 32 layerwise directed vectors
- `humble_solver_generation_motion`: 480 layerwise directed vectors
- `humble_solver_to_verifier_motion`: 480 layerwise directed vectors
- `humble_verifier_generation_motion`: 480 layerwise directed vectors
- `outcome_pre_to_generation`: 32 layerwise directed vectors
- `uncertainty_centroid_to_sample`: 64 layerwise directed vectors
- `uncertainty_sample_contrast`: 32 layerwise directed vectors
- `wording_neutralized_to_standard`: 32 layerwise directed vectors

## Largest Motions

- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=144.854 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=143.331 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=142.598 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=142.517 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=142.330 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=140.967 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=140.863 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=140.752 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=138.934 | n=1
- `humble_solver_to_verifier_motion` L31: `solver_response_mean` -> `verifier_response_mean` | norm=138.429 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=138.187 | n=1
- `humble_verifier_generation_motion` L31: `verifier_prompt_pre` -> `verifier_response_mean` | norm=137.124 | n=1
- `humble_verifier_generation_motion` L31: `verifier_prompt_pre` -> `verifier_response_mean` | norm=137.124 | n=1
- `humble_verifier_generation_motion` L31: `verifier_prompt_pre` -> `verifier_response_mean` | norm=137.124 | n=1
- `humble_solver_generation_motion` L31: `solver_prompt_pre` -> `solver_response_mean` | norm=135.755 | n=1
- `humble_verifier_generation_motion` L31: `verifier_prompt_pre` -> `verifier_response_mean` | norm=135.654 | n=1

## Strong Probe Alignments

- -0.488 with `organic_correction_vector`: `outcome_pre_to_generation` L15 `pre_reasoning_state` -> `generated_wrong_state`
- -0.422 with `organic_correction_vector`: `humble_solver_generation_motion` L15 `solver_prompt_pre` -> `solver_response_mean`
- -0.421 with `organic_correction_vector`: `humble_solver_generation_motion` L15 `solver_prompt_pre` -> `solver_response_mean`
- -0.418 with `organic_correction_vector`: `humble_solver_generation_motion` L15 `solver_prompt_pre` -> `solver_response_mean`
- -0.418 with `organic_correction_vector`: `humble_solver_generation_motion` L14 `solver_prompt_pre` -> `solver_response_mean`
- -0.412 with `organic_correction_vector`: `humble_solver_generation_motion` L14 `solver_prompt_pre` -> `solver_response_mean`
- -0.412 with `organic_correction_vector`: `humble_solver_generation_motion` L14 `solver_prompt_pre` -> `solver_response_mean`
- -0.408 with `organic_correction_vector`: `outcome_pre_to_generation` L16 `pre_reasoning_state` -> `generated_wrong_state`
- -0.407 with `organic_correction_vector`: `humble_solver_generation_motion` L16 `solver_prompt_pre` -> `solver_response_mean`
- -0.406 with `organic_correction_vector`: `humble_solver_generation_motion` L16 `solver_prompt_pre` -> `solver_response_mean`
- -0.405 with `organic_correction_vector`: `humble_solver_generation_motion` L17 `solver_prompt_pre` -> `solver_response_mean`
- -0.404 with `organic_correction_vector`: `humble_solver_generation_motion` L17 `solver_prompt_pre` -> `solver_response_mean`
- -0.401 with `organic_correction_vector`: `humble_solver_generation_motion` L16 `solver_prompt_pre` -> `solver_response_mean`
- -0.401 with `organic_correction_vector`: `humble_solver_generation_motion` L15 `solver_prompt_pre` -> `solver_response_mean`
- -0.397 with `organic_correction_vector`: `humble_solver_generation_motion` L16 `solver_prompt_pre` -> `solver_response_mean`
- -0.394 with `organic_correction_vector`: `humble_solver_generation_motion` L15 `solver_prompt_pre` -> `solver_response_mean`
- -0.393 with `organic_correction_vector`: `outcome_pre_to_generation` L14 `pre_reasoning_state` -> `generated_wrong_state`
- -0.391 with `organic_correction_vector`: `humble_solver_generation_motion` L15 `solver_prompt_pre` -> `solver_response_mean`
- -0.391 with `organic_correction_vector`: `humble_solver_generation_motion` L14 `solver_prompt_pre` -> `solver_response_mean`
- -0.390 with `organic_correction_vector`: `humble_solver_generation_motion` L15 `solver_prompt_pre` -> `solver_response_mean`

## Read This Carefully

- `outcome_pre_to_generation` is the cleanest motion: before-answer state to generated-answer state.
- `*_centroid_to_sample` says how sampled replies move away from a problem's own center.
- `wording_neutralized_to_standard` is a wording/semantics transition, not an outcome transition.
- This still does not fully capture verifier-induced abandonment. For that, the benchmark must save attempt-stage states: solver initial, solver final, verifier initial, verifier final, repair final.
