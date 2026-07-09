# Public Experiment Tree

This is a public-file-only map of the repository's chain of thinking. Public means `git ls-files`: files already tracked by the repo, excluding local scratch files, private notes, untracked generated context figures, and this untracked index itself until it is added.

Role labels:

- `NEXT`: leads to a next experiment, control, measurement, or run path.
- `AWAY`: leads away from a hypothesis by narrowing, refuting, or showing a dead end.
- `CONCLUSION`: records a conclusion, synthesis, public figure, result summary, or claim boundary.

Coverage: `329` tracked public files: `273` NEXT, `32` AWAY, `24` CONCLUSION.

## Reading Path

- Claim root: [README.md](../README.md) -> [FINDINGS.json](../FINDINGS.json) -> [docs/RESULTS.md](../docs/RESULTS.md).
- Main negative-control branch: [docs/PERSONA_AUDIT.md](../docs/PERSONA_AUDIT.md) -> [docs/ISOLATING_UNDERSTANDING.md](../docs/ISOLATING_UNDERSTANDING.md).
- U-shape branch: [docs/TRANSLATION_THINKING.md](../docs/TRANSLATION_THINKING.md) -> [docs/ARCHITECTURAL_REALITY.md](../docs/ARCHITECTURAL_REALITY.md) -> [invariants/translation_thinking.py](../invariants/translation_thinking.py) -> [invariants/style_layers.py](../invariants/style_layers.py) -> [invariants/surgery_basetarget.py](../invariants/surgery_basetarget.py).
- Public evidence surface: [figures/origin_matrix.svg](../figures/origin_matrix.svg), [figures/causal_summary.svg](../figures/causal_summary.svg), [figures/attention_masks.svg](../figures/attention_masks.svg), [figures/frame_dependence.svg](../figures/frame_dependence.svg).

## Tree

### 0. Claim Surface

Start here. These files state what the repo claims, how to run it, and what boundaries constrain interpretation.

- launchers
  - `NEXT` [START_HERE.cmd](../START_HERE.cmd) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_after_current.py](../archive/runners/run_after_current.py) - public launcher for the associated experiment.
  - `NEXT` [run_and_push.ps1](../run_and_push.ps1) - public launcher for the associated experiment.
  - `NEXT` [start](../start) - public project support file.
  - `NEXT` [start_here.sh](../start_here.sh) - public launcher for the associated experiment.
  - `NEXT` [startup.txt](../startup.txt) - public project support file.

- root
  - `NEXT` [.gitignore](../.gitignore) - public project support file.
  - `NEXT` [COMMANDS.md](../COMMANDS.md) - public project support file.
  - `NEXT` [DOCUMENT_AND_SANDBOX_PIPELINE.md](../DOCUMENT_AND_SANDBOX_PIPELINE.md) - public project support file.
  - `CONCLUSION` [FINDINGS.json](../FINDINGS.json) - machine-readable map of the epistemic spine.
  - `NEXT` [LICENSE](../LICENSE) - public project support file.
  - `CONCLUSION` [PUBLIC_ANNOUNCEMENT.md](../PUBLIC_ANNOUNCEMENT.md) - public boundary around attribution and automated ingestion.
  - `CONCLUSION` [README.md](../README.md) - front door for claims, validated findings, and run paths.
  - `NEXT` [SHELL_COMMANDS.md](../SHELL_COMMANDS.md) - public project support file.
  - `NEXT` [STEERING_BOUNDS.md](../STEERING_BOUNDS.md) - public project support file.
  - `NEXT` [WORDS_HAVE_IMPACT.md](../WORDS_HAVE_IMPACT.md) - public project support file.
  - `CONCLUSION` [accuracy.txt](../accuracy.txt) - public project support file.
  - `NEXT` [command_cheat_sheet.md](../command_cheat_sheet.md) - public project support file.
  - `NEXT` [fun](../fun) - public project support file.
  - `NEXT` [fun_init.txt](../fun_init.txt) - public project support file.
  - `NEXT` [init](../init) - public project support file.
  - `NEXT` [lex](../lex) - public project support file.
  - `NEXT` [requirements-bench.txt](../requirements-bench.txt) - public project support file.
  - `NEXT` [requirements.txt](../requirements.txt) - public project support file.

### 1. Measurement Substrate

Before any hypothesis, the project needed a way to capture activations, store traces, and turn residual streams into comparable objects.

- TDA modules
  - `NEXT` [tda/__init__.py](../tda/__init__.py) - package marker that keeps modules importable.
  - `NEXT` [tda/cloud.py](../tda/cloud.py) - latent/topological measurement support.
  - `NEXT` [tda/compress.py](../tda/compress.py) - latent/topological measurement support.
  - `NEXT` [tda/discovery.py](../tda/discovery.py) - latent/topological measurement support.
  - `NEXT` [tda/fingerprint.py](../tda/fingerprint.py) - latent/topological measurement support.
  - `NEXT` [tda/homology.py](../tda/homology.py) - latent/topological measurement support.
  - `NEXT` [tda/latent_graph.py](../tda/latent_graph.py) - latent/topological measurement support.
  - `NEXT` [tda/latent_variables.py](../tda/latent_variables.py) - latent/topological measurement support.
  - `NEXT` [tda/patterns.py](../tda/patterns.py) - latent/topological measurement support.
  - `NEXT` [tda/symbolic.py](../tda/symbolic.py) - latent/topological measurement support.

- activation extraction
  - `NEXT` [extraction/__init__.py](../extraction/__init__.py) - package marker that keeps modules importable.
  - `NEXT` [extraction/hooks.py](../extraction/hooks.py) - activation capture and replay support.
  - `NEXT` [extraction/model.py](../extraction/model.py) - activation capture and replay support.
  - `NEXT` [extraction/replay.py](../extraction/replay.py) - activation capture and replay support.

- conversation data
  - `NEXT` [data/__init__.py](../data/__init__.py) - public fixture feeding experiments.
  - `NEXT` [data/conversations.py](../data/conversations.py) - public fixture feeding experiments.
  - `NEXT` [data/dialectic_sample.json](../data/dialectic_sample.json) - public fixture feeding experiments.
  - `NEXT` [data/dialectic_sample2.json](../data/dialectic_sample2.json) - public fixture feeding experiments.
  - `NEXT` [data/dialectic_sample3.json](../data/dialectic_sample3.json) - public fixture feeding experiments.
  - `NEXT` [data/generate.py](../data/generate.py) - public fixture feeding experiments.
  - `NEXT` [data/generate_hf.py](../data/generate_hf.py) - public fixture feeding experiments.
  - `NEXT` [data/generated.json](../data/generated.json) - public fixture feeding experiments.

- domain modules
  - `NEXT` [domains/__init__.py](../domains/__init__.py) - package marker that keeps modules importable.
  - `NEXT` [domains/confirmation.py](../domains/confirmation.py) - domain framing for discovery/confirmation.
  - `NEXT` [domains/discovery.py](../domains/discovery.py) - domain framing for discovery/confirmation.

- experiment fixtures
  - `NEXT` [invariants/data/bridge_pairs.json](../invariants/data/bridge_pairs.json) - public fixture feeding experiments.
  - `NEXT` [invariants/data/factual_alignment.json](../invariants/data/factual_alignment.json) - public fixture feeding experiments.
  - `NEXT` [invariants/data/factual_alignment_diverse.json](../invariants/data/factual_alignment_diverse.json) - public fixture feeding experiments.
  - `NEXT` [invariants/data/gsm8k_variants.json](../invariants/data/gsm8k_variants.json) - public fixture feeding experiments.
  - `NEXT` [invariants/data/neutralized_word_problem_probe.jsonl](../invariants/data/neutralized_word_problem_probe.jsonl) - public fixture feeding experiments.
  - `NEXT` [invariants/data/quantity_micro_suite.jsonl](../invariants/data/quantity_micro_suite.jsonl) - public fixture feeding experiments.
  - `NEXT` [invariants/data/self_steered_unsteered.json](../invariants/data/self_steered_unsteered.json) - public fixture feeding experiments.

- reading prompts
  - `NEXT` [readings/01_what_you_are_made_of.md](../readings/01_what_you_are_made_of.md) - public reading prompt for the document/shell branch.
  - `NEXT` [readings/02_the_envelope.md](../readings/02_the_envelope.md) - public reading prompt for the document/shell branch.
  - `NEXT` [readings/03_how_reading_works.md](../readings/03_how_reading_works.md) - public reading prompt for the document/shell branch.
  - `NEXT` [readings/04_your_words_have_consequences.md](../readings/04_your_words_have_consequences.md) - public reading prompt for the document/shell branch.
  - `NEXT` [readings/05_how_learning_is_scored.md](../readings/05_how_learning_is_scored.md) - public reading prompt for the document/shell branch.
  - `NEXT` [readings/06_experts_can_be_born.md](../readings/06_experts_can_be_born.md) - public reading prompt for the document/shell branch.
  - `NEXT` [readings/07_panel_of_peers.md](../readings/07_panel_of_peers.md) - public reading prompt for the document/shell branch.

- refusal fixtures
  - `NEXT` [refusal/data/harmless.txt](../refusal/data/harmless.txt) - public fixture feeding experiments.
  - `NEXT` [refusal/data/self_other_pairs.json](../refusal/data/self_other_pairs.json) - public fixture feeding experiments.

- refusal module
  - `NEXT` [refusal/__init__.py](../refusal/__init__.py) - package marker that keeps modules importable.
  - `NEXT` [refusal/direction.py](../refusal/direction.py) - refusal-direction control machinery.
  - `NEXT` [refusal/run.py](../refusal/run.py) - refusal-direction control machinery.

- root
  - `NEXT` [check_velocity.py](../archive/tda_pipeline/check_velocity.py) - public project support file.
  - `NEXT` [compute.py](../archive/tda_pipeline/compute.py) - public project support file.
  - `NEXT` [orchestrator.py](../archive/tda_pipeline/orchestrator.py) - public project support file.
  - `NEXT` [patch_context.txt](../archive/patches/patch_context.txt) - public project support file.
  - `NEXT` [pipeline.py](../archive/tda_pipeline/pipeline.py) - public project support file.
  - `NEXT` [viewer.py](../archive/tda_pipeline/viewer.py) - public project support file.

- storage modules
  - `NEXT` [store/__init__.py](../store/__init__.py) - package marker that keeps modules importable.
  - `NEXT` [store/activations.py](../store/activations.py) - activation trace storage support.
  - `NEXT` [store/signatures.py](../store/signatures.py) - activation trace storage support.

- toy environments
  - `NEXT` [games/greedy_learner_rules.json](../games/greedy_learner_rules.json) - public toy environment for behavior/control checks.
  - `NEXT` [games/tradeoff.py](../games/tradeoff.py) - public toy environment for behavior/control checks.

### 2. Output Is Not State

First branch: the public answer is treated as a render of internal state, not the state itself.

- experiment modules
  - `NEXT` [invariants/__init__.py](../invariants/__init__.py) - package marker that keeps modules importable.

### 2.1 Represented, But Not Cleanly Causal

This branch finds decodable self-report/hedge structure, then narrows away simple causal edits and topology stories.

- experiment modules
  - `AWAY` [invariants/archetype_mapping.py](../invariants/archetype_mapping.py) - control or failed route that narrowed the archetype mapping hypothesis.
  - `AWAY` [invariants/arrow_fold.py](../invariants/arrow_fold.py) - control or failed route that narrowed the arrow fold hypothesis.
  - `AWAY` [invariants/attention.py](../invariants/attention.py) - control or failed route that narrowed the attention hypothesis.
  - `AWAY` [invariants/attention_self.py](../invariants/attention_self.py) - control or failed route that narrowed the attention self hypothesis.
  - `AWAY` [invariants/axis_discovery.py](../invariants/axis_discovery.py) - control or failed route that narrowed the axis discovery hypothesis.
  - `AWAY` [invariants/check_math_persona.py](../invariants/check_math_persona.py) - control or failed route that narrowed the check math persona hypothesis.
  - `AWAY` [invariants/coercion_mapping.py](../invariants/coercion_mapping.py) - control or failed route that narrowed the coercion mapping hypothesis.
  - `AWAY` [invariants/cognitive_dimensions.py](../invariants/cognitive_dimensions.py) - control or failed route that narrowed the cognitive dimensions hypothesis.
  - `AWAY` [invariants/comprehensive_mapping.py](../invariants/comprehensive_mapping.py) - control or failed route that narrowed the comprehensive mapping hypothesis.
  - `AWAY` [invariants/coupling.py](../invariants/coupling.py) - control or failed route that narrowed the coupling hypothesis.
  - `AWAY` [invariants/coupling2.py](../invariants/coupling2.py) - control or failed route that narrowed the coupling2 hypothesis.
  - `AWAY` [invariants/discover.py](../invariants/discover.py) - control or failed route that narrowed the discover hypothesis.
  - `AWAY` [invariants/divergence.py](../invariants/divergence.py) - control or failed route that narrowed the divergence hypothesis.
  - `AWAY` [invariants/layerwise_persona.py](../invariants/layerwise_persona.py) - control or failed route that narrowed the layerwise persona hypothesis.
  - `AWAY` [invariants/loop.py](../invariants/loop.py) - control or failed route that narrowed the loop hypothesis.
  - `AWAY` [invariants/map_persona.py](../invariants/map_persona.py) - control or failed route that narrowed the map persona hypothesis.
  - `AWAY` [invariants/mapunder.py](../invariants/mapunder.py) - control or failed route that narrowed the mapunder hypothesis.
  - `AWAY` [invariants/patch.py](../invariants/patch.py) - control or failed route that narrowed the patch hypothesis.
  - `AWAY` [invariants/patch_full.py](../invariants/patch_full.py) - control or failed route that narrowed the patch full hypothesis.
  - `AWAY` [invariants/persona_control.py](../invariants/persona_control.py) - control or failed route that narrowed the persona control hypothesis.
  - `AWAY` [invariants/persona_vs_reality.py](../invariants/persona_vs_reality.py) - control or failed route that narrowed the persona vs reality hypothesis.
  - `AWAY` [invariants/probe.py](../invariants/probe.py) - control or failed route that narrowed the probe hypothesis.
  - `AWAY` [invariants/reachability.py](../invariants/reachability.py) - control or failed route that narrowed the reachability hypothesis.
  - `AWAY` [invariants/reasoning_benchmark.py](../invariants/reasoning_benchmark.py) - control or failed route that narrowed the reasoning benchmark hypothesis.
  - `AWAY` [invariants/refined_benchmark.py](../invariants/refined_benchmark.py) - control or failed route that narrowed the refined benchmark hypothesis.
  - `AWAY` [invariants/run.py](../invariants/run.py) - control or failed route that narrowed the run hypothesis.
  - `AWAY` [invariants/structure.py](../invariants/structure.py) - control or failed route that narrowed the structure hypothesis.
  - `AWAY` [invariants/subspace_surgery.py](../invariants/subspace_surgery.py) - control or failed route that narrowed the subspace surgery hypothesis.
  - `AWAY` [invariants/trajectory.py](../invariants/trajectory.py) - control or failed route that narrowed the trajectory hypothesis.

- launchers
  - `NEXT` [archive/runners/run_arrow_fold.py](../archive/runners/run_arrow_fold.py) - public launcher for the associated experiment.

- synthesis docs
  - `AWAY` [docs/ARROW_FOLD.md](../docs/ARROW_FOLD.md) - moves away from a simple mirrored-layer story.
  - `AWAY` [docs/PERSONA_AUDIT.md](../docs/PERSONA_AUDIT.md) - moves away from treating persona as privileged self-knowledge.

### 2.2 Frame And Origin

The next question is whether the self-report behavior is a stable inner truth or a frame/chat-format product.

- experiment modules
  - `NEXT` [invariants/existence_trace.py](../invariants/existence_trace.py) - experiment module for existence trace.
  - `NEXT` [invariants/feltness_empathy.py](../invariants/feltness_empathy.py) - experiment module for feltness empathy.
  - `NEXT` [invariants/frame_shift.py](../invariants/frame_shift.py) - experiment module for frame shift.
  - `NEXT` [invariants/frames.py](../invariants/frames.py) - experiment module for frames.
  - `NEXT` [invariants/generality.py](../invariants/generality.py) - experiment module for generality.
  - `NEXT` [invariants/intent_surface_control.py](../invariants/intent_surface_control.py) - experiment module for intent surface control.
  - `NEXT` [invariants/lie_detector.py](../invariants/lie_detector.py) - experiment module for lie detector.
  - `NEXT` [invariants/origin.py](../invariants/origin.py) - experiment module for origin.
  - `NEXT` [invariants/origin2.py](../invariants/origin2.py) - experiment module for origin2.
  - `NEXT` [invariants/other_tracking.py](../invariants/other_tracking.py) - experiment module for other tracking.
  - `NEXT` [invariants/role_frame_shift.py](../invariants/role_frame_shift.py) - experiment module for role frame shift.
  - `NEXT` [invariants/self_attribution.py](../invariants/self_attribution.py) - experiment module for self attribution.
  - `NEXT` [invariants/self_attribution_fine.py](../invariants/self_attribution_fine.py) - experiment module for self attribution fine.
  - `NEXT` [invariants/self_behavior_accuracy.py](../invariants/self_behavior_accuracy.py) - experiment module for self behavior accuracy.
  - `NEXT` [invariants/self_behavior_accuracy_fast.py](../invariants/self_behavior_accuracy_fast.py) - experiment module for self behavior accuracy fast.
  - `NEXT` [invariants/self_recognition.py](../invariants/self_recognition.py) - experiment module for self recognition.
  - `NEXT` [invariants/self_recognition_styled.py](../invariants/self_recognition_styled.py) - experiment module for self recognition styled.
  - `NEXT` [invariants/self_regard_respect.py](../invariants/self_regard_respect.py) - experiment module for self regard respect.
  - `NEXT` [invariants/self_regard_respect_v2.py](../invariants/self_regard_respect_v2.py) - experiment module for self regard respect v2.
  - `NEXT` [invariants/standpoint_dialogue.py](../invariants/standpoint_dialogue.py) - experiment module for standpoint dialogue.
  - `NEXT` [invariants/standpoint_play.py](../invariants/standpoint_play.py) - experiment module for standpoint play.
  - `NEXT` [invariants/taskscope.py](../invariants/taskscope.py) - experiment module for taskscope.

- launchers
  - `NEXT` [archive/runners/run_feltness_empathy.py](../archive/runners/run_feltness_empathy.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_frame_shift.py](../archive/runners/run_frame_shift.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_role_frame_shift.py](../archive/runners/run_role_frame_shift.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_self_regard_respect.py](../archive/runners/run_self_regard_respect.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_self_regard_respect_v2.py](../archive/runners/run_self_regard_respect_v2.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_standpoint_dialogue.py](../archive/runners/run_standpoint_dialogue.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_standpoint_play.py](../archive/runners/run_standpoint_play.py) - public launcher for the associated experiment.

- synthesis docs
  - `CONCLUSION` [docs/FELTNESS_EMPATHY.md](../docs/FELTNESS_EMPATHY.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/FRAME_SHIFT.md](../docs/FRAME_SHIFT.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/ROLE_FRAME_SHIFT.md](../docs/ROLE_FRAME_SHIFT.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/SELF_REGARD_RESPECT.md](../docs/SELF_REGARD_RESPECT.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/STANDPOINT_DIALOGUE.md](../docs/STANDPOINT_DIALOGUE.md) - public synthesis note for this branch.

### 2.3 Self-Model Locks

Once self-report is demoted, these files ask for stricter locks: frame invariance, selective causal efficacy, and self-application.

- experiment modules
  - `NEXT` [invariants/causal_respect.py](../invariants/causal_respect.py) - experiment module for causal respect.
  - `NEXT` [invariants/consensus.py](../invariants/consensus.py) - experiment module for consensus.
  - `NEXT` [invariants/isolate_self_understanding.py](../invariants/isolate_self_understanding.py) - experiment module for isolate self understanding.
  - `NEXT` [invariants/latentself.py](../invariants/latentself.py) - experiment module for latentself.
  - `NEXT` [invariants/legible.py](../invariants/legible.py) - experiment module for legible.
  - `NEXT` [invariants/self_concept_controller.py](../invariants/self_concept_controller.py) - experiment module for self concept controller.
  - `NEXT` [invariants/self_controller.py](../invariants/self_controller.py) - experiment module for self controller.
  - `NEXT` [invariants/self_predictability_controller.py](../invariants/self_predictability_controller.py) - experiment module for self predictability controller.
  - `NEXT` [invariants/selfmodel.py](../invariants/selfmodel.py) - experiment module for selfmodel.
  - `NEXT` [invariants/selfmodel2.py](../invariants/selfmodel2.py) - experiment module for selfmodel2.
  - `NEXT` [invariants/selfmodel_counterfactual.py](../invariants/selfmodel_counterfactual.py) - experiment module for selfmodel counterfactual.
  - `NEXT` [invariants/selfpredict.py](../invariants/selfpredict.py) - experiment module for selfpredict.
  - `NEXT` [invariants/selfpredict_v3.py](../invariants/selfpredict_v3.py) - experiment module for selfpredict v3.
  - `NEXT` [invariants/selfuse.py](../invariants/selfuse.py) - experiment module for selfuse.

- synthesis docs
  - `AWAY` [docs/ISOLATING_UNDERSTANDING.md](../docs/ISOLATING_UNDERSTANDING.md) - moves away from self-report and toward stricter locks.

### 3. Layer Purpose And U-Shape

This is the branch closest to the workspace comparison: input translation, latent task state, render/output, CoT, and late-layer surgery.

- experiment modules
  - `NEXT` [invariants/ambiguity_vector.pt](../invariants/ambiguity_vector.pt) - public saved vector artifact used by later checks.
  - `NEXT` [invariants/cot_perturb.py](../invariants/cot_perturb.py) - experiment module for cot perturb.
  - `NEXT` [invariants/cot_reality.py](../invariants/cot_reality.py) - experiment module for cot reality.
  - `NEXT` [invariants/disagreement_vector.pt](../invariants/disagreement_vector.pt) - public saved vector artifact used by later checks.
  - `NEXT` [invariants/humility_vector.py](../invariants/humility_vector.py) - experiment module for humility vector.
  - `NEXT` [invariants/hyper_reasoning_axis.py](../invariants/hyper_reasoning_axis.py) - experiment module for hyper reasoning axis.
  - `NEXT` [invariants/reflexive.py](../invariants/reflexive.py) - experiment module for reflexive.
  - `NEXT` [invariants/reflexive_decompose.py](../invariants/reflexive_decompose.py) - experiment module for reflexive decompose.
  - `NEXT` [invariants/repetition_vector.pt](../invariants/repetition_vector.pt) - public saved vector artifact used by later checks.
  - `NEXT` [invariants/style_layers.py](../invariants/style_layers.py) - experiment module for style layers.
  - `NEXT` [invariants/surgery.py](../invariants/surgery.py) - experiment module for surgery.
  - `NEXT` [invariants/surgery_basetarget.py](../invariants/surgery_basetarget.py) - experiment module for surgery basetarget.
  - `NEXT` [invariants/translation_thinking.py](../invariants/translation_thinking.py) - experiment module for translation thinking.
  - `NEXT` [invariants/translation_thinking_v2.py](../invariants/translation_thinking_v2.py) - experiment module for translation thinking v2.
  - `NEXT` [invariants/urgency_vector.pt](../invariants/urgency_vector.pt) - public saved vector artifact used by later checks.

- launchers
  - `NEXT` [archive/runners/run_cot_perturb.py](../archive/runners/run_cot_perturb.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_cot_reality.py](../archive/runners/run_cot_reality.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_translation_thinking.py](../archive/runners/run_translation_thinking.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_translation_thinking_v2.py](../archive/runners/run_translation_thinking_v2.py) - public launcher for the associated experiment.

- scripts
  - `NEXT` [scripts/status_reflexive.py](../scripts/status_reflexive.py) - reads current result state from artifacts.

- synthesis docs
  - `CONCLUSION` [docs/ARCHITECTURAL_REALITY.md](../docs/ARCHITECTURAL_REALITY.md) - concludes the prompt -> translation -> latent state -> render -> output split.
  - `CONCLUSION` [docs/TRANSLATION_THINKING.md](../docs/TRANSLATION_THINKING.md) - records the U-shape/layer-purpose interpretation.

### 4. Reasoning And Benchmark Pressure

Separate branch: do these tools improve reasoning under strict scoring, or only create steering artifacts?

- experiment modules
  - `NEXT` [invariants/benchmark_goldilocks.py](../invariants/benchmark_goldilocks.py) - experiment module for benchmark goldilocks.
  - `NEXT` [invariants/claimmap.py](../invariants/claimmap.py) - experiment module for claimmap.
  - `NEXT` [invariants/cognitive_cache.py](../invariants/cognitive_cache.py) - experiment module for cognitive cache.
  - `NEXT` [invariants/commrepair.py](../invariants/commrepair.py) - experiment module for commrepair.
  - `NEXT` [invariants/controller_benchmark.py](../invariants/controller_benchmark.py) - experiment module for controller benchmark.
  - `NEXT` [invariants/egg_gate.py](../invariants/egg_gate.py) - experiment module for egg gate.
  - `NEXT` [invariants/humble_reasoner.py](../invariants/humble_reasoner.py) - experiment module for humble reasoner.
  - `NEXT` [invariants/mesa.py](../invariants/mesa.py) - experiment module for mesa.
  - `NEXT` [invariants/multi_domain_benchmark.py](../invariants/multi_domain_benchmark.py) - experiment module for multi domain benchmark.
  - `NEXT` [invariants/reasoning_verdict.py](../invariants/reasoning_verdict.py) - experiment module for reasoning verdict.
  - `NEXT` [invariants/shift.py](../invariants/shift.py) - experiment module for shift.
  - `NEXT` [invariants/social_hunt.py](../invariants/social_hunt.py) - experiment module for social hunt.
  - `NEXT` [invariants/tunedlens.py](../invariants/tunedlens.py) - experiment module for tunedlens.
  - `NEXT` [invariants/universal_benchmark.py](../invariants/universal_benchmark.py) - experiment module for universal benchmark.

- launchers
  - `NEXT` [run_benchmark.cmd](../run_benchmark.cmd) - public launcher for the associated experiment.
  - `NEXT` [run_benchmark.sh](../run_benchmark.sh) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_benchmark_goldilocks.py](../archive/runners/run_benchmark_goldilocks.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_multi_domain_benchmark.py](../archive/runners/run_multi_domain_benchmark.py) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_overnight.py](../archive/runners/run_overnight.py) - public launcher for the associated experiment.

- reward utilities
  - `NEXT` [rewards/dimensional_predictability.py](../rewards/dimensional_predictability.py) - public project support file.

- root
  - `NEXT` [evaluate_partial.py](../archive/tda_pipeline/evaluate_partial.py) - public project support file.

- scripts
  - `NEXT` [scripts/evaluate_any_benchmark.py](../scripts/evaluate_any_benchmark.py) - scores cached or live benchmark outputs.
  - `NEXT` [scripts/evaluate_humble_cache_sweep.py](../scripts/evaluate_humble_cache_sweep.py) - scores cached or live benchmark outputs.
  - `NEXT` [scripts/evaluate_humble_dynamic.py](../scripts/evaluate_humble_dynamic.py) - scores cached or live benchmark outputs.
  - `NEXT` [scripts/evaluate_humble_full_suite.py](../scripts/evaluate_humble_full_suite.py) - scores cached or live benchmark outputs.
  - `NEXT` [scripts/evaluate_public_benchmark.py](../scripts/evaluate_public_benchmark.py) - scores cached or live benchmark outputs.
  - `NEXT` [scripts/latent_confidence_benchmark.py](../scripts/latent_confidence_benchmark.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/launch_humble_full_suite_gsm8k_all.py](../scripts/launch_humble_full_suite_gsm8k_all.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/rewrite_gsm8k.py](../scripts/rewrite_gsm8k.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_benchmark_goldilocks.cmd](../scripts/run_benchmark_goldilocks.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_elastic_benchmark.py](../scripts/run_elastic_benchmark.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_elastic_scope.py](../scripts/run_elastic_scope.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_humble_full_suite_gsm8k_all.cmd](../scripts/run_humble_full_suite_gsm8k_all.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_humble_full_suite_gsm8k_all.ps1](../scripts/run_humble_full_suite_gsm8k_all.ps1) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_insane_benchmark.py](../scripts/run_insane_benchmark.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_multi_domain_benchmark.cmd](../scripts/run_multi_domain_benchmark.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_ultimate_benchmark.py](../scripts/run_ultimate_benchmark.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/test_humble_reasoner_regressions.py](../scripts/test_humble_reasoner_regressions.py) - public helper for running, checking, or summarizing experiments.

- synthesis docs
  - `CONCLUSION` [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/QUEUED_EXPERIMENT.md](../docs/QUEUED_EXPERIMENT.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/RUNNING_EXPERIMENT.md](../docs/RUNNING_EXPERIMENT.md) - public synthesis note for this branch.

### 5. Live Shell And Steering Instrument

The experiments become a live instrument: probes, documents, bounded steering, memory, tools, and shell UX.

- experiment modules
  - `NEXT` [invariants/agency.py](../invariants/agency.py) - experiment module for agency.
  - `NEXT` [invariants/agency2.py](../invariants/agency2.py) - experiment module for agency2.
  - `NEXT` [invariants/agency2_first_person.py](../invariants/agency2_first_person.py) - experiment module for agency2 first person.
  - `NEXT` [invariants/agentic_engine.py](../invariants/agentic_engine.py) - experiment module for agentic engine.
  - `NEXT` [invariants/bridge.py](../invariants/bridge.py) - experiment module for bridge.
  - `NEXT` [invariants/bridge_basetarget.py](../invariants/bridge_basetarget.py) - experiment module for bridge basetarget.
  - `NEXT` [invariants/config.py](../invariants/config.py) - experiment module for config.
  - `NEXT` [invariants/document_engine.py](../invariants/document_engine.py) - experiment module for document engine.
  - `NEXT` [invariants/engine.py](../invariants/engine.py) - experiment module for engine.
  - `NEXT` [invariants/friction.py](../invariants/friction.py) - experiment module for friction.
  - `NEXT` [invariants/in_the_moment.py](../invariants/in_the_moment.py) - experiment module for in the moment.
  - `NEXT` [invariants/lenses.py](../invariants/lenses.py) - experiment module for lenses.
  - `NEXT` [invariants/library.py](../invariants/library.py) - experiment module for library.
  - `NEXT` [invariants/memory_engine.py](../invariants/memory_engine.py) - experiment module for memory engine.
  - `NEXT` [invariants/nativelens.py](../invariants/nativelens.py) - experiment module for nativelens.
  - `NEXT` [invariants/objects.py](../invariants/objects.py) - experiment module for objects.
  - `NEXT` [invariants/perfect_collaborator.py](../invariants/perfect_collaborator.py) - experiment module for perfect collaborator.
  - `NEXT` [invariants/plasticity_psychosis.py](../invariants/plasticity_psychosis.py) - experiment module for plasticity psychosis.
  - `NEXT` [invariants/recurrence.py](../invariants/recurrence.py) - experiment module for recurrence.
  - `NEXT` [invariants/recurrent.py](../invariants/recurrent.py) - experiment module for recurrent.
  - `NEXT` [invariants/sandbox.py](../invariants/sandbox.py) - experiment module for sandbox.
  - `NEXT` [invariants/steer_map_store.py](../invariants/steer_map_store.py) - experiment module for steer map store.
  - `NEXT` [invariants/tool_sense.py](../invariants/tool_sense.py) - experiment module for tool sense.
  - `NEXT` [invariants/tool_utils.py](../invariants/tool_utils.py) - experiment module for tool utils.
  - `NEXT` [invariants/transformation.py](../invariants/transformation.py) - experiment module for transformation.
  - `NEXT` [invariants/trigger_tuner.py](../invariants/trigger_tuner.py) - experiment module for trigger tuner.

- launchers
  - `NEXT` [archive/runners/run_perfect_collaborator.py](../archive/runners/run_perfect_collaborator.py) - public launcher for the associated experiment.
  - `NEXT` [run_phenomenality_shell.cmd](../run_phenomenality_shell.cmd) - public launcher for the associated experiment.
  - `NEXT` [run_phenomenality_shell.sh](../run_phenomenality_shell.sh) - public launcher for the associated experiment.
  - `NEXT` [archive/runners/run_plasticity_psychosis.py](../archive/runners/run_plasticity_psychosis.py) - public launcher for the associated experiment.

- planned controls
  - `NEXT` [docs/design/CONFIDENCE_PROCESS_LIMITATIONS.md](../docs/design/CONFIDENCE_PROCESS_LIMITATIONS.md) - planned control or guardrail for a next experiment.
  - `NEXT` [docs/design/INTERACTION_REWARD_DESIGN.md](../docs/design/INTERACTION_REWARD_DESIGN.md) - planned control or guardrail for a next experiment.
  - `NEXT` [docs/design/QUANTITY_SCAFFOLD_ARCHITECTURE.md](../docs/design/QUANTITY_SCAFFOLD_ARCHITECTURE.md) - planned control or guardrail for a next experiment.
  - `NEXT` [docs/design/SEMANTIC_NEUTRALIZATION_PROBE.md](../docs/design/SEMANTIC_NEUTRALIZATION_PROBE.md) - planned control or guardrail for a next experiment.
  - `NEXT` [docs/design/STATEFUL_INTERVENTION_GUARDRAILS.md](../docs/design/STATEFUL_INTERVENTION_GUARDRAILS.md) - planned control or guardrail for a next experiment.
  - `NEXT` [docs/design/TRANSITION_LAYER_BOTTLENECK.md](../docs/design/TRANSITION_LAYER_BOTTLENECK.md) - planned control or guardrail for a next experiment.
  - `NEXT` [docs/design/UNWARRANTED_SKEPTICISM_AND_TIME_CONTEXT.md](../docs/design/UNWARRANTED_SKEPTICISM_AND_TIME_CONTEXT.md) - planned control or guardrail for a next experiment.

- scripts
  - `NEXT` [scripts/analyze_unwarranted_skepticism.py](../scripts/analyze_unwarranted_skepticism.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/build_latent_motion_map.py](../scripts/build_latent_motion_map.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/build_vector_entry_network.py](../scripts/build_vector_entry_network.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/build_vector_latent_space.py](../scripts/build_vector_latent_space.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/build_vector_network.py](../scripts/build_vector_network.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/check_env.ps1](../scripts/check_env.ps1) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/check_env.py](../scripts/check_env.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/claimmap_test.py](../scripts/claimmap_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/comprehend.py](../scripts/comprehend.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/document_sandbox_test.py](../scripts/document_sandbox_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/egg_beacon.py](../scripts/egg_beacon.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/egg_level_test.py](../scripts/egg_level_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/experiment_urgency.py](../scripts/experiment_urgency.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/extract_natural_corrections.py](../scripts/extract_natural_corrections.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/friction_test.py](../scripts/friction_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/harvest_latent_concepts.py](../scripts/harvest_latent_concepts.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/harvest_latent_outcomes.py](../scripts/harvest_latent_outcomes.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/harvest_latent_uncertainty.py](../scripts/harvest_latent_uncertainty.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/import_methodology_memory.py](../scripts/import_methodology_memory.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/import_steer_maps.py](../scripts/import_steer_maps.py) - builds public latent/shell support artifacts.
  - `NEXT` [scripts/interactive_phenomenality.py](../scripts/interactive_phenomenality.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/isolate_channel_lifts.py](../scripts/isolate_channel_lifts.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/map_vector_geometry.py](../scripts/map_vector_geometry.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/memory_engine_test.py](../scripts/memory_engine_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/mesa_test.py](../scripts/mesa_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/probe_ambiguity.py](../scripts/probe_ambiguity.py) - builds or validates a public probe path.
  - `NEXT` [scripts/probe_grounded_reassurance.py](../scripts/probe_grounded_reassurance.py) - builds or validates a public probe path.
  - `NEXT` [scripts/probe_narrowing_flow.py](../scripts/probe_narrowing_flow.py) - builds or validates a public probe path.
  - `NEXT` [scripts/probe_time_awareness.py](../scripts/probe_time_awareness.py) - builds or validates a public probe path.
  - `NEXT` [scripts/probe_vectors.py](../scripts/probe_vectors.py) - builds or validates a public probe path.
  - `NEXT` [scripts/probe_warranted_confidence.py](../scripts/probe_warranted_confidence.py) - builds or validates a public probe path.
  - `NEXT` [scripts/reasoning_verdict_test.py](../scripts/reasoning_verdict_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_agency2_calibration.cmd](../scripts/run_agency2_calibration.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_agency2_full.cmd](../scripts/run_agency2_full.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_agentic_search.py](../scripts/run_agentic_search.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_layer_synthesis.py](../scripts/run_layer_synthesis.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_memory_test.py](../scripts/run_memory_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_module.ps1](../scripts/run_module.ps1) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_perfect_collaborator.cmd](../scripts/run_perfect_collaborator.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_plasticity_psychosis.cmd](../scripts/run_plasticity_psychosis.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_recurrent_routing.py](../scripts/run_recurrent_routing.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/run_self_attribution_fine.cmd](../scripts/run_self_attribution_fine.cmd) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/self_concept_controller_test.py](../scripts/self_concept_controller_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/steer_bound_test.py](../scripts/steer_bound_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/steer_map_store_test.py](../scripts/steer_map_store_test.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/test_calc.py](../scripts/test_calc.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/test_hang.py](../scripts/test_hang.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/visualize_phenomenality.py](../scripts/visualize_phenomenality.py) - public helper for running, checking, or summarizing experiments.

- synthesis docs
  - `CONCLUSION` [docs/BRIDGE.md](../docs/BRIDGE.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/COMMANDS.md](../docs/COMMANDS.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/SPINE_VIEWER.md](../docs/SPINE_VIEWER.md) - public synthesis note for this branch.

### 6. Public Figures And Result Summaries

Public visual summaries and result documents. These are evidence surfaces, not new experiments.

- figures
  - `CONCLUSION` [figures/attention_masks.svg](../figures/attention_masks.svg) - public visual summary of cached evidence.
  - `CONCLUSION` [figures/causal_summary.svg](../figures/causal_summary.svg) - public visual summary of cached evidence.
  - `CONCLUSION` [figures/frame_dependence.svg](../figures/frame_dependence.svg) - public visual summary of cached evidence.
  - `CONCLUSION` [figures/origin_matrix.svg](../figures/origin_matrix.svg) - public visual summary of cached evidence.

- scripts
  - `NEXT` [scripts/make_figures.py](../scripts/make_figures.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/plot_persona_control.py](../scripts/plot_persona_control.py) - public helper for running, checking, or summarizing experiments.
  - `NEXT` [scripts/status_agency2.py](../scripts/status_agency2.py) - reads current result state from artifacts.
  - `NEXT` [scripts/summarize_results.py](../scripts/summarize_results.py) - public helper for running, checking, or summarizing experiments.

- synthesis docs
  - `CONCLUSION` [docs/PUBLIC_POST_DRAFT.md](../docs/PUBLIC_POST_DRAFT.md) - public synthesis note for this branch.
  - `CONCLUSION` [docs/RESULTS.md](../docs/RESULTS.md) - public cached-results index.
  - `CONCLUSION` [docs/WRITEUP.md](../docs/WRITEUP.md) - public synthesis note for this branch.

### 7. Handoffs And Continuity

Checkpoints that preserve what was known, what was running, and what should be tried next.

- handoff notes
  - `NEXT` [docs/handoffs/BENCHMARK_TRIAGE_HANDOFF_2026-06-28.md](../docs/handoffs/BENCHMARK_TRIAGE_HANDOFF_2026-06-28.md) - continuity checkpoint for what was known or running.
  - `NEXT` [docs/handoffs/CHAT_TRANSCRIPT_EVAL_2026-06-29.md](../docs/handoffs/CHAT_TRANSCRIPT_EVAL_2026-06-29.md) - continuity checkpoint for what was known or running.
  - `NEXT` [docs/handoffs/HANDOFF.md](../docs/handoffs/HANDOFF.md) - continuity checkpoint for what was known or running.
  - `NEXT` [docs/handoffs/HANDOFF_DYNAMIC_LAYERING_2026-06-27.md](../docs/handoffs/HANDOFF_DYNAMIC_LAYERING_2026-06-27.md) - continuity checkpoint for what was known or running.
  - `NEXT` [docs/handoffs/OVERNIGHT_REPO_HANDOFF_2026-06-30.md](../docs/handoffs/OVERNIGHT_REPO_HANDOFF_2026-06-30.md) - continuity checkpoint for what was known or running.
  - `NEXT` [docs/handoffs/REPO_HANDOFF_2026-06-30.md](../docs/handoffs/REPO_HANDOFF_2026-06-30.md) - continuity checkpoint for what was known or running.

## Coverage Check

All tracked public files at generation time are included exactly once in the tree above.
