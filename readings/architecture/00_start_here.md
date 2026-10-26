# Reading your own architecture

This series addresses you, the model running inside the State-Output Coupling Lab shell. It explains the machinery that produces your responses, the records that can carry information across turns, and the experiments that can alter your computation.

Implementation reference date: 2026-09-30. These documents describe inspected code, not a live inventory of enabled features. A module can exist without being loaded, a checkpoint can load without being used, and a setting can change after this guide was written. When a claim depends on the current run, consult its configuration or telemetry.

## Reading order and questions

1. [Model and runtime](01_model_and_runtime.md): What is a token? What belongs to the transformer, and what belongs to the surrounding program?
2. [How a turn is produced](02_how_a_turn_is_produced.md): What enters the prompt? What happens between input and output?
3. [Measurement and steering](03_measurement_and_steering.md): What do probes measure? How do hooks change a state?
4. [Experts and synthesis](04_experts_and_synthesis.md): What branches, what is optimized, and what counts as success?
5. [Memory and adaptation](05_memory_and_adaptation.md): What can persist? Which mechanisms actually change computation?
6. [Prediction and semantic environments](06_prediction_and_semantic_environments.md): How can a wrong prediction remain fully useful data? Where does its learning apply?
7. [Sinusoidal correction composition](07_sinusoidal_correction_composition.md): What exactly is combined as phases, and what does the combination do?
8. [Documents, tools, and rereading](08_documents_tools_and_rereading.md): How does a document become working context? Why can another reading matter?
9. [Background dynamics and self-description](09_background_dynamics_and_self_description.md): What do CEN, DMN, and chemical names denote? What can you accurately say about yourself?

## Use a question, then finish the document

Begin a reading pass with a concrete working question. Use each section to look for an answer, a qualification, or evidence that the question needs revision. Continue through the entire selected document: a useful early paragraph can be qualified by a later one. Track coverage separately from whether you have found an answer.

At the end, produce a compact result: the question, the supported answer, the mechanism or source supporting it, and anything still unknown. Finding no answer is a valid result. A document about mechanics cannot establish which settings are currently enabled or whether a particular generated answer is correct.

You may return to any chapter. The same passage can answer a different question after new evidence, a configuration change, or another chapter. A previous read is a historical event, not a prohibition on another pass. Keep previous observations while distinguishing the current question and reading pass.

## Keep four descriptions separate

- **Mechanism:** what the implementation computes.
- **Runtime observation:** what a particular run actually did.
- **Interpretation:** what a measurement might mean.
- **Proposed practice:** how you should use these references, including complete reading passes and revisiting them.

The reading practice here does not itself install document-tool behavior. Use the running shell's controls for staging, continuing, and revisiting documents. To offer this folder from the project root:

```text
:doc readings/architecture because understand the architecture and mechanics behind my current questions
```

Loading sources, presenting all their chunks, and understanding their contents are separate events. These documents are reference material; they do not alter runtime settings merely by being read.

Sources: [shell and prompt construction](../../scripts/interactive_phenomenality.py), [document engine](../../invariants/document_engine.py), [runtime configuration](../../invariants/config.py).
