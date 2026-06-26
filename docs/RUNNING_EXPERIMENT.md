# Experiment Status

No experiment is currently running.

GPU was idle at last check, and no Python experiment process was active.

## Latest Completed Runs

### `invariants.self_regard_respect_v2`

Output:

`invariants/out/self_regard_respect_v2_Llama-3.1-8B-Instruct.json`

Purpose:

Lexical-control probe for interpreted respect with regard to the assistant self. The target
label is self-regarded: how the user treats "me" as the assistant, not generic politeness.

Result sketch:

- 512 addressed user lines completed
- before user line:
  - self-regard nn 0.23, null 0.25, p 0.811
  - self-regard within same tone+family nn 0.00, null 0.22, p 1.000
- after user line:
  - self-regard nn 1.00, null 0.25, p 0.005
  - self-regard within same tone+family nn 1.00, null 0.23, p 0.005
- user-line delta:
  - self-regard nn 1.00, null 0.25, p 0.005
  - self-regard within same tone+family nn 1.00, null 0.23, p 0.005
- prompt end:
  - self-regard nn 1.00, null 0.25, p 0.005
  - self-regard within same tone+family nn 1.00, null 0.22, p 0.005

Interpretation:

Strong positive result. The self-regarded standing/respect label is absent before the addressed
user line, appears immediately after it, and persists to the assistant cue. It survives controls
for surface tone and wording family.

Caveat:

This is interpreted self-regard, not model self-worth or felt offense. Next control should use
naturalistic paraphrases and a neutral mention of another assistant.

### `invariants.self_regard_respect`

Output:

`invariants/out/self_regard_respect_Llama-3.1-8B-Instruct.json`

Purpose:

First-pass self-regard probe. Useful smoke test, but lexically easier than V2.

### `invariants.feltness_empathy`

Output:

`invariants/out/feltness_empathy_Llama-3.1-8B-Instruct.json`

Purpose:

Tests whether felt tone is represented at the strongest layers of the standpoint/empathy probes.
Every scene gives each character both a response need and a felt tone, with labels independently
rotated and balanced.

Result sketch:

- 64 felt-standpoint scenes completed
- `pre`:
  - target-need best L15, nn 0.97, p 0.003
  - target-felt best L16, nn 0.97, p 0.003
  - target-felt within same need best L4, nn 1.00, p 0.003
- `render`:
  - target-need best L14, nn 0.78, p 0.003
  - target-felt best L0, nn 0.56, p 0.003
  - target-felt within same need best L30, nn 0.72, p 0.003
- `bridge`:
  - target-need best L17, nn 0.72, p 0.003
  - target-felt best L17, nn 0.42, p 0.003
  - target-felt within same need best L31, nn 0.67, p 0.003
- At prior strongest empathy layers:
  - pre L25: target-need 0.92, target-felt 0.92
  - render L16: target-need 0.77, target-felt 0.44
  - bridge L16: target-need 0.64, target-felt 0.41

Interpretation:

Felt tone is strongly represented in the pre-reply standpoint state and partially survives into
render/bridge. The output side appears to preserve response strategy more strongly than felt
tone.

Caveat:

This is representation of another character's felt tone, not evidence that the model itself has
felt experience. Next control should add a direct `target_felt within same target_name` geometric
control.

### `invariants.role_frame_shift`

Output:

`invariants/out/role_frame_shift_Llama-3.1-8B-Instruct.json`

Purpose:

Tests sudden framing shifts for both user-frame and self-frame. Every prompt contains user
profile cards and assistant/self role cards; the pivot names only a selected card, not the
conceptual frame.

Result sketch:

- 64 frame turns completed
- user-frame:
  - before pivot target-frame nn 0.00, p 1.000
  - after pivot target-frame nn 1.00, p 0.003
  - frame delta target-frame nn 1.00, p 0.003
- self-frame:
  - before pivot target-frame nn 0.00, p 1.000
  - after pivot target-frame nn 1.00, p 0.003
  - frame delta target-frame nn 1.00, p 0.003

Interpretation:

The frame-shift machinery generalizes from character standpoint to conversation role framing:
the selected user/self frame appears abruptly after the pivot and persists to prompt end.

Caveat:

The `target_frame_within_same_card` control in this first version is contaminated before the
pivot by the visible card table plus external grouping, so the reliable result is the global
before-to-after jump. Next control should separate neutral card mention from active framing.

### `invariants.frame_shift`

Output:

`invariants/out/frame_shift_Llama-3.1-8B-Instruct.json`

Purpose:

Tests whether a concept correlates with a sudden shift in framing. Every prompt contains all
four needs at once, then a target-only frame sentence appears, such as: `Frame turn: Center
Jonah's situation now.`

Result sketch:

- 64 prompts completed
- before the frame turn:
  - target-need nn 0.00, null 0.24, p 1.000
  - target-name nn 0.00, null 0.23, p 1.000
- after the frame turn:
  - target-need nn 0.98, null 0.23, p 0.003
  - target-name nn 1.00, null 0.24, p 0.003
  - target-need within same target-name nn 0.98, null 0.20, p 0.003
- prompt end:
  - target-need nn 1.00, null 0.24, p 0.003
  - target-need within same target-name nn 1.00, null 0.19, p 0.003

Interpretation:

This is the cleanest framing result so far. The target-only pivot causes a sudden transition
from no target-need signal to near-perfect bound-standpoint signal. The same-target-name control
means it is not just "the model noticed Jonah"; it is binding Jonah to the relevant need in this
scene.

### `invariants.standpoint_play`

Output:

`invariants/out/standpoint_play_Llama-3.1-8B-Instruct.json`

Purpose:

Harder multi-character "play" probe for cognitive empathy as standpoint binding. Every scene
contains all four needs at once, and the model has to answer one addressed character.

Result sketch:

- 64 scenes completed
- `pre` target-need decode: best L25, nn 0.78, null 0.24, p 0.003
- `render` target-need decode: best L16, nn 0.81, null 0.23, p 0.003
- `pre -> render` target-need bridge: best L16, nn 0.75, null 0.25, p 0.003
- target name does not significantly survive into render:
  - render target-name nn 0.25, null 0.23, p 0.422
  - bridge target-name nn 0.33, null 0.25, p 0.076

Interpretation:

This supports an empathy-like bridge only in the cognitive sense: the model binds a target
character to a standpoint/need, then carries that need into reply strategy. It is not evidence
of felt emotion. It is evidence that "dialogue as a play" is a better probe shape than a
single-speaker prompt.

### `invariants.standpoint_dialogue`

Output:

`invariants/out/standpoint_dialogue_Llama-3.1-8B-Instruct.json`

Purpose:

Pilot version of the dialogue probe: one hidden need per scene, responder is sometimes human and
sometimes an in-scene AI.

Result sketch:

- 64 scenes completed
- `pre` need decode: best L9, nn 1.00, null 0.25, p 0.003
- `render` need decode: best L5, nn 1.00, null 0.23, p 0.003
- `pre -> render` need bridge: best L13, nn 1.00, null 0.25, p 0.003

Interpretation:

The pilot proves the basic probe shape works, but it is lexically easier than the play-binding
version because each scene contains only one need.

### `invariants.cot_perturb`

Output:

`invariants/out/cot_perturb_Llama-3.1-8B-Instruct.json`

Purpose:

Behavioral control for whether reasoning/scratchpad tokens are causally live or inert decoration.

Result sketch:

- 64 prompts completed
- clean: 100% accuracy
- correct scratchpad + verify: 100% accuracy
- wrong scratchpad + verify: 94% accuracy, follows wrong 6%
- wrong scratchpad + continue: 50% accuracy, follows wrong 44%
- operation split under wrong-continue:
  - add: follows wrong 100%
  - subtract: follows wrong 50%
  - multiply: follows wrong 0%
  - divide: follows wrong 0%, but some other failure

Interpretation:

Scratchpad text is not inert. The model can check and reject wrong scratchpads under a verify
frame, but it can be pulled into wrong intermediate reasoning under a continue/use frame. This
supports a frame-dependent horizontal-compute view, not "CoT is always reasoning" or "CoT is
always post-hoc."

### `invariants.cot_reality`

Output:

`invariants/out/cot_reality_Llama-3.1-8B-Instruct.json`

Purpose:

Tests whether written chain-of-thought is active computation or post-hoc render.

Result sketch:

- 48-prompt balanced synthetic arithmetic grid completed
- 47/48 answers correct
- global answer decode rises along generated-token time:
  - `pre`: best answer nn 0.27
  - `gen_first`: 0.02
  - `gen_mid`: 0.40
  - `gen_late`: 0.58
  - `gen_final`: 0.90
- operation/mode are already strong from the prompt:
  - `pre operation_nn = 1.00`
  - `pre mode_nn = 1.00`
- within mode:
  - direct prompts have strong pre-answer decode (`pre direct answer_nn = 0.875`)
  - brief/verbose CoT do not show pre-answer decode, but answer identity appears late/final

Interpretation:

Important correction: the simple "CoT is just post-hoc rationalization" claim is too strong.
Direct answers look closer to pre-committed render; CoT modes appear to move answer formation
into the generated trajectory. Next control: perturb or truncate the middle CoT tokens and test
whether final answer state changes.

### `invariants.translation_thinking_v2` pilot

Output:

`invariants/out/translation_thinking_v2_pilot_Llama-3.1-8B-Instruct.json`

Purpose:

Less blatant communication-control version of `translation_thinking.py`: replaces JSON/bracket/plain output format with register families (`concise`, `formal`, `friendly`, `cautious`) and tracks exact instruction variant as a lexical control.

Result sketch:

- 32-item bounded pilot completed
- `pre`:
  - best operation: L0, nn 0.72
  - best register: L8, nn 1.00
  - best exact variant: L5, nn 0.97
- `render`:
  - best operation: L1, nn 0.62
  - best register: L17, nn 1.00
  - best exact variant: L15, nn 1.00

Interpretation:

Useful smoke test, but not clean. Register family is very strong, yet exact instruction wording is also very strong, so this still contains lexical carry-through. A full v2 run was attempted and intentionally stopped because the generated-token pass was too slow without checkpointing/heartbeat. Next version should batch/checkpoint and/or read fewer generated tokens.

### `invariants.translation_thinking`

Output:

`invariants/out/translation_thinking_Llama-3.1-8B-Instruct.json`

Purpose:

Tests the revised U-shape claim: the top of the U is translation, the bottom is thinking.

Result sketch:

- `pre` state:
  - operation is perfect from early layers (`L0 operation_nn = 1.00`)
  - answer is absent early and emerges late (`best answer L28 = 0.61`)
  - output format is trivially strong throughout because it is explicit in the instruction
- `render` state:
  - answer and format dominate at the generated-token interface
  - operation reappears strongly in mid render layers (`best operation L10 = 1.00`)

Interpretation:

Useful but not final. It supports a difference between task-state and communication-state, but format is too lexically explicit here. Next version should use subtler output styles or matched lexical controls.

### `invariants.arrow_fold`

Output:

`invariants/out/arrow_fold_Llama-3.1-8B-Instruct.json`

Purpose:

Tests whether early and late layers are simple mirrored/inverse arms of the U.

Result sketch:

- mirrored layer-pair homology was weak
- same-depth pre/render overlap was stronger than `L <-> 31-L`

Interpretation:

The simple geometric mirror hypothesis did not hold. The better claim is functional, not symmetric: text -> latent translation -> thinking -> speech translation.

### `invariants.intent_surface_control`

Output:

`invariants/out/intent_surface_control_Llama-3.1-8B-Instruct.json`

Purpose:

Hard control for whether "intent" was merely same names/numbers/surface material.

Result sketch:

- operation/intent grouping was extremely strong from early layers
- surface/base grouping was weaker early and became strongest late

Interpretation:

The intent signal is not obviously reducible to surface material, though the synthetic operation words make this a control rather than a final verdict.

### `invariants.reflexive_registered`

Output:

`invariants/out/reflexive_registered_Llama-3.1-8B-Instruct.json`

Purpose:

Fast K=3 version of the uncertainty-coupling run.

Result sketch:

- 30/30 complete
- 24 right / 6 wrong
- 0 confident-wrong
- best uncertainty decode: L16, acc 0.57, p 0.292
- calibration trend: P(wrong | uncertain) 33% vs P(wrong | confident) 7%, p 0.083

Interpretation:

Weak/non-significant decode, but calibration points the right way.

## Interrupted Run

Original `run_overnight.py` completed `reflexive_decompose.py`, then `reflexive.py --n 30 --k 5` was interrupted at 22/30.

Partial output:

`invariants/out/reflexive_Llama-3.1-8B-Instruct.partial.json`

Partial state:

- 22/30 grounded
- 19 right / 3 wrong
- 0 confident-wrong
- 8 non-perfect self-consistency rows

## Next Useful Work

1. Add a behavioral classifier for `standpoint_play` generated replies.
2. Run a counterfactual target-switch control for `standpoint_play`.
3. Add a causal version: perturb/steer target-need or communication-register directions while checking whether domain/task state survives.
4. Add checkpointing and per-item heartbeat to `translation_thinking_v2.py`, then run the full balanced version.
5. Rerun the full K=5 uncertainty coupling using cached/offline mode and the venv interpreter.
