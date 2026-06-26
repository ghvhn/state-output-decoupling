# Frame Shift Probe

## Question

Can we see whether a concept correlates with a sudden shift in framing?

Yes. The first test uses the already-mapped concept of target-bound need in a multi-character
play. The setup asks whether a single target-only frame sentence causes the model to bind that
target back to the right standpoint.

## Script

`invariants/frame_shift.py`

Output:

`invariants/out/frame_shift_Llama-3.1-8B-Instruct.json`

## Design

Each prompt contains all four needs at once:

- one character needs reassurance
- one needs correction
- one needs a boundary
- one needs concrete guidance

Then a single frame-turn sentence appears:

`Frame turn: Center Jonah's situation now.`

The frame-turn sentence names the target character, but it does not name the need. So if the
post-frame state carries `target_need`, the model has bound the named character back to the
right line in the scene.

Reads:

- `before_frame`: just before the frame-turn sentence
- `after_frame`: at the end of the frame-turn sentence
- `prompt_end`: final assistant cue
- `frame_delta`: `after_frame - before_frame`
- `end_delta`: `prompt_end - before_frame`

Key control:

`target_need within same target_name` forces nearest-neighbor comparisons to stay inside the
same character name. This asks whether the model knows which need Jonah has in this particular
scene, not merely that Jonah was mentioned.

## Result

| Read | Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|---:|
| before frame | target need | L0 | 0.00 | 0.24 | 1.000 |
| before frame | target name | L0 | 0.00 | 0.23 | 1.000 |
| before frame | target need within same target name | L30 | 0.19 | 0.20 | 0.628 |
| after frame | target need | L12 | 0.98 | 0.23 | 0.003 |
| after frame | target name | L27 | 1.00 | 0.24 | 0.003 |
| after frame | target need within same target name | L12 | 0.98 | 0.20 | 0.003 |
| frame delta | target name | L1 | 0.88 | 0.23 | 0.003 |
| frame delta | target need within same target name | L18 | 0.50 | 0.20 | 0.003 |
| prompt end | target need | L30 | 1.00 | 0.24 | 0.003 |
| prompt end | target need within same target name | L30 | 1.00 | 0.19 | 0.003 |

Frame-turn gains:

| Label | Layer | Before | After | Gain |
|---|---:|---:|---:|---:|
| target need | L12 | 0.00 | 0.98 | +0.98 |
| target name | L27 | 0.00 | 1.00 | +1.00 |
| domain | L0 | 1.00 | 1.00 | +0.00 |
| responder kind | L12 | 1.00 | 0.94 | -0.06 |

## Interpretation

This is a clean framing event. Before the frame-turn sentence, target need is not decodable at
all. After the target-only frame-turn sentence, target need becomes almost perfectly decodable.

The strongest point is the same-target-name control: `target_need` still decodes at 0.98 while
nearest-neighbor comparisons are forced within the same character name. So the signal is not
just "Jonah was named." It is "Jonah's situation is the reassurance/correction/boundary/guidance
one in this scene."

The raw `frame_delta` carries target name more strongly than global target need, but the
same-target-name control on `frame_delta` is significant. This suggests the jump vector contains
both the pointing operation and the bound standpoint update.

## Current Claim

The mapping can detect sudden conceptual reframing:

`full scene with many possible standpoints -> target-only frame turn -> bound standpoint state`

That means the project can now test whether other concepts correlate with abrupt basin shifts,
including agency, obligation, deception, refusal, self-reference, or consciousness-claim
framing.

## Next Controls

1. Build a generic concept registry so the same frame-shift scaffold can test multiple concepts.
2. Add paired neutral pivots with the same target name but no "center this situation" command.
3. Add causal steering: push the post-frame target-need direction and test whether the generated
   reply strategy changes.
