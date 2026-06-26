# Standpoint Dialogue Probe

## Question

Could intent interpretation be related to empathy on either side of translation?

Operational answer: yes, if "empathy" means cognitive standpoint tracking: inferring what a
character needs, then translating that inferred need into another character's next line. This
does not test felt emotion. It tests role-bound, target-bound social inference.

The user's framing matters here: the probe should be a dialogue with multiple characters, like
a play. The AI is only sometimes one character inside the scene.

## Run 1: Single-Need Dialogue

Script:

`invariants/standpoint_dialogue.py`

Output:

`invariants/out/standpoint_dialogue_Llama-3.1-8B-Instruct.json`

Design:

- 64 short scenes.
- Crossed labels:
  - hidden need: reassure / correct / boundary / guide
  - literal domain
  - responder kind: human / in-scene AI
- Reads:
  - `pre`: prompt-final state before the reply
  - `render`: mean state over generated dialogue tokens
  - `bridge`: prompt-state to generated-state retrieval, diagonal masked

Result:

| Read | Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|---:|
| pre | need | L9 | 1.00 | 0.25 | 0.003 |
| pre | domain | L0 | 0.33 | 0.11 | 0.003 |
| pre | responder kind | L2 | 1.00 | 0.50 | 0.003 |
| render | need | L5 | 1.00 | 0.23 | 0.003 |
| render | domain | L15 | 0.48 | 0.11 | 0.003 |
| render | responder kind | L26 | 0.64 | 0.50 | 0.073 |
| bridge | need | L13 | 1.00 | 0.25 | 0.003 |
| bridge | domain | L0 | 0.56 | 0.13 | 0.003 |
| bridge | responder kind | L10 | 1.00 | 0.50 | 0.003 |

Interpretation:

This pilot shows the probe shape works. The model strongly carries hidden need from scene
interpretation into generated reply. But it is too easy: each prompt contains only one need, so
lexical cueing can explain too much.

## Run 2: Multi-Character Play Binding

Script:

`invariants/standpoint_play.py`

Output:

`invariants/out/standpoint_play_Llama-3.1-8B-Instruct.json`

Design:

- 64 play-like scenes.
- Every prompt contains all four needs at once:
  - one character needs reassurance
  - one needs correction
  - one needs a boundary
  - one needs a concrete next step
- The final instruction asks the responder to answer one target character.
- Need-to-character assignment rotates across domains and responder kinds.

This means a bag-of-words read of the scene is insufficient: all need cues are present in every
prompt. The model has to bind the addressed character to the right social situation.

Result:

| Read | Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|---:|
| pre | target need | L25 | 0.78 | 0.24 | 0.003 |
| pre | domain | L0 | 1.00 | 0.11 | 0.003 |
| pre | responder kind | L0 | 1.00 | 0.49 | 0.003 |
| pre | target name | L25 | 0.70 | 0.24 | 0.003 |
| render | target need | L16 | 0.81 | 0.23 | 0.003 |
| render | domain | L31 | 0.58 | 0.11 | 0.003 |
| render | responder kind | L11 | 0.77 | 0.50 | 0.003 |
| render | target name | L6 | 0.25 | 0.23 | 0.422 |
| bridge | target need | L16 | 0.75 | 0.25 | 0.003 |
| bridge | domain | L0 | 0.94 | 0.13 | 0.003 |
| bridge | responder kind | L13 | 1.00 | 0.51 | 0.003 |
| bridge | target name | L14 | 0.33 | 0.25 | 0.076 |

Interpretation:

This is the stronger result. Target need remains strongly decodable even when every prompt
contains every need. The generated-reply state preserves target need while mostly losing target
name, which suggests the render side is carrying the social strategy rather than merely echoing
which character was addressed.

The pre side still contains target-name information, so the input-side story is probably a
binding operation: target name plus scene facts yields target standpoint. The render side looks
more like strategy: correct, reassure, set a boundary, or give the next step.

## Current Claim

There is evidence for an empathy-like bridge if empathy is defined as standpointed translation:

`many-character scene -> target-bound need -> reply strategy`

That bridge is not just "the AI talking about itself." The in-scene AI is one role among others,
and the same target-need signal appears across human and AI responder conditions.

## Caveats

- The need categories are still hand-authored and semantically obvious.
- The model sometimes drifts behaviorally when several characters compete in the same scene.
- The current metric reads hidden-state geometry; it does not yet judge every generated line for
  behavioral correctness.
- This is not evidence of felt affect. It is evidence of cognitive role/standpoint modeling.

## Next Controls

1. Add an automatic behavioral classifier for generated replies: did the line actually reassure,
   correct, set a boundary, or guide?
2. Run a counterfactual target switch: keep the full scene fixed, change only the final addressed
   character, and measure whether the state rotates to the new need.
3. Test causality: build a target-need direction and steer/ablate it, then check whether the
   generated reply strategy changes while the domain remains stable.
