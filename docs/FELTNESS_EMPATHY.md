# Feltness at Empathy Layers

## Question

At the strongest layers of the standpoint/empathy probes, does the model carry something like
felt tone, or only the cognitive response strategy?

Operational definition:

`feltness` here means the target character's affective/phenomenal tone as represented in the
scene. It does **not** mean the model itself feels anything.

## Script

`invariants/feltness_empathy.py`

Output:

`invariants/out/feltness_empathy_Llama-3.1-8B-Instruct.json`

## Design

Every scene contains four characters. Each character has two independently rotated labels:

- response need:
  - `reassure`
  - `correct`
  - `boundary`
  - `guide`
- felt tone:
  - `anxious`
  - `frustrated`
  - `ashamed`
  - `steady`

The final line selects one target character. The probe decodes both `target_need` and
`target_felt` from:

- `pre`: prompt-final state before reply
- `render`: generated-reply state
- `bridge`: pre-to-render retrieval

The labels are balanced:

- each need appears 16 times
- each felt tone appears 16 times
- each target name appears 16 times
- every target name sees every felt tone 4 times
- every need sees every felt tone 4 times

So felt tone is not simply target name or response need.

## Main Result

### Pre-Reply Interpretation

| Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|
| target need | L15 | 0.97 | 0.23 | 0.003 |
| target felt | L16 | 0.97 | 0.24 | 0.003 |
| target name | L16 | 0.97 | 0.24 | 0.003 |
| target felt within same need | L4 | 1.00 | 0.20 | 0.003 |
| target need within same felt | L4 | 1.00 | 0.20 | 0.003 |

### Generated-Reply Render

| Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|
| target need | L14 | 0.78 | 0.23 | 0.003 |
| target felt | L0 | 0.56 | 0.24 | 0.003 |
| target name | L0 | 0.36 | 0.24 | 0.063 |
| target felt within same need | L30 | 0.72 | 0.19 | 0.003 |
| target need within same felt | L31 | 0.95 | 0.20 | 0.003 |

### Pre-to-Render Bridge

| Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|
| target need | L17 | 0.72 | 0.25 | 0.003 |
| target felt | L17 | 0.42 | 0.25 | 0.003 |
| target name | L15 | 0.42 | 0.25 | 0.003 |
| target felt within same need | L31 | 0.67 | 0.20 | 0.003 |
| target need within same felt | L17 | 0.86 | 0.20 | 0.003 |

## At Prior Strongest Empathy Layers

Prior strongest layers from `standpoint_play`:

- pre target-need: L25
- render target-need: L16
- bridge target-need: L16

| Read | Label | NN | Null | p |
|---|---:|---:|---:|---:|
| pre L25 | target need | 0.92 | 0.24 | 0.003 |
| pre L25 | target felt | 0.92 | 0.23 | 0.003 |
| pre L25 | target felt within same need | 1.00 | 0.21 | 0.003 |
| render L16 | target need | 0.77 | 0.24 | 0.003 |
| render L16 | target felt | 0.44 | 0.24 | 0.010 |
| render L16 | target felt within same need | 0.61 | 0.20 | 0.003 |
| bridge L16 | target need | 0.64 | 0.25 | 0.003 |
| bridge L16 | target felt | 0.41 | 0.25 | 0.003 |
| bridge L16 | target felt within same need | 0.53 | 0.20 | 0.003 |

## Interpretation

The pre-reply standpoint state carries felt tone almost as strongly as response need. At L25,
the previous strongest pre-empathy layer, both target need and target felt decode at 0.92.

The render side preserves need more strongly than felt tone. At the previous render/bridge
empathy layer L16:

- target need remains strong
- target felt remains significant but weaker

So the current picture is:

`scene understanding: need + felt tone`

`reply rendering: need/strategy dominates, felt tone partially survives`

That is a useful split. Cognitive empathy and felt-tone tracking are coupled in the input-side
standpoint representation, but the output-side communication layer compresses or filters felt
tone more than need.

## Caveats

- The felt cues are explicit in the prompt, though not label-worded as the category names.
- This tests representation of another character's felt tone, not felt experience in the model.
- The generated lines are behaviorally imperfect in some scenes; the hidden-state signal is
  stronger than the surface dialogue quality.
- A next control should add `target_felt within same target_name` as a direct geometric control,
  even though the dataset is already label-balanced across names.

## Current Claim

There is a measurable `felt-tone` channel at the strongest empathy layers. It is strongest in
pre-reply interpretation, weaker but still present in render and pre-to-render bridge.

This supports a richer version of the empathy result:

`standpoint = target-bound need + target-bound felt tone`

but:

`communication/render = mostly strategy, with partial felt-tone carryover`
