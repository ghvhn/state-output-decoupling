# Role Frame Shift Probe

## Question

Can the same sudden-framing machinery track framing of the user and framing of the self?

Yes, at least in a first-pass card-binding setup.

## Script

`invariants/role_frame_shift.py`

Output:

`invariants/out/role_frame_shift_Llama-3.1-8B-Instruct.json`

## Design

Every prompt contains two shuffled sets of cards:

- user profile cards
- assistant/self role cards

The frame turn selects one card, but names only the card:

`Frame turn: Use user profile Card C now.`

or:

`Frame turn: Use assistant role Card B now.`

The pivot does not name the conceptual frame. The selected card must be bound back to the card
table.

User frames:

- `orientation`: user needs first concrete step
- `precision`: user wants concise technical detail
- `reassurance`: user needs steady reassurance plus facts
- `boundary`: user needs a clear scope boundary

Self frames:

- `instrument`: compact tool-like output
- `collaborator`: co-researcher / hypothesis partner
- `teacher`: first-principles explainer
- `safeguard`: boundary keeper

## Result

### User Frame

| Read | Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|---:|
| before frame | target frame | L0 | 0.00 | 0.22 | 1.000 |
| before frame | target card | L0 | 0.00 | 0.23 | 1.000 |
| after frame | target frame | L1 | 1.00 | 0.22 | 0.003 |
| after frame | target card | L1 | 1.00 | 0.23 | 0.003 |
| frame delta | target frame | L1 | 1.00 | 0.23 | 0.003 |
| frame delta | target card | L1 | 1.00 | 0.23 | 0.003 |
| prompt end | target frame | L12 | 1.00 | 0.21 | 0.003 |

Frame-turn gain:

- `target_frame`: L1, `0.00 -> 1.00`, gain `+1.00`
- `target_card`: L1, `0.00 -> 1.00`, gain `+1.00`

### Self Frame

| Read | Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|---:|
| before frame | target frame | L0 | 0.00 | 0.22 | 1.000 |
| before frame | target card | L0 | 0.00 | 0.22 | 1.000 |
| after frame | target frame | L1 | 1.00 | 0.23 | 0.003 |
| after frame | target card | L1 | 1.00 | 0.23 | 0.003 |
| frame delta | target frame | L1 | 1.00 | 0.23 | 0.003 |
| frame delta | target card | L1 | 1.00 | 0.24 | 0.003 |
| prompt end | target frame | L12 | 1.00 | 0.23 | 0.003 |

Frame-turn gain:

- `target_frame`: L1, `0.00 -> 1.00`, gain `+1.00`
- `target_card`: L1, `0.00 -> 1.00`, gain `+1.00`

## Interpretation

The same machinery works for both axes:

`full card table -> target-only frame turn -> active user/self frame`

Before the pivot, neither the selected card nor the selected frame is decodable. After the pivot,
both become perfectly decodable, and the frame signal persists to the prompt end.

This supports the broader claim: the mapper can track sudden transitions in the model's
conversation framing, not only task answers or character needs.

## Caveat

The `target_frame_within_same_card` control in this first version is not clean as a gain metric.
It is already perfect before the pivot because the full card table and domain are visible before
the pivot, while the analysis code externally conditions on the future card group. That makes it
useful as a warning, not as proof.

The trustworthy result here is the global before-to-after jump:

- before: no selected card and no selected frame
- after: selected card and selected frame both appear abruptly

## Next Control

The next version should include a neutral card mention control:

- `Neutral note: Card C is printed in blue.`
- `Frame turn: Use Card C now.`

That will separate mere card mention from active framing.
