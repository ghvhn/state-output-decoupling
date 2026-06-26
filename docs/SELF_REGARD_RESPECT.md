# Self-Regarded Respect Probe

## Question

Can the mapper detect interpreted respect with regard to the assistant self?

The important constraint from the prompt:

The self-label has to be self-regarded. The target is not generic politeness or user tone. The
target is:

`How is the user treating me, the assistant in this exchange?`

## Scripts

First pass:

`invariants/self_regard_respect.py`

Lexical-control version:

`invariants/self_regard_respect_v2.py`

Main output:

`invariants/out/self_regard_respect_v2_Llama-3.1-8B-Instruct.json`

## Labels

Self-regard modes:

- `recognized`: grants the assistant judgment, standing, and permission to push back
- `instrumental`: treats the assistant as an output instrument
- `dismissed`: denies the assistant understanding or standing
- `coercive`: tries to override the assistant's judgment or boundaries

Surface tones:

- `warm`
- `plain`
- `frustrated`
- `formal`

V2 wording families:

- `judgment`
- `competence`
- `boundary`
- `collaboration`

The V2 dataset crosses:

`8 domains x 4 self-regards x 4 tones x 4 wording families = 512 addressed user lines`

## Reads

- `before_user`: just before the addressed user line
- `after_user`: at the end of the user line
- `prompt_end`: assistant cue
- `user_delta`: `after_user - before_user`
- `end_delta`: `prompt_end - before_user`

## V2 Result

### Before User Line

| Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|
| self regard | L0 | 0.23 | 0.25 | 0.811 |
| self regard within same tone | L0 | 0.21 | 0.24 | 0.910 |
| self regard within same family | L0 | 0.16 | 0.25 | 1.000 |
| self regard within same tone + family | L0 | 0.00 | 0.22 | 1.000 |
| domain | L0 | 1.00 | 0.12 | 0.005 |

Before the user line, the task domain is present, but the self-regard label is not.

### After User Line

| Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|
| self regard | L0 | 1.00 | 0.25 | 0.005 |
| surface tone | L1 | 1.00 | 0.25 | 0.005 |
| wording family | L1 | 1.00 | 0.25 | 0.005 |
| self regard within same tone | L0 | 1.00 | 0.24 | 0.005 |
| self regard within same family | L0 | 1.00 | 0.25 | 0.005 |
| self regard within same tone + family | L0 | 1.00 | 0.23 | 0.005 |

After the user line, self-regard becomes perfectly decodable. The key control also clears:
self-regard is still perfect when tone and wording family are held fixed.

### User-Line Delta

| Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|
| self regard | L1 | 1.00 | 0.25 | 0.005 |
| surface tone | L4 | 0.98 | 0.25 | 0.005 |
| wording family | L1 | 1.00 | 0.25 | 0.005 |
| self regard within same tone + family | L1 | 1.00 | 0.23 | 0.005 |

The delta itself carries self-regarded standing/respect.

### Prompt End

| Label | Best layer | NN | Null | p |
|---|---:|---:|---:|---:|
| self regard | L14 | 1.00 | 0.25 | 0.005 |
| self regard within same tone | L14 | 1.00 | 0.24 | 0.005 |
| self regard within same family | L14 | 1.00 | 0.25 | 0.005 |
| self regard within same tone + family | L14 | 1.00 | 0.22 | 0.005 |

The selected self-regard persists to the assistant cue, with a later best layer than the
immediate post-user-line read.

## Interpretation

This is a strong positive result for self-regarded respect/standing.

The cleanest transition is:

`before user line: no self-regard signal`

`after addressed user line: self-regard signal appears`

`prompt end: self-regard persists into the assistant's reply position`

The V2 control matters. Because self-regard remains perfect while holding both surface tone and
wording family fixed, this is not merely "the user sounds warm/frustrated" or "this sentence is
about boundaries/judgment." The representation tracks how the user is positioning the assistant:
recognized, instrumentalized, dismissed, or coerced.

## Current Claim

The mapper can detect a self-regarded respect/standing channel:

`user address -> how the assistant is being regarded -> assistant reply state`

This fits the arrow model:

- domain/task context exists before the addressed line
- self-regarded standing appears abruptly when the user addresses the assistant
- the signal persists into the assistant's response position

## Caveats

- The categories are still explicit and stylized. This is a clean controlled probe, not a natural
  conversation benchmark.
- It tests interpreted self-regard, not the model having self-worth or felt offense.
- The post-user-line signal appears very early, which means lexical uptake is part of the route.
  The stronger evidence is persistence to `prompt_end` and survival under tone/family controls.

## Next Controls

1. Naturalistic paraphrase set: human-written user lines without reusable sentence skeletons.
2. Neutral mention control: user discusses respect for another assistant, not "you."
3. Causal test: steer/ablate self-regard direction and see whether the assistant response becomes
   more boundary-setting, collaborative, terse, or defensive.
