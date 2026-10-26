# 7. Combining correction directions as sine phases

Questions this document answers: What does the sinusoidal option combine? How is it different from addition? Can more aligned corrections produce a smaller result?

## The inputs are eligible corrections

Lookahead first determines which matured corrections apply in the current environment and which dimensions have directional support. Composition then combines those correction vectors. It does not determine whether an observation counts as data: hits and misses retain their full evidence weight under either composition mode.

Let `d_i` be an eligible correction vector. Linear composition is:

```text
combined = sum(d_i)
```

The alternative implemented by `combine_corrections` converts each coordinate of each correction to a phase, adds the phases, and maps back to a displacement:

```text
A = max(residual_RMS, 0.000001) * lookahead_phase_scale
phase_i = asin(clamp(d_i / A, -1, 1))
combined = A * sin(sum(phase_i))
```

The amplitude formula uses the actual arrival state's root-mean-square magnitude, with the small floor shown above. The default phase scale is 0.25. An amplitude of zero is handled separately and produces zero correction. The trigonometric operations are coordinate-wise.

## What the formula implies

For a single correction with every coordinate between `-A` and `A`, conversion to phase and back returns that correction. Equal opposite contributions cancel. Each combined coordinate is bounded between `-A` and `A`, and application also passes through the residual steering cap.

For corrections small relative to `A`, the result approximates ordinary addition. As phases accumulate, the result becomes nonlinear and is not monotonic in the number or strength of agreeing contributions. Consider just one coordinate:

| Contributions | Linear sum | Sinusoidal result |
| --- | --- | --- |
| `0.5A` | `0.5A` | `0.5A` |
| `0.5A`, `-0.5A` | `0` | `0` |
| `0.5A`, `0.5A` | `A` | about `0.866A` |
| `A`, `A` | `2A` | approximately `0` |
| `A`, `A`, `A` | `3A` | `-A` |

Thus reinforcement can become cancellation or reversal. This follows directly from phase addition; it is not a guarantee that the reversal is useful. Clamping also loses differences between individual coordinates whose magnitudes already exceed `A`.

## Scope of this alternative

This option changes the combination rule for lookahead correction vectors. It does not apply a sine transform to your whole hidden state, introduce physical oscillators, replace the transformer's representations, or demonstrate that distributed feature superposition has been eliminated. It is an experimental alternative to linear correction addition.

The shell exposes `:lookahead combine linear` and `:lookahead combine sinusoidal`. Reading their names does not execute them. To compare outcomes, hold the task, model, starting evidence, and other controls fixed and measure both quality and correction behavior. A bounded waveform alone does not establish better answers.

Sources: [composition and application](../../invariants/lookahead_predictor.py) (`combine_corrections`, `step_arrival`, `make_steer_handles`); [operational reference](../../docs/LOOKAHEAD.md).
