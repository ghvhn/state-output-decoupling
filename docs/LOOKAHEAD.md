# Context-local future prediction

The apparatus forecasts token IDs and internal-state waypoints at arbitrary
positive horizons, then evaluates them only when those future tokens arrive.
It observes external input and self-generated output on separate timelines.
It never changes the current token logits to make a forecast come true.

## Evidence and locality

Every matured prediction contributes weight **1**, whether its token forecast
matched or missed. Correctness is telemetry, not a validity filter. A missed
forecast teaches the actual transition and token just as fully as a hit.

Learning uses online conditional transition means, per-dimension variance, and
token frequencies. It does not globally fine-tune shared neural weights. The
key is **stream + semantic neighborhood + source token + horizon**. Neighborhoods
use cosine similarity of internal states with fixed centers (default cutoff
0.90, adjustable with `:tune lookahead_environment_similarity <value>`).
Fixed centers avoid gradually chaining unrelated contexts together. A source
token can therefore have different expected future directions in different
contexts. Neighborhood similarity is an operational proxy for semantics.

Each ticket retains its originating environment. Even if the arriving state has
moved elsewhere, its outcome updates that original environment with full weight.
Statistics are not transferred to the new environment. Pending forecasts and
display history are bounded; accumulated evidence counts and moments persist.
When pending capacity is full, no additional forecast is issued. Existing
forecasts are not silently evicted. Tickets beyond a completed sequence are
unobserved, not classified as false.

External text uses frozen token embeddings and a causal context average. Self
output uses one consistent intermediate-layer sensor. Their state estimates and
corrections remain separate because these are different coordinate systems.
Fresh prompts and document/tool payloads are ingested once, without replaying
historical prompt material as new evidence. An emitted final token is scored and
counted even when no forward pass exists for its residual; no latent target is
invented in that case.

## Directional adjustment

Prediction error and ClaimMap-axis drift are available after maturity. A proposed
correction is eligible only in the corresponding neighborhood and named context.
Its coordinates are further gated by the positive alignment between the expected
trajectory and the trajectory actually taken. Unsupported or opposite directions
remain inactive. Eligible adjustments may span many residual dimensions and the
selected layer band; applicability is local even when the vector has broad support.

Dispersion produces a separate certainty score per dimension:
`certainty = 1 / (1 + sample_variance)`. This is a dispersion score, not a calibrated
probability. At least two latent observations are needed to estimate it. It scales
the intervention, never the data weight. Cold neighborhoods observe and learn
without injecting an invented direction. Historical prefill tokens are unchanged.
Only the current token receives an adjustment, within the existing steering cap.

## Linear and sinusoidal composition

```
:lookahead combine linear
:lookahead combine sinusoidal
```

For eligible correction vectors `d_i`, linear composition is `sum(d_i)`.
Sinusoidal composition maps each coordinate to a phase:

```
A = residual_RMS * lookahead_phase_scale
phi_i = asin(clamp(d_i / A, -1, 1))
combined = A * sin(sum(phi_i))
```

The default amplitude coefficient is 0.25. Zero amplitude gives no adjustment.
One in-range contribution is unchanged; opposing phases cancel; multiple phases
can reinforce or interfere. Each combined coordinate is bounded by `A`, followed
by the usual steering cap. The internal state itself is not passed through sine.
Composition is selectable and saved; its effect on answer quality is not established
by the implementation tests.

## Controls

| Command or degree | Effect |
| --- | --- |
| `:lookahead on [alpha]` | Enable supported post-maturity steering |
| `:lookahead off` | Stop steering while continuing observation and learning |
| `:lookahead status` | Evidence count, current mode, checkpoint, composition, latest outcomes |
| `:lookahead k N` | Base positive forecast horizon; generated schedules are nonsequential |
| `degree diff` | Scale the learned expected displacement |
| `degree horizon` | Scale automatic horizon scheduling |
| `degree direction` | Change per-dimension certainty shaping for interventions |
| `degree distance` | Scale correction magnitude |
| `degree logits` | Change forecast probability sharpness, leaving generation logits untouched |
| `degree factor` | Scale overall correction magnitude |

Use `:lookahead degree <name> <value>` for the degree rows. Settings never change
the unit weight of a learning observation. ClaimMap attachment is explicit through
normal ClaimMap commands; `:lookahead claimmap detach` removes it.

`init_lookahead.txt` is an explicit macro which resets travel chemistry and enables
lookahead settings. CEN remains off by default. Learned summaries are saved after
generation to `invariants/out/lookahead_<model-key>.pt`; model identity and width
separate checkpoints. Named boundaries are also available through the Python
`set_environment(name)` API. Different sequences share only matching learned
summaries, never pending self-output tickets.

## Validation

The offline tests cover causality, equal treatment of hits/misses, locality across
states/tokens/horizons/streams, directional gating, checkpoint restoration,
sinusoidal identities, input validation, and real Hugging Face generation with a
small locally initialized Llama model on CPU and CUDA in half precision. They verify exact generated-token counting
across 24-token chunks, tool input, final tokens, and cleanup on errors.

These checks do not measure quality or throughput on the full 8B model. The older
gravity suite already failed before this initiative because `build_space_effects`
and related features are absent; this work does not claim to repair that subsystem.
