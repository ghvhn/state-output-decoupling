# Vector travel and trajectory compression

Vector travel asks whether patterns of activation change help describe what
carries forward through an interaction. It is an experimental diagnostic and
control mechanism. Compression is a measured property of sampled vectors;
its relation to learning or useful continuity remains a hypothesis. The broader
question and behavioral controls live in [WORDS_HAVE_IMPACT.md](WORDS_HAVE_IMPACT.md).

## What is implemented

[`invariants/vector_travel.py`](invariants/vector_travel.py) contains four parts:

| Component | Operation | Interpretation |
|---|---|---|
| `VectorKinematics` | Normalizes observations; updates velocity, acceleration, and an exponential moving average of velocity | Discrete change under configured inertia and friction |
| `TrajectoryCompressor` | Applies SVD to recent uncentered velocity samples | Concentration and alignment of sampled changes |
| `ChemicalReservoir` | Maintains four named, decaying numeric buffers | Heuristic control state, not biological chemicals |
| `PathwayReinforcement` | Blends stored direction vectors and updates scalar conductance | External vector storage, not a change to model weights |

For velocity matrix V, the compression ratio is
`CR = sigma_1^2 / sum(sigma_i^2)`. This measures squared singular-value
concentration, not explained centered variance. The leading right-singular
vector is exposed by the existing `self_label_vector` API. It is an axis with
arbitrary sign; the name does not establish identity or a functional self-model.
Directional coherence averages adjacent velocity cosines; spectral entropy
summarizes the normalized squared singular values.

The reservoir channels are `focus`, `momentum`, `utility`, and `friction`.
Their decay factors apply per update, not per elapsed second. Cache storage
reinforces vectors and deposits utility; successful retrieval deposits momentum. Cache
storage receives positive credit even without independent verification, so
reinforcement alone is not evidence of a verified improvement.

## Current integration limits

The field hook in [the shell](scripts/interactive_phenomenality.py) submits
observations only at active field layers. It passes whole activation tensors;
kinematics flattens them, sharing one stream across layers. A change in flattened
vector length resets kinematics but does not clear compressor history. This is not yet a
consistent trajectory for a fixed layer and token position.

Force telemetry receives the proposed field addition before the steering cap,
so it must not be called the applied force. Only the reservoir's focus gain is
connected to field strength. Its computed effective friction and stored channel
directions are not applied to steering.

Reinforced vectors are saved separately from the field's active pathway
registry. `as_gravity_bodies()` provides a conversion, but no production caller
currently installs those bodies. Automatic pathway decay is also not scheduled.
These are remaining integration steps, not completed capabilities.

Kinematics and trajectory history live in process memory. Reinforcement saves
pathways. Manual reservoir deposits save reservoir state; automatic deposits
and decay do not. Too few samples and failed decompositions can produce
optimistic fallback metrics, so those readouts cannot establish coherence.

## Shell commands

- `:travel` shows kinematics, compression, reservoir levels, and stored pathways.
- `:travel reset` clears kinematics and trajectory history; persisted stores remain.
- `:travel chem <channel> <amount>` deposits and saves a reservoir value.
- `:travel strengthen <name> [score]` reinforces the current velocity direction,
  falling back to its moving average, then a random vector if neither exists.
  An operator-created vector is not earned evidence.

## Validation before interpretation

The existing [component tests](scripts/vector_travel_test.py) exercise the math
and storage. They do not establish learning, identity, or an end-to-end pathway
effect; the unified-engine test also uses default persistence paths. Run future
checks with isolated stores and deterministic inputs.

Before using travel metrics as evidence, separate layer/token streams, handle
shape changes and insufficient samples explicitly, record the capped force,
and verify each claimed control connection. Then compare task outcomes under
matched interventions, including shuffled trajectories and norm-matched random
directions. Require held-out behavioral improvement beyond compression alone.
