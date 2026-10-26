# The State Reference Body

All physical state is decoupled from conversational prose. The shell does not prepend technical diagnostics, mechanics, or probe listings into the operator's chat turn. The model is not fed a linguistic self-description to recite or dissect.

Instead, state exists as a reference body in the gravitational field:

```python
# The reference body @state occupies an exact coordinate in latent space
st_anchor = steer_bodies()["anchors"]["state"]
```

## Shell Language Boundary

The shell language is the public surface for changing controllable state. A
proposed altered configuration must be expressible as a runnable `:` command,
with the command form carrying the effect, arguments, and reason where one is
needed. Ordinary prose can describe a state, but cannot be treated as a state
transition.

This applies to durable operational configuration: probes, tuning, steering,
macros, exposure, and game rules. The self macro records its supported session
configuration as replayable commands. Ephemeral observations such as a live
activation coordinate, a generation-time reading, or a probe score remain
measurements; they can be read through `<<STATE>>` or `<<PROBE: ...>>`, but are
not configuration changes to replay.

Before a command generator proposes a state change, it can ask
`<<HELP: :command args>>`. The returned explanation uses the same parser
contract as the shell, so the command has a plain-language account and a
grammar check before an operator executes it.

ClaimMap follows the same boundary. Its human-readable comparison is an
operator/log observation, never a conversational prompt block. The only
model-side effect is its explicitly enabled, bounded `claimmap_alpha` steering
path.

## Latent Kinematics and Continuous Field Coupling

At the end of each generation, the model's actual activations define the latent coordinates of `state`. The continuous gravity manifold then tracks displacement and motion relative to this reference:

- `pos:@state`: mean cosine similarity between live activations and the recent state
- `dist:@state`: 1 - pos:@state (displacement in representation space)
- `vel:@state`: turn-to-turn rate of drift relative to state
- `accel:@state`: change in drift velocity

Physical forces, spring bands (`:steer band`), and relativistic local clocks (`:steer time`) can attach directly to `state` and external status streams (`status:cpu_pct`, `status:tot_routing_total`).

## On-Demand Reference

When the model intentionally needs to inspect its physical or operational status, it can reach for its reference body via:

```text
<<STATE>>
<<PROBE: state>>
```

This returns a factual reference body readout as a tool consequence, keeping conversational messages clean while maintaining complete empirical grounding.
