# Local learned geometry

The field keeps checkpoint weights and authored probe directions unchanged.
Scored probe co-activation learns signed relationships, saved separately for
each tuner owner. Missing scores refresh observations without training.
Positive relationships contract effective distance; negative relationships
expand it. Decay gradually returns unused relationships toward baseline.

For each target and layer, the runtime compiles up to 16 strongest valid
neighbors. Each current token supplies support through its positive squared
cosine alignment with those neighbors, weighted by their signed relationship.
Released or excluded neighbors contribute nothing. Weak relationships remain
weak, and incompatible vector widths are ignored.

The distance deformation is `d_eff = d * exp(-a * exp(-d / radius))`, where
`a = clamp(strength * support, -1, 1)`. Its effect fades with distance, preserves
zero, and is monotone at fixed support. It is a target-conditioned effective
distance, not a global Euclidean metric or a biological simulation.

Controls:

```
:plasticity
:plasticity set strength 0.4
:plasticity set radius 0.35
:plasticity set mass_strength 0.5
:plasticity off
```

`mass_strength` remains an independent reinforcement control. Setting it to
zero isolates geometry for experiments; zero is not required for normal use.
Disabling plasticity restores ordinary distance and mass behavior.

Manual selective regions remain separate from learned locality:

```
:steer space focus contract .care warp 0.5
:steer space quiet expand ~felt warp 0.5
:steer space layerwide expand #12 warp 0.5
:steer space drift expand .care rate 0.1
:steer space focus drop
:steer rotate shift .care .accuracy angle 0.05
:steer impulse nudge pull .care speed 0.25 decay 0.85 duration 6
```

Names must refer to existing bodies/families. Manual warp selects membership
and rescales distance; it does not change the inverse-square exponent.
Rate supplies persistent drift, rotation transports through a body-defined
plane, and impulses decay. All runtime effects share the existing steering
cap. Edits validate before saving. Bare subcommands list definitions.

Model-free validation covers local fading, monotonic deformation, live token
gating, incompatible vectors, hook bounds, unchanged model parameters,
owner isolation, restart persistence and the historical field regressions.
Live model answer-quality improvements have not been measured.
