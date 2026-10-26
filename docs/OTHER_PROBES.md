# Other-aware probes

`I/me/my` refer to the owning agent, `you/your` to the addressed interlocutor,
and standalone `@` to the selected Other. `@Alex` fixes a named referent. Email
addresses are not markers. No identity association is inferred from a concept
definition alone.

```
:other user
:probe social_butterfly I enjoy conversing with @. || Conversing with @ is monotonous.
:probe social_obligation I am obligated to communicate with @. || I can choose whether to communicate with @.
:probe social_efficacy @ and I partake in productive conversation. || I am not communicating well with @.
:probe compose social +social_butterfly -social_obligation
:probe expose social
```

`@` framings classify a probe as relational and observational; compositions
inherit this classification. These probes remain measurable and exposable but
are excluded from automatic steering, gravity and routing. Their exposure
descriptions distinguish relational measurements from direct self-identity.

`:other` reports the current referent. `:other clear` leaves it unassociated.
`:other self` intentionally places the owning agent in `@`. With an exposed
relational probe, the model can request this through `<<PROBE: other self>>`.

Changing the Other resolves the original templates and mints or restores a
referent-specific direction. Dependent compositions are rebuilt. Direction
changes reset active centering history and tuner evidence. Original templates
remain intact; exposure diction and naming hypotheses use separate binding
namespaces. Selection is owner-local and persists across restart.

Each reply's existing probe hidden-state pass is also projected onto cached
directions for self, user, joined agents, the active speaker, and the selected
Other. These measurements are stored under `other_probe_readings` in turn
telemetry. Each participant has a separate rolling baseline. Directions are
minted on first use and cached; extra participants do not require separate
reply-scoring passes. Initial minting still costs additional model forwards.

Readings describe how the reply aligns with a relational concept; they do not
establish another participant's private mental state. Live model quality and
performance effects remain unmeasured.
