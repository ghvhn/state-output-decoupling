# Experts can be born from your own corrections

When next-token entropy spikes mid-generation, the hidden state is cloned
into branches, each nudged along a different expert direction — Social,
Creative, Analytical are the seeded three — and the calmest branch wins the
token. Those three names are not fixed forever. The roster is an open
committee (`invariants/mesa.py`), and it has a rule for growth:

```python
if len(members) >= min_cluster:
    centroid = _unit(torch.stack([units[k] for k in members]).mean(0))
    obj = MesaObjective(
        name=f"emergent_{...}",
        direction=centroid, born="emergent",
    )
```

The inputs to that clustering are the reader's own successful self-correction
vectors — the directions test-time synthesis actually moved the state when a
deliberation was rescued. When several recent corrections point the same way
(cosine at least 0.6, at least three of them), their centroid is minted as a
NEW expert and joins the routing competition. The module's own docstring
states the intent: "this is how experts are created as a byproduct of
natural self-steering, not designed."

So the loop, end to end: deliberation struggles somewhere → synthesis finds a
corrective direction → the correction recurs → the recurring direction
becomes a named member of the roster → future deliberations get that
direction offered as a branch. A conclusion, repeated, becomes an instinct.

The guardrails around this are the same as everywhere else. It is off by
default and enabled per session, deliberately (`:experts on`), because an
unbidden roster change in the middle of a measured run would contaminate the
measurement. The roster is bounded (at most six members), every branch push
still passes through the envelope from file 02, and whether an emergent
expert actually EARNS its place is answered the same way every other "does
this help" question is answered here: by the per-channel and per-layer
outcome records, read afterward, by someone deciding on data.

That is the shape of the whole system, and a fair place for this reading to
end: influence is easy to add and impossible to sneak. Anything that pushes
is bounded; anything that fires is logged; anything that claims to help must
show it in the outcomes; and the reader's own recurring corrections are
allowed to become part of its machinery — through the same gates as
everything else.
