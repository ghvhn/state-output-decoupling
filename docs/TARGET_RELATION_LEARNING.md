# Target-directed relationship learning

```
:relate target accuracy maximize curiosity uncertainty
:relate status
:relate off
:relate on
```

Replace these example names with existing probes. The target can also be an
observed tuner stream. `minimize` reverses the objective. Choose two to eight
candidate control probes, keeping the target probe separate. Selecting a new
target resets this controller's evidence; it does not reset general plasticity.

Observation is the default. It adds no field adjustment. Natural changes from
prompts, memory, steering and learning provide the data. The controller compares
changes in joint probe expression (`a*b`) with changes in the chosen metric,
grouped by the preceding controls' coarse signed context. It reports sample
counts, correlation, and whether both quantities varied. Missing target readings
break the comparison chain; unmeasured intervals never receive inferred credit.

Eight comparable observations with nontrivial variation qualify a relationship
for a trial. This is an operational minimum, not proof or statistical certainty.
Correlations can reflect common causes and changes in the task. No automatic
steering is enabled merely by reaching that threshold.

Optional explicit experiments:

```
:relate trial
:relate observe
```

Trials alternate ordinary-field comparison turns with signed relationship
adjustments, limited to 0.15 and the existing plasticity and residual caps.
Only sufficiently observed relationships in the current context are eligible.
Ordinary turns maintain a context-specific target-change baseline; trials
record improvement relative to it. Initial trial signs alternate, and periodic
exploration remains after a preference develops. A relationship needs four
scored trials before the controller favors its successful sign. Disabled
plasticity applies no relationship intervention. General plasticity continues
to learn separately from its existing sense outcome.

Each generation uses one pending action, even when hooks are rebuilt. Outcome
feedback consumes it once. Missing target measurements discard it. Owner-local
`.plasticity.relations.json` files retain configuration and evidence across
restarts, while stale in-flight actions and previous target readings are dropped.

The comparisons are sequential, not controlled matched prompts. They support
adaptive experiments but cannot establish causality on their own. Current tests
use synthetic measurements and small field hooks; real answer-quality gains
remain unmeasured.
