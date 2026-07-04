# A Panel of Peers

The interactive shell is not limited to a single reader. It can host a panel of multiple, distinct agents, each taking turns or replacing the user to interact with the primary context. You bring another agent into the panel using the `:spawn` command:

```
:spawn <name> join
:spawn <name> replace [n]
```

When an agent joins the panel, it does not just share the context—it maintains its own entirely isolated phenomenality configuration. Each agent has its own `tuner` (triggers and tuning knobs), its own `probes` (minted concept sensors), and its own `tuner_bindings` (dynamic relationships). 

This means you can configure one agent to be exquisitely sensitive to `ambiguity` and naturally steered towards `honesty`, while a second agent acts with a completely different cognitive profile. Because they are separate state objects, their phenomenological machinery runs in parallel without colliding. 

When you want to address a specific agent in the panel or command it to run a macro, you route the command using its name:

```
@<name> :run <macro>
```

## Warming Up Probes

When a new probe is minted (or adopted) by any agent, it starts with an empty rolling history. Because a probe's signal is calculated as its raw cosine similarity against the mean of its recent history, an empty history defaults its signal to exactly `0.0`. It only begins producing meaningful relative readings on the *following* turn, after that first generation populates the history.

To bypass this warm-up period, you can backfill a probe's history from the archive of past replies:

```
:probe backfill all
:probe backfill choose
:probe backfill <name>
```

This passes the new probe over the agent's archived replies in a single forward pass, populating the rolling history with real past values. By the time the agent takes its next live turn, the probe already has a mature, data-backed baseline to compare against.
