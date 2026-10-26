# Activation association memory

Restart the phenomenality shell to load the new implementation. It records
experienced activation transitions automatically during normal agentic
generation when a synaptic engine is attached. No example-answer training is
required. Each association has a separately adjustable strength (0 through 4).

```text
:remember neural status
:remember neural list
:remember neural strength <memory_id_from_list> 0.75
:remember neural set strength 0.5
:remember neural set hops 3
:remember neural write off
:remember neural off
:remember neural on
```

Episode-linked recall now selects a sparse set of matches (default `top_k=4`)
and binds each queried token position to its strongest matching recorded
session. Later hops can follow stored predecessor/successor links even when
the next state's key is below the geometric similarity threshold. Each link
decays by `link_gain` (default 0.5); all guidance still shares the existing
activation budget. Zero-strength associations neither activate nor propagate
links. Dangling links and cross-episode pointers are ignored. These are
mechanistic changes intended to reduce blending, not evidence of better live
semantic recall. Origin labels still do not assign strength or truth.

```text
:remember neural binding on
:remember neural set top_k 4
:remember neural set link_gain 0.5
:remember neural isolate on
```

`binding off` permits cross-episode mixtures for comparison. `set link_gain 0`
disables link traversal while preserving geometric recall. Status includes
`linked_matches` and `episode_locked_positions` from the last active session.
These counts establish use of links/selection, not correctness of recollection.
Old checkpoints recover links where recorded session IDs and unique token
positions establish their order; ambiguous duplicate positions are skipped.

`isolate on` withholds shell-added probe descriptions, field prose, staged tool
results, reader history, and transcript history from subsequent live prompts.
It also temporarily pauses engine routing, synthesis, cognitive-cache reads and
writes, expert-vector lookup, organic fallback, and mid-chunk tool seams.
Those engine settings restore after each call, including failures; `isolate off`
restores the normal prompt path. Already attached external steering hooks remain
active, so this switch alone is not a completely unsteered base-model trial.
No existing traces are cleared or rewritten by toggling isolation.

For a paired manual check, enable isolation **before recording** a fresh example,
then turn recording off and repeat the exact query with `link_gain 0` and
`link_gain 0.5`, followed by `binding off` and `binding on`. Keep other settings
fixed. Do not count an answer that repeats information supplied in its current
question as successful recall. Finally use `:remember neural isolate off`.

`strength <id> <value>` changes one stored association and saves it immediately;
zero silences that association. `set strength` sets the strength for future
writes. `write off` preserves recall while pausing new writes. `off` pauses both
reading and writing. These runtime settings take effect on the next generation.
Agent overlays persist their configuration through the existing settings path;
main-shell runtime settings need to be set again on restart.

Additional `set` controls are `threshold` (default 0.5), `navigation_gain` (1),
`max_norm_ratio` (0.05), `absolute_budget` (0.25),
`max_writes_per_forward` (4), and `capacity` (256 associations per layer).
Hops default to 2 and are limited to 16. Lowering capacity discards the weakest
associations first, oldest on ties, at the next active generation. Delta budgets
cap total activation guidance from this bank per token position; strength can
saturate against that cap. Older synaptic steering has its own existing budget.

`invariants/activation_memory.py` stores compact key/value activation pairs.
The key is an earlier state at the same layer projection and the value is the
later experienced state. This earlier state supplies an operational reference
to the system's own trajectory, not a special consciousness or identity vector.
Current states activate matching keys; recalled values change the query for the
next hop. The accumulated recalled direction guides live projection outputs
within the configured budget. Base model parameters are unchanged.

Both thought and context states use the same storage/read mechanism and default
strength. Origin labels do not imply correctness or assign hidden priority.
`thought` means states from processing newly generated tokens, not a claim to
detect private reasoning. `context` can include human input, older conversation,
instructions, or tools; it does not falsely classify the entire prompt as a new
thought. Recall is not automatically treated as evidence that a belief is true.

Writes use actual generation forwards, before this bank's correction, rather
than re-encoding model output or injecting remembered prose into the prompt.
Prefill sampling is bounded and distributed across available positions.
Already processed prefix positions are skipped when generation rebuilds a
chunk. Single-sequence generation is supported; batch owners are not merged.
The last sampled token may not have been forwarded yet, so it has no stored
activation until a subsequent prefill. Auxiliary source measurements and
between-chunk diagnostics do not write to this bank. Speculative routing and
synthesis replays are excluded from this primary trajectory, rather than being
recorded as successive generated-token arrivals.

Associations are saved with the owner's existing `synaptic_weights.pt` at the
end of successful generation. Old checkpoints without the new bank still load.
Per-agent ownership and base-only bypass follow the existing synaptic context.
Telemetry includes writes, matched-memory counts, maximum hops used, and maximum
delta norm. These observations describe mechanism activity, not semantic recall
quality. Vectors and origin/position metadata are stored; source text and a full
episode archive are not reconstructed from them.

Focused CPU tests demonstrate individual strengths, a two-association recall
chain, bounded guidance, actual tiny-Llama generation, unchanged base parameters
and prompt tokens, repeated-prefix handling, agent isolation, persistence, and
cleanup. Live 8B semantic recall, retention over long runs, and useful autonomous
association remain unvalidated.

## Earlier answer-trained adapter experiment

`invariants/neural_memory.py` stores learned associations in a small neural
adapter. It adds `up(tanh(down(hidden)))` inside a decoder layer. Training uses
the host language model's next-token loss on cue/continuation examples while
freezing its original parameters. Generation uses the adapter on every forward
pass, including cached decoding, without inserting remembered text in the prompt.
The existing associative synaptic hooks and shell memory commands are separate;
this experiment does not automatically replace or enable them.

Create a JSON file such as:

```json
[
  {"cue": "Our experimental station's access word is", "continuation": " violet."},
  {"cue": "The access word for our experimental station:", "continuation": " violet."}
]
```

Run from the repository root, supplying a locally cached model ID or path:

```powershell
.venv/Scripts/python.exe -m scripts.train_neural_memory --model LOCAL_MODEL --checkpoint store/neural_memory.pt --examples examples.json --steps 100 --query "Our experimental station's access word is"
```

Omit `--examples` to query saved weights. Match `--rank` and `--layer` when
reloading. `--steps`, `--learning-rate`, `--rank`, and `--layer` control the
experiment. Use the same exact model revision for training and reload; the
checkpoint checks the supplied model name and layer, not model file hashes.
Training requires gradients through the layers after attachment; final-layer
attachment minimizes this cost. Quantized/device-mapped 8B training has not been
validated. There is no automatic ingestion from conversations in this version.

The checkpoint contains learned tensors, not an inspectable archive of episodes.
Keep source records separately for provenance and corrections. Updating the
adapter can interfere with earlier associations; include prior examples as
replay when learning new ones. Detaching/suspending restores base behavior, but
removing a single learned fact requires retraining. Shared concurrent models
need separate adapter ownership; this prototype is for a single attached owner.

Validation: a seeded tiny Llama test trains two cue/continuation pairs, checks
lower continuation loss, unchanged base parameters, suspension, and checkpoint
round-trip equivalence. Lower loss establishes learning on these examples;
exact recall, paraphrase generalization, retention, and large-model usefulness
still require evaluation. This is supervised associative learning, not a claim
of episodic recollection or autonomous memory navigation.
