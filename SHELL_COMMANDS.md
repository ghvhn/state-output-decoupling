# Interactive Shell Commands

Reference for `scripts/interactive_phenomenality.py` (launch via
`run_phenomenality_shell.cmd`). House rules that hold everywhere: readouts
are free and honest about insufficient evidence; anything that MOVES the
system is a deliberate command (nothing auto-drifts); tuned values persist
across sessions; anything folded into the model's context is minimal and in
its own first-person voice.

> **Grammar authority:** exact argument order and types live in
> [docs/COMMANDS.md](docs/COMMANDS.md), regenerated at every shell start.
> Its "Declared argument contracts" section is rendered from the same
> `CommandSpec` objects that PARSE those commands, so it cannot drift from
> behavior; this file is the narrative companion (what each family is for
> and the house physics behind it).

## Session & context

- **`exit` / `quit`** — close cleanly (logs `shell_closed`; an interrupted
  session can auto-resume its unanswered message next launch).
- **`:context`** — rolling current-session transcript state.
  **`:context on|off`** include/exclude recent turns from the prompt;
  **`:context clear`** empties it. Capped at 50 turns / 16,000 chars; when
  it overflows the LOWEST-sense pair is evicted first (compaction by earned
  sense, not arrival order), so the conversation stays the one context that
  scales.
- **`:context resume [last|<session_id>]`** — restore a prior shell's
  transcript. `last` = the most recent session with stored turns, closed or
  not (a crashed shell is exactly the one worth resuming). Restored turns
  re-enter with their earned sense scores, so eviction treats them the same
  as live ones.
- **`:history`** — pointer: transcript is `:context`, long-term is `:memory`.
- **`:startup [list]`** — view all declared launch macros in `startup/` and whether startup auto-run is `ENABLED` or `DISABLED`.
- **`:startup on` / `:startup off`** — enable or disable executing `startup/` macros on launch.
- **`:startup add <path>`** — declare a macro to run on launch (copies it into `startup/`).
- **`:startup remove <name>`** — remove a macro from `startup/`.
- **`:startup run`** — immediately execute all declared macros in `startup/`.

## Command syntax & sigils

- **Command Chaining (`;`) & Prompt Indicator (`'; '` / `' ;'`)**:
  - **Chained commands** touch the semicolon on the right (no space after `;`, e.g. `:clock;travel`, `:clock ;travel`, `:timestamps on;:clock;travel`), or explicitly begin with a colon (e.g. `:clock; :travel`, `:clock ; :travel`).
  - **Semicolon followed by a space without a colon (`'; '`)** transitions into an **actual prompt turn** to the LLM (e.g. `:tune !claimmap_alpha 0.05 ; What are your thoughts on this?` or `:clock;travel ; How far did we move?`).
  - A leading `;` on a line (e.g. `; explain your steps` or `;explain`) treats the line directly as a prompt turn.
  - Semicolons inside arguments can be escaped with `\;`.
  - **Space-colon (` :`) does NOT split commands** — colons in arguments (e.g. `:suggest :clock`, `status:time`, `probe:care`) are preserved safely without unintended command breaks.
- **Command Provenance & Rationale (`#` / `because`)** — trailing `#` (or ` because `) on any command is peeled off as the explanation/rationale:
  - **Space required BEFORE `#`** (`" #"`), but no space required after `#` (e.g. `:tune !claimmap_alpha 0.05 #stabilizing intent`, `:clock #myreason`, and `:clock # stabilizing` are all cleanly recognized).
  - Without a space before `#` (e.g. `:clock#myreason`), `#` is treated as part of the argument, not provenance.
  - Recorded into cognitive memory event provenance (`command_because`).
  - Informs `:suggest`, `:spawn`, `:fix`, `:solve`, and `:doc` of the operator's motivating rationale.
  - Standalone lines starting with `#` are treated as comments. Literal hashes inside arguments can be escaped with `\#`.
- **Identifier Sigils** — concise single-character punctuation marks define namespaces:
  - **`$` $\to$ Anchor (Absolute Reference / Landmark)** (e.g. `$origin`, `$truth`; `@` supported as alias).
  - **`@` $\to$ Subagent Addressing** (e.g. `@agent :cmd`).
  - **`!` $\to$ Knob / Tuner Trigger** (e.g. `!claimmap_alpha`, `!lookahead_alpha`, `!prioritize_alpha`).
  - **`%` $\to$ Process Status / Metric** (e.g. `%cpu_pct`, `%ram_pct`, `%time`, `%gpu`, `%vram`).
  - **`?` $\to$ Probe** (e.g. `?care`, `?epistemic_openness`, `?accuracy`).
  - **`L<n>` $\to$ Model Layer** (e.g. `L16`, `L24`; `#16` supported as alias).
  - **`~` $\to$ G Family Container** (e.g. `~affect`).
  - **`^` $\to$ Trigger Lift** (e.g. `^mytrigger`).
  - **`*` $\to$ Trigger Outcome** (e.g. `*mytrigger`).
- **Polarity & Correlation (`+` and `-`)** — leading `+` and `-` represent positive or negative correlation/direction (applying a $+1.0$ or $-1.0$ scale multiplier):
  - `+?care` — positive attraction to the `care` probe.
  - `-?hallucination` — negative repulsion from the `hallucination` probe.
  - `+!claimmap_alpha` / `-!lookahead_alpha` — positive/negative knob scaling.
  - `-%cpu_pct` — negative status dampening.


## Long-term memory (explicit tool, never hidden prompt context)

- **`:memory`** — status: path, counts, scope.
- **`:memory recent [n]`** — last n records.
- **`:memory search <query>`** — searches everything (turns, events, document
  chunks); shows YOU the hits, touches nothing.
- **`:memory use <query>`** — searches and stages the result for the model's
  next turn, one turn only.
- **`:memory boundary`** — marks a session boundary for "recent" scoping.

## Instruments

- **`:claimmap <text A> || <text B>`** — measures both texts' per-layer
  activation geometry; stages the felt rendering (first person, no numbers)
  for the next turn. Steer delta staged too; pushes nothing until
  `claimmap_alpha` > 0.
- **`:methodmap <query>`** — searches sanitized methodology memories; stages
  matches.
- **`:steermap`** — steer-map store summary: top action/layer/step groups
  with acceptance-aware success rates.
- **`:slice [axes...] [drop|export <path>]`** — slice what the cognitive
  cache carries by any axis it records: `time A..B` (entries are stamped at
  store; older ones get bounds from the order-preserving synthesis-trace
  join), `probe <name>`, `scope <s>`, `reason <r>`, `layers <lo> <hi>`,
  `steps A..B`, `index A..B`. The matching slice always lists first; `drop`
  excises it only after writing a backup (and refuses with no axis — no
  accidental full wipe); `export <path>` saves the slice to a `.pt` without
  touching the cache. Bare `:slice` summarizes. This is the surgical tool
  for cache hygiene — e.g. removing diagnostic-ping noise identified by
  time window: `:slice time 2026-07-09T13:54..2026-07-09T13:56 drop`.

## Steering readout & calibration

- **`:steer`** — envelope dashboard: current cap/band; observed push
  distribution + clip rate; data-implied cap (refuses under 64 observations);
  synthesis-evidence band vs layer-steer band (transfer-free, per-layer
  table); channel-lift table (fired vs unfired outcomes per channel).
- **`:calibrate <name> [pct|intent|<anchor>|<a>+<b>|band args]`** — the
  calibration front door: request any knob's calibration BY NAME; the
  system evaluates the request deterministically and refuses unsafe ones —
  circular strength/budget knobs (they shape the distribution they'd
  calibrate to), binary outcome streams (no percentile exists), vacuous
  caps (p100 never binds), and under-evidenced bars (<10 observed signals,
  or the route's own floor). Safe routes: observed thresholds by
  percentile, paired anchors (one or several joined with `+`),
  `steer_cap_fraction` from push telemetry, `steer_band` from per-layer
  outcomes, `conversation_productive intent` from the intent axis, and
  `<knob> outcome` for budgets. Operator-only by construction: the model's
  words can never loosen its own bounds. See the probe section for paired
  and multi-anchor calibration; **`:suggest`** lists the moves the accrued
  evidence already backs.
- **`:tune`** — every knob with value, fire rate, signal distribution, lift.
- **`:tune <name> <value>`** — set a knob.
- **`:tune <name> auto [percentile]`** — calibrate a threshold to its own
  observed distribution.
- **`:tune steer_cap_fraction auto [pct]`** — cap from observed push ratios
  (default p95).
- **`:tune conversation_productive auto intent`** — set the productive bar
  RELATIVE TO INTENT-SHAPING: the sense cut between turns that settled
  intent (lowered ambiguity+disagreement vs the prior turn) and turns that
  did not. The plain `auto 50` quantile remains available; `intent` anchors
  the bar to the decomposed reward's intent axis instead.
- **`:tune steer_band auto [min_events] [gold|conversation|any] [synthesis|layersteer]`**
  — band from per-layer outcomes, by evidence lane and kind.

Knobs you'll actually touch:

| Knob | Default | Moves |
|---|---|---|
| `claimmap_alpha` | 0.0 (off) | ClaimMap steer strength; start ~0.02, watch |
| `memory_alpha` | 0.0 (off) | memory-retrieval steer strength |
| `claimmap_tension` | ~0.18 | auto-ClaimMap fire threshold |
| `memory_need` | 0.05 | state-triggered memory-gap threshold; the gap (ambiguity+disagreement−flow−warranted) is now sampled per turn, so it calibrates/anchors like any stream |
| `steer_cap_fraction` | 0.5 | envelope: max push / residual norm |
| `steer_band_lo/hi` | 0.40/0.70 | depth window for band steers |
| `steer_fraction` | 0.25 | agentic delta/branch scale |
| `steer_layer_sweep` | 0 (off) | sweep WIDTH: 0 = full band, 1 = one least-tested layer (pure isolation), k = deterministic overlay of the k least-tested (outcomes score the group; sweep_width recorded per event) |
| `conversation_productive` | 0.0 | sense threshold labeling turns productive |
| `response_tokens` | 512 | reply token budget (cut-off suggestions reference it) |
| `reading_settled_streak` | 2 | consecutive productive reading turns that end a 'satisfied' auto-read |
| `routing_events` | 4 | expert competitions allowed per reply (the consultation budget) |
| `calibration_gain` | 0.5 | how far a calibration MOVES the value: the first calibration of a knob adopts the computed target, each later one moves this fraction of the way toward it (EMA smoothing, so one noisy snapshot never overwrites accrued tuning). 1.0 = overwrite. Applies to percentile and paired calibrations; outcome (discrete argmax) and cap/band stay jumps |
| `expert_proof_weight` | 0.0 (off) | how hard ToT routing favors PROVEN experts: branch entropy is discounted by each expert's accrued route success rate (centered, so unproven experts stay neutral). 0 = pure entropy. Earn it: `:calibrate expert_proof_weight outcome` after trying a couple of values. The routing trace logs `entropy_winner` vs the proof-picked winner, so "did proof override entropy, and did it help?" is answerable |
| `routing_loops` | 3 | expert competitions allowed per token |
| `routing_depth` | 1 | maximum sequential branch/select stages within each competition; 0 disables routing; later stages run while selected entropy exceeds `routing_entropy` |
| `routing_layer_span` | 0 | last N layers of the detected plateau-to-boundary region to replay per stage, clamped to that region; 0 replays the whole region |
| `routing_entropy` | 2.0 | next-token entropy that triggers an expert competition |
| `synthesis_events` | 1 | test-time synthesis events allowed per reply |
| `synthesis_steps` | 60 | optimizer steps per synthesis event |
| `plateau_epsilon` | 0.05 | plateau-velocity trigger that starts a synthesis |
| `intent_settling` | 0.0 | observed stream: prev minus current ambiguity+disagreement (>0 = intent shaped) |
| `eot_urgency` | 0.05 | P(end-of-turn) below this at the budget = "cut off mid-thought" |
| `sandbox_success` | 0.5 | observation stream of real execution outcomes |
| `words_had_impact` | 0.5 | observation stream of word-caused turns |
| `max_committee_size` | 6 | max number of emergent experts in the roster |
| `max_rounds` | 5 | max verification rounds the reasoner gets per solve |
| `required_agreement` | 3 | required number of agreeing paths for consensus |
| `max_tool_calls` | 8 | max tool uses allowed per reply |
| `max_new_tokens` | 220 | base generation token budget for the underlying engine |
| `repair_token_multiplier` | 2.0 | token budget multiplier granted to verifier repair attempts |
| `max_elapsed_sec` | -1.0 | hard timeout limit for generation (if > 0) |
| `oracle_max_elapsed_sec` | 60.0 | timeout limit for oracle reasoning synthesis |
| `verifier_time_reserve_sec` | 20.0 | time forcibly reserved for verification before hitting the global timeout |
| `relax_agreement_under_urgency` | 0.0 | boolean flag (>= 1.0 is true) to lower the required agreement when time is short |
| `stop_on_critical_urgency` | 1.0 | boolean flag (>= 1.0 is true) to forcefully halt generation if frantic urgency is reached |

## Command knobs (criteria & metric)

Every exposed tool and every command carries its own knob pair, registered
lazily on first use (typos never mint a knob) and at exposure:

- **`cmd_<word>`** (metric) — observed on every generated reply: signal 1
  when the command ran since the last reply, 0 otherwise; credited with the
  turn's sense. Its fired-vs-unfired lift answers *"are turns that use this
  command more productive?"* and composes with the whole outcome surface:
  `:figure lift:cmd_doc`, field source `lift:cmd_doc`, the `:tune` listing.
- **`cmd_<word>_criteria`** (criteria) — the activation bar. When an
  exposure's `because` clause names a probe, the shell observes that probe's
  live signal into this knob each reply, so the bar calibrates against a
  real distribution (`:calibrate cmd_<word>_criteria <pct>`) rather than
  being asserted; with no named probe it stays a manual dial.

## Gravity field (steering as physics)

Tuning, steering, and calibration share one frame: every probe is a body — a
mass at its direction on the unit sphere. Each token's hidden state is pulled
along the sphere tangent, pull ∝ mass/(d+eps)² with d = 1−cos. Tuning sets the
field's live constants (G, envelope cap, doppler, law step); calibration
places a body's firing horizon from its own observed orbits; every push stays
inside the envelope cap.

- **`:steer field|gravity init|on|off [G] | status`** — the master switch.
  `init` loads/migrates every field surface without inventing laws or
  unfreezing bodies; while on, the field REPLACES pin/mix steering.
- **`:steer mass <probe|all> <source|auto|hold>`** — a body's raw
  gravitational coefficient (negative = repulsor; effective masses normalize
  to |sum|=1 when the field applies). `auto` returns it to live signed evidence
  lift each turn; `hold` snapshots the current raw mass as a constant.
- **`:steer g <node> <source|off> [scale [offset]]`** — personal G: bind one
  node's coupling to a live source instead of global G (`prioritize_alpha`).
  Sources: number, `probe:<name>`, `knob:<name>`, `status:ram|vram`,
  `lift:<trigger>`, `outcome:<trigger>`, `family:<name>`, `global`, or field
  geometry `pos:<body>`, `dist:<body>`, `vel:<body>`, `speed:<body>`,
  `accel:<body>`. Bodies can be probes, `@anchor`, `impulse:<name>`,
  `space:<name>`, or `band:<name>`.
- **`:steer time <node> <source|off> [scale [offset]]`** — local clock: the
  node's gravity evolves on its own timeline (rate clamped 0.05..20).
- **`:steer impulse <name> toward|away <probe|@anchor> [speed <source> [scale [offset]]] [time <source> [scale [offset]]] [decay <r>] [duration <tau>] [mode velocity|distance|force|force_distance]`** — transient tangent velocity: an initial shove that decays while evolving on any live source, not wall-clock time.
- **`:steer freeze <probe|@anchor> [off]`** — inertial coefficient: the body
  keeps PULLING but stops MOVING itself, so it cannot habituate to its own
  gravity. Frozen probes and anchors keep their baselines and latent positions fixed and stop responding to laws; `off` thaws them so they are mobile under laws again.
- **`:steer pole <body> <family[+|-]> [off]`** / **`:steer law <famA> <famB>
  <k|off>`** — selective magnetism: poled bodies interact only where a law
  couples their families (k>0 like repels/unlike attracts; unlisted pairs
  inert; `:tune gravity_law_step` integrates, moving unfrozen anchors and unfrozen typed probes).
- **`:steer bodies`** — the whole physics at a glance: masses, personal Gs,
  qualities, sets, impulses, spaces, rotations, bands, recent positions,
  exclusions, families, clocks, shapes, poles, laws, knobs.

**The outcome channel is mappable — correlatable and steerable.** Every
tracked stream is credited each turn with its own signal and the turn's
outcome, so each credit history carries "did this actually help" per stream:

- *Draw it*: `:figure outcome:<trigger>` (credited outcomes in credit order),
  `:figure lift:<trigger>` (rolling fired-vs-unfired lift against a 0-line),
  `:figure outcomes` (every credited trigger at once).
- *Correlate it*: `:figure patterns` reports `credited_outcome` — the
  signal↔outcome Pearson r per stream — and outcome maps carry `signal_r=`
  next to the binary lift.
- *Steer by it*: `lift:<trigger>` / `outcome:<trigger>` are field sources —
  a body's G (or a clock, or a quality strength) can be driven by proven
  productivity instead of assertion. Outcomes still never move a threshold
  on their own; routing them into force is an explicit command.

## Down-the-line lookahead apparatus & ClaimMap coupling

- **`:lookahead [status]`** — inspect waypoint ledger, dual-stream mode, ClaimMap coupling, and arrival discrepancy telemetry.
- **`:lookahead on [alpha]` / `:lookahead off`** — toggle post-mature layer adjustment on/off.
- **`:lookahead k <val>`** — set default lookahead horizon offset (out-of-order waypoint mapping).
- **`:lookahead claimmap [detach]`** — inspect or detach active ClaimMap tension axis.
- **`:lookahead degree <param> <val>`** — configure `diff`, `horizon`, `direction`, `distance`, `logits`, or `factor`. All values must be finite and nonnegative; these never discount evidence.
- **`:lookahead combine linear|sinusoidal`** — sum applicable correction directions or compose them as sine phases. Linear is the default. `:tune lookahead_phase_scale 0.25` sets sine amplitude as a fraction of residual RMS.
- **`:lookahead init` / `:trajectory` / `:run init_lookahead.txt`** — execute the full lookahead & ClaimMap initialization macro.

Every matured prediction is one full observation, hit or miss. Estimates are
conditioned on the source token, a neighborhood of similar internal states,
stream, and future horizon. Corrections require both a matching environment and
a supported current trajectory. `:lookahead off` leaves observation/learning on.
Learned summaries and composition persist in a model-specific checkpoint.
See [LOOKAHEAD.md](docs/LOOKAHEAD.md) for the exact equations and limitations.

## Central executive network (CEN & DMN)

- **Default state**: **OFF** on boot (eliminates background thread polling & GIL contention).
- **`:cen [status]`** — inspect whether CEN/DMN threads are active or dormant.
- **`:cen on` / `:cen off`** — dynamically enable or disable background executive threads.
- **`:cen episodes`** — view consolidated thought episodes in the DMN repository.
- **`:cen gamma <val>`** — set epistemic friction scaling coefficient.

## Documents & reading

- **`:doc <path> [because <why>]`** — load and automatically read a full pass.
  The source text and provenance enter the conversation without synthesized
  reading instructions or invented questions. Accepts a directory (.md/.txt/.py) or
  glob; `{filename}` and `{last_updated}` in the reason are substituted per file.
- **Reader and talker phases.** Reading generates unspoken continuations,
  recorded as `document_reader_thought` events separately from assistant speech.
  A bounded window of recent source text and these generated thoughts accompanies
  subsequent reading and speaking turns. The shell returns to normal speaking
  when the pass completes, reading is stopped, or an error occurs. This separation
  is runtime routing and record provenance, not a trained latent reader/speaker
  switch; it adds no semantic instruction telling the model how to read.
- **`:doc read [n] [state|reply|order|interleave|updated] [+probe -probe] [satisfied|<stop_probe>]`**
  resumes unfinished coverage. With no count or stop condition it reads every
  remaining passage, including documents longer than 20 chunks. The default
  `state` mode ranks unread passages by cosine similarity between the most
  recent actual generation residual vector and each raw passage's activation
  vector. Cold starts or invalid state fall back to file order. This is a
  relevance proxy, not learned curiosity or a guarantee of answers; it adds
  no reward or curiosity probe. Explicit `reply` uses word overlap; `order`
  uses file order, `interleave` alternates documents, and `updated` selects
  newest files first. An explicit count limits the run to
  1-20 passages. Signed probes rank up to 160 unread candidates using cached,
  read-only projections. A bare probe or `satisfied` deliberately enables an
  early stop; unread passages remain available to resume.
- **`:doc reread [modifiers]`** — start a new pass with the current conversation
  as context, even partway through a previous pass. All `read` modifiers work.
  State mode reranks from the current generation state and clears the previous
  pass's cached candidate activation vectors.
  Reopening a completed document, or using `read` on a fully completed library,
  also starts a fresh pass. Reading history is retained; stored source text is
  deduplicated independently, so rereading does not duplicate document records.
- **Resume survives shell restarts.** Progress is recorded only after a
  nonempty reader continuation for the exact staged passage. A generation error or
  empty continuation pauses reading without crediting unprocessed text. Reloading
  the file restores the latest compatible pass. Changed text or chunk boundaries
  get separate coverage. Old presentation-only records stay in history without
  being mistaken for completed processing.
- **`:doc next`** — read one unread passage in file order, filling any earlier
  gaps; if the current document is complete, begin a fresh pass.
- **`:doc status`** — show each document's unread count for the current pass.
- **`:doc inject`** — stage the entire library for the next conversation turn,
  within the prompt budget. It starts a fresh pass and credits coverage only
  after a successful response. Oversized libraries must be read in passages.
- **`:doc stop`** — pause between passages and return to speaking; `:doc read` resumes. The input
  listener starts during automatic reading so commands can be queued while
  the model reads.

## Sandbox & agency

- **`:sandbox on|off|status`** — default off. When on, a fenced ```python
  block in the model's reply executes for real (isolated interpreter, 10s
  timeout); next turn carries "I ran the code I wrote; exit_code=…" with the
  actual output. Logged with provenance; success observed.
- **`:impact`** — agency ledger: rate of word-caused turns, consequence
  trail, and the lift readout (does experienced impact track better
  deliberation?).
- **`:listen on|off|status`** — speak while it thinks. A reader thread queues
  every line you type so nothing is ever dropped; with listen on, lines typed
  DURING a reply are drained at the next chunk seam (the same seam tools fire
  on) and appended to the model's live stream as a legible
  `[Operator interjects: …]`. The model reads the interjection and continues —
  whether it redirects or folds it in is its own choice; nothing is
  force-stopped. With listen off, mid-reply typing still isn't lost — it lands
  as the next turn's input. Opt-in: until `:listen on`, input is the plain
  unchanged path. (The reader thread, once started, stays for the session.)
- **`:clock`** — generation cost sensor. Every turn prints a `[Clock]` line
  (wall-time for the reply, tok/s, and memory: live allocation / total
  reserved / this turn's peak). Memory is VRAM on a CUDA box and this
  process's resident RAM on a CPU-only box, so the sensor is live either way
  — CPU-only runs are not stuck at zero. `:clock` on its own shows the current
  footprint, the last turn's timing, and the accrued distributions. The
  wall-time and reserved-GB are observed as the streams `generation_seconds`
  and `vram_gb` — so they anchor and calibrate like any sensor
  (`:calibrate conversation_productive generation_seconds` asks whether
  slower turns cohere worse), and their sense-lift is in the readout.

## Named-concept probes

A probe is a per-layer activation direction that scores every reply from
then on (one extra forward per turn), centered against its own rolling
history and paired with sense. Every way of making one is operator-only —
the model authoring the instrument that judges it would be circular. All
persist to `invariants/out/probes/`, reload at startup, backfill, and
calibrate identically. `:probe` lists (marking which are exposed to the
model); `:probe drop <name>` deactivates (its observed stream is kept).

- **`:probe define <name>`** or **`:probe <name> define`** — show the original stored
  definition and contrastive framings for a probe, resolving adopted and composed
  root sources, layers, and replay commands.
- **`:probe explain <name> [activation|logits|evidence|ping [alpha]|prompt]`** or
  **`:probe <name> explain`** — default to an activation-only explanation from an empty
  native user message and assistant header. The selected probe supplies the subject;
  separate operation and reference directions supply candidate controls at the assistant
  decision position. There is no textual request, answer prefix, or measured-example
  requirement. `activation` selects this mode explicitly. `logits` measures local
  directional distance and signed next-token effects from an exact blank native
  assistant decision at the last selected layer, without generation, written
  instructions, probe framings, or measured-example training. The result describes
  predictive geometry on identical fresh, unsteered counterfactual histories,
  not a verified definition or a measurement of the current live conversation.
  It is retained only in the
  dedicated operator trace. `evidence` uses the empirical
  definition path and requires at least four varying measured examples
  (`:probe backfill <name>` imports archived replies). `ping` provides an activation
  nudge sanity check; `prompt` combines probe steering with a textual definition request
  and the `This probe measures ...` opening. These modes supply no stored framings
  or operator seed text to the model. Output form is checked; generic chat is an
  unqualified continuation and token-limit output is incomplete. Passing activation
  explanations remain unverified candidates and do not train naming. Dedicated
  `.activation-explain.jsonl` operator traces retain attempts separately from
  conversational and behavioral records.
  - `:tune probe_explain_subject <value>` — subject gain, default `0.02`, range `0..1`.
  - `:tune probe_explain_operation <value>` — signed operation gain, default `0.12`,
    range `-1..1`; zero disables that control.
  - `:tune probe_explain_reference <value>` — signed reference gain, default `0.04`,
    range `-1..1`; zero disables that control.
  - `:tune probe_explain_map 1` — use a learned current-state map for sustained
    operation steering. Default `0` retains the original decision pulse. The map
    predicts a raw correction at each assistant prediction position, at the last
    selected layer. Subject and reference remain decision pulses, sharing its cap.
    This map actuates residual states; it does not measure directional logit distance.
  - `:tune probe_explain_map_tokens <3..64>` — calibration tokens per trajectory,
    default `24`. Calibration pairs use exactly the same assistant token prefix
    on the blank-source and written-task sides. Task text stays out of runtime.
  - `:tune probe_explain_logits_epsilon <value>` — logit perturbation step,
    default `0.01`, strict finite range `0 < value <= 0.1`. It persists and is
    excluded from automatic and outcome calibration.
- **`:probe defining [show|on|off]`** — inspect the owning tuner's saved defining
  profile or toggle its reuse. Definition attempts record the exact request
  settings, output-form result and activation changes before generation. After
  three distinct successful captures, compatible learned steering is tried
  automatically; a rejected result falls back to the explicit request. `off`
  stops reuse while keeping observations. The profile survives shell restarts.
- **`:probe define-experiment <name>`** — run the earlier activation-only
  controller selected with `:tune probe_define_process 0|1|2|3`.
- **`:probe <name> <framing with it> || <framing without it>`** — mint from
  two contrastive framings YOU write: the direction separating them at the
  last token, plus the per-layer latent midpoint $m_\ell = \frac{1}{2}(h_A + h_B)$, poles $(h_A, h_B)$, separation scale $s_\ell = \|h_A - h_B\|$, and per-layer/per-token scoring profiles.
- **`:place <probe> <+|->`** — nudge both the probe's per-layer unit direction and its latent-space position toward (`+`) or away from (`-`) the last assistant response.
- **`:probe adopt <dim> [<dim> ...]`** — turn stored vectors into
  reply-scoring probes: the sensor vectors (`ambiguity`, `disagreement`,
  `validated_flow`, `warranted_confidence`, `urgency`, …), the
  `organic_correction` vector, or any saved probe. Grab several at once;
  a later `:probe backfill <any>` scores them all in one pass. These read
  the same axis in the REPLY that the `phen_*` streams read in the
  reasoning states.
- **`:probe compose <name> <signed mix>`** — mint from a signed, weighted
  combination of existing dimensions and probes, e.g.
  `:probe compose memory_need ambiguity + disagreement - validated_flow - warranted_confidence`
  or `:probe compose eager_but_lost 0.5*curiosity - understanding`. Terms
  resolve from active probes → stored sensor vectors → saved probe files.
  The math is honest: projecting on the normalized signed sum equals the
  weighted sum of the per-term cosines, so the mix measures exactly the
  relationship it names. Refused if the terms share no layers.
- **band suffix** — `mint`, `adopt`, and `compose` all accept a trailing
  `band <lo> <hi>` (inclusive layer indices) to set READING depth
  explicitly, independent of the live steer band, e.g.
  `:probe adopt ambiguity band 22 26`. Sensor vectors carry all layers so
  they re-read at any depth; a saved probe is honestly clipped to the
  layers it actually has.
- **`:probe backfill <name> [n]`** — retro-score up to `n` archived replies
  (default: all) in chronological order, one shared forward per reply
  scoring EVERY active probe (joint rows, deduped by timestamp). Rebuilds
  the named probe's signal stream + credit channel from the whole record
  and seeds its rolling history, so a probe minted today arrives
  pre-calibrated against everything you've already run. Reply text is all
  it needs — no live session required.
- **`:probe expose <name> [off]`** — let the MODEL consult this sensor
  itself (default: hidden). Once exposed it can read the instrument but
  never shape it (minting/composing/dropping/calibrating stay operator
  acts). See `<<PROBE: …>>` below. Exposure persists per-probe. Caveat: a
  sensor the model can read is one it can optimize against — expose
  deliberately, and compare its hidden-vs-exposed lift.

## Calibration front door

- **`:suggest`** — scan the accrued state for READY next moves, not only
  calibrations — each with its numbers and the exact command, computed but
  never applied, grouped by kind:
  - **Explored, not committed** — a circular knob explored at ≥2 values whose
    best beats current, never locked in (`:calibrate <knob> outcome`). This is
    the "I forgot to run it" catch.
  - **Calibrations ready** — thresholds mis-set on their own distribution
    (fire ≤5% or ≥95%), and every target←anchor cut the joint table supports
    (ranked by median separation).
  - **Capabilities never tried** — an opt-in knob (`claimmap_alpha`,
    `memory_alpha`, `expert_proof_weight`, `steer_layer_sweep`) with <2 tried
    values, so no verdict can be earned yet (`:tune <knob> <value>`).
  - **Probes short on evidence** — an active probe with few paired turns vs a
    deep archive (`:probe backfill <name>`).
  - **Signals worth exposing** — an unexposed probe with a strong, evidenced
    sense-lift (`:probe expose <name>`).

  **`:suggest apply`** auto-queues only the safe measurement/calibration moves
  (calibrate/commit/backfill); explore and expose change behavior or surface
  state to the model, so they stay a deliberate hand.
- Goal/autocomplete suggestions prefer the shortest runnable command. If a
  generated trailing `because` turns into an abstract dead-end, the shell
  rewinds that tail and stages the command head.
- **Calibration augments, it doesn't overwrite.** A percentile or paired
  calibration moves the value PART of the way toward the freshly computed
  target (`calibration_gain`, default 0.5) rather than replacing it — the
  first calibration of a knob adopts the target fully (blending with an
  unearned default is meaningless), and each subsequent one is an EMA step, so
  a single noisy snapshot can't wipe out accrued tuning and repeated
  calibrations converge. `:tune <name> <value>` is still a direct, authoritative
  overwrite; only calibrations blend. (Outcome calibration jumps to the proven
  tried value — blending a discrete budget would land on a value you never
  tested.)
- **`:calibrate <target> <anchor>` — paired calibration, both ways.** Any
  threshold stream can be the target OR the anchor, by name (aliases:
  `productive`→sense, `intent`, `impact`, bare probe names, and the
  reasoning-state sensors `phen_ambiguity` / `phen_disagreement` /
  `phen_validated_flow` / … plus `memory_need`, all now observed per turn).
  The target's bar becomes the cut, in its own units, between anchor-fired
  and anchor-unfired turns, over the persisted per-turn signals table
  (`invariants/out/turn_signals.jsonl`). So
  `:calibrate conversation_productive situational_authority` anchors
  productivity to the concept, and the reverse asks what authority level
  distinguishes productive turns. Refused until 5+5 paired rows carry both
  streams; probes are always labeled minted hypotheses. Bare probe names
  also work as plain percentile targets (`:calibrate situational_authority 80`).
- **Multi-anchor (joint):** join anchors with `+` and they AND together —
  a turn counts as anchor-fired only when EVERY named stream fired, e.g.
  `:calibrate curiosity understanding+ambiguity`. Needs the anchors on the
  same rows, so `:probe backfill` (joint rows) is what makes it usable
  retroactively.
- **`:calibrate <knob> outcome`** — the legitimate route for strength/budget
  knobs (alphas, fractions, routing/synthesis budgets, response tokens):
  every turn pairs each knob's value-in-force with the turn's sense, and
  this picks the TRIED value whose turns went best. Refused without real
  exploration (>=2 distinct tried values, >=5 sensed turns each): no
  exploration, no verdict. Self-percentiles for these stay refused.

## Model-invocable tags (its words, its impact)

The model can reach for tools itself; results return same-turn with a
first-person causal lead-in (`Because I asked …:`):

- **`<<MEMORY: query>>`** — long-term memory search.
- **`<<CLAIMMAP: A || B>>`** — felt comparison of two framings.
- **`<<METHODMAP: query>>`** — methodology lookup.
- **`<<DOC: words>>`** — asks to keep reading using the same state relevance
  ranking as `:doc read`, with order fallback. Its words remain event context,
  without becoming injected reading instructions.
- **`<<PROBE: name>>`** — read an EXPOSED sensor's own last-turn reading
  (with its bar and lift). `<<PROBE: name1, name2>>` or `<<PROBE: all>>`
  for several; `<<PROBE: name || candidate words>>` scores hypothetical
  words against the sensor's recent baseline WITHOUT observing, crediting,
  or touching history (a hypothetical read leaves no trace). Reading only,
  and only for probes the operator exposed; unexposed names are refused.

## Config-level switches (not shell commands)

- `emergent_experts_enabled` (AgenticConfig, default False) — allows the
  mesa Committee to mint new experts from recurring self-correction deltas
  (bounded roster). Off by default so benchmark lanes keep the fixed-roster
  control.
- `TDA_DISABLED_STEER_CHANNELS` (env) — per-channel ablation for isolation
  runs (`scripts/isolate_channel_lifts.py` drives it).
- `TDA_STEER_CAP_FRACTION`, `TDA_STEER_BAND_LO/HI`, `TDA_STEER_FRACTION`,
  `TDA_URGENCY_MAX_COEFFICIENT` (env) — startup defaults for the same knobs.

- `:plasticity [top N|pair A B|history N|on|off|reset [A B]|set KNOB VALUE|save]` -- inspect and tune persistent co-activation geometry. Set `mass_strength 0` for distance-only modulation; requires gravity steering for field effects.
