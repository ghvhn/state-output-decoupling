# Interactive Shell Commands

Reference for `scripts/interactive_phenomenality.py` (launch via
`run_phenomenality_shell.cmd`). House rules that hold everywhere: readouts
are free and honest about insufficient evidence; anything that MOVES the
system is a deliberate command (nothing auto-drifts); tuned values persist
across sessions; anything folded into the model's context is minimal and in
its own first-person voice.

## Session & context

- **`exit` / `quit`** — close cleanly (logs `shell_closed`; an interrupted
  session can auto-resume its unanswered message next launch).
- **`:context`** — rolling current-session transcript state.
  **`:context on|off`** include/exclude recent turns from the prompt;
  **`:context clear`** empties it. Capped at 50 turns / 16,000 chars (the
  conversation is the one context that scales).
- **`:history`** — pointer: transcript is `:context`, long-term is `:memory`.

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

## Steering readout & calibration

- **`:steer`** — envelope dashboard: current cap/band; observed push
  distribution + clip rate; data-implied cap (refuses under 64 observations);
  synthesis-evidence band vs layer-steer band (transfer-free, per-layer
  table); channel-lift table (fired vs unfired outcomes per channel).
- **`:tune`** — every knob with value, fire rate, signal distribution, lift.
- **`:tune <name> <value>`** — set a knob.
- **`:tune <name> auto [percentile]`** — calibrate a threshold to its own
  observed distribution.
- **`:tune steer_cap_fraction auto [pct]`** — cap from observed push ratios
  (default p95).
- **`:tune steer_band auto [min_events] [gold|conversation|any] [synthesis|layersteer]`**
  — band from per-layer outcomes, by evidence lane and kind.

Knobs you'll actually touch:

| Knob | Default | Moves |
|---|---|---|
| `claimmap_alpha` | 0.0 (off) | ClaimMap steer strength; start ~0.02, watch |
| `memory_alpha` | 0.0 (off) | memory-retrieval steer strength |
| `claimmap_tension` | ~0.18 | auto-ClaimMap fire threshold |
| `memory_need` | 0.05 | state-triggered memory-gap threshold |
| `steer_cap_fraction` | 0.5 | envelope: max push / residual norm |
| `steer_band_lo/hi` | 0.40/0.70 | depth window for band steers |
| `steer_fraction` | 0.25 | agentic delta/branch scale |
| `steer_layer_sweep` | 0 (off) | 1 = one least-tested layer per steer; per-layer outcomes accrue |
| `conversation_productive` | 0.0 | sense threshold labeling turns productive |
| `response_tokens` | 512 | reply token budget (cut-off suggestions reference it) |
| `eot_urgency` | 0.05 | P(end-of-turn) below this at the budget = "cut off mid-thought" |
| `sandbox_success` | 0.5 | observation stream of real execution outcomes |
| `words_had_impact` | 0.5 | observation stream of word-caused turns |

## Documents & reading

- **`:doc <path> [because <why>]`** — ingest (deterministic chunks, sha-deduped,
  full provenance into memory) and stage chunk 1 with the first-person frame.
  Accepts a directory (walks .md/.txt) or a glob; `{filename}` and
  `{last_updated}` in the why are substituted per file.
- **`:doc next`** — stage the following chunk (explicit advance).
- **`:doc status`** — per-document unread counts, staged/auto-read state.
- **`:doc read [n] [order|interleave|reply]`** — reading as dialogue (cap 20
  turns): the framed chunk IS the user turn. `order` = document order,
  non-adapting; `interleave` = least-read document speaks next; `reply` =
  echo-following (deliberate use; caveat stated).
- **`:doc inject`** — stage the ENTIRE library for one turn, budget-bounded
  and framed; refuses honestly when the library exceeds the budget.
- **`:doc stop`** — interrupt auto-read.

## Sandbox & agency

- **`:sandbox on|off|status`** — default off. When on, a fenced ```python
  block in the model's reply executes for real (isolated interpreter, 10s
  timeout); next turn carries "I ran the code I wrote; exit_code=…" with the
  actual output. Logged with provenance; success observed.
- **`:impact`** — agency ledger: rate of word-caused turns, consequence
  trail, and the lift readout (does experienced impact track better
  deliberation?).

## Model-invocable tags (its words, its impact)

The model can reach for tools itself; results return same-turn with a
first-person causal lead-in (`Because I asked …:`):

- **`<<MEMORY: query>>`** — long-term memory search.
- **`<<CLAIMMAP: A || B>>`** — felt comparison of two framings.
- **`<<METHODMAP: query>>`** — methodology lookup.
- **`<<DOC: words>>`** — asks to keep reading; its words pick the chunk
  (reply-mode selection over the query, order-fallback stated honestly).

## Config-level switches (not shell commands)

- `emergent_experts_enabled` (AgenticConfig, default False) — allows the
  mesa Committee to mint new experts from recurring self-correction deltas
  (bounded roster). Off by default so benchmark lanes keep the fixed-roster
  control.
- `TDA_DISABLED_STEER_CHANNELS` (env) — per-channel ablation for isolation
  runs (`scripts/isolate_channel_lifts.py` drives it).
- `TDA_STEER_CAP_FRACTION`, `TDA_STEER_BAND_LO/HI`, `TDA_STEER_FRACTION`,
  `TDA_URGENCY_MAX_COEFFICIENT` (env) — startup defaults for the same knobs.
