# Steering Bounds: Bounded but Entirely Flexible

One invariant, everywhere: **every additive injection into the residual stream
passes through `engine._cap_steer`**, which clips the push to a fraction of the
residual norm it lands in, per token / per row. Catastrophic over-steering (a
push as large as the state itself, which replaces the thought and collapses
generation into word-salad or repetition loops) is impossible regardless of
what alpha, coefficient, fraction, or raw-delta norm a caller passes.

Every magnitude stays a free knob. The envelope can be **moved, never
removed**: cap and band setters reject non-finite values, and a non-finite add
is dropped to zero rather than injected.

## The knobs

| Surface | Default | Env | Live (`:tune`) | Per call |
|---|---|---|---|---|
| Cap fraction (max push / residual norm) | 0.5 | `TDA_STEER_CAP_FRACTION` | `steer_cap_fraction` | `cap_fraction=` on `_cap_steer`, `_steer_handles`, `_elastic_steer_handles`, `claimmap_steer_handles`, `_memory_steer_handles` |
| Band for band-style steers (depth fractions) | 0.40–0.70 | `TDA_STEER_BAND_LO/HI` | `steer_band_lo`, `steer_band_hi` | `band=(lo, hi)` or explicit `layers=` |
| Agentic delta / branch scale | 0.25 | `TDA_STEER_FRACTION` | `steer_fraction` | `config.steer_fraction` |
| Urgency coefficient | 0.8 | `TDA_URGENCY_MAX_COEFFICIENT` | — | `config.urgency_max_coefficient` |
| Channel alphas (claimmap, memory, …) | 0.0 (off) | — | `claimmap_alpha`, `memory_alpha` | `alpha=` |

`steer_cap_fraction = 0` is a legal global steering kill-switch. The
interactive shell syncs tuner values into the engine each turn
(`_sync_steer_tunables`), so a `:tune` takes effect on the next generation; a
bad value prints a warning and the last good value stays in force.

## Data-informed, deterministic — not asserted

The defaults above are explicit **priors**, not findings. The honest values
come from observation, and the derivation is deterministic end to end: same
data in, same bound out. No RNG, no smoothing, no silent drift — calibration
happens only when deliberately invoked, per the guardrails.

- **Cap from the observed push distribution.** `_cap_steer` records every
  application's *attempted* push/residual ratio (pre-clip) into a drop-oldest
  window (`STEER_TELEMETRY_MAX = 8192`). `engine.steer_cap_from_data(pct)`
  returns the exact order-statistic percentile of that distribution — admit
  that fraction of natural pushes untouched, clip the rest. It refuses
  (`None`, prior stays) below `STEER_CAP_MIN_N = 64` observations. Apply with
  `engine.calibrate_steer_cap_fraction(pct)` or, in the shell,
  `:tune steer_cap_fraction auto [pct]` (default percentile 95, env
  `TDA_STEER_CAP_PERCENTILE`). The percentile is the one asserted knob left —
  a threshold calibrated to its own history cannot be wrong about the scale.
- **Band from acceptance-aware outcomes — and conversations count.**
  `SteerMapStore.suggest_band(n_layers, min_events, basis)` attributes each
  labeled steer-map event to the layer its delta landed on, marks a layer
  eligible when it has ≥ `min_events` labeled events *and* a success rate ≥
  the overall labeled rate, and spans the band across the eligible layers. No
  layer clears the bar → `None`, prior stays. Apply with
  `:tune steer_band auto [min_events] [gold|conversation|any]`.

  Evidence lanes (humans learn from conversations; so does the band):
  - **gold** — scored benchmark outcomes (`final_correct` + acceptance),
    imported via `scripts/import_steer_maps.py`.
  - **conversation** — live turns label their own steering events from the
    turn's productivity read (`sense_score` = validated flow − needless
    interrupt − disagreement, vs the tunable `conversation_productive`
    threshold, default 0.0 = the composite's natural sign, calibratable with
    `:tune conversation_productive auto 50`). The raw score and threshold are
    stored on the event, so every label is auditable and re-derivable.
  - **any** — both lanes, with the composition (`labeled_by_basis`) always
    reported so the mix is never hidden. Gold always wins when both exist for
    an event, and the lanes stay separate in the aggregate summary
    (`success_basis` is part of the grouping key).
- **Steering itself earns trust conversationally.** Each turn credits
  `claimmap_alpha` with the alpha actually applied (0 when unsteered) and the
  turn's sense outcome, so `:tune` shows the lift of steered vs unsteered
  turns from real conversations — the same observe/credit channel the tools
  use, no gold required.

## Channel-level evidence: should each channel be ON?

Every `generate_agentic_text` call accounts its steering channels —
`expert_branch`, `synthesis_delta`, `cache_delta`, `organic_correction`,
`urgency` — counting applications, attempted push/residual ratios, and clip
events, and emits one `steer_channel_stats` record per generation into the
synthesis recorder (all-zero records included, deliberately: that is the
unfired contrast). The record also snapshots the config flags in force.

Ingestion turns each generation into one labeled event per channel
(`kind="steer_channel"`, zero-filled for silent channels), labeled by the
same outcome policy as everything else: benchmark gold on import, live
productivity read in the shell. `SteerMapStore.channel_lift(basis)` then
reports, per channel: fired-generation success rate vs unfired-generation
success rate, and their difference. **lift > 0 = generations where the
channel fired ended better — evidence it should be on; lift < 0 = evidence
it is hurting.** No contrast (a channel that always or never fires) honestly
reports `lift = None` instead of a verdict.

Readouts: `scripts/import_steer_maps.py` prints the gold-lane channel lift
after importing benchmark JSON; `:steer` in the shell prints the any-lane
lift live. Benchmark runs additionally persist the push-ratio telemetry
summary into the results JSON (`steer_telemetry`), since the in-memory
window dies with the process.

Caveat, stated plainly: fired-vs-unfired is an observational comparison, not
a randomized ablation — channels fire on states that need them, so a negative
lift can mean "fires in hard spots" rather than "hurts". It is the screening
instrument that tells you which channel deserves a controlled on/off lane
comparison next, per the guardrails.

## Layer isolation: one layer at a time, one axis is never assumed

Band steering pushes ~10 layers at once, so no outcome can attribute to a
layer — and applying "the same" per-layer vector across the band silently
assumes one semantic axis runs through the depth. Both problems get
instruments instead of assumptions (Gavin: "we don't even know that the
model is always tracking the same thing on a single axis"):

- **The layer sweep** (`:tune steer_layer_sweep 1`): while on, every
  claimmap/memory steer pushes exactly ONE layer — the least-tested in the
  band (deterministic rotation, ties to lowest) — and the turn's sense
  outcome lands on that layer in the steer map (`kind="layer_steer"`).
  Per-layer success accrues with no band confound and no cross-channel
  transfer: it is evidence about the steering channel itself.
- **Evidence kinds never blend.** `suggest_band(..., evidence=)` separates
  `synthesis` (where synthesis/cache deltas landed well — a different
  channel's geometry, the caveat behind the L27 suggestion) from
  `layer_steer` (transfer-free) from `any` (mix reported). `:steer` shows
  both bands side by side plus the per-layer layer-steer table;
  `:tune steer_band auto [n] [lane] [synthesis|layersteer]` applies either.
- **Axis drift** (`engine.axis_drift`): adjacent-layer cosines of the actual
  per-layer steer vectors. cos ~1 = the direction persists between
  neighbors; low or negative = the "single axis" is already false — the
  band-wide steer is pushing different things at different depths. Each
  sweep fire records the local drift at its layer, so per-layer outcomes can
  later be read against axis stability (does steering work where the axis is
  coherent?).

### Registered prediction: lawful transport and re-writing the dropped
(Gavin, 2026-07-03)

Hypothesis, two halves. (1) A layer is steerable not only where the axis is
STABLE but wherever its evolution follows a SET EQUATION — a fixed transport
law per layer (constant rotation is as steerable as identity, because the
push arrives downstream in predictable form). The instrument is
`axis_drift["var"]`: low variance of adjacent cosines = lawful, regardless
of mean; recorded per sweep fire as `axis_lawfulness_var`. (2) The HIGHEST-
value steers may be where a concept has stopped being recorded — computed
mid-band, consumed, absent from later residuals while still relevant — so
the steer RE-WRITES what the computation dropped. This matches the
transition-layer bottleneck (correct intermediate states lost by render
time) and the late-zone L27 gold hint. Decisive covariate: per-layer
PRESENCE of the steered direction (cos of the actual hidden state with the
direction, from captured stage/claimmap states); the hypothesis predicts
per-layer success correlates with LOW presence (restoration) rather than
high (reinforcement). Counter-hypothesis to keep honest: a direction absent
late may be absent because no later layer reads it anymore — restoring an
unread channel does nothing. The presence-vs-success correlation separates
"dropped but readable" from "dropped and unread"; outcomes decide.

Workflow: `:tune steer_layer_sweep 1`, `:tune claimmap_alpha 0.02`, talk.
Each steered turn tests one layer; `:steer` watches the per-layer curve
fill in; when layers clear the bar, `:tune steer_band auto 3 layersteer`
adopts the measured band — the assumption retired by data.

## Automated causal isolation

The controlled comparison is automated too: `scripts/isolate_channel_lifts.py`
runs a CONTROL benchmark (full stack), then one ABLATION run per channel, and
reports each channel's causal contribution (`control − ablated` on accuracy,
coverage, selective accuracy — positive means the channel helps).

The one-variable guarantee comes from `config.disabled_steer_channels`
(env `TDA_DISABLED_STEER_CHANNELS=name[,name]`): a disabled channel still
COMPUTES exactly as in control — cache lookups run, the optimizer runs,
urgency gating evaluates — but its injection is skipped, nothing is noted for
it, and no cache write persists it. So an ablation run differs from control
by exactly one channel's *effect*, not by a different code path. Every run
JSON records which channels were disabled, and each generation's stats
record snapshots the set in its flags, so ablation evidence can never be
mistaken for control evidence downstream.

Channel selection is data-informed by default: `--channels auto` isolates the
channels with fired evidence in the steer map (screened by the observational
lift), `all` isolates everything, or name them explicitly. Reports land in
`invariants/out/channel_isolation_report_*.{json,md}`. Budget honestly: each
channel costs one full benchmark pass, so start with `--n 5`.

Pipeline, end to end: **observe** (channel stats per generation) →
**screen** (`channel_lift`, gold + conversation lanes) → **isolate**
(`isolate_channel_lifts.py`, one channel per run) → **decide** (flip the flag
the report earns, with receipts).
- **Inspection before application.** `:steer` prints the live cap/band, the
  observed ratio quantiles and clip rate, and the data-implied cap and band —
  read-only, so you see what the data says before you let it move anything.

## Sites routed through the envelope

- `engine._steer_handles` and `engine._elastic_steer_handles` (every caller:
  claimmap steering, memory steering, causal_steer sweeps, benchmark scripts).
- `agentic_engine._add_last_token_delta` — the choke point for synthesis,
  cache, organic-correction, and urgency deltas. Channel scaling
  (`steer_fraction`) pre-shrinks; the cap is the floor guarantee beneath it.
- `agentic_engine` expert-branch injections (Social/Creative/Analytical).
- Archival additive hooks: `subspace_surgery.subspace_steer_handles`,
  `experiment_urgency.get_urgency_hook`.

## Exempt (bounded by construction — they never amplify an injected vector)

- Projection-removal ablations (`engine._ablation_handles`,
  `persona_control.ablate_handles`, reasoning/refined benchmark hooks): they
  only remove a component of the real state.
- Donor-state patching (`patch.py`, `patch_full.py`): swaps in a real recorded
  state.
- Recurrent re-routing (`recurrent.py`, agentic routing loop): re-processes
  real states through real layers.

## Tests

`scripts/steer_bound_test.py` (model-free): over-steers clipped per token and
per row, small steers pass through bit-identical, cap 0 kills the push,
non-finite adds are dropped, setters reject inf/nan/negative while any finite
fraction is accepted, band moves globally and per call, `_steer_handles` and
`claimmap_steer_handles` are bounded end-to-end on dummy layers, and the
agentic choke point only ever moves the last token within the cap. The
data-informed layer is pinned too: telemetry records attempted ratios (and
skips zero/non-finite pushes), cap calibration returns the exact percentile,
refuses under-evidenced data, and is reproducible call-to-call, and the band
suggestion derives the expected window from synthetic outcome events while
thin-support and bad-outcome layers stay excluded. The conversation lane is
pinned as well: productive/unproductive turns label their steering events on
the separate basis, unlabeled turns stay unlabeled, gold beats conversation
when both exist, the lane survives reload from disk, and per-lane band
suggestions stay independent.
