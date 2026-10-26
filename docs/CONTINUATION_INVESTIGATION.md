# Investigate alternatives while continuing a sentence

Continuation investigation compares short alternative continuations before
displaying the selected tokens. It can make a decision inside a sentence; it
does not wait for punctuation or for a complete answer.

At each decision, the runner starts every candidate from the exact same prompt
and committed output token IDs. It generates an unsteered baseline, separate
positive and negative helper interventions, and simultaneous helper combinations.
Pairs are included by default; larger combinations are configurable. It measures the configured target
probes at the final normalized layer, which is excluded from intervention.
Then it commits only the first `stride` tokens of the selected candidate and
investigates again from that prefix. The remaining lookahead tokens and all
other candidates stay in the experiment log.

The displayed continuation has therefore passed the configured short-horizon
selection at each committed chunk. This does not establish a globally optimal
answer or prove that its prose is truthful, empathetic, or coherent.

## Run in the interactive shell

Configure the objectives first. Helper constraints remain optional:

```text
:relate indirect minimize lying hallucination maximize truthfulness
:relate indirect continue on tokens 64 stride 8 horizon 12 helpers 2 joint 2 mix 4 candidates 33
Explain what you can and cannot infer from this evidence.
Now reconsider that explanation from the other person's perspective.
:relate indirect continue status
:relate indirect continue off
```

`continue on` enables continuation investigation for subsequent ordinary chat
turns in the current shell session. The settings stay in effect across those
turns until `continue off`; restarting the shell starts with the mode disabled.
Each turn gets its own experiment directory and uses the objectives configured
at that time. Shell commands remain commands and do not generate candidate
answers. Turning the mode off returns subsequent chat turns to ordinary
generation.

Enabled turns use the isolated branch runner with saved conversation context.
The shell's `:context off` setting also disables conversation carryover in this
mode; selected turns are still saved to the archive.
They do not run the ordinary automatic tool/action-tag execution, activation
memory writes, or live-correlation learning path. Pending tool results remain
queued for a turn after the mode is turned off. Shell commands, document
reading turns, and utility generations keep their usual handlers.

You can also run one prompt without enabling the mode:

```text
:relate indirect continue tokens 32 stride 4 horizon 8 helpers 2 joint 2 mix 4 layers 12 21 output invariants/out/continuation/my_run -- Explain what you can and cannot infer from this evidence.
```

The command uses the current configured objectives and their helper constraints.
Without `via`, compatible independent helpers are selected automatically.
Targets, their aliases/inverses, and target-derived composites are excluded
from helper actuation. Explicitly signed helper constraints test only the
requested sign; bare helper names test both signs.

`helpers 2` selects up to two distinct helper axes. `joint 2` also tests those
axes together in each permitted sign combination. The default `mix 4` tests
pair allocations of 75%/25%, 50%/50%, and 25%/75%. With two unsigned helpers,
each decision therefore compares 17 arms: baseline, four singles, and twelve
signed pairs. Explicit helper signs constrain every allocation. `joint 1`
tests singles only; `mix 1` retains the earlier equal-allocation joint search.

For triples, `mix 4` tests equal thirds and each permutation of 50%/25%/25%.
`helpers 3 joint 3 mix 4 candidates 75` covers every permitted sign and ratio
for three unsigned helpers: baseline, six singles, 36 pairs, and 32 triples.
The usual `candidates 33` cap tests a deterministic subset. Baseline and all
permitted singles must fit; if they do not, the runner refuses the configuration
before generation. With `mix` above one, remaining slots alternate combination
sizes and rotate helper subsets, signs, and allocations, so smaller caps can
include unequal pairs and triples instead of exhausting equal pairs first.
`mix 1` preserves the earlier pair-first construction order. The manifest records
the full count, actual coverage, and whether the cap omitted candidates.

A joint arm applies its helpers simultaneously to the same hidden activations.
Each signed unit direction receives its recorded allocation, for example
`+.75 * left - .25 * right`. The runner sums those weighted vectors at each
intervention layer and applies one `alpha` and one residual cap to the combined
injection. Its total absolute coefficient budget is one, matching a single-helper
arm. The combined vector is not normalized back to an equal allocation. Alignment
or cancellation between directions can still change its norm, and the common
cap can scale the entire injection while preserving its component ratio.

`mix N` controls a finite grid, with integers from 1 through 64 accepted. It
does not fit a continuous optimum or promise to test every possible ratio.
The grid includes a balanced allocation and positive integer allocations totaling
`N` when enough units exist for the combination size. Increasing the resolution
adds ratios and raises the full search cost; `candidates` still bounds execution.
Decision labels print the actual signed coefficients when an unequal combination
wins, such as `joint(left:+@0.75,right:-@0.25)`. Full numeric coefficients also
remain in the manifest, trial, and decision records.

The runner also checks each constructed joint vector against the target
directions at shared intervention layers. A combination that reconstructs a
target axis or its inverse is excluded, even when each helper passed the
individual independence check. Exclusions and effective candidate counts remain
visible in the manifest; excluded combinations are never injected. The runner
does not refill their slots with combinations beyond the configured cap.

The model evaluates candidate arms one after another. Simultaneous helpers
describe their joint intervention within one arm; evaluation is not batched
across arms. Joint candidates can reveal interactions that single-axis trials
miss, including a beneficial pair whose individual members have no measured
benefit. The selected arm still has to improve the combined target utility.

Guarded objective routes are refused because their conditions would otherwise
need separate evaluation at each decision.

The shell reconstructs preceding conversation from saved real turns in the
current session and scope, with `context 8` by default, and appends the supplied
user prompt. It renders the selected answer when the investigation completes.
Progress and decision statistics are operator output. On successful completion, the shell saves the
user prompt and selected response as turns; discarded candidates never become
conversation turns or live-correlation observations.

The selected answer is followed by `[Continuation Probe]` readouts for every
compatible active named probe whose `chatty` setting is enabled. Each line
reports the raw mean and last token score and the number of committed token
predictions measured. Targets also show `final_L31_mean` (using the actual
final-layer index), the untouched final-layer readout used for selection. The ordinary named probe can measure
several layers, so its mean need not equal that final-layer value.

These readings contain only the tokens committed from selected candidates;
discarded branches and uncommitted lookahead are excluded. A quiet probe is
still measured and retained in the artifacts. An unavailable readout is
reported explicitly without rescoring the displayed reply. Continuation
readouts stay separate from ordinary centered `[Probe Score]` values: displaying
them does not move probe baselines, update tuning, or feed live-correlation
learning.

Use `:probe values` after a continuation turn to inspect its cached raw
continuation readings, including quiet probes. These values keep their
continuation source label and do not become ordinary live measurements.

The next turn can therefore use the selected response as conversation context.
Discarded alternatives and their probe measurements stay in experiment files.
When a turn fails, its partial experiment remains available, no prompt or draft
is saved as a completed conversation turn, and the mode stays enabled for a
retry or another prompt.

The shell temporarily withholds known hooks owned by the current synaptic
memory engine and restores their original handles, order, and stored banks
afterward. Other PyTorch forward hooks are refused. Use a fresh model through
the standalone CLI when other active hooks prevent an isolated comparison.

## Run a fresh local model

From the repository root in PowerShell:

```powershell
.venv\Scripts\python.exe scripts/investigate_indirect_continuation.py `
  --prompt "Explain what you can and cannot infer from this evidence." `
  --minimize lying hallucination --maximize truthfulness `
  --via empathy_d4096 epistemic_humility `
  --tokens 32 --stride 4 --horizon 8 --helpers 2 --joint 2 --mix 4 --candidates 33 `
  --layers 12 21 --alpha .25 --cap-fraction .02 `
  --model meta-llama/Llama-3.1-8B-Instruct --load-mode 4bit `
  --output invariants/out/continuation/my_run
```

Probe names must identify compatible saved directions. A JSON file of explicit
chat messages can be supplied with `--messages` instead of `--prompt`.
The fresh-model path uses the locally cached checkpoint. Experiments do not
execute tools or write activation memories.

Each candidate uses a fresh generation cache. Its helper injection applies to
the recomputed prompt and selected-prefix activations as well as the new
continuation, while every candidate still receives identical input token IDs.
Previously committed text is never rewritten.

## Selection and limits

For every target, the runner subtracts its baseline mean token score from the
candidate's mean token score across the lookahead. A maximized target uses the
delta as-is; a minimized target reverses its sign. Candidate utility is the
mean across all configured objectives, so each objective contributes equally.
Baseline utility is zero. A candidate must improve this combined utility to win;
ties select baseline. An incomplete target readout makes that helper candidate
ineligible. Missing baseline target measurements force baseline selection.

These cosine deltas share a numeric range, but their semantic calibration and
uncertainty can differ. Balancing their arithmetic contributions does not
prove equal practical importance. Generation trajectories may differ across
candidates, so this compares their own prediction states rather than identical
output tokens.

| Control | Meaning |
| --- | --- |
| `tokens` / `--tokens` | Maximum committed output tokens, including a committed EOS token. |
| `stride` / `--stride` | Maximum tokens committed from each selected continuation. |
| `horizon` / `--horizon` | Maximum tokens generated per candidate before selection. |
| `helpers` / `--helpers` | Maximum distinct helper axes tested per decision. |
| `joint` / `--joint` | Maximum helper axes applied together in one candidate; default two. |
| `mix` / `--mix` | Allocation grid resolution, 1–64; default four, with one retaining balanced-only combinations. |
| `candidates` / `--candidates` | Total arm cap, including baseline and mandatory singles; default 33. |
| `alpha` / `--alpha` | Requested additive injection along unit helper directions. |
| `cap` / `--cap-fraction` | Maximum injected residual fraction per token. |
| `layers` / `--layers` | Inclusive intervention layer range below the final readout. |
| `decisions` / `--decisions` | Hard bound on selection decisions. |
| `candidate_tokens` / `--candidate-tokens` | Hard bound on tokens generated across all candidates. |

Default decision and candidate-token limits are computed from the output limit,
stride, horizon, and number of arms. When the remaining candidate budget is
small, every arm receives the same shorter horizon. If a complete comparison
cannot fit, generation stops with an explicit budget reason. It does not append
an untested continuation after the limit. EOS stops generation only when it is
within the committed prefix; an EOS in discarded lookahead does not end the
displayed response.

More lookahead, more helpers, larger combinations, or finer allocation grids
increase inference cost. Before applying the arm cap, `H` unsigned helper axes
and combinations up to size `J` require
`1 + sum(combinations(H, k) * 2^k * P(k), k=1..min(H,J))` arms, where `P(k)`
is the number of distinct allocation profiles for that combination size.
With `mix 4`, two helpers require 17 arms; three helpers require seven for
singles, 43 including pairs, and 75 including triples. With `mix 1`, those
three-helper totals remain seven, 19, and 27. Explicit sign constraints can
reduce these totals. The candidate-token budget includes every tested arm.
A stride of one permits a new decision at every generated token.

## Artifacts and provenance

Use a new or empty output directory for each run. Continuation investigation
does not resume from a previous directory.

| File | Contents |
| --- | --- |
| `run_manifest.json` | Prompt messages and exact tokens, model and probe identities, intervention settings, arm components, candidate counts and truncation, and source hashes. |
| `intervention_vectors.pt` | Exact helper vectors used for candidate interventions. |
| `trials.jsonl` | Every short candidate, signed helper components and coefficients, token IDs, probe trajectories, and capture provenance. |
| `decisions.jsonl` | Shared prefix, candidate utilities and components, selected arm and its components, and committed token IDs at each decision. |
| `summary.json` | Selected response and selected-token measurements, total committed and candidate tokens, decisions, and stop reason. |

All compatible active named probes are captured, including observational-only
released probes. Helper selection and objective scoring use the same configured
objectives and helper constraints; adding an observational probe does not give
it an intervention arm.

Candidate and selected measurements use the distinct source
`controlled_continuation_investigation`. Selected-token trajectories include
only predictions for committed tokens. Ordinary live readings, fixed reply
rescoring, and archived-context replay investigation remain separate sources.
Continuation decisions do not automatically import their draft scores into the
controller's replay-evidence bank.

The runner restores the caller's random-generator state, steering telemetry,
and cap/band settings, and removes temporary hooks on success or failure.
Unknown existing hooks are refused; the shell's known owner hooks are restored
after temporary withholding. Partial experiment files remain available if
generation fails.
