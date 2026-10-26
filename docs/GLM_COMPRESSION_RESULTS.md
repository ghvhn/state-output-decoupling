# Measured GLM compression results

The native GLM-5.3 teacher capture succeeded. The experimental hybrid combines
INT4 storage, an explicitly protected probe span, and calibration PCA directions
in that span's orthogonal complement. It roughly halves held-out normalized
response error versus INT4 in both linear and sampled nonlinear tests. Fitted
seed-pole contrast error falls by 95.99% in the nonlinear test. Plain PCA at
the same rank still has lower general response error; the hybrid preserves
the designated contrasts much better. These are scoped numerical comparisons,
not a validated compressed GLM decoder.

## Evidence and scope

- Teacher: `zai-org/GLM-5.3`, revision
  `aca966e4e02791568aa6a4ced368624b3d897f42`.
- All 147 source files were staged and SHA256-verified on Modal:
  755,663,668,166 bytes. Native FP8/TP8 GPU capture saved 410 contexts and
  all 8,336 selected positions.
- Local full-vocabulary reconstruction exported all 8,336 positions.
  Two saved anchors passed the declared BF16 tolerances with 100% argmax
  agreement. Recorded reply text was teacher-forced input; these are not
  newly generated GLM responses.
- Compression source: eight layer-3 gate matrices, each `[2048, 6144]`,
  with 100,687,872 original FP8/scales bytes.
- Fit: 1,022 states in 32 whole trajectories, selected under a 1,024-state
  budget. Evaluation: all 2,432 held-out states in 76 trajectories, including
  1,216 measured adjacent pairs. Seed calibration supplied 77 probe contrasts.

## Linear comparison

The complete sixteen-candidate run finished in 138.25 seconds on the CPU.
These principal candidates share the identical INT4 codes and scales.

| Repair | Rank | Serialized MB | Held-out state error | Held-out difference error | Native span error |
| --- | ---: | ---: | ---: | ---: | ---: |
| INT4 baseline | — | 53.481 | 0.093944 | 0.088268 | 0.088483 |
| Native probe span | 77 | 60.447 | 0.086430 | 0.085727 | 0.0000002993 |
| Calibration PCA | 77 | 60.693 | 0.047626 | 0.064012 | 0.084828 |
| Seeded random | 77 | 60.426 | 0.093470 | 0.087715 | 0.087935 |
| Native span + complement PCA | 154 | 67.660 | 0.046511 | 0.062887 | 0.0000002992 |
| Calibration PCA | 154 | 67.632 | 0.042841 | 0.057733 | 0.083045 |
| Seeded random | 154 | 67.365 | 0.092652 | 0.087132 | 0.087386 |

MB is decimal storage; all errors are relative RMSE. Repairs within each rank
share raw tensor bytes; their evidence metadata changes total serialized size.
The seed witness bank is stored separately and is not included in these sizes.

The hybrid is **32.80% smaller than the original FP8 sample**. Relative to
INT4, it uses **26.51% more bytes** and reduces held-out state error by
**50.49%**, and state-difference error by **28.75%**. The native span is
preserved to numerical rounding. This is an explicit constraint/storage
tradeoff; the same-size PCA control has lower general response error.

The protected-span test uses the same bank as fitting. It establishes
reproduction of those measured directions, not validated probe meanings or
generalization to new concepts. State differences are not per-token velocity.
Trajectory weights are balanced; conversations and source categories are not.
Provenance and stimulation weights are all one in this selected dataset.

All eight matrices are applied equally to every input, including counterfactual
expert responses. These experts carry only about 3.22% of actual router weight.
The comparison does not establish compressed-decoder language quality,
generation behavior, full-model storage feasibility, or runtime savings.
The held-out set is now a development comparison because its results motivated
the hybrid; a fresh sealed test set is required for a final quality claim.

## Nonlinear expert comparison

The seven predeclared INT4/repair candidates were then evaluated without any
refitting. Original up and down weights were downloaded as two bounded,
checksum-verified 100,687,872-byte samples. The CPU reference follows
`down(SiLU(gate(x)) * up(x))`, matching the saved configuration and
[the pinned expert implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py).
Only the gate weights change. All earlier selected states and the 154 original
measured seed-pole inputs are used, rather than reconstructed pole vectors.
The complete run finished in 68.125 seconds.

| Gate repair | Rank | Held-out output error | Held-out output-difference error | Fitted-seed output-contrast error |
| --- | ---: | ---: | ---: | ---: |
| INT4 baseline | — | 0.052569 | 0.061625 | 0.057769 |
| Native probe span | 77 | 0.046941 | 0.057928 | 0.036639 |
| Calibration PCA | 77 | 0.026734 | 0.038484 | 0.035770 |
| Seeded random | 77 | 0.051762 | 0.061350 | 0.057462 |
| Native span + complement PCA | 154 | 0.026139 | 0.037789 | 0.002319 |
| Calibration PCA | 154 | 0.024078 | 0.034810 | 0.034457 |
| Seeded random | 154 | 0.051625 | 0.060757 | 0.056913 |

Errors are relative RMSE. Seed contrast is the mean of per-probe normalized
squared output-contrast errors, followed by a square root; it is different
from the preceding linear span's global Frobenius error. Temporal differences
are `f(next) - f(previous)`, and seed contrasts are `f(positive) - f(negative)`.

Against INT4, the hybrid reduces held-out normalized output RMSE by **50.28%**,
output-difference RMSE by **38.68%**, and fitted-seed normalized contrast RMSE
by **95.99%**. Equal-rank PCA remains **7.88% better on general held-out output
RMSE**, while its fitted-seed contrast error is about 14.86 times the hybrid's.
Its serialized artifact is 28,672 bytes smaller; raw tensor payloads match.

Native-span repair alone has near-zero linear span error but 3.66% nonlinear
seed-contrast RMSE. Preserving the gate's difference between two examples does
not preserve its absolute response at each example. A common gate offset can
affect SiLU differently at the two endpoints, and their up responses differ.
The hybrid's calibration complement plausibly reduces this context offset;
that explanation is consistent with these measurements and is not a separate
causal finding or evidence that probe meanings were learned.

Storage savings still refer to gate weights. Keeping the two original
up/down groups adds 201,375,744 fixed raw source bytes: the hybrid's sampled
three-projection total is 269,036,194 bytes versus 302,063,616 original bytes,
a **10.93%** reduction before witness/metadata overhead for those unchanged
source groups. No full-model compression ratio is established.

The evaluator uses CPU FP32 dequantized matrices and FP64 aggregate metrics.
It does not reproduce native dynamic-FP8 activation rounding, router mixtures,
shared experts, residual connections, later layers or vocabulary scores under
compressed weights. The seed endpoints were used to fit the native span, so
their contrasts are preservation evidence, not held-out concept generalization.
All correct, incorrect and unknown predictions remain evidence. The largest
declared numerical tensor estimate is 802,867,828 bytes under a 1 GiB budget;
read-only source mappings, allocator and BLAS overhead are outside this estimate.

## Artifacts and validation

- `invariants/out/glm_capture_trial_20261002/hybrid_comparison/results.json`:
  all 16 actual candidate results and separately scoped metrics.
- `invariants/out/glm_capture_trial_20261002/hybrid_figure/comparison.png`
  and `.svg`: reviewed figure with all 48 finite metrics.
- `hybrid_figure/plot_manifest.json`: input/output hashes, actual selected
  counts, complete source archive, and both matched repair groups.
- `invariants/out/glm_capture_trial_20261002/nonlinear_comparison/results.json`:
  seven saved-artifact evaluations, actual raw-pole contrasts and their scopes.
- `invariants/out/glm_capture_trial_20261002/nonlinear_figure/comparison.png`
  and `.svg`: reviewed nonlinear figure, with a seven-row CSV and hash manifest.
- Seven related suites passed **325 tests**, including split exclusions,
  signed span repair, equivalent-span invariance, protected-prefix corruption,
  exact baseline reuse, byte accounting and preallocation bounds.
- The nonlinear evaluation/acquisition/CLI integration passed **185 tests**,
  covering lazy loading, no refitting, original BF16 endpoint bytes, projection
  roles, source/configuration hashes, finite metrics and atomic failure handling.
  These suites overlap earlier checks; the counts are not additive.

The last authenticated billing receipt, at 00:59 EDT on 2026-10-03, reports
$26.73840081 metered usage, $0 paid, and $3.30 credit remaining. A later
read-only refresh failed because the Modal token expired. Subsequent hybrid
and nonlinear evaluation ran locally; no new cloud compute was started.
Billing remains provisional.

The [capture trial](GLM_CAPTURE_TRIAL.md) contains implementation details,
reproduction commands, cloud receipts and the earlier failed-run history.

A separate [regional response-transport follow-up](REGIONAL_GLM_RESPONSE_TRANSPORT.md)
now compresses all three sampled projections with the same INT4 base and adds
a small calibrated output-error adapter. Shared region maps reduce its held-out
output error by 20.39% and retain 45.35% sample storage savings versus original
FP8/scales. Its baseline and changed projections differ from the gate-only
comparisons above; their error values are not interchangeable. The
[Antigravity audit](ANTIGRAVITY_COMPRESSION_AUDIT.md) distinguishes the original
additive/angular prototypes from this new measured test and corrects their
storage accounting. Both follow-ups ran locally.
