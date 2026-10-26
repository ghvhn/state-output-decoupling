# Formula-generated changes across geometric levels

This prototype fits a compact family of affine changes to quantized GLM expert
responses. A continuous radial coordinate generates the map; discrete ordered
bands provide a response holdout test. These are measured geometric levels,
without assigned semantic meanings or electron-shell physics.

The completed local comparison supports prediction across a withheld response
level: a quadratic formula reduced that level's output error by **3.06%** versus
a constant map. Reversing the radial ordering worsened error. Separately stored
level maps remained slightly more accurate at nearly identical storage, so this
experiment establishes no material compression advantage for the formula.

The follow-up [rotation-related response laws](ROTATION_RELATED_RESPONSE_LAWS.md)
addresses rotation between directions explicitly and records accumulated path
travel separately from periodic angular maps.

## Completed native-data comparison

On 2026-10-03, the CPU run completed in **23.594 seconds** with six threads.
It used the same pinned `zai-org/GLM-5.3` layer-3 original gate/up/down weights
and saved INT4 control as the [regional experiment](REGIONAL_GLM_RESPONSE_TRANSPORT.md).
The source revision is `aca966e4e02791568aa6a4ced368624b3d897f42`.

- **1,022 calibration inputs**, in 32 complete trajectories, defined geometry.
  **251 band-1 response records** were excluded from every response fit;
  **771 response records** remained. Sparse regions retained their records but
  supplied no local maps.
- Region counts were **9, 10, 523, 480**; only the latter two had sufficient
  permitted-response support. Their quadratic design and aggregate response
  bases had full measured ranks of 51 and 16, respectively.
- **2,432 held-out states**, in 76 disjoint trajectories, supplied **1,216
  actual adjacent pairs**. **638 states** occupied the withheld level, all within
  supported regions. **547 pairs** touched that level, **456** entered/left it,
  and **91** had both endpoints in it. **51 states** elsewhere used the global
  fallback under the common support policy.
- All three original projection payloads total **302,063,616 bytes**. The
  unchanged saved INT4 control totals **160,443,759 actual serialized bytes**.
  No further GPU or cloud compute was used.

All errors below are relative RMSE. The last column is the actual serialized
adapter size, in decimal MB; the fixed INT4 base is common to every candidate.

| Candidate | All-state error | Withheld-level error | Output-difference error | Expert 7 withheld-level error | Adapter MB |
|---|---:|---:|---:|---:|---:|
| INT4 base | 0.110701 | 0.112790 | 0.127240 | 0.159496 | 0 |
| Global correction | 0.091283 | 0.095229 | 0.120225 | 0.136534 | 3.927 |
| Regional offsets | 0.091345 | 0.095398 | 0.120200 | 0.136715 | 4.715 |
| Constant map | 0.089308 | 0.092850 | 0.118088 | 0.136039 | 4.729 |
| Linear formula | 0.087418 | 0.090717 | 0.116892 | 0.135316 | 4.729 |
| Quadratic formula | **0.086780** | **0.090009** | **0.116392** | **0.135046** | **4.729** |
| Identity control | 0.091345 | 0.095398 | 0.120200 | 0.136715 | 4.729 |
| Reflected-level control | 0.143708 | 0.167048 | 0.162868 | 0.163609 | 4.729 |
| Separate level maps | 0.085641 | 0.088776 | 0.115510 | 0.134507 | 4.729 |

The quadratic reduces full-set error by **2.83%** versus the constant map and
**21.61%** versus the INT4 base. Its withheld-level error improves **3.06%**
over the constant map and **20.20%** over INT4. It also improves withheld-level
expert-7 error by **0.73%** versus the constant map; that expert remains partially
calibrated outside the withheld level. A linear formula supplies most of the
improvement; quadratic error is another **0.78% lower** on the withheld level.

Actual endpoint-difference errors for pairs touching the withheld level are
0.122779 for constant, 0.120665 for quadratic and 0.119777 for separate maps.
For entering/leaving pairs they are 0.123287, 0.121283 and 0.120398. Thus the
formula helps both point outputs and measured transitions, while the table
remains slightly better. Reflection raises withheld-level error to 0.167048,
above even the base's 0.112790, despite identical geometric support and payload.

The quadratic adapter occupies **4,728,667 actual bytes** and the compact table
**4,728,973 bytes**, a difference of only **306 bytes**. Both have **13,056 raw
coefficient bytes** at these settings. The complete sampled INT4-plus-quadratic
representation occupies **165,172,426 bytes**, **45.32% below** the sampled
original FP8/scales payload and **2.95% above** its INT4 base. This sample ratio
does not establish whole-model feasibility; most adapter storage is in global
bases and offsets, rather than the generating law.

Immutable settings, evidence, source/control receipts, fit diagnostics, saved
adapters and results are in
`invariants/out/glm_capture_trial_20261002/shell_transport_20261003/`.
The [reviewed figure](../invariants/out/glm_capture_trial_20261002/shell_transport_figure_20261003/comparison.png)
and its SVG/receipt are in the neighboring figure directory.

## What the formula changes

For normalized calibration PCA coordinates `z`, region `k` has center `c_k`
and maximum calibration radius `R_k`. The local direction is `v = z - c_k`
and the continuous coordinate is `s = norm(v) / R_k`. Within-region calibration
radial quantiles define four ordered bands. The formula uses the continuous
coordinate, rather than fitting a polynomial through three band-average maps.

Each expert retains its global response-error basis `B_e`, affine intercept
`m_e` and regional offset `b_e,k`. Its correction is:

```text
predicted_error_e(z) = m_e + z B_e + b_e,k
                    + [v, 1] (H_k,0 + s H_k,1 + s² H_k,2) B_e
```

The final row in each `H` generates an affine shift. Other rows transform the
full local direction, so equal radius does not imply equal response. The shared
`H` matrices are learned across experts 0 through 6; expert 7 evaluates partial
reuse of the law through its independently calibrated global basis and offset.
This correction acts on sampled response errors. It does not change native
hidden states, model weights or a decoder's generated text.

The shared fit minimizes the summed response error of the included experts
with a scaled Frobenius ridge penalty. Its two Gram eigensystems solve
`G_design H G_basis + lambda H = cross` directly in FP64, avoiding a large
Kronecker allocation. Degree, ridge and shell exclusion are fixed before
evaluating results. Effective design ranks are recorded; a regularized solution
does not imply a unique or physically meaningful law.

## Response holdout and controls

Band 1, the second of four ordered bands, supplies no response targets to the
global intercept/basis, offsets, polynomial or separately fitted maps. Its
calibration inputs still define PCA geometry and band boundaries. This is an
unseen-response interpolation test, not wholly unseen input geometry. Separate
held-out conversations and complete trajectories supply the evaluation inputs.
Expert 7's responses from the other bands may fit its own basis and offsets,
but cannot fit the shared operators.

The candidates are the unchanged all-INT4 base, global correction, regional
offsets, constant affine map, linear formula, quadratic formula, identity,
reflected radial coordinate, and separately stored level maps. Constant,
linear, quadratic, identity and reflected controls retain matching coefficient
tensor shapes. Lower-degree methods have zero higher-degree terms; this padding
is an explicit byte-matching control, rather than their minimal encoding.

The separately fitted table selects the closest fitted ordered level for the
withheld band, choosing the lower level on ties. No withheld response determines
that choice. Its saved tensors contain only the three fitted levels, omitting
the unfitted band. At four levels and degree two, it has the same coefficient
count as the polynomial; extra table flags and metadata are measured. Generating
today's small maps is a behavioral test, without a substantial storage claim.
All regional methods share one support mask. Each active region
must have enough permitted records for the quadratic design and enough records
in every non-withheld band for its independent table map. Outside the calibrated
radial interval, or beyond the orthogonal support radius, every regional method
uses the global correction. Reflection uses the same calibrated interval.

Every correct, incorrect and unknown prediction remains a valid input record.
The first diagnostic requires uniform provenance and stimulation weights.
Response fits weight permitted records uniformly. Evaluation conditions the
existing full-split trajectory/feature weights on each subset; it does not
reassign one vote per trajectory after selecting a band. Empty subsets have
unavailable error, rather than zero error.

## Evaluation and reproduction

The runner reuses the previously saved all-three-projection INT4 block-128
control. It verifies the exact gate/up/down source receipts, pinned architecture,
codec context and physical files. It measures original dequantized FP32 CPU
expert responses and saved INT4 responses before fitting new corrections.
Adapters are evaluated after an exact restricted tensor save/load round trip.
Byte totals include final metadata and actual saved files.

Metrics include whole-set and withheld-band point output errors, expert 7,
actual next-minus-previous output differences, pairs touching the withheld band,
pairs entering/leaving it, and pairs whose endpoints both occupy that band.
Each endpoint independently selects its region and continuous coordinate.

```text
.venv\Scripts\python.exe scripts\shell_transport_glm.py --out invariants\out\glm_capture_trial_20261002\new_shell_run --threads 6
```

The implementation is in `tda/shell_response_transport.py`, with the CPU runner
in `scripts/shell_transport_glm.py` and scientific renderer in
`scripts/plot_shell_transport_glm.py`. Original checkpoints, shell settings,
Antigravity prototypes and earlier candidate artifacts are preserved.

On 2026-10-03 the shell-law, runner, regional, nonlinear GLM, capture-calibration
and angular-codec suites passed **212 tests** in 7.16 seconds. New checks compare
the matrix solve with an independently constructed ridge objective, recover a
known continuous law on an excluded radial interval, and prove bitwise
invariance of fitted tensors when withheld responses change. They also check
expert exclusion, identical geometry/support, actual endpoint differences,
empty subsets, compact-table storage, overflow and exact save/load byte totals.

This development context set was inspected in previous compression work.
Router mixtures, shared experts, native FP8 activation arithmetic, later layers,
logits, generation, unseen semantic relations and inference speed remain outside
this diagnostic. A useful numerical law would require fresh confirmation and
runtime integration before it could become a model compression method.
