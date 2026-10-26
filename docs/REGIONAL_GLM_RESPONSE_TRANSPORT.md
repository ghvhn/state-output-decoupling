# Regional GLM response transport

The first local test supports a small, context-dependent reuse mechanism. A
compact map learned from seven sampled experts improves the eighth expert's
quantization-error correction slightly, and assigning the maps to the wrong
regions makes the aggregate error substantially worse. These are numerical
response results, not certified semantic analogies or a runnable full GLM model.

The subsequent [shell-law experiment](SHELL_GENERATED_RESPONSE_LAWS.md) learns
a formula that generates a shared map from continuous radius. It withholds one
ordered level's response targets from every fit and compares against constant,
linear, reflected and compact table controls. The formula improves withheld-level
prediction over a constant map; a table still performs slightly better at
nearly the same storage.

## Why this follows the simile idea

Antigravity's original prototypes share numerical components inside weight
blocks. The [audit](ANTIGRAVITY_COMPRESSION_AUDIT.md) records the original requests,
implementation and measured errors, with corrected storage accounting.

Here the reusable component is a transformation of a response, conditioned on
the current activation neighborhood. Every sampled expert keeps its own small
global response-error basis. A shared region operator changes how that basis is
used. This tests whether related changes can be reused across different expert
functions without storing a separate full local map for every expert/region.

It extends the earlier source-dependent [definition conversion](RELATIONAL_DEFINITION_CONVERSIONS.md)
idea to a measurable compression error. Those earlier generated definitions
remained unqualified; their existence is not evidence of successful analogy.

Regions in this test are calibration-only clusters in activation coordinates.
They have no assigned meanings. The same region labels across training and
evaluation are an interpolation test; transfer to an entirely unseen semantic
region was not tested. Expert 7 is excluded only from the shared-operator fit:
its global basis and regional offsets are still calibrated from its responses.

## Completed native-data experiment

- Original teacher: `zai-org/GLM-5.3`, revision
  `aca966e4e02791568aa6a4ced368624b3d897f42`.
- Layer 3, routed experts 0 through 7, original gate/up/down matrices and their
  FP8 scales. All original sample checksums were verified. Original payload:
  **302,063,616 bytes**, for three projections across all eight experts.
- Measured native normalized expert inputs: **1,022 calibration states in 32
  trajectories**, and **2,432 held-out states in 76 trajectories**, with **1,216
  measured adjacent pairs**. Record, trajectory and conversation splits remain
  disjoint. Recorded replies were teacher-forced, not newly generated GLM text.
- Base: the same saved symmetric INT4 control for all three projections,
  block size 128, FP32 scales. Actual serialized base storage:
  **160,443,759 bytes**. This is simple rounding, not GPTQ/AWQ or a best available
  quantized checkpoint comparison.
- Calibration-only rank-16 input PCA, four geometric clusters, fixed ridge
  `0.01`, seed `0`. Cluster counts were **9, 10, 523, 480**. The two clusters
  with fewer than rank+2 examples were retained but received no fitted local
  operator. **14 held-out states** used a global fallback because of support
  or disabled-region checks. No examples were discarded.
- Fits weight records uniformly. Evaluation uses the existing trajectory- and
  feature-balanced relative-RMSE metric. Provenance and stimulation weights are
  all one. Correct, incorrect and unknown predictions remain valid records.
- Successful run finished in **20.859 seconds**, on the CPU with six threads.
  Its declared numerical array estimate is recorded in `settings.json`, under
  a 4 GiB budget; this excludes source mappings, allocator and BLAS overhead.
  An initial run declined to fit the two sparse regions. The revised guard
  preserves those observations and uses the global fallback instead.

## Actual comparison

All errors are relative RMSE. Storage includes the three saved INT4 projections
and the separately saved adapter's tensors and final metadata. MB is decimal.

| Candidate | Output error | Output-difference error | Expert 7 output error | Stored MB |
|---|---:|---:|---:|---:|
| All three projections INT4 | 0.110701 | 0.127240 | 0.157804 | 160.444 |
| Global residual correction | 0.090113 | 0.119298 | 0.129464 | 164.280 |
| Regional offsets, global slope | 0.090144 | 0.119216 | 0.129504 | 165.068 |
| Shared regional transport | **0.088125** | **0.116969** | **0.128638** | **165.073** |
| Identity-operator control | 0.090144 | 0.119216 | 0.129504 | 165.073 |
| Wrong-region operator control | 0.146564 | 0.152994 | 0.159028 | 165.073 |
| Independent local maps | 0.086044 | 0.114606 | 0.125420 | 177.651 |

The shared adapter stores **4,551,012 raw tensor bytes**, with
**4,628,793 actual serialized bytes**. Its four 16-by-16 operators occupy only
**4,096 raw bytes**; input coordinates, global response bases and offsets make
up the rest. Identity and wrong-region controls have the same tensor shapes and
payload bytes; their different metadata labels add two or six serialized bytes.

The shared correction reduces held-out output error by about **20.39%** versus
the identical all-INT4 base, at about **2.88% more serialized storage**. Including
the adapter, the sampled block remains **45.35% smaller** than original FP8/scales.
These are sample counts and sample storage savings, not a whole-model ratio.

The shared operator improves aggregate output error by about **2.24%** versus
the identity/offset control, and expert 7's output error by about **0.67%**. This
is limited transfer of a calibrated numerical correction across expert
functions. It does not establish a reusable meaning or a causal language skill.

Independent local maps have the lowest error but cost another **12.58 MB**.
Wrong-region maps cause error to rise above the all-INT4 baseline, demonstrating
that the association between the map and its measured neighborhood matters.
The wrong-region control cyclically rebinds the four operators, including the
two identity fallbacks; offsets and geometric region selection stay fixed.

Actual predicted endpoint differences are evaluated after each endpoint selects
its region. There are **509 within-region pairs** and **707 crossing pairs**.
Shared maps reduce both types of difference error versus the INT4 baseline;
wrong-region maps especially worsen crossing-pair error. This checks one
consequence of hard region boundaries without proving smooth behavior everywhere.

## Mechanism and scope

For calibrated input coordinates `z`, an expert's global correction is
`mean_e + z @ B_e`. Region `k` adds its calibrated offset and
`(z - center_k) @ (T_k - I) @ B_e`. All experts reuse `T_k`; it is fitted from
experts 0–6 only. Independent local maps replace that shared operator with a
separate slope correction for each expert/region. Outside the declared support
radii, every regional method falls back to its global correction.

The outputs are original dequantized FP32 expert responses minus the responses
of the saved all-projection INT4 codec. Reconstruction and correction are
evaluated through `down(SiLU(gate(x)) * up(x))`. Source context hashes bind
the exact three original projections, architecture and saved INT4 artifacts.
Adapters are saved, restricted-loaded, checked byte-for-byte, and evaluated
after that round trip. They do not change original checkpoints or live shell
settings.

This prototype does not reproduce native dynamic-FP8 activation rounding,
router mixtures, shared experts, residual connections, later layers or a
complete decoder. Every expert sees every input with equal expert weighting,
including counterfactual activations. It evaluates sampled CPU functions; an
output correction would require a new runtime integration to become a model
replacement. Fewer stored bytes do not establish faster inference.

This held-out set had already been inspected in earlier compression work.
It is a development comparison. Fresh conversations and explicitly designed
cross-region relation tasks, followed by frozen-decoder logit and generation
checks, remain necessary for a broad claim.

## Reproduce

Use a new output directory:

```text
.venv\Scripts\python.exe scripts\region_transport_glm.py --out invariants\out\glm_capture_trial_20261002\new_region_run --threads 6
```

The completed settings, source receipts, evidence, saved codecs/adapters,
all seven candidate results and status are in
`invariants/out/glm_capture_trial_20261002/region_transport_20261003/`.
The reviewed figure is in the neighboring `region_transport_figure_final/` directory.

## Verification

On 2026-10-03, the regional, nonlinear GLM evaluation, capture calibration and
both angular-codec suites passed **192 tests** in 8.82 seconds. The regional
tests cover expert exclusion, incorrect bindings, exact endpoint differences,
split and scope checks, support fallbacks, retained predictions and serialization.
The angular tests also check lossless seven-bit storage, custom grids, padding,
bounded decoding, corruption and exact final byte totals. These regression
checks complement the native-data run; they do not establish decoder quality.

```text
.venv\Scripts\python.exe -m pytest tests/test_region_response_transport.py tests/test_glm_moe_sample_eval.py tests/test_calibrate_glm_compression.py tests/test_angular_frame_packing.py tests/test_angular_frame_quant.py -q -p no:cacheprovider --basetemp invariants/out/new_compression_regression
```
