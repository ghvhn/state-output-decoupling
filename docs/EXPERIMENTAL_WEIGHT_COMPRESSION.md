# Experimental weight compression

The repository now has an experimental **weight** compression path in addition
to its existing activation PCA and trajectory storage. It exports scoped matrix
group factors and a simple block quantization control. These artifacts are not
drop-in Hugging Face checkpoints. The production interactive model loader has
not been changed.

The completed measurements and limitations are recorded in
`invariants/out/compression_runs/20261002_assessment.md`. The first real GLM
sample and complete small-decoder trials have finished; the full GLM export
remains unimplemented and unvalidated.

## Lessons used from our language experiments

The definition experiment in
`invariants/out/definition_runs/20261002_structure_aligned/assessment.md`
changed residual states while producing identical greedy text in several
conditions. It also reported source states outside calibration support. That
supports measuring multiple levels of preservation, and checking transfer
against measured source contexts. It does not prove that any particular
internal direction is an emotion, meaning, or causal explanation.

| Observation or prior decision | Compression consequence |
| --- | --- |
| Internal movement can leave chosen words unchanged | Measure full token distributions, observed-token likelihood and prediction margins; argmax agreement is insufficient |
| Relationships and transformations matter as well as positions | Fit both responses at measured states and responses to within-trajectory differences; compare with point-only and shuffled-successor controls |
| A measured coordinate is not a certified meaning | No probe name or positive/negative semantic interpretation is a compression target |
| Every correct and incorrect prediction is valid scoped evidence | Keep correctness separate from provenance/stimulation weights; never discard a record because its token prediction was wrong |
| Learned relationships apply within their measured environment | Pin model revision, layer, projection, layout, matrix slot order and input site; reject cross-environment pairings and split leakage |
| A local improvement must survive language production | Add an optional correction through a complete decoder, against frozen measured teacher distributions and their changes |

These are design constraints and testable hypotheses. They do not certify a new
general theory of neural networks.

## Implemented paths

`tda/expert_compress.py` streams flattened matrices from one exact group in two
CPU passes. With expert/matrix-slot mean `M`, coefficient columns `U` and basis
matrices `B`, decoding is `W_hat[e] = M + U[e] @ B`. Weight PCA selects `U` from
the centered weight Gram matrix. FP32 fitting, FP16/BF16/FP32 factor storage,
source FP8 scale bytes, actual serialized bytes and dense reconstruction costs
are accounted separately. A memory budget is checked before reading weights.

`tda/trajectory_compress.py` selects the same-size `U` using centered expert
responses `W[e] x` and responses to measured differences
`W[e] (x_next - x_previous)`. Each record/pair is energy normalized and degree
balanced. A small optional weight-energy floor provides a control for directions
poorly represented by calibration. Exact environment and trajectory pair
scopes are enforced. Reversing both endpoints leaves this quadratic objective
unchanged; that is not a detector of temporal causality.

`tda/predictive_compress.py` optionally corrects the mean and basis through a
caller-supplied differentiable decoder. Coefficients, rank and storage dtype
stay fixed. The objective combines full-vocabulary teacher KL with changes in
centered token log odds over a teacher top-K union plus observed targets.
Wrong teacher predictions still contribute. Only calibration evidence selects
the checkpoint; explicit held-out record and trajectory IDs must be excluded.
Storage precision is restored and actual weight error is recomputed afterward.
The callback is an integration point, not a GLM inference implementation.

`tda/block_quant.py` is an explicit symmetric INT4/INT8 rounding control with
FP32 block scales. INT4 uses levels -7 through 7 and packed nibbles. It is not
NF4, GPTQ or AWQ. Container bytes, scales and padded tails are included.

All exports use new atomic directories, source scope checks, checksums,
restricted tensor loading and bounded storage/container validation. Source
weights are retained; no model checkpoint or live conversational cache is
modified by fitting.

## Bounded GLM-5.3 acquisition

The official checkpoint is pinned to
`aca966e4e02791568aa6a4ced368624b3d897f42`. The saved metadata inventories
141 shards, **755,632,050,320 bytes**, and 753,329,940,480 declared parameters.
The initial local Transformers 4.57.6 registry did not support `glm_moe_dsa`.
Configuration dispatch alone would not establish runtime compatibility either.

The downloaded sample contains layer 3 routed experts 0 through 7, `gate_proj`
only, each `[2048, 6144]`, plus FP32 `[16, 48]` scales. Original payload storage
is **100,687,872 bytes**. Downloaded bytes including bounded shard headers are
**101,005,096**. Exact HTTP 206 ranges, complete safetensors headers, pinned
shard lengths, tensor offsets/shapes/dtypes and payload checksums are checked.
The FP8 values are multiplied by their corresponding 128×128 block scales.

This is a real source-weight sample, not a runnable model. The later native
FP8/TP8 teacher capture completed all 410 contexts and 8,336 selected positions
on Modal, with original output-head reconstruction qualified locally. Recorded
replies were teacher-forced input, rather than newly generated GLM replies.
Native measured seed contrasts and scoped gate-input trajectories now support
actual compression comparisons. See [the capture trial](GLM_CAPTURE_TRIAL.md)
for receipts and complete-archive results. A compressed GLM decoder adapter and
full-model language validation remain unimplemented. Gate-only errors cannot assess
`down(SiLU(gate(x)) * up(x))`, routing, shared experts, attention or the decoder.

A subsequent local evaluation downloaded the unchanged up/down groups and
measured seven saved gate candidates through the sampled nonlinear expert
formula in CPU FP32. The hybrid roughly halves held-out normalized output
error versus INT4 and reduces fitted-seed nonlinear contrast RMSE by 95.99%.
Equal-rank PCA still has lower general output error. This adds nonlinear
sample evidence while leaving native dynamic-FP8 kernels, router mixtures and
the compressed decoder unvalidated. See [the measured results](GLM_COMPRESSION_RESULTS.md)
for actual byte counts, separate calibration/held-out metrics and reviewed figures.

## Research overlap and the hypothesis we are testing

Activation-aware low-rank compression already exists:
[SVD-LLM](https://arxiv.org/abs/2403.07378) uses truncation-aware whitening and
sequential updates. [IO-SVD](https://arxiv.org/abs/2605.15626) uses a KL-aware
input/output metric and loss-aware allocation. [AIR](https://arxiv.org/abs/2606.19993)
uses backward influence in low-rank corrections. These are relevant established
comparisons; predictive sensitivity alone is not our novelty claim.

[LorExperts/BTExperts](https://arxiv.org/abs/2608.07814) investigates functional
expert communities distinct from weight similarity, and retains experts with
local low-rank corrections. Expert relationships alone are not new either.

The extension here is a scoped experimental combination of measured trajectory
contrasts, validity independent of accuracy, provenance/stimulation weights,
and direct correction of predictive changes. Its value must be demonstrated
against weight-only, point-only, shuffled and quantization controls at reported
byte budgets. We have not established worldwide novelty or superiority.

## Reproduction

Metadata only:

```text
.venv\Scripts\python.exe scripts\glm_download_preflight.py --output invariants\out\glm_5_3_preflight
```

New bounded sample destination:

```text
.venv\Scripts\python.exe scripts\glm_weight_sample.py --metadata invariants\out\glm_5_3_preflight --output invariants\out\compression_runs\new_glm_sample --byte-cap 117440512
```

CPU sample comparison, always a new output directory:

```text
.venv\Scripts\python.exe scripts\expert_compression_test.py --sample invariants\out\compression_runs\20261002_glm_gate_sample --out invariants\out\compression_runs\new_glm_storage
```

Complete cached GPT-2 CPU language pilot:

```text
.venv\Scripts\python.exe scripts\compression_language_test.py --out invariants\out\compression_runs\new_language_trial --layer 5 --rank 2 --max-tokens 48 --threads 6 --predictive-correction
```

The GPT-2 pilot uses twelve **attention query heads**, not MoE experts. It
preserves their slot order, key/value weights and bias, changes one query
projection only in memory, evaluates a complete learned decoder, and restores
the original query weights. Its purpose-written training and evaluation texts
are separate trajectories. Iteration after inspecting the first evaluation is
exploratory; these short texts cannot establish broad language quality.

A sealed evaluation can load the already-fixed artifacts with no fitting:

```text
.venv\Scripts\python.exe scripts\compression_language_test.py --out invariants\out\compression_runs\new_sealed_evaluation --layer 5 --rank 2 --max-tokens 64 --threads 6 --predictive-correction --evaluation-fixture tests\fixtures\compression_language_sealed.json --reuse-artifacts invariants\out\compression_runs\20261002_gpt2_predictive_budgeted
```

## Before any complete GLM export

The [shell-generated response laws](SHELL_GENERATED_RESPONSE_LAWS.md) extend
the regional diagnostic with a continuous affine matrix polynomial in measured
radius. One level's responses are excluded from every fit. The formula improves
that level's output error by 3.06% over a constant map, while separately fitted
maps remain slightly better at nearly identical storage. This tests prediction
across a withheld response level; it supplies no material storage advantage or
semantic/full-decoder claim. All controls reuse the identical saved INT4 base.

The [regional response experiment](REGIONAL_GLM_RESPONSE_TRANSPORT.md) is a
separate sampled CPU path: all three projections use saved INT4 weights, and a
small source-bound adapter predicts the resulting nonlinear output error.
Calibration-only activation neighborhoods select shared 16-by-16 maps learned
from seven experts. Reuse improves the eighth expert's correction slightly;
incorrect region bindings worsen error. Actual storage and output/difference
metrics are reported together. The data supports conditional numerical reuse,
not certified semantic regions or a production compressed GLM decoder.

The [rotation-related comparison](ROTATION_RELATED_RESPONSE_LAWS.md) tests whether
rotating an input direction should rotate its local correction, with a response
sector withheld from every fit. It also records observed path travel beyond one
turn separately from static periodic maps. These are numerical experiments on
the saved samples, with no semantic-rotation or whole-model feasibility claim.

The original [Antigravity compression attempt](ANTIGRAVITY_COMPRESSION_AUDIT.md)
has also been checked against its code and original logs. Its additive scheme
reduced error with more block parameters; its historical storage claim omitted
those parameters. Seven-bit packing of the 125 angular states is implemented in
`tda/angular_frame_packing.py`: a native prefix test reduced codes plus unchanged
scales by 10%, with bitwise-identical reconstruction after save/load. Its actual
configuration and container costs are counted. This storage extension does not
improve the angular codebook's accuracy. Globally shared additive subvalues
would require a separate fit and accuracy test.

Retaining a full download plus even idealized 4-bit output would exceed the
previously measured local free-disk budget. Raw idealized 2-bit output also
exceeds local combined RAM and VRAM. Those floors omit scales, buffers and
runtime overhead. The full original checkpoint was instead staged on a cloud
Volume: all 147 files and 755,663,668,166 bytes were SHA256-verified, and the
native eight-H200 teacher capture completed. The local archive is about 2.83 GB;
the full checkpoint was not downloaded to the laptop.

A complete compressed export still needs a staged reader/writer plan, verified
reconstruction, GLM-compatible compressed execution and held-out
likelihood/logit/task checks. Keeping both full source and compressed output
would also need a renewed storage allowance check. Sample storage ratios cannot
be multiplied by the checkpoint size to establish feasibility. Full source
staging and original-teacher capture succeeded; a complete compressed GLM
checkpoint and its language quality have not been validated.
