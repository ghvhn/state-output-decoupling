# Native GLM capture trial

This trial preserves reusable evidence from the original GLM-5.3 FP8 teacher
for probe calibration and compression experiments. It does not export a
compressed GLM checkpoint or establish its language quality.

The pinned teacher is `zai-org/GLM-5.3` at
`aca966e4e02791568aa6a4ced368624b3d897f42`. Source weights occupy
755,632,050,320 bytes; staged native metadata adds 31,617,846 bytes. Downloads
happen inside a dedicated Modal Volume on CPUs. The full checkpoint stays off
the user's laptop. GPU loading time counts against the capture deadline.

## Evidence

`invariants/out/glm_capture_trial_20261002_priority` contains 410 contexts,
137,933 input tokens, 8,336 selected positions and 4,096 adjacent-position pairs.
There are 256 conversation/reference contexts and both text poles of 77 unique
existing probes, with 334 calibration contexts and 76 held-out contexts.
Cheap paired probe seeds are captured first, making an interrupted run useful
for native probe calibration before the split-balanced conversation contexts.
Conversation/source identities are split before measurements; probe seeds
belong only to calibration. Old activation vectors do not become GLM vectors.

Full conversation records supply the corpus. Operator readings, commands and
logs stay out of prompt content. Local review files preserve source text and
operator records separately. Tokens retain actual private content; hashes do
not anonymize it. Only the selected `corpus.json` and four allowlisted runtime
source files are uploaded, rather than full archives or the repository.

Recorded conversation replies supply teacher-forced input sequences to GLM;
some were originally produced by the previous local model. The measured states
are GLM's responses to those sequences. Observed next-token labels describe the
recorded corpus, rather than newly generated GLM replies. Their likelihood and
accuracy measure fit to that corpus, not free-generation quality.

The capture saves decoder outputs at layers 3, 12, 24, 36, 48, 60 and 77, the
exact layer-3 normalized gate/up input, router choices/weights and final
normalized readout states. First-batch logical tensors must agree across all
eight ranks. The shared gate/up input is not an expert's down-projection input;
the latter also depends on that expert's weights and nonlinearity.

The unchanged BF16 output head is saved in bounded CPU blocks. Each final
state has 6,144 components versus 154,880 vocabulary logits, making
equal-precision state storage about 25.2 times smaller per position. Saved
full-vocabulary anchors check local reconstruction. Floating-point rounding
can differ across matrix shapes/kernels, so bit-identical results are not
promised. Readout validation covers only the captured original teacher's final
projection. Modified-decoder quality and runtime savings require other checks.

Unobserved next tokens in probe framings carry explicit observation flags and
a `-1` sentinel. They calibrate geometric contrasts but do not enter accuracy
or likelihood scores. Incorrect observed predictions remain valid evidence,
preserving their trajectory, semantic environment and provenance.

## Budget

The user has $30 free credit and explicitly selected a $5 net spending backup.
The launcher defaults to $0 paid authorization. It permits the backup only
when recorded in the verified receipt, limits total trial usage to $35, and
reserves $2 before allocating compute. The provider's independent workspace
usage cap cannot enlarge this trial ceiling.

There is one eight-H200 capture container, no warm pool or elastic retries,
hard CPU/RAM limits, a short idle window and a bounded process. Loading,
capture and saving share a deadline. The worker stops when the corpus finishes
or its deadline/byte budget is reached. Completed chunks remain readable after
an interruption. Loading failure does not trigger another GPU run.

The checkpoint and capture must remain inside verified free Volume storage.
Storage can accrue charges after compute stops if that allowance is exceeded;
a compute spend limit alone does not bound storage. The receipt distinguishes
operator credit/spend-limit evidence from SDK billing/workspace/inventory
checks. Starter's published GPU concurrency is a default, not a personalized
measurement; allocation remains subject to Modal's server-side eligibility.

Rates and limit semantics checked on 2026-10-02:
[Modal pricing](https://modal.com/pricing),
[budgets](https://modal.com/docs/guide/budgets),
[weight staging](https://modal.com/docs/guide/model-weights).

## Tools

| Tool | Purpose |
| --- | --- |
| `scripts/prepare_glm_capture.py` | Offline pinned tokenization and sealed splits |
| `scripts/modal_glm_capture.py` | Offline plan, CPU build/staging, bounded capture |
| `tda/glm_capture_runtime.py` | Native FP8/TP8 teacher and selected-state hooks |
| `tda/glm_capture_archive.py` | Atomic bounded evidence and checked tensor loading |
| `scripts/fetch_glm_capture.py` | Bounded artifact download from one cloud capture |
| `tda/glm_capture_calibration.py` | Native probe axes and scoped gate/up calibration records |
| `scripts/reconstruct_glm_capture.py` | CPU verification and full-vocabulary readout export |
| `scripts/calibrate_glm_compression.py` | Bounded native PCA/trajectory/probe-span/INT4 comparison and probe residual repair |

An isolated client lives under `invariants/out/modal_client`; the project's
environment is unchanged. The cloud image uses Python 3.12, PyTorch
2.11.0+cu128, Transformers 5.15.0, accelerate 1.15.0, kernels **0.16.0**, Hub
1.33.0 and safetensors 0.8.0. A cloud CPU check caught that kernels 0.17 is
incompatible with Transformers 5.15's FP8 guard. After correcting the version,
the cloud CPU check passed and warmed the native FP8 kernel cache. A subsequent
source audit caught that native TP8 stores vocabulary-row shards of the output
head, while its forward gathers full logits. Validation now checks the actual
`colwise_gather_output` plan and local BF16 shard. A fresh cloud CPU receipt
passed after that correction. These results
do not establish successful GLM GPU loading or forwards.

Default offline plan:

```powershell
& .\invariants\out\modal_client\Scripts\python.exe scripts\modal_glm_capture.py --corpus invariants/out/glm_capture_trial_20261002_priority/corpus.json
```

Cloud modes use `--mode build`, `--mode stage`, then `--mode capture`, together
with `--workspace gavinthepowell --budget-evidence <fresh_receipt.json>`.
Capture also requires the selected corpus and matching build/stage receipts.
Default invocation allocates no cloud resources.

After downloading the archive:

```powershell
& .\.venv\Scripts\python.exe scripts\reconstruct_glm_capture.py <archive_directory>
& .\.venv\Scripts\python.exe scripts\reconstruct_glm_capture.py <archive_directory> --out <new_readout_directory>
```

After obtaining the original capture archive, the complete thirteen-variant
comparison uses a new output directory:

```powershell
& .\.venv\Scripts\python.exe scripts/calibrate_glm_compression.py --sample invariants/out/compression_runs/20261002_glm_gate_sample --archive <archive_directory> --out <new_comparison_directory> --ranks "0,1,2" --max-train 1024 --max-heldout 2432 --threads 6 --budget 1073741824 --row-chunk 16 --probe-span --probe-residual --probe-residual-controls --residual-random-seed 0
```

Use the fetched original archive, rather than the head-only preflight or the
reconstructed-logit directory. Partial captures need measured calibration and
held-out trajectories and sufficient calibration rank for the matched PCA
control; missing evidence stops the comparison.

## Validation

The capture/calibration/readout suites initially passed 164 tests. Two added
regressions validate native output-head shards and reject wrong axes or plans;
the updated runtime/launcher suites passed 69 tests. Tests cover provenance,
split leakage, exact measurement sites, missing labels,
router pairing, deadlines, corruption, memory/storage bounds and interrupted
writes. The eight existing compression suites passed 276 tests after integration.

A complete cached GPT-2 CPU experiment captured actual normalized states and
reconstructed all 50,257 vocabulary logits. Two anchors had maximum error
0.0000534058, mean error 0.00000505404 and 100% argmax agreement. Float32
tolerances were 0.0001 absolute and 0.00001 relative. The stricter bit-exact
check failed and is retained. All 17 positions, including incorrect
predictions, were exported. This validates a learned model's archive/readout
path; it does not validate GLM or a compressed model.

The local result is in
`invariants/out/glm_capture_readout_cpu_validation/validation_result.json`.
Cloud completion receipts are kept beside the trial plan. The baseline GLM
GPU capture now has a completion receipt below; a compressed decoder's language
quality requires its own validation.

A cloud CPU benchmark exported the actual unchanged 1,903,165,440-byte BF16
head from its verified first shard in 13.302 seconds; Volume commit added
3.635 seconds. This fits the 45-second save reserve in that measured CPU
environment. The separate head-only fallback is preserved under
`/head-preflight/<revision>/archive`. The receipt is
`invariants/out/glm_capture_trial_20261002/head_export_benchmark.json`;
it contains no decoder-state or GPU-forward claim.

All 147 native checkpoint files completed staging with verified SHA256 hashes,
totaling 755,663,668,166 bytes. The first GPU attempt reached process startup
but stalled before model loading: PyTorch's standalone rendezvous advertised
the container hostname, which did not resolve with external networking blocked.
The app was stopped to bound its cost. The launcher now uses a static literal
loopback address and local Gloo/NCCL bootstrap interfaces. The CPU build also
requires a real eight-rank Gloo reduction before recording a passing receipt.
The updated launcher/runtime tests passed 71 checks. A fresh CPU receipt was
obtained before the next GPU call; the failed attempt is retained separately.

The next attempt passed process startup but failed before model loading because
kernel version metadata was absent offline. CPU preflight now retains the
official `v4` reference and pins the build to commit
`8c178950e8710e26a2210e4e909cd02dc16c8715`. A fresh offline CPU check additionally
caught that Hub 1.33 demands every build variant during kernels' initial
snapshot lookup. The supported `LOCAL_KERNELS` loader selects the same verified
native build from the pinned snapshot, with normal dependency validation.
Preflight compares online/offline artifact hashes, checks module-path containment
and rejects changed commits. It records offline success before GPU admission.
The first two failed GPU attempts have separate receipts; neither captured decoder states.

The third attempt passed offline kernel checks and process startup, then failed
before weight loading because Transformers 5.15 no longer accepts `tp_plan` as
a top-level loading argument. The loader now uses
`transformers.distributed.DistributedConfig(tp_size=8)`. A new eight-rank CPU
proof saves and reloads a tiny, unquantized GLM with this same loading path and
an explicit CPU device mesh. Its native sharded BF16 output head produces finite,
gathered full-vocabulary scores that agree across ranks. The build receipt
`invariants/out/glm_capture_trial_20261002/build_result_native_loading.json`
records these passing checks and the source fingerprint. This is a loading and
head-forward proof; it does not validate the decoder's native FP8 GPU execution.
The third failure is retained in `capture_attempt_3_result.json`. The updated
runtime and launcher suites passed 76 tests.

A fourth allocation encountered Modal's `failed to get new inputs` error
before worker execution. It was stopped after repeated container failures.
A cheap CPU-only function then received both a tiny input and the exact
1,158,609-byte authorized corpus with its original 147-file metadata request,
under the same blocked-network policy. The successful delivery receipt is
`input_delivery_result.json`; the stopped allocation is recorded separately
in `capture_attempt_4_failure.json`. This does not validate GPU execution.

The fifth allocation completed the native FP8/TP8 capture on 2026-10-03.
Run `20261003T042222-51abd806ab79` saved all 8,336 selected positions in 205
chunks across all 410 contexts. The first batch passed logical-tensor agreement
across eight ranks and retained two full-vocabulary anchors. The native output
head was saved, and the GPU app stopped with about 663 seconds left in its
execution deadline. The completion receipt is `capture_result.json` and the
app is [the completed GPU run](https://modal.com/apps/gavinthepowell/main/ap-5HKQ23ELlIoSz9C6rHEZVn).
The first post-run billing refresh reported $26.73840081 metered usage and zero
paid charges; reporting is provisional. Local readout qualification and actual
compression comparisons are separate from this successful baseline capture.

The downloaded archive contains 357 checksum-verified artifacts totaling
2,827,846,647 bytes. CPU reconstruction passed its two full-vocabulary anchors
at absolute tolerance 0.03125 and relative tolerance 0.01, with maximum error
0.03125, mean error 0.0000108651 and 100% argmax agreement. All 8,336 positions
were exported in native BF16, totaling 2,582,653,897 serialized bytes. There
are 8,182 observed next-token targets and 154 unobserved probe-prefix positions;
all predictions are retained. The complete local report is
`invariants/out/glm_capture_trial_20261002/readout/report.json`.

`native_probe_bank(archive, tensor_name=...)` fits empirical contrast axes only
from paired calibration seed poles. It records the exact teacher and state-site
fingerprint, diagnostics for incomplete/zero contrasts and unvalidated semantic
meanings. `gate_calibration_records(archive, expert_identity, split=...)` returns
records and within-trajectory pairs accepted by the existing trajectory fitter.
Calibration and held-out sets are selected separately. Shared gate/up inputs
are explicitly rejected for down projections, and missing prediction correctness
is retained as unknown rather than inferred from another model's responses.

`fit_probe_span_group` separately protects the orthonormal span of paired native
seed contrasts with one whole-family response normalization. Sign flips,
equivalent basis rotations and duplicated axes cannot add objective mass.
Probe evidence retains endpoint IDs, raw contrast magnitudes and actual
capture checksums. It remains a static, dequantized linear projection proxy;
the endpoint comparisons are not temporal transitions.

The CPU comparison launcher optionally enables it with `--probe-span`, next to
equal-rank PCA/trajectory controls and INT4. It selects whole trajectories,
holds evaluation examples out of fitting, and reports actual serialized bytes,
separate static-seed and temporal response errors, and native routing coverage.
Explicit token gaps distinguish state differences from per-token velocity.
These projection metrics do not certify a modified nonlinear decoder or language.

`--probe-residual` adds a second geometry-based candidate: repair the identical
saved INT4 baseline only along the orthonormal native probe span. If that span
has basis `Q`, the candidate is `Wq + (W - Wq) Q Q.T`. Shared input factors and
per-expert FP32 correction coefficients are serialized and counted in its size.
The correction rank is independent of the shared-expert ranks. It protects
signed linear responses and their mixtures in the measured span; it does not
claim that seed meanings are validated or that the full decoder is preserved.
Uncorrected INT4 and repaired INT4 receive the same static-seed and held-out
response evaluations. Both optional candidates can be enabled together with
`--probe-span --probe-residual`.

`--probe-residual-controls` additionally compares calibration-only uncentered
PCA input directions and a reproducible Gaussian/QR random span. All three
repairs use the identical INT4 codes/scales, FP32 coefficients and the native
probe-span rank. Tensor payload bytes must match; actual serialized metadata
differences are reported. A PCA span with insufficient numerical rank is
rejected explicitly. These controls test whether selecting probe directions
helps beyond storing more coefficients. They carry distinct origins and never
present random or PCA directions as measured probe seeds.

The updated comparison, native calibration and probe-span suites passed 68
tests, including exact INT4 baseline reuse, correction artifact accounting,
signed span response restoration, unchanged off-span values, held-out split
separation and explicit nonunit token gaps.

The broader residual/span/native-calibration/quantization/compression/trajectory
and CLI run passed 268 tests after adding the controls. Low-rank quantization repair and activation-aware
calibration have established precedents in [LoftQ](https://arxiv.org/abs/2310.08659),
[LQ-LoRA](https://arxiv.org/abs/2311.12023) and
[QERA](https://arxiv.org/abs/2410.06040). Here the experimental choice is a fixed
span selected from scoped native seed contrasts, compared with separate genuine
trajectory and held-out evidence. Seed-span improvement is expected by
construction; held-out improvement at comparable total bytes requires measurement.

## Complete-archive comparison

The actual thirteen-variant comparison is saved under
`invariants/out/glm_capture_trial_20261002/comparison`. The 1,024-position
calibration budget selected 1,022 states from 32 whole trajectories, with 512
genuine adjacent pairs. All 2,432 held-out conversation states from 76
trajectories were evaluated, including 1,216 adjacent pairs. The remaining
calibration trajectories were not used by this bounded trial. Separately,
the 154 measured seed endpoints supplied a 77-dimensional native contrast span.
Selected calibration trajectories cover 18 conversation/source identities;
held-out trajectories cover 32. Record, trajectory, conversation and semantic
environment identifiers do not overlap between these selected splits.
All measured pair token gaps in this run are one.

| Candidate | Serialized bytes | Held-out state RMSE | Held-out state-difference RMSE | Static native-span RMSE |
| --- | ---: | ---: | ---: | ---: |
| INT4 | 53,481,253 | 0.093944 | 0.088268 | 0.088483 |
| INT4 + native probe repair, rank 77 | 60,447,394 | 0.086430 | 0.085727 | 0.0000002993 |
| INT4 + calibration PCA repair, rank 77 | 60,693,154 | 0.047626 | 0.064012 | 0.084828 |
| INT4 + random repair, rank 77 | 60,426,402 | 0.093470 | 0.087715 | 0.087935 |

All errors in this table are relative RMSE. State and state-difference errors
use the existing trajectory/provenance/stimulation weighting; static span
error uses one whole-span Frobenius normalization. A state difference is not
per-token velocity. The three rank-77 repairs have identical raw tensor bytes;
serialized sizes differ because their evidence metadata differs. Original
FP8 weights and block scales for these eight gate projections occupy
100,687,872 bytes. These sizes exclude the separately stored seed witness bank.
Provenance and stimulation weights are all one in this selected dataset.
Each trajectory receives total point mass one and total pair mass one;
this does not equalize conversations or source categories. First-occurrence
selection is deterministic and is not a randomized sample. No uncertainty
interval or population-wide statistical claim is supplied.

Probe repair restores the measured span to numerical rounding, as expected
from its construction, and modestly improves held-out responses. Calibration
PCA is substantially better on broader held-out responses, while failing to
protect the designated span to the same extent. The much smaller shared-expert
rank-0/1/2 candidates have substantially larger response errors in this trial.
This result motivates testing a hybrid with explicit protected-span constraints
and equal-rank, equal-tensor-byte controls.

The expanded CLI accepts up to 8,192 records and 4,096 adjacent pairs per split,
with bounded tensor allocation checks before fitting or state copying. The
updated CLI, residual and trajectory suites passed 134 tests, including a
weighted evaluation with 2,432 states. The numerical tensor budget excludes
allocator and BLAS workspace overhead; it is not a process-memory guarantee.

These are counterfactual linear responses of eight equally weighted experts
at one layer. Actual native routing coverage is reported separately. The trial
does not fit with router weights, reproduce native dynamic FP8 activation
rounding, evaluate the nonlinear expert block, or validate a compressed decoder.
The held-out states were never used to fit these candidates. Using this held-out
result to select a subsequent method makes it a development comparison; an
independent test corpus is required for an unbiased final quality claim.
The sampled experts appeared at 228/1,022 calibration positions and
512/2,432 held-out positions. Together they carried about 3.22% of actual
scaled router weight in each split. Evaluating all eight matrices at every
state therefore covers many counterfactual responses and does not estimate
the error of the actually routed block.

## Protected-span hybrid

`--probe-hybrid --hybrid-extra-rank 77` extends the comparison with three
rank-154 INT4 repairs: native probes plus calibration complement PCA, plain
calibration PCA, and a seeded random span. It requires `--probe-residual` and
the identical saved INT4 baseline. Omitting `--hybrid-extra-rank` requests as
many extra directions as the native probe rank.

The hybrid preserves the original orthonormal native basis `Q` unchanged.
It projects selected calibration states to `X_perp = X - (X Q) Q.T`, fits
uncentered PCA there, reprojects and orthogonalizes only the extra directions
`R`, and repairs quantization error on `[Q, R]`. No dense hidden-by-hidden
projector is formed. Held-out states supply exclusion identities only. Extra
directions with insufficient numerical spectrum are rejected without padding.
The source-relative rank floor excludes projection roundoff from the usable
complement spectrum.

The plain-PCA and random controls use the same total correction rank, FP32
coefficients, INT4 codes/scales and raw tensor bytes. Their total serialized
sizes include their own distinct metadata. These rank-154 candidates require
more storage than the earlier rank-77 repairs; improvement versus rank 77
alone cannot establish a superior direction-selection method.

Native span and calibration evidence retain separate fingerprints and the
exact teacher/input site. Generic fitting and serialization verify the native
prefix checksum, protected/extra ranks and complementary spectrum bounds.
The combined basis has a distinct origin; PCA directions are never labeled as
measured probe seed axes. The seven related suites passed 325 tests, including
held-out nonuse, signed repair, equivalent-span invariance, exact INT4 reuse,
equal-rank payload accounting and corrupted scope/prefix rejection.

The full sixteen-variant native comparison completed in 138.25 seconds under
`invariants/out/glm_capture_trial_20261002/hybrid_comparison`, with the same
1,022 calibration states, 2,432 held-out states and 77 native seed axes.

| Rank-154 candidate | Serialized bytes | Held-out state RMSE | Held-out state-difference RMSE | Static native-span RMSE |
| --- | ---: | ---: | ---: | ---: |
| INT4 + native span/complement PCA hybrid | 67,660,450 | 0.046511 | 0.062887 | 0.0000002992 |
| INT4 + calibration PCA | 67,631,778 | 0.042841 | 0.057733 | 0.083045 |
| INT4 + seeded random span | 67,365,026 | 0.092652 | 0.087132 | 0.087386 |

Each rank-154 repair has 67,354,624 raw tensor bytes, compared with
60,416,000 bytes at rank 77. The hybrid reduces held-out state RMSE by 50.49%
and state-difference RMSE by 28.75% versus INT4. Its serialized size is 32.80%
below the original FP8 sample and 26.51% above INT4. It preserves the fitted
native span to numerical rounding. Equal-rank PCA still has lower general
response errors, so this is a measurable constraint/storage tradeoff rather
than a general superiority result. Probe semantics remain unvalidated.

The audited figure is
`invariants/out/glm_capture_trial_20261002/hybrid_figure/comparison.png`, with a
matching SVG and `plot_manifest.json`. All 48 plotted metrics are finite;
the manifest records the complete source archive, actual selected counts and
two separately matched correction-rank groups.

## Sampled nonlinear validation

The original eight-expert up and down groups were separately downloaded from
the same pinned checkpoint with exact HTTP ranges and unchanged 128 MiB caps.
Each has 100,687,872 source payload bytes and 101,005,096 total network bytes
including headers. Up matrices are `[2048, 6144]` with `[16, 48]` scales;
down matrices are `[6144, 2048]` with `[48, 16]` scales. The existing source
reader now supports all three explicit projection types while retaining
default gate-manifest compatibility. Its 55 focused tests passed.

The local nonlinear reference is `down(SiLU(gate(x)) * up(x))`, using original
up/down weights and a candidate gate. This follows the mathematical expert
sequence in [the pinned Transformers implementation](https://raw.githubusercontent.com/huggingface/transformers/v5.15.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py)
and the saved configuration's `hidden_act="silu"`. CPU FP32 dequantized
matrices do not reproduce native dynamic-FP8 activation rounding. Routing,
shared experts, residual connections and subsequent decoder layers remain
outside this sampled comparison.

Actual measured seed poles are loaded from archive rows, retaining their
centers, identities and original state checksums. A nonlinear pole contrast
is `f(positive) - f(negative)`; it cannot be obtained by evaluating only the
linear difference direction. Likewise temporal output differences are
`f(next) - f(previous)`, rather than `f(next - previous)`.

The seven predeclared INT4 and repair candidates completed this evaluation in
68.125 seconds with no refitting, the same 1,022 calibration/2,432 held-out
states and all 154 raw seed endpoints. Results are in
`invariants/out/glm_capture_trial_20261002/nonlinear_comparison/results.json`.
Hybrid rank 154 has held-out output RMSE 0.0261387 and output-difference RMSE
0.0377885, versus INT4's 0.0525693 and 0.0616247. Its fitted-seed output-contrast
RMSE is 0.00231890, versus INT4's 0.0577692 and equal-rank PCA's 0.0344574.
PCA rank 154 still has lower held-out output RMSE, 0.0240777.

Protecting a linear difference leaves a common endpoint offset unconstrained.
SiLU and multiplication by differing up responses can turn that common offset
into different nonlinear errors. Native rank-77 repair therefore has nearly
exact linear span responses but nonlinear fitted-seed contrast RMSE 0.0366390.
The hybrid's additional calibration directions plausibly reduce this offset;
this is consistent with the result, rather than an independently tested causal
explanation or a validation of probe meanings.

The nonlinear/helper/acquisition/CLI integration suites passed 185 tests.
Maximum declared numerical tensor workspace is 802,867,828 bytes under a
1 GiB budget. Read-only source mappings, allocator and BLAS overhead are excluded;
this is not a process-RSS limit. All candidate payloads and source receipts were
checked; candidates are loaded one at a time and outputs publish atomically.

Reproduce the seven fixed-candidate nonlinear evaluation with a new output:

```powershell
& .\.venv\Scripts\python.exe scripts/evaluate_glm_moe_compression.py --sample invariants/out/compression_runs/20261002_glm_gate_sample --up-sample invariants/out/glm_capture_trial_20261002/up_sample --down-sample invariants/out/glm_capture_trial_20261002/down_sample --comparison invariants/out/glm_capture_trial_20261002/hybrid_comparison --archive invariants/out/glm_capture_trial_20261002/archive --out <new_nonlinear_directory> --threads 6 --budget 1073741824 --row-chunk 16 --batch-records 128
```

The concise [measured results](GLM_COMPRESSION_RESULTS.md) distinguish the
linear span, actual nonlinear pole contrasts, held-out state differences,
gate-only storage savings and the smaller savings for the sampled whole block.
