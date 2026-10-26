# Antigravity compression audit

Audit date: 2026-10-03. This is a read-only assessment of the original code,
local conversation records and captured command output. The original prototypes,
reports and candidate artifacts were not changed or rerun for this audit.

Antigravity implemented numerical decomposition prototypes and recorded lower
errors for its additive quantizer. Those results do not establish an equal-byte
compression advantage, a semantic-region mapping, or improved model language.
Its reported full-block storage reduction contradicts its implemented payload.

## Original request and provenance

The relevant Antigravity session is `7ecc4cf2-99d9-4ff8-905f-0581efbfa0fc`.
Its local evidence directory is:

```text
C:\Users\Gavin Powell\.gemini\antigravity\brain\7ecc4cf2-99d9-4ff8-905f-0581efbfa0fc
```

In that directory, `.system_generated/logs/transcript_full.jsonl` records:

- **Line 1, step 0, user input, 2026-10-03 16:37 Eastern:** the user asks to
  continue GLM-5.3 compression and use “similies” to represent tensor values as
  shared smaller components. The example is `10.3 = 5sub1 + 5sub2 + 0.3` versus
  `7.3 = 5sub1 + 2 + 0.3`.
- **Line 221, step 220, user input, 17:06 Eastern:** the user asks to add
  trajectories so subcomponents combine through vector addition.
- **Line 239, step 238, user input, 17:11 Eastern:** the user proposes
  `a + b@67degrees + c@35degrees`.

These are direct user requests. The interpretation as scalar additive
quantization and then an angular codebook is the assistant's implementation
choice. Nonorthogonal vector addition remains linear in its coefficients;
discrete code selection is nonlinear, but that does not demonstrate semantic
or trajectory-dependent operators.

The assistant's historical summary is
`full_block_additive_compression_results.md` in the same session directory.
Its error figures are corroborated by command logs below. Its claims of the
“exact same 4-bit representation” and roughly 47% storage reduction are not
supported by the accounting. The recorded full-block errors concern expert 0,
whereas its storage summary discusses the eight-expert sample.

## Implemented mechanisms

[`tda/subgroup_additive_quant.py`](../tda/subgroup_additive_quant.py), lines
88–163, fits blockwise scalar reconstructions of the form
`w = offset + sum(bit_k * subvalue_k)`. Four bits choose among 16 additive levels.
Each subgroup independently stores four FP32 subvalues and one FP32 offset.
It alternates code assignment and least-squares fitting to the original weights.
It does not use semantic-region labels or measured model responses as its fit
objective. Lines 55–57 and 113–116 give the actual tensor layout and payload.

[`tda/angular_frame_quant.py`](../tda/angular_frame_quant.py), lines 43–63 and
105–180, constructs a fixed two-dimensional grid from three directions, at
0°, 67° and 35°, using five coefficient levels per component. The default grid
has 125 points. One uint8 code represents a pair of scalar weights, with one
FP32 scale per block. It fits nearest grid points and scales. This is a numerical
codebook; it does not discover semantic regions or reusable analogical relations.

The older Antigravity vector-travel prototype is separate from weight
compression. [VECTOR_TRAVEL_AND_COMPRESSION.md](../VECTOR_TRAVEL_AND_COMPRESSION.md)
describes its implemented SVD of recent uncentered activation velocities and its
integration limits. Spectral concentration is not a stored model-weight codec
or evidence that a named semantic transformation transfers.

## Results recorded in the original logs

The following paths are relative to the session directory above; line numbers
refer to the physical local files.

`.system_generated/tasks/task-114.log`, **lines 1–3**, reports projection weight
relative Frobenius errors:

| Projection | Symmetric INT4 | Additive | Logged reduction |
|---|---:|---:|---:|
| Gate | 0.1015 | 0.0755 | 25.58% |
| Up | 0.1012 | 0.0753 | 25.54% |
| Down | 0.1050 | 0.0788 | 25.02% |

The projection benchmark uses the first **32,768 flattened values per sampled
expert**, not complete projection matrices; see
[`scripts/evaluate_additive_quant_glm.py`](../scripts/evaluate_additive_quant_glm.py),
lines 19–36. Eight experts times 32,768 values is **262,144 values**. The script's
comment describing approximately 2.68 million parameters is inaccurate.

`.system_generated/tasks/task-212.log`, **lines 2, 4 and 7**, explicitly records
512 heldout states, 512 calibration states, **expert 0**, and block size 32.
Its **lines 17–25** report the following nonlinear output errors:

| Configuration on expert 0 | Logged relative output RMSE |
|---|---:|
| INT4 gate; original up/down | 0.02824 |
| Additive gate; original up/down | 0.02374 |
| INT4 + rank-154 repair gate; original up/down | 0.01408 |
| Additive + rank-154 repair gate; original up/down | 0.01046 |
| All three projections INT4 | 0.07971 |
| All three projections additive | 0.06544 |
| INT4-repaired gate; additive up/down | 0.05892 |
| Additive-repaired gate; additive up/down | 0.05848 |

The corresponding `.system_generated/steps/212/output.txt`, **line 2**, records
exit code 0. These are corroborated historical
measurements, not a new reproduction. They show lower errors in that comparison.
They do not show an equal-storage gain.

`.system_generated/steps/268/output.txt`, **lines 7, 10 and 13**, records the
angular comparison on the same bounded gate prefix:

| Quantizer | Weight relative Frobenius error |
|---|---:|
| Symmetric INT4 | 0.1015 |
| Scalar additive | 0.0755 |
| Angular frame | 0.1913 |

The angular prototype was worse than INT4 in this recorded test. The printed
“4.0 bits/parameter” column, lines 18–20, counts code bits while omitting block
parameters.

## Corrected storage accounting

At block size 32, INT4 stores 16 packed-code bytes plus a 4-byte scale:
**20 bytes per 32 weights = 5 effective bits per weight**. Additive stores
16 packed-code bytes, a 4-byte offset and four 4-byte subvalues:
**36 bytes per 32 weights = 9 effective bits per weight**. The additive payload
is therefore **80% larger than its INT4 control**, despite identical code depth.
The default angular codec stores 16 code bytes plus a 4-byte scale per block,
also **5 effective bits per weight**.

Each complete source projection contains `8 * 2048 * 6144 = 100,663,296` weight
values across the eight sampled experts. Original FP8 weights plus their FP32
block scales occupy **100,687,872 bytes per projection**. Applying the prototype's
payload formula at block size 32 gives:

| Representation | One projection, eight experts | Three projections, eight experts |
|---|---:|---:|
| Original FP8 weights/scales | 100,687,872 | 302,063,616 |
| Fresh INT4, block 32 | 62,914,560 | 188,743,680 |
| Original byte-index angular, block 32 | 62,914,560 | 188,743,680 |
| New seven-bit angular packing, block 32 | 56,623,104 | 169,869,312 |
| Additive, block 32 | 113,246,208 | 339,738,624 |

These additive totals are **computed tensor payloads extrapolated to all eight
experts**, not measurements of a saved eight-expert additive artifact. They
exclude serialization headers, metadata and any rank-154 repair. The single
expert actually fitted in the nonlinear run has 42,467,328 additive bytes across
three projections versus 37,757,952 original FP8/scales bytes. Either scope gives
**12.47% expansion**, before repair and serialization costs. Thus the historical
claim of approximately 160 MB and 47% reduction cannot describe this configuration.

The prior saved INT4 control used **block size 128**, costing 4.25 effective
bits per weight and 53,477,376 tensor bytes per projection across eight experts.
Different block size, calibration subset and metric prevent attributing the
historical comparison to that saved control.

Both prototype save functions also calculate `serialized_total_bytes` before
writing the larger final metadata containing those byte counts:
`subgroup_additive_quant.py`, lines 238–245, and `angular_frame_quant.py`, lines
252–259. Consequently, their reported serialized totals undercount the physical
files. Future comparisons must sum actual stored files after serialization.

### Subsequent lossless packing extension

The new [`tda/angular_frame_packing.py`](../tda/angular_frame_packing.py)
stores the same default 125 grid indices in seven bits per two-weight pair.
At block size 32, 14 index bytes and the unchanged four-byte FP32 scale give
**18 bytes per block = 4.5 effective bits per weight**. This is **10% fewer
tensor bytes** than the original angular codec or this block-32 INT4 control.
It preserves fitted indices, scales and reconstructed weights; it does not
improve the angular grid's measured accuracy. The angular rows above are
payload formulas over the declared sample shape, not a saved full-sample test.

Fixed angles have no per-block overhead. Their shared configuration, coefficient
levels, serialization and metadata are still stored and counted in the new
artifact format. Exact source identity, checksums, canonical padding, valid
indices, bounded decoding and actual final file totals are checked during
save/load. The original quantizer and its artifacts remain unchanged.

A separate [native-prefix verification](../invariants/out/glm_capture_trial_20261002/angular_packing_20261003/verification.json)
fitted the original default angular grid once on the first 32,768 weights of
each of eight gate experts: **262,144 values**, copied into an explicitly scoped
dequantized-FP32 prefix identity. Codes plus scales shrank from **163,840 to
147,456 bytes**. The saved packed artifact occupies **151,506 actual bytes**:
149,285 for tensors, 173 for configuration and 2,048 for metadata. Decoded
weights and FP32 scales remained bitwise identical after packing and restricted
reload. Prefix weight relative Frobenius error remained **0.19128686**. This
CPU verification evaluates storage and reconstruction, not native-FP8 execution
or language behavior. The combined compression regression suites passed
**192 tests**; no GPU or new cloud compute was used.

Sharing four scalar additive subvalues globally would also change its payload:
four-bit codes plus a block-32 FP32 offset cost five bits per weight, plus the
global subvalues. This is a different, more constrained quantizer. Its previous
accuracy results used separately fitted subvalues in every block and cannot be
transferred to that proposed global-sharing scheme.

## Comparison scope and transfer limits

[`scripts/compare_swiglu_full_block.py`](../scripts/compare_swiglu_full_block.py):

- **Lines 36–37:** selects the first 512 records in each split rather than the
  earlier whole-trajectory selection.
- **Lines 54 and 72–74:** reads and fits only expert 0.
- **Lines 78–90:** freshly fits both codecs at block size 32.
- **Lines 98–106:** freshly derives a rank-154 native/PCA basis from those 512
  calibration records, then constructs new dense gate repairs. It does not load
  the previously saved hybrid gate candidate.
- **Lines 68 and 116–117:** computes the square root of total output-error energy
  divided by total reference-output energy. It does not use the prior
  trajectory-balanced, provenance/stimulation-weighted mean of per-example
  relative errors.

The earlier saved evaluation covers eight experts equally and 2,432 heldout
states with 1,216 actual within-trajectory output differences, plus separate raw
seed-pole contrasts. The historical script reports only point-output error;
it does not measure those transitions or seed contrasts. Its “Codex baseline”
label refers to a fresh approximation using a related construction, not the
same saved artifact. Neither experiment establishes full-decoder behavior,
logit quality, native FP8 GPU arithmetic, a router-weighted MoE layer, or runtime
savings.

The closest existing source-conditioned relation learner is
[`invariants/relational_conversion.py`](../invariants/relational_conversion.py),
lines 1–7: `delta(x) = mean_delta + (x - mean_x) @ X.T @ coefficients`.
Its [definition experiment](../invariants/out/definition_runs/20261001_relations_384/assessment.md)
retained a conditional component, but all generated attempts remained
unqualified. This demonstrates implemented conditional geometry, not successful
semantic analogy transfer. A cluster's location or a probe's operator name does
not validate its meaning.

## Completed bounded local follow-up

The [regional response experiment](REGIONAL_GLM_RESPONSE_TRANSPORT.md) uses an
**all-three-projection INT4 block-128 baseline**
and a compact correction to its nonlinear output error. It fits a calibration-only
PCA representation of rank 16 and four geometric regions, with shared regional
operators learned on experts 0–6 and tested on expert 7. Heldout data retain
whole trajectories. Identity and wrong-region operators use identical tensor
payload bytes, allowing a binding control without changing capacity; their
metadata labels differ by a few bytes.

On 2,432 development held-out states, the shared correction reduced output
error by 20.39% versus that same INT4 base, using 2.88% additional serialized
storage. Binding the operators to the wrong regions worsened error. The modest
expert-7 improvement over the identity control was 0.67%; its global basis and
offsets were still calibrated. The linked report includes actual endpoint
differences, region crossings, storage, source receipts and scope limits.
Expert transfer is distinct from semantic transfer into an unseen region.
A compact response correction also remains distinct from a serialized replacement
of GLM's weights or an accelerated decoder. Names for geometric regions and
probe meanings require independent behavioral validation.
