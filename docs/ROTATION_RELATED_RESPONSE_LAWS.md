# Related directions, periodic corrections, and observed winding

This experiment addresses rotation **between directions**, rather than radial
distance alone. It learns an angular relation between quantization-response
corrections for sampled GLM expert functions. A separate path measurement can
retain travel beyond 360 degrees when recorded intermediate directions support
it. Neither a geometric region nor an angle is assigned a semantic meaning.

The completed CPU comparison found that the rotation correction improves
withheld-sector error by **13.99%** versus INT4, but is **0.68% worse** than the
radial correction. Separate sector maps perform best. The capture contains no
supported observed path over 360 degrees; synthetic multi-turn tests pass.

## Completed GLM sample comparison

The final local run on 2026-10-04 completed in **17.078 seconds**, using six CPU
threads and the pinned `zai-org/GLM-5.3` layer-3 samples at revision
`aca966e4e02791568aa6a4ced368624b3d897f42`. No new GPU or cloud compute was used.

- **1,022 calibration inputs** from 32 trajectories define geometry. **264
  sector-1 response records** are excluded from every response fit; **758**
  remain for the common global fits. The two active local charts use 353 and
  373 permitted, phase-supported records.
- Region counts are **9, 10, 523, 480**. The first chart lacks a two-direction
  span; the second lacks sector coverage. Both retain global fallback. The
  remaining charts support all three fitted sectors and the withheld sector.
- **2,432 heldout states**, in 76 disjoint trajectories, supply **1,216 actual
  adjacent pairs**. The withheld sector contains **413 states**, **408** with
  local support. **364 pairs** touch it, **315** enter/leave it, and **49** have
  both endpoints there. **66 states** overall lack supported local geometry.
- The covariant canonical design has effective rank **48 of 51** in both active
  regions under the same Gram-eigenvalue threshold as other fits. This captures
  the three canonically eliminated columns rather than counting FP32 roundoff
  as independent evidence. Normal solve residuals are below `2.8e-14`.

Errors are relative RMSE of sampled expert outputs, not language error rates.
Adapter sizes below are actual serialized decimal MB, including metadata; the
same **160,443,759-byte** saved INT4 base is common to every row.

| Candidate | All-state error | Withheld-angle error | Output-difference error | Expert 7 withheld-angle error | Adapter MB |
|---|---:|---:|---:|---:|---:|
| INT4 base | 0.110701 | 0.114637 | 0.127240 | 0.161761 | 0 |
| Global correction | 0.091236 | 0.099686 | 0.121036 | 0.148843 | 3.931 |
| Regional offsets | 0.091267 | 0.099728 | 0.121002 | 0.149002 | 4.717 |
| Radial quadratic | 0.086760 | 0.097934 | 0.117329 | 0.147570 | 4.731 |
| Covariant rotation | 0.087024 | 0.098605 | 0.117572 | 0.148121 | 4.731 |
| Reversed-angle control | 0.089035 | 0.099154 | 0.119701 | 0.148680 | 4.731 |
| Free Fourier map | 0.086449 | 0.099582 | 0.117376 | 0.147801 | 4.783 |
| Shifted-angle Fourier control | 0.096028 | 0.098285 | 0.125470 | 0.146937 | 4.783 |
| Separate sector maps | **0.086035** | **0.096399** | **0.116718** | **0.146364** | 4.731 |

The correct rotation law beats its reversed-angle control by **0.55%** on the
withheld sector and reduces whole-set error by **21.39%** versus INT4. However,
its withheld-sector error is **0.68% higher** than radial, and separate maps
improve another **2.24%** over rotation. Actual entering/leaving-pair errors are
0.124067 for radial, 0.124588 for rotation and 0.123018 for separate maps. The
rotation restriction has not demonstrated an advantage over the simpler law.

The larger Fourier map slightly improves whole-set output error over radial,
but worsens withheld-sector error. Its shifted-angle control even improves
that subset while worsening whole-set and transition errors. The controls
therefore do not establish a consistent predictive angular relation across
this data, and no tuning was performed after inspecting them.

The covariant adapter occupies **4,730,795 actual bytes**. Together with INT4 it
totals **165,174,554 bytes**, **45.32% below** the **302,063,616-byte** original
sampled FP8/scales payload. The table is only **375 bytes larger**. Rotation and
table each store **13,056 raw coefficient bytes**; Fourier stores **65,280**.
These results establish no material storage advantage for the covariant rule.
Most adapter storage is in common bases and offsets. This sample ratio cannot
be extrapolated into a runnable full compressed model.

The independent winding audit finds **226 supported calibration connections**
and **500 heldout connections**, with longest supported segments of **four
recorded positions** in each split. Maximum absolute signed accumulation is
**192.04 degrees** in calibration and **179.73 degrees** in heldout. No supported
observed segment exceeds 360 degrees. These counts are smaller than raw recorded
contiguous runs because calibrated support and chart continuity are required.
They leave unobserved travel undetermined.

Final settings, source/control receipts, evidence, adapters, diagnostics and
results are in
`invariants/out/glm_capture_trial_20261002/rotation_transport_20261004_verified/`.
The [reviewed comparison figure](../invariants/out/glm_capture_trial_20261002/rotation_transport_figure_20261004_verified/comparison.png)
has SVG, CSV and hash receipts beside it. Earlier radial and angular artifacts
are preserved. A diagnostic-only rank correction prompted the final immutable
receipt run; every fitted tensor hash and every evaluation metric matches the
preceding angular run exactly.

## What the rotation law enforces

Calibration-only, standardized rank-16 PCA coordinates are divided into four
regions. Each region with sufficient direction span has one fixed oriented
two-dimensional plane, measured from its local unit directions. Directions
retain their other fourteen coordinates. The plane and its orientation are
saved; they do not change between neighboring positions or candidates.

For local row direction `v`, phase `theta`, and normalized full local radius
`rho`, the shared correction in latent response coordinates is:

```text
H(rho) = H0 + rho H1 + rho² H2
delta(v,rho,theta) = ([v R(theta), 1] H(rho)) R(theta).T
```

`R(theta)` rotates only the calibrated plane and leaves its orthogonal
complement unchanged. With plane columns `p,q`, its matrix is:

```text
R(theta) = I + (cos(theta)-1)(p p.T + q q.T)
             + sin(theta)(q p.T - p q.T)
```

The row-vector convention is deliberate: `v R(theta)` canonicalizes the
measured phase, and multiplication by `R(theta).T` restores it. The resulting
local law satisfies:

```text
delta(v R(alpha).T,rho,theta+alpha) = delta(v,rho,theta) R(alpha).T
```

Thus a rotated input has a correspondingly rotated local correction. The
complete expert correction adds a global affine response-error map and a
regional offset, then projects `delta` through that expert's calibrated output
basis. Covariance applies to the local latent correction; it is not a symmetry
claim about GLM's full response, language, or learned meanings.

The exact fit uses a separate rotated output Gram for each calibration record.
At rank 16 and degree two the bounded normal system has 816 unknowns, solved in
FP64. Its unregularized design can be rank deficient: canonicalizing the plane
removes one varying coordinate. Ridge permits a prediction without identifying
a unique underlying law. Effective ranks and solve residuals are retained.

## What makes 450 degrees different from 90 degrees

An endpoint direction alone supplies its phase modulo a full turn. Identical
endpoints cannot reveal whether the route included extra revolutions. The
static correction is periodic: `theta` and `theta + 2*pi*k` give the same map.

The separate path audit starts each supported segment at zero and sums signed
principal increments:

```text
increment = atan2(sin(next_phase-previous_phase),
                 cos(next_phase-previous_phase))
accumulated_rotation += increment
```

Synthetic ordered paths `0,90` and `0,90,180,270,360,450` finish at the same
direction but retain 90 and 450 degrees of signed travel. Periodic sine/cosine
features agree; segment-relative net turns differ, 0.25 versus 1.25. Reversed
travel changes the sign. Total absolute travel is also stored separately, so a
turn followed by its reversal can have zero net rotation and nonzero travel.

The audit resets at a token-position gap, trajectory/environment/region-plane
change, undefined or unsupported phase, or an ambiguous 180-degree increment.
It never connects separated sampled pairs into a fabricated continuous path.
Integer winding is reported only when the **projected phase** closes; that does
not require full direction, radius, or latent state closure. Open paths retain
fractional net turns and a separate branch excess relative to their principal
endpoint difference.

This reconstructs the shortest-arc path of the observations, under the stated
assumption that each resolved increment represents the principal turn. Even
consecutive sampled states cannot rule out hidden extra revolutions between
them. Absolute earlier winding is unknown. Winding is recorded as path state,
but does not fit or steer any response-correction candidate in this experiment.

## Comparison fixed before evaluation

Four equal angular sectors define a partial response holdout. Sector 1,
`[90,180)` degrees in each calibrated plane, supplies no response target to any
global basis, intercept, regional offset, or shared map. Its calibration inputs
still define geometry. Separate heldout conversations and trajectories supply
evaluation inputs; this is a withheld-response test rather than entirely unseen
input geometry. Expert 7 supplies no responses to shared operators, although
its own basis and offsets may use permitted-sector responses.

The nine candidates are the unchanged all-INT4 control, global correction,
regional offsets, radial quadratic, covariant rotation quadratic, reversed-angle
rotation control, free Fourier map, shifted-angle Fourier control, and separate
sector maps. All use identical geometry and geometric support. Correct and
incorrect angular controls reuse identical fitted coefficient arrays.

The Fourier control combines radial powers `1,rho,rho²` with harmonics
`1,cos(theta),sin(theta),cos(2*theta),sin(2*theta)`. It has fifteen affine blocks
per region, versus three for radial/rotation/table maps, and its greater capacity
and physical storage must be reported. The compact table stores only the three
fitted sectors; it uses the nearest fitted circular sector center, with the lower
actual sector ID on exact ties.

A region needs at least `3*(rank+1)+1` permitted responses and at least `rank+2`
responses in every nonwithheld sector. Prediction also requires at least 5%
of the direction norm in the plane, a measured radial interval, and bounded
out-of-PCA residual. Unsupported states use the global correction. A calibration
region without two independent directions has no defined angle or local map;
its records can still fit the global fallback because they occupy no withheld
angular sector. An inactive storage placeholder is accompanied by an explicit
undefined-plane mask and never supplies measured angles.

The runner verifies original gate/up/down source receipts, pinned architecture,
and the prior saved all-three-projection INT4 block-128 control. Every adapter is
evaluated after an exact restricted tensor save/load round trip. Reported bytes
include final metadata and actual files. Evaluation uses FP32 CPU nonlinear
expert functions, with actual next-minus-previous output differences from the
recorded adjacent pairs. No additional cloud or GPU compute is used.

The implementation is in `tda/rotation_geometry.py` and
`tda/rotation_response_transport.py`; the local runner is
`scripts/rotation_transport_glm.py`. The earlier radial experiment remains in
[Formula-generated changes across geometric levels](SHELL_GENERATED_RESPONSE_LAWS.md).

```text
.venv\Scripts\python.exe scripts\rotation_transport_glm.py --out invariants\out\glm_capture_trial_20261002\new_rotation_run --threads 6
```

The rotation geometry/law/runner, radial law/runner, regional, nonlinear GLM,
capture-calibration and angular-codec suites passed **260 tests in 6.86 seconds**
on 2026-10-04. Tests independently construct the exact rotating-output ridge
objective, recover a known synthetic law on a withheld sector, prove fitted
tensors remain unchanged when withheld targets change, and check expert
exclusion, identical support, degenerate-plane fallback, actual paired outputs,
periodicity, covariance, multi-turn observed paths, gap/ambiguity resets, scope,
working-memory guards, numerical rank and exact serialized round trips.

Router mixtures, shared experts, native FP8 activation arithmetic, later layers,
logits, generation and inference speed remain outside this test. This development
set was inspected during earlier compression work; fresh confirmation and
runtime integration are required before claiming a model-compression method.
