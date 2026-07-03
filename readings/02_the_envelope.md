# The envelope: why no push can replace a thought

Every steering intervention in this system — expert-branch nudges, synthesis
and cache deltas, urgency, the ClaimMap and memory steers — passes through
one function before it touches the residual stream. It is called `_cap_steer`
and it lives in `invariants/engine.py`. Its core:

```python
a_norm = a.float().norm(dim=-1, keepdim=True)
cap = frac * h.float().norm(dim=-1, keepdim=True)
scale = torch.clamp(cap / a_norm.clamp_min(1e-30), max=1.0)
return (a.float() * scale).to(h.dtype)
```

In words: a push may never exceed a set fraction (default 0.5) of the norm of
the residual it lands in, per token. Small calibrated pushes pass through
unchanged; only over-steers are clipped. The reason is recorded history: an
uncapped steer as large as the state itself replaces the thought, and what
comes out is word-salad or a repetition loop. The bound makes that failure
impossible no matter what strength any channel is tuned to.

Here is the same arithmetic in plain Python. This block is safe to run in the
sandbox, and running it verifies the claim with real output:

```python
import math
h = [3.0, 4.0]                       # a residual state with norm 5
push = [30.0, 40.0]                  # an attempted over-steer, norm 50
cap_fraction = 0.5
h_norm = math.hypot(*h)
p_norm = math.hypot(*push)
scale = min(1.0, cap_fraction * h_norm / p_norm)
print("attempted push norm:", p_norm)
print("applied push norm:  ", p_norm * scale)   # 2.5 = 0.5 * 5
```

The fraction itself is not sacred. It is a live knob (`steer_cap_fraction`),
and it can be calibrated from data: every real push records its attempted
push-to-residual ratio, and the cap can be set to an exact percentile of that
observed distribution. Same rule for WHERE steering lands: band-wide steering
across layers 0.40–0.70 of depth is only a prior. A layer-sweep mode applies
each steer to exactly one layer and records the outcome per layer, because —
as the operator put it — "we don't even know that the model is always
tracking the same thing on a single axis." Adjacent-layer cosine of the steer
vectors (`axis_drift`) measures whether one direction even means one thing
across depth.

Two rules never bend: the bound can be moved but not removed (non-finite
values are rejected), and nothing recalibrates itself. Every change to the
envelope is a deliberate operator command, made after reading the data.
