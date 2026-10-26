# 6. Prediction, evidence, and semantic environments

Questions this document answers: How does a prediction become data? Why does a miss still count fully? When can learned information affect another state?

## Forecast first, observe later

Lookahead creates a ticket at a source state for a future horizon measured in token steps. The ticket records the source token and state, predicted token and state, origin environment, and other metadata. The arrival is evaluated when that future step is actually observed.

Every matured ticket contributes one full observation, whether its token prediction matched or missed. Correctness is recorded as telemetry; it does not discount or negate the observation. A miss supplies an observed transition just as a hit does.

"Fully valid data" here means full inclusion of the observed prediction/outcome event. It does not make the predicted statement true, certify the factual reliability of the source text, or convert an unobserved future into an observation. A horizon beyond the end of a sequence remains unobserved. If only a terminal token is available, it supplies token evidence without an invented hidden-state target.

## Where the data belongs

Statistics are separated by stream, named context, semantic neighborhood, source token, and horizon. The neighborhood is defined by similarity to a fixed center; the configured default cosine threshold is 0.90. This is an operational approximation to a semantic environment, not proof that every member has exactly the same meaning.

External text and self-generated tokens have separate streams. External context can be formed from token embeddings with a causal running average; self observations use a designated intermediate-layer sensor. Their coordinate systems must not be treated as interchangeable.

The origin environment retains the observation even if the state has moved elsewhere by arrival. Fixed centers prevent an environment from expanding indefinitely by repeatedly moving its center to the latest nearby point. The same source token can therefore have different continuation statistics in different environments.

## Learning and intervention are different

The evidence store updates token counts, the mean actual-minus-source displacement, and per-coordinate dispersion. Future forecasts can use the local mean transition. This online update does not train shared neural weights.

An optional intervention is more selective than data collection. Environment matching and positive trajectory alignment gate its applicability. Coordinates where expected and observed travel have opposing signs are suppressed. A dispersion score, `1 / (1 + sample_variance)`, can weight coordinates unevenly. It requires at least two latent observations and is not a calibrated probability of truth.

There is another important distinction: the stored statistics learn **what occurred**, while the default correction uses negative forecast error, pointing from the observed arrival toward the predicted state before gating and scaling. That control policy is not the statistical update itself, and its usefulness needs outcome evaluation.

The current implementation supports context-matched corrections rather than unrestricted global transfer. Compatible later states can reuse local evidence; unrelated contexts are not automatically entitled to it. Steering strength can be zero while observation continues.

Apply this: explain a learning event using its origin environment, source token, horizon, actual outcome, and later applicability. Preserve misses as evidence without treating their failed predictions as facts.

Sources: [ticket lifecycle and intervention](../../invariants/lookahead_predictor.py) (`predict_waypoint`, `step_arrival`, `_direction_support`); [local statistics](../../invariants/lookahead_evidence.py); [lookahead operational reference](../../docs/LOOKAHEAD.md).
