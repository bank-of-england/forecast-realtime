# Density Forecast Plan

Status: proposed. No production changes have been made.

## Decision

Add predictive quantiles to `ForecastModel.forecast()`, `predict()`, and
`RealTimeModel.forecast()` as a separate output mode. Point and quantile
results are not mixed in the stored realtime results:

- `quantiles=False` keeps the current point-forecast behaviour and stores point
   forecasts in `data.forecasts`.
- `quantiles=True` or a supplied probability sequence stores only native-metric
   quantile rows in `runner.quantiles`, including a `quantile` column. It does
   not store point rows in the same result.

The model may calculate a point forecast internally during a density request,
but realtime does not publish that point result alongside the quantiles.
Level reconstruction is disabled automatically for density requests.

Quantiles are the public density result. The library will not expose or store
predictive draws. Models may calculate quantiles analytically or use joint paths
inside an adapter before discarding them.

The first release supports quantiles for a model's native forecast metric. It
rejects density requests that would require realtime to derive a metric from
quantiles at several uncertain future dates.

## Public Contract

Add a keyword-only `quantiles` argument:

```python
result = model.forecast(steps=8, quantiles=True)
result = model.forecast(steps=8, quantiles=[0.05, 0.5, 0.95])
runner.forecast(y_variables=["target"], steps=8, quantiles=True)
```

`quantiles=False` keeps current behaviour. `quantiles=True` requests the default
probabilities `(0.16, 0.5, 0.84)`. A supplied sequence must contain distinct,
finite probabilities strictly between zero and one. Reject empty sequences and
Boolean entries. Sort valid probabilities before use.

For direct model calls, `ForecastResult` remains the point-forecast DataFrame
with a `quantiles` attribute containing `date`, `variable`, `quantile`, and
`value`, or `None` for a point-only request. At the realtime boundary, the
requested output mode is exclusive: `runner.quantiles` stores the equivalent
quantile rows, including source, metric, frequency, vintage, and horizon, while
point-only runs continue to use `data.forecasts`.

`reconstruct_levels` applies only to point mode. When `quantiles` is true or a
sequence is supplied, realtime disables level reconstruction regardless of the
`reconstruct_levels` value. Quantiles remain in the model's native forecast
metric; realtime never derives level quantiles from marginal quantile curves.

Validate complete date-variable-probability coverage, unique keys, finite
values, and non-crossing quantiles. The point forecast need not equal the median.
Keep decomposition as a decomposition of the point forecast only; reject
`decomp=True` for quantile-only realtime runs until both output modes can be
published together.

Update the `forecast()` and `predict()` docstrings to document the exclusive
output modes and add this TODO: point and quantile results cannot currently be
published together. Supporting both will require predictive draws (or an
equivalent joint-path representation) and a flag stating whether those draws
preserve temporal dependence.

## Model Contract

A density-capable `_forecast()` returns point forecasts and quantiles in an
enriched `ForecastResult`; realtime selects one output mode when publishing the
result. Existing array and DataFrame returns remain valid for point forecasts.
The shared finaliser preserves and validates quantiles.

Use one forecast call to produce points and quantiles. Do not refit, run a second
BVAR simulation, or cache the last density result. Preserve the existing
`forecast()` and `predict()` dispatch behaviour.

Models use one of two internal approaches:

1. **Direct quantiles.** The model calculates the requested probabilities without
   generating paths. `ForecastOLS` uses this approach.
2. **Native path summarisation.** A backend generates joint paths, applies a
   transformation it supports, and calculates quantiles before returning control
   to the adapter. BVAR uses this approach.

Do not add a shared draw payload or public sample-storage API. A future feature
that needs realtime to transform arbitrary predictive paths should introduce a
private path capability with its own contract.

## ForecastOLS

Extend `ForecastOLS`; do not add density support to Ridge, Lasso, or ElasticNet.

Initially support intercept-only and fixed-regressor OLS without target lags.
Treat future regressors as known. Imputed and supplied regressor paths add no
regressor uncertainty.

For a full-rank design with independent Gaussian, constant-variance errors:

$$
\nu = n-k, \qquad
s^2 = \frac{\sum_{i=1}^{n}(y_i-x_i^\top\hat\beta)^2}{\nu}, \qquad
Q_p(y_*\mid x_*) = x_*^\top\hat\beta
+ t_\nu^{-1}(p)\,s\sqrt{1+x_*^\top(X^\top X)^{-1}x_*}.
$$

Here, `n` is the number of retained estimation observations and `k` is the
number of design columns, including an intercept when fitted. The leading `1`
includes future observation noise.

Capture the fitted design, residual variance, and covariance factor in consistent
units. Respect formulas, retained dummies, missing-row selection, and scaling.
Use stable linear algebra, not an explicit inverse. Density requests require a
full-rank design and positive residual degrees of freedom. Point-only forecasts
retain their current behaviour for rank-deficient fits. A zero residual variance
produces a degenerate distribution.

Calculate OLS quantiles with SciPy's Student-t inverse CDF. Do not refit the
model to calculate uncertainty.

Reject OLS density requests for target-lag recursion and direct strategies in
this release. Both require a joint multi-horizon uncertainty model.

## BVAR

The BVAR backend already generates joint forecast paths. Its forecast arrays
contain effective history followed by the requested horizon. The adapter selects
the forecast tail for realtime results.

BVAR applies supported forecast transformations to each joint path before it
calculates quantiles. Its formatted result has `date`, `quantile`, `variable`,
and `value` columns. The adapter should use this behaviour, return the requested
quantiles, and discard the paths. `point_only=True` produces one deterministic
path, so density requests must reject it.

Keep `forecasts_type` as the choice between a point mean and median. Reject a
density request when BVAR does not support the requested transformation. Realtime
must never derive joint behaviour from marginal quantile curves.

Correct the wrapper's burn-in default before testing conditional densities.
`ForecastBVAR` requests `N_draws=5000` but retains `n_samples=1000`; the backend
caps the forecast draw count at the retained count. Preserve an omitted `N_burn`
as `None` so the backend derives burn-in from the effective count. Reject invalid
explicit burn-in values.

## Realtime Boundary

Pass density options through `ForecastTask` run controls and return quantiles in
`ForecastRunResult`. Do not pass orchestration options to model fitting.

Apply the same forecast calendar, publication masks, horizon cut-offs, source
labels, and native metric mappings used for point forecasts. Summarise BVAR paths
inside the worker so that full arrays do not cross process boundaries.

Point and quantile publication are mutually exclusive. In point mode, retain the
existing level-reconstruction path and write `data.forecasts`. In quantile mode,
skip level reconstruction and `data.add_forecasts`; write only native-metric
quantile rows, with the `quantile` column, to `runner.quantiles`. Never
reconstruct paths from quantile curves. In simulation runs, append the
configured input simulation identifiers to `runner.quantiles`; these identify
input scenarios, not predictive paths.

Reject density requests for transformations that the model or backend cannot
produce directly in its native metric. Realtime must not derive a metric from
quantiles at several uncertain future dates.

Defer forecast-tree densities until the project defines cross-model dependence.
Do not combine matching marginal quantiles as if they were joint draws.

## `forecast-evaluation` Boundary

`DensityForecastData` currently has two separate defects:

1. `_add_density_forecasts()` overwrites the supplied metric with `levels`.
2. `_prepare_density_forecasts()` calculates growth along marginal quantile
   curves. A ratio of marginal quantiles is generally not a quantile of a ratio.

For example, two equally likely level paths, `(100, 200)` and `(200, 100)`, have
flat marginal quantile curves. The current conversion reports zero growth, though
the path growth is either `+100%` or `-50%`.

The first release keeps authoritative density rows in `runner.quantiles` and does
not use derived-density ingestion. A later `forecast-evaluation` change should
preserve supplied metrics, accept precomputed metric-specific quantiles, and stop
manufacturing derived density rows. `compute_levels=False` is not a workaround.

Do not use `sample_from_density()` to recreate paths. Marginal quantiles do not
identify their temporal dependence.

## Tests

Use OLS for general density tests and keep native BVAR coverage small.

1. Compare OLS prediction intervals with `statsmodels.get_prediction()`, covering
   intercept-only fits, scaling, formulas, and dummies.
2. Test invalid probabilities, insufficient observations, rank failures, and
   unchanged point forecasts.
3. Test result validation, multiple vintages, publication masks, horizon
   cut-offs, repeat calls, decomposition, and native-metric simulations with
   small OLS fixtures.
4. Test quantile-mode output selection: level reconstruction is disabled, the
   stored rows retain the native metric and `quantile` column, and point rows
   are not published. Confirm rejected density requests leave point forecasts
   and caller data unchanged.
5. Test sequential and spawned parallel runs with OLS, including one small real
   process-pool case.
6. Add one BVAR integration test with two variables, one lag, short history, few
   posterior draws, `nb_restart=0`, fixed seeds, and no progress bars. Compare
   wrapper quantiles with the backend's formatted output for unconditional and
   conditional forecasts, including one BVAR-supported transformation.
7. Test BVAR selection, burn-in forwarding, mode-only rejection, and draw shapes
   with synthetic backend outputs.
8. In `forecast-evaluation`, test preserved native metrics, no invented derived
   densities, exclusive quantile storage, and unchanged state after rejection.

## Delivery

1. Add the request and result contract with analytical OLS quantiles.
2. Add realtime collection, validation, and rejection of unsupported transformed
   densities.
3. Add the BVAR adapter and burn-in correction.
4. Update API, usage, and model-author documentation.
5. Add private joint-path support and `forecast-evaluation` ingestion only when
   publishing points and quantiles together becomes a concrete requirement. The
   joint-path contract must state whether temporal dependence is preserved.

After each slice, run focused tests with `pytest -n auto` in the
`forecast-realtime` conda environment. Then run the full suite and the
repository's lint, formatting, docstring, and documentation checks. Use the
`forecast-evaluation` environment for its separate work.

Suggested commit messages, without committing:

- `feat: add predictive quantiles to model and realtime forecasts`
- `fix: preserve density metrics without transforming marginal quantiles`