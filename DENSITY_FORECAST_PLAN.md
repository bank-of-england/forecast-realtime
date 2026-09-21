# Density Forecast Implementation Plan

Status: proposed; no production changes implemented.

## Objective

Add predictive quantiles to `ForecastModel.forecast()`, `predict()` and
`RealTimeModel.forecast()` without changing existing point forecasts. Use
`ForecastOLS` for fast analytical and end-to-end tests. Keep BVAR coverage to one
small native integration test, supported by inexpensive adapter tests.

## Verified Opera Contracts

The Opera `bvar`, `forecast-evaluation` and `forecast-realtime` skills were read
alongside the relevant local source.

- BVAR defaults to quantiles `[0.16, 0.5, 0.84]`. Its formatted output has columns
  `date, quantile, variable, value`.
- BVAR forecast arrays contain effective history followed by the requested future
  periods. Select only the forecast tail when constructing realtime results.
- BVAR applies forecast transformations to individual draws before calculating
  quantiles. `point_only=True` produces a deterministic path, not predictive
  uncertainty.
- `DensityForecastData` stores density rows separately from point rows and accepts
  a numeric `quantile` column. Its mean-from-quantiles conversion is unimplemented;
  retain the model's point forecast rather than deriving it from a quantile grid.
- `SimulationData` identifies input paths using `draw` and `scenario` by default.
  Those identifiers are not posterior or predictive draw identifiers.

## Public API and Results

Add keyword-only `quantiles=False | True | Sequence[float]`, preserving existing
positional arguments, including the tree's positional `context` argument.

```python
result = model.forecast(steps=8, quantiles=True)
result = model.forecast(steps=8, quantiles=[0.05, 0.5, 0.95])
runner.forecast(y_variables=["target"], steps=8, quantiles=True)
```

- `False` preserves current behaviour and avoids density-specific work.
- `True` selects `(0.16, 0.5, 0.84)`.
- Explicit probabilities must be finite, distinct and strictly between zero and
  one. Reject empty sequences and Boolean entries; normalise valid probabilities
  into ascending order. Evaluation accepts endpoints, but analytical Student-t
  quantiles there are infinite.
- `ForecastResult` remains a point-forecast DataFrame. Add `result.quantiles` with
  columns `date, variable, quantile, value`, or `None` for point-only calls.
- Validate complete date/target/probability coverage, unique keys, finite values
  and non-crossing quantiles. Point forecasts need not equal the median.
- `runner.quantiles` contains the corresponding long realtime table with source,
  metric, frequency, vintage and horizon metadata.
- Reject unsupported density requests before realtime tasks run. Decomposition
  remains a decomposition of the point forecast, not of each quantile.

## Model Execution Contract

Allow density-capable `_forecast()` implementations to return an enriched
`ForecastResult` containing point forecasts, quantiles and an optional private
joint predictive draw payload. Existing array/DataFrame returns remain valid for
point-only requests. Extend the shared finaliser to preserve and validate the
payload rather than discarding it when wrapping the result.

Use one forecast call for point and density output. Avoid a second fit, a second
BVAR simulation or a mutable cache of the last density result. Preserve public
`predict()` and `forecast()` overrides through the existing dispatch adapters.

Represent joint draws with a documented `(draw, step, target)` ordering aligned
to the point forecast's dates and targets. Keep this internal initially; do not
introduce a general distribution framework or a public sample-storage API.

## Fast ForecastOLS Densities

Extend the existing `ForecastOLS`, rather than introducing a test-only model or
replacing its NumPy coefficient solver. Keep OLS uncertainty calculations out of
Ridge, Lasso and ElasticNet unless those models later implement their own methods.

### Analytical quantiles

Initially support intercept-only and fixed-regressor OLS with no target lags,
using the existing recursive strategy without recursive target feedback. Future
regressors are treated as known; imputed or supplied regressor paths do not add
regressor uncertainty to these densities.

For a full-column-rank design with independent Gaussian, constant-variance errors:

$$
\nu = n-k, \qquad
s^2 = \frac{\sum_{i=1}^{n}(y_i-x_i^\top\hat\beta)^2}{\nu}, \qquad
Q_p(y_*\mid x_*) = x_*^\top\hat\beta
  + t_\nu^{-1}(p)\,s\sqrt{1+x_*^\top(X^\top X)^{-1}x_*}.
$$

Here `n` is the number of retained estimation observations and `k` is the number
of design columns, including the intercept when fitted. The `1 +` includes future
observation noise; omitting it gives uncertainty in the conditional mean instead.

Capture the actual fitted design, residual variance and covariance factor in
consistent units. Respect formulas, retained dummies, missing-row selection and
scaling. Use stable linear algebra rather than an explicit matrix inverse.
Require positive residual degrees of freedom and full column rank for density
requests; leave existing point-only rank-deficient fits unchanged. Permit a
zero-residual-variance fit to produce a degenerate predictive distribution.

Calculate quantiles directly with SciPy's Student-t inverse CDF. This is fast,
deterministic and needs neither posterior sampling nor optimisation. The density
path must not refit coefficients; uncertainty statistics may be computed lazily
from the retained estimation design.

### Joint draws for transformation tests

Provide optional seeded joint predictive draws when the realtime transformation
path requires them. Under the same Gaussian regression assumptions, draw one
variance and one coefficient vector per path, then fresh observation errors for
each future period:

$$
\sigma_d^2 = \nu s^2 / \chi^2_{\nu,d}, \qquad
\beta_d\mid\sigma_d^2 \sim
N\left(\hat\beta,\sigma_d^2(X^\top X)^{-1}\right), \qquad
y_{d,h}=x_h^\top\beta_d+\epsilon_{d,h}.
$$

This gives the usual Student-t predictive marginals under the reference-prior
Gaussian regression model. Sharing coefficients and variance within a path
preserves their contribution to cross-horizon dependence. Independently sampling
each marginal Student-t distribution would lose that dependence.

Use a private NumPy generator and explicit draw-count and seed controls when
sampling is requested. Ordinary analytical quantiles should require no draws.
Use a few hundred draws for orchestration tests that compare identical seeded
paths; do not use such tests to claim accurate tail estimation. Compute expected
transformed quantiles from those same paths instead of imposing fragile Monte
Carlo accuracy tolerances.

Initially reject target-lag recursion and direct-strategy density requests.
Recursive densities need simulated lag feedback; direct densities need
horizon-specific uncertainty estimates and an explicit joint dependence model.

## BVAR Adapter

Reuse the conditional or unconditional draws from the existing backend call.
Calculate point summaries and quantiles from the same draws, keep only the
requested future rows and preserve native joint paths for reconstruction.
Leave `forecasts_type` controlling the point mean or median.

Reject density requests with `mode_only=True`. Validate that sampling leaves
enough retained draws to describe uncertainty.

Fix the relevant burn-in default before testing conditional densities:
`ForecastBVAR` currently defaults to `n_samples=1000`, `N_draws=5000` and an
explicit `N_burn=2500`. Native forecasting caps draws at the stored posterior
count, making that conditional burn-in invalid. Preserve an omitted burn-in as
`None` so the backend can resolve it against the effective count; reject invalid
explicit settings rather than silently changing them.

## Density Ingestion Bug

There are two separate correctness issues in the current local
`forecast_evaluation.data.DensityForecastData` implementation:

1. `_add_density_forecasts()` copies input values and unconditionally assigns
   `metric="levels"`. Inputs labelled `pop` or `yoy` are therefore misrepresented.
2. `_prepare_density_forecasts()` calculates percentage changes along each
   marginal quantile curve. A ratio of marginal quantiles is generally not a
   quantile of the ratio; the joint distribution across dates is required.

For example, consider two equally likely level paths across two dates:
`(100, 200)` and `(200, 100)`. Both dates have the same marginal distribution, so
every same-probability quantile curve is flat and the current conversion gives
zero growth. Actual path growth is either `+100%` or `-50%`.

The distinction does not prohibit every quantile transformation. A strictly
increasing pointwise transformation, such as exponentiation, preserves quantile
order. Growth relative to a known positive historical level also has a fixed
denominator. Growth between two uncertain future levels does not.

### Required evaluation integration change

Plan a separate, tested change in `forecast-evaluation` to preserve explicitly
supplied metrics and stop automatically manufacturing density metrics from
marginal quantile paths. It should accept precomputed metric-specific quantiles
without requiring private-table mutation. Do not treat `compute_levels=False` as
a workaround: the current density ingestion path still constructs derived rows.

Realtime should transform coherent paths using history available at each vintage
and only then calculate quantiles. Do not use `sample_from_density()` to recreate
the original paths: marginal quantiles do not identify their dependence.

Until corrected ingestion is available, keep authoritative density rows in
`runner.quantiles` and do not publish misleading derived metrics. Afterwards,
use `add_density_forecasts()` where the caller's data object supports it, while
continuing to store model point forecasts independently through `add_forecasts()`.
Preserve the caller's data object and declare any required dependency version.

## Realtime, Transformations and Simulations

- Carry density options in `ForecastTask` run controls and quantile output in
  `ForecastRunResult`; do not forward orchestration options into model fitting.
- Apply the same dates, publication masks, per-variable horizon cutoffs, source
  labels and native metric mappings as point forecasts.
- Reconstruct complete predictive paths before trimming intermediate horizons
  needed by cumulative transformations. Never reconstruct quantile curves.
- Summarise draws inside workers so full BVAR arrays need not cross process
  boundaries. Retain native quantiles when reconstruction is disabled.
- Leave existing point forecasts unchanged: a transformed point forecast need
  not equal the mean or median of the transformed predictive distribution.
- In simulation runs, append the configured input simulation identifiers to
  `runner.quantiles`. Keep those identities distinct from predictive draw axes.
  Retain path-local calendars, filters and models; do not replace panel objects.
- Defer forecast-tree densities until cross-model dependence has a defined
  contract. Do not combine corresponding marginal quantiles as joint draws.

## Test Strategy

OLS is the primary density test model. Reuse existing test modules and fixtures;
avoid multiplying expensive native-model runs across general contract tests.

1. Add exact OLS checks against `statsmodels.get_prediction()` observation
   intervals, including intercept-only fits, scaling, formulas and dummies.
   Check invalid probabilities, sample-size/rank failures and unchanged points.
2. Use small OLS fixtures for result validation, multiple vintages, publication
   masking, cutoffs, repeat calls, decomposition coexistence and simulation paths.
3. Test draw-wise reconstruction with deterministic paths, including the
   counterexample above and a two-step differenced series. Add one seeded OLS
   integration through the real transformation path.
4. Test sequential and spawned parallel equality with OLS. Keep one small real
   process-pool case and use existing lightweight execution fixtures elsewhere.
5. Add one small native BVAR integration test: two variables, one lag, short
   history and horizon, few posterior draws, `nb_restart=0`, fixed seeds and no
   progress bars. Reuse one fit for unconditional and conditional calls; compare
   wrapper quantiles to native draws and formatted output without refitting.
6. Test BVAR adapter selection, burn-in forwarding, mode-only rejection and draw
   shape handling with synthetic backend outputs, without sampling or compiling
   native kernels. Do not use BVAR for generic realtime or simulation tests.
7. In `forecast-evaluation`, test preservation of supplied metrics, no invented
   derived densities, coexistence with points and unchanged state on rejection.

## Delivery Order and Verification

1. Add the request/result contract and analytical OLS densities with focused tests.
2. Add OLS joint paths and realtime density collection, transformation and
   simulation coverage.
3. Add the BVAR adapter and burn-in correction with one small native integration.
4. Correct density ingestion in `forecast-evaluation`, then enable and test public
   storage integration against a compatible version.
5. Update API and usage documentation, model-author guidance and Opera skills.

Run focused tests with `pytest -n auto` in the `forecast-realtime` conda environment
after each slice, then the full suite and the repository's lint, formatting,
docstring and documentation checks. Use the `forecast-evaluation` environment for
that package's separate changes and tests. Record unsupported density modes
explicitly rather than silently falling back to point forecasts.

Suggested commit messages, without committing:

- `feat: add predictive quantiles to model and realtime forecasts`
- `fix: preserve density metrics without transforming marginal quantiles`