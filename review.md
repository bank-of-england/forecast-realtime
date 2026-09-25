# Review: `feature/density_forecasts` → `dev`

**Verdict:** ready to merge. 1271 tests pass (12 skipped); `ruff check` and
`ruff format --check` are clean. Nothing still refers to the removed names.

## Worth fixing

### 1. Declare `_supports_quantiles` on `ForecastModel`

The docs say the default is `False`, but the base class never declares the
flag; `getattr(..., False)` supplies the default in
`forecast_model.py` (`_predict_data`) and `real_time_model.py` (`forecast`).
Declare `_supports_quantiles: bool = False` beside the other capability
flags, read it directly, and delete the redundant `= False` in
`ForecastRidge`, `ForecastLasso` and `ForecastElasticNet`. Keep the opt-out
in `ForecastBridgeOLS`, which inherits from `ForecastOLS`.

### 2. Import from the defining module

- `real_time_model.py` imports `_normalise_quantiles` from `.forecast_model`.
- `forecast_tree.py` imports `ForecastResult` from `forecast_model`.

Both work only because `forecast_model` happens to import them. Import from
`.forecast_result`.

### 3. One validator for point and quantile results

A quantile result is a point result with one more key column. Today the two
share `_order_forecast_keys` and `_validate_calendar` but differ at the edges
for no principled reason:

| Check                         | Point                           | Quantile                    |
| ----------------------------- | ------------------------------- | --------------------------- |
| Column names                  | exact list, exact order         | exact set, any order        |
| `date` has datetime dtype     | checked                         | not checked                 |
| Calendar source               | supplied, or derived from frame | must be supplied            |
| Calendar checked              | before or after ordering        | before ordering             |

Only three differences follow from the contract itself:

- the key gains `quantile`;
- quantile values must be finite, whereas point values may be `NaN`;
- quantiles must not cross.

Everything else should be one path:

```python
@staticmethod
def _validate_result(forecast, dates, variables, steps, origin,
                     include_origin, probabilities=None):
    keys = ["date", "variable"] + (["quantile"] if probabilities else [])
    columns = keys + ["value"]
    if (not isinstance(forecast, pd.DataFrame)
            or not forecast.columns.is_unique
            or set(forecast.columns) != set(columns)):
        raise ValueError(f"Forecasts must have columns {columns}.")
    if not pd.api.types.is_datetime64_any_dtype(forecast["date"]):
        raise TypeError("Forecast date must have a datetime dtype.")
    if dates is None:
        dates = pd.DatetimeIndex(forecast["date"].drop_duplicates()).sort_values()
    ForecastResult._validate_calendar(dates, steps, origin, include_origin)
    result = ForecastResult._order_forecast_keys(
        forecast.loc[:, columns], dates, variables, probabilities
    )
    if probabilities:
        values = result["value"].to_numpy(float).reshape(-1, len(probabilities))
        if not np.isfinite(values).all():
            raise ValueError("Quantile forecast values must be finite.")
        if (np.diff(values, axis=1) < 0).any():
            raise ValueError("Quantile forecasts must not cross.")
    return result
```

This replaces `_validate_point_result` and `_validate_quantile_result`. It
also makes the column rule the same for both: any order is accepted and the
output order is fixed.

One decision remains: whether quantile hooks may, like point hooks, return
their own dates. If they may, `_finalise_forecast` passes `forecast_dates=None`
for both and the "quantiles require an explicit calendar" rule disappears. If
they may not, keep supplying the calendar for quantiles; the validator above
handles both cases.

### 4. OLS prediction intervals: match R and statsmodels

#### Today

`ForecastOLS` returns plug-in quantiles:

$$\hat{y}_0 + s\,\Phi^{-1}(q), \qquad s^2 = \frac{\text{RSS}}{n - k}$$

where $k$ is `N_regressors`. The intervals omit coefficient uncertainty
and uncertainty in $s$, so they are narrower than those of R's
`predict.lm(interval = "prediction")` or statsmodels'
`get_prediction().summary_frame()` (`obs_ci_*`). They also count $k$
design columns rather than the rank, so they diverge from statsmodels on
rank-deficient designs. Direct strategies and target lags are rejected.

#### Target

Use the textbook OLS prediction interval, conditional on known $x_0$:

$$\hat{y}_0 + s\,\sqrt{1 + x_0^\top (X^\top X)^{+} x_0}\;\, t^{-1}_{n - r}(q),
\qquad s^2 = \frac{\text{RSS}}{n - r}$$

where $r = \operatorname{rank}(X)$ and $(X^\top X)^{+}$ is the
pseudo-inverse. With a full-rank design this equals R and statsmodels
exactly; with a rank-deficient one it equals statsmodels, which also uses
the pseudo-inverse and $n - r$. The median stays the point forecast.

#### Direct and recursive

The formula applies to any single OLS regression whose $x_0$ is observed.
That covers both strategies, with one exception:

| Strategy  | Target lags | Regression for horizon $h$           | $x_0$                      | Closed form? |
| --------- | ----------- | ------------------------------------ | -------------------------- | ------------ |
| Recursive | none        | one regression, $y_t$ on $x_t$       | future row $h$, supplied   | yes          |
| Direct    | any         | one per horizon, $y_{t+h}$ on $x_t$  | origin row, observed       | yes          |
| Recursive | some        | one regression, iterated             | contains forecast $\hat{y}$ | no           |

- **Direct.** Each horizon is its own regression with its own $\beta_h$,
  $s_h$, $(X_h^\top X_h)^{+}$ and $n_h - r_h$, all fitted in
  `LinearRegression._fit`. Target lags are fine: at the origin they are
  observed data, not forecasts. The per-horizon residuals are serially
  correlated (MA($h-1$)), but that affects joint paths, not the marginal
  quantile per horizon; this is what running R per horizon would give.
- **Recursive without target lags.** One regression; each future row is a
  separate one-step prediction with its own leverage.
- **Recursive with target lags.** Later rows contain earlier forecasts, so
  errors compound through the lag polynomial and no closed form exists
  (one would simulate). Keep rejecting this case.

The rule becomes one condition: reject quantiles only for
`forecast_strategy == "recursive"` with `y_lags > 0`.

#### Implementation

All changes are in `models/ols.py`; `LinearRegression` is untouched.

1. **Fit.** After `super()._fit`, rebuild the raw design (intercept plus
   `X`) and, for each fitted $\beta$ (`beta_` for recursive, each
   `betas_[h]` for direct, on `y[h:]` and `design[:n-h]`), store the
   residual variance, the degrees of freedom $n - r$ and
   $(X^\top X)^{+}$ (computed as `pinv(X) @ pinv(X).T`, which is
   numerically stabler than inverting $X^\top X$). Replace `_std_error`
   with this per-horizon mapping.
2. **Forecast.** Compute the point forecast as now. Rebuild the rows it
   used with `_select_forecast_rows` (the first row repeated for direct,
   the first `steps` rows for recursive, ones for an intercept-only fit),
   prepend the intercept, and compute
   `mean + s * sqrt(1 + x0 @ XtX_pinv @ x0) * t.ppf(q, n - r)` per row with
   the terms of that row's horizon.
3. **Errors.** Keep "positive residual degrees of freedom", now per
   horizon; change the lag rejection to "recursive quantiles do not
   support target lags".

#### Tests

- Compare against statsmodels `get_prediction(x0).summary_frame(alpha)`:
  `obs_ci_lower`, `mean` and `obs_ci_upper` for quantiles $\alpha/2$, $0.5$
  and $1 - \alpha/2$. This replaces the current tests that re-derive our own
  formula, so the reference becomes independent.
- Direct: for each horizon, fit statsmodels on `model.y[h:]` and
  `model.X[:n-h]`, with and without `y_lags`.
- Rank-deficient: compare against statsmodels (it uses the pseudo-inverse).
- The realtime test that relies on direct models being rejected needs
  another failing model, such as a subclass that raises for quantiles.
- Unchanged: zero residual variance gives degenerate quantiles; point
  forecasts are unaffected by a quantile call.

#### Docs

Rewrite the `ForecastOLS` section of `docs/models.md`: the new formula,
that it matches R and statsmodels, that direct strategies are supported,
and that recursive target lags are not. Keep the statement that future
regressor paths are treated as known.

#### Out of scope

Uncertainty about future regressors. R and statsmodels also treat $x_0$ as
known. Options for later: model $x$ jointly (BVAR), simulate $x$ paths, or
calibrate quantiles from past real-time forecast errors.

## Nits

- `_validate_decomposition` has `if column not in result: continue`, which
  can never trigger: both columns are required a few lines earlier.
- `_is_fitted` is checked in both `forecast()` and `_predict_data()`, and
  `steps` is validated in `_predict_data`, `ForecastResult` and the tree. The
  checks in `_predict_data` and `ForecastResult` are enough.
- `RealTimeModel.forecast(quantiles=...)` resets `native_forecasts` and
  `decompositions` but leaves forecasts from any earlier point run in
  `self.data`. `add_forecasts` accumulates by design, so this may be intended;
  if so, say so in the docstring.
