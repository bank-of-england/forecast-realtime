# Outlier Dummies Strategy

This document explains how `forecast_realtime` handles **outlier (point)
dummies** — one-off indicator variables that neutralise the influence of an
exceptional observation (e.g. a COVID quarter) on the fitted model. For the
broader forecasting design see [forecasting_strategy.md](forecasting_strategy.md);
for the API quick-start see the [Home](index.md); for the package
reference see the [API Reference](api.md).

These are **point dummies for outliers only** — not seasonal dummies and not
multi-period regime/step dummies. A dummy is `1` on a single date and `0`
everywhere else.

## Why outlier dummies

A single extreme observation can distort an estimated relationship: in OLS it
drags the fitted line; in a scaled, regularised model it also inflates the
standardisation statistics. Adding an unpenalised point dummy for that date
absorbs the observation, so the remaining coefficients are estimated as if the
outlier were not there. For a pure point dummy in plain OLS this is exactly
equivalent to dropping that row, but the dummy approach generalises to the
scaled/regularised case and keeps a coefficient for every column.

## API

Dummies are supplied at fit/forecast time (not in the model constructor),
mirroring the `y_lags` / `X_lags` pattern. The `dummies` argument is accepted by:

- `ForecastModel.fit(..., dummies=...)`
- `RealTimeModel.forecast(..., dummies=...)` — passed through to `model.fit()`
  at every vintage.

The value is either:

- **A list of dates** — each becomes a column named by its period:

  ```python
  model.fit(y, X=X, dummies=["2020-06-30"])
  # → column "D_2020Q2" (quarterly), "D_2020M6" (monthly),
  #   "D_2020" (annual), or "D_<ISO date>" as a fallback.
  ```

- **A dict `{name: date}`** — for a custom column name:

  ```python
  model.fit(y, X=X, dummies={"covid": "2020-06-30"})
  # → column "covid"
  ```

The column name for the list form is derived from the date and the **inferred
frequency of the index**, so quarterly/monthly/annual data are labelled
appropriately.

## How it works

- The dummy columns are built deterministically from the `DatetimeIndex`
  (`1.0` on the dummy date, `0.0` elsewhere). Because they are regenerated from
  the index at **both** fit and forecast time, they need **no imputation** —
  this is the key advantage over passing an outlier indicator as an ordinary
  `X` regressor.
- Dummies are concatenated to the design matrix **before** the formula is
  applied, so a formula's right-hand side can select or drop dummies by name
  (`y ~ x1 + D_2020M6` keeps only that dummy; `y ~ .` or no formula keeps all).
- Decomposition is automatic: because dummies are ordinary design columns, the
  forecast decomposition picks them up as components with no extra work.

## All-zero dummy dropping (real-time safety)

In a real-time vintage loop a requested dummy date may fall **outside** the
data available at a given vintage (still in the forecast horizon, or before the
sample starts). That would create an all-zero column — a meaningless zero
coefficient and a singular design for solve-based models.

To prevent this, `fit()` drops any dummy column that is all-zero over the fit
window and records the survivors on `self._dummy_cols`. `forecast()` then
filters the rebuilt dummies to those same survivors, so the fit and forecast
designs stay aligned.

## Regularised models — unpenalised and unscaled dummies (FWL)

For the regularised linear models (`Ridge`, `Lasso`, `ElasticNet`) the dummies
must **not** be shrunk by the penalty, and must **not** be standardised — an
outlier indicator that is itself penalised or rescaled defeats the purpose.

scikit-learn's `Ridge`/`Lasso` apply a single `alpha` to every coefficient and
have no per-feature penalty (no glmnet-style `penalty.factor`), so this is
handled in `LinearRegression._fit` via **Frisch-Waugh-Lovell (FWL)
partialling** whenever dummies are present:

1. Split the design into an **unpenalised block** `U` (intercept + dummies) and
   a **penalised block** `P` (the ordinary regressors).
2. Residualise `P` and `y` with respect to `U`.
3. Scale **only** `P` (when `scale=True`), run the subclass solver on the
   residualised, scaled problem, and rescale the coefficients back.
4. Recover the intercept/dummy coefficients by OLS on the partial residual.

The result is exact: dummies carry **no** L1/L2 penalty and are **never**
scaled, even with `scale=True`, while the ordinary regressors are scaled and
regularised as usual. The plain OLS path is numerically unchanged, and the
no-dummy path is untouched. It works for both recursive and direct strategies
(per-horizon FWL in the direct case), and the reassembled full-length
coefficient vector aligns with the design columns so `forecast()` and the
decomposition need no special handling.

## Important note

Calling `model._forecast_decomp(X=raw_X)` directly **bypasses** dummy
augmentation (the raw `X` lacks the dummy column) and raises a
"Missing regressors" error. Always go through `forecast(decomp=True)`, which
augments the design with the dummies first.
