# Models

All built-in models live under `rt.models` and inherit from `ForecastModel`.
Lags are supplied at fit/forecast time (`y_lags`/`X_lags`), not in the
constructor.

Optional models are imported when their public attribute is first accessed.
Install the extra for the model you use, for example `pip install -e
.[ridge]`, `pip install -e .[xgboost]` or `pip install -e .[bvar]`. If a
required dependency is unavailable, attribute access raises
`ModuleNotFoundError` naming the model, dependency and installation extra.

| Model | Description |
|-------|-------------|
| `ForecastOLS`, `ForecastRidge`, `ForecastLasso`, `ForecastElasticNet` | Linear models (shared `LinearRegression` base; support `scale`, recursive/direct strategy, `formula`) |
| `RandomForest`, `XGBoost` | Tree models (shared `TreeRegression` base; support `standardise`, `formula`, recursive/direct strategy) |
| `ForecastBVAR` | Bayesian VAR with conditional forecasting |
| `ForecastMIDAS`, `ForecastMultiMIDAS`, `ForecastMIDASCombo` | Mixed-frequency (MIDAS) models |
| `ForecastRlm` | Wrapper for an R-based `lm()` model |

```python
import forecast_evaluation as fe
import forecast_realtime as rt

forecast_data = fe.ForecastData(load_fer=True)
ols = rt.models.ForecastOLS(label="ols", formula="cpisa ~ gdpkp")
ridge = rt.models.ForecastRidge(cv=5, scale=True)

# Pass several models at once — each is run across every vintage
rt_model = rt.RealTimeModel(data=forecast_data, models=[ols, ridge])
```

## Example with a built-in model

```python
import forecast_evaluation as fe
import forecast_realtime as rt

forecast_data = fe.ForecastData(load_fer=True)

ridge_model = rt.models.ForecastRidge(label="Ridge", cv=5, scale=True)
lasso_model = rt.models.ForecastLasso(label="LASSO", cv=5, scale=True)

rt_model = rt.RealTimeModel(data=forecast_data, models=[ridge_model, lasso_model])

rt_model.forecast(
    y_variables=["cpisa"],
    X_variables=["gdpkp", "unemp"],
    data_transformation={"cpisa": "pop", "gdpkp": "pop", "unemp": "pop"},
    steps=12,
    y_lags=4,
    first_vintage="2015-01-01",
    X_imputation="last",
)

# Optional interactive dashboard:
# rt_model.data.run_dashboard()
```

## Cross-validation for regularised models

`ForecastRidge`, `ForecastLasso` and `ForecastElasticNet` accept the `cv`
argument for selecting the regularisation parameter. When `cv` is an integer,
it is passed to scikit-learn's estimator and uses ordinary K-fold
cross-validation. This selects the penalty within each real-time vintage; it
does not replace the outer evaluation across forecast vintages.

For a time-ordered validation scheme, pass a scikit-learn-compatible splitter
explicitly:

```python
from sklearn.model_selection import TimeSeriesSplit

ridge = rt.models.ForecastRidge(cv=TimeSeriesSplit(n_splits=5), scale=True)
```

Ordinary K-fold cross-validation is not automatically invalid for time-series
models. Bergmeir, Hyndman and Koo (2018) show that standard K-fold
cross-validation can be used for purely autoregressive models when the
candidate models have uncorrelated errors. This result is not a blanket
guarantee for models with arbitrary exogenous regressors, non-stationarity,
structural breaks or serially correlated errors. Choose a splitter that
matches the information available at the forecast origin and the forecasting
task; `TimeSeriesSplit` and other custom splitters are available when that is
appropriate.

Reference: Bergmeir, C., Hyndman, R. J. and Koo, B. (2018), "A note on the
validity of cross-validation for evaluating autoregressive time series
prediction", *Computational Statistics & Data Analysis*, 120, 70-83.
[doi:10.1016/j.csda.2017.11.003](https://doi.org/10.1016/j.csda.2017.11.003).

## Models in other languages

Models written in R, MATLAB or Julia can be wrapped with `RModel`,
`MATLABModel` and `JuliaModel`. You provide `fit` and `forecast` functions in
the target language; the wrapper handles data exchange (Parquet), parameters
and the subprocess.

The supplied R, MATLAB or Julia script is executed by the corresponding
runtime and must therefore be trusted code. The wrapper protects generated
command and path literals, but it does not sandbox the external script.

```python
import forecast_realtime as rt
from forecast_realtime import RModel, MATLABModel, JuliaModel

model = RModel("my_model.R", p=4, shrinkage=0.5)

ets = rt.models.RFableETS(error="A", trend="A", season="N")
arima = rt.models.RFableARIMA(p=1, d=0, q=0)
```

See [adding_a_model.md](adding_a_model.md) for the expected function
signatures.

### Regressor imputation

When `X_imputation` is requested, every regressor must have at least one
observed value. An all-missing regressor is rejected with a `ValueError`
rather than being silently removed from the fitted specification. Models that
own their own missing-value handling are not sent through this generic
imputation path.

### Fable models

`RFableModel`, `RFableETS` and `RFableARIMA` wrap the R `fable` package.
They require an R installation with the `arrow`, `fable`, `fabletools` and
`tsibble` packages available; tests and fits invoke `Rscript` directly.

- `spec` (`RFableModel` only): a generic R fable model expression, e.g.
  `"ARIMA(value ~ 1 + pdq(1, 0, 0))"`. The response column is always named
  `value`; regressors keep their Python column names. `spec` is parsed and
  evaluated as trusted R code. `RFableETS` and `RFableARIMA` build this
  expression for you from their keyword arguments.
- `index`: how the date index is converted to a tsibble index — one of
  `"auto"` (inferred from the data's spacing), `"quarter"`, `"month"` or
  `"date"`.
- `allow_xreg` / `xreg`: `RFableModel` accepts an `allow_xreg` flag that
  controls whether regressor columns in `X` are permitted. `RFableARIMA`
  instead takes an `xreg` string naming the regressor term to add to the
  ARIMA formula (e.g. `xreg="indicator"`), which also sets `allow_xreg=True`.
  `xreg` is an R expression and must come from a trusted source.

```python
ets_monthly = rt.models.RFableETS(error="A", trend="A", season="N", index="month")
```
