# Usage

The snippets below use this shared setup. Each model is run for every data
vintage by the same `RealTimeModel` instance.

```python
import forecast_evaluation as fe
import forecast_realtime as rt

forecast_data = fe.ForecastData(load_fer=True)
model_1 = rt.models.ForecastOLS(label="OLS")
model_2 = rt.models.ForecastRidge(label="Ridge", cv=5, scale=True)
rt_model = rt.RealTimeModel(data=forecast_data, models=[model_1, model_2])
```

## Lags

The linear, tree and neural models support autoregressive (`y_lags`) and
distributed (`X_lags`) lags, supplied at forecast time:

```python
rt_model.forecast(
    y_variables=["cpisa"],
    X_variables=["gdpkp", "unemp"],
    data_transformation={"cpisa": "pop", "gdpkp": "pop", "unemp": "pop"},
    steps=12,
    y_lags=4,  # append y_{t-1} … y_{t-4}
    X_lags={"oil": 4, "fx": 1},  # per-regressor lag counts (an int applies to all)
)
```

`y_lags=k` appends `_y_lag1 … _y_lagk`; `X_lags` appends `col_lag1 … col_lagk`
per regressor.

## Outlier dummies

Pass `dummies` to add one-off **point dummies** (value `1` on a single date, `0`
elsewhere) for outliers such as the COVID quarter. Supply either a list of dates
or a `{name: date}` mapping; the same argument works on `ForecastModel.fit(...)`
and `RealTimeModel.forecast(...)`.

```python
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "pop"},
    steps=12,
    dummies=["2020-06-30"],  # or {"covid": "2020-06-30"}
)
```

Dummies are rebuilt from the `DatetimeIndex` at both fit and forecast time (no
imputation), follow formula selection, and appear as ordinary components in the
decomposition. For `ForecastRidge`/`ForecastLasso`/`ForecastElasticNet` they are
left unpenalised and unscaled. See [dummies_strategy.md](dummies_strategy.md).

## Regressor imputation

Regressors are often **ragged** — columns end at different dates and/or fall
short of the forecast horizon. Set `X_imputation` to fill those gaps at both fit
and forecast time:

```python
rt_model.forecast(
    y_variables=["cpisa"],
    X_variables=["gdpkp"],
    data_transformation={"cpisa": "pop", "gdpkp": "pop"},
    steps=12,
    X_imputation="last",
)  # None | "zero" | "last" | "mean" | "ar1_t"
```

| Value | Fill rule |
|-------|-----------|
| `None` (default) | Disabled — X passed through as-is |
| `"zero"` | Fill with `0` |
| `"last"` | Repeat the last observed value (random walk) |
| `"mean"` | In-sample column mean |
| `"ar1_t"` | Simulate from an AR(1) fitted by ML with Student-t innovations |

Columns containing no observed values are rejected when imputation is enabled.
Provide at least one observed value for each regressor you want to estimate.

`X_imputation` is applied only when the model's
`_needs_ragged_edge_imputation` class attribute is `True`. This is the default
for `ForecastModel` subclasses, so `RealTimeModel` applies the selected
strategy to their ragged-edge X data. When a model sets
`_needs_ragged_edge_imputation = False`, `RealTimeModel` does not apply
`X_imputation`; the model is responsible for handling its own ragged edge.
Models that determine publication availability and forecast dates from raw X
data themselves, such as the MIDAS family, use this setting. The flag does not
enable imputation unless `X_imputation` is also supplied.

## Data transformations

`data_transformation` maps each variable to the space the model is estimated in.
Forecasts are returned in that space and automatically back-transformed to
levels where possible (`reconstruct_levels=True` by default).

Models receive the transformed inputs described by the call-level
`data_transformation` or by a model-specific mapping. Forecast target values
use the same metric as their transformed target input, so no separate output
metric argument is needed:

```python
model = rt.models.ForecastOLS(
    data_transformation={"cpisa": "diff"},
)
```
When several models are compared, each model-specific mapping takes precedence
over the call-level fallback. Metrics are applied after forecasts are combined
and melted, so each source is reconstructed from its own fitted target metric.

Transformation frequency is inferred independently from each raw y/X column's
dates. The forecast horizon frequency is also inferred from the selected target
variables; pass `step_frequency` only when those variables have mixed or
ambiguous frequencies. It does not control input transformations. If a raw
column has an ambiguous frequency, provide it through the resolved
`input_frequencies` mapping passed to the model.

| Transform | Description |
|-----------|-------------|
| `"levels"` | Raw levels |
| `"pop"` | Period-on-period growth |
| `"yoy"` | Year-on-year growth |
| `"logs"` | Log levels |
| `"log diff"` | Log difference |
| `"diff"` | First difference |

## News decomposition

Set `decomp=True` to attribute each forecast revision to **news** (newly
released data), **reestimation** (parameter changes from refitting), and
**interaction** (the residual cross-term). Results are stored on
`rt_model.decompositions`, separately from the forecasts. Decomposition requires
the model to implement `_forecast_decomp()`; models without that method return
`None`.

```python
rt_model.forecast(
    y_variables=["cpisa"],
    X_variables=["gdpkp"],
    data_transformation={"cpisa": "pop", "gdpkp": "pop"},
    steps=12,
    decomp=True,
    X_imputation="last",
)
print(rt_model.decompositions)
```

See [forecasting_strategy.md](forecasting_strategy.md) for the full
methodology.

## Parallel execution

The vintage loop can run in parallel across models and vintage batches:

```python
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "levels"},
    steps=4,
    parallel=True,
)  # auto batch size
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "levels"},
    steps=4,
    parallel=True,
    max_workers=8,
)  # cap the worker count
```

With `parallel=True`, `ForecastTree` callable transforms and model instances
must be pickleable for `ProcessPoolExecutor`; module-level callables are the
usual choice. Sequential mode (`parallel=False`) also supports local
callables.
