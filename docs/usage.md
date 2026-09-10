# Usage

The snippets below use the package's simulated mixed-frequency real-time data.
Each example creates a fresh `NowcastData` and `RealTimeModel` so its forecasts
do not affect the next example.

```python
import forecast_evaluation as fe
import forecast_realtime as rt

sample_data = rt.generate_synthetic_data(
    N=2,
    first_period="2015-01-31",
    endpoint="2024-12-31",
)
```

## Lags

The linear, tree and neural models support autoregressive (`y_lags`) and
distributed (`X_lags`) lags, supplied at forecast time:

```python
forecast_data = fe.NowcastData(outturns_data=sample_data.copy())
rt_model = rt.RealTimeModel(
    data=forecast_data,
    models=rt.models.ForecastRidge(cv=5, scale=True),
)
rt_model.forecast(
    y_variables=["quarterly_1"],
    X_variables=["quarterly_2"],
    data_transformation={"quarterly_1": "pop", "quarterly_2": "pop"},
    steps=2,
    y_lags=4,  # append y_{t-1} … y_{t-4}
    X_lags={"quarterly_2": 2},  # an int applies the same count to every regressor
    X_imputation="last",
    first_vintage="2024-01-31",
    last_vintage="2024-06-30",
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
forecast_data = fe.NowcastData(outturns_data=sample_data.copy())
rt_model = rt.RealTimeModel(
    data=forecast_data,
    models=rt.models.ForecastOLS(label="OLS"),
)
rt_model.forecast(
    y_variables=["quarterly_1"],
    data_transformation={"quarterly_1": "pop"},
    steps=2,
    dummies=["2020-06-30"],  # or {"outlier": "2020-06-30"}
    first_vintage="2024-01-31",
    last_vintage="2024-06-30",
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
forecast_data = fe.NowcastData(outturns_data=sample_data.copy())
rt_model = rt.RealTimeModel(
    data=forecast_data,
    models=rt.models.ForecastRidge(cv=5, scale=True),
)
rt_model.forecast(
    y_variables=["quarterly_1"],
    X_variables=["quarterly_2"],
    data_transformation={"quarterly_1": "pop", "quarterly_2": "pop"},
    steps=2,
    X_imputation="last",
    first_vintage="2024-01-31",
    last_vintage="2024-06-30",
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
    data_transformation={"quarterly_1": "diff"},
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
forecast_data = fe.NowcastData(outturns_data=sample_data.copy())
rt_model = rt.RealTimeModel(
    data=forecast_data,
    models=rt.models.ForecastRidge(cv=5, scale=True),
)
rt_model.forecast(
    y_variables=["quarterly_1"],
    X_variables=["quarterly_2"],
    data_transformation={"quarterly_1": "pop", "quarterly_2": "pop"},
    steps=2,
    decomp=True,
    X_imputation="last",
    first_vintage="2024-01-31",
    last_vintage="2024-06-30",
)
print(rt_model.decompositions)
```

See [forecasting_strategy.md](forecasting_strategy.md) for the full
methodology.

## Model-owned source conditioning

Use the model's `conditioning` argument when models in one run need different
sources or durations. The resolver handles each model's policy independently.
`periods` counts from the first forecast period, so it is positive and
inclusive in ordinary language (`periods=3` means periods 1 to 3). The older
run-level `y_steps_ahead` and `X_steps_ahead` arguments keep their zero-based
inclusive convention (`0` means one period).

The precedence rules are deliberately strict: `conditioning=None` inherits the
run fallback, a non-empty mapping replaces it completely, and `conditioning={}`
disables external conditioning. `None` and `{}` are also distinct in the
legacy source and horizon mappings. Treat a missing source label as invalid.
When a known source has no path for a variable or vintage, preserve the existing
missing-path behaviour.

Only models that explicitly opt in to target conditioning can accept explicit
future y constraints. Recursive regression and tree models without a supporting
root reject unsupported-y conditioning rather than ignoring it. Direct model
calls do not consult conditioning policies: they use the explicitly supplied frames.

For the tree-specific routing rules and the `ForecastContext.y_published`
migration, see [ForecastTree](forecast_tree.md) and [Adding a New Model](adding_a_model.md).

## Parallel execution

The vintage loop can run in parallel across models and vintage batches:

```python
if __name__ == "__main__":
    forecast_data = fe.NowcastData(outturns_data=sample_data.copy())
    rt_model = rt.RealTimeModel(
        data=forecast_data,
        models=rt.models.ForecastOLS(label="OLS"),
    )
    rt_model.forecast(
        y_variables=["quarterly_1"],
        data_transformation={"quarterly_1": "levels"},
        steps=2,
        y_lags=4,
        parallel=True,
        max_workers=2,
        first_vintage="2024-01-31",
        last_vintage="2024-06-30",
    )
```

With `parallel=True`, `ForecastTree` callable transforms and model instances
must be pickleable for `ProcessPoolExecutor`; module-level callables are the
usual choice. Sequential mode (`parallel=False`) also supports local
callables.
