# Forecasting Strategy

This document explains *how* `forecast_realtime` produces forecasts and the
strategic choices available to you when running a real-time exercise. For the
API quick-start see the [Home](index.md); for the model interface see
[adding_a_model.md](adding_a_model.md); for the package reference see
[API Reference](api.md).

## Real-time forecasting

A real-time exercise replays history as it was actually observed. For every
historical *vintage* (the data set as it existed on a given date),
`RealTimeModel` refits the model on the data available at that point and
produces a forecast. This avoids look-ahead bias: a forecast dated 2015-Q1
uses only data available in 2015-Q1.

```
For each model, for each vintage:
    extract data at vintage → transform → fit → forecast → store
```

Because each vintage receives a fresh deep copy of the model, vintages do not
share state.

## Multi-step strategies: recursive vs direct

Linear and ML models support two strategies for forecasting more than one step
ahead, controlled by the `forecast_strategy` constructor argument.

### Recursive (default)

A single model is estimated and rolled forward: the horizon-1 forecast is fed
back in as a lagged input to produce horizon 2, and so on. This is parsimonious
and uses one set of parameters for all horizons.

```python
model = rt.models.ForecastOLS(forecast_strategy="recursive")
```

### Direct

A separate model is estimated for each horizon (`y_{t+h}` regressed on the
features available at `t`). This can be more robust at longer horizons but
needs more data. The number of horizons must be declared up front via `steps`:

```python
model = rt.models.ForecastOLS(forecast_strategy="direct", steps=8)
```

|                    | Recursive                     | Direct                  |
| ------------------ | ----------------------------- | ----------------------- |
| Models estimated   | 1                             | one per horizon         |
| Parameters         | shared across horizons        | horizon-specific        |
| Data efficiency    | higher                        | lower                   |
| Error accumulation | compounds through feedback    | independent per horizon |

## Lags

Autoregressive and distributed lags are part of the forecasting strategy and
are supplied at fit/forecast time (not in the constructor):

- `y_lags=k` adds `y_{t-1} … y_{t-k}` as features (`_y_lag1 … _y_lagk`).
- `X_lags=k`, or a `{col: k}` dict for per-regressor counts, adds lagged
  regressors (`col_lag1 … col_lagk`).

During recursive forecasting the model rolls these lags forward automatically.

## Conditioning on known future paths

Sometimes you know (or assume) the future path of the target itself for the
first few horizons — for example, a published nowcast for the current quarter.
Provide these via `y_steps_ahead` (how many horizons to condition) and
`y_sources` (which forecast source to read):

```python
rt_model.forecast(
    y_variables=["cpisa", "unemp"],
    data_transformation={"cpisa": "levels", "unemp": "levels"},
    steps=4,
    y_steps_ahead={"cpisa": 1, "unemp": 0},  # horizons to condition on
    y_sources={"cpisa": "mpr", "unemp": "mpr"},  # source of those values
)
```

The model is then conditioned on those values when producing the remaining
horizons (used, for example, by the conditional BVAR).

## Regressor forecast paths

Exogenous regressors (`X_variables`) also need values over the forecast
horizon. There are two options:

1. **Supply a forecast path** via `X_steps_ahead` (how many horizons of the
   regressor to use) and `X_sources` (which forecast source to read) — e.g.
   oil futures or consensus FX.
2. **Let the model iterate** using the regressor's own lags. When no future
  value is available, `X_imputation` fills it when supplied.
  The default is no imputation (`None`); linear models require complete future
  regressors unless an imputation strategy is selected.

```python
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "levels", "oil": "levels", "fx": "levels"},
    steps=11,
    X_variables=["oil", "fx"],
    X_steps_ahead={"oil": 10, "fx": 10},
    X_sources={"oil": "futures", "fx": "consensus"},
)
```

Conditioning (`y_*`) and regressor paths (`X_*`) can be used together: the `X`
arguments feed the design matrix, while the `y` arguments condition the target.

### Model-owned conditioning

When several models in one real-time run need different conditioning sources,
put the policy on each model. This keeps the source data, vintages and forecast
horizon shared while allowing each specification to be compared fairly:

```python
models = [
  rt.models.ForecastBVAR(label="baseline", conditioning={}),
  rt.models.ForecastBVAR(
    label="source_a_first_period",
    conditioning={"y": {"gdp": {"source": "A", "periods": 1}}},
  ),
  rt.models.ForecastBVAR(
    label="source_b_first_three_periods",
    conditioning={"y": {"gdp": {"source": "B", "periods": 3}}},
  ),
]
rt_model = rt.RealTimeModel(data=forecast_data, models=models)
rt_model.forecast(y_variables=["gdp"], steps=4)
```

Here `periods` is a positive count: `1` constrains the first forecast period
and `3` constrains the first three. A model setting of `None` (or an omitted
setting) inherits the run-level `y_sources`/`X_sources` fallback. A non-empty
setting replaces that fallback as a whole; it is not merged by role or
variable. An explicit `{}` disables external conditioning for both roles.
Omitting a role from a replacement policy also selects no external source for
that role, while ordinary published observations, future X availability and
the configured X imputation remain available.

Conditioning does not introduce per-model vintage ranges or output horizons:
those remain shared run controls.

The legacy run-level arguments retain their inclusive, zero-based meaning:
`y_steps_ahead={"gdp": 0}` selects one period and `2` selects three. An omitted
mapping and an explicit empty mapping are distinct, including for X paths, so
do not collapse them with truthiness. A legacy horizon of `None` still selects
its source without imposing an explicit target constraint.

Conditioning policies select source paths only during realtime orchestration.
Direct `model.forecast(y=..., X=...)` and `model.predict(...)` calls continue to
use the frames supplied by the caller: they do not look up sources or
clip those frames with the model's policy. Models that cannot enforce target
conditioning reject an explicit future y constraint; historical rows, empty
frames and all-NaN future paths do not constitute a constraint. See
[Adding a New Model](adding_a_model.md) for the author contract and context
migration.

## Forecast horizons

- `forecast_horizon=0` is the first information horizon returned by the model;
  calendar distance is recorded separately in `target_minus_vintage`.
- `steps` sets how many horizons to produce.
- `target_minus_vintage` is the calendar distance from the forecast vintage to
  the target period and may be negative for a valid nowcast.
- `first_forecast_horizon` (an int, or a per-variable dict) is deprecated. It
  is retained only as a calendar-relative fitting/output cutoff for
  compatibility; omit it for the default latest-target fit.
- Legacy conditioning and regressor horizons must be less than `steps`;
  model-owned `periods` must not exceed `steps`.

## Data transformations and level reconstruction

Each variable is modelled in the space given by `data_transformation`:

| Transform   | Description                |
| ----------- | -------------------------- |
| `"levels"`  | Raw levels                 |
| `"pop"`     | Period-on-period growth    |
| `"yoy"`     | Year-on-year growth        |
| `"logs"`    | Log levels                 |
| `"log diff"`| Log difference             |
| `"diff"`    | First difference           |

Models always return forecasts in that same space, and `RealTimeModel`
reconstructs levels afterwards (`reconstruct_levels=True` by default) when a
levels outturn is available. This keeps model code simple — a model never has
to worry about back-transformation.

Each model returns target forecasts in the same metric space as its transformed
target input. A model-specific `data_transformation` mapping takes precedence
over the call-level mapping supplied to `RealTimeModel.forecast()`. This allows
models compared in one run to use different metrics for the same variable;
`RealTimeModel` applies each fitted model's target metric and reconstructs each
source independently.

Forecast trees follow a root-specific rule: callable roots default to
`"levels"`, while model-backed roots inherit the wrapped model's fitted target
metric.

## News decomposition

When `decomp=True`, every forecast revision between consecutive vintages is
attributed to three sources:

```
news          = forecast(old params, new data)  − previous forecast
reestimation  = forecast(new params, old data)  − previous forecast
interaction   = total revision − news − reestimation
```

- **news** — the effect of newly released or revised data.
- **reestimation** — the effect of parameters changing when the model is refit.
- **interaction** — the residual cross-term combining both effects.

The first forecast for a horizon is a *level* decomposition (each component's
contribution sums to the forecast); subsequent vintages produce *revision*
decompositions. Results are written to `rt_model.decompositions`, separately
from the forecasts. Only models that implement `_forecast_decomp()` produce a
decomposition; others return `None`.

## Parallel execution

The vintage loop runs sequentially by default. Pass `parallel=True` to run the
cartesian product of models and vintage batches concurrently; `batch_size` is
auto-tuned (override it explicitly if needed) and `max_workers` caps the number
of workers.

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

## Choosing a strategy

- **Short series / want parsimony** → recursive.
- **Longer horizons with enough data** → consider direct.
- **Known near-term path of the target** → condition with
  `y_steps_ahead` / `y_sources`.
- **Have external regressor forecasts** → supply `X_steps_ahead` / `X_sources`;
  otherwise rely on lags plus `X_imputation`.
- **Need to explain revisions** → enable `decomp=True`.
- **Large vintage span** → enable `parallel=True`.
