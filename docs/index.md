# Real-time Forecasting

`forecast_realtime` produces forecasts from historical data vintages. The Bank
of England develops the project on top of the `forecast_evaluation` package.

---

## Overview

The package replays a forecasting exercise across historical vintages. At each
vintage, it extracts the available data, prepares model inputs, fits one or
more models, and records forecasts with their vintage metadata.

The framework supports ordinary Python models, mixed-frequency MIDAS models,
forecast trees, data transformations, conditioning paths, and wrappers for
models implemented in R, MATLAB, or Julia.

### Key features

- **Vintage-aware forecasting** — fit and forecast models using only the data
  available at each historical vintage.
- **A common model contract** — use the same `fit()` and `forecast()` interface
  for built-in and custom models.
- **Input preparation** — apply lags, dummies, formulas, metric selection, and
  transformations before model estimation.
- **Mixed-frequency models** — use MIDAS models that own ragged-edge alignment
  and forecast-date construction.
- **Forecast trees** — combine model leaves with fixed callable transforms or
  learned `ForecastModel` transforms.
- **News decompositions** — attribute revisions to new data, reestimation, and
  their interaction when a model supports decomposition.

---

## Installation

### Install from PyPI

```bash
pip install forecast_realtime
```

Install the model integrations you need with an extra:

```bash
pip install "forecast_realtime[models]"
```

### Set up a development environment

```bash
git clone https://github.com/bank-of-england/forecast-realtime.git
cd forecast-realtime
conda create --name forecast-realtime
conda activate forecast-realtime
conda install pip
pip install -e ".[dev,docs]"
pre-commit install
```

Verify the installation with:

```bash
pytest
zensical build --strict
```

Two classes organize the package: **ForecastModel** and **RealTimeModel**.

## ForecastModel

`ForecastModel` defines the contract for every forecasting model. Its public
`fit()` and `forecast()` methods validate data and construct design matrices;
subclasses implement `_fit()` and `_forecast()`.

```
ForecastModel (ABC)
│
├── fit(y, X, y_lags, X_lags, dummies, **kwargs)   # public — builds lags/dummies, validates
│   └── _fit(y, X, **kwargs)                       # abstract — subclass implements this
│
├── forecast(steps, X, y, decomp, **kwargs)        # public — validates inputs/outputs
│   ├── _forecast(steps, X, y, **kwargs)           # abstract — subclass implements this
│   └── _forecast_decomp(steps, X, y, **kwargs)    # optional — components, or None
│
└── test(checks)                                   # concrete — diagnostic checks
```

The base class provides this validation:

- `fit()` checks that `y` (and `X`, if supplied) is a `pd.DataFrame`, builds the
  lagged design matrix and dummies, then stores `self.y` after fitting.
- `forecast()` checks that `steps` is a positive integer and verifies the output
  is a `pd.DataFrame` of shape `(steps, n_variables)` indexed by a
  `DatetimeIndex` (one date per horizon).

`_fit()` and `_forecast()` always receive `y` and `X` as pandas DataFrames.
See [adding_a_model.md](adding_a_model.md) for the full interface.

## RealTimeModel

`RealTimeModel` combines a `ForecastData` object from `forecast_evaluation`
with one or more `ForecastModel` instances. The `models` argument accepts one
model or a list. The class then runs each model across historical data vintages.

```
RealTimeModel(data, models)
│
└── forecast(y_variables, data_transformation, frequency, label, steps, ...)
        │
        For each model, for each vintage date in the dataset:
            1. Extract data available at that vintage (y and optionally X)
            2. Apply data_transformation to outturns and forecasts
            3. Deep-copy the ForecastModel (avoid state leaking)
            4. model.fit(y=y_vintage, X=X_fit, y_lags=y_lags, X_lags=X_lags)
            5. Build conditioning paths:
               - y_forecasts (pd.DataFrame) from y_steps_ahead/y_sources
               - X_forecast (pd.DataFrame) from X_steps_ahead/X_sources
            6. model.forecast(steps=steps, X=X_forecast, y=y_forecasts)
            7. If decomp=True, capture the news breakdown
            8. Store forecasts with vintage_date and forecast_horizon
        │
        ├── Adds forecasts to data.forecasts (ready for evaluation)
        └── Adds decompositions to rt_model.decompositions (when decomp=True)
```

The vintage loop runs sequentially by default; pass `parallel=True` to run
models and vintage batches concurrently.

---

## Package structure

```
src/forecast_realtime/
├── forecast_model.py       # ForecastModel contract and result validation
├── real_time_model.py      # Vintage loop and forecast orchestration
├── forecast_tree.py        # Tree-based forecast composition
├── data_transformation.py  # Input metrics and transformation pipelines
├── formula.py              # Formula-based variable selection
├── external_model.py       # R, MATLAB, and Julia wrappers
└── models/                 # Built-in model implementations
```

---

## Contents

- [API Reference](api.md) — public classes and built-in model interfaces.
- [Usage](usage.md) — lags, dummies, imputation, transformations, news
  decomposition, and parallel execution.
- [Models](models.md) — built-in models and R, MATLAB, and Julia wrappers.
- [Forecast trees](forecast_tree.md) — compose model leaves and transforms.
- [Adding a model](adding_a_model.md) — the complete `ForecastModel` contract.
- [Forecasting strategy](forecasting_strategy.md) — forecasting methodology.
- [Dummy strategy](dummies_strategy.md) — outlier dummy handling.
- [Input metric transformations](input_metric_transformation_plan.md) — input
  metric selection and transformation design.
