# Real-time Forecast Package

A Python package for real-time orchestration of forecasting models.

The main object, **RealTimeModel**, combines a **ForecastData** object (from the
`forecast_evaluation` package) with one or more **ForecastModel** objects.
`ForecastModel` is an abstract base class with
`_fit()`, `_forecast()` and an optional `_forecast_decomp()` method. All
built-in models inherit from it, and you can subclass it to wrap any Python (or
R, MATLAB, or Julia) forecasting model.

## Installation

```bash
pip install "forecast-realtime[models]"
```

## Quick demo

```python
import forecast_evaluation as fe
import forecast_realtime as rt

forecast_data = fe.ForecastData(load_fer=True)

ridge = rt.models.ForecastRidge(label="Ridge", cv=5, scale=True)
lasso = rt.models.ForecastLasso(label="LASSO", cv=5, scale=True)

rt_model = rt.RealTimeModel(
    data=forecast_data,
    models=[ridge, lasso],
)

rt_model.forecast(
    y_variables=["cpisa"],
    X_variables=["gdpkp"],
    data_transformation={"cpisa": "pop", "gdpkp": "pop"},
    steps=12,
    y_lags=4,
    X_imputation="last",
)

# Optional interactive dashboard:
# rt_model.data.run_dashboard()
```

### Marimo notebook

Install the notebook and model dependencies from the repository root:

```bash
pip install -e ".[models,notebooks]"
```

Open the demo as an editable notebook with visible code cells and outputs:

```bash
marimo edit notebooks/demo_models.py
```

## Add your own model

Subclass `ForecastModel` and implement `_fit()` and `_forecast()` (plus
`_forecast_decomp()` if you want news decompositions). `y` and `X` arrive as
pandas DataFrames, and the base class handles validation, lags, dummies and
forecast dates.

```python
import numpy as np
import pandas as pd

from forecast_realtime import ForecastModel


class MyOLS(ForecastModel):
    """Small OLS model showing the custom-model authoring pattern."""

    def _fit(self, y, X=None, **kwargs):
        # y and X are passed as pandas DataFrames
        if X is None:
            raise ValueError("MyOLS requires X")
        X = X.to_numpy(dtype=float)
        y = y.to_numpy(dtype=float)

        # OLS estimate: beta = (X'X)^-1 X'y
        self.beta = np.linalg.inv(X.T @ X) @ X.T @ y

        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        if X is None:
            raise ValueError("MyOLS requires future X")
        # ForecastModel passes the historical and future design rows.
        future_X = X.loc[X.index > self.last_y_fit_date].iloc[:steps]
        return future_X.to_numpy(dtype=float) @ self.beta
```

Pass it straight to `RealTimeModel`:

```python
rt_model = rt.RealTimeModel(data=forecast_data, models=[MyOLS()])
```

## Documentation

- [docs/index.md](docs/index.md) — how `ForecastModel` and `RealTimeModel` work.
- [docs/models.md](docs/models.md) — built-in models and R/MATLAB/Julia wrappers.
- [docs/usage.md](docs/usage.md) — lags, dummies, imputation, transformations,
  news decomposition and parallel execution.
- [adding_a_model.md](docs/adding_a_model.md) — the full `ForecastModel` interface.
- [forecasting_strategy.md](docs/forecasting_strategy.md) — forecasting methodology.
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup and workflow.

## Data Classification
Bank of England Data Classification: OFFICIAL BLUE