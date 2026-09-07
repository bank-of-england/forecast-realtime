# Real-time Forecast Package

A Python package for real-time orchestration of forecasting models.

## Installation

```bash
pip install "forecast-realtime[models]"
```

## Import your data

Outturns use a long-form DataFrame with `date`, `frequency`, `variable`,
`value`, `vintage_date` and `metric` columns. This walkthrough uses the bundled
data generator; replace `sample_data` with your own DataFrame.

```python
import forecast_evaluation as fe
import forecast_realtime as rt

sample_data = rt.generate_synthetic_data(
    N=2,
    first_period="2015-01-31",
    endpoint="2024-12-31",
)
print(sample_data.head().to_string(index=False))
forecast_data = fe.NowcastData(outturns_data=sample_data)
```

The first five generated outturn rows are:

```text
        date frequency  variable      value vintage_date metric
2015-01-31         M monthly_1 101.577869   2024-01-31 levels
2015-01-31         M monthly_2 101.256092   2024-01-31 levels
2015-02-28         M monthly_1 101.293703   2024-01-31 levels
2015-02-28         M monthly_1  -0.279752   2024-01-31    pop
2015-02-28         M monthly_2  98.456195   2024-01-31 levels
```

## Use an existing model

```python
existing_model = rt.models.ForecastRidge(label="Ridge", cv=5, scale=True)
```

## Add your own model

Subclass `ForecastModel` and implement `_fit()` and `_forecast()`. `y` and `X`
arrive as pandas DataFrames.

```python
import numpy as np


class MyOLS(rt.ForecastModel):
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


custom_model = MyOLS(label="My OLS")
```

## Forecast

Pass one or more models to `RealTimeModel`, then call `forecast()`:

```python
rt_model = rt.RealTimeModel(
    data=forecast_data,
    models=[existing_model, custom_model],
)

rt_model.forecast(
    y_variables=["quarterly_1"],
    X_variables=["quarterly_2"],
    data_transformation={"quarterly_1": "pop", "quarterly_2": "pop"},
    steps=2,
    X_imputation="last",
    first_vintage="2024-01-31",
    last_vintage="2024-06-30",
)
print(rt_model.data.forecasts.head().to_string(index=False))
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