# Adding a New Model

This guide explains how to integrate a new forecasting model into the ecosystem. All model integration is done through the **`forecast_realtime`** package, which provides the base class, orchestration and caching infrastructure.

---

## Where to Put the Code

Put your model in your own project; you do not need to edit or install files
inside `forecast_realtime`. To try the Python moving-average example below,
create this file:

```text
my_forecast_project/
└── run_forecast.py
```

Paste both Python blocks from [Example: Moving Average in
Python](#example-moving-average-in-python) into `run_forecast.py`: put the
`MovingAverage` class first and the real-time forecast code after it. Then run:

```console
cd my_forecast_project
python run_forecast.py
```

For a larger project, move the class to a separate module and import it into
`run_forecast.py`. No package registration is required. Models written in R,
MATLAB, or Julia need an additional external script; [Where to Put an External
Script](#where-to-put-an-external-script) explains how to supply its path.

---

## Interface Requirements

Any model that participates in the ecosystem must subclass `ForecastModel` and provide three things:

### `__init__(...)` — Configuration

Store all hyperparameters and settings that `_fit` and `_forecast` will need. This is the only place to accept user-facing arguments (e.g. regularisation strength, window size, number of estimators). It should also initialise any placeholders for fitted state (e.g. `self.model = None`) so the object is fully described before any data are seen.

Always call `super().__init__(label=label, formula=formula)` to register the
model label and optional transformation mapping when your model exposes it:

```python
def __init__(
    self,
    my_param=1.0,
    label=None,
    formula=None,
    data_transformation=None,
):
    super().__init__(
        label=label,
        formula=formula,
        data_transformation=data_transformation,
    )
    self.my_param = my_param
```

- **`label`**: string tag attached to all forecasts produced by this model instance. Defaults to the class name. Overridden by `RealTimeModel.forecast(label=...)`.
- **`formula`**: R-style formula string (e.g. `"cpisa ~ gdpkp + unemp"` or `"cpisa ~ ."`) that selects which y and X columns are used. Applied after lag augmentation. `None` = use all columns.
- **`data_transformation`**: optional plain mapping from each input variable to
    the metric used by this model. It takes precedence over the call-level
    `data_transformation` mapping. Forecasts use the same metric as their target
    input; `RealTimeModel` reconstructs levels where possible.

> **Lag features are not constructor parameters.** `y_lags` and `X_lags` are passed to `ForecastModel.fit()` (or `RealTimeModel.forecast()`), which builds the lagged design matrix before calling `_fit()`. Do not handle lag construction in `__init__` or `_fit()`.

### `_fit(y, X=None, **kwargs)` — Estimation

Estimate the model using historical data (`y`), for example by fitting regression coefficients, training tree splits, or computing summary statistics from the observed time series. After this call, the model should be ready to produce forecasts.

`_fit()` receives the **already-processed** design matrix. If `y_lags` or `X_lags` were specified, `ForecastModel.fit()` has already appended the lag columns to `X` and dropped NaN rows before calling `_fit()`. Do not call `build_lagged_design` or do any lag construction inside `_fit()`.

**Inputs:**

| Argument | Type | Description |
|----------|------|-------------|
| `y` | `pd.DataFrame` | Target variable(s). **Index:** `DatetimeIndex`. **Values:** already transformed (e.g. growth rates). When lags are used this is the NaN-dropped aligned target; otherwise the full training history. |
| `X` | `pd.DataFrame` or `None` | Design matrix, potentially augmented with lag columns. Column order: base X cols, then `_y_lag1…_y_lagk`, then `col_lag1…col_lagk` per X column. `None` if no regressors and `y_lags=0`. |
| `**kwargs` | | Extra keyword arguments forwarded from `RealTimeModel.forecast(..., **kwargs)`. `y_lags` and `X_lags` are **not** present here — they are consumed by `ForecastModel.fit()`. |

**Example `y` (quarterly, single variable, `data_transformation={"cpisa": "pop"}`):**

```text
              cpisa
date
2014-03-31    0.6
2014-06-30    0.8
2014-09-30    0.5
2014-12-31    0.9
2015-03-31    0.7
```

**Example `y` (quarterly, multivariate):**

```text
              cpisa    gdpkp
date
2014-03-31    0.6      0.4
2014-06-30    0.8      0.3
2014-09-30    0.5      0.7
2014-12-31    0.9      0.2
```

**Must return** `self`.

---

### `_forecast(steps, X=None, y=None, **kwargs)` — Forecasting

Produce multi-step-ahead forecasts using the fitted model. This method is called after `_fit()` and should return predicted values for the next `steps` periods. Each row of the output corresponds to a forecast horizon (row 0 is the nowcast of the current period, row 1 is one period ahead, and so on).

**Inputs:**

| Argument | Type | Description |
|----------|------|-------------|
| `steps` | `int` | Number of periods ahead to forecast (always ≥ 1). |
| `X` | `pd.DataFrame` or `None` | Design matrix over history **and** the forecast horizon, indexed by a `DatetimeIndex`; the last `steps` rows hold the regressor forecasts. Column order matches the `X` passed to `_fit`. `None` if no `X_variables` (or no `X_cond_variables`) were specified. |
| `y` | `pd.DataFrame` or `None` | Target history plus conditioning paths over the horizon, indexed by a `DatetimeIndex`. Column order matches the `y` passed to `_fit`. Entries set to `NaN` are unconstrained; non-NaN entries pin that variable/horizon to an externally supplied value (e.g. MPR projections). `None` if no `y_cond_variables` were specified. |
| `**kwargs` | | Additional keyword arguments. |

**Output:**

Must return a `pd.DataFrame` of shape `(steps, n_y_variables)`:
- **Index:** a `pd.DatetimeIndex` (name `"date"`) of length `steps`, one date per horizon. The subclass must provide this index. AR-style models can call `self._wrap_forecast(arr, steps)` to wrap an `(steps, n_vars)` ndarray with dates inferred from `self.y.index`; mixed-frequency models (e.g. MIDAS) build the DataFrame with their own anchor dates.
- **Rows** correspond to forecast horizons 0, 1, …, steps−1 (horizon 0 = nowcast of the current period).
- **Columns** must match the order and count of columns in the `y` DataFrame that was passed to `_fit()`.
- Values must be in the metric declared by the model's `data_transformation`
    (or the call-level fallback). `RealTimeModel` handles back-transformation to
    levels automatically; model code must not back-transform its own output.

**Example output for `steps=4`, 1 variable:**

```python
pd.DataFrame(
    [[0.7], [0.6], [0.5], [0.4]],
    index=pd.date_range("2024-03-31", periods=4, freq="QE", name="date"),
    columns=["cpisa"],
)  # shape: (4, 1)
```

**Example output for `steps=4`, 2 variables:**

```python
pd.DataFrame(
    [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.4, 0.6]],
    index=pd.date_range("2024-03-31", periods=4, freq="QE", name="date"),
    columns=["cpisa", "gdpkp"],
)  # shape: (4, 2)
```

The base class validates that the return value is a `DataFrame` with a `DatetimeIndex`, exactly `steps` rows, and the same number of columns as the fitted `y`.

---

### `_forecast_decomp(steps, X=None, y=None, **kwargs)` — Forecast Decomposition (Optional)

Break down forecast revisions into interpretable components. When a forecast is updated between data vintages, the revision can be split into:

- **News**: revision from new data released
- **Reestimation**: revision from model refit (parameter changes, not new data)
- **Interaction**: cross-term combining both effects

This method is **optional**. If not implemented, return `None` and the model will not produce decompositions.

**Inputs:**

| Argument | Type | Description |
|----------|------|-------------|
| `steps` | `int` | Number of periods ahead to forecast (same as `_forecast`). |
| `X` | `pd.DataFrame` or `None` | Full augmented design matrix (same as passed to `_forecast`). |
| `y` | `pd.DataFrame` or `None` | Conditioning paths (same as passed to `_forecast`). |
| `**kwargs` | | Additional keyword arguments. |

**Output (minimal contract):**

Return `pd.DataFrame` or `None`:

- If decomposition not supported: return `None`
- If decomposition computed, one row per component per horizon:
  - `forecast_horizon` (int): 0-based horizon index
  - `component` (str): name of the component (e.g. `'intercept'`, `'gdpkp'`, `'cpisa_lag1'`)
  - `contribution` (float): additive effect — values must sum to the total forecast for each horizon
  - `weight` (float or NaN): model coefficient (NaN if not applicable, e.g. black-box models)

`RealTimeModel` augments these rows with metadata (`variable`, `date`, `vintage_date`, `frequency`, `source`, `forecast_metric`, `decomposition`, `revision_source`, `base_vintage_date`) before storing in `rt_model.decompositions`. The model does **not** need to return these columns.

**Example output (OLS with 2 regressors + intercept, `steps=4`):**

```python
pd.DataFrame(
    {
        "forecast_horizon": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        "component": ["intercept", "payrolls", "ip"] * 4,
        "contribution": [-0.9, 0.5, 0.06] * 4,
        "weight": [np.nan, 0.5, 0.1] * 4,
    }
)  # 12 rows × 4 cols; each horizon has 3 components
```

**Important notes:**

- `forecast_horizon` is 0-based (0 = nowcast)
- `contribution` values **must sum to the total forecast** for each horizon
- `weight` can be `NaN` for non-parametric or black-box models
- Simple models (e.g. moving average) can return `None` and skip decomposition
- Do **not** include `news`, `revision_source`, `vintage_date`, or other metadata — `RealTimeModel` adds those

---

You can also implement the model in another language; see [Language Interoperability](#language-interoperability) below.

---

## Example: Moving Average in Python

A moving average model forecasts every horizon as the mean of the last `window_size` observations. This is the simplest useful example to illustrate the interface.

### Step 1: Subclass `ForecastModel`

```python
import numpy as np
import pandas as pd
from forecast_realtime import ForecastModel


class MovingAverage(ForecastModel):
    """Moving-average forecast: predict the mean of the last `window_size` observations.

    Parameters
    ----------
    window_size : int
        Number of trailing observations to average.
    """

    def __init__(self, window_size: int = 4, label=None):
        super().__init__(label=label)
        self.window_size = window_size
        self.window_mean = None  # Has shape (n_variables,) after fitting.

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame = None, **kwargs):
        """Compute the mean of the last `window_size` rows of y.

        Parameters
        ----------
        y : pd.DataFrame
            DatetimeIndex, one column per variable.
            Values are already transformed (e.g. growth rates).
        X : ignored
        """
        tail = y.iloc[-self.window_size :]  # last window_size rows
        self.window_mean = tail.mean().values  # Has shape (n_variables,).
        return self

    def _forecast(
        self,
        steps: int,
        X: np.ndarray = None,
        y: np.ndarray = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Return the moving-average value for every horizon.

        Parameters
        ----------
        steps : int
            Number of horizons to forecast.
        X, y : ignored

        Returns
        -------
        pd.DataFrame, shape (steps, n_variables)
            Same value repeated for each step, indexed by a DatetimeIndex
            of length ``steps`` (built by ``_wrap_forecast`` from
            ``self.y.index``).
        """
        # Tile the mean across all forecast horizons, then wrap with the
        # standard inferred-date DataFrame.
        arr = np.tile(self.window_mean, (steps, 1))
        return self._wrap_forecast(arr, steps)

    def _forecast_decomp(
        self,
        steps: int,
        X: np.ndarray = None,
        y: np.ndarray = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Return decomposition: for moving average, all contribution from 'window_mean' component."""
        components = []
        for h in range(steps):
            for var_idx, var_col in enumerate(self.y.columns):
                components.append(
                    {
                        "forecast_horizon": h,
                        "component": "window_mean",
                        "contribution": self.window_mean[var_idx],
                        "weight": np.nan,
                    }
                )
        return pd.DataFrame(components)
```

### Step 2: Run Real-time Forecasts

```python
import forecast_evaluation as fe
import forecast_realtime as rt

forecast_data = fe.ForecastData(load_fer=True)

ma_model = MovingAverage(window_size=4)
rt_model = rt.RealTimeModel(data=forecast_data, models=ma_model)

# Run forecasts (optionally with decomposition)
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "pop"},
    steps=8,
    label="MA(4)",
    first_vintage="2015-01-01",
    decomp=False,  # Set to True to enable decomposition
)

# Optional interactive dashboard:
# rt_model.data.run_dashboard()

# If decomp=True, access decompositions:
# print(rt_model.decompositions)  # DataFrame with component breakdown
```

---

## Example: OLS with Decomposition

Ordinary Least Squares (OLS) forecasts with interpretable component decomposition. This example shows how `_forecast_decomp()` breaks down forecast revisions into data news, parameter reestimation, and interaction effects.

### Step 1: Subclass `ForecastModel` with decomposition support

```python
import numpy as np
import pandas as pd
from forecast_realtime import ForecastModel
from sklearn.linear_model import LinearRegression


class SimpleOLS(ForecastModel):
    """OLS regression with forecast decomposition support.

    Parameters
    ----------
    fit_intercept : bool
        Whether to include an intercept term.
    """

    def __init__(self, fit_intercept: bool = True, label=None, formula=None):
        super().__init__(label=label, formula=formula)
        self.fit_intercept = fit_intercept
        self.model = None
        self.intercept_ = None
        self.coef_ = None

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame = None, **kwargs):
        """Fit OLS to y and X."""
        if X is None or X.shape[1] == 0:
            raise ValueError("SimpleOLS requires X_variables")

        self.model = LinearRegression(fit_intercept=self.fit_intercept)
        self.model.fit(X, y.values)
        self.intercept_ = self.model.intercept_
        self.coef_ = self.model.coef_
        return self

    def _forecast(self, steps: int, X=None, y=None, **kwargs) -> pd.DataFrame:
        """Forecast using OLS: y = intercept + X @ coef."""
        if X is None:
            raise ValueError("SimpleOLS requires X (future regressors)")

        future_X = X.loc[X.index > self.last_y_fit_date].iloc[:steps]
        forecasts = future_X.to_numpy(dtype=float) @ self.coef_.T + self.intercept_
        return forecasts

    def _forecast_decomp(self, steps: int, X=None, y=None, **kwargs) -> pd.DataFrame:
        """Decompose forecast into intercept + regressor components.

        Returns one row per component per horizon with columns:
        forecast_horizon, component, contribution, weight.
        """
        if X is None:
            return None

        components = []
        future_X = X.loc[X.index > self.last_y_fit_date].iloc[:steps]
        X_cols = list(future_X.columns)

        for h in range(steps):
            # Intercept contribution
            components.append(
                {
                    "forecast_horizon": h,
                    "component": "intercept",
                    "contribution": float(self.intercept_),
                    "weight": np.nan,
                }
            )

            # Regressor contributions
            for col_idx, col_name in enumerate(X_cols):
                x_value = future_X.iloc[h, col_idx]
                contribution = float(self.coef_[col_idx]) * x_value
                components.append(
                    {
                        "forecast_horizon": h,
                        "component": col_name,
                        "contribution": contribution,
                        "weight": float(self.coef_[col_idx]),
                    }
                )

        return pd.DataFrame(components)
```

### Step 2: Run OLS with decomposition enabled

```python
import forecast_evaluation as fe
import forecast_realtime as rt

forecast_data = fe.ForecastData(load_fer=True)

ols_model = SimpleOLS(fit_intercept=True)
rt_model = rt.RealTimeModel(data=forecast_data, models=ols_model)

rt_model.forecast(
    y_variables=["cpisa"],
    X_variables=["gdpkp", "unemp"],
    data_transformation={"cpisa": "pop", "gdpkp": "pop", "unemp": "levels"},
    steps=12,
    label="OLS",
    first_vintage="2015-01-01",
    X_imputation="last",
    decomp=True,  # Enable decomposition
)

# Access decomposition results
print(rt_model.decompositions)
# Output includes forecast_horizon, variable, date, vintage_date, frequency,
# source, forecast_metric, decomposition, revision_source, base_vintage_date,
# component, contribution and weight columns.
# Shows how each regressor + intercept contributed to each horizon's forecast
```

---

## Language Interoperability

Models written in R, Julia, MATLAB, or another language integrate through the
provided classes in `forecast_realtime`:

| Language | Class          | CLI executable |
|----------|----------------|----------------|
| R        | `RModel`       | `Rscript`      |
| MATLAB   | `MATLABModel`  | `matlab`       |
| Julia    | `JuliaModel`   | `julia`        |

All three inherit from `ExternalModel`. It manages the temporary directory,
Parquet I/O, parameter deserialisation, command dispatch, subprocess
execution, and forecast output. You provide the model logic.

External scripts are trusted executable code. They are launched as subprocesses
without a shell, and generated path literals are quoted for the target
language, but this does not sandbox the script or its parameters. Fable
`spec` and `xreg` values have the same trusted R-expression contract.

**You only write two functions:** `fit(y, X, params)` → returns a model object, and `forecast(model, steps, X, y, params)` → returns a data frame / matrix. The argument order mirrors `ForecastModel._fit` / `_forecast`, with `model` standing in for `self` and `params` for `**kwargs`. `X` is the regressors (`NULL` / `[]` / `nothing` when there are none); at forecast time it holds the future regressor values (one row per step). Both `y` and `X` include a `date` column containing their pandas index. Use this column to align time-series data, and exclude it from numerical regressors unless the model explicitly uses dates.

### What the Package Handles for You

1. `fit()` writes `y.parquet` (and optionally `X.parquet`) to a temporary directory, loads `y` and `X` into data frames (`X` is `NULL` / `[]` / `nothing` when absent), deserialises your keyword arguments into `params`, calls your `fit(y, X, params)` function, and **saves the returned model object** to disk (`model.rds` / `model.mat` / `model.jls`). Their pandas indexes are stored as a `date` column.
2. `forecast()` loads `y` and the future `X`, **deserialises the saved model**, calls your `forecast(model, steps, X, y, params)` function, takes the returned data frame / matrix and **writes it to `forecasts.parquet`**, then returns the result as a `pd.DataFrame` (the base class wraps it with the standard inferred-date `DatetimeIndex`).
3. The temporary directory is **automatically deleted** when the model object is garbage-collected.

Your functions never touch `cache_dir`, `saveRDS`, `write_parquet`, or any other file I/O — the runner scripts handle all of that.

### Where to Put an External Script

Save the external script wherever you keep your model code. When you create an
`RModel`, `MATLABModel`, or `JuliaModel`, pass the path to that script. The
script does not have to share a directory with your Python file.

Keeping both files together is a simple option. For the R example below, the
project would look like this:

```text
my_forecast_project/
├── run_forecast.py
└── ma_model.R
```

Use `ma_model.m` for MATLAB or `ma_model.jl` for Julia. Put the Python wrapper
code in `run_forecast.py`, and build the script path from that file's location:

```python
from pathlib import Path

script = Path(__file__).resolve().with_name("ma_model.R")
model = RModel(str(script), window_size=4)
```

This path works whether you run `python run_forecast.py` from the project
directory or invoke the file from elsewhere. A bare path such as
`"ma_model.R"` depends on the process's current working directory and may fail
later when R, MATLAB, or Julia starts.

If you keep external scripts in a subdirectory, include it in the path:

```text
my_forecast_project/
├── run_forecast.py
└── models/
    └── ma_model.R
```

```python
script = Path(__file__).resolve().parent / "models" / "ma_model.R"
model = RModel(str(script), window_size=4)
```

In a notebook, `__file__` is unavailable. Define the project directory
explicitly and build the path from it:

```python
project_dir = Path("/absolute/path/to/my_forecast_project")
model = RModel(str(project_dir / "ma_model.R"), window_size=4)
```

The package neither searches for the script nor copies it into your project.
The examples below use the two-file layout above. For a complete MATLAB wrapper,
see `tests/models/matlab_scripts/demo_forecast_lm_matlab.py`.

### Function Signatures Your Script Must Define

| Language | `fit` | `forecast` |
|----------|-------|------------|
| R | `fit(y, X, params)` → returns a model object (e.g. a list) | `forecast(model, steps, X, y, params)` → returns a `data.frame` |
| MATLAB | `result = my_model('fit', y, X, params)` → returns a struct | `result = my_model('forecast', model, steps, X, y, params)` → returns a table |
| Julia | `fit(y, X, params)` → returns any serialisable object | `forecast(model, steps, X, y, params)` → returns a `DataFrame` |

---

## Example: Moving Average in R

`RModel` takes the path to your `.R` script plus any keyword arguments you want forwarded as parameters:

```python
from pathlib import Path

import forecast_evaluation as fe
import forecast_realtime as rt
from forecast_realtime import RModel

forecast_data = fe.ForecastData(load_fer=True)

# Resolve the script relative to this file so it works from any directory
# "window_size=4" becomes params$window_size inside the R script
model = RModel(str(Path(__file__).parent / "ma_model.R"), window_size=4)
rt_model = rt.RealTimeModel(data=forecast_data, models=model)
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "pop"},
    steps=8,
    label="MA(4) R",
    first_vintage="2015-01-01",
)
```

The R script `ma_model.R` looks like:

```r
# ma_model.R — only defines fit() and forecast()

fit <- function(y, X, params) {
  window_size <- as.integer(params$window_size)
    y <- y[, setdiff(colnames(y), "date"), drop = FALSE]

  n       <- nrow(y)
  tail_df <- y[max(1, n - window_size + 1):n, , drop = FALSE]

  window_mean <- sapply(tail_df, mean)

  # Return a model object — the runner saves it to model.rds
  list(window_mean = window_mean, col_names = colnames(y))
}

forecast <- function(model, steps, X, y, params) {
  window_mean <- model$window_mean
  n_vars         <- length(window_mean)

  fcst <- matrix(rep(window_mean, each = steps), nrow = steps, ncol = n_vars)

  out <- as.data.frame(fcst)
  colnames(out) <- model$col_names
  # Return a data.frame — the runner writes it to forecasts.parquet
  out
}
```

---

## Example: Moving Average in MATLAB

`MATLABModel` takes the path to your `.m` file. The file's stem is called as a MATLAB function:

```python
from pathlib import Path

import forecast_evaluation as fe
import forecast_realtime as rt
from forecast_realtime import MATLABModel

forecast_data = fe.ForecastData(load_fer=True)

# Resolve the script relative to this file so it works from any directory
# "window_size=4" becomes params.window_size inside the MATLAB function
model = MATLABModel(str(Path(__file__).parent / "ma_model.m"), window_size=4)
rt_model = rt.RealTimeModel(data=forecast_data, models=model)
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "pop"},
    steps=8,
    label="MA(4) MATLAB",
    first_vintage="2015-01-01",
)
```

The MATLAB function `ma_model.m` looks like:

```matlab
function result = ma_model(action, varargin)
    % fit:      result = ma_model('fit', y, X, params)
    % forecast: result = ma_model('forecast', model, steps, X, y, params)

    if strcmp(action, 'fit')
        y      = varargin{1};
        X      = varargin{2};  % regressors (empty [] when none)
        params = varargin{3};
        window_size = params.window_size;

        y_arr = table2array(y);
        n     = size(y_arr, 1);

        tail_y         = y_arr(max(1, n - window_size + 1):n, :);
        window_mean = mean(tail_y, 1);

        % Return a model struct — the runner saves it to model.mat
        result.window_mean = window_mean;
        result.col_names      = y.Properties.VariableNames;

    elseif strcmp(action, 'forecast')
        model = varargin{1};
        steps = varargin{2};
        X     = varargin{3};   % future regressors (one row per step)
        y     = varargin{4};

        fcst = repmat(model.window_mean, steps, 1);

        % Return a table — the runner writes it to forecasts.parquet
        result = array2table(fcst, 'VariableNames', model.col_names);
    end
end
```

---

## Example: Moving Average in Julia

`JuliaModel` takes the path to your `.jl` script:

```python
from pathlib import Path

import forecast_evaluation as fe
import forecast_realtime as rt
from forecast_realtime import JuliaModel

forecast_data = fe.ForecastData(load_fer=True)

# Resolve the script relative to this file so it works from any directory
# "window_size=4" becomes params["window_size"] inside the Julia script
model = JuliaModel(str(Path(__file__).parent / "ma_model.jl"), window_size=4)
rt_model = rt.RealTimeModel(data=forecast_data, models=model)
rt_model.forecast(
    y_variables=["cpisa"],
    data_transformation={"cpisa": "pop"},
    steps=8,
    label="MA(4) Julia",
    first_vintage="2015-01-01",
)
```

The Julia script `ma_model.jl` looks like:

```julia
# ma_model.jl — only defines fit() and forecast()

using Statistics

function fit(y, X, params)
    window_size = Int(params[:window_size])

    col_names = names(y)
    n         = nrow(y)

    tail_start     = max(1, n - window_size + 1)
    window_mean = [mean(Float64.(y[tail_start:n, c])) for c in col_names]

    # Return a model object — the runner serialises it to model.jls
    Dict("window_mean" => window_mean,
         "col_names" => col_names)
end

function forecast(model, steps, X, y, params)
    window_mean = model["window_mean"]

    fcst = repeat(transpose(window_mean), steps, 1)

    # Return a DataFrame — the runner writes it to forecasts.parquet
    DataFrame(fcst, model["col_names"])
end
```

---

## Testing

Every model wrapper must include a test (in `tests/models/`) that verifies the wrapper produces **exactly the same results** as the native package when called directly. This ensures the wrapper is a transparent pass-through with no unintended side effects. Examples can be found in `tests/models/test_midas.py` and `tests/models/test_bvar.py`.

---

## Debugging External Models

All external model classes support an interactive debug mode that launches the language's REPL with `y`, `X` and `params` already loaded, and your `fit()` / `forecast()` called automatically.

External model scripts are executed as code by R, MATLAB or Julia. Only use
scripts from a trusted source; path and command quoting does not sandbox the
script itself. For fable wrappers, `spec` and `xreg` are also trusted R
expressions and are evaluated by the R process.

Pass `debug="fit"` or `debug="forecast"` when creating the model:

```python
from pathlib import Path

script = str(Path(__file__).parent / "ma_model.R")

model = RModel(script, debug="fit", window_size=4)
model.fit(y)  # opens an interactive R REPL, calls fit()

model = RModel(script, debug="forecast", window_size=4)
model.fit(y)  # runs fit normally
model.forecast(4)  # opens an interactive R REPL, calls forecast()
```

Add breakpoints in your script before running:

| Language | Breakpoint command             | Notes                                    |
|----------|--------------------------------|------------------------------------------|
| R        | `browser()`                    | Pause and inspect; `n` to step, `c` to continue, `Q` to quit |
| MATLAB   | Set breakpoints in the editor  | `debug="fit"` opens the MATLAB desktop   |
| Julia    | `@bp` or `@infiltrate`         | Requires `Debugger.jl` or `Infiltrator.jl` |
