"""Integration tests for fable models through forecast_realtime."""

import shutil
import subprocess

import forecast_evaluation as fe
import numpy as np
import pandas as pd
import pytest

import forecast_realtime as rt

_FABLE_R_CODE = r"""
suppressPackageStartupMessages({
  library(arrow)
  library(fable)
  library(fabletools)
  library(tsibble)
})
args <- commandArgs(trailingOnly = TRUE)
y <- arrow::read_parquet(args[[1]])
spec <- args[[2]]
h <- as.integer(args[[3]])
freq <- args[[5]]
index <- if (freq == "Q") {
  tsibble::yearquarter(as.Date(y$date))
} else if (freq == "M") {
  tsibble::yearmonth(as.Date(y$date))
} else {
  stop(sprintf("Unknown frequency: %s", freq))
}
data <- tsibble::as_tsibble(
  data.frame(index = index, value = as.numeric(y$target)),
  index = index
)
model_spec <- eval(parse(text = spec), envir = globalenv())
model <- fabletools::model(data, fable_model = model_spec)
forecast <- fabletools::forecast(model, h = h)
result <- data.frame(value = as.numeric(as.data.frame(forecast)[[".mean"]]))
    arrow::write_parquet(result, args[[4]])
"""

_FABLE_R_CODE_XREG = r"""
suppressPackageStartupMessages({
  library(arrow)
  library(fable)
  library(fabletools)
  library(tsibble)
})
args <- commandArgs(trailingOnly = TRUE)
y <- arrow::read_parquet(args[[1]])
xreg <- arrow::read_parquet(args[[2]])
spec <- args[[3]]
h <- as.integer(args[[4]])
freq <- args[[5]]
out_path <- args[[6]]

make_index <- function(dates) {
  if (freq == "Q") {
    tsibble::yearquarter(as.Date(dates))
  } else if (freq == "M") {
    tsibble::yearmonth(as.Date(dates))
  } else {
    stop(sprintf("Unknown frequency: %s", freq))
  }
}

y_dates <- as.Date(y$date)
last_date <- max(y_dates)

xreg_dates <- as.Date(xreg$date)
history_mask <- xreg_dates <= last_date
future_mask <- xreg_dates > last_date

train <- data.frame(
  index = make_index(y_dates),
  value = as.numeric(y$target),
  regressor = as.numeric(xreg$regressor[history_mask])
)
data <- tsibble::as_tsibble(train, index = index)

model_spec <- eval(parse(text = spec), envir = globalenv())
model <- fabletools::model(data, fable_model = model_spec)

future <- data.frame(
  index = make_index(xreg_dates[future_mask]),
  regressor = as.numeric(xreg$regressor[future_mask])
)
new_data <- tsibble::as_tsibble(future, index = index)

forecast <- fabletools::forecast(model, new_data = new_data)
result <- data.frame(value = as.numeric(as.data.frame(forecast)[[".mean"]]))
arrow::write_parquet(result, out_path)
"""


def _r_fable_available() -> bool:
    """Return True when the R runtime and fable dependencies are installed."""
    if shutil.which("Rscript") is None:
        return False
    try:
        result = subprocess.run(
            [
                "Rscript",
                "-e",
                "cat(all(vapply(c('arrow', 'fable', 'fabletools', 'tsibble'), "
                "requireNamespace, logical(1), quietly=TRUE)))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and result.stdout.strip() == "TRUE"


requires_r_fable = pytest.mark.skipif(
    not _r_fable_available(),
    reason=(
        "Rscript and the R packages arrow, fable, fabletools, and tsibble are required"
    ),
)


@pytest.fixture
def quarterly_series():
    dates = pd.date_range("2000-03-31", periods=20, freq="QE")
    values = (
        100.0 + np.linspace(0.0, 10.0, len(dates)) + 0.2 * np.sin(np.arange(len(dates)))
    )
    return pd.DataFrame({"target": values}, index=dates)


@pytest.fixture
def monthly_series():
    dates = pd.date_range("2000-01-31", periods=24, freq="ME")
    values = (
        100.0 + np.linspace(0.0, 10.0, len(dates)) + 0.2 * np.sin(np.arange(len(dates)))
    )
    return pd.DataFrame({"target": values}, index=dates)


def _direct_fable_forecast(y, spec, steps, tmp_path, freq="Q"):
    """Run the same fixed fable specification directly in a fresh R process."""
    input_path = tmp_path / "direct_y.parquet"
    script_path = tmp_path / "direct_fable.R"
    output_path = tmp_path / "direct_forecast.parquet"
    y.rename_axis("date").reset_index().to_parquet(input_path, index=False)
    script_path.write_text(_FABLE_R_CODE)
    result = subprocess.run(
        [
            "Rscript",
            str(script_path),
            str(input_path),
            spec,
            str(steps),
            str(output_path),
            freq,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Direct fable R process failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return pd.read_parquet(output_path)["value"].to_numpy()


def _direct_fable_forecast_xreg(y, X, spec, steps, tmp_path, freq="Q"):
    """Run a fixed fable ARIMA-with-regressor specification directly in R."""
    y_path = tmp_path / "direct_xreg_y.parquet"
    x_path = tmp_path / "direct_xreg_x.parquet"
    script_path = tmp_path / "direct_fable_xreg.R"
    output_path = tmp_path / "direct_xreg_forecast.parquet"
    y.rename_axis("date").reset_index().to_parquet(y_path, index=False)
    X.rename_axis("date").reset_index().to_parquet(x_path, index=False)
    script_path.write_text(_FABLE_R_CODE_XREG)
    result = subprocess.run(
        [
            "Rscript",
            str(script_path),
            str(y_path),
            str(x_path),
            spec,
            str(steps),
            freq,
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Direct fable R process failed (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return pd.read_parquet(output_path)["value"].to_numpy()


def _forecast_data(y, vintage, freq="Q"):
    return fe.ForecastData(
        outturns_data=pd.DataFrame(
            {
                "date": y.index,
                "variable": "target",
                "vintage_date": vintage,
                "frequency": freq,
                "value": y["target"].to_numpy(),
                "metric": "levels",
            }
        ),
        metric="levels",
        compute_levels=False,
    )


@requires_r_fable
@pytest.mark.parametrize(
    "model_factory",
    [
        lambda index: rt.models.RFableETS(error="A", trend="A", season="N", index=index),
        lambda index: rt.models.RFableARIMA(p=1, d=0, q=0, index=index),
    ],
    ids=["ets", "arima"],
)
@pytest.mark.parametrize("freq", ["Q", "M"])
def test_fable_matches_direct_r_and_realtime(
    model_factory,
    freq,
    quarterly_series,
    monthly_series,
    tmp_path,
):
    """Native fable, direct wrapper, and RealTimeModel give identical values."""
    index = "quarter" if freq == "Q" else "month"
    series = quarterly_series if freq == "Q" else monthly_series
    offset = pd.offsets.QuarterEnd(1) if freq == "Q" else pd.offsets.MonthEnd(1)

    model = model_factory(index)
    steps = 4
    direct = _direct_fable_forecast(series, model.spec, steps, tmp_path, freq)

    model.fit(series)
    wrapped = model.forecast(steps=steps)["target"].to_numpy()

    vintage = series.index[-1] + offset
    realtime = rt.RealTimeModel(
        data=_forecast_data(series, vintage, freq),
        models=model_factory(index),
    )
    realtime.forecast(
        y_variables=["target"],
        data_transformation={"target": "levels"},
        steps=steps,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )
    realtime_values = (
        realtime.data.forecasts.loc[
            (realtime.data.forecasts["source"] == type(model).__name__)
            & (realtime.data.forecasts["variable"] == "target")
            & (realtime.data.forecasts["metric"] == "levels"),
        ]
        .sort_values("date")["value"]
        .to_numpy()
    )

    np.testing.assert_array_equal(wrapped, direct)
    np.testing.assert_array_equal(realtime_values, direct)


@requires_r_fable
def test_fable_arima_with_xreg(quarterly_series, tmp_path):
    """RFableARIMA with a regressor matches a direct R call using the same xreg."""
    steps = 4
    dates = quarterly_series.index
    history_regressor = pd.DataFrame(
        {"regressor": np.linspace(1.0, 2.0, len(dates))},
        index=dates,
    )
    future_dates = pd.date_range(
        dates[-1] + pd.offsets.QuarterEnd(1), periods=steps, freq="QE"
    )
    future_regressor = pd.DataFrame(
        {"regressor": np.linspace(2.1, 2.5, steps)}, index=future_dates
    )
    X = pd.concat([history_regressor, future_regressor])

    model = rt.models.RFableARIMA(p=1, d=0, q=0, xreg="regressor", index="quarter")
    model.fit(quarterly_series, X)
    wrapped = model.forecast(steps=steps, X=X)["target"].to_numpy()

    direct = _direct_fable_forecast_xreg(
        quarterly_series, X, model.spec, steps, tmp_path, "Q"
    )

    assert wrapped.shape == (steps,)
    assert np.all(np.isfinite(wrapped))
    np.testing.assert_allclose(wrapped, direct)


def test_rfablemodel_rejects_invalid_spec_and_index():
    with pytest.raises(ValueError):
        rt.models.RFableModel("")
    with pytest.raises(ValueError):
        rt.models.RFableModel("   ")
    with pytest.raises(ValueError):
        rt.models.RFableModel(123)
    with pytest.raises(ValueError):
        rt.models.RFableModel("ETS(value ~ error('A'))", index="not-a-real-index")


def test_rfableets_rejects_invalid_components():
    with pytest.raises(ValueError):
        rt.models.RFableETS(error="bad")
    with pytest.raises(ValueError):
        rt.models.RFableETS(trend="bad")
    with pytest.raises(ValueError):
        rt.models.RFableETS(season="bad")
    with pytest.raises(ValueError):
        rt.models.RFableETS(season="N", period=4)
    with pytest.raises(ValueError):
        rt.models.RFableETS(season=None, period=4)


def test_rfablemodel_rejects_date_regressor_name_before_serialising(quarterly_series):
    model = rt.models.RFableModel(
        "TSLM(value ~ trend() + season())",
        index="quarter",
        allow_xreg=True,
    )
    X = pd.DataFrame({"date": np.linspace(1.0, 2.0, len(quarterly_series))})
    X.index = quarterly_series.index

    with pytest.raises(ValueError, match="reserved for the input index"):
        model.fit(quarterly_series, X)


@requires_r_fable
@pytest.mark.parametrize("reserved_name", ["value", "index"])
def test_rfablemodel_rejects_reserved_regressor_names(
    quarterly_series,
    reserved_name,
):
    model = rt.models.RFableModel(
        "TSLM(value ~ trend() + season())",
        index="quarter",
        allow_xreg=True,
    )
    X = pd.DataFrame({reserved_name: np.linspace(1.0, 2.0, len(quarterly_series))})
    X.index = quarterly_series.index

    with pytest.raises(subprocess.CalledProcessError):
        model.fit(quarterly_series, X)


def test_rfablearima_rejects_invalid_seasonal_args():
    with pytest.raises(TypeError):
        rt.models.RFableARIMA(seasonal="yes")
    with pytest.raises(ValueError):
        rt.models.RFableARIMA(seasonal=False, P=1)
    with pytest.raises(ValueError):
        rt.models.RFableARIMA(seasonal=False, D=1)
    with pytest.raises(ValueError):
        rt.models.RFableARIMA(seasonal=False, Q=1)
    with pytest.raises(ValueError):
        rt.models.RFableARIMA(seasonal=False, period=4)
    with pytest.raises(ValueError):
        rt.models.RFableARIMA(xreg="")
    with pytest.raises(ValueError):
        rt.models.RFableARIMA(xreg="   ")
