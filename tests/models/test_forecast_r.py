"""Tests for ForecastRlm (lm-based AR via R), including debug REPL flow."""

import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forecast_realtime.models import ForecastRlm


def _r_arrow_available() -> bool:
    """Return ``True`` when ``Rscript`` and the R ``arrow`` package are available."""
    if shutil.which("Rscript") is None:
        return False
    try:
        result = subprocess.run(
            ["Rscript", "-e", "cat(requireNamespace('arrow', quietly=TRUE))"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return "TRUE" in result.stdout
    except Exception:
        return False


# Skip if Rscript is not on PATH or R 'arrow' package is not installed.
pytestmark = pytest.mark.skipif(
    not _r_arrow_available(),
    reason="Rscript not found on PATH or R 'arrow' package not installed",
)


@pytest.fixture
def quarterly_data():
    """20 quarters of synthetic GDP data."""
    dates = pd.date_range("2000-03-31", periods=20, freq="QE")
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(20) * 0.5)
    return pd.DataFrame({"gdp": values}, index=dates)


def test_forecast_r_lm(quarterly_data):
    """Fit lm AR(2), forecast 4 steps, and verify output shape."""
    model = ForecastRlm(lags=2, debug=None)

    # estimation
    model.fit(quarterly_data)
    fc = model.forecast(steps=4)

    assert isinstance(fc, pd.DataFrame)
    assert fc.shape == (4, 1)
    assert not fc.isna().any().any()


def test_forecast_r_lm_uses_future_regressor_rows():
    """Forecast-time X changes must affect the R model forecast."""
    dates = pd.date_range("2000-03-31", periods=20, freq="QE")
    x_train = np.sin(np.arange(20, dtype=float))
    y = pd.DataFrame(
        {"gdp": 100.0 + 0.3 * np.arange(20) + 2.0 * x_train},
        index=dates,
    )
    X_train = pd.DataFrame({"indicator": x_train}, index=dates)
    future_dates = pd.date_range(
        dates[-1] + pd.offsets.QuarterEnd(), periods=4, freq="QE"
    )
    X_future = pd.DataFrame(
        {"indicator": [10.0, 11.0, 12.0, 13.0]},
        index=future_dates,
    )
    X_all = pd.concat([X_train, X_future])

    model = ForecastRlm()
    model.fit(y, X=X_train)
    baseline = model.forecast(steps=4, X=X_all).to_numpy()

    changed_X = X_all.copy()
    changed_X.iloc[-4:, 0] += 100.0
    changed = model.forecast(steps=4, X=changed_X).to_numpy()

    assert not np.allclose(baseline, changed)


def test_fitted_values_aligned_to_training_index(quarterly_data):
    """fitted_values_ should be a Series aligned to the training y index."""
    model = ForecastRlm(lags=2, debug=None)
    model.fit(quarterly_data)

    assert isinstance(model.fitted_values_, pd.Series)
    assert len(model.fitted_values_) == len(model.y)
    assert list(model.fitted_values_.index) == list(model.y.index)


class TestDebugFitR:
    """debug='fit' should generate an init script and invoke R."""

    def test_generates_init_script(self, quarterly_data):
        model = ForecastRlm(lags=2, debug="fit")

        with patch("subprocess.run"):
            model.fit(quarterly_data)

        init_path = model.cache_dir / "_debug_init.R"
        assert init_path.exists(), "_debug_init.R was not created"

        content = init_path.read_text()
        assert "library(arrow)" in content
        assert "read_params" in content
        assert "source(" in content
        assert "fit(y, X, params)" in content
        assert "saveRDS(" in content

    def test_calls_R(self, quarterly_data):
        model = ForecastRlm(lags=2, debug="fit")

        with patch("subprocess.run") as mock_run:
            model.fit(quarterly_data)

        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "R"
        assert "--no-save" in cmd
        assert "--no-restore" in cmd
        assert "R_PROFILE_USER" in mock_run.call_args.kwargs.get("env", {})

    def test_writes_y_parquet(self, quarterly_data):
        model = ForecastRlm(lags=2, debug="fit")

        with patch("subprocess.run"):
            model.fit(quarterly_data)

        assert (model.cache_dir / "y.parquet").exists(), (
            "y.parquet was not written before debug REPL"
        )


class TestDebugForecastRlm:
    """debug='forecast' should run fit normally, then open a debug REPL for forecast."""

    def test_fit_runs_normally(self, quarterly_data):
        model = ForecastRlm(lags=2, debug="forecast")
        model.fit(quarterly_data)
        assert (model.cache_dir / "model.rds").exists()

    def test_generates_init_script(self, quarterly_data):
        model = ForecastRlm(lags=2, debug="forecast")
        model.fit(quarterly_data)

        def mock_matlab_run(*args, **kwargs):
            pd.DataFrame({"gdp": [0.0] * 4}).to_parquet(
                model.cache_dir / "forecasts.parquet"
            )

        with patch("subprocess.run", side_effect=mock_matlab_run):
            model.forecast(steps=4)

        content = (model.cache_dir / "_debug_init.R").read_text()
        assert "forecast(model, 4L, X, y, params)" in content
        assert "readRDS(" in content
        assert "write_parquet(" in content

    def test_calls_R(self, quarterly_data):
        model = ForecastRlm(lags=2, debug="forecast")
        model.fit(quarterly_data)

        def mock_r_run(*args, **kwargs):
            pd.DataFrame({"gdp": [0.0] * 4}).to_parquet(
                model.cache_dir / "forecasts.parquet"
            )

        with patch("subprocess.run", side_effect=mock_r_run) as mock_run:
            model.forecast(steps=4)

        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "R"
        assert "R_PROFILE_USER" in mock_run.call_args.kwargs.get("env", {})
