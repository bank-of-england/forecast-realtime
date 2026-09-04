"""Tests for MATLABModel (least-squares trend model), including debug flow."""

import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forecast_realtime.external_model import MATLABModel

_MATLAB_SCRIPT = str(Path(__file__).parent / "matlab_scripts" / "forecast_lm.m")

_TIMEOUT = 600.0

requires_matlab = pytest.mark.skipif(
    shutil.which("matlab") is None,
    reason="matlab not found on PATH",
)


@pytest.fixture(scope="module")
def quarterly_data():
    """20 quarters of synthetic GDP data."""
    dates = pd.date_range("2000-03-31", periods=20, freq="QE")
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(20) * 0.5)
    return pd.DataFrame({"gdp": values}, index=dates)


@pytest.fixture(scope="module")
def fitted_model(quarterly_data):
    """A model fitted once in MATLAB and shared by the read-only tests."""
    model = MATLABModel(_MATLAB_SCRIPT, subprocess_timeout=_TIMEOUT)
    model.fit(quarterly_data)
    return model


@requires_matlab
def test_forecast_matlab_lm(fitted_model):
    """Forecast 4 steps from the fitted MATLAB model and check the output."""
    fc = fitted_model.forecast(steps=4)

    assert isinstance(fc, pd.DataFrame)
    assert fc.shape == (4, 1)
    assert not fc.isna().any().any()


@requires_matlab
def test_model_mat_written_on_fit(fitted_model):
    """The runner should serialise the fitted model struct to model.mat."""
    assert (fitted_model.cache_dir / "model.mat").exists()


@requires_matlab
def test_fitted_values_not_produced(fitted_model):
    """The MATLAB runner emits no fitted values, so fitted_values_ stays None."""
    assert fitted_model.fitted_values_ is None


@requires_matlab
def test_forecast_matlab_uses_future_regressor_rows():
    """Forecast-time X changes must affect the MATLAB model forecast."""
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

    model = MATLABModel(_MATLAB_SCRIPT, subprocess_timeout=_TIMEOUT)
    model.fit(y, X=X_train)
    baseline = model.forecast(steps=4, X=X_all).to_numpy()

    changed_X = X_all.copy()
    changed_X.iloc[-4:, 0] += 100.0
    changed = model.forecast(steps=4, X=changed_X).to_numpy()

    assert not np.allclose(baseline, changed)


class TestDebugFitMATLAB:
    """debug='fit' should build an init command and invoke MATLAB.

    The subprocess is patched out, so these never launch MATLAB and need no
    installation.
    """

    def test_generates_init_command(self, quarterly_data):
        model = MATLABModel(_MATLAB_SCRIPT, debug="fit")

        with patch("subprocess.run") as mock_run:
            model.fit(quarterly_data)

        source = mock_run.call_args.args[0][2]
        assert "addpath(" in source
        assert "user_function = str2func(" in source
        assert "parquetread(" in source
        assert "model = user_function('fit', y, X, params)" in source
        assert "save(fullfile(cache_dir, 'model.mat'), 'model')" in source

    def test_calls_matlab(self, quarterly_data):
        model = MATLABModel(_MATLAB_SCRIPT, debug="fit")

        with patch("subprocess.run") as mock_run:
            model.fit(quarterly_data)

        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "matlab"
        assert cmd[1] == "-r"

    def test_writes_y_parquet(self, quarterly_data):
        model = MATLABModel(_MATLAB_SCRIPT, debug="fit")

        with patch("subprocess.run"):
            model.fit(quarterly_data)

        assert (model.cache_dir / "y.parquet").exists(), (
            "y.parquet was not written before the debug session"
        )


@requires_matlab
class TestDebugForecastMATLAB:
    """debug='forecast' should fit normally, then debug only the forecast stage."""

    @pytest.fixture(scope="class")
    @classmethod
    def debug_model(cls, quarterly_data):
        """Fitted once: only the forecast stage is under test here."""
        model = MATLABModel(_MATLAB_SCRIPT, debug="forecast", subprocess_timeout=_TIMEOUT)
        model.fit(quarterly_data)
        return model

    @staticmethod
    def _stub_forecast_output(model):
        """Return a subprocess stand-in that writes the expected forecast file."""

        def mock_matlab_run(*args, **kwargs):
            pd.DataFrame({"gdp": [0.0] * 4}).to_parquet(
                model.cache_dir / "forecasts.parquet"
            )

        return mock_matlab_run

    def test_fit_runs_normally(self, debug_model):
        assert (debug_model.cache_dir / "model.mat").exists()

    def test_generates_init_command(self, debug_model):
        with patch(
            "subprocess.run", side_effect=self._stub_forecast_output(debug_model)
        ) as mock_run:
            debug_model.forecast(steps=4)

        source = mock_run.call_args.args[0][2]
        assert "loaded = load(fullfile(cache_dir, 'model.mat'), 'model')" in source
        assert "user_function('forecast', loaded.model, 4, X, y, params)" in source
        assert "parquetwrite(fullfile(cache_dir, 'forecasts.parquet'), result)" in source

    def test_calls_matlab(self, debug_model):
        with patch(
            "subprocess.run", side_effect=self._stub_forecast_output(debug_model)
        ) as mock_run:
            debug_model.forecast(steps=4)

        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd[0] == "matlab"
        assert cmd[1] == "-r"
