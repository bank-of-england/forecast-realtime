"""Tests for MovingAverage models (Python and R versions).

Exercises the MA examples from adding_a_model.md to ensure they work
end-to-end against the ForecastModel / RModel interface.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecast_realtime import ForecastModel
from forecast_realtime.external_model import RModel


# ---------------------------------------------------------------------------
# Python MovingAverage (copied verbatim from adding_a_model.md)
# ---------------------------------------------------------------------------
class MovingAverage(ForecastModel):
    """Moving-average forecast: predict the mean of the last `window_size` observations.

    Parameters
    ----------
    window_size : int
        Number of trailing observations to average.
    """

    def __init__(self, window_size: int = 4):
        self.window_size = window_size
        self.forecast_value = None  # will be shape (n_variables,)

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
        self.forecast_value = tail.mean().values  # shape (n_variables,)
        return self

    def _forecast(self, steps: int, X: np.ndarray = None, **kwargs) -> pd.DataFrame:
        """Return the moving-average value for every horizon.

        Parameters
        ----------
        steps : int
            Number of horizons to forecast.
        X : ignored

        Returns
        -------
        pd.DataFrame, shape (steps, n_variables)
            Same value repeated for each step, indexed by a DatetimeIndex.
        """
        # Tile the mean across all forecast horizons, then wrap with the
        # default inferred dates (next ``steps`` periods after ``self.y``).
        arr = np.tile(self.forecast_value, (steps, 1))
        return self._wrap_forecast(arr, steps)


class BareArrayMovingAverage(MovingAverage):
    """Same model, but returning a bare ndarray from ``_forecast``.

    ``ForecastModel.forecast`` is expected to attach the standard inferred
    dates and column names itself, so a simple model need not know about
    ``_wrap_forecast`` at all.
    """

    def _forecast(self, steps: int, X: np.ndarray = None, **kwargs) -> np.ndarray:
        return np.tile(self.forecast_value, (steps, 1))


class SeriesMovingAverage(MovingAverage):
    """Single-variable model returning a 1-D array from ``_forecast``."""

    def _forecast(self, steps: int, X: np.ndarray = None, **kwargs) -> np.ndarray:
        return np.repeat(self.forecast_value[0], steps)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def quarterly_data():
    """20 quarters of synthetic GDP data."""
    dates = pd.date_range("2000-03-31", periods=20, freq="QE")
    np.random.seed(42)
    values = 100 + np.cumsum(np.random.randn(20) * 0.5)
    return pd.DataFrame({"gdp": values}, index=dates)


# ---------------------------------------------------------------------------
# Python MA tests
# ---------------------------------------------------------------------------
class TestMovingAveragePython:
    """Test the Python MovingAverage model."""

    def test_shape(self, quarterly_data):
        model = MovingAverage(window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=8)

        assert isinstance(fc, pd.DataFrame)
        assert fc.shape == (8, 1)

    def test_values_equal_window_mean(self, quarterly_data):
        model = MovingAverage(window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=3)

        expected = quarterly_data["gdp"].iloc[-4:].mean()
        np.testing.assert_allclose(fc.values[:, 0], expected)

    def test_all_horizons_identical(self, quarterly_data):
        model = MovingAverage(window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=6)

        # Every row should be the same value
        np.testing.assert_array_equal(fc.values[0], fc.values[-1])

    def test_multivariate(self):
        dates = pd.date_range("2000-03-31", periods=10, freq="QE")
        np.random.seed(0)
        y = pd.DataFrame(
            {"a": np.random.randn(10), "b": np.random.randn(10)},
            index=dates,
        )
        model = MovingAverage(window_size=3)
        model.fit(y)
        fc = model.forecast(steps=4)

        assert fc.shape == (4, 2)
        np.testing.assert_allclose(fc.values[0, 0], y["a"].iloc[-3:].mean())
        np.testing.assert_allclose(fc.values[0, 1], y["b"].iloc[-3:].mean())


# ---------------------------------------------------------------------------
# Array returns from _forecast are wrapped by the base class
# ---------------------------------------------------------------------------
class TestBareArrayForecast:
    """``_forecast`` may return an array; ``forecast()`` wraps it."""

    def test_matches_explicitly_wrapped_model(self, quarterly_data):
        wrapped = MovingAverage(window_size=4).fit(quarterly_data).forecast(steps=8)
        bare = BareArrayMovingAverage(window_size=4).fit(quarterly_data).forecast(steps=8)

        pd.testing.assert_frame_equal(bare, wrapped)

    def test_index_and_columns_are_attached(self, quarterly_data):
        model = BareArrayMovingAverage(window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=5)

        assert isinstance(fc.index, pd.DatetimeIndex)
        assert fc.index.name == "date"
        assert list(fc.columns) == list(quarterly_data.columns)
        assert fc.index[0] > quarterly_data.index[-1]

    def test_one_dimensional_array_is_reshaped(self, quarterly_data):
        model = SeriesMovingAverage(window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=6)

        assert fc.shape == (6, 1)
        np.testing.assert_allclose(
            fc.values[:, 0], quarterly_data["gdp"].iloc[-4:].mean()
        )

    def test_wrong_length_array_raises(self, quarterly_data):
        class ShortMovingAverage(MovingAverage):
            def _forecast(self, steps, X=None, **kwargs):
                return np.tile(self.forecast_value, (steps - 1, 1))

        model = ShortMovingAverage(window_size=4)
        model.fit(quarterly_data)

        with pytest.raises(ValueError, match="expected"):
            model.forecast(steps=4)

    def test_none_raises(self, quarterly_data):
        class NoneMovingAverage(MovingAverage):
            def _forecast(self, steps, X=None, **kwargs):
                return None

        model = NoneMovingAverage(window_size=4)
        model.fit(quarterly_data)

        with pytest.raises(TypeError, match="returned None"):
            model.forecast(steps=4)


# ---------------------------------------------------------------------------
# R MA tests
# ---------------------------------------------------------------------------
_MA_R_SCRIPT = str(Path(__file__).parent / "r_scripts" / "ma_model.R")


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


@pytest.mark.skipif(
    not _r_arrow_available(),
    reason="Rscript not found on PATH or R 'arrow' package not installed",
)
class TestMovingAverageR:
    """Test the R MovingAverage model via RModel."""

    def test_shape(self, quarterly_data):
        model = RModel(_MA_R_SCRIPT, window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=8)

        assert isinstance(fc, pd.DataFrame)
        assert fc.shape == (8, 1)

    def test_values_equal_window_mean(self, quarterly_data):
        model = RModel(_MA_R_SCRIPT, window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=3)

        expected = quarterly_data["gdp"].iloc[-4:].mean()
        np.testing.assert_allclose(fc.values[:, 0], expected, rtol=1e-6)

    def test_all_horizons_identical(self, quarterly_data):
        model = RModel(_MA_R_SCRIPT, window_size=4)
        model.fit(quarterly_data)
        fc = model.forecast(steps=6)

        np.testing.assert_array_almost_equal(fc.values[0], fc.values[-1])

    def test_matches_python(self, quarterly_data):
        """R and Python MA should produce identical forecasts."""
        py_model = MovingAverage(window_size=4)
        py_model.fit(quarterly_data)
        py_fc = py_model.forecast(steps=4)

        r_model = RModel(_MA_R_SCRIPT, window_size=4)
        r_model.fit(quarterly_data)
        r_fc = r_model.forecast(steps=4)

        np.testing.assert_allclose(r_fc.values, py_fc.values, rtol=1e-6)
