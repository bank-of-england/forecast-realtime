"""Tests for the uniform in-sample fitted-values contract on ForecastModel."""

import pandas as pd
import pytest

from forecast_realtime.forecast_model import ForecastModel


class _DummyModel(ForecastModel):
    """Minimal concrete subclass used only to exercise the base contract."""

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return pd.DataFrame()


def test_fitted_values_default_is_none():
    """A freshly constructed instance has fitted_values_ set to None."""
    model = _DummyModel()

    assert model.fitted_values_ is None


def test_fitted_values_property_raises_before_fit():
    """Accessing fitted_values before fitting raises a clear AttributeError."""
    model = _DummyModel()

    with pytest.raises(AttributeError, match="not been fitted"):
        _ = model.fitted_values


def test_fitted_values_property_returns_stored_series():
    """Once fitted_values_ is populated, the property returns it unchanged."""
    model = _DummyModel()
    expected = pd.Series([1.0, 2.0, 3.0])
    model.fitted_values_ = expected

    pd.testing.assert_series_equal(model.fitted_values, expected)
