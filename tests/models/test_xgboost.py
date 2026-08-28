"""Tests for XGBoost model."""

import pytest

pytest.importorskip("xgboost")

import numpy as np
import pandas as pd

import forecast_realtime as rt

from .sample_regression import sample_regression_data


def test_xgb_recursive_noiseless():
    """XGBoost recovers the true relationship on noiseless linear data."""
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=200,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=42,
    )

    model = rt.models.XGBoost(n_estimators=200, random_state=42)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    assert isinstance(forecasts, pd.DataFrame)
    assert forecasts.shape == (len(y_test), 1)
    assert isinstance(forecasts.index, pd.DatetimeIndex)
    np.testing.assert_allclose(forecasts.values, y_test.values, atol=0.5)


def test_xgb_recursive_with_ar():
    """Recursive forecasting with autoregressive lag."""
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=200,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        b_ar=0.5,
        noise_std=0,
        forecast_type="recursive",
        random_seed=42,
    )

    model = rt.models.XGBoost(n_estimators=200, random_state=42)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    assert isinstance(forecasts, pd.DataFrame)
    assert forecasts.shape == (len(y_test), 1)
    np.testing.assert_allclose(forecasts.values, y_test.values, atol=1.0)


def test_xgb_direct_h0():
    """Direct forecasting at horizon 0."""
    horizon = 0
    steps = horizon + 1

    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=200,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=42,
    )

    model = rt.models.XGBoost(
        n_estimators=200, random_state=42, forecast_strategy="direct", steps=steps
    )
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    assert forecasts.shape == (steps, 1)
    np.testing.assert_allclose(forecasts.iloc[0].values, y_test.iloc[0].values, atol=0.5)


def test_xgb_direct_h1():
    """Direct forecasting at horizon 1."""
    horizon = 1
    steps = horizon + 1

    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=200,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=42,
    )

    model = rt.models.XGBoost(
        n_estimators=200, random_state=42, forecast_strategy="direct", steps=steps
    )
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    assert forecasts.shape == (steps, 1)
    np.testing.assert_allclose(
        forecasts.iloc[horizon].values, y_test.iloc[horizon].values, atol=0.5
    )


def test_xgb_standardise_matches_unstandardised():
    """Standardised and unstandardised fits produce similar forecasts."""
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=200,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=42,
    )

    model_raw = rt.models.XGBoost(n_estimators=200, random_state=42)
    model_raw.fit(y_train, X=X_train)
    fc_raw = model_raw.forecast(steps=len(y_test), X=X_test)

    model_std = rt.models.XGBoost(n_estimators=200, random_state=42, standardise=True)
    model_std.fit(y_train, X=X_train)
    fc_std = model_std.forecast(steps=len(y_test), X=X_test)

    np.testing.assert_allclose(fc_raw.values, fc_std.values, atol=0.5)


def test_xgb_feature_importance():
    """Signal features get higher importance than noise."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    noise_col = rng.standard_normal(n)
    y = 2.0 * x1 - 1.0 * x2

    X = pd.DataFrame({"x1": x1, "x2": x2, "noise": noise_col}, index=dates)
    y_df = pd.DataFrame({"target": y}, index=dates)

    model = rt.models.XGBoost(n_estimators=200, random_state=42)
    model.fit(y_df, X=X)

    imp = model.model.feature_importances_
    assert imp[0] > imp[2], "x1 importance should exceed noise"
    assert imp[1] > imp[2], "x2 importance should exceed noise"


def test_xgb_requires_x():
    """Raises ValueError when X is None."""
    y_train, _, _, _, _ = sample_regression_data(n_train=50, n_test=5, random_seed=42)
    model = rt.models.XGBoost()
    try:
        model.fit(y_train, X=None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_fitted_values_recovers_insample():
    """fitted_values_ recovers y_train in-sample on noiseless data."""
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=200,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=42,
    )

    model = rt.models.XGBoost(n_estimators=200, random_state=42)
    model.fit(y_train, X=X_train)

    fitted = model.fitted_values_.dropna()
    np.testing.assert_allclose(
        fitted.to_numpy(), y_train.loc[fitted.index].to_numpy().ravel(), atol=0.5
    )
