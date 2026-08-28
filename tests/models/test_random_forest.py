"""Tests for RandomForest model."""

import numpy as np
import pandas as pd
import pytest

import forecast_realtime as rt

from .sample_regression import sample_ar2_data, sample_regression_data


class _LagEchoEstimator:
    """Stub estimator returning the ``target_lag2`` column of the design row."""

    def __init__(self, lag2_position):
        self.lag2_position = lag2_position

    def predict(self, X):
        return np.asarray(X)[:, self.lag2_position]


class _CaptureEstimator:
    """Estimator that records fit and prediction arrays for scaling tests."""

    def fit(self, X, y):
        self.X_fit = np.asarray(X).copy()
        self.y_fit = np.asarray(y).copy()
        return self

    def predict(self, X):
        self.X_predict = np.asarray(X).copy()
        return np.zeros(len(X))


class _CaptureForest(rt.models.RandomForest):
    def _build_estimator(self):
        return _CaptureEstimator()


def test_rf_recursive_two_y_lags_uses_correct_period():
    """Each lag column holds y from its own period as the forecast rolls forward.

    The stub estimator returns the value in ``target_lag2``,
    i.e. y two periods before the forecast date. The three-step path should
    therefore be ``[y[T-1], y[T], forecast[0]]``. If every lag column were
    overwritten with the latest prediction, steps 1 and 2 would repeat the
    previous prediction instead.
    """
    y_train, X_train, _, X_test, _ = sample_ar2_data(n_train=100, n_test=3)

    model = rt.models.RandomForest(n_estimators=5, random_state=0)
    model.fit(y_train, X=X_train, y_lags=2)

    lag2_position = list(model.X.columns).index("target_lag2")
    model.model = _LagEchoEstimator(lag2_position)

    forecasts = model.forecast(steps=3, X=X_test).values.ravel()

    expected = [
        y_train.iloc[-2, 0],  # two periods before T+1 is y[T-1]
        y_train.iloc[-1, 0],  # two periods before T+2 is y[T]
        forecasts[0],  # two periods before T+3 is the first forecast
    ]
    np.testing.assert_allclose(forecasts, expected, atol=1e-10)


def test_rf_direct_standardise_uses_horizon_specific_scalers():
    """Each direct horizon uses the scalers fitted on its training sample."""
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    X_train = pd.DataFrame({"x1": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]}, index=dates[:6])
    y_train = pd.DataFrame(
        {"target": [10.0, 20.0, 40.0, 80.0, 160.0, 320.0]}, index=dates[:6]
    )
    X_future = pd.DataFrame({"x1": [64.0]}, index=dates[6:7])
    steps = 3

    model = _CaptureForest(forecast_strategy="direct", steps=steps, standardise=True)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_future)

    for horizon in range(steps):
        X_sample = X_train.iloc[: len(X_train) - horizon]
        y_sample = y_train.iloc[horizon:]
        X_scaler = model._X_scalers_[horizon]
        y_scaler = model._y_scalers_[horizon]
        estimator = model.models_[horizon]

        np.testing.assert_allclose(X_scaler.mean_, X_sample.mean().values)
        np.testing.assert_allclose(y_scaler.mean_, y_sample.mean().values)
        np.testing.assert_allclose(estimator.X_fit, X_scaler.transform(X_sample.values))
        np.testing.assert_allclose(
            estimator.X_predict, X_scaler.transform(X_future.values)
        )
        np.testing.assert_allclose(forecasts.iloc[horizon, 0], y_scaler.mean_[0])


def test_rf_direct_rejects_excessive_forecast_horizon():
    """Direct tree regression requires every requested horizon to be fitted."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    y = pd.DataFrame({"target": np.arange(1.0, 6.0)}, index=dates)
    X = pd.DataFrame({"driver": np.arange(1.0, 6.0)}, index=dates)

    model = _CaptureForest(forecast_strategy="direct", steps=2)
    model.fit(y, X=X)

    with pytest.raises(ValueError, match="fitted for 2.*3.*larger horizon"):
        model.forecast(steps=3, X=X.iloc[[-1]])


def test_rf_recursive_two_y_lags_noiseless():
    """RandomForest tracks a noiseless AR(2) path over three recursive steps."""
    y_train, X_train, y_test, X_test, _ = sample_ar2_data(n_train=4000, n_test=3)

    model = rt.models.RandomForest(n_estimators=200, random_state=42)
    model.fit(y_train, X=X_train, y_lags=2)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    assert forecasts.shape == (len(y_test), 1)
    np.testing.assert_allclose(forecasts.values, y_test.values, atol=0.5)


def test_rf_recursive_noiseless():
    """RandomForest recovers the true relationship on noiseless linear data."""
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=10000,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=42,
    )

    model = rt.models.RandomForest(n_estimators=100, random_state=42)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    assert isinstance(forecasts, pd.DataFrame)
    assert forecasts.shape == (len(y_test), 1)
    assert isinstance(forecasts.index, pd.DatetimeIndex)
    np.testing.assert_allclose(forecasts.values, y_test.values, atol=0.05)


def test_rf_recursive_with_ar():
    """Recursive forecasting with autoregressive lag."""
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=5000,
        n_test=4,
        b1=2.0,
        b2=-1.0,
        b_ar=0.5,
        noise_std=0,
        forecast_type="recursive",
        random_seed=42,
    )

    model = rt.models.RandomForest(n_estimators=200, random_state=42)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    assert isinstance(forecasts, pd.DataFrame)
    assert forecasts.shape == (len(y_test), 1)
    np.testing.assert_allclose(forecasts.values, y_test.values, atol=0.1)


def test_rf_direct_h0():
    """Direct forecasting at horizon 0."""
    horizon = 0
    steps = horizon + 1

    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=10000,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=42,
    )

    model = rt.models.RandomForest(
        n_estimators=100, random_state=42, forecast_strategy="direct", steps=steps
    )
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    assert forecasts.shape == (steps, 1)
    np.testing.assert_allclose(forecasts.iloc[0].values, y_test.iloc[0].values, atol=0.01)


def test_rf_direct_h1():
    """Direct forecasting at horizon 1."""
    horizon = 1
    steps = horizon + 1

    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=10000,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=42,
    )

    model = rt.models.RandomForest(
        n_estimators=100, random_state=42, forecast_strategy="direct", steps=steps
    )
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    assert forecasts.shape == (steps, 1)
    np.testing.assert_allclose(
        forecasts.iloc[horizon].values, y_test.iloc[horizon].values, atol=0.01
    )


def test_rf_standardise_matches_unstandardised():
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

    model_raw = rt.models.RandomForest(n_estimators=200, random_state=42)
    model_raw.fit(y_train, X=X_train)
    fc_raw = model_raw.forecast(steps=len(y_test), X=X_test)

    model_std = rt.models.RandomForest(
        n_estimators=200, random_state=42, standardise=True
    )
    model_std.fit(y_train, X=X_train)
    fc_std = model_std.forecast(steps=len(y_test), X=X_test)

    np.testing.assert_allclose(fc_raw.values, fc_std.values, atol=0.01)


def test_rf_feature_importance():
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

    model = rt.models.RandomForest(n_estimators=200, random_state=42)
    model.fit(y_df, X=X)

    imp = model.model.feature_importances_
    # x1 and x2 should each have higher importance than noise
    assert imp[0] > imp[2], "x1 importance should exceed noise"
    assert imp[1] > imp[2], "x2 importance should exceed noise"


def test_rf_requires_x():
    """Raises ValueError when X is None."""
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=50, n_test=5, random_seed=42
    )
    model = rt.models.RandomForest()
    try:
        model.fit(y_train, X=None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_fitted_values_recovers_insample():
    """fitted_values_ recovers y_train in-sample on noiseless data."""
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=10000,
        n_test=5,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=42,
    )

    model = rt.models.RandomForest(n_estimators=100, random_state=42)
    model.fit(y_train, X=X_train)

    fitted = model.fitted_values_.dropna()
    # Bootstrap aggregation means in-sample predictions are not an exact
    # interpolation, unlike the out-of-sample forecast; atol reflects the
    # measured in-sample deviation for this deterministic (seeded) fit.
    np.testing.assert_allclose(
        fitted.to_numpy(), y_train.loc[fitted.index].to_numpy().ravel(), atol=0.7
    )
