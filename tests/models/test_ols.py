"""Tests for OLS model."""

import numpy as np
import pandas as pd
import pytest

import forecast_realtime as rt
from forecast_realtime._utils import build_lagged_design, regularise_missing_rows

from ..schemas import minimal_decomposition_schema
from .sample_regression import sample_ar2_data, sample_ar4_data, sample_regression_data


def test_build_lagged_design_uses_previous_calendar_period():
    """A missing quarter produces a missing lag instead of skipping it."""
    index = pd.to_datetime(["2020-03-31", "2020-09-30"])
    y = pd.DataFrame({"target": [10.0, 30.0]}, index=index)

    regularised_y = regularise_missing_rows(y, {"target": "Q"})
    design = build_lagged_design(regularised_y, None, y_lags=1, X_lags=0)

    assert pd.isna(design.loc[pd.Timestamp("2020-09-30"), "target_lag1"])


def test_ols_scaling():
    """Test basic recursive fit with synthetic data."""
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    # adding more scale to the data to test scaling
    X_train = X_train
    y_train = y_train
    y_test = y_test
    X_test = X_test

    model = rt.models.ForecastOLS(forecast_strategy="recursive", scale=True)

    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    assert np.allclose(forecasts, y_test, atol=1e-8)


def test_ols_recursive():
    """Test basic recursive fit with synthetic data."""
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="recursive")

    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    assert np.allclose(forecasts, y_test, atol=1e-8)


def test_ols_rejects_multiple_left_hand_side_variables():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame(
        {
            "target_a": [1.0, 2.0, 3.0, 4.0],
            "target_b": [2.0, 3.0, 4.0, 5.0],
        },
        index=index,
    )
    X = pd.DataFrame({"driver": [1.0, 2.0, 3.0, 4.0]}, index=index)

    with pytest.raises(ValueError) as exc_info:
        rt.models.ForecastOLS().fit(y, X=X)

    assert "ForecastOLS cannot handle multiple left-hand-side variables; " in str(
        exc_info.value
    )


def test_ols_rejects_missing_values_by_default():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame({"target": [1.0, np.nan, 3.0, 4.0]}, index=index)
    X = pd.DataFrame({"driver": [1.0, 2.0, 3.0, 4.0]}, index=index)

    with pytest.raises(ValueError, match="drop_nans=True"):
        rt.models.ForecastOLS().fit(y, X=X)


def test_ols_aligns_leading_missing_rows_before_nan_check():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame({"target": [np.nan, 2.0, 3.0, 4.0]}, index=index)
    X = pd.DataFrame({"driver": [np.nan, 2.0, 3.0, 4.0]}, index=index)

    model = rt.models.ForecastOLS()
    model.fit(y, X=X)

    assert model.y.index[0] == index[1]


def test_ols_formula_selects_columns_before_aligning_start_dates():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "unused_target": [np.nan, np.nan, 3.0, 4.0],
        },
        index=index,
    )
    X = pd.DataFrame(
        {
            "used": [10.0, 20.0, 30.0, 40.0],
            "unused": [np.nan, np.nan, 30.0, 40.0],
        },
        index=index,
    )

    model = rt.models.ForecastOLS(formula="target ~ used").fit(y, X=X)

    assert model.y.index[0] == index[0]
    assert list(model.X.columns) == ["used"]


def test_ols_drop_nans_controls_interior_missing_rows_after_alignment():
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    y = pd.DataFrame({"target": [np.nan, 2.0, np.nan, 4.0]}, index=index)
    X = pd.DataFrame({"driver": [np.nan, 2.0, 3.0, 4.0]}, index=index)

    with pytest.raises(ValueError, match="drop_nans=True"):
        rt.models.ForecastOLS().fit(y, X=X)

    model = rt.models.ForecastOLS(drop_nans=True)
    model.fit(y, X=X)
    assert model.y.index[0] == index[1]


@pytest.mark.parametrize(
    "model_class, model_options",
    [
        (rt.models.ForecastOLS, {}),
        (rt.models.ForecastRidge, {"alpha": 1e-8}),
        (rt.models.ForecastLasso, {"alpha": 1e-8}),
        (rt.models.ForecastElasticNet, {"alpha": 1e-8}),
    ],
)
def test_linear_models_reject_mixed_frequencies(model_class, model_options):
    y_index = pd.date_range("2020-03-31", periods=4, freq="QE")
    X_index = pd.date_range("2020-01-31", periods=12, freq="ME")
    y = pd.DataFrame({"target": np.arange(4.0)}, index=y_index)
    X = pd.DataFrame({"driver": np.arange(12.0)}, index=X_index)

    with pytest.raises(ValueError, match="does not support mixed frequencies"):
        model_class(**model_options).fit(y, X=X, frequency="Q")


def test_linear_models_reject_mixed_frequencies_for_pretransformed_inputs():
    y_index = pd.date_range("2020-03-31", periods=4, freq="QE")
    X_index = pd.date_range("2020-01-31", periods=12, freq="ME")
    y = pd.DataFrame({"target": np.arange(4.0)}, index=y_index)
    X = pd.DataFrame({"driver": np.arange(12.0)}, index=X_index)

    with pytest.raises(ValueError, match="does not support mixed frequencies"):
        rt.models.ForecastOLS(data_transformation={"target": "pop", "driver": "pop"}).fit(
            y,
            X=X,
            frequency="Q",
            y_input_metrics={"target": "pop"},
            X_input_metrics={"driver": "pop"},
        )


def test_linear_forecast_rows_pass_stored_target_frequency_without_conditioning(
    monkeypatch,
):
    y_index = pd.date_range("2020-03-31", periods=3, freq="QE")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=y_index)
    X = pd.DataFrame({"driver": [4.0, 5.0, 6.0]}, index=y_index)
    model = rt.models.ForecastOLS()
    model.fit(y, X=X, frequency="Q")

    observed_frequencies = []
    infer_forecast_dates = model._infer_forecast_dates

    def record_frequency(y_index, steps, frequency=None, start=None):
        observed_frequencies.append(frequency)
        return infer_forecast_dates(
            y_index,
            steps,
            frequency=frequency,
            start=start,
        )

    monkeypatch.setattr(model, "_infer_forecast_dates", record_frequency)
    X_with_future = pd.concat(
        [X, pd.DataFrame({"driver": [7.0]}, index=pd.to_datetime(["2020-12-31"]))]
    )

    rows = model._select_forecast_rows(X_with_future, model.last_y_fit_date, 1, None)

    assert observed_frequencies == ["Q"]
    assert rows.index.equals(pd.DatetimeIndex([pd.Timestamp("2020-12-31")]))


def test_ols_recursive_intercept_only():
    """Recursive intercept-only forecasts use the fitted intercept."""
    index = pd.date_range("2020-01-01", periods=4, freq="D")
    y = pd.DataFrame({"target": [1.0, 2.0, 4.0, 5.0]}, index=index)

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y)
    forecasts = model.forecast(steps=2)

    expected = y["target"].mean()
    assert np.allclose(forecasts["target"], expected)


def test_ols_direct_intercept_only():
    """Direct intercept-only forecasts use each horizon's fitted intercept."""
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    y = pd.DataFrame({"target": [1.0, 2.0, 4.0, 5.0, 8.0]}, index=index)

    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=3)
    model.fit(y)
    forecasts = model.forecast(steps=3)

    expected = [model.betas_[h].reshape(-1)[0] for h in range(3)]
    assert np.allclose(forecasts["target"], expected)


def test_ols_direct_rejects_excessive_forecast_horizon():
    """Direct OLS requires a fitted model for every requested horizon."""
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    y = pd.DataFrame({"target": np.arange(1.0, 6.0)}, index=index)
    X = pd.DataFrame({"driver": np.arange(1.0, 6.0)}, index=index)

    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=2)
    model.fit(y, X=X)

    with pytest.raises(ValueError, match="fitted for 2.*3.*larger horizon"):
        model.forecast(steps=3, X=X.iloc[[-1]])


def test_ols_direct_intercept_only_rejects_excessive_forecast_horizon():
    """Intercept-only direct OLS validates the requested horizon too."""
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    y = pd.DataFrame({"target": np.arange(1.0, 6.0)}, index=index)

    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=2)
    model.fit(y)

    with pytest.raises(ValueError, match="fitted for 2.*3.*larger horizon"):
        model.forecast(steps=3)


@pytest.mark.parametrize(
    "model_class, model_options",
    [
        (rt.models.ForecastRidge, {"alpha": 1e-8}),
        (rt.models.ForecastLasso, {"alpha": 1e-8}),
        (rt.models.ForecastElasticNet, {"alpha": 1e-8, "l1_ratio": 0.5}),
    ],
)
def test_regularised_direct_models_reject_excessive_forecast_horizon(
    model_class, model_options
):
    """Regularised linear models inherit direct-horizon validation."""
    index = pd.date_range("2020-01-01", periods=5, freq="D")
    y = pd.DataFrame({"target": np.arange(1.0, 6.0)}, index=index)
    X = pd.DataFrame({"driver": np.arange(1.0, 6.0)}, index=index)

    model = model_class(forecast_strategy="direct", steps=2, **model_options)
    model.fit(y, X=X)

    with pytest.raises(ValueError, match="fitted for 2.*3.*larger horizon"):
        model.forecast(steps=3, X=X.iloc[[-1]])


def test_ols_recursive_with_ar():
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        b_ar=0.5,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    assert np.isclose(beta[3], true_coef["b_ar"], atol=1e-8)
    assert np.allclose(forecasts, y_test, atol=1e-8)


def test_ols_direct_h0():
    """Test basic direct fit with synthetic data."""
    # Test direct forecasting for a single horizon (h=1)
    horizon = 0
    steps = horizon + 1

    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    beta = model.betas_[horizon]
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    assert np.allclose(forecasts.iloc[0], y_test.iloc[0], atol=1e-8)


def test_ols_direct_h1():
    """Test basic direct fit with synthetic data."""
    # Test direct forecasting for a single horizon (h=1)
    horizon = 1
    steps = horizon + 1
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    beta = model.betas_[horizon]
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    assert np.allclose(forecasts.iloc[horizon], y_test.iloc[horizon], atol=1e-8)


def test_ols_recursive_two_y_lags():
    """Recursive forecasts with ``y_lags=2`` use the right value in each lag.

    With a noiseless AR(2) DGP the fitted coefficients are exact, so the
    three-step recursive path must reproduce the true path. This fails if
    every lag column is overwritten with the latest prediction.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_ar2_data(n_train=200, n_test=3)

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train, y_lags=2)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_.ravel()
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["a1"], atol=1e-8)
    assert np.isclose(beta[3], true_coef["a2"], atol=1e-8)
    assert np.allclose(forecasts.values, y_test.values, atol=1e-8)


def test_ols_recursive_four_y_lags():
    """Recursive forecasts with ``y_lags=4`` track all four lags independently.

    With a noiseless AR(4) DGP the fitted coefficients are exact, so the
    five-step recursive path must reproduce the true path.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_ar4_data(n_train=200, n_test=5)

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train, y_lags=4)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_.ravel()
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["a1"], atol=1e-8)
    assert np.isclose(beta[3], true_coef["a2"], atol=1e-8)
    assert np.isclose(beta[4], true_coef["a3"], atol=1e-8)
    assert np.isclose(beta[5], true_coef["a4"], atol=1e-8)
    assert np.allclose(forecasts.values, y_test.values, atol=1e-8)


def test_ols_decomposition_two_y_lags_matches_forecast():
    """Decomposition with ``y_lags=2`` reproduces the recursive forecast path."""
    y_train, X_train, y_test, X_test, _ = sample_ar2_data(n_train=200, n_test=3)

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train, y_lags=2)

    forecast = model.forecast(steps=len(y_test), X=X_test)
    X_aug = build_lagged_design(y_train, X_test, y_lags=2, X_lags=0)
    decomps = model._forecast_decomp(steps=len(y_test), X=X_aug)

    # Rows are emitted one block of components per recursive step.
    steps = len(y_test)
    n_components = decomps.shape[0] // steps
    totals = decomps["contribution"].values.reshape(steps, n_components).sum(axis=1)

    assert np.allclose(totals, forecast.values.ravel(), atol=1e-8)
    assert np.allclose(totals, y_test.values.ravel(), atol=1e-8)


def test_ols_decomposition_recursive_reconstructs_forecast():
    """Contributions sum to the forecast value at every horizon (recursive)."""
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train)

    forecast = model.forecast(steps=len(y_test), X=X_test)
    decomps = model._forecast_decomp(steps=len(y_test), X=X_test)

    assert decomps is not None
    minimal_decomposition_schema.validate(decomps)

    # Check that contributions sum to forecast for each horizon
    for h in range(len(y_test)):
        decomp_sum = decomps.loc[decomps["forecast_horizon"] == h, "contribution"].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-9)


def test_ols_recursive_decomposition_ignores_surplus_forecast_rows():
    """Decomposition uses the same requested horizon rows as forecasting."""
    y_train, X_train, _, X_test, _ = sample_regression_data(
        n_train=100,
        n_test=6,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )
    steps = 2

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train)

    forecast = model.forecast(steps=steps, X=X_test)
    decomp = model._forecast_decomp(steps=steps, X=X_test)

    assert decomp["forecast_horizon"].unique().tolist() == [0, 1]
    totals = decomp.groupby("forecast_horizon")["contribution"].sum()
    np.testing.assert_allclose(totals.to_numpy(), forecast.iloc[:, 0], atol=1e-9)


def test_ols_decomposition_direct_reconstructs_forecast():
    """Contributions sum to the forecast value at every horizon (direct)."""
    horizon = 1
    steps = horizon + 1

    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    model.fit(y_train, X=X_train)

    forecast = model.forecast(steps=steps, X=X_test)
    decomps = model._forecast_decomp(steps=steps, X=X_test)

    assert decomps is not None
    minimal_decomposition_schema.validate(decomps)

    # Check that contributions sum to forecast for each horizon
    for h in range(steps):
        decomp_sum = decomps.loc[decomps["forecast_horizon"] == h, "contribution"].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-9)


def test_ols_direct_decomposition_uses_first_forecast_row():
    """Direct decomposition mirrors the single row used by forecasting."""
    horizon = 1
    steps = horizon + 1
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    model.fit(y_train, X=X_train)
    forecast = model.forecast(steps=steps, X=X_test)
    decomp = model._forecast_decomp(steps=steps, X=X_test)

    n_components = model.N_regressors
    assert len(decomp) == steps * n_components
    assert decomp["forecast_horizon"].tolist() == [0] * n_components + [1] * n_components

    X_first = np.concatenate(([1.0], X_test.iloc[0].to_numpy()))
    for h in range(steps):
        horizon_decomp = decomp[decomp["forecast_horizon"] == h]
        expected_contributions = X_first * model.betas_[h].reshape(-1)
        np.testing.assert_allclose(
            horizon_decomp["contribution"], expected_contributions, atol=1e-9
        )
        np.testing.assert_allclose(
            horizon_decomp["contribution"].sum(), forecast.iloc[h, 0], atol=1e-9
        )


def test_ols_recursive_lag_decomposition_labels_each_horizon():
    """Recursive lag decomposition mirrors each forecast horizon."""
    y_train, X_train, y_test, X_test, _ = sample_ar2_data(n_train=200, n_test=3)

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train, y_lags=2)
    forecast = model.forecast(steps=len(y_test), X=X_test)
    X_aug = build_lagged_design(y_train, X_test, y_lags=2, X_lags=0)
    decomp = model._forecast_decomp(steps=len(y_test), X=X_aug)

    n_components = model.N_regressors
    assert len(decomp) == len(y_test) * n_components
    assert decomp["forecast_horizon"].tolist() == [
        h for h in range(len(y_test)) for _ in range(n_components)
    ]

    for h in range(len(y_test)):
        horizon_decomp = decomp[decomp["forecast_horizon"] == h]
        np.testing.assert_allclose(
            horizon_decomp["contribution"].sum(), forecast.iloc[h, 0], atol=1e-9
        )


def test_ols_dummy_absorbs_outlier():
    """A point dummy on an outlier date absorbs the outlier.

    With a noiseless DGP, injecting a large outlier into one in-sample
    observation biases the OLS coefficients. Adding a point dummy on that
    date absorbs the outlier exactly, so the remaining coefficients are
    recovered and the dummy coefficient equals the injected magnitude.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    # Inject a large outlier into a single in-sample observation.
    outlier_date = y_train.index[50]
    outlier_size = 100.0
    y_train.loc[outlier_date] = y_train.loc[outlier_date] + outlier_size

    # Without a dummy the outlier biases the slope coefficient.
    biased = rt.models.ForecastOLS(forecast_strategy="recursive")
    biased.fit(y_train, X=X_train)
    assert not np.isclose(biased.beta_[1], true_coef["b1"], atol=1e-6)

    # With a point dummy on the outlier date the coefficients are recovered.
    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train, dummies=[outlier_date])

    # Daily index -> ISO-date fallback name, appended as the last design column.
    dummy_name = f"D_{outlier_date.date()}"
    assert dummy_name in model.X.columns
    assert list(model.X.columns)[-1] == dummy_name

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    # The dummy coefficient equals the injected outlier magnitude.
    assert np.isclose(beta[-1], outlier_size, atol=1e-8)

    # Forecasts on clean test data are exact (the dummy is zero over the horizon).
    forecasts = model.forecast(steps=len(y_test), X=X_test)
    np.testing.assert_allclose(forecasts.values.ravel(), y_test.values.ravel(), atol=1e-8)


def test_ols_dummy_with_scaling_recovers_coefs():
    """A dummy with scale=True does not perturb coefficient estimation.

    The dummy is fitted via Frisch-Waugh-Lovell, so it is unpenalised and never
    scaled while the structural regressors are scaled. On a noiseless DGP the
    structural coefficients equal the true values and the dummy absorbs the
    whole injected outlier, confirming scaling and the dummy are independent.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    # Inject a large outlier and flag it with a dummy.
    outlier_date = y_train.index[50]
    outlier_size = 100.0
    y_train.loc[outlier_date] = y_train.loc[outlier_date] + outlier_size
    dummy_name = f"D_{outlier_date.date()}"

    model = rt.models.ForecastOLS(forecast_strategy="recursive", scale=True)
    model.fit(y_train, X=X_train, dummies=[outlier_date])

    cols = ["intercept"] + list(model.X.columns)
    beta = model.beta_.ravel()

    # Structural coefficients equal the true DGP values despite scaling.
    assert np.isclose(beta[cols.index("intercept")], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[cols.index("x1")], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[cols.index("x2")], true_coef["b2"], atol=1e-8)
    # The dummy absorbs the whole outlier (unpenalised and unscaled).
    assert np.isclose(beta[cols.index(dummy_name)], outlier_size, atol=1e-8)

    forecasts = model.forecast(steps=len(y_test), X=X_test)
    np.testing.assert_allclose(forecasts.values.ravel(), y_test.values.ravel(), atol=1e-8)

    """Each dummy date gets its own design column and coefficient.

    Two outliers of different sign and magnitude are each absorbed by their
    own dummy coefficient, leaving the structural coefficients unbiased.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    # Inject two outliers of different sign/size at distinct in-sample dates.
    date_a, date_b = y_train.index[20], y_train.index[70]
    size_a, size_b = 30.0, -50.0
    y_train.loc[date_a] = y_train.loc[date_a] + size_a
    y_train.loc[date_b] = y_train.loc[date_b] + size_b

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train, dummies=[date_a, date_b])

    # One extra design column (and coefficient) per dummy date.
    name_a = f"D_{date_a.date()}"
    name_b = f"D_{date_b.date()}"
    assert name_a in model.X.columns
    assert name_b in model.X.columns
    assert model.X.shape[1] == X_train.shape[1] + 2

    beta = model.beta_
    # Structural coefficients are recovered despite the two outliers.
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    # Each dummy coefficient equals its own injected magnitude.
    cols = list(model.X.columns)
    # +1 offset because beta_[0] is the intercept.
    assert np.isclose(beta[cols.index(name_a) + 1], size_a, atol=1e-8)
    assert np.isclose(beta[cols.index(name_b) + 1], size_b, atol=1e-8)


def test_ols_all_zero_dummy_dropped():
    """A dummy whose date is outside the fit window is dropped.

    In a real-time loop a dummy date may not yet be observed for an early
    vintage, leaving an all-zero column. Such columns carry no information,
    so they are dropped at fit; the forecast design stays aligned and the
    surviving dummy still absorbs its own outlier.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    in_window = y_train.index[40]  # observed -> kept
    out_window = y_test.index[5]  # in the forecast horizon -> all-zero at fit
    outlier_size = 25.0
    y_train.loc[in_window] = y_train.loc[in_window] + outlier_size

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train, dummies=[in_window, out_window])

    in_name = f"D_{in_window.date()}"
    out_name = f"D_{out_window.date()}"

    # Only the in-window dummy survives.
    assert model._dummy_cols == [in_name]
    assert in_name in model.X.columns
    assert out_name not in model.X.columns

    # Structural coefficients recovered; surviving dummy absorbs its outlier.
    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-8)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-8)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-8)
    assert np.isclose(beta[-1], outlier_size, atol=1e-8)

    # Forecast still runs and matches truth (design stays aligned with fit).
    forecasts = model.forecast(steps=len(y_test), X=X_test)
    np.testing.assert_allclose(forecasts.values.ravel(), y_test.values.ravel(), atol=1e-8)


def test_ols_formula_selects_dummies():
    """A formula keeps only the dummies named on its right-hand side.

    With ``y ~ x1 + D_<date>`` the listed dummy is kept while an unlisted
    dummy (and the unlisted regressor ``x2``) are dropped from the design.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    listed, unlisted = y_train.index[30], y_train.index[60]
    y_train.loc[listed] = y_train.loc[listed] + 40.0
    y_train.loc[unlisted] = y_train.loc[unlisted] - 20.0

    listed_name = f"D_{listed.date()}"
    unlisted_name = f"D_{unlisted.date()}"

    model = rt.models.ForecastOLS(formula=f"target ~ x1 + {listed_name}")
    model.fit(y_train, X=X_train, dummies=[listed, unlisted])

    # Only x1 and the listed dummy survive; x2 and the unlisted dummy are gone.
    assert list(model.X.columns) == ["x1", listed_name]
    assert model._dummy_cols == [listed_name]
    assert unlisted_name not in model.X.columns

    # Forecast design stays aligned with the fitted design.
    forecasts = model.forecast(steps=len(y_test), X=X_test)
    assert len(forecasts) == len(y_test)
    assert np.isfinite(forecasts.values).all()


# ============================================================================
# Forecast-availability checks (LinearRegression._forecast)
# ============================================================================
# X is filtered to rows after the last fitted date before prediction; these
# tests cover the resulting availability check for both forecast strategies:
# recursive needs `steps` future rows, direct only needs the first one.


def test_forecast_no_future_rows_raises_recursive():
    """Recursive strategy: X has no rows after the last fitted date."""
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=100, n_test=10, forecast_type="recursive", random_seed=123
    )
    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train)

    with pytest.raises(ValueError, match="need 5"):
        model.forecast(steps=5, X=X_train)


def test_forecast_no_future_rows_raises_direct():
    """Direct strategy: X has no rows after the last fitted date."""
    horizon = 3
    steps = horizon + 1
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        forecast_type="direct",
        horizon=horizon,
        random_seed=123,
    )
    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    model.fit(y_train, X=X_train)

    with pytest.raises(ValueError, match="need 1"):
        model.forecast(steps=steps, X=X_train)


def test_forecast_fewer_future_rows_than_steps_raises_recursive():
    """Recursive strategy needs `steps` future rows; fewer must raise."""
    y_train, X_train, _, X_test, _ = sample_regression_data(
        n_train=100, n_test=10, forecast_type="recursive", random_seed=123
    )
    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train)

    with pytest.raises(ValueError, match="need 5"):
        model.forecast(steps=5, X=X_test.iloc[:3])


def test_forecast_fewer_future_rows_than_steps_ok_direct():
    """Direct strategy only needs the first future row, regardless of steps."""
    horizon = 3
    steps = horizon + 1
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=123,
    )
    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    model.fit(y_train, X=X_train)

    # Only one future row available, but direct only ever uses X_aug.iloc[[0]].
    forecasts = model.forecast(steps=steps, X=X_test.iloc[:1])

    assert len(forecasts) == steps
    np.testing.assert_allclose(forecasts.iloc[horizon], y_test.iloc[horizon], atol=1e-8)


def test_forecast_exactly_steps_future_rows_ok_recursive():
    """Recursive strategy succeeds with exactly `steps` future rows."""
    steps = 5
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )
    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train)

    forecasts = model.forecast(steps=steps, X=X_test.iloc[:steps])

    assert len(forecasts) == steps
    np.testing.assert_allclose(
        forecasts.values.ravel(), y_test.iloc[:steps, 0].values, atol=1e-8
    )


def test_forecast_exactly_steps_future_rows_ok_direct():
    """Direct strategy succeeds with exactly `steps` future rows too (it only
    needs the first one, so this is well within its requirement)."""
    horizon = 3
    steps = horizon + 1
    y_train, X_train, y_test, X_test, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="direct",
        horizon=horizon,
        random_seed=123,
    )
    model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    model.fit(y_train, X=X_train)

    forecasts = model.forecast(steps=steps, X=X_test.iloc[:steps])

    assert len(forecasts) == steps
    np.testing.assert_allclose(forecasts.iloc[horizon], y_test.iloc[horizon], atol=1e-8)


def test_fitted_values_recovers_insample():
    """fitted_values_ recovers y_train in-sample on noiseless data."""
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastOLS(forecast_strategy="recursive")
    model.fit(y_train, X=X_train)

    fitted = model.fitted_values_.dropna()
    np.testing.assert_allclose(
        fitted.to_numpy(), y_train.loc[fitted.index].to_numpy().ravel(), atol=1e-8
    )
