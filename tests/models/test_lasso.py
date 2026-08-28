"""Tests for Lasso model."""

import numpy as np
import pytest

import forecast_realtime as rt

from ..schemas import minimal_decomposition_schema
from .sample_regression import sample_regression_data


def test_lasso_recursive():
    """Recover true betas with recursive forecasting and near-zero penalty."""
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastLasso(forecast_strategy="recursive", alpha=1e-8)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts, y_test, atol=1e-4)


def test_lasso_scaling():
    """Recover true betas with scale=True and near-zero penalty."""
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastLasso(forecast_strategy="recursive", scale=True, alpha=1e-8)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts, y_test, atol=1e-4)


def test_lasso_recursive_with_ar():
    """Recover true betas including autoregressive lag."""
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

    model = rt.models.ForecastLasso(forecast_strategy="recursive", alpha=1e-8)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.isclose(beta[3], true_coef["b_ar"], atol=1e-4)
    assert np.allclose(forecasts, y_test, atol=1e-3)


def test_lasso_direct_h0():
    """Direct forecasting at horizon 0."""
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

    model = rt.models.ForecastLasso(forecast_strategy="direct", steps=steps, alpha=1e-8)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    beta = model.betas_[horizon]
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts.iloc[0], y_test.iloc[0], atol=1e-4)


def test_lasso_direct_h1():
    """Direct forecasting at horizon 1."""
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

    model = rt.models.ForecastLasso(forecast_strategy="direct", steps=steps, alpha=1e-8)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    beta = model.betas_[horizon]
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts.iloc[horizon], y_test.iloc[horizon], atol=1e-4)


def test_lasso_cv():
    """Cross-validated Lasso runs and populates best_alpha."""
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=100,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0.1,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastLasso(forecast_strategy="recursive", cv=5)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    assert model.alphas is None
    assert model.best_alpha is not None
    assert model.best_alpha > 0
    assert forecasts.shape[0] == len(y_test)


@pytest.mark.parametrize("alphas", [1, 1.0, np.array(1.0)])
def test_lasso_cv_rejects_scalar_alphas(alphas):
    """Cross-validated Lasso requires an array-like candidate grid."""
    with pytest.raises(TypeError, match="alphas must be array-like"):
        rt.models.ForecastLasso(cv=5, alphas=alphas)


def test_lasso_cv_accepts_explicit_alpha_grid():
    """Cross-validated Lasso selects from an explicit candidate grid."""
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        noise_std=0.1,
        forecast_type="recursive",
        random_seed=123,
    )
    alphas = np.array([0.01, 0.1, 1.0])

    model = rt.models.ForecastLasso(cv=5, alphas=alphas, scale=True)
    model.fit(y_train, X=X_train)

    assert model.best_alpha in alphas


def test_lasso_cv_applies_selected_penalty():
    """CV selects from the supplied grid and shrinks noisy coefficients."""
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=60,
        n_test=10,
        cst=0,
        b1=0,
        b2=0,
        noise_std=1,
        forecast_type="recursive",
        random_seed=2026,
    )
    alphas = np.array([1.0, 100.0])

    cv_model = rt.models.ForecastLasso(cv=5, alphas=alphas, scale=True)
    unpenalised_model = rt.models.ForecastLasso(alpha=1e-8, scale=True)
    cv_model.fit(y_train, X=X_train)
    unpenalised_model.fit(y_train, X=X_train)

    assert cv_model.best_alpha in alphas
    assert np.linalg.norm(cv_model.beta_[1:]) < 0.1 * np.linalg.norm(
        unpenalised_model.beta_[1:]
    )


def test_lasso_decomposition_recursive_reconstructs_forecast():
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

    model = rt.models.ForecastLasso(forecast_strategy="recursive", alpha=1e-8)
    model.fit(y_train, X=X_train)

    forecast = model.forecast(steps=len(y_test), X=X_test)
    decomps = model._forecast_decomp(steps=len(y_test), X=X_test)

    assert decomps is not None
    minimal_decomposition_schema.validate(decomps)

    for h in range(len(y_test)):
        decomp_sum = decomps.loc[decomps["forecast_horizon"] == h, "contribution"].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-9)


def test_lasso_decomposition_direct_reconstructs_forecast():
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

    model = rt.models.ForecastLasso(forecast_strategy="direct", steps=steps, alpha=1e-8)
    model.fit(y_train, X=X_train)

    forecast = model.forecast(steps=steps, X=X_test)
    decomps = model._forecast_decomp(steps=steps, X=X_test)

    assert decomps is not None
    minimal_decomposition_schema.validate(decomps)

    for h in range(steps):
        decomp_sum = decomps.loc[decomps["forecast_horizon"] == h, "contribution"].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-9)


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

    model = rt.models.ForecastLasso(forecast_strategy="recursive", alpha=1e-8)
    model.fit(y_train, X=X_train)

    fitted = model.fitted_values_.dropna()
    np.testing.assert_allclose(
        fitted.to_numpy(), y_train.loc[fitted.index].to_numpy().ravel(), atol=1e-4
    )
