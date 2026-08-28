"""Tests for Ridge model."""

import numpy as np
import pytest

import forecast_realtime as rt

from ..schemas import minimal_decomposition_schema
from .sample_regression import sample_regression_data


def test_ridge_cv_uses_default_alphas_when_omitted():
    """K-fold CV uses RidgeCV's candidate grid when alphas is omitted."""
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        noise_std=0.1,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastRidge(cv=5, scale=True)
    model.fit(y_train, X=X_train)

    assert model.alpha is None
    assert model.alphas is None
    assert model.alpha_ in (0.1, 1.0, 10.0)


def test_ridge_cv_applies_selected_penalty():
    """The default CV grid selects a penalty that shrinks noisy coefficients."""
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

    cv_model = rt.models.ForecastRidge(cv=5, scale=True)
    unpenalised_model = rt.models.ForecastRidge(alpha=1e-12, scale=True)
    cv_model.fit(y_train, X=X_train)
    unpenalised_model.fit(y_train, X=X_train)

    assert cv_model.alpha_ > 1e-8
    assert np.linalg.norm(cv_model.beta_[1:]) < 0.95 * np.linalg.norm(
        unpenalised_model.beta_[1:]
    )


@pytest.mark.parametrize("alphas", [1, 1.0, np.array(1.0)])
def test_ridge_cv_rejects_scalar_alphas(alphas):
    """K-fold RidgeCV requires an array-like candidate grid."""
    with pytest.raises(TypeError, match="alphas must be array-like"):
        rt.models.ForecastRidge(cv=5, alphas=alphas)


def test_ridge_fixed_alpha_without_cv():
    """A fixed alpha uses Ridge rather than cross-validation."""
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=100,
        n_test=10,
        noise_std=0.1,
        forecast_type="recursive",
        random_seed=123,
    )

    model = rt.models.ForecastRidge(alpha=2.0, scale=True)
    model.fit(y_train, X=X_train)

    assert model.alpha == 2.0
    assert model.alpha_ == 2.0


def test_ridge_rejects_alpha_grid_without_cv():
    """Candidate alpha grids are only valid when cross-validation is enabled."""
    with pytest.raises(TypeError, match="alphas can only be set"):
        rt.models.ForecastRidge(alphas=[0.1, 1.0])


def test_ridge_rejects_unknown_alpha_scaling():
    """Only the documented loss conventions are accepted."""
    with pytest.raises(ValueError, match="alpha_scaling must be one of"):
        rt.models.ForecastRidge(alpha_scaling="bogus")


def test_ridge_alpha_scalings_are_reparametrisations():
    """ "mean" and "sum" are the same penalty expressed on different loss scales.

    sklearn's Ridge penalises the summed squared residual, so alpha=a on the
    "mean" scale is a*n on the "sum" scale. Both must produce identical
    coefficients.
    """
    n_train = 200
    y_train, X_train, _, _, _ = sample_regression_data(
        n_train=n_train,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=1.0,
        forecast_type="recursive",
        random_seed=123,
    )

    a = 0.5
    mean_model = rt.models.ForecastRidge(alpha=a, alpha_scaling="mean", scale=True)
    sum_model = rt.models.ForecastRidge(
        alpha=a * n_train, alpha_scaling="sum", scale=True
    )
    for model in (mean_model, sum_model):
        model.fit(y_train, X=X_train)

    np.testing.assert_allclose(mean_model.beta_, sum_model.beta_, rtol=1e-10)

    # alpha_ is reported back on whichever scale the user supplied.
    assert mean_model.alpha_ == a
    assert sum_model.alpha_ == a * n_train


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore:Linear regression models with a zero l1")
def test_ridge_mean_scaling_matches_elasticnet_l1_ratio_zero():
    """The "mean" scale is exactly ElasticNet's penalty at l1_ratio=0.

    ElasticNet minimises
    ``1/(2n)||y - Xw||^2 + alpha*l1_ratio*||w||_1 + 0.5*alpha*(1-l1_ratio)*||w||^2``,
    which at l1_ratio=0 reduces to ``(1/n)||y - Xw||^2 + alpha*||w||^2`` once
    scaled by 2n. Pinning this equivalence is what makes ``alpha`` directly
    comparable with ForecastLasso/ForecastElasticNet, which use the same
    normalisation. We compare against sklearn's ElasticNet rather than reuse
    it as the solver, since coordinate descent is slower and far less accurate
    at small alpha than Ridge's closed form.
    """
    from sklearn.linear_model import ElasticNet

    rng = np.random.default_rng(0)
    n, k = 300, 8
    X = rng.normal(size=(n, k))
    y = X @ rng.normal(size=k) + rng.normal(scale=2.0, size=n)

    alpha = 0.5
    reference = ElasticNet(
        alpha=alpha, l1_ratio=0.0, fit_intercept=False, max_iter=500_000, tol=1e-14
    ).fit(X, y)

    model = rt.models.ForecastRidge(
        alpha=alpha, alpha_scaling="mean", fit_intercept=False
    )
    beta = model._fit_reg(y=y, X=X).ravel()

    np.testing.assert_allclose(beta, reference.coef_, atol=1e-8)


def test_ridge_mean_scaling_is_stable_across_sample_size():
    """The "mean" scale keeps a fixed alpha's shrinkage comparable as n grows.

    Under sklearn's raw "sum" convention the same alpha bites progressively
    less as the sample grows, because the residual sum scales with n while the
    penalty does not. Normalising by n removes that drift.
    """
    fitted = {}
    for scaling in ("sum", "mean"):
        norms = []
        for n_train in (100, 400):
            y_train, X_train, _, _, _ = sample_regression_data(
                n_train=n_train,
                n_test=10,
                b1=2.0,
                b2=-1.0,
                noise_std=1.0,
                forecast_type="recursive",
                random_seed=123,
            )
            model = rt.models.ForecastRidge(alpha=1.0, alpha_scaling=scaling, scale=True)
            model.fit(y_train, X=X_train)
            norms.append(np.linalg.norm(model.beta_[1:]))
        fitted[scaling] = norms

    def drift(norms):
        return abs(norms[1] - norms[0]) / norms[0]

    assert drift(fitted["mean"]) < drift(fitted["sum"])


def test_ridge_scaling():
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

    model = rt.models.ForecastRidge(forecast_strategy="recursive", scale=True, alpha=1e-7)

    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts, y_test, atol=1e-4)


def test_ridge_recursive():
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

    model = rt.models.ForecastRidge(forecast_strategy="recursive", alpha=1e-7)

    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts, y_test, atol=1e-4)


def test_ridge_recursive_with_ar():
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

    model = rt.models.ForecastRidge(forecast_strategy="recursive", alpha=1e-7)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=len(y_test), X=X_test)

    beta = model.beta_

    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.isclose(beta[3], true_coef["b_ar"], atol=1e-4)
    assert np.allclose(forecasts, y_test, atol=1e-4)


def test_ridge_direct_h0():
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

    model = rt.models.ForecastRidge(forecast_strategy="direct", steps=steps, alpha=1e-7)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    beta = model.betas_[horizon]
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts.iloc[0], y_test.iloc[0], atol=1e-4)


def test_ridge_direct_h1():
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

    model = rt.models.ForecastRidge(forecast_strategy="direct", steps=steps, alpha=1e-7)
    model.fit(y_train, X=X_train)
    forecasts = model.forecast(steps=steps, X=X_test)

    beta = model.betas_[horizon]
    assert np.isclose(beta[0], true_coef["cst"], atol=1e-4)
    assert np.isclose(beta[1], true_coef["b1"], atol=1e-4)
    assert np.isclose(beta[2], true_coef["b2"], atol=1e-4)
    assert np.allclose(forecasts.iloc[horizon], y_test.iloc[horizon], atol=1e-4)


def test_ridge_decomposition_recursive_reconstructs_forecast():
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

    model = rt.models.ForecastRidge(forecast_strategy="recursive", alpha=1e-7)
    model.fit(y_train, X=X_train)

    forecast = model.forecast(steps=len(y_test), X=X_test)
    decomps = model._forecast_decomp(steps=len(y_test), X=X_test)

    assert decomps is not None
    minimal_decomposition_schema.validate(decomps)

    # Check that contributions sum to forecast for each horizon
    for h in range(len(y_test)):
        decomp_sum = decomps.loc[decomps["forecast_horizon"] == h, "contribution"].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-9)


def test_ridge_decomposition_direct_reconstructs_forecast():
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

    model = rt.models.ForecastRidge(forecast_strategy="direct", steps=steps, alpha=1e-7)
    model.fit(y_train, X=X_train)

    forecast = model.forecast(steps=steps, X=X_test)
    decomps = model._forecast_decomp(steps=steps, X=X_test)

    assert decomps is not None
    minimal_decomposition_schema.validate(decomps)

    # Check that contributions sum to forecast for each horizon
    for h in range(steps):
        decomp_sum = decomps.loc[decomps["forecast_horizon"] == h, "contribution"].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-9)


def test_ridge_dummy_is_unpenalised_and_unscaled():
    """A dummy with scale=True and near-zero alpha recovers the true coefs.

    The dummy is fitted via Frisch-Waugh-Lovell so it carries no L2 penalty and
    is not standardised. With a negligible alpha on noiseless data, neither
    scaling nor the dummy should perturb coefficient estimation: the structural
    coefficients equal the true DGP values and the dummy absorbs the whole
    injected outlier.
    """
    y_train, X_train, y_test, X_test, true_coef = sample_regression_data(
        n_train=120,
        n_test=10,
        b1=2.0,
        b2=-1.0,
        noise_std=0,
        forecast_type="recursive",
        random_seed=123,
    )

    alpha = np.array([1e-8])

    # Inject a large outlier and flag it with a dummy.
    outlier_date = y_train.index[30]
    y_dirty = y_train.copy()
    y_dirty.loc[outlier_date] = y_dirty.loc[outlier_date] + 100.0
    dummy_name = f"D_{outlier_date.date()}"

    model = rt.models.ForecastRidge(scale=True, alpha=alpha[0])
    model.fit(y_dirty, X=X_train, dummies=[outlier_date])
    cols = ["intercept"] + list(model.X.columns)
    beta = model.beta_.ravel()

    # Structural coefficients equal the true DGP values: neither scaling nor the
    # dummy affected estimation.
    assert np.isclose(beta[cols.index("intercept")], true_coef["cst"], atol=1e-6)
    assert np.isclose(beta[cols.index("x1")], true_coef["b1"], atol=1e-6)
    assert np.isclose(beta[cols.index("x2")], true_coef["b2"], atol=1e-6)

    # The dummy absorbs the whole outlier (unpenalised, so no shrinkage).
    assert np.isclose(beta[cols.index(dummy_name)], 100.0, atol=1e-6)

    # Forecasting still runs and stays finite.
    forecasts = model.forecast(steps=len(y_test), X=X_test)
    assert len(forecasts) == len(y_test)
    assert np.isfinite(forecasts.values).all()


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

    model = rt.models.ForecastRidge(forecast_strategy="recursive", alpha=1e-7)
    model.fit(y_train, X=X_train)

    fitted = model.fitted_values_.dropna()
    np.testing.assert_allclose(
        fitted.to_numpy(), y_train.loc[fitted.index].to_numpy().ravel(), atol=1e-4
    )
