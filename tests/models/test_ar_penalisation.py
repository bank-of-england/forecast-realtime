"""Numerical contracts for generated, optionally unpenalised target lags."""

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.model_selection import TimeSeriesSplit, check_cv

from forecast_realtime.models import (
    ForecastElasticNet,
    ForecastLasso,
    ForecastOLS,
    ForecastRidge,
)

REGULARISED = [ForecastRidge, ForecastLasso, ForecastElasticNet]


@pytest.fixture
def ar_data():
    """Generate non-centred AR data with exogenous signal and a point outlier."""
    rng = np.random.default_rng(763)
    index = pd.date_range("2000-01-31", periods=124, freq="ME")
    X = pd.DataFrame(
        {"x": rng.normal(2, 3, len(index)), "z": rng.normal(-1, 0.6, len(index))},
        index=index,
    )
    values = np.zeros(len(index))
    for t in range(2, len(index)):
        values[t] = (
            1.5
            + 0.6 * values[t - 1]
            - 0.15 * values[t - 2]
            + 0.8 * X.iloc[t, 0]
            - 1.2 * X.iloc[t, 1]
            + rng.normal(scale=0.4)
            + (8 if t == 40 else 0)
        )
    return pd.DataFrame({"y": values[:120]}, index=index[:120]), X


def _design(model, horizon=0):
    """Return the raw estimation arrays, aligned to one direct horizon."""
    target = model.y_fit.to_numpy()
    design = model.X.loc[model.y_fit.index].to_numpy()
    names = list(model.X.columns)
    if model.fit_intercept:
        design = np.column_stack([np.ones(len(design)), design])
        names = ["intercept", *names]
    return target[horizon:], design[:-horizon] if horizon else design, names


def _ridge_reference(target, design, names, exempt, alpha, normalisation, scale):
    """Solve the raw augmented least-squares problem with explicit penalty rows."""
    penalised = np.array([name not in exempt for name in names])
    weights = penalised.astype(float)
    if scale and penalised.any():
        U = design[:, ~penalised]
        residual = (
            design[:, penalised]
            - U @ np.linalg.lstsq(U, design[:, penalised], rcond=None)[0]
        )
        std = residual.std(axis=0)
        weights[penalised] = np.where(std == 0, 1, std)
    strength = alpha * (len(target) if normalisation == "mean" else 1)
    augmented = np.vstack([design, np.sqrt(strength) * np.diag(weights)])
    response = np.vstack([target, np.zeros((design.shape[1], 1))])
    return np.linalg.lstsq(augmented, response, rcond=None)[0]


@pytest.mark.parametrize("model_class", REGULARISED)
def test_constructor_default_and_explicit_ar_penalty(model_class, ar_data):
    """The appended constructor option defaults to exemption and reaches fitting."""
    parameters = inspect.signature(model_class).parameters
    assert list(parameters)[-1] == "penalise_ar"
    assert parameters["penalise_ar"].default is False
    y, X = ar_data
    models = [
        model_class(alpha=100, **option)
        for option in ({}, {"penalise_ar": False}, {"penalise_ar": True})
    ]
    for model in models:
        model.fit(y, X=X, y_lags=2, dummies=[y.index[40]])
    np.testing.assert_array_equal(models[0].beta_, models[1].beta_)
    assert models[0].penalise_ar is False
    assert models[2].penalise_ar is True
    ar_idx = [list(models[0].X.columns).index(f"y_lag{lag}") + 1 for lag in (1, 2)]
    assert np.linalg.norm(models[2].beta_[ar_idx]) < np.linalg.norm(
        models[0].beta_[ar_idx]
    )
    for model in models:
        target, design, names = _design(model)
        dummy_idx = [names.index(name) for name in model._dummy_cols]
        np.testing.assert_allclose(
            design[:, dummy_idx].T @ (target - design @ model.beta_), 0, atol=1e-10
        )


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("normalisation", ["sum", "mean"])
@pytest.mark.parametrize("lags,dummies", [(1, False), (2, True)])
def test_ridge_matches_augmented_least_squares(
    ar_data, fit_intercept, scale, normalisation, lags, dummies
):
    """Every direct horizon matches independently weighted raw-space least squares."""
    y, X = ar_data
    alpha = 4.0 if normalisation == "sum" else 0.8
    model = ForecastRidge(
        alpha=alpha,
        alpha_scaling=normalisation,
        fit_intercept=fit_intercept,
        scale=scale,
        forecast_strategy="direct",
        steps=3,
    )
    model.fit(y, X=X, y_lags=lags, dummies=[y.index[40]] if dummies else None)
    exempt = {
        "intercept",
        *(f"y_lag{lag}" for lag in range(1, lags + 1)),
        *model._dummy_cols,
    }
    for horizon, beta in model.betas_.items():
        target, design, names = _design(model, horizon)
        expected = _ridge_reference(
            target, design, names, exempt, alpha, normalisation, scale
        )
        assert beta.shape == (len(names), 1)
        np.testing.assert_allclose(beta, expected, atol=1e-10, rtol=1e-10)
    _, design, _ = _design(model)
    np.testing.assert_allclose(model.fitted_values_, (design @ model.betas_[0]).ravel())
    assert model.alpha_ == alpha


@pytest.mark.parametrize("model_class", [ForecastLasso, ForecastElasticNet])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("alpha", [0.05, 100.0])
def test_l1_penalties_satisfy_optimality_conditions(
    ar_data, model_class, fit_intercept, scale, alpha
):
    """Exempt terms have zero score; other terms satisfy the L1/L2 KKT conditions."""
    y, X = ar_data
    model = model_class(alpha=alpha, scale=scale, fit_intercept=fit_intercept)
    model.fit(y, X=X, y_lags=2, dummies=[y.index[40]])
    target, design, names = _design(model)
    penalised = np.array([name in ("x", "z") for name in names])
    U, P = design[:, ~penalised], design[:, penalised]
    residual = target - design @ model.beta_
    np.testing.assert_allclose(U.T @ residual, 0, atol=1e-8)
    P_res = P - U @ np.linalg.lstsq(U, P, rcond=None)[0]
    y_res = target - U @ np.linalg.lstsq(U, target, rcond=None)[0]
    P_std = P_res.std(axis=0) if scale else np.ones(P.shape[1])
    y_std = y_res.std(axis=0) if scale else np.ones(1)
    if scale and fit_intercept:
        P_res = P_res - P_res.mean(axis=0)
        y_res = y_res - y_res.mean(axis=0)
    P_in, y_in = P_res / P_std, y_res / y_std
    beta = model.beta_[penalised].ravel() * P_std / y_std
    score = (P_in.T @ (y_in.ravel() - P_in @ beta)) / len(target)
    ratio = getattr(model, "l1_ratio", 1.0)
    score -= alpha * (1 - ratio) * beta
    active = np.abs(beta) > 1e-9
    np.testing.assert_allclose(
        score[active], alpha * ratio * np.sign(beta[active]), atol=2e-5
    )
    assert np.all(np.abs(score[~active]) <= alpha * ratio + 2e-5)
    if alpha == 100:
        np.testing.assert_array_equal(beta, 0)
        np.testing.assert_allclose(
            model.beta_[~penalised], np.linalg.lstsq(U, target, rcond=None)[0], atol=1e-10
        )


@pytest.mark.parametrize("scale", [False, True])
def test_only_exact_retained_generated_lags_are_exempt(ar_data, scale):
    """Exogenous lags, formula-like labels and prefix-sharing names stay penalised."""
    y, X = ar_data
    X = X.assign(y_lag99=X["z"] ** 2, y_lag1_extra=X["x"] ** 2)
    X["y_lag1:x"] = X["x"] * X["z"]
    X["I(y_lag1 ** 2)"] = X["z"] ** 3
    names = [
        "y_lag99",
        "y_lag2",
        "x",
        "x_lag1",
        "y_lag1_extra",
        "y_lag1:x",
        "I(y_lag1 ** 2)",
    ]
    model = ForecastRidge(alpha=0.8, scale=scale, formula="y ~ " + " + ".join(names))
    model.fit(y, X=X, y_lags=2, X_lags={"x": 1})
    target, design, actual_names = _design(model)
    assert actual_names == ["intercept", *names]
    expected = _ridge_reference(
        target, design, actual_names, {"intercept", "y_lag2"}, 0.8, "mean", scale
    )
    np.testing.assert_allclose(model.beta_, expected, atol=1e-10)


@pytest.mark.parametrize("model_class", REGULARISED)
@pytest.mark.parametrize("cv", [3, TimeSeriesSplit(n_splits=3)])
@pytest.mark.parametrize("strategy", ["recursive", "direct"])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_internal_cv_matches_training_fold_reference(
    ar_data, model_class, cv, strategy, scale, fit_intercept
):
    """CV selects and refits penalties using independent training-fold solutions."""
    y, X = ar_data
    alphas = [0.001, 0.1, 1.0, 10.0]
    model = model_class(
        cv=cv,
        alphas=alphas,
        forecast_strategy=strategy,
        steps=2,
        scale=scale,
        fit_intercept=fit_intercept,
    )
    model.fit(y, X=X, y_lags=2, dummies=[y.index[40]])
    betas = model.betas_ if strategy == "direct" else {0: model.beta_}
    for horizon, beta in betas.items():
        target, design, names = _design(model, horizon)
        alpha, expected = _cv_reference(model, target, design, names, alphas)
        np.testing.assert_allclose(beta, expected, atol=1e-8)
    assert getattr(model, model._penalty_attribute) == alpha
    assert model.alpha is None
    assert model.alphas == alphas
    assert repr(model.cv) == repr(cv)
    result = model.forecast(steps=2, X=X, decomp=True)
    np.testing.assert_allclose(
        result.decomposition.groupby("forecast_horizon")["contribution"].sum(),
        result.iloc[:, 0],
        atol=1e-9,
    )


def _cv_reference(model, target, design, names, alphas):
    """Select by raw validation MSE without calling any production fit helpers."""
    exempt = {"intercept", *model._dummy_cols}
    if not model.penalise_ar:
        exempt.update(f"y_lag{lag}" for lag in range(1, model.y_lags + 1))
    penalised = np.array([name not in exempt for name in names])

    def fit(y, X, alpha, ratio):
        if isinstance(model, ForecastRidge):
            return _ridge_reference(
                y, X, names, exempt, alpha, model.alpha_scaling, model.scale
            )
        U, P = X[:, ~penalised], X[:, penalised]
        yr = y - U @ np.linalg.lstsq(U, y, rcond=None)[0]
        Pr = P - U @ np.linalg.lstsq(U, P, rcond=None)[0]
        ps, ys = np.ones(P.shape[1]), np.ones(1)
        if model.scale:
            ps, ys = Pr.std(axis=0), yr.std(axis=0)
            ps, ys = np.where(ps == 0, 1, ps), np.where(ys == 0, 1, ys)
            if model.fit_intercept:
                Pr, yr = Pr - Pr.mean(axis=0), yr - yr.mean(axis=0)
        if alphas is None:
            maximum = np.abs((Pr / ps).T @ (yr / ys)).max() / (len(y) * ratio)
            alpha = max(alpha * maximum, np.finfo(float).resolution)
        solver = (
            Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
            if isinstance(model, ForecastLasso)
            else ElasticNet(
                alpha=alpha, l1_ratio=ratio, fit_intercept=False, max_iter=10000
            )
        )
        bp = solver.fit(Pr / ps, (yr / ys).ravel()).coef_ * ys / ps
        beta = np.zeros((X.shape[1], 1))
        beta[penalised, 0] = bp
        beta[~penalised] = np.linalg.lstsq(U, y - P @ bp[:, None], rcond=None)[0]
        return beta

    candidates = [
        (alpha, ratio)
        for ratio in np.atleast_1d(getattr(model, "l1_ratio", 1.0))
        for alpha in (np.logspace(0, -3, 100) if alphas is None else alphas)
    ]
    splits = list(check_cv(model.cv).split(design, target.ravel()))
    losses = [
        np.mean(
            [
                np.mean(
                    (
                        target[valid]
                        - design[valid] @ fit(target[train], design[train], alpha, ratio)
                    )
                    ** 2
                )
                for train, valid in splits
            ]
        )
        for alpha, ratio in candidates
    ]
    alpha, ratio = candidates[int(np.argmin(losses))]
    beta = fit(target, design, alpha, ratio)
    if alphas is None:
        U, P = design[:, ~penalised], design[:, penalised]
        yr = target - U @ np.linalg.lstsq(U, target, rcond=None)[0]
        Pr = P - U @ np.linalg.lstsq(U, P, rcond=None)[0]
        if model.scale:
            ps, ys = Pr.std(axis=0), yr.std(axis=0)
            if model.fit_intercept:
                Pr, yr = Pr - Pr.mean(axis=0), yr - yr.mean(axis=0)
            Pr = Pr / np.where(ps == 0, 1, ps)
            yr = yr / np.where(ys == 0, 1, ys)
        alpha = max(
            alpha * np.abs(Pr.T @ yr).max() / (len(target) * ratio),
            np.finfo(float).resolution,
        )
    return alpha, beta


@pytest.mark.parametrize("model_class", [ForecastLasso, ForecastElasticNet])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("automatic", [False, True])
def test_l1_cv_grid_matches_reference(ar_data, model_class, scale, automatic):
    """Automatic grids adapt to training data; explicit grids retain their values."""
    y, X = ar_data
    alphas = None if automatic else [0.03, 0.3, 3.0]
    options = {"l1_ratio": [0.3, 0.8]} if model_class is ForecastElasticNet else {}
    model = model_class(cv=TimeSeriesSplit(3), scale=scale, alphas=alphas, **options)
    model.fit(y, X=X, y_lags=2, dummies=[y.index[40]])
    alpha, expected = _cv_reference(model, *_design(model), alphas)
    assert model.best_alpha == pytest.approx(alpha)
    np.testing.assert_allclose(model.beta_, expected, atol=1e-8)
    if options:
        assert model.l1_ratio == options["l1_ratio"]


@pytest.mark.parametrize("normalisation", ["sum", "mean"])
@pytest.mark.parametrize("lags", [0, 2])
def test_ridge_dummy_cv_matches_reference(ar_data, normalisation, lags):
    """Dummy and AR CV use original-unit validation and each fold's loss scale."""
    y, X = ar_data
    model = ForecastRidge(
        cv=TimeSeriesSplit(3),
        alphas=[0.02, 0.2, 2.0],
        scale=True,
        alpha_scaling=normalisation,
        forecast_strategy="direct",
        steps=2,
    )
    model.fit(y, X=X, y_lags=lags, dummies=[y.index[40]])
    for horizon, beta in model.betas_.items():
        alpha, expected = _cv_reference(model, *_design(model, horizon), model.alphas)
        np.testing.assert_allclose(beta, expected, atol=1e-9)
    assert model.alpha_ == alpha


@pytest.mark.parametrize("model_class", REGULARISED)
def test_validation_values_do_not_enter_training_fits(model_class, monkeypatch):
    """Projection, scaling and automatic candidates depend only on training rows."""
    rng = np.random.default_rng(451)
    design = np.column_stack([np.ones(50), rng.normal(size=(50, 3))])
    y = rng.normal(size=(50, 1))
    train, valid = np.arange(35), np.arange(35, 50)
    original_prepare = model_class._partialled_problem
    original_fit = model_class._fit_reg
    preparations, fits = [], []

    def prepare(self, target, X, unpen, pen):
        preparations.append((target.copy(), X.copy()))
        return original_prepare(self, target, X, unpen, pen)

    def fit(self, y, X):
        fits.append((self.alpha, y.copy(), X.copy()))
        return original_fit(self, y, X)

    monkeypatch.setattr(model_class, "_partialled_problem", prepare)
    monkeypatch.setattr(model_class, "_fit_reg", fit)
    for changed in (False, True):
        target, X = y.copy(), design.copy()
        if changed:
            target[valid] += 500
            X[valid, 1:] *= 100
        model = model_class(cv=[(train, valid)], scale=True)
        model._fit_partialled(target, X, [0, 1], [2, 3])
    # One preparation for the training fold and one for the final refit per run.
    assert len(preparations) == 4
    for left, right in zip(preparations[0], preparations[2], strict=True):
        np.testing.assert_array_equal(left, right)
    count = len(fits) // 2
    assert count == 101
    for before, after in zip(fits[: count - 1], fits[count:-1], strict=True):
        for left, right in zip(before, after, strict=True):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("alphas", [[], [np.nan], [np.inf], [-1], [[0.1]]])
def test_fwl_cv_rejects_invalid_grids(ar_data, alphas):
    """Invalid explicit grids fail clearly before fitting candidates."""
    y, X = ar_data
    with pytest.raises(ValueError, match="alphas must"):
        ForecastRidge(cv=3, alphas=alphas).fit(y, X=X, y_lags=1)


def test_zero_ratio_requires_explicit_grid(ar_data):
    """Pure L2 has no automatic L1 path but accepts explicit alpha candidates."""
    y, X = ar_data
    with pytest.raises(ValueError, match="Supply alphas when l1_ratio=0"):
        ForecastElasticNet(cv=3, l1_ratio=0).fit(y, X=X, y_lags=1)

    model = ForecastElasticNet(cv=3, l1_ratio=0, alphas=[0.1, 1.0])
    model.fit(y, X=X, y_lags=1)
    target, design, names = _design(model)
    expected = _ridge_reference(
        target,
        design,
        names,
        {"intercept", "y_lag1"},
        model.best_alpha,
        "mean",
        False,
    )
    np.testing.assert_allclose(model.beta_, expected, atol=1e-3)


@pytest.mark.parametrize("model_class", REGULARISED)
def test_zero_signal_cv_returns_finite_coefficients(model_class):
    """Zero residual targets and rank-deficient exempt blocks remain well defined."""
    design = np.column_stack([np.ones(30), np.ones(30), np.arange(30)])
    model = model_class(cv=3, scale=True)
    beta = model._fit_partialled(np.zeros((30, 1)), design, [0, 1], [2])
    np.testing.assert_array_equal(beta, 0)
    assert np.isfinite(getattr(model, model._penalty_attribute))


@pytest.mark.parametrize("model_class", REGULARISED)
@pytest.mark.parametrize("cv", [None, 3])
def test_collinear_penalised_column_is_zero(ar_data, model_class, cv):
    """A regressor explained by exempt lags cannot amplify projection round-off."""
    y, _ = ar_data
    X = y.shift(1).rename(columns={"y": "copy"})
    model = model_class(scale=True, cv=cv, alpha=0.1)
    model.fit(y, X=X, y_lags=1)
    names = ["intercept", *model.X.columns]
    assert model.beta_[names.index("copy"), 0] == 0
    future = pd.DataFrame(
        {"copy": [y.iloc[-1, 0] + 1]},
        index=pd.date_range(y.index[-1], periods=2, freq="ME")[1:],
    )
    forecast = model.forecast(steps=1, X=future)
    expected = model.beta_[0, 0] + model.beta_[names.index("y_lag1"), 0] * y.iloc[-1, 0]
    np.testing.assert_allclose(forecast.iloc[0, 0], expected)


@pytest.mark.parametrize("model_class", REGULARISED)
@pytest.mark.parametrize("penalise_ar", [False, True])
def test_generated_lag_name_collision_is_rejected(ar_data, model_class, penalise_ar):
    """User regressors cannot silently inherit a generated lag's exemption."""
    y, X = ar_data
    X = X.rename(columns={"x": "y_lag1"})
    with pytest.raises(ValueError, match="duplicate generated target lags"):
        model_class(penalise_ar=penalise_ar).fit(y, X=X, y_lags=1)


def test_unequal_folds_and_gap_match_reference(ar_data):
    """Unequal validation sizes retain equal fold weights and honour splitter gaps."""
    y, X = ar_data
    splits = [
        (np.arange(35), np.arange(40, 45)),
        (np.arange(55), np.arange(60, 80)),
    ]
    model = ForecastRidge(cv=splits, alphas=[0.001, 0.1, 10.0], scale=True)
    model.fit(y, X=X, y_lags=2, dummies=[y.index[40]])
    alpha, expected = _cv_reference(model, *_design(model), model.alphas)
    assert model.alpha_ == alpha
    np.testing.assert_allclose(model.beta_, expected, atol=1e-9)


def test_equal_scores_keep_first_supplied_alpha():
    """Exact ties preserve user grid order rather than sorting the candidates."""
    model = ForecastRidge(cv=3, alphas=[2.0, 0.1, 10.0])
    design = np.column_stack([np.ones(30), np.arange(30)])
    model._fit_partialled(np.zeros((30, 1)), design, [0], [1])
    assert model.alpha_ == 2.0


@pytest.mark.parametrize("model_class", REGULARISED)
@pytest.mark.parametrize(
    "case", ["no_lags", "formula_excluded", "penalised_ar", "dummies"]
)
def test_cv_accepts_unaffected_designs(ar_data, model_class, case):
    """CV also accepts designs without exempt target lags."""
    y, X = ar_data
    model = model_class(
        cv=3,
        alphas=[0.1, 1.0],
        scale=True,
        penalise_ar=case == "penalised_ar",
        formula="y ~ x" if case == "formula_excluded" else None,
    )
    model.fit(
        y,
        X=X,
        y_lags=2 if case in ("formula_excluded", "penalised_ar") else 0,
        dummies=[y.index[40]] if case == "dummies" else None,
    )
    assert getattr(model, model._penalty_attribute) in (0.1, 1.0)


@pytest.mark.parametrize("model_class", REGULARISED)
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("strategy", ["recursive", "direct"])
@pytest.mark.parametrize("cv", [None, 3, TimeSeriesSplit(n_splits=3)])
def test_ar_only_refit_skips_cv_and_clears_selected_penalty(
    ar_data, model_class, fit_intercept, strategy, cv, monkeypatch
):
    """AR-only refits use least squares at every horizon and clear stale CV state."""
    y, X = ar_data
    model = model_class(
        cv=3,
        alphas=[0.1, 1.0],
        fit_intercept=fit_intercept,
        forecast_strategy=strategy,
        steps=3,
    )
    model.fit(y, X=X, dummies=[y.index[40]])
    assert getattr(model, model._penalty_attribute) is not None
    model.scale = True
    model.cv = cv

    def unexpected(*args, **kwargs):
        pytest.fail("An entirely unpenalised fit must not call the regularised solver")

    monkeypatch.setattr(model_class, "_fit_reg", unexpected)
    model.fit(y, y_lags=2)
    assert getattr(model, model._penalty_attribute) is None
    betas = model.betas_ if strategy == "direct" else {0: model.beta_}
    for horizon, beta in betas.items():
        target, design, _ = _design(model, horizon)
        np.testing.assert_allclose(beta, np.linalg.lstsq(design, target, rcond=None)[0])


@pytest.mark.parametrize("model_class", [*REGULARISED, ForecastOLS])
@pytest.mark.parametrize("with_X", [False, True])
def test_rank_deficient_ar_block_uses_least_squares(ar_data, model_class, with_X):
    """Dependent exempt columns keep a finite minimum-norm least-squares solution."""
    y, X = ar_data
    y = y.copy()
    y["y"] = np.tile([1.0, 2.0], len(y) // 2)
    models = []
    for flag in (False, True):
        model = model_class(penalise_ar=flag)
        model.fit(y, X=X if with_X else None, y_lags=2)
        models.append(model)
    model = models[0]
    target, design, names = _design(model)
    assert np.linalg.matrix_rank(design) < design.shape[1]
    expected = _ridge_reference(
        target,
        design,
        names,
        {"intercept", "y_lag1", "y_lag2"},
        0 if model_class is ForecastOLS else 0.1,
        "mean",
        False,
    )
    np.testing.assert_allclose(model.beta_, expected, atol=1e-10)
    if model_class is ForecastOLS:
        np.testing.assert_array_equal(models[0].beta_, models[1].beta_)


@pytest.mark.parametrize("model_class", REGULARISED)
@pytest.mark.parametrize(
    "case,scale",
    [
        ("no_X", False),
        ("named_lag", False),
        ("named_lag", True),
        ("excluded_lags", False),
        ("excluded_lags", True),
    ],
)
def test_no_retained_generated_lags_ignore_flag(ar_data, model_class, case, scale):
    """Neither an arbitrary lag label nor a formula-excluded lag earns exemption."""
    y, X = ar_data
    X = None if case == "no_X" else X.rename(columns={"z": "y_lag1"})
    models = []
    for flag in (False, True):
        model = model_class(
            penalise_ar=flag,
            formula="y ~ x" if case == "excluded_lags" else None,
            scale=scale,
        )
        model.fit(y, X=X, y_lags=2 if case == "excluded_lags" else 0)
        models.append(model)
    np.testing.assert_array_equal(models[0].beta_, models[1].beta_)


@pytest.mark.parametrize("model_class", [ForecastRidge, ForecastOLS])
def test_scaled_no_intercept_dummy_fit_keeps_residual_means(ar_data, model_class):
    """The shared FWL correction also applies to existing dummy-only fits."""
    y, X = ar_data
    model = model_class(scale=True, fit_intercept=False)
    model.fit(y, X=X, dummies=[y.index[40]])
    target, design, names = _design(model)
    expected = _ridge_reference(
        target,
        design,
        names,
        set(model._dummy_cols),
        0 if model_class is ForecastOLS else 0.1,
        "mean",
        True,
    )
    np.testing.assert_allclose(model.beta_, expected, atol=1e-10)


@pytest.mark.parametrize("model_class", REGULARISED)
@pytest.mark.parametrize("strategy", ["recursive", "direct"])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_raw_fitted_values_forecasts_and_decomposition(
    ar_data, model_class, strategy, scale, fit_intercept
):
    """Reconstructed coefficients drive raw-space fits, forecasts and contributions."""
    y, X = ar_data
    model = model_class(
        alpha=0.4,
        scale=scale,
        forecast_strategy=strategy,
        steps=4,
        fit_intercept=fit_intercept,
    )
    model.fit(y, X=X, y_lags=2, dummies=[y.index[40]])
    _, design, names = _design(model)
    beta = model.betas_[0] if strategy == "direct" else model.beta_
    np.testing.assert_allclose(model.fitted_values_, (design @ beta).ravel())
    result = model.forecast(steps=4, X=X, decomp=True)
    expected = []
    recent = list(y.iloc[-2:, 0].to_numpy()[::-1])
    future = X.loc[X.index > y.index[-1]]
    for horizon in range(4):
        row = future.iloc[0 if strategy == "direct" else horizon].to_dict()
        row.update(intercept=1.0, y_lag1=recent[0], y_lag2=recent[1])
        row.update(dict.fromkeys(model._dummy_cols, 0.0))
        coefficients = model.betas_[horizon] if strategy == "direct" else model.beta_
        value = (np.array([row[name] for name in names]) @ coefficients).item()
        expected.append(value)
        if strategy == "recursive":
            recent = [value, recent[0]]
    np.testing.assert_allclose(result.iloc[:, 0], expected, atol=1e-10)
    totals = result.decomposition.groupby("forecast_horizon")["contribution"].sum()
    np.testing.assert_allclose(totals, expected, atol=1e-10)
