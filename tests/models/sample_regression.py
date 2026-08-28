"""Ground-truth synthetic regression data generation for model testing.

Generates bivariate regression data with known true coefficients:
    y[t] = b1*X1[t] + b2*X2[t] + eps[t]

Supports both recursive and direct forecasting scenarios.
"""

import numpy as np
import pandas as pd


def sample_regression_data(
    n_train: int = 500,
    n_test: int = 50,
    cst: float = 1.0,
    b1: float = 2.0,
    b2: float = -1.0,
    b_ar: float = 0.0,
    noise_std: float = 0.001,
    forecast_type: str = "recursive",
    horizon: int | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Generate synthetic regression data with known true model.

    Parameters
    ----------
    n_train : int
        Number of training observations
    n_test : int
        Number of test observations
    cst : float
        Constant term in the regression model
    b1 : float
        True coefficient for X1: y = cst + b1*X1 + b2*X2 + noise.
    b2 : float
        True coefficient for X2: y = cst + b1*X1 + b2*X2 + noise.
    b_ar : float
        Coefficient for autoregressive terms of y
    noise_std : float
        Standard deviation of Gaussian noise
    forecast_type : str
        "recursive" or "direct" (for per-horizon models)
    horizon : int | None
        For direct forecasting, a single horizon to model.
        Default is None (becomes 1 in _sample_direct).
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]
        Training and test data with ground truth coefficients
    """
    rng = np.random.default_rng(random_seed)

    if forecast_type == "recursive":
        return _sample_recursive(n_train, n_test, cst, b1, b2, b_ar, noise_std, rng)
    elif forecast_type == "direct":
        return _sample_direct(n_train, n_test, cst, b1, b2, noise_std, horizon, rng)
    else:
        raise ValueError(f"Unknown forecast_type: {forecast_type}")


def sample_ar2_data(
    n_train: int = 200,
    n_test: int = 3,
    cst: float = 0.5,
    b1: float = 1.0,
    a1: float = 0.6,
    a2: float = -0.3,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Generate noiseless AR(2)-with-exogenous data.

        y[t] = cst + b1*x1[t] + a1*y[t-1] + a2*y[t-2]

    Used to check recursive forecasting with ``y_lags=2``: the second lag must
    carry the value from two periods ago, not a copy of the first lag.

    Parameters
    ----------
    n_train : int
        Number of training observations.
    n_test : int
        Number of test observations.
    cst : float
        Constant term.
    b1 : float
        Exogenous coefficient.
    a1 : float
        First autoregressive coefficient.
    a2 : float
        Second autoregressive coefficient.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]
        Training and test data with ground truth coefficients.
    """
    rng = np.random.default_rng(random_seed)
    n_total = n_train + n_test
    dates = pd.date_range("2020-01-01", periods=n_total, freq="D")

    x1 = rng.standard_normal(n_total)
    y = np.zeros(n_total)
    y[0] = cst + b1 * x1[0]
    y[1] = cst + b1 * x1[1] + a1 * y[0]
    for t in range(2, n_total):
        y[t] = cst + b1 * x1[t] + a1 * y[t - 1] + a2 * y[t - 2]

    X_all = pd.DataFrame({"x1": x1}, index=dates)
    y_all = pd.DataFrame({"target": y}, index=dates)

    true_coef = {"cst": cst, "b1": b1, "a1": a1, "a2": a2}
    return (
        y_all.iloc[:n_train],
        X_all.iloc[:n_train],
        y_all.iloc[n_train:],
        X_all.iloc[n_train:],
        true_coef,
    )


def sample_ar4_data(
    n_train: int = 200,
    n_test: int = 5,
    cst: float = 0.5,
    b1: float = 1.0,
    a1: float = 0.4,
    a2: float = -0.2,
    a3: float = 0.15,
    a4: float = -0.1,
    random_seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Generate noiseless AR(4)-with-exogenous data.

        y[t] = cst + b1*x1[t] + a1*y[t-1] + a2*y[t-2] + a3*y[t-3] + a4*y[t-4]

    Used to check recursive forecasting with ``y_lags=4``: all four lags must
    be tracked independently so the multi-step path is exact.

    Parameters
    ----------
    n_train : int
        Number of training observations.
    n_test : int
        Number of test observations.
    cst : float
        Constant term.
    b1 : float
        Exogenous coefficient.
    a1 : float
        First autoregressive coefficient.
    a2 : float
        Second autoregressive coefficient.
    a3 : float
        Third autoregressive coefficient.
    a4 : float
        Fourth autoregressive coefficient.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]
        Training and test data with ground truth coefficients.
    """
    rng = np.random.default_rng(random_seed)
    n_total = n_train + n_test
    dates = pd.date_range("2020-01-01", periods=n_total, freq="D")

    x1 = rng.standard_normal(n_total)
    y = np.zeros(n_total)
    for t in range(4):
        y[t] = cst + b1 * x1[t]
    for t in range(4, n_total):
        y[t] = (
            cst
            + b1 * x1[t]
            + a1 * y[t - 1]
            + a2 * y[t - 2]
            + a3 * y[t - 3]
            + a4 * y[t - 4]
        )

    X_all = pd.DataFrame({"x1": x1}, index=dates)
    y_all = pd.DataFrame({"target": y}, index=dates)

    true_coef = {"cst": cst, "b1": b1, "a1": a1, "a2": a2, "a3": a3, "a4": a4}
    return (
        y_all.iloc[:n_train],
        X_all.iloc[:n_train],
        y_all.iloc[n_train:],
        X_all.iloc[n_train:],
        true_coef,
    )


def _sample_recursive(n_train, n_test, cst, b1, b2, b_ar, noise_std, rng):
    """Sample recursive (standard) regression data.

    X[t] ~ N(0, 1) iid
    y[t] = b1*X1[t] + b2*X2[t] + eps[t]
    """
    if b_ar > 0:
        n_ar = 1
    else:
        n_ar = 0

    n_train -= n_ar

    n_total = n_train + n_test

    # Generate exogenous features
    X_all = rng.standard_normal((n_total, 2 + n_ar))
    dates = pd.date_range("2020-01-01", periods=n_total, freq="D")

    # Generate target
    eps = rng.normal(0, noise_std, n_total)

    y_all = np.zeros(n_total)
    for t in range(n_total):
        X1_t = X_all[t, 0]
        X2_t = X_all[t, 1]
        eps_t = eps[t]
        if t < 1 or b_ar == 0:
            y_all[t] = cst + b1 * X1_t + b2 * X2_t + eps_t
        else:
            y_ar = y_all[t - 1]
            y_all[t] = cst + b1 * X1_t + b2 * X2_t + b_ar * y_ar + eps_t
            X_all[t, 2] = y_ar

    if b_ar > 0:
        # Remove the first observation used for AR lag
        dates = dates[1:]
        X_all = X_all[1:, :]
        y_all = y_all[1:]

    X_names = ["x1", "x2"]
    if b_ar > 0:
        X_names.append("_y_lag1")

    # Training data
    X_train = pd.DataFrame(X_all[:n_train], index=dates[:n_train], columns=X_names)
    y_train = pd.DataFrame(y_all[:n_train], index=dates[:n_train], columns=["target"])

    # Test data
    X_test = pd.DataFrame(X_all[n_train:], index=dates[n_train:], columns=X_names)
    y_test = pd.DataFrame(y_all[n_train:], index=dates[n_train:], columns=["target"])

    true_coef = {"b1": b1, "b2": b2, "b_ar": b_ar, "cst": cst, "noise_std": noise_std}
    return y_train, X_train, y_test, X_test, true_coef


def _sample_direct(n_train, n_test, cst, b1, b2, noise_std, horizon, rng):
    """Sample direct (per-horizon) regression data for a single horizon.

    For a given horizon h:
        DGP: y[t] = cst + b1*X1[t-h] + b2*X2[t-h] + eps[t]

    Training: fit y[t] ~ X[t-h]
    Forecasting: predict y[t+h] using X[t] (which equals X[(t+h)-h])

    Returns training data with lagged features aligned to targets,
    and test data for forecasting from current X.
    """
    if horizon is None:
        horizon = 1

    max_h = horizon + 1
    n_total = n_train + max_h + n_test

    # Generate exogenous features
    X_all = rng.standard_normal((n_total, 2))
    dates = pd.date_range("2020-01-01", periods=n_total, freq="D")

    # Generate noise
    eps = rng.normal(0, noise_std, n_total)

    # Generate y[t] = cst + b1*X1[t-h] + b2*X2[t-h] + eps[t]
    # Start from index max_h (need max_h periods of lagged X history)
    y_all = np.zeros(n_total)
    X_lagged = np.zeros((n_total, 2))
    for t in range(max_h, n_total):
        h = horizon
        X_lagged[t] = X_all[t - h]
        y_all[t] = cst + b1 * X_lagged[t, 0] + b2 * X_lagged[t, 1] + eps[t]

    # Training: Create pairs (X[t], y[t])
    t_indices = np.arange(max_h, n_train + max_h)
    X_train = X_all[t_indices]  # X[t]
    y_targets = y_all[t_indices]  # y[t]

    X_train = pd.DataFrame(
        X_train, index=pd.DatetimeIndex(dates[t_indices]), columns=["x1", "x2"]
    )
    y_train = pd.DataFrame(
        y_targets, index=pd.DatetimeIndex(dates[t_indices]), columns=["target"]
    )

    # Test: Use X at position n_train + max_h (final feature row)
    test_idx = n_train + max_h
    X_test = pd.DataFrame(
        [X_all[test_idx]], index=[dates[test_idx]], columns=["x1", "x2"]
    )

    # y_test: ground truth y[t] for t = n_train + max_h + 1, ..., n_train + max_h + n_test
    # When forecasting from X[test_idx], we predict y[test_idx + h]
    y_test_values = []
    y_test_dates_list = []
    for i in range(0, n_test + 1):
        t = test_idx + i
        if t < n_total:
            y_val = cst + b1 * X_all[t - horizon, 0] + b2 * X_all[t - horizon, 1] + eps[t]
            y_test_values.append(y_val)
            y_test_dates_list.append(dates[t])

    y_test = pd.DataFrame(
        y_test_values, index=pd.DatetimeIndex(y_test_dates_list), columns=["target"]
    )

    true_coef = {"b1": b1, "b2": b2, "cst": cst, "noise_std": noise_std}
    return y_train, X_train, y_test, X_test, true_coef
