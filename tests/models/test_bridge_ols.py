"""Tests for ForecastBridgeOLS model."""

import numpy as np
import pandas as pd
import pytest

import forecast_realtime as rt

from ..schemas import minimal_decomposition_schema


def _quarterly_dates(n_q, start="2019-03-31"):
    return pd.date_range(start, periods=n_q, freq="QE")


def _monthly_dates(n_q, start="2019-01-31"):
    return pd.date_range(start, periods=n_q * 3, freq="ME")


def test_bridge_monthly_to_quarterly_mean_matches_manual_ols():
    """A monthly X aggregated to quarterly matches a manual OLS fit on the
    manually-averaged quarterly regressor, for both coefficients and
    forecasts."""
    rng = np.random.default_rng(42)
    n_q = 24
    n_train = 18

    q_dates = _quarterly_dates(n_q)
    m_dates = _monthly_dates(n_q)

    x_monthly = rng.standard_normal(len(m_dates))
    X_monthly = pd.DataFrame({"x1": x_monthly}, index=m_dates)

    x_quarterly_mean = X_monthly["x1"].resample("QE").mean()
    assert (x_quarterly_mean.index == q_dates).all()

    cst, b1 = 1.0, 2.0
    y = pd.DataFrame({"target": cst + b1 * x_quarterly_mean.values}, index=q_dates)
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    X_quarterly_manual = pd.DataFrame({"x1": x_quarterly_mean.values}, index=q_dates)

    manual_model = rt.models.ForecastOLS(forecast_strategy="recursive")
    manual_model.fit(y_train, X=X_quarterly_manual.iloc[:n_train])
    manual_forecast = manual_model.forecast(steps=n_q - n_train, X=X_quarterly_manual)

    bridge_model = rt.models.ForecastBridgeOLS(forecast_strategy="recursive")
    bridge_model.fit(y_train, X=X_monthly)
    bridge_forecast = bridge_model.forecast(steps=n_q - n_train, X=X_monthly)

    assert np.allclose(manual_model.beta_, bridge_model.beta_)
    assert np.allclose(manual_forecast.values, bridge_forecast.values, atol=1e-8)
    assert np.allclose(bridge_forecast.values.ravel(), y_test.values.ravel(), atol=1e-8)


def test_bridge_infers_frequency_per_column():
    """A quarterly column (no aggregation needed) mixed with a monthly column
    (aggregated) are both handled correctly with no explicit frequency
    parameter."""
    rng = np.random.default_rng(7)
    n_q = 20
    n_train = 15

    q_dates = _quarterly_dates(n_q)
    m_dates = _monthly_dates(n_q)

    x1_monthly = rng.standard_normal(len(m_dates))
    x2_quarterly = rng.standard_normal(n_q)

    union_index = pd.DatetimeIndex(sorted(set(m_dates) | set(q_dates)))
    X_full = pd.DataFrame(index=union_index)
    X_full.loc[m_dates, "x1"] = x1_monthly
    X_full.loc[q_dates, "x2"] = x2_quarterly

    x1_quarterly_mean = pd.Series(x1_monthly, index=m_dates).resample("QE").mean()

    cst, b1, b2 = 0.5, 1.5, -2.0
    y_values = cst + b1 * x1_quarterly_mean.values + b2 * x2_quarterly
    y = pd.DataFrame({"target": y_values}, index=q_dates)
    y_train = y.iloc[:n_train]

    X_manual = pd.DataFrame(
        {"x1": x1_quarterly_mean.values, "x2": x2_quarterly}, index=q_dates
    )
    manual_model = rt.models.ForecastOLS(forecast_strategy="recursive")
    manual_model.fit(y_train, X=X_manual.iloc[:n_train])

    bridge_model = rt.models.ForecastBridgeOLS(forecast_strategy="recursive")
    bridge_model.fit(y_train, X=X_full)

    assert np.allclose(manual_model.beta_, bridge_model.beta_)


def test_bridge_rejects_non_quarterly_target():
    """A monthly y raises ValueError: ForecastBridgeOLS only supports a
    quarterly target."""
    rng = np.random.default_rng(17)
    n_q = 8
    m_dates = _monthly_dates(n_q)
    y = pd.DataFrame({"target": np.arange(len(m_dates), dtype=float)}, index=m_dates)

    X_monthly = pd.DataFrame({"x1": rng.standard_normal(len(m_dates))}, index=m_dates)
    with pytest.raises(ValueError, match="quarterly target"):
        rt.models.ForecastBridgeOLS().fit(y, X=X_monthly)


def test_bridge_rejects_x_frequency_neither_monthly_nor_quarterly():
    """Weekly or daily X regressors against a quarterly y raise ValueError:
    ForecastBridgeOLS only supports monthly or quarterly regressors."""
    rng = np.random.default_rng(3)
    n_q = 8
    q_dates = _quarterly_dates(n_q)
    y = pd.DataFrame({"target": np.arange(n_q, dtype=float)}, index=q_dates)

    weekly_dates = pd.date_range("2019-01-01", periods=n_q * 13, freq="W")
    X_weekly = pd.DataFrame(
        {"x1": rng.standard_normal(len(weekly_dates))}, index=weekly_dates
    )
    with pytest.raises(ValueError):
        rt.models.ForecastBridgeOLS().fit(y, X=X_weekly)

    daily_dates = pd.date_range("2019-01-01", periods=n_q * 91, freq="D")
    X_daily = pd.DataFrame(
        {"x1": rng.standard_normal(len(daily_dates))}, index=daily_dates
    )
    with pytest.raises(ValueError):
        rt.models.ForecastBridgeOLS().fit(y, X=X_daily)


def test_bridge_partial_quarter_is_not_averaged_without_padding():
    """A trailing partial quarter (only 2 of 3 months present, no padding) is
    left as NaN rather than averaged from incomplete data."""
    rng = np.random.default_rng(11)
    n_q = 5
    q_dates = _quarterly_dates(n_q)
    m_dates = _monthly_dates(n_q)

    x_vals = rng.standard_normal(len(m_dates))
    X_monthly = pd.DataFrame({"x1": x_vals}, index=m_dates)
    # drop the final month, leaving the trailing quarter with only 2/3 months
    X_partial = X_monthly.iloc[:-1]

    y = pd.DataFrame({"target": rng.standard_normal(n_q)}, index=q_dates)

    model = rt.models.ForecastBridgeOLS()
    aggregated = model._aggregate_X(X_partial, y)

    assert aggregated["x1"].iloc[:-1].notna().all()
    assert pd.isna(aggregated["x1"].iloc[-1])


def test_bridge_padded_partial_quarter_matches_manual_padded_mean():
    """A genuine trailing partial quarter (only 2 of 3 months naturally
    present), padded with a last-value carry-forward for the missing final
    month (mirroring what ``forecast_realtime._utils.impute_X`` would produce
    for ``X_imputation="last"``), aggregates to the manual mean computed using
    that same padded value.
    """
    rng = np.random.default_rng(13)
    n_q = 5
    q_dates = _quarterly_dates(n_q)
    m_dates = _monthly_dates(n_q)

    x_vals = rng.standard_normal(len(m_dates))
    X_monthly_full = pd.DataFrame({"x1": x_vals}, index=m_dates)

    # naturally short by one month: the trailing quarter only has 2/3 months
    X_partial = X_monthly_full.iloc[:-1]

    # pad the missing final month via last-value carry-forward, as
    # forecast_realtime._utils.impute_X(method="last") would
    last_value = X_partial["x1"].iloc[-1]
    padded_index = m_dates[-1:]
    X_padded = pd.concat(
        [X_partial, pd.DataFrame({"x1": [last_value]}, index=padded_index)]
    )

    y = pd.DataFrame({"target": rng.standard_normal(n_q)}, index=q_dates)

    model = rt.models.ForecastBridgeOLS()
    aggregated = model._aggregate_X(X_padded, y)

    manual = X_padded["x1"].resample("QE").mean()
    assert np.allclose(aggregated["x1"].values, manual.values)
    # sanity: the padded value differs from the true dropped observation, so
    # this genuinely exercises the padded mean rather than the original one
    assert last_value != x_vals[-1]


def test_bridge_quarterly_regressors_match_forecast_ols():
    """All-quarterly X and y (no aggregation needed) produce identical
    results to ForecastOLS directly."""
    rng = np.random.default_rng(5)
    n_q = 20
    n_train = 15
    q_dates = _quarterly_dates(n_q)

    X = pd.DataFrame({"x1": rng.standard_normal(n_q)}, index=q_dates)
    cst, b1 = 0.3, 1.2
    y = pd.DataFrame({"target": cst + b1 * X["x1"].values}, index=q_dates)
    y_train, X_train = y.iloc[:n_train], X.iloc[:n_train]

    ols_model = rt.models.ForecastOLS(forecast_strategy="recursive")
    ols_model.fit(y_train, X=X_train)
    ols_forecast = ols_model.forecast(steps=n_q - n_train, X=X)

    bridge_model = rt.models.ForecastBridgeOLS(forecast_strategy="recursive")
    bridge_model.fit(y_train, X=X_train)
    bridge_forecast = bridge_model.forecast(steps=n_q - n_train, X=X)

    assert np.allclose(ols_model.beta_, bridge_model.beta_)
    assert np.allclose(ols_forecast.values, bridge_forecast.values, atol=1e-8)


def test_bridge_accepts_model_owned_data_transformation():
    model = rt.models.ForecastBridgeOLS(
        data_transformation={"target": "levels", "x1": "levels"}
    )

    assert model.data_transformation == {
        "target": "levels",
        "x1": "levels",
    }


def test_bridge_decomposition_schema_and_reconstructs_forecast():
    """The public ``forecast(decomp=True)`` entry point, fed raw monthly X,
    aggregates upstream of decomposition and its components reconstruct the
    plain forecast."""
    rng = np.random.default_rng(23)
    n_q = 24
    n_train = 18
    q_dates = _quarterly_dates(n_q)
    m_dates = _monthly_dates(n_q)

    x_monthly = rng.standard_normal(len(m_dates))
    X_monthly = pd.DataFrame({"x1": x_monthly}, index=m_dates)
    x_quarterly_mean = X_monthly["x1"].resample("QE").mean()

    cst, b1 = 1.0, 2.0
    y = pd.DataFrame({"target": cst + b1 * x_quarterly_mean.values}, index=q_dates)
    y_train = y.iloc[:n_train]
    steps = n_q - n_train

    bridge_model = rt.models.ForecastBridgeOLS(forecast_strategy="recursive")
    bridge_model.fit(y_train, X=X_monthly)
    forecast = bridge_model.forecast(steps=steps, X=X_monthly, decomp=True)
    decomps = forecast.decomposition

    assert decomps is not None
    minimal_decomposition_schema.validate(decomps)

    for h in range(steps):
        decomp_sum = decomps.loc[decomps["forecast_horizon"] == h, "contribution"].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-8)


def test_bridge_formula_selects_aggregated_regressors():
    """A formula selects a subset of the aggregated regressor columns."""
    rng = np.random.default_rng(29)
    n_q = 20
    n_train = 15
    q_dates = _quarterly_dates(n_q)
    m_dates = _monthly_dates(n_q)

    x1_monthly = rng.standard_normal(len(m_dates))
    x2_monthly = rng.standard_normal(len(m_dates))
    X_monthly = pd.DataFrame({"x1": x1_monthly, "x2": x2_monthly}, index=m_dates)
    x1_quarterly_mean = pd.Series(x1_monthly, index=m_dates).resample("QE").mean()

    cst, b1 = 0.4, 1.7
    y = pd.DataFrame({"target": cst + b1 * x1_quarterly_mean.values}, index=q_dates)
    y_train = y.iloc[:n_train]

    manual_model = rt.models.ForecastOLS(forecast_strategy="recursive")
    manual_X = pd.DataFrame({"x1": x1_quarterly_mean.values}, index=q_dates)
    manual_model.fit(y_train, X=manual_X.iloc[:n_train])

    bridge_model = rt.models.ForecastBridgeOLS(
        forecast_strategy="recursive", formula="target ~ x1"
    )
    bridge_model.fit(y_train, X=X_monthly)

    assert list(bridge_model.X.columns) == ["x1"]
    assert np.allclose(manual_model.beta_, bridge_model.beta_)


def test_bridge_x_lags_built_after_aggregation():
    """``X_lags`` are built from the aggregated (quarterly) column, not from
    the raw monthly one -- the reason ``fit()``/``forecast()`` are overridden
    rather than ``_fit()``/``_forecast()``."""
    rng = np.random.default_rng(31)
    n_q = 20
    n_train = 15
    q_dates = _quarterly_dates(n_q)
    m_dates = _monthly_dates(n_q)

    x_monthly = rng.standard_normal(len(m_dates))
    X_monthly = pd.DataFrame({"x1": x_monthly}, index=m_dates)
    x_quarterly_mean = X_monthly["x1"].resample("QE").mean()

    cst, b0, b1 = 0.2, 1.0, -0.5
    y_values = cst + b0 * x_quarterly_mean.values
    y_values[1:] += b1 * x_quarterly_mean.values[:-1]
    y = pd.DataFrame({"target": y_values}, index=q_dates)
    y_train = y.iloc[:n_train]

    bridge_model = rt.models.ForecastBridgeOLS(forecast_strategy="recursive")
    bridge_model.fit(y_train, X=X_monthly, X_lags=1)

    # the lag column exists and matches a lag of the *quarterly* mean series,
    # not a lag of the raw monthly values
    assert "x1_lag1" in bridge_model.X.columns
    expected_lag1 = x_quarterly_mean.shift(1).reindex(bridge_model.X.index)
    pd.testing.assert_series_equal(
        bridge_model.X["x1_lag1"], expected_lag1, check_names=False
    )

    manual_X = pd.DataFrame(
        {"x1": x_quarterly_mean.values, "x1_lag1": x_quarterly_mean.shift(1).values},
        index=q_dates,
    )
    manual_model = rt.models.ForecastOLS(forecast_strategy="recursive")
    manual_model.fit(y_train, X=manual_X.loc[y_train.index])

    assert np.allclose(manual_model.beta_, bridge_model.beta_)
