"""Tests for ForecastMIDAS wrapper."""

import pytest

pytest.importorskip("nowcast_midas")

import nowcast_midas as nm
import numpy as np
import pandas as pd

import forecast_realtime as rt

from ..schemas import minimal_decomposition_schema


def test_midas_native_fit_and_forecast():
    """Test that native nowcast_midas.MIDAS and
    ForecastMIDAS wrapper produce identical results."""
    # Generate synthetic quarterly target and monthly regressors
    np.random.seed(42)
    n_quarters = 60
    n_months = n_quarters * 3 + 12

    dates_q = pd.date_range("2010-03-31", periods=n_quarters, freq="QE")
    dates_m = pd.date_range("2009-01-31", periods=n_months, freq="ME")

    # Monthly regressor (AR(1) process)
    x_monthly = np.zeros(n_months)
    x_monthly[0] = 1.0
    for i in range(1, n_months):
        x_monthly[i] = 0.8 * x_monthly[i - 1] + np.random.randn()

    # Quarterly target correlated with monthly lags
    y_quarterly = np.zeros(n_quarters)
    for i, qd in enumerate(dates_q):
        mask = dates_m <= qd
        if mask.sum() >= 3:
            y_quarterly[i] = 0.5 * x_monthly[mask][-3:].mean() + np.random.randn() * 0.2

    # Build DataFrames in ForecastModel format (DatetimeIndex)
    y = pd.DataFrame({"gdp": y_quarterly}, index=dates_q)
    X = pd.DataFrame({"indicator": x_monthly}, index=dates_m)

    # --- Native nowcast_midas ---
    target = pd.DataFrame({"date": dates_q, "value": y_quarterly})
    regressors = pd.DataFrame({"date": dates_m, "value": x_monthly})

    native_model = nm.MIDAS(
        method="almon",
        n_lags=6,
        n_pars_weights=2,
        estimator="ols",
        horizons=[0, 1, 2, 3],
    )
    native_model.fit(target=target, regressors=regressors)
    native_forecasts_df = native_model.forecast(regressors)

    # Build native output array (steps, 1) matching wrapper format
    native_forecasts = np.full((4, 1), np.nan)
    for _, row in native_forecasts_df.iterrows():
        h = int(row["horizon"])
        if h < 4:
            native_forecasts[h, 0] = row["forecast"]

    # --- ForecastMIDAS wrapper ---
    wrapper = rt.models.ForecastMIDAS(
        method="almon", n_lags=6, estimator="ols", horizons=[0, 1, 2, 3]
    )
    wrapper.fit(y=y, X=X)
    wrapper_forecasts = wrapper.forecast(steps=4)

    np.testing.assert_array_equal(native_forecasts, wrapper_forecasts.values)


def _make_midas_data(seed=42, n_quarters=60):
    """Build a synthetic quarterly target and monthly regressor."""
    rng = np.random.default_rng(seed)
    n_months = n_quarters * 3 + 12
    dates_q = pd.date_range("2010-03-31", periods=n_quarters, freq="QE")
    dates_m = pd.date_range("2009-01-31", periods=n_months, freq="ME")

    x_monthly = np.zeros(n_months)
    x_monthly[0] = 1.0
    for i in range(1, n_months):
        x_monthly[i] = 0.8 * x_monthly[i - 1] + rng.standard_normal()

    y_quarterly = np.zeros(n_quarters)
    for i, qd in enumerate(dates_q):
        mask = dates_m <= qd
        if mask.sum() >= 3:
            y_quarterly[i] = (
                0.5 * x_monthly[mask][-3:].mean() + rng.standard_normal() * 0.2
            )

    y = pd.DataFrame({"gdp": y_quarterly}, index=dates_q)
    X = pd.DataFrame({"indicator": x_monthly}, index=dates_m)
    return y, X


def test_midas_handles_missing_regressor_internally():
    """MIDAS owns alignment and receives its missing regressor observations."""
    y, X = _make_midas_data()
    X.iloc[20, 0] = np.nan

    model = rt.models.ForecastMIDAS(
        method="almon", n_lags=6, estimator="ols", horizons=[0, 1, 2]
    )
    assert model._handles_missing_values is True

    model.fit(y=y, X=X)
    forecast = model.forecast(steps=3, X=X)

    assert model._regressors["value"].isna().sum() == 1
    assert forecast.shape == (3, 1)
    assert forecast.notna().all().all()


def test_midas_ignores_trailing_missing_regressor_dates():
    y, X = _make_midas_data()
    observed_X = X.iloc[:-2]
    padded_X = X.copy()
    padded_X.iloc[-2:, 0] = np.nan

    observed_model = rt.models.ForecastMIDAS(horizons=[0, 1, 2])
    observed_model.fit(y=y, X=observed_X)
    expected = observed_model.forecast(steps=3, X=observed_X)

    padded_model = rt.models.ForecastMIDAS(horizons=[0, 1, 2])
    padded_model.fit(y=y, X=padded_X)
    actual = padded_model.forecast(steps=3, X=padded_X)

    pd.testing.assert_frame_equal(actual, expected)
    assert padded_model._regressors["date"].max() == observed_X.index.max()


def test_midas_rejects_incomplete_backend_forecasts(monkeypatch):
    """A missing backend horizon raises before a NaT-indexed result is built."""
    y, X = _make_midas_data()
    model = rt.models.ForecastMIDAS(
        method="almon", n_lags=6, estimator="ols", horizons=[0, 1, 2]
    )
    model.fit(y=y, X=X)
    original_forecast = model.model.forecast

    def incomplete_forecast(regressors):
        return original_forecast(regressors).query("horizon < 2")

    monkeypatch.setattr(model.model, "forecast", incomplete_forecast)
    with pytest.raises(ValueError, match=r"horizon\(s\) \[2\]"):
        model.forecast(steps=3, X=X)


# ------------------------------------------------------------------ #
# Decomposition tests                                                #
# ------------------------------------------------------------------ #


def test_midas_decomposition_schema_and_components():
    """forecast(decomp=True) populates a schema-valid decomposition table."""
    y, X = _make_midas_data()
    model = rt.models.ForecastMIDAS(
        method="almon", n_lags=6, estimator="ols", horizons=[0, 1, 2]
    )
    model.fit(y=y, X=X)

    forecast = model.forecast(steps=3, X=X, decomp=True)
    decomp_df = forecast.decomposition

    assert decomp_df is not None
    minimal_decomposition_schema.validate(decomp_df)

    # The MIDAS regressor block is named after the X column.
    components = set(decomp_df["component"])
    assert "intercept" in components
    assert "indicator" in components


def test_midas_decomposition_reconstructs_forecast():
    """Contributions sum to the forecast value at every horizon."""
    y, X = _make_midas_data()
    model = rt.models.ForecastMIDAS(
        method="almon", n_lags=6, estimator="ols", horizons=[0, 1, 2]
    )
    model.fit(y=y, X=X)

    forecast = model.forecast(steps=3, X=X, decomp=True)
    decomp_df = forecast.decomposition
    minimal_decomposition_schema.validate(decomp_df)

    for h in range(3):
        decomp_sum = decomp_df.loc[
            decomp_df["forecast_horizon"] == h, "contribution"
        ].sum()
        np.testing.assert_allclose(decomp_sum, forecast.iloc[h, 0], atol=1e-9)


def test_midas_decomposition_none_without_flag():
    """No decomposition is stored when decomp=False."""
    y, X = _make_midas_data()
    model = rt.models.ForecastMIDAS(
        method="almon", n_lags=6, estimator="ols", horizons=[0]
    )
    model.fit(y=y, X=X)

    forecast = model.forecast(steps=1, X=X, decomp=False)
    assert forecast.decomposition is None


# ------------------------------------------------------------------ #
# fitted_values_ tests                                                #
# ------------------------------------------------------------------ #


def test_fitted_values_matches_backend_horizon0():
    """fitted_values_ equals the backend's horizon-0 in-sample fit."""
    y, X = _make_midas_data()
    model = rt.models.ForecastMIDAS(
        method="almon", n_lags=6, estimator="ols", horizons=[0, 1, 2]
    )
    model.fit(y=y, X=X)

    backend_fit = model.model.fits_[0].fitted_values
    common_index = model.fitted_values_.dropna().index.intersection(backend_fit.index)
    assert len(common_index) > 0
    np.testing.assert_allclose(
        model.fitted_values_.loc[common_index].to_numpy(),
        backend_fit.loc[common_index].to_numpy(),
        atol=1e-9,
    )
