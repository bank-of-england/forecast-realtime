"""Tests for the ForecastMultiMIDAS wrapper (decomposition focus)."""

import pytest

pytest.importorskip("nowcast_midas")

import numpy as np
import pandas as pd

import forecast_realtime as rt

from ..schemas import minimal_decomposition_schema


def _make_multimidas_data(seed=42, n_quarters=60):
    """Build a synthetic quarterly target and two monthly regressors."""
    rng = np.random.default_rng(seed)
    n_months = n_quarters * 3 + 12
    dates_q = pd.date_range("2010-03-31", periods=n_quarters, freq="QE")
    dates_m = pd.date_range("2009-01-31", periods=n_months, freq="ME")

    def _ar1(phi):
        x = np.zeros(n_months)
        x[0] = 1.0
        for i in range(1, n_months):
            x[i] = phi * x[i - 1] + rng.standard_normal()
        return x

    pmi = _ar1(0.8)
    ip = _ar1(0.6)

    y_quarterly = np.zeros(n_quarters)
    for i, qd in enumerate(dates_q):
        mask = dates_m <= qd
        if mask.sum() >= 3:
            y_quarterly[i] = (
                0.5 * pmi[mask][-3:].mean()
                + 0.3 * ip[mask][-3:].mean()
                + rng.standard_normal() * 0.2
            )

    y = pd.DataFrame({"gdp": y_quarterly}, index=dates_q)
    X = pd.DataFrame({"PMI": pmi, "IP": ip}, index=dates_m)
    return y, X


def test_multimidas_decomposition_schema_and_components():
    """forecast(decomp=True) populates a schema-valid decomposition table."""
    y, X = _make_multimidas_data()
    model = rt.models.ForecastMultiMIDAS(
        variables=["PMI", "IP"], method="almon", n_lags=6, horizons=[0, 1, 2]
    )
    model.fit(y=y, X=X)

    forecast = model.forecast(steps=3, X=X, decomp=True)
    decomp_df = forecast.decomposition

    assert decomp_df is not None
    minimal_decomposition_schema.validate(decomp_df)

    components = set(decomp_df["component"])
    assert {"intercept", "PMI", "IP"}.issubset(components)


def test_multimidas_rejects_incomplete_backend_forecasts(monkeypatch):
    """A missing backend horizon raises before a NaT-indexed result is built."""
    y, X = _make_multimidas_data()
    model = rt.models.ForecastMultiMIDAS(
        variables=["PMI", "IP"], method="almon", n_lags=6, horizons=[0, 1, 2]
    )
    model.fit(y=y, X=X)
    original_forecast = model.model.forecast

    def incomplete_forecast(regressors):
        return original_forecast(regressors).query("horizon < 2")

    monkeypatch.setattr(model.model, "forecast", incomplete_forecast)
    with pytest.raises(ValueError, match=r"horizon\(s\) \[2\]"):
        model.forecast(steps=3, X=X)


def test_multimidas_decomposition_reconstructs_forecast():
    """Contributions sum to the forecast value at every horizon."""
    y, X = _make_multimidas_data()
    model = rt.models.ForecastMultiMIDAS(
        variables=["PMI", "IP"], method="almon", n_lags=6, horizons=[0, 1, 2]
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


def test_multimidas_decomposition_none_without_flag():
    """No decomposition is stored when decomp=False."""
    y, X = _make_multimidas_data()
    model = rt.models.ForecastMultiMIDAS(
        variables=["PMI", "IP"], method="almon", n_lags=6, horizons=[0]
    )
    model.fit(y=y, X=X)

    forecast = model.forecast(steps=1, X=X, decomp=False)
    assert forecast.decomposition is None


# ------------------------------------------------------------------ #
# fitted_values_ tests                                                #
# ------------------------------------------------------------------ #


def test_fitted_values_matches_backend_horizon0():
    """fitted_values_ equals the backend's horizon-0 in-sample fit."""
    y, X = _make_multimidas_data()
    model = rt.models.ForecastMultiMIDAS(
        variables=["PMI", "IP"], method="almon", n_lags=6, horizons=[0, 1, 2]
    )
    model.fit(y=y, X=X)

    fit0 = model.model.fits_[0]
    backend_fit = pd.Series(
        np.asarray(fit0.fitted_values).ravel(), index=pd.DatetimeIndex(fit0.dates)
    )
    common_index = model.fitted_values_.dropna().index.intersection(backend_fit.index)
    assert len(common_index) > 0
    np.testing.assert_allclose(
        model.fitted_values_.loc[common_index].to_numpy(),
        backend_fit.loc[common_index].to_numpy(),
        atol=1e-9,
    )
