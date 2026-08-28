"""Tests for ForecastMIDASCombo wrapper."""

import pytest

pytest.importorskip("nowcast_midas")

import nowcast_midas as nm
import numpy as np
import pandas as pd
from nowcast_midas.specs import ComboSpec, MidasSpec, OLSSpec
from nowcast_midas.utils import sample_combo_data

import forecast_realtime as rt


def _long_regressors_to_wide(regressors: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format regressors DataFrame returned by
    :func:`sample_combo_data` into the wide DatetimeIndex DataFrame the
    forecast_realtime framework feeds models as ``X``."""
    return regressors.pivot(index="date", columns="variable", values="value")


def test_midas_combo_native_fit_and_forecast():
    """Native nowcast_midas.MidasCombo and ForecastMIDASCombo wrapper produce
    identical forecasts on a well-specified DGP."""
    monthly_vars = ["PMI", "IP"]
    quarterly_vars = ["UNEMP"]
    horizons = 3

    # Canonical sc-midas DGP utility (long-format target & regressors)
    target, regressors, _ = sample_combo_data(
        n_quarters=80,
        n_lags=6,
        monthly_vars=monthly_vars,
        quarterly_vars=quarterly_vars,
        noise=0.2,
        seed=11,
        method="almon",
        outlier_date=None,
    )

    midas_specs = [
        MidasSpec(variable=v, method="almon", n_lags=6, estimator="ols")
        for v in monthly_vars
    ]
    ols_specs = [OLSSpec(variable=v, n_lags=1) for v in quarterly_vars]
    combo_specs = ComboSpec(
        name="combo",
        sources=midas_specs + ols_specs,
        method="average",
    )

    # --- Native nowcast_midas ---------------------------------------------
    native_model = nm.MidasCombo(
        combo_specs=combo_specs,
        horizons=horizons,
    )
    native_model.fit(target=target, regressors=regressors)
    native_df = native_model.forecast()

    # Extract "combo" spec forecasts sorted by horizon
    combo_df = native_df[native_df["spec"] == "combo"].sort_values("horizon")
    native_forecasts = np.full((horizons, 1), np.nan)
    native_forecasts[: len(combo_df), 0] = combo_df["value"].values

    # --- ForecastMIDASCombo wrapper --------------------------------------
    target_var = target["variable"].iloc[0]
    y = pd.DataFrame(
        {target_var: target["value"].to_numpy()},
        index=pd.DatetimeIndex(target["date"].to_numpy()),
    )
    X = _long_regressors_to_wide(regressors)

    wrapper = rt.models.ForecastMIDASCombo(
        combo_specs=combo_specs,
        horizons=horizons,
        regressor_frequencies={
            **{v: "ME" for v in monthly_vars},
            **{v: "QE" for v in quarterly_vars},
        },
    )
    wrapper.fit(y=y, X=X)
    wrapper_forecasts = wrapper.forecast(steps=horizons)

    np.testing.assert_array_equal(native_forecasts, wrapper_forecasts.values)


def test_midas_combo_uses_shared_realtime_input_frequencies(monkeypatch):
    """Frequencies resolved by the real-time API replace local inference."""
    wrapper, _ = _make_midas_combo_wrapper()

    def fail_inference(series):
        raise AssertionError("MIDASCombo must use the shared frequency map")

    monkeypatch.setattr(wrapper, "_infer_frequency", fail_inference, raising=False)
    wrapper.regressor_frequencies = None
    target = wrapper._raw_y_history.columns[0]
    input_frequencies = {target: "Q", "PMI": "M", "IP": "M", "UNEMP": "Q"}
    wrapper.fit(
        y=wrapper._raw_y_history,
        X=wrapper._raw_X_history,
        input_frequencies=input_frequencies,
    )

    long_regressors = wrapper._X_to_long(wrapper._raw_X_history)

    assert dict(long_regressors.groupby("variable")["frequency"].first()) == {
        "PMI": "ME",
        "IP": "ME",
        "UNEMP": "QE",
    }


def test_midas_combo_rejects_incomplete_backend_forecasts(monkeypatch):
    """A missing root-combination horizon raises a useful contract error."""
    wrapper, _ = _make_midas_combo_wrapper()
    original_forecast = wrapper.model.forecast

    def incomplete_forecast():
        return original_forecast().query("spec != 'combo' or horizon < 2")

    monkeypatch.setattr(wrapper.model, "forecast", incomplete_forecast)
    with pytest.raises(ValueError, match=r"horizon\(s\) \[2\]"):
        wrapper.forecast(steps=3)


# ------------------------------------------------------------------ #
# fitted_values_ tests                                                #
# ------------------------------------------------------------------ #


def _make_midas_combo_wrapper():
    """Build a fitted ForecastMIDASCombo wrapper and its training y."""
    monthly_vars = ["PMI", "IP"]
    quarterly_vars = ["UNEMP"]
    horizons = 3

    target, regressors, _ = sample_combo_data(
        n_quarters=80,
        n_lags=6,
        monthly_vars=monthly_vars,
        quarterly_vars=quarterly_vars,
        noise=0.2,
        seed=11,
        method="almon",
        outlier_date=None,
    )

    midas_specs = [
        MidasSpec(variable=v, method="almon", n_lags=6, estimator="ols")
        for v in monthly_vars
    ]
    ols_specs = [OLSSpec(variable=v, n_lags=1) for v in quarterly_vars]
    combo_specs = ComboSpec(
        name="combo",
        sources=midas_specs + ols_specs,
        method="average",
    )

    target_var = target["variable"].iloc[0]
    y = pd.DataFrame(
        {target_var: target["value"].to_numpy()},
        index=pd.DatetimeIndex(target["date"].to_numpy()),
    )
    X = _long_regressors_to_wide(regressors)

    wrapper = rt.models.ForecastMIDASCombo(
        combo_specs=combo_specs,
        horizons=horizons,
        regressor_frequencies={
            **{v: "ME" for v in monthly_vars},
            **{v: "QE" for v in quarterly_vars},
        },
    )
    wrapper.fit(y=y, X=X)
    return wrapper, y


def test_fitted_values_matches_backend_horizon0():
    """fitted_values_ equals the backend's horizon-0 root combination fit."""
    wrapper, _ = _make_midas_combo_wrapper()

    root_name = wrapper.combo_specs.name
    backend_fit = wrapper.model.fitted_[root_name][0]
    common_index = wrapper.fitted_values_.dropna().index.intersection(
        backend_fit.dropna().index
    )
    assert len(common_index) > 0
    np.testing.assert_allclose(
        wrapper.fitted_values_.loc[common_index].to_numpy(),
        backend_fit.loc[common_index].to_numpy(),
        atol=1e-9,
    )
