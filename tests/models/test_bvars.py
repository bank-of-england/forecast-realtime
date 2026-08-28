# Load ForecastData with FER dataset
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bvar")

import bvar as bv
import forecast_evaluation as fe

forecast_data = fe.ForecastData(load_fer=True)


def test_bvar_declares_missing_values_unsupported():
    """RealTimeModel must complete-case BVAR estimation data."""
    import forecast_realtime as rt

    assert rt.models.ForecastBVAR._handles_missing_values is False


def test_bvar_rejects_density_forecasts():
    """Density output remains disabled until it has a dedicated contract."""
    import forecast_realtime as rt

    with pytest.raises(ValueError, match="mean.*median.*density.*not supported"):
        rt.models.ForecastBVAR(forecasts_type="density")


@pytest.mark.parametrize("forecasts_type", ["mean", "median"])
def test_bvar_accepts_point_forecast_types(forecasts_type):
    """Mean and median are the supported BVAR forecast summaries."""
    import forecast_realtime as rt

    model = rt.models.ForecastBVAR(forecasts_type=forecasts_type)

    assert model.forecasts_type == forecasts_type


def test_realtime_bvar_receives_no_missing_estimation_values(monkeypatch):
    """Leading and internal gaps are removed before BVAR estimation."""
    import forecast_realtime as rt

    captured = {}

    def capture_optimisation(self, data, **kwargs):
        captured["optimisation"] = data.copy()

    def capture_sampling(self, data, **kwargs):
        captured["sampling"] = data.copy()

    def capture_fitted_values(self):
        data = captured["sampling"]
        self.fitted_values = np.zeros((1, len(data) - self.n_lags, len(data.columns)))

    monkeypatch.setattr(bv.BVAR, "optimise_hyperparameters", capture_optimisation)
    monkeypatch.setattr(bv.BVAR, "sample", capture_sampling)
    monkeypatch.setattr(bv.BVAR, "compute_fitted_values", capture_fitted_values)

    class BVARWithoutNativeForecast(rt.models.ForecastBVAR):
        def _forecast(self, steps=1, X=None, y=None, **kwargs):
            return np.zeros((steps, len(self.y.columns)))

    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    frames = []
    for variable, missing_dates in {
        "target_a": [],
        "target_b": [dates[0], dates[5]],
    }.items():
        frame = pd.DataFrame(
            {
                "date": dates,
                "variable": variable,
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": np.arange(12.0),
                "metric": "levels",
            }
        )
        frames.append(frame[~frame["date"].isin(missing_dates)])

    data = fe.ForecastData(outturns_data=pd.concat(frames, ignore_index=True))
    realtime_model = rt.RealTimeModel(
        data=data,
        models=BVARWithoutNativeForecast(progressbar=False, mode_only=True),
    )
    realtime_model.forecast(
        y_variables=["target_a", "target_b"],
        data_transformation={"target_a": "levels", "target_b": "levels"},
        steps=1,
        first_forecast_horizon=1,
        first_vintage=str(dates[-1].date()),
        last_vintage=str(dates[-1].date()),
    )

    expected_index = pd.DatetimeIndex(
        dates.delete([0, 5]).astype("datetime64[ns]"), name="date"
    )
    for estimation_data in captured.values():
        assert not estimation_data.isna().any().any()
        pd.testing.assert_index_equal(estimation_data.index, expected_index)


def test_bvar_native_unconditional():
    """Test that native bvar and ForecastBVAR wrapper produce identical results."""
    import forecast_realtime as rt

    variables = ["cpisa", "gdpkp"]
    outturns = forecast_data.outturns.copy()
    outturns = outturns[outturns["variable"].isin(variables)]

    vintage = pd.Timestamp("2015-03-31")
    y_vintage = outturns[outturns["vintage_date"] <= vintage].copy()
    y_vintage = y_vintage.sort_values("vintage_date", ascending=False).drop_duplicates(
        subset=["date", "variable"], keep="first"
    )
    y_vintage = y_vintage.pivot(index="date", columns="variable", values="value")
    y_vintage = y_vintage[y_vintage.index < vintage].dropna()

    H = 4

    # --- Native bvar ---
    prior = bv.NaturalConjugate(minnesota=True, soc=True, sur=True)
    native_model = bv.BVAR(
        n_lags=5, model=prior, stationary=True, optimisation_method="ml"
    )
    native_model.optimise_hyperparameters(y_vintage, nb_restart=5, random_state=0)
    native_model.sample(
        data=y_vintage,
        N_draws=1000,
        point_only=True,
        progressbar=False,
        random_state=0,
    )
    native_model.forecast(
        H=H,
        point_only=True,
        N_draws=5000,
        N_burn=2500,
        progressbar=False,
        random_state=0,
    )
    native_forecasts = np.mean(native_model.forecast_unconditional, axis=0)[-H:]

    # --- ForecastBVAR wrapper ---
    wrapper = rt.models.ForecastBVAR(
        stationary=True,
        n_lags=5,
        mode_only=True,
        optim_random_state=0,
        sampling_random_state=0,
        forecast_random_state=0,
    )
    wrapper.fit(y=y_vintage)
    wrapper_forecasts = wrapper.forecast(steps=H)

    np.testing.assert_array_equal(native_forecasts, wrapper_forecasts.values)


def test_bvar_native_conditional():
    """Test that native bvar and
    ForecastBVAR wrapper produce identical conditional results."""
    import forecast_realtime as rt

    variables = ["cpisa", "gdpkp"]
    outturns = forecast_data.outturns.copy()
    outturns = outturns[outturns["variable"].isin(variables)]
    forecasts = forecast_data.forecasts.copy()
    forecasts = forecasts[
        (forecasts["variable"].isin(variables)) & (forecasts["source"] == "mpr")
    ]

    vintage = pd.Timestamp("2015-03-31")

    # Training data
    y_vintage = outturns[outturns["vintage_date"] <= vintage].copy()
    y_vintage = y_vintage.sort_values("vintage_date", ascending=False).drop_duplicates(
        subset=["date", "variable"], keep="first"
    )
    y_vintage = y_vintage.pivot(index="date", columns="variable", values="value")
    y_vintage = y_vintage[y_vintage.index < vintage].dropna()

    # Conditioning forecasts
    fcst_vintage = forecasts[forecasts["vintage_date"] <= vintage].copy()
    fcst_vintage = fcst_vintage.sort_values(
        "vintage_date", ascending=False
    ).drop_duplicates(subset=["date", "variable"], keep="first")
    fcst_pivot = fcst_vintage.pivot(index="date", columns="variable", values="value")
    fcst_pivot = fcst_pivot[fcst_pivot.index >= vintage]

    # Build conditioning matrix: cpisa conditioned for 2 steps, gdpkp for 1
    H = 4
    y_columns = list(y_vintage.columns)
    constraint_mean = np.full((H, len(y_columns)), np.nan)
    conditioning = {"cpisa": 1, "gdpkp": 0}
    for var, steps_ahead in conditioning.items():
        adjusted = steps_ahead + 1
        col_idx = y_columns.index(var)
        if var in fcst_pivot.columns:
            constraint_mean[:adjusted, col_idx] = fcst_pivot[var].values[:adjusted]

    # --- Native bvar ---
    prior = bv.NaturalConjugate(minnesota=True, soc=True, sur=True)
    native_model = bv.BVAR(
        n_lags=5, model=prior, stationary=True, optimisation_method="ml"
    )
    native_model.optimise_hyperparameters(y_vintage, nb_restart=5, random_state=0)
    native_model.sample(
        data=y_vintage,
        N_draws=1000,
        point_only=True,
        progressbar=False,
        random_state=0,
    )
    native_model.forecast(
        H=H,
        constraint_mean=constraint_mean,
        point_only=True,
        method="andersson_et_al",
        N_draws=5000,
        N_burn=2500,
        progressbar=False,
        random_state=0,
    )
    native_forecasts = np.mean(native_model.forecast_conditional, axis=0)[-H:]

    # --- ForecastBVAR wrapper ---
    wrapper = rt.models.ForecastBVAR(
        stationary=True,
        n_lags=5,
        mode_only=True,
        optim_random_state=0,
        sampling_random_state=0,
        forecast_random_state=0,
    )
    wrapper.fit(y=y_vintage)

    # make constraint_mean a DataFrame with proper dates
    constraint_mean_df = pd.DataFrame(
        constraint_mean,
        columns=y_vintage.columns,
        index=pd.date_range(start=vintage, periods=H, freq="QE"),
    )
    wrapper_forecasts = wrapper.forecast(steps=H, y=constraint_mean_df)

    np.testing.assert_array_equal(native_forecasts, wrapper_forecasts.values)


def _build_multivariate_data(n_lags):
    """Small, fast synthetic multivariate dataset for fitted-values tests.

    T=100 is used (rather than a smaller sample) to keep the closed-form
    Natural Conjugate posterior well-conditioned; very short samples can
    otherwise produce numerically unstable (NaN/overflow) fitted values.
    """
    y, _, _, _ = bv.simulate_var(T=100, n=2, n_lags=n_lags, levels=False, seed=0)
    y.columns = ["var_a", "var_b"]
    return y


def _fit_fast_wrapper(y, n_lags):
    import forecast_realtime as rt

    wrapper = rt.models.ForecastBVAR(
        stationary=True,
        n_lags=n_lags,
        nb_restart=0,
        n_samples=50,
        mode_only=True,
        progressbar=False,
        optim_random_state=0,
        sampling_random_state=0,
    )
    wrapper.fit(y=y)
    return wrapper


def test_fitted_values_recovers_backend():
    """fitted_values_ recovers the native bvar posterior-mean in-sample fit."""
    n_lags = 1
    y = _build_multivariate_data(n_lags)

    wrapper = _fit_fast_wrapper(y, n_lags)

    expected = pd.DataFrame(
        wrapper.bvar.fitted_values.mean(axis=0),
        index=y.index[n_lags:],
        columns=y.columns,
    )

    fitted = wrapper.fitted_values_.dropna()
    np.testing.assert_allclose(
        fitted.to_numpy(), expected.loc[fitted.index].to_numpy(), atol=1e-9
    )
