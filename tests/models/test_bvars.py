"""Tests for ForecastBVAR using deterministic synthetic data."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("bvar")

import bvar as bv
import forecast_evaluation as fe

import forecast_realtime as rt
from forecast_realtime.forecast_tree import _point_forecast_to_wide


class _SyntheticBVAR:
    """Small backend that exposes the wrapper's forecast contract."""

    def __init__(self, n_lags=1, **kwargs):
        self.n_lags = n_lags
        self.is_fitted = False
        self.point_only = False
        self.forecast_calls = []
        self.forecast_unconditional = None
        self.forecast_conditional = None
        self.df_forecasts_unconditional = None
        self.df_forecasts_conditional = None

    def optimise_hyperparameters(self, data, **kwargs):
        return None

    def sample(self, data, N_draws, point_only, **kwargs):
        self.data = data.copy()
        self.N_draws = N_draws
        self.point_only = point_only
        self.is_fitted = True
        self.fitted_values = np.zeros((2, len(data) - self.n_lags, len(data.columns)))

    def compute_fitted_values(self):
        return None

    def _forecast_array(self, history, H, offset):
        arrays = []
        for draw in range(2):
            values = np.zeros((len(history) + H, history.shape[1]))
            values[: len(history)] = history
            values[len(history) :] = (
                offset
                + draw
                + np.arange(H * history.shape[1]).reshape(H, history.shape[1])
            )
            arrays.append(values)
        return np.asarray(arrays)

    def _formatted_table(self, history, H, quantiles, offset):
        history_periods = pd.PeriodIndex(history.index, freq="Q")
        future_periods = pd.period_range(
            start=history_periods[-1] + 1,
            periods=H,
            freq=history_periods.freq,
        )
        periods = history_periods.append(future_periods)
        rows = []
        for quantile in quantiles:
            for period_index, period in enumerate(periods):
                for variable_index, variable in enumerate(history.columns):
                    rows.append(
                        {
                            "date": period,
                            "variable": variable,
                            "quantile": quantile,
                            "value": offset
                            + period_index
                            + variable_index * 10
                            + quantile,
                        }
                    )
        return pd.DataFrame(rows)

    def forecast(
        self,
        H,
        constraint_mean=None,
        method=None,
        N_draws=5000,
        N_burn=None,
        point_only=False,
        format=False,
        quantiles=None,
        base_value=None,
        progressbar=False,
        transformations=None,
        random_state=None,
    ):
        history = self.data.iloc[self.n_lags :]
        quantiles = [0.16, 0.5, 0.84] if quantiles is None else list(quantiles)
        conditional = constraint_mean is not None
        self.forecast_unconditional = self._forecast_array(history.to_numpy(), H, 0)
        self.forecast_conditional = (
            self._forecast_array(history.to_numpy(), H, 100) if conditional else None
        )
        self.df_forecasts_unconditional = self._formatted_table(history, H, quantiles, 0)
        self.df_forecasts_conditional = (
            self._formatted_table(history, H, quantiles, 100) if conditional else None
        )
        self.forecast_calls.append(
            {
                "H": H,
                "constraint_mean": (
                    None
                    if constraint_mean is None
                    else np.asarray(constraint_mean).copy()
                ),
                "N_draws": N_draws,
                "N_burn": N_burn,
                "point_only": point_only,
                "format": format,
                "quantiles": list(quantiles) if format else None,
                "transformations": transformations,
                "unconditional_table": self.df_forecasts_unconditional.copy(),
                "conditional_table": (
                    None
                    if self.df_forecasts_conditional is None
                    else self.df_forecasts_conditional.copy()
                ),
            }
        )


def _synthetic_history():
    dates = pd.date_range("2010-03-31", periods=12, freq="QE")
    return pd.DataFrame(
        {
            "var_a": np.linspace(100.0, 111.0, len(dates)),
            "var_b": np.linspace(80.0, 86.0, len(dates)),
        },
        index=dates,
    )


def _fit_synthetic_bvar(monkeypatch, **kwargs):
    monkeypatch.setattr(bv, "BVAR", _SyntheticBVAR)
    options = {
        "n_lags": 1,
        "nb_restart": 0,
        "n_samples": 12,
        "N_draws": 12,
        "progressbar": False,
    }
    options.update(kwargs)
    model = rt.models.ForecastBVAR(**options)
    model.fit(_synthetic_history())
    return model


def _density_dates(model, steps):
    return model._forecast_dates(
        model._fitted_model_configuration.forecast_origin,
        steps,
    )


def _native_density_tail(table, dates, steps):
    periods = table["date"].drop_duplicates().sort_values().iloc[-steps:]
    result = table.loc[table["date"].isin(periods)].copy()
    result["date"] = result["date"].map(dict(zip(periods, dates, strict=True)))
    return (
        result[["date", "variable", "quantile", "value"]]
        .sort_values(["date", "variable", "quantile"])
        .reset_index(drop=True)
    )


def test_bvar_declares_missing_values_unsupported():
    """RealTimeModel must complete-case BVAR estimation data."""
    import forecast_realtime as rt

    assert rt.models.ForecastBVAR._handles_missing_values is False


def test_bvar_rejects_density_forecasts_type():
    """Point summaries remain separate from the quantile request option."""
    import forecast_realtime as rt

    with pytest.raises(ValueError, match="mean.*median.*quantiles"):
        rt.models.ForecastBVAR(forecasts_type="density")


@pytest.mark.parametrize("forecasts_type", ["mean", "median"])
def test_bvar_accepts_point_forecast_types(forecasts_type):
    """Mean and median are the supported BVAR forecast summaries."""
    import forecast_realtime as rt

    model = rt.models.ForecastBVAR(forecasts_type=forecasts_type)

    assert model.forecasts_type == forecasts_type


def test_bvar_density_selects_tail_maps_dates_and_clears_backend_results(monkeypatch):
    """Density forecasts use formatted tails for both forecast modes."""
    model = _fit_synthetic_bvar(monkeypatch)
    steps = 2
    variables = list(model.y.columns)
    dates = _density_dates(model, steps)
    probabilities = [0.25, 0.75]

    point_before = model.forecast(steps=steps)
    unconditional = model.forecast(steps=steps, quantiles=probabilities)
    call = model.bvar.forecast_calls[-1]

    assert call["format"] is True
    assert call["quantiles"] == probabilities
    assert call["constraint_mean"] is None
    assert call["N_burn"] is None
    assert len(call["unconditional_table"]["date"].unique()) == (
        len(model.y) - model.n_lags + steps
    )
    assert list(unconditional.columns) == ["date", "variable", "quantile", "value"]
    assert len(unconditional) == steps * len(variables) * len(probabilities)
    assert pd.DatetimeIndex(unconditional["date"].unique()).equals(dates)
    assert set(unconditional["variable"]) == set(variables)
    assert set(unconditional["quantile"]) == set(probabilities)
    pd.testing.assert_frame_equal(
        unconditional.sort_values(["date", "variable", "quantile"]).reset_index(
            drop=True
        ),
        _native_density_tail(call["unconditional_table"], dates, steps),
        check_dtype=False,
    )
    assert all(
        getattr(model.bvar, attribute) is None
        for attribute in (
            "forecast_unconditional",
            "forecast_conditional",
            "df_forecasts_unconditional",
            "df_forecasts_conditional",
        )
    )

    conditioning = pd.DataFrame(np.nan, index=dates, columns=variables)
    conditioning.iloc[0, 0] = model.y.iloc[-1, 0]
    conditional = model.forecast(steps=steps, y=conditioning, quantiles=True)
    call = model.bvar.forecast_calls[-1]

    assert call["constraint_mean"].shape == (steps, len(variables))
    np.testing.assert_allclose(
        call["constraint_mean"][:, 0], conditioning.iloc[:, 0], equal_nan=True
    )
    assert set(conditional["date"]) == set(dates)
    assert set(conditional["quantile"]) == {0.16, 0.5, 0.84}
    pd.testing.assert_frame_equal(
        conditional.sort_values(["date", "variable", "quantile"]).reset_index(drop=True),
        _native_density_tail(call["conditional_table"], dates, steps),
        check_dtype=False,
    )
    assert all(
        getattr(model.bvar, attribute) is None
        for attribute in (
            "forecast_unconditional",
            "forecast_conditional",
            "df_forecasts_unconditional",
            "df_forecasts_conditional",
        )
    )

    point_after = model.forecast(steps=steps)
    pd.testing.assert_frame_equal(point_before, point_after)


@pytest.mark.parametrize("burn", [-1, 12, 12.0, True])
def test_bvar_rejects_invalid_explicit_burn(burn):
    """Explicit burn-in must be an integer below the effective draw count."""
    with pytest.raises(ValueError, match="N_burn must be an integer"):
        rt.models.ForecastBVAR(n_samples=12, N_draws=12, N_burn=burn)


@pytest.mark.parametrize("burn", [0, 11])
def test_bvar_accepts_explicit_burn_at_valid_bounds(burn):
    """The inclusive lower and exclusive upper burn-in bounds are accepted."""
    model = rt.models.ForecastBVAR(n_samples=12, N_draws=12, N_burn=burn)

    assert model.N_burn == burn


def test_bvar_density_rejects_mode_only(monkeypatch):
    """Posterior quantiles require a sampled, rather than point-only, backend."""
    model = _fit_synthetic_bvar(monkeypatch, mode_only=True)

    with pytest.raises(ValueError, match="point_only"):
        model.forecast(steps=2, quantiles=True)
    assert not model.bvar.forecast_calls


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


def test_bvar_matches_native_unconditional_and_conditional_forecasts(
    sample_realtime_ragged,
    request,
):
    """Both forecast paths match native BVAR using the same fitted models."""
    variables = ["quarterly_1", "quarterly_2"]
    outturns = sample_realtime_ragged.query("metric == 'levels'").copy()
    outturns = outturns[outturns["variable"].isin(variables)]

    vintage = pd.Timestamp("2024-06-30")
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
    native_model.optimise_hyperparameters(y_vintage, nb_restart=0, random_state=0)
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
        nb_restart=0,
        mode_only=True,
        optim_random_state=0,
        sampling_random_state=0,
        forecast_random_state=0,
    )
    wrapper.fit(y=y_vintage)
    wrapper_forecasts = wrapper.forecast(steps=H)

    np.testing.assert_allclose(
        native_forecasts,
        _point_forecast_to_wide(wrapper_forecasts, variables).to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
    compiled_unconditional = wrapper_forecasts.copy()

    # Build deterministic conditioning paths: quarterly_1 for two steps and
    # quarterly_2 for one.
    y_columns = list(y_vintage.columns)
    constraint_mean = np.full((H, len(y_columns)), np.nan)
    conditioning = {"quarterly_1": 1, "quarterly_2": 0}
    for var, steps_ahead in conditioning.items():
        adjusted = steps_ahead + 1
        col_idx = y_columns.index(var)
        constraint_mean[:adjusted, col_idx] = y_vintage[var].iloc[-1]

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

    # make constraint_mean a DataFrame with proper dates
    constraint_mean_df = pd.DataFrame(
        constraint_mean,
        columns=y_vintage.columns,
        index=pd.date_range(
            start=y_vintage.index[-1] + pd.offsets.QuarterEnd(),
            periods=H,
            freq="QE",
        ),
    )
    wrapper_forecasts = wrapper.forecast(steps=H, y=constraint_mean_df)

    np.testing.assert_allclose(
        native_forecasts,
        _point_forecast_to_wide(wrapper_forecasts, variables).to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # Other contract tests use this same kernel without its compilation overhead.
    request.getfixturevalue("bvar_python_kernel")
    pd.testing.assert_frame_equal(
        wrapper.forecast(steps=H), compiled_unconditional, rtol=1e-12, atol=1e-12
    )
    pd.testing.assert_frame_equal(
        wrapper.forecast(steps=H, y=constraint_mean_df),
        wrapper_forecasts,
        rtol=1e-12,
        atol=1e-12,
    )


def _build_native_density_data():
    rng = np.random.default_rng(20260922)
    values = np.empty((48, 2))
    values[0] = [100.0, 80.0]
    for row in range(1, len(values)):
        values[row] = (
            np.array([0.65, 0.7]) * values[row - 1]
            + np.array([35.0, 24.0])
            + rng.normal(scale=[0.25, 0.2])
        )
    return pd.DataFrame(
        values,
        columns=["var_a", "var_b"],
        index=pd.date_range("2012-03-31", periods=len(values), freq="QE"),
    )


def test_bvar_density_matches_native_formatted_tails():
    """Compiled native density tails match the wrapper in both forecast modes."""
    history = _build_native_density_data()
    probabilities = [0.25, 0.5, 0.75]
    steps = 2
    model = rt.models.ForecastBVAR(
        stationary=True,
        n_lags=1,
        nb_restart=0,
        n_samples=16,
        N_draws=16,
        N_burn=8,
        progressbar=False,
        optim_random_state=11,
        sampling_random_state=13,
        forecast_random_state=17,
    )
    model.fit(history)

    oracle = deepcopy(model.bvar)
    oracle.data_transformation = {"var_a": "levels", "var_b": "levels"}
    dates = _density_dates(model, steps)
    conditioning = pd.DataFrame(np.nan, index=dates, columns=history.columns)
    conditioning.loc[dates[0], "var_a"] = history.iloc[-1, 0] * 1.01
    conditioning.loc[dates[1], "var_a"] = history.iloc[-1, 0] * 1.02

    for supplied_conditioning in (None, conditioning):
        wrapper_result = model.forecast(
            steps=steps,
            y=supplied_conditioning,
            quantiles=probabilities,
        )
        constraint_mean = (
            None if supplied_conditioning is None else supplied_conditioning.to_numpy()
        )
        oracle.forecast(
            H=steps,
            constraint_mean=constraint_mean,
            point_only=False,
            method=model.method,
            N_draws=model.N_draws,
            N_burn=model.N_burn,
            base_value=model.base_value,
            progressbar=False,
            random_state=model.forecast_random_state,
            format=True,
            quantiles=probabilities,
        )
        table = (
            oracle.df_forecasts_conditional
            if constraint_mean is not None
            else oracle.df_forecasts_unconditional
        )
        assert isinstance(table["date"].iloc[0], pd.Period)
        expected = _native_density_tail(table, dates, steps)
        actual = (
            wrapper_result[["date", "variable", "quantile", "value"]]
            .sort_values(["date", "variable", "quantile"])
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            rtol=1e-10,
            atol=1e-10,
        )


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
