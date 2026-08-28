"""Integration tests using synthetic data with publication lags."""

import forecast_evaluation as fe
import numpy as np
import pandas as pd
import pytest

import forecast_realtime as rt
from forecast_realtime._utils import impute_X

from .schemas import decomposition_schema


@pytest.mark.parametrize(
    ("fixture_name", "frequency", "expected_lags"),
    [
        (
            "quarterly_outturns",
            "Q",
            {"quarterly_target": 1, "quarterly_fast": 0, "quarterly_slow": 2},
        ),
        (
            "monthly_outturns",
            "M",
            {"monthly_target": 1, "monthly_fast": 0, "monthly_slow": 3},
        ),
        (
            "mixed_frequency_outturns",
            None,
            {"quarterly_target": 1, "monthly_fast": 0, "monthly_slow": 2},
        ),
    ],
)
def test_lagged_dataset_has_expected_first_release_lags(
    request, fixture_name, frequency, expected_lags
):
    """Each synthetic series has a known lag and one later revision."""
    outturns = request.getfixturevalue(fixture_name)

    for variable, expected_lag in expected_lags.items():
        variable_data = outturns[outturns["variable"] == variable]
        variable_frequency = variable_data["frequency"].iloc[0]
        assert frequency is None or variable_frequency == frequency
        assert variable_data.groupby("date").size().eq(2).all()

        first_release = variable_data.sort_values("vintage_date").iloc[0]
        actual_lag = (
            pd.Period(first_release["vintage_date"], freq=variable_frequency)
            - pd.Period(first_release["date"], freq=variable_frequency)
        ).n
        assert actual_lag == expected_lag


@pytest.mark.parametrize(
    ("fixture_name", "frequency", "y_variable", "X_variables"),
    [
        (
            "quarterly_outturns",
            "Q",
            "quarterly_target",
            ["quarterly_fast", "quarterly_slow"],
        ),
        (
            "monthly_outturns",
            "M",
            "monthly_target",
            ["monthly_fast", "monthly_slow"],
        ),
    ],
)
def test_realtime_ols_handles_lagged_single_frequency_data(
    request, fixture_name, frequency, y_variable, X_variables
):
    """OLS forecasts remain finite across vintages with ragged data edges."""
    outturns = request.getfixturevalue(fixture_name)
    data = fe.ForecastData(
        outturns_data=outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())
    transformations = {variable: "levels" for variable in [y_variable, *X_variables]}

    model.forecast(
        y_variables=[y_variable],
        X_variables=X_variables,
        data_transformation=transformations,
        steps=2,
        first_vintage="2017-12-31",
        last_vintage="2018-12-31",
        X_imputation="last",
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            (frame["source"] == "ForecastOLS")
            & (frame["variable"] == y_variable)
            & (frame["metric"] == "levels")
        )
    ]

    assert forecasts["vintage_date"].nunique() >= 5
    assert forecasts.groupby("vintage_date").size().eq(2).all()
    assert forecasts["forecast_horizon"].isin([0, 1]).all()
    assert np.isfinite(forecasts["value"]).all()


@pytest.mark.parametrize(
    ("fixture_name", "y_variable", "X_variables", "frequency"),
    [
        ("monthly_outturns", "monthly_target", ["monthly_fast"], "M"),
        ("quarterly_outturns", "quarterly_target", ["quarterly_fast"], "Q"),
    ],
)
def test_realtime_ols_lags_work_at_monthly_and_quarterly_frequency(
    request, fixture_name, y_variable, X_variables, frequency
):
    """OLS builds and forecasts both y and X lags at the target frequency."""
    outturns = request.getfixturevalue(fixture_name)
    data = fe.ForecastData(
        outturns_data=outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())

    model.forecast(
        y_variables=[y_variable],
        X_variables=X_variables,
        data_transformation={
            y_variable: "levels",
            **{x: "levels" for x in X_variables},
        },
        steps=2,
        y_lags=1,
        X_lags=1,
        first_vintage="2017-12-31",
        last_vintage="2018-12-31",
        X_imputation="last",
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            frame["source"].eq("ForecastOLS")
            & frame["variable"].eq(y_variable)
            & frame["metric"].eq("levels")
        )
    ]
    assert not forecasts.empty
    assert forecasts["frequency"].eq(frequency).all()
    assert forecasts.groupby("vintage_date").size().eq(2).all()
    assert forecasts["forecast_horizon"].isin([0, 1]).all()
    assert np.isfinite(forecasts["value"]).all()


@pytest.mark.parametrize(
    ("fixture_name", "y_variable", "X_variables"),
    [
        ("monthly_outturns", "monthly_target", ["monthly_fast"]),
        ("quarterly_outturns", "quarterly_target", ["quarterly_fast"]),
    ],
)
def test_realtime_ols_decomposition_reconstructs_lagged_forecasts(
    request, fixture_name, y_variable, X_variables
):
    """Lagged OLS decomposition reconstructs monthly and quarterly forecasts."""
    outturns = request.getfixturevalue(fixture_name)
    data = fe.ForecastData(
        outturns_data=outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())

    model.forecast(
        y_variables=[y_variable],
        X_variables=X_variables,
        data_transformation={
            y_variable: "levels",
            **{x: "levels" for x in X_variables},
        },
        steps=2,
        y_lags=1,
        X_lags=1,
        first_vintage="2017-12-31",
        last_vintage="2018-12-31",
        X_imputation="last",
        decomp=True,
        reconstruct_levels=False,
    )

    decomposition_schema.validate(model.decompositions)
    level_decompositions = model.decompositions.loc[
        model.decompositions["decomposition"].eq("level")
    ]
    assert {f"{y_variable}_lag1", f"{X_variables[0]}_lag1"}.issubset(
        set(level_decompositions["component"])
    )
    forecasts = model.data.forecasts.loc[
        lambda frame: (
            frame["source"].eq("ForecastOLS")
            & frame["variable"].eq(y_variable)
            & frame["metric"].eq("levels")
        )
    ]
    totals = level_decompositions.groupby(["vintage_date", "date", "forecast_horizon"])[
        "contribution"
    ].sum()
    forecast_values = forecasts.set_index(["vintage_date", "date", "forecast_horizon"])[
        "value"
    ]
    np.testing.assert_allclose(
        totals.sort_index().to_numpy(),
        forecast_values.reindex(totals.index).sort_index().to_numpy(),
        atol=1e-8,
    )


def test_realtime_ols_forecasts_three_missing_target_months(monthly_outturns):
    """The default start forecasts all three months missing from the target."""
    test_vintage = pd.Timestamp("2018-12-31")
    outturns = monthly_outturns.copy()
    target_mask = outturns["variable"].eq("monthly_target")
    outturns.loc[target_mask, "vintage_date"] += pd.offsets.MonthEnd(2)

    expected_dates = pd.date_range("2018-10-31", periods=3, freq="ME")
    available_target_dates = pd.DatetimeIndex(
        outturns.loc[
            target_mask & (outturns["vintage_date"] <= test_vintage), "date"
        ].unique()
    )
    assert available_target_dates.max() == expected_dates[0] - pd.offsets.MonthEnd(1)
    assert set(expected_dates).isdisjoint(available_target_dates)

    data = fe.ForecastData(
        outturns_data=outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())
    model.forecast(
        y_variables=["monthly_target"],
        X_variables=["monthly_fast"],
        data_transformation={"monthly_target": "levels", "monthly_fast": "levels"},
        steps=3,
        first_vintage=str(test_vintage.date()),
        last_vintage=str(test_vintage.date()),
        X_imputation="last",
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            (frame["source"] == "ForecastOLS")
            & (frame["variable"] == "monthly_target")
            & (frame["metric"] == "levels")
        )
    ].sort_values("date")

    assert forecasts["date"].tolist() == expected_dates.tolist()
    assert forecasts["forecast_horizon"].tolist() == [0, 1, 2]
    assert np.isfinite(forecasts["value"]).all()


def test_realtime_ols_mixed_frequency_formula_decomposition(
    sample_outturns,
):
    """OLS decomp uses complete quarterly rows after formula selection."""
    y_variable = "quarterly_a"
    quarterly_regressors = ["quarterly_b", "quarterly_c"]
    X_variables = [*quarterly_regressors, "monthly_a"]

    data = fe.ForecastData(
        outturns_data=sample_outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(
        data=data,
        models=rt.models.ForecastOLS(
            formula=f"{y_variable} ~ " + " + ".join(quarterly_regressors),
        ),
    )

    model.forecast(
        y_variables=[y_variable],
        X_variables=X_variables,
        data_transformation={
            y_variable: "levels",
            **{x: "levels" for x in X_variables},
        },
        steps=2,
        first_vintage="2018-06-30",
        last_vintage="2019-06-30",
        X_imputation="last",
        decomp=True,
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            (frame["source"] == "ForecastOLS")
            & (frame["variable"] == y_variable)
            & (frame["metric"] == "levels")
        )
    ]
    decompositions = model.decompositions.loc[
        lambda frame: frame["source"] == "ForecastOLS"
    ]

    assert not forecasts.empty
    assert forecasts["forecast_horizon"].isin([0, 1]).all()
    assert np.isfinite(forecasts["value"]).all()
    assert not decompositions.empty
    assert decompositions["forecast_horizon"].isin([0, 1]).all()
    assert np.isfinite(decompositions["contribution"]).all()
    level_decompositions = decompositions.loc[decompositions["decomposition"] == "level"]
    totals = level_decompositions.groupby(["vintage_date", "date", "forecast_horizon"])[
        "contribution"
    ].sum()
    forecast_values = forecasts.set_index(["vintage_date", "date", "forecast_horizon"])[
        "value"
    ]
    np.testing.assert_allclose(
        totals.sort_index().to_numpy(),
        forecast_values.reindex(totals.index).sort_index().to_numpy(),
        atol=1e-8,
    )


def test_realtime_ols_formula_ignores_unselected_mixed_frequency_targets(
    sample_outturns,
):
    """Formula-selected OLS ignores requested target columns at other frequencies."""
    data = fe.ForecastData(
        outturns_data=sample_outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(
        data=data,
        models=rt.models.ForecastOLS(formula="quarterly_a ~ quarterly_b"),
    )

    model.forecast(
        y_variables=["quarterly_a", "monthly_a"],
        X_variables=["quarterly_b"],
        data_transformation={
            "quarterly_a": "levels",
            "monthly_a": "levels",
            "quarterly_b": "levels",
        },
        steps=1,
        step_frequency="Q",
        first_vintage="2018-06-30",
        last_vintage="2018-06-30",
        X_imputation="last",
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            frame["source"].eq("ForecastOLS")
            & frame["variable"].eq("quarterly_a")
            & frame["metric"].eq("levels")
        )
    ]
    assert not forecasts.empty
    assert np.isfinite(forecasts["value"]).all()


@pytest.mark.parametrize("X_variable", ["monthly_fast", "monthly_slow"])
def test_realtime_midas_handles_mixed_frequency_lagged_data(
    mixed_frequency_outturns, X_variable
):
    """MIDAS forecasts a quarterly target from ragged monthly regressors."""
    pytest.importorskip("nowcast_midas")

    data = fe.ForecastData(
        outturns_data=mixed_frequency_outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(
        data=data,
        models=rt.models.ForecastMIDAS(
            method="almon",
            n_lags=3,
            estimator="ols",
            horizons=[0, 1],
        ),
    )

    model.forecast(
        y_variables=["quarterly_target"],
        X_variables=[X_variable],
        data_transformation={"quarterly_target": "levels", X_variable: "levels"},
        steps=2,
        first_vintage="2017-12-31",
        last_vintage="2018-12-31",
        X_imputation="last",
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            (frame["source"] == "ForecastMIDAS")
            & (frame["variable"] == "quarterly_target")
            & (frame["metric"] == "levels")
        )
    ]

    assert forecasts["vintage_date"].nunique() >= 5
    assert forecasts.groupby("vintage_date").size().between(1, 2).all()
    assert forecasts.groupby("vintage_date")["forecast_horizon"].agg(set).eq({0, 1}).all()
    assert forecasts["forecast_horizon"].isin([0, 1]).all()
    representative_forecasts = forecasts.loc[
        forecasts["vintage_date"].eq(pd.Timestamp("2018-12-31"))
    ]
    assert len(representative_forecasts) == 2
    assert set(representative_forecasts["forecast_horizon"]) == {0, 1}
    assert np.isfinite(forecasts["value"]).all()


def _pivot_latest_vintage(outturns, variables, vintage):
    """Select the latest release for each date at or before ``vintage``."""
    subset = outturns[outturns["variable"].isin(variables)]
    subset = subset[subset["vintage_date"] <= vintage]
    subset = subset.sort_values("vintage_date", ascending=False).drop_duplicates(
        subset=["date", "variable"], keep="first"
    )
    return subset.pivot(index="date", columns="variable", values="value")


def _manual_bridge_ols_forecast(
    outturns, y_variable, X_variables, vintage, last_observed_date, steps, X_imputation
):
    """Replicate one iteration of ``RealTimeModel``'s vintage loop directly
    with pandas + ``ForecastBridgeOLS``, mirroring the vintage selection,
    fit-time imputation, and forecast-time imputation performed by the loop."""
    y_wide = _pivot_latest_vintage(outturns, [y_variable], vintage)
    X_wide = _pivot_latest_vintage(outturns, X_variables, vintage)
    y_fit = y_wide[y_wide.index <= last_observed_date]

    X_fit = X_wide
    if X_imputation is not None:
        last_valid_dates = [X_fit[col].last_valid_index() for col in X_fit.columns]
        last_valid_dates = [d for d in last_valid_dates if d is not None]
        target_date = max([*last_valid_dates, y_fit.index[-1]])
        X_fit = impute_X(
            X_fit,
            target_date,
            steps=0,
            method=X_imputation,
            frequencies={column: "M" for column in X_fit},
        )

    manual_model = rt.models.ForecastBridgeOLS()
    manual_model.fit(y_fit, X=X_fit)

    X_forecast = X_fit
    if X_imputation is not None:
        forecast_target_period = pd.Period(manual_model.last_y_fit_date, freq="Q") + steps
        forecast_target_date = forecast_target_period.to_timestamp(how="end").normalize()
        X_forecast = impute_X(
            X_forecast,
            forecast_target_date,
            steps=0,
            method=X_imputation,
            frequencies={column: "M" for column in X_forecast},
        )

    return manual_model.forecast(steps=steps, X=X_forecast)


def test_realtime_bridge_ols_mixed_frequency_quarterly_target(mixed_frequency_outturns):
    """Bridge OLS forecasts a quarterly target from ragged monthly regressors,
    matching a manually vintage-looped Bridge OLS fit/forecast."""
    y_variable = "quarterly_target"
    X_variables = ["monthly_fast", "monthly_slow"]
    outturns = mixed_frequency_outturns

    data = fe.ForecastData(
        outturns_data=outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=rt.models.ForecastBridgeOLS())

    model.forecast(
        y_variables=[y_variable],
        X_variables=X_variables,
        data_transformation={
            y_variable: "levels",
            **{x: "levels" for x in X_variables},
        },
        steps=2,
        first_vintage="2017-12-31",
        last_vintage="2018-12-31",
        X_imputation="last",
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            (frame["source"] == "ForecastBridgeOLS")
            & (frame["variable"] == y_variable)
            & (frame["metric"] == "levels")
        )
    ]

    assert forecasts["vintage_date"].nunique() >= 5
    assert forecasts["forecast_horizon"].isin([0, 1]).all()
    assert np.isfinite(forecasts["value"]).all()

    # Manually replicate the vintage loop for a single vintage: select the
    # latest release for y/X as of that vintage, aggregate X to quarterly,
    # fit and forecast with Bridge OLS directly.
    check_vintage = pd.Timestamp("2018-12-31")
    last_observed_date = pd.Timestamp("2018-09-30")
    manual_forecast = _manual_bridge_ols_forecast(
        outturns,
        y_variable,
        X_variables,
        vintage=check_vintage,
        last_observed_date=last_observed_date,
        steps=2,
        X_imputation="last",
    )

    model_forecasts = forecasts.loc[
        lambda frame: frame["vintage_date"] == check_vintage
    ].sort_values("forecast_horizon")

    assert model_forecasts["forecast_horizon"].tolist() == [0, 1]
    np.testing.assert_allclose(
        model_forecasts["value"].to_numpy(),
        manual_forecast.to_numpy().ravel(),
        atol=1e-8,
    )


def test_realtime_bridge_ols_respects_publication_lags(mixed_frequency_outturns):
    """At an early vintage, ``monthly_slow`` (2-month publication lag) has an
    incomplete current quarter while ``monthly_fast`` (0-month lag) is
    complete. Without imputation, Bridge OLS's own complete-quarter rule
    leaves the nowcast regressor NaN (and the resulting forecast NaN);
    with ``X_imputation`` enabled, the padded value is incorporated and the
    forecast matches a manual computation using the same padding."""
    y_variable = "quarterly_target"
    X_variables = ["monthly_fast", "monthly_slow"]
    outturns = mixed_frequency_outturns
    check_vintage = pd.Timestamp("2018-03-31")
    last_observed_date = pd.Timestamp("2017-12-31")

    # Sanity-check the fixture's publication-lag assumption for this vintage:
    # monthly_slow's current quarter (Jan-Mar 2018) is incomplete, while
    # monthly_fast's is complete.
    X_wide = _pivot_latest_vintage(outturns, X_variables, check_vintage)
    q1_2018 = X_wide.loc["2018-01-31":"2018-03-31"]
    assert q1_2018["monthly_fast"].notna().sum() == 3
    assert q1_2018["monthly_slow"].notna().sum() < 3

    def _run(X_imputation):
        data = fe.ForecastData(
            outturns_data=outturns,
            metric="levels",
            compute_levels=False,
            data_check=False,
        )
        model = rt.RealTimeModel(data=data, models=rt.models.ForecastBridgeOLS())
        model.forecast(
            y_variables=[y_variable],
            X_variables=X_variables,
            data_transformation={
                y_variable: "levels",
                **{x: "levels" for x in X_variables},
            },
            steps=1,
            first_vintage="2017-12-31",
            last_vintage="2018-12-31",
            X_imputation=X_imputation,
        )
        all_forecasts = model.data.forecasts
        if "source" not in all_forecasts.columns:
            # No forecasts survived at all (e.g. every vintage's nowcast was
            # NaN and got dropped), so there's nothing to filter.
            return all_forecasts.iloc[0:0]
        return all_forecasts.loc[
            lambda frame: (
                (frame["source"] == "ForecastBridgeOLS")
                & (frame["variable"] == y_variable)
                & (frame["metric"] == "levels")
                & (frame["vintage_date"] == check_vintage)
                & (frame["forecast_horizon"] == 0)
            )
        ]

    # Without imputation: monthly_slow's incomplete current quarter leaves the
    # aggregated regressor (and hence the nowcast) NaN for every vintage, so
    # NaN forecasts are dropped and none are produced for this variable.
    no_impute_forecast = _run(None)
    assert len(no_impute_forecast) == 0

    # With imputation: the padded value is incorporated, matching a manual
    # fit/forecast that pads X the same way before aggregating to quarterly.
    imputed_forecast = _run("last")
    assert len(imputed_forecast) == 1
    assert np.isfinite(imputed_forecast["value"].iloc[0])

    manual_forecast = _manual_bridge_ols_forecast(
        outturns,
        y_variable,
        X_variables,
        vintage=check_vintage,
        last_observed_date=last_observed_date,
        steps=1,
        X_imputation="last",
    )

    np.testing.assert_allclose(
        imputed_forecast["value"].to_numpy(),
        manual_forecast.to_numpy().ravel(),
        atol=1e-8,
    )


def test_realtime_bridge_decomposition_supports_mixed_regressors_and_lags(
    sample_outturns,
):
    """Bridge decomposition includes mixed regressors after lag construction."""
    data = fe.ForecastData(
        outturns_data=sample_outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(
        data=data,
        models=rt.models.ForecastBridgeOLS(
            formula="quarterly_a ~ monthly_a + quarterly_b",
        ),
    )

    model.forecast(
        y_variables=["quarterly_a"],
        X_variables=["monthly_a", "quarterly_b"],
        data_transformation={
            "quarterly_a": "levels",
            "monthly_a": "levels",
            "quarterly_b": "levels",
        },
        steps=2,
        y_lags=1,
        X_lags=1,
        first_vintage="2018-06-30",
        last_vintage="2019-06-30",
        X_imputation="last",
        decomp=True,
        reconstruct_levels=False,
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            frame["source"].eq("ForecastBridgeOLS")
            & frame["variable"].eq("quarterly_a")
            & frame["metric"].eq("levels")
        )
    ]
    decompositions = model.decompositions

    assert not forecasts.empty
    assert forecasts["forecast_horizon"].isin([0, 1]).all()
    assert np.isfinite(forecasts["value"]).all()
    assert decompositions is not None
    assert not decompositions.empty

    level_decompositions = decompositions.loc[decompositions["decomposition"].eq("level")]
    assert {"monthly_a", "quarterly_b"}.issubset(set(level_decompositions["component"]))
    totals = level_decompositions.groupby(["vintage_date", "date", "forecast_horizon"])[
        "contribution"
    ].sum()
    forecast_values = forecasts.set_index(["vintage_date", "date", "forecast_horizon"])[
        "value"
    ]
    np.testing.assert_allclose(
        totals.sort_index().to_numpy(),
        forecast_values.reindex(totals.index).sort_index().to_numpy(),
        atol=1e-8,
    )
