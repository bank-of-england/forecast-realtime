import copy
import inspect
import time
import warnings

import forecast_evaluation as fe
import numpy as np
import pandas as pd
import pytest

import forecast_realtime as rt
from forecast_realtime import ForecastModel
from forecast_realtime import real_time_model as real_time_model_module
from forecast_realtime._utils import _ar1_t_impute, impute_X
from forecast_realtime.data_transformation import DataTransformationPipeline
from forecast_realtime.forecast_tree import ForecastTree, TreeNode
from forecast_realtime.real_time_model import _level_contributions

from .schemas import decomposition_schema


class _RequiredForecastOptionModel(ForecastModel):
    """Require one model option during both fitting and forecasting."""

    def _fit(self, y, X=None, forecast_value=None, **kwargs):
        if forecast_value is None:
            raise ValueError("forecast_value is required")
        return self

    def _forecast(self, steps, X=None, y=None, forecast_value=None, **kwargs):
        if forecast_value is None:
            raise ValueError("forecast_value is required")
        return np.full((steps, len(self.y.columns)), forecast_value)


class _CaptureFitModel(ForecastModel):
    """Capture the vintage-specific transformed history supplied to fitting."""

    captured_fit_y: list[pd.DataFrame] = []

    def _fit(self, y, X=None, **kwargs):
        _CaptureFitModel.captured_fit_y.append(y.copy())
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        return pd.DataFrame({self.y.columns[0]: np.zeros(steps)})


@pytest.mark.parametrize(
    ("transformation", "expected_first", "expected_last"),
    [
        ("levels", [110.0, 110.0], [110.0, 121.0, 132.0]),
        ("pop", [0.0], [10.0, (132.0 / 121.0 - 1) * 100]),
    ],
)
def test_realtime_fitting_uses_as_of_revisions_before_transformation(
    transformation, expected_first, expected_last
):
    """Revisions become available only at their release vintage."""
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-02-29",
                    "2020-03-31",
                    "2020-03-31",
                ]
            ),
            "variable": "target",
            "vintage_date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-02-29",
                    "2020-03-31",
                    "2020-03-31",
                    "2020-04-30",
                ]
            ),
            "frequency": "M",
            "value": [100.0, 110.0, 110.0, 121.0, 121.0, 132.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    _CaptureFitModel.captured_fit_y = []
    model = rt.RealTimeModel(
        data=data,
        models=_CaptureFitModel(
            data_transformation={"target": transformation},
        ),
    )

    model.forecast(
        y_variables=["target"],
        steps=1,
        first_vintage="2020-02-29",
        last_vintage="2020-04-30",
    )

    assert len(_CaptureFitModel.captured_fit_y) == 3
    np.testing.assert_allclose(
        _CaptureFitModel.captured_fit_y[0]["target"].to_numpy(), expected_first
    )
    np.testing.assert_allclose(
        _CaptureFitModel.captured_fit_y[-1]["target"].to_numpy(), expected_last
    )


def _get_vintage_data(
    forecast_data,
    vintage,
    y_variables,
    X_variables=None,
    last_observed_date=None,
):
    """Extract y and X DataFrames for a single vintage, mirroring RealTimeModel."""
    outturns = getattr(forecast_data, "_raw_outturns", forecast_data.outturns).copy()
    all_vars = y_variables + (X_variables or [])
    outturns = outturns[outturns["variable"].isin(all_vars)]

    at_vintage = outturns[outturns["vintage_date"] <= vintage].copy()
    at_vintage = at_vintage.sort_values("vintage_date", ascending=False).drop_duplicates(
        subset=["date", "variable"], keep="first"
    )

    y_out = at_vintage[at_vintage["variable"].isin(y_variables)]
    y_wide = y_out.pivot(index="date", columns="variable", values="value")
    if last_observed_date is None:
        last_observed_date = vintage - pd.offsets.MonthEnd(1)
    y_wide = y_wide[y_wide.index <= last_observed_date]

    X_wide = None
    if X_variables:
        X_out = at_vintage[at_vintage["variable"].isin(X_variables)]
        X_wide = X_out.pivot(index="date", columns="variable", values="value")

    return y_wide, X_wide


def test_realtime_forwards_model_options_to_fit_and_forecast():
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "variable": "target",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [1.0, 2.0, 3.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=_RequiredForecastOptionModel())

    model.forecast(
        y_variables=["target"],
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        forecast_value=7.0,
    )

    forecast = model.data.forecasts.loc[
        (model.data.forecasts["variable"] == "target")
        & (model.data.forecasts["forecast_horizon"] == 0),
        "value",
    ]
    assert (forecast == 7.0).any()


def test_realtime_infers_step_frequency_from_y_variables():
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "variable": "target",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [1.0, 2.0, 3.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=_RequiredForecastOptionModel())

    model.forecast(
        y_variables=["target"],
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        forecast_value=7.0,
    )

    assert (model.data.forecasts["value"] == 7.0).any()


def test_realtime_infers_long_input_frequencies_once_per_forecast(monkeypatch):
    vintages = pd.to_datetime(["2020-01-31", "2020-02-29"])
    dates = pd.date_range("2019-11-30", periods=4, freq="ME")
    outturns = pd.DataFrame(
        {
            "date": list(dates) * len(vintages),
            "variable": "target",
            "vintage_date": np.repeat(vintages, len(dates)),
            "frequency": "M",
            "value": np.arange(float(len(dates) * len(vintages))),
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    calls = []
    infer_frequencies = real_time_model_module.infer_long_variable_frequencies

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return infer_frequencies(*args, **kwargs)

    monkeypatch.setattr(
        real_time_model_module,
        "infer_long_variable_frequencies",
        spy,
    )
    model = rt.RealTimeModel(data=data, models=_RequiredForecastOptionModel())

    model.forecast(
        y_variables=["target"],
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintages[0].date()),
        last_vintage=str(vintages[-1].date()),
        forecast_value=7.0,
    )

    assert len(calls) == 1


def test_realtime_requires_step_frequency_for_mixed_y_frequencies():
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-02-29", "2020-03-31", "2020-06-30"]
            ),
            "variable": ["monthly", "monthly", "quarterly", "quarterly"],
            "vintage_date": vintage,
            "frequency": ["M", "M", "Q", "Q"],
            "value": [1.0, 2.0, 1.5, 2.5],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=_RequiredForecastOptionModel())

    with pytest.raises(ValueError, match="step_frequency"):
        model.forecast(y_variables=["monthly", "quarterly"])


@pytest.mark.parametrize(
    ("model_class", "model_options"),
    [
        (rt.models.ForecastOLS, {}),
        (rt.models.ForecastRidge, {"alpha": 1e-8}),
        (rt.models.ForecastLasso, {"alpha": 1e-8}),
        (rt.models.ForecastElasticNet, {"alpha": 1e-8}),
    ],
)
def test_realtime_linear_models_reject_mixed_frequency_data(
    mixed_frequency_outturns, model_class, model_options
):
    """Unsupported linear models reject mixed-frequency inputs via the API."""
    data = fe.ForecastData(
        outturns_data=mixed_frequency_outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=model_class(**model_options))

    with pytest.raises(ValueError, match="does not support mixed frequencies"):
        model.forecast(
            y_variables=["quarterly_target"],
            X_variables=["monthly_fast"],
            data_transformation={
                "quarterly_target": "levels",
                "monthly_fast": "levels",
            },
            steps=1,
            first_vintage="2018-12-31",
            last_vintage="2018-12-31",
            X_imputation="last",
        )


@pytest.mark.parametrize(
    ("model_class", "model_options"),
    [
        (rt.models.ForecastOLS, {}),
        (rt.models.ForecastRidge, {"alpha": 1e-8}),
        (rt.models.ForecastLasso, {"alpha": 1e-8}),
        (rt.models.ForecastElasticNet, {"alpha": 1e-8}),
    ],
)
def test_realtime_linear_models_reject_explicit_step_frequency_bypass(
    mixed_frequency_outturns, model_class, model_options
):
    """An explicit forecast frequency cannot hide mixed-frequency input data."""
    data = fe.ForecastData(
        outturns_data=mixed_frequency_outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=model_class(**model_options))

    with pytest.raises(ValueError, match="does not support mixed frequencies"):
        model.forecast(
            y_variables=["quarterly_target"],
            X_variables=["monthly_fast"],
            data_transformation={
                "quarterly_target": "levels",
                "monthly_fast": "levels",
            },
            steps=1,
            step_frequency="M",
            first_vintage="2018-12-31",
            last_vintage="2018-12-31",
            X_imputation="last",
        )


def test_realtime_linear_model_accepts_explicit_same_frequency_step(
    monthly_outturns,
):
    """A same-frequency explicit forecast frequency is accepted."""
    data = fe.ForecastData(
        outturns_data=monthly_outturns,
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())

    model.forecast(
        y_variables=["monthly_target"],
        X_variables=["monthly_fast"],
        data_transformation={
            "monthly_target": "levels",
            "monthly_fast": "levels",
        },
        steps=1,
        step_frequency="M",
        first_vintage="2018-12-31",
        last_vintage="2018-12-31",
        X_imputation="last",
    )

    forecasts = model.data.forecasts.loc[
        lambda frame: (
            (frame["source"] == "ForecastOLS")
            & (frame["variable"] == "monthly_target")
            & (frame["metric"] == "levels")
        )
    ]

    assert len(forecasts) == 1
    assert np.isfinite(forecasts["value"]).all()


def test_forecast_rejects_dated_dataframe_with_reordered_target_columns():
    """Dated DataFrame forecasts must preserve fitted target order."""

    class ReorderedForecastModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return pd.DataFrame(
                [[10.0, 20.0]],
                index=pd.DatetimeIndex(["2020-03-31"]),
                columns=["target_b", "target_a"],
            )

    model = ReorderedForecastModel()
    model.fit(
        pd.DataFrame(
            [[1.0, 2.0], [2.0, 3.0]],
            index=pd.date_range("2020-01-31", periods=2, freq="ME"),
            columns=["target_a", "target_b"],
        )
    )

    with pytest.raises(ValueError, match="Forecast columns must match"):
        model.forecast(steps=1)


@pytest.mark.parametrize("parallel", [False, True])
def test_forecast_raises_domain_error_when_all_vintages_are_skipped(parallel):
    """Explain when no selected vintage has usable target data."""
    vintage = pd.Timestamp("2020-01-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 101.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())

    with pytest.raises(
        ValueError,
        match=(
            r"No forecasts could be produced for y_variables=\['gdp'\]"
            r" and X_variables=None across"
        ),
    ):
        model.forecast(
            y_variables=["gdp"],
            data_transformation={"gdp": "levels"},
            steps=1,
            first_forecast_horizon=0,
            first_vintage=str(vintage.date()),
            last_vintage=str(vintage.date()),
            parallel=parallel,
            max_workers=1,
        )


def test_nowcast_overlap_uses_forecast_as_the_differencing_base():
    """An overlapping nowcast must not become a second observation in ``diff``.

    The high-level forecast output echoes conditioning values.  Therefore, the
    reconstructed levels verify that the forecast path, rather than the
    overlapping outturn, supplies the base for the April difference.
    """

    class ConditioningEchoModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return y.tail(steps)

    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    nowcasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-04-30"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "source": "nowcast",
            "frequency": "M",
            "value": [50.0, 60.0],
            "metric": "levels",
            "forecast_horizon": [0, 1],
        }
    )
    data = fe.NowcastData(
        outturns_data=outturns,
        forecasts_data=nowcasts,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=ConditioningEchoModel())

    model.forecast(
        y_variables=["gdp"],
        y_steps_ahead={"gdp": 1},
        y_sources={"gdp": "nowcast"},
        data_transformation={"gdp": "diff"},
        steps=2,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        drop_transformation_nans=False,
    )

    reconstructed = model.data.forecasts.loc[
        lambda df: (
            (df["source"] == "ConditioningEchoModel")
            & (df["variable"] == "gdp")
            & (df["metric"] == "levels")
            & (df["vintage_date"] == vintage)
        ),
        ["date", "value"],
    ].sort_values("date")

    # The forecast path is [50, 60], so its differences reconstruct to those
    # same levels instead of treating the March outturn (121) as its base.
    np.testing.assert_allclose(reconstructed["value"], [50.0, 60.0])


@pytest.mark.parametrize(
    ("argument", "variables", "steps_ahead", "sources"),
    [
        ("y_steps_ahead", ["monthly_a"], {"monthly_a": 2}, {"monthly_a": "source"}),
        (
            "X_steps_ahead",
            ["monthly_a", "monthly_b"],
            {"monthly_b": 2},
            {"monthly_b": "source"},
        ),
    ],
)
def test_conditioning_horizon_must_be_less_than_steps(
    forecast_data, argument, variables, steps_ahead, sources
):
    """Conditioning horizons are zero-based and cannot equal ``steps``."""
    forecast_kwargs = {
        "y_variables": [variables[0]],
        "data_transformation": {variable: "levels" for variable in variables},
        "frequency": "M",
        "steps": 2,
        "first_forecast_horizon": 0,
        "first_vintage": "2020-01-31",
        "last_vintage": "2020-01-31",
        argument: steps_ahead,
    }
    if argument == "y_steps_ahead":
        forecast_kwargs["y_sources"] = sources
    else:
        forecast_kwargs["X_variables"] = [variables[1]]
        forecast_kwargs["X_sources"] = sources

    model = rt.RealTimeModel(data=forecast_data, models=rt.models.ForecastOLS())

    with pytest.raises(ValueError, match="range 0..1"):
        model.forecast(**forecast_kwargs)


@pytest.mark.parametrize("argument", ["y_steps_ahead", "X_steps_ahead"])
def test_conditioning_horizon_rejects_boolean(forecast_data, argument):
    """Booleans must not pass as integer conditioning horizons."""
    forecast_kwargs = {
        "y_variables": ["monthly_a"],
        "data_transformation": {"monthly_a": "levels", "monthly_b": "levels"},
        "frequency": "M",
        "steps": 2,
        "first_forecast_horizon": 0,
        "first_vintage": "2020-01-31",
        "last_vintage": "2020-01-31",
        argument: {"monthly_a" if argument == "y_steps_ahead" else "monthly_b": True},
    }
    if argument == "y_steps_ahead":
        forecast_kwargs["y_sources"] = {"monthly_a": "source"}
    else:
        forecast_kwargs["X_variables"] = ["monthly_b"]
        forecast_kwargs["X_sources"] = {"monthly_b": "source"}

    model = rt.RealTimeModel(data=forecast_data, models=rt.models.ForecastOLS())

    with pytest.raises(ValueError, match="range 0..1"):
        model.forecast(**forecast_kwargs)


def test_conditioning_horizon_steps_minus_one_covers_all_steps(forecast_data):
    """The largest valid zero-based horizon conditions the full forecast path."""
    vintage = pd.Timestamp("2020-01-31")
    conditioning = pd.DataFrame(
        {
            "date": pd.date_range(vintage, periods=2, freq="ME"),
            "variable": "monthly_a",
            "vintage_date": vintage,
            "source": "source",
            "value": [50.0, 60.0],
            "frequency": "M",
            "forecast_horizon": [0, 1],
        }
    )
    forecast_data.add_forecasts(conditioning, metric="levels")
    _SpyModel.captured_y = []
    _SpyModel.captured_X = []
    model = rt.RealTimeModel(data=forecast_data, models=_SpyModel())

    model.forecast(
        y_variables=["monthly_a"],
        y_steps_ahead={"monthly_a": 1},
        y_sources={"monthly_a": "source"},
        data_transformation={"monthly_a": "levels"},
        steps=2,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    seen_y = _SpyModel.captured_y[0]
    np.testing.assert_allclose(
        seen_y.loc[conditioning["date"], "monthly_a"], [50.0, 60.0]
    )


def test_sparse_conditioning_dates_remain_aligned_to_calendar(forecast_data):
    """A missing source month remains missing at its expected horizon."""
    vintage = pd.Timestamp("2020-01-31")
    conditioning = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-03-31"]),
            "variable": "monthly_a",
            "vintage_date": vintage,
            "source": "source",
            "value": [50.0, 70.0],
            "frequency": "M",
            "forecast_horizon": [0, 2],
        }
    )
    forecast_data.add_forecasts(conditioning, metric="levels")
    _SpyModel.captured_y = []
    _SpyModel.captured_X = []
    model = rt.RealTimeModel(data=forecast_data, models=_SpyModel())

    model.forecast(
        y_variables=["monthly_a"],
        y_steps_ahead={"monthly_a": 2},
        y_sources={"monthly_a": "source"},
        data_transformation={"monthly_a": "levels"},
        steps=3,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    seen_y = _SpyModel.captured_y[0]
    assert seen_y.loc[pd.Timestamp("2020-02-29"), "monthly_a"] != 70.0
    assert seen_y.loc[pd.Timestamp("2020-03-31"), "monthly_a"] == 70.0


def test_reconstruct_diff_uses_latest_prior_vintage_level_base():
    """A ``diff`` forecast is reconstructed even without a same-vintage level.

    The target's level series was last released at an earlier vintage; the
    reconstruction must use that earlier-vintage level as its base instead of
    silently skipping the variable.
    """
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "frequency": "M",
            "value": [100.0, 110.0],
            "metric": "levels",
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-03-31"]),
            "frequency": "M",
            "value": [5.0],
            "metric": "diff",
        }
    )

    result = DataTransformationPipeline({"gdp": "diff"}).reconstruct_levels(
        forecasts=forecasts,
        outturns=outturns,
        y_variables=["gdp"],
    )

    reconstructed = result.loc[
        (result["metric"] == "levels") & (result["date"] == pd.Timestamp("2020-03-31")),
        "value",
    ]

    assert len(reconstructed) == 1
    np.testing.assert_allclose(reconstructed.iloc[0], 115.0)


def test_reconstruct_log_diff_uses_latest_prior_vintage_level_base():
    """A ``log diff`` forecast is reconstructed even without a same-vintage level."""
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "frequency": "M",
            "value": [100.0, 110.0],
            "metric": "levels",
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-03-31"]),
            "frequency": "M",
            "value": [np.log(1.05)],
            "metric": "log diff",
        }
    )

    result = DataTransformationPipeline({"gdp": "log diff"}).reconstruct_levels(
        forecasts=forecasts,
        outturns=outturns,
        y_variables=["gdp"],
    )

    reconstructed = result.loc[
        (result["metric"] == "levels") & (result["date"] == pd.Timestamp("2020-03-31")),
        "value",
    ]

    assert len(reconstructed) == 1
    np.testing.assert_allclose(reconstructed.iloc[0], 115.5)


def test_reconstruct_levels_selects_latest_revision_among_prior_vintages():
    """The most recent pre-vintage level revision is used as the base.

    Three revisions of the same base date exist at different vintages, all
    at or before the vintage being processed. The rows are deliberately
    ordered with the latest revision first, so a fix that merely relaxes the
    equality filter to ``<=`` without deduplicating by vintage would pick the
    wrong (earliest) revision.
    """
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-29", "2020-02-29", "2020-02-29"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-03-10", "2020-02-29", "2020-02-15"]),
            "frequency": "M",
            "value": [112.0, 110.0, 108.0],
            "metric": "levels",
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-03-31"]),
            "frequency": "M",
            "value": [5.0],
            "metric": "diff",
        }
    )

    result = DataTransformationPipeline({"gdp": "diff"}).reconstruct_levels(
        forecasts=forecasts,
        outturns=outturns,
        y_variables=["gdp"],
    )

    reconstructed = result.loc[
        (result["metric"] == "levels") & (result["date"] == pd.Timestamp("2020-03-31")),
        "value",
    ]

    assert len(reconstructed) == 1
    np.testing.assert_allclose(reconstructed.iloc[0], 117.0)


def test_reconstruct_levels_excludes_future_vintage_revision():
    """A level revision released after the vintage being processed is ignored."""
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-29", "2020-02-29"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-02-29", "2020-04-15"]),
            "frequency": "M",
            "value": [110.0, 999.0],
            "metric": "levels",
        }
    )
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-03-31"]),
            "frequency": "M",
            "value": [5.0],
            "metric": "diff",
        }
    )

    result = DataTransformationPipeline({"gdp": "diff"}).reconstruct_levels(
        forecasts=forecasts,
        outturns=outturns,
        y_variables=["gdp"],
    )

    reconstructed = result.loc[
        (result["metric"] == "levels") & (result["date"] == pd.Timestamp("2020-03-31")),
        "value",
    ]

    assert len(reconstructed) == 1
    np.testing.assert_allclose(reconstructed.iloc[0], 115.0)


def test_realtime_ols_with_regressors(forecast_data):
    """RealTimeModel OLS with regressors matches a direct OLS fit on the same vintage."""
    y_variables = ["monthly_a"]
    X_variables = ["monthly_b", "monthly_c"]
    all_vars = y_variables + X_variables
    data_transformation = {var: "levels" for var in all_vars}
    steps = 4
    test_vintage = pd.Timestamp("2020-01-31")

    y_manual, X_manual = _get_vintage_data(
        forecast_data,
        test_vintage,
        y_variables,
        X_variables,
        last_observed_date=test_vintage,
    )

    # --- Run RealTimeModel ---
    ols_model = rt.models.ForecastOLS(drop_nans=True)
    rt_model = rt.RealTimeModel(data=forecast_data, models=ols_model)
    rt_model.forecast(
        y_variables=y_variables,
        X_variables=X_variables,
        data_transformation=data_transformation,
        steps=steps,
        first_vintage=str(test_vintage.date()),
        last_vintage=str(test_vintage.date()),
        X_imputation="zero",
    )

    rt_forecasts = rt_model.data.forecasts
    rt_ols = (
        rt_forecasts[
            (rt_forecasts["source"] == "ForecastOLS")
            & (rt_forecasts["vintage_date"] == test_vintage)
            & (rt_forecasts["variable"] == "monthly_a")
            & (rt_forecasts["metric"] == "levels")
        ]
        .sort_values("date")["value"]
        .values
    )

    # --- Reproduce manually ---
    manual_model = rt.models.ForecastOLS(drop_nans=True)
    manual_model.fit(
        y_manual,
        X=X_manual,
        data_transformation=data_transformation,
        X_imputation="zero",
    )

    X_forecast = pd.DataFrame(
        0.0,
        columns=X_variables,
        index=pd.date_range(
            start=test_vintage + pd.offsets.MonthEnd(1), periods=steps, freq="ME"
        ),
    )
    manual_forecasts = manual_model.forecast(steps=steps, X=X_forecast)

    # --- Compare ---
    np.testing.assert_allclose(rt_ols, manual_forecasts.values.ravel(), atol=1e-10)


def test_realtime_formula_selects_target_from_requested_variables(forecast_data):
    """A formula may select a subset of the requested target variables."""
    model = rt.models.ForecastOLS(
        label="formula_ols",
        formula="monthly_a ~ monthly_c",
    )
    rt_model = rt.RealTimeModel(data=forecast_data, models=model)
    rt_model.forecast(
        y_variables=["monthly_a", "monthly_b"],
        X_variables=["monthly_b", "monthly_c"],
        data_transformation={
            "monthly_a": "levels",
            "monthly_b": "levels",
            "monthly_c": "levels",
        },
        steps=1,
        first_vintage="2020-01-31",
        last_vintage="2020-01-31",
        X_imputation="zero",
    )

    forecasts = rt_model.data.forecasts.query("source == 'formula_ols'")
    assert set(forecasts["variable"]) == {"monthly_a"}


def test_realtime_formula_does_not_warn_for_unselected_targets(forecast_data):
    model = rt.models.ForecastOLS(
        label="formula_ols",
        formula="monthly_a ~ monthly_c",
    )
    rt_model = rt.RealTimeModel(data=forecast_data, models=model)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        rt_model.forecast(
            y_variables=["monthly_a", "monthly_b"],
            X_variables=["monthly_b", "monthly_c"],
            data_transformation={
                "monthly_a": "levels",
                "monthly_b": "levels",
                "monthly_c": "levels",
            },
            steps=1,
            first_vintage="2020-01-31",
            last_vintage="2020-01-31",
            X_imputation="zero",
        )

    assert not any(
        "following y_variables are not present" in str(warning.message)
        for warning in captured
    )


def test_realtime_ols_dummy_absorbs_outlier(sample_outturns):
    """RealTimeModel OLS with an outlier dummy matches a manual OLS fit.

    Injects a large outlier into ``monthly_a`` at a known in-sample date,
    then runs RealTimeModel with a point dummy on that date. The result must
    match a manual OLS fit using the same dummy, and the deterministic dummy
    column (named after its monthly period) must be part of the design.
    """
    y_variables = ["monthly_a"]
    X_variables = ["monthly_b", "monthly_c"]
    all_vars = y_variables + X_variables
    data_transformation = {var: "levels" for var in all_vars}
    steps = 4
    test_vintage = pd.Timestamp("2020-01-31")
    outlier_date = pd.Timestamp("2018-06-30")
    dummies = [outlier_date]

    # Inject a large outlier into monthly_a at a known in-sample date.
    outturns = sample_outturns.copy()
    mask = (outturns["variable"] == "monthly_a") & (outturns["date"] == outlier_date)
    assert mask.any(), "Outlier date must exist for monthly_a"
    outturns.loc[mask, "value"] = outturns.loc[mask, "value"] + 100.0

    # Build a ForecastData from the perturbed outturns (mirrors the fixture).
    fd = fe.ForecastData(outturns_data=outturns, metric="levels", compute_levels=False)

    # --- Run RealTimeModel with the dummy ---
    rt_model = rt.RealTimeModel(data=fd, models=rt.models.ForecastOLS())
    rt_model.forecast(
        y_variables=y_variables,
        X_variables=X_variables,
        data_transformation=data_transformation,
        steps=steps,
        first_vintage=str(test_vintage.date()),
        last_vintage=str(test_vintage.date()),
        first_forecast_horizon=0,
        dummies=dummies,
        X_imputation="zero",
    )

    rt_forecasts = rt_model.data.forecasts
    rt_ols = (
        rt_forecasts[
            (rt_forecasts["source"] == "ForecastOLS")
            & (rt_forecasts["vintage_date"] == test_vintage)
            & (rt_forecasts["variable"] == "monthly_a")
            & (rt_forecasts["metric"] == "levels")
        ]
        .sort_values("date")["value"]
        .values
    )
    assert len(rt_ols) == steps
    assert np.isfinite(rt_ols).all()

    # --- Reproduce manually with the same dummy ---
    y_manual, X_manual = _get_vintage_data(fd, test_vintage, y_variables, X_variables)

    manual_model = rt.models.ForecastOLS()
    manual_model.fit(y_manual, X=X_manual, dummies=dummies)

    # Monthly index -> period-style dummy name, appended as the last column.
    assert "D_2018M6" in manual_model.X.columns
    assert list(manual_model.X.columns)[-1] == "D_2018M6"

    X_forecast = pd.DataFrame(
        np.zeros((steps, len(X_variables))),
        columns=X_variables,
        index=pd.date_range(start=test_vintage, periods=steps, freq="ME"),
    )
    manual_forecasts = manual_model.forecast(steps=steps, X=X_forecast)

    # --- Compare ---
    np.testing.assert_allclose(rt_ols, manual_forecasts.values.ravel(), atol=1e-10)


def test_realtime_ols_decomposition_with_regressors(forecast_data):
    """Verify OLS decomposition collection and schema validation.

    Runs two consecutive vintages to test decomposition row collection.
    """
    y_variables = ["monthly_a"]
    X_variables = ["monthly_b", "monthly_c"]
    all_vars = y_variables + X_variables
    data_transformation = {var: "levels" for var in all_vars}
    steps = 2

    # Two consecutive vintages so forecasts from vintage 1 are revised in vintage 2
    first_vintage = pd.Timestamp("2020-01-31")
    second_vintage = pd.Timestamp("2020-02-29")

    ols_model = rt.models.ForecastOLS()
    rt_model = rt.RealTimeModel(data=forecast_data, models=ols_model)
    rt_model.forecast(
        y_variables=y_variables,
        X_variables=X_variables,
        data_transformation=data_transformation,
        steps=steps,
        first_vintage=str(first_vintage.date()),
        last_vintage=str(second_vintage.date()),
        decomp=True,
        X_imputation="zero",
    )

    # Verify decompositions exist and have expected structure
    assert rt_model.decompositions is not None
    assert isinstance(rt_model.decompositions, pd.DataFrame)

    decomp = rt_model.decompositions

    # Validate against schema
    decomposition_schema.validate(decomp)

    # Check that decompositions exist for both vintages
    assert len(decomp) > 0
    assert (decomp["source"] == "ForecastOLS").all()
    assert (decomp["variable"] == "monthly_a").all()

    # Check for level rows; revision decompositions are not yet implemented.
    level_rows = decomp[decomp["decomposition"] == "level"]

    assert len(level_rows) > 0, "Should have level decomposition rows"

    # Verify level rows have NaT for base_vintage_date and revision_source
    assert level_rows["base_vintage_date"].isna().all()
    assert level_rows["revision_source"].isna().all()

    # Verify contribution sum matches forecast for each horizon+vintage combo
    forecasts_df = forecast_data.forecasts[
        (forecast_data.forecasts["source"] == "OLS_decomp")
        & (forecast_data.forecasts["variable"] == "monthly_a")
        & (forecast_data.forecasts["metric"] == "levels")
    ]

    for vintage in [first_vintage, second_vintage]:
        vintage_level_decomp = level_rows[level_rows["vintage_date"] == vintage]
        for h in range(steps):
            decomp_h = vintage_level_decomp[vintage_level_decomp["forecast_horizon"] == h]
            if len(decomp_h) > 0:
                contrib_sum = decomp_h["contribution"].sum()
                # Find the corresponding forecast value
                fc_h = forecasts_df[
                    (forecasts_df["vintage_date"] == vintage)
                    & (forecasts_df["forecast_horizon"] == h)
                ]
                if len(fc_h) > 0:
                    forecast_val = fc_h["value"].iloc[0]
                    # Allow small numerical tolerance
                    np.testing.assert_allclose(
                        contrib_sum,
                        forecast_val,
                        atol=1e-6,
                        err_msg=f"Decomposition sum mismatch at {vintage}, h={h}",
                    )


def test_revision_counterfactual_does_not_mutate_fitted_model():
    """Counterfactual decomposition must not mutate the fitted model."""
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=dates)
    model = _NativeMetricConstantModel()
    model.fit(y, data_transformation={"target": "levels"}, frequency="M")
    forecast = model.forecast(steps=2, decomp=True)
    decomposition = forecast.decomposition
    forecast_origin = forecast.forecast_origin

    _level_contributions(
        model,
        y_history=y,
        X_history=None,
        y_conditioning=None,
        X_conditioning=None,
        forecast_origin=model.last_y_fit_date,
        steps=2,
        dates=forecast.index,
        data_transformation={"target": "levels"},
        frequency="M",
        X_imputation=None,
        y_variables=["target"],
    )

    assert model.forecast(steps=2, decomp=True).decomposition is not decomposition
    assert model.forecast(steps=2).forecast_origin == forecast_origin
    pd.testing.assert_frame_equal(model._raw_y_history, y)


class _NativeMetricConstantModel(ForecastModel):
    """Deterministic single-target model with a one-component decomposition,
    for asserting decomposition metadata describes the model's own
    (possibly transformed) output space."""

    def _fit(self, y, X=None, **kwargs):
        self.level = float(y.iloc[-1, 0])
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        return np.full((steps, 1), self.level)

    def _forecast_decomp(self, steps, X=None, y=None, **kwargs):
        return pd.DataFrame(
            {
                "forecast_horizon": range(steps),
                "component": "level",
                "contribution": self.level,
                "weight": np.nan,
            }
        )


def test_realtime_decomposition_reports_native_transformed_metric(forecast_data):
    """Decomposition metadata must describe the model's own transformed
    output space (e.g. 'diff'), never silently claim 'levels', and must stay
    identical whether or not levels reconstruction of the point forecast is
    requested."""
    y_variables = ["monthly_a"]
    data_transformation = {"monthly_a": "diff"}
    vintage = pd.Timestamp("2020-01-31")

    def _run(reconstruct_levels):
        rt_model = rt.RealTimeModel(
            data=copy.deepcopy(forecast_data), models=_NativeMetricConstantModel()
        )
        rt_model.forecast(
            y_variables=y_variables,
            data_transformation=data_transformation,
            steps=2,
            first_vintage=str(vintage.date()),
            last_vintage=str(vintage.date()),
            decomp=True,
            reconstruct_levels=reconstruct_levels,
        )
        return rt_model.decompositions

    decomp_reconstructed = _run(True)
    decomp_native = _run(False)

    decomposition_schema.validate(decomp_reconstructed)
    decomposition_schema.validate(decomp_native)

    assert (decomp_reconstructed["forecast_metric"] == "diff").all()
    pd.testing.assert_frame_equal(
        decomp_reconstructed.reset_index(drop=True),
        decomp_native.reset_index(drop=True),
    )


class _TwoTargetMeanModel(ForecastModel):
    """Deterministic multi-target model: per-target mean, with a component
    named identically across targets to exercise decomposition
    target-identity handling."""

    def _fit(self, y, X=None, **kwargs):
        self.means = y.mean()
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        return np.tile(self.means.to_numpy(), (steps, 1))

    def _forecast_decomp(self, steps, X=None, y=None, **kwargs):
        rows = []
        for variable, mean in self.means.items():
            for h in range(steps):
                rows.append(
                    {
                        "forecast_horizon": h,
                        "component": "mean",
                        "contribution": float(mean),
                        "weight": np.nan,
                        "variable": variable,
                    }
                )
        return pd.DataFrame(rows)


class _TwoTargetNoVariableModel(_TwoTargetMeanModel):
    """Multi-target model whose ``_forecast_decomp`` omits ``variable``."""

    def _forecast_decomp(self, steps, X=None, y=None, **kwargs):
        rows = super()._forecast_decomp(steps=steps, X=X, y=y, **kwargs)
        return rows.drop(columns=["variable"])


def test_realtime_decomposition_multi_target_identity(forecast_data):
    """Decomposition rows for a multi-target model must be attributed to
    their own target variable and its own transformation metric."""
    y_variables = ["monthly_a", "monthly_b"]
    data_transformation = {"monthly_a": "levels", "monthly_b": "diff"}
    vintage = pd.Timestamp("2020-01-31")

    rt_model = rt.RealTimeModel(data=forecast_data, models=_TwoTargetMeanModel())
    rt_model.forecast(
        y_variables=y_variables,
        data_transformation=data_transformation,
        steps=2,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        decomp=True,
    )

    decomp = rt_model.decompositions
    decomposition_schema.validate(decomp)

    assert set(decomp["variable"]) == set(y_variables)

    metric_by_variable = decomp.drop_duplicates("variable").set_index("variable")[
        "forecast_metric"
    ]
    assert metric_by_variable["monthly_a"] == "levels"
    assert metric_by_variable["monthly_b"] == "diff"

    # Each variable keeps its own contribution, never the other's.
    contribution_a = decomp.loc[
        decomp["variable"] == "monthly_a", "contribution"
    ].unique()
    contribution_b = decomp.loc[
        decomp["variable"] == "monthly_b", "contribution"
    ].unique()
    assert len(contribution_a) == 1
    assert len(contribution_b) == 1
    assert contribution_a[0] != contribution_b[0]


def test_realtime_decomposition_multi_target_requires_variable_column(forecast_data):
    """A multi-target model's decomposition must self-identify its target
    variable per row; RealTimeModel cannot infer it and must raise clearly
    rather than silently emitting an invalid/ambiguous row."""
    y_variables = ["monthly_a", "monthly_b"]
    vintage = pd.Timestamp("2020-01-31")

    rt_model = rt.RealTimeModel(data=forecast_data, models=_TwoTargetNoVariableModel())
    with pytest.raises(ValueError, match="variable"):
        rt_model.forecast(
            y_variables=y_variables,
            data_transformation={"monthly_a": "levels", "monthly_b": "levels"},
            steps=2,
            first_vintage=str(vintage.date()),
            last_vintage=str(vintage.date()),
            decomp=True,
        )


def test_realtime_decomposition_revision_metadata_multi_target(forecast_data):
    """Revision decomposition must not conflate components across different
    target variables that happen to share a component name."""
    y_variables = ["monthly_a", "monthly_b"]
    data_transformation = {"monthly_a": "levels", "monthly_b": "levels"}
    first_vintage = pd.Timestamp("2020-01-31")
    second_vintage = pd.Timestamp("2020-02-29")

    rt_model = rt.RealTimeModel(data=forecast_data, models=_TwoTargetMeanModel())
    rt_model.forecast(
        y_variables=y_variables,
        data_transformation=data_transformation,
        steps=2,
        first_vintage=str(first_vintage.date()),
        last_vintage=str(second_vintage.date()),
        decomp=True,
    )

    decomp = rt_model.decompositions
    decomposition_schema.validate(decomp)

    revision_rows = decomp[decomp["decomposition"] == "revision"]
    assert len(revision_rows) > 0

    # Exactly one revision row per (vintage transition, date, component,
    # revision_source, variable); a merge that drops "variable" as a join
    # key would produce a cross-product duplicating these rows across both
    # target variables.
    counts = revision_rows.groupby(
        [
            "vintage_date",
            "base_vintage_date",
            "date",
            "component",
            "revision_source",
            "variable",
        ]
    ).size()
    assert (counts == 1).all()

    for variable in y_variables:
        rows = revision_rows[revision_rows["variable"] == variable]
        assert (rows["forecast_metric"] == data_transformation[variable]).all()


def test_first_forecast_horizon_one(forecast_data):
    """With first_forecast_horizon=0, current-period data is included in fitting."""
    y_variables = ["monthly_b"]
    steps = 2
    test_vintage = pd.Timestamp("2020-01-31")

    ols_model = rt.models.ForecastOLS()
    rt_model = rt.RealTimeModel(data=forecast_data, models=ols_model)
    rt_model.forecast(
        y_variables=y_variables,
        X_variables=["monthly_a"],
        data_transformation={"monthly_a": "levels", "monthly_b": "levels"},
        steps=steps,
        first_forecast_horizon=0,
        first_vintage=str(test_vintage.date()),
        last_vintage=str(test_vintage.date()),
        X_imputation="zero",
    )

    rt_forecasts = rt_model.data.forecasts
    rt_ols = rt_forecasts[
        (rt_forecasts["source"] == "ForecastOLS")
        & (rt_forecasts["vintage_date"] == test_vintage)
        & (rt_forecasts["variable"] == "monthly_b")
        & (rt_forecasts["metric"] == "levels")
    ].sort_values("date")

    # Reproduce manually: include current period (index <= vintage)
    outturns = forecast_data.outturns.copy()
    outturns = outturns[
        (outturns["variable"].isin(["monthly_a", "monthly_b"]))
        & (outturns["metric"] == "levels")
    ]
    at_vintage = outturns[outturns["vintage_date"] <= test_vintage].copy()
    at_vintage = at_vintage.sort_values("vintage_date", ascending=False).drop_duplicates(
        subset=["date", "variable"], keep="first"
    )
    y_wide = at_vintage.pivot(index="date", columns="variable", values="value")
    y_wide = y_wide[y_wide.index <= test_vintage]  # include current period

    manual_model = rt.models.ForecastOLS()

    # The real-time loop fits on the target's own observed dates only (its
    # pivot never manufactures a NaN row from the regressor's later date), so
    # replicate that here rather than fitting on the combined pivot's
    # artificial NaN row for the current period.
    manual_model.fit(y_wide[["monthly_b"]].dropna(), X=y_wide[["monthly_a"]])
    manual_forecasts = manual_model.forecast(steps=1, X=y_wide[["monthly_a"]])

    np.testing.assert_allclose(
        rt_ols["value"].values[0], manual_forecasts.values.ravel(), atol=1e-10
    )


def test_forecast_horizon_is_relative_to_last_observation():
    """Output horizons use the final fitted observation as their base."""

    class ConstantModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            self.constant = float(y.iloc[-1, 0])
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.full((steps, 1), self.constant)

    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=ConstantModel())

    model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "levels"},
        steps=2,
        first_forecast_horizon=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    forecasts = model.data.forecasts.loc[
        lambda df: (
            (df["source"] == "ConstantModel")
            & (df["variable"] == "gdp")
            & (df["metric"] == "levels")
        ),
        ["date", "forecast_horizon"],
    ].sort_values("date")

    assert forecasts["date"].tolist() == [
        pd.Timestamp("2020-04-30"),
        pd.Timestamp("2020-05-31"),
    ]
    assert forecasts["forecast_horizon"].tolist() == [0, 1]


def test_default_first_forecast_horizon_uses_available_target_data():
    """The default starts at the first period after the final fitted target."""

    class ConstantModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            self.constant = float(y.iloc[-1, 0])
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.full((steps, 1), self.constant)

    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "variable": ["gdp", "gdp"],
            "vintage_date": [vintage, vintage],
            "frequency": "M",
            "value": [100.0, 110.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=ConstantModel())

    model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "levels"},
        steps=2,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    forecasts = model.data.forecasts.loc[
        lambda df: (
            (df["source"] == "ConstantModel")
            & (df["variable"] == "gdp")
            & (df["metric"] == "levels")
        ),
        ["date", "forecast_horizon"],
    ].sort_values("date")

    assert forecasts["date"].tolist() == [
        pd.Timestamp("2020-03-31"),
        pd.Timestamp("2020-04-30"),
    ]
    assert forecasts["forecast_horizon"].tolist() == [0, 1]


def test_first_forecast_horizon_supports_early_estimates():
    """An explicit early start retains first estimates before the vintage period."""

    class ConstantModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            self.constant = float(y.iloc[-1, 0])
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.full((steps, 1), self.constant)

    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2019-11-30",
                    "2019-12-31",
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",
                ]
            ),
            "variable": ["gdp", "gdp", "gdp", "gdp", "gdp"],
            "vintage_date": [vintage, vintage, vintage, vintage, vintage],
            "frequency": "M",
            "value": [80.0, 90.0, 100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=ConstantModel())

    model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "levels"},
        steps=2,
        first_forecast_horizon=-1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    forecasts = model.data.forecasts.loc[
        lambda df: (
            (df["source"] == "ConstantModel")
            & (df["variable"] == "gdp")
            & (df["metric"] == "levels")
        ),
        ["date", "forecast_horizon"],
    ].sort_values("date")

    assert forecasts["date"].tolist() == [
        pd.Timestamp("2020-02-29"),
        pd.Timestamp("2020-03-31"),
    ]
    assert forecasts["forecast_horizon"].tolist() == [0, 1]


def test_first_forecast_horizon_dict_filters_vintage_relative_targets():
    """Per-variable cutoffs do not change the emitted information horizon."""

    class MultiConstantModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            self.values = y.iloc[-1].to_numpy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.tile(self.values, (steps, 1))

    vintage = pd.Timestamp("2020-03-31")
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    outturns = pd.DataFrame(
        {
            "date": dates.tolist() * 2,
            "variable": ["gdp_a"] * 3 + ["gdp_b"] * 3,
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0, 200.0, 210.0, 221.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=MultiConstantModel())

    model.forecast(
        y_variables=["gdp_a", "gdp_b"],
        data_transformation={"gdp_a": "levels", "gdp_b": "levels"},
        steps=2,
        first_forecast_horizon={"gdp_a": 0, "gdp_b": 1},
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    forecasts = model.data.forecasts.loc[
        lambda df: (df["source"] == "MultiConstantModel") & (df["metric"] == "levels"),
        ["variable", "date", "forecast_horizon"],
    ].sort_values(["variable", "date"])

    assert forecasts.loc[
        forecasts["variable"] == "gdp_a", "forecast_horizon"
    ].tolist() == [0, 1]
    assert forecasts.loc[
        forecasts["variable"] == "gdp_b", "forecast_horizon"
    ].tolist() == [1]
    assert forecasts.loc[forecasts["variable"] == "gdp_b", "date"].tolist() == [
        pd.Timestamp("2020-04-30")
    ]


@pytest.mark.parametrize(
    ("first_forecast_horizon", "error", "message"),
    [
        ({"monthly_a": 0, "monthly_b": 1, "extra": 2}, ValueError, "match"),
        ({"monthly_a": 0, "monthly_b": True}, TypeError, "non-boolean integers"),
        ({"monthly_a": 0, "monthly_b": 1.5}, TypeError, "non-boolean integers"),
    ],
)
def test_first_forecast_horizon_dict_validates_exact_integer_values(
    forecast_data, first_forecast_horizon, error, message
):
    model = rt.RealTimeModel(data=forecast_data, models=_RequiredForecastOptionModel())

    with pytest.raises(error, match=message):
        model.forecast(
            y_variables=["monthly_a", "monthly_b"],
            steps=1,
            first_forecast_horizon=first_forecast_horizon,
        )


def test_impute_forecast_X_diverging_series_lengths():
    """Each column is padded/trimmed according to its own shortage.

    Regression test: previously, columns with a different number of observed
    future values relative to each other could be padded/trimmed using the
    wrong length, corrupting columns that did not need it. Here three columns
    each require a different treatment:
    - ``col_short`` : missing 2 future values -> padded.
    - ``col_exact`` : already has exactly the required number of values -> untouched.
    - ``col_surplus`` : has one extra value beyond what's required -> trimmed.
    """
    last_y_fit_date = pd.Timestamp("2020-01-31")
    steps = 3  # required last forecast date: 2020-04-30

    index = pd.date_range(start="2019-10-31", periods=8, freq="ME")  # Oct19..May20

    col_short = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, np.nan, np.nan, np.nan], index=index
    )  # valid through Feb20 -> shortage of 2
    col_exact = pd.Series(
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, np.nan], index=index
    )  # valid through Apr20 -> shortage of 0
    col_surplus = pd.Series(
        [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0], index=index
    )  # valid through May20 -> one value too many, needs trimming

    X_forecast = pd.DataFrame(
        {
            "col_short": col_short,
            "col_exact": col_exact,
            "col_surplus": col_surplus,
        }
    )

    result = impute_X(
        X_forecast,
        last_y_fit_date,
        steps,
        method="last",
        frequencies={column: "M" for column in X_forecast},
    )

    expected_index = pd.date_range(start="2019-10-31", end="2020-04-30", freq="ME")
    pd.testing.assert_index_equal(result.index, expected_index)

    # col_short: original 5 values, padded with 2 copies of the last value (5.0)
    np.testing.assert_allclose(
        result["col_short"].values, [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0]
    )

    # col_exact: untouched, no padding or trimming
    np.testing.assert_allclose(
        result["col_exact"].values, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    )

    # col_surplus: the extra (May20) value is trimmed, other columns unaffected
    np.testing.assert_allclose(
        result["col_surplus"].values, [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0]
    )


def test_impute_X_mixed_monthly_quarterly_columns_pad_independently():
    """Each column is padded on its own frequency, not the wide index's frequency.

    Regression test: a quarterly column mixed with a monthly column in the
    same wide ``X`` used to be padded using the frequency inferred from the
    combined (monthly-granularity) index, producing fake monthly fill values
    between its true quarter-ends. Each column must instead reach its own
    per-column target date using its own frequency.
    """
    last_date = pd.Timestamp("2020-04-30")
    steps = 0

    monthly = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range("2019-09-30", periods=4, freq="ME"),
    )
    quarterly = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.date_range("2019-09-30", periods=3, freq="QE-DEC"),
    )
    X = pd.concat([monthly.rename("monthly"), quarterly.rename("quarterly")], axis=1)

    result = impute_X(
        X,
        last_date,
        steps,
        method="last",
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    monthly_dates = result["monthly"].dropna().index
    expected_monthly_dates = pd.date_range("2019-09-30", "2020-04-30", freq="ME")
    pd.testing.assert_index_equal(monthly_dates, expected_monthly_dates)

    quarterly_dates = result["quarterly"].dropna().index
    expected_quarterly_dates = pd.to_datetime(
        ["2019-09-30", "2019-12-31", "2020-03-31", "2020-06-30"]
    )
    pd.testing.assert_index_equal(quarterly_dates, expected_quarterly_dates)


def test_impute_X_mixed_frequency_steps_advance_correctly_per_column():
    """With a forecast horizon, each column's own frequency offset is used to
    advance its target date, not the wide index's frequency."""
    last_date = pd.Timestamp("2020-01-31")
    steps = 2

    monthly = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range("2019-10-31", periods=4, freq="ME"),
    )
    quarterly = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.date_range("2019-06-30", periods=3, freq="QE-DEC"),
    )
    X = pd.concat(
        [monthly.rename("monthly"), quarterly.rename("quarterly")], axis=1, sort=False
    )

    result = impute_X(
        X,
        last_date,
        steps,
        method="last",
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    assert result["monthly"].dropna().index[-1] == pd.Timestamp("2020-03-31")
    assert result["quarterly"].dropna().index[-1] == pd.Timestamp("2020-06-30")


def test_impute_X_mixed_frequency_trims_surplus_independently():
    """A quarterly column with surplus data is trimmed to its own quarterly
    target, independent of what the monthly column's own target is."""
    last_date = pd.Timestamp("2020-01-31")
    steps = 0

    monthly = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range("2019-10-31", periods=4, freq="ME"),
    )
    quarterly = pd.Series(
        [10.0, 20.0, 30.0, 40.0, 50.0],
        index=pd.date_range("2019-06-30", periods=5, freq="QE-DEC"),
    )
    X = pd.concat(
        [monthly.rename("monthly"), quarterly.rename("quarterly")], axis=1, sort=False
    )

    result = impute_X(
        X,
        last_date,
        steps,
        method="last",
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    quarterly_dates = result["quarterly"].dropna().index
    expected_quarterly_dates = pd.to_datetime(
        ["2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31"]
    )
    pd.testing.assert_index_equal(quarterly_dates, expected_quarterly_dates)

    monthly_dates = result["monthly"].dropna().index
    expected_monthly_dates = pd.date_range("2019-10-31", "2020-01-31", freq="ME")
    pd.testing.assert_index_equal(monthly_dates, expected_monthly_dates)


@pytest.mark.parametrize("method", ["zero", "last", "mean", "ar1_t"])
def test_impute_X_mixed_frequency_all_methods_avoid_error(method):
    """Each supported imputation method pads mixed-frequency columns onto
    their own frequency without error, leaving quarterly padding only on
    quarter-end dates."""
    last_date = pd.Timestamp("2020-01-31")
    steps = 1

    monthly = pd.Series(
        np.arange(1.0, 9.0),
        index=pd.date_range("2019-06-30", periods=8, freq="ME"),
    )
    quarterly = pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.date_range("2019-06-30", periods=3, freq="QE-DEC"),
    )
    X = pd.concat([monthly.rename("monthly"), quarterly.rename("quarterly")], axis=1)

    result = impute_X(
        X,
        last_date,
        steps,
        method=method,
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    monthly_dates = result["monthly"].dropna().index
    expected_monthly_dates = pd.date_range("2019-06-30", "2020-02-29", freq="ME")
    pd.testing.assert_index_equal(monthly_dates, expected_monthly_dates)
    assert np.isfinite(result["monthly"].dropna()).all()

    quarterly_dates = result["quarterly"].dropna().index
    expected_quarterly_dates = pd.to_datetime(
        ["2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31"]
    )
    pd.testing.assert_index_equal(quarterly_dates, expected_quarterly_dates)
    assert np.isfinite(result["quarterly"].dropna()).all()


@pytest.mark.parametrize(
    "optimiser_result",
    [
        type("FailedResult", (), {"success": False, "x": [1.0, 2.0, 3.0, 4.0]})(),
        type(
            "NonFiniteResult",
            (),
            {"success": True, "x": [np.nan, 2.0, 3.0, 4.0]},
        )(),
    ],
)
def test_ar1_t_uses_finite_ols_fallback_for_invalid_optimiser(
    monkeypatch, optimiser_result
):
    monkeypatch.setattr(
        "forecast_realtime._utils.minimize", lambda *args, **kwargs: optimiser_result
    )
    observed = pd.Series(np.arange(1.0, 9.0))

    fill = _ar1_t_impute(observed, shortage=4, rng=np.random.default_rng(0))

    assert len(fill) == 4
    assert np.isfinite(fill).all()


def test_ar1_t_constant_and_short_series_repeat_last_value():
    rng = np.random.default_rng(0)

    assert _ar1_t_impute([1.0] * 8, 3, rng) == [1.0, 1.0, 1.0]
    assert _ar1_t_impute([1.0, 2.0, 3.0], 3, rng) == [3.0, 3.0, 3.0]


def test_impute_X_preserves_internal_nan_from_original_index():
    """A genuine internal gap (mid-series NaN, strictly between a column's
    own first and last valid date) must survive imputation unchanged.

    This is distinct from a *surplus* date beyond a column's own required
    target (covered by ``test_impute_forecast_X_diverging_series_lengths``,
    where such a trimmed-away date must NOT reappear as an all-NaN row).
    """
    last_date = pd.Timestamp("2020-02-29")
    steps = 0

    # col_a has a genuine internal gap at 2019-12-31, strictly between its
    # own first (2019-10-31) and last (2020-02-29) valid dates.
    col_a = pd.Series(
        [1.0, 2.0, np.nan, 4.0, 5.0],
        index=pd.date_range("2019-10-31", periods=5, freq="ME"),
    )
    X = col_a.rename("col_a").to_frame()

    result = impute_X(
        X,
        last_date,
        steps,
        method="last",
        frequencies={"col_a": "M"},
    )

    internal_gap_date = pd.Timestamp("2019-12-31")
    assert internal_gap_date in result.index
    assert np.isnan(result.loc[internal_gap_date, "col_a"])
    np.testing.assert_allclose(result["col_a"].dropna().values, [1.0, 2.0, 4.0, 5.0])


def test_impute_X_rejects_all_missing_columns():
    """All-missing regressors are rejected rather than silently removed."""
    last_date = pd.Timestamp("2020-02-29")
    index = pd.date_range("2019-10-31", periods=5, freq="ME")
    X = pd.DataFrame(
        {
            "usable": [1.0, 2.0, 3.0, np.nan, np.nan],
            "missing": [np.nan] * len(index),
        },
        index=index,
    )

    with pytest.raises(ValueError, match="missing"):
        impute_X(X, last_date, method="last", frequencies={"usable": "M"})


def test_impute_X_rejects_entirely_all_missing_design():
    """An entirely all-missing design is rejected during imputation."""
    index = pd.date_range("2019-10-31", periods=3, freq="ME")
    X = pd.DataFrame({"missing": [np.nan] * len(index)}, index=index)

    with pytest.raises(ValueError, match="missing"):
        impute_X(
            X,
            pd.Timestamp("2020-01-31"),
            method="zero",
            frequencies={"missing": "M"},
        )


def test_impute_X_uses_supplied_frequency_with_two_observations():
    """A supplied frequency controls imputation even with two observations."""
    last_date = pd.Timestamp("2020-01-31")
    steps = 1
    X = pd.DataFrame(
        {"only_two": [1.0, 2.0]},
        index=pd.to_datetime(["2019-12-31", "2020-01-31"]),
    )

    result = impute_X(
        X,
        last_date,
        steps,
        method="last",
        frequencies={"only_two": "M"},
    )

    expected_dates = pd.to_datetime(["2019-12-31", "2020-01-31", "2020-02-29"])
    pd.testing.assert_index_equal(result["only_two"].dropna().index, expected_dates)
    np.testing.assert_allclose(result["only_two"].dropna().values, [1.0, 2.0, 2.0])


def test_impute_X_requires_explicit_frequency_mapping():
    last_date = pd.Timestamp("2020-01-31")
    X = pd.DataFrame({"only_one": [1.0]}, index=pd.to_datetime(["2019-12-31"]))

    with pytest.raises(TypeError, match="frequencies"):
        impute_X(X, last_date, steps=0, method="last")


def test_models_not_needing_imputation_skip_realtime_imputation():
    """A model flagged ``_needs_ragged_edge_imputation = False`` must receive the raw,
    un-imputed X at both fit and forecast time, even when ``X_imputation`` is
    requested.

    ``reg2`` is genuinely ragged (no March observation), so pivoting it
    alongside the longer ``reg1`` naturally leaves a NaN at March. Without the
    opt-out this NaN would be papered over by ``impute_X``'s "last" method
    (``reg2`` has three evenly-spaced prior points so its month-end frequency
    is inferred cleanly, landing the fabricated fill exactly on 31 March);
    with the opt-out, the raw NaN must survive unchanged.
    """

    # RealTimeModel deep-copies the model per vintage, so captured X must be
    # recorded outside the instance (a closure variable, not an attribute) to
    # survive the copy and remain visible to the test.
    captured = {"fit_X": None, "forecast_X": None}

    class RaggednessStub(ForecastModel):
        _needs_ragged_edge_imputation = False

        def _fit(self, y, X=None, **kwargs):
            captured["fit_X"] = X
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            captured["forecast_X"] = X
            return pd.DataFrame(
                np.zeros((steps, 1)),
                columns=y.columns,
                index=pd.date_range(
                    start=y.index[-1] + pd.offsets.MonthEnd(1),
                    periods=steps,
                    freq="ME",
                ),
            )

    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",  # gdp
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",  # reg1 (full history)
                    "2019-12-31",
                    "2020-01-31",
                    "2020-02-29",  # reg2 (ragged - no March)
                ]
            ),
            "variable": ["gdp"] * 3 + ["reg1"] * 3 + ["reg2"] * 3,
            "vintage_date": vintage,
            "frequency": "M",
            "value": [10.0, 20.0, 30.0, 100.0, 200.0, 300.0, 0.5, 1.0, 2.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, metric="levels", compute_levels=False)

    stub = RaggednessStub()
    rt_model = rt.RealTimeModel(data=data, models=stub)
    rt_model.forecast(
        y_variables=["gdp"],
        X_variables=["reg1", "reg2"],
        data_transformation={"gdp": "levels", "reg1": "levels", "reg2": "levels"},
        steps=1,
        first_forecast_horizon={"gdp": 0},
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        X_imputation="last",
    )

    assert captured["fit_X"] is not None
    assert pd.isna(captured["fit_X"].loc[pd.Timestamp("2020-03-31"), "reg2"])
    assert captured["forecast_X"] is not None
    assert pd.isna(captured["forecast_X"].loc[pd.Timestamp("2020-03-31"), "reg2"])


@pytest.mark.parametrize("forecast_strategy", ["recursive", "direct"])
@pytest.mark.parametrize("method", ["zero", "last", "mean", "ar1_t"])
def test_realtime_ols_X_imputation_methods_avoid_error(
    forecast_data, method, forecast_strategy
):
    """Each supported X_imputation strategy fills the missing future regressor
    rows, so forecasting succeeds instead of hitting the availability error,
    for both the recursive and direct forecasting strategies."""
    y_variables = ["monthly_a"]
    X_variables = ["monthly_b", "monthly_c"]
    data_transformation = {var: "levels" for var in y_variables + X_variables}
    steps = 4
    test_vintage = pd.Timestamp("2020-01-31")

    if forecast_strategy == "direct":
        ols_model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    else:
        ols_model = rt.models.ForecastOLS()
    rt_model = rt.RealTimeModel(data=forecast_data, models=ols_model)
    rt_model.forecast(
        y_variables=y_variables,
        X_variables=X_variables,
        data_transformation=data_transformation,
        steps=steps,
        first_vintage=str(test_vintage.date()),
        last_vintage=str(test_vintage.date()),
        first_forecast_horizon=0,
        X_imputation=method,
    )

    rt_forecasts = rt_model.data.forecasts
    rt_ols = rt_forecasts[
        (rt_forecasts["source"] == "ForecastOLS")
        & (rt_forecasts["vintage_date"] == test_vintage)
        & (rt_forecasts["variable"] == "monthly_a")
        & (rt_forecasts["metric"] == "levels")
    ]
    assert len(rt_ols) == steps
    assert np.isfinite(rt_ols["value"]).all()


@pytest.mark.parametrize("forecast_strategy", ["recursive", "direct"])
def test_realtime_ols_X_steps_ahead_conditioning_avoids_error(
    forecast_data, forecast_strategy
):
    """Regressor-conditioning forecasts (X_steps_ahead/X_sources) can cover the
    forecast horizon by themselves, without needing X_imputation at all, for
    both the recursive and direct forecasting strategies."""
    y_variables = ["monthly_a"]
    X_variables = ["monthly_b"]
    data_transformation = {"monthly_a": "levels", "monthly_b": "levels"}
    steps = 1
    test_vintage = pd.Timestamp("2020-01-31")

    # A conditioning forecast for monthly_b, "contemporaneously available" at
    # test_vintage, covering every plausible first-forecast date so the test
    # doesn't depend on the exact horizon-0 convention.
    conditioning = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "monthly_b",
            "vintage_date": test_vintage,
            "source": "nowcast_source",
            "value": 1.23,
            "frequency": "M",
            "forecast_horizon": [0, 1, 2],
        }
    )
    forecast_data.add_forecasts(conditioning, metric="levels")

    if forecast_strategy == "direct":
        ols_model = rt.models.ForecastOLS(forecast_strategy="direct", steps=steps)
    else:
        ols_model = rt.models.ForecastOLS()
    rt_model = rt.RealTimeModel(data=forecast_data, models=ols_model)

    # No X_imputation: the conditioning path alone must cover the horizon.
    rt_model.forecast(
        y_variables=y_variables,
        X_variables=X_variables,
        X_steps_ahead={"monthly_b": 0},
        X_sources={"monthly_b": "nowcast_source"},
        data_transformation=data_transformation,
        steps=steps,
        first_vintage=str(test_vintage.date()),
        last_vintage=str(test_vintage.date()),
        first_forecast_horizon=0,
    )

    rt_forecasts = rt_model.data.forecasts
    rt_ols = rt_forecasts[
        (rt_forecasts["source"] == "ForecastOLS")
        & (rt_forecasts["vintage_date"] == test_vintage)
        & (rt_forecasts["variable"] == "monthly_a")
        & (rt_forecasts["metric"] == "levels")
    ]
    assert len(rt_ols) == steps
    assert np.isfinite(rt_ols["value"]).all()


# ============================================================================
# Forecast-source filtering (regression coverage for the row-wise mask in
# RealTimeModel.forecast() and the DataTransformationPipeline.filter()
# helper, ahead of vectorising both).
# ============================================================================


class _SpyModel(ForecastModel):
    """Records the y/X it receives so tests can inspect what reached it.

    ``RealTimeModel`` fits a fresh ``copy.deepcopy`` of the model per vintage,
    so an instance attribute set inside ``_forecast`` would be lost on the
    copy; class-level lists survive since the class itself is not copied.
    """

    captured_y: list = []
    captured_X: list = []
    captured_fit_y: list = []

    def _fit(self, y, X=None, **kwargs):
        _SpyModel.captured_fit_y.append(y.copy())
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        _SpyModel.captured_y.append(y.copy() if y is not None else None)
        _SpyModel.captured_X.append(X.copy() if X is not None else None)
        # No DatetimeIndex: _wrap_forecast labels the next `steps` periods
        # after the last fitted observation, regardless of whether y/X carry
        # conditioned future rows.
        return pd.DataFrame({"target": [0.0] * steps})


@pytest.mark.parametrize(
    "conditioning_source, expect_conditioning_applied",
    [("nowcast", True), ("other_source", False)],
)
def test_y_conditioning_forecast_source_filter(
    conditioning_source, expect_conditioning_applied
):
    """``y_sources`` only lets matching-source conditioning values reach the
    model; a mismatched source is filtered out and conditioning is skipped."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "target",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [10.0, 11.0, 12.0],
            "metric": "levels",
        }
    )
    conditioning = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-04-30"]),
            "variable": "target",
            "vintage_date": vintage,
            "source": conditioning_source,
            "frequency": "M",
            "value": [50.0, 60.0],
            "metric": "levels",
            "forecast_horizon": [0, 1],
        }
    )
    data = fe.NowcastData(
        outturns_data=outturns,
        forecasts_data=conditioning,
        compute_levels=False,
        data_check=False,
    )
    _SpyModel.captured_y = []
    _SpyModel.captured_X = []
    model = rt.RealTimeModel(data=data, models=_SpyModel())

    model.forecast(
        y_variables=["target"],
        y_steps_ahead={"target": 1},
        y_sources={"target": "nowcast"},
        data_transformation={"target": "levels"},
        steps=2,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    seen_y = _SpyModel.captured_y[0]
    conditioning_reached_model = seen_y["target"].isin([50.0, 60.0]).any()
    assert conditioning_reached_model == expect_conditioning_applied


@pytest.mark.parametrize(
    "conditioning_source, expect_conditioning_applied",
    [("nowcast_source", True), ("other_source", False)],
)
def test_X_conditioning_forecast_source_filter(
    conditioning_source, expect_conditioning_applied
):
    """``X_sources`` only lets matching-source regressor forecasts reach the
    model; a mismatched source is filtered out and conditioning is skipped."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"] * 2),
            "variable": ["target"] * 3 + ["driver"] * 3,
            "vintage_date": vintage,
            "frequency": "M",
            "value": [10.0, 11.0, 12.0, 1.0, 2.0, 3.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, metric="levels", compute_levels=False)
    conditioning = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-04-30"]),
            "variable": "driver",
            "vintage_date": vintage,
            "source": conditioning_source,
            "frequency": "M",
            "value": [500.0, 600.0],
            "forecast_horizon": [0, 1],
        }
    )
    data.add_forecasts(conditioning, metric="levels")

    _SpyModel.captured_y = []
    _SpyModel.captured_X = []
    model = rt.RealTimeModel(data=data, models=_SpyModel())

    model.forecast(
        y_variables=["target"],
        X_variables=["driver"],
        X_steps_ahead={"driver": 1},
        X_sources={"driver": "nowcast_source"},
        data_transformation={"target": "levels", "driver": "levels"},
        steps=2,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    seen_X = _SpyModel.captured_X[0]
    conditioning_reached_model = seen_X["driver"].isin([500.0, 600.0]).any()
    assert conditioning_reached_model == expect_conditioning_applied


def test_mixed_frequency_X_conditioning_keeps_months_for_quarterly_target():
    """Quarterly forecasts retain all monthly X values in a conditioned quarter."""
    vintage = pd.Timestamp("2020-12-31")
    quarterly_dates = pd.date_range("2020-03-31", periods=4, freq="QE")
    monthly_dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    outturns = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": quarterly_dates,
                    "variable": "target",
                    "vintage_date": vintage,
                    "frequency": "Q",
                    "value": [10.0, 11.0, 12.0, 13.0],
                    "metric": "levels",
                }
            ),
            pd.DataFrame(
                {
                    "date": monthly_dates,
                    "variable": "driver",
                    "vintage_date": vintage,
                    "frequency": "M",
                    "value": np.arange(1.0, 13.0),
                    "metric": "levels",
                }
            ),
        ],
        ignore_index=True,
    )
    conditioning = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-31", periods=3, freq="ME"),
            "variable": "driver",
            "vintage_date": vintage,
            "source": "monthly_conditioning",
            "frequency": "M",
            "value": [101.0, 102.0, 103.0],
            "forecast_horizon": 0,
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    data._raw_forecasts = conditioning.copy()
    data._raw_forecasts["target_minus_vintage"] = (
        data._raw_forecasts["date"].dt.to_period("M") - vintage.to_period("M")
    ).map(lambda period: period.n)
    data._raw_forecasts["unique_id"] = "monthly_conditioning"
    _SpyModel.captured_y = []
    _SpyModel.captured_X = []
    model = rt.RealTimeModel(data=data, models=_SpyModel())

    model.forecast(
        y_variables=["target"],
        X_variables=["driver"],
        X_steps_ahead={"driver": 0},
        X_sources={"driver": "monthly_conditioning"},
        data_transformation={"target": "levels", "driver": "levels"},
        steps=1,
        step_frequency="Q",
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    seen_X = _SpyModel.captured_X[0]
    conditioned = seen_X.loc[
        seen_X.index.isin(pd.date_range("2021-01-31", periods=3, freq="ME")),
        "driver",
    ]
    expected_dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    assert conditioned.index.tolist() == expected_dates.tolist()
    np.testing.assert_allclose(conditioned.to_numpy(), [101.0, 102.0, 103.0])


def test_quarterly_Y_and_X_conditioning_reaches_model_together():
    """Quarterly Y and X conditioning paths are applied together."""
    vintage = pd.Timestamp("2020-12-31")
    future_date = pd.Timestamp("2021-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-03-31",
                    "2020-06-30",
                    "2020-09-30",
                    "2020-12-31",
                ]
                * 2
            ),
            "variable": ["target"] * 4 + ["driver"] * 4,
            "vintage_date": vintage,
            "frequency": "Q",
            "value": [10.0, 11.0, 12.0, 13.0, 1.0, 2.0, 3.0, 4.0],
            "metric": "levels",
        }
    )
    conditioning = pd.DataFrame(
        {
            "date": [future_date, future_date],
            "variable": ["target", "driver"],
            "vintage_date": vintage,
            "source": ["target_conditioning", "driver_conditioning"],
            "frequency": "Q",
            "value": [101.0, 201.0],
            "forecast_horizon": 0,
            "metric": "levels",
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        forecasts_data=conditioning,
        compute_levels=False,
        data_check=False,
    )
    _SpyModel.captured_y = []
    _SpyModel.captured_X = []
    model = rt.RealTimeModel(data=data, models=_SpyModel())

    model.forecast(
        y_variables=["target"],
        X_variables=["driver"],
        y_steps_ahead={"target": 0},
        y_sources={"target": "target_conditioning"},
        X_steps_ahead={"driver": 0},
        X_sources={"driver": "driver_conditioning"},
        data_transformation={"target": "levels", "driver": "levels"},
        steps=1,
        step_frequency="Q",
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    assert _SpyModel.captured_y[0].loc[future_date, "target"] == 101.0
    assert _SpyModel.captured_X[0].loc[future_date, "driver"] == 201.0


def test_realtime_metric_selection_rejects_ambiguous_available_metrics():
    """Ambiguous available metrics require a requested transformation."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]
            ),
            "variable": "target",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [10.0, 100.0, 11.0, 101.0],
            "metric": ["pop", "yoy", "pop", "yoy"],
        }
    )
    data = fe.ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    model = rt.RealTimeModel(data=data, models=_SpyModel())

    with pytest.raises(
        ValueError,
        match=(r"target.*ambiguous.*available metrics: \['pop', 'yoy'\]"),
    ):
        model.forecast(
            y_variables=["target"],
            steps=1,
            first_vintage=str(vintage.date()),
            last_vintage=str(vintage.date()),
        )


@pytest.mark.parametrize("metric", ["pop", "levels"])
def test_forecast_accepts_forecast_data_filtered_to_pop_or_levels(metric):
    """Realtime forecasting works from either stored metric representation."""
    sample_data = rt.generate_synthetic_data(N=1, publication_lags=False)
    forecast_data = fe.ForecastData(
        outturns_data=sample_data,
        compute_levels=False,
        data_check=False,
    )
    forecast_data.filter(
        variables=["monthly_1"],
        metrics=[metric],
        start_vintage="2024-01-31",
        end_vintage="2024-02-29",
    )

    rt_model = rt.RealTimeModel(
        data=forecast_data,
        models=rt.models.ForecastOLS(label=f"OLS-{metric}"),
    )
    rt_model.forecast(
        y_variables=["monthly_1"],
        data_transformation={"monthly_1": "pop"},
        steps=1,
        first_forecast_horizon=1,
        first_vintage="2024-01-31",
        last_vintage="2024-02-29",
    )

    results = rt_model.data.forecasts
    assert not results.empty
    assert results["variable"].eq("monthly_1").all()
    assert results["metric"].eq("pop").any()


def test_filter_by_variables_and_metric_keeps_matching_rows():
    """Rows whose variable is requested and whose metric matches are kept."""
    data = pd.DataFrame(
        {
            "variable": ["gdp", "gdp", "unemp"],
            "metric": ["levels", "yoy", "levels"],
            "value": [100.0, 5.0, 4.0],
        }
    )

    result = DataTransformationPipeline({"gdp": "levels", "unemp": "levels"}).filter(
        data, ["gdp", "unemp"]
    )

    assert result["variable"].tolist() == ["gdp", "unemp"]
    assert result["metric"].tolist() == ["levels", "levels"]


def test_filter_by_variables_and_metric_drops_mismatched_metric():
    """A requested variable with the wrong metric for its transformation is
    dropped, even though other rows for the same variable are kept."""
    data = pd.DataFrame(
        {
            "variable": ["gdp", "gdp"],
            "metric": ["levels", "yoy"],
            "value": [100.0, 5.0],
        }
    )

    result = DataTransformationPipeline({"gdp": "yoy"}).filter(data, ["gdp"])

    assert result["metric"].tolist() == ["yoy"]
    assert result["value"].tolist() == [5.0]


def test_filter_by_variables_and_metric_drops_unrelated_variable():
    """A variable outside *variables* is excluded even with a matching metric."""
    data = pd.DataFrame(
        {
            "variable": ["gdp", "unemp"],
            "metric": ["levels", "levels"],
            "value": [100.0, 4.0],
        }
    )

    result = DataTransformationPipeline({"gdp": "levels", "unemp": "levels"}).filter(
        data, ["gdp"]
    )

    assert result["variable"].tolist() == ["gdp"]


def test_filter_by_variables_and_metric_raises_for_unmapped_variable():
    """A requested variable absent from ``data_transformation`` raises
    ``KeyError``. Public callers (``RealTimeModel.forecast``) validate that
    every variable has a transformation before reaching this helper, so this
    documents the current contract rather than a supported public usage."""
    data = pd.DataFrame({"variable": ["gdp"], "metric": ["levels"], "value": [100.0]})

    with pytest.raises(KeyError):
        DataTransformationPipeline({}).filter(data, ["gdp"])


def test_filter_by_variables_and_metric_preserves_missing_values():
    """NaN values in otherwise-matching rows survive the filter untouched."""
    data = pd.DataFrame(
        {
            "variable": ["gdp", "gdp"],
            "metric": ["levels", "levels"],
            "value": [np.nan, 100.0],
        }
    )

    result = DataTransformationPipeline({"gdp": "levels"}).filter(data, ["gdp"])

    assert result["value"].isna().tolist() == [True, False]


def test_filter_by_variables_and_metric_empty_frame():
    """An empty input with the required columns returns an empty result."""
    data = pd.DataFrame(columns=["variable", "metric", "value"])

    result = DataTransformationPipeline({"gdp": "levels"}).filter(data, ["gdp"])

    assert result.empty


def test_filter_by_variables_and_metric_returns_copy_preserving_index():
    """The filtered result is an independent copy that keeps the original
    (non-default) index of the matching rows."""
    data = pd.DataFrame(
        {
            "variable": ["gdp", "unemp"],
            "metric": ["levels", "levels"],
            "value": [100.0, 4.0],
        },
        index=[10, 20],
    )

    result = DataTransformationPipeline({"gdp": "levels", "unemp": "levels"}).filter(
        data, ["gdp"]
    )

    assert result.index.tolist() == [10]

    result.loc[10, "value"] = -1.0
    assert data.loc[10, "value"] == 100.0


# ============================================================================
# Parallel Execution Tests
# ============================================================================
# Tests for vintage-level and model-level parallelisation in RealTimeModel.
# Validates that sequential and parallel execution produce identical results.


class TestParallelExecution:
    """Test suite for parallel vs sequential execution equivalence."""

    def _run_forecast_and_return(
        self,
        data,
        model,
        label,
        y_variables=["cpisa"],
        X_variables=["unemp"],
        data_transformation=None,
        parallel=False,
        batch_size=10,
        y_lags=4,
    ):
        """Execute forecast and return (forecasts, elapsed_time) tuple."""
        if data_transformation is None:
            # pop by default for y_variables and X_variables
            data_transformation = {var: "pop" for var in y_variables + X_variables}

        rt_model = rt.RealTimeModel(data=data.copy(), models=model)

        start_time = time.time()
        rt_model.forecast(
            y_variables=y_variables,
            X_variables=X_variables,
            data_transformation=data_transformation,
            steps=8,
            label=label,
            first_vintage="2015-03-31",
            last_vintage="2018-12-31",
            parallel=parallel,
            batch_size=batch_size,
            y_lags=y_lags,
            X_imputation="zero",
        )
        elapsed = time.time() - start_time

        return rt_model.data.forecasts.copy(), elapsed

    def _assert_forecasts_equal(self, actual, expected, normalize_source=None):
        """Assert two forecast DataFrames are equal after sorting. Reduces duplication."""
        assert actual.shape[0] == expected.shape[0], (
            f"Shape mismatch: {actual.shape[0]} vs {expected.shape[0]}"
        )

        actual_sorted = actual.sort_values(
            ["vintage_date", "date", "variable"]
        ).reset_index(drop=True)
        expected_sorted = expected.sort_values(
            ["vintage_date", "date", "variable"]
        ).reset_index(drop=True)

        if normalize_source:
            actual_sorted["source"] = normalize_source

        pd.testing.assert_frame_equal(
            actual_sorted[
                [
                    "date",
                    "vintage_date",
                    "forecast_horizon",
                    "variable",
                    "value",
                    "metric",
                ]
            ].reset_index(drop=True),
            expected_sorted[
                [
                    "date",
                    "vintage_date",
                    "forecast_horizon",
                    "variable",
                    "value",
                    "metric",
                ]
            ].reset_index(drop=True),
            rtol=1e-10,
            atol=1e-12,
        )

    @pytest.fixture
    def setup_data(self):
        """Load and prepare test data for parallel execution tests."""
        forecast_data = fe.ForecastData(load_fer=True)
        forecast_data.filter(
            variables=["cpisa", "unemp"],
            start_vintage="2015-01-01",
            end_vintage="2020-12-31",
        )
        return forecast_data

    @pytest.fixture
    def setup_models(self):
        """Create test models for parallel execution tests."""
        return [
            rt.models.ForecastOLS(forecast_strategy="recursive", label="ForecastOLS_rec"),
            rt.models.ForecastOLS(
                forecast_strategy="direct", steps=8, label="ForecastOLS_dir"
            ),
        ]

    def test_single_model_sequential_vs_parallel_vintages(self, setup_data):
        """
        Test single model sequential vs parallel vintage execution equivalence.

        Validates vintage-level parallelisation correctness and reports timing.
        """
        model = rt.models.ForecastOLS()

        # Sequential and parallel execution with timing
        forecasts_seq, time_seq = self._run_forecast_and_return(
            setup_data, model, "Seq", parallel=False
        )
        forecasts_par, time_par = self._run_forecast_and_return(
            setup_data, model, "Par", parallel=True, batch_size=2
        )

        # Report timing
        speedup = time_seq / time_par if time_par > 0 else float("inf")
        print("\nSingle Model Timing:")
        print(f"  Sequential: {time_seq:.3f}s")
        print(f"  Parallel:   {time_par:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")

        # Validate equivalence
        self._assert_forecasts_equal(
            forecasts_par, forecasts_seq, normalize_source="ForecastOLS - Seq"
        )

    def test_multiple_models_sequential_vs_parallel(self, setup_data, setup_models):
        """
        Test multiple models sequential vs parallel execution equivalence.

        Validates model-level parallelisation and reports timing.
        """
        # Sequential: run each model separately and merge (current ml_models.py pattern)
        models_seq_results = {}
        total_time_seq = 0.0
        for model in setup_models:
            model_label = model.label
            forecasts, elapsed = self._run_forecast_and_return(
                copy.deepcopy(setup_data),
                model,
                None,
                parallel=False,
            )
            total_time_seq += elapsed
            # Filter to only this model's forecasts (exclude baseline forecasts
            # that came with setup_data)
            models_seq_results[model_label] = forecasts[
                forecasts["source"] == model_label
            ]

        data_seq_merged = pd.concat(list(models_seq_results.values()), ignore_index=True)

        # Parallel: list-based multi-model approach (uses single method call)
        rt_par = rt.RealTimeModel(data=setup_data.copy(), models=setup_models)
        start_time = time.time()
        rt_par.forecast(
            y_variables=["cpisa"],
            X_variables=["unemp"],
            data_transformation={"cpisa": "pop", "unemp": "pop"},
            steps=8,
            first_vintage="2015-03-31",
            last_vintage="2018-12-31",
            parallel=True,
            y_lags=4,
            X_imputation="zero",
        )
        time_par = time.time() - start_time
        data_par = rt_par.data.forecasts.copy()
        # Filter to only the models we just ran (exclude baseline forecasts)
        model_labels = [m.label for m in setup_models]
        data_par = data_par[data_par["source"].isin(model_labels)]

        # Report timing
        speedup = total_time_seq / time_par if time_par > 0 else float("inf")
        print("\nMultiple Models Timing:")
        print(f"  Sequential (sum): {total_time_seq:.3f}s")
        print(f"  Parallel:         {time_par:.3f}s")
        print(f"  Speedup:          {speedup:.2f}x")

        # Validate equivalence per model
        assert data_seq_merged.shape[0] == data_par.shape[0]

        data_seq_sorted = data_seq_merged.sort_values(
            ["source", "vintage_date", "date", "variable"]
        ).reset_index(drop=True)
        data_par_sorted = data_par.sort_values(
            ["source", "vintage_date", "date", "variable"]
        ).reset_index(drop=True)

        for model in setup_models:
            model_label = model.label
            seq_model = data_seq_sorted[data_seq_sorted["source"] == model_label]
            par_model = data_par_sorted[data_par_sorted["source"] == model_label]
            self._assert_forecasts_equal(par_model, seq_model)

    def test_backwards_compatibility_single_model(self, setup_data):
        """
        Test that existing single-model code still works (backwards compatibility).
        """
        model = rt.models.ForecastOLS()
        forecasts, time_seq = self._run_forecast_and_return(
            setup_data, model, "Seq", parallel=False
        )

        # Also test parallel mode and report timing
        _, time_par = self._run_forecast_and_return(
            setup_data, model, "Par", parallel=True, batch_size=2
        )

        speedup = time_seq / time_par if time_par > 0 else float("inf")
        print("\nBackwards Compatibility (Single Model) Timing:")
        print(f"  Sequential: {time_seq:.3f}s")
        print(f"  Parallel:   {time_par:.3f}s")
        print(f"  Speedup:    {speedup:.2f}x")

        # Should work without errors
        assert forecasts is not None
        assert not forecasts.empty
        # Filter to only ForecastOLS forecasts (data may contain older baseline forecasts)
        ols_forecasts = forecasts[forecasts["source"] == "ForecastOLS - Seq"]
        assert not ols_forecasts.empty, "No ForecastOLS forecasts found"
        assert (ols_forecasts["source"] == "ForecastOLS - Seq").all() or (
            ols_forecasts["source"] == "ForecastOLS"
        ).all()

    @pytest.mark.skip(
        reason="This test is a demonstration of parallelization"
        " benefits with a large workload. too slow."
    )
    def test_50_models_large_scale(self, setup_data):
        """
        Test parallelization with 50 models (all same ForecastOLS model).

        Demonstrates parallelization benefit with significant workload.
        Uses default parameters: parallel=True with auto batch_size.
        """
        # Create 50 identical ForecastOLS models with explicit labels
        models_50 = [
            rt.models.ForecastOLS(label=f"ForecastOLS_{i:02d}") for i in range(50)
        ]

        # Run with parallel=True and auto batch_size (batch_size=None)
        rt_model = rt.RealTimeModel(data=setup_data.copy(), models=models_50)

        start_time = time.time()
        rt_model.forecast(
            y_variables=["cpisa"],
            X_variables=["unemp"],
            data_transformation={"cpisa": "pop", "unemp": "pop"},
            steps=8,
            first_vintage="2015-03-31",
            last_vintage="2018-12-31",
            parallel=True,
            # batch_size=None -> auto-computed based on num_workers
            y_lags=4,
        )
        time_parallel = time.time() - start_time

        start_time = time.time()
        rt_model.forecast(
            y_variables=["cpisa"],
            X_variables=["unemp"],
            data_transformation={"cpisa": "pop", "unemp": "pop"},
            steps=8,
            first_vintage="2015-03-31",
            last_vintage="2018-12-31",
            parallel=False,
            y_lags=4,
        )
        time_sequential = time.time() - start_time

        forecasts = rt_model.data.forecasts.copy()
        print("\n50 Models Parallel (auto batch_size) Timing:")
        print(f"  Time: {time_parallel:.3f}s")
        print("50 Models Sequential Timing:")
        print(f"  Time: {time_sequential:.3f}s")

        # Validate results
        assert forecasts is not None
        assert not forecasts.empty
        ols_forecasts = forecasts[forecasts["source"].str.startswith("ForecastOLS")]
        ols_models = ols_forecasts["source"].unique()
        assert len(ols_models) == len(models_50), (
            f"Expected {len(models_50)} ForecastOLS models, got {len(ols_models)}"
        )


# ============================================================================
# Missing-value capability alignment
# ============================================================================
# Models that do not handle missing values receive complete-case estimation
# data; models that do handle them receive the raw aligned panel.

GAP_VINTAGE = pd.Timestamp("2003-02-28")


def _gap_outturns(
    drop_dates=None, y_drop_dates=None, second_driver=False, extra_x_months=0
):
    """Build a single-vintage monthly panel of a target plus one or two regressors.

    ``ForecastData`` refuses null values in the long panel, so a gap can only be
    expressed by removing rows. Whether that surfaces as a missing row or as a
    NaN depends on the company the regressor keeps: with a second regressor
    still covering the date, the pivot leaves a NaN in the first one's column.

    Args:
        drop_dates : list[pd.Timestamp] or None
            Dates whose ``driver`` rows are removed.
        y_drop_dates : list[pd.Timestamp] or None
            Dates whose ``target`` rows are removed.
        second_driver : bool
            Whether to add a ``driver2`` regressor spanning every date.
        extra_x_months : int
            Months by which the regressors lead the target (ragged edge).
    """
    dates = pd.date_range("2000-01-31", periods=36, freq="ME")
    rng = np.random.default_rng(0)
    x_values = rng.normal(size=36)
    y_values = 2.0 + 0.5 * x_values + rng.normal(scale=0.01, size=36)

    # The target stops earlier when the regressors are made to lead it.
    y_dates = dates[: 36 - extra_x_months]
    y_frame = pd.DataFrame(
        {
            "date": y_dates,
            "variable": "target",
            "vintage_date": GAP_VINTAGE,
            "frequency": "M",
            "value": y_values[: len(y_dates)],
            "metric": "levels",
        }
    )
    if y_drop_dates is not None:
        y_frame = y_frame[~y_frame["date"].isin(y_drop_dates)]
    frames = [y_frame]

    x_frame = pd.DataFrame(
        {
            "date": dates,
            "variable": "driver",
            "vintage_date": GAP_VINTAGE,
            "frequency": "M",
            "value": x_values,
            "metric": "levels",
        }
    )
    if drop_dates is not None:
        x_frame = x_frame[~x_frame["date"].isin(drop_dates)]
    frames.append(x_frame)

    if second_driver:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "variable": "driver2",
                    "vintage_date": GAP_VINTAGE,
                    "frequency": "M",
                    "value": rng.normal(size=36),
                    "metric": "levels",
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _run_gap_forecast(outturns, **forecast_kwargs):
    """Run a one-vintage OLS forecast over the supplied panel."""
    x_variables = [v for v in ("driver", "driver2") if v in set(outturns["variable"])]
    data = fe.ForecastData(outturns_data=outturns)
    rt_model = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS(drop_nans=True))
    rt_model.forecast(
        y_variables=["target"],
        X_variables=x_variables,
        data_transformation={v: "levels" for v in ["target", *x_variables]},
        steps=1,
        first_vintage=str(GAP_VINTAGE.date()),
        last_vintage=str(GAP_VINTAGE.date()),
        X_imputation="last",
        **forecast_kwargs,
    )
    return rt_model


def test_forecast_allows_missing_row_for_capable_model():
    """A capable model receives a panel with a missing regressor date.

    Absent rows also leave the index unevenly spaced, which would otherwise
    surface much later as an opaque frequency-inference failure inside ragged
    edge imputation.
    """
    outturns = _gap_outturns(drop_dates=[pd.Timestamp("2001-09-30")])

    rt_model = _run_gap_forecast(outturns)

    assert rt_model.data.forecasts is not None


def test_forecast_allows_internal_gap_for_capable_model():
    """A capable model receives a panel with an internal regressor gap.

    The date sits inside the observed span, so it is missing data rather than a
    ragged publication edge. The model owns how that gap is handled.
    """
    outturns = _gap_outturns(drop_dates=[pd.Timestamp("2001-09-30")], second_driver=True)

    rt_model = _run_gap_forecast(outturns)

    assert rt_model.data.forecasts is not None


def test_forecast_allows_internal_gap_without_policy_flag():
    """Missing-value handling is selected by the model capability."""
    outturns = _gap_outturns(drop_dates=[pd.Timestamp("2001-09-30")], second_driver=True)

    rt_model = _run_gap_forecast(outturns)

    assert rt_model.data.forecasts is not None


def test_forecast_allows_gap_in_target_for_capable_model():
    """A capable model receives a panel with an internal target gap.

    The model owns how missing target observations are handled.
    """
    outturns = _gap_outturns(y_drop_dates=[pd.Timestamp("2001-09-30")])

    rt_model = _run_gap_forecast(outturns)

    assert rt_model.data.forecasts is not None


def test_forecast_permits_ragged_leading_indicator():
    """A regressor published ahead of the target is not an internal gap."""
    outturns = _gap_outturns(extra_x_months=2)

    rt_model = _run_gap_forecast(outturns)

    assert rt_model.data.forecasts is not None


def test_forecast_does_not_expose_allow_missing_values():
    """Missing-value policy is controlled by the model capability."""
    parameters = inspect.signature(rt.RealTimeModel.forecast).parameters

    assert "allow_missing_values" not in parameters


def _run_missing_capability_forecast(handles_missing_values):
    """Capture the estimation panel received by a model capability variant."""
    dates = pd.date_range("2000-01-31", periods=12, freq="ME")
    frames = []
    values = {
        "target_a": np.arange(12.0),
        "target_b": np.arange(12.0) + 10.0,
        "driver_a": np.arange(12.0) + 20.0,
        "driver_b": np.arange(12.0) + 30.0,
    }
    missing_dates = {
        "target_a": [],
        "target_b": [dates[0], dates[4]],
        "driver_a": [],
        "driver_b": [dates[1], dates[6]],
    }
    for variable, variable_values in values.items():
        frame = pd.DataFrame(
            {
                "date": dates,
                "variable": variable,
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": variable_values,
                "metric": "levels",
            }
        )
        frame = frame[~frame["date"].isin(missing_dates[variable])]
        frames.append(frame)

    captured = {"y": None, "X": None}

    class CapturingModel(ForecastModel):
        _handles_missing_values = handles_missing_values

        def _fit(self, y, X=None, **kwargs):
            captured["y"] = y.copy()
            captured["X"] = X.copy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.zeros((steps, 2))

    data = fe.ForecastData(outturns_data=pd.concat(frames, ignore_index=True))
    model = rt.RealTimeModel(data=data, models=CapturingModel())
    model.forecast(
        y_variables=["target_a", "target_b"],
        X_variables=["driver_a", "driver_b"],
        data_transformation={variable: "levels" for variable in values},
        steps=1,
        first_forecast_horizon=1,
        first_vintage=str(dates[-1].date()),
        last_vintage=str(dates[-1].date()),
    )
    return captured


def test_model_not_handling_missing_values_receives_complete_cases():
    """NaN-intolerant models receive jointly aligned, complete y/X rows."""
    captured = _run_missing_capability_forecast(handles_missing_values=False)

    assert not captured["y"].isna().any().any()
    assert not captured["X"].isna().any().any()
    pd.testing.assert_index_equal(captured["y"].index, captured["X"].index)
    assert len(captured["y"]) == 8


def test_model_handling_missing_values_receives_raw_panel():
    """MIDAS-like models retain ownership of their internal NaN alignment."""
    captured = _run_missing_capability_forecast(handles_missing_values=True)

    assert captured["y"].isna().any().any()
    assert captured["X"].isna().any().any()


def test_model_not_handling_missing_values_receives_complete_lagged_design():
    """Lag construction must not reintroduce missing estimation rows."""
    dates = pd.date_range("2000-01-31", periods=6, freq="ME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "variable": "target",
            "vintage_date": dates[-1],
            "frequency": "M",
            "value": np.arange(6.0),
            "metric": "levels",
        }
    )
    captured = {}

    class CapturingModel(ForecastModel):
        _handles_missing_values = False

        def _fit(self, y, X=None, **kwargs):
            captured["y"] = y.copy()
            captured["X"] = X.copy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.zeros((steps, 1))

    data = fe.ForecastData(outturns_data=frame)
    model = rt.RealTimeModel(data=data, models=CapturingModel())
    model.forecast(
        y_variables=["target"],
        data_transformation={"target": "levels"},
        steps=1,
        first_forecast_horizon=1,
        first_vintage=str(dates[-1].date()),
        last_vintage=str(dates[-1].date()),
        y_lags=1,
    )

    assert not captured["y"].isna().any().any()
    assert not captured["X"].isna().any().any()
    expected_index = pd.DatetimeIndex(dates[1:].astype("datetime64[ns]"), name="date")
    pd.testing.assert_index_equal(captured["y"].index, expected_index)
    pd.testing.assert_index_equal(captured["X"].index, expected_index)


def test_model_not_handling_values_preserves_internal_calendar_gaps_for_lags():
    """Missing calendar periods must not be compressed before lag creation."""
    dates = pd.date_range("2000-01-31", periods=6, freq="ME")
    frames = []
    for variable, values, missing_dates in [
        ("target", np.arange(6.0), []),
        ("driver", np.arange(10.0, 16.0), [dates[2]]),
    ]:
        frame = pd.DataFrame(
            {
                "date": dates,
                "variable": variable,
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": values,
                "metric": "levels",
            }
        )
        frames.append(frame[~frame["date"].isin(missing_dates)])

    captured = {}

    class CapturingModel(ForecastModel):
        _handles_missing_values = False

        def _fit(self, y, X=None, **kwargs):
            captured["y"] = y.copy()
            captured["X"] = X.copy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.zeros((steps, 1))

    data = fe.ForecastData(outturns_data=pd.concat(frames, ignore_index=True))
    model = rt.RealTimeModel(data=data, models=CapturingModel())
    model.forecast(
        y_variables=["target"],
        X_variables=["driver"],
        data_transformation={"target": "levels", "driver": "levels"},
        steps=1,
        first_forecast_horizon=1,
        first_vintage=str(dates[-1].date()),
        last_vintage=str(dates[-1].date()),
        X_lags=1,
    )

    expected_index = pd.DatetimeIndex([dates[1], dates[4], dates[5]], name="date").astype(
        "datetime64[ns]"
    )
    pd.testing.assert_index_equal(captured["y"].index, expected_index)
    pd.testing.assert_index_equal(captured["X"].index, expected_index)
    assert captured["X"].loc[dates[4], "driver_lag1"] == 13.0


def test_model_not_handling_missing_values_filters_after_formula_selection():
    """Missing values in formula-excluded regressors must not drop rows."""
    dates = pd.date_range("2000-01-31", periods=6, freq="ME")
    frames = []
    for variable, values, missing_dates in [
        ("target", np.arange(6.0), []),
        ("used", np.arange(10.0, 16.0), []),
        ("unused", np.arange(20.0, 26.0), [dates[2]]),
    ]:
        frame = pd.DataFrame(
            {
                "date": dates,
                "variable": variable,
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": values,
                "metric": "levels",
            }
        )
        frames.append(frame[~frame["date"].isin(missing_dates)])

    captured = {}

    class CapturingModel(ForecastModel):
        _handles_missing_values = False

        def _fit(self, y, X=None, **kwargs):
            captured["y"] = y.copy()
            captured["X"] = X.copy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.zeros((steps, 1))

    data = fe.ForecastData(outturns_data=pd.concat(frames, ignore_index=True))
    model = rt.RealTimeModel(data=data, models=CapturingModel(formula="target ~ used"))
    model.forecast(
        y_variables=["target"],
        X_variables=["used", "unused"],
        data_transformation={
            "target": "levels",
            "used": "levels",
            "unused": "levels",
        },
        steps=1,
        first_forecast_horizon=1,
        first_vintage=str(dates[-1].date()),
        last_vintage=str(dates[-1].date()),
    )

    expected_index = pd.DatetimeIndex(dates.astype("datetime64[ns]"), name="date")
    pd.testing.assert_index_equal(captured["y"].index, expected_index)
    pd.testing.assert_index_equal(captured["X"].index, expected_index)
    assert list(captured["X"].columns) == ["used"]


def test_model_not_handling_missing_values_keeps_pre_target_x_lag_history():
    """X history before the first target date must remain available for lags."""
    dates = pd.date_range("2000-01-31", periods=6, freq="ME")
    frames = [
        pd.DataFrame(
            {
                "date": dates[1:],
                "variable": "target",
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": np.arange(5.0),
                "metric": "levels",
            }
        ),
        pd.DataFrame(
            {
                "date": dates,
                "variable": "driver",
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": np.arange(10.0, 16.0),
                "metric": "levels",
            }
        ),
    ]
    captured = {}

    class CapturingModel(ForecastModel):
        _handles_missing_values = False

        def _fit(self, y, X=None, **kwargs):
            captured["y"] = y.copy()
            captured["X"] = X.copy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.zeros((steps, 1))

    data = fe.ForecastData(outturns_data=pd.concat(frames, ignore_index=True))
    model = rt.RealTimeModel(data=data, models=CapturingModel())
    model.forecast(
        y_variables=["target"],
        X_variables=["driver"],
        data_transformation={"target": "levels", "driver": "levels"},
        steps=1,
        first_forecast_horizon=1,
        first_vintage=str(dates[-1].date()),
        last_vintage=str(dates[-1].date()),
        X_lags=1,
    )

    expected_index = pd.DatetimeIndex(dates[1:].astype("datetime64[ns]"), name="date")
    pd.testing.assert_index_equal(captured["y"].index, expected_index)
    pd.testing.assert_index_equal(captured["X"].index, expected_index)
    np.testing.assert_allclose(captured["X"]["driver_lag1"], np.arange(10.0, 15.0))


def test_model_not_handling_missing_values_drops_incomplete_future_x_rows():
    """Incomplete future X rows must not be reintroduced for forecasting."""
    dates = pd.date_range("2000-01-31", periods=8, freq="ME")
    frames = [
        pd.DataFrame(
            {
                "date": dates[:6],
                "variable": "target",
                "vintage_date": dates[5],
                "frequency": "M",
                "value": np.arange(6.0),
                "metric": "levels",
            }
        ),
        pd.DataFrame(
            {
                "date": dates,
                "variable": "driver_a",
                "vintage_date": dates[5],
                "frequency": "M",
                "value": np.arange(10.0, 18.0),
                "metric": "levels",
            }
        ),
        pd.DataFrame(
            {
                "date": dates[[0, 1, 2, 3, 4, 5, 7]],
                "variable": "driver_b",
                "vintage_date": dates[5],
                "frequency": "M",
                "value": np.arange(20.0, 27.0),
                "metric": "levels",
            }
        ),
    ]
    captured = {}

    class CapturingModel(ForecastModel):
        _handles_missing_values = False

        def _fit(self, y, X=None, **kwargs):
            captured["fit_X"] = X.copy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            captured["forecast_X"] = X.copy()
            return np.zeros((steps, 1))

    data = fe.ForecastData(outturns_data=pd.concat(frames, ignore_index=True))
    model = rt.RealTimeModel(data=data, models=CapturingModel())
    model.forecast(
        y_variables=["target"],
        X_variables=["driver_a", "driver_b"],
        data_transformation={
            "target": "levels",
            "driver_a": "levels",
            "driver_b": "levels",
        },
        steps=2,
        first_forecast_horizon=1,
        first_vintage=str(dates[5].date()),
        last_vintage=str(dates[5].date()),
    )

    assert dates[6] not in captured["forecast_X"].index
    assert not captured["forecast_X"].isna().any().any()
    assert dates[6] not in captured["fit_X"].index


def test_model_not_handling_missing_values_anchors_after_trailing_target_gap():
    """A trailing target gap must not become part of the forecast history."""
    dates = pd.date_range("2000-01-31", periods=6, freq="ME")
    frames = [
        pd.DataFrame(
            {
                "date": dates[:-1],
                "variable": "target",
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": np.arange(5.0),
                "metric": "levels",
            }
        ),
        pd.DataFrame(
            {
                "date": dates,
                "variable": "driver",
                "vintage_date": dates[-1],
                "frequency": "M",
                "value": np.arange(10.0, 16.0),
                "metric": "levels",
            }
        ),
    ]
    captured = {}

    class CapturingModel(ForecastModel):
        _handles_missing_values = False

        def _fit(self, y, X=None, **kwargs):
            captured["fit_y"] = y.copy()
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            captured["forecast_y"] = y.copy()
            return np.zeros((steps, 1))

    data = fe.ForecastData(outturns_data=pd.concat(frames, ignore_index=True))
    model = rt.RealTimeModel(data=data, models=CapturingModel())
    model.forecast(
        y_variables=["target"],
        X_variables=["driver"],
        data_transformation={"target": "levels", "driver": "levels"},
        steps=1,
        first_forecast_horizon=1,
        first_vintage=str(dates[-1].date()),
        last_vintage=str(dates[-1].date()),
    )

    assert captured["fit_y"].index[-1] == dates[-2]
    assert captured["forecast_y"].index[-1] == dates[-2]
    assert not captured["forecast_y"].isna().any().any()


# ============================================================================
# Per-model data transformation pipeline resolution (Phase 3)
# ============================================================================
# Each model in one RealTimeModel run resolves its own pipeline: the model's
# own `data_transformation` when set, otherwise the call-level
# `data_transformation` mapping as a fallback.


class _PipelineSpyModel(ForecastModel):
    """Records the fitting/conditioning ``y`` each labelled instance receives.

    ``RealTimeModel`` fits a fresh ``copy.deepcopy`` of the model per vintage,
    so captures are keyed by ``label`` on a class-level dict (surviving the
    copy) rather than on the instance, so two differently-configured
    instances of this class don't share captures.
    """

    captured: dict = {}

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        _PipelineSpyModel.captured.setdefault(self.label, []).append(
            y.copy() if y is not None else None
        )
        return pd.DataFrame({col: [0.0] * steps for col in self.y.columns})


def test_two_models_receive_independently_transformed_data_sequential():
    """Each model transforms an independent copy of the same raw data.

    ``LevelsModel`` has no model-owned pipeline (falls back to the
    call-level ``data_transformation``: "levels"); ``DiffModel`` owns a
    "diff" pipeline for the same variable. ``drop_transformation_nans=False``
    is passed explicitly, so the leading differencing NaN produced for
    ``DiffModel`` is retained rather than dropped.
    """
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    _PipelineSpyModel.captured = {}
    levels_model = _PipelineSpyModel(label="LevelsModel")
    diff_model = _PipelineSpyModel(label="DiffModel", data_transformation={"gdp": "diff"})

    rt_model = rt.RealTimeModel(data=data, models=[levels_model, diff_model])
    rt_model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "levels"},
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        drop_transformation_nans=False,
    )

    levels_y = _PipelineSpyModel.captured["LevelsModel"][0]
    diff_y = _PipelineSpyModel.captured["DiffModel"][0]

    np.testing.assert_allclose(levels_y["gdp"].to_numpy(), [100.0, 110.0, 121.0])
    np.testing.assert_allclose(
        diff_y["gdp"].to_numpy(), [np.nan, 10.0, 11.0], equal_nan=True
    )

    # Output metric assignment must be correct for each model, and level
    # reconstruction (using each model's own mapping) must not be skipped or
    # corrupted by the other model's mapping.
    forecasts = rt_model.data.forecasts
    levels_row = forecasts[
        (forecasts["source"] == "LevelsModel") & (forecasts["metric"] == "levels")
    ]
    diff_row = forecasts[
        (forecasts["source"] == "DiffModel") & (forecasts["metric"] == "levels")
    ]
    assert len(levels_row) == 1
    assert len(diff_row) == 1
    np.testing.assert_allclose(levels_row["value"].iloc[0], 0.0)
    # DiffModel forecasts a 0.0 diff; reconstructed level = last outturn (121) + 0.
    np.testing.assert_allclose(diff_row["value"].iloc[0], 121.0)


def test_two_models_receive_independently_transformed_data_parallel():
    """The same per-model dispatch holds with ``parallel=True``.

    Each (model, vintage_batch) worker must receive that model's own
    transformed data/vintages/mapping rather than one shared transformed
    panel, so the reconstructed output must match the sequential case.
    """
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    _PipelineSpyModel.captured = {}
    levels_model = _PipelineSpyModel(label="LevelsModel")
    diff_model = _PipelineSpyModel(label="DiffModel", data_transformation={"gdp": "diff"})

    rt_model = rt.RealTimeModel(data=data, models=[levels_model, diff_model])
    rt_model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "levels"},
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        drop_transformation_nans=False,
        parallel=True,
        batch_size=1,
        max_workers=1,
    )

    forecasts = rt_model.data.forecasts
    levels_row = forecasts[
        (forecasts["source"] == "LevelsModel") & (forecasts["metric"] == "levels")
    ]
    diff_row = forecasts[
        (forecasts["source"] == "DiffModel") & (forecasts["metric"] == "levels")
    ]
    assert len(levels_row) == 1
    assert len(diff_row) == 1
    np.testing.assert_allclose(levels_row["value"].iloc[0], 0.0)
    np.testing.assert_allclose(diff_row["value"].iloc[0], 121.0)


def test_model_own_pipeline_missing_variable_mapping_raises_with_model_context():
    """A model-owned pipeline missing a requested variable raises the
    existing clear validation error, tied to the offending model's label."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    bad_model = _PipelineSpyModel(
        label="BadModel", data_transformation={"other_var": "levels"}
    )
    rt_model = rt.RealTimeModel(data=data, models=bad_model)

    with pytest.raises(
        ValueError,
        match=r"Model 'BadModel':.*data_transformation must contain all y_variables",
    ):
        rt_model.forecast(
            y_variables=["gdp"],
            data_transformation={"gdp": "levels"},
            steps=1,
            first_vintage=str(vintage.date()),
            last_vintage=str(vintage.date()),
        )


def test_formula_model_does_not_require_unselected_variable_mappings():
    """A formula model does not need mappings for unused shared inputs."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-31",
                    "2020-02-29",
                    "2020-03-31",
                ]
                * 2
            ),
            "variable": ["gdp"] * 3 + ["unused"] * 3,
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0, 10.0, 20.0, 30.0],
            "metric": "levels",
        }
    )
    unused_pop = outturns[outturns["variable"] == "unused"].copy()
    unused_pop["metric"] = "pop"
    unused_pop["value"] = [1.0, 2.0, 3.0]
    outturns = pd.concat([outturns, unused_pop], ignore_index=True)
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    model = _PipelineSpyModel(
        label="FormulaModel",
        formula="gdp ~ gdp",
        data_transformation={"gdp": "levels"},
    )
    rt_model = rt.RealTimeModel(data=data, models=model)

    rt_model.forecast(
        y_variables=["gdp", "unused"],
        X_variables=["gdp", "unused"],
        data_transformation={"gdp": "levels"},
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )


def test_conditioning_forecast_transformed_on_each_models_own_scale():
    """A ``y`` conditioning path is transformed on each model's own scale.

    The same nowcast (a levels value) reaches ``LevelsModel`` unchanged, but
    reaches ``DiffModel`` as the difference against the preceding outturn,
    since the conditioning forecast is transformed by each model's own
    pipeline rather than a single shared transformation.
    """
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    nowcast = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-30"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "source": "nowcast",
            "frequency": "M",
            "value": [130.0],
            "metric": "levels",
            "forecast_horizon": [0],
        }
    )
    data = fe.NowcastData(
        outturns_data=outturns,
        forecasts_data=nowcast,
        compute_levels=False,
        data_check=False,
    )

    _PipelineSpyModel.captured = {}
    levels_model = _PipelineSpyModel(label="LevelsModel")
    diff_model = _PipelineSpyModel(label="DiffModel", data_transformation={"gdp": "diff"})

    rt_model = rt.RealTimeModel(data=data, models=[levels_model, diff_model])
    rt_model.forecast(
        y_variables=["gdp"],
        y_steps_ahead={"gdp": 0},
        y_sources={"gdp": "nowcast"},
        data_transformation={"gdp": "levels"},
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        drop_transformation_nans=False,
    )

    levels_y = _PipelineSpyModel.captured["LevelsModel"][0]
    diff_y = _PipelineSpyModel.captured["DiffModel"][0]

    np.testing.assert_allclose(levels_y.loc[pd.Timestamp("2020-04-30"), "gdp"], 130.0)
    # DiffModel's own pipeline differences the conditioning path against the
    # preceding (March) outturn on the model's own scale: 130 - 121 = 9.
    np.testing.assert_allclose(diff_y.loc[pd.Timestamp("2020-04-30"), "gdp"], 9.0)


def test_call_level_data_transformation_optional_when_model_pipeline_covers_all():
    """``data_transformation`` may be omitted entirely when every model's own
    pipeline covers the requested y/X variables."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    _PipelineSpyModel.captured = {}
    model = _PipelineSpyModel(
        label="OwnPipelineModel", data_transformation={"gdp": "levels"}
    )

    rt_model = rt.RealTimeModel(data=data, models=model)
    rt_model.forecast(
        y_variables=["gdp"],
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    forecasts = rt_model.data.forecasts
    assert (forecasts["source"] == "OwnPipelineModel").any()


def _tree_first_component(components):
    return next(iter(components.values()))


class _TreePipelineLeaf(ForecastModel):
    """Pickleable leaf used to verify model-owned tree pipelines in dispatch."""

    captured = {}

    def __init__(self, label, data_transformation):
        super().__init__(label=label, data_transformation=data_transformation)

    def _fit(self, y, X=None, **kwargs):
        _TreePipelineLeaf.captured.setdefault(self.label, []).append(y.copy())
        self.fitted_values_ = y.copy()
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        return pd.DataFrame(
            {column: [self.y[column].iloc[-1]] * steps for column in self.y.columns},
            index=pd.date_range(self.y.index[-1], periods=steps + 1, freq="ME")[1:],
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_realtime_tree_dispatch_accepts_complete_leaf_pipelines_without_fallback(
    parallel,
):
    vintage = pd.Timestamp("2020-03-31")
    dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    outturns = pd.DataFrame(
        {
            "date": list(dates),
            "variable": ["monthly"] * len(dates),
            "vintage_date": vintage,
            "frequency": "M",
            "value": list(np.linspace(100.0, 114.0, len(dates))),
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)
    monthly_leaf = _TreePipelineLeaf("monthly_leaf", {"monthly": "diff"})
    quarterly_leaf = _TreePipelineLeaf("quarterly_leaf", {"monthly": "levels"})
    tree = ForecastTree(
        TreeNode(
            transform=_tree_first_component,
            children=[monthly_leaf, quarterly_leaf],
            name="root",
            target="monthly",
        ),
        label="mixed_tree",
    )
    _TreePipelineLeaf.captured = {}

    rt.RealTimeModel(data=data, models=tree).forecast(
        y_variables=["monthly"],
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        parallel=parallel,
        max_workers=1,
        reconstruct_levels=False,
        drop_transformation_nans=False,
    )

    forecasts = data.forecasts[
        (data.forecasts["source"] == "mixed_tree")
        & (data.forecasts["metric"] == "levels")
    ]
    assert len(forecasts) == 1
    np.testing.assert_allclose(forecasts["value"], 1.0)
    if not parallel:
        assert set(_TreePipelineLeaf.captured) == {"monthly_leaf", "quarterly_leaf"}
        np.testing.assert_allclose(
            _TreePipelineLeaf.captured["monthly_leaf"][0]["monthly"].to_numpy(),
            [np.nan] + [1.0] * 13,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            _TreePipelineLeaf.captured["quarterly_leaf"][0]["monthly"].to_numpy(),
            np.linspace(100.0, 113.0, len(dates) - 1),
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_realtime_tree_owned_input_pipeline_does_not_reconstruct_root_output(
    parallel,
):
    vintage = pd.Timestamp("2020-03-31")
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    outturns = pd.DataFrame(
        {
            "date": dates,
            "variable": "monthly",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)
    leaf = _TreePipelineLeaf("diff_leaf", {"monthly": "levels"})
    tree = ForecastTree(
        TreeNode(
            transform=_tree_first_component,
            children=[leaf],
            name="root",
            target="monthly",
        ),
        label="diff_tree",
        data_transformation={"monthly": "diff"},
    )

    rt.RealTimeModel(data=data, models=tree).forecast(
        y_variables=["monthly"],
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        parallel=parallel,
        max_workers=1,
        drop_transformation_nans=False,
    )

    forecasts = data.forecasts[
        (data.forecasts["source"] == "diff_tree")
        & (data.forecasts["variable"] == "monthly")
    ]
    level_forecasts = forecasts[forecasts["metric"] == "levels"]
    assert len(level_forecasts) == 1
    np.testing.assert_allclose(level_forecasts["value"].iloc[0], 110.0)


def test_realtime_skips_vintage_with_no_usable_transformed_y():
    vintages = pd.to_datetime(["2020-03-31", "2020-06-30"])
    later_dates = pd.date_range("2018-03-31", periods=25, freq="ME")
    outturns = pd.DataFrame(
        {
            "date": list(pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))
            + list(later_dates),
            "variable": "gdp",
            "vintage_date": [vintages[0]] * 3 + [vintages[1]] * len(later_dates),
            "frequency": "M",
            "value": [100.0, 101.0, 102.0]
            + list(np.linspace(100.0, 124.0, len(later_dates))),
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    with pytest.warns(UserWarning, match="No usable transformed y"):
        rt.RealTimeModel(data=data, models=_PipelineSpyModel(label="skip_test")).forecast(
            y_variables=["gdp"],
            data_transformation={"gdp": "yoy"},
            steps=1,
            first_forecast_horizon=0,
            first_vintage=str(vintages[0].date()),
            last_vintage=str(vintages[1].date()),
            drop_transformation_nans=True,
        )


def test_model_without_pipeline_uses_identity_input_when_fallback_is_omitted():
    """A model without either pipeline uses raw levels as its input."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    plain_model = _PipelineSpyModel(label="PlainModel")
    rt_model = rt.RealTimeModel(data=data, models=plain_model)

    rt_model.forecast(
        y_variables=["gdp"],
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    assert (data.forecasts["source"] == "PlainModel").any()


def test_call_level_data_transformation_optional_with_mixed_models():
    """One model owns a full pipeline; a second needs the call-level
    fallback, which must still be validated/applied for that model only."""
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)

    _PipelineSpyModel.captured = {}
    own_pipeline_model = _PipelineSpyModel(
        label="OwnPipelineModel", data_transformation={"gdp": "diff"}
    )
    fallback_model = _PipelineSpyModel(label="FallbackModel")

    rt_model = rt.RealTimeModel(data=data, models=[own_pipeline_model, fallback_model])
    rt_model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "levels"},
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        drop_transformation_nans=False,
    )

    forecasts = rt_model.data.forecasts
    assert (forecasts["source"] == "OwnPipelineModel").any()
    assert (forecasts["source"] == "FallbackModel").any()


def test_reconstruct_levels_false_preserves_native_metric_forecast():
    """When level reconstruction is skipped, a forecast in a metric that
    ``ForecastData`` cannot store directly (e.g. ``diff``) is preserved on
    ``RealTimeModel.native_forecasts`` instead of being silently dropped."""

    class DiffConstantModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.full((steps, 1), 5.0)

    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)
    model = rt.RealTimeModel(data=data, models=DiffConstantModel())

    model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "diff"},
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        reconstruct_levels=False,
    )

    assert model.data.forecasts.empty
    native = model.native_forecasts
    assert native is not None
    row = native[(native["variable"] == "gdp") & (native["metric"] == "diff")]
    assert len(row) == 1
    np.testing.assert_allclose(row["value"].iloc[0], 5.0)


def test_reconstruct_levels_true_does_not_populate_native_forecasts():
    """When reconstruction succeeds, there is no unrepresentable native row
    left over, so ``native_forecasts`` stays ``None``."""

    class DiffConstantModel(ForecastModel):
        def _fit(self, y, X=None, **kwargs):
            return self

        def _forecast(self, steps, X=None, y=None, **kwargs):
            return np.full((steps, 1), 5.0)

    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    data = fe.ForecastData(outturns_data=outturns, compute_levels=False, data_check=False)
    model = rt.RealTimeModel(data=data, models=DiffConstantModel())

    model.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "diff"},
        steps=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        reconstruct_levels=True,
    )

    assert not model.data.forecasts.empty
    assert model.native_forecasts is None
