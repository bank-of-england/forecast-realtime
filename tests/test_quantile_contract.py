import forecast_evaluation as fe
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import forecast_realtime as rt
from forecast_realtime._realtime_forecasting import ForecastRunResult
from forecast_realtime.forecast_model import (
    ForecastModel,
    ForecastResult,
    _normalise_quantiles,
)


class _FitQuantileSpyOLS(rt.models.ForecastOLS):
    """Record fit options while retaining the OLS density implementation."""

    fit_kwargs = []

    def _fit(self, y, X=None, **kwargs):
        type(self).fit_kwargs.append(kwargs.copy())
        return super()._fit(y, X=X, **kwargs)


class _OriginInclusiveOLS(rt.models.ForecastOLS):
    """Small density model whose forecast calendar includes its origin."""

    _forecast_dates_include_origin = True


class _OriginInclusiveMultiModel(ForecastModel):
    """Return deterministic multi-target forecasts on an origin-inclusive calendar."""

    _forecast_dates_include_origin = True
    _supports_quantiles = True

    def _fit(self, y, X=None, **kwargs):
        self._forecast_values = y.iloc[-1].to_numpy(dtype=float)
        return self

    def _forecast(
        self,
        steps=1,
        X=None,
        y=None,
        forecast_origin=None,
        quantiles=None,
        **kwargs,
    ):
        dates = self._forecast_dates(forecast_origin, steps)
        point = np.tile(self._forecast_values, (steps, 1))
        if quantiles is None:
            return pd.DataFrame(point, index=dates, columns=self.y.columns)
        rows = [
            {
                "date": date,
                "variable": variable,
                "quantile": probability,
                "value": value + probability,
            }
            for date, values in zip(dates, point, strict=True)
            for variable, value in zip(self.y.columns, values, strict=True)
            for probability in quantiles
        ]
        return pd.DataFrame(rows)


class _CorruptQuantileForecastOLS(rt.models.ForecastOLS):
    """Corrupt raw quantile output from the forecast hook."""

    corruption = None

    def _forecast(self, *args, quantiles=None, **kwargs):
        forecast = super()._forecast(*args, quantiles=quantiles, **kwargs)
        if quantiles is None or self.corruption is None:
            return forecast

        forecast = forecast.copy()
        probabilities = sorted(forecast["quantile"].unique())
        first_date = forecast["date"].iloc[0]
        first_variable = forecast["variable"].iloc[0]
        first_key = (forecast["date"] == first_date) & (
            forecast["variable"] == first_variable
        )
        if self.corruption == "nonfinite":
            forecast.loc[
                first_key & (forecast["quantile"] == probabilities[0]), "value"
            ] = np.inf
        elif self.corruption == "crossing":
            forecast.loc[
                first_key & (forecast["quantile"] == probabilities[0]), "value"
            ] = 1.0
            forecast.loc[
                first_key & (forecast["quantile"] == probabilities[-1]), "value"
            ] = -1.0
        elif self.corruption == "duplicate":
            forecast = pd.concat([forecast, forecast.iloc[[0]]], ignore_index=True)
        elif self.corruption == "missing":
            forecast = forecast.iloc[1:].reset_index(drop=True)
        else:
            raise AssertionError(f"Unknown corruption: {self.corruption}")
        return forecast


def _realtime_outturns():
    dates = pd.date_range("2018-01-31", periods=48, freq="ME")
    vintages = pd.to_datetime(["2021-06-30", "2021-07-31", "2021-08-31"])
    values = 100.0 + 0.75 * np.arange(len(dates))
    rows = []
    for vintage_number, vintage in enumerate(vintages):
        available_dates = dates[dates <= vintage]
        for date, value in zip(
            available_dates, values[: len(available_dates)], strict=True
        ):
            rows.append(
                {
                    "date": date,
                    "variable": "target",
                    "vintage_date": vintage,
                    "frequency": "M",
                    "value": value + 0.2 * vintage_number,
                    "metric": "levels",
                }
            )
    return pd.DataFrame(rows)


def _realtime_data():
    return fe.ForecastData(
        outturns_data=_realtime_outturns(),
        metric="levels",
        compute_levels=False,
        data_check=False,
    )


def _realtime_multitarget_data():
    outturns = _realtime_outturns()
    other = outturns.loc[outturns["date"] < outturns["vintage_date"]].copy()
    other["variable"] = "other"
    other["value"] += 10.0
    return fe.ForecastData(
        outturns_data=pd.concat([outturns, other], ignore_index=True),
        metric="levels",
        compute_levels=False,
        data_check=False,
    )


def _realtime_options(**overrides):
    options = {
        "y_variables": ["target"],
        "data_transformation": {"target": "levels"},
        "steps": 3,
        "first_forecast_horizon": 0,
        "first_vintage": "2021-06-30",
        "last_vintage": "2021-08-31",
    }
    options.update(overrides)
    return options


def _caller_forecasts():
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2021-06-30")],
            "vintage_date": [pd.Timestamp("2021-06-30")],
            "forecast_horizon": [0],
            "variable": ["target"],
            "value": [999.0],
            "metric": ["levels"],
            "source": ["caller"],
            "frequency": ["M"],
        }
    )


def test_realtime_rejects_bridge_density_before_fitting():
    runner = rt.RealTimeModel(_realtime_data(), rt.models.ForecastBridgeOLS())
    with pytest.raises(ValueError, match="does not support quantile"):
        runner.forecast(**_realtime_options(), quantiles=True)
    assert runner.quantiles is None
    assert runner.data.forecasts.empty


@pytest.mark.parametrize("parallel", [False, True])
def test_realtime_rejected_density_model_keeps_other_models(
    parallel, inline_executor
):
    runner = rt.RealTimeModel(
        _realtime_data(),
        [
            rt.models.ForecastOLS(label="fixed"),
            rt.models.ForecastOLS(label="direct", forecast_strategy="direct", steps=3),
        ],
    )
    original = runner.data.forecasts.copy(deep=True)
    with pytest.warns(UserWarning, match="Model 'direct' failed"):
        runner.forecast(**_realtime_options(parallel=parallel), quantiles=True)
    assert runner.quantiles is not None
    assert not runner.quantiles.empty
    assert set(runner.quantiles["source"]) == {"fixed"}
    pd.testing.assert_frame_equal(runner.data.forecasts, original)


def test_realtime_density_with_no_emitted_rows_raises_no_forecasts_error():
    empty = ForecastRunResult(pd.DataFrame(), None, False)
    with pytest.raises(ValueError, match="No forecasts could be produced"):
        rt.RealTimeModel._aggregate_forecast_results(
            [empty],
            outturns=pd.DataFrame(),
            y_variables=["target"],
            X_variables=None,
            reconstruct_levels=False,
            first_vintage="2021-06-30",
            last_vintage="2021-08-31",
            quantiles=True,
        )


def _regression_data(n_train=12, n_future=3):
    index = pd.date_range("2020-01-31", periods=n_train + n_future, freq="ME")
    regressor_values = np.linspace(-1.5, 1.5, len(index))
    noise = np.random.default_rng(20260922).normal(0, 0.35, len(index))
    target_values = 2.5 + 1.7 * regressor_values + noise
    target = pd.DataFrame({"target": target_values[:n_train]}, index=index[:n_train])
    regressors = pd.DataFrame({"x": regressor_values[:n_train]}, index=index[:n_train])
    future = pd.DataFrame({"x": regressor_values[n_train:]}, index=index[n_train:])
    return target, regressors, future


def _quantile_rows():
    dates = pd.date_range("2021-01-31", periods=2, freq="ME")
    variables = ["target", "other"]
    probabilities = (0.1, 0.9)
    rows = [
        {
            "date": date,
            "variable": variable,
            "quantile": probability,
            "value": date_number + variable_number + probability,
        }
        for date_number, date in enumerate(dates)
        for variable_number, variable in enumerate(variables)
        for probability in probabilities
    ]
    return dates, variables, probabilities, pd.DataFrame(rows)


def _assert_matches_residual_se_quantiles(
    model,
    future,
    training_target,
    training_design,
    future_design,
    probabilities=(0.05, 0.5, 0.95),
):
    result = model.forecast(
        steps=len(future), X=future, quantiles=list(reversed(probabilities))
    )
    assert list(result.columns) == ["date", "variable", "quantile", "value"]
    assert result["quantile"].drop_duplicates().tolist() == list(probabilities)

    actual = result.pivot(index="date", columns="quantile", values="value")
    fitted = sm.OLS(
        training_target,
        sm.add_constant(training_design, has_constant="add"),
    ).fit()
    residual_se = np.sqrt(np.sum(fitted.resid**2) / fitted.df_resid)
    np.testing.assert_allclose(model._std_error, residual_se)
    expected = np.asarray(
        fitted.predict(sm.add_constant(future_design, has_constant="add"))
    )[:, None] + residual_se * stats.norm.ppf(probabilities)

    assert actual.index.equals(future.index)
    np.testing.assert_allclose(
        actual[probabilities[0]], expected[:, 0], rtol=1e-10, atol=1e-9
    )
    np.testing.assert_allclose(
        actual[probabilities[-1]], expected[:, -1], rtol=1e-10, atol=1e-9
    )
    np.testing.assert_allclose(actual[0.5], expected[:, 1], rtol=1e-10, atol=1e-9)


@pytest.mark.parametrize(
    "quantiles",
    [
        [],
        [True, 0.5],
        [np.bool_(False), 0.5],
        [0.2, 0.2],
        [np.nan],
        [np.inf],
        [-np.inf],
        [0.0],
        [1.0],
    ],
)
def test_normalise_quantiles_rejects_invalid_probabilities(quantiles):
    with pytest.raises(ValueError):
        _normalise_quantiles(quantiles)


def test_normalise_quantiles_sorts_sequences_and_defaults():
    assert _normalise_quantiles(False) is None
    assert _normalise_quantiles(True) == (0.16, 0.5, 0.84)
    assert _normalise_quantiles([0.84, 0.16, 0.5]) == (0.16, 0.5, 0.84)


def test_forecast_result_reorders_complete_quantile_keys():
    dates, variables, probabilities, expected = _quantile_rows()
    shuffled = expected.sample(frac=1, random_state=7)

    result = ForecastResult(
        shuffled,
        expected_columns=variables,
        steps=len(dates),
        forecast_origin=pd.Timestamp("2020-12-31"),
        quantiles=probabilities,
        forecast_dates=dates,
    )

    pd.testing.assert_frame_equal(result.forecast, expected)


@pytest.mark.parametrize(
    ("operation", "dimension"),
    [
        ("missing", "date"),
        ("missing", "variable"),
        ("missing", "quantile"),
        ("duplicate", None),
        ("extra", "date"),
        ("extra", "variable"),
        ("extra", "quantile"),
    ],
)
def test_forecast_result_rejects_invalid_quantile_key_coverage(operation, dimension):
    dates, variables, probabilities, rows = _quantile_rows()
    if operation == "missing":
        missing_values = {
            "date": dates[0],
            "variable": variables[0],
            "quantile": probabilities[0],
        }
        invalid = rows.loc[rows[dimension] != missing_values[dimension]]
    elif operation == "duplicate":
        invalid = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    else:
        extra_values = {
            "date": dates[-1] + pd.offsets.MonthEnd(),
            "variable": "unexpected",
            "quantile": 0.5,
        }
        extra = rows.iloc[[0]].copy()
        extra[dimension] = extra_values[dimension]
        invalid = pd.concat([rows, extra], ignore_index=True)

    with pytest.raises(ValueError, match="Quantile"):
        ForecastResult(
            invalid,
            expected_columns=variables,
            steps=len(dates),
            forecast_origin=pd.Timestamp("2020-12-31"),
            quantiles=probabilities,
            forecast_dates=dates,
        )


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_forecast_result_rejects_nonfinite_quantile_values(invalid_value):
    dates, variables, probabilities, rows = _quantile_rows()
    rows.loc[0, "value"] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        ForecastResult(
            rows,
            expected_columns=variables,
            steps=len(dates),
            forecast_origin=pd.Timestamp("2020-12-31"),
            quantiles=probabilities,
            forecast_dates=dates,
        )


def test_forecast_result_rejects_crossing_quantile_values():
    dates, variables, probabilities, rows = _quantile_rows()
    rows.loc[rows["quantile"] == probabilities[0], "value"] = 2.0
    rows.loc[rows["quantile"] == probabilities[-1], "value"] = 1.0

    with pytest.raises(ValueError, match="must not cross"):
        ForecastResult(
            rows,
            expected_columns=variables,
            steps=len(dates),
            forecast_origin=pd.Timestamp("2020-12-31"),
            quantiles=probabilities,
            forecast_dates=dates,
        )


@pytest.mark.parametrize("decomp", [False, True])
def test_forecast_result_rejects_quantile_decomposition(decomp):
    dates, variables, probabilities, rows = _quantile_rows()

    with pytest.raises(ValueError, match="decomp=True"):
        ForecastResult(
            rows,
            expected_columns=variables,
            steps=len(dates),
            forecast_origin=pd.Timestamp("2020-12-31"),
            decomposition=None if decomp else pd.DataFrame(),
            quantiles=probabilities,
            forecast_dates=dates,
            decomp=decomp,
        )


@pytest.mark.parametrize(
    ("forecast_dates", "error_type", "message"),
    [
        (pd.Index(["2021-01-31", "2021-02-28"]), TypeError, "DatetimeIndex"),
        (
            pd.date_range("2021-02-28", periods=2, freq="ME"),
            ValueError,
            "cover every",
        ),
    ],
)
def test_forecast_result_requires_exact_quantile_calendar(
    forecast_dates, error_type, message
):
    dates, variables, probabilities, rows = _quantile_rows()

    with pytest.raises(error_type, match=message):
        ForecastResult(
            rows,
            expected_columns=variables,
            steps=len(dates),
            forecast_origin=pd.Timestamp("2020-12-31"),
            quantiles=probabilities,
            forecast_dates=forecast_dates,
        )


def test_ols_density_defaults_are_sorted_and_point_mode_remains_separate():
    target, regressors, future = _regression_data()
    model = rt.models.ForecastOLS().fit(target, X=regressors)

    point = model.forecast(steps=len(future), X=future)
    default = model.forecast(steps=len(future), X=future, quantiles=True)
    custom = model.forecast(steps=len(future), X=future, quantiles=[0.84, 0.16, 0.5])

    assert list(default["quantile"].drop_duplicates()) == [0.16, 0.5, 0.84]
    assert list(custom["quantile"].drop_duplicates()) == [0.16, 0.5, 0.84]
    assert list(point.columns) == ["date", "variable", "value"]
    assert "quantile" not in point.columns


def test_ols_density_is_deterministic():
    target, regressors, future = _regression_data()
    first = rt.models.ForecastOLS().fit(target, X=regressors)
    second = rt.models.ForecastOLS().fit(target, X=regressors)
    first_result = first.forecast(steps=len(future), X=future, quantiles=True)
    pd.testing.assert_frame_equal(
        first_result,
        second.forecast(steps=len(future), X=future, quantiles=True),
    )


def test_ols_rank_deficient_density_uses_residual_se_and_preserves_point_forecast():
    target, regressors, future = _regression_data()
    rank_deficient = regressors.assign(duplicate=regressors["x"])
    future_rank_deficient = future.assign(duplicate=future["x"])
    model = rt.models.ForecastOLS().fit(target, X=rank_deficient)
    expected_point = model.forecast(steps=len(future), X=future_rank_deficient)

    density = model.forecast(
        steps=len(future), X=future_rank_deficient, quantiles=[0.1, 0.9]
    )

    assert np.isfinite(density["value"]).all()
    pd.testing.assert_frame_equal(
        model.forecast(steps=len(future), X=future_rank_deficient), expected_point
    )


def test_ols_nonpositive_residual_degrees_of_freedom_preserves_point_forecast():
    index = pd.date_range("2020-01-31", periods=3, freq="ME")
    target = pd.DataFrame({"target": [1.0, 2.0]}, index=index[:2])
    regressors = pd.DataFrame({"x": [0.0, 1.0]}, index=index[:2])
    future = pd.DataFrame({"x": [2.0]}, index=index[2:])
    model = rt.models.ForecastOLS().fit(target, X=regressors)
    expected_point = model.forecast(steps=1, X=future)

    with pytest.raises(ValueError, match="positive residual degrees of freedom"):
        model.forecast(steps=1, X=future, quantiles=[0.1, 0.9])

    pd.testing.assert_frame_equal(model.forecast(steps=1, X=future), expected_point)


@pytest.mark.parametrize(
    ("model_kwargs", "fit_kwargs"),
    [
        ({}, {"y_lags": 1}),
        ({"forecast_strategy": "direct", "steps": 2}, {}),
    ],
)
def test_ols_density_rejects_target_lags_and_direct_strategy(model_kwargs, fit_kwargs):
    target, regressors, future = _regression_data()
    model = rt.models.ForecastOLS(**model_kwargs).fit(target, X=regressors, **fit_kwargs)

    with pytest.raises(ValueError, match="target lags or direct strategies"):
        model.forecast(steps=2, X=future, quantiles=[0.1, 0.9])


@pytest.mark.parametrize(
    ("model_name", "model_kwargs"),
    [
        ("ForecastRidge", {"alpha": 0.1}),
        ("ForecastLasso", {"alpha": 0.01}),
        ("ForecastElasticNet", {"alpha": 0.01, "l1_ratio": 0.5}),
    ],
)
def test_regularised_models_reject_density(model_name, model_kwargs):
    pytest.importorskip("sklearn")
    target, regressors, future = _regression_data()
    model = getattr(rt.models, model_name)(**model_kwargs).fit(target, X=regressors)

    assert not model.forecast(steps=len(future), X=future).empty
    with pytest.raises(ValueError, match="does not support quantile forecasts"):
        model.forecast(steps=len(future), X=future, quantiles=[0.1, 0.5, 0.9])


def _identity_component(components):
    return components["leaf"]


def test_forecast_tree_rejects_density():
    target, regressors, future = _regression_data()
    leaf = rt.models.ForecastOLS(label="leaf")
    tree = rt.ForecastTree(
        rt.TreeNode(
            transform=_identity_component,
            children=[leaf],
            name="root",
            target="target",
        )
    ).fit(target, X=regressors)

    with pytest.raises(ValueError, match="does not support quantile forecasts"):
        tree.forecast(steps=len(future), X=future, quantiles=[0.1, 0.9])


def test_ols_zero_residual_variance_produces_degenerate_quantiles():
    index = pd.date_range("2020-01-31", periods=10, freq="ME")
    regressor_values = np.linspace(-2.0, 2.0, len(index))
    target = pd.DataFrame({"target": 4.0 + 2.0 * regressor_values[:7]}, index=index[:7])
    regressors = pd.DataFrame({"x": regressor_values[:7]}, index=index[:7])
    future = pd.DataFrame({"x": regressor_values[7:]}, index=index[7:])
    model = rt.models.ForecastOLS().fit(target, X=regressors)

    point = model.forecast(steps=len(future), X=future)
    density = model.forecast(steps=len(future), X=future, quantiles=[0.1, 0.5, 0.9])
    values = density.pivot(index="date", columns="quantile", values="value")

    np.testing.assert_allclose(
        values.to_numpy(), np.repeat(point["value"].to_numpy()[:, None], 3, axis=1)
    )


def test_ols_forecast_accepts_an_explicit_forecast_context():
    target, regressors, future = _regression_data()
    model = rt.models.ForecastOLS().fit(target, X=regressors)
    context = rt.ForecastContext(
        y_history=target,
        X_history=regressors,
        X_conditioning=future,
        forecast_origin=target.index[-1],
    )
    probabilities = [0.1, 0.5, 0.9]

    expected = model.forecast(steps=len(future), X=future, quantiles=probabilities)
    result = model.forecast(context=context, steps=len(future), quantiles=probabilities)

    pd.testing.assert_frame_equal(result, expected)


def test_ols_formula_selection_matches_statsmodels_prediction_intervals():
    target, regressors, future = _regression_data()
    training_design = regressors.rename(columns={"x": "used"})
    training_design["unused"] = np.linspace(2.0, 3.0, len(training_design))
    future_design = future.rename(columns={"x": "used"})
    future_design["unused"] = np.linspace(3.0, 3.5, len(future_design))
    model = rt.models.ForecastOLS(formula="target ~ used").fit(target, X=training_design)

    _assert_matches_residual_se_quantiles(
        model,
        future_design,
        target["target"],
        training_design[["used"]],
        future_design[["used"]],
    )


def test_ols_dummy_selection_matches_statsmodels_prediction_intervals():
    target, regressors, future = _regression_data()
    dummy_date = target.index[4]
    training_dummy = (regressors.index == dummy_date).astype(float)
    future_dummy = (future.index == dummy_date).astype(float)
    training_design = regressors.assign(event=training_dummy)
    future_design = future.assign(event=future_dummy)
    model = rt.models.ForecastOLS().fit(
        target, X=regressors, dummies={"event": dummy_date}
    )

    _assert_matches_residual_se_quantiles(
        model,
        future,
        target["target"],
        training_design,
        future_design,
    )


def test_ols_missing_row_selection_matches_statsmodels_prediction_intervals():
    target, regressors, future = _regression_data()
    target_with_missing = target.copy()
    regressors_with_missing = regressors.copy()
    target_with_missing.iloc[2, 0] = np.nan
    regressors_with_missing.iloc[7, 0] = np.nan
    complete = (
        target_with_missing["target"].notna() & regressors_with_missing["x"].notna()
    )
    model = rt.models.ForecastOLS(drop_nans=True).fit(
        target_with_missing, X=regressors_with_missing
    )

    _assert_matches_residual_se_quantiles(
        model,
        future,
        target_with_missing.loc[complete, "target"],
        regressors_with_missing.loc[complete],
        future,
    )


def _sorted_realtime_rows(frame, columns):
    return (
        frame.loc[:, columns]
        .sort_values(["vintage_date", "date", "variable"])
        .reset_index(drop=True)
    )


def _sorted_quantile_rows(frame):
    return frame.sort_values(
        ["vintage_date", "date", "variable", "quantile"]
    ).reset_index(drop=True)


def test_realtime_density_preserves_caller_forecasts_and_does_not_pass_quantiles_to_fit():
    caller_data = fe.ForecastData(
        outturns_data=_realtime_outturns(),
        forecasts_data=_caller_forecasts(),
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    caller_before = caller_data.forecasts.copy(deep=True)
    _FitQuantileSpyOLS.fit_kwargs = []
    runner = rt.RealTimeModel(
        data=caller_data,
        models=_FitQuantileSpyOLS(label="density_spy"),
    )

    runner.forecast(**_realtime_options(quantiles=True))

    pd.testing.assert_frame_equal(caller_data.forecasts, caller_before)
    assert runner.data.forecasts.equals(caller_before)
    assert runner.quantiles is not None
    assert runner.quantiles["vintage_date"].nunique() == 3
    assert list(runner.quantiles["quantile"].drop_duplicates()) == [0.16, 0.5, 0.84]
    assert set(runner.quantiles["metric"]) == {"levels"}
    assert runner.native_forecasts is None
    assert runner.decompositions is None
    assert _FitQuantileSpyOLS.fit_kwargs
    assert all("quantiles" not in kwargs for kwargs in _FitQuantileSpyOLS.fit_kwargs)


def test_realtime_quantile_median_matches_point_metadata_and_calendar():
    point_data = _realtime_data()
    point_runner = rt.RealTimeModel(data=point_data, models=rt.models.ForecastOLS())
    point_runner.forecast(**_realtime_options(first_forecast_horizon=1))

    density_runner = rt.RealTimeModel(
        data=_realtime_data(), models=rt.models.ForecastOLS()
    )
    density_runner.forecast(
        **_realtime_options(first_forecast_horizon=1, quantiles=[0.1, 0.5, 0.9])
    )

    metadata = [
        "date",
        "vintage_date",
        "forecast_horizon",
        "variable",
        "metric",
        "source",
        "frequency",
    ]
    point = point_data.forecasts.query("source == 'ForecastOLS' and metric == 'levels'")
    median = density_runner.quantiles.query("quantile == 0.5")
    pd.testing.assert_frame_equal(
        _sorted_realtime_rows(point, metadata),
        _sorted_realtime_rows(median, metadata),
        check_dtype=False,
    )
    point_values = _sorted_realtime_rows(point, metadata + ["value"])
    median_values = _sorted_realtime_rows(median, metadata + ["value"])
    np.testing.assert_allclose(point_values["value"], median_values["value"])


def test_realtime_origin_inclusive_density_preserves_point_horizons():
    probabilities = [0.1, 0.5, 0.9]
    point_data = _realtime_data()
    point_runner = rt.RealTimeModel(
        data=point_data,
        models=_OriginInclusiveOLS(label="origin"),
    )
    point_runner.forecast(**_realtime_options(first_forecast_horizon=None))

    density_runner = rt.RealTimeModel(
        data=_realtime_data(),
        models=_OriginInclusiveOLS(label="origin"),
    )
    density_runner.forecast(
        **_realtime_options(
            first_forecast_horizon=None,
            quantiles=probabilities,
        )
    )

    columns = [
        "date",
        "vintage_date",
        "forecast_horizon",
        "variable",
        "metric",
        "source",
        "frequency",
    ]
    point = point_data.forecasts.query("source == 'origin' and metric == 'levels'")
    median = density_runner.quantiles.query("quantile == 0.5")
    pd.testing.assert_frame_equal(
        _sorted_realtime_rows(point, columns),
        _sorted_realtime_rows(median, columns),
        check_dtype=False,
    )
    assert set(point["forecast_horizon"]) == {1, 2}
    assert set(median["forecast_horizon"]) == {1, 2}
    assert set(density_runner.quantiles["quantile"]) == set(probabilities)


def test_realtime_origin_inclusive_density_masks_targets_independently():
    probabilities = [0.1, 0.5, 0.9]
    options = _realtime_options(
        y_variables=["target", "other"],
        data_transformation={"target": "levels", "other": "levels"},
        first_forecast_horizon=None,
    )
    point_data = _realtime_multitarget_data()
    rt.RealTimeModel(
        data=point_data,
        models=_OriginInclusiveMultiModel(label="multi"),
    ).forecast(**options)

    density_runner = rt.RealTimeModel(
        data=_realtime_multitarget_data(),
        models=_OriginInclusiveMultiModel(label="multi"),
    )
    density_runner.forecast(**(options | {"quantiles": probabilities}))

    columns = [
        "date",
        "vintage_date",
        "forecast_horizon",
        "variable",
        "metric",
        "source",
        "frequency",
    ]
    point = point_data.forecasts.query("metric == 'levels'")
    median = density_runner.quantiles.query("quantile == 0.5")
    pd.testing.assert_frame_equal(
        _sorted_realtime_rows(point, columns),
        _sorted_realtime_rows(median, columns),
        check_dtype=False,
    )
    assert set(point.loc[point["variable"] == "target", "forecast_horizon"]) == {2}
    assert set(point.loc[point["variable"] == "other", "forecast_horizon"]) == {1, 2}


@pytest.mark.parametrize(
    ("first_forecast_horizon", "minimum_emitted_horizon"),
    [(None, 1), (2, 2)],
)
def test_realtime_density_masks_published_rows_and_applies_cutoffs(
    first_forecast_horizon, minimum_emitted_horizon
):
    probabilities = [0.1, 0.5, 0.9]
    runner = rt.RealTimeModel(data=_realtime_data(), models=rt.models.ForecastOLS())
    runner.forecast(
        **_realtime_options(
            first_forecast_horizon=first_forecast_horizon,
            quantiles=probabilities,
        )
    )

    quantiles = runner.quantiles
    assert quantiles is not None
    horizons = [
        (pd.Period(date, freq="M") - pd.Period(vintage, freq="M")).n
        for date, vintage in zip(
            quantiles["date"], quantiles["vintage_date"], strict=True
        )
    ]
    assert min(horizons) >= minimum_emitted_horizon
    assert set(quantiles["quantile"]) == set(probabilities)

    grouped = quantiles.groupby(["vintage_date", "date", "variable"], sort=False)
    assert grouped.ngroups > 0
    assert all(sorted(rows["quantile"].tolist()) == probabilities for _, rows in grouped)


@pytest.mark.parametrize("transformation", ["logs", "diff", "pop"])
def test_realtime_density_keeps_native_metric_without_reconstruction(transformation):
    data = _realtime_data()
    forecasts_before = data.forecasts.copy(deep=True)
    runner = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())

    runner.forecast(
        **_realtime_options(
            data_transformation={"target": transformation},
            first_forecast_horizon=1,
            reconstruct_levels=True,
            quantiles=[0.1, 0.5, 0.9],
        )
    )

    pd.testing.assert_frame_equal(data.forecasts, forecasts_before)
    assert data.forecasts.empty
    assert runner.native_forecasts is None
    assert runner.quantiles is not None
    assert set(runner.quantiles["metric"]) == {transformation}
    assert np.isfinite(runner.quantiles["value"]).all()


def test_realtime_density_repeat_and_point_transitions_preserve_forecast_state():
    data = fe.ForecastData(
        outturns_data=_realtime_outturns(),
        forecasts_data=_caller_forecasts(),
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    caller_before = data.forecasts.copy(deep=True)
    runner = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())
    density_options = _realtime_options(
        first_forecast_horizon=1, quantiles=[0.1, 0.5, 0.9]
    )

    runner.forecast(**density_options)
    first_density = runner.quantiles.copy(deep=True)
    runner.forecast(**density_options)
    pd.testing.assert_frame_equal(
        _sorted_quantile_rows(runner.quantiles),
        _sorted_quantile_rows(first_density),
    )
    pd.testing.assert_frame_equal(data.forecasts, caller_before)

    runner.forecast(**_realtime_options(first_forecast_horizon=1))
    point_forecasts = data.forecasts.copy(deep=True)
    assert (point_forecasts["source"] == "caller").any()
    assert (point_forecasts["source"] == "ForecastOLS").any()

    runner.forecast(**density_options)
    pd.testing.assert_frame_equal(data.forecasts, point_forecasts)
    pd.testing.assert_frame_equal(
        _sorted_quantile_rows(runner.quantiles),
        _sorted_quantile_rows(first_density),
    )


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ({"quantiles": [0.1, 0.1]}, "distinct"),
        ({"quantiles": True, "decomp": True}, "decomp=True"),
        ({"quantiles": True, "steps": 0}, "positive integer"),
    ],
)
def test_realtime_density_invalid_requests_leave_state_unchanged(invalid, message):
    data = fe.ForecastData(
        outturns_data=_realtime_outturns(),
        forecasts_data=_caller_forecasts(),
        metric="levels",
        compute_levels=False,
        data_check=False,
    )
    runner = rt.RealTimeModel(data=data, models=rt.models.ForecastOLS())
    runner.forecast(**_realtime_options(quantiles=[0.1, 0.5, 0.9]))
    quantiles_before = runner.quantiles.copy(deep=True)
    forecasts_before = data.forecasts.copy(deep=True)

    request = _realtime_options(quantiles=[0.1, 0.5, 0.9])
    request.update(invalid)
    with pytest.raises(ValueError, match=message):
        runner.forecast(**request)

    pd.testing.assert_frame_equal(runner.quantiles, quantiles_before)
    pd.testing.assert_frame_equal(data.forecasts, forecasts_before)
    assert runner.native_forecasts is None
    assert runner.decompositions is None


def test_realtime_density_sequential_matches_spawned_parallel_execution():
    probabilities = [0.1, 0.5, 0.9]
    sequential = rt.RealTimeModel(data=_realtime_data(), models=rt.models.ForecastOLS())
    sequential.forecast(**_realtime_options(quantiles=probabilities, parallel=False))

    parallel = rt.RealTimeModel(data=_realtime_data(), models=rt.models.ForecastOLS())
    parallel.forecast(
        **_realtime_options(
            quantiles=probabilities,
            parallel=True,
            batch_size=1,
            max_workers=2,
        )
    )

    pd.testing.assert_frame_equal(
        _sorted_quantile_rows(parallel.quantiles),
        _sorted_quantile_rows(sequential.quantiles),
    )
    assert parallel.data.forecasts.empty


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("nonfinite", "finite"),
        ("crossing", "must not cross"),
        ("duplicate", "unique"),
        ("missing", "cover every"),
    ],
)
@pytest.mark.parametrize("parallel", [False, True])
def test_realtime_rejects_invalid_raw_quantile_forecast(
    corruption, message, parallel, inline_executor
):
    probabilities = [0.1, 0.5, 0.9]
    data = _realtime_data()
    model = _CorruptQuantileForecastOLS(label="override")
    runner = rt.RealTimeModel(data=data, models=model)

    runner.forecast(**_realtime_options(first_forecast_horizon=1))
    point_before = data.forecasts.copy(deep=True)
    runner.forecast(
        **_realtime_options(
            first_forecast_horizon=1,
            quantiles=probabilities,
        )
    )
    quantiles_before = runner.quantiles.copy(deep=True)

    model.corruption = corruption
    options = _realtime_options(
        first_forecast_horizon=1,
        quantiles=probabilities,
        parallel=parallel,
    )
    if parallel:
        options.update(batch_size=1, max_workers=2)
    with pytest.raises(ValueError, match=message):
        runner.forecast(**options)

    pd.testing.assert_frame_equal(runner.quantiles, quantiles_before)
    pd.testing.assert_frame_equal(data.forecasts, point_before)
