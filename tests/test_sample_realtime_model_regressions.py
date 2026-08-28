"""End-to-end regressions for generated realtime data and non-optional models."""

import copy
import pickle

import numpy as np
import pandas as pd
from forecast_evaluation import ForecastData, NowcastData

import forecast_realtime as rt
from forecast_realtime.forecast_tree import ForecastTree, TreeNode

_RESULT_COLUMNS = [
    "date",
    "vintage_date",
    "forecast_horizon",
    "variable",
    "metric",
    "frequency",
    "source",
    "value",
]
_KEY_COLUMNS = [
    "date",
    "vintage_date",
    "forecast_horizon",
    "variable",
    "metric",
    "frequency",
    "source",
]


def levels_only(panel, variables):
    """Select independent levels rows for the requested variables."""
    return panel.loc[
        panel["variable"].isin(variables) & panel["metric"].eq("levels")
    ].copy()


def native_sources(panel, source_metrics):
    """Select one stored source metric independently for each variable."""
    mask = pd.Series(False, index=panel.index)
    for variable, metric in source_metrics.items():
        mask |= panel["variable"].eq(variable) & panel["metric"].eq(metric)
    return panel.loc[mask].copy()


def wide_vintage_levels(panel, variables, vintage):
    """Return one exact vintage's levels observations in wide form."""
    selected = levels_only(panel, variables)
    selected = selected.loc[selected["vintage_date"].eq(pd.Timestamp(vintage))]
    return selected.pivot(index="date", columns="variable", values="value")[variables]


def forecast_data(panel):
    """Create a fresh mutable ForecastData container for a source view."""
    return ForecastData(
        outturns_data=copy.deepcopy(panel),
        compute_levels=False,
        data_check=False,
    )


def native_result(realtime_model, label, metric):
    """Return one model's native output, regardless of ForecastData storage."""
    if metric in {"levels", "pop", "yoy"}:
        forecasts = realtime_model.data.forecasts
    else:
        forecasts = realtime_model.native_forecasts

    if forecasts is None or forecasts.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    return forecasts.loc[
        forecasts["source"].eq(label) & forecasts["metric"].eq(metric)
    ].copy()


def assert_forecasts_equal(left, right, *, rtol=1e-10, atol=1e-12):
    """Compare forecast metadata and values after enforcing uniqueness."""
    left = left[_RESULT_COLUMNS].copy()
    right = right[_RESULT_COLUMNS].copy()
    for frame in (left, right):
        assert np.isfinite(frame["value"].to_numpy(dtype=float)).all()
        assert not frame.duplicated(_KEY_COLUMNS).any()

    left = left.sort_values(_KEY_COLUMNS).reset_index(drop=True)
    right = right.sort_values(_KEY_COLUMNS).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left[_KEY_COLUMNS], right[_KEY_COLUMNS], check_dtype=False
    )
    np.testing.assert_allclose(
        left["value"].to_numpy(dtype=float),
        right["value"].to_numpy(dtype=float),
        rtol=rtol,
        atol=atol,
    )


def snapshot_forecasts(frame):
    """Normalise forecast records for a compact, deterministic snapshot."""
    result = frame[_RESULT_COLUMNS].sort_values(_KEY_COLUMNS).reset_index(drop=True)
    for column in ("date", "vintage_date"):
        result[column] = result[column].dt.strftime("%Y-%m-%d")
    result["value"] = result["value"].round(10)
    return result.to_dict(orient="records")


def _monthly_ols_models():
    return [
        rt.models.ForecastOLS(
            label="ols_levels",
            formula="monthly_1 ~ monthly_2",
            data_transformation={
                "monthly_1": "levels",
                "monthly_2": "levels",
            },
        ),
        rt.models.ForecastOLS(
            label="ols_pop",
            formula="monthly_1 ~ monthly_2",
            data_transformation={"monthly_1": "pop", "monthly_2": "pop"},
        ),
    ]


def _run_monthly_ols(panel):
    realtime = rt.RealTimeModel(forecast_data(panel), _monthly_ols_models())
    realtime.forecast(
        y_variables=["monthly_1"],
        X_variables=["monthly_2"],
        data_transformation={"monthly_1": "levels", "monthly_2": "levels"},
        steps=2,
        first_vintage="2024-11-30",
        last_vintage="2024-12-31",
        X_imputation="last",
        reconstruct_levels=False,
    )
    return realtime


def test_monthly_ols_levels_and_pop_match_levels_only_sources(
    sample_realtime_complete, snapshot
):
    original = sample_realtime_complete.copy(deep=True)
    full = _run_monthly_ols(sample_realtime_complete)
    levels = _run_monthly_ols(
        levels_only(sample_realtime_complete, ["monthly_1", "monthly_2"])
    )

    for label, metric in [("ols_levels", "levels"), ("ols_pop", "pop")]:
        full_result = native_result(full, label, metric)
        levels_result = native_result(levels, label, metric)
        assert len(full_result) == len(levels_result) == 4
        assert set(full_result["metric"]) == {metric}
        assert_forecasts_equal(full_result, levels_result)

    all_results = pd.concat(
        [
            native_result(full, "ols_levels", "levels"),
            native_result(full, "ols_pop", "pop"),
        ],
        ignore_index=True,
    )
    assert snapshot_forecasts(all_results) == snapshot
    pd.testing.assert_frame_equal(sample_realtime_complete, original)


def test_direct_monthly_ols_native_logs_match_levels_derived_logs(
    sample_realtime_complete,
):
    variables = ["monthly_1", "monthly_2"]
    levels = wide_vintage_levels(sample_realtime_complete, variables, "2024-11-30")
    assert (levels > 0).all().all()
    logs = np.log(levels)

    levels_model = rt.models.ForecastOLS(
        formula="monthly_1 ~ monthly_2",
        data_transformation={"monthly_1": "logs", "monthly_2": "logs"},
    )
    native_model = rt.models.ForecastOLS(
        formula="monthly_1 ~ monthly_2",
        data_transformation={"monthly_1": "logs", "monthly_2": "logs"},
    )
    levels_model.fit(
        levels[["monthly_1"]],
        levels[["monthly_2"]],
        frequency="M",
        X_imputation="last",
        y_input_metrics={"monthly_1": "levels"},
        X_input_metrics={"monthly_2": "levels"},
    )
    native_model.fit(
        logs[["monthly_1"]],
        logs[["monthly_2"]],
        frequency="M",
        X_imputation="last",
        y_input_metrics={"monthly_1": "logs"},
        X_input_metrics={"monthly_2": "logs"},
    )

    pd.testing.assert_frame_equal(levels_model.y, native_model.y)
    pd.testing.assert_frame_equal(levels_model.X, native_model.X)
    np.testing.assert_allclose(levels_model.beta_, native_model.beta_)
    pd.testing.assert_frame_equal(
        levels_model.fitted_values.to_frame(), native_model.fitted_values.to_frame()
    )
    for model in (levels_model, native_model):
        transformation = model._fitted_model_configuration.data_transformation
        assert dict(transformation.data_transformation) == {
            "monthly_1": "logs",
            "monthly_2": "logs",
        }

    future_index = pd.date_range("2024-12-31", periods=2, freq="ME")
    future_levels = pd.DataFrame(
        np.repeat(levels.iloc[[-1]].to_numpy(), 2, axis=0),
        index=future_index,
        columns=variables,
    )
    levels_forecast = levels_model.forecast(
        steps=2, X=future_levels[["monthly_2"]], X_imputation="last"
    )
    native_forecast = native_model.forecast(
        steps=2, X=np.log(future_levels[["monthly_2"]]), X_imputation="last"
    )
    pd.testing.assert_frame_equal(levels_forecast, native_forecast)


def _run_bridge(panel, pipeline):
    model = rt.models.ForecastBridgeOLS(
        label="bridge",
        formula="quarterly_1 ~ monthly_1 + quarterly_2",
        data_transformation=pipeline,
    )
    realtime = rt.RealTimeModel(forecast_data(panel), model)
    realtime.forecast(
        y_variables=["quarterly_1"],
        X_variables=["monthly_1", "quarterly_2"],
        steps=2,
        first_vintage="2024-09-30",
        last_vintage="2024-09-30",
        X_imputation="last",
        reconstruct_levels=False,
    )
    return realtime


def test_bridge_ols_levels_match_full_and_levels_only_panels(
    sample_realtime_complete,
):
    variables = ["quarterly_1", "monthly_1", "quarterly_2"]
    pipeline = {variable: "levels" for variable in variables}
    full = _run_bridge(sample_realtime_complete, pipeline)
    levels = _run_bridge(levels_only(sample_realtime_complete, variables), pipeline)

    full_result = native_result(full, "bridge", "levels")
    levels_result = native_result(levels, "bridge", "levels")
    assert len(full_result) == len(levels_result) == 2
    assert_forecasts_equal(full_result, levels_result)


def test_bridge_ols_mixed_native_sources_match_levels_derived_sources(
    sample_realtime_complete,
):
    variables = ["quarterly_1", "monthly_1", "quarterly_2"]
    pipeline = {
        "quarterly_1": "logs",
        "monthly_1": "pop",
        "quarterly_2": "levels",
    }
    levels = _run_bridge(levels_only(sample_realtime_complete, variables), pipeline)
    mixed = _run_bridge(
        native_sources(
            sample_realtime_complete,
            {
                "quarterly_1": "levels",
                "monthly_1": "pop",
                "quarterly_2": "levels",
            },
        ),
        pipeline,
    )

    levels_result = native_result(levels, "bridge", "logs")
    mixed_result = native_result(mixed, "bridge", "logs")
    assert len(levels_result) == len(mixed_result) == 2
    assert_forecasts_equal(levels_result, mixed_result)


def test_quarterly_ols_pop_matches_levels_and_native_pop_sources(
    sample_realtime_complete,
):
    """Quarterly ``pop`` uses quarterly periods for derived and native inputs."""
    variables = ["quarterly_1", "quarterly_2"]
    model_kwargs = {
        "label": "quarterly_pop",
        "formula": "quarterly_1 ~ quarterly_2",
        "data_transformation": {
            "quarterly_1": "pop",
            "quarterly_2": "pop",
        },
    }

    levels_run = rt.RealTimeModel(
        forecast_data(levels_only(sample_realtime_complete, variables)),
        rt.models.ForecastOLS(**model_kwargs),
    )
    levels_run.forecast(
        y_variables=["quarterly_1"],
        X_variables=["quarterly_2"],
        steps=2,
        first_vintage="2024-09-30",
        last_vintage="2024-09-30",
        X_imputation="last",
        reconstruct_levels=False,
    )

    native_run = rt.RealTimeModel(
        forecast_data(
            native_sources(
                sample_realtime_complete,
                {variable: "pop" for variable in variables},
            )
        ),
        rt.models.ForecastOLS(**model_kwargs),
    )
    native_run.forecast(
        y_variables=["quarterly_1"],
        X_variables=["quarterly_2"],
        steps=2,
        first_vintage="2024-09-30",
        last_vintage="2024-09-30",
        X_imputation="last",
        reconstruct_levels=False,
    )

    levels_result = native_result(levels_run, "quarterly_pop", "pop")
    native_result_frame = native_result(native_run, "quarterly_pop", "pop")
    assert len(levels_result) == len(native_result_frame) == 2
    assert_forecasts_equal(levels_result, native_result_frame)


def test_bridge_ols_mixed_levels_and_pop_match_derived_sources(
    sample_realtime_complete,
):
    """Mixed monthly/quarterly levels and ``pop`` inputs are equivalent."""
    variables = ["quarterly_1", "monthly_1", "quarterly_2"]
    pipeline = {
        "quarterly_1": "levels",
        "monthly_1": "pop",
        "quarterly_2": "levels",
    }
    levels = _run_bridge(levels_only(sample_realtime_complete, variables), pipeline)
    native = _run_bridge(
        native_sources(
            sample_realtime_complete,
            {
                "quarterly_1": "levels",
                "monthly_1": "pop",
                "quarterly_2": "levels",
            },
        ),
        pipeline,
    )

    levels_result = native_result(levels, "bridge", "levels")
    native_result_frame = native_result(native, "bridge", "levels")
    assert len(levels_result) == len(native_result_frame) == 2
    assert_forecasts_equal(levels_result, native_result_frame)


def _make_monthly_tree():
    levels_leaf = rt.models.ForecastOLS(
        label="levels_leaf",
        formula="monthly_1 ~ monthly_2",
        data_transformation={"monthly_1": "levels", "monthly_2": "levels"},
    )
    logs_leaf = rt.models.ForecastOLS(
        label="logs_leaf",
        formula="monthly_1 ~ monthly_2",
        data_transformation={"monthly_1": "logs", "monthly_2": "logs"},
    )
    stack = rt.models.ForecastOLS(
        label="stack",
        formula="monthly_1 ~ levels_leaf + logs_leaf",
        data_transformation={
            "monthly_1": "levels",
            "levels_leaf": "levels",
            "logs_leaf": "logs",
        },
    )
    return ForecastTree(
        TreeNode(transform=stack, children=[levels_leaf, logs_leaf], name="stack"),
        label="monthly_tree",
    )


def _run_monthly_tree(panel, *, parallel):
    realtime = rt.RealTimeModel(forecast_data(panel), _make_monthly_tree())
    realtime.forecast(
        y_variables=["monthly_1"],
        X_variables=["monthly_2"],
        steps=2,
        first_vintage="2024-11-30",
        last_vintage="2024-12-31",
        y_lags=1,
        X_imputation="last",
        reconstruct_levels=False,
        parallel=parallel,
        max_workers=2 if parallel else None,
        batch_size=1 if parallel else None,
    )
    return realtime


def test_monthly_tree_with_levels_and_logs_is_parallel_equivalent(
    sample_realtime_complete,
):
    tree = _make_monthly_tree()
    pickle.dumps(tree)

    sequential = _run_monthly_tree(sample_realtime_complete, parallel=False)
    process_parallel = _run_monthly_tree(sample_realtime_complete, parallel=True)
    sequential_result = native_result(sequential, "monthly_tree", "levels")
    parallel_result = native_result(process_parallel, "monthly_tree", "levels")

    assert len(sequential_result) == len(parallel_result) == 4
    assert_forecasts_equal(sequential_result, parallel_result)


def _run_monthly_pop_decomposition(panel):
    model = rt.models.ForecastOLS(
        label="ols_pop",
        formula="monthly_1 ~ monthly_2",
        data_transformation={"monthly_1": "pop", "monthly_2": "pop"},
    )
    realtime = rt.RealTimeModel(forecast_data(panel), model)
    realtime.forecast(
        y_variables=["monthly_1"],
        X_variables=["monthly_2"],
        steps=2,
        first_vintage="2024-11-30",
        last_vintage="2024-12-31",
        X_imputation="last",
        reconstruct_levels=False,
        decomp=True,
        parallel=False,
    )
    return realtime


def test_monthly_pop_decomposition_matches_for_native_and_derived_sources(
    sample_realtime_complete,
):
    full = _run_monthly_pop_decomposition(sample_realtime_complete)
    levels = _run_monthly_pop_decomposition(
        levels_only(sample_realtime_complete, ["monthly_1", "monthly_2"])
    )
    full_point = native_result(full, "ols_pop", "pop")
    levels_point = native_result(levels, "ols_pop", "pop")
    assert_forecasts_equal(full_point, levels_point)

    assert full.decompositions is not None
    assert levels.decompositions is not None
    decomp_keys = [
        "date",
        "vintage_date",
        "forecast_horizon",
        "variable",
        "component",
        "decomposition",
        "revision_source",
        "base_vintage_date",
        "forecast_metric",
        "source",
        "frequency",
    ]
    decomp_value_columns = ["contribution", "weight", "news"]
    for frame in (full.decompositions, levels.decompositions):
        assert set(frame["forecast_metric"]) == {"pop"}
        assert np.isfinite(frame["contribution"].to_numpy(dtype=float)).all()
        assert not frame.duplicated(decomp_keys).any()

    full_decomp = full.decompositions.sort_values(decomp_keys).reset_index(drop=True)
    levels_decomp = levels.decompositions.sort_values(decomp_keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        full_decomp[decomp_keys], levels_decomp[decomp_keys], check_dtype=False
    )
    for column in decomp_value_columns:
        np.testing.assert_allclose(
            full_decomp[column].fillna(0).to_numpy(dtype=float),
            levels_decomp[column].fillna(0).to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )

    for decomp in (full.decompositions, levels.decompositions):
        level_decomp = decomp[decomp["decomposition"] == "level"]
        for (vintage, horizon, variable), group in level_decomp.groupby(
            ["vintage_date", "forecast_horizon", "variable"]
        ):
            point = native_result(
                full if decomp is full.decompositions else levels,
                "ols_pop",
                "pop",
            )
            expected = point.loc[
                (point["vintage_date"] == vintage)
                & (point["forecast_horizon"] == horizon)
                & (point["variable"] == variable),
                "value",
            ]
            if not expected.empty:
                np.testing.assert_allclose(group["contribution"].sum(), expected.iloc[0])

    second_vintage = pd.Timestamp("2024-12-31")
    full_second_keys = full.decompositions.loc[
        full.decompositions["vintage_date"].eq(second_vintage), decomp_keys
    ]
    levels_second_keys = levels.decompositions.loc[
        levels.decompositions["vintage_date"].eq(second_vintage), decomp_keys
    ]
    pd.testing.assert_frame_equal(
        full_second_keys.sort_values(decomp_keys).reset_index(drop=True),
        levels_second_keys.sort_values(decomp_keys).reset_index(drop=True),
        check_dtype=False,
    )


def _conditioning_forecasts(panel, metric):
    rows = []
    for vintage in pd.to_datetime(["2024-10-31", "2024-11-30"]):
        source_vintage = vintage - pd.offsets.MonthEnd(1)
        date = vintage + pd.offsets.MonthEnd(1)
        value = (
            panel.loc[
                panel["variable"].eq("monthly_1")
                & panel["metric"].eq(metric)
                & panel["date"].eq(date),
            ]
            .sort_values("vintage_date")["value"]
            .iloc[-1]
        )
        rows.append(
            {
                "date": date,
                "vintage_date": source_vintage,
                "variable": "monthly_1",
                "frequency": "M",
                "value": value,
                "forecast_horizon": 1,
                "source": "sample_conditioning",
                "metric": metric,
            }
        )
    return pd.DataFrame(rows)


def _conditioned_forecast_data(panel, metric, conditioning_panel=None):
    conditioning_panel = panel if conditioning_panel is None else conditioning_panel
    nowcast = NowcastData(
        outturns_data=copy.deepcopy(conditioning_panel),
        forecasts_data=_conditioning_forecasts(conditioning_panel, metric),
        compute_levels=False,
        data_check=False,
    )
    return ForecastData(
        outturns_data=copy.deepcopy(panel),
        forecasts_data=copy.deepcopy(nowcast._raw_forecasts),
        compute_levels=False,
        data_check=False,
    )


def _run_conditioned_pop(panel, conditioning_metric, conditioning_panel=None):
    model = rt.models.ForecastOLS(
        label="conditioned_pop",
        data_transformation={"monthly_1": "pop"},
    )
    realtime = rt.RealTimeModel(
        _conditioned_forecast_data(panel, conditioning_metric, conditioning_panel), model
    )
    realtime.forecast(
        y_variables=["monthly_1"],
        steps=2,
        y_lags=1,
        y_steps_ahead={"monthly_1": 0},
        y_sources={"monthly_1": "sample_conditioning"},
        first_vintage="2024-10-31",
        last_vintage="2024-11-30",
        reconstruct_levels=False,
    )
    return realtime


def test_monthly_pop_conditioning_matches_native_pop_conditioning(
    sample_realtime_complete,
):
    variables = ["monthly_1"]
    levels = levels_only(sample_realtime_complete, variables)
    native = native_sources(sample_realtime_complete, {"monthly_1": "pop"})
    levels_run = _run_conditioned_pop(levels, "levels")
    native_run = _run_conditioned_pop(native, "pop")

    levels_result = native_result(levels_run, "conditioned_pop", "pop")
    native_result_frame = native_result(native_run, "conditioned_pop", "pop")
    assert_forecasts_equal(levels_result, native_result_frame)


def test_monthly_pop_accepts_levels_history_with_native_pop_conditioning(
    sample_realtime_complete,
):
    levels = levels_only(sample_realtime_complete, ["monthly_1"])
    native = native_sources(sample_realtime_complete, {"monthly_1": "pop"})

    mixed_run = _run_conditioned_pop(levels, "pop", sample_realtime_complete)
    native_run = _run_conditioned_pop(native, "pop")

    mixed_result = native_result(mixed_run, "conditioned_pop", "pop")
    native_result_frame = native_result(native_run, "conditioned_pop", "pop")
    assert_forecasts_equal(mixed_result, native_result_frame)


def _run_ragged_bridge(panel, *, parallel):
    variables = ["quarterly_1", "monthly_1", "quarterly_2"]
    model = rt.models.ForecastBridgeOLS(
        label="ragged_bridge",
        formula="quarterly_1 ~ monthly_1 + quarterly_2",
        data_transformation={variable: "levels" for variable in variables},
    )
    realtime = rt.RealTimeModel(forecast_data(panel), model)
    realtime.forecast(
        y_variables=["quarterly_1"],
        X_variables=["monthly_1", "quarterly_2"],
        steps=2,
        first_vintage="2024-12-31",
        last_vintage="2025-01-31",
        X_imputation="last",
        reconstruct_levels=False,
        parallel=parallel,
        max_workers=2 if parallel else None,
        batch_size=1 if parallel else None,
    )
    return realtime


def test_ragged_bridge_ols_is_parallel_equivalent(sample_realtime_ragged):
    sequential = _run_ragged_bridge(sample_realtime_ragged, parallel=False)
    process_parallel = _run_ragged_bridge(sample_realtime_ragged, parallel=True)
    sequential_result = native_result(sequential, "ragged_bridge", "levels")
    parallel_result = native_result(process_parallel, "ragged_bridge", "levels")

    assert not sequential_result.empty
    assert set(sequential_result["vintage_date"]) == set(parallel_result["vintage_date"])
    assert_forecasts_equal(sequential_result, parallel_result)
