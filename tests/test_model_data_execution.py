"""Focused phase-six integration tests for private ``ModelData`` execution."""

from __future__ import annotations

import copy
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

import forecast_realtime as rt
from forecast_realtime._model_data import ModelData, ModelInputRequirements
from forecast_realtime._realtime_forecasting import ForecastTask
from forecast_realtime.real_time_model import _run_forecast_task

_MAPPING = {"target": "levels", "feature": "levels"}
_FORECAST_COLUMNS = [
    "date",
    "vintage_date",
    "forecast_horizon",
    "variable",
    "value",
    "metric",
    "source",
    "frequency",
]
_DECOMPOSITION_KEYS = [
    "date",
    "vintage_date",
    "base_vintage_date",
    "forecast_horizon",
    "variable",
    "decomposition",
    "revision_source",
    "component",
]


def _archive(
    vintages: pd.DatetimeIndex,
    *,
    missing_target_vintage: pd.Timestamp | None = None,
) -> pd.DataFrame:
    dates = pd.date_range("2023-01-31", periods=12, freq="ME")
    rows = []
    for vintage_number, vintage in enumerate(vintages):
        target = 20.0 + np.arange(len(dates)) * 0.4 + vintage_number * 0.2
        feature = 5.0 + np.arange(len(dates)) * 0.15 + vintage_number * 0.1
        if vintage == missing_target_vintage:
            target = np.full(len(dates), np.nan)
        for variable, values in (("target", target), ("feature", feature)):
            rows.extend(
                {
                    "date": date,
                    "variable": variable,
                    "vintage_date": vintage,
                    "frequency": "M",
                    "value": float(value),
                    "metric": "levels",
                    "source": "survey",
                }
                for date, value in zip(dates, values, strict=True)
            )
    return pd.DataFrame(rows)


def _selected_data(
    outturns: pd.DataFrame,
    *,
    y_sources: dict[str, str] | None = None,
    X_sources: dict[str, str] | None = None,
    forecasts: pd.DataFrame | None = None,
) -> ModelData:
    archive = ModelData.from_archive(
        outturns,
        forecasts,
        frequencies={"target": "M", "feature": "M"},
    )
    requirements = (
        ModelInputRequirements(
            consumer="phase-six-ols",
            y=(("target", "levels"),),
            X=(("feature", "levels"),),
            explicit=True,
        ),
    )
    return archive.select(
        requirements,
        y_sources=y_sources,
        X_sources=X_sources,
    )


def _common_options(*, decomp: bool = False) -> dict:
    return {
        "y_steps_ahead": None,
        "X_steps_ahead": None,
        "steps": 2,
        "label": None,
        "first_forecast_horizon": None,
        "frequency": "M",
        "y_lags": 0,
        "X_lags": 0,
        "dummies": None,
        "decomp": decomp,
        "X_imputation": "last",
        "drop_transformation_nans": True,
    }


def _make_tasks(
    data: ModelData,
    vintages: np.ndarray,
    *,
    parallel: bool,
    batch_size: int | None,
    decomp: bool = False,
) -> list[ForecastTask]:
    model = rt.models.ForecastOLS(
        label="phase-six-ols",
        formula="target ~ feature",
        data_transformation=_MAPPING,
    )
    return rt.RealTimeModel._build_forecast_tasks(
        [(model, _MAPPING, data)],
        vintages,
        _common_options(decomp=decomp),
        batch_size=batch_size,
        parallel=parallel,
        max_workers=2,
        model_kwargs={},
    )


def _run_spawned_groups(groups: list[list[ForecastTask]]):
    tasks = [task for group in groups for task in group]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=context, max_workers=2) as pool:
        results = list(pool.map(_run_forecast_task, tasks))
    grouped = []
    offset = 0
    for group in groups:
        grouped.append(results[offset : offset + len(group)])
        offset += len(group)
    return grouped


def _sorted_forecasts(results) -> pd.DataFrame:
    frame = pd.concat([result.forecasts for result in results], ignore_index=True)
    return (
        frame[_FORECAST_COLUMNS]
        .sort_values(["vintage_date", "date", "forecast_horizon", "variable", "source"])
        .reset_index(drop=True)
    )


def _assert_forecasts_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        left.drop(columns="value"),
        right.drop(columns="value"),
        check_dtype=False,
    )
    np.testing.assert_allclose(left["value"], right["value"], rtol=1e-12, atol=1e-12)


def _sorted_decompositions(results) -> pd.DataFrame:
    frame = pd.concat(
        [
            result.decompositions
            for result in results
            if result.decompositions is not None
        ],
        ignore_index=True,
    )
    return frame.sort_values(_DECOMPOSITION_KEYS).reset_index(drop=True)


def _assert_decompositions_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        left,
        right,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def _fit_ar1_t_from_model_data(args):
    data, input_order, mapping = args
    selected = data.subset(["target"], list(input_order))
    model = rt.models.ForecastOLS(
        formula="target ~ " + " + ".join(input_order),
        data_transformation=mapping,
    )
    model._fit_from_data(
        selected,
        data_transformation=mapping,
        frequency="M",
        X_imputation="ar1_t",
    )
    forecast = model.forecast(steps=2, X=selected.to_wide("X"), X_imputation="ar1_t")
    return model._prepared_X_history, forecast.forecast


def test_forecast_task_serialisation_preserves_bindings_and_isolation():
    vintage = pd.Timestamp("2024-01-31")
    outturns = _archive(pd.DatetimeIndex([vintage]))
    forecasts = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-29")],
            "variable": ["target"],
            "vintage_date": [vintage],
            "frequency": ["M"],
            "value": [25.0],
            "metric": ["levels"],
            "source": ["forecast-feed"],
        }
    )
    data = _selected_data(
        outturns,
        forecasts=forecasts,
        y_sources={"target": "missing-feed"},
        X_sources=None,
    )
    task = ForecastTask(
        model=rt.models.ForecastOLS(label="serialisation"),
        data=data,
        data_transformation=_MAPPING,
        vintages=np.array([vintage]),
        options=_common_options(),
        model_kwargs={},
    )

    copied = copy.deepcopy(task)
    restored = pickle.loads(pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL))

    for candidate in (copied, restored):
        assert candidate.data.metrics("y") == {"target": "levels"}
        assert candidate.data.metrics("X") == {"feature": "levels"}
        assert candidate.data.frequencies("y") == {"target": "M"}
        assert candidate.data.frequencies("X") == {"feature": "M"}
        assert candidate.data.to_wide("X", "conditioning") is None
        empty_y_conditioning = candidate.data.to_wide("y", "conditioning")
        assert empty_y_conditioning is not None
        assert empty_y_conditioning.empty
        history = candidate.data.to_long()
        assert set(history["source"]) == {"survey"}

    assert set(vars(task)) == {
        "model",
        "data",
        "data_transformation",
        "vintages",
        "options",
        "model_kwargs",
    }
    assert not {"outturns", "forecasts", "metric_dictionaries"}.intersection(vars(task))
    assert not {"outturns", "forecasts", "metric_dictionaries"}.intersection(
        vars(restored.data)
    )

    for candidate, value in ((copied, -999.0), (restored, -998.0)):
        projection = candidate.data.as_of(vintage).to_wide("y")
        projection.iloc[0, 0] = value
        assert task.data.to_wide("y").iloc[0, 0] == 20.0


def test_spawned_worker_matches_sequential_across_vintage_batch_sizes():
    vintages = pd.date_range("2024-01-31", periods=4, freq="ME").to_numpy()
    data = _selected_data(_archive(pd.DatetimeIndex(vintages)))
    sequential = _make_tasks(
        data,
        vintages,
        parallel=False,
        batch_size=1,
    )
    sequential_forecasts = _sorted_forecasts(
        [_run_forecast_task(task) for task in sequential]
    )
    groups = [
        _make_tasks(data, vintages, parallel=True, batch_size=size)
        for size in (1, 2, len(vintages))
    ]
    spawned_groups = _run_spawned_groups(groups)

    assert len(sequential) == 1
    for tasks, results in zip(groups, spawned_groups, strict=True):
        assert (
            len(tasks)
            == (len(vintages) + tasks[0].vintages.size - 1) // tasks[0].vintages.size
        )
        _assert_forecasts_equal(sequential_forecasts, _sorted_forecasts(results))


def test_spawned_decomposition_skips_bad_release_and_reconciles_all_rows():
    vintages = pd.date_range("2024-01-31", periods=3, freq="ME")
    data = _selected_data(_archive(vintages, missing_target_vintage=vintages[1]))
    task = _make_tasks(
        data,
        vintages.to_numpy(),
        parallel=False,
        batch_size=1,
        decomp=True,
    )[0]

    with pytest.warns(UserWarning, match="Skipping this vintage"):
        sequential_result = _run_forecast_task(task)
    (spawned_result,) = _run_spawned_groups([[task]])[0]

    _assert_forecasts_equal(
        _sorted_forecasts([sequential_result]),
        _sorted_forecasts([spawned_result]),
    )
    sequential_decomp = _sorted_decompositions([sequential_result])
    spawned_decomp = _sorted_decompositions([spawned_result])
    _assert_decompositions_equal(sequential_decomp, spawned_decomp)

    level = sequential_decomp.loc[sequential_decomp["decomposition"].eq("level")]
    revision = sequential_decomp.loc[sequential_decomp["decomposition"].eq("revision")]
    assert set(level["vintage_date"]) == {vintages[0], vintages[2]}
    assert set(revision["vintage_date"]) == {vintages[2]}
    assert set(revision["base_vintage_date"]) == {vintages[0]}
    assert set(revision["revision_source"]) == {
        "news",
        "reestimation",
        "interaction",
    }
    assert level["base_vintage_date"].isna().all()
    assert level["revision_source"].isna().all()
    assert level["forecast_metric"].eq("levels").all()
    assert level["frequency"].eq("M").all()

    forecast_keys = ["date", "vintage_date", "forecast_horizon", "variable"]
    forecasts = _sorted_forecasts([sequential_result])
    level_keys = level[forecast_keys].drop_duplicates().sort_values(forecast_keys)
    forecast_key_frame = (
        forecasts[forecast_keys].drop_duplicates().sort_values(forecast_keys)
    )
    pd.testing.assert_frame_equal(
        level_keys.reset_index(drop=True),
        forecast_key_frame.reset_index(drop=True),
        check_dtype=False,
    )

    level_totals = level.groupby(forecast_keys, sort=True)["contribution"].sum()
    forecast_totals = forecasts.groupby(forecast_keys, sort=True)["value"].sum()
    np.testing.assert_allclose(level_totals.to_numpy(), forecast_totals.to_numpy())

    current_level = level.loc[level["vintage_date"].eq(vintages[2])]
    base_level = level.loc[level["vintage_date"].eq(vintages[0])]
    revision_keys = ["date", "forecast_horizon", "variable"]
    current_totals = current_level.groupby(revision_keys)["contribution"].sum()
    base_totals = base_level.groupby(revision_keys)["contribution"].sum()
    revision_totals = revision.groupby(revision_keys)["contribution"].sum()
    np.testing.assert_allclose(
        revision_totals.sort_index().to_numpy(),
        (current_totals - base_totals).sort_index().to_numpy(),
    )
    assert len(revision) == len(current_level) * 3


def test_public_decomposition_guard_remains_rejected():
    vintage = pd.Timestamp("2024-01-31")
    outturns = _archive(pd.DatetimeIndex([vintage]))
    realtime = rt.RealTimeModel(
        ForecastData(
            outturns_data=outturns,
            compute_levels=False,
            data_check=False,
        ),
        rt.models.ForecastOLS(),
    )

    with pytest.raises(ValueError, match="decomp=True.*parallel=True"):
        realtime.forecast(
            y_variables=["target"],
            X_variables=["feature"],
            data_transformation=_MAPPING,
            steps=1,
            decomp=True,
            parallel=True,
        )


def test_sequential_task_construction_keeps_all_vintages_in_one_batch():
    vintages = pd.date_range("2024-01-31", periods=4, freq="ME").to_numpy()
    data = _selected_data(_archive(pd.DatetimeIndex(vintages)))
    tasks = _make_tasks(data, vintages, parallel=False, batch_size=1)

    assert len(tasks) == 1
    np.testing.assert_array_equal(tasks[0].vintages, vintages)


@pytest.mark.parametrize("input_order", [("x_a", "x_b"), ("x_b", "x_a")])
def test_multicolumn_ar1_t_preparation_matches_direct_and_realtime_paths(
    input_order,
):
    vintage = pd.Timestamp("2021-01-31")
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    values = {
        "target": 20.0 + np.arange(12) * 0.3,
        "x_a": 4.0 + np.arange(10) * 0.2,
        "x_b": 8.0 + np.arange(9) * 0.15,
    }
    rows = []
    for variable, series in values.items():
        for date, value in zip(dates[: len(series)], series, strict=True):
            rows.append(
                {
                    "date": date,
                    "variable": variable,
                    "vintage_date": vintage,
                    "frequency": "M",
                    "value": value,
                    "metric": "levels",
                }
            )
    outturns = pd.DataFrame(rows)
    archive = ModelData.from_archive(
        outturns,
        frequencies={"target": "M", "x_a": "M", "x_b": "M"},
    )
    requirements = (
        ModelInputRequirements(
            consumer="ar1-parity",
            y=(("target", "levels"),),
            X=(("x_a", "levels"), ("x_b", "levels")),
            explicit=True,
        ),
    )
    selected = archive.select(requirements).subset(["target"], list(input_order))
    selected_at_vintage = selected.as_of(vintage)
    mapping = {"target": "levels", "x_a": "levels", "x_b": "levels"}
    formula = "target ~ " + " + ".join(input_order)
    projected_X = selected_at_vintage.to_wide("X")

    direct = rt.models.ForecastOLS(
        formula=formula,
        data_transformation=mapping,
    )
    direct.fit(
        selected_at_vintage.to_wide("y"),
        selected_at_vintage.to_wide("X"),
        data_transformation=mapping,
        frequency="M",
        X_imputation="ar1_t",
        input_frequencies={"target": "M", "x_a": "M", "x_b": "M"},
        y_input_metrics={"target": "levels"},
        X_input_metrics={"x_a": "levels", "x_b": "levels"},
    )
    realtime = rt.models.ForecastOLS(
        formula=formula,
        data_transformation=mapping,
    )
    realtime._fit_from_data(
        selected_at_vintage,
        data_transformation=mapping,
        frequency="M",
        X_imputation="ar1_t",
    )

    assert list(selected_at_vintage.columns("X")) == list(input_order)
    pd.testing.assert_frame_equal(
        direct._prepared_y_history,
        realtime._prepared_y_history,
    )
    pd.testing.assert_frame_equal(
        direct._prepared_X_history,
        realtime._prepared_X_history,
    )
    direct_forecast = direct.forecast(
        steps=2,
        X=projected_X,
        X_imputation="ar1_t",
    ).forecast
    realtime_forecast = realtime.forecast(
        steps=2,
        X=projected_X,
        X_imputation="ar1_t",
    ).forecast
    repeated_forecast = realtime.forecast(
        steps=2,
        X=projected_X,
        X_imputation="ar1_t",
    ).forecast

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=context, max_workers=1) as pool:
        spawned_prepared_X, spawned_forecast = next(
            iter(
                pool.map(
                    _fit_ar1_t_from_model_data,
                    [(selected_at_vintage, input_order, mapping)],
                )
            )
        )

    pd.testing.assert_frame_equal(
        direct._prepared_X_history,
        spawned_prepared_X,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    pd.testing.assert_frame_equal(direct_forecast, realtime_forecast)
    pd.testing.assert_frame_equal(realtime_forecast, repeated_forecast)
    pd.testing.assert_frame_equal(
        realtime_forecast,
        spawned_forecast,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
