"""Tests for deterministic realtime input-metric selection."""

import pandas as pd
import pytest
from forecast_evaluation import ForecastData

from forecast_realtime._realtime_forecasting import ForecastRunResult
from forecast_realtime.forecast_model import ForecastModel
from forecast_realtime.real_time_model import (
    RealTimeModel,
    _select_input_metrics,
)


class _NoopModel(ForecastModel):
    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        return pd.DataFrame()


@pytest.fixture
def metric_rows():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]
            ),
            "variable": ["gdp", "gdp", "gdp", "gdp"],
            "metric": ["levels", "pop", "levels", "pop"],
            "value": [100.0, 1.0, 110.0, 2.0],
        }
    )


def test_requested_metric_is_selected_before_deduplication(metric_rows):
    selected, input_metrics = _select_input_metrics(metric_rows, ["gdp"], {"gdp": "pop"})

    assert input_metrics == {"gdp": "pop"}
    assert selected["metric"].unique().tolist() == ["pop"]


def test_levels_are_selected_when_requested_metric_is_derivable(metric_rows):
    levels_only = metric_rows[metric_rows["metric"] == "levels"]

    selected, input_metrics = _select_input_metrics(levels_only, ["gdp"], {"gdp": "pop"})

    assert input_metrics == {"gdp": "levels"}
    assert selected["metric"].unique().tolist() == ["levels"]


def test_levels_request_prefers_levels_over_other_metrics(metric_rows):
    selected, input_metrics = _select_input_metrics(
        metric_rows, ["gdp"], {"gdp": "levels"}
    )

    assert input_metrics == {"gdp": "levels"}
    assert selected["metric"].unique().tolist() == ["levels"]


def test_selection_is_independent_of_row_order(metric_rows):
    forward, forward_metrics = _select_input_metrics(metric_rows, ["gdp"], {"gdp": "pop"})
    reverse, reverse_metrics = _select_input_metrics(
        metric_rows.iloc[::-1], ["gdp"], {"gdp": "pop"}
    )

    pd.testing.assert_frame_equal(
        forward.sort_values(["date", "value"]).reset_index(drop=True),
        reverse.sort_values(["date", "value"]).reset_index(drop=True),
    )
    assert forward_metrics == reverse_metrics == {"gdp": "pop"}


def test_no_requested_metric_rejects_ambiguous_input(metric_rows):
    with pytest.raises(ValueError, match="gdp.*ambiguous.*levels.*pop"):
        _select_input_metrics(metric_rows, ["gdp"], None)


def test_unavailable_requested_metric_has_actionable_error(metric_rows):
    with pytest.raises(ValueError, match="gdp.*index.*levels.*pop"):
        _select_input_metrics(metric_rows, ["gdp"], {"gdp": "index"})


def test_realtime_selects_metrics_using_model_owned_mapping(monkeypatch):
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"]
            ),
            "variable": "gdp",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 1.0, 110.0, 2.0],
            "metric": ["levels", "pop", "levels", "pop"],
        }
    )
    data = ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    )
    realtime = RealTimeModel(
        data=data,
        models=_NoopModel(data_transformation={"gdp": "pop"}),
    )
    captured_tasks = []

    def capture_tasks(tasks, *, parallel, max_workers):
        captured_tasks.extend(tasks)
        result = pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-04-30"]),
                "vintage_date": vintage,
                "forecast_horizon": [0],
                "variable": ["gdp"],
                "value": [3.0],
                "metric": ["levels"],
                "source": ["noop"],
                "frequency": ["M"],
            }
        )
        return [
            ForecastRunResult(
                forecasts=result,
                decompositions=None,
                all_vintages_skipped=False,
            )
        ]

    monkeypatch.setattr(
        RealTimeModel,
        "_execute_forecast_tasks",
        staticmethod(capture_tasks),
    )

    realtime.forecast(
        y_variables=["gdp"],
        data_transformation={"gdp": "levels"},
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
        reconstruct_levels=False,
    )

    task = captured_tasks[0]
    assert task.data_transformation == {"gdp": "pop"}
    assert task.input_metrics == {"gdp": "pop"}
    assert task.common["outturns"]["metric"].unique().tolist() == ["pop"]
