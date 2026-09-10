"""Pending semantic contracts for model input data."""

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

import forecast_realtime as rt
from forecast_realtime.forecast_model import ForecastModel
from forecast_realtime.forecast_tree import ForecastTree, TreeNode


class _RecordingModel(ForecastModel):
    """Small pickleable model whose preparation inputs remain observable."""

    _supports_target_conditioning = True

    fit_calls = []
    prepare_calls = []

    @classmethod
    def reset(cls):
        cls.fit_calls = []
        cls.prepare_calls = []

    def _fit(self, y, X=None, **kwargs):
        type(self).fit_calls.append(
            {
                "label": self.label,
                "y": y.copy(),
                "X": None if X is None else X.copy(),
            }
        )
        self.fitted_values_ = y.copy()
        return self

    def _prepare_forecast_inputs(self, y, X):
        type(self).prepare_calls.append(
            {
                "label": self.label,
                "y": None if y is None else y.copy(),
                "X": None if X is None else X.copy(),
            }
        )
        return y, X

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return np.zeros((steps, len(self.y.columns)))


def _outturns(variable, dates, values, vintage):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "variable": variable,
            "vintage_date": pd.Timestamp(vintage),
            "frequency": "M",
            "value": values,
            "metric": "levels",
        }
    )


def _data(outturns):
    return ForecastData(
        outturns_data=outturns,
        metric="levels",
        compute_levels=False,
    )


def _forecasts(variable, dates, values, vintage, source, metric="levels"):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "variable": variable,
            "vintage_date": pd.Timestamp(vintage),
            "source": source,
            "frequency": "M",
            "value": values,
            "forecast_horizon": range(len(dates)),
            "target_minus_vintage": [
                (pd.Period(date, freq="M") - pd.Period(vintage, freq="M")).n
                for date in dates
            ],
            "metric": metric,
        }
    )


def test_realtime_rejects_conflicting_same_date_vintage_values():
    outturns = _outturns(
        "target",
        ["2020-01-31", "2020-02-29"],
        [100.0, 110.0],
        "2020-02-29",
    )
    data = _data(outturns)
    data._raw_outturns = pd.concat(
        [
            outturns,
            outturns.iloc[[0]].assign(value=101.0),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match=r"(?i)(conflict|duplicate)"):
        rt.RealTimeModel(data=data, models=_RecordingModel()).forecast(
            y_variables=["target"],
            steps=1,
            first_vintage="2020-02-29",
            last_vintage="2020-02-29",
            first_forecast_horizon=0,
            data_transformation={"target": "levels"},
        )


def test_realtime_keeps_y_and_x_conditioning_sources_separate():
    dates = ["2020-01-31", "2020-02-29", "2020-03-31"]
    data = _data(_outturns("z", dates, [1.0, 2.0, 3.0], "2020-03-31"))
    data._raw_forecasts = pd.concat(
        [
            _forecasts("z", ["2020-04-30"], [101.0], "2020-03-31", "y_source"),
            _forecasts("z", ["2020-04-30"], [202.0], "2020-03-31", "X_source"),
        ],
        ignore_index=True,
    )
    _RecordingModel.reset()

    rt.RealTimeModel(data=data, models=_RecordingModel()).forecast(
        y_variables=["z"],
        X_variables=["z"],
        y_steps_ahead={"z": 0},
        y_sources={"z": "y_source"},
        X_steps_ahead={"z": 0},
        X_sources={"z": "X_source"},
        steps=1,
        first_vintage="2020-03-31",
        last_vintage="2020-03-31",
        first_forecast_horizon=1,
        data_transformation={"z": "levels"},
    )

    prepared = _RecordingModel.prepare_calls[-1]
    assert prepared["y"].loc[pd.Timestamp("2020-04-30"), "z"] == 101.0
    assert prepared["X"].loc[pd.Timestamp("2020-04-30"), "z"] == 202.0


def _select_leaf(children):
    return children["leaf"].copy()


def test_mapping_free_tree_retains_realtime_regressor_inputs():
    data = _data(
        pd.concat(
            [
                _outturns(
                    "target",
                    ["2020-01-31", "2020-02-29", "2020-03-31"],
                    [1.0, 2.0, 3.0],
                    "2020-03-31",
                ),
                _outturns(
                    "regressor",
                    ["2020-01-31", "2020-02-29", "2020-03-31"],
                    [10.0, 20.0, 30.0],
                    "2020-03-31",
                ),
            ],
            ignore_index=True,
        )
    )
    _RecordingModel.reset()
    leaf = _RecordingModel(label="leaf")
    tree = ForecastTree(
        spec=TreeNode(
            transform=_select_leaf,
            children=[leaf],
            name="root",
            target="target",
        )
    )

    rt.RealTimeModel(data=data, models=tree).forecast(
        y_variables=["target"],
        X_variables=["regressor"],
        steps=1,
        first_vintage="2020-03-31",
        last_vintage="2020-03-31",
    )

    assert not data.forecasts.empty
    leaf_preparation = [
        call for call in _RecordingModel.prepare_calls if call["label"] == "leaf"
    ][-1]
    assert leaf_preparation["X"] is not None
    assert list(leaf_preparation["X"].columns) == ["regressor"]


def test_diff_forecast_conditioning_preserves_published_level_path():
    dates = ["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"]
    data = _data(_outturns("target", dates, [100.0, 110.0, 121.0, 133.1], "2020-04-30"))
    data._raw_forecasts = _forecasts(
        "target",
        ["2020-05-31"],
        [5.0],
        "2020-04-30",
        "diff_source",
        metric="diff",
    )
    _RecordingModel.reset()

    rt.RealTimeModel(data=data, models=_RecordingModel()).forecast(
        y_variables=["target"],
        y_steps_ahead={"target": 1},
        y_sources={"target": "diff_source"},
        steps=2,
        first_vintage="2020-04-30",
        last_vintage="2020-04-30",
        first_forecast_horizon=0,
        data_transformation={"target": "diff"},
    )

    prepared_y = _RecordingModel.prepare_calls[-1]["y"]
    assert prepared_y.loc[pd.Timestamp("2020-04-30"), "target"] == pytest.approx(12.1)
    assert prepared_y.loc[pd.Timestamp("2020-05-31"), "target"] == pytest.approx(5.0)


def test_multivariate_child_dispatches_each_native_metric_once():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 12.0, 17.0, 25.0],
        },
        index=dates,
    )
    _RecordingModel.reset()
    child = _RecordingModel(
        label="child",
        data_transformation={"a": "levels", "b": "diff"},
    )
    stacker = _RecordingModel(
        label="stacker",
        formula="a ~ child_a + child_b",
        data_transformation={
            "a": "levels",
            "child_a": "levels",
            "child_b": "diff",
        },
    )
    tree = ForecastTree(spec=TreeNode(transform=stacker, children=[child], name="stack"))

    tree.fit(y)

    stacker_fit = [
        call for call in _RecordingModel.fit_calls if call["label"] == "stacker"
    ][-1]
    assert list(stacker_fit["X"].columns) == ["child_a", "child_b"]
    np.testing.assert_allclose(stacker_fit["X"]["child_b"], [2.0, 5.0, 8.0])
