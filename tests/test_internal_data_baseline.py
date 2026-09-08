"""Characterise the stage-one internal data contracts.

These tests record current public behaviour at the model, tree, realtime and
long-form transformation boundaries. They are deliberately small so later
internal-data changes can distinguish structural work from semantic changes.
"""

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

import forecast_realtime as rt
import forecast_realtime._model_data as model_data
from forecast_realtime._utils import impute_X
from forecast_realtime.data_transformation import DataTransformationPipeline
from forecast_realtime.forecast_model import ForecastModel
from forecast_realtime.forecast_tree import ForecastTree, TreeNode


class _RecordingModel(ForecastModel):
    """Small model that records the public preparation boundary."""

    def _fit(self, y, X=None, **kwargs):
        self.fitted_values_ = y.copy()
        return self

    def _prepare_forecast_inputs(self, y, X):
        self.received_forecast_y = None if y is None else y.copy()
        self.received_forecast_X = None if X is None else X.copy()
        return y, X

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return np.zeros((steps, len(self.y.columns)))


class _PublicOverrideModel(_RecordingModel):
    """Record calls to public ``fit`` and ``forecast`` before delegation."""

    fit_calls = []
    forecast_calls = []
    predict_calls = []

    def fit(self, *args, **kwargs):
        type(self).fit_calls.append(1)
        return super().fit(*args, **kwargs)

    def forecast(self, *args, **kwargs):
        type(self).forecast_calls.append(1)
        return super().forecast(*args, **kwargs)

    def predict(self, *args, **kwargs):
        type(self).predict_calls.append(1)
        return super().predict(*args, **kwargs)


def _monthly_y(values, start="2020-01-31"):
    return pd.DataFrame(
        {"target": values},
        index=pd.date_range(start, periods=len(values), freq="ME"),
    )


@pytest.mark.parametrize("pipeline", [None, {"target": "levels"}])
@pytest.mark.parametrize("conditioning", ["none", "empty", "all_nan"])
def test_conditioning_intent_reaches_forecast_preparation_hook(pipeline, conditioning):
    """None, empty and all-missing conditioning retain their current intent."""
    model = _RecordingModel(data_transformation=pipeline)
    model.fit(_monthly_y([1.0, 2.0, 3.0]), frequency="M" if pipeline else None)

    if conditioning == "none":
        supplied = None
    elif conditioning == "empty":
        supplied = pd.DataFrame(
            columns=["target"],
            index=pd.DatetimeIndex([], name="date"),
        )
    else:
        supplied = pd.DataFrame(
            {"target": [np.nan]},
            index=pd.DatetimeIndex(["2020-04-30"], name="date"),
        )

    model.forecast(steps=1, y=supplied)
    received = model.received_forecast_y

    if pipeline is None and conditioning == "none":
        assert received is None
    else:
        assert received is not None
        assert list(received.index) == list(
            pd.date_range(
                "2020-01-31",
                periods=4 if conditioning == "all_nan" else 3,
                freq="ME",
            )
        )
        if conditioning == "all_nan":
            assert pd.isna(received.iloc[-1, 0])


def test_direct_daily_inputs_without_transformation_round_trip():
    model = _RecordingModel()
    index = pd.date_range("2020-01-01", periods=3, freq="D")

    model.fit(pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=index))
    result = model.forecast(steps=2)

    assert result.index.equals(pd.date_range("2020-01-04", periods=2, freq="D"))


def test_period_index_direct_fit_and_forecast_but_not_explicit_transformation():
    index = pd.period_range("2020Q1", periods=3, freq="Q")
    model = _RecordingModel()

    model.fit(pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=index))
    result = model.forecast(steps=2)

    assert result.index.equals(pd.DatetimeIndex(["2020-12-31", "2021-03-31"]))

    with pytest.raises(ValueError, match="DatetimeIndex"):
        _RecordingModel().fit(
            pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=index),
            data_transformation={"target": "diff"},
            frequency="Q",
        )


def test_pipeline_constructor_and_projection_do_not_alias_callers():
    mapping = {"target": "levels"}
    pipeline = DataTransformationPipeline(mapping)
    mapping["target"] = "diff"
    assert pipeline.data_transformation == {"target": "levels"}

    source = pd.DataFrame(
        {"variable": ["target"], "metric": ["levels"], "value": [1.0]},
        index=[7],
    )
    projection = pipeline.filter(source, ["target"])
    projection.loc[7, "value"] = 99.0

    assert source.loc[7, "value"] == 1.0
    assert projection.index.tolist() == [7]


def test_apply_retains_rows_metadata_order_and_supplied_vintage_boundaries():
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": ["target"] * 3,
            "vintage_date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-02-29"]),
            "frequency": ["M"] * 3,
            "value": [100.0, 110.0, 121.0],
            "metric": ["levels"] * 3,
            "source": ["survey"] * 3,
            "forecast_horizon": [0, 0, 1],
            "caller_note": ["first", "second", "third"],
        },
        index=[11, 4, 8],
    )

    transformed, forecasts = DataTransformationPipeline({"target": "diff"}).apply(
        outturns=outturns,
        forecasts=None,
        y_variables=["target"],
        X_variables=None,
    )

    assert forecasts is None
    assert list(transformed.columns) == list(outturns.columns)
    assert isinstance(transformed.index, pd.RangeIndex)
    assert transformed.index.tolist() == [0, 1, 2, 3]
    assert transformed["date"].dtype == outturns["date"].dtype
    assert transformed["vintage_date"].dtype == outturns["vintage_date"].dtype
    assert transformed["forecast_horizon"].dtype == outturns["forecast_horizon"].dtype
    assert transformed.loc[:2, "date"].tolist() == outturns["date"].tolist()
    assert transformed.loc[:2, "caller_note"].tolist() == outturns["caller_note"].tolist()
    assert transformed.loc[:2, "metric"].tolist() == ["levels"] * 3

    derived = transformed.loc[transformed["metric"] == "diff"].iloc[0]
    assert derived["date"] == pd.Timestamp("2020-03-31")
    assert derived["vintage_date"] == pd.Timestamp("2020-02-29")
    assert derived["source"] == "survey"
    assert derived["frequency"] == "M"
    assert derived["forecast_horizon"] == 1
    assert derived["caller_note"] == "third"
    assert derived["value"] == 11.0


def test_ar1_t_imputation_consumes_one_seeded_rng_in_declared_column_order(monkeypatch):
    calls = []

    def controlled_imputer(observed, shortage, rng):
        draws = rng.random(shortage)
        calls.append((observed.name, shortage, draws.copy()))
        return draws.tolist()

    monkeypatch.setattr(model_data, "_ar1_t_impute", controlled_imputer)
    index = pd.date_range("2019-10-31", periods=4, freq="ME")
    X = pd.DataFrame(
        {
            "short_column": [1.0, 2.0, 3.0, np.nan],
            "long_column": [10.0, 20.0, 30.0, 40.0],
        },
        index=index,
    )

    first = impute_X(
        X,
        pd.Timestamp("2020-01-31"),
        steps=2,
        method="ar1_t",
        random_state=17,
        frequencies={"short_column": "M", "long_column": "M"},
    )
    first_calls = calls[:]
    calls.clear()
    second = impute_X(
        X,
        pd.Timestamp("2020-01-31"),
        steps=2,
        method="ar1_t",
        random_state=17,
        frequencies={"short_column": "M", "long_column": "M"},
    )

    expected_rng = np.random.default_rng(17)
    expected_short = expected_rng.random(3)
    expected_long = expected_rng.random(2)
    assert [(name, shortage) for name, shortage, _ in first_calls] == [
        ("short_column", 3),
        ("long_column", 2),
    ]
    np.testing.assert_allclose(first_calls[0][2], expected_short)
    np.testing.assert_allclose(first_calls[1][2], expected_long)
    pd.testing.assert_frame_equal(first, second)


def test_realtime_reaches_public_fit_and_predict_not_forecast():
    _PublicOverrideModel.fit_calls = []
    _PublicOverrideModel.forecast_calls = []
    _PublicOverrideModel.predict_calls = []
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "variable": ["target"] * 3,
            "vintage_date": [vintage] * 3,
            "frequency": ["M"] * 3,
            "value": [1.0, 2.0, 3.0],
            "metric": ["levels"] * 3,
        }
    )
    realtime = rt.RealTimeModel(
        data=ForecastData(outturns_data=outturns, compute_levels=False, data_check=False),
        models=_PublicOverrideModel(),
    )

    realtime.forecast(
        y_variables=["target"],
        data_transformation={"target": "levels"},
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    assert _PublicOverrideModel.fit_calls
    assert _PublicOverrideModel.predict_calls
    assert not _PublicOverrideModel.forecast_calls


def test_forecast_tree_reaches_component_public_fit_and_forecast_overrides():
    _PublicOverrideModel.fit_calls = []
    _PublicOverrideModel.forecast_calls = []
    leaf = _PublicOverrideModel(label="leaf")
    tree = ForecastTree(
        TreeNode(
            transform=lambda components: next(iter(components.values())),
            children=[leaf],
            name="root",
            target="target",
        )
    )
    y = _monthly_y([1.0, 2.0, 3.0])

    tree.fit(y)
    tree.forecast(steps=1)

    assert _PublicOverrideModel.fit_calls
    assert _PublicOverrideModel.forecast_calls
