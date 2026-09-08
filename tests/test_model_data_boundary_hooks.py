import inspect
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

import forecast_realtime as rt
from forecast_realtime._model_data import ModelData
from forecast_realtime.forecast_model import ForecastContext, ForecastModel
from forecast_realtime.forecast_tree import ForecastTree, TreeNode


class _BoundaryModel(ForecastModel):
    validation_calls = 0

    def __init__(self, *args, fail_validation=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_validation = fail_validation

    def _validate_fit_inputs(self, y, X):
        type(self).validation_calls += 1
        self.validation_marker = "candidate"
        y.iloc[0, 0] = -999.0
        if self.fail_validation:
            raise RuntimeError("validation hook failed")
        return y, X

    def _fit(self, y, X=None, **kwargs):
        self.fitted_values_ = y.copy()
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return np.zeros((steps, len(self.y.columns)))


class _RecordingModel(ForecastModel):
    def _fit(self, y, X=None, **kwargs):
        self.fitted_values_ = y.copy()
        return self

    def _prepare_forecast_inputs(self, y, X):
        self.received_forecast_y = None if y is None else y.copy()
        self.received_forecast_X = None if X is None else X.copy()
        return y, X

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return np.zeros((steps, len(self.y.columns)))


class _RealtimeRecordingModel(_RecordingModel):
    prepared_X = []

    def _prepare_forecast_inputs(self, y, X):
        type(self).prepared_X.append(None if X is None else X.copy())
        return super()._prepare_forecast_inputs(y, X)


class _ContextOverrideModel(_RecordingModel):
    def __init__(self, *args, replacement=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.replacement = replacement

    def predict(self, context, *args, **kwargs):
        if self.replacement:
            context = replace(
                context,
                y_conditioning=context.y_conditioning.assign(target=888.0),
            )
        else:
            context.y_conditioning.loc[:, "target"] = 777.0
        return super().predict(context, *args, **kwargs)


def _monthly_y(values, start="2020-01-31"):
    return pd.DataFrame(
        {"target": values},
        index=pd.date_range(start, periods=len(values), freq="ME"),
    )


def _realtime_data():
    vintage = pd.Timestamp("2020-03-31")
    outturns = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=3, freq="ME"),
            "variable": "target",
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0],
            "metric": "levels",
        }
    )
    return ForecastData(
        outturns_data=outturns,
        compute_levels=False,
        data_check=False,
    ), vintage


def test_realtime_formula_skips_unselected_X_conditioning_sources():
    vintage = pd.Timestamp("2020-03-31")
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    outturns = pd.DataFrame(
        {
            "date": list(dates) * 3,
            "variable": ["target"] * 3 + ["chosen"] * 3 + ["unused"] * 3,
            "vintage_date": vintage,
            "frequency": "M",
            "value": [100.0, 110.0, 121.0, 10.0, 20.0, 30.0, 1.0, 2.0, 3.0],
            "metric": "levels",
        }
    )
    conditioning = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-04-30", "2020-05-31"] * 2),
            "variable": ["chosen"] * 2 + ["unused"] * 2,
            "vintage_date": vintage,
            "source": ["chosen-source"] * 2 + ["unused-source"] * 2,
            "frequency": "M",
            "value": [40.0, 50.0, 400.0, 500.0],
            "forecast_horizon": [1, 2, 1, 2],
        }
    )
    data = ForecastData(
        outturns_data=outturns,
        forecasts_data=conditioning,
        compute_levels=False,
        data_check=False,
    )
    _RealtimeRecordingModel.prepared_X = []

    rt.RealTimeModel(
        data=data,
        models=_RealtimeRecordingModel(formula="target ~ chosen"),
    ).forecast(
        y_variables=["target"],
        X_variables=["chosen", "unused"],
        X_steps_ahead={"chosen": 1, "unused": 1},
        X_sources={"chosen": "chosen-source", "unused": "unused-source"},
        data_transformation={
            "target": "levels",
            "chosen": "levels",
            "unused": "levels",
        },
        steps=2,
        first_forecast_horizon=1,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )

    prepared_X = _RealtimeRecordingModel.prepared_X[-1]
    assert list(prepared_X.columns) == ["chosen"]
    expected_horizon = pd.date_range("2020-04-30", periods=2, freq="ME").as_unit("ns")
    assert prepared_X.index.equals(pd.date_range("2020-01-31", periods=5, freq="ME"))
    pd.testing.assert_frame_equal(
        prepared_X.loc[expected_horizon],
        pd.DataFrame({"chosen": [40.0, 50.0]}, index=expected_horizon),
    )


def test_failed_validation_hook_isolated_from_fitted_model_and_model_data():
    source = _monthly_y([100.0, 110.0, 121.0])
    model = _BoundaryModel()
    _BoundaryModel.validation_calls = 0
    model.fit(source)
    model.validation_marker = "published"
    fitted_before = model.fitted_values_.copy()
    shared = ModelData.from_wide(y=source)

    model.fail_validation = True
    with pytest.raises(RuntimeError, match="validation hook failed"):
        model._fit_from_data(shared)

    assert _BoundaryModel.validation_calls == 2
    assert model.validation_marker == "published"
    pd.testing.assert_frame_equal(model.fitted_values_, fitted_before)
    pd.testing.assert_frame_equal(shared.to_wide("y"), source)
    pd.testing.assert_frame_equal(source, _monthly_y([100.0, 110.0, 121.0]))


def test_validation_hook_is_reached_once_through_realtime_and_tree_paths():
    realtime_data, vintage = _realtime_data()
    _BoundaryModel.validation_calls = 0
    rt.RealTimeModel(data=realtime_data, models=_BoundaryModel()).forecast(
        y_variables=["target"],
        data_transformation={"target": "levels"},
        steps=1,
        first_forecast_horizon=0,
        first_vintage=str(vintage.date()),
        last_vintage=str(vintage.date()),
    )
    assert _BoundaryModel.validation_calls == 1

    _BoundaryModel.validation_calls = 0
    tree = ForecastTree(
        TreeNode(
            transform=lambda components: next(iter(components.values())),
            children=[_BoundaryModel(label="leaf")],
            name="root",
            target="target",
        )
    )
    tree.fit(_monthly_y([100.0, 110.0, 121.0]))
    assert _BoundaryModel.validation_calls == 1


@pytest.mark.parametrize("positional_context", [False, True])
def test_tree_forecast_preserves_the_public_context_position(positional_context):
    history = _monthly_y([100.0, 110.0, 121.0])
    conditioning = _monthly_y([130.0, 140.0], start="2020-04-30")
    leaf = _RecordingModel(label="leaf")
    tree = ForecastTree(
        TreeNode(
            transform=lambda components: components["leaf"],
            children=[leaf],
            name="root",
            target="target",
        )
    ).fit(history)
    context = ForecastContext(
        y_history=history,
        X_history=None,
        y_conditioning=conditioning,
        forecast_origin=history.index[-1],
    )

    result = (
        tree.forecast(2, None, None, False, context)
        if positional_context
        else tree.forecast(steps=2, context=context)
    )

    assert result.forecast.index.equals(conditioning.index)
    pd.testing.assert_frame_equal(
        leaf.received_forecast_y.loc[conditioning.index], conditioning
    )


@pytest.mark.parametrize("replacement", [False, True])
def test_public_predict_override_preserves_context_changes(replacement):
    model = _ContextOverrideModel(replacement=replacement)
    model.fit(_monthly_y([100.0, 110.0, 121.0]))
    conditioning = _monthly_y([130.0], start="2020-04-30")

    model.forecast(steps=1, y=conditioning)

    assert model.received_forecast_y.loc[pd.Timestamp("2020-04-30"), "target"] == (
        888.0 if replacement else 777.0
    )
    assert conditioning.loc[pd.Timestamp("2020-04-30"), "target"] == 130.0


def test_forecast_context_exposes_published_paths_and_reconciles_native_conditioning():
    fit_history = _monthly_y([100.0, 110.0, 121.0])
    full_history = _monthly_y([100.0, 110.0, 121.0, 133.1])
    native_future = _monthly_y([5.0], start="2020-05-31")
    base = ModelData.from_wide(y=fit_history, frequencies={"target": "M"})
    future = ModelData.from_wide(
        y_conditioning=native_future,
        frequencies={"target": "M"},
        y_conditioning_input_metrics={"target": "diff"},
    )
    raw = base.with_conditioning(future).published_after(
        ModelData.from_wide(y=full_history, frequencies={"target": "M"}),
        pd.Timestamp("2020-03-31"),
    )

    published_only = ForecastContext._from_data(
        base.published_after(
            ModelData.from_wide(y=full_history, frequencies={"target": "M"}),
            pd.Timestamp("2020-03-31"),
        ),
        pd.Timestamp("2020-03-31"),
    )
    assert (
        published_only.y_conditioning.loc[pd.Timestamp("2020-04-30"), "target"] == 133.1
    )

    context = ForecastContext._from_data(raw, pd.Timestamp("2020-03-31"))
    expected = pd.DataFrame(
        {"target": [12.1, 5.0]},
        index=pd.date_range("2020-04-30", periods=2, freq="ME"),
    )
    pd.testing.assert_frame_equal(context.y_conditioning, expected)
    assert context.y_conditioning_input_metrics == {"target": "diff"}

    model = _RecordingModel(data_transformation={"target": "diff"})
    model.fit(fit_history, frequency="M")
    model.predict(context, steps=2)
    pd.testing.assert_frame_equal(model.received_forecast_y.loc[expected.index], expected)
    pd.testing.assert_frame_equal(native_future, _monthly_y([5.0], start="2020-05-31"))
    pd.testing.assert_frame_equal(full_history, _monthly_y([100.0, 110.0, 121.0, 133.1]))


def test_projection_frequency_mutation_does_not_annotate_raw_data_or_forecasting():
    index = pd.DatetimeIndex(
        pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]), freq=None
    )
    source = pd.DataFrame({"target": [100.0, 110.0, 121.0]}, index=index)
    model = _RecordingModel()
    model.fit(source)

    projection = model._raw_data.to_wide("y")
    projection.index.freq = "ME"
    assert model._raw_data.index("y").freq is None
    assert source.index.freq is None

    model.forecast(steps=1)

    assert model._raw_data.index("y").freq is None
    assert source.index.freq is None


def test_unknown_input_metric_is_rejected_without_formula():
    with pytest.raises(ValueError, match="y_input_metrics contains variables"):
        _RecordingModel().fit(
            _monthly_y([100.0, 110.0, 121.0]),
            y_input_metrics={"unknown": "levels"},
        )


def test_formula_ignores_input_metric_entries_for_unselected_columns():
    y = pd.DataFrame(
        {
            "target": [100.0, 110.0, 121.0],
            "unused": [1.0, 2.0, 3.0],
        },
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    X = pd.DataFrame(
        {"feature": [10.0, 20.0, 30.0], "unused_feature": [4.0, 5.0, 6.0]},
        index=y.index,
    )
    model = _RecordingModel(formula="target ~ feature")

    model.fit(
        y,
        X,
        y_input_metrics={"target": "levels", "unused": "diff"},
        X_input_metrics={"feature": "levels", "unused_feature": "diff"},
    )

    assert list(model.y.columns) == ["target"]
    assert list(model.X.columns) == ["feature"]


@pytest.mark.parametrize(
    ("mapping", "error", "message"),
    [
        ({"target": "unknown"}, ValueError, "supported metrics"),
        (["target"], TypeError, "dict\\[str, str\\]"),
        ({"target": 1}, TypeError, "map str to str"),
    ],
)
def test_invalid_input_metric_mappings_keep_their_errors(mapping, error, message):
    with pytest.raises(error, match=message):
        _RecordingModel().fit(_monthly_y([100.0, 110.0, 121.0]), y_input_metrics=mapping)


@pytest.mark.parametrize("frequency", ["D", "W"])
def test_pipeline_rejects_explicit_non_monthly_or_quarterly_frequency(frequency):
    history = _monthly_y([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Unsupported frequency"):
        _RecordingModel().fit(
            history,
            data_transformation={"target": "levels"},
            frequency=frequency,
        )


def test_mapping_free_daily_fit_and_forecast_remain_supported():
    daily = pd.DataFrame(
        {"target": [1.0, 2.0, 3.0]},
        index=pd.date_range("2020-01-01", periods=3, freq="D"),
    )
    result = _RecordingModel()

    result.fit(daily)
    forecast = result.forecast(steps=2)

    assert forecast.index.equals(pd.date_range("2020-01-04", periods=2, freq="D"))


def test_public_model_signatures_and_context_shape_remain_stable():
    assert tuple(inspect.signature(ForecastModel.fit).parameters) == (
        "self",
        "y",
        "X",
        "y_lags",
        "X_lags",
        "dummies",
        "data_transformation",
        "frequency",
        "X_imputation",
        "input_frequencies",
        "y_input_metrics",
        "X_input_metrics",
        "drop_transformation_nans",
        "kwargs",
    )
    assert tuple(inspect.signature(ForecastModel.forecast).parameters) == (
        "self",
        "steps",
        "X",
        "y",
        "decomp",
        "data_transformation",
        "frequency",
        "X_imputation",
        "context",
        "kwargs",
    )
    assert tuple(inspect.signature(ForecastModel.predict).parameters) == (
        "self",
        "context",
        "steps",
        "decomp",
        "data_transformation",
        "frequency",
        "X_imputation",
        "kwargs",
    )
    assert tuple(inspect.signature(ForecastTree.forecast).parameters) == (
        "self",
        "steps",
        "X",
        "y",
        "decomp",
        "context",
        "kwargs",
    )
    assert "_model_data" not in {field.name for field in fields(ForecastContext)}
