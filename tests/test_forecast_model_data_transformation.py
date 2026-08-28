"""Tests for the model-owned data transformation mapping on
``ForecastModel``.

The model setting has the same mapping shape as the call-level
``data_transformation`` argument. ``DataTransformationPipeline`` remains an
internal execution helper.
"""

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

from forecast_realtime._utils import regularise_missing_rows
from forecast_realtime.forecast_model import ForecastModel, NoUsableTransformedYError
from forecast_realtime.models.ols import ForecastOLS
from forecast_realtime.tree_regression import TreeRegression


class _StubModel(ForecastModel):
    """Minimal concrete subclass used only to exercise the base contract."""

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return pd.DataFrame()


def test_model_without_transformation_uses_identity_transform():
    model = _StubModel()

    assert model.resolve_input_data_transformation() is None


def test_model_owned_transformation_overrides_call_fallback():
    model = _StubModel(data_transformation={"gdp": "diff"})

    assert model.data_transformation == {"gdp": "diff"}
    assert model.resolve_input_data_transformation(
        {"gdp": "logs"}
    ).data_transformation == {"gdp": "diff"}


def test_model_owned_transformation_is_a_plain_mapping():
    model = _StubModel(data_transformation={"gdp": "diff", "x": "logs"})

    assert model.data_transformation == {"gdp": "diff", "x": "logs"}


def test_model_owned_transformation_survives_copy_and_pickle():
    model = _StubModel(data_transformation={"gdp": "diff"})

    copied = copy.deepcopy(model)
    restored = pickle.loads(pickle.dumps(model))

    assert copied.data_transformation == {"gdp": "diff"}
    assert restored.data_transformation == {"gdp": "diff"}


class _RecordingModel(ForecastModel):
    """Concrete subclass that records exactly what ``_fit``/``_forecast`` receive."""

    def _fit(self, y, X=None, **kwargs):
        self.received_fit_y = y.copy()
        self.received_fit_X = X.copy() if X is not None else None
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        self.received_forecast_y = y.copy() if y is not None else None
        self.received_forecast_X = X.copy() if X is not None else None
        self.received_forecast_origin = kwargs.get("forecast_origin")
        return np.zeros((steps, len(self.y.columns)))


class _LegacyStubModel(ForecastModel):
    """Concrete subclass that never calls ``super().__init__()``."""

    def __init__(self):
        pass

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return pd.DataFrame()


def test_explicit_forecast_frequency_overrides_period_index_frequency():
    index = pd.period_range("2020Q1", periods=3, freq="Q")

    dates = ForecastModel._infer_forecast_dates(index, steps=2, frequency="M")

    assert dates.equals(
        pd.DatetimeIndex([pd.Timestamp("2020-10-31"), pd.Timestamp("2020-11-30")])
    )


def test_data_transformation_defaults_to_none():
    model = _StubModel()

    assert model.data_transformation is None


def test_formula_selects_raw_inputs_before_transformation_and_forecast_preparation():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "unused_target": [np.nan, np.nan, 3.0, 4.0],
        },
        index=index,
    )
    X = pd.DataFrame(
        {
            "used": [10.0, 20.0, 30.0, 40.0],
            "unused": [np.nan, np.nan, 30.0, 40.0],
        },
        index=index,
    )
    mapping = {
        "target": "levels",
        "unused_target": "diff",
        "used": "levels",
        "unused": "diff",
    }
    model = _RecordingModel(formula="target ~ used", data_transformation=mapping)

    model.fit(
        y,
        X,
        y_input_metrics={column: "levels" for column in y},
        X_input_metrics={column: "levels" for column in X},
    )
    future_index = pd.date_range("2020-05-31", periods=1, freq="ME")
    model.forecast(
        steps=1,
        X=pd.DataFrame({"used": [50.0], "unused": [np.nan]}, index=future_index),
    )

    assert model.received_fit_y.index[0] == index[0]
    assert list(model.received_fit_y.columns) == ["target"]
    assert list(model.received_fit_X.columns) == ["used"]
    assert list(model.received_forecast_X.columns) == ["used"]


def test_data_transformation_accepts_dict():
    model = _StubModel(data_transformation={"gdp": "diff"})

    assert model.data_transformation == {"gdp": "diff"}


def test_data_transformation_copies_mapping():
    mapping = {"gdp": "diff"}
    model = _StubModel(data_transformation=mapping)
    mapping["gdp"] = "levels"

    assert model.data_transformation == {"gdp": "diff"}


def test_data_transformation_invalid_constructor_value_raises():
    with pytest.raises(TypeError, match="data_transformation"):
        _StubModel(data_transformation=["gdp"])


def test_data_transformation_invalid_constructor_key_raises():
    with pytest.raises(TypeError, match="data_transformation"):
        _StubModel(data_transformation={1: "diff"})


def test_data_transformation_invalid_constructor_value_type_raises():
    with pytest.raises(TypeError, match="data_transformation"):
        _StubModel(data_transformation={"gdp": 1})


def test_data_transformation_assignable_after_construction():
    model = _StubModel()

    model.data_transformation = {"gdp": "log diff"}

    assert model.data_transformation == {"gdp": "log diff"}


def test_data_transformation_invalid_assignment_raises():
    model = _StubModel()

    with pytest.raises(TypeError, match="data_transformation"):
        model.data_transformation = "diff"


def test_data_transformation_invalid_assignment_key_raises():
    model = _StubModel()

    with pytest.raises(TypeError, match="data_transformation"):
        model.data_transformation = {("gdp",): "diff"}


def test_data_transformation_invalid_assignment_value_type_raises():
    model = _StubModel()

    with pytest.raises(TypeError, match="data_transformation"):
        model.data_transformation = {"gdp": ["diff"]}


def test_data_transformation_none_reset_is_allowed():
    model = _StubModel(data_transformation={"gdp": "diff"})

    model.data_transformation = None

    assert model.data_transformation is None


def test_data_transformation_survives_deepcopy():
    model = _StubModel(data_transformation={"gdp": "diff"})

    cloned = copy.deepcopy(model)

    assert cloned.data_transformation is not model.data_transformation
    assert cloned.data_transformation == model.data_transformation


def test_data_transformation_survives_pickle_roundtrip():
    model = _StubModel(data_transformation={"gdp": "diff"})

    restored = pickle.loads(pickle.dumps(model))

    assert restored.data_transformation == model.data_transformation


# --------------------------------------------------------------------------- #
# Phase 4: fit()/forecast() resolve and apply the pipeline.                    #
# --------------------------------------------------------------------------- #


def _levels_y(values, start="2020-01-31", freq="ME"):
    return pd.DataFrame(
        {"gdp": values}, index=pd.date_range(start, periods=len(values), freq=freq)
    )


@pytest.mark.parametrize(
    ("frequency", "dates", "expected_gap"),
    [
        ("M", ["2020-01-01", "2020-03-01"], "2020-02-01"),
        ("M", ["2020-01-31", "2020-03-31"], "2020-02-29"),
        ("Q", ["2020-01-01", "2020-07-01"], "2020-04-01"),
        ("Q", ["2020-03-31", "2020-09-30"], "2020-06-30"),
    ],
)
def test_regularise_missing_rows_preserves_source_anchor(frequency, dates, expected_gap):
    data = pd.DataFrame({"value": [1.0, 3.0]}, index=pd.to_datetime(dates))

    result = regularise_missing_rows(data, {"value": frequency})

    assert pd.Timestamp(expected_gap) in result.index
    assert result.index[result.index.isin(pd.to_datetime(dates))].equals(
        pd.DatetimeIndex(dates)
    )


def test_forecast_reuses_fitted_frequency_mapping_for_regularisation(monkeypatch):
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    history = _levels_y([100.0, 110.0, 121.0])
    model.fit(history, frequency="M")

    def fail_inference(*args, **kwargs):
        raise AssertionError("prediction must reuse fitted frequency mappings")

    monkeypatch.setattr(
        "forecast_realtime.forecast_model.infer_variable_frequencies",
        fail_inference,
    )
    model.forecast(
        steps=1,
        y=pd.DataFrame(
            {"gdp": [130.0]},
            index=pd.to_datetime(["2020-04-30"]),
        ),
    )


def test_forecast_imputation_uses_stored_target_frequency():
    y = pd.DataFrame(
        {"target": [100.0, 110.0, 121.0, 130.0]},
        index=pd.date_range("2019-06-30", periods=4, freq="QE-DEC"),
    )
    X = pd.DataFrame(
        {"driver": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        index=pd.date_range("2019-10-31", periods=6, freq="ME"),
    )
    model = _RecordingModel(data_transformation={"target": "levels", "driver": "levels"})
    model.fit(y, X, frequency="Q", X_imputation="last")

    assert model._fitted_model_configuration.data_transformation.X_frequency_mapping == {
        "driver": "M"
    }

    model.forecast(
        steps=1,
        X=pd.DataFrame(
            {"driver": [7.0]},
            index=pd.to_datetime(["2020-04-30"]),
        ),
    )

    assert model.received_forecast_X.index[-1] == pd.Timestamp("2020-06-30")


def test_forecast_dummies_use_stored_target_frequency_without_X_design():
    y = pd.DataFrame(
        {"target": [100.0, 110.0, 121.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    X = pd.DataFrame(
        {"driver": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        index=pd.date_range("2019-10-31", periods=6, freq="ME"),
    )
    model = _RecordingModel()
    model.fit(y, X, frequency="Q", dummies=["2020-01-31"])

    model.forecast(steps=1)

    assert model.received_forecast_X.index[-1] == pd.Timestamp("2020-06-30")


def test_nan_intolerant_fit_and_forecast_regularisation_preserves_mixed_anchors():
    class _NoMissingValuesModel(_RecordingModel):
        _handles_missing_values = False

        def _forecast(self, steps=1, X=None, y=None, **kwargs):
            self.received_forecast_y = y.copy() if y is not None else None
            return pd.DataFrame(
                np.zeros((steps, 1)),
                index=pd.date_range("2020-07-31", periods=steps, freq="ME"),
                columns=self.y.columns,
            )

    y_dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    x_dates = pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"])
    panel_index = y_dates.union(x_dates)
    y = pd.DataFrame(
        {
            "target": pd.Series([100.0, 110.0, 115.0, 120.0], index=y_dates).reindex(
                panel_index
            )
        },
        index=panel_index,
    )
    X = pd.DataFrame(
        {"driver": pd.Series([10.0, 20.0, 30.0], index=x_dates).reindex(panel_index)},
        index=panel_index,
    )
    model = _NoMissingValuesModel()

    model.fit(
        y,
        X,
    )

    assert pd.Timestamp("2020-03-31") in model._y_history.index
    assert pd.Timestamp("2020-06-30") in model._X_history.index

    conditioning = pd.DataFrame({"target": [140.0]}, index=pd.to_datetime(["2020-06-30"]))
    model.forecast(
        steps=1,
        y=conditioning,
    )

    assert pd.Timestamp("2020-05-31") in model.received_forecast_y.index
    assert pd.Timestamp("2020-06-30") in model.received_forecast_y.index


def test_fit_applies_instance_owned_pipeline_before_fit():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, frequency="M")

    # First differencing NaN row dropped by default (drop_transformation_nans=True).
    np.testing.assert_allclose(model.received_fit_y["gdp"].to_numpy(), [10.0, 11.0])
    assert list(model.received_fit_y.index) == list(y.index[1:])


def test_fit_defaults_missing_input_metric_mappings_to_levels():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, frequency="M")

    transformation = model._fitted_model_configuration.data_transformation
    assert transformation.y_input_metric_mapping == {"gdp": "levels"}
    np.testing.assert_allclose(model.received_fit_y["gdp"].to_numpy(), [10.0, 11.0])


def test_direct_fit_without_pipeline_stores_level_input_defaults():
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, frequency="M")

    transformation = model._fitted_model_configuration.data_transformation
    assert transformation.data_transformation is None
    assert transformation.y_input_metric_mapping == {"gdp": "levels"}


def test_direct_fit_rejects_derived_input_when_levels_are_requested_by_default():
    model = _RecordingModel()
    y = pd.DataFrame(
        {"gdp": [10.0, 11.0, 15.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )

    with pytest.raises(
        ValueError, match="Cannot transform variable 'gdp' from metric 'diff' to 'levels'"
    ):
        model.fit(y, y_input_metrics={"gdp": "diff"}, frequency="M")


def test_fit_persists_explicit_input_metric_mappings_and_uses_identity():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = pd.DataFrame(
        {"gdp": [10.0, 11.0, 15.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )

    model.fit(y, y_input_metrics={"gdp": "diff"}, frequency="M")

    transformation = model._fitted_model_configuration.data_transformation
    assert transformation.y_input_metric_mapping == {"gdp": "diff"}
    np.testing.assert_allclose(model.received_fit_y["gdp"].to_numpy(), y["gdp"])


def test_forecast_reuses_fitted_input_metric_mapping_for_conditioning():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = pd.DataFrame(
        {"gdp": [10.0, 11.0, 15.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    model.fit(y, y_input_metrics={"gdp": "diff"}, frequency="M")

    conditioning = pd.DataFrame({"gdp": [17.0]}, index=pd.DatetimeIndex(["2020-04-30"]))
    model.forecast(steps=1, y=conditioning)

    np.testing.assert_allclose(
        model.received_forecast_y["gdp"].to_numpy(), [10.0, 11.0, 15.0, 17.0]
    )


def test_fit_stores_target_metrics_from_data_transformation():
    model = _RecordingModel(data_transformation={"gdp": "levels", "inflation": "diff"})
    y = pd.DataFrame(
        {
            "gdp": [100.0, 110.0, 121.0],
            "inflation": [2.0, 2.1, 2.2],
        },
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )

    model.fit(y, frequency="M")

    assert model.native_metric_mapping() == {"gdp": "levels", "inflation": "diff"}


def test_fit_uses_explicit_transformation_frequency_for_each_raw_column():
    monthly_dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    quarterly_dates = pd.date_range("2019-03-31", periods=5, freq="QE")
    index = monthly_dates.union(quarterly_dates)
    y = pd.DataFrame(
        {
            "monthly": pd.Series(np.arange(100.0, 115.0), index=monthly_dates).reindex(
                index
            ),
            "quarterly": pd.Series(
                [100.0, 102.0, 104.0, 106.0, 110.0], index=quarterly_dates
            ).reindex(index),
        },
        index=index,
    )
    model = _RecordingModel(data_transformation={"monthly": "yoy", "quarterly": "yoy"})

    model.fit(
        y,
        frequency="M",
        input_frequencies={"monthly": "M", "quarterly": "Q"},
        drop_transformation_nans=False,
    )

    np.testing.assert_allclose(
        model.received_fit_y.loc[pd.Timestamp("2020-03-31"), "monthly"],
        (114.0 / 102.0 - 1) * 100,
    )
    np.testing.assert_allclose(
        model.received_fit_y.loc[pd.Timestamp("2020-03-31"), "quarterly"],
        (110.0 / 100.0 - 1) * 100,
    )


def test_fit_applies_call_level_fallback_pipeline_when_no_instance_pipeline():
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, data_transformation={"gdp": "diff"}, frequency="M")

    np.testing.assert_allclose(model.received_fit_y["gdp"].to_numpy(), [10.0, 11.0])


def test_forecast_reuses_call_level_fallback_pipeline_from_fit():
    model = _RecordingModel()
    index = pd.date_range("2020-01-31", periods=3, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=index)
    X = pd.DataFrame({"driver": [10.0, 20.0, 30.0]}, index=index)

    model.fit(
        y,
        X,
        data_transformation={"target": "levels", "driver": "diff"},
        frequency="M",
    )

    future = pd.DataFrame(
        {"driver": [40.0]},
        index=pd.date_range("2020-04-30", periods=1, freq="ME"),
    )
    model.forecast(steps=1, X=future)

    assert model.received_forecast_X.loc[future.index, "driver"].item() == 10.0


def test_forecast_rejects_conflicting_fit_time_transformation():
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y, data_transformation={"gdp": "diff"}, frequency="M")

    with pytest.raises(ValueError, match="data_transformation.*conflicts"):
        model.forecast(
            steps=1,
            data_transformation={"gdp": "levels"},
            frequency="M",
        )


def test_fitted_transformation_copies_caller_mapping_and_survives_pickle():
    model = _RecordingModel()
    transformation = {"gdp": "diff"}
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y, data_transformation=transformation, frequency="M")
    transformation["gdp"] = "levels"

    restored = pickle.loads(pickle.dumps(model))
    conditioning = _levels_y([130.0], start="2020-04-30", freq="ME")
    restored.forecast(steps=1, y=conditioning)

    np.testing.assert_allclose(
        restored.received_forecast_y.loc[conditioning.index, "gdp"].to_numpy(),
        [9.0],
    )


def test_forecast_uses_fitted_configuration_after_public_state_mutation():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y, frequency="M")

    model.last_y_fit_date = pd.Timestamp("2021-12-31")
    model._forecast_frequency = "Q"
    conditioning = _levels_y([130.0], start="2020-04-30", freq="ME")

    forecast = model.forecast(steps=1, y=conditioning)

    assert forecast.index[0] == pd.Timestamp("2020-04-30")
    np.testing.assert_allclose(
        model.received_forecast_y.loc[conditioning.index, "gdp"].to_numpy(),
        [9.0],
    )


def test_fit_drops_leading_nan_row_for_pop_by_default():
    """The "pop" transformation leaves a leading NaN row, so it must be dropped as with
    "diff"/"log diff" when drop_transformation_nans defaults to True."""
    model = _RecordingModel(data_transformation={"gdp": "pop"})
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, frequency="M")

    np.testing.assert_allclose(model.received_fit_y["gdp"].to_numpy(), [10.0, 10.0])
    assert list(model.received_fit_y.index) == list(y.index[1:])


def test_fit_retains_leading_nan_row_for_pop_when_flag_false():
    model = _RecordingModel(data_transformation={"gdp": "pop"})
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, frequency="M", drop_transformation_nans=False)

    assert np.isnan(model.received_fit_y["gdp"].iloc[0])
    assert list(model.received_fit_y.index) == list(y.index)


def test_fit_drops_all_leading_nan_rows_for_yoy_by_default():
    """The "yoy" transformation leaves 12 leading NaN rows, all of which
    must be dropped by default."""
    values = list(np.linspace(100, 111, 13))
    model = _RecordingModel(data_transformation={"cpi": "yoy"})
    y = pd.DataFrame(
        {"cpi": values}, index=pd.date_range("2019-01-31", periods=13, freq="ME")
    )

    model.fit(y, frequency="M")

    assert len(model.received_fit_y) == 1
    assert not model.received_fit_y["cpi"].isna().any()
    np.testing.assert_allclose(
        model.received_fit_y["cpi"].iloc[0], (values[12] / values[0] - 1) * 100
    )


def test_fit_short_yoy_target_raises_before_X_imputation():
    y = pd.DataFrame(
        {"cpi": [100.0, 101.0, 102.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    X = pd.DataFrame({"driver": [1.0, 2.0]}, index=y.index[:2])
    model = _RecordingModel(data_transformation={"cpi": "yoy", "driver": "levels"})

    with pytest.raises(NoUsableTransformedYError, match="after transformation"):
        model.fit(y, X, frequency="M", X_imputation="last")


def test_fit_retains_all_leading_nan_rows_for_yoy_when_flag_false():
    values = list(np.linspace(100, 111, 13))
    model = _RecordingModel(data_transformation={"cpi": "yoy"})
    y = pd.DataFrame(
        {"cpi": values}, index=pd.date_range("2019-01-31", periods=13, freq="ME")
    )

    model.fit(y, frequency="M", drop_transformation_nans=False)

    assert len(model.received_fit_y) == 13
    assert model.received_fit_y["cpi"].iloc[:12].isna().all()


def test_fit_yoy_monthly_missing_calendar_period_preserves_valid_first_observation():
    """An early missing calendar month must not push the leading-NaN drop past
    a genuinely defined yoy observation.

    Regression test: dropping the metric's nominal calendar lag (12 rows for
    monthly "yoy") assumes no calendar gaps. With month 3 (2019-03) entirely
    missing, the raw frame has only 13 rows, so ``iloc[12:]`` would keep only
    the last row and incorrectly discard the defined 2020-01 observation.
    """
    dates = pd.date_range("2019-01-31", periods=14, freq="ME").delete(2)
    values = [100.0 + i for i in range(14)]
    del values[2]
    model = _RecordingModel(data_transformation={"cpi": "yoy"})
    y = pd.DataFrame({"cpi": values}, index=dates)

    model.fit(y, frequency="M")

    assert len(model.received_fit_y) == 2
    assert not model.received_fit_y["cpi"].isna().any()
    np.testing.assert_allclose(
        model.received_fit_y["cpi"].to_numpy(),
        [(112.0 / 100.0 - 1) * 100, (113.0 / 101.0 - 1) * 100],
    )


def test_fit_yoy_quarterly_leading_raw_nan_drops_full_undefined_prefix():
    """Raw leading ``NaN`` values (present rows, undefined values) can leave
    more leading rows undefined than the metric's nominal calendar lag.

    Regression test: dropping only the nominal lag (4 rows for quarterly
    "yoy") would leave two more rows undefined here, because the fifth and
    sixth quarters still divide by the raw ``NaN`` first/second quarters.
    """
    dates = pd.date_range("2019-03-31", periods=8, freq="QE")
    values = [np.nan, np.nan, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    model = _RecordingModel(data_transformation={"gdp": "yoy"})
    y = pd.DataFrame({"gdp": values}, index=dates)

    model.fit(y, frequency="Q")

    assert len(model.received_fit_y) == 2
    assert not model.received_fit_y["gdp"].isna().any()
    np.testing.assert_allclose(
        model.received_fit_y["gdp"].to_numpy(),
        [(106.0 / 102.0 - 1) * 100, (107.0 / 103.0 - 1) * 100],
    )


def test_fit_instance_pipeline_takes_precedence_over_fallback():
    model = _RecordingModel(data_transformation={"gdp": "levels"})
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, data_transformation={"gdp": "diff"}, frequency="M")

    np.testing.assert_allclose(
        model.received_fit_y["gdp"].to_numpy(), [100.0, 110.0, 121.0]
    )


def test_fit_without_any_pipeline_leaves_data_unchanged():
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y)

    pd.testing.assert_frame_equal(model.received_fit_y, y)


def test_fit_with_pipeline_infers_frequency_without_global_frequency():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y)

    np.testing.assert_allclose(model.received_fit_y["gdp"].to_numpy(), [10.0, 11.0])


def test_fit_stores_raw_history_distinct_from_prepared_history():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = _levels_y([100.0, 110.0, 121.0])

    model.fit(y, frequency="M")

    np.testing.assert_allclose(
        model._raw_y_history["gdp"].to_numpy(), [100.0, 110.0, 121.0]
    )
    np.testing.assert_allclose(model.y["gdp"].to_numpy(), [10.0, 11.0])


def test_forecast_conditioning_uses_final_raw_historical_observation():
    """The first forecasted difference must use the last raw fitted level, not
    a value from a previously-transformed frame."""
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y, frequency="M")

    conditioning = _levels_y([130.0], start="2020-04-30", freq="ME")
    model.forecast(steps=1, y=conditioning, frequency="M")

    np.testing.assert_allclose(model.received_forecast_y["gdp"].to_numpy()[-1:], [9.0])


def test_forecast_passes_fitted_target_boundary_with_conditioning():
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y)

    conditioning = _levels_y([130.0, 140.0], start="2020-04-30", freq="ME")
    model.forecast(steps=2, y=conditioning)

    assert model.received_forecast_origin == pd.Timestamp("2020-03-31")
    assert model.received_forecast_y.index[-1] == pd.Timestamp("2020-05-31")


def test_forecast_does_not_double_transform():
    model = _RecordingModel(data_transformation={"gdp": "diff"})
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y, frequency="M")

    conditioning = _levels_y([130.0, 140.0], start="2020-04-30", freq="ME")
    model.forecast(steps=2, y=conditioning, frequency="M")

    # 130-121=9, then 140-130=10; NOT diff-of-diff.
    np.testing.assert_allclose(
        model.received_forecast_y["gdp"].to_numpy()[-2:], [9.0, 10.0]
    )


def test_forecast_regularises_missing_rows_before_lag_construction():
    class _NoMissingValuesModel(_RecordingModel):
        _handles_missing_values = False

        def _forecast(self, steps=1, X=None, y=None, **kwargs):
            self.received_forecast_y = y.copy() if y is not None else None
            return pd.DataFrame(
                np.zeros((steps, 1)),
                index=pd.date_range("2020-05-31", periods=steps, freq="ME"),
                columns=self.y.columns,
            )

    history = pd.DataFrame(
        {"gdp": [100.0, 110.0, 130.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-04-30"]),
    )
    model = _NoMissingValuesModel(data_transformation={"gdp": "levels"})
    model.fit(history, y_lags=1)

    model.forecast(
        steps=1,
        y=pd.DataFrame({"gdp": [140.0]}, index=pd.to_datetime(["2020-05-31"])),
    )

    assert pd.Timestamp("2020-03-31") in model.received_forecast_y.index
    assert pd.isna(model.received_forecast_y.loc[pd.Timestamp("2020-03-31"), "gdp"])


def test_forecast_uses_mixed_target_frequencies_for_conditioning():
    monthly_dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    quarterly_dates = pd.date_range("2019-03-31", periods=5, freq="QE")
    index = monthly_dates.union(quarterly_dates)
    y = pd.DataFrame(
        {
            "monthly": pd.Series(np.arange(100.0, 115.0), index=monthly_dates).reindex(
                index
            ),
            "quarterly": pd.Series(
                [100.0, 102.0, 104.0, 106.0, 108.0], index=quarterly_dates
            ).reindex(index),
        },
        index=index,
    )
    model = _RecordingModel(data_transformation={"monthly": "yoy", "quarterly": "yoy"})
    model.fit(
        y,
        frequency="M",
        input_frequencies={"monthly": "M", "quarterly": "Q"},
        drop_transformation_nans=False,
    )

    conditioning = pd.DataFrame(
        {
            "monthly": [115.0],
            "quarterly": [110.0],
        },
        index=pd.DatetimeIndex(["2020-04-30"]),
    )
    model.forecast(steps=1, y=conditioning, frequency="M")

    np.testing.assert_allclose(
        model.received_forecast_y.loc[pd.Timestamp("2020-04-30"), "monthly"],
        (115.0 / 103.0 - 1) * 100,
    )
    assert np.isnan(
        model.received_forecast_y.loc[pd.Timestamp("2020-04-30"), "quarterly"]
    )


def test_forecast_without_pipeline_merges_raw_history_with_conditioning():
    """An untransformed (no pipeline) autoregressive model must still see its
    raw fitted history ahead of a conditioning path, not the conditioning
    path alone."""
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y)

    conditioning = _levels_y([130.0], start="2020-04-30", freq="ME")
    model.forecast(steps=1, y=conditioning)

    np.testing.assert_allclose(
        model.received_forecast_y["gdp"].to_numpy(), [100.0, 110.0, 121.0, 130.0]
    )


def test_forecast_without_pipeline_merges_raw_X_history_with_future():
    """The same raw-history merge applies to X when no pipeline is set."""
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])
    X = pd.DataFrame({"x": [10.0, 20.0, 30.0]}, index=y.index)
    model.fit(y, X)

    X_future = pd.DataFrame(
        {"x": [40.0]}, index=pd.date_range("2020-04-30", periods=1, freq="ME")
    )
    model.forecast(steps=1, X=X_future)

    np.testing.assert_allclose(
        model.received_forecast_X["x"].to_numpy(), [10.0, 20.0, 30.0, 40.0]
    )


@pytest.mark.parametrize("pipeline", [None, {"gdp": "levels", "x": "levels"}])
def test_forecast_without_X_preserves_absent_regressor_intent(pipeline):
    model = _RecordingModel(data_transformation=pipeline)
    y = _levels_y([100.0, 110.0, 121.0])
    X = pd.DataFrame({"x": [10.0, 20.0, 30.0]}, index=y.index)
    model.fit(y, X)

    model.forecast(steps=1)

    assert model.received_forecast_X is None


def test_forecast_without_pipeline_future_wins_on_overlap():
    """A conditioning row sharing a date with fitted history takes precedence,
    matching the pipeline-resolved merge contract."""
    model = _RecordingModel()
    y = _levels_y([100.0, 110.0, 121.0])
    model.fit(y)

    conditioning = _levels_y([999.0], start="2020-03-31", freq="ME")
    model.forecast(steps=1, y=conditioning)

    np.testing.assert_allclose(
        model.received_forecast_y["gdp"].to_numpy(), [100.0, 110.0, 999.0]
    )


@pytest.mark.parametrize("forecast_strategy", ["recursive", "direct"])
def test_ols_forecast_rows_use_fitted_target_boundary_with_conditioning(
    forecast_strategy,
):
    model_kwargs = {"forecast_strategy": forecast_strategy}
    if forecast_strategy == "direct":
        model_kwargs["steps"] = 2
    model = ForecastOLS(**model_kwargs)
    y = _levels_y([100.0, 110.0, 121.0])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=y.index)
    model.fit(y, X)

    conditioning = _levels_y([130.0, 140.0], start="2020-04-30", freq="ME")
    X_future = pd.DataFrame(
        {"x": [4.0, 5.0]},
        index=pd.date_range("2020-04-30", periods=2, freq="ME"),
    )

    model.forecast(steps=2, X=X_future, y=conditioning)

    assert model.last_y_fit_date == pd.Timestamp("2020-03-31")
    assert list(
        model._select_forecast_rows(
            pd.concat([X, X_future]), model.last_y_fit_date, 2, conditioning
        ).index
    ) == list(X_future.index)


@pytest.mark.parametrize("forecast_strategy", ["recursive", "direct"])
def test_tree_forecast_rows_use_fitted_target_boundary_with_conditioning(
    forecast_strategy,
):
    class RecordingTree(TreeRegression):
        captured_predictions = []

        def _build_estimator(self):
            captured_predictions = self.captured_predictions

            class Estimator:
                def fit(self, X, y):
                    return self

                def predict(self, X):
                    captured_predictions.append(np.asarray(X).copy())
                    return np.zeros(len(X))

            return Estimator()

    model_kwargs = {"forecast_strategy": forecast_strategy}
    if forecast_strategy == "direct":
        model_kwargs["steps"] = 2
    model = RecordingTree(**model_kwargs)
    y = _levels_y([100.0, 110.0, 121.0])
    X = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=y.index)
    model.fit(y, X)

    conditioning = _levels_y([130.0, 140.0], start="2020-04-30", freq="ME")
    X_future = pd.DataFrame(
        {"x": [4.0, 5.0]},
        index=pd.date_range("2020-04-30", periods=2, freq="ME"),
    )

    model.forecast(steps=2, X=X_future, y=conditioning)

    assert model.last_y_fit_date == pd.Timestamp("2020-03-31")
    if forecast_strategy == "direct":
        assert len(RecordingTree.captured_predictions) >= 3
        np.testing.assert_allclose(RecordingTree.captured_predictions[-1], [[4.0]])
    else:
        np.testing.assert_allclose(RecordingTree.captured_predictions[-1], [[4.0], [5.0]])


def test_fit_ragged_edge_imputation_applied_after_transformation():
    """Ragged-edge padding must use the transformed metric, not raw levels."""
    model = _RecordingModel(data_transformation={"gdp": "diff", "x": "diff"})
    y = _levels_y([100.0, 110.0, 121.0, 130.0, 145.0])
    X = pd.DataFrame(
        {"x": [10.0, 20.0, 40.0, 70.0]},
        index=pd.date_range("2020-01-31", periods=4, freq="ME"),
    )

    model.fit(y, X, frequency="M", X_imputation="last")

    # x diffs to [10.0, 20.0, 30.0] at Feb/Mar/Apr (Jan dropped as the leading
    # transformation NaN); ragged-edge padding then repeats the last
    # transformed value (30.0) up to gdp's last fitted date (May), not a raw
    # level.
    np.testing.assert_allclose(
        model.received_fit_X["x"].to_numpy(), [10.0, 20.0, 30.0, 30.0]
    )


def test_fit_missing_row_regularisation_applied_after_transformation():
    """Regularisation materialises the absent March row so lag construction
    treats April as two periods after February, not one."""

    class _NoMissingValuesModel(_RecordingModel):
        _handles_missing_values = False

    y = pd.DataFrame(
        {"gdp": [100.0, 110.0, 130.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-04-30"]),
    )

    regularised_model = _NoMissingValuesModel(data_transformation={"gdp": "levels"})
    regularised_model.fit(y, y_lags=1, frequency="M")

    # April's lag1 falls on the regularised (NaN) March row, so only February
    # (lag1 = January) survives complete-case estimation.
    assert len(regularised_model.received_fit_y) == 1
    assert regularised_model.received_fit_y.index[0] == pd.Timestamp("2020-02-29")

    inferred_model = _NoMissingValuesModel()
    inferred_model.fit(y, y_lags=1)

    assert len(inferred_model.received_fit_y) == 1


def test_fit_missing_row_regularisation_noop_without_frequency():
    class _NoMissingValuesModel(_RecordingModel):
        _handles_missing_values = False

    model = _NoMissingValuesModel()
    y = pd.DataFrame(
        {"gdp": [100.0, 110.0, 130.0]},
        index=pd.to_datetime(["2020-01-31", "2020-02-29", "2020-04-30"]),
    )

    model.fit(y)

    assert pd.Timestamp("2020-03-31") not in model.received_fit_y.index
