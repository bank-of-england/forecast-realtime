import numpy as np
import pandas as pd
import pytest

from forecast_realtime.forecast_model import (
    ForecastModel,
    ForecastResult,
    _point_forecast_to_wide,
)
from forecast_realtime.models.ols import ForecastOLS


class _MutatingFitModel(ForecastModel):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.fit_attempts = 0

    def _fit(self, y, X=None, **kwargs):
        self.fit_attempts += 1
        self.fitted_values_ = y.copy()
        if self.fail:
            self.marker = "failed candidate"
            raise RuntimeError("fit failed")
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return self._wrap_forecast(
            np.full((steps, self.y.shape[1]), self.y.iloc[-1, 0]), steps
        )


class _DatedForecastModel(ForecastModel):
    def __init__(self, index_builder):
        super().__init__()
        self.index_builder = index_builder

    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps=1, X=None, y=None, forecast_origin=None, **kwargs):
        index = self.index_builder(forecast_origin, steps)
        return pd.DataFrame(
            {"target": np.arange(float(steps))},
            index=index,
        )


class _DesignRecordingModel(ForecastModel):
    def _fit(self, y, X=None, **kwargs):
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        return np.zeros((steps, len(self.y.columns)))


def _monthly_dates(origin, steps):
    return pd.date_range(origin, periods=steps + 1, freq="ME")[1:]


@pytest.mark.parametrize("include_origin", [False, True])
def test_long_point_result_preserves_hook_payload_and_metadata(include_origin):
    import pickle

    origin = pd.Timestamp("2020-01-31")
    dates = pd.DatetimeIndex(
        [origin if include_origin else pd.Timestamp("2020-02-07"), "2020-04-13"],
        name="date",
    )
    payload = pd.DataFrame({"value": [1.0, np.nan], "date": [3.0, 4.0]}, index=dates)
    model = _DesignRecordingModel().fit(
        pd.DataFrame(
            {"value": [1.0, 2.0, 3.0], "date": [4.0, 5.0, 6.0]},
            index=pd.date_range("2019-11-30", periods=3, freq="ME"),
        )
    )
    model._forecast_dates_include_origin = include_origin
    result = model._finalise_forecast(payload, 2, origin)
    wide = result.forecast.pivot(index="date", columns="variable", values="value")
    pd.testing.assert_frame_equal(
        wide[payload.columns].rename_axis(columns=None), payload
    )
    assert result["variable"].tolist() == ["value", "date", "value", "date"]
    assert isinstance(result.index, pd.RangeIndex)
    assert result.forecast_origin == origin
    assert result.decomposition is None
    for restored in [result.copy(), result.iloc[:2], pickle.loads(pickle.dumps(result))]:
        assert restored.forecast_origin == origin
    assert type(result.forecast) is pd.DataFrame


@pytest.mark.parametrize("method", ["forecast", "predict"])
@pytest.mark.parametrize(
    "corruption, message",
    [
        ("none", None),
        ("missing", "cover every date"),
        ("duplicate", "complete and unique"),
        ("unknown", "cover every date"),
        ("missing_date", "complete and unique"),
        ("string_date", "datetime dtype"),
        ("origin", "strictly after"),
        ("short", "2 dates"),
        ("wide", "Point forecasts must have columns"),
        ("quantile", "Point forecasts must have columns"),
    ],
)
def test_public_point_overrides_validate_once(monkeypatch, method, corruption, message):
    history = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0], "date": [4.0, 5.0, 6.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    model = _DesignRecordingModel().fit(history)
    original = getattr(ForecastModel, method)
    calls = []

    def override(self, *args, **kwargs):
        calls.append(method)
        result = original(self, *args, **kwargs)
        frame = result.forecast.iloc[::-1].reset_index(drop=True)
        if corruption == "missing":
            frame = frame.iloc[1:]
        elif corruption == "duplicate":
            frame = pd.concat([frame, frame.iloc[[0]]])
        elif corruption == "unknown":
            frame.loc[0, "variable"] = "unknown"
        elif corruption == "missing_date":
            frame.loc[0, "date"] = pd.NaT
        elif corruption == "string_date":
            frame["date"] = frame["date"].astype(str)
        elif corruption == "origin":
            frame.loc[frame["date"] == frame["date"].min(), "date"] = history.index[-1]
        elif corruption == "short":
            frame = frame.iloc[:2]
        elif corruption == "wide":
            frame = _point_forecast_to_wide(frame)
        elif corruption == "quantile":
            frame["quantile"] = np.nan
        return ForecastResult(frame, forecast_origin=result.forecast_origin)

    monkeypatch.setattr(_DesignRecordingModel, method, override)
    boundary = (
        model._forecast_from_data if method == "forecast" else model._predict_from_data
    )
    options = {"forecast_origin": history.index[-1], "steps": 2}
    if message:
        with pytest.raises((ValueError, TypeError), match=message):
            boundary(model._raw_data, **options)
    else:
        result = boundary(model._raw_data, **options)
        assert isinstance(result.index, pd.RangeIndex)
        assert result["variable"].tolist() == ["value", "date", "value", "date"]
        assert result["date"].is_monotonic_increasing
        np.testing.assert_array_equal(result["value"], np.zeros(4))
    assert calls == [method]


def test_long_decomposition_reconciles_by_date_and_target():
    dates = pd.to_datetime(["2020-02-07", "2020-04-13"])
    forecast = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "variable": ["z", "a"] * 2,
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    decomposition = pd.DataFrame(
        {
            "forecast_horizon": [1, 0, 1, 0],
            "variable": ["a", "a", "z", "z"],
            "component": "total",
            "contribution": [4.0, 2.0, 3.0, 1.0],
            "weight": 1.0,
        }
    )
    result = ForecastResult(
        forecast.iloc[::-1],
        decomposition=decomposition,
        forecast_origin=pd.Timestamp("2020-01-31"),
        steps=2,
        expected_columns=["z", "a"],
    )
    pd.testing.assert_frame_equal(result.forecast, forecast)
    pd.testing.assert_frame_equal(result.decomposition, decomposition)
    with pytest.raises(ValueError, match="long point forecast"):
        _point_forecast_to_wide(result.assign(quantile=0.5))


def test_forecast_with_lagged_X_reuses_raw_prepared_history():
    index = pd.date_range("2020-01-31", periods=8, freq="ME")
    y = pd.DataFrame({"target": np.arange(8.0)}, index=index)
    X = pd.DataFrame({"driver": np.arange(8.0)}, index=index)
    model = ForecastOLS().fit(y, X, X_lags=1)

    future = pd.DataFrame(
        {"driver": [8.0]}, index=pd.date_range("2020-09-30", periods=1, freq="ME")
    )
    forecast = model.forecast(steps=1, X=future)

    assert len(forecast) == 1


def test_formula_selects_target_before_building_y_lags():
    index = pd.date_range("2020-01-31", periods=8, freq="ME")
    y = pd.DataFrame({"a": np.arange(8.0), "b": np.arange(10.0, 18.0)}, index=index)
    model = ForecastOLS(formula="b ~ b_lag1").fit(y, y_lags=1)

    assert list(model.X.columns) == ["b_lag1"]


def test_short_month_end_index_keeps_dummy_names_and_calendar_dates():
    index = pd.date_range("2020-01-31", periods=2, freq="ME")
    y = pd.DataFrame({"target": [1.0, 10.0]}, index=index)
    X = pd.DataFrame({"driver": [1.0, 2.0]}, index=index)
    model = ForecastOLS().fit(y, X, dummies=[index[1]])

    future = pd.DataFrame(
        {"driver": [3.0]}, index=pd.date_range("2020-03-31", periods=1, freq="ME")
    )
    forecast = model.forecast(steps=1, X=future)

    assert forecast["date"].iloc[0] == pd.Timestamp("2020-03-31")


def test_dummy_names_use_target_frequency_for_mixed_design_index():
    target_index = pd.date_range("2020-01-31", periods=4, freq="QE")
    regressor_index = pd.date_range("2020-01-31", periods=10, freq="ME")
    y = pd.DataFrame({"target": np.arange(4.0)}, index=target_index)
    X = pd.DataFrame({"monthly_driver": np.arange(10.0)}, index=regressor_index)

    model = _DesignRecordingModel().fit(y, X=X, dummies=[target_index[1]])

    assert "D_2020Q2" in model.X.columns


def test_direct_fit_rejects_unknown_X_imputation():
    index = pd.date_range("2020-01-31", periods=3, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=index)
    with pytest.raises(ValueError, match="X_imputation"):
        ForecastOLS().fit(y, X_imputation="unknown")


def test_trailing_missing_target_does_not_advance_regression_origin():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, np.nan]}, index=index)
    X = pd.DataFrame({"driver": [1.0, 2.0, 3.0, 4.0]}, index=index)
    model = ForecastOLS(drop_nans=True).fit(y, X)

    assert model.last_y_fit_date == index[-2]


def test_array_forecast_starts_after_final_target_used_for_fitting():
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0]}, index=index[:-1])

    model = ForecastOLS().fit(y)
    forecast = model.forecast(steps=1)

    assert forecast["date"].iloc[0] == index[-1]


@pytest.mark.parametrize(
    "index_builder, error_message",
    [
        (
            lambda origin, steps: pd.DatetimeIndex(
                [origin + pd.offsets.MonthEnd(1)] * steps
            ),
            "duplicate dates",
        ),
        (
            lambda origin, steps: pd.DatetimeIndex(
                list(reversed(_monthly_dates(origin, steps)))
            ),
            "sorted in increasing order",
        ),
        (
            lambda origin, steps: pd.DatetimeIndex(
                [origin + pd.offsets.MonthEnd(1), pd.NaT]
            ),
            "missing dates",
        ),
        (
            lambda origin, steps: pd.DatetimeIndex(
                [origin, origin + pd.offsets.MonthEnd(1)]
            ),
            "strictly after",
        ),
    ],
)
def test_dated_forecast_validates_index_contract(index_builder, error_message):
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    y = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]}, index=index)

    model = _DatedForecastModel(index_builder).fit(y)

    with pytest.raises(ValueError, match=error_message):
        model.forecast(steps=2)


def _valid_forecast_result_decomposition():
    return pd.DataFrame(
        {
            "forecast_horizon": [0, 0, 1, 1],
            "component": ["first", "second", "first", "second"],
            "contribution": [0.25, 0.75, 1.25, 0.75],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _forecast_result(decomposition, expected_columns=None):
    expected_columns = expected_columns or ["target"]
    dates = pd.date_range("2020-02-29", periods=2, freq="ME")
    forecast = pd.DataFrame(
        [
            {"date": date, "variable": column, "value": value}
            for value, date in zip([1.0, 2.0], dates, strict=True)
            for column in expected_columns
        ]
    )
    return ForecastResult(
        forecast,
        decomposition=decomposition,
        forecast_origin=pd.Timestamp("2020-01-31"),
        steps=2,
        expected_columns=expected_columns,
    )


def test_forecast_result_validates_and_preserves_single_target_decomposition():
    result = _forecast_result(_valid_forecast_result_decomposition())

    assert list(result.decomposition.columns) == [
        "forecast_horizon",
        "component",
        "contribution",
        "weight",
    ]


@pytest.mark.parametrize(
    "mutate, error_type, error_message",
    [
        (
            lambda decomposition: decomposition.drop(columns="contribution"),
            ValueError,
            "missing required columns",
        ),
        (
            lambda decomposition: decomposition.assign(forecast_horizon=[0, 0, 2, 2]),
            ValueError,
            "range 0..1",
        ),
        (
            lambda decomposition: decomposition.assign(
                contribution=[0.25, "bad", 1.25, 0.75]
            ),
            TypeError,
            "contribution values must be numeric",
        ),
        (
            lambda decomposition: pd.concat([decomposition, decomposition.iloc[[0]]]),
            ValueError,
            "at most one contribution",
        ),
        (
            lambda decomposition: decomposition.assign(
                contribution=[0.0, 0.0, 1.25, 0.75]
            ),
            ValueError,
            "must reconcile",
        ),
    ],
)
def test_forecast_result_rejects_invalid_decomposition(mutate, error_type, error_message):
    decomposition = mutate(_valid_forecast_result_decomposition())

    with pytest.raises(error_type, match=error_message):
        _forecast_result(decomposition)


def test_forecast_result_requires_target_identity_for_multi_target_decomposition():
    decomposition = _valid_forecast_result_decomposition()

    with pytest.raises(ValueError, match="Multi-target"):
        _forecast_result(
            decomposition, expected_columns=["first_target", "second_target"]
        )


def test_forecast_requires_fit():
    with pytest.raises(AttributeError, match="fit"):
        ForecastOLS().forecast()


def test_failed_initial_fit_does_not_publish_partial_state():
    model = _MutatingFitModel(fail=True)
    y = pd.DataFrame(
        {"target": [1.0, 2.0]},
        index=pd.date_range("2020-01-31", periods=2, freq="ME"),
    )

    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit(y)

    assert model._is_fitted is False
    assert model.fit_attempts == 0
    assert not hasattr(model, "marker")
    with pytest.raises(AttributeError, match="fit"):
        model.forecast()


def test_failed_refit_preserves_the_previous_fitted_state():
    model = _MutatingFitModel()
    first_y = pd.DataFrame(
        {"target": [1.0, 2.0]},
        index=pd.date_range("2020-01-31", periods=2, freq="ME"),
    )
    second_y = pd.DataFrame(
        {"target": [10.0, 20.0, 30.0]},
        index=pd.date_range("2020-03-31", periods=3, freq="ME"),
    )

    model.fit(first_y)
    previous_forecast = model.forecast()
    previous_y = model.y.copy()
    previous_fit_date = model.last_y_fit_date
    model.fail = True

    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit(second_y)

    pd.testing.assert_frame_equal(model.forecast(), previous_forecast)
    pd.testing.assert_frame_equal(model.y, previous_y)
    assert model.last_y_fit_date == previous_fit_date
    assert model.fit_attempts == 1
    assert not hasattr(model, "marker")
