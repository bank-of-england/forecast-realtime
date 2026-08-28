import numpy as np
import pandas as pd
import pytest

from forecast_realtime.forecast_model import ForecastModel, ForecastResult
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

    assert forecast.index[0] == pd.Timestamp("2020-03-31")


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

    assert forecast.index[0] == index[-1]


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
    forecast = pd.DataFrame(
        {column: [1.0, 2.0] for column in (expected_columns or ["target"])},
        index=pd.date_range("2020-02-29", periods=2, freq="ME"),
    )
    return ForecastResult(
        forecast,
        decomposition=decomposition,
        forecast_origin=pd.Timestamp("2020-01-31"),
        steps=2,
        expected_columns=expected_columns or ["target"],
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
