"""Focused tests for the raw wide-input API on ``DataTransformationPipeline``.

``RealTimeModel`` will eventually select a raw vintage and pass raw wide
``y``/``X`` frames (levels, one column per variable) straight into model
``fit()``/``forecast()``. These tests cover the pipeline methods that
prepare those raw frames: ``transform_fit_inputs()`` for a single training
snapshot and ``transform_forecast_inputs()`` for historical data plus
appended conditioning/future rows treated as one trajectory.

The existing long-form ``apply``/``filter``/``reconstruct_levels`` contract
(covered in ``tests/test_data_transformation.py``) is unchanged; these tests
target only the new wide-input methods.
"""

import numpy as np
import pandas as pd
import pytest

from forecast_realtime.data_transformation import DataTransformationPipeline


def _dates(*strings):
    return pd.DatetimeIndex(pd.to_datetime(strings), name="date")


# ---------------------------------------------------------------------------
# transform_fit_inputs
# ---------------------------------------------------------------------------


def test_transform_fit_inputs_levels_identity():
    """A "levels" mapping returns the same values, in a new frame."""
    y = pd.DataFrame(
        {"gdp": [100.0, 110.0, 121.0]},
        index=_dates("2020-01-31", "2020-02-29", "2020-03-31"),
    )
    pipeline = DataTransformationPipeline({"gdp": "levels"})

    y_out, X_out = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    assert X_out is None
    np.testing.assert_allclose(y_out["gdp"].to_numpy(), y["gdp"].to_numpy())
    assert y_out is not y


def test_transform_fit_inputs_identity_for_already_requested_metric():
    y = pd.DataFrame(
        {"gdp": [10.0, 11.0, 15.0]},
        index=_dates("2020-01-31", "2020-02-29", "2020-03-31"),
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
        y_input_metrics={"gdp": "diff"},
    )

    pd.testing.assert_frame_equal(y_out, y)


def test_transform_fit_inputs_identity_applies_to_y_and_X_source_metrics():
    dates = _dates("2020-01-31", "2020-02-29", "2020-03-31")
    y = pd.DataFrame({"gdp": [10.0, 11.0, 15.0]}, index=dates)
    X = pd.DataFrame({"unemp": [4.0, 4.2, 4.1]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff", "unemp": "logs"})

    y_out, X_out = pipeline.transform_fit_inputs(
        y,
        X,
        y_variables=["gdp"],
        X_variables=["unemp"],
        frequency="M",
        frequencies={"gdp": "M", "unemp": "M"},
        y_input_metrics={"gdp": "diff"},
        X_input_metrics={"unemp": "logs"},
    )

    pd.testing.assert_frame_equal(y_out, y)
    pd.testing.assert_frame_equal(X_out, X)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("logs", np.log([100.0, 110.0, 125.0])),
        ("diff", [np.nan, 10.0, 15.0]),
        ("log diff", [np.nan, np.log(1.1), np.log(125.0 / 110.0)]),
        ("pop", [np.nan, 10.0, (125.0 / 110.0 - 1) * 100.0]),
        ("yoy", [np.nan, np.nan, np.nan]),
    ],
)
def test_transform_fit_inputs_converts_levels_to_requested_metric(metric, expected):
    values = [100.0, 110.0, 125.0]
    y = pd.DataFrame(
        {"gdp": values},
        index=_dates("2020-01-31", "2020-02-29", "2020-03-31"),
    )
    pipeline = DataTransformationPipeline({"gdp": metric})

    y_out, _ = pipeline.transform_fit_inputs(
        y,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
        y_input_metrics={"gdp": "levels"},
    )

    np.testing.assert_allclose(y_out["gdp"].to_numpy(), expected, equal_nan=True)


@pytest.mark.parametrize("requested_metric", ["levels", "logs", "log diff", "pop", "yoy"])
def test_transform_fit_inputs_rejects_unsupported_derived_source(requested_metric):
    y = pd.DataFrame(
        {"gdp": [10.0, 11.0, 15.0]},
        index=_dates("2020-01-31", "2020-02-29", "2020-03-31"),
    )
    pipeline = DataTransformationPipeline({"gdp": requested_metric})

    with pytest.raises(
        ValueError, match="Cannot transform variable 'gdp' from metric 'diff'"
    ):
        pipeline.transform_fit_inputs(
            y,
            y_variables=["gdp"],
            frequency="M",
            frequencies={"gdp": "M"},
            y_input_metrics={"gdp": "diff"},
        )


def test_transform_fit_inputs_levels_derived_conversion_preserves_calendar_gap():
    dates = _dates("2020-01-31", "2020-02-29", "2020-04-30")
    y = pd.DataFrame({"gdp": [100.0, 110.0, 130.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
        y_input_metrics={"gdp": "levels"},
    )

    np.testing.assert_allclose(y_out["gdp"].iloc[1], 10.0)
    assert np.isnan(y_out["gdp"].iloc[2])


def test_transform_fit_inputs_diff():
    y = pd.DataFrame(
        {"gdp": [100.0, 110.0, 125.0]},
        index=_dates("2020-01-31", "2020-02-29", "2020-03-31"),
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].iloc[1:].to_numpy(), [10.0, 15.0])


def test_transform_fit_inputs_log_diff():
    values = [100.0, 110.0, 125.0]
    y = pd.DataFrame(
        {"gdp": values}, index=_dates("2020-01-31", "2020-02-29", "2020-03-31")
    )
    pipeline = DataTransformationPipeline({"gdp": "log diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    expected = np.diff(np.log(values))
    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].iloc[1:].to_numpy(), expected)


def test_transform_fit_inputs_logs():
    values = [100.0, 110.0, 125.0]
    y = pd.DataFrame(
        {"gdp": values}, index=_dates("2020-01-31", "2020-02-29", "2020-03-31")
    )
    pipeline = DataTransformationPipeline({"gdp": "logs"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    np.testing.assert_allclose(y_out["gdp"].to_numpy(), np.log(values))


def test_transform_fit_inputs_pop():
    values = [100.0, 110.0, 121.0]
    y = pd.DataFrame(
        {"gdp": values}, index=_dates("2020-01-31", "2020-02-29", "2020-03-31")
    )
    pipeline = DataTransformationPipeline({"gdp": "pop"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    expected = pd.Series(values).pct_change().to_numpy() * 100
    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].to_numpy()[1:], expected[1:])


def test_transform_fit_inputs_yoy_uses_quarterly_periods():
    values = [100.0, 102.0, 104.0, 106.0, 110.0]
    y = pd.DataFrame(
        {"gdp": values},
        index=_dates(
            "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31"
        ),
    )
    pipeline = DataTransformationPipeline({"gdp": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="Q", frequencies={"gdp": "Q"}
    )

    assert y_out["gdp"].iloc[:4].isna().all()
    np.testing.assert_allclose(y_out["gdp"].iloc[4], (110.0 / 100.0 - 1) * 100)


def test_transform_fit_inputs_yoy_uses_monthly_periods():
    values = list(np.linspace(100, 111, 13))
    y = pd.DataFrame(
        {"cpi": values}, index=pd.date_range("2019-01-31", periods=13, freq="ME")
    )
    pipeline = DataTransformationPipeline({"cpi": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["cpi"], frequency="M", frequencies={"cpi": "M"}
    )

    assert y_out["cpi"].iloc[:12].isna().all()
    np.testing.assert_allclose(y_out["cpi"].iloc[12], (values[12] / values[0] - 1) * 100)


def test_transform_fit_inputs_yoy_accepts_per_variable_frequencies():
    dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    quarterly_dates = pd.date_range("2019-03-31", periods=5, freq="QE")
    y = pd.DataFrame(
        {
            "monthly": list(np.linspace(100, 114, 15)),
            "quarterly": pd.Series(
                [100.0, 102.0, 104.0, 106.0, 110.0], index=quarterly_dates
            ).reindex(dates),
        },
        index=dates,
    )
    pipeline = DataTransformationPipeline({"monthly": "yoy", "quarterly": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y,
        y_variables=["monthly", "quarterly"],
        frequency="M",
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    assert y_out["monthly"].iloc[:12].isna().all()
    np.testing.assert_allclose(y_out["monthly"].iloc[12], (112 / 100 - 1) * 100)
    assert y_out["quarterly"].dropna().empty is False
    np.testing.assert_allclose(
        y_out["quarterly"].dropna().iloc[-1], (110 / 100 - 100 / 100) * 100
    )


def test_transform_fit_inputs_uses_per_variable_frequencies():
    monthly_dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    quarterly_dates = pd.date_range("2019-03-31", periods=5, freq="QE")
    index = monthly_dates.union(quarterly_dates)
    y = pd.DataFrame(
        {
            "monthly": pd.Series(
                np.arange(100.0, 100.0 + len(monthly_dates)), index=monthly_dates
            ).reindex(index),
            "quarterly": pd.Series(
                [100.0, 102.0, 104.0, 106.0, 110.0], index=quarterly_dates
            ).reindex(index),
        },
        index=index,
    )
    pipeline = DataTransformationPipeline({"monthly": "yoy", "quarterly": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y,
        y_variables=["monthly", "quarterly"],
        frequency="M",
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    np.testing.assert_allclose(
        y_out.loc[pd.Timestamp("2020-03-31"), "monthly"],
        (114.0 / 102.0 - 1) * 100,
    )
    np.testing.assert_allclose(
        y_out.loc[pd.Timestamp("2020-03-31"), "quarterly"],
        (110.0 / 100.0 - 1) * 100,
    )


def test_transform_fit_inputs_uses_per_variable_X_frequencies():
    y = pd.DataFrame(
        {"target": [1.0] * 5},
        index=pd.date_range("2019-03-31", periods=5, freq="QE"),
    )
    monthly_dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    quarterly_dates = pd.date_range("2019-03-31", periods=5, freq="QE")
    x_index = monthly_dates.union(quarterly_dates)
    X = pd.DataFrame(
        {
            "monthly": pd.Series(
                np.arange(100.0, 100.0 + len(monthly_dates)), index=monthly_dates
            ).reindex(x_index),
            "quarterly": pd.Series(
                [100.0, 102.0, 104.0, 106.0, 110.0], index=quarterly_dates
            ).reindex(x_index),
        },
        index=x_index,
    )
    pipeline = DataTransformationPipeline(
        {"target": "levels", "monthly": "yoy", "quarterly": "yoy"}
    )

    _, X_out = pipeline.transform_fit_inputs(
        y,
        X,
        y_variables=["target"],
        X_variables=["monthly", "quarterly"],
        frequency="Q",
        frequencies={"target": "Q", "monthly": "M", "quarterly": "Q"},
    )

    np.testing.assert_allclose(
        X_out.loc[pd.Timestamp("2020-03-31"), "monthly"],
        (114.0 / 102.0 - 1) * 100,
    )
    np.testing.assert_allclose(
        X_out.loc[pd.Timestamp("2020-03-31"), "quarterly"],
        (110.0 / 100.0 - 1) * 100,
    )


def test_transform_forecast_inputs_uses_mixed_column_frequencies():
    history_index = pd.date_range("2018-01-31", periods=27, freq="ME")
    future_index = pd.DatetimeIndex(["2020-06-30"])
    y_history = pd.DataFrame(
        {
            "monthly": list(np.linspace(100, 126, 27)),
            "quarterly": pd.Series(
                [100.0] * 9,
                index=pd.date_range("2018-03-31", periods=9, freq="QE"),
            ).reindex(history_index),
        },
        index=history_index,
    )
    y_conditioning = pd.DataFrame(
        {"monthly": [127.0], "quarterly": [110.0]}, index=future_index
    )
    pipeline = DataTransformationPipeline({"monthly": "yoy", "quarterly": "yoy"})

    _, conditioning_out, _, _ = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        y_variables=["monthly", "quarterly"],
        frequency="M",
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    np.testing.assert_allclose(conditioning_out["monthly"].iloc[0], (127 / 117 - 1) * 100)
    np.testing.assert_allclose(
        conditioning_out["quarterly"].iloc[0], (110 / 100 - 1) * 100
    )


def test_transform_forecast_inputs_uses_explicit_column_frequencies():
    monthly_history_dates = pd.date_range("2019-01-31", periods=15, freq="ME")
    quarterly_history_dates = pd.date_range("2019-03-31", periods=5, freq="QE")
    history_index = monthly_history_dates.union(quarterly_history_dates)
    y_history = pd.DataFrame(
        {
            "monthly": pd.Series(
                np.arange(100.0, 100.0 + len(monthly_history_dates)),
                index=monthly_history_dates,
            ).reindex(history_index),
            "quarterly": pd.Series(
                [100.0, 102.0, 104.0, 106.0, 108.0], index=quarterly_history_dates
            ).reindex(history_index),
        },
        index=history_index,
    )
    conditioning = pd.DataFrame(
        {
            "monthly": [115.0, np.nan],
            "quarterly": [np.nan, 110.0],
        },
        index=_dates("2020-04-30", "2020-06-30"),
    )
    pipeline = DataTransformationPipeline({"monthly": "yoy", "quarterly": "yoy"})

    _, conditioning_out, _, _ = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=conditioning,
        y_variables=["monthly", "quarterly"],
        frequency="M",
        frequencies={"monthly": "M", "quarterly": "Q"},
    )

    np.testing.assert_allclose(
        conditioning_out.loc[pd.Timestamp("2020-04-30"), "monthly"],
        (115.0 / 103.0 - 1) * 100,
    )
    np.testing.assert_allclose(
        conditioning_out.loc[pd.Timestamp("2020-06-30"), "quarterly"],
        (110.0 / 102.0 - 1) * 100,
    )


def test_transform_forecast_inputs_uses_explicit_X_frequencies():
    history_index = pd.date_range("2019-01-31", periods=15, freq="ME")
    y_history = pd.DataFrame({"target": [1.0] * 15}, index=history_index)
    quarterly_history_dates = pd.date_range("2019-03-31", periods=5, freq="QE")
    x_history_index = history_index.union(quarterly_history_dates)
    X_history = pd.DataFrame(
        {
            "monthly": pd.Series(np.arange(100.0, 115.0), index=history_index).reindex(
                x_history_index
            ),
            "quarterly": pd.Series(
                [100.0, 102.0, 104.0, 106.0, 108.0], index=quarterly_history_dates
            ).reindex(x_history_index),
        },
        index=x_history_index,
    )
    X_future = pd.DataFrame(
        {"monthly": [115.0], "quarterly": [110.0]},
        index=_dates("2020-04-30"),
    )
    pipeline = DataTransformationPipeline(
        {"target": "levels", "monthly": "yoy", "quarterly": "yoy"}
    )

    _, _, _, future_out = pipeline.transform_forecast_inputs(
        y_history,
        X_history=X_history,
        X_future=X_future,
        y_variables=["target"],
        X_variables=["monthly", "quarterly"],
        frequency="Q",
        frequencies={"target": "Q", "monthly": "M", "quarterly": "Q"},
    )

    np.testing.assert_allclose(
        future_out.loc[pd.Timestamp("2020-04-30"), "monthly"],
        (115.0 / 103.0 - 1) * 100,
    )
    assert np.isnan(future_out.loc[pd.Timestamp("2020-04-30"), "quarterly"])


def test_transform_fit_inputs_diff_treats_missing_month_as_undefined():
    """A missing calendar month means the following diff has no valid base."""
    dates = _dates("2020-01-31", "2020-02-29", "2020-04-30")  # March missing
    y = pd.DataFrame({"gdp": [100.0, 110.0, 130.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].iloc[1], 10.0)
    assert np.isnan(y_out["gdp"].iloc[2])


def test_transform_fit_inputs_diff_treats_missing_quarter_as_undefined():
    dates = _dates("2020-03-31", "2020-06-30", "2020-12-31")  # Q3 missing
    y = pd.DataFrame({"gdp": [100.0, 105.0, 120.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="Q", frequencies={"gdp": "Q"}
    )

    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].iloc[1], 5.0)
    assert np.isnan(y_out["gdp"].iloc[2])


def test_transform_fit_inputs_month_start_diff_preserves_anchor_and_gap():
    dates = _dates("2020-01-01", "2020-02-01", "2020-04-01")  # March missing
    y = pd.DataFrame({"gdp": [100.0, 110.0, 130.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequencies={"gdp": "M"}
    )

    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].iloc[1], 10.0)
    assert np.isnan(y_out["gdp"].iloc[2])


def test_transform_fit_inputs_month_start_yoy_preserves_anchor():
    values = list(np.linspace(100.0, 112.0, 13))
    y = pd.DataFrame(
        {"cpi": values}, index=pd.date_range("2019-01-01", periods=13, freq="MS")
    )
    pipeline = DataTransformationPipeline({"cpi": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["cpi"], frequencies={"cpi": "M"}
    )

    assert y_out["cpi"].iloc[:12].isna().all()
    np.testing.assert_allclose(y_out["cpi"].iloc[12], (112.0 / 100.0 - 1) * 100)


def test_transform_fit_inputs_month_start_yoy_gap_is_undefined():
    dates = _dates("2019-02-01", "2019-12-01", "2020-01-01")  # January 2019 missing
    y = pd.DataFrame({"cpi": [100.0, 101.0, 113.0]}, index=dates)
    pipeline = DataTransformationPipeline({"cpi": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["cpi"], frequencies={"cpi": "M"}
    )

    assert y_out["cpi"].isna().all()


def test_transform_fit_inputs_quarter_start_diff_preserves_anchor_and_gap():
    dates = _dates("2020-01-01", "2020-04-01", "2020-10-01")  # Q3 missing
    y = pd.DataFrame({"gdp": [100.0, 105.0, 120.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequencies={"gdp": "Q"}
    )

    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].iloc[1], 5.0)
    assert np.isnan(y_out["gdp"].iloc[2])


def test_transform_fit_inputs_quarter_start_yoy_preserves_anchor():
    values = [100.0, 102.0, 104.0, 106.0, 110.0]
    y = pd.DataFrame(
        {"gdp": values},
        index=pd.date_range("2019-01-01", periods=5, freq="QS"),
    )
    pipeline = DataTransformationPipeline({"gdp": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequencies={"gdp": "Q"}
    )

    assert y_out["gdp"].iloc[:4].isna().all()
    np.testing.assert_allclose(y_out["gdp"].iloc[4], (110.0 / 100.0 - 1) * 100)


def test_transform_fit_inputs_quarter_start_yoy_gap_is_undefined():
    dates = _dates("2019-04-01", "2019-10-01", "2020-01-01")  # January 2019 missing
    y = pd.DataFrame({"gdp": [100.0, 105.0, 110.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequencies={"gdp": "Q"}
    )

    assert y_out["gdp"].isna().all()


def test_transform_fit_inputs_pop_treats_missing_month_as_undefined():
    dates = _dates("2020-01-31", "2020-02-29", "2020-04-30")  # March missing
    y = pd.DataFrame({"gdp": [100.0, 110.0, 130.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "pop"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    assert np.isnan(y_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_out["gdp"].iloc[1], 10.0)
    assert np.isnan(y_out["gdp"].iloc[2])


def test_transform_fit_inputs_yoy_missing_month_uses_calendar_not_position():
    """A month missing between a target and its true year-ago date must not
    let the year-on-year growth silently shift onto the wrong observation."""
    dates = pd.date_range("2019-01-31", periods=24, freq="ME")
    values = [100.0 + i for i in range(24)]
    y = pd.DataFrame({"gdp": values}, index=dates)
    y = y.drop(index=dates[17])  # drop June 2020, between Dec 2019 and Dec 2020
    pipeline = DataTransformationPipeline({"gdp": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    dec_2020 = y_out["gdp"].loc[pd.Timestamp("2020-12-31")]
    correct = (123.0 / 111.0 - 1) * 100.0  # Dec-2020 (123) vs Dec-2019 (111)
    np.testing.assert_allclose(dec_2020, correct)


def test_transform_fit_inputs_yoy_missing_quarter_uses_calendar_not_position():
    dates = pd.date_range("2019-03-31", periods=8, freq="QE")
    values = [100.0 + i for i in range(8)]
    y = pd.DataFrame({"gdp": values}, index=dates)
    y = y.drop(index=dates[5])  # drop 2020-Q2, between 2019-Q4 and 2020-Q4
    pipeline = DataTransformationPipeline({"gdp": "yoy"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="Q", frequencies={"gdp": "Q"}
    )

    q4_2020 = y_out["gdp"].loc[pd.Timestamp("2020-12-31")]
    correct = (107.0 / 103.0 - 1) * 100.0  # 2020-Q4 (107) vs 2019-Q4 (103)
    np.testing.assert_allclose(q4_2020, correct)


def test_transform_fit_inputs_transforms_y_and_x_independently():
    dates = _dates("2020-01-31", "2020-02-29", "2020-03-31")
    y = pd.DataFrame({"gdp": [100.0, 110.0, 125.0]}, index=dates)
    X = pd.DataFrame({"unemp": [4.0, 4.2, 4.1]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff", "unemp": "levels"})

    y_out, X_out = pipeline.transform_fit_inputs(
        y,
        X,
        y_variables=["gdp"],
        X_variables=["unemp"],
        frequency="M",
        frequencies={"gdp": "M", "unemp": "M"},
    )

    np.testing.assert_allclose(y_out["gdp"].iloc[1:].to_numpy(), [10.0, 15.0])
    np.testing.assert_allclose(X_out["unemp"].to_numpy(), X["unemp"].to_numpy())


def test_transform_fit_inputs_does_not_mutate_caller_frames():
    dates = _dates("2020-01-31", "2020-02-29", "2020-03-31")
    y = pd.DataFrame({"gdp": [100.0, 110.0, 125.0]}, index=dates)
    X = pd.DataFrame({"unemp": [4.0, 4.2, 4.1]}, index=dates)
    y_before = y.copy()
    X_before = X.copy()
    pipeline = DataTransformationPipeline({"gdp": "diff", "unemp": "diff"})

    pipeline.transform_fit_inputs(
        y,
        X,
        y_variables=["gdp"],
        X_variables=["unemp"],
        frequency="M",
        frequencies={"gdp": "M", "unemp": "M"},
    )

    pd.testing.assert_frame_equal(y, y_before)
    pd.testing.assert_frame_equal(X, X_before)


def test_transform_fit_inputs_is_deterministic_across_calls():
    """The pipeline holds no mutable state; repeat calls on fresh raw copies agree."""
    dates = _dates("2020-01-31", "2020-02-29", "2020-03-31")
    y = pd.DataFrame({"gdp": [100.0, 110.0, 125.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    first, _ = pipeline.transform_fit_inputs(
        y.copy(), y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )
    second, _ = pipeline.transform_fit_inputs(
        y.copy(), y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    pd.testing.assert_frame_equal(first, second)


def test_transform_fit_inputs_missing_y_mapping_raises():
    y = pd.DataFrame({"gdp": [100.0]}, index=_dates("2020-01-31"))
    pipeline = DataTransformationPipeline({})

    with pytest.raises(
        ValueError, match="data_transformation must contain all y_variables"
    ):
        pipeline.transform_fit_inputs(y, y_variables=["gdp"], frequency="M")


def test_transform_fit_inputs_missing_column_raises():
    y = pd.DataFrame({"other": [100.0]}, index=_dates("2020-01-31"))
    pipeline = DataTransformationPipeline({"gdp": "levels"})

    with pytest.raises(ValueError, match="missing columns"):
        pipeline.transform_fit_inputs(y, y_variables=["gdp"], frequency="M")


def test_transform_fit_inputs_non_datetime_index_raises():
    y = pd.DataFrame({"gdp": [100.0, 110.0]})
    pipeline = DataTransformationPipeline({"gdp": "levels"})

    with pytest.raises(ValueError, match="DatetimeIndex"):
        pipeline.transform_fit_inputs(y, y_variables=["gdp"], frequency="M")


def test_transform_fit_inputs_duplicate_index_raises():
    y = pd.DataFrame({"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-01-31"))
    pipeline = DataTransformationPipeline({"gdp": "levels"})

    with pytest.raises(ValueError, match="duplicate"):
        pipeline.transform_fit_inputs(y, y_variables=["gdp"], frequency="M")


def test_transform_fit_inputs_X_variables_without_X_raises():
    y = pd.DataFrame({"gdp": [100.0]}, index=_dates("2020-01-31"))
    pipeline = DataTransformationPipeline({"gdp": "levels", "unemp": "levels"})

    with pytest.raises(ValueError, match="X_variables"):
        pipeline.transform_fit_inputs(
            y, y_variables=["gdp"], X_variables=["unemp"], frequency="M"
        )


def test_transform_fit_inputs_X_without_X_variables_raises():
    dates = _dates("2020-01-31")
    y = pd.DataFrame({"gdp": [100.0]}, index=dates)
    X = pd.DataFrame({"unemp": [4.0]}, index=dates)
    pipeline = DataTransformationPipeline({"gdp": "levels", "unemp": "levels"})

    with pytest.raises(ValueError, match="X"):
        pipeline.transform_fit_inputs(y, X, y_variables=["gdp"], frequency="M")


def test_transform_fit_inputs_invalid_frequency_raises():
    y = pd.DataFrame({"gdp": [100.0]}, index=_dates("2020-01-31"))
    pipeline = DataTransformationPipeline({"gdp": "levels"})

    with pytest.raises(ValueError, match="frequency"):
        pipeline.transform_fit_inputs(y, y_variables=["gdp"], frequency="W")


# ---------------------------------------------------------------------------
# transform_forecast_inputs
# ---------------------------------------------------------------------------


def test_transform_forecast_inputs_diff_uses_preceding_history_observation():
    """The first conditioning-row diff uses the last raw history observation."""
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    y_conditioning = pd.DataFrame({"gdp": [130.0]}, index=_dates("2020-03-31"))
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_hist_out, y_cond_out, _, _ = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
    )

    assert np.isnan(y_hist_out["gdp"].iloc[0])
    np.testing.assert_allclose(y_hist_out["gdp"].iloc[1], 10.0)
    np.testing.assert_allclose(y_cond_out["gdp"].iloc[0], 20.0)


def test_transform_forecast_inputs_combines_levels_history_with_native_future():
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    y_conditioning = pd.DataFrame({"gdp": [20.0]}, index=_dates("2020-03-31"))
    pipeline = DataTransformationPipeline({"gdp": "pop"})

    y_hist_out, y_cond_out, _, _ = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        y_variables=["gdp"],
        frequencies={"gdp": "M"},
        y_input_metrics={"gdp": "levels"},
        y_conditioning_input_metrics={"gdp": "pop"},
    )

    np.testing.assert_allclose(y_hist_out["gdp"].to_numpy(), [np.nan, 10.0])
    np.testing.assert_allclose(y_cond_out["gdp"].to_numpy(), [20.0])


def test_transform_forecast_inputs_diff_treats_missing_boundary_month_as_undefined():
    """A calendar gap spanning the history/conditioning boundary must still
    leave the first conditioning diff undefined."""
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    y_conditioning = pd.DataFrame(
        {"gdp": [130.0]}, index=_dates("2020-04-30")
    )  # March missing
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    _, y_cond_out, _, _ = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
    )

    assert np.isnan(y_cond_out["gdp"].iloc[0])


def test_transform_forecast_inputs_overlap_precedence_prefers_conditioning():
    """A conditioning row sharing a date with history wins for differencing."""
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    y_conditioning = pd.DataFrame({"gdp": [111.0]}, index=_dates("2020-02-29"))
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    _, y_cond_out, _, _ = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
    )

    np.testing.assert_allclose(y_cond_out["gdp"].iloc[0], 11.0)


def test_transform_forecast_inputs_overlays_subset_columns_cell_by_cell():
    history_dates = _dates("2020-01-31", "2020-02-29")
    future_date = _dates("2020-02-29")
    y_history = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=history_dates)
    X_history = pd.DataFrame({"x": [5.0, 6.0], "z": [7.0, 8.0]}, index=history_dates)
    y_conditioning = pd.DataFrame({"a": [20.0]}, index=future_date)
    X_future = pd.DataFrame({"x": [60.0]}, index=future_date)
    pipeline = DataTransformationPipeline(
        {"a": "levels", "b": "levels", "x": "levels", "z": "levels"}
    )

    y_history_out, y_conditioning_out, X_history_out, X_future_out = (
        pipeline.transform_forecast_inputs(
            y_history,
            y_conditioning=y_conditioning,
            X_history=X_history,
            X_future=X_future,
            y_variables=["a", "b"],
            X_variables=["x", "z"],
            frequency="M",
            frequencies={"a": "M", "b": "M", "x": "M", "z": "M"},
        )
    )

    assert y_history_out.loc[future_date[0], "a"] == 20.0
    assert y_history_out.loc[future_date[0], "b"] == 4.0
    assert y_conditioning_out.loc[future_date[0], "a"] == 20.0
    assert X_history_out.loc[future_date[0], "x"] == 60.0
    assert X_history_out.loc[future_date[0], "z"] == 8.0
    assert X_future_out.loc[future_date[0], "x"] == 60.0


def test_transform_forecast_inputs_preserves_history_for_overlapping_future_NaN():
    dates = _dates("2020-01-31", "2020-02-29")
    y_history = pd.DataFrame({"gdp": [100.0, 110.0]}, index=dates)
    y_conditioning = pd.DataFrame({"gdp": [np.nan]}, index=dates[-1:])
    X_history = pd.DataFrame({"driver": [1.0, 2.0]}, index=dates)
    X_future = pd.DataFrame({"driver": [np.nan]}, index=dates[-1:])
    pipeline = DataTransformationPipeline({"gdp": "levels", "driver": "levels"})

    y_history_out, y_conditioning_out, X_history_out, X_future_out = (
        pipeline.transform_forecast_inputs(
            y_history,
            y_conditioning=y_conditioning,
            X_history=X_history,
            X_future=X_future,
            y_variables=["gdp"],
            X_variables=["driver"],
            frequency="M",
            frequencies={"gdp": "M", "driver": "M"},
        )
    )

    assert y_history_out.loc[dates[-1], "gdp"] == 110.0
    assert y_conditioning_out.loc[dates[-1], "gdp"] == 110.0
    assert X_history_out.loc[dates[-1], "driver"] == 2.0
    assert X_future_out.loc[dates[-1], "driver"] == 2.0


def test_transform_forecast_inputs_keeps_future_only_NaN():
    history_dates = _dates("2020-01-31", "2020-02-29")
    future_date = _dates("2020-03-31")
    y_history = pd.DataFrame({"gdp": [100.0, 110.0]}, index=history_dates)
    y_conditioning = pd.DataFrame({"gdp": [np.nan]}, index=future_date)
    X_history = pd.DataFrame({"driver": [1.0, 2.0]}, index=history_dates)
    X_future = pd.DataFrame({"driver": [np.nan]}, index=future_date)
    pipeline = DataTransformationPipeline({"gdp": "levels", "driver": "levels"})

    _, y_conditioning_out, _, X_future_out = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        X_history=X_history,
        X_future=X_future,
        y_variables=["gdp"],
        X_variables=["driver"],
        frequency="M",
        frequencies={"gdp": "M", "driver": "M"},
    )

    assert pd.isna(y_conditioning_out.loc[future_date[0], "gdp"])
    assert pd.isna(X_future_out.loc[future_date[0], "driver"])


def test_transform_forecast_inputs_x_future_alongside_y_conditioning():
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    y_conditioning = pd.DataFrame({"gdp": [130.0]}, index=_dates("2020-03-31"))
    X_history = pd.DataFrame(
        {"unemp": [4.0, 4.2]}, index=_dates("2020-01-31", "2020-02-29")
    )
    X_future = pd.DataFrame({"unemp": [4.3]}, index=_dates("2020-03-31"))
    pipeline = DataTransformationPipeline({"gdp": "diff", "unemp": "levels"})

    y_hist_out, y_cond_out, X_hist_out, X_fut_out = pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        X_history=X_history,
        X_future=X_future,
        y_variables=["gdp"],
        X_variables=["unemp"],
        frequency="M",
        frequencies={"gdp": "M", "unemp": "M"},
    )

    np.testing.assert_allclose(y_cond_out["gdp"].iloc[0], 20.0)
    np.testing.assert_allclose(X_fut_out["unemp"].iloc[0], 4.3)
    np.testing.assert_allclose(
        X_hist_out["unemp"].to_numpy(), X_history["unemp"].to_numpy()
    )


def test_transform_forecast_inputs_without_conditioning_returns_none():
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_hist_out, y_cond_out, X_hist_out, X_fut_out = pipeline.transform_forecast_inputs(
        y_history,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
    )

    assert y_cond_out is None
    assert X_hist_out is None
    assert X_fut_out is None
    np.testing.assert_allclose(y_hist_out["gdp"].iloc[1], 10.0)


def test_transform_forecast_inputs_does_not_mutate_caller_frames():
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    y_conditioning = pd.DataFrame({"gdp": [130.0]}, index=_dates("2020-03-31"))
    y_history_before = y_history.copy()
    y_conditioning_before = y_conditioning.copy()
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    pipeline.transform_forecast_inputs(
        y_history,
        y_conditioning=y_conditioning,
        y_variables=["gdp"],
        frequency="M",
        frequencies={"gdp": "M"},
    )

    pd.testing.assert_frame_equal(y_history, y_history_before)
    pd.testing.assert_frame_equal(y_conditioning, y_conditioning_before)


def test_transform_forecast_inputs_conditioning_unknown_column_raises():
    y_history = pd.DataFrame(
        {"gdp": [100.0, 110.0]}, index=_dates("2020-01-31", "2020-02-29")
    )
    y_conditioning = pd.DataFrame({"unemp": [4.0]}, index=_dates("2020-03-31"))
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    with pytest.raises(ValueError, match="columns"):
        pipeline.transform_forecast_inputs(
            y_history,
            y_conditioning=y_conditioning,
            y_variables=["gdp"],
            frequency="M",
            frequencies={"gdp": "M"},
        )


# ---------------------------------------------------------------------------
# Consolidation: one direct wide arithmetic core, no long-form intermediate
# ---------------------------------------------------------------------------


def test_wide_methods_do_not_depend_on_long_form_wide_conversion_helpers():
    """The legacy wide<->long conversion helpers are gone entirely, so the
    wide-input methods cannot route transformed values through them."""
    import forecast_realtime.data_transformation as data_transformation

    assert not hasattr(data_transformation, "_wide_to_long")
    assert not hasattr(data_transformation, "_long_to_wide")


def test_transform_fit_inputs_diff_uses_shared_arithmetic_core(monkeypatch):
    """The direct wide path and the legacy long-form ``difference_by_vintage``
    share a single ``_difference_series`` implementation for "diff"."""
    import forecast_realtime.data_transformation as data_transformation

    calls = []
    original = data_transformation._difference_series

    def spy(values, logarithmic=False):
        calls.append(logarithmic)
        return original(values, logarithmic=logarithmic)

    monkeypatch.setattr(data_transformation, "_difference_series", spy)

    y = pd.DataFrame(
        {"gdp": [100.0, 110.0, 125.0]},
        index=_dates("2020-01-31", "2020-02-29", "2020-03-31"),
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )

    assert calls == [False]
    np.testing.assert_allclose(y_out["gdp"].iloc[1:].to_numpy(), [10.0, 15.0])


def test_transform_fit_inputs_yoy_uses_shared_growth_core(monkeypatch):
    """The direct wide path and the legacy long-form ``growth_by_vintage``
    share a single ``_growth_series`` implementation for "yoy"/"pop"."""
    import forecast_realtime.data_transformation as data_transformation

    calls = []
    original = data_transformation._growth_series

    def spy(values, periods):
        calls.append(periods)
        return original(values, periods=periods)

    monkeypatch.setattr(data_transformation, "_growth_series", spy)

    values = [100.0, 102.0, 104.0, 106.0, 110.0]
    y = pd.DataFrame(
        {"gdp": values},
        index=_dates(
            "2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31"
        ),
    )
    pipeline = DataTransformationPipeline({"gdp": "yoy"})

    pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="Q", frequencies={"gdp": "Q"}
    )

    assert calls == [4]


def test_logs_uses_shared_arithmetic_core(monkeypatch):
    """The long-form ``apply()`` path and the direct wide path share a single
    ``_logs_series`` implementation for "logs"."""
    import forecast_realtime.data_transformation as data_transformation

    calls = []
    original = data_transformation._logs_series

    def spy(values):
        calls.append(True)
        return original(values)

    monkeypatch.setattr(data_transformation, "_logs_series", spy)

    values = [100.0, 110.0, 125.0]
    pipeline = DataTransformationPipeline({"gdp": "logs"})

    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime("2020-03-31"),
            "frequency": "M",
            "value": values,
            "metric": "levels",
        }
    )
    outturns_out, _ = pipeline.apply(
        outturns=outturns, forecasts=None, y_variables=["gdp"], X_variables=None
    )
    logs_rows = outturns_out[outturns_out["metric"] == "logs"]
    np.testing.assert_allclose(logs_rows["value"].to_numpy(), np.log(values))

    y = pd.DataFrame(
        {"gdp": values}, index=_dates("2020-01-31", "2020-02-29", "2020-03-31")
    )
    y_out, _ = pipeline.transform_fit_inputs(
        y, y_variables=["gdp"], frequency="M", frequencies={"gdp": "M"}
    )
    np.testing.assert_allclose(y_out["gdp"].to_numpy(), np.log(values))

    assert calls == [True, True]
