"""Focused tests for the reusable ``DataTransformationPipeline``.

The pipeline packages the real-time transformation, metric filtering,
differencing and level-reconstruction logic that previously lived only as
module-level helpers in ``RealTimeModel``. Broad behavioural coverage of
that logic already exists in ``tests/test_real_time_model.py``; these tests
target the new pipeline API itself (construction, picklability and
delegation), not the full transformation matrix.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from forecast_realtime.data_transformation import DataTransformationPipeline


def _levels_frame(variable, dates, vintages, values):
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "variable": variable,
            "vintage_date": pd.to_datetime(vintages),
            "frequency": "M",
            "value": values,
            "metric": "levels",
        }
    )


def test_pipeline_rejects_non_dict_transformation():
    with pytest.raises(
        TypeError, match=r"data_transformation must be a dict\[str, str\] mapping"
    ):
        DataTransformationPipeline(["levels"])


def test_pipeline_is_picklable_and_reusable_after_restore():
    """The pipeline must hold only plain data to cross a process-pool boundary."""
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    restored = pickle.loads(pickle.dumps(pipeline))

    assert restored.data_transformation == pipeline.data_transformation

    outturns = _levels_frame(
        "gdp", ["2020-01-31", "2020-02-29"], ["2020-02-29"] * 2, [100.0, 110.0]
    )
    outturns_out, forecasts_out = restored.apply(
        outturns=outturns, forecasts=None, y_variables=["gdp"], X_variables=None
    )

    assert forecasts_out is None
    assert set(outturns_out.loc[outturns_out["variable"] == "gdp", "metric"]) == {
        "levels",
        "diff",
    }


def test_apply_computes_diff_from_levels():
    outturns = _levels_frame(
        "gdp", ["2020-01-31", "2020-02-29"], ["2020-02-29"] * 2, [100.0, 110.0]
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    outturns_out, forecasts_out = pipeline.apply(
        outturns=outturns, forecasts=None, y_variables=["gdp"], X_variables=None
    )

    diff_rows = outturns_out[outturns_out["metric"] == "diff"]
    assert forecasts_out is None
    assert len(diff_rows) == 1
    np.testing.assert_allclose(diff_rows["value"].iloc[0], 10.0)


def test_apply_missing_y_variable_mapping_raises():
    outturns = _levels_frame("gdp", ["2020-01-31"], ["2020-01-31"], [100.0])
    pipeline = DataTransformationPipeline({})

    with pytest.raises(
        ValueError, match="data_transformation must contain all y_variables"
    ):
        pipeline.apply(
            outturns=outturns, forecasts=None, y_variables=["gdp"], X_variables=None
        )


def test_apply_diff_treats_missing_month_as_undefined():
    """A missing calendar month means the following diff has no valid base,
    so no "diff" row is produced for it (matches the dropna-on-undefined
    long-form contract)."""
    outturns = _levels_frame(
        "gdp",
        ["2020-01-31", "2020-02-29", "2020-04-30"],  # March missing
        ["2020-04-30"] * 3,
        [100.0, 110.0, 130.0],
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    outturns_out, _ = pipeline.apply(
        outturns=outturns, forecasts=None, y_variables=["gdp"], X_variables=None
    )

    diff_rows = outturns_out[outturns_out["metric"] == "diff"]
    assert list(diff_rows["date"]) == [pd.Timestamp("2020-02-29")]
    np.testing.assert_allclose(diff_rows["value"].iloc[0], 10.0)


def test_apply_pop_treats_missing_quarter_as_undefined():
    outturns = _levels_frame(
        "gdp",
        ["2020-03-31", "2020-06-30", "2020-12-31"],  # Q3 missing
        ["2020-12-31"] * 3,
        [100.0, 105.0, 120.0],
    )
    outturns["frequency"] = "Q"
    pipeline = DataTransformationPipeline({"gdp": "pop"})

    outturns_out, _ = pipeline.apply(
        outturns=outturns, forecasts=None, y_variables=["gdp"], X_variables=None
    )

    pop_rows = outturns_out[outturns_out["metric"] == "pop"]
    assert list(pop_rows["date"]) == [pd.Timestamp("2020-06-30")]
    np.testing.assert_allclose(pop_rows["value"].iloc[0], 5.0)


def test_apply_differences_overlapping_outturn_and_forecast_at_same_date():
    """A forecast landing on an outturn date is treated as a single trajectory point."""
    outturns = _levels_frame(
        "gdp", ["2020-01-31", "2020-02-29"], ["2020-02-29"] * 2, [100.0, 110.0]
    )
    forecasts = _levels_frame("gdp", ["2020-02-29"], ["2020-02-29"], [111.0])

    pipeline = DataTransformationPipeline({"gdp": "diff"})
    _outturns_out, forecasts_out = pipeline.apply(
        outturns=outturns, forecasts=forecasts, y_variables=["gdp"], X_variables=None
    )

    forecast_diff = forecasts_out.loc[forecasts_out["metric"] == "diff", "value"]
    assert len(forecast_diff) == 1
    np.testing.assert_allclose(forecast_diff.iloc[0], 11.0)


def test_filter_keeps_only_required_metric_per_variable():
    data = pd.DataFrame(
        {
            "variable": ["gdp", "gdp", "cpi"],
            "metric": ["levels", "diff", "levels"],
            "value": [100.0, 10.0, 1.5],
        }
    )
    pipeline = DataTransformationPipeline({"gdp": "diff", "cpi": "levels"})

    result = pipeline.filter(data, ["gdp", "cpi"])

    assert list(result["metric"]) == ["diff", "levels"]


def test_filter_raises_for_unmapped_variable():
    data = pd.DataFrame({"variable": ["gdp"], "metric": ["levels"], "value": [1.0]})
    pipeline = DataTransformationPipeline({})

    with pytest.raises(KeyError):
        pipeline.filter(data, ["gdp"])


def test_reconstruct_levels_rebuilds_from_diff():
    outturns = _levels_frame(
        "gdp", ["2020-01-31", "2020-02-29"], ["2020-02-29"] * 2, [100.0, 110.0]
    )
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-03-31"]),
            "frequency": "M",
            "value": [5.0],
            "metric": "diff",
        }
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    result = pipeline.reconstruct_levels(
        forecasts=forecasts, outturns=outturns, y_variables=["gdp"], frequency="M"
    )

    reconstructed = result.loc[
        (result["metric"] == "levels") & (result["date"] == pd.Timestamp("2020-03-31")),
        "value",
    ]
    assert len(reconstructed) == 1
    np.testing.assert_allclose(reconstructed.iloc[0], 115.0)


def test_apply_is_deterministic_across_pipeline_instances():
    """Separate pipeline instances built from the same mapping behave identically."""
    outturns = _levels_frame(
        "gdp", ["2020-01-31", "2020-02-29"], ["2020-02-29"] * 2, [100.0, 110.0]
    )
    pipeline = DataTransformationPipeline({"gdp": "diff"})

    other_outturns, other_forecasts = DataTransformationPipeline({"gdp": "diff"}).apply(
        outturns=outturns.copy(),
        forecasts=None,
        y_variables=["gdp"],
        X_variables=None,
    )
    pipeline_outturns, pipeline_forecasts = pipeline.apply(
        outturns=outturns.copy(), forecasts=None, y_variables=["gdp"], X_variables=None
    )

    pd.testing.assert_frame_equal(
        other_outturns.reset_index(drop=True), pipeline_outturns.reset_index(drop=True)
    )
    assert other_forecasts is None and pipeline_forecasts is None


def test_filter_and_reconstruct_are_deterministic_across_pipeline_instances():
    data = pd.DataFrame(
        {"variable": ["gdp", "gdp"], "metric": ["levels", "diff"], "value": [100.0, 10.0]}
    )

    other_filtered = DataTransformationPipeline({"gdp": "diff"}).filter(data, ["gdp"])
    pipeline_filtered = DataTransformationPipeline({"gdp": "diff"}).filter(data, ["gdp"])

    pd.testing.assert_frame_equal(
        other_filtered.reset_index(drop=True), pipeline_filtered.reset_index(drop=True)
    )

    outturns = _levels_frame(
        "gdp", ["2020-01-31", "2020-02-29"], ["2020-02-29"] * 2, [100.0, 110.0]
    )
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31"]),
            "variable": "gdp",
            "vintage_date": pd.to_datetime(["2020-03-31"]),
            "frequency": "M",
            "value": [5.0],
            "metric": "diff",
        }
    )

    other_reconstructed = DataTransformationPipeline({"gdp": "diff"}).reconstruct_levels(
        forecasts=forecasts.copy(), outturns=outturns, y_variables=["gdp"], frequency="M"
    )
    pipeline_reconstructed = DataTransformationPipeline(
        {"gdp": "diff"}
    ).reconstruct_levels(
        forecasts=forecasts.copy(), outturns=outturns, y_variables=["gdp"], frequency="M"
    )

    pd.testing.assert_frame_equal(
        other_reconstructed.reset_index(drop=True),
        pipeline_reconstructed.reset_index(drop=True),
    )
