import copy
import pickle

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from forecast_realtime._model_data import ModelData, ModelInputRequirements


def test_from_wide_preserves_missing_paths_layout_and_metadata():
    index = pd.period_range("2024-01", periods=2, freq="M", name="period")
    columns = pd.Index(["b", "a"], name="signal")
    y = pd.DataFrame(
        {
            "b": pd.Series([1, 2], index=index, dtype="Int64"),
            "a": pd.Series([1.5, 2.5], index=index, dtype="float64"),
        },
        index=index,
    )
    y.columns = columns
    empty = pd.DataFrame(index=index)
    empty.columns = pd.Index([], name="signal")
    missing = pd.DataFrame(
        {"missing": pd.Series([pd.NA, pd.NA], index=index, dtype="Float64")},
        index=index,
    )
    missing.columns = pd.Index(["missing"], name="signal")

    data = ModelData.from_wide(y=y, X=empty, y_conditioning=missing)

    assert not data.has_path("X", "conditioning")
    assert data.has_path("X", "history")
    assert data.to_wide("X", "history").empty
    assert_frame_equal(data.to_wide("y"), y)
    assert_frame_equal(data.to_wide("y", "conditioning"), missing)
    assert data.to_wide("X", "conditioning") is None
    assert "vintage_date" not in data.to_long()


def test_from_wide_roundtrips_a_period_index_without_inventing_dates():
    index = pd.period_range("2025Q1", periods=3, freq="Q", name="quarter")
    frame = pd.DataFrame(
        {"z": pd.Series([1.0, 2.0, 3.0], index=index, dtype="float32")},
        index=index,
    )
    frame.columns = pd.Index(["z"], name="variable")

    result = ModelData.from_wide(y=frame).to_wide()

    assert_frame_equal(result, frame)
    assert isinstance(result.index, pd.PeriodIndex)
    assert "vintage_date" not in ModelData.from_wide(y=frame).to_long()


def test_from_wide_and_projections_do_not_alias_caller_frames():
    index = pd.date_range("2024-01-01", periods=2, freq="D", name="date")
    frame = pd.DataFrame({"z": [1.0, 2.0]}, index=index)
    data = ModelData.from_wide(y=frame)

    frame.iloc[0, 0] = 99.0
    frame.columns = pd.Index(["changed"], name="other")
    projection = data.to_wide()
    projection.iloc[0, 0] = 88.0
    projection.index = pd.date_range("2030-01-01", periods=2, name="other_date")

    expected = pd.DataFrame({"z": [1.0, 2.0]}, index=index)
    assert_frame_equal(data.to_wide(), expected)


def test_from_wide_roundtrips_heterogeneous_dtypes_without_aliasing():
    index = pd.date_range("2024-01-31", periods=2, freq="ME", name="date")
    columns = pd.Index(["signed", "nullable", "decimal"], name="signal")
    frame = pd.DataFrame(
        {
            "signed": pd.Series([2**53 + 1, 2**63 - 2], index=index, dtype="int64"),
            "nullable": pd.Series([pd.NA, 7], index=index, dtype="Int64"),
            "decimal": pd.Series([1.25, 2.5], index=index, dtype="float64"),
        },
        index=index,
    )
    frame.columns = columns
    expected = frame.copy(deep=True)
    data = ModelData.from_wide(y=frame)

    frame.iloc[0, 0] = 0
    frame.columns = pd.Index(["changed"] * 3, name="other")
    result = data.to_wide()

    assert result.index.equals(expected.index)
    assert result.columns.equals(expected.columns)
    assert result.dtypes.equals(expected.dtypes)
    assert_frame_equal(result, expected, check_exact=True)

    result.iloc[1, 0] = 0
    result.index = pd.date_range("2030-01-31", periods=2, freq="ME")
    result.columns = pd.Index(["changed"] * 3)
    assert_frame_equal(data.to_wide(), expected, check_exact=True)


def test_deepcopy_and_pickle_isolate_model_storage():
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    data = ModelData.from_wide(y=pd.DataFrame({"z": [1.0, 2.0]}, index=index))

    copied = copy.deepcopy(data)
    copied._observations.loc[0, "value"] = 10.0
    restored = pickle.loads(pickle.dumps(data))
    restored._observations.loc[0, "value"] = 20.0

    assert data.to_wide().iloc[0, 0] == 1.0
    assert copied.to_wide().iloc[0, 0] == 10.0
    assert restored.to_wide().iloc[0, 0] == 20.0


def test_archive_selection_uses_independent_y_and_x_source_streams():
    dates = pd.to_datetime(["2024-01-31", "2024-01-31"])
    forecasts = pd.DataFrame(
        {
            "date": dates,
            "variable": ["z", "z"],
            "value": [10.0, 20.0],
            "vintage_date": pd.to_datetime(["2024-01-15", "2024-01-15"]),
            "source": ["y-feed", "x-feed"],
            "metric": ["levels", "levels"],
        }
    )
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"]),
            "variable": ["z"],
            "value": [1.0],
            "vintage_date": pd.to_datetime(["2024-02-01"]),
            "metric": ["levels"],
        }
    )
    requirements = (
        ModelInputRequirements(
            consumer="leaf",
            y=(("z", "levels"),),
            X=(("z", "levels"),),
        ),
    )

    selected = ModelData.from_archive(outturns, forecasts).select(
        requirements,
        y_sources={"z": "y-feed"},
        X_sources={"z": "x-feed"},
    )

    assert selected.to_wide("y", "conditioning").iloc[0, 0] == 10.0
    assert selected.to_wide("X", "conditioning").iloc[0, 0] == 20.0
    assert selected.metrics("y", "conditioning") == {"z": "levels"}
    assert selected.metrics("X", "conditioning") == {"z": "levels"}


def test_as_of_uses_latest_release_without_future_or_metric_fallback():
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-31", "2024-01-31", "2024-01-31", "2024-02-29", "2024-03-31"]
            ),
            "variable": ["z"] * 5,
            "value": [10.0, 11.0, 900.0, 20.0, 30.0],
            "vintage_date": pd.to_datetime(
                ["2024-02-01", "2024-03-01", "2024-03-01", "2024-03-01", "2024-04-01"]
            ),
            "metric": ["levels", "levels", "logs", "levels", "levels"],
        }
    )
    requirements = (ModelInputRequirements(consumer="leaf", y=(("z", "levels"),)),)
    selected = ModelData.from_archive(outturns).select(requirements)

    early = selected.as_of("2024-03-15")
    late = selected.as_of("2024-04-15")

    assert_frame_equal(
        early.to_wide(),
        pd.DataFrame(
            {"z": [11.0, 20.0]},
            index=pd.DatetimeIndex(
                pd.to_datetime(["2024-01-31", "2024-02-29"]), name="date"
            ),
        ).rename_axis(columns="variable"),
    )
    assert late.to_wide().loc[pd.Timestamp("2024-03-31"), "z"] == 30.0
    assert pd.Timestamp("2024-03-31") not in early.to_wide().index
    assert_frame_equal(
        selected.subset(["z"]).as_of("2024-03-15").to_wide(), early.to_wide()
    )


def test_published_after_stacks_incremental_history_before_combining_transforms():
    cutoff = pd.Timestamp("2020-03-31")
    fit_history = pd.DataFrame(
        {"target": [100.0, 110.0, 121.0]},
        index=pd.date_range("2020-01-31", periods=3, freq="ME"),
    )
    full_history = pd.DataFrame(
        {"target": [100.0, 110.0, 121.0, 133.1, 140.0]},
        index=pd.date_range("2020-01-31", periods=5, freq="ME"),
    )
    additional_history = full_history.iloc[[3]].copy()
    base = ModelData.from_wide(y=fit_history, frequencies={"target": "M"})

    published = base.published_after(
        ModelData.from_wide(y=full_history, frequencies={"target": "M"}),
        cutoff,
    )
    stacked = published.published_after(
        ModelData.from_wide(y=additional_history, frequencies={"target": "M"}),
        cutoff,
    )

    assert_frame_equal(stacked.to_wide("y", "published"), full_history.iloc[3:])

    expected = pd.DataFrame(
        {"target": [np.nan, 10.0, 11.0, 133.1 - 121.0, 140.0 - 133.1]},
        index=full_history.index,
    )
    transformed = stacked.transform({"target": "diff"}, combine=True).to_wide()
    assert_frame_equal(transformed, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_archive_rejects_conflicting_duplicate_values_and_metadata():
    base = {
        "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
        "variable": ["z", "z"],
        "vintage_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
        "metric": ["levels", "levels"],
    }
    values_conflict = pd.DataFrame({**base, "value": [1.0, 2.0]})
    metadata_conflict = pd.DataFrame(
        {**base, "value": [1.0, 1.0], "release_note": ["first", "second"]}
    )

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        ModelData.from_long(values_conflict, semantics="archive")
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        ModelData.from_long(metadata_conflict, semantics="archive")


def test_identical_archive_duplicates_are_collapsed():
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "variable": ["z", "z"],
            "value": [1.0, 1.0],
            "vintage_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
            "release_note": ["same", "same"],
        }
    )

    result = ModelData.from_archive(rows).to_long()

    assert_frame_equal(result, rows.iloc[[0]])


def test_trajectory_roundtrip_preserves_extra_metadata_and_duplicate_rows():
    rows = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "variable": ["z", "z"],
            "value": [1.0, 1.0],
            "vintage_date": pd.to_datetime(["2024-02-01", "2024-02-01"]),
            "source": ["feed", "feed"],
            "metric": ["levels", "levels"],
            "release_note": ["same", "same"],
        }
    )

    result = ModelData.from_long(rows, semantics="trajectory").to_long()

    assert_frame_equal(result, rows)


def test_as_of_rejects_wide_and_trajectory_inputs():
    index = pd.date_range("2024-01-01", periods=1, freq="D")
    wide = ModelData.from_wide(y=pd.DataFrame({"z": [1.0]}, index=index))
    trajectory = ModelData.from_long(
        pd.DataFrame(
            {
                "date": index,
                "variable": ["z"],
                "value": [1.0],
                "vintage_date": pd.to_datetime(["2024-02-01"]),
            }
        ),
        semantics="trajectory",
    )

    with pytest.raises(ValueError, match="revision archive"):
        wide.as_of("2024-02-01")
    with pytest.raises(ValueError, match="revision archive"):
        trajectory.as_of("2024-02-01")


def test_select_resolves_conflicting_requirements_to_a_common_levels_source():
    outturns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31"] * 3),
            "variable": ["z"] * 3,
            "value": [1.0, 2.0, 3.0],
            "vintage_date": pd.to_datetime(["2024-02-01"] * 3),
            "metric": ["levels", "logs", "diff"],
        }
    )
    requirements = (
        ModelInputRequirements(consumer="levels-leaf", y=(("z", "logs"),)),
        ModelInputRequirements(consumer="change-leaf", y=(("z", "diff"),)),
    )

    selected = ModelData.from_archive(outturns).select(requirements)

    assert requirements[0].y == (("z", "logs"),)
    assert requirements[1].y == (("z", "diff"),)
    assert selected.metrics("y") == {"z": "levels"}
    assert selected.to_wide().iloc[0, 0] == 1.0


def test_select_keeps_same_variable_y_and_x_provenance_separate():
    forecasts = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-31", "2024-01-31"]),
            "variable": ["z", "z"],
            "value": [10.0, 20.0],
            "vintage_date": pd.to_datetime(["2024-01-15", "2024-01-15"]),
            "source": ["y-feed", "x-feed"],
            "metric": ["levels", "levels"],
        }
    )
    requirements = (
        ModelInputRequirements(
            consumer="leaf",
            y=(("z", "levels"),),
            X=(("z", "levels"),),
        ),
    )
    selected = ModelData.from_archive(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-31"]),
                "variable": ["other"],
                "value": [1.0],
                "vintage_date": pd.to_datetime(["2024-02-01"]),
            }
        ),
        forecasts,
    ).select(
        requirements,
        y_sources={"z": "y-feed"},
        X_sources={"z": "x-feed"},
    )

    y_stream = selected._layouts["y", "conditioning"][0][0]
    x_stream = selected._layouts["X", "conditioning"][0][0]
    assert y_stream != x_stream
    assert selected._catalogue[y_stream]["source"] == "y-feed"
    assert selected._catalogue[x_stream]["source"] == "x-feed"
