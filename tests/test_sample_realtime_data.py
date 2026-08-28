import numpy as np
import pandas as pd
import pytest

import forecast_realtime
from forecast_realtime.sample_realtime_data import (
    FIRST_PERIOD,
    SNAPSHOT_END,
    _aggregate_quarterly,
    _build_covariance,
    _create_monthly_snapshots,
    _generate_synthetic_mixed_frequency_data,
    _get_vintage,
)


def test_stage_one_has_valid_long_structure_and_default_ranges():
    data = _generate_synthetic_mixed_frequency_data(N=2, seed=12)

    assert list(data.columns) == ["date", "frequency", "variable", "value"]
    assert data["date"].is_monotonic_increasing
    assert set(data["frequency"]) == {"M", "Q"}
    assert set(data.loc[data["frequency"] == "M", "variable"]) == {
        "monthly_1",
        "monthly_2",
    }
    assert set(data.loc[data["frequency"] == "Q", "variable"]) == {
        "quarterly_1",
        "quarterly_2",
    }
    assert data["date"].min() == pd.Timestamp(FIRST_PERIOD)
    assert data.loc[data["frequency"] == "M", "date"].max() == pd.Timestamp("2025-12-31")
    assert data.loc[data["frequency"] == "Q", "date"].min() == pd.Timestamp("1980-06-30")
    assert data.loc[data["frequency"] == "Q", "date"].max() == pd.Timestamp("2025-12-31")
    assert data.loc[data["frequency"] == "M"].groupby("variable").size().eq(552).all()
    assert data.loc[data["frequency"] == "Q"].groupby("variable").size().eq(183).all()


def test_covariance_modes_encode_dense_and_sparse_intent_for_any_supported_n():
    for n in (1, 2, 10, 31):
        dense = _build_covariance(n, "dense")
        sparse = _build_covariance(n, "sparse")

        assert np.allclose(np.diag(dense), 1)
        assert np.all(np.linalg.eigvalsh(dense) >= -1e-12)
        dense_off_diagonal = dense[~np.eye(2 * n, dtype=bool)]
        assert np.all(dense_off_diagonal != 0)

        assert np.allclose(np.diag(sparse), 1)
        assert np.all(np.linalg.eigvalsh(sparse) >= -1e-12)
        sparse_off_diagonal = sparse[~np.eye(2 * n, dtype=bool)]
        assert np.count_nonzero(sparse_off_diagonal) < len(sparse_off_diagonal) / 2


def test_stage_one_modes_and_seeds_are_distinct_and_reproducible():
    dense = _generate_synthetic_mixed_frequency_data(N=3, mode="dense", seed=7)
    dense_again = _generate_synthetic_mixed_frequency_data(N=3, mode="dense", seed=7)
    sparse = _generate_synthetic_mixed_frequency_data(N=3, mode="sparse", seed=7)

    pd.testing.assert_frame_equal(dense, dense_again)
    assert not np.array_equal(dense["value"].to_numpy(), sparse["value"].to_numpy())


def test_quarterly_aggregation_uses_chronological_mariano_murasawa_weights():
    dates = pd.date_range("1980-01-31", periods=8, freq="ME")
    values = pd.Series(np.arange(1, 9, dtype=float), index=dates)

    aggregated = _aggregate_quarterly(values)

    assert aggregated.index.tolist() == [pd.Timestamp("1980-06-30")]
    np.testing.assert_allclose(
        aggregated.iloc[0], np.dot([1, 2, 3, 2, 1], [2, 3, 4, 5, 6]) / 3
    )


def test_snapshots_compute_period_on_period_growth_for_monthly_and_quarterly():
    snapshots = _create_monthly_snapshots(
        _generate_synthetic_mixed_frequency_data(N=1, seed=12),
        publication_lags=False,
    )
    latest = snapshots[snapshots["vintage_date"] == SNAPSHOT_END]

    for frequency in ("M", "Q"):
        levels = latest[
            (latest["frequency"] == frequency) & (latest["metric"] == "levels")
        ].set_index("date")["value"]
        pop = latest[
            (latest["frequency"] == frequency) & (latest["metric"] == "pop")
        ].set_index("date")["value"]
        expected = levels.pct_change(fill_method=None).dropna() * 100.0

        pd.testing.assert_index_equal(pop.index, expected.index)
        np.testing.assert_allclose(pop.to_numpy(), expected.to_numpy())


def test_stage_one_validation():
    with pytest.raises(ValueError, match="N must be a positive integer"):
        _generate_synthetic_mixed_frequency_data(N=0)
    with pytest.raises(ValueError, match="mode must be"):
        _generate_synthetic_mixed_frequency_data(mode="elastic")
    with pytest.raises(ValueError, match="endpoint must be"):
        _generate_synthetic_mixed_frequency_data(endpoint="1979-12-31")


def test_snapshots_have_expected_structure_ranges_and_historical_inclusion():
    stage_one = _generate_synthetic_mixed_frequency_data(N=2, seed=4)
    snapshots = _create_monthly_snapshots(stage_one, publication_lags=False)

    assert list(snapshots.columns) == [
        "date",
        "frequency",
        "variable",
        "value",
        "vintage_date",
        "metric",
    ]
    assert set(snapshots["metric"]) == {"levels", "pop"}
    assert snapshots["vintage_date"].min() == pd.Timestamp("2024-01-31")
    assert snapshots["vintage_date"].max() == pd.Timestamp(SNAPSHOT_END)
    assert snapshots["date"].min() == pd.Timestamp(FIRST_PERIOD)
    assert snapshots["vintage_date"].is_monotonic_increasing

    first_snapshot = snapshots[snapshots["vintage_date"] == "2024-01-31"]
    assert first_snapshot["date"].min() == pd.Timestamp(FIRST_PERIOD)
    assert set(first_snapshot["variable"]) == {
        "monthly_1",
        "monthly_2",
        "quarterly_1",
        "quarterly_2",
    }
    assert snapshots[snapshots["vintage_date"] == SNAPSHOT_END][
        "date"
    ].max() == pd.Timestamp("2025-12-31")


def test_snapshots_keep_fixed_window_for_non_default_stage_one_endpoint():
    stage_one = _generate_synthetic_mixed_frequency_data(
        N=2, seed=4, endpoint="2025-06-30"
    )
    snapshots = _create_monthly_snapshots(stage_one, publication_lags=False)

    expected_vintages = pd.date_range("2024-01-31", SNAPSHOT_END, freq="ME")
    assert snapshots["vintage_date"].drop_duplicates().tolist() == list(expected_vintages)

    source_in_window = stage_one[stage_one["date"] >= "2024-01-01"]
    first_release = snapshots.groupby(["date", "frequency", "variable"], as_index=False)[
        "vintage_date"
    ].min()
    first_release = first_release.merge(
        source_in_window[["date", "frequency", "variable"]],
        on=["date", "frequency", "variable"],
        how="right",
    )
    assert (
        first_release["vintage_date"].dt.to_period("M")
        == first_release["date"].dt.to_period("M")
    ).all()

    target = pd.Timestamp("2025-06-30")
    target_source = stage_one[stage_one["date"] == target]
    target_release = snapshots[
        (snapshots["vintage_date"] == target) & (snapshots["date"] == target)
    ]
    assert set(target_release["variable"]) == set(target_source["variable"])


def test_snapshots_apply_independent_calendar_month_lags_and_no_revisions():
    stage_one = _generate_synthetic_mixed_frequency_data(N=2, seed=4)
    snapshots = _create_monthly_snapshots(stage_one, publication_lags=True, seed=8)

    observed = snapshots[
        (snapshots["date"] >= "2024-01-01") & (snapshots["metric"] == "levels")
    ]
    first_release = observed.groupby(["variable", "date"], as_index=False).first()
    first_release["lag"] = (
        first_release["vintage_date"].dt.to_period("M")
        - first_release["date"].dt.to_period("M")
    ).apply(lambda period: period.n)
    assert first_release["lag"].map(lambda lag: isinstance(lag, (int, np.integer))).all()
    assert first_release["lag"].between(0, 6).all()
    assert first_release.groupby("variable")["lag"].nunique().eq(1).all()
    assert first_release.groupby("variable")["lag"].first().nunique() > 1

    levels = snapshots[snapshots["metric"] == "levels"]
    assert (
        levels.groupby(["vintage_date", "variable", "date"])["value"].nunique().max() == 1
    )
    assert (
        snapshots[snapshots["vintage_date"] == SNAPSHOT_END]["date"]
        .ge(pd.Timestamp(FIRST_PERIOD))
        .all()
    )
    source_values = stage_one.set_index(["variable", "date"])["value"]
    snapshot_values = levels.set_index(["variable", "date"])["value"]
    np.testing.assert_allclose(
        snapshot_values.to_numpy(),
        source_values.reindex(snapshot_values.index).to_numpy(),
    )


def test_snapshot_validation_and_get_vintage():
    stage_one = _generate_synthetic_mixed_frequency_data(N=1, seed=22)
    snapshots = _create_monthly_snapshots(stage_one, publication_lags=False)

    vintage = _get_vintage(snapshots, "2024-01-31")
    expected = stage_one[stage_one["date"] <= "2024-01-31"]
    actual = (
        vintage[vintage["metric"] == "levels"]
        .set_index(["variable", "date"])["value"]
        .sort_index()
    )
    expected_values = expected.set_index(["variable", "date"])["value"].sort_index()
    pd.testing.assert_series_equal(actual, expected_values, check_names=False)
    assert vintage["vintage_date"].eq(pd.Timestamp("2024-01-31")).all()

    with pytest.raises(ValueError, match="publication_lags must be a bool"):
        _create_monthly_snapshots(stage_one, publication_lags=1)
    with pytest.raises(ValueError, match="required columns"):
        _create_monthly_snapshots(stage_one.drop(columns="value"))


def test_get_vintage_preserves_same_variable_and_date_at_different_frequencies():
    snapshots = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-31"),
                "frequency": "Q",
                "variable": "shared",
                "value": 40,
                "vintage_date": pd.Timestamp("2024-02-29"),
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "frequency": "M",
                "variable": "shared",
                "value": 10,
                "vintage_date": pd.Timestamp("2024-01-31"),
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "frequency": "M",
                "variable": "shared",
                "value": 20,
                "vintage_date": pd.Timestamp("2024-02-29"),
            },
            {
                "date": pd.Timestamp("2024-01-31"),
                "frequency": "Q",
                "variable": "shared",
                "value": 30,
                "vintage_date": pd.Timestamp("2024-01-31"),
            },
        ]
    ).sample(frac=1, random_state=7)

    vintage = _get_vintage(snapshots, "2024-02-29")

    assert vintage[["frequency", "value"]].to_dict("records") == [
        {"frequency": "M", "value": 20},
        {"frequency": "Q", "value": 40},
    ]


def test_snapshots_reject_observations_released_after_fixed_window():
    stage_one = _generate_synthetic_mixed_frequency_data(N=1, endpoint="2026-07-31")

    with pytest.raises(ValueError, match="after the final snapshot"):
        _create_monthly_snapshots(stage_one, publication_lags=False)


def test_public_api_exposes_only_the_composed_synthetic_data_function():
    snapshots = forecast_realtime.generate_synthetic_data(
        N=1, publication_lags=False, first_period="2023-01-31", endpoint="2024-12-31"
    )

    assert "vintage_date" in snapshots
    assert set(snapshots["metric"]) == {"levels", "pop"}
    assert "generate_synthetic_data" in forecast_realtime.__all__
    old_names = {
        "generate_synthetic_mixed_frequency_data",
        "generate_mixed_frequency_data",
        "create_mixed_frequency_data",
        "create_monthly_snapshots",
        "create_realtime_snapshots",
        "generate_realtime_snapshots",
        "create_realtime_mixed_freq_data",
        "get_vintage",
    }
    assert old_names.isdisjoint(forecast_realtime.__all__)
    assert all(not hasattr(forecast_realtime, name) for name in old_names)
