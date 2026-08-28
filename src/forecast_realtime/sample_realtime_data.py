"""Synthetic mixed-frequency data and monthly real-time snapshots.

The data is created in three stages:

1. Sample all monthly latent series jointly from a multivariate normal
    distribution. This includes both the variables that will remain monthly and
    the variables that will later be observed quarterly.
2. Aggregate the quarterly latent series from their monthly paths using the
    Mariano-Murasawa five-month weights, retaining observations at quarter ends.
3. Build monthly vintages by applying each series' publication lag. Once an
    observation is released, it is carried forward unchanged into later
    vintages, so the resulting panel represents a real-time information set.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data_transformation import apply_transformations

SEED = 20260101
N = 10
FIRST_PERIOD = "1980-01-31"
ENDPOINT = "2025-12-31"
YEAR = 2024
MAX_PUBLICATION_LAG = 6
SNAPSHOT_START = "2024-01-31"
SNAPSHOT_END = "2026-06-30"

_MONTHLY_COLUMNS = ["date", "frequency", "variable", "value"]
_SNAPSHOT_COLUMNS = [*(_MONTHLY_COLUMNS), "vintage_date", "metric"]
_MM_WEIGHTS = np.array([1 / 3, 2 / 3, 1, 2 / 3, 1 / 3])


def _validate_stage_one_inputs(n: int, mode: str, first_period, endpoint) -> None:
    if type(n) is not int or n <= 0:
        raise ValueError("N must be a positive integer")
    if mode not in {"dense", "sparse"}:
        raise ValueError("mode must be either 'dense' or 'sparse'")
    if pd.Timestamp(endpoint).to_period("M") < pd.Timestamp(first_period).to_period("M"):
        raise ValueError("endpoint must be on or after first_period")


def _build_covariance(n: int, mode: str) -> np.ndarray:
    """Build a unit-diagonal covariance for the requested correlation mode."""
    dimension = 2 * n
    if mode == "dense":
        covariance = np.full((dimension, dimension), 0.05, dtype=float)
        np.fill_diagonal(covariance, 1.0)
        return covariance

    covariance = np.eye(dimension, dtype=float)
    if dimension == 2:
        return covariance
    for first in range(0, dimension - 1, 2):
        covariance[first, first + 1] = 0.35
        covariance[first + 1, first] = 0.35
    return covariance


def _month_ends(first_period, endpoint) -> pd.DatetimeIndex:
    start = pd.Timestamp(first_period).to_period("M").to_timestamp(how="end")
    end = pd.Timestamp(endpoint).to_period("M").to_timestamp(how="end")
    return pd.date_range(start.normalize(), end.normalize(), freq="ME")


def _aggregate_quarterly(series: pd.Series) -> pd.Series:
    """Aggregate a monthly path using chronological Mariano-Murasawa weights."""
    series = series.sort_index()
    rolling = series.rolling(window=5, min_periods=5).apply(
        lambda values: float(np.dot(_MM_WEIGHTS, values)), raw=True
    )
    quarter_ends = rolling.index[rolling.index.month.isin((3, 6, 9, 12))]
    return rolling.loc[quarter_ends].dropna()


def _long_frame(
    values: pd.DataFrame, frequency: str, variables: Iterable[str]
) -> pd.DataFrame:
    frame = (
        values.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="variable", value_name="value")
    )
    frame.insert(1, "frequency", frequency)
    frame["variable"] = pd.Categorical(
        frame["variable"], categories=list(variables), ordered=True
    )
    return frame


def _generate_synthetic_mixed_frequency_data(
    N: int = N,
    mode: str = "dense",
    seed: int = SEED,
    first_period=FIRST_PERIOD,
    endpoint=ENDPOINT,
) -> pd.DataFrame:
    """Sample monthly latent paths and derive quarterly paths.

    The one joint draw contains ``N`` monthly and ``N`` quarterly latent
    series. Quarterly observations are complete five-month weighted windows,
    reported at quarter ends. The returned table is in long format.
    """
    _validate_stage_one_inputs(N, mode, first_period, endpoint)
    monthly_dates = _month_ends(first_period, endpoint)
    monthly_variables = [f"monthly_{index}" for index in range(1, N + 1)]
    quarterly_variables = [f"quarterly_{index}" for index in range(1, N + 1)]

    rng = np.random.default_rng(seed)
    latent = rng.multivariate_normal(
        mean=np.zeros(2 * N),
        cov=_build_covariance(N, mode),
        size=len(monthly_dates),
        check_valid="raise",
    )
    monthly_levels = (
        pd.DataFrame(
            latent[:, :N], index=monthly_dates, columns=monthly_variables
        ).cumsum()
        + 100.0
    )
    quarterly_sample = pd.DataFrame(
        latent[:, N:], index=monthly_dates, columns=quarterly_variables
    )
    quarterly = pd.concat(
        {
            variable: _aggregate_quarterly(quarterly_sample[variable].cumsum() + 100.0)
            for variable in quarterly_variables
        },
        axis=1,
    )

    result = pd.concat(
        [
            _long_frame(monthly_levels, "M", monthly_variables),
            _long_frame(quarterly, "Q", quarterly_variables),
        ],
        ignore_index=True,
    )
    result["variable"] = result["variable"].astype(str)
    return result.sort_values(["date", "frequency", "variable"]).reset_index(drop=True)[
        _MONTHLY_COLUMNS
    ]


def _validate_stage_two_inputs(data: pd.DataFrame, publication_lags: bool) -> None:
    missing = set(_MONTHLY_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"required columns are missing: {sorted(missing)}")
    if type(publication_lags) is not bool:
        raise ValueError("publication_lags must be a bool")
    if data.duplicated(["date", "frequency", "variable"]).any():
        raise ValueError("data must contain one value per date, frequency and variable")


def _create_monthly_snapshots(
    data: pd.DataFrame,
    publication_lags: bool = True,
    seed: int = SEED,
    year: int = YEAR,
) -> pd.DataFrame:
    """Create complete monthly vintages of a Stage 1 data set.

    Each series receives one independent lag in the inclusive range 0--6
    months when ``publication_lags`` is true. Values are never revised: an
    observation is repeated in every later snapshot after its first release.
    """
    _validate_stage_two_inputs(data, publication_lags)
    if type(year) is not int or year < 1:
        raise ValueError("year must be a positive integer")
    if year != YEAR:
        raise ValueError("year must be 2024 for the fixed snapshot range")

    source = data[_MONTHLY_COLUMNS].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["_period"] = source["date"].dt.to_period("M")
    variables = sorted(source["variable"].unique())
    if publication_lags:
        rng = np.random.default_rng(seed)
        lags = dict(
            zip(variables, rng.integers(0, MAX_PUBLICATION_LAG + 1, len(variables)))
        )
    else:
        lags = dict.fromkeys(variables, 0)
    source["_release_period"] = source["_period"] + source["variable"].map(lags)

    snapshot_start = pd.Period(SNAPSHOT_START, freq="M")
    snapshot_end = pd.Period(SNAPSHOT_END, freq="M")
    if source["_release_period"].min() > snapshot_start:
        raise ValueError("data has no observations available by the first snapshot")
    if source["_release_period"].max() > snapshot_end:
        raise ValueError("data contains observations released after the final snapshot")
    snapshots = pd.period_range(snapshot_start, snapshot_end, freq="M")
    rows = []
    for vintage_period in snapshots:
        available = source[source["_release_period"] <= vintage_period].copy()
        if available.empty:
            continue
        available["vintage_date"] = vintage_period.to_timestamp(how="end").normalize()
        available["metric"] = "levels"
        variables = available["variable"].drop_duplicates().tolist()
        available = apply_transformations(
            available,
            variables,
            {variable: "pop" for variable in variables},
        )
        rows.append(available[_SNAPSHOT_COLUMNS])

    if not rows:
        return pd.DataFrame(columns=_SNAPSHOT_COLUMNS)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["vintage_date", "date", "frequency", "variable"])
        .reset_index(drop=True)
    )


def generate_synthetic_data(
    N: int = N,
    mode: str = "dense",
    seed: int = SEED,
    first_period=FIRST_PERIOD,
    endpoint=ENDPOINT,
    publication_lags: bool = True,
    year: int = YEAR,
) -> pd.DataFrame:
    """Create the composed synthetic mixed-frequency real-time data set."""
    return _create_realtime_mixed_freq_data(
        seed=seed,
        N=N,
        mode=mode,
        first_period=first_period,
        endpoint=endpoint,
        publication_lags=publication_lags,
        year=year,
    )


def _create_realtime_mixed_freq_data(
    seed: int = SEED,
    N: int = N,
    mode: str = "dense",
    first_period=FIRST_PERIOD,
    endpoint=ENDPOINT,
    publication_lags: bool = True,
    year: int = YEAR,
) -> pd.DataFrame:
    """Create the composed synthetic mixed-frequency real-time data set."""
    stage_one = _generate_synthetic_mixed_frequency_data(
        N=N,
        mode=mode,
        seed=seed,
        first_period=first_period,
        endpoint=endpoint,
    )
    return _create_monthly_snapshots(
        stage_one, publication_lags=publication_lags, seed=seed, year=year
    )


def _get_vintage(data: pd.DataFrame, vintage_date: str | pd.Timestamp) -> pd.DataFrame:
    """Return the latest value available for each observation at a vintage."""
    as_of = pd.Timestamp(vintage_date)
    available = data.loc[data["vintage_date"] <= as_of]
    if available.empty:
        return data.iloc[0:0].copy()
    identity_columns = ["variable", "frequency", "date"]
    if "metric" in available.columns:
        identity_columns.append("metric")
    sort_columns = ["vintage_date", "variable", "frequency", "date"]
    if "metric" in available.columns:
        sort_columns.append("metric")
    return (
        available.sort_values(sort_columns, kind="stable")
        .drop_duplicates(identity_columns, keep="last")
        .sort_values(sort_columns[1:], kind="stable")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    panel = generate_synthetic_data()
    print(panel.head())
    print(f"\nrows: {len(panel):,}")
    print(
        f"vintages: {panel['vintage_date'].min():%Y-%m-%d} -> "
        f"{panel['vintage_date'].max():%Y-%m-%d}"
    )
