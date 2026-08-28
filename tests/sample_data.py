import numpy as np
import pandas as pd


def create_sample_mixed_freq_outturns() -> pd.DataFrame:
    """Create sample outturns with mixed-frequency data for testing.

    Generates outturns for six variables over 10 years (2015–2024),
    with two quarterly revisions per observation:

    - **quarterly_a** (Q, 6-week publication lag)
    - **quarterly_b** (Q, 4-week publication lag)
    - **quarterly_c** (Q, 2-week publication lag)
    - **monthly_a** (M, 0-week publication lag)
    - **monthly_b** (M, 1-week publication lag)
    - **monthly_c** (M, 2-week publication lag)

    Returns
    -------
    pd.DataFrame
        Columns: date, variable, vintage_date, frequency, value.
    """
    np.random.seed(42)

    quarterly_vars = {
        "quarterly_a": 42,  # 6 weeks
        "quarterly_b": 28,  # 4 weeks
        "quarterly_c": 14,  # 2 weeks
    }
    monthly_vars = {
        "monthly_a": 0,  # 0 weeks
        "monthly_b": 7,  # 1 week
        "monthly_c": 14,  # 2 weeks
    }

    rows = []

    # Quarterly variables
    q_dates = pd.date_range("2015-03-31", "2024-12-31", freq="QE")
    for var, pub_lag_days in quarterly_vars.items():
        values = np.random.randn(len(q_dates))
        for i, target_date in enumerate(q_dates):
            first_release = target_date + pd.Timedelta(days=pub_lag_days)
            # Two vintages: first release and one quarterly revision
            revision = first_release + pd.offsets.QuarterEnd(1)
            for vintage_date in [first_release, revision]:
                rows.append(
                    {
                        "date": target_date,
                        "variable": var,
                        "vintage_date": vintage_date,
                        "frequency": "Q",
                        "value": round(values[i] + np.random.normal(0, 0.01), 4),
                    }
                )

    # Monthly variables
    m_dates = pd.date_range("2015-01-31", "2024-12-31", freq="ME")
    for var, pub_lag_days in monthly_vars.items():
        values = np.random.randn(len(m_dates))
        for i, target_date in enumerate(m_dates):
            first_release = target_date + pd.Timedelta(days=pub_lag_days)
            # Two vintages: first release and one monthly revision
            revision = first_release + pd.offsets.MonthEnd(1)
            for vintage_date in [first_release, revision]:
                rows.append(
                    {
                        "date": target_date,
                        "variable": var,
                        "vintage_date": vintage_date,
                        "frequency": "M",
                        "value": round(values[i] + np.random.normal(0, 0.01), 4),
                    }
                )

    return pd.DataFrame(rows)


def _create_lagged_outturns(
    dates: pd.DatetimeIndex,
    values: dict[str, np.ndarray],
    frequency: str,
    publication_lags: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    """Create first releases and one revision for each synthetic series."""
    offset_class = pd.offsets.MonthEnd if frequency == "M" else pd.offsets.QuarterEnd
    rng = np.random.default_rng(seed)
    rows = []

    for variable, series in values.items():
        lag = publication_lags[variable]
        release_offset = offset_class(lag)
        revision_offset = offset_class(1)
        for date, value in zip(dates, series):
            first_release = date + release_offset
            revision = first_release + revision_offset
            for vintage_date, revision_noise in (
                (first_release, 0.0),
                (revision, rng.normal(0.0, 0.01)),
            ):
                rows.append(
                    {
                        "date": date,
                        "variable": variable,
                        "vintage_date": vintage_date,
                        "frequency": frequency,
                        "value": float(value + revision_noise),
                    }
                )

    return pd.DataFrame(rows)


def create_sample_quarterly_outturns() -> pd.DataFrame:
    """Create quarterly data with three different publication lags.

    The target is published one quarter after its target period. The fast
    regressor is available in the target quarter, while the slow regressor is
    published two quarters later. Every observation has one later revision.
    """
    rng = np.random.default_rng(101)
    dates = pd.date_range("2014-03-31", periods=44, freq="QE")
    fast = rng.normal(size=len(dates))
    slow = rng.normal(size=len(dates))
    target = 1.0 + 0.7 * fast - 0.35 * slow + rng.normal(scale=0.1, size=len(dates))

    return _create_lagged_outturns(
        dates=dates,
        values={
            "quarterly_target": target,
            "quarterly_fast": fast,
            "quarterly_slow": slow,
        },
        frequency="Q",
        publication_lags={
            "quarterly_target": 1,
            "quarterly_fast": 0,
            "quarterly_slow": 2,
        },
        seed=102,
    )


def create_sample_monthly_outturns() -> pd.DataFrame:
    """Create monthly data with three different publication lags.

    The target is published one month after its target period. The fast
    regressor is available in the target month, while the slow regressor is
    published three months later. Every observation has one later revision.
    """
    rng = np.random.default_rng(201)
    dates = pd.date_range("2014-01-31", periods=132, freq="ME")
    fast = rng.normal(size=len(dates))
    slow = rng.normal(size=len(dates))
    target = 1.0 + 0.6 * fast - 0.25 * slow + rng.normal(scale=0.1, size=len(dates))

    return _create_lagged_outturns(
        dates=dates,
        values={
            "monthly_target": target,
            "monthly_fast": fast,
            "monthly_slow": slow,
        },
        frequency="M",
        publication_lags={
            "monthly_target": 1,
            "monthly_fast": 0,
            "monthly_slow": 3,
        },
        seed=202,
    )


def create_sample_mixed_frequency_outturns() -> pd.DataFrame:
    """Create quarterly target data with monthly regressors and different lags.

    The quarterly target is published one quarter late. The fast monthly
    regressor is available in its target month, while the slow monthly
    regressor is published two months late. Every observation has one later
    revision.
    """
    rng = np.random.default_rng(301)
    monthly_dates = pd.date_range("2013-01-31", periods=144, freq="ME")
    quarterly_dates = pd.date_range("2014-03-31", periods=44, freq="QE")
    monthly_fast = rng.normal(size=len(monthly_dates))
    monthly_slow = rng.normal(size=len(monthly_dates))

    quarterly_signal = []
    for date in quarterly_dates:
        available = monthly_dates <= date
        quarterly_signal.append(
            0.7 * monthly_fast[available][-3:].mean()
            + 0.3 * monthly_slow[available][-3:].mean()
        )
    quarterly_target = np.asarray(quarterly_signal) + rng.normal(
        scale=0.1, size=len(quarterly_dates)
    )

    quarterly = _create_lagged_outturns(
        dates=quarterly_dates,
        values={"quarterly_target": quarterly_target},
        frequency="Q",
        publication_lags={"quarterly_target": 1},
        seed=302,
    )
    monthly = _create_lagged_outturns(
        dates=monthly_dates,
        values={
            "monthly_fast": monthly_fast,
            "monthly_slow": monthly_slow,
        },
        frequency="M",
        publication_lags={"monthly_fast": 0, "monthly_slow": 2},
        seed=303,
    )
    return pd.concat([quarterly, monthly], ignore_index=True)
