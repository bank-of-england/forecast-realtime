"""Fast factories for synthetic real-time test inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_realtime.sample_realtime_data import (
    ENDPOINT,
    FIRST_PERIOD,
    MAX_PUBLICATION_LAG,
    SEED,
    SNAPSHOT_END,
    SNAPSHOT_START,
    _generate_synthetic_mixed_frequency_data,
)
from forecast_realtime.sample_realtime_data import (
    N as DEFAULT_N,
)

_OUTPUT_COLUMNS = [
    "date",
    "frequency",
    "variable",
    "value",
    "vintage_date",
    "metric",
]


def generate_synthetic_data(
    N: int = DEFAULT_N,
    seed: int = SEED,
    first_period=FIRST_PERIOD,
    endpoint=ENDPOINT,
    publication_lags: bool = True,
) -> pd.DataFrame:
    """Build model-test inputs without transforming every vintage separately."""
    source = _generate_synthetic_mixed_frequency_data(
        N=N,
        seed=seed,
        first_period=first_period,
        endpoint=endpoint,
    )
    variables = sorted(source["variable"].unique())
    if publication_lags:
        rng = np.random.default_rng(seed)
        lags = dict(
            zip(variables, rng.integers(0, MAX_PUBLICATION_LAG + 1, len(variables)))
        )
    else:
        lags = dict.fromkeys(variables, 0)
    source["_release_period"] = source["date"].dt.to_period("M") + source["variable"].map(
        lags
    )

    vintage_periods = pd.period_range(SNAPSHOT_START, SNAPSHOT_END, freq="M")

    levels = source.assign(metric="levels")
    pop = source.sort_values(["frequency", "variable", "date"], kind="stable").copy()
    pop["value"] = (
        pop.groupby(["frequency", "variable"], sort=False)["value"]
        .pct_change(fill_method=None)
        .mul(100.0)
    )
    pop = pop.dropna(subset=["value"]).assign(metric="pop")
    transformed = pd.concat([levels, pop], ignore_index=True)

    vintages = pd.DataFrame(
        {
            "_vintage_period": vintage_periods,
            "vintage_date": vintage_periods.to_timestamp(how="end").normalize(),
        }
    )
    result = transformed.merge(vintages, how="cross")
    result = result.loc[result["_release_period"] <= result["_vintage_period"]]
    return (
        result.sort_values(
            ["vintage_date", "date", "frequency", "variable"], kind="stable"
        )
        .reset_index(drop=True)
        .loc[:, _OUTPUT_COLUMNS]
    )
