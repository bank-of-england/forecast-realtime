"""Pandera schemas for forecast_realtime data contracts.

Schemas define the structure and validation rules for key output tables
(decompositions, forecasts, etc.) produced by the real-time forecasting pipeline.
"""

import numpy as np
import pandera.pandas as pa
from pandera.pandas import Check, Column

REVISION_SOURCES = ["news", "reestimation", "interaction"]

# Minimal decomposition schema: raw output from ForecastModel._forecast_decomp()
# (before RealTimeModel augmentation with metadata)
minimal_decomposition_schema = pa.DataFrameSchema(
    {
        "forecast_horizon": Column(int),
        "component": Column(str),
        "contribution": Column(float, nullable=False),
        "weight": Column(float, nullable=True),
    },
    strict=False,  # Allow extra columns (will be augmented by RealTimeModel)
)

decomposition_schema = pa.DataFrameSchema(
    {
        "variable": Column(str),
        "date": Column("datetime64[ns]"),
        "forecast_horizon": Column(int),
        "frequency": Column(str, Check.isin(["Q", "M"])),
        "source": Column(str),
        "vintage_date": Column("datetime64[ns]"),
        "base_vintage_date": Column("datetime64[ns]", nullable=True),
        "decomposition": Column(str, Check.isin(["level", "revision"])),
        "component": Column(str),
        "revision_source": Column(str, Check.isin(REVISION_SOURCES), nullable=True),
        "contribution": Column(float, nullable=False),
        "weight": Column(float, nullable=True),
        "news": Column(float, nullable=True),
        "forecast_metric": Column(str),
    },
    strict=True,
    checks=[
        # decomposition flag must be consistent with base_vintage_date
        # and revision_source.
        Check(
            lambda df: (df["decomposition"] == "level") == df["base_vintage_date"].isna(),
            error="decomposition must be 'level' iff base_vintage_date is NaT",
        ),
        Check(
            lambda df: (
                (df["decomposition"] == "revision") == df["revision_source"].notna()
            ),
            error="revision_source must be set iff decomposition is 'revision'",
        ),
        # weight * news must reconstruct contribution where both are provided.
        Check(
            lambda df: (
                df[["weight", "news"]].isna().any(axis=1)
                | np.isclose(df["weight"] * df["news"], df["contribution"], atol=1e-8)
            ),
            error="contribution must equal weight * news where both are provided",
        ),
    ],
)
