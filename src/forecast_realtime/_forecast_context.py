"""Private implementation of the public forecast context contract."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastContext:
    """Raw fitted history and future paths for one prediction request."""

    y_history: pd.DataFrame
    X_history: pd.DataFrame | None
    y_conditioning: pd.DataFrame | None = None
    """Raw explicit target constraints, never merged with published observations."""
    X_conditioning: pd.DataFrame | None = None
    forecast_origin: pd.Timestamp | None = None
    y_conditioning_input_metrics: dict[str, str] | None = None
    X_conditioning_input_metrics: dict[str, str] | None = None
    y_published: pd.DataFrame | None = None
    """Published target observations after the fitted history."""
    y_published_input_metrics: dict[str, str] | None = None
    """Input units of published targets, independent of explicit constraint units."""

    @classmethod
    def _from_data(cls, data, forecast_origin):
        """Materialise raw frames for a tree's context-based forecast hook."""
        return cls(
            data.to_wide("y"),
            data.to_wide("X"),
            data.to_wide("y", "conditioning"),
            data.to_wide("X", "conditioning"),
            forecast_origin,
            data.metrics("y", "conditioning"),
            data.metrics("X", "conditioning"),
            data.to_wide("y", "published"),
            data.metrics("y", "published"),
        )
