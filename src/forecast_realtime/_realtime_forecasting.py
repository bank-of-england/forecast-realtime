"""Private containers for realtime forecast execution."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForecastTask:
    """Pickleable work item submitted to a realtime forecast worker."""

    model: object
    data_transformation: object
    vintages: np.ndarray
    common: dict
    input_metrics: dict[str, str] = None
    y_input_metrics: dict[str, str] = None
    X_input_metrics: dict[str, str] = None
    y_conditioning_input_metrics: dict[str, str] = None
    X_conditioning_input_metrics: dict[str, str] = None


@dataclass(frozen=True)
class ForecastRunResult:
    """Completed worker outputs before aggregation and storage."""

    forecasts: object
    decompositions: object
    all_vintages_skipped: bool
    native_forecasts: object = None
