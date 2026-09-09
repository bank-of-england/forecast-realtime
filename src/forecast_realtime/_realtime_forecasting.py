"""Private containers for realtime forecast execution."""

from dataclasses import dataclass

import numpy as np

from ._model_data import ModelData


@dataclass(frozen=True)
class ForecastTask:
    """Pickleable work item submitted to a realtime forecast worker."""

    model: object
    data: ModelData
    data_transformation: object
    vintages: np.ndarray
    options: dict
    model_kwargs: dict


@dataclass(frozen=True)
class ForecastRunResult:
    """Completed worker outputs before aggregation and storage."""

    forecasts: object
    decompositions: object
    all_vintages_skipped: bool
    native_forecasts: object = None
