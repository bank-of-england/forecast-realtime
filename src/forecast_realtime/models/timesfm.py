"""Local forecasting with Google's TimesFM 3.0 model."""

from functools import lru_cache

import numpy as np
import pandas as pd

from forecast_realtime.forecast_model import ForecastModel


@lru_cache(maxsize=4)
def _load_forecaster(model_id, device):
    try:
        from timesfm3 import TimesFM3Forecaster
    except ModuleNotFoundError as error:
        if error.name != "timesfm3":
            raise
        raise ModuleNotFoundError(
            'Install TimesFM 3.0 with pip install "forecast-realtime[timesfm]".',
            name="timesfm3",
        ) from error
    return TimesFM3Forecaster.from_pretrained(model_id, device=device)


class ForecastTimesFM(ForecastModel):
    """Forecast locally with pretrained TimesFM 3.0 weights.

    Fit stores the context and loads the pretrained model without fine-tuning.
    Point forecasts are medians. Requested quantiles are linearly interpolated
    within the checkpoint's output grid; extrapolation is not supported.
    The backend interpolates missing observations and owns its
    multi-horizon inference; ``forecast_strategy`` is not supported.
    Other constructor options are passed to ``ForecastModel``.

    TimesFM 3.0 weights have a separate non-commercial, non-production licence.
    Commercial or production use of downloaded weights is not permitted.
    The wrapper does not support the different TimesFM 2.5 API or checkpoints.

    Parameters
    ----------
    model_id : str
        Hugging Face checkpoint or local TimesFM 3.0 checkpoint directory.
        Defaults to ``google/timesfm-3.0-pytorch``, approximately 330 million
        parameters with nine quantiles from 0.1 to 0.9.
    device : str
        PyTorch device. Defaults to ``cpu``.
    """

    _supports_quantiles = True
    _handles_mixed_frequencies = False

    def __init__(
        self, model_id: str = "google/timesfm-3.0-pytorch", device: str = "cpu", **kwargs
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.device = device

    def _fit(self, y, X=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unexpected fit options: {sorted(kwargs)}")
        _load_forecaster(self.model_id, self.device)
        return self

    def _forecast(
        self, steps=1, X=None, y=None, forecast_origin=None, quantiles=None, **kwargs
    ):
        if kwargs:
            raise TypeError(f"Unexpected forecast options: {sorted(kwargs)}")
        backend = _load_forecaster(self.model_id, self.device)
        if quantiles is not None:
            levels = np.asarray(backend.config.quantiles, dtype=float)
            for probability in quantiles:
                if probability < levels[0] - 1e-8 or probability > levels[-1] + 1e-8:
                    raise ValueError(f"Backend does not provide quantile {probability}.")
        history = self.y if y is None else y
        history = history.loc[:forecast_origin].reindex(columns=self.y.columns)
        if isinstance(history.index, pd.PeriodIndex):
            history = history.copy()
            history.index = history.index.to_timestamp(how="end").normalize()
        dates = self._forecast_dates(forecast_origin, steps)
        design = self._fitted_model_configuration.design
        frequency = design.frequency
        if frequency in ("M", "Q"):
            frequency += "S" if design.month_start else "E"
        calendar = pd.date_range(history.index[0], forecast_origin, freq=frequency)
        history = history.reindex(calendar)
        context = history.to_numpy(dtype=float, copy=True).T
        past_covariates = None
        covariates = None
        if self.X is not None:
            if X is None:
                X = self.X
            if isinstance(X.index, pd.PeriodIndex):
                X = X.copy()
                X.index = X.index.to_timestamp(how="end").normalize()
            regressors = X.reindex(index=calendar.append(dates), columns=self.X.columns)
            known = regressors.loc[dates].notna().any()
            if (~known).any():
                past_covariates = (
                    regressors.loc[calendar, ~known].to_numpy(dtype=float, copy=True).T
                )
            if known.any():
                covariates = regressors.loc[:, known].to_numpy(dtype=float, copy=True).T
        prediction = backend.predict(
            context,
            horizon=steps,
            past_only_covariates=past_covariates,
            past_future_covariates=covariates,
            return_quantiles=quantiles is not None,
        )
        values = prediction.forecast if quantiles is None else prediction.quantiles
        values = np.asarray(values)
        if len(self.y.columns) == 1 and values.ndim == (1 if quantiles is None else 2):
            values = values[None, ...]
        expected = (len(self.y.columns), steps)
        if quantiles is not None:
            expected += (len(levels),)
        if values.shape != expected:
            raise ValueError(
                f"Backend forecasts have shape {values.shape}; expected {expected}."
            )
        if quantiles is None:
            return values.T
        requested = np.clip(quantiles, levels[0], levels[-1])
        values = np.apply_along_axis(
            lambda row: np.interp(requested, levels, row), -1, values
        )
        values = values.transpose(1, 0, 2)
        return pd.DataFrame(
            {
                "date": dates.repeat(len(self.y.columns) * len(quantiles)),
                "variable": np.tile(np.repeat(self.y.columns, len(quantiles)), steps),
                "quantile": np.tile(quantiles, steps * len(self.y.columns)),
                "value": values.reshape(-1),
            }
        )
