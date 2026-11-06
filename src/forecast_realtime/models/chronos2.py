"""Local forecasting with Amazon's Chronos-2 models."""

from functools import lru_cache

import numpy as np
import pandas as pd

from forecast_realtime.forecast_model import ForecastModel


@lru_cache(maxsize=4)
def _load_forecaster(model_id, device):
    try:
        from chronos import Chronos2Pipeline
    except ModuleNotFoundError as error:
        if error.name != "chronos":
            raise
        raise ModuleNotFoundError(
            'Install Chronos-2 with pip install "forecast-realtime[chronos2]".',
            name="chronos",
        ) from error
    return Chronos2Pipeline.from_pretrained(model_id, device_map=device)


class ForecastChronos2(ForecastModel):
    """Forecast locally with pretrained Chronos-2 weights.

    Fit stores the context and loads the pretrained model without fine-tuning.
    Point forecasts are medians. The backend interpolates requested quantiles;
    probabilities outside its trained grid are clipped to the grid endpoints.
    Forecasting uses the backend's native multi-horizon inference and quantile-path
    rollout for longer horizons. ``forecast_strategy`` is not supported.
    Other constructor options are passed to ``ForecastModel``.

    Parameters
    ----------
    model_id : str
        Hugging Face checkpoint or local checkpoint directory.
        Defaults to ``amazon/chronos-2``. Available checkpoints:

        - ``amazon/chronos-2``: approximately 120 million parameters.
        - ``autogluon/chronos-2-synth``: approximately 120 million parameters.
        - ``autogluon/chronos-2-small``: approximately 28 million parameters.

    device : str
        PyTorch device. Defaults to ``cpu``.
    """

    _supports_quantiles = True
    _handles_mixed_frequencies = False

    def __init__(self, model_id: str = "amazon/chronos-2", device: str = "cpu", **kwargs):
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
        inputs = {"target": history.to_numpy(dtype=float, copy=True).T}
        if self.X is not None:
            if X is None:
                X = self.X
            if isinstance(X.index, pd.PeriodIndex):
                X = X.copy()
                X.index = X.index.to_timestamp(how="end").normalize()
            covariates = X.reindex(index=calendar.append(dates), columns=self.X.columns)
            inputs["past_covariates"] = {
                name: covariates.loc[calendar, name].to_numpy(dtype=float, copy=True)
                for name in self.X.columns
            }
            inputs["future_covariates"] = {
                name: covariates.loc[dates, name].to_numpy(dtype=float, copy=True)
                for name in self.X.columns
                if covariates.loc[dates, name].notna().any()
            }
        probabilities = [0.5] if quantiles is None else list(quantiles)
        backend = _load_forecaster(self.model_id, self.device)
        distributions, points = backend.predict_quantiles(
            inputs=[inputs], prediction_length=steps, quantile_levels=probabilities
        )
        values = points[0] if quantiles is None else distributions[0]
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        expected = (len(self.y.columns), steps)
        if quantiles is not None:
            expected += (len(probabilities),)
        if values.shape != expected:
            raise ValueError(
                f"Backend forecasts have shape {values.shape}; expected {expected}."
            )
        if quantiles is None:
            return values.T
        values = values.transpose(1, 0, 2)
        return pd.DataFrame(
            {
                "date": dates.repeat(len(self.y.columns) * len(probabilities)),
                "variable": np.tile(np.repeat(self.y.columns, len(probabilities)), steps),
                "quantile": np.tile(probabilities, steps * len(self.y.columns)),
                "value": values.reshape(-1),
            }
        )
