"""Local forecasting with The Forecasting Company's t0 model."""

from functools import lru_cache

import numpy as np
import pandas as pd

from forecast_realtime.forecast_model import ForecastModel


@lru_cache(maxsize=4)
def _load_forecaster(model_id, device):
    try:
        from t0 import T0Forecaster
    except ModuleNotFoundError as error:
        if error.name != "t0":
            raise
        raise ModuleNotFoundError(
            'Install the local t0 model with pip install "forecast-realtime[t0]".',
            name="t0",
        ) from error
    return T0Forecaster.from_pretrained(model_id).to(device).eval()


class ForecastT0(ForecastModel):
    """Forecast locally with pretrained t0 weights.

    Fit stores the context and loads the pretrained model; forecasting reuses it.
    Point forecasts are medians. No API key is required.
    Other constructor options are passed to ``ForecastModel``.

    Parameters
    ----------
    model_id : str
        Hugging Face checkpoint. Defaults to ``theforecastingcompany/t0-gamma``.
    device : str
        PyTorch device. Defaults to ``cpu``.
    """

    _supports_quantiles = True
    _handles_mixed_frequencies = False

    def __init__(
        self,
        model_id: str = "theforecastingcompany/t0-gamma",
        device: str = "cpu",
        **kwargs,
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
        context = history.to_numpy(dtype=float, copy=True).T[None, ...]
        covariates = None
        if self.X is not None:
            if X is None:
                X = self.X
            if isinstance(X.index, pd.PeriodIndex):
                X = X.copy()
                X.index = X.index.to_timestamp(how="end").normalize()
            covariates = (
                X.reindex(index=calendar.append(dates), columns=self.X.columns)
                .to_numpy(dtype=float, copy=True)
                .T[None, ...]
            )
        prediction = _load_forecaster(self.model_id, self.device).predict(
            context,
            horizon=steps,
            quantile_levels=(0.5,) if quantiles is None else tuple(quantiles),
            future_covariates=covariates,
        )
        values = prediction.median if quantiles is None else prediction.quantiles
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        values = np.asarray(values)
        if quantiles is None:
            return values[0].T
        levels = np.asarray(prediction.quantile_levels, dtype=float)
        expected = (1, len(self.y.columns), steps, len(levels))
        if values.shape != expected:
            raise ValueError(
                f"Backend quantiles have shape {values.shape}; expected {expected}."
            )
        selected = []
        for probability in quantiles:
            matches = np.flatnonzero(np.isclose(levels, probability, rtol=0, atol=1e-8))
            if len(matches) != 1:
                raise ValueError(f"Backend does not provide quantile {probability}.")
            selected.append(int(matches[0]))
        values = np.take(values[0], selected, axis=-1).transpose(1, 0, 2)
        return pd.DataFrame(
            {
                "date": dates.repeat(len(self.y.columns) * len(quantiles)),
                "variable": np.tile(np.repeat(self.y.columns, len(quantiles)), steps),
                "quantile": np.tile(quantiles, steps * len(self.y.columns)),
                "value": values.reshape(-1),
            }
        )
