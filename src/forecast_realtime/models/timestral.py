"""Local forecasting with Mistral's Timestral models."""

from functools import lru_cache

import numpy as np
import pandas as pd

from forecast_realtime.forecast_model import ForecastModel


@lru_cache(maxsize=4)
def _load_forecaster(model_id, device):
    try:
        from timestral import TimestralForecaster
    except ModuleNotFoundError as error:
        if error.name != "timestral":
            raise
        raise ModuleNotFoundError(
            "Install Timestral with pip install "
            "git+https://github.com/mistralai/timestral.git.",
            name="timestral",
        ) from error
    return TimestralForecaster.from_pretrained(model_id, device=device)


class ForecastTimestral(ForecastModel):
    """Forecast locally with pretrained Timestral weights.

    Fit stores the context and loads the pretrained model; forecasting reuses it.
    Point forecasts are medians. Quantiles must be supplied by the checkpoint.
    The backend predicts horizons jointly within each block and rolls longer
    horizons forward using predicted medians. The linear-model option
    ``forecast_strategy`` is not supported; backend block recursion remains enabled.
    Other constructor options are passed to ``ForecastModel``.

    Parameters
    ----------
    model_id : str
        Hugging Face checkpoint. Defaults to ``mistralai/Timestral-209M``.
    device : str
        PyTorch device. Defaults to ``cpu``.
    """

    _supports_quantiles = True
    _handles_mixed_frequencies = False

    def __init__(
        self, model_id: str = "mistralai/Timestral-209M", device: str = "cpu", **kwargs
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
        prediction = _load_forecaster(self.model_id, self.device).forecast(
            context,
            prediction_length=steps,
            known_covariates=covariates,
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
