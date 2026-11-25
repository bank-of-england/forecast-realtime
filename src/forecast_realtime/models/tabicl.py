"""Local forecasting with TabICLv2."""

import sys
from contextvars import ContextVar
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from forecast_realtime.forecast_model import ForecastModel

_show_series_progress = ContextVar("tabicl_show_series_progress", default=True)


def _series_progress(iterable, **options):
    if (
        options.get("desc") == "Predicting time series"
        and not _show_series_progress.get()
    ):
        options["disable"] = True
    return tqdm(iterable, **options)


def _timestamps(frame):
    if isinstance(frame.index, pd.PeriodIndex):
        frame = frame.set_axis(frame.index.to_timestamp(how="end").normalize())
    return frame


class ForecastTabICLv2(ForecastModel):
    """Forecast each target series independently with the local TabICLv2 forecaster.

    ``fit()`` prepares ``TabICLForecaster`` without fine-tuning its weights.
    ``forecast()`` uses the history through the forecast origin to predict the
    whole horizon in one call. It returns medians by default or the requested
    quantiles. Supply predictor history as ``X`` at fit time and known future
    values as ``X`` at forecast time. Predictors with no values in the horizon
    are ignored.

    Parameters
    ----------
    device : str
        Inference device. Use ``"cuda"`` for a CUDA-capable GPU. This argument
        takes precedence over the device in ``estimator_kwargs``.
    model_id : str | Path | None
        Published regression checkpoint filename (a string) or an existing local
        checkpoint file. An explicit selection takes precedence over checkpoint
        settings in ``estimator_kwargs``. Omit it to keep the backend defaults.
    estimator_kwargs : dict | None
        Options for the underlying TabICL regressor, passed to the forecaster
        as ``tabicl_config``.
    progressbar : bool
        Show TabICL's per-series prediction bar. Default is False.

    Notes
    -----
    Additional keyword arguments, such as ``label`` and ``formula``, go to
    ``ForecastModel``.

    Install the optional backend with ``pip install "forecast-realtime[tabicl]"``.
    """

    _supports_quantiles = True
    _handles_mixed_frequencies = False

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_id: str | Path | None = None,
        estimator_kwargs: dict | None = None,
        progressbar: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model_id = model_id
        self.estimator_kwargs = dict(estimator_kwargs or {})
        self.progressbar = progressbar

    def _fit(self, y, X=None):
        try:
            from tabicl import TabICLForecaster
        except ModuleNotFoundError as error:
            if error.name != "tabicl":
                raise
            raise ModuleNotFoundError(
                'Install TabICLv2 with pip install "forecast-realtime[tabicl]".',
                name="tabicl",
            ) from error
        config = {**self.estimator_kwargs, "device": self.device}
        if self.model_id is not None:
            path = Path(self.model_id)
            if path.is_file():
                config["model_path"] = path
            elif isinstance(self.model_id, str) and self.model_id == path.name:
                config.pop("model_path", None)
                config["checkpoint_version"] = self.model_id
            else:
                raise FileNotFoundError(f"TabICL checkpoint file not found: {path}")
        self._backend = TabICLForecaster(point_estimate="median", tabicl_config=config)
        dispatch = sys.modules.get("tabicl.forecast._dispatch")
        if dispatch is not None and getattr(dispatch, "tqdm", None) is tqdm:
            dispatch.tqdm = _series_progress
        return self

    def _forecast(self, steps=1, X=None, y=None, forecast_origin=None, quantiles=None):
        history = _timestamps(self.y if y is None else y)
        history = history.loc[:forecast_origin, self.y.columns]
        context = history.rename_axis(index="timestamp", columns="item_id")
        context = context.stack(future_stack=True).rename("target").reset_index()
        dates = self._forecast_dates(forecast_origin, steps)
        index = pd.MultiIndex.from_product(
            [dates, history.columns], names=["timestamp", "item_id"]
        )

        future = index.to_frame(index=False)
        if self.X is not None:
            regressors = _timestamps(self.X if X is None else X).reindex(
                columns=self.X.columns
            )
            horizon = regressors.reindex(dates).dropna(axis=1, how="all")
            names = {
                name: f"covariate_{number}" for number, name in enumerate(horizon.columns)
            }
            if names:
                context = context.join(
                    regressors[list(names)].rename(columns=names), on="timestamp"
                )
                future = future.join(horizon.rename(columns=names), on="timestamp")

        inferred = history.index.inferred_freq if len(history) >= 3 else None
        regular = inferred is not None and pd.date_range(
            history.index[-1], periods=steps + 1, freq=inferred
        )[1:].equals(dates)
        options = (
            {"prediction_length": steps}
            if regular and len(future.columns) == 2
            else {"future_df": future}
        )
        token = _show_series_progress.set(self.progressbar)
        try:
            prediction = self._backend.predict_df(
                context_df=context,
                quantiles=[0.5] if quantiles is None else list(quantiles),
                **options,
            ).reorder_levels(index.names)
        finally:
            _show_series_progress.reset(token)
        if (
            not prediction.index.is_unique
            or len(prediction) != len(index)
            or not index.isin(prediction.index).all()
        ):
            raise ValueError(
                "Backend forecasts do not match the requested dates and targets."
            )
        prediction = prediction.reindex(index)
        if quantiles is None:
            return prediction["target"].to_numpy().reshape(steps, len(history.columns))
        prediction.columns = prediction.columns.map(str)
        values = prediction[[str(level) for level in quantiles]]
        values = values.set_axis(quantiles, axis=1)
        values = values.rename_axis(index=["date", "variable"], columns="quantile")
        return values.stack(future_stack=True).rename("value").reset_index()
