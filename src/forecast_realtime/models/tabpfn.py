"""Local forecasting with the TabPFN-TS pipeline."""

from pathlib import Path

import pandas as pd

from forecast_realtime.forecast_model import ForecastModel


def _timestamps(frame):
    if isinstance(frame.index, pd.PeriodIndex):
        frame = frame.set_axis(frame.index.to_timestamp(how="end").normalize())
    return frame


class ForecastTabPFN(ForecastModel):
    """Forecast each target series independently with the local TabPFN-TS pipeline.

    ``fit()`` prepares the pipeline without fine-tuning the checkpoint.
    ``forecast()`` sends the history through the forecast origin and requests
    the whole horizon in one call. It returns medians by default. To request
    an 80% prediction interval, pass ``quantiles=[0.1, 0.9]`` to ``forecast()``.
    Supply predictor history as ``X`` at fit time and known future values as
    ``X`` at forecast time. Predictors with no values in the horizon are ignored.

    Parameters
    ----------
    device : str
        Inference device. Defaults to ``"cpu"``. Use ``"cuda"`` for a CUDA-capable
        GPU. This argument takes precedence over the device in ``estimator_kwargs``.
    model_id : str | Path | None
        TabPFN checkpoint filename or local path. For TabPFN-3.5-Fast, use
        ``model_id="tabpfn-v3.5-fast-20260909.safetensors"``. This overrides
        ``estimator_kwargs["model_path"]``. If neither is set, TabPFN-TS
        chooses the checkpoint. The older ``model_path`` keyword is an alias;
        specify only one of ``model_id`` and ``model_path``.
    estimator_kwargs : dict | None
        Options for the underlying TabPFN regressor, such as ``n_estimators``.

    Notes
    -----
    Additional keyword arguments, such as ``label`` and ``formula``, go to
    ``ForecastModel``.

    Install with ``pip install "forecast-realtime[tabpfn]"``. Some checkpoints
    require Prior Labs authorisation. Set ``model_id="tabpfn-v2-regressor.ckpt"``
    to use a public checkpoint. There is no hosted fallback.
    """

    _supports_quantiles = True
    _handles_mixed_frequencies = False

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_id: str | Path | None = None,
        estimator_kwargs: dict | None = None,
        **kwargs,
    ):
        model_path = kwargs.pop("model_path", None)
        if model_id is not None and model_path is not None:
            raise TypeError("Specify only one of model_id and model_path.")
        super().__init__(**kwargs)
        self.device = device
        self.model_id = model_id if model_id is not None else model_path
        self.estimator_kwargs = dict(estimator_kwargs or {})

    @property
    def model_path(self):
        """Compatibility alias for ``model_id``."""
        return self.model_id

    @model_path.setter
    def model_path(self, value):
        self.model_id = value

    def _fit(self, y, X=None):
        try:
            from tabpfn_time_series import (
                TABPFN_DEFAULT_CONFIG,
                TabPFNMode,
                TabPFNTSPipeline,
            )
        except ModuleNotFoundError as error:
            if error.name != "tabpfn_time_series":
                raise
            raise ModuleNotFoundError(
                'Install TabPFN with pip install "forecast-realtime[tabpfn]".',
                name="tabpfn_time_series",
            ) from error
        config = {
            **TABPFN_DEFAULT_CONFIG,
            **self.estimator_kwargs,
            "device": self.device,
        }
        if self.model_id is not None:
            config["model_path"] = self.model_id
        self._backend = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_model_config=config,
        )
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
        prediction = self._backend.predict_df(
            context_df=context,
            quantiles=[0.5] if quantiles is None else list(quantiles),
            **options,
        ).reorder_levels(index.names)
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
