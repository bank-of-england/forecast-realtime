"""Local TabPFN regression with the TabPFN-TS 1.4 default time features."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

from forecast_realtime.forecast_model import ForecastModel

_MAX_CONTEXT = 32768
_SEASONAL_SLOTS = 12


def _timestamps(frame):
    if isinstance(frame.index, pd.PeriodIndex):
        frame = frame.set_axis(frame.index.to_timestamp(how="end").normalize())
    return frame


def _seasonal_periods(values):
    """Estimate the reference model's twelve spectral peaks from observed targets."""
    size = len(values)
    if size < 2:
        return []
    position = np.arange(size)
    trend = np.polyval(np.polyfit(position, values, 1), position)
    spectrum = np.abs(rfft((values - trend) * np.hanning(size), n=2 * size))
    spectrum[0] = 0
    peaks = find_peaks(spectrum, height=0.05 * spectrum.max())[0]
    if not peaks.size:
        peaks = np.arange(spectrum.size)
    peaks = peaks[np.argsort(spectrum[peaks])[::-1][:_SEASONAL_SLOTS]]
    peaks = peaks[peaks != 0]
    periods, first = np.unique(np.rint(1 / rfftfreq(2 * size)[peaks]), return_index=True)
    return periods[np.argsort(-spectrum[peaks[first]], kind="stable")]


def _covariate_window(calendars):
    """Use a regular series' frequency, or the most common step in gapped series."""
    deltas = []
    for dates in calendars:
        if len(dates) >= 3 and (frequency := pd.infer_freq(dates)) is not None:
            offset = pd.tseries.frequencies.to_offset(frequency)
            break
        deltas.extend(dates[1:] - dates[:-1])
    else:
        if not deltas:
            return 4
        step = pd.Series(deltas).mode().iloc[0]
        days = step / pd.Timedelta(days=1)
        if 28 <= days <= 31:
            return 12
        if 89 <= days <= 92:
            return 4
        offset = pd.tseries.frequencies.to_offset(step)
    name = offset.rule_code.split("-")[0].upper()
    cycles = {"MIN": 1440, "H": 24, "D": 7, "B": 7, "ME": 12, "MS": 12}
    return max(4, round(cycles.get(name, 4) / offset.n))


def _feature_matrix(target, dates, predictors, window):
    """Build context and horizon together; only observed targets determine periods."""
    calendar = target.index.append(dates)
    position = np.arange(len(calendar))
    blocks = []
    if predictors is not None:
        known = predictors.reindex(calendar)
        blocks.append(known.to_numpy(na_value=np.nan))
        numeric = known.select_dtypes(include=["number", "bool"]).astype(float)
        if len(numeric.columns):
            blocks.extend(
                features.to_numpy(dtype=np.float32)
                for features in (
                    numeric.shift(1),
                    numeric.shift(window),
                    numeric.rolling(window, min_periods=1).mean(),
                )
            )
    blocks.extend((position, calendar.year))
    components = (
        (calendar.second, 59),
        (calendar.minute, 59),
        (calendar.hour, 23),
        (calendar.day_of_week, 6),
        (calendar.day - 1, 29.5),
        (calendar.day_of_year - 1, 364),
        (calendar.isocalendar().week - 1, 51),
        (calendar.month - 1, 11),
    )
    for component, divisor in components:
        angle = 2 * np.pi * np.asarray(component, dtype=np.int32) / divisor
        blocks.extend(
            (np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32))
        )
    periodic = np.zeros((len(calendar), 2 * _SEASONAL_SLOTS), dtype=np.float32)
    for slot, period in enumerate(_seasonal_periods(target.to_numpy(dtype=float))):
        angle = 2 * np.pi * position / period
        periodic[:, 2 * slot] = np.sin(angle)
        periodic[:, 2 * slot + 1] = np.cos(angle)
    return np.column_stack([*blocks, periodic])


class ForecastTabPFN(ForecastModel):
    """Forecast each target series independently with a local TabPFNRegressor.

    ``fit()`` prepares the regressor configuration without fine-tuning the checkpoint.
    ``forecast()`` sends the history through the forecast origin and requests
    the whole horizon in one call. It returns medians by default. To request
    an 80% prediction interval, pass ``quantiles=[0.1, 0.9]`` to ``forecast()``.
    Supply predictor history as ``X`` at fit time and known future values as
    ``X`` at forecast time. Predictors with no values in the horizon are ignored.
    The adapter builds the default TabPFN-TS 1.4 time and known-predictor features
    locally; it does not require the time-series pipeline or the hosted client.

    Parameters
    ----------
    device : str
        Inference device. Defaults to ``"cpu"``. Use ``"cuda"`` for a CUDA-capable
        GPU. This argument takes precedence over the device in ``estimator_kwargs``.
    model_id : str | Path | None
        TabPFN checkpoint filename or local path. For TabPFN-3.5-Fast, use
        ``model_id="tabpfn-v3.5-fast-20260909.safetensors"``. This overrides
        ``estimator_kwargs["model_path"]``. The default is the TabPFN-3.5 regressor
        checkpoint. The older ``model_path`` keyword is an alias;
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
            from tabpfn import TabPFNRegressor
        except ModuleNotFoundError as error:
            if error.name != "tabpfn":
                raise
            raise ModuleNotFoundError(
                'Install TabPFN with pip install "forecast-realtime[tabpfn]".',
                name="tabpfn",
            ) from error
        config = {**self.estimator_kwargs, "device": self.device}
        if self.model_id is not None:
            config["model_path"] = self.model_id
        if config.get("model_path") is None:
            from tabpfn.constants import ModelVersion

            config["model_path"] = TabPFNRegressor.create_default_for_version(
                ModelVersion("v3.5")
            ).model_path
        self._regressor_class = TabPFNRegressor
        self._regressor_config = config
        return self

    def _forecast(self, steps=1, X=None, y=None, forecast_origin=None, quantiles=None):
        history = _timestamps(self.y if y is None else y)
        history = history.loc[:forecast_origin, self.y.columns]
        dates = self._forecast_dates(forecast_origin, steps)
        predictors = None
        if self.X is not None:
            regressors = _timestamps(self.X if X is None else X).reindex(
                columns=self.X.columns
            )
            horizon = regressors.reindex(dates).dropna(axis=1, how="all")
            if len(horizon.columns):
                predictors = regressors[horizon.columns]

        targets = [
            (series.dropna() if series.count() > 1 else series.fillna(0)).tail(
                _MAX_CONTEXT
            )
            for _, series in history.items()
        ]
        window = (
            _covariate_window(target.index.append(dates) for target in targets)
            if predictors is not None
            else 4
        )
        levels = [0.5] if quantiles is None else list(quantiles)
        forecasts = []
        for target in targets:
            features = _feature_matrix(target, dates, predictors, window)
            model = self._regressor_class(**self._regressor_config)
            model.fit(features[: len(target)], target.to_numpy())
            output = model.predict(
                features[len(target) :], output_type="main", quantiles=levels
            )
            values = np.asarray(
                [output["median"]] if quantiles is None else output["quantiles"]
            )
            if values.shape != (len(levels), steps):
                raise ValueError(f"Unexpected TabPFN forecast shape: {values.shape}.")
            forecasts.append(values.T)
        values = np.stack(forecasts, axis=1)
        if quantiles is None:
            return values[:, :, 0]
        index = pd.MultiIndex.from_product(
            [dates, history.columns, levels], names=["date", "variable", "quantile"]
        )
        return pd.Series(values.ravel(), index=index, name="value").reset_index()
