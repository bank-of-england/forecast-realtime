"""ForecastModel wrapper for ``nowcast_midas.MIDAS``."""

import nowcast_midas as nm
import numpy as np
import pandas as pd

from forecast_realtime._utils import validate_forecast_horizons
from forecast_realtime.forecast_model import ForecastModel


def _to_midas_regressors(X: pd.DataFrame) -> pd.DataFrame:
    series = X.iloc[:, 0]
    last_observation = series.last_valid_index()
    if last_observation is None:
        raise ValueError("MIDAS requires at least one non-missing regressor value")
    series = series.loc[:last_observation]
    return pd.DataFrame({"date": series.index, "value": series.to_numpy()})


class ForecastMIDAS(ForecastModel):
    """MIDAS regression wrapper for the forecast_realtime framework.

    Parameters
    ----------
    method : str
        Weighting scheme: ``'almon'``, ``'exp_almon'``, ``'beta'``,
        or ``'unrestricted'``. Default ``'almon'``.
    n_lags : int
        Number of high-frequency (monthly) lags. Default 6.
    n_pars_weights : int
        Number of weight-shape parameters for exp_almon/almon. Default 2.
    estimator : str | None
        ``'ols'`` or ``'nls'``. Defaults to ``'ols'`` for
        unrestricted/almon, ``'nls'`` otherwise.
    horizons : list | None
        Horizons for direct multi-step forecasting. Each horizon
        is fitted as a separate model: ``y[t+h] ~ X[t]``.
        If ``None`` (default), horizons are derived from ``steps``
        at fit time.
    start_lag : int
        Index of the first lag to include (default 0). When
        ``start_lag=1`` the most recent monthly observation (lag 0)
        is skipped.
    dummy_periods : list | None
        Optional list of low-frequency dates (quarter ends) to include as
        outlier dummies in the regression. Default None.
    n_ar_lags : int
        Number of autoregressive lags of the target to include as
        additional regressors (default 0 = no AR terms). When > 0 the
        model becomes
        ``y[t+h] = alpha + beta * X[t]'w + gamma'D[t+h]
        + sum_{k=1..p} phi_k * y[t+h-k] + eps`` with ``p = n_ar_lags``.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target and regressors.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    """

    _handles_mixed_frequencies = True

    # MIDAS infers its own info-date/horizon from ragged-edge regressor data.
    _needs_ragged_edge_imputation = False
    _forecast_dates_include_origin = True

    def __init__(
        self,
        method: str = "almon",
        n_lags: int = 6,
        n_pars_weights: int = 2,
        estimator: str | None = None,
        horizons: list | None = None,
        start_lag: int = 0,
        dummy_periods: list | None = None,
        n_ar_lags: int = 0,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
    ) -> None:

        super().__init__(
            label=label,
            formula=formula,
            data_transformation=data_transformation,
        )

        self.method = method
        self.n_lags = n_lags
        self.n_pars_weights = n_pars_weights
        self.estimator = estimator
        self.horizons = horizons
        self.start_lag = start_lag
        self.dummy_periods = dummy_periods
        self.n_ar_lags = n_ar_lags

        # Internal nowcast_midas model(s) — one created per fit() call
        self.model: nm.MIDAS | None = None
        # Store regressors for forecasting
        self._regressors: pd.DataFrame | None = None

    def _fit(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame | None = None,
        **kwargs,
    ):
        """Fit the MIDAS model.

        Parameters
        ----------
        y : pd.DataFrame
            Low-frequency target with DatetimeIndex and one column.
            The index dates are interpreted as quarterly end-of-period dates.
        X : pd.DataFrame, optional
            High-frequency (monthly) regressors with DatetimeIndex and one
            column. Required for MIDAS — raises ValueError if not provided.

        Returns
        -------
        self
        """
        if X is None:
            raise ValueError(
                "MIDAS requires high-frequency regressors (X). "
                "Pass monthly data as X to fit()."
            )

        incoming_index = y.index
        target_name = y.columns[0]

        # Convert y (DatetimeIndex DataFrame) to nowcast_midas format
        target = pd.DataFrame({"date": y.index, "value": y.iloc[:, 0].values})

        # Convert X (DatetimeIndex DataFrame) to nowcast_midas format
        regressors = _to_midas_regressors(X)

        # Store y and regressors for use in forecasting/refitting
        self.y = y
        self._regressors = regressors.copy()
        # Remember the regressor's name for decomposition component labels
        self._x_name = str(X.columns[0])

        # Determine horizons: explicit > steps kwarg > default [0]
        steps = kwargs.pop("steps", None)
        if self.horizons is not None:
            horizons = self.horizons
        elif steps is not None:
            horizons = list(range(steps))
        else:
            horizons = [0]

        self.model = nm.MIDAS(
            method=self.method,
            n_lags=self.n_lags,
            n_pars_weights=self.n_pars_weights,
            estimator=self.estimator,
            horizons=horizons,
            start_lag=self.start_lag,
            n_ar_lags=self.n_ar_lags,
            dummy_periods=self.dummy_periods,
        )

        self.model.fit(target=target, regressors=regressors)

        if 0 in self.model.fits_:
            self.fitted_values_ = self.model.fits_[0].fitted_values.reindex(
                incoming_index
            )
            self.fitted_values_.name = target_name
        else:
            self.fitted_values_ = pd.Series(
                np.nan, index=incoming_index, name=target_name
            )

        return self

    def _forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Produce multi-step forecasts using the fitted MIDAS model.

        Uses direct forecasting: one model per horizon. If the model was
        not fitted with enough horizons, it is re-fitted to cover all
        requested steps.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        X : pd.DataFrame, optional
            Updated high-frequency regressors for the forecast period.
            If None, uses the regressors stored from fit().
        y : pd.DataFrame, optional
            Conditioning paths (unused; kept for API compatibility).
        **kwargs
            Additional model-specific arguments.

        Returns
        -------
        pd.DataFrame
            Forecasts indexed by a ``DatetimeIndex`` of model-anchored
            forecast dates, with the same column name as ``y``. Missing
            horizons are filled with NaN.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Use stored regressors if none provided for forecast
        if X is not None and isinstance(X, pd.DataFrame):
            regressors = _to_midas_regressors(X)
        elif X is not None and isinstance(X, np.ndarray):
            # If X is a numpy array (from RealTimeModel loop), we can't
            # use it directly for MIDAS forecasting — use stored regressors
            regressors = self._regressors
        else:
            regressors = self._regressors

        # Check if we need to refit with more horizons
        max_fitted_horizon = max(self.model.fits_.keys())
        if steps - 1 > max_fitted_horizon:
            # Refit with the required horizons
            target = pd.DataFrame(
                {"date": self.y.index, "value": self.y.iloc[:, 0].values}
            )
            self.model = nm.MIDAS(
                method=self.method,
                n_lags=self.n_lags,
                n_pars_weights=self.n_pars_weights,
                estimator=self.estimator,
                horizons=list(range(steps)),
                start_lag=self.start_lag,
                n_ar_lags=self.n_ar_lags,
                dummy_periods=self.dummy_periods,
            )
            self.model.fit(target=target, regressors=self._regressors)

        # Produce forecasts for each horizon
        forecasts_df = self.model.forecast(regressors)
        validate_forecast_horizons(
            forecasts_df["horizon"], steps, self.__class__.__name__
        )

        # Build full output array (steps, 1) with NaNs for missing horizons.
        # forecasts_df may be sparse (e.g., some horizons don't converge);
        # we fill all positions and extract model-anchored forecast dates.
        # The realtime loop reads these dates from the returned DataFrame to
        # label horizons correctly (info_date may lie in an earlier quarter
        # than the vintage when X has publication lag).
        forecasts = np.full((steps, 1), np.nan)
        dates: list = [pd.NaT] * steps
        for _, row in forecasts_df.iterrows():
            h = int(row["horizon"])
            if h < steps:
                forecasts[h, 0] = row["value"]
                dates[h] = pd.Timestamp(row["date"])

        return pd.DataFrame(
            forecasts,
            index=pd.DatetimeIndex(dates, name="date"),
            columns=self.y.columns,
        )

    def _forecast_decomp(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Return the MIDAS forecast decomposition in the model contract."""
        # Build regressors identically to _forecast().
        if X is not None and isinstance(X, pd.DataFrame):
            regressors = _to_midas_regressors(X)
        else:
            regressors = self._regressors

        if regressors is None:
            return None

        raw = self.model.forecast_decomp(
            regressors,
            regressor_name=getattr(self, "_x_name", "X"),
        )

        if raw is None or raw.empty:
            return None

        raw = raw[raw["horizon"] < steps].copy()
        if raw.empty:
            return None

        raw = raw.rename(columns={"horizon": "forecast_horizon"})
        return raw[["forecast_horizon", "component", "contribution", "weight"]]
