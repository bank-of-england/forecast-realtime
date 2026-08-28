"""ForecastModel wrapper for ``nowcast_midas.MultiMIDAS``."""

import nowcast_midas as nm
import numpy as np
import pandas as pd

from forecast_realtime._utils import validate_forecast_horizons
from forecast_realtime.forecast_model import ForecastModel


class ForecastMultiMIDAS(ForecastModel):
    """Multi-regressor MIDAS regression wrapper for the forecast_realtime framework.

    Parameters
    ----------
    variables : list
        Regressors to include.  Pass a plain string to use the shared
        defaults; pass a :class:`~nowcast_midas.specs.VariableSpec` to override
        any parameter for that regressor.  Use
        ``VariableSpec(..., frequency='QE')`` for quarterly regressors.
    method : str
        Shared weighting scheme for monthly variables given as plain strings:
        ``'almon'``, ``'exp_almon'``, ``'beta'``, or ``'unrestricted'``.
        Default ``'almon'``.
    n_lags : int
        Shared number of lags (default 3).
    n_pars_weights : int
        Shared weight-shape parameters for polynomial schemes (default 2).
    estimator : str | None
        Shared estimator override.  ``None`` (default) chooses automatically
        per variable based on method.
    horizons : list | None
        Horizons for direct multi-step forecasting.  Each horizon is fitted
        as a separate model: ``y[t+h] ~ X[t]``.  If ``None`` (default),
        horizons are derived from ``steps`` at fit time.
    start_lag : int
        Shared starting lag index (default 0).
    dummy_periods : list | None
        Optional list of low-frequency dates (quarter ends) to include as
        outlier dummies in the regression.  Default ``None``.
    n_ar_lags : int
        Number of autoregressive lags of the target to include as additional
        regressors (default 0 = no AR terms).
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target and regressors.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    """

    _handles_mixed_frequencies = True

    # MultiMIDAS infers its own info-date/horizon from ragged-edge regressor data.
    _needs_ragged_edge_imputation = False
    _forecast_dates_include_origin = True

    def __init__(
        self,
        variables: list,
        method: str = "almon",
        n_lags: int = 3,
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

        self.variables = variables
        self.method = method
        self.n_lags = n_lags
        self.n_pars_weights = n_pars_weights
        self.estimator = estimator
        self.horizons = horizons
        self.start_lag = start_lag
        self.dummy_periods = dummy_periods
        self.n_ar_lags = n_ar_lags

        # Internal nowcast_midas model — created per fit() call
        self.model: nm.MultiMIDAS | None = None
        # Store regressors for forecasting
        self._regressors: pd.DataFrame | None = None

    @staticmethod
    def _X_to_long(X: pd.DataFrame) -> pd.DataFrame:
        """Convert a wide regressor DataFrame to long format.

        Parameters
        ----------
        X : pd.DataFrame
            Wide DataFrame with DatetimeIndex, one column per regressor.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with ``date``, ``variable``, and ``value``
            columns, dropping NaN rows.
        """
        rows = []
        for col in X.columns:
            series = X[col].dropna()
            rows.append(
                pd.DataFrame(
                    {
                        "date": series.index,
                        "variable": col,
                        "value": series.to_numpy(),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    def _fit(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame | None = None,
        **kwargs,
    ):
        """Fit the MultiMIDAS model.

        Parameters
        ----------
        y : pd.DataFrame
            Low-frequency target with DatetimeIndex and one column.
            The index dates are interpreted as quarterly end-of-period dates.
        X : pd.DataFrame, optional
            Wide DataFrame with DatetimeIndex and one column per regressor.
            Required — raises ValueError if not provided.

        Returns
        -------
        self
        """
        if X is None:
            raise ValueError(
                "MultiMIDAS requires high-frequency regressors (X). "
                "Pass a wide DataFrame (one column per regressor) as X to fit()."
            )

        incoming_index = y.index
        target_name = y.columns[0]

        # Convert y (DatetimeIndex DataFrame) to nowcast_midas format
        target = pd.DataFrame({"date": y.index, "value": y.iloc[:, 0].values})

        # Convert wide X to long format for nowcast_midas
        regressors = self._X_to_long(X)

        # Store y and regressors for use in forecasting/refitting
        self.y = y
        self._regressors = regressors.copy()

        # Determine horizons: explicit > steps kwarg > default [0]
        steps = kwargs.pop("steps", None)
        if self.horizons is not None:
            horizons = self.horizons
        elif steps is not None:
            horizons = list(range(steps))
        else:
            horizons = [0]

        self.model = nm.MultiMIDAS(
            variables=self.variables,
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
            fit0 = self.model.fits_[0]
            self.fitted_values_ = pd.Series(
                np.asarray(fit0.fitted_values).ravel(),
                index=pd.DatetimeIndex(fit0.dates),
                name=target_name,
            ).reindex(incoming_index)
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
        """Produce multi-step forecasts using the fitted MultiMIDAS model.

        Uses direct forecasting: one model per horizon.  If the model was
        not fitted with enough horizons, it is re-fitted to cover all
        requested steps.

        Parameters
        ----------
        steps : int
            Number of steps ahead to forecast.
        X : pd.DataFrame, optional
            Updated wide regressor DataFrame for the forecast period.
            If None, uses the regressors stored from fit().

        Returns
        -------
        pd.DataFrame
            Forecasts as a DataFrame with DatetimeIndex (forecast dates)
            and the same column name as ``y``.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Use stored regressors if none provided for forecast
        if X is not None and isinstance(X, pd.DataFrame):
            regressors = self._X_to_long(X)
        else:
            regressors = self._regressors

        # Check if we need to refit with more horizons
        max_fitted_horizon = max(self.model.fits_.keys())
        if steps - 1 > max_fitted_horizon:
            target = pd.DataFrame(
                {"date": self.y.index, "value": self.y.iloc[:, 0].values}
            )
            self.model = nm.MultiMIDAS(
                variables=self.variables,
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

        forecasts = np.full((steps, 1), np.nan)
        dates: list = [pd.NaT] * steps
        for _, row in forecasts_df.iterrows():
            h = int(row["horizon"])
            if h < steps:
                forecasts[h, 0] = row["forecast"]
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
        """Return the MultiMIDAS forecast decomposition in the model contract."""
        # Build regressors identically to _forecast().
        if X is not None and isinstance(X, pd.DataFrame):
            regressors = self._X_to_long(X)
        else:
            regressors = self._regressors

        if regressors is None:
            return None

        raw = self.model.forecast_decomp(regressors)

        if raw is None or raw.empty:
            return None

        raw = raw[raw["horizon"] < steps].copy()
        if raw.empty:
            return None

        raw = raw.rename(columns={"horizon": "forecast_horizon"})
        return raw[["forecast_horizon", "component", "contribution", "weight"]]
