"""Ordinary Least Squares regression model for time series forecasting."""

import numpy as np
import pandas as pd
from scipy.stats import norm

from forecast_realtime.linear_regression import LinearRegression


class ForecastOLS(LinearRegression):
    """Plain vanilla OLS model: y = Xβ + ε.

    Fits a single-equation OLS regression of y (one variable) on X
    (regressors) using ``numpy.linalg.lstsq``.

    Parameters
    ----------
    fit_intercept : bool
        Whether to include an intercept term. Default is True.
    """

    _supports_quantiles = True

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None, **kwargs):
        super()._fit(y, X, **kwargs)
        residuals = y.iloc[:, 0].to_numpy(dtype=float) - self.fitted_values_.to_numpy(
            dtype=float
        )
        degrees_freedom = len(residuals) - self.N_regressors
        self._std_error = (
            float(np.sqrt(residuals @ residuals / degrees_freedom))
            if degrees_freedom > 0
            else None
        )
        return self

    def _fit_reg(self, y: np.ndarray, X: np.ndarray):
        """Fit one horizon; ``fit()`` calls this helper for multiple horizons."""
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    def _forecast(
        self, steps=None, X=None, y=None, forecast_origin=None, quantiles=None, **kwargs
    ):
        if quantiles is None:
            return super()._forecast(
                steps=steps, X=X, y=y, forecast_origin=forecast_origin, **kwargs
            )
        if (
            self.forecast_strategy == "direct"
            or self._fitted_model_configuration.design.y_lags
        ):
            raise ValueError(
                "Linear regression quantiles do not support target lags or "
                "direct strategies."
            )
        if self._std_error is None:
            raise ValueError(
                "Linear regression quantiles require positive residual "
                "degrees of freedom."
            )

        point = super()._forecast(
            steps=steps, X=X, y=y, forecast_origin=forecast_origin, **kwargs
        )
        mean = point.iloc[:, 0].to_numpy(dtype=float)
        values = mean[:, None] + self._std_error * norm.ppf(quantiles)
        return pd.DataFrame(
            {
                "date": np.repeat(point.index, len(quantiles)),
                "variable": self.y.columns[0],
                "quantile": np.tile(quantiles, len(point)),
                "value": values.reshape(-1),
            }
        )
