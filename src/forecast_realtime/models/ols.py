"""Ordinary Least Squares regression model for time series forecasting."""

import numpy as np

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

    def _fit_reg(self, y: np.ndarray, X: np.ndarray):
        """Fit one horizon; ``fit()`` calls this helper for multiple horizons."""
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta
