"""Ordinary Least Squares regression model for time series forecasting."""

import numpy as np
import pandas as pd
from scipy.stats import t

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

    def _fit_reg(self, y: np.ndarray, X: np.ndarray):
        """Fit one horizon; ``fit()`` calls this helper for multiple horizons."""
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return beta

    def _forecast(
        self, steps=None, X=None, y=None, forecast_origin=None, quantiles=None, **kwargs
    ):
        point = super()._forecast(
            steps=steps, X=X, y=y, forecast_origin=forecast_origin, **kwargs
        )
        if quantiles is None:
            return point

        direct = self.forecast_strategy == "direct"
        if not direct and self._fitted_model_configuration.design.y_lags:
            raise ValueError("Recursive quantiles do not support target lags.")
        rows = self._forecast_design(len(point), X, y, forecast_origin).to_numpy(float)
        horizons = range(len(point)) if direct else [0]
        errors = [self._prediction_error(h, rows) for h in horizons]
        scale = np.concatenate([scale for scale, _ in errors])
        df = np.concatenate([df for _, df in errors])

        values = point.to_numpy(float) + scale[:, None] * t.ppf(quantiles, df[:, None])
        return pd.DataFrame(
            {
                "date": np.repeat(point.index, len(quantiles)),
                "variable": self.y.columns[0],
                "quantile": np.tile(quantiles, len(point)),
                "value": values.reshape(-1),
            }
        )

    def _prediction_error(self, h: int, rows: np.ndarray):
        """Return the prediction standard errors and residual degrees of freedom.

        Matches R's ``predict.lm(interval="prediction")`` and statsmodels'
        ``obs_ci``, treating ``rows`` as known.
        """
        n = len(self.y) - h
        X = np.ones((n, 1)) if self.X is None else self.X.to_numpy(float)[:n]
        if self.X is not None and self.fit_intercept:
            X = np.column_stack([np.ones(n), X])
        y = self.y.to_numpy(float).ravel()[h:]
        direct = self.forecast_strategy == "direct"
        beta = np.ravel(self.betas_[h] if direct else self.beta_)

        df = n - np.linalg.matrix_rank(X)
        if df <= 0:
            raise ValueError(
                "Linear regression quantiles require positive residual "
                "degrees of freedom."
            )
        residuals = y - X @ beta
        leverage = np.sum((rows @ np.linalg.pinv(X)) ** 2, axis=1)  # x0ᵀ(XᵀX)⁺x0
        scale = np.sqrt(residuals @ residuals / df * (1 + leverage))
        return scale, np.full(len(rows), df)
