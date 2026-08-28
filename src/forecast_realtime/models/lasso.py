"""Lasso regression with optional cross-validated alpha for time series forecasting."""

import numpy as np
from sklearn.linear_model import Lasso as _Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import BaseCrossValidator

from forecast_realtime.linear_regression import LinearRegression


class ForecastLasso(LinearRegression):
    """Lasso regression with optional CV-selected alpha: y = Xβ + ε.

    Fits a single-equation Lasso regression of y (one variable) on X
    (regressors) using ``sklearn.linear_model.Lasso`` or ``LassoCV``. The
    regularisation parameter (alpha) can be fixed or automatically selected
    via cross-validation.

    Parameters
    ----------
    fit_intercept : bool
        Whether to include an intercept term. Default is True.
    forecast_strategy : str
        Forecasting strategy to use ("recursive" or "direct"). Default is
        "recursive".
    steps : int | None
        For direct forecasting, the horizon to fit. Required when
        ``forecast_strategy="direct"``.
    scale : bool
        Whether to scale X and y before fitting. Default is False.
    alpha : float | None
        Regularisation strength. If ``None`` and ``cv`` is also ``None``,
        defaults to 0.1. Ignored when ``cv`` is set.
    cv : int | BaseCrossValidator | None
        Cross-validation splitter or number of splits. Default is ``None``.
    label : str | None
        Name used to identify the model's forecasts. Defaults to the class
        name.
    alphas : np.ndarray | list | None
        Alpha values to try when ``cv`` is set. If None, use ``LassoCV``'s
        default candidates. Must be array-like rather than a scalar.
    formula : str | None
        Optional formula selecting the target and regressors.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    drop_nans : bool
        Whether to remove rows containing missing values before fitting.
    align_start_dates : bool
        Whether to align the starts of the target and regressor series.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        forecast_strategy: str = "recursive",
        steps: int | None = None,
        scale: bool = False,
        alpha: float | None = None,
        cv: int | BaseCrossValidator | None = None,
        label: str | None = None,
        alphas: np.ndarray | list | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        drop_nans: bool = False,
        align_start_dates: bool = True,
    ):
        super().__init__(
            fit_intercept=fit_intercept,
            forecast_strategy=forecast_strategy,
            steps=steps,
            scale=scale,
            label=label if label is not None else self.__class__.__name__,
            formula=formula,
            data_transformation=data_transformation,
            drop_nans=drop_nans,
            align_start_dates=align_start_dates,
        )
        if cv is not None and alphas is not None and np.asarray(alphas).ndim == 0:
            raise TypeError("alphas must be array-like when cv is not None")

        self.alpha = alpha
        self.alphas = alphas
        self.cv = cv
        self.best_alpha = None

    def _fit_reg(self, y: np.ndarray, X: np.ndarray):
        """Fit one horizon, optionally selecting alpha by cross-validation.

        ``fit()`` calls this helper for multiple horizons.
        """
        if self.cv is not None:
            if self.alphas is None:
                model = LassoCV(
                    cv=self.cv,
                    max_iter=10000,
                    fit_intercept=False,
                )
            else:
                model = LassoCV(
                    alphas=self.alphas,
                    cv=self.cv,
                    max_iter=10000,
                    fit_intercept=False,
                )
        else:
            a = self.alpha if self.alpha is not None else 0.1
            model = _Lasso(alpha=a, max_iter=10000, fit_intercept=False)

        model.fit(X, np.asarray(y).ravel())

        if self.cv is not None:
            self.best_alpha = model.alpha_

        # Return coefficients in same format as OLS (k x 1)
        beta = model.coef_.reshape(-1, 1)
        return beta
