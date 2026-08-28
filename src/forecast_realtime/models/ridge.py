"""Ridge regression with optional cross-validated alpha for forecasting."""

import numpy as np
from sklearn.linear_model import Ridge as _Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import BaseCrossValidator

from forecast_realtime.linear_regression import LinearRegression

# Multiplier taking alpha from each loss convention to sklearn's sum-of-squares one.
ALPHA_SCALINGS = {
    "sum": lambda n: 1.0,
    "mean": lambda n: float(n),
}


class ForecastRidge(LinearRegression):
    """Ridge regression with optional cross-validated alpha: y = Xβ + ε.

    Fits a single-equation Ridge regression of y (one variable) on X
    (regressors) using ``sklearn.linear_model.Ridge`` or
    ``sklearn.linear_model.RidgeCV``. Set ``cv`` to select the regularisation
    parameter by cross-validation.

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
        Regularisation strength, expressed on the scale set by
        ``alpha_scaling``. If ``None`` and ``cv`` is also ``None``, defaults
        to 0.1. Ignored when ``cv`` is set.
    cv : int | BaseCrossValidator | None
        Cross-validation splitter or number of splits. Default is ``None``.
    label : str | None
        Name used to identify the model's forecasts. Defaults to the class
        name.
    alphas : np.ndarray | list | None
        Alpha values to try when ``cv`` is set.
    alpha_scaling : str
        Loss normalisation used for ``alpha``. Default is ``"mean"``.
    formula : str | None
        Optional formula selecting the target and regressors.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    drop_nans : bool
        Whether to remove rows containing missing values before fitting.
    align_start_dates : bool
        Whether to align the starts of the target and regressor series.

    Attributes
    ----------
    alpha_ : float or None
        Penalty actually applied, reported on the ``alpha_scaling`` scale.
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
        alpha_scaling: str = "mean",
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
        if cv is None and alphas is not None:
            raise TypeError("alphas can only be set when cv is not None")
        if cv is not None and alphas is not None and np.asarray(alphas).ndim == 0:
            raise TypeError("alphas must be array-like when cv is not None")
        if alpha_scaling not in ALPHA_SCALINGS:
            raise ValueError(
                f"alpha_scaling must be one of {sorted(ALPHA_SCALINGS)}, "
                f"got {alpha_scaling!r}"
            )

        self.alpha = alpha
        self.alphas = alphas
        self.alpha_scaling = alpha_scaling
        self.cv = cv
        self.alpha_ = None

    def _fit_reg(self, y: np.ndarray, X: np.ndarray):
        """Fit a single-horizon Ridge model, optionally selecting alpha by CV.

        This is a helper function for fit, which handles multiple steps.

        ``alpha``/``alphas`` are converted from the ``alpha_scaling`` loss
        convention to the sum-of-squares scale sklearn expects, and
        ``alpha_`` is converted back so it is reported on the scale the user
        supplied.
        """
        # Under CV the folds train on fewer than n rows, so the rescaling is
        # exact only for the final refit.
        factor = ALPHA_SCALINGS[self.alpha_scaling](X.shape[0])

        if self.cv is None:
            alpha = self.alpha if self.alpha is not None else 0.1
            model = _Ridge(alpha=alpha * factor, fit_intercept=False)
        else:
            alphas = self.alphas if self.alphas is not None else (0.1, 1.0, 10.0)
            model = RidgeCV(
                alphas=np.asarray(alphas, dtype=float) * factor,
                fit_intercept=False,
                cv=self.cv,
            )
        model.fit(X, np.asarray(y).ravel())

        self.alpha_ = (model.alpha if self.cv is None else model.alpha_) / factor

        # Return coefficients in same format as OLS (k x 1)
        beta = model.coef_.reshape(-1, 1)
        return beta
