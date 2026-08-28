"""Ordinary Least Squares regression model for time series forecasting."""

import numpy as np
import pandas as pd

from forecast_realtime._utils import init_recent_y
from forecast_realtime.forecast_model import X_IMPUTATION_METHODS, ForecastModel


class LinearRegression(ForecastModel):
    """OLS regression with optional y and X lags appended to X.

    This base class provides helpers for:
        - direct and recursive forecasting
        - scaling of X and y

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
    label : str | None
        Name used to identify the model's forecasts. Defaults to the class
        name.
    formula : str | None
        Optional patsy-style formula selecting the regressors. Default is None.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    drop_nans : bool
        Whether to remove rows containing missing values before fitting.
    align_start_dates : bool
        Whether to align the starts of the target and regressor series.
    """

    _handles_mixed_frequencies = False
    _supports_multivariate_y = False

    def __init__(
        self,
        fit_intercept: bool = True,
        forecast_strategy: str = "recursive",
        steps: int | None = None,
        scale: bool = False,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        drop_nans: bool = False,
        align_start_dates: bool = True,
    ):
        label = label if label is not None else self.__class__.__name__
        super().__init__(
            label=label,
            formula=formula,
            data_transformation=data_transformation,
            align_start_dates=align_start_dates,
        )
        self.fit_intercept = fit_intercept
        self.model = None
        self.forecast_strategy = forecast_strategy
        self.steps = steps
        self.scale = scale
        self.drop_nans = drop_nans

        # populated by fit when forecast_strategy="recursive"
        self.model = None
        self.beta_ = None

        # populated by fit when forecast_strategy="direct"
        self.models_ = {}
        self.betas_ = {}

        # if you select direct forecasting, you must provide steps
        if self.forecast_strategy == "direct" and steps is None:
            raise ValueError("For direct forecasting, you must provide a list of steps.")

    def _select_forecast_rows(self, X, last_y_index, steps, y):
        """Select rows matching the target forecast dates when available."""
        X_aug = X[X.index > last_y_index].copy()
        forecast_history = self.y
        configuration = getattr(self, "_fitted_model_configuration", None)
        forecast_frequency = (
            configuration.data_transformation.frequency
            if configuration is not None
            else getattr(self, "_forecast_frequency", None)
        )
        if last_y_index != forecast_history.index[-1]:
            forecast_dates = self._infer_forecast_dates(
                pd.DatetimeIndex([last_y_index]),
                steps,
                frequency=forecast_frequency,
            )
        else:
            forecast_dates = self._infer_forecast_dates(
                forecast_history.index,
                steps,
                frequency=forecast_frequency,
            )
        target_rows = X_aug.reindex(forecast_dates)
        if not target_rows.isna().any().any():
            return target_rows
        return X_aug

    def _prepare_estimation_inputs(self, y, X):
        panel = y if X is None else pd.concat([y, X.reindex(y.index)], axis=1)
        if self.align_start_dates:
            complete = ~panel.isna().any(axis=1)
            if complete.any():
                first_complete = complete.idxmax()
                panel = panel.loc[first_complete:]
                y = y.loc[first_complete:]
                if X is not None:
                    X = X.loc[X.index >= first_complete]
        if not self.drop_nans and panel.isna().any().any():
            raise ValueError(
                "Linear regression requires y and X to contain no NaNs after "
                "transformation, imputation, and aligning start dates; set "
                "drop_nans=True to drop incomplete observations."
            )
        common_indices = panel.index[~panel.isna().any(axis=1)]
        return y.loc[common_indices], X.loc[common_indices] if X is not None else None

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None, **kwargs):
        """Fit OLS: y = Xβ + ε.

        Parameters
        ----------
        y : pd.DataFrame
            Single-column DataFrame with the dependent variable.
        X : pd.DataFrame, optional
            Regressor matrix (includes lag features if y_lags or X_lags > 0).

        Returns
        -------
        self
            Fitted model instance.
        """
        # Keep the estimation index for the fitted-values output.
        incoming_index = y.index
        self.last_y_fit_date = y.index[-1]
        # TODO: Allow model-specific interpolation and support NumPy and pandas inputs.

        if X is not None:
            if X.ndim == 1:
                X = X.reshape(-1, 1)
        else:
            if not self.fit_intercept:
                raise ValueError("Must provide X or set fit_intercept=True")

        # Add intercept column if needed
        if self.fit_intercept:
            n_rows = X.shape[0] if X is not None else y.shape[0]
            ones = np.ones((n_rows, 1))
            X_design = np.column_stack([ones, X]).copy() if X is not None else ones
        else:
            X_design = X.copy()

        self.N_regressors = X_design.shape[1]

        # Raw (unscaled, intercept-included) design, kept aside for computing
        # fitted values in the raw target space: self.beta_/self.betas_ are
        # always rescaled back to apply directly to this design (see
        # _rescale_beta_X and _fit_partialled).
        X_design_raw = X_design.copy()

        # X here excludes the constant.
        y_ori = y.copy()

        # Columns to exempt from regularisation and scaling: the intercept and
        # any outlier dummies. When dummies are present we fit via
        # Frisch-Waugh-Lovell so that only the penalised block is scaled and
        # regularised, while the dummies (and intercept) are left untouched.
        dummy_cols = [
            c for c in (getattr(self, "_dummy_cols", None) or []) if c in X.columns
        ]
        if dummy_cols:
            # Fit via Frisch-Waugh-Lovell so dummies (and intercept) are left
            # unpenalised and unscaled, while only the penalised block is scaled.
            col_list = list(X.columns)
            offset = 1 if self.fit_intercept else 0
            dummy_pos = [col_list.index(c) + offset for c in dummy_cols]
            unpen_idx = ([0] if self.fit_intercept else []) + dummy_pos
            pen_idx = [j for j in range(self.N_regressors) if j not in unpen_idx]
            y_in = np.asarray(y_ori, dtype=float).reshape(-1, 1)
        else:
            # No dummies: scale the whole design uniformly and rescale betas.
            X_design, y_in = self._scale(X_design, y)

        if self.forecast_strategy == "direct":
            # Fit a separate OLS model for each horizon
            self.betas_ = {}
            self.y_fits = {}
            for h in range(self.steps + 1):
                y_h = y_in[h:]
                X_h = X_design[:-h] if h > 0 else X_design
                if dummy_cols:
                    self.betas_[h] = self._fit_partialled(y_h, X_h, unpen_idx, pen_idx)
                else:
                    self.betas_[h] = self._rescale_beta_X(self._fit_reg(y=y_h, X=X_h))
                self.y_fits[h] = y_ori[h:]
            self.y_fit = self.y_fits[0]

            # Horizon-0 fitted values: h=0 uses the full (unshifted) design
            # and target, so X_design_raw/betas_[0] line up with y_fit.index.
            fitted = (X_design_raw @ self.betas_[0]).ravel()
            fitted_index = self.y_fit.index
        else:
            # Fit a single OLS model for recursive forecasting
            if dummy_cols:
                self.beta_ = self._fit_partialled(y_in, X_design, unpen_idx, pen_idx)
            else:
                self.beta_ = self._rescale_beta_X(self._fit_reg(y=y_in, X=X_design))
            self.y_fit = y_ori
            fitted = (X_design_raw @ self.beta_).ravel()
            fitted_index = self.y_fit.index

        fitted_values = pd.Series(fitted, index=fitted_index, name=self.y_fit.columns[0])
        self.fitted_values_ = fitted_values.reindex(incoming_index)

        return self

    def _forecast(
        self,
        steps: int | None = None,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        forecast_origin=None,
        **kwargs,
    ):
        """Forecast using the fitted OLS model(s).

        Parameters
        ----------
        steps : int
            Number of forecast steps.
        X : pd.DataFrame, optional
            Full augmented design matrix (includes all history and lags);
            rows after ``self.y.index[-1]`` are used for the forecast.
        y : pd.DataFrame, optional
            Target history plus conditioning paths. Default is None.
        kwargs : dict
            Additional keyword arguments (not used).

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame.
        """
        if steps is None:
            steps = self.steps

        self._validate_direct_forecast_steps(steps)

        if X is None:
            if not self.fit_intercept:
                raise ValueError("Forecast X matrix is None and fit_intercept=False")
            # Intercept-only forecast
            if self.forecast_strategy == "direct":
                intercepts = [
                    np.asarray(self.betas_[h]).reshape(-1)[0] for h in range(steps)
                ]
                forecasts = np.asarray(intercepts).reshape(-1, 1)
            else:
                intercept = np.asarray(self.beta_).reshape(-1)[0]
                forecasts = np.full((steps, 1), intercept)
            return self._wrap_forecast(forecasts, steps, forecast_origin=forecast_origin)

        # Filter X to rows after y (forecast rows only).
        last_y_index = (
            forecast_origin
            if forecast_origin is not None
            else self._fitted_model_configuration.forecast_origin
        )

        # Availability check: recursive needs `steps` future rows (one per
        # horizon); direct only ever uses the first future row (each horizon
        # has its own fitted beta for the origin-t regressors).
        X_aug = self._select_forecast_rows(X, last_y_index, steps, y)
        required_rows = 1 if self.forecast_strategy == "direct" else steps
        if X_aug.shape[0] < required_rows:
            methods = "/".join(X_IMPUTATION_METHODS)
            raise ValueError(
                f"{self.__class__.__name__}._forecast: X has {X_aug.shape[0]} "
                f"row(s) after {last_y_index}, need {required_rows}. Extend X, "
                f"set X_imputation ({methods}), or use X_steps_ahead/X_sources."
            )

        # Add intercept column if needed
        if self.fit_intercept:
            X_aug.insert(0, "intercept", 1.0)

        # X_aug and self.beta_ should have compatible shapes
        # TODO: replace this check with an exact name check
        if X_aug.shape[1] != self.N_regressors:
            raise ValueError(
                "Missing regressors during forecasting.",
                f" Expected {self.N_regressors} features, got {X_aug.shape[1]}.",
            )

        # Filter to forecast rows
        if self.forecast_strategy == "direct":
            # Direct forecasting uses only the first row of X_aug for each horizon.
            forecasts = []
            for h in range(steps):
                beta_h = self.betas_[h]
                forecast_h = _forecast_reg(beta=beta_h, X=X_aug.iloc[[0]])
                forecasts.append(forecast_h)

            # Concatenate forecasts for all horizons
            forecasts = np.concatenate(forecasts, axis=0)
        else:
            # If X contains y lags, use a recursive loop.
            # forecasts are used as regressor for the next step
            if f"{self.y_name}_lag1" in X_aug.columns:
                forecasts = []
                # count the number of lags
                n_lags = sum(
                    1 for col in X_aug.columns if col.startswith(f"{self.y_name}_lag")
                )
                # The last few values of y, newest first: recent_y[0] is y one
                # period back, recent_y[1] two periods back, and so on. Starts
                # from actual data and is updated with forecasts as we go.
                recent_y = init_recent_y(X_aug, self.y_name, n_lags)
                for step in range(steps):
                    # Prepare the row for forecasting
                    X_row = X_aug.iloc[step : step + 1].copy()
                    # Each lag column takes the value from its own period:
                    # y_lag2 is y two periods before this forecast date, not a
                    # repeat of y_lag1.
                    for lag in range(1, n_lags + 1):
                        X_row[f"{self.y_name}_lag{lag}"] = recent_y[lag - 1]

                    forecast_step = _forecast_reg(beta=self.beta_, X=X_row)
                    # Move one period forward: the new forecast becomes the
                    # most recent value and the oldest one drops off.
                    # forecast_step is a (1, 1) frame, so extract the scalar to
                    # avoid a mis-aligned assignment (-> NaN).
                    last_y = float(np.asarray(forecast_step).reshape(-1)[0])
                    recent_y = [last_y] + recent_y[:-1]
                    forecasts.append(forecast_step)
                forecasts = np.concatenate(forecasts, axis=0)
            else:
                # X_aug may extend further than requested (e.g. a ragged
                # regressor published ahead of others); only the first
                # `steps` rows are used.
                forecasts = _forecast_reg(beta=self.beta_, X=X_aug.iloc[:steps])

        return self._wrap_forecast(forecasts, steps, forecast_origin=forecast_origin)

    def _forecast_decomp(
        self,
        steps: int | None = None,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        forecast_origin=None,
        **kwargs,
    ):
        """Decompose the forecast into components (intercept + regressors + lags).

        Returns a DataFrame with one row per component per horizon, showing
        the contribution of each feature to the forecast.

        This is a **minimal decomposition contract**. RealTimeModel augments
        with metadata (variable, date, forecast_horizon, frequency, source,
        vintage_date, decomposition, revision_source, forecast_metric).

        Columns returned:
        - forecast_horizon (int): steps ahead (0..steps-1, relative position in forecast)
        - component (str): name of component (e.g. 'intercept', 'payrolls', 'y_lag1')
        - contribution (float): additive component of forecast
        - weight (float): coefficient/weight

        # TODO: Factor out the logic shared with forecast().

        Parameters
        ----------
        steps : int
            Number of steps to decompose.
        X : pd.DataFrame or np.ndarray, optional
            Full augmented design matrix (includes all history and lags).
            If DataFrame, rows after self.y.index[-1] are used for decomposition.
        y : pd.DataFrame, optional
            Unused (for API compatibility).

        Returns
        -------
        pd.DataFrame or None
            Decomposition rows (minimal contract), or None if no regressors
            (intercept-only model).
        """
        if steps is None:
            steps = self.steps

        self._validate_direct_forecast_steps(steps)

        # Add intercept column if needed
        # Filter X to rows after y (forecast rows only).
        last_y_index = (
            forecast_origin
            if forecast_origin is not None
            else y.index[-1]
            if y is not None
            else self.y_fit.index[-1]
        )

        # Add intercept column if needed
        X_aug = self._select_forecast_rows(X, last_y_index, steps, y)
        if self.fit_intercept:
            X_aug.insert(0, "intercept", 1.0)

        # X_aug and self.beta_ should have compatible shapes
        if X_aug.shape[1] != self.N_regressors:
            raise ValueError(
                "Missing regressors during forecasting.",
                f" Expected {self.N_regressors} features, got {X_aug.shape[1]}.",
            )

        # Filter to forecast rows
        if self.forecast_strategy == "direct":
            decomp = []
            for h in range(steps):
                beta_h = self.betas_[h]
                forecast_h = _forecast_decomp_reg(beta=beta_h, X=X_aug.iloc[[0]])
                # Override the horizon.
                forecast_h["forecast_horizon"] = h
                decomp.append(forecast_h)

            # Concatenate forecasts for all horizons
            decomp = pd.concat(decomp, ignore_index=True)
        else:
            # If X contains y lags, use a recursive loop.
            # forecasts are used as regressor for the next step
            if f"{self.y_name}_lag1" in X_aug.columns:
                decomp = []
                # count the number of lags
                n_lags = sum(
                    1 for col in X_aug.columns if col.startswith(f"{self.y_name}_lag")
                )
                # The last few values of y, newest first: recent_y[0] is y one
                # period back, recent_y[1] two periods back, and so on.
                recent_y = init_recent_y(X_aug, self.y_name, n_lags)
                for step in range(steps):
                    # Prepare the row for forecasting
                    X_row = X_aug.iloc[step : step + 1].copy()
                    # Each lag column takes the value from its own period.
                    for lag in range(1, n_lags + 1):
                        X_row[f"{self.y_name}_lag{lag}"] = recent_y[lag - 1]

                    forecast_step = _forecast_decomp_reg(beta=self.beta_, X=X_row)
                    forecast_step["forecast_horizon"] = step
                    # Move one period forward using this step's total forecast
                    # (the sum of the component contributions).
                    last_y = float(forecast_step["contribution"].sum())
                    recent_y = [last_y] + recent_y[:-1]
                    decomp.append(forecast_step)
                decomp = pd.concat(decomp, ignore_index=True)
            else:
                decomp = _forecast_decomp_reg(beta=self.beta_, X=X_aug.iloc[:steps])

        return decomp

    def _validate_direct_forecast_steps(self, steps):
        """Ensure direct forecasts do not exceed the fitted horizon."""
        if self.forecast_strategy == "direct" and steps > self.steps:
            raise ValueError(
                f"{self.__class__.__name__} was fitted for {self.steps} forecast "
                f"step(s), but {steps} were requested. Fit with a larger horizon."
            )

    def _fit_partialled(self, y, X_design, unpen_idx, pen_idx):
        """Fit regularised coefficients with unpenalised columns separated.

        Parameters
        ----------
        y : np.ndarray
            Target column, shape (n,) or (n, 1).
        X_design : np.ndarray
            Full design matrix (intercept + regressors + dummies).
        unpen_idx : list[int]
            Column indices to leave unpenalised and unscaled.
        pen_idx : list[int]
            Column indices to scale and regularise.

        Returns
        -------
        np.ndarray
            Full-length beta (N_regressors, 1) aligned with ``X_design``.
        """
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        U = X_design[:, unpen_idx]
        P = X_design[:, pen_idx]

        # Residualise the penalised block and y with respect to U.
        P_res = P - U @ np.linalg.lstsq(U, P, rcond=None)[0]
        y_res = y - U @ np.linalg.lstsq(U, y, rcond=None)[0]

        # Scale only the penalised (regularised) block. The mean shift is
        # harmless: it is reabsorbed by the unpenalised coefficients below.
        if self.scale and P.shape[1] > 0:
            P_mean = np.nanmean(P_res, axis=0)
            P_std = np.nanstd(P_res, axis=0)
            P_std = np.where(P_std == 0, 1.0, P_std)
            y_mean = np.nanmean(y_res, axis=0)
            y_std = np.nanstd(y_res, axis=0)
            y_std = np.where(y_std == 0, 1.0, y_std)
            P_in = (P_res - P_mean) / P_std
            y_in = (y_res - y_mean) / y_std
        else:
            P_in, y_in = P_res, y_res
            P_std = np.ones(P.shape[1])
            y_std = np.ones(1)

        # Penalised solve on the residualised (and optionally scaled) problem.
        if P.shape[1] > 0:
            beta_p_scaled = np.asarray(self._fit_reg(y=y_in, X=P_in)).reshape(-1, 1)
            beta_p = (y_std / P_std).reshape(-1, 1) * beta_p_scaled
        else:
            beta_p = np.zeros((0, 1))

        # Recover the unpenalised coefficients (intercept + dummies) by OLS on
        # the partial residual, using the raw (unscaled) penalised block.
        beta_u = np.linalg.lstsq(U, y - P @ beta_p, rcond=None)[0]

        # Reassemble into a full-length beta in the original column order.
        beta = np.zeros((X_design.shape[1], 1))
        if pen_idx:
            beta[pen_idx] = beta_p
        beta[unpen_idx] = beta_u
        return beta

    def _scale(self, X: np.ndarray, y: np.ndarray):
        """Standardise X and y using z-score normalisation.

        Stores mean and std for later rescaling during forecasting.

        # if self.intercept is True note that this function
        # uses the original X not the augmented design matrix with intercept.

        Parameters
        ----------
        X : np.ndarray
            Design matrix to scale.
        y : np.ndarray
            Target variable to scale.

        Returns
        -------
        X_scaled : np.ndarray
            Scaled design matrix.
        y_scaled : np.ndarray
            Scaled target variable.
        """

        if not self.scale:
            return X, y

        # Handle 1D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.fit_intercept:
            # Remove intercept column for scaling
            X = X[:, 1:]

        # Compute and store scaling parameters
        self._X_mean = np.nanmean(X, axis=0)
        self._X_std = np.nanstd(X, axis=0)
        self._y_mean = np.nanmean(y, axis=0)
        self._y_std = np.nanstd(y, axis=0)

        self._X_std = np.where(self._X_std == 0, 1.0, self._X_std)
        self._y_std = np.where(self._y_std == 0, 1.0, self._y_std)

        # Scale
        X_scaled = (X - self._X_mean) / self._X_std

        y_scaled = (y - self._y_mean) / self._y_std

        return X_scaled, y_scaled

    def _rescale_beta_X(self, beta: np.ndarray) -> np.ndarray:
        """Rescale beta fitted on scaled data back to original space.

        Returns a beta vector [intercept, b_1, ..., b_p] applicable
        directly to unscaled X (with a prepended intercept column),
        producing forecasts in the original y scale.
        """

        if not self.scale:
            return beta

        # beta has shape (p,) — no intercept, fit on scaled X
        beta = beta.flatten()
        beta_orig = (self._y_std / self._X_std) * beta  # (p,)
        intercept = (self._y_mean - self._X_mean @ beta_orig).item()  # scalar
        return np.concatenate([[intercept], beta_orig])  # (1+p,)


def _forecast_reg(beta, X):
    """Helper. Projection forward in a regression model."""

    forecasts = X @ beta
    return forecasts


def _forecast_decomp_reg(beta, X):
    """Decompose the forecast into components (intercept + regressors + lags).

    Returns a DataFrame with one row per component per horizon, showing
    the contribution of each feature to the forecast.

    This is a **minimal decomposition contract**. RealTimeModel augments
    with metadata (variable, date, forecast_horizon, frequency, source,
    vintage_date, decomposition, revision_source, forecast_metric).

    Columns returned:
    - forecast_horizon (int): steps ahead (0..steps-1, relative position in forecast)
    - component (str): name of component (e.g. 'intercept', 'payrolls', 'y_lag1')
    - contribution (float): additive component of forecast
    - weight (float): coefficient/weight

    Parameters
    ----------
    beta : np.ndarray
        Fitted coefficient vector (intercept first when present), shape (k, 1).
    X : pd.DataFrame or np.ndarray
        Forecast-row design matrix (one row per horizon, columns aligned
        with ``beta``).

    Returns
    -------
    pd.DataFrame or None
        Decomposition rows (minimal contract), or None if no regressors
        (intercept-only model).
    """
    # Contributions: each row × coefficients (broadcasting)
    contributions = X * beta.T

    # Get feature names
    feature_names = X.columns.tolist()

    # Flatten into rows
    rows = []
    for h in range(X.shape[0]):
        for feat_idx, feat_name in enumerate(feature_names):
            rows.append(
                {
                    "forecast_horizon": h,
                    "component": feat_name,
                    "contribution": contributions.iloc[h, feat_idx],
                    "weight": beta[feat_idx].item(),
                }
            )
    return pd.DataFrame(rows)
