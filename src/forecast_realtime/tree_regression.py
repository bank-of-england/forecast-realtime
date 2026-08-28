"""Base class for tree-based time-series regressors."""

from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from forecast_realtime._utils import init_recent_y
from forecast_realtime.forecast_model import ForecastModel


class TreeRegression(ForecastModel):
    """Base class for tree-based forecasting models.

    Provides helpers for:
        - direct and recursive forecasting
        - optional standardisation of X and y

    Subclasses must implement :meth:`_build_estimator` which returns a
    *new, unfitted* sklearn-compatible estimator (must expose ``.fit()``
    and ``.predict()``).

    Parameters
    ----------
    forecast_strategy : str
        Forecasting strategy ("recursive" or "direct"). Default "recursive".
    steps : int | None
        For direct forecasting, the maximum horizon to fit. Required when
        ``forecast_strategy="direct"``.
    standardise : bool
        Whether to standardise X and y before fitting. Default False.
    label : str | None
        Name used to identify the model's forecasts. Defaults to the class
        name.
    formula : str | None
        Optional patsy-style formula selecting the regressors. Default None.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    """

    def __init__(
        self,
        forecast_strategy: str = "recursive",
        steps: int | None = None,
        standardise: bool = False,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
    ):
        label = label if label is not None else self.__class__.__name__
        super().__init__(
            label=label,
            formula=formula,
            data_transformation=data_transformation,
        )
        self.forecast_strategy = forecast_strategy
        self.steps = steps
        self.standardise = standardise

        # populated by _fit
        self.model = None  # recursive
        self.models_ = {}  # direct (keyed by horizon h)

        # scalers
        self._X_scaler = None
        self._y_scaler = None
        self._X_scalers_ = {}
        self._y_scalers_ = {}

        if self.forecast_strategy == "direct" and steps is None:
            raise ValueError("For direct forecasting, you must provide steps.")

    # -------------------------------------------------------------- #
    # Abstract method for subclasses                                  #
    # -------------------------------------------------------------- #

    @abstractmethod
    def _build_estimator(self):
        """Return a *new, unfitted* sklearn-compatible estimator.

        The returned object must expose ``.fit(X, y)`` and
        ``.predict(X)`` methods.
        """
        ...

    # -------------------------------------------------------------- #
    # Standardisation helpers                                         #
    # -------------------------------------------------------------- #

    def _fit_standardisers(
        self, y: np.ndarray, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit scalers on training data and return standardised arrays.

        Parameters
        ----------
        y : np.ndarray, shape (n,)
        X : np.ndarray, shape (n, p)

        Returns
        -------
        y_scaled, X_scaled : tuple[np.ndarray, np.ndarray]
        """
        if not self.standardise:
            self._X_scaler = None
            self._y_scaler = None
            return y, X

        self._y_scaler = StandardScaler()
        y = self._y_scaler.fit_transform(y.reshape(-1, 1)).ravel()

        self._X_scaler = StandardScaler()
        X = self._X_scaler.fit_transform(X)

        return y, X

    def _standardise_X(self, X: np.ndarray, scaler=None) -> np.ndarray:
        """Transform X using the fitted scaler."""
        if not self.standardise:
            return X
        scaler = self._X_scaler if scaler is None else scaler
        if scaler is None:
            return X
        return scaler.transform(X)

    def _inverse_standardise_y(self, y: np.ndarray, scaler=None) -> np.ndarray:
        """Inverse-transform predictions back to the original y scale."""
        if not self.standardise:
            return y
        scaler = self._y_scaler if scaler is None else scaler
        if scaler is None:
            return y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return scaler.inverse_transform(y).ravel()

    # -------------------------------------------------------------- #
    # _fit                                                            #
    # -------------------------------------------------------------- #

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None, **kwargs):
        """Fit the tree-based model.

        Parameters
        ----------
        y : pd.DataFrame
            Single-column target variable.
        X : pd.DataFrame, optional
            Regressor matrix (includes lag features built by
            :meth:`ForecastModel.fit`).

        Returns
        -------
        self
        """
        if X is None:
            raise ValueError(
                f"{self.__class__.__name__} requires at least one feature "
                "(X or y_lags > 0)."
            )

        # Full training index as received, before any row filtering. Used at
        # the end to reindex the fitted values so dropped rows show as NaN.
        incoming_index = y.index

        # Drop NaNs
        y_idx = y.index[~y.isna().any(axis=1)]
        X_idx = X.index[~X.isna().any(axis=1)]
        common = y_idx.intersection(X_idx)
        y = y.loc[common]
        X = X.loc[common]
        if len(common):
            self.last_y_fit_date = common[-1]

        y_ori = y.copy()

        y_arr = y.iloc[:, 0].astype(float).values
        X_arr = X.astype(float).values

        if self.forecast_strategy == "direct":
            self.models_ = {}
            self.y_fits = {}
            self._X_scalers_ = {}
            self._y_scalers_ = {}
            fitted_h0 = None
            for h in range(self.steps + 1):
                y_h_raw = y_arr[h:]
                X_h_raw = X_arr[: len(y_h_raw)]
                y_h, X_h = self._fit_standardisers(y_h_raw, X_h_raw)
                self._X_scalers_[h] = self._X_scaler
                self._y_scalers_[h] = self._y_scaler
                est = self._build_estimator()
                est.fit(X_h, y_h)
                self.models_[h] = est
                self.y_fits[h] = y_ori.iloc[h:]
                if h == 0:
                    # Horizon-0 in-sample fit: X_h/y_h are the full cleaned
                    # design/target, so rows align 1:1 with self.y_fits[0].
                    preds = self._inverse_standardise_y(np.atleast_1d(est.predict(X_h)))
                    fitted_h0 = pd.Series(
                        np.asarray(preds).ravel(),
                        index=self.y_fits[0].index,
                        name=self.y_fits[0].columns[0],
                    )
            self.y_fit = self.y_fits[0]
        else:
            self._X_scalers_ = {}
            self._y_scalers_ = {}
            y_arr, X_arr = self._fit_standardisers(y_arr, X_arr)
            self.model = self._build_estimator()
            self.model.fit(X_arr, y_arr)
            self.y_fit = y_ori

            # In-sample fit on the same cleaned/standardised design used for
            # fitting, mapped back to the _fit target space.
            preds = self._inverse_standardise_y(np.atleast_1d(self.model.predict(X_arr)))
            fitted_h0 = pd.Series(
                np.asarray(preds).ravel(),
                index=self.y_fit.index,
                name=self.y_fit.columns[0],
            )

        self.fitted_values_ = fitted_h0.reindex(incoming_index)

        return self

    # -------------------------------------------------------------- #
    # _forecast                                                       #
    # -------------------------------------------------------------- #

    def _forecast(
        self,
        steps: int | None = None,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        forecast_origin=None,
        **kwargs,
    ):
        """Forecast using the fitted tree model(s).

        Parameters
        ----------
        steps : int
            Number of forecast steps.
        X : pd.DataFrame or None
            Full augmented design matrix (includes history and lags).
        y : pd.DataFrame or None
            Target history plus conditioning paths (used to determine the
            last fitted date).

        Returns
        -------
        pd.DataFrame
        """
        if steps is None:
            steps = self.steps

        self._validate_direct_forecast_steps(steps)

        if X is None:
            raise ValueError(f"{self.__class__.__name__} requires X for forecasting.")

        last_y_index = (
            forecast_origin
            if forecast_origin is not None
            else getattr(self, "last_y_fit_date", self.y_fit.index[-1])
        )

        if self.forecast_strategy == "direct":
            return self._forecast_direct(steps, X, last_y_index, forecast_origin)
        return self._forecast_recursive(steps, X, last_y_index, forecast_origin)

    def _forecast_direct(self, steps, X, last_y_index, forecast_origin):
        """Direct forecasting: use last observed X row with horizon-specific models."""
        # "Now" is the first row after the fitted anchor (a ragged-edge X may
        # lead the target); only fall back to the last historical row when X
        # does not extend past the anchor at all.
        X_future = X[X.index > last_y_index]
        X_last = (
            X_future.iloc[[0]].copy()
            if not X_future.empty
            else X[X.index <= last_y_index].iloc[[-1]].copy()
        )

        forecasts = []
        for h in range(steps):
            est = self.models_[h]
            X_last_arr = self._standardise_X(
                X_last.astype(float).values, scaler=self._X_scalers_[h]
            )
            pred = est.predict(X_last_arr)
            pred = self._inverse_standardise_y(
                np.atleast_1d(pred), scaler=self._y_scalers_[h]
            )
            forecasts.append(pred[0])

        return self._wrap_forecast(
            np.array(forecasts).reshape(-1, 1), steps, forecast_origin=forecast_origin
        )

    def _validate_direct_forecast_steps(self, steps):
        """Ensure direct forecasts do not exceed the fitted horizon."""
        if self.forecast_strategy == "direct" and steps > self.steps:
            raise ValueError(
                f"{self.__class__.__name__} was fitted for {self.steps} forecast "
                f"step(s), but {steps} were requested. Fit with a larger horizon."
            )

    def _forecast_recursive(self, steps, X, last_y_index, forecast_origin):
        """Recursive forecasting: step forward, updating y lags each step."""
        X_aug = X[X.index > last_y_index].copy()

        # Pad if fewer future X rows than steps
        if len(X_aug) < steps:
            if X_aug.empty:
                X_hist = X[X.index <= last_y_index]
                base_row = X_hist.iloc[[-1]] if not X_hist.empty else X.iloc[[-1]]
            else:
                base_row = X_aug.iloc[[-1]]
            n_extra = steps - len(X_aug)
            X_aug = pd.concat([X_aug] + [base_row] * n_extra, ignore_index=True)

        y_name = getattr(self, "y_name", None)
        has_y_lags = y_name and f"{y_name}_lag1" in X_aug.columns

        if has_y_lags:
            n_lags = sum(1 for col in X_aug.columns if col.startswith(f"{y_name}_lag"))
            # The last few values of y, newest first: recent_y[0] is y one
            # period back, recent_y[1] two periods back, and so on. Starts from
            # actual data and is updated with forecasts as we go.
            recent_y = init_recent_y(X_aug, y_name, n_lags)
            forecasts = []
            for step in range(steps):
                X_row = X_aug.iloc[step : step + 1].copy()
                # Each lag column takes the value from its own period: y_lag2
                # is y two periods before this forecast date, not a repeat of
                # y_lag1.
                for lag in range(1, n_lags + 1):
                    X_row[f"{y_name}_lag{lag}"] = recent_y[lag - 1]
                X_row_arr = self._standardise_X(X_row.astype(float).values)
                pred = self.model.predict(X_row_arr)
                pred = self._inverse_standardise_y(np.atleast_1d(pred))
                last_pred = float(pred[0])
                # Move one period forward: the new forecast becomes the most
                # recent value and the oldest one drops off.
                recent_y = [last_pred] + recent_y[:-1]
                forecasts.append(last_pred)
            forecasts = np.array(forecasts).reshape(-1, 1)
        else:
            X_arr = self._standardise_X(X_aug.iloc[:steps].astype(float).values)
            forecasts = self.model.predict(X_arr)
            forecasts = self._inverse_standardise_y(np.atleast_1d(forecasts))
            forecasts = forecasts.reshape(-1, 1)

        return self._wrap_forecast(forecasts, steps, forecast_origin=forecast_origin)
