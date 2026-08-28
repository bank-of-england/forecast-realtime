"""Wrapper around the bvar package for Bayesian VAR forecasting."""

from typing import Literal

import numpy as np
import pandas as pd

from forecast_realtime.forecast_model import ForecastModel


class ForecastBVAR(ForecastModel):
    """Bayesian VAR model wrapper for unconditional and conditional forecasting.

    Wraps the ``bvar`` package's ``BVAR`` class, exposing a simplified interface
    compatible with ``ForecastModel``.

    Parameters
    ----------
    stationary : bool
        If True, treat all variables as stationary. Default is True.
    forecasts_type : Literal["mean", "median"]
        How to summarise the posterior forecast distribution. Default is "mean".
    n_lags : int
        Number of lags in the VAR model. Default is 1.
    model : str
        Prior model type. Default is "natural_conjugate".
    minnesota : bool
        Use Minnesota prior. Default is True.
    soc : bool
        Use sum-of-coefficients prior. Default is True.
    sur : bool
        Use single-unit-root prior. Default is True.
    covid : bool
        Include COVID dummy variables. Default is False.
    covid_dates : list
        Dates for COVID dummies. Default is None (uses package defaults).
    optimisation_method : str
        Method for hyperparameter optimisation. Default is "ml".
    cv_options : dict | None
        Options for cross-validation (if ``optimisation_method="cross_validation"``).
    nb_restart : int
        Number of random restarts for the hyperparameter optimiser. Default is 5.
    n_samples : int
        Number of posterior draws to retain. Default is 1000.
    progressbar : bool
        Show progress bar during sampling. Default is True.
    mode_only : bool
        If True, only compute the posterior mode (fast). Default is False.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target variables.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    method : str
        Algorithm for conditional forecasting. Default is "andersson_et_al".
    N_draws : int
        Number of draws for forecast uncertainty simulation. Default is 5000.
    N_burn : int | None
        Burn-in draws to discard during forecasting. Default is ``N_draws // 2``.
    base_value : np.ndarray | None
        Base value for converting differenced forecasts back to levels.
    optim_random_state : int | None
        Random seed passed to ``BVAR.optimise_hyperparameters()``. Default is 42.
    sampling_random_state : int | None
        Random seed passed to ``BVAR.sample()``. Default is 42.
    forecast_random_state : int | None
        Random seed passed to ``BVAR.forecast()``. Default is 42.
    """

    _handles_mixed_frequencies = True

    _handles_missing_values = False

    def __init__(
        self,
        stationary: bool = True,
        forecasts_type: Literal["mean", "median"] = "mean",
        n_lags: int = 1,
        model: str = "natural_conjugate",
        minnesota: bool = True,
        soc: bool = True,
        sur: bool = True,
        covid: bool = False,
        covid_dates: list = None,
        optimisation_method: str = "ml",
        cv_options: dict | None = None,
        nb_restart: int = 5,
        n_samples: int = 1000,
        progressbar: bool = True,
        mode_only: bool = False,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        # Forecast parameters
        method: str = "andersson_et_al",
        N_draws: int = 5000,
        N_burn: int | None = None,
        base_value: np.ndarray | None = None,
        # Seeding
        optim_random_state: int | None = 42,
        sampling_random_state: int | None = 42,
        forecast_random_state: int | None = 42,
    ):
        if forecasts_type not in ("mean", "median"):
            raise ValueError(
                "forecasts_type must be 'mean' or 'median'; density forecasts "
                "are not supported."
            )

        import bvar as bv

        super().__init__(
            label=label,
            formula=formula,
            data_transformation=data_transformation,
        )

        # ── Prior model ───────────────────────────────────────────────
        if model == "independent_niw":
            sampling_model = bv.IndependentNIW(
                minnesota=minnesota,
                soc=soc,
                sur=sur,
                covid=covid,
                covid_dates=covid_dates,
            )
        else:
            sampling_model = bv.NaturalConjugate(
                minnesota=minnesota,
                soc=soc,
                sur=sur,
                covid=covid,
                covid_dates=covid_dates,
            )

        # ── BVAR instance ────────────────────────────────────────────
        self.bvar = bv.BVAR(
            n_lags=n_lags,
            model=sampling_model,
            stationary=stationary,
            optimisation_method=optimisation_method,
        )

        # ── Store parameters for fit / forecast calls ────────────────
        self.n_lags = n_lags
        self.optimisation_method = optimisation_method
        self.cv_options = cv_options
        self.nb_restart = nb_restart

        self.n_samples = n_samples
        self.progressbar = progressbar
        self.mode_only = mode_only

        self.forecasts_type = forecasts_type
        self.method = method
        self.N_draws = N_draws
        self.N_burn = N_burn if N_burn is not None else N_draws // 2
        self.base_value = base_value

        self.optim_random_state = optim_random_state
        self.sampling_random_state = sampling_random_state
        self.forecast_random_state = forecast_random_state

    # ──────────────────────────────────────────────────────────────────
    # ForecastModel interface
    # ──────────────────────────────────────────────────────────────────

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None):
        """
        Fit the BVAR model: optimise hyperparameters, then sample from the posterior.

        Parameters
        ----------
        y : pd.DataFrame
            Time series data with DatetimeIndex and variables in columns.
        X : pd.DataFrame, optional
            Not used by this model (accepted for interface compatibility).

        Returns
        -------
        self
        """
        # Optimise hyperparameters
        self.bvar.optimise_hyperparameters(
            y,
            nb_restart=self.nb_restart,
            cv_options=self.cv_options,
            random_state=self.optim_random_state,
        )

        # Sample from posterior
        self.bvar.sample(
            data=y,
            N_draws=self.n_samples,
            point_only=self.mode_only,
            progressbar=self.progressbar,
            random_state=self.sampling_random_state,
        )

        # Compute in-sample fitted values across posterior draws, shape
        # (N_draws, T_effective, n_variables), and reduce to a point estimate
        # consistent with forecasts_type.
        self.bvar.compute_fitted_values()
        fitted_draws = self.bvar.fitted_values

        if self.forecasts_type == "median":
            fitted_point = np.median(fitted_draws, axis=0)
        else:
            fitted_point = np.mean(fitted_draws, axis=0)

        self.fitted_values_ = pd.DataFrame(
            fitted_point,
            index=y.index[self.n_lags :],
            columns=y.columns,
        ).reindex(y.index)

        # store last training date for filtering conditioning data
        self.last_y_fit_date = y.index[-1]

        return self

    def _forecast(
        self,
        steps: int = 1,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        forecast_origin=None,
        **kwargs,
    ):
        """
        Produce unconditional or conditional forecasts.

        Parameters
        ----------
        steps : int
            Forecast horizon (number of steps ahead). Default is 1.
        X : np.ndarray, optional
            Exogenous regressor forecasts. Not used by BVAR.
        y : np.ndarray, shape (steps, n), optional
            Conditioning paths for the mean. NaN entries are unconstrained.
            Default is None (unconditional forecast).

        Returns
        -------
        np.ndarray
            Forecast array of shape ``(steps, n_variables)``.
        """
        if not self.bvar.is_fitted:
            raise RuntimeError(
                "Model must be fitted before forecasting. Call fit() first."
            )

        # keep only data after the last training date for conditioning
        if y is not None:
            origin = forecast_origin or self._fitted_model_configuration.forecast_origin
            y = y[y.index > origin]
            y = None if y.empty else y.to_numpy()

        self.bvar.forecast(
            H=steps,
            constraint_mean=y,
            point_only=self.mode_only,
            method=self.method,
            N_draws=self.N_draws,
            N_burn=self.N_burn,
            base_value=self.base_value,
            progressbar=False,
            random_state=self.forecast_random_state,
        )

        # Pick the right forecast array
        if y is not None:
            forecasts = self.bvar.forecast_conditional
        else:
            forecasts = self.bvar.forecast_unconditional

        # Summarise the posterior distribution
        if self.forecasts_type == "mean":
            forecasts = np.mean(forecasts, axis=0)
        else:
            forecasts = np.median(forecasts, axis=0)

        # Return only the forecast horizon rows
        return self._wrap_forecast(
            forecasts[-steps:], steps, forecast_origin=forecast_origin
        )
