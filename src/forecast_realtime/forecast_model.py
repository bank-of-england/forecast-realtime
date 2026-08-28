"""Core classes and contracts for forecast models."""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecast_realtime._utils import (
    build_dummies,
    build_lagged_design,
    impute_X,
    regularise_missing_rows,
)
from forecast_realtime.data_transformation import (
    DataTransformationPipeline,
    FittedDataTransformation,
    _validate_mapping_coverage,
    _validate_metric_mapping,
    combine_history_and_future,
    infer_frequency_from_dates,
    infer_variable_frequencies,
    leading_nan_row_count,
)
from forecast_realtime.formula import Formula

# Single source of truth for RealTimeModel's X_imputation strategies, shared
# with the validation check in RealTimeModel.forecast() and the error below.
X_IMPUTATION_METHODS = ("zero", "last", "mean", "ar1_t")


class NoUsableTransformedYError(ValueError):
    """Raised when input preparation leaves no usable transformed target rows."""


@dataclass(frozen=True)
class ForecastContext:
    """Raw fitted history and future paths for one prediction request."""

    y_history: pd.DataFrame
    X_history: pd.DataFrame | None
    y_conditioning: pd.DataFrame | None = None
    X_conditioning: pd.DataFrame | None = None
    forecast_origin: pd.Timestamp | None = None
    y_conditioning_input_metrics: dict[str, str] | None = None
    X_conditioning_input_metrics: dict[str, str] | None = None


@dataclass(frozen=True)
class FittedModelConfiguration:
    """Immutable configuration captured when a model is successfully fitted."""

    data_transformation: FittedDataTransformation
    y_columns: tuple[str, ...]
    X_columns: tuple[str, ...] | None
    y_lags: int
    X_lags: int | tuple[tuple[str, int], ...]
    dummies: tuple | tuple[tuple[str, object], ...] | None
    dummy_definitions: tuple[tuple[str, object], ...] | None
    dummy_columns: tuple[str, ...]
    forecast_origin: pd.Timestamp
    drop_transformation_nans: bool


@dataclass(frozen=True)
class _FitDesignState:
    """Design data and compatibility metadata produced during fitting."""

    y_history: pd.DataFrame
    X_history: pd.DataFrame | None
    y_name: str
    X_names: list[str] | None
    dummies: list | dict | None
    dummy_definitions: dict | None
    dummy_columns: list[str]
    last_y_fit_date: pd.Timestamp | pd.Period


class ForecastResult(pd.DataFrame):
    """DataFrame-compatible forecast with explicit prediction metadata."""

    _metadata = ["decomposition", "forecast_origin"]

    @property
    def _constructor(self):
        return ForecastResult

    @property
    def forecast(self) -> pd.DataFrame:
        """Return the forecast values without result metadata."""
        return pd.DataFrame(self)

    def __init__(
        self,
        forecast=None,
        decomposition: pd.DataFrame | None = None,
        forecast_origin: pd.Timestamp | None = None,
        steps: int | None = None,
        expected_columns: list[str] | None = None,
        forecast_dates_include_origin: bool = False,
    ):
        super().__init__(forecast)
        if steps is not None or expected_columns is not None:
            if steps is None or expected_columns is None or forecast_origin is None:
                raise ValueError(
                    "steps, expected_columns, and forecast_origin must be provided "
                    "together to validate a ForecastResult."
                )
            self._validate_forecast(
                steps,
                expected_columns,
                forecast_origin,
                forecast_dates_include_origin,
            )
            decomposition = self._validate_decomposition(
                decomposition,
                steps,
                expected_columns,
            )
        self.decomposition = decomposition
        self.forecast_origin = forecast_origin

    def _validate_forecast(
        self,
        steps: int,
        expected_columns: list[str],
        forecast_origin: pd.Timestamp,
        forecast_dates_include_origin: bool,
    ) -> None:
        """Validate the shape, target and date contract of this forecast."""
        if len(self) != steps:
            raise ValueError(f"Forecast must have {steps} rows, got {len(self)}")
        if list(self.columns) != expected_columns:
            raise ValueError(
                "Forecast columns must match the fitted target columns in order; "
                f"expected {expected_columns}, got {list(self.columns)}"
            )
        if not isinstance(self.index, pd.DatetimeIndex):
            raise TypeError("Forecast must be indexed by a DatetimeIndex.")
        if self.index.hasnans:
            raise ValueError("Forecast index must not contain missing dates.")
        if self.index.has_duplicates:
            raise ValueError("Forecast index must not contain duplicate dates.")
        if not self.index.is_monotonic_increasing:
            raise ValueError("Forecast index must be sorted in increasing order.")

        origin = pd.Timestamp(forecast_origin)
        invalid_dates = (
            self.index < origin if forecast_dates_include_origin else self.index <= origin
        )
        if invalid_dates.any():
            raise ValueError(
                "Forecast dates do not satisfy their declared relationship to the "
                "fitted forecast origin; ordinary forecasts must be strictly "
                "after that origin."
            )

    def _validate_decomposition(
        self,
        decomposition: pd.DataFrame | None,
        steps: int,
        expected_columns: list[str],
    ) -> pd.DataFrame | None:
        """Validate and normalise a model-local additive decomposition."""
        if decomposition is None:
            return None
        if not isinstance(decomposition, pd.DataFrame):
            raise TypeError("decomposition must be a pandas DataFrame or None.")

        required_columns = {
            "forecast_horizon",
            "component",
            "contribution",
            "weight",
        }
        missing_columns = required_columns - set(decomposition.columns)
        if missing_columns:
            raise ValueError(
                f"decomposition is missing required columns: {sorted(missing_columns)}"
            )
        if decomposition.empty:
            raise ValueError("decomposition must contain at least one row.")

        result = decomposition.copy()
        horizon = result["forecast_horizon"]
        if horizon.isna().any() or not pd.api.types.is_integer_dtype(horizon):
            raise TypeError("decomposition forecast_horizon must be integers.")
        if ((horizon < 0) | (horizon >= steps)).any():
            raise ValueError(
                f"decomposition forecast_horizon must be in the range 0..{steps - 1}."
            )

        for column in ("contribution", "weight"):
            if column not in result:
                continue
            values = result[column]
            numeric = pd.to_numeric(values, errors="coerce")
            if column == "contribution" and numeric.isna().any():
                raise TypeError("decomposition contribution values must be numeric.")
            if (
                column == "weight"
                and values.notna().any()
                and numeric[values.notna()].isna().any()
            ):
                raise TypeError("decomposition weight values must be numeric.")
            if np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
                raise ValueError(f"decomposition {column} values must be finite.")
            if column == "contribution":
                result[column] = numeric
            elif values.notna().any():
                result.loc[values.notna(), column] = numeric[values.notna()]

        for column in ("component", "variable"):
            if column not in result:
                continue
            if (
                result[column].isna().any()
                or not result[column].map(lambda value: isinstance(value, str)).all()
            ):
                raise TypeError(
                    f"decomposition {column} values must be non-missing strings."
                )

        if "variable" not in result:
            if len(expected_columns) != 1:
                raise ValueError(
                    "Multi-target decompositions must include a 'variable' column."
                )
            variables = pd.Series(expected_columns[0], index=result.index)
        else:
            variables = result["variable"]
        if not set(variables).issubset(expected_columns):
            unknown = sorted(set(variables) - set(expected_columns))
            raise ValueError(
                f"decomposition contains unknown target variable(s): {unknown}"
            )

        decomposition_keys = pd.DataFrame(
            {
                "forecast_horizon": result["forecast_horizon"],
                "variable": variables,
                "component": result["component"],
            }
        )
        duplicate_keys = decomposition_keys.duplicated()
        if duplicate_keys.any():
            raise ValueError(
                "decomposition must contain at most one contribution per "
                "forecast_horizon, variable, and component."
            )

        expected_pairs = pd.MultiIndex.from_product(
            [range(steps), expected_columns],
            names=["forecast_horizon", "variable"],
        )
        actual_pairs = pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "forecast_horizon": result["forecast_horizon"],
                    "variable": variables,
                }
            ).drop_duplicates()
        )
        if not expected_pairs.isin(actual_pairs).all():
            raise ValueError(
                "decomposition must reconcile every forecast horizon for every "
                "target variable."
            )

        totals = (
            result.assign(_variable=variables)
            .groupby(["forecast_horizon", "_variable"])["contribution"]
            .sum()
        )
        forecast_values = self.to_numpy(dtype=float)
        for horizon_value in range(steps):
            for variable_index, variable in enumerate(expected_columns):
                total = totals.loc[(horizon_value, variable)]
                forecast_value = forecast_values[horizon_value, variable_index]
                if not np.isclose(total, forecast_value, equal_nan=True):
                    raise ValueError(
                        "decomposition contributions must reconcile to the "
                        "forecast for every horizon and target variable."
                    )
        return result


_FITTED_VALUES_NOT_FITTED_MSG = (
    "fitted_values are not available; the model has not been fitted yet "
    "(call fit() first)."
)


def _validate_X_imputation(value: str | None) -> None:
    if value is not None and value not in X_IMPUTATION_METHODS:
        raise ValueError(
            f"X_imputation must be None or one of {X_IMPUTATION_METHODS}; got {value!r}"
        )


def _coerce_data_transformation(
    value: dict[str, str] | None,
) -> DataTransformationPipeline | None:
    """Normalise a call-level ``data_transformation`` fallback argument.

    Mirrors ``ForecastModel.data_transformation``'s own setter
    validation for a call-level mapping.
    """
    if value is None:
        return value
    if isinstance(value, dict):
        return DataTransformationPipeline(value)
    raise TypeError(
        "data_transformation must be None or a dict[str, str] mapping; "
        f"got {type(value).__name__}."
    )


def _restrict_mapping(value: dict | None, columns) -> dict | None:
    """Return mapping entries belonging to the selected input columns."""
    if value is None:
        return None
    return {column: value[column] for column in columns if column in value}


def _freeze_option(value):
    """Copy the small option structures that form part of fitted state."""
    if isinstance(value, dict):
        return tuple((key, _freeze_option(item)) for key, item in value.items())
    if isinstance(value, list):
        return tuple(_freeze_option(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_option(item) for item in value)
    return value


def _thaw_option(value):
    """Recreate a mutable option structure for an internal model hook."""
    if isinstance(value, tuple):
        if value and all(isinstance(item, tuple) and len(item) == 2 for item in value):
            return {key: _thaw_option(item) for key, item in value}
        return [_thaw_option(item) for item in value]
    return value


def _validate_fitted_override(name: str, supplied, fitted) -> None:
    """Reject a forecast preprocessing option that differs from fit-time state."""
    if supplied is not None and supplied != fitted:
        raise ValueError(
            f"{name} conflicts with the preprocessing configuration used by fit()"
        )


class ForecastModel(ABC):
    """Abstract base class for forecast models.

    Subclasses implement ``_fit`` and ``_forecast``; the public ``fit`` and
    ``forecast`` handle validation, lag/dummy construction and output shaping.

    Forecast date contract
    ~~~~~~~~~~~~~~~~~~~~~~
    ``_forecast`` may return an ``(steps, n_vars)`` array-like, which
    :meth:`_wrap_forecast` labels with the next ``steps`` periods after the
    effective fitting origin. Models anchored elsewhere must return a DataFrame
    with their own ``DatetimeIndex`` whose dates are strictly after that origin.
    ``RealTimeModel`` derives horizons from these dates, so wrong dates give
    wrong horizons.

    Data transformation configuration
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``data_transformation`` is an optional, model-owned mapping
    (constructor argument or assignable property). ``RealTimeModel.forecast()``
    resolves one transformation per model, preferring this setting when set
    and falling back to the call-level mapping otherwise.

    Ragged-edge regressor handling
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``_needs_ragged_edge_imputation`` defaults to ``True``. When it is ``True``,
    the ``X_imputation`` option supplied to ``fit()`` or ``forecast()`` can be
    applied to missing future regressor values. Models that handle their own
    ragged edge should set it to ``False``; the framework then leaves ``X``
    unchanged and does not apply ``X_imputation``.
    """

    # Most models need RealTimeModel to pad ragged X data before fitting.
    _needs_ragged_edge_imputation: bool = True

    _forecast_dates_include_origin: bool = False

    # True means the model owns the treatment of NaNs passed to _fit(). Models
    # that require complete estimation data opt out so RealTimeModel aligns y
    # and X on complete-case rows before fitting.
    _handles_missing_values: bool = True

    _handles_mixed_frequencies: bool = True

    _supports_multivariate_y: bool = True

    def __init__(
        self,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        align_start_dates: bool = False,
    ):
        """
        Args:
            label : str, optional
                Name used by RealTimeModel to tag forecasts. Defaults to the
                class name.
            formula : str, optional
                R-style variable selection, e.g. "cpisa ~ gdpkp + unemp".
                Defaults to using all y and X variables.
            data_transformation : dict, optional
                Optional model-owned mapping from variable to required metric.
                When set, ``RealTimeModel.forecast()`` uses this transformation
                for the model instead of the call-level mapping.
        """
        self.label = label if label is not None else self.__class__.__name__
        self._formula = Formula(formula) if formula else None
        self.fitted_values_ = None
        self._is_fitted = False
        self.data_transformation = data_transformation
        self.align_start_dates = align_start_dates

    @property
    def data_transformation(self) -> dict[str, str] | None:
        """Optional model-owned variable-to-metric transformation mapping.

        Resolved per-model by ``RealTimeModel.forecast()``, which prefers this
        setting when set and falls back to its call-level mapping.
        """
        mapping = getattr(self, "_data_transformation", None)
        return None if mapping is None else dict(mapping)

    @data_transformation.setter
    def data_transformation(self, value):
        self._data_transformation = _validate_metric_mapping(value, "data_transformation")

    def resolve_target_variables(self, y_variables: list[str]) -> list[str]:
        """Return the requested variables this model treats as targets."""
        if self._formula is not None:
            return list(self._formula.y_cols)
        return list(y_variables)

    def resolve_input_data_transformation(
        self,
        data_transformation: dict[str, str] | None = None,
        *,
        y_variables: list[str] | None = None,
        X_variables: list[str] | None = None,
    ) -> DataTransformationPipeline | None:
        """Resolve the model input pipeline, preserving ``None`` to represent
        the identity transformation.

        A model-owned pipeline takes precedence over the call-level fallback.
        When neither is configured, raw levels are the identity input. If
        variables are supplied, the resolved mapping is validated against the
        model input roles using the same coverage rules as the pipeline.
        """
        mapping = self.data_transformation
        pipeline = (
            DataTransformationPipeline(mapping)
            if mapping is not None
            else _coerce_data_transformation(data_transformation)
        )

        if pipeline is not None and y_variables is not None:
            formula = getattr(self, "_formula", None)
            if formula is not None:
                y_variables = list(formula.y_cols)
                if not formula.has_wildcard:
                    X_variables = [
                        variable
                        for variable in (X_variables or [])
                        if variable in formula.X_cols
                    ]
            _validate_mapping_coverage(
                pipeline.data_transformation,
                y_variables,
                X_variables,
            )
        return pipeline

    def native_metric_mapping(
        self, target_variables: list[str] | None = None
    ) -> dict[str, str]:
        """Return the metric space used by the fitted target outputs."""
        target_variables = list(target_variables or self.y.columns)
        configuration = self._fitted_model_configuration
        mapping = dict(configuration.data_transformation.data_transformation or {})
        return {
            variable: mapping.get(variable, "levels") for variable in target_variables
        }

    @staticmethod
    def _drop_missing_estimation_rows(y, X):
        """Return complete-case estimation inputs for a NaN-intolerant model."""
        X_on_y = X.reindex(y.index) if X is not None else None
        estimation_panel = y if X_on_y is None else pd.concat([y, X_on_y], axis=1)
        complete_index = estimation_panel.index[~estimation_panel.isna().any(axis=1)]

        if complete_index.empty:
            raise ValueError(
                "No complete observations remain after aligning y and X for a "
                "model that does not handle missing values."
            )

        y_complete = y.loc[complete_index]
        if X is None:
            return y_complete, None

        return y_complete, X.loc[complete_index]

    @property
    def fitted_values(self) -> pd.Series | pd.DataFrame:
        """In-sample fitted values produced during ``fit()``.

        Returns
        -------
        pd.Series | pd.DataFrame
            The in-sample fitted values stored on ``self.fitted_values_``
            by the subclass's ``_fit()`` implementation.

        Raises
        ------
        AttributeError
            If the model has not been fitted yet (or the subclass does
            not populate ``fitted_values_``).
        """
        if self.fitted_values_ is None:
            raise AttributeError(_FITTED_VALUES_NOT_FITTED_MSG)
        return self.fitted_values_

    @abstractmethod
    def _fit(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame | None = None,
        **kwargs,
    ):
        """
        Internal fit implementation. Subclasses must override this.

        Args:
            y : pd.DataFrame
                Targets, indexed by a DatetimeIndex.
            X : pd.DataFrame, optional
                Design matrix (lags and dummies already built by ``fit``).
            **kwargs
                Model-specific arguments.

        Returns:
            self (required).
        """
        ...

    @abstractmethod
    def _forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Internal forecast implementation. Subclasses must override this.

        Args:
            steps : int
                Number of steps ahead to forecast.
            X : pd.DataFrame, optional
                Design matrix over history plus the forecast horizon.
            y : pd.DataFrame, optional
                Conditioning paths for y, shape (steps, n_y_vars); NaN entries
                are unconstrained.
            **kwargs
                Model-specific arguments.

        Returns:
            pd.DataFrame or array-like
                Forecasts for ``steps`` periods, one column per target. A plain
                ``(steps, n_vars)`` array is enough - ``forecast()`` attaches
                the standard dates. Models forecasting other periods must return
                a DataFrame with their own ``DatetimeIndex``, which is passed
                through untouched. See the class docstring.
        """
        ...

    def _forecast_decomp(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Optional hook returning the additive components of the latest forecast.

        Subclasses that support decomposition return one row per step per
        component with columns:

        - ``forecast_horizon`` (int): 0..steps-1
        - ``component`` (str): e.g. 'intercept', 'x1_lag1', 'residual'
        - ``contribution`` (float): additive contribution to the forecast
        - ``weight`` (float, nullable): coefficient, linear models only

        RealTimeModel adds the remaining metadata (variable, date, frequency,
        source, vintage dates, decomposition type, revision source,
        forecast_metric) when writing the ``decompositions`` table.

        Args:
            steps : int
                Number of steps ahead being forecast.
            X : pd.DataFrame, optional
                Design matrix, as passed to ``_forecast``.
            y : pd.DataFrame, optional
                Conditioning paths, as passed to ``_forecast``.
            **kwargs
                Model-specific arguments.

        Returns:
            pd.DataFrame or None
                None if the model does not support decomposition (the default,
                e.g. black-box external models).
        """
        return None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _prepare_fit_inputs(
        self, y: pd.DataFrame, X: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply model-specific preparation before design construction."""
        return y, X

    def _prepare_forecast_inputs(
        self, y: pd.DataFrame | None, X: pd.DataFrame | None
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Apply model-specific preparation before forecast design selection."""
        return y, X

    def _prepare_estimation_inputs(
        self, y: pd.DataFrame, X: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply model-specific filtering after formula selection."""
        return y, X

    def _finalise_forecast(
        self,
        forecast,
        steps: int,
        forecast_origin: pd.Timestamp,
        decomp: bool = False,
        decomp_kwargs: dict | None = None,
    ) -> ForecastResult:
        """Validate, normalise, and wrap a model forecast result."""
        if forecast is None:
            raise TypeError(
                f"{self.__class__.__name__}._forecast returned None; it must "
                f"return forecasts for {steps} step(s)."
            )

        if not (
            isinstance(forecast, pd.DataFrame)
            and isinstance(forecast.index, pd.DatetimeIndex)
        ):
            forecast = self._wrap_forecast(
                forecast,
                steps,
                forecast_origin=forecast_origin,
            )

        if forecast.index.name != "date":
            forecast = forecast.copy()
            forecast.index.name = "date"

        decomposition = None
        if decomp:
            decomposition = self._forecast_decomp(
                steps=steps,
                **(decomp_kwargs or {}),
            )

        configuration = getattr(self, "_fitted_model_configuration", None)
        expected_columns = (
            list(configuration.y_columns)
            if configuration is not None
            else list(self.y.columns)
        )
        return ForecastResult(
            forecast,
            decomposition=decomposition,
            forecast_origin=forecast_origin,
            steps=steps,
            expected_columns=expected_columns,
            forecast_dates_include_origin=self._forecast_dates_include_origin,
        )

    @staticmethod
    def _validate_datetime_frame(frame: pd.DataFrame, role: str) -> None:
        if not isinstance(frame.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            raise TypeError(f"{role} must be indexed by a DatetimeIndex.")
        if frame.index.has_duplicates:
            raise ValueError(f"{role} index must not contain duplicate dates.")
        if not frame.index.is_monotonic_increasing:
            raise ValueError(f"{role} index must be sorted in increasing order.")

    @staticmethod
    def _dummy_spec(dummies: list | dict, columns: list[str]) -> dict:
        if isinstance(dummies, dict):
            return dict(dummies)
        return dict(zip(columns, dummies, strict=True))

    @classmethod
    def _infer_forecast_dates(
        cls,
        y_index: pd.DatetimeIndex | pd.PeriodIndex,
        steps: int,
        frequency: str | None = None,
        start: pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        """Build ``steps`` consecutive dates after the selected start.

        ``frequency`` is the resolved frequency of the forecast horizon.
        """
        if len(y_index) == 0:
            raise ValueError("Cannot infer forecast dates from empty y_index.")
        start = y_index[-1] if start is None else pd.Timestamp(start)
        if isinstance(y_index, pd.PeriodIndex):
            period = pd.Period(start, freq=frequency)
            periods = pd.PeriodIndex(period + np.arange(1, steps + 1))
            return periods.to_timestamp(how="end").normalize()
        if frequency in ("M", "Q"):
            anchor = "start" if y_index.is_month_start.all() else "end"
            period = pd.Period(start, freq=frequency)
            dates = [period + i for i in range(1, steps + 1)]
            return pd.DatetimeIndex(
                [date.to_timestamp(how=anchor).normalize() for date in dates]
            )
        offset = pd.tseries.frequencies.to_offset(frequency)
        return pd.DatetimeIndex([start + (i + 1) * offset for i in range(steps)])

    def _wrap_forecast(
        self,
        arr: np.ndarray,
        steps: int,
        forecast_origin: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Wrap an ``(steps, n_vars)`` array-like into the standard DataFrame,
        using :meth:`_infer_forecast_dates` for the index and ``self.y.columns``
        for the columns.

        ``forecast()`` applies this automatically, so models rarely call it.
        Dates are anchored to ``last_y_fit_date`` unless an explicit origin is
        supplied by the prediction context.
        """
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        configuration = getattr(self, "_fitted_model_configuration", None)
        expected_columns = (
            list(configuration.y_columns)
            if configuration is not None
            else list(self.y.columns)
        )
        n_cols = len(expected_columns)
        if arr.shape != (steps, n_cols):
            methods = "/".join(X_IMPUTATION_METHODS)
            raise ValueError(
                f"{self.__class__.__name__}._forecast returned array of shape "
                f"{arr.shape}, expected ({steps}, {n_cols}). If rows are short, "
                f"X may not extend past {self.last_y_fit_date} - extend X, set "
                f"X_imputation ({methods}), or use X_steps_ahead/X_sources."
            )
        forecast_origin = (
            (
                configuration.forecast_origin
                if configuration is not None
                else self.last_y_fit_date
            )
            if forecast_origin is None
            else forecast_origin
        )
        forecast_frequency = (
            configuration.data_transformation.frequency
            if configuration is not None
            else getattr(self, "_forecast_frequency", None)
        )
        dates = self._infer_forecast_dates(
            self.y.index,
            steps,
            frequency=forecast_frequency,
            start=forecast_origin,
        )

        # TODO: Decide whether forecast dates should come from self.y_forecast.index.
        return pd.DataFrame(arr, index=dates, columns=expected_columns)

    def fit(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame | None = None,
        y_lags: int = 0,
        X_lags: int | dict = 0,
        dummies: list | dict | None = None,
        data_transformation: dict[str, str] | None = None,
        frequency: str | None = None,
        X_imputation: str | None = None,
        input_frequencies: dict[str, str] | None = None,
        y_input_metrics: dict[str, str] | None = None,
        X_input_metrics: dict[str, str] | None = None,
        drop_transformation_nans: bool = True,
        **kwargs,
    ):
        """Fit the model and return ``self`` after successful estimation."""
        candidate = copy.deepcopy(self)
        candidate._fit_impl(
            y=y,
            X=X,
            y_lags=y_lags,
            X_lags=X_lags,
            dummies=dummies,
            data_transformation=data_transformation,
            frequency=frequency,
            X_imputation=X_imputation,
            input_frequencies=input_frequencies,
            y_input_metrics=y_input_metrics,
            X_input_metrics=X_input_metrics,
            drop_transformation_nans=drop_transformation_nans,
            **kwargs,
        )
        self._commit_fit(candidate)
        return self

    def _commit_fit(self, candidate: "ForecastModel") -> None:
        """Publish a successfully fitted candidate on this model instance."""
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    def _validate_fit_inputs(
        self, y: pd.DataFrame, X: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Validate, formula-select, and copy the raw frames used by fitting."""
        if not isinstance(y, pd.DataFrame):
            raise TypeError("y must be a pandas DataFrame")
        if y.empty:
            raise ValueError("y must not be empty")
        self._validate_datetime_frame(y, "y")

        if X is not None:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("X must be a pandas DataFrame or None")
            if X.empty:
                raise ValueError("X must not be empty if provided")
            self._validate_datetime_frame(X, "X")

        if getattr(self, "_formula", None):
            y, X = self._formula.extract_available_inputs(y, X)

        if not self._supports_multivariate_y and y.shape[1] > 1:
            raise ValueError(
                f"{self.__class__.__name__} cannot handle multiple left-hand-side "
                f"variables; select one variable in `forecast(y_variables=)` or use "
                f"the formula argument of {self.__class__.__name__}"
            )

        return y.copy(), X.copy() if X is not None else None

    def _resolve_fit_transformation(
        self,
        raw_y: pd.DataFrame,
        raw_X: pd.DataFrame | None,
        data_transformation: dict[str, str] | None,
        frequency: str | None,
        X_imputation: str | None,
        input_frequencies: dict[str, str] | None,
        y_input_metrics: dict[str, str] | None,
        X_input_metrics: dict[str, str] | None,
    ) -> FittedDataTransformation:
        """Resolve the transformation and its fit-time frequency metadata."""
        pipeline = self.resolve_input_data_transformation(
            data_transformation,
            y_variables=list(raw_y.columns),
            X_variables=list(raw_X.columns) if raw_X is not None else None,
        )
        y_frequency_map = {
            variable: input_frequencies[variable] for variable in raw_y.columns
        }
        X_frequency_map = (
            {variable: input_frequencies[variable] for variable in raw_X.columns}
            if raw_X is not None
            else {}
        )
        y_frequency_values = {
            y_frequency_map[variable]
            for variable in raw_y.columns
            if y_frequency_map[variable] is not None
        }
        y_frequency = (
            next(iter(y_frequency_values)) if len(y_frequency_values) == 1 else None
        )
        target_frequency = frequency or y_frequency
        if raw_X is not None and not self._handles_mixed_frequencies:
            X_frequency_variables = list(raw_X.columns)
            if getattr(self, "_formula", None) and not self._formula.has_wildcard:
                X_frequency_variables = [
                    variable
                    for variable in self._formula.X_cols
                    if variable in raw_X.columns
                ]
            X_frequency_values = {
                X_frequency_map[variable] for variable in X_frequency_variables
            }
            if (
                y_frequency is not None
                and X_frequency_values
                and (X_frequency_values != {y_frequency})
            ):
                raise ValueError(
                    f"{self.__class__.__name__} does not support mixed frequencies: "
                    f"y is {y_frequency!r}, X has {sorted(X_frequency_values)!r}."
                )
        return FittedDataTransformation.from_fit(
            pipeline,
            y_variables=list(raw_y.columns),
            X_variables=list(raw_X.columns) if raw_X is not None else None,
            y_input_metrics=y_input_metrics,
            X_input_metrics=X_input_metrics,
            y_frequencies=y_frequency_map,
            X_frequencies=X_frequency_map,
            frequency=target_frequency,
            X_imputation=X_imputation,
            pipeline_source=(
                "model"
                if self.data_transformation is not None
                else "fallback"
                if data_transformation is not None
                else "identity"
            ),
        )

    def _prepare_training_data(
        self,
        raw_y: pd.DataFrame,
        raw_X: pd.DataFrame | None,
        transformation: FittedDataTransformation,
        X_imputation: str | None,
        drop_transformation_nans: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Transform, impute, regularise, and model-prepare training data."""
        pipeline = transformation.pipeline
        y, X = transformation.transform_fit_inputs(raw_y, raw_X)
        if pipeline is not None:
            y_variables = list(transformation.y_variables)
            X_variables = (
                list(transformation.X_variables)
                if transformation.X_variables is not None
                else None
            )
            if drop_transformation_nans:
                y_leading_rows = leading_nan_row_count(y, y_variables)
                if y_leading_rows:
                    y = y.iloc[y_leading_rows:]
                if X is not None:
                    X_leading_rows = leading_nan_row_count(X, X_variables)
                    if X_leading_rows:
                        X = X.iloc[X_leading_rows:]
                if y.empty or y.dropna(how="all").empty:
                    raise NoUsableTransformedYError(
                        "No usable transformed y observations remain after "
                        "transformation."
                    )

        if (
            X is not None
            and X_imputation is not None
            and self._needs_ragged_edge_imputation
        ):
            last_valid_dates = [X[col].last_valid_index() for col in X.columns]
            last_valid_dates = [date for date in last_valid_dates if date is not None]
            target_date = max(last_valid_dates + [y.index[-1]])
            X = impute_X(
                X,
                target_date,
                steps=0,
                method=X_imputation,
                frequencies=transformation.X_frequency_mapping,
            )

        if (
            not self._handles_missing_values
            or transformation.y_frequency_mapping
            or transformation.X_frequency_mapping
        ):
            y = regularise_missing_rows(y, transformation.y_frequency_mapping)
            if X is not None:
                X = regularise_missing_rows(X, transformation.X_frequency_mapping)

        return self._prepare_fit_inputs(y, X)

    def _build_fit_design(
        self,
        prepared_y: pd.DataFrame,
        prepared_X: pd.DataFrame | None,
        y_lags: int,
        X_lags: int | dict,
        dummies: list | dict | None,
        target_frequency: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, _FitDesignState]:
        """Build the estimation design and fit the model."""
        if prepared_y.empty or prepared_y.dropna(how="all").empty:
            raise NoUsableTransformedYError(
                "No usable transformed y observations remain after model preparation."
            )

        self.X_lags = X_lags
        self.y_lags = y_lags
        self.X_names = list(prepared_X.columns) if prepared_X is not None else None
        y_fit = (
            self._formula.extract_y(prepared_y)
            if getattr(self, "_formula", None)
            else prepared_y
        )
        self.y_name = list(y_fit.columns)[0]
        self._prepared_y_history = y_fit.copy()
        self._prepared_X_history = prepared_X.copy() if prepared_X is not None else None

        if y_lags or X_lags:
            X_design_df = build_lagged_design(
                y_fit,
                prepared_X,
                y_lags,
                X_lags,
            )
        else:
            X_design_df = prepared_X

        self.dummies = dummies
        dummy_definitions = None
        dummy_columns = []
        dummy_names = []
        if dummies:
            dummy_index = X_design_df.index if X_design_df is not None else y_fit.index
            D = build_dummies(dummy_index, dummies, target_frequency)
            dummy_names = list(D.columns)
            dummy_definitions = self._dummy_spec(dummies, dummy_names)
            X_design_df = (
                D if X_design_df is None else pd.concat([X_design_df, D], axis=1)
            )

        if getattr(self, "_formula", None):
            y_fit = self._formula.extract_y(y_fit)
            X_design_df = self._formula.extract_X(X_design_df)

        if dummy_names and X_design_df is not None:
            present = [column for column in dummy_names if column in X_design_df.columns]
            zero = [column for column in present if not (X_design_df[column] != 0).any()]
            if zero:
                X_design_df = X_design_df.drop(columns=zero)
            dummy_columns = [column for column in present if column not in zero]

        y_estimation, X_estimation = self._prepare_estimation_inputs(y_fit, X_design_df)
        if not self._handles_missing_values:
            y_estimation, X_estimation = self._drop_missing_estimation_rows(
                y_estimation, X_estimation
            )

        last_y_fit_date = (
            y_estimation.index[-1]
            if not self._handles_missing_values
            or not y_estimation.index.equals(y_fit.index)
            else y_fit.index[-1]
        )
        self._dummy_definitions = dummy_definitions
        self._dummy_cols = dummy_columns
        return (
            y_estimation,
            X_estimation,
            _FitDesignState(
                y_history=y_fit,
                X_history=X_design_df,
                y_name=self.y_name,
                X_names=self.X_names,
                dummies=dummies,
                dummy_definitions=dummy_definitions,
                dummy_columns=dummy_columns,
                last_y_fit_date=last_y_fit_date,
            ),
        )

    def _store_fitted_configuration(
        self,
        y_estimation: pd.DataFrame,
        X_estimation: pd.DataFrame | None,
        transformation: FittedDataTransformation,
        design_state: _FitDesignState,
        drop_transformation_nans: bool,
    ) -> None:
        """Store fitted data and the immutable configuration used to forecast."""
        self._y_history = design_state.y_history
        self._X_history = design_state.X_history
        self.y = y_estimation
        self.X = X_estimation
        self.y_estimation = y_estimation
        self.X_estimation = X_estimation
        self._forecast_frequency = transformation.frequency
        self._fitted_model_configuration = FittedModelConfiguration(
            data_transformation=transformation,
            y_columns=tuple(y_estimation.columns),
            X_columns=tuple(X_estimation.columns) if X_estimation is not None else None,
            y_lags=self.y_lags,
            X_lags=_freeze_option(self.X_lags),
            dummies=_freeze_option(design_state.dummies)
            if design_state.dummies is not None
            else None,
            dummy_definitions=(
                _freeze_option(design_state.dummy_definitions)
                if design_state.dummy_definitions is not None
                else None
            ),
            dummy_columns=tuple(design_state.dummy_columns),
            forecast_origin=(
                self.last_y_fit_date.to_timestamp(how="end").normalize()
                if isinstance(self.last_y_fit_date, pd.Period)
                else pd.Timestamp(self.last_y_fit_date)
            ),
            drop_transformation_nans=drop_transformation_nans,
        )
        self._is_fitted = True

    def _fit_impl(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame | None = None,
        y_lags: int = 0,
        X_lags: int | dict = 0,
        dummies: list | dict | None = None,
        data_transformation: dict[str, str] | None = None,
        frequency: str | None = None,
        X_imputation: str | None = None,
        input_frequencies: dict[str, str] | None = None,
        y_input_metrics: dict[str, str] | None = None,
        X_input_metrics: dict[str, str] | None = None,
        drop_transformation_nans: bool = True,
        **kwargs,
    ):
        """Fit the model by orchestrating the preparation stages."""
        raw_y, raw_X = self._validate_fit_inputs(y, X)
        if input_frequencies is None:
            explicit_frequency = frequency is not None
            requested_metrics = data_transformation or self.data_transformation or {}
            y_input_metrics = y_input_metrics or {}
            X_input_metrics = X_input_metrics or {}
            y_frequency_variables = [
                variable
                for variable in raw_y.columns
                if requested_metrics.get(variable) in {"diff", "log diff", "pop", "yoy"}
                and y_input_metrics.get(variable, "levels") == "levels"
            ]
            X_frequency_variables = (
                [
                    variable
                    for variable in raw_X.columns
                    if requested_metrics.get(variable)
                    in {"diff", "log diff", "pop", "yoy"}
                    and X_input_metrics.get(variable, "levels") == "levels"
                ]
                if raw_X is not None
                else []
            )
            calendar_dates = (
                raw_y.index.to_timestamp()
                if isinstance(raw_y.index, pd.PeriodIndex)
                else raw_y.index
            )
            calendar_index = (
                calendar_dates.is_month_start.all() or calendar_dates.is_month_end.all()
            )
            if calendar_index and (
                explicit_frequency
                or y_lags
                or X_lags
                or dummies
                or X_imputation
                or not self._handles_missing_values
            ):
                y_frequency_variables = list(raw_y.columns)
                X_frequency_variables = list(raw_X.columns) if raw_X is not None else []
            y_frequency_variables = (
                list(raw_y.columns) if frequency is not None else y_frequency_variables
            )
            X_frequency_variables = (
                list(raw_X.columns)
                if frequency is not None and raw_X is not None
                else X_frequency_variables
            )
            input_frequencies = {
                variable: None
                for variable in [
                    *raw_y.columns,
                    *(raw_X.columns if raw_X is not None else []),
                ]
            }
            input_frequencies.update(
                infer_variable_frequencies(raw_y, y_frequency_variables, "raw y")
            )
            if raw_X is not None:
                input_frequencies.update(
                    infer_variable_frequencies(raw_X, X_frequency_variables, "raw X")
                )
            inferred_frequency = (
                raw_y.index.freqstr
                if isinstance(raw_y.index, pd.PeriodIndex)
                else raw_y.index.inferred_freq
            )
            if frequency is None and calendar_index and not inferred_frequency:
                frequency = infer_frequency_from_dates(raw_y.index, "raw y")
            elif frequency is None and inferred_frequency:
                inferred_frequency = inferred_frequency.replace("Q-", "QE-")
                rule_code = pd.tseries.frequencies.to_offset(
                    inferred_frequency
                ).rule_code.upper()
                frequency = (
                    "M"
                    if rule_code.startswith(("ME", "MS"))
                    else "Q"
                    if rule_code.startswith(("QE", "QS"))
                    else inferred_frequency
                )
        _validate_X_imputation(X_imputation)
        if getattr(self, "_formula", None):
            y_input_metrics = _restrict_mapping(y_input_metrics, raw_y.columns)
            X_columns = raw_X.columns if raw_X is not None else []
            X_input_metrics = _restrict_mapping(X_input_metrics, X_columns)
        self._raw_y_history = raw_y
        self._raw_X_history = raw_X
        fitted_transformation = self._resolve_fit_transformation(
            raw_y,
            raw_X,
            data_transformation,
            frequency,
            X_imputation,
            input_frequencies,
            y_input_metrics,
            X_input_metrics,
        )
        self._input_frequencies = {
            **fitted_transformation.y_frequency_mapping,
            **fitted_transformation.X_frequency_mapping,
        }
        prepared_y, prepared_X = self._prepare_training_data(
            raw_y,
            raw_X,
            fitted_transformation,
            X_imputation,
            drop_transformation_nans,
        )
        y_estimation, X_estimation, design_state = self._build_fit_design(
            prepared_y,
            prepared_X,
            y_lags,
            X_lags,
            dummies,
            fitted_transformation.frequency,
        )
        self.last_y_fit_date = design_state.last_y_fit_date
        fitted = self._fit(y=y_estimation, X=X_estimation, **kwargs)
        if fitted is not self:
            raise TypeError(
                f"{self.__class__.__name__}._fit must return self; "
                f"got {type(fitted).__name__ if fitted is not None else 'None'}."
            )
        self._store_fitted_configuration(
            y_estimation,
            X_estimation,
            fitted_transformation,
            design_state,
            drop_transformation_nans,
        )
        return self

    def forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        decomp: bool = False,
        data_transformation: dict[str, str] | None = None,
        frequency: str | None = None,
        X_imputation: str | None = None,
        context: ForecastContext | None = None,
        **kwargs,
    ) -> ForecastResult:
        """Forecast using the fitted history without changing model state."""
        if not getattr(self, "_is_fitted", False):
            raise AttributeError("Model has not been fitted yet; call fit() first.")
        if context is None:
            context = ForecastContext(
                y_history=self._raw_y_history,
                X_history=self._raw_X_history,
                y_conditioning=y,
                X_conditioning=X,
                forecast_origin=self._fitted_model_configuration.forecast_origin,
                y_conditioning_input_metrics=None,
                X_conditioning_input_metrics=None,
            )
        return self.predict(
            context,
            steps=steps,
            decomp=decomp,
            data_transformation=data_transformation,
            frequency=frequency,
            X_imputation=X_imputation,
            **kwargs,
        )

    def predict(
        self,
        context: ForecastContext,
        steps: int = 1,
        decomp: bool = False,
        data_transformation: dict[str, str] | None = None,
        frequency: str | None = None,
        X_imputation: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate forecasts.

        Args:
            steps : int
                Number of steps ahead to forecast. Default 1.
            X : pd.DataFrame, optional
                Exogenous regressors, extended over the forecast horizon.
                Combined with the raw ``X`` history stored by ``fit()`` (future
                value wins on any overlapping date) before transforming, so
                the first transformed value is anchored to the final raw
                fitted observation. This merge happens whether or not a
                pipeline is resolved, so an untransformed autoregressive model
                still sees its full raw history.
            y : pd.DataFrame, optional
                Conditioning paths for y, shape (steps, n_y_vars); NaN entries
                are unconstrained. Combined with the raw ``y`` history the same
                way, whether or not a pipeline is resolved.
            decomp : bool, optional
                If True, include decomposition rows from ``_forecast_decomp()``
                in the returned ``ForecastResult``. Default False.
            data_transformation : dict, optional
                Call-level fallback transformation, used only when this model
                has no model-owned ``data_transformation`` of its own. Should
                match whatever was passed to ``fit()`` for this model.
            frequency : str, optional
                Legacy target frequency metadata ("M" or "Q"). Input
                transformation frequency is inferred from each raw column.
            X_imputation : str, optional
                Ragged-edge imputation strategy applied to ``X`` after any
                transformation, extending it to cover the forecast horizon,
                when ``self._needs_ragged_edge_imputation`` is True. If that
                attribute is False, the model handles its own ragged edge and
                this option is not applied.
        Returns:
            pd.DataFrame
                ``steps`` rows indexed by a DatetimeIndex named ``"date"``, one
                column per target variable.
        """
        # Validate that steps is an integer greater than zero
        if not isinstance(steps, int) or steps <= 0:
            raise ValueError("'Steps' must be an integer greater than zero")
        if not getattr(self, "_is_fitted", False):
            raise AttributeError("Model has not been fitted yet; call fit() first.")
        configuration = self._fitted_model_configuration
        fitted_transformation = configuration.data_transformation
        fitted_pipeline_mapping = (
            dict(fitted_transformation.data_transformation)
            if fitted_transformation.data_transformation is not None
            else {}
        )
        forecast_pipeline = (
            _coerce_data_transformation(data_transformation)
            if data_transformation is not None
            else None
        )
        if (
            data_transformation is not None
            and fitted_transformation.pipeline_source != "model"
        ):
            _validate_fitted_override(
                "data_transformation",
                (
                    forecast_pipeline.data_transformation
                    if forecast_pipeline is not None
                    else None
                ),
                fitted_pipeline_mapping,
            )
        _validate_fitted_override("frequency", frequency, fitted_transformation.frequency)
        _validate_fitted_override(
            "X_imputation", X_imputation, fitted_transformation.X_imputation
        )
        effective_X_imputation = (
            fitted_transformation.X_imputation if X_imputation is None else X_imputation
        )
        if context.y_conditioning is not None:
            if not isinstance(context.y_conditioning, pd.DataFrame):
                raise TypeError("y must be a pandas DataFrame or None")
            self._validate_datetime_frame(context.y_conditioning, "y")
        if context.X_conditioning is not None:
            if not isinstance(context.X_conditioning, pd.DataFrame):
                raise TypeError("X must be a pandas DataFrame or None")
            self._validate_datetime_frame(context.X_conditioning, "X")
        _validate_X_imputation(X_imputation)

        raw_y_history = context.y_history
        raw_X_history = context.X_history
        if not isinstance(raw_y_history, pd.DataFrame):
            raise TypeError("context.y_history must be a pandas DataFrame.")
        self._validate_datetime_frame(raw_y_history, "context.y_history")
        if raw_X_history is not None:
            if not isinstance(raw_X_history, pd.DataFrame):
                raise TypeError("context.X_history must be a pandas DataFrame or None.")
            self._validate_datetime_frame(raw_X_history, "context.X_history")
        raw_y_conditioning = context.y_conditioning
        raw_X_conditioning = context.X_conditioning
        if getattr(self, "_formula", None):
            raw_y_history = self._formula.extract_y(raw_y_history)
            if raw_y_conditioning is not None:
                raw_y_conditioning = raw_y_conditioning[
                    [
                        column
                        for column in fitted_transformation.y_variables
                        if column in raw_y_conditioning.columns
                    ]
                ]

            if fitted_transformation.X_variables is None:
                raw_X_history = None
                raw_X_conditioning = None
            else:
                X_columns = list(fitted_transformation.X_variables)
                if raw_X_history is not None:
                    raw_X_history = raw_X_history[
                        [column for column in X_columns if column in raw_X_history]
                    ]
                if raw_X_conditioning is not None:
                    raw_X_conditioning = raw_X_conditioning[
                        [column for column in X_columns if column in raw_X_conditioning]
                    ]
        y_conditioning_input_metrics = (
            context.y_conditioning_input_metrics
            if context.y_conditioning_input_metrics is not None
            else fitted_transformation.y_input_metric_mapping
        )
        X_conditioning_input_metrics = (
            context.X_conditioning_input_metrics
            if context.X_conditioning_input_metrics is not None
            else fitted_transformation.X_input_metric_mapping
        )
        if getattr(self, "_formula", None):
            y_conditioning_input_metrics = _restrict_mapping(
                y_conditioning_input_metrics, fitted_transformation.y_variables
            )
            X_conditioning_input_metrics = _restrict_mapping(
                X_conditioning_input_metrics, fitted_transformation.X_variables or ()
            )
        forecast_origin = (
            context.forecast_origin
            if context.forecast_origin is not None
            else raw_y_history.index[-1]
        )
        pipeline = fitted_transformation.pipeline

        if pipeline is not None:
            y_history, y_conditioning, X_history, X_conditioning = (
                fitted_transformation.transform_forecast_inputs(
                    y_history=raw_y_history,
                    y_conditioning=raw_y_conditioning,
                    X_history=raw_X_history,
                    X_future=(raw_X_conditioning if raw_X_history is not None else None),
                    y_conditioning_input_metrics=y_conditioning_input_metrics,
                    X_conditioning_input_metrics=X_conditioning_input_metrics,
                )
            )
            y_input = combine_history_and_future(y_history, y_conditioning)
            if raw_X_history is not None and raw_X_conditioning is not None:
                X_input = combine_history_and_future(X_history, X_conditioning)
            else:
                X_input = None
        else:
            # No transformation to apply, but an untransformed autoregressive
            # model still needs its raw fitted history ahead of an explicitly
            # supplied conditioning/future path, with the future value
            # winning on any overlapping (backcast) date - the same contract
            # as the pipeline-resolved branch above. Only merge when the
            # caller actually supplied a conditioning/future value: passing
            # None is how a model explicitly opts out of using that input for
            # this call (e.g. an external-process model distinguishing "no
            # future regressors" from "reuse the fit regressors").
            y_input = (
                combine_history_and_future(raw_y_history, raw_y_conditioning)
                if raw_y_conditioning is not None
                else None
            )
            X_input = (
                combine_history_and_future(raw_X_history, raw_X_conditioning)
                if raw_X_history is not None and raw_X_conditioning is not None
                else None
            )

        if y_input is not None and (
            not self._handles_missing_values or fitted_transformation.y_frequency_mapping
        ):
            y_input = regularise_missing_rows(
                y_input, fitted_transformation.y_frequency_mapping
            )
        if (
            X_input is not None
            and raw_X_history is not None
            and (
                not self._handles_missing_values
                or fitted_transformation.X_frequency_mapping
            )
        ):
            X_input = regularise_missing_rows(
                X_input, fitted_transformation.X_frequency_mapping
            )

        if (
            X_input is not None
            and effective_X_imputation is not None
            and self._needs_ragged_edge_imputation
        ):
            target_frequency = fitted_transformation.frequency
            target_period = pd.Period(forecast_origin, freq=target_frequency) + steps
            target_date = target_period.to_timestamp(how="end").normalize()
            X_input = impute_X(
                X_input,
                target_date,
                steps=0,
                method=effective_X_imputation,
                frequencies=fitted_transformation.X_frequency_mapping,
            )

        y_input, X_input = self._prepare_forecast_inputs(y_input, X_input)

        if getattr(self, "_formula", None) and y_input is not None:
            y_input = self._formula.extract_y(y_input)

        # The public attributes remain mirrored for model implementations, but
        # prediction decisions use the captured values so later caller
        # mutation cannot alter a fit.
        X_lags = _thaw_option(configuration.X_lags)
        y_lags = configuration.y_lags

        # Build augmented design matrix (full history) using build_lagged_design
        if y_lags or X_lags:
            # Use combined inputs when supplied, otherwise use the fitted history.
            y_design_input = (
                y_input
                if y_input is not None
                else getattr(
                    self, "_prepared_y_history", getattr(self, "_y_history", self.y)
                )
            )
            X_design_input = (
                X_input
                if X_input is not None
                else getattr(
                    self, "_prepared_X_history", getattr(self, "_X_history", self.X)
                )
            )

            # Build augmented design matrix with full history
            X_design = build_lagged_design(
                y_design_input,
                X_design_input,
                y_lags,
                X_lags,
            )
        else:
            X_design = X_input

        # Append all requested dummies BEFORE the formula so it can reference
        # them by name. They are regenerated here (not imputed) because they are
        # deterministic functions of the date index, keeping the forecast design
        # aligned with fit over history + horizon.
        dummies = _thaw_option(configuration.dummies)
        dummy_definitions = _thaw_option(configuration.dummy_definitions)
        dummy_names = []
        if dummies:
            target_frequency = fitted_transformation.frequency
            if X_design is not None:
                dummy_index = X_design.index
            else:
                y_design_history = (
                    y_input
                    if y_input is not None
                    else getattr(
                        self, "_prepared_y_history", getattr(self, "_y_history", self.y)
                    )
                )
                forecast_dates = self._infer_forecast_dates(
                    y_design_history.index,
                    steps,
                    frequency=target_frequency,
                )
                dummy_index = y_design_history.index.append(forecast_dates)
            D = build_dummies(
                dummy_index,
                dummy_definitions or dummies,
                target_frequency,
            )
            dummy_names = list(D.columns)
            X_design = D if X_design is None else pd.concat([X_design, D], axis=1)

        # Apply formula if provided
        if getattr(self, "_formula", None):
            X_design = self._formula.extract_X(X_design)

        # Keep exactly the dummy columns retained at fit time (all-zero or
        # formula-excluded dummies were dropped there), so the forecast design
        # matrix matches the fitted one.
        if dummy_names and X_design is not None:
            keep = set(configuration.dummy_columns)
            drop = [c for c in dummy_names if c in X_design.columns and c not in keep]
            if drop:
                X_design = X_design.drop(columns=drop)

        if not self._handles_missing_values and X_design is not None:
            X_design = X_design.dropna()

        # Pass full history to _forecast; models handle filtering to forecast rows
        hook_kwargs = dict(kwargs)
        hook_kwargs["forecast_origin"] = forecast_origin
        forecast = self._forecast(steps=steps, X=X_design, y=y_input, **hook_kwargs)
        return self._finalise_forecast(
            forecast,
            steps,
            forecast_origin,
            decomp=decomp,
            decomp_kwargs={"X": X_design, "y": y_input, **hook_kwargs},
        )
