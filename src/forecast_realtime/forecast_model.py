"""Core classes and contracts for forecast models."""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from forecast_realtime._data_transformation import (
    DataTransformationPipeline,
    FittedDataTransformation,
)
from forecast_realtime._model_data import (
    ModelData,
    ModelInputRequirements,
    _validate_mapping_coverage,
    _validate_metric_mapping,
)
from forecast_realtime._utils import (
    build_dummies,
    build_lagged_design,
    dummy_items,
    resolve_X_lags,
)
from forecast_realtime.formula import Formula

from ._conditioning import _parse_conditioning
from ._forecast_context import ForecastContext
from .forecast_result import ForecastResult, _normalise_quantiles

# Single source of truth for RealTimeModel's X_imputation strategies, shared
# with the validation check in RealTimeModel.forecast() and the error below.
X_IMPUTATION_METHODS = ("zero", "last", "mean", "ar1_t")


class NoUsableTransformedYError(ValueError):
    """Raised when input preparation leaves no usable transformed target rows."""


@dataclass(frozen=True)
class DesignSpec:
    """Frozen options that define a model's fitted design matrix."""

    y_lags: int = 0
    X_lags: tuple[tuple[str, int], ...] = ()
    dummies: tuple[tuple[str, pd.Timestamp], ...] = ()
    columns: tuple[str, ...] | None = None
    frequency: str | None = None
    period_index: bool = False
    month_start: bool = False

    def build(self, y, X, index=None, formula=None) -> pd.DataFrame | None:
        """Apply the fitted lags, dummies, formula, and column order."""
        if index is not None:
            design_index = y.index.union(index).sort_values()
            if X is not None:
                design_index = design_index.union(X.index).sort_values()
            y = y.reindex(design_index)
            if X is not None:
                X = X.reindex(design_index)

        X_lags = dict(self.X_lags)
        if self.y_lags or any(X_lags.values()):
            design = build_lagged_design(y, X, self.y_lags, X_lags)
        else:
            design = X

        if self.dummies:
            dummy_index = design.index if design is not None else y.index
            if isinstance(dummy_index, pd.PeriodIndex):
                dummy_index = dummy_index.to_timestamp(how="end").normalize()
            dummies = build_dummies(
                pd.DatetimeIndex(dummy_index),
                dict(self.dummies),
                self.frequency,
            )
            design = dummies if design is None else pd.concat([design, dummies], axis=1)

        if formula is not None:
            design = formula.extract_X(design)
        if design is not None and self.columns is not None:
            design = design.reindex(columns=self.columns)
        return design


@dataclass(frozen=True)
class FittedModelConfiguration:
    """Immutable configuration captured when a model is successfully fitted."""

    inputs: FittedDataTransformation
    design: DesignSpec
    y_columns: tuple[str, ...]
    X_columns: tuple[str, ...] | None
    forecast_origin: pd.Timestamp


_FITTED_VALUES_NOT_FITTED_MSG = (
    "fitted_values are not available; the model has not been fitted yet "
    "(call fit() first)."
)


def _validate_X_imputation(value: str | None) -> None:
    if value is not None and value not in X_IMPUTATION_METHODS:
        raise ValueError(
            f"X_imputation must be None or one of {X_IMPUTATION_METHODS}; got {value!r}"
        )


def _restrict_mapping(value: dict | None, columns) -> dict | None:
    """Return mapping entries belonging to the selected input columns."""
    if value is None:
        return None
    return {column: value[column] for column in columns if column in value}


def _as_origin(date) -> pd.Timestamp:
    """Convert a forecast origin to its calendar timestamp."""
    if isinstance(date, pd.Period):
        return date.to_timestamp(how="end").normalize()
    return pd.Timestamp(date)


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

    _supports_target_conditioning: bool = False

    _supports_quantiles: bool = False

    def __init__(
        self,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        align_start_dates: bool = False,
        conditioning: dict | None = None,
    ):
        """Configure input selection, transformation and conditioning.

        Parameters
        ----------
        label : str | None
            Forecast label. Defaults to the class name.
        formula : str | None
            Variable selection, e.g. ``"cpisa ~ gdpkp + unemp"``.
            None uses all supplied y and X variables.
        data_transformation : dict[str, str] | None
            Model-owned variable-to-metric mapping, preferred over the run fallback.
        align_start_dates : bool
            Align input series to their latest common starting date.
        conditioning : dict | None
            Sources and positive durations by role and variable.
            None inherits the run fallback; an empty mapping disables it.
            Direct forecasts use supplied frames independently of this policy.
        """
        self.label = label if label is not None else self.__class__.__name__
        self._formula = Formula(formula) if formula else None
        self.fitted_values_ = None
        self._is_fitted = False
        self.data_transformation = data_transformation
        self.align_start_dates = align_start_dates
        self._conditioning = _parse_conditioning(conditioning, self.label)

    @property
    def conditioning(self):
        """Immutable conditioning policy; direct forecasts use supplied frames instead."""
        return self._conditioning

    def _conditioning_variables(self, requirements, y_variables):
        """Return raw inputs eligible for source conditioning."""
        return {
            role: {
                variable
                for request in requirements
                for variable, _ in request.items(role)
            }
            for role in ("y", "X")
        }

    def _validate_target_conditioning(self, variables):
        """Reject explicit target requests the model cannot enforce."""
        if variables and not self._supports_target_conditioning:
            raise ValueError(
                f"Model {self.label!r} does not support target conditioning "
                f"for variables {sorted(variables)}."
            )

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
        if getattr(self, "_formula", None) is not None:
            return list(self._formula.y_cols)
        return list(y_variables)

    @property
    def y_lags(self) -> int:
        return self._fitted_model_configuration.design.y_lags

    @property
    def X_lags(self) -> dict[str, int]:
        return dict(self._fitted_model_configuration.design.X_lags)

    @property
    def y_name(self) -> str:
        return self._fitted_model_configuration.y_columns[0]

    @property
    def last_y_fit_date(self) -> pd.Timestamp:
        return self._fitted_model_configuration.forecast_origin

    @property
    def _dummy_cols(self) -> list[str]:
        design = self._fitted_model_configuration.design
        if design.columns is None:
            return []
        fitted_columns = set(design.columns)
        return [name for name, _ in design.dummies if name in fitted_columns]

    def _resolve_mapping(
        self, fallback: dict[str, str] | None = None
    ) -> tuple[dict[str, str] | None, str]:
        """Resolve model, call-level, or identity mapping in precedence order."""
        mapping = self.data_transformation
        if mapping is not None:
            return mapping, "model"
        if fallback is not None:
            return _validate_metric_mapping(fallback, "data_transformation"), "fallback"
        return None, "identity"

    def input_requirements(
        self,
        y_variables,
        X_variables=None,
        data_transformation=None,
    ) -> tuple[ModelInputRequirements, ...]:
        """Report every formula-selected input, including implicit levels requests."""
        mapping, source = self._resolve_mapping(data_transformation)
        if getattr(self, "_formula", None) is not None:
            y_variables, X_variables = self._formula.input_columns(X_variables)
        if mapping is not None:
            _validate_mapping_coverage(mapping, y_variables, X_variables)
        return (
            ModelInputRequirements(
                consumer=self.label,
                y=tuple((v, (mapping or {}).get(v, "levels")) for v in y_variables),
                X=tuple(
                    (v, (mapping or {}).get(v, "levels")) for v in (X_variables or [])
                ),
                explicit=source != "identity",
            ),
        )

    @property
    def _raw_y_history(self):
        """Retain the established inspection path as an isolated projection."""
        return self._raw_data.to_wide("y")

    @property
    def _input_frequencies(self):
        """Expose resolved input calendars to model-specific fit hooks."""
        return self._raw_data.frequencies("y") | self._raw_data.frequencies("X")

    @property
    def _raw_X_history(self):
        return self._raw_data.to_wide("X")

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
        mapping, _ = self._resolve_mapping(data_transformation)
        if mapping is not None and y_variables is not None:
            formula = getattr(self, "_formula", None)
            if formula is not None:
                y_variables, X_variables = formula.input_columns(X_variables)
            _validate_mapping_coverage(
                mapping,
                y_variables,
                X_variables,
            )
        return DataTransformationPipeline(mapping) if mapping is not None else None

    def native_metric_mapping(
        self, target_variables: list[str] | None = None
    ) -> dict[str, str]:
        """Return the metric space used by the fitted target outputs."""
        target_variables = list(target_variables or self.y.columns)
        configuration = self._fitted_model_configuration
        mapping = configuration.inputs.mapping or {}
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
        quantiles=None,
    ) -> ForecastResult:
        """Normalise hook output, validate the result and attach metadata."""
        if quantiles is None:
            forecast = self._normalise_point_forecast(forecast, steps, forecast_origin)
        decomposition = (
            self._forecast_decomp(steps=steps, **(decomp_kwargs or {}))
            if decomp and quantiles is None
            else None
        )
        return ForecastResult(
            forecast,
            expected_columns=list(self._fitted_model_configuration.y_columns),
            steps=steps,
            forecast_origin=forecast_origin,
            decomposition=decomposition,
            quantiles=False if quantiles is None else quantiles,
            forecast_dates_include_origin=self._forecast_dates_include_origin,
            decomp=decomp,
        )

    def _normalise_point_forecast(self, forecast, steps, forecast_origin):
        """Validate the wide hook contract and convert it to a long table."""
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

        ForecastResult._validate_calendar(
            forecast.index, steps, forecast_origin, self._forecast_dates_include_origin
        )
        expected_columns = list(self._fitted_model_configuration.y_columns)
        if list(forecast.columns) != expected_columns:
            raise ValueError(
                "Forecast columns must match the fitted target columns in order; "
                f"expected {expected_columns}, got {list(forecast.columns)}"
            )
        return pd.DataFrame(
            {
                "date": forecast.index.repeat(len(expected_columns)),
                "variable": np.tile(expected_columns, len(forecast)),
                "value": forecast.to_numpy().reshape(-1),
            }
        )

    def _forecast_dates(self, origin, steps):
        """Return the authoritative calendar for a fitted model forecast."""
        configuration = self._fitted_model_configuration
        origin = _as_origin(configuration.forecast_origin if origin is None else origin)
        design = configuration.design
        if design.period_index:
            calendar = pd.PeriodIndex([pd.Period(origin, freq=design.frequency)])
        else:
            anchor = (
                origin.to_period(design.frequency).to_timestamp()
                if design.month_start and design.frequency in ("M", "Q")
                else origin
            )
            calendar = pd.DatetimeIndex([anchor])
        dates = self._infer_forecast_dates(
            calendar,
            steps,
            frequency=design.frequency,
            start=origin,
        )
        if self._forecast_dates_include_origin:
            dates = pd.DatetimeIndex([origin]).append(dates[:-1])
        return dates

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
        using :meth:`_forecast_dates` for the index and fitted target columns.

        ``forecast()`` applies this automatically, so models rarely call it.
        Dates are anchored to ``last_y_fit_date`` unless an explicit origin is
        supplied by the prediction context.
        """
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        configuration = self._fitted_model_configuration
        expected_columns = list(configuration.y_columns)
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
            configuration.forecast_origin if forecast_origin is None else forecast_origin
        )
        dates = self._forecast_dates(forecast_origin, steps)

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
        ModelData.validate_frame(y, "y")
        if X is not None:
            ModelData.validate_frame(X, "X")
        if getattr(self, "_formula", None) is not None:
            y_columns, X_columns = self._formula.input_columns(
                X.columns if X is not None else None
            )
            y_input_metrics = _restrict_mapping(y_input_metrics, y_columns)
            if X_columns is not None and not self._formula.has_wildcard:
                X_input_metrics = _restrict_mapping(X_input_metrics, X_columns)
        data = ModelData.from_wide(
            y,
            X,
            frequencies=input_frequencies,
            y_input_metrics=y_input_metrics,
            X_input_metrics=X_input_metrics,
        )
        source_data = kwargs.pop("_model_data", None)
        if source_data is not None:
            data = data.with_input_metadata(source_data, fields=("source", "owner"))
        return self._fit_data(
            data,
            y_lags=y_lags,
            X_lags=X_lags,
            dummies=dummies,
            data_transformation=data_transformation,
            frequency=frequency,
            X_imputation=X_imputation,
            drop_transformation_nans=drop_transformation_nans,
            **kwargs,
        )

    def _fit_data(self, data: ModelData, **kwargs):
        """Fit a candidate through the single data-based implementation."""
        candidate = copy.deepcopy(self)
        data = candidate._select_fit_data(data)
        candidate._fit_data_impl(data, **kwargs)
        self._commit_fit(candidate)
        return self

    def _fit_from_data(self, data: ModelData, **kwargs):
        """Retain public fit overrides without making frames the internal payload."""
        if type(self).fit is not ForecastModel.fit:
            return self.fit(
                data.to_wide("y"),
                data.to_wide("X"),
                input_frequencies={**data.frequencies("y"), **data.frequencies("X")},
                y_input_metrics=data.metrics("y"),
                X_input_metrics=data.metrics("X"),
                _model_data=data,
                **kwargs,
            )
        return self._fit_data(data, **kwargs)

    def _commit_fit(self, candidate: "ForecastModel") -> None:
        """Publish a successfully fitted candidate on this model instance."""
        self.__dict__.clear()
        self.__dict__.update(candidate.__dict__)

    def _validate_forecast_options(
        self, data_transformation, frequency, X_imputation
    ) -> None:
        """Reject public preprocessing overrides that conflict with fitted state."""
        configuration = self._fitted_model_configuration
        inputs = configuration.inputs
        if data_transformation is not None and inputs.pipeline_source != "model":
            mapping = _validate_metric_mapping(data_transformation, "data_transformation")
            _validate_fitted_override(
                "data_transformation", mapping, inputs.mapping or {}
            )
        _validate_fitted_override("frequency", frequency, configuration.design.frequency)
        _validate_X_imputation(X_imputation)
        _validate_fitted_override("X_imputation", X_imputation, inputs.X_imputation)

    def _select_fit_data(self, data):
        """Validate fitting capabilities and select raw formula inputs once."""
        for role in ("y", "X"):
            if data.has_path(role) and (
                not len(data.index(role)) or not len(data.columns(role))
            ):
                raise ValueError(
                    f"{role} must not be empty" + (" if provided" if role == "X" else "")
                )
        y_columns = list(data.columns("y"))
        X_columns = list(data.columns("X")) if data.has_path("X") else None
        if getattr(self, "_formula", None) is not None:
            self._formula._validate_y_columns(y_columns)
            y_columns, X_columns = self._formula.input_columns(X_columns)
        if not self._supports_multivariate_y and len(y_columns) > 1:
            raise ValueError(
                f"{type(self).__name__} cannot handle multiple left-hand-side "
                f"variables; select one variable in `forecast(y_variables=)` or use "
                f"the formula argument of {type(self).__name__}"
            )
        return data.history().subset(y_columns, X_columns)

    def _resolve_fit_transformation(
        self,
        data: ModelData,
        mapping: dict[str, str] | None,
        mapping_source: str,
        X_imputation: str | None,
        drop_transformation_nans: bool,
    ) -> FittedDataTransformation:
        """Resolve the transformation and validate input frequency compatibility."""
        y_columns = list(data.columns("y"))
        X_columns = list(data.columns("X")) if data.has_path("X") else None
        if mapping is not None:
            formula = getattr(self, "_formula", None)
            if formula is not None:
                y_columns, X_columns = formula.input_columns(X_columns)
            _validate_mapping_coverage(mapping, y_columns, X_columns)
        y_frequency_map = data.frequencies("y")
        X_frequency_map = data.frequencies("X")
        y_frequency_values = {
            y_frequency_map[variable]
            for variable in y_columns
            if y_frequency_map[variable] is not None
        }
        y_frequency = (
            next(iter(y_frequency_values)) if len(y_frequency_values) == 1 else None
        )
        if X_columns is not None and not self._handles_mixed_frequencies:
            X_frequency_values = set(X_frequency_map.values())
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
            mapping,
            y_variables=y_columns,
            X_variables=X_columns,
            X_imputation=X_imputation,
            pipeline_source=mapping_source,
            drop_transformation_nans=drop_transformation_nans,
        )

    def _prepare_training_data(
        self,
        data: ModelData,
        transformation: FittedDataTransformation,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Transform, impute, regularise, and model-prepare training data."""
        mapping = transformation.mapping
        prepared = data.transform(mapping)
        if mapping is not None and transformation.drop_transformation_nans:
            prepared = prepared.trim_undefined_prefix()
            if not len(prepared.index("y")) or not prepared.last_valid_dates("y"):
                raise NoUsableTransformedYError(
                    "No usable transformed y observations remain after transformation."
                )
        if transformation.X_imputation is not None and self._needs_ragged_edge_imputation:
            target = max(prepared.last_valid_dates("X") + [prepared.index("y")[-1]])
            prepared = prepared.impute(target, method=transformation.X_imputation)
        prepared = prepared.regularise()
        return self._prepare_fit_inputs(prepared.to_wide("y"), prepared.to_wide("X"))

    def _build_fit_design(
        self,
        prepared_y: pd.DataFrame,
        prepared_X: pd.DataFrame | None,
        y_lags: int,
        X_lags: int | dict,
        dummies: list | dict | None,
        target_frequency: str | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None, DesignSpec]:
        """Build the estimation design and its frozen specification."""
        if prepared_y.empty or prepared_y.dropna(how="all").empty:
            raise NoUsableTransformedYError(
                "No usable transformed y observations remain after model preparation."
            )

        y_fit = (
            self._formula.extract_y(prepared_y)
            if getattr(self, "_formula", None)
            else prepared_y
        )
        X_lags_map = (
            resolve_X_lags(X_lags, prepared_X.columns) if prepared_X is not None else {}
        )
        design_index = (
            y_fit.index
            if prepared_X is None
            else y_fit.index.union(prepared_X.index).sort_values()
        )
        dummy_spec = tuple(
            (name, _as_origin(date))
            for name, date in (dummy_items(dummies, target_frequency) if dummies else ())
        )
        specification = DesignSpec(
            y_lags=y_lags,
            X_lags=tuple(X_lags_map.items()),
            dummies=dummy_spec,
            frequency=target_frequency,
            period_index=isinstance(y_fit.index, pd.PeriodIndex),
            month_start=bool(
                isinstance(y_fit.index, pd.DatetimeIndex)
                and y_fit.index.is_month_start.all()
            ),
        )
        formula = getattr(self, "_formula", None)
        X_design_df = specification.build(
            y_fit,
            prepared_X,
            index=design_index,
            formula=formula,
        )

        dummy_names = [name for name, _ in specification.dummies]
        if dummy_names and X_design_df is not None:
            present = [column for column in dummy_names if column in X_design_df.columns]
            zero = [column for column in present if not (X_design_df[column] != 0).any()]
            if zero:
                X_design_df = X_design_df.drop(columns=zero)

        specification = replace(
            specification,
            columns=tuple(X_design_df.columns) if X_design_df is not None else None,
        )

        y_estimation, X_estimation = self._prepare_estimation_inputs(y_fit, X_design_df)
        if not self._handles_missing_values:
            y_estimation, X_estimation = self._drop_missing_estimation_rows(
                y_estimation, X_estimation
            )

        return (
            y_estimation,
            X_estimation,
            specification,
        )

    def _store_fitted_configuration(
        self,
        y_estimation: pd.DataFrame,
        X_estimation: pd.DataFrame | None,
        inputs: FittedDataTransformation,
        design: DesignSpec,
    ) -> None:
        """Publish fitted data and the immutable configuration used to forecast."""
        self.y = y_estimation
        self.X = X_estimation
        self._fitted_model_configuration = FittedModelConfiguration(
            inputs=inputs,
            design=design,
            y_columns=tuple(y_estimation.columns),
            X_columns=tuple(X_estimation.columns) if X_estimation is not None else None,
            forecast_origin=_as_origin(y_estimation.index[-1]),
        )

    def _fit_data_impl(
        self,
        data: ModelData,
        y_lags: int = 0,
        X_lags: int | dict = 0,
        dummies: list | dict | None = None,
        data_transformation: dict[str, str] | None = None,
        frequency: str | None = None,
        X_imputation: str | None = None,
        drop_transformation_nans: bool = True,
        **kwargs,
    ):
        """Fit the model by orchestrating the preparation stages."""
        _validate_X_imputation(X_imputation)
        mapping, mapping_source = self._resolve_mapping(data_transformation)
        data, frequency = data.resolve_frequencies(
            mapping,
            frequency,
            require_calendars=bool(
                y_lags
                or X_lags
                or dummies
                or X_imputation
                or not self._handles_missing_values
            ),
        )
        self._raw_data = data
        fitted_transformation = self._resolve_fit_transformation(
            data,
            mapping,
            mapping_source,
            X_imputation,
            drop_transformation_nans,
        )
        prepared_y, prepared_X = self._prepare_training_data(data, fitted_transformation)
        y_estimation, X_estimation, design = self._build_fit_design(
            prepared_y,
            prepared_X,
            y_lags,
            X_lags,
            dummies,
            frequency,
        )
        self._store_fitted_configuration(
            y_estimation,
            X_estimation,
            fitted_transformation,
            design,
        )
        fitted = self._fit(y=y_estimation, X=X_estimation, **kwargs)
        if fitted is not self:
            raise TypeError(
                f"{self.__class__.__name__}._fit must return self; "
                f"got {type(fitted).__name__ if fitted is not None else 'None'}."
            )
        self._is_fitted = True
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
        *,
        quantiles: bool | list[float] = False,
        **kwargs,
    ) -> ForecastResult:
        """Return long point forecasts, or native-metric quantiles when requested.

        Extend models through _forecast(), not by replacing this orchestration.
        Point hooks remain arrays or wide tables; ForecastResult validates output.
        """
        if not self._is_fitted:
            raise AttributeError("Model has not been fitted yet; call fit() first.")
        self._validate_forecast_options(data_transformation, frequency, X_imputation)
        if context is None:
            for role, frame in (("y", y), ("X", X)):
                if frame is not None:
                    ModelData.validate_frame(frame, role)
            future = ModelData.from_wide(
                y_conditioning=y,
                X_conditioning=X,
                frequencies={
                    **self._raw_data.frequencies("y"),
                    **self._raw_data.frequencies("X"),
                },
                y_conditioning_input_metrics=self._raw_data.metrics("y"),
                X_conditioning_input_metrics=self._raw_data.metrics("X"),
            )
            return self._predict_data(
                self._raw_data.with_conditioning(future),
                forecast_origin=self._fitted_model_configuration.forecast_origin,
                steps=steps,
                decomp=decomp,
                quantiles=quantiles,
                **kwargs,
            )
        return self._predict_data(
            ModelData.from_context(context, self._raw_data),
            forecast_origin=context.forecast_origin,
            steps=steps,
            decomp=decomp,
            quantiles=quantiles,
            **kwargs,
        )

    def _validate_explicit_target_path(self, data, forecast_origin, steps):
        """Validate raw explicit constraints before input preparation."""
        frame = data.to_wide("y", "conditioning")
        if frame is None or frame.empty or not frame.notna().any().any():
            return None
        origin = forecast_origin if forecast_origin is not None else data.index("y")[-1]
        dates = self._forecast_dates(origin, steps)
        if isinstance(frame.index, pd.PeriodIndex):
            frame.index = frame.index.to_timestamp(how="end").normalize()
        active = frame.loc[frame.index.isin(dates)].dropna(axis=1, how="all")
        self._validate_target_conditioning(set(active.columns))
        unknown = set(active.columns) - set(self._fitted_model_configuration.y_columns)
        if unknown:
            raise ValueError(
                f"Model {self.label!r}: conditioning variables {sorted(unknown)} "
                "are not fitted targets."
            )
        return active

    def _predict_data(
        self,
        data: ModelData,
        *,
        forecast_origin=None,
        steps=1,
        decomp=False,
        quantiles=False,
        **kwargs,
    ):
        """Prepare labelled inputs, construct the design and validate the result."""
        probabilities = _normalise_quantiles(quantiles)
        if probabilities is not None:
            if not self._supports_quantiles:
                raise ValueError(
                    f"{type(self).__name__} does not support quantile forecasts."
                )
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be an integer greater than zero")
        if not getattr(self, "_is_fitted", False):
            raise AttributeError("Model has not been fitted yet; call fit() first.")
        configuration = self._fitted_model_configuration
        self._validate_explicit_target_path(data, forecast_origin, steps)
        fitted_inputs = configuration.inputs
        effective_X_imputation = fitted_inputs.X_imputation
        if getattr(self, "_formula", None):
            data = data.subset(fitted_inputs.y_variables, fitted_inputs.X_variables)
        forecast_origin = (
            forecast_origin if forecast_origin is not None else data.index("y")[-1]
        )
        mapping = fitted_inputs.mapping
        data = data.with_fitted_metadata(self._raw_data)
        prepared = data.transform(mapping, combine=True).regularise()
        if effective_X_imputation is not None and self._needs_ragged_edge_imputation:
            target_frequency = configuration.design.frequency
            target_period = pd.Period(forecast_origin, freq=target_frequency) + steps
            target_date = target_period.to_timestamp(how="end").normalize()
            prepared = prepared.impute(target_date, method=effective_X_imputation)
        y_input, X_input = self._prepare_forecast_inputs(
            prepared.to_wide("y"), prepared.to_wide("X")
        )

        if getattr(self, "_formula", None) and y_input is not None:
            y_input = self._formula.extract_y(y_input)

        design = configuration.design
        has_lags = bool(design.y_lags or any(lag for _, lag in design.X_lags))
        y_design_input, X_design_input = y_input, X_input
        if y_design_input is None or (has_lags and X_design_input is None):
            history = data.history().transform(mapping).regularise()
            if y_design_input is None:
                y_design_input = history.to_wide("y")
                if getattr(self, "_formula", None):
                    y_design_input = self._formula.extract_y(y_design_input)
            if has_lags and X_design_input is None:
                _, X_design_input = self._prepare_forecast_inputs(
                    None, history.to_wide("X")
                )
        design_index = None
        if X_input is None and (has_lags or design.dummies):
            design_index = self._forecast_dates(forecast_origin, steps)
        X_design = design.build(
            y_design_input,
            X_design_input,
            index=design_index,
            formula=getattr(self, "_formula", None),
        )

        if not self._handles_missing_values and X_design is not None:
            X_design = X_design.dropna()

        # Pass full history to _forecast; models handle filtering to forecast rows
        hook_kwargs = dict(kwargs)
        hook_kwargs["forecast_origin"] = forecast_origin
        if probabilities is not None:
            hook_kwargs["quantiles"] = probabilities
        forecast = self._forecast(steps=steps, X=X_design, y=y_input, **hook_kwargs)
        return self._finalise_forecast(
            forecast,
            steps,
            forecast_origin,
            decomp=decomp,
            decomp_kwargs={"X": X_design, "y": y_input, **hook_kwargs},
            quantiles=probabilities,
        )
