import copy
import os
import pickle
import warnings
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral

import numpy as np
import pandas as pd
from forecast_evaluation import ForecastData
from tqdm import tqdm

from ._realtime_forecasting import ForecastRunResult, ForecastTask
from .data_transformation import (
    DataTransformationPipeline,
    infer_long_variable_frequencies,
)
from .forecast_model import (
    X_IMPUTATION_METHODS,
    ForecastContext,
    ForecastModel,
    NoUsableTransformedYError,
)

_DERIVABLE_FROM_LEVELS = frozenset({"levels", "logs", "log diff", "diff", "pop", "yoy"})


def _select_input_metrics(
    data: pd.DataFrame,
    variables: list[str],
    requested_metrics: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Select one deterministic input metric for each requested variable."""
    if "metric" not in data.columns:
        return data.copy(), {variable: "levels" for variable in variables}

    requested_metrics = requested_metrics or {}
    selected_metrics = {}
    available_variables = set(data["variable"])

    for variable in variables:
        if variable not in available_variables:
            continue

        available = sorted(data.loc[data["variable"] == variable, "metric"].unique())
        requested = requested_metrics.get(variable)
        if requested is not None:
            if requested in available:
                selected = requested
            elif "levels" in available and requested in _DERIVABLE_FROM_LEVELS:
                selected = "levels"
            else:
                raise ValueError(
                    f"Cannot select input metric for variable '{variable}': "
                    f"requested metric '{requested}' is unavailable; available "
                    f"metrics: {available}."
                )
        elif len(available) == 1:
            selected = available[0]
        else:
            raise ValueError(
                f"Input metrics for variable '{variable}' are ambiguous; "
                f"available metrics: {available}."
            )
        selected_metrics[variable] = selected

    metric_by_variable = data["variable"].map(selected_metrics)
    selected_rows = data.loc[
        data["variable"].isin(selected_metrics) & data["metric"].eq(metric_by_variable)
    ].copy()
    return selected_rows, selected_metrics


def _select_tree_input_metrics(
    data: pd.DataFrame,
    variables: list[str],
    requirements: dict[str, tuple[str, ...]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Select a source that can be dispatched to heterogeneous tree leaves."""
    selected_variables = [
        variable for variable in variables if requirements.get(variable)
    ]
    requested = {}
    for variable in selected_variables:
        options = requirements.get(variable, ())
        if len(options) <= 1:
            if options:
                requested[variable] = options[0]
            continue

        available = set(data.loc[data["variable"] == variable, "metric"])
        if "levels" in available:
            requested[variable] = "levels"
            continue

        native = sorted(available.intersection(options))
        if len(native) == 1:
            requested[variable] = native[0]
            continue
        raise ValueError(
            f"Cannot select a common raw metric for tree variable '{variable}': "
            f"leaves require {list(options)}, available metrics are "
            f"{sorted(available)}. Retain levels or provide one native source."
        )

    return _select_input_metrics(data, selected_variables, requested)


def _run_forecast_task(task: ForecastTask) -> ForecastRunResult:
    """Run one pickleable task in either sequential or spawned execution."""
    result = _loop_through_vintages(
        model=task.model,
        data_transformation=task.data_transformation,
        input_metrics=task.input_metrics,
        y_input_metrics=task.y_input_metrics,
        X_input_metrics=task.X_input_metrics,
        y_conditioning_input_metrics=task.y_conditioning_input_metrics,
        X_conditioning_input_metrics=task.X_conditioning_input_metrics,
        vintages=task.vintages,
        **task.common,
    )
    return ForecastRunResult(*result)


def _validate_conditioning(role, selected_variables, horizons, sources, steps):
    horizon_name = f"{role}_steps_ahead"
    source_name = f"{role}_sources"

    if horizons is not None:
        if selected_variables is None:
            raise ValueError(
                f"{role}_variables must be provided when {horizon_name} is specified."
            )

        if not set(horizons.keys()).issubset(set(selected_variables)):
            extra_keys = set(horizons.keys()) - set(selected_variables)
            raise ValueError(
                f"Keys of {horizon_name} must be a subset of {role}_variables. "
                f"Extra keys: {extra_keys}"
            )

        invalid = [
            horizon
            for horizon in horizons.values()
            if horizon is not None
            and (type(horizon) is not int or not 0 <= horizon < steps)
        ]
        if invalid:
            raise ValueError(
                f"{horizon_name} values must be None or integers in the range "
                f"0..{steps - 1}; got {invalid}"
            )

        if sources is None:
            raise ValueError(
                f"{source_name} must be provided when {horizon_name} is specified."
            )

        if set(sources.keys()) != set(horizons.keys()):
            raise ValueError(
                f"Keys of {source_name} must match {horizon_name} exactly. "
                f"Got {set(sources.keys())}, "
                f"but expected {set(horizons.keys())}"
            )

    if horizons is None and sources is not None:
        raise ValueError(
            f"{source_name} is provided but {horizon_name} is None. "
            f"Please provide {horizon_name} to use {source_name}."
        )


def _expected_conditioning_dates(first_forecast_date, steps, frequency):
    first_period = pd.Period(first_forecast_date, freq=frequency)
    return (
        pd.period_range(first_period, periods=steps, freq=frequency)
        .to_timestamp(how="end")
        .normalize()
    )


def _conditioning_dates_by_variable(
    first_forecast_date,
    steps,
    target_frequency,
    variables,
    frequency_map,
):
    """Build target-horizon date groups at each conditioned variable's frequency."""
    target_period = pd.Period(first_forecast_date, freq=target_frequency)
    target_periods = pd.period_range(target_period, periods=steps, freq=target_frequency)
    dates_by_variable = {}
    for variable in variables:
        variable_frequency = frequency_map[variable]
        dates_by_variable[variable] = [
            pd.period_range(
                target_period.asfreq(variable_frequency, how="start")
                if target_periods.empty
                else period.asfreq(variable_frequency, how="start"),
                period.asfreq(variable_frequency, how="end"),
                freq=variable_frequency,
            )
            .to_timestamp(how="end")
            .normalize()
            for period in target_periods
        ]
    return dates_by_variable


def _resolve_step_frequency(
    input_frequencies: dict[str, str],
    y_variables: list[str],
    step_frequency: str | None,
) -> str:
    """Resolve the horizon frequency from the selected target frequencies."""
    if step_frequency is not None:
        return step_frequency

    frequencies = {variable: [input_frequencies[variable]] for variable in y_variables}
    unique_frequencies = {
        frequency for values in frequencies.values() for frequency in values
    }
    if len(unique_frequencies) != 1 or any(
        len(values) != 1 for values in frequencies.values()
    ):
        raise ValueError(
            "step_frequency must be provided when y_variables have mixed or "
            f"ambiguous frequencies; found {frequencies}."
        )
    return next(iter(unique_frequencies))


class RealTimeModel:
    """A class for producing real-time forecasts."""

    def __init__(
        self,
        data,
        models: ForecastModel | list[ForecastModel] = None,
    ):
        """
        Initialise RealTimeModel with a ForecastData object.

        Args:
            data : ForecastData
                An instance of the ForecastData class
            models : ForecastModel or list[ForecastModel]
                A single forecasting model or a list of models. Labels come from each
                model's `label` attribute, or from its class name when no label is set.
                Labels become the `source` in forecasts.
        """
        if not isinstance(data, ForecastData):
            raise TypeError("data must be an instance of ForecastData")

        if models is None:
            raise TypeError("'models' argument is required")

        # Accept single model or list of models
        if isinstance(models, list):
            for m in models:
                if not isinstance(m, ForecastModel):
                    raise TypeError(
                        f"All models in list must be ForecastModel instances. "
                        f"Got {type(m)}"
                    )
            self.models = models
        elif isinstance(models, ForecastModel):
            self.models = [models]
        else:
            raise TypeError(
                "models must be a ForecastModel instance or list[ForecastModel]"
            )

        # Validate that all model labels are unique
        labels = [m.label for m in self.models]
        if len(labels) != len(set(labels)):
            duplicates = [label for label in set(labels) if labels.count(label) > 1]
            raise ValueError(
                f"Model labels must be unique. Found duplicates: {duplicates}. "
                f"Use the 'label' argument when creating models to differentiate them, "
                f"e.g., ForecastRidge(label='Ridge-1'), ForecastRidge(label='Ridge-2')"
            )

        self.data = data
        self.decompositions = None
        self.native_forecasts = None

    def forecast(
        self,
        y_variables: list[str],
        step_frequency: str | None = None,
        data_transformation: dict[str, str] | None = None,
        label: str | None = None,
        steps: int = 1,
        first_forecast_horizon: dict[str, int] | int | None = None,
        X_variables: list[str] | None = None,
        y_steps_ahead: dict[str, int | None] | None = None,
        y_sources: dict[str, str] | None = None,
        X_steps_ahead: dict[str, int | None] | None = None,
        X_sources: dict[str, str] | None = None,
        y_lags: int = 0,
        X_lags: int | dict = 0,
        dummies: list | dict | None = None,
        first_vintage: str | None = None,
        last_vintage: str | None = None,
        reconstruct_levels: bool = True,
        parallel: bool = False,
        batch_size: int | None = None,
        max_workers: int | None = None,
        decomp: bool = False,
        X_imputation: str | None = None,
        drop_transformation_nans: bool = True,
        **kwargs,
    ):
        """Produce forecasts in real-time.

        Args:
            y_variables : List[str]
                The labels of the variable(s) to include in y.
                Must be a subset of the variables in the ForecastData object.
            step_frequency : str | None, optional
                The frequency used for forecast steps (e.g. "M" for monthly,
                "Q" for quarterly). When omitted, it is inferred from the
                selected y variables. It must be provided when those variables
                have mixed or ambiguous frequencies.
            data_transformation : dict[str, str] | None, optional
                A dictionary mapping each variable to the type of
                data transformation to apply before forecasting.
                The values must be one of "levels", "pop", "yoy", "logs",
                "log diff" or "diff". Used only as a fallback for models that
                have no model-owned ``data_transformation``; may be
                omitted (default None) when every model's own pipeline covers
                all requested y/X variables. Any model without a pipeline that
                covers its own variables raises a clear error naming that
                model.
            label : str | None, optional
                Extra label to name the forecast on top of each model's label attribute.
            steps : int >= 1, optional
                The number of steps ahead to forecast.
                If None, no conditioning is applied.
            first_forecast_horizon : dict[str, int] | int | None, optional
                The first target period to return, measured from the vintage
                period. At the selected frequency, the horizon is ``target
                period - vintage period``: 0 is the vintage period, -1 is one
                period before it, and 1 is one period after it.

                If None, the model is fitted through the latest period for
                which all selected y variables have observations in that
                vintage, and forecasting starts in the next period. If several
                y variables are selected, the one with the shortest available
                history therefore sets this default starting point.

                If an int, the same horizon is used for every y variable. If a
                dict, it maps each y variable to its own first returned
                horizon. The model still uses one fitting cutoff for all y
                variables, set by the smallest value in the dict; returned
                rows are then filtered using each variable's own value.
            X_variables : List[str] | None, optional
                The labels of the variable(s) to include in X.
                Must be a subset of the variables in the ForecastData object.
            y_steps_ahead : dict[str, int | None] | None, optional
                A dictionary mapping y variable names to the number
                of forecast steps ahead to use as conditioning paths.
                Values are zero-based horizons in ``0..steps-1`` or None.
                Keys must be a subset of y_variables.
            y_sources : dict[str, str] | None, optional
                Source of the forecasts to use for conditioning.
                Keys must match y_steps_ahead keys.
            X_steps_ahead : dict[str, int | None] | None, optional
                A dictionary mapping X variable names to the number
                of forecast steps ahead to use as regressor forecasts.
                Values are zero-based horizons in ``0..steps-1`` or None.
                Keys must be a subset of X_variables.
            X_sources : dict[str, str] | None, optional
                Source of the forecasts to use for X regressors.
                Keys must match X_steps_ahead keys.
            first_vintage : str, optional
                The first vintage to use for forecasting. Defaults to None,
                which means the earliest vintage in the data will be used.
            last_vintage : str, optional
                The last vintage to use for forecasting.
                Defaults to None, which means the latest vintage in the data will be used.
            y_lags : int, optional
                Number of autoregressive lags of y to append to X before fitting.
                When provided, overrides any ``y_lags`` set on the model itself.
                Default 0 (no lags).
            X_lags : int or dict[str, int], optional
                Lags of X to append before fitting. An ``int`` applies the same
                lag count to every X column; a ``dict`` maps each column name to
                its lag count. When provided, overrides any ``X_lags`` set on the
                model itself. Default 0 (no lags).
            dummies : list or dict, optional
                Outlier (point) dummies appended to the design matrix at fit and
                forecast time. Either a list of dates (each becomes a 0/1 column
                that is 1 only on that date) or a dict mapping a chosen column
                name to a date. Being deterministic functions of the date index,
                they need no imputation over the forecast horizon. Default None.
            reconstruct_levels : bool, optional
                If True (default), reconstruct levels from logs / log diff / diff
                forecasts when the underlying levels series is available in the
                outturns. Set to False to skip the reconstruction step; native
                "diff"/"log diff"/"logs" forecast rows, which ``ForecastData``
                cannot store directly, are then preserved on
                ``self.native_forecasts`` instead of being dropped.
            parallel : bool, optional
                Parallelisation strategy:
                - False (default): sequential execution over models
                - True: parallelise with ProcessPoolExecutor across
                  (model, vintage_batch) combinations for full 2D scaling.
            batch_size : int | None, optional
                Number of vintages per worker task. If None (default),
                computed as ceil(len(vintages) / num_workers) for optimal
                load balancing. Tune manually for specific hardware/data.
            max_workers : int | None, optional
                Max worker processes for ProcessPoolExecutor (None = all CPUs).
            decomp : bool, optional
                If True, collect decomposition rows from models and augment with
                metadata. Stored in self.decompositions. Default False.
            X_imputation : str or None, optional
                Strategy used to fill missing future regressor values when the
                provided X has fewer rows ahead than ``steps``.
                This is applied only for models whose
                ``_needs_ragged_edge_imputation`` attribute is True. Models
                that set it to False handle their own ragged edge, so this
                option is not applied to them.
                - ``None`` (default): no imputation (X is passed through as-is)
                - ``"zero"``          : fill with 0
                - ``"last"``          : repeat the last observed value (random-walk)
                - ``"mean"``          : fill with the in-sample column mean
                - ``"ar1_t"``         : simulate from an AR(1) fit with Student-t errors
            drop_transformation_nans : bool, optional
                If True (default), drop the first row of ``y``/``X`` for any
                variable using a "diff" or "log diff" ``data_transformation``,
                since that transformation leaves the first observation NaN.
                Set to False to keep it.
            **kwargs : dict
                Additional keyword arguments to pass.
        """
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")

        # Validate the extra forecast label
        if label is not None and not isinstance(label, str):
            raise TypeError(f"label must be a string; got {type(label)}")

        # Validate parallel parameter
        if parallel not in (False, True):
            raise ValueError(f"parallel must be False or True; got {parallel!r}")

        # Decomposition cannot run in parallel: the revision decomposition for a
        # given vintage is derived from the immediately preceding vintage's run,
        # so vintages must be processed sequentially in order.
        if decomp and parallel:
            raise ValueError(
                "decomp=True is not supported with parallel=True: the revision "
                "decomposition for each vintage depends on the previous vintage's "
                "run, which requires sequential processing. Set parallel=False."
            )

        # Validate the future-regressor imputation strategy
        # (None disables imputation entirely)
        if X_imputation not in (None, *X_IMPUTATION_METHODS):
            raise ValueError(
                f"X_imputation must be None or one of {X_IMPUTATION_METHODS}; "
                f"got {X_imputation!r}"
            )

        kwargs = copy.deepcopy(kwargs)
        if parallel:
            try:
                pickle.dumps(kwargs)
            except (pickle.PicklingError, TypeError, AttributeError) as error:
                raise TypeError(
                    "Additional model options must be pickleable when parallel=True."
                ) from error

        # outturns = self.data.outturns.copy()
        # forecasts = self.data.forecasts.copy()
        # TODO: remove this raw-table workaround once forecast-evaluation fixes
        # duplicate derived/native outturn metrics and the fixed version is
        # required by this project.
        # ForecastData prepares derived metrics on its public tables. Real-time
        # input selection must use the source rows supplied by the caller so a
        # levels-only panel is transformed by the model pipeline, rather than
        # selecting ForecastData's evaluation-derived metric as native input.
        outturns = getattr(self.data, "_raw_outturns", self.data.outturns).copy()
        forecasts = getattr(self.data, "_raw_forecasts", self.data.forecasts).copy()

        # Validate y_variables and X_variables.
        if not isinstance(y_variables, list):
            raise ValueError("y_variables must be a list of variable names")

        if X_variables is not None and not isinstance(X_variables, list):
            raise ValueError("X_variables must be a list of variable names")

        # Validate y_variables and X_variables
        if not set(y_variables).issubset(outturns["variable"].unique()):
            raise ValueError(
                f"y_variables must be a subset of the variables in the outturns. "
                f"Got {y_variables}, but expected a subset of "
                f"{outturns['variable'].unique()}"
            )

        if X_variables is not None and not set(X_variables).issubset(
            outturns["variable"].unique()
        ):
            raise ValueError(
                f"X_variables must be a subset of the variables in the outturns. "
                f"Got {X_variables}, but expected a subset of "
                f"{outturns['variable'].unique()}"
            )

        _validate_conditioning("y", y_variables, y_steps_ahead, y_sources, steps)
        _validate_conditioning("X", X_variables, X_steps_ahead, X_sources, steps)

        # Normalise first_forecast_horizon to a dict (when provided)
        if first_forecast_horizon is None:
            ffh_dict = None
        else:
            if isinstance(first_forecast_horizon, Integral) and not isinstance(
                first_forecast_horizon, bool
            ):
                ffh_dict = {var: first_forecast_horizon for var in y_variables}
            elif isinstance(first_forecast_horizon, dict):
                ffh_dict = dict(first_forecast_horizon)
            else:
                raise TypeError(
                    "first_forecast_horizon must be an int, a dict mapping "
                    "variable names to int, or None"
                )
            expected_keys = set(y_variables)
            actual_keys = set(ffh_dict)
            if actual_keys != expected_keys:
                missing = expected_keys - actual_keys
                extra = actual_keys - expected_keys
                raise ValueError(
                    "first_forecast_horizon keys must match y_variables exactly. "
                    f"Missing: {missing}; Extra: {extra}"
                )
            invalid_values = {
                variable: value
                for variable, value in ffh_dict.items()
                if isinstance(value, bool) or not isinstance(value, Integral)
            }
            if invalid_values:
                raise TypeError(
                    "first_forecast_horizon values must be non-boolean integers. "
                    f"Invalid values: {invalid_values}"
                )

        # Validate training period
        if first_vintage is not None:
            first_vintage = pd.to_datetime(first_vintage)
        else:
            first_vintage = outturns["vintage_date"].min()

        if last_vintage is not None:
            last_vintage = pd.to_datetime(last_vintage)
        else:
            last_vintage = outturns["vintage_date"].max()

        if last_vintage < first_vintage:
            raise ValueError("first_vintage must be before or equal to last_vintage")

        if first_vintage < outturns["vintage_date"].min():
            raise ValueError(
                "first_vintage cannot be before the earliest date in the outturns"
            )

        if last_vintage > outturns["vintage_date"].max():
            raise ValueError(
                "last_vintage cannot be after the latest date in the outturns"
            )

        all_variables = list(dict.fromkeys(y_variables + (X_variables or [])))
        input_frequencies = infer_long_variable_frequencies(
            outturns, all_variables, "raw inputs"
        )
        step_frequency = _resolve_step_frequency(
            input_frequencies, y_variables, step_frequency
        )

        # keep only relevant variables
        outturns = outturns[
            outturns["variable"].isin(y_variables + (X_variables if X_variables else []))
        ]

        # filter forecasts based on conditioning/regressor sources
        all_forecast_sources = {}
        if y_sources is not None:
            all_forecast_sources.update(y_sources)
        if X_sources is not None:
            all_forecast_sources.update(X_sources)

        if all_forecast_sources and not forecasts.empty:
            mask = forecasts["variable"].isin(all_forecast_sources) & (
                forecasts["source"] == forecasts["variable"].map(all_forecast_sources)
            )
            forecasts = forecasts[mask]

        # Vintage range shared by every model: outturns/forecasts stay raw
        # (untransformed) until each model's own public fit()/forecast() call,
        # so there is no model-specific transformation left to shrink this
        # range up front; any vintage a model cannot usably fit is skipped
        # inside _loop_through_vintages instead.
        vintages = outturns["vintage_date"].unique()
        vintages = np.sort(
            vintages[(vintages >= first_vintage) & (vintages <= last_vintage)]
        )

        # Resolve and select input metrics independently for each model before
        # any realtime vintage deduplication or long-to-wide pivoting.
        all_variables = y_variables + (X_variables or [])
        resolved_model_data = []
        for model in self.models:
            y_conditioning_input_metrics = {}
            X_conditioning_input_metrics = {}
            try:
                pipeline = model.resolve_input_data_transformation(
                    data_transformation,
                    y_variables=y_variables,
                    X_variables=X_variables,
                )
                formula = getattr(model, "_formula", None)
                model_y_variables = model.resolve_target_variables(y_variables)
                model_X_variables = (
                    (
                        [
                            variable
                            for variable in (X_variables or [])
                            if variable in formula.X_cols
                        ]
                        if not formula.has_wildcard
                        else X_variables or []
                    )
                    if formula is not None
                    else X_variables or []
                )
                tree_requirements = getattr(model, "input_metric_requirements", None)
                if tree_requirements is not None:
                    requirements = tree_requirements(
                        y_variables,
                        X_variables,
                        data_transformation=data_transformation,
                    )
                    selected_outturns, input_metrics = _select_tree_input_metrics(
                        outturns, all_variables, requirements
                    )
                else:
                    requested_metrics = (
                        pipeline.data_transformation if pipeline is not None else None
                    )
                    if not requested_metrics and data_transformation is None:
                        available_variables = set(outturns["variable"])
                        level_defaults = {
                            variable: "levels"
                            for variable in [
                                *model_y_variables,
                                *model_X_variables,
                            ]
                            if variable in available_variables
                            and "levels"
                            in set(
                                outturns.loc[outturns["variable"] == variable, "metric"]
                            )
                        }
                        requested_metrics = level_defaults or None
                    selected_outturns, input_metrics = _select_input_metrics(
                        outturns,
                        [*model_y_variables, *model_X_variables],
                        requested_metrics,
                    )
                forecast_variables = []
                if "variable" in forecasts.columns:
                    conditioning_variables = set(
                        (y_steps_ahead or {}) | (X_steps_ahead or {})
                    )
                    forecast_variables = [
                        variable
                        for variable in [*model_y_variables, *model_X_variables]
                        if variable in conditioning_variables
                        and variable in set(forecasts["variable"])
                    ]
                if tree_requirements is not None:
                    selected_forecasts, conditioning_input_metrics = (
                        _select_tree_input_metrics(
                            forecasts,
                            forecast_variables,
                            requirements,
                        )
                    )
                else:
                    selected_forecasts, conditioning_input_metrics = (
                        _select_input_metrics(
                            forecasts, forecast_variables, requested_metrics
                        )
                    )
                y_conditioning_input_metrics = {
                    variable: metric
                    for variable, metric in conditioning_input_metrics.items()
                    if variable in y_variables
                }
                X_conditioning_input_metrics = {
                    variable: metric
                    for variable, metric in conditioning_input_metrics.items()
                    if variable in (X_variables or [])
                }
            except ValueError as error:
                raise ValueError(f"Model '{model.label}': {error}") from error
            effective_transformation = (
                pipeline.data_transformation if pipeline is not None else None
            )
            resolved_model_data.append(
                (
                    model,
                    effective_transformation,
                    selected_outturns,
                    selected_forecasts,
                    input_metrics,
                    y_conditioning_input_metrics,
                    X_conditioning_input_metrics,
                )
            )

        # Inject lag parameters into kwargs so they are forwarded to model._fit()
        if not isinstance(y_lags, int) or y_lags < 0:
            raise ValueError("y_lags must be a non-negative integer")
        if not isinstance(X_lags, (int, dict)):
            raise ValueError("X_lags must be an int or a dict")
        if isinstance(X_lags, int) and X_lags < 0:
            raise ValueError("X_lags must be non-negative when specified as int")
        if isinstance(X_lags, dict) and any(
            not isinstance(v, int) or v < 0 for v in X_lags.values()
        ):
            raise ValueError("X_lags dict values must be non-negative integers")

        # Shared forecast arguments contain raw outturns and conditioning forecasts.
        # The realtime loop selects vintage-specific paths; ForecastModel owns
        # transformation, lag construction, formula selection, and design matrices.
        common = dict(
            outturns=outturns,
            forecasts=forecasts,
            y_variables=y_variables,
            X_variables=X_variables,
            y_steps_ahead=y_steps_ahead,
            X_steps_ahead=X_steps_ahead,
            steps=steps,
            label=label,
            first_forecast_horizon=ffh_dict,
            frequency=step_frequency,
            y_lags=y_lags,
            X_lags=X_lags,
            dummies=dummies,
            decomp=decomp,
            X_imputation=X_imputation,
            input_frequencies=input_frequencies,
            drop_transformation_nans=drop_transformation_nans,
            **kwargs,
        )

        tasks = self._build_forecast_tasks(
            resolved_model_data,
            vintages,
            common,
            batch_size=batch_size,
            parallel=parallel,
            max_workers=max_workers,
        )
        task_results = self._execute_forecast_tasks(
            tasks,
            parallel=parallel,
            max_workers=max_workers,
        )
        result = self._aggregate_forecast_results(
            task_results,
            outturns=outturns,
            y_variables=y_variables,
            X_variables=X_variables,
            frequency=step_frequency,
            reconstruct_levels=reconstruct_levels,
            first_vintage=first_vintage,
            last_vintage=last_vintage,
        )

        self.data.add_forecasts(
            result.forecasts,
            compute_levels=reconstruct_levels,
        )
        self.decompositions = result.decompositions
        self.native_forecasts = result.native_forecasts
        self.first_vintage = first_vintage
        self.last_vintage = last_vintage
        self.y_lags = copy.deepcopy(y_lags)
        self.X_lags = copy.deepcopy(X_lags)
        self.dummies = copy.deepcopy(dummies)
        self.kwargs = copy.deepcopy(kwargs)
        return self

    @staticmethod
    def _build_forecast_tasks(
        resolved_model_data,
        vintages,
        common,
        *,
        batch_size,
        parallel,
        max_workers,
    ):
        """Build the same worker tasks for sequential and parallel runs."""
        if parallel:
            num_workers = max(max_workers or os.cpu_count() or 1, 1)
            batch_size = batch_size or max(
                1, (len(vintages) + num_workers - 1) // num_workers
            )
        else:
            batch_size = len(vintages) or 1

        batches = [
            vintages[start : start + batch_size]
            for start in range(0, len(vintages), batch_size)
        ] or [vintages]
        tasks = []
        y_variables = common["y_variables"]
        X_variables = common["X_variables"]
        for (
            model,
            data_transformation,
            model_outturns,
            model_forecasts,
            input_metrics,
            y_conditioning_input_metrics,
            X_conditioning_input_metrics,
        ) in resolved_model_data:
            model_common = copy.deepcopy(common)
            model_common.update(
                outturns=model_outturns,
                forecasts=model_forecasts,
            )
            tasks.extend(
                ForecastTask(
                    model=model,
                    data_transformation=data_transformation,
                    vintages=batch,
                    common=copy.deepcopy(model_common),
                    input_metrics=input_metrics,
                    y_input_metrics={
                        variable: metric
                        for variable, metric in input_metrics.items()
                        if variable in y_variables
                    },
                    X_input_metrics={
                        variable: metric
                        for variable, metric in input_metrics.items()
                        if variable in (X_variables or [])
                    },
                    y_conditioning_input_metrics=y_conditioning_input_metrics,
                    X_conditioning_input_metrics=X_conditioning_input_metrics,
                )
                for batch in batches
            )
        return tasks

    @staticmethod
    def _execute_forecast_tasks(tasks, *, parallel, max_workers):
        """Schedule tasks without aggregating or publishing their results."""
        if not parallel:
            return [_run_forecast_task(task) for task in tasks]

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_run_forecast_task, task) for task in tasks]
            return [future.result() for future in futures]

    @staticmethod
    def _aggregate_forecast_results(
        task_results,
        *,
        outturns,
        y_variables,
        X_variables,
        frequency,
        reconstruct_levels,
        first_vintage,
        last_vintage,
    ):
        """Aggregate completed worker output without mutating the realtime model."""
        if not task_results or all(
            result.all_vintages_skipped for result in task_results
        ):
            _raise_no_forecasts_error(
                y_variables,
                X_variables,
                np.array([first_vintage, last_vintage]),
            )

        forecasts = pd.concat(
            [result.forecasts for result in task_results], ignore_index=True
        )
        decompositions_list = [
            result.decompositions
            for result in task_results
            if result.decompositions is not None
        ]
        decompositions = (
            pd.concat(decompositions_list, ignore_index=True)
            if decompositions_list
            else None
        )

        if reconstruct_levels:
            forecasts = _reconstruct_forecasts_from_metrics(
                forecasts, outturns, frequency
            )
            native_forecasts = None
        else:
            native_forecasts = forecasts[
                ~forecasts["metric"].isin(["levels", "pop", "yoy"])
            ].copy()
            native_forecasts = native_forecasts if not native_forecasts.empty else None

        stored_forecasts = forecasts[
            forecasts["metric"].isin(["levels", "pop", "yoy"])
        ].copy()
        return ForecastRunResult(
            forecasts=stored_forecasts,
            decompositions=decompositions,
            all_vintages_skipped=False,
            native_forecasts=native_forecasts,
        )


def _loop_through_vintages(
    outturns,
    forecasts,
    model,
    y_variables,
    X_variables,
    y_steps_ahead,
    X_steps_ahead,
    steps,
    first_forecast_horizon,
    label,
    vintages,
    frequency,
    data_transformation,
    X_imputation,
    input_frequencies,
    y_lags=0,
    X_lags=0,
    dummies=None,
    decomp=False,
    drop_transformation_nans=True,
    input_metrics=None,
    y_input_metrics=None,
    X_input_metrics=None,
    y_conditioning_input_metrics=None,
    X_conditioning_input_metrics=None,
    **kwargs,
):
    """Loop through vintages and produce forecasts.

    The loop selects raw outturns and conditioning forecasts and constructs
    vintage-specific paths. Each model's public ``fit()``/``forecast()`` owns
    transformation, lag construction, formula selection, and design matrices.

    Returns:
        tuple: (forecasts_df, decomp_df_or_None, all_vintages_skipped)
    """
    if y_input_metrics is None and input_metrics is not None:
        y_input_metrics = {
            variable: metric
            for variable, metric in input_metrics.items()
            if variable in y_variables
        }
    if X_input_metrics is None and input_metrics is not None:
        X_input_metrics = {
            variable: metric
            for variable, metric in input_metrics.items()
            if variable in (X_variables or [])
        }

    # Get the target vintages.
    y_all_vintages = outturns.copy()
    y_all_vintages = y_all_vintages[y_all_vintages["variable"].isin(y_variables)]

    # Get regressor vintages for fitting.
    if X_variables is not None:
        X_fit_all_vintages = outturns.copy()
        X_fit_all_vintages = X_fit_all_vintages[
            X_fit_all_vintages["variable"].isin(X_variables)
        ]

    # Get conditioning forecasts for target variables.
    if y_steps_ahead is not None and forecasts is not None:
        y_cond_variables = list(y_steps_ahead.keys())
        y_cond_all_vintages = forecasts.copy()
        y_cond_all_vintages = y_cond_all_vintages[
            y_cond_all_vintages["variable"].isin(y_cond_variables)
        ]
    else:
        y_cond_all_vintages = None

    # Get regressor forecasts for models that support them.
    if X_steps_ahead is not None and forecasts is not None:
        X_cond_variables = list(X_steps_ahead.keys())
        X_cond_all_vintages = forecasts.copy()
        X_cond_all_vintages = X_cond_all_vintages[
            X_cond_all_vintages["variable"].isin(X_cond_variables)
        ]
    else:
        X_cond_all_vintages = None

    min_ffh = min(first_forecast_horizon.values()) if first_forecast_horizon else 0

    forecasts_list = []
    forecast_metrics = None
    decomp_rows = []  # Collect decomposition rows if decomp=True
    prev_vintage_state = {}  # Track previous vintage's model state and decomp

    for vintage in tqdm(vintages, desc=f"Running {model.label}"):
        # target variable: select closest vintage <= current vintage
        y_vintage = y_all_vintages[y_all_vintages["vintage_date"] <= vintage].copy()
        y_vintage = y_vintage.sort_values(
            "vintage_date", ascending=False
        ).drop_duplicates(subset=["date", "variable"], keep="first")
        y_vintage = y_vintage.pivot(index="date", columns="variable", values="value")
        vintage_period = pd.Period(vintage, freq=frequency)

        model_formula = getattr(model, "_formula", None)
        y_vintage_for_cutoff = (
            model_formula.extract_y(y_vintage) if model_formula else y_vintage
        )

        # Compute min_ffh for this vintage
        if first_forecast_horizon is None and not y_vintage.empty:
            y_vintage_no_nan = y_vintage_for_cutoff.dropna()  # Drops rows with ANY NaN
            if not y_vintage_no_nan.empty:
                min_last_date = y_vintage_no_nan.index.max()
            else:
                min_last_date = y_vintage.index.min()
            min_last_period = pd.Period(min_last_date, freq=frequency)
            min_ffh_vintage = (min_last_period - vintage_period).n + 1
        else:
            min_ffh_vintage = min_ffh

        last_observed_period = vintage_period + min_ffh_vintage - 1
        last_observed_date = last_observed_period.to_timestamp(how="end").normalize()
        first_forecast_date = (
            (vintage_period + min_ffh_vintage).to_timestamp(how="end").normalize()
        )
        y_fit = y_vintage[y_vintage.index <= last_observed_date]
        expected_y_dates = _expected_conditioning_dates(
            first_forecast_date, steps, frequency
        )
        y_forecasts = y_vintage.reindex(expected_y_dates)

        if y_fit.empty:
            warnings.warn(
                f"No usable transformed y: no y data available for variables "
                f"{y_variables} at vintage {vintage}. "
                "Skipping this vintage.",
                UserWarning,
            )
            continue

        if model_formula:
            target_rows = model_formula.extract_y(y_fit).notna().any(axis=1)
            y_fit = y_fit.loc[target_rows]
            if y_fit.empty:
                warnings.warn(
                    f"No usable transformed y: formula target is unavailable "
                    f"at vintage {vintage}. Skipping this vintage.",
                    UserWarning,
                )
                continue

        # regressor: select closest vintage <= current vintage
        if X_variables is not None:
            X_fit_vintage = X_fit_all_vintages[
                X_fit_all_vintages["vintage_date"] <= vintage
            ].copy()
            X_fit_vintage = X_fit_vintage.sort_values(
                "vintage_date", ascending=False
            ).drop_duplicates(subset=["date", "variable"], keep="first")
            X_fit = X_fit_vintage.pivot(index="date", columns="variable", values="value")
            if X_fit.empty:
                warnings.warn(
                    f"No X data available for variables {X_variables} "
                    f"at vintage {vintage}. Skipping this vintage.",
                    UserWarning,
                )
                continue

            # Raw dates, rather than ForecastData metadata, determine each
            # column's frequency during model input preparation.
        else:
            X_fit = None

        y_model = y_fit
        X_model = X_fit
        X_frequency_map = {
            variable: input_frequencies[variable]
            for variable in (X_steps_ahead or {})
            if variable in input_frequencies
        }

        # Create a copy of the forecast model for this vintage to avoid
        # state modification carrying over to the next iteration
        model_vintage = copy.deepcopy(model)

        # =======================
        # Fitting
        # =======================
        # Estimate the model with target-only y and the selected regressors.
        try:
            model_vintage.fit(
                y=y_model,
                X=X_model,
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
        except NoUsableTransformedYError:
            warnings.warn(
                f"No usable transformed y observations at vintage {vintage}; "
                "skipping this vintage.",
                UserWarning,
            )
            continue
        model_target_variables = list(model_vintage.y.columns)
        forecast_metrics = model_vintage.native_metric_mapping(
            target_variables=model_target_variables
        )
        y_forecast_base = y_model
        if not model_vintage._handles_missing_values:
            y_forecast_base = y_model.loc[y_model.index <= model_vintage.last_y_fit_date]

        # Forecasts
        # =======================
        # Conditioning on Y
        # =======================

        # Build the conditioning paths if provided
        if y_cond_all_vintages is not None:
            y_cond_vintage = y_cond_all_vintages[
                y_cond_all_vintages["vintage_date"] <= vintage
            ].copy()
            y_cond_vintage = y_cond_vintage.sort_values(
                "vintage_date", ascending=False
            ).drop_duplicates(subset=["date", "variable"], keep="first")
            y_cond_vintage = y_cond_vintage.pivot(
                index="date", columns="variable", values="value"
            )
            y_cond_vintage.index = pd.to_datetime(y_cond_vintage.index).normalize()
            # filter out variables which have been selected but not
            # available in this vintage
            y_cond_available = y_cond_vintage.columns.tolist()
            y_steps_ahead_available = {
                var: steps_ahead
                for var, steps_ahead in y_steps_ahead.items()
                if var in y_cond_available
            }
            y_cond = y_cond_vintage.reindex(expected_y_dates)
            y_forecasts = y_forecasts.reindex(columns=y_forecast_base.columns)

            # Explicit conditioning forecasts take precedence over published values.
            for var, steps_ahead in y_steps_ahead_available.items():
                if steps_ahead is not None and var in y_forecasts.columns:
                    conditioned_dates = expected_y_dates[: steps_ahead + 1]
                    published = y_forecasts.loc[conditioned_dates, var]
                    y_forecasts.loc[conditioned_dates, var] = y_cond.loc[
                        conditioned_dates, var
                    ].combine_first(published)

        if y_forecasts.isna().all().all():
            y_forecasts = None
        y_forecast_input_metrics = (
            {
                **(y_input_metrics or {}),
                **(y_conditioning_input_metrics or {}),
            }
            if y_forecasts is not None
            else None
        )

        # =======================
        # Conditioning on X
        # =======================

        # Build the X regressor forecasts if provided (OLS/MIDAS)
        # TODO: This is not tested
        if X_cond_all_vintages is not None:
            X_fcst_vintage = X_cond_all_vintages[
                X_cond_all_vintages["vintage_date"] <= vintage
            ].copy()
            X_fcst_vintage = X_fcst_vintage.sort_values(
                "vintage_date", ascending=False
            ).drop_duplicates(subset=["date", "variable"], keep="first")
            X_fcst_vintage = X_fcst_vintage.pivot(
                index="date", columns="variable", values="value"
            )
            X_fcst_vintage.index = pd.to_datetime(X_fcst_vintage.index).normalize()
            conditioning_dates = _conditioning_dates_by_variable(
                first_forecast_date,
                steps,
                frequency,
                list(X_steps_ahead),
                X_frequency_map,
            )
            expected_dates = pd.DatetimeIndex(
                sorted(
                    {
                        date
                        for dates in conditioning_dates.values()
                        for horizon_dates in dates
                        for date in horizon_dates
                    }
                )
            )

            # filter out variables which have been selected but not
            # available in this vintage
            X_fcst_available = X_fcst_vintage.columns.tolist()
            X_steps_ahead_available = {
                var: sa for var, sa in X_steps_ahead.items() if var in X_fcst_available
            }
            X_fcst = X_fcst_vintage.reindex(expected_dates)

            # Create regressor forecast DataFrame with the complete expected date grid
            # matching X_fit column order
            X_columns = list(X_model.columns)
            X_forecast = pd.DataFrame(np.nan, index=expected_dates, columns=X_columns)

            # Fill in the forecasts using X_steps_ahead
            for var, sa in X_steps_ahead_available.items():
                if sa is not None and var in X_columns:
                    conditioned_dates = pd.DatetimeIndex(
                        [
                            date
                            for horizon_dates in conditioning_dates[var][: sa + 1]
                            for date in horizon_dates
                        ]
                    )
                    X_forecast.loc[conditioned_dates, var] = X_fcst.loc[
                        conditioned_dates, var
                    ]
        else:
            X_forecast = X_model

        # =======================
        # Forecasting
        # =======================

        # Forecast dates come from the model and may reflect publication lags.
        model_result = model_vintage.predict(
            ForecastContext(
                y_history=model_vintage._raw_y_history,
                X_history=model_vintage._raw_X_history,
                y_conditioning=y_forecasts,
                X_conditioning=X_forecast,
                forecast_origin=model_vintage.last_y_fit_date,
                y_conditioning_input_metrics=y_forecast_input_metrics,
                X_conditioning_input_metrics=(
                    X_conditioning_input_metrics
                    if X_cond_all_vintages is not None
                    else X_input_metrics
                ),
            ),
            steps=steps,
            decomp=decomp,
            data_transformation=data_transformation,
            frequency=frequency,
            X_imputation=X_imputation,
            **kwargs,
        )
        model_forecast = model_result.forecast

        # =======================
        # Forecast decomposition
        # =======================
        # Collect decompositions if requested
        if decomp and model_result.decomposition is not None:
            label_decomp = model_vintage.label
            if label is not None:
                label_decomp += " - " + label
            row_decomp = _augment_level_decomp(
                model_result.decomposition,
                dates=model_forecast.index,
                vintage=vintage,
                label=label_decomp,
                y_variables=model_target_variables,
                data_transformation=forecast_metrics,
                frequency=input_frequencies,
            )
            decomp_rows.append(row_decomp)

            # Revision decompositions: derived purely from level decomps of the
            # current and previous vintages (Level 1) plus two counterfactual
            # level decomp from the two fitted models (Level 2). No model
            # internals are touched — only the model's own forecast(decomp=True).
            if prev_vintage_state:
                revision_rows = _compute_revision_decompositions(
                    current_decomp=row_decomp,
                    current_model=model_vintage,
                    current_state={
                        "y_history": model_vintage._raw_y_history,
                        "X_history": model_vintage._raw_X_history,
                        "y_conditioning": y_forecasts,
                        "X_conditioning": X_forecast,
                        "y_conditioning_input_metrics": y_forecast_input_metrics,
                        "X_conditioning_input_metrics": (
                            X_conditioning_input_metrics
                            if X_cond_all_vintages is not None
                            else X_input_metrics
                        ),
                        "forecast_origin": model_vintage.last_y_fit_date,
                    },
                    current_dates=model_forecast.index,
                    prev_state=prev_vintage_state,
                    steps=steps,
                    data_transformation=data_transformation,
                    frequency=frequency,
                    X_imputation=X_imputation,
                    **kwargs,
                )
                decomp_rows.extend(revision_rows)

            # Store lightweight state for the next iteration. ``model_vintage``
            # is a fresh deepcopy each loop, so a reference is enough.
            prev_vintage_state = {
                "vintage_date": vintage,
                "model": model_vintage,
                "y_history": model_vintage._raw_y_history.copy(),
                "X_history": (
                    model_vintage._raw_X_history.copy()
                    if model_vintage._raw_X_history is not None
                    else None
                ),
                "y_conditioning": (
                    y_forecasts.copy() if y_forecasts is not None else None
                ),
                "X_conditioning": (X_forecast.copy() if X_forecast is not None else None),
                "y_conditioning_input_metrics": copy.deepcopy(y_forecast_input_metrics),
                "X_conditioning_input_metrics": copy.deepcopy(
                    X_conditioning_input_metrics
                    if X_cond_all_vintages is not None
                    else X_input_metrics
                ),
                "forecast_origin": model_vintage.last_y_fit_date,
                "forecast_index": model_forecast.index,
                "decomp": row_decomp,
            }
            # =======================
            # =======================

        # Validate that this model's requested targets are present in filtered data.
        expected_y_variables = model.resolve_target_variables(y_variables)
        missing_variables = set(expected_y_variables) - set(y_fit.columns)
        if missing_variables:
            warnings.warn(
                f"The following y_variables are not present in vintage {vintage}:"
                f"{missing_variables}. "
                f"Proceeding with available variables only.",
                UserWarning,
            )

        # Build the per-vintage long table. Date comes straight from the
        # model's returned index. forecast_evaluation defines the horizon
        # relative to the final target observation used for fitting.
        forecast_df = model_forecast.copy()
        output_variables = list(forecast_df.columns)
        if first_forecast_horizon is None:
            published = y_vintage.reindex(
                index=forecast_df.index, columns=output_variables
            ).notna()
            forecast_df = forecast_df.mask(published)
        forecast_df = forecast_df.reset_index()  # date column from index
        forecast_df["date"] = pd.to_datetime(forecast_df["date"]).dt.normalize()
        forecast_df["vintage_date"] = vintage
        if model_vintage._forecast_dates_include_origin:
            forecast_df["forecast_horizon"] = np.arange(len(forecast_df))
        else:
            forecast_df["forecast_horizon"] = [
                (pd.Period(d, freq=frequency) - last_observed_period).n - 1
                for d in forecast_df["date"]
            ]

        # save
        forecasts_list.append(forecast_df)

    if not forecasts_list:
        return (
            pd.DataFrame(
                columns=[
                    "date",
                    "vintage_date",
                    "forecast_horizon",
                    "variable",
                    "value",
                    "metric",
                    "source",
                    "frequency",
                ]
            ),
            None,
            True,
        )

    # concatenate all forecasts into a single dataframe
    all_forecasts = pd.concat(forecasts_list, ignore_index=True)

    # reorder columns to have date and vintage_date first
    all_forecasts = all_forecasts[
        ["date", "vintage_date", "forecast_horizon"] + output_variables
    ]

    # melt the dataframe to long format
    all_forecasts = all_forecasts.melt(
        id_vars=["date", "vintage_date", "forecast_horizon"],
        var_name="variable",
        value_name="value",
    )

    # Filter per-variable by the vintage-relative cutoff. The emitted
    # forecast_horizon is relative to last_observed_period, so it must not be
    # compared directly with first_forecast_horizon.
    if first_forecast_horizon is not None:
        all_forecasts["_target_minus_vintage"] = [
            (pd.Period(date, freq=frequency) - pd.Period(vintage_date, freq=frequency)).n
            for date, vintage_date in zip(
                all_forecasts["date"],
                all_forecasts["vintage_date"],
                strict=False,
            )
        ]
        all_forecasts["_min_h"] = all_forecasts["variable"].map(first_forecast_horizon)
        all_forecasts = all_forecasts[
            all_forecasts["_target_minus_vintage"] >= all_forecasts["_min_h"]
        ].copy()
        all_forecasts = all_forecasts.drop(columns=["_target_minus_vintage", "_min_h"])

    all_forecasts["metric"] = all_forecasts["variable"].map(forecast_metrics)

    # add additional columns to work with ForecastData
    label_forecast = model_vintage.label
    if label is not None:
        label_forecast += " - " + label
    all_forecasts["source"] = label_forecast
    all_forecasts["frequency"] = all_forecasts["variable"].map(input_frequencies)

    # drop missing values
    all_forecasts = all_forecasts.dropna()

    # Prepare decompositions if collected
    all_decompositions = None
    if decomp and decomp_rows:
        all_decompositions = pd.concat(decomp_rows, ignore_index=True)

    # Return tuple: (forecasts_df, decomp_df_or_None, all_vintages_skipped)
    return (all_forecasts, all_decompositions, False)


def _reconstruct_forecasts_from_metrics(
    forecasts: pd.DataFrame, outturns: pd.DataFrame, frequency: str | None
) -> pd.DataFrame:
    """Reconstruct level forecasts independently for each model output mapping."""
    reconstructed = []
    for source, source_forecasts in forecasts.groupby("source", sort=False):
        metrics = source_forecasts[["variable", "metric"]].drop_duplicates()
        output_mapping = dict(zip(metrics["variable"], metrics["metric"], strict=False))
        reconstructed.append(
            DataTransformationPipeline(output_mapping).reconstruct_levels(
                forecasts=source_forecasts,
                outturns=outturns,
                y_variables=list(output_mapping),
                frequency=frequency,
            )
        )
    return pd.concat(reconstructed, ignore_index=True) if reconstructed else forecasts


def _raise_no_forecasts_error(y_variables, X_variables, vintages):
    """Raise a useful error when no selected vintage produced a forecast."""
    if len(vintages):
        vintage_range = (
            f"{pd.Timestamp(vintages[0]).date()} to {pd.Timestamp(vintages[-1]).date()}"
        )
    else:
        vintage_range = "the selected range"
    raise ValueError(
        "No forecasts could be produced for y_variables="
        f"{y_variables} and X_variables={X_variables} across {vintage_range}; "
        "every selected vintage was skipped because usable y or X data was "
        "unavailable."
    )


def _augment_level_decomp(
    raw_decomp,
    dates,
    vintage,
    label,
    y_variables,
    data_transformation,
    frequency,
):
    """
    The model returns the minimal columns (``forecast_horizon``,
    ``component``, ``contribution``, ``weight``); here we attach the
    absolute target ``date`` and the metadata the schema requires for a
    ``level`` decomposition (``level`` meaning a single-vintage
    decomposition, as opposed to a ``revision`` one - it says nothing about
    whether the contributions themselves are in levels units).

    A single-target model does not need to identify its own ``variable``;
    a multi-target model must, since RealTimeModel cannot otherwise tell
    which target each row belongs to. ``forecast_metric`` is always the
    model's own native output metric for that variable (e.g. ``"diff"``),
    never assumed to be ``"levels"``, since decomposition contributions are
    never reconstructed to levels.
    """
    out = raw_decomp.copy()
    if "variable" in out.columns:
        unknown = sorted(set(out["variable"].unique()) - set(y_variables))
        if unknown:
            raise ValueError(
                f"Model '{label}' _forecast_decomp() returned decomposition "
                f"rows for variable(s) {unknown}, not in y_variables "
                f"{y_variables}."
            )
    elif len(y_variables) == 1:
        out["variable"] = y_variables[0]
    else:
        raise ValueError(
            f"Model '{label}' forecasts {len(y_variables)} target variables "
            f"{y_variables}, but its _forecast_decomp() did not return a "
            "'variable' column identifying which target each decomposition "
            "row belongs to. Multi-target models must label each row."
        )
    out["vintage_date"] = vintage
    out["decomposition"] = "level"
    out["revision_source"] = pd.NA
    out["revision_source"] = out["revision_source"].astype("string")
    out["forecast_metric"] = out["variable"].map(data_transformation)
    out["base_vintage_date"] = pd.NaT
    out["source"] = label
    out["frequency"] = out["variable"].map(frequency)
    if "news" not in out.columns:
        out["news"] = np.nan
    out["date"] = dates[out["forecast_horizon"].to_numpy()]
    return out


def _level_contributions(
    model,
    y_history,
    X_history,
    y_conditioning,
    X_conditioning,
    forecast_origin,
    steps,
    dates,
    data_transformation,
    frequency,
    X_imputation,
    y_variables,
    y_conditioning_input_metrics=None,
    X_conditioning_input_metrics=None,
    **kwargs,
):
    """Counterfactual level decomposition for an already-fitted ``model``.

    Re-runs the model's own ``forecast(decomp=True)`` for a supplied raw
    history and future conditioning path, then relabels the relative forecast
    horizons onto the absolute target ``dates``. The fitted model is copied
    before its forecast context is installed, so forecast bookkeeping and
    model-specific caches cannot leak into another vintage.

    ``y_variables`` disambiguates rows across targets for a multi-target
    model: used to attach ``variable`` when the model's own
    ``_forecast_decomp()`` did not (single-target models), so every
    downstream merge can join on ``variable`` and never conflate two
    targets' same-named components.

    Returns a DataFrame with columns ``[date, component, variable,
    contribution]`` or ``None`` if the model produced no decomposition.
    """
    # Counterfactuals are deliberately isolated from the fitted model held by
    # the vintage loop. Some model implementations cache forecast state even
    # when their public hook appears read-only.
    counterfactual_model = copy.deepcopy(model)
    result = counterfactual_model.predict(
        ForecastContext(
            y_history=y_history.copy(),
            X_history=X_history.copy() if X_history is not None else None,
            y_conditioning=(
                y_conditioning.copy() if y_conditioning is not None else None
            ),
            X_conditioning=(
                X_conditioning.copy() if X_conditioning is not None else None
            ),
            forecast_origin=forecast_origin,
            y_conditioning_input_metrics=copy.deepcopy(y_conditioning_input_metrics),
            X_conditioning_input_metrics=copy.deepcopy(X_conditioning_input_metrics),
        ),
        steps=steps,
        decomp=True,
        data_transformation=data_transformation,
        frequency=frequency,
        X_imputation=X_imputation,
        **kwargs,
    )

    raw = result.decomposition
    if raw is None:
        return None
    out = raw[["forecast_horizon", "component", "contribution"]].copy()
    out["date"] = dates[out["forecast_horizon"].to_numpy()]
    if "variable" in raw.columns:
        out["variable"] = raw["variable"].to_numpy()
    elif len(y_variables) == 1:
        out["variable"] = y_variables[0]
    else:
        raise ValueError(
            "Counterfactual decomposition covers multiple target variables "
            f"{y_variables} but has no 'variable' column to disambiguate "
            "them; the model's _forecast_decomp() must label each row for "
            "multi-target output."
        )
    return out[["date", "component", "variable", "contribution"]]


def _compute_revision_decompositions(
    current_decomp,
    current_model,
    current_state,
    current_dates,
    prev_state,
    steps,
    data_transformation,
    frequency,
    X_imputation,
    **kwargs,
):
    """Decompose the revision between two consecutive vintages.

    The contract is entirely model-agnostic: every quantity is a *level*
    decomposition produced by the model itself, so the only operation here
    is subtraction.

    Notation, with ``beta`` the fitted parameters and ``X`` the regressor
    paths, ``D(beta, X)`` the level contribution per (date, component):

    - ``A = D(beta_new, X_new)`` — current vintage (``current_decomp``)
    - ``B = D(beta_old, X_old)`` — previous vintage (``prev_state['decomp']``)
    - ``C = D(beta_old, X_new)`` — previous model on the new data
    - ``E = D(beta_new, X_old)`` — current model on the old data

    Level 1 (total revision):  ``revision = A - B``
    Level 2 (attribution):
        ``news         = C - B``  (effect of new data, old parameters)
        ``reestimation = E - B``  (effect of re-estimation, old data)
        ``interaction  = revision - news - reestimation``

    Each is emitted as a ``revision`` row tagged with its
    ``revision_source`` so the three sources sum back to the revision.

    Every merge joins on ``variable`` as well as ``date``/``component``, so a
    multi-target model whose targets share a component name (e.g. both have
    an ``'intercept'``) never has one target's contribution matched against
    the other's.
    """
    prev_decomp = prev_state["decomp"]
    prev_model = prev_state["model"]
    prev_y_history = prev_state["y_history"]
    prev_X_history = prev_state["X_history"]
    prev_y_conditioning = prev_state["y_conditioning"]
    prev_X_conditioning = prev_state["X_conditioning"]
    prev_y_conditioning_input_metrics = prev_state["y_conditioning_input_metrics"]
    prev_X_conditioning_input_metrics = prev_state["X_conditioning_input_metrics"]
    prev_origin = prev_state["forecast_origin"]
    prev_dates = prev_state["forecast_index"]
    prev_vintage_date = prev_state["vintage_date"]
    current_vintage = current_decomp["vintage_date"].iloc[0]
    y_variables = sorted(set(current_decomp["variable"]) | set(prev_decomp["variable"]))
    merge_keys = ["date", "component", "variable"]

    # Level 1: align current (A) and previous (B) level decomps on the
    # overlapping (date, component, variable) triples and subtract.
    A = current_decomp[merge_keys + ["contribution"]].rename(
        columns={"contribution": "c_new_new"}
    )
    B = prev_decomp[merge_keys + ["contribution"]].rename(
        columns={"contribution": "c_old_old"}
    )
    merged = A.merge(B, on=merge_keys, how="outer")
    if merged.empty:
        return []
    merged["c_new_new"] = merged["c_new_new"].fillna(0.0)
    merged["c_old_old"] = merged["c_old_old"].fillna(0.0)

    # Level 2: counterfactual level decomps from the two fitted models.
    C = _level_contributions(
        prev_model,
        current_state["y_history"],
        current_state["X_history"],
        current_state["y_conditioning"],
        current_state["X_conditioning"],
        current_state["forecast_origin"],
        steps,
        current_dates,
        data_transformation,
        frequency,
        X_imputation,
        y_variables,
        y_conditioning_input_metrics=current_state["y_conditioning_input_metrics"],
        X_conditioning_input_metrics=current_state["X_conditioning_input_metrics"],
        **kwargs,
    )
    E = _level_contributions(
        current_model,
        prev_y_history,
        prev_X_history,
        prev_y_conditioning,
        prev_X_conditioning,
        prev_origin,
        steps,
        prev_dates,
        data_transformation,
        frequency,
        X_imputation,
        y_variables,
        y_conditioning_input_metrics=prev_y_conditioning_input_metrics,
        X_conditioning_input_metrics=prev_X_conditioning_input_metrics,
        **kwargs,
    )

    if C is not None:
        merged = merged.merge(
            C.rename(columns={"contribution": "c_old_new"}),
            on=merge_keys,
            how="left",
        )
        merged["c_old_new"] = merged["c_old_new"].fillna(0.0)
    else:
        merged["c_old_new"] = merged["c_old_old"]
    if E is not None:
        merged = merged.merge(
            E.rename(columns={"contribution": "c_new_old"}),
            on=merge_keys,
            how="left",
        )
        merged["c_new_old"] = merged["c_new_old"].fillna(0.0)
    else:
        merged["c_new_old"] = merged["c_old_old"]

    revision = merged["c_new_new"] - merged["c_old_old"]
    news = merged["c_old_new"] - merged["c_old_old"]
    reestimation = merged["c_new_old"] - merged["c_old_old"]
    interaction = revision - news - reestimation

    # Carry forward per-(date, component, variable) metadata from the
    # current level decomp.
    meta_cols = merge_keys + [
        "forecast_horizon",
        "forecast_metric",
        "source",
        "frequency",
        "weight",
    ]
    meta = current_decomp[meta_cols].drop_duplicates(merge_keys)
    previous_meta = prev_decomp[meta_cols].drop_duplicates(merge_keys)
    meta = meta.merge(
        previous_meta,
        on=merge_keys,
        how="outer",
        suffixes=("", "_previous"),
    )
    for column in meta_cols:
        if column in merge_keys:
            continue
        previous_column = f"{column}_previous"
        if previous_column in meta:
            meta[column] = meta[column].combine_first(meta[previous_column])
            meta = meta.drop(columns=previous_column)
    meta["forecast_horizon"] = meta["forecast_horizon"].astype(int)

    def _make_rows(source, values):
        rows = merged[merge_keys].merge(meta, on=merge_keys)
        rows["contribution"] = values.to_numpy()
        # Vectorized computation of shock size: news = contribution / weight
        rows["news"] = np.where(
            (rows["weight"].notna()) & (rows["weight"] != 0),
            rows["contribution"] / rows["weight"],
            np.nan,
        )
        rows["decomposition"] = "revision"
        rows["revision_source"] = source
        rows["base_vintage_date"] = prev_vintage_date
        rows["vintage_date"] = current_vintage
        return rows

    rows = pd.concat(
        [
            _make_rows("news", news),
            _make_rows("reestimation", reestimation),
            _make_rows("interaction", interaction),
        ],
        ignore_index=True,
    )
    return [rows]


def example_ridge():

    import forecast_evaluation as fe

    import forecast_realtime as rt

    # Load ForecastData with FER dataset
    forecast_data = fe.ForecastData(load_fer=True)

    # Create Ridge model with 5-fold cross-validation
    forecast_model = rt.models.ForecastRidge(label="Ridge", cv=5)

    # Create RealTimeModel for real-time forecasting
    rt_model = rt.RealTimeModel(
        data=forecast_data,
        models=[forecast_model],
    )

    # Run real-time forecast
    rt_model.forecast(
        y_variables=["cpisa"],
        data_transformation={"cpisa": "pop"},
        steps=12,
        label="Ridge",
        first_vintage="2015-01-01",
        X_imputation="last",
    )

    rt_model.data.run_dashboard()


def example_bvar():

    import forecast_evaluation as fe

    import forecast_realtime as rt

    # Load ForecastData with FER dataset
    forecast_data = fe.ForecastData(load_fer=True)

    # filter dates
    forecast_data.filter(start_date="1991-01-31")

    # Load a BVAR model with the features you want
    bvar_model = rt.models.ForecastBVAR(
        stationary=True, n_lags=5, mode_only=True, nb_restart=5, covid=True
    )

    # Create RealTimeModel for real-time forecasting
    rt_model = rt.RealTimeModel(
        data=forecast_data,
        models=[bvar_model],
    )

    # Run real-time forecast
    rt_model.forecast(
        y_variables=["cpisa", "unemp", "gdpkp"],
        data_transformation={"cpisa": "pop", "unemp": "levels", "gdpkp": "pop"},
        y_steps_ahead={
            "cpisa": 1,
            "unemp": 0,
        },  # conditioning assumption (quarter-ahead starting with the nowcast)
        y_sources={
            "cpisa": "mpr",
            "unemp": "mpr",
        },  # which forecasts to use as conditioning assumptions
        steps=13,  # forecasts up to 3-year ahead
        label="Conditional BVAR",
        first_vintage="2015-03-31",
    )

    rt_model.data.run_dashboard(host="127.0.0.2")


if __name__ == "__main__":
    example_bvar()
