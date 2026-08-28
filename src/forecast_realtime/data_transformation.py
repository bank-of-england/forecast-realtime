"""Data-transformation utilities and pickleable pipeline classes."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _ordered_trajectory(group: pd.DataFrame) -> pd.DataFrame:
    """Return one chronologically ordered row for each date in a vintage."""
    if "_type" in group.columns:
        trajectory = group.assign(
            _precedence=group["_type"].ne("forecast").astype(int)
        ).sort_values(["date", "_precedence"], kind="stable")
    else:
        trajectory = group.sort_values("date", kind="stable")

    return trajectory.drop_duplicates(subset="date", keep="first").copy()


def _logs_series(values: pd.Series) -> pd.Series:
    """Return the natural logarithm of a series."""
    return np.log(values)


def _difference_series(values: pd.Series, logarithmic: bool = False) -> pd.Series:
    """Return the first difference of a series, optionally after logging it."""
    return (_logs_series(values) if logarithmic else values).diff()


def _growth_series(values: pd.Series, periods: int) -> pd.Series:
    """Return percentage growth over ``periods`` steps."""
    return values.pct_change(periods=periods, fill_method=None) * 100.0


def _reconstruct_additive(last_level: float, changes: pd.Series) -> pd.Series:
    """Reconstruct levels from additive changes."""
    return last_level + changes.cumsum()


def _reconstruct_logarithmic(last_level: float, changes: pd.Series) -> pd.Series:
    """Reconstruct levels from logarithmic changes."""
    return np.exp(np.log(last_level) + changes.cumsum())


def _calendar_align(values: pd.Series, frequency: str) -> pd.Series:
    """Reindex a series to the complete calendar grid at ``frequency``."""
    first = values.first_valid_index()
    last = values.last_valid_index()
    if first is None or last is None:
        return values

    valid_dates = pd.DatetimeIndex(values.dropna().index)
    if valid_dates.is_month_start.all():
        timestamp_anchor = "start"
    elif valid_dates.is_month_end.all():
        timestamp_anchor = "end"
    else:
        raise ValueError(
            "Cannot calendar-align values: dates must consistently use "
            "month-start or month-end anchors."
        )

    period_values = values.groupby(values.index.to_period(frequency), sort=True).last()
    periods = pd.period_range(
        start=period_values.index.min(),
        end=period_values.index.max(),
        freq=frequency,
    )
    complete_index = periods.to_timestamp(how=timestamp_anchor).normalize()
    aligned = period_values.reindex(periods)
    aligned.index = complete_index
    return aligned


def _reconstruct_levels_by_vintage(
    forecasts: pd.DataFrame,
    levels_outturns: pd.DataFrame,
    reconstruct: Callable[[float, pd.Series], pd.Series],
) -> list[pd.DataFrame]:
    """Reconstruct forecast levels independently for each forecast vintage."""
    reconstructed_groups = []

    for vintage, vintage_forecasts in forecasts.groupby("vintage_date", sort=False):
        vintage_forecasts = vintage_forecasts.sort_values("date").copy()

        first_fcst_date = vintage_forecasts["date"].min()
        as_of_levels = levels_outturns[
            levels_outturns["vintage_date"] <= vintage
        ].sort_values("vintage_date", ascending=False)
        as_of_levels = as_of_levels.drop_duplicates(subset="date", keep="first")
        last_level_data = as_of_levels[as_of_levels["date"] < first_fcst_date]

        if last_level_data.empty:
            continue

        last_level = last_level_data.sort_values("date").iloc[-1]["value"]
        vintage_forecasts["value"] = reconstruct(last_level, vintage_forecasts["value"])
        vintage_forecasts["metric"] = "levels"
        reconstructed_groups.append(vintage_forecasts)

    return reconstructed_groups


def difference_by_vintage(data: pd.DataFrame, logarithmic: bool = False) -> pd.DataFrame:
    """Return first differences for each date within each vintage."""
    differenced_groups = []

    for _, group in data.groupby("vintage_date"):
        group = group.copy()
        trajectory = _ordered_trajectory(group)
        frequency = _resolve_frequency(trajectory["frequency"])
        aligned = _calendar_align(trajectory.set_index("date")["value"], frequency)
        differenced = _transform_metric(
            aligned,
            "log diff" if logarithmic else "diff",
            frequency,
        )

        # Preserve both row types for downstream fitting and conditioning, but
        # give overlapping rows the single date-level transformation.
        group["value"] = group["date"].map(differenced)
        differenced_groups.append(group)

    return pd.concat(differenced_groups, ignore_index=True)


def growth_by_vintage(data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Return percentage growth over ``periods`` steps for each vintage."""
    grown_groups = []

    for _, group in data.groupby("vintage_date"):
        group = group.copy()
        trajectory = _ordered_trajectory(group)
        frequency = _resolve_frequency(trajectory["frequency"])
        aligned = _calendar_align(trajectory.set_index("date")["value"], frequency)
        grown = _growth_series(aligned, periods=periods)

        group["value"] = group["date"].map(grown)
        grown_groups.append(group)

    return pd.concat(grown_groups, ignore_index=True)


# Periods per year for each supported frequency, used to resolve the "yoy" lag.
_PERIODS_PER_YEAR = {"M": 12, "Q": 4}


def _resolve_frequency(frequency: pd.Series) -> str:
    """Return the single supported frequency value in a frequency column."""
    unique_frequencies = frequency.dropna().unique()
    if len(unique_frequencies) != 1:
        raise ValueError(
            "Cannot resolve a single frequency for calendar-based transformation; "
            f"got {list(unique_frequencies)}."
        )
    resolved = unique_frequencies[0]
    if resolved not in _PERIODS_PER_YEAR:
        raise ValueError(
            f"Unsupported frequency '{resolved}' for calendar-based transformation; "
            f"expected one of {list(_PERIODS_PER_YEAR)}."
        )
    return resolved


def _periods_per_year(frequency: pd.Series) -> int:
    """Resolve the year-on-year lag from a single-valued ``frequency`` column."""
    return _PERIODS_PER_YEAR[_resolve_frequency(frequency)]


def leading_nan_row_count(frame: pd.DataFrame, columns: list[str]) -> int:
    """Rows to drop so every column in *columns* starts at its first defined value.

    Measured from the actual transformed data rather than assumed from a
    metric's nominal calendar lag: an early missing calendar period can push
    a "diff"/"pop"/"yoy" column's genuine first defined observation later
    than its nominal lag would suggest (or, conversely, may leave fewer leading
    rows are undefined than the nominal lag). Only the *leading* (prefix) run
    of undefined rows is measured per column, so interior missing
    observations are never counted. Returns 0 for a column with no leading
    ``NaN`` (e.g. "levels"/"logs").
    """
    counts = []
    for column in columns:
        values = frame[column]
        first_valid = values.first_valid_index()
        counts.append(
            len(values) if first_valid is None else values.index.get_loc(first_valid)
        )
    return max(counts, default=0)


def apply_transformations(
    data: pd.DataFrame,
    variables: list[str],
    data_transformation: dict[str, str],
) -> pd.DataFrame:
    """Add the requested metrics for variables in long-form data.

    Args:
        data : pd.DataFrame
            The data to transform (outturns or forecasts)
        variables : list[str]
            The variables to check
        data_transformation : dict[str, str]
            Dictionary mapping variables to their required transformations

    Returns:
        pd.DataFrame : Data with computed transformations added
    """
    new_rows = []

    for var in variables:
        required_metric = data_transformation[var]
        var_data = data[data["variable"] == var]
        available_metrics = var_data["metric"].unique()

        # If required metric already exists, skip
        if required_metric in available_metrics:
            continue

        if "levels" in available_metrics and required_metric in _VALID_METRICS:
            levels_data = var_data[var_data["metric"] == "levels"].copy()
            transformed = _transform_long_metric(levels_data, required_metric)
            if required_metric in _CALENDAR_DEPENDENT_METRICS:
                transformed = transformed.dropna(subset=["value"])
            transformed["metric"] = required_metric
            new_rows.append(transformed)
        else:
            raise ValueError(
                f"Cannot compute transformation '{required_metric}'"
                f"for variable '{var}'. "
                f"Available metrics: {list(available_metrics)}. "
                f"Please ensure 'levels' or an appropriate base metric is available."
            )

    # Append new transformations to the data
    if new_rows:
        data = pd.concat([data] + new_rows, ignore_index=True)

    return data


# Frequencies accepted by the raw wide-input methods, matching the "Q"/"M"
# convention used elsewhere in the outturns/forecasts schema.
_VALID_FREQUENCIES = tuple(_PERIODS_PER_YEAR)
_VALID_METRICS = ("levels", "logs", "diff", "log diff", "pop", "yoy")
_CALENDAR_DEPENDENT_METRICS = {"diff", "log diff", "pop", "yoy"}


def _ordered_items(mapping: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    """Freeze a metric mapping in a deterministic order."""
    return tuple(sorted((mapping or {}).items()))


@dataclass(frozen=True)
class ModelInputRequirements:
    """Immutable requested metrics for a model's raw input variables."""

    y: tuple[tuple[str, str], ...] = ()
    X: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mappings(
        cls,
        y: dict[str, str] | None = None,
        X: dict[str, str] | None = None,
    ) -> "ModelInputRequirements":
        return cls(_ordered_items(y), _ordered_items(X))

    @property
    def y_mapping(self) -> dict[str, str]:
        return dict(self.y)

    @property
    def X_mapping(self) -> dict[str, str]:
        return dict(self.X)


@dataclass(frozen=True)
class InputMetricMapping:
    """Immutable source metrics for one input role."""

    values: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, mapping: dict[str, str] | None) -> "InputMetricMapping":
        return cls(_ordered_items(mapping))

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self.values)


@dataclass(frozen=True)
class RawInputBundle:
    """Raw history and conditioning data with role-specific provenance."""

    y_history: pd.DataFrame
    X_history: pd.DataFrame | None = None
    y_conditioning: pd.DataFrame | None = None
    X_conditioning: pd.DataFrame | None = None
    y_history_metrics: InputMetricMapping = InputMetricMapping()
    X_history_metrics: InputMetricMapping = InputMetricMapping()
    y_conditioning_metrics: InputMetricMapping = InputMetricMapping()
    X_conditioning_metrics: InputMetricMapping = InputMetricMapping()


@dataclass(frozen=True)
class PreparedModelInputs:
    """Transformed y/X frames handed to a model preparation hook."""

    y: pd.DataFrame
    X: pd.DataFrame | None
    y_metric: str | None = None
    X_metrics: tuple[tuple[str, str], ...] = ()


def _validate_frequency(frequency: str) -> None:
    if frequency not in _VALID_FREQUENCIES:
        raise ValueError(
            f"Unsupported frequency '{frequency}'; expected one of "
            f"{list(_VALID_FREQUENCIES)}."
        )


def infer_frequency_from_dates(dates: pd.DatetimeIndex, context: str = "data") -> str:
    """Infer the package frequency (``"M"`` or ``"Q"``) from raw dates."""
    if isinstance(dates, pd.PeriodIndex):
        dates = dates.to_timestamp(how="end").normalize()
    else:
        dates = pd.DatetimeIndex(dates)
    dates = dates.dropna().sort_values().unique()
    if len(dates) < 2:
        raise ValueError(
            f"Cannot infer frequency for {context}: at least two non-null "
            "dates are required."
        )
    month_numbers = dates.year * 12 + dates.month
    gaps = np.diff(month_numbers)
    if np.any(gaps <= 0):
        raise ValueError(
            f"Cannot infer supported frequency for {context}: dates do not form "
            "monthly or quarterly observations."
        )

    if not (dates.is_month_end.all() or dates.is_month_start.all()):
        raise ValueError(
            f"Cannot infer supported frequency for {context}: dates must be "
            "month-start or month-end observations."
        )
    quarter_end_months = dates.month.isin([3, 6, 9, 12]).all()
    quarter_start_months = dates.month.isin([1, 4, 7, 10]).all()

    inferred = pd.infer_freq(dates) if len(dates) >= 3 else None
    if inferred is not None:
        offset = pd.tseries.frequencies.to_offset(inferred)
        rule_code = offset.rule_code.upper()
        if rule_code.startswith(("ME", "MS")) and offset.n == 1:
            return "M"
        if rule_code.startswith(("QE", "QS")) and offset.n == 1:
            return "Q"

    if len(dates) == 2:
        gap = int(gaps[0])
        if gap == 1:
            return "M"
        if gap == 3 and (quarter_end_months or quarter_start_months):
            return "Q"
    elif (
        (quarter_end_months or quarter_start_months)
        and np.all(gaps % 3 == 0)
        and np.any(gaps == 3)
    ):
        return "Q"
    elif not quarter_end_months and not quarter_start_months:
        return "M"

    raise ValueError(
        f"Cannot infer supported frequency for {context} from dates "
        f"with calendar-month gaps {gaps.tolist()}; frequency is ambiguous or "
        "unsupported. Provide more observations with a regular frequency."
    )


def infer_variable_frequencies(
    frame: pd.DataFrame, variables: list[str], context: str
) -> dict[str, str]:
    """Infer one frequency per variable from its non-null raw observations."""
    frequencies = {}
    for variable in variables:
        values = frame[variable].dropna()
        if not values.empty:
            frequencies[variable] = infer_frequency_from_dates(
                values.index, f"{context} column '{variable}'"
            )
    return frequencies


def infer_long_variable_frequencies(
    data: pd.DataFrame, variables: list[str], context: str
) -> dict[str, str]:
    """Infer one frequency per variable from a long-form date column."""
    frequencies = {}
    for variable in variables:
        rows = data.loc[data["variable"].eq(variable)]
        dates = pd.DatetimeIndex(rows["date"])
        frequencies[variable] = infer_frequency_from_dates(
            dates, f"{context} column '{variable}'"
        )
    return frequencies


def _validate_x_role(X: pd.DataFrame | None, X_variables: list[str] | None) -> None:
    """X and X_variables must be given together, or not at all."""
    if X is not None and not X_variables:
        raise ValueError("X was given without X_variables.")
    if X_variables and X is None:
        raise ValueError("X_variables was given without X.")


def combine_history_and_future(
    history: pd.DataFrame, future: pd.DataFrame | None
) -> pd.DataFrame:
    """Overlay future/conditioning values on history one cell at a time.

    A present future value overrides a historical value at the same date,
    while missing or omitted future values retain the history. Future-only
    missing values remain missing.
    """
    if future is None or future.empty:
        return history
    combined = future.combine_first(history).sort_index()
    combined = combined.reindex(columns=history.columns)
    if isinstance(combined.index, pd.DatetimeIndex) and combined.index.freq is None:
        combined.index.freq = combined.index.inferred_freq
    return combined


def _validate_wide_frame(
    frame: pd.DataFrame,
    variables: list[str],
    role: str,
    *,
    require_all: bool,
) -> None:
    """Validate a raw wide input frame for *role* (e.g. ``"y"``, ``"X_future"``).

    Args:
        frame : pd.DataFrame
            The candidate wide frame.
        variables : list[str]
            The variables allowed as columns of *frame*.
        role : str
            A label used in error messages (e.g. ``"y"``, ``"y_conditioning"``).
        require_all : bool
            Whether every entry of *variables* must be a column of *frame*.
            ``False`` allows a conditioning/future frame to cover only a
            subset of the variables.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            f"{role} must be a pandas DataFrame; got {type(frame).__name__}."
        )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{role} must be indexed by a DatetimeIndex.")
    if frame.index.has_duplicates:
        raise ValueError(f"{role} index must not contain duplicate dates.")

    unknown = [c for c in frame.columns if c not in variables]

    if require_all:
        missing = [v for v in variables if v not in frame.columns]
        if missing:
            raise ValueError(f"{role} is missing columns for variables: {missing}.")

    if unknown:
        raise ValueError(
            f"{role} has columns not in the configured variables: {unknown}."
        )


def _validate_mapping_coverage(
    data_transformation: dict[str, str],
    y_variables: list[str],
    X_variables: list[str] | None,
    context: str | None = None,
) -> None:
    """*data_transformation* must map every y/X variable to a required metric."""
    mapping_context = f" for {context}" if context is not None else ""
    if not set(y_variables).issubset(data_transformation.keys()):
        raise ValueError(
            f"data_transformation{mapping_context} must contain all y_variables. "
            f"Got {list(data_transformation.keys())},"
            f"but expected to include {y_variables}"
        )

    if X_variables is not None and not set(X_variables).issubset(
        data_transformation.keys()
    ):
        raise ValueError(
            f"data_transformation{mapping_context} must contain all X_variables. "
            f"Got {list(data_transformation.keys())},"
            f"but expected to include {X_variables}"
        )


def _resolve_input_metric_mapping(
    mapping: dict[str, str] | None,
    variables: list[str],
    name: str,
) -> dict[str, str]:
    """Validate source metrics and default omitted variables to levels."""
    validated = _validate_metric_mapping(mapping, name) or {}
    unknown = [variable for variable in validated if variable not in variables]
    if unknown:
        raise ValueError(
            f"{name} contains variables not present in the configured inputs: "
            f"{unknown}. Expected only {variables}."
        )
    invalid = [
        (variable, metric)
        for variable, metric in validated.items()
        if metric not in _VALID_METRICS
    ]
    if invalid:
        raise ValueError(
            f"{name} must use supported metrics {list(_VALID_METRICS)}; got {invalid}."
        )
    return {variable: validated.get(variable, "levels") for variable in variables}


def _validate_metric_mapping(
    mapping: dict[str, str] | None, name: str
) -> dict[str, str] | None:
    """Validate and copy a variable-to-metric mapping."""
    if mapping is None:
        return None
    if not isinstance(mapping, dict):
        raise TypeError(
            f"{name} must be a dict[str, str] mapping; got {type(mapping).__name__}."
        )
    bad_items = [
        (key, value)
        for key, value in mapping.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if bad_items:
        raise TypeError(
            f"{name} must map str to str (variable -> metric); "
            f"got non-str key/value pairs: {bad_items}."
        )
    return dict(mapping)


def _transform_metric(values: pd.Series, metric: str, frequency: str | None) -> pd.Series:
    """Compute ``metric`` from a chronologically ordered levels series."""
    if metric == "levels":
        return values.copy()
    if metric == "logs":
        return _logs_series(values)
    if metric == "log diff":
        return _difference_series(_calendar_align(values, frequency), logarithmic=True)
    if metric == "diff":
        return _difference_series(_calendar_align(values, frequency))
    if metric == "pop":
        return _growth_series(_calendar_align(values, frequency), periods=1)
    if metric == "yoy":
        return _growth_series(
            _calendar_align(values, frequency), periods=_PERIODS_PER_YEAR[frequency]
        )
    raise ValueError(f"Unsupported metric '{metric}' for raw wide-input transformation.")


def _transform_long_metric(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Apply the canonical trajectory converter to one long-form variable."""
    transformed_groups = []
    for _, group in data.groupby("vintage_date", sort=False):
        group = group.copy()
        trajectory = _ordered_trajectory(group)
        frequency = None
        if metric in _CALENDAR_DEPENDENT_METRICS:
            frequency = _resolve_frequency(trajectory["frequency"])
        values = trajectory.set_index("date")["value"]
        converted = _transform_metric(values, metric, frequency)
        group["value"] = group["date"].map(converted)
        transformed_groups.append(group)
    if not transformed_groups:
        return data.copy()
    return pd.concat(transformed_groups, ignore_index=True)


def _transform_variable(
    history: pd.Series,
    future: pd.Series | None,
    metric: str,
    input_metric: str = "levels",
    frequency: str | None = None,
    *,
    future_input_metric: str | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    """Transform one variable's source metric into ``metric``.

    Returns transformed history and, when supplied, transformed future values.
    """
    future_input_metric = (
        input_metric if future_input_metric is None else future_input_metric
    )
    if future is not None and input_metric != future_input_metric:
        if input_metric == "levels" and future_input_metric == metric:
            transformed_history = _transform_metric(history, metric, frequency)
            combined = combine_history_and_future(
                transformed_history.to_frame(), future.to_frame()
            ).iloc[:, 0]
            return combined.reindex(history.index), combined.reindex(future.index)
        raise ValueError(
            f"Cannot combine variable '{history.name}' from source metrics "
            f"'{input_metric}' and '{future_input_metric}' for requested "
            f"metric '{metric}'; history must be levels and future values "
            "must already use the requested metric."
        )

    if input_metric == metric and future_input_metric == metric:
        combined = combine_history_and_future(
            history.to_frame(), future.to_frame() if future is not None else None
        ).iloc[:, 0]
        return (
            combined.reindex(history.index),
            combined.reindex(future.index) if future is not None else None,
        )
    if input_metric != "levels" or future_input_metric != "levels":
        raise ValueError(
            f"Cannot transform variable '{history.name}' from metric "
            f"'{input_metric}' or '{future_input_metric}' to '{metric}'; "
            "only levels-derived conversions "
            "are supported."
        )

    combined = combine_history_and_future(
        history.to_frame(), future.to_frame() if future is not None else None
    ).iloc[:, 0]

    transformed = _transform_metric(combined, metric, frequency)

    history_out = transformed.reindex(history.index)
    future_out = transformed.reindex(future.index) if future is not None else None
    return history_out, future_out


def _transform_wide_variables(
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    variables: list[str],
    data_transformation: dict[str, str],
    input_metrics: dict[str, str] | None = None,
    future_input_metrics: dict[str, str] | None = None,
    frequencies: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Transform selected wide history and future variables independently."""
    history_out = {}
    future_out = {} if future is not None else None

    for var in variables:
        metric = data_transformation[var]
        future_col = future[var] if future is not None and var in future.columns else None
        history_out[var], future_col_out = _transform_variable(
            history[var],
            future_col,
            metric,
            (input_metrics or {}).get(var, "levels"),
            frequencies[var],
            future_input_metric=(future_input_metrics or {}).get(
                var, (input_metrics or {}).get(var, "levels")
            ),
        )
        if future_out is not None and future_col_out is not None:
            future_out[var] = future_col_out

    history_out = pd.DataFrame(history_out, index=history.index)[variables]
    if future_out is not None:
        columns = [v for v in variables if v in future_out]
        future_out = pd.DataFrame(future_out, index=future.index)[columns]

    return history_out, future_out


class DataTransformationPipeline:
    """Transform, filter and reconstruct data for real-time forecasts.

    Wraps a ``data_transformation`` mapping (variable -> required metric,
    e.g. ``{"gdp": "diff"}``). Instances hold only plain data (a dict) so
    they can be pickled and resolved once per model.

    The pipeline accepts wide model inputs and long-form forecast metadata.
    It stores the variable-to-metric mapping supplied at construction.
    """

    def __init__(self, data_transformation: dict[str, str]):
        validated = _validate_metric_mapping(data_transformation, "data_transformation")
        if validated is None:
            raise ValueError("data_transformation must be a dictionary")
        self.data_transformation = validated

    def apply(
        self,
        outturns: pd.DataFrame,
        forecasts: pd.DataFrame | None,
        y_variables: list[str],
        X_variables: list[str] | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Apply variable-specific transformations to long-form data.

        Args:
            outturns : pd.DataFrame
                The outturns dataframe
            forecasts : pd.DataFrame or None
                The forecasts dataframe
            y_variables : list[str]
                The y variables
            X_variables : list[str] or None
                The X variables

        Returns:
            tuple : (filtered_outturns, filtered_forecasts)
        """
        data_transformation = self.data_transformation
        _validate_mapping_coverage(data_transformation, y_variables, X_variables)

        # Collect all variables that need transformations
        all_variables = list(y_variables)
        if X_variables is not None:
            all_variables = all_variables + [
                v for v in X_variables if v not in y_variables
            ]

        # Apply transformations
        has_forecasts = (
            forecasts is not None
            and isinstance(forecasts, pd.DataFrame)
            and not forecasts.empty
        )

        if has_forecasts:
            # Combine outturns with forecasts before transforming so that
            # diff-based transformations have the preceding outturn as base.
            # Transform the combined data, then split back.
            forecasts_tagged = forecasts.copy()
            forecasts_tagged["_type"] = "forecast"
            outturns_tagged = outturns.copy()
            outturns_tagged["_type"] = "outturn"
            combined = pd.concat([outturns_tagged, forecasts_tagged], ignore_index=True)
            combined = apply_transformations(combined, all_variables, data_transformation)
            outturns = combined[combined["_type"] == "outturn"].drop(columns=["_type"])
            forecasts = combined[combined["_type"] == "forecast"].drop(columns=["_type"])
        else:
            outturns = apply_transformations(outturns, all_variables, data_transformation)
            forecasts = None

        return outturns, forecasts

    def filter(self, data: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
        """Filter a DataFrame to rows whose variable is in *variables* and whose
        metric matches the required transformation for that variable.

        Args:
            data : pd.DataFrame
                DataFrame containing at least ``variable`` and ``metric`` columns.
            variables : list[str]
                The variables to keep.

        Returns:
            pd.DataFrame : Filtered copy of *data*.
        """
        data_transformation = self.data_transformation

        variable_mask = data["variable"].isin(variables)
        unmapped = set(data.loc[variable_mask, "variable"]) - set(data_transformation)
        if unmapped:
            raise KeyError(next(iter(unmapped)))

        required_metric = data["variable"].map(data_transformation)
        mask = variable_mask & (data["metric"] == required_metric)
        return data[mask].copy()

    def reconstruct_levels(
        self,
        forecasts: pd.DataFrame,
        outturns: pd.DataFrame,
        y_variables: list[str],
        frequency: str | None = None,
    ) -> pd.DataFrame:
        """Reconstruct levels from logs or log differences if levels data is available.

        Args:
            forecasts : pd.DataFrame
                The forecasts dataframe with transformed values
            outturns : pd.DataFrame
                The outturns dataframe (already filtered)
            y_variables : list[str]
                The y variables
            frequency : str
                The frequency of the data

        Returns:
            pd.DataFrame : Forecasts with additional level reconstructions
        """
        data_transformation = self.data_transformation
        new_rows = []

        for var in y_variables:
            transformation = data_transformation[var]

            # Skip if already in levels or if reconstruction not needed
            if transformation == "levels":
                continue

            # Check if levels exist in outturns for this variable
            var_outturns = outturns[outturns["variable"] == var]
            available_metrics = var_outturns["metric"].unique()

            if "levels" not in available_metrics:
                continue

            # Get forecasts for this variable
            var_forecasts = forecasts[forecasts["variable"] == var].copy()

            if transformation == "logs":
                # Reconstruct levels from logs: levels = exp(logs)
                var_forecasts["value"] = np.exp(var_forecasts["value"])
                var_forecasts["metric"] = "levels"
                new_rows.append(var_forecasts)

            elif transformation == "log diff":
                levels_outturns = var_outturns[var_outturns["metric"] == "levels"]
                new_rows.extend(
                    _reconstruct_levels_by_vintage(
                        var_forecasts, levels_outturns, _reconstruct_logarithmic
                    )
                )

            elif transformation == "diff":
                levels_outturns = var_outturns[var_outturns["metric"] == "levels"]
                new_rows.extend(
                    _reconstruct_levels_by_vintage(
                        var_forecasts, levels_outturns, _reconstruct_additive
                    )
                )

        # Append reconstructed levels to forecasts
        if new_rows:
            forecasts = pd.concat([forecasts] + new_rows, ignore_index=True)

        return forecasts

    def transform_fit_inputs(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame | None = None,
        *,
        y_variables: list[str],
        X_variables: list[str] | None = None,
        frequency: str | None = None,
        frequencies: dict[str, str] | None = None,
        y_input_metrics: dict[str, str] | None = None,
        X_input_metrics: dict[str, str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Transform raw wide fit inputs (levels) into the configured metrics.

        Intended for the raw wide ``y``/``X`` frames a model receives after
        ``RealTimeModel`` has selected a vintage: one ``DatetimeIndex``-keyed
        column per variable, holding levels. Each column is transformed
        according to this pipeline's ``data_transformation`` mapping (e.g.
        ``"diff"``, ``"log diff"``, ``"logs"``, ``"pop"``, ``"yoy"`` or the
        ``"levels"`` identity).

        Args:
            y : pd.DataFrame
                Raw levels, ``DatetimeIndex``-keyed, one column per
                ``y_variables`` entry.
            X : pd.DataFrame or None, optional
                Raw levels for ``X_variables``, same shape convention as *y*.
            y_variables : list[str]
                The y variables; every entry must be a column of *y* and a
                key of ``data_transformation``.
            X_variables : list[str] or None, optional
                The X variables; required (and only allowed) together with
                *X*.
            frequency : str, optional
                Legacy fallback data frequency (``"M"`` or ``"Q"``). The
                transformation frequency is inferred from each raw column.
        Returns:
            tuple : ``(y_out, X_out)``, new DataFrames in the configured
            metric space. *X_out* is ``None`` when *X* is ``None``.
        """
        _validate_wide_frame(y, y_variables, "y", require_all=True)
        _validate_x_role(X, X_variables)
        if X is not None:
            _validate_wide_frame(X, X_variables, "X", require_all=True)
        _validate_mapping_coverage(self.data_transformation, y_variables, X_variables)
        y_input_metrics = _resolve_input_metric_mapping(
            y_input_metrics, y_variables, "y_input_metrics"
        )
        X_input_metrics = _resolve_input_metric_mapping(
            X_input_metrics, X_variables or [], "X_input_metrics"
        )
        if frequency is not None:
            _validate_frequency(frequency)

        y_frequency_map = {variable: frequencies[variable] for variable in y_variables}
        x_frequency_map = (
            {variable: frequencies[variable] for variable in X_variables}
            if X is not None
            else None
        )

        y_out, _ = _transform_wide_variables(
            y,
            None,
            y_variables,
            self.data_transformation,
            input_metrics=y_input_metrics,
            frequencies=y_frequency_map,
        )
        X_out = None
        if X is not None:
            X_out, _ = _transform_wide_variables(
                X,
                None,
                X_variables,
                self.data_transformation,
                input_metrics=X_input_metrics,
                frequencies=x_frequency_map,
            )
        return y_out, X_out

    def transform_forecast_inputs(
        self,
        y_history: pd.DataFrame,
        y_conditioning: pd.DataFrame | None = None,
        X_history: pd.DataFrame | None = None,
        X_future: pd.DataFrame | None = None,
        *,
        y_variables: list[str],
        X_variables: list[str] | None = None,
        frequency: str | None = None,
        frequencies: dict[str, str] | None = None,
        y_input_metrics: dict[str, str] | None = None,
        X_input_metrics: dict[str, str] | None = None,
        y_conditioning_input_metrics: dict[str, str] | None = None,
        X_conditioning_input_metrics: dict[str, str] | None = None,
    ) -> tuple[
        pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None
    ]:
        """Transform raw wide history plus conditioning/future rows as one trajectory.

        Historical raw levels (*y_history*/*X_history*) and appended raw
        conditioning/future rows (*y_conditioning*/*X_future*) are
        transformed together, so a differencing or growth transformation
        applied to the first conditioning row uses the preceding raw
        historical observation as its base. A conditioning row that shares a
        date with a historical row (a backcast overlap) takes precedence for
        that date, matching :meth:`apply`.

        Args:
            y_history : pd.DataFrame
                Raw historical levels, ``DatetimeIndex``-keyed, one column
                per ``y_variables`` entry.
            y_conditioning : pd.DataFrame or None, optional
                Raw conditioning/future levels for a subset of
                ``y_variables``, appended after *y_history*.
            X_history : pd.DataFrame or None, optional
                Raw historical levels for ``X_variables``.
            X_future : pd.DataFrame or None, optional
                Raw future levels for a subset of ``X_variables``, appended
                after *X_history*.
            y_variables : list[str]
                The y variables; every entry must be a column of *y_history*
                and a key of ``data_transformation``.
            X_variables : list[str] or None, optional
                The X variables; required (and only allowed) together with
                *X_history*.
            frequency : str, optional
                Legacy fallback data frequency (``"M"`` or ``"Q"``). The
                transformation frequency is inferred from each raw column.
        Returns:
            tuple : ``(y_history_out, y_conditioning_out, X_history_out,
            X_future_out)``, new DataFrames in the configured metric space.
            Each conditioning/future output is ``None`` when its raw input
            was ``None``.
        """
        _validate_wide_frame(y_history, y_variables, "y_history", require_all=True)
        _validate_x_role(X_history, X_variables)
        if X_history is not None:
            _validate_wide_frame(X_history, X_variables, "X_history", require_all=True)
        if y_conditioning is not None:
            _validate_wide_frame(
                y_conditioning, y_variables, "y_conditioning", require_all=False
            )
        if X_future is not None:
            _validate_wide_frame(X_future, X_variables, "X_future", require_all=False)
        _validate_mapping_coverage(self.data_transformation, y_variables, X_variables)
        y_input_metrics = _resolve_input_metric_mapping(
            y_input_metrics, y_variables, "y_input_metrics"
        )
        y_conditioning_input_metrics = _resolve_input_metric_mapping(
            y_conditioning_input_metrics,
            y_variables,
            "y_conditioning_input_metrics",
        )
        X_input_metrics = _resolve_input_metric_mapping(
            X_input_metrics, X_variables or [], "X_input_metrics"
        )
        X_conditioning_input_metrics = _resolve_input_metric_mapping(
            X_conditioning_input_metrics,
            X_variables or [],
            "X_conditioning_input_metrics",
        )
        if frequency is not None:
            _validate_frequency(frequency)

        y_frequency_map = {variable: frequencies[variable] for variable in y_variables}
        x_frequency_map = None
        if X_history is not None:
            x_frequency_map = {
                variable: frequencies[variable] for variable in X_variables
            }

        y_history_out, y_conditioning_out = _transform_wide_variables(
            y_history,
            y_conditioning,
            y_variables,
            self.data_transformation,
            input_metrics=y_input_metrics,
            future_input_metrics=y_conditioning_input_metrics,
            frequencies=y_frequency_map,
        )
        X_history_out = X_future_out = None
        if X_history is not None:
            X_history_out, X_future_out = _transform_wide_variables(
                X_history,
                X_future,
                X_variables,
                self.data_transformation,
                input_metrics=X_input_metrics,
                future_input_metrics=X_conditioning_input_metrics,
                frequencies=x_frequency_map,
            )
        return y_history_out, y_conditioning_out, X_history_out, X_future_out


@dataclass(frozen=True)
class FittedDataTransformation:
    """Immutable input-transformation configuration captured during fitting."""

    data_transformation: tuple[tuple[str, str], ...] | None
    y_input_metrics: tuple[tuple[str, str], ...]
    X_input_metrics: tuple[tuple[str, str], ...]
    y_variables: tuple[str, ...]
    X_variables: tuple[str, ...] | None
    y_frequencies: tuple[tuple[str, str], ...]
    X_frequencies: tuple[tuple[str, str], ...]
    frequency: str | None
    X_imputation: str | None
    pipeline_source: str
    y_conditioning_input_metrics: tuple[tuple[str, str], ...] = ()
    X_conditioning_input_metrics: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_fit(
        cls,
        pipeline: DataTransformationPipeline | None,
        *,
        y_variables: list[str],
        X_variables: list[str] | None,
        y_frequencies: dict[str, str],
        X_frequencies: dict[str, str],
        frequency: str | None,
        X_imputation: str | None,
        pipeline_source: str,
        y_input_metrics: dict[str, str] | None = None,
        X_input_metrics: dict[str, str] | None = None,
        y_conditioning_input_metrics: dict[str, str] | None = None,
        X_conditioning_input_metrics: dict[str, str] | None = None,
    ) -> "FittedDataTransformation":
        resolved_y_input_metrics = _resolve_input_metric_mapping(
            y_input_metrics, y_variables, "y_input_metrics"
        )
        resolved_X_input_metrics = _resolve_input_metric_mapping(
            X_input_metrics, X_variables or [], "X_input_metrics"
        )
        resolved_y_conditioning_input_metrics = (
            _resolve_input_metric_mapping(
                y_conditioning_input_metrics,
                y_variables,
                "y_conditioning_input_metrics",
            )
            if y_conditioning_input_metrics is not None
            else {}
        )
        resolved_X_conditioning_input_metrics = (
            _resolve_input_metric_mapping(
                X_conditioning_input_metrics,
                X_variables or [],
                "X_conditioning_input_metrics",
            )
            if X_conditioning_input_metrics is not None
            else {}
        )
        return cls(
            data_transformation=(
                _ordered_items(pipeline.data_transformation)
                if pipeline is not None
                else None
            ),
            y_input_metrics=_ordered_items(resolved_y_input_metrics),
            X_input_metrics=_ordered_items(resolved_X_input_metrics),
            y_variables=tuple(y_variables),
            X_variables=tuple(X_variables) if X_variables is not None else None,
            y_frequencies=_ordered_items(y_frequencies),
            X_frequencies=_ordered_items(X_frequencies),
            frequency=frequency,
            X_imputation=X_imputation,
            pipeline_source=pipeline_source,
            y_conditioning_input_metrics=_ordered_items(
                resolved_y_conditioning_input_metrics
            ),
            X_conditioning_input_metrics=_ordered_items(
                resolved_X_conditioning_input_metrics
            ),
        )

    @property
    def pipeline(self) -> DataTransformationPipeline | None:
        """Recreate a short-lived pipeline without exposing fitted state."""
        if self.data_transformation is None:
            return None
        return DataTransformationPipeline(dict(self.data_transformation))

    @property
    def y_frequency_mapping(self) -> dict[str, str]:
        return dict(self.y_frequencies)

    @property
    def X_frequency_mapping(self) -> dict[str, str]:
        return dict(self.X_frequencies)

    @property
    def y_input_metric_mapping(self) -> dict[str, str]:
        return dict(self.y_input_metrics)

    @property
    def X_input_metric_mapping(self) -> dict[str, str]:
        return dict(self.X_input_metrics)

    @property
    def y_conditioning_input_metric_mapping(self) -> dict[str, str]:
        return dict(self.y_conditioning_input_metrics)

    @property
    def X_conditioning_input_metric_mapping(self) -> dict[str, str]:
        return dict(self.X_conditioning_input_metrics)

    def _validate_no_pipeline_inputs(self) -> None:
        """Reject derived sources when the implicit requested metric is levels."""
        for mapping in (self.y_input_metric_mapping, self.X_input_metric_mapping):
            for variable, input_metric in mapping.items():
                if input_metric != "levels":
                    raise ValueError(
                        f"Cannot transform variable '{variable}' from metric "
                        f"'{input_metric}' to 'levels'; only levels-derived "
                        "conversions are supported."
                    )

    def transform_fit_inputs(
        self, y: pd.DataFrame, X: pd.DataFrame | None
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        pipeline = self.pipeline
        if pipeline is None:
            self._validate_no_pipeline_inputs()
            return y.copy(), X.copy() if X is not None else None

        return pipeline.transform_fit_inputs(
            y,
            X,
            y_variables=list(self.y_variables),
            X_variables=(
                list(self.X_variables) if self.X_variables is not None else None
            ),
            frequency=self.frequency,
            frequencies={
                **self.y_frequency_mapping,
                **self.X_frequency_mapping,
            },
            y_input_metrics=self.y_input_metric_mapping,
            X_input_metrics=self.X_input_metric_mapping,
        )

    def transform_forecast_inputs(
        self,
        y_history: pd.DataFrame,
        y_conditioning: pd.DataFrame | None,
        X_history: pd.DataFrame | None,
        X_future: pd.DataFrame | None,
        y_conditioning_input_metrics: dict[str, str] | None = None,
        X_conditioning_input_metrics: dict[str, str] | None = None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        pd.DataFrame | None,
        pd.DataFrame | None,
    ]:
        pipeline = self.pipeline
        if pipeline is None:
            self._validate_no_pipeline_inputs()
            return y_history, y_conditioning, X_history, X_future

        return pipeline.transform_forecast_inputs(
            y_history=y_history,
            y_conditioning=y_conditioning,
            X_history=X_history,
            X_future=X_future,
            y_variables=list(self.y_variables),
            X_variables=(
                list(self.X_variables) if self.X_variables is not None else None
            ),
            frequency=self.frequency,
            frequencies={
                **self.y_frequency_mapping,
                **self.X_frequency_mapping,
            },
            y_input_metrics=self.y_input_metric_mapping,
            X_input_metrics=self.X_input_metric_mapping,
            y_conditioning_input_metrics=(
                self.y_conditioning_input_metric_mapping
                if y_conditioning_input_metrics is None
                and self.y_conditioning_input_metric_mapping
                else y_conditioning_input_metrics
                if y_conditioning_input_metrics is not None
                else self.y_input_metric_mapping
            ),
            X_conditioning_input_metrics=(
                self.X_conditioning_input_metric_mapping
                if X_conditioning_input_metrics is None
                and self.X_conditioning_input_metric_mapping
                else X_conditioning_input_metrics
                if X_conditioning_input_metrics is not None
                else self.X_input_metric_mapping
            ),
        )


# Keep the existing name as the implementation while exposing the architectural
# name used by callers that reason about a fitted preprocessing plan.
ResolvedTransformationPlan = FittedDataTransformation
