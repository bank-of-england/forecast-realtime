"""Data-transformation utilities and pickleable pipeline classes."""

from dataclasses import dataclass

import pandas as pd

from ._model_data import (
    ModelData,
    _resolve_input_metric_mapping,
    _validate_mapping_coverage,
    _validate_metric_mapping,
)
from ._model_data import infer_frequency_from_dates as infer_frequency_from_dates
from ._model_data import (
    infer_long_variable_frequencies as infer_long_variable_frequencies,
)


def difference_by_vintage(data: pd.DataFrame, logarithmic: bool = False) -> pd.DataFrame:
    """Return first differences for each date within each vintage."""
    if not data.groupby("vintage_date").ngroups:
        raise ValueError("No objects to concatenate")
    return (
        ModelData.from_long(data)
        .convert_trajectories("log diff" if logarithmic else "diff", sort_vintages=True)
        .to_long()
    )


def growth_by_vintage(data: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Return percentage growth over ``periods`` steps for each vintage."""
    if not data.groupby("vintage_date").ngroups:
        raise ValueError("No objects to concatenate")
    return (
        ModelData.from_long(data)
        .convert_trajectories("diff", periods=periods, sort_vintages=True)
        .to_long()
    )


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
    return ModelData.from_long(data).derive(variables, data_transformation).to_long()


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
    ) -> pd.DataFrame:
        """Reconstruct levels from logs or differences using available level history.

        Args:
            forecasts : pd.DataFrame
                The forecasts dataframe with transformed values
            outturns : pd.DataFrame
                The outturns dataframe (already filtered)
            y_variables : list[str]
                The y variables

        Returns:
            pd.DataFrame : Forecasts with additional level reconstructions
        """
        return (
            ModelData.from_long(forecasts)
            .reconstruct_levels(
                ModelData.from_long(outturns, semantics="archive"),
                y_variables,
                self.data_transformation,
            )
            .to_long()
        )

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
                Forecast-step frequency (``"M"`` or ``"Q"``), independent of
                each variable's transformation calendar.
            frequencies : dict[str, str], optional
                Source calendars by variable; omitted calendars are inferred
                from each historical column's non-null observations.
            y_input_metrics : dict[str, str], optional
                Source metrics for y; omitted entries default to levels.
            X_input_metrics : dict[str, str], optional
                Source metrics for X; omitted entries default to levels.
        Returns:
            tuple : ``(y_out, X_out)``, new DataFrames in the configured
            metric space. *X_out* is ``None`` when *X* is ``None``.
        """
        y_out, _, X_out, _ = self.transform_forecast_inputs(
            y_history=y,
            X_history=X,
            y_variables=y_variables,
            X_variables=X_variables,
            frequency=frequency,
            frequencies=frequencies,
            y_input_metrics=y_input_metrics,
            X_input_metrics=X_input_metrics,
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
                Forecast-step frequency (``"M"`` or ``"Q"``), independent of
                each variable's transformation calendar.
            frequencies : dict[str, str], optional
                Source calendars by variable; omitted calendars are inferred
                from each historical column's non-null observations.
            y_input_metrics : dict[str, str], optional
                Source metrics for y history; omitted entries default to levels.
            X_input_metrics : dict[str, str], optional
                Source metrics for X history; omitted entries default to levels.
            y_conditioning_input_metrics : dict[str, str], optional
                Source metrics for y conditioning; omitted entries default to levels.
            X_conditioning_input_metrics : dict[str, str], optional
                Source metrics for future X; omitted entries default to levels.
        Returns:
            tuple : ``(y_history_out, y_conditioning_out, X_history_out,
            X_future_out)``, new DataFrames in the configured metric space.
            Each conditioning/future output is ``None`` when its raw input
            was ``None``.
        """
        if X_history is not None and not X_variables:
            raise ValueError("X was given without X_variables.")
        if X_variables and X_history is None:
            raise ValueError("X_variables was given without X.")
        for frame, variables, role, require_all in (
            (y_history, y_variables, "y_history", True),
            (X_history, X_variables or [], "X_history", True),
            (y_conditioning, y_variables, "y_conditioning", False),
            (X_future, X_variables or [], "X_future", False),
        ):
            if frame is not None or role == "y_history":
                ModelData.validate_frame(
                    frame,
                    role,
                    allow_period=False,
                    require_sorted=False,
                    variables=variables,
                    require_all=require_all,
                )
        _validate_mapping_coverage(self.data_transformation, y_variables, X_variables)
        y_conditioning_input_metrics = _resolve_input_metric_mapping(
            y_conditioning_input_metrics,
            y_variables,
            "y_conditioning_input_metrics",
        )
        X_conditioning_input_metrics = _resolve_input_metric_mapping(
            X_conditioning_input_metrics,
            X_variables or [],
            "X_conditioning_input_metrics",
        )
        data = ModelData.from_wide(
            y=y_history.loc[:, y_variables],
            X=X_history.loc[:, X_variables] if X_history is not None else None,
            y_conditioning=(
                y_conditioning.loc[
                    :, [v for v in y_variables if v in y_conditioning.columns]
                ]
                if y_conditioning is not None
                else None
            ),
            X_conditioning=(
                X_future.loc[:, [v for v in X_variables if v in X_future.columns]]
                if X_future is not None
                else None
            ),
            frequencies=frequencies,
            y_input_metrics=y_input_metrics,
            X_input_metrics=X_input_metrics,
            y_conditioning_input_metrics=y_conditioning_input_metrics,
            X_conditioning_input_metrics=X_conditioning_input_metrics,
        )
        data, _ = data.resolve_frequencies(self.data_transformation, frequency)
        transformed = data.transform(self.data_transformation)
        return (
            transformed.to_wide("y", "history"),
            transformed.to_wide("y", "conditioning"),
            transformed.to_wide("X", "history"),
            transformed.to_wide("X", "conditioning"),
        )


@dataclass(frozen=True)
class FittedDataTransformation:
    """Immutable input-transformation configuration captured during fitting."""

    data_transformation: tuple[tuple[str, str], ...] | None
    y_variables: tuple[str, ...]
    X_variables: tuple[str, ...] | None
    frequency: str | None
    X_imputation: str | None
    pipeline_source: str

    @classmethod
    def from_fit(
        cls,
        pipeline: DataTransformationPipeline | None,
        *,
        y_variables: list[str],
        X_variables: list[str] | None,
        frequency: str | None,
        X_imputation: str | None,
        pipeline_source: str,
    ) -> "FittedDataTransformation":
        return cls(
            data_transformation=(
                tuple(sorted(pipeline.data_transformation.items()))
                if pipeline is not None
                else None
            ),
            y_variables=tuple(y_variables),
            X_variables=tuple(X_variables) if X_variables is not None else None,
            frequency=frequency,
            X_imputation=X_imputation,
            pipeline_source=pipeline_source,
        )
