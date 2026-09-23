"""Validation and metadata for long forecast results."""

import numpy as np
import pandas as pd


def _normalise_quantiles(quantiles):
    """Return sorted probabilities, or None for point forecasts."""
    if quantiles is False:
        return None
    if quantiles is True:
        return (0.16, 0.5, 0.84)
    try:
        values = tuple(quantiles)
    except TypeError as error:
        raise ValueError(
            "quantiles must be a Boolean or a probability sequence."
        ) from error
    if not values or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.number))
        or not np.isreal(value)
        for value in values
    ):
        raise ValueError("quantiles must contain non-Boolean probabilities.")
    values = np.asarray(values, dtype=float)
    if (
        not np.isfinite(values).all()
        or (values <= 0).any()
        or (values >= 1).any()
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError(
            "quantiles must be distinct finite probabilities between 0 and 1."
        )
    return tuple(sorted(values.tolist()))


class ForecastResult(pd.DataFrame):
    """Long forecast table with origin and decomposition metadata.

    Point results contain ``date``, ``variable`` and ``value``. Quantile results
    also contain ``quantile`` before ``value``. Rows use a RangeIndex and follow
    date, fitted target and ascending probability order; steps count dates.

    Construction validates and orders the forecast against explicit expectations.
    Slices and copies are ordinary DataFrames without result metadata. Validation
    does not protect against later mutation of the result.
    """

    _metadata = ["decomposition", "forecast_origin"]

    @property
    def forecast(self) -> pd.DataFrame:
        """Return the forecast values without result metadata."""
        return pd.DataFrame(self)

    def __init__(
        self,
        forecast: pd.DataFrame,
        *,
        expected_columns: list[str],
        steps: int,
        forecast_origin: pd.Timestamp,
        decomposition: pd.DataFrame | None = None,
        quantiles: bool | list[float] = False,
        forecast_dates: pd.DatetimeIndex | None = None,
        forecast_dates_include_origin: bool = False,
        decomp: bool = False,
    ):
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be an integer greater than zero")
        if not expected_columns or not pd.Index(expected_columns).is_unique:
            raise ValueError("Expected target columns must be non-empty and unique.")
        forecast_origin = pd.Timestamp(forecast_origin)
        if pd.isna(forecast_origin):
            raise ValueError("forecast_origin must be a non-missing timestamp.")
        probabilities = _normalise_quantiles(quantiles)
        if probabilities is None:
            forecast = ForecastResult._validate_point_result(
                forecast,
                steps,
                expected_columns,
                forecast_origin,
                forecast_dates,
                forecast_dates_include_origin,
            )
            decomposition = ForecastResult._validate_decomposition(
                forecast, decomposition, steps, expected_columns
            )
        else:
            if decomp or decomposition is not None:
                raise ValueError("decomp=True is not supported for quantile forecasts.")
            if forecast_dates is None:
                raise ValueError("Quantile forecasts require an expected calendar.")
            ForecastResult._validate_calendar(
                forecast_dates, steps, forecast_origin, forecast_dates_include_origin
            )
            forecast = ForecastResult._validate_quantile_result(
                forecast, forecast_dates, expected_columns, probabilities
            )
        super().__init__(forecast)
        self.decomposition = decomposition
        self.forecast_origin = forecast_origin

    @staticmethod
    def _order_forecast_keys(forecast, dates, variables, probabilities=None):
        """Check complete, unique keys and return rows in the requested order."""
        keys = ["date", "variable"]
        dimensions = [dates, variables]
        mode, coverage = "Point", "date and fitted target"
        if probabilities is not None:
            keys.append("quantile")
            dimensions.append(probabilities)
            mode, coverage = "Quantile", "date, variable and probability"
        if forecast[keys].isna().any().any() or forecast.duplicated(keys).any():
            raise ValueError(f"{mode} forecast keys must be complete and unique.")
        expected = pd.MultiIndex.from_product(dimensions, names=keys)
        result = pd.DataFrame(forecast).set_index(keys)
        if len(result) != len(expected) or not expected.isin(result.index).all():
            raise ValueError(f"{mode} forecasts must cover every {coverage}.")
        return result.reindex(expected).reset_index()

    @staticmethod
    def _validate_quantile_result(forecast, dates, variables, probabilities):
        """Validate complete marginal quantiles on the requested forecast calendar."""
        columns = ["date", "variable", "quantile", "value"]
        if (
            not isinstance(forecast, pd.DataFrame)
            or not forecast.columns.is_unique
            or set(forecast.columns) != set(columns)
        ):
            raise ValueError(f"Quantile forecasts must have columns {columns}.")
        result = ForecastResult._order_forecast_keys(
            forecast.loc[:, columns], dates, variables, probabilities
        )
        values = result["value"].to_numpy(dtype=float).reshape(-1, len(probabilities))
        if not np.isfinite(values).all():
            raise ValueError("Quantile forecast values must be finite.")
        if (np.diff(values, axis=1) < 0).any():
            raise ValueError("Quantile forecasts must not cross.")
        return result

    @staticmethod
    def _validate_point_result(
        forecast,
        steps: int,
        expected_columns: list[str],
        forecast_origin: pd.Timestamp,
        forecast_dates: pd.DatetimeIndex | None,
        forecast_dates_include_origin: bool,
    ) -> pd.DataFrame:
        """Validate complete point keys without treating missing values as absent."""
        if not isinstance(forecast, pd.DataFrame) or list(forecast.columns) != [
            "date",
            "variable",
            "value",
        ]:
            raise ValueError(
                "Point forecasts must have columns ['date', 'variable', 'value']."
            )
        if not pd.api.types.is_datetime64_any_dtype(forecast["date"]):
            raise TypeError("Forecast date must have a datetime dtype.")
        dates = (
            pd.DatetimeIndex(forecast["date"].drop_duplicates()).sort_values()
            if forecast_dates is None
            else forecast_dates
        )
        if forecast_dates is not None:
            ForecastResult._validate_calendar(
                dates, steps, forecast_origin, forecast_dates_include_origin
            )
        ordered = ForecastResult._order_forecast_keys(forecast, dates, expected_columns)
        if forecast_dates is None:
            ForecastResult._validate_calendar(
                dates, steps, forecast_origin, forecast_dates_include_origin
            )
        return ordered

    @staticmethod
    def _validate_calendar(dates, steps, forecast_origin, forecast_dates_include_origin):
        """Validate the date contract shared by hook payloads and long results."""
        if len(dates) != steps:
            raise ValueError(f"Forecast must have {steps} dates, got {len(dates)}")
        if not isinstance(dates, pd.DatetimeIndex):
            raise TypeError("Forecast must be indexed by a DatetimeIndex.")
        if dates.hasnans:
            raise ValueError("Forecast index must not contain missing dates.")
        if dates.has_duplicates:
            raise ValueError("Forecast index must not contain duplicate dates.")
        if not dates.is_monotonic_increasing:
            raise ValueError("Forecast index must be sorted in increasing order.")

        origin = pd.Timestamp(forecast_origin)
        invalid_dates = (
            dates < origin if forecast_dates_include_origin else dates <= origin
        )
        if invalid_dates.any():
            raise ValueError(
                "Forecast dates do not satisfy their declared relationship to the "
                "fitted forecast origin; ordinary forecasts must be strictly "
                "after that origin."
            )

    @staticmethod
    def _validate_decomposition(
        forecast,
        decomposition: pd.DataFrame | None,
        steps: int,
        expected_columns: list[str],
    ) -> pd.DataFrame | None:
        """Validate and normalise a model-local additive decomposition."""
        if decomposition is None:
            return None
        if not isinstance(decomposition, pd.DataFrame):
            raise TypeError("decomposition must be a pandas DataFrame or None.")
        if not decomposition.columns.is_unique:
            raise ValueError("decomposition columns must be unique.")

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
        forecast_values = forecast.set_index(["date", "variable"])["value"]
        for horizon_value, date in enumerate(forecast["date"].drop_duplicates()):
            for variable in expected_columns:
                total = totals.loc[(horizon_value, variable)]
                forecast_value = forecast_values.loc[(date, variable)]
                if not np.isclose(total, forecast_value, equal_nan=True):
                    raise ValueError(
                        "decomposition contributions must reconcile to the "
                        "forecast for every horizon and target variable."
                    )
        return result
