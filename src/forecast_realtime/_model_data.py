"""Private observations, provenance and calendar operations for model inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t

_VALID_METRICS = ("levels", "logs", "diff", "log diff", "pop", "yoy")
_CALENDAR_DEPENDENT_METRICS = {"diff", "log diff", "pop", "yoy"}
_PERIODS_PER_YEAR = {"M": 12, "Q": 4}
_VALID_FREQUENCIES = tuple(_PERIODS_PER_YEAR)


@dataclass(frozen=True)
class ModelInputRequirements:
    """Ordered input requests for one consumer, independent of its observations."""

    consumer: str
    y: tuple[tuple[str, str], ...] = ()
    X: tuple[tuple[str, str], ...] = ()
    explicit: bool = False
    X_kind: str = "raw"

    def items(self, role: str):
        """Return raw requests for a role without including synthetic regressors."""
        return () if role == "X" and self.X_kind == "component" else getattr(self, role)


def _select_input_metrics(data, variables, requested_metrics=None):
    """Select a metric before release selection, without per-vintage fallback."""
    if "metric" not in data:
        return data.copy(), {variable: "levels" for variable in variables}
    requested_metrics = requested_metrics or {}
    selected = {}
    for variable in variables:
        available = sorted(data.loc[data["variable"].eq(variable), "metric"].unique())
        if not available:
            continue
        requested = requested_metrics.get(variable)
        if requested is not None:
            if requested in available:
                metric = requested
            elif "levels" in available and requested in _VALID_METRICS:
                metric = "levels"
            else:
                raise ValueError(
                    f"Cannot select input metric for variable '{variable}': "
                    f"requested metric '{requested}' is unavailable; available "
                    f"metrics: {available}."
                )
        elif len(available) == 1:
            metric = available[0]
        else:
            raise ValueError(
                f"Input metrics for variable '{variable}' are ambiguous; "
                f"available metrics: {available}."
            )
        selected[variable] = metric
    return data.loc[data["metric"].eq(data["variable"].map(selected))].copy(), selected


class ModelData:
    """Own a long observation table and isolated projections for model hooks.

    The catalogue holds stream labels once. Layouts bind ordered stream ids to
    role/path pairs and retain input indexes, including supplied empty inputs.
    Neither a caller-owned frame nor a mutable projection is kept as storage.
    Internal views share read-only tables; operations replace rather than edit them.
    """

    def __init__(
        self,
        observations,
        catalogue,
        *,
        layouts=None,
        semantics="wide",
        metadata=None,
        long_columns=None,
        long_dtypes=None,
    ):
        self._observations = observations
        self._catalogue = tuple(catalogue)
        self._layouts = layouts or {}
        self._semantics = semantics
        self._metadata = metadata
        self._long_columns = long_columns
        self._long_dtypes = long_dtypes

    def _replace(self, **changes):
        state = {
            "observations": self._observations,
            "catalogue": self._catalogue,
            "layouts": self._layouts,
            "semantics": self._semantics,
            "metadata": self._metadata,
            "long_columns": self._long_columns,
            "long_dtypes": self._long_dtypes,
        }
        return type(self)(**(state | changes))

    @classmethod
    def from_wide(
        cls,
        y=None,
        X=None,
        *,
        y_conditioning=None,
        X_conditioning=None,
        frequencies=None,
        y_input_metrics=None,
        X_input_metrics=None,
        y_conditioning_input_metrics=None,
        X_conditioning_input_metrics=None,
        y_published=None,
        y_published_input_metrics=None,
    ):
        """Copy direct inputs without inventing release dates or dropping NaNs."""
        frames = {
            ("y", "history"): (y, y_input_metrics),
            ("X", "history"): (X, X_input_metrics),
            ("y", "conditioning"): (y_conditioning, y_conditioning_input_metrics),
            ("X", "conditioning"): (X_conditioning, X_conditioning_input_metrics),
            ("y", "published"): (y_published, y_published_input_metrics),
        }
        paths = {}
        for (role, kind), (frame, metrics) in frames.items():
            if frame is None:
                continue
            cls.validate_frame(frame, role, require_sorted=False)
            metrics = _resolve_input_metric_mapping(
                metrics,
                list(frame.columns)
                if kind == "history"
                else list(metrics or frame.columns),
                f"{role}_{kind + '_' if kind != 'history' else ''}input_metrics",
            )
            series = []
            for column in frame.columns:
                series.append(
                    (
                        dict(
                            variable=column,
                            source=None,
                            metric=(metrics or {}).get(column, "levels"),
                            kind=kind,
                            frequency=(frequencies or {}).get(column),
                        ),
                        frame[column],
                    )
                )
            paths[role, kind] = (frame.index, frame.columns, series)
        return cls._from_paths(paths)

    @staticmethod
    def validate_frame(
        frame,
        role,
        *,
        allow_period=True,
        require_sorted=True,
        variables=None,
        require_all=True,
    ):
        """Validate date indexes and optional column coverage at an input boundary."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{role} must be a pandas DataFrame")
        index_types = (
            (pd.DatetimeIndex, pd.PeriodIndex) if allow_period else pd.DatetimeIndex
        )
        if not isinstance(frame.index, index_types):
            raise TypeError(f"{role} must be indexed by a DatetimeIndex.")
        if frame.index.has_duplicates:
            raise ValueError(f"{role} index must not contain duplicate dates.")
        if require_sorted and not frame.index.is_monotonic_increasing:
            raise ValueError(f"{role} index must be sorted in increasing order.")
        if frame.columns.has_duplicates:
            raise ValueError(f"{role} must not contain duplicate columns.")
        if variables is not None:
            missing = [variable for variable in variables if variable not in frame]
            if require_all and missing:
                raise ValueError(f"{role} is missing columns for variables: {missing}.")
            unknown = [column for column in frame if column not in variables]
            if unknown:
                raise ValueError(
                    f"{role} has columns not in the configured variables: {unknown}."
                )

    @classmethod
    def from_long(cls, data, *, semantics="trajectory", frequencies=None):
        """Copy labelled trajectories or an explicitly declared revision archive."""
        if semantics not in {"trajectory", "archive"}:
            raise ValueError("Long inputs must declare trajectory or archive semantics.")
        required = {"date", "value", "vintage_date"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Long observations are missing columns: {sorted(missing)}")
        labels = data.copy()
        defaults = {
            "variable": "",
            "source": None,
            "metric": "levels",
            "_type": "history",
        }
        for column, default in defaults.items():
            if column not in labels:
                labels[column] = default
        keys = ["variable", "source", "metric", "_type"]
        catalogue = []
        row_streams = np.empty(len(labels), dtype=int)
        for stream, (identity, positions) in enumerate(
            labels.groupby(keys, dropna=False, sort=False).indices.items()
        ):
            group = labels.iloc[positions]
            variable, source, metric, kind = identity
            frequency_values = (
                group["frequency"].dropna().unique() if "frequency" in group else []
            )
            catalogue.append(
                dict(
                    variable=variable,
                    source=None if pd.isna(source) else source,
                    metric=metric,
                    kind=kind,
                    frequency=(frequencies or {}).get(
                        variable,
                        frequency_values[0] if len(frequency_values) == 1 else None,
                    ),
                    dtype=group["value"].dtype,
                )
            )
            row_streams[positions] = stream
        table = labels[["date", "vintage_date", "value"]].reset_index(drop=True)
        table.insert(0, "series", row_streams)
        duplicate_keys = ["series", "date", "vintage_date"]
        duplicate = table.duplicated(duplicate_keys, keep=False)
        if duplicate.any():
            candidates = labels.reset_index(drop=True).loc[duplicate].copy()
            candidates["series"] = row_streams[duplicate]
            distinct = candidates.drop_duplicates()
            if distinct.duplicated(duplicate_keys, keep=False).any():
                raise ValueError(
                    "Conflicting duplicate observations for a series/date/vintage key."
                )
        metadata = data.drop(
            columns=["date", "vintage_date", "value", *[c for c in keys if c in data]]
        ).copy()
        # Row references preserve identical supplied rows without duplicate observations.
        row_ids = table.groupby(duplicate_keys, dropna=False, sort=False).ngroup()
        table["_row"] = row_ids
        table = table.drop_duplicates(duplicate_keys).set_index("_row")
        metadata["_row"] = row_ids.to_numpy()
        if semantics == "archive":
            metadata = metadata.loc[~metadata["_row"].duplicated()]
        return cls(
            table,
            catalogue,
            semantics=semantics,
            metadata=metadata,
            long_columns=data.columns.copy(),
            long_dtypes=data.dtypes.copy(),
        )

    @classmethod
    def from_archive(cls, outturns, forecasts=None, *, frequencies=None):
        """Adapt raw revision archives, retaining history and supplied sources."""
        rows = [outturns.assign(_type="history")]
        if forecasts is not None and not forecasts.empty:
            rows.append(forecasts.assign(_type="conditioning"))
        result = cls.from_long(
            pd.concat(rows, ignore_index=True),
            semantics="archive",
            frequencies=frequencies,
        )
        return result._replace(
            long_columns=result._long_columns.drop("_type"),
            long_dtypes=result._long_dtypes.drop("_type"),
        )

    def select(self, requirements, *, y_sources=None, X_sources=None):
        """Bind role-specific requests, preserving a common metric across consumers."""
        layouts = {}
        catalogue = pd.DataFrame(self._catalogue)
        for role, sources in (("y", y_sources), ("X", X_sources)):
            requests = {}
            explicit_variables = set()
            for consumer in requirements:
                for variable, metric in consumer.items(role):
                    requests.setdefault(variable, set()).add(metric)
                    if consumer.explicit:
                        explicit_variables.add(variable)
            for kind in ("history", "conditioning"):
                ids = []
                for variable, metrics in requests.items():
                    candidates = catalogue.loc[
                        catalogue["variable"].eq(variable) & catalogue["kind"].eq(kind)
                    ]
                    if kind == "conditioning":
                        if not sources or variable not in sources:
                            continue
                        candidates = candidates.loc[
                            candidates["source"].eq(sources[variable])
                        ]
                    if candidates.empty:
                        continue
                    if len(metrics) > 1:
                        available = set(candidates["metric"])
                        native = available.intersection(metrics)
                        if "levels" in available:
                            metric = "levels"
                        elif len(native) == 1:
                            metric = next(iter(native))
                        else:
                            raise ValueError(
                                f"Cannot select a common raw metric for tree variable "
                                f"'{variable}': leaves require {sorted(metrics)}, "
                                f"available metrics are {sorted(available)}. "
                                "Retain levels or provide one native source."
                            )
                    else:
                        metric = next(iter(metrics))
                        if (
                            metric == "levels"
                            and variable not in explicit_variables
                            and "levels" not in set(candidates["metric"])
                        ):
                            metric = None
                    selected, _ = _select_input_metrics(
                        candidates, [variable], {variable: metric}
                    )
                    if len(selected) != 1:
                        raise ValueError(
                            f"Input sources for variable '{variable}' are ambiguous."
                        )
                    ids.extend(selected.index)
                if (
                    kind == "history"
                    and requests
                    or kind == "conditioning"
                    and sources is not None
                ):
                    ids.sort(key=lambda stream: self._catalogue[stream]["variable"])
                    columns = pd.Index(
                        [self._catalogue[s]["variable"] for s in ids], name="variable"
                    )
                    layouts[role, kind] = (tuple(ids), None, columns)
        selected_ids = {stream for ids, _, _ in layouts.values() for stream in ids}
        rows = self._observations.loc[self._observations["series"].isin(selected_ids)]
        metadata = self._metadata
        if metadata is not None:
            metadata = metadata.loc[metadata["_row"].isin(rows.index)]
        return self._replace(observations=rows, metadata=metadata, layouts=layouts)

    def as_of(self, vintage):
        """Select the latest available release in each selected archive stream."""
        if self._semantics != "archive":
            raise ValueError(
                "as_of() requires a revision archive, not a supplied trajectory "
                "or wide input."
            )
        rows = self._observations
        if self._layouts:
            ids = {s for streams, _, _ in self._layouts.values() for s in streams}
            rows = rows.loc[rows["series"].isin(ids)]
        rows = rows.loc[rows["vintage_date"].le(pd.Timestamp(vintage))]
        rows = rows.sort_values("vintage_date", ascending=False, kind="stable")
        rows = rows.drop_duplicates(["series", "date"])
        return self._replace(observations=rows)

    def has_path(self, role, kind="history"):
        """Distinguish omitted inputs from supplied empty or all-missing paths."""
        return (role, kind) in self._layouts

    def _path(self, role, kind="history"):
        ids, index, columns = self._layouts[role, kind]
        if index is None:
            rows = self._observations.loc[self._observations["series"].isin(ids)]
            present = set(rows["series"])
            ids = tuple(stream for stream in ids if stream in present)
            columns = pd.Index(
                [self._catalogue[s]["variable"] for s in ids], name=columns.name
            )
            index = pd.DatetimeIndex(rows["date"].unique(), name="date").sort_values()
        return ids, index, columns

    def columns(self, role, kind="history"):
        """Return the declared columns of a supplied path."""
        return (
            self._path(role, kind)[2].copy(deep=True)
            if self.has_path(role, kind)
            else None
        )

    def index(self, role, kind="history"):
        """Return an isolated date index for a supplied path."""
        return self._path(role, kind)[1].copy(deep=True)

    def _series_values(self, stream, index):
        selected = self._observations["series"].to_numpy() == stream
        dates = pd.Index(self._observations["date"].array[selected])
        if dates.has_duplicates:
            raise ValueError(
                "Select an archive vintage before projecting a wide trajectory."
            )
        entry = self._catalogue[stream]
        values = pd.Series(
            self._observations["value"].array[selected],
            index=dates,
            name=entry["variable"],
            dtype=entry["dtype"],
        )
        return values.reindex(index)

    def to_wide(self, role="y", kind="history"):
        """Materialise an isolated DataFrame at a model or compatibility boundary."""
        if not self.has_path(role, kind):
            return None
        ids, index, columns = self._path(role, kind)
        result = pd.DataFrame(
            {
                column: self._series_values(stream, index)
                for column, stream in zip(columns, ids, strict=True)
            },
            index=index.copy(deep=True),
        )
        result.columns = columns.copy(deep=True)
        return result

    def metrics(self, role, kind="history"):
        """Return labelled source units, never inferred from merged values."""
        if not self.has_path(role, kind):
            return {}
        return {
            self._catalogue[s]["variable"]: self._catalogue[s]["metric"]
            for s in self._layouts[role, kind][0]
        }

    def frequencies(self, role, kind="history"):
        """Return the per-series calendar independently of the forecast step."""
        if not self.has_path(role, kind):
            return {}
        return {
            self._catalogue[s]["variable"]: self._catalogue[s]["frequency"]
            for s in self._layouts[role, kind][0]
        }

    def with_dates(self, role, index, kind="history"):
        """Restrict a role's date support without changing shared observations."""
        ids, _, columns = self._path(role, kind)
        return self._replace(
            layouts=self._layouts | {(role, kind): (ids, index.copy(deep=True), columns)}
        )

    def history(self):
        """Return the history bindings without conditioning paths."""
        return self._replace(
            layouts={
                key: value for key, value in self._layouts.items() if key[1] == "history"
            }
        )

    def subset(self, y_variables, X_variables=None):
        """Select ordered consumer inputs without copying their observation table."""
        layouts = {}
        for (role, kind), (ids, index, columns) in self._layouts.items():
            variables = y_variables if role == "y" else X_variables
            if variables is None:
                continue
            by_variable = {self._catalogue[s]["variable"]: s for s in ids}
            selected = [variable for variable in variables if variable in by_variable]
            layouts[role, kind] = (
                tuple(by_variable[v] for v in selected),
                index,
                pd.Index(selected, name=columns.name),
            )
        return self._replace(layouts=layouts)

    def with_conditioning(self, future):
        """Attach replacement labelled paths without mutating fitted history."""
        if not any(kind != "history" for _, kind in future._layouts):
            return self.history()
        paths = dict(self.history()._paths())
        paths.update(
            {key: value for key, value in future._paths() if key[1] != "history"}
        )
        return self._from_paths(paths)

    def without_target_conditioning(self):
        """Retain published observations and all non-target paths for tree children."""
        return self._replace(
            layouts={
                key: value
                for key, value in self._layouts.items()
                if key != ("y", "conditioning")
            }
        )

    def with_input_metadata(
        self, source, *, fields=("source", "owner", "metric", "frequency")
    ):
        """Restore selected provenance fields after a DataFrame input hook."""
        catalogue = list(self._catalogue)
        for role, kind in self._layouts:
            if not source.has_path(role, kind):
                continue
            source_ids = source._path(role, kind)[0]
            labels = {
                source._catalogue[s]["variable"]: source._catalogue[s] for s in source_ids
            }
            for stream in self._path(role, kind)[0]:
                entry = catalogue[stream]
                if entry["variable"] in labels:
                    catalogue[stream] = entry | {
                        key: value
                        for key, value in labels[entry["variable"]].items()
                        if key in fields
                    }
        return self._replace(catalogue=catalogue)

    def with_fitted_metadata(self, fitted):
        """Reuse fitted calendars and component ownership, not replacement units."""
        catalogue = list(self._catalogue)
        for role, kind in self._layouts:
            labels = {
                fitted._catalogue[s]["variable"]: fitted._catalogue[s]
                for s in fitted._layouts.get((role, "history"), ((), None, None))[0]
            }
            for stream in self._path(role, kind)[0]:
                entry = catalogue[stream]
                source = labels.get(entry["variable"], {})
                catalogue[stream] = entry | {
                    "frequency": entry["frequency"] or source.get("frequency"),
                    "owner": source.get("owner", entry.get("owner", "raw")),
                }
        return self._replace(catalogue=catalogue)

    @classmethod
    def from_components(cls, target, components, *, kind="history"):
        """Bind actual named output columns with their native units and calendars."""
        paths = dict(target._paths(role="y"))
        series = []
        index = None
        for name, frame, metrics, frequencies in components:
            cls.validate_frame(frame, name)
            index = frame.index if index is None else index.union(frame.index)
            for column in frame.columns:
                output = name if len(frame.columns) == 1 else f"{name}_{column}"
                series.append(
                    (
                        dict(
                            variable=output,
                            source=name,
                            metric=metrics[column],
                            frequency=frequencies.get(column),
                            kind=kind,
                            owner="component",
                        ),
                        frame[column].rename(output),
                    )
                )
        columns = pd.Index([entry["variable"] for entry, _ in series])
        if columns.has_duplicates:
            raise ValueError("Named component outputs must have unique columns.")
        paths["X", kind] = (index.sort_values(), columns, series)
        return cls._from_paths(paths)

    def published_after(self, history, cutoff, *, role="y"):
        """Retain available observations beyond a fitting cutoff as a labelled path."""
        ids, index, columns = history._path(role)
        index = index[index > cutoff]
        if not len(index):
            return self
        paths = dict(self._paths())
        old_index, _, old_values = paths.get((role, "published"), (index, columns, []))
        published = {entry["variable"]: values for entry, values in old_values}
        paths[role, "published"] = (
            index.union(old_index).sort_values(),
            columns,
            [
                (
                    history._catalogue[s],
                    _overlay_values(
                        history._series_values(s, index),
                        published.get(history._catalogue[s]["variable"]),
                    ),
                )
                for s in ids
            ],
        )
        return self._from_paths(paths)

    def condition(
        self, first_date, steps, frequency, *, y_steps_ahead=None, X_steps_ahead=None
    ):
        """Select published and supplied future paths without mixing their units."""
        first = pd.Period(first_date, freq=frequency)
        periods = pd.period_range(first, periods=steps, freq=frequency)
        y_dates = periods.to_timestamp(how="end").normalize()
        paths = {}
        ids, _, columns = self._path("y")
        published = [(self._catalogue[s], self._series_values(s, y_dates)) for s in ids]
        for role, horizons in (("y", y_steps_ahead), ("X", X_steps_ahead)):
            if not self.has_path(role):
                continue
            if role == "X" and horizons is None:
                _, path = next(self._paths(role="X", kind="history"))
                paths["X", "conditioning"] = path
                continue
            if horizons is None:
                continue
            frequency_map = self.frequencies(role)
            dates_by_variable = {}
            for variable, horizon in horizons.items():
                if variable not in frequency_map:
                    continue
                variable_frequency = frequency if role == "y" else frequency_map[variable]
                dates_by_variable[variable] = [
                    pd.period_range(
                        period.asfreq(variable_frequency, how="start"),
                        period.asfreq(variable_frequency, how="end"),
                        freq=variable_frequency,
                    )
                    .to_timestamp(how="end")
                    .normalize()
                    for period in periods
                ]
            dates = (
                y_dates
                if role == "y"
                else pd.DatetimeIndex(
                    sorted(
                        {
                            date
                            for groups in dates_by_variable.values()
                            for group in groups
                            for date in group
                        }
                    )
                )
            )
            condition_ids = (
                self._path(role, "conditioning")[0]
                if self.has_path(role, "conditioning")
                else ()
            )
            available = {self._catalogue[s]["variable"]: s for s in condition_ids}
            history_ids, _, role_columns = self._path(role)
            output = []
            for stream in history_ids:
                entry = self._catalogue[stream]
                variable = entry["variable"]
                horizon = horizons.get(variable)
                if variable in available and horizon is not None:
                    supplied = self._catalogue[available[variable]]
                    allowed = pd.DatetimeIndex(
                        [
                            date
                            for group in dates_by_variable[variable][: horizon + 1]
                            for date in group
                        ]
                    )
                    values = self._series_values(available[variable], allowed).reindex(
                        dates
                    )
                    output.append((supplied, values))
                else:
                    output.append((entry, pd.Series(np.nan, index=dates, name=variable)))
            if role == "X" or any(values.notna().any() for _, values in output):
                paths[role, "conditioning"] = (dates, role_columns, output)
        if any(values.notna().any() for _, values in published):
            paths["y", "published"] = (y_dates, columns, published)
        return self._from_paths(paths)

    @classmethod
    def from_context(cls, context, fitted):
        """Adapt the public context using the fitted source units and calendars."""
        for role, frame in (
            ("context.y_history", context.y_history),
            ("context.X_history", context.X_history),
            ("y", context.y_conditioning),
            ("X", context.X_conditioning),
            ("context.y_published", context.y_published),
        ):
            if frame is not None or role == "context.y_history":
                cls.validate_frame(frame, role)
        return cls.from_wide(
            context.y_history,
            context.X_history,
            y_conditioning=context.y_conditioning,
            X_conditioning=context.X_conditioning,
            y_published=context.y_published,
            y_published_input_metrics=(
                context.y_published_input_metrics
                if context.y_published_input_metrics is not None
                else fitted.metrics("y")
            ),
            frequencies={**fitted.frequencies("y"), **fitted.frequencies("X")},
            y_input_metrics=fitted.metrics("y"),
            X_input_metrics=fitted.metrics("X"),
            y_conditioning_input_metrics=(
                context.y_conditioning_input_metrics
                if context.y_conditioning_input_metrics is not None
                else fitted.metrics("y")
            ),
            X_conditioning_input_metrics=(
                context.X_conditioning_input_metrics
                if context.X_conditioning_input_metrics is not None
                else fitted.metrics("X")
            ),
        )

    def resolve_frequencies(
        self, mapping, frequency=None, *, require_calendars=False, require_step=True
    ):
        """Resolve necessary calendars once, leaving ordinary daily data valid."""
        if mapping is not None and frequency is not None:
            _validate_frequency(frequency)
        catalogue = list(self._catalogue)
        y_index = self.index("y")
        dates = y_index.to_timestamp() if isinstance(y_index, pd.PeriodIndex) else y_index
        calendar_index = dates.is_month_start.all() or dates.is_month_end.all()
        for role, kind in self._layouts:
            if kind != "history":
                continue
            ids, index, _ = self._path(role, kind)
            for stream in ids:
                entry = self._catalogue[stream]
                needed = (
                    (require_step and frequency is not None)
                    or (calendar_index and require_calendars)
                    or (
                        (mapping or {}).get(entry["variable"])
                        in _CALENDAR_DEPENDENT_METRICS
                        and entry["metric"] == "levels"
                    )
                )
                if entry["frequency"] is not None or not needed:
                    continue
                observed = self._series_values(stream, index).dropna()
                if not observed.empty:
                    resolved = infer_frequency_from_dates(
                        observed.index, f"raw {role} column '{entry['variable']}'"
                    )
                    catalogue[stream] = entry | {"frequency": resolved}
        if require_step and frequency is None:
            target_calendars = set(self.frequencies("y").values()) - {None}
            if len(target_calendars) == 1:
                frequency = next(iter(target_calendars))
        if require_step and frequency is None:
            inferred = (
                y_index.freqstr
                if isinstance(y_index, pd.PeriodIndex)
                else y_index.inferred_freq
            )
            if calendar_index and not inferred:
                frequency = infer_frequency_from_dates(y_index, "raw y")
            elif inferred:
                rule = pd.tseries.frequencies.to_offset(
                    inferred.replace("Q-", "QE-")
                ).rule_code.upper()
                frequency = (
                    "M"
                    if rule.startswith(("ME", "MS"))
                    else "Q"
                    if rule.startswith(("QE", "QS"))
                    else inferred
                )
        if mapping is not None and frequency is not None:
            _validate_frequency(frequency)
        return self._replace(catalogue=catalogue), frequency

    def trim_undefined_prefix(self):
        """Drop only leading undefined observations, independently for each role."""
        result = self
        for role in ("y", "X"):
            if not self.has_path(role):
                continue
            ids, index, _ = self._path(role)
            leading = []
            for stream in ids:
                values = self._series_values(stream, index)
                positions = np.flatnonzero(values.notna().to_numpy())
                leading.append(int(positions[0]) if len(positions) else len(index))
            result = result.with_dates(role, index[max(leading, default=0) :])
        return result

    def last_valid_dates(self, role):
        """Return each observed series' final date in declaration order."""
        if not self.has_path(role):
            return []
        ids, index, _ = self._path(role)
        return [
            date
            for stream in ids
            if (date := self._series_values(stream, index).last_valid_index()) is not None
        ]

    @classmethod
    def _from_paths(cls, paths):
        """Store calculated series once, retaining each path's declared axes."""
        catalogue, dates, values, streams, layouts = [], [], [], [], {}
        for key, (index, columns, series) in paths.items():
            ids = []
            for entry, column in series:
                ids.append(len(catalogue))
                catalogue.append(entry | {"dtype": column.dtype, "kind": key[1]})
                dates.append(column.index)
                values.append(column)
                streams.append(np.full(len(column), ids[-1], dtype=int))
            layouts[key] = (tuple(ids), index.copy(deep=True), columns.copy(deep=True))
        if len({str(column.dtype) for column in values}) > 1:
            values = [column.astype(object) for column in values]
        observations = pd.DataFrame(
            {
                "series": np.concatenate(streams) if streams else np.array([], dtype=int),
                "date": dates[0].append(dates[1:]) if dates else pd.DatetimeIndex([]),
                "value": pd.concat(values, ignore_index=True).array if values else [],
            }
        )
        return cls(observations, catalogue, layouts=layouts)

    def _paths(self, *, role=None, kind=None):
        for path_role, path_kind in self._layouts:
            if (
                role is not None
                and role != path_role
                or kind is not None
                and kind != path_kind
            ):
                continue
            ids, index, columns = self._path(path_role, path_kind)
            yield (
                (path_role, path_kind),
                (
                    index,
                    columns,
                    [(self._catalogue[s], self._series_values(s, index)) for s in ids],
                ),
            )

    def transform(self, mapping=None, *, combine=False):
        """Convert bound paths, optionally returning complete forecast trajectories."""

        def requested(entry):
            if mapping is not None:
                return mapping[entry["variable"]]
            return entry["metric"] if entry.get("owner") == "component" else "levels"

        layouts = {key: self._path(*key) for key in self._layouts}
        if mapping is not None:
            for (role, kind), (ids, index, columns) in layouts.items():
                if not isinstance(index, pd.DatetimeIndex):
                    raise ValueError(f"{role} must be indexed by a DatetimeIndex.")
                history_columns = layouts[role, "history"][2]
                unknown = columns.difference(history_columns).tolist()
                if unknown:
                    raise ValueError(
                        f"{role} has columns not in the configured variables: {unknown}."
                    )
                layouts[role, kind] = (ids, index, columns.rename(None))
        if not combine and all(kind == "history" for _, kind in layouts):
            if all(
                requested(self._catalogue[s]) == self._catalogue[s]["metric"]
                for ids, _, _ in layouts.values()
                for s in ids
            ):
                return self._replace(layouts=layouts)

        paths = dict(self._paths())
        output_paths = {}
        for role in ("y", "X"):
            history = paths.get((role, "history"))
            if history is None:
                continue
            if (
                combine
                and not (mapping is not None and role == "y")
                and not any(
                    (role, kind) in paths for kind in ("published", "conditioning")
                )
            ):
                continue
            index, columns, series = history
            future = {
                entry["variable"]: (entry, values)
                for entry, values in paths.get((role, "conditioning"), (None, None, []))[
                    2
                ]
            }
            published = {
                entry["variable"]: (entry, values)
                for entry, values in paths.get((role, "published"), (None, None, []))[2]
            }
            converted = {kind: [] for kind in ("history", "published", "conditioning")}
            combined_index = index
            if combine:
                for kind in ("published", "conditioning"):
                    if (role, kind) in paths and len(paths[role, kind][1]):
                        combined_index = combined_index.union(
                            paths[role, kind][0]
                        ).sort_values()
                if (
                    isinstance(combined_index, pd.DatetimeIndex)
                    and combined_index.freq is None
                ):
                    combined_index = combined_index.copy(deep=True)
                    combined_index.freq = combined_index.inferred_freq
            for entry, values in series:
                variable = entry["variable"]
                metric = requested(entry)
                if mapping is None and metric == "levels" and entry["metric"] != "levels":
                    raise ValueError(
                        f"Cannot transform variable '{variable}' from metric "
                        f"'{entry['metric']}' to 'levels'; only levels-derived "
                        "conversions are supported."
                    )
                base = values
                if variable in published:
                    published_entry, published_values = published[variable]
                    if published_entry["metric"] != entry["metric"]:
                        raise ValueError(
                            "Published observations must retain their history "
                            "source metric."
                        )
                    base = _overlay_values(base, published_values)
                future_entry, future_values = future.get(variable, (entry, None))
                trajectory = _transform_trajectory(
                    base,
                    future_values,
                    metric,
                    entry["metric"],
                    entry["frequency"],
                    future_input_metric=future_entry["metric"],
                )
                output_entry = entry | {"metric": metric}
                converted["history"].append(
                    (
                        output_entry,
                        trajectory.reindex(combined_index if combine else index),
                    )
                )
                if combine:
                    continue
                if variable in published:
                    converted["published"].append(
                        (output_entry, trajectory.reindex(published[variable][1].index))
                    )
                if future_values is not None:
                    converted["conditioning"].append(
                        (
                            future_entry | {"metric": metric},
                            trajectory.reindex(future_values.index),
                        )
                    )
            for kind, output in converted.items():
                if (role, kind) not in paths or combine and kind != "history":
                    continue
                _, path_index, path_columns = layouts[role, kind]
                output_columns = pd.Index(
                    [entry["variable"] for entry, _ in output], name=path_columns.name
                )
                output_paths[role, kind] = (
                    combined_index if combine else path_index,
                    output_columns,
                    output,
                )
        return self._from_paths(output_paths)

    def regularise(self):
        """Materialise internal calendar gaps on each series' own frequency."""
        paths = {}
        changed = False
        for key, (index, columns, series) in self._paths():
            index_name = index.name
            result = [
                (entry, _regularise_series(values, entry["frequency"]))
                for entry, values in series
            ]
            changed |= any(
                old is not new for (_, old), (_, new) in zip(series, result, strict=True)
            )
            for _, values in result:
                index = index.union(values.index)
            paths[key] = (index.sort_values().rename(index_name), columns, result)
        return self._from_paths(paths) if changed else self

    def impute(self, last_date, steps=0, method="zero", random_state=0, *, role="X"):
        """Extend ragged series with one generator consumed in declaration order."""
        paths = dict(self._paths())
        key = (role, "history")
        if key not in paths:
            return self
        index, columns, series = paths[key]
        all_missing = [
            entry["variable"] for entry, values in series if values.isna().all()
        ]
        if all_missing:
            raise ValueError(
                f"Cannot impute regressors with no observations: {all_missing}. "
                "Provide at least one finite value for each regressor."
            )
        if not series or not len(index):
            return self
        rng = np.random.default_rng(random_state)
        result = [
            (
                entry,
                _impute_series(values, last_date, steps, method, rng, entry["frequency"]),
            )
            for entry, values in series
        ]
        if all(old is new for (_, old), (_, new) in zip(series, result, strict=True)):
            return self
        index = result[0][1].index
        for _, values in result[1:]:
            index = index.union(values.index)
        paths[key] = (index.sort_values(), columns, result)
        return self._from_paths(paths)

    def convert_trajectories(self, metric, *, periods=None, sort_vintages=False):
        """Convert supplied vintage groups without archive backfilling."""
        data = self.to_long()
        groups = []
        for _, group in data.groupby("vintage_date", sort=sort_vintages):
            trajectory = _ordered_trajectory(group)
            frequency = (
                _resolve_frequency(trajectory["frequency"])
                if metric in _CALENDAR_DEPENDENT_METRICS or periods is not None
                else None
            )
            values = trajectory.set_index("date")["value"]
            converted = (
                _growth_series(_calendar_align(values, frequency), periods)
                if periods is not None
                else _transform_metric(values, metric, frequency)
            )
            groups.append(group.assign(value=group["date"].map(converted)))
        result = pd.concat(groups, ignore_index=True) if groups else data.copy()
        return type(self).from_long(result)

    def derive(self, variables, mapping):
        """Retain supplied rows and append requested metrics with their metadata."""
        data = self.to_long()
        derived = []
        for variable in variables:
            metric = mapping[variable]
            rows = data.loc[data["variable"].eq(variable)]
            available = rows["metric"].unique()
            if metric in available:
                continue
            if "levels" not in available or metric not in _VALID_METRICS:
                raise ValueError(
                    f"Cannot compute transformation '{metric}'for variable '{variable}'. "
                    f"Available metrics: {list(available)}. Please ensure 'levels' "
                    "or an appropriate base metric is available."
                )
            levels = type(self).from_long(rows.loc[rows["metric"].eq("levels")])
            converted = levels.convert_trajectories(metric).to_long()
            if metric in _CALENDAR_DEPENDENT_METRICS:
                converted = converted.dropna(subset=["value"])
            derived.append(converted.assign(metric=metric))
        result = pd.concat([data, *derived], ignore_index=True) if derived else data
        return type(self).from_long(result)

    def reconstruct_levels(self, history, variables, mapping):
        """Append supported level reconstructions from vintage-safe anchors."""
        forecasts = self.to_long()
        outturns = history.to_long()
        reconstructed = []
        for variable in variables:
            metric = mapping[variable]
            levels = outturns.loc[
                outturns["variable"].eq(variable) & outturns["metric"].eq("levels")
            ]
            if levels.empty or metric not in {"logs", "diff", "log diff"}:
                continue
            rows = forecasts.loc[forecasts["variable"].eq(variable)]
            if metric == "logs":
                reconstructed.append(
                    rows.assign(value=np.exp(rows["value"]), metric="levels")
                )
                continue
            archive = type(self).from_long(levels, semantics="archive")
            group_keys = ["vintage_date"] + (["source"] if "source" in rows else [])
            for key, group in rows.groupby(group_keys, sort=False):
                group = group.sort_values("date")
                vintage = key[0] if isinstance(key, tuple) else key
                anchors = archive.as_of(vintage).to_long()
                anchors = anchors.loc[
                    anchors["date"].lt(group["date"].min())
                ].sort_values("date")
                if not anchors.empty:
                    reconstruct = (
                        _reconstruct_additive
                        if metric == "diff"
                        else _reconstruct_logarithmic
                    )
                    values = reconstruct(anchors.iloc[-1]["value"], group["value"])
                    reconstructed.append(group.assign(value=values, metric="levels"))
        result = (
            pd.concat([forecasts, *reconstructed], ignore_index=True)
            if reconstructed
            else forecasts
        )
        return type(self).from_long(result)

    def to_long(self):
        """Restore labelled rows, original layout and caller metadata."""
        if self._metadata is None:
            rows = self._observations.copy()
            rows["variable"] = rows["series"].map(
                lambda s: self._catalogue[s]["variable"]
            )
            rows["metric"] = rows["series"].map(lambda s: self._catalogue[s]["metric"])
            return rows.drop(columns="series").reset_index(drop=True)
        metadata = self._metadata.loc[
            self._metadata["_row"].isin(self._observations.index)
        ]
        rows = self._observations.loc[metadata["_row"]].copy()
        rows.index = metadata.index
        result = metadata.drop(columns="_row").copy()
        for column in ("date", "vintage_date", "value"):
            result[column] = rows[column].array
        for column, key in (
            ("variable", "variable"),
            ("metric", "metric"),
            ("source", "source"),
            ("_type", "kind"),
        ):
            if column in self._long_columns:
                result[column] = [self._catalogue[s][key] for s in rows["series"]]
        return result.reindex(columns=self._long_columns).astype(self._long_dtypes)


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


def _validate_mapping_coverage(
    data_transformation: dict[str, str],
    y_variables: list[str],
    X_variables: list[str] | None,
) -> None:
    """*data_transformation* must map every y/X variable to a required metric."""
    if not set(y_variables).issubset(data_transformation.keys()):
        raise ValueError(
            "data_transformation must contain all y_variables. "
            f"Got {list(data_transformation.keys())},"
            f"but expected to include {y_variables}"
        )

    if X_variables is not None and not set(X_variables).issubset(
        data_transformation.keys()
    ):
        raise ValueError(
            "data_transformation must contain all X_variables. "
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


def _transform_trajectory(
    history: pd.Series,
    future: pd.Series | None,
    metric: str,
    input_metric: str = "levels",
    frequency: str | None = None,
    *,
    future_input_metric: str | None = None,
) -> pd.Series:
    """Reconcile source units before overlaying a complete variable trajectory."""
    future_input_metric = (
        input_metric if future_input_metric is None else future_input_metric
    )
    if future is not None and input_metric != future_input_metric:
        if input_metric == "levels" and future_input_metric == metric:
            transformed_history = _transform_metric(history, metric, frequency)
            return _overlay_values(transformed_history, future)
        raise ValueError(
            f"Cannot combine variable '{history.name}' from source metrics "
            f"'{input_metric}' and '{future_input_metric}' for requested "
            f"metric '{metric}'; history must be levels and future values "
            "must already use the requested metric."
        )

    if input_metric == metric and future_input_metric == metric:
        return _overlay_values(history, future)
    if input_metric != "levels" or future_input_metric != "levels":
        raise ValueError(
            f"Cannot transform variable '{history.name}' from metric "
            f"'{input_metric}' or '{future_input_metric}' to '{metric}'; "
            "only levels-derived conversions "
            "are supported."
        )

    combined = _overlay_values(history, future)

    return _transform_metric(combined, metric, frequency)


def _ar1_t_impute(observed, shortage, rng):
    """Simulate future values from a Student-t AR(1) model.

    The last observed value is repeated when the model cannot be fitted.

    Args:
        observed : array-like
            The observed (in-sample) values of the column to extrapolate.
        shortage : int
            Number of future values to simulate.
        rng : np.random.Generator
            Random number generator used to draw the Student-t innovations.

    Returns:
        list[float] : The ``shortage`` simulated future values.
    """
    if shortage <= 0:
        return []

    values = np.asarray(observed, dtype=float)
    values = values[np.isfinite(values)]

    def _last_value_fallback():
        last = values[-1] if len(values) else 0.0
        return [float(last)] * shortage

    if len(values) < 5:
        # Too few observations to fit the model; repeat the last observed value.
        return _last_value_fallback()

    if np.all(values == values[0]):
        return _last_value_fallback()

    y_t = values[1:]
    y_lag = values[:-1]

    # OLS starting values for the intercept, persistence and innovation scale.
    design = np.column_stack([np.ones_like(y_lag), y_lag])
    try:
        beta, *_ = np.linalg.lstsq(design, y_t, rcond=None)
        c0, phi0 = beta[0], beta[1]
        resid = y_t - design @ beta
        sigma0 = np.sqrt(np.sum(resid**2) / max(len(resid) - 2, 1))
    except (np.linalg.LinAlgError, ValueError):
        return _last_value_fallback()

    phi0 = np.clip(phi0, -0.99, 0.99)
    if not np.isfinite(c0) or not np.isfinite(phi0) or not np.isfinite(sigma0):
        return _last_value_fallback()
    sigma0 = sigma0 if sigma0 > 0 else 1.0

    # Unconstrained parametrisation for the maximum-likelihood fit:
    #   phi   = tanh(z_phi)           -> |phi| < 1  (stationary)
    #   scale = exp(log_scale)        -> scale > 0
    #   nu    = 2 + exp(log_nu_excess) -> nu > 2     (finite variance)
    x0 = np.array(
        [
            np.arctanh(np.clip(phi0, -0.99, 0.99)),
            c0,
            np.log(sigma0),
            np.log(3.0),  # start at nu = 5
        ]
    )

    def _transformed_parameters(params):
        z_phi, c, log_scale, log_nu_excess = params
        return c, np.tanh(z_phi), np.exp(log_scale), 2.0 + np.exp(log_nu_excess)

    def _valid_parameters(params):
        try:
            c, phi, scale, nu = _transformed_parameters(params)
        except (FloatingPointError, OverflowError, ValueError):
            return None
        if (
            not np.isfinite(c)
            or not np.isfinite(phi)
            or abs(phi) >= 1
            or not np.isfinite(scale)
            or scale <= 0
            or not np.isfinite(nu)
            or nu <= 2
        ):
            return None
        return c, phi, scale, nu

    def _simulate(params):
        transformed = _valid_parameters(params)
        if transformed is None:
            return None
        c, phi, scale, nu = transformed
        fill = []
        x_prev = values[-1]
        for _ in range(shortage):
            eps = rng.standard_t(nu) * scale
            x_prev = c + phi * x_prev + eps
            if not np.isfinite(x_prev):
                return None
            fill.append(float(x_prev))
        return fill

    def _neg_log_likelihood(params):
        z_phi, c, log_scale, log_nu_excess = params
        phi = np.tanh(z_phi)
        scale = np.exp(log_scale)
        nu = 2.0 + np.exp(log_nu_excess)
        mu = c + phi * y_lag
        return -np.sum(student_t.logpdf(y_t, df=nu, loc=mu, scale=scale))

    try:
        result = minimize(_neg_log_likelihood, x0, method="Nelder-Mead")
    except (ValueError, FloatingPointError):
        return _simulate(x0) or _last_value_fallback()

    try:
        params = np.asarray(getattr(result, "x", []), dtype=float)
    except (TypeError, ValueError):
        params = np.empty(0)
    if (
        not getattr(result, "success", False)
        or params.shape != (4,)
        or not np.all(np.isfinite(params))
    ):
        params = x0

    fill = _simulate(params)
    if fill is None and not np.array_equal(params, x0):
        fill = _simulate(x0)
    return fill or _last_value_fallback()


def _impute_series(values, last_date, steps, method, rng, frequency):
    """Pad or trim a single calendar without consuming unnecessary draws."""
    if frequency is None:
        raise ValueError(f"No frequency was supplied for X column '{values.name}'.")
    original = values if values.index.is_monotonic_increasing else values.sort_index()
    observed = original.dropna()
    offset = pd.tseries.frequencies.to_offset(
        {"M": "ME", "Q": "QE-DEC"}.get(frequency, frequency)
    )
    target = last_date + offset * steps
    last = observed.index[-1]
    shortage = (pd.Period(target, freq=offset) - pd.Period(last, freq=offset)).n
    if shortage == 0 and last == original.index[-1]:
        return original
    if shortage <= 0:
        trimmed_last = observed.iloc[: len(observed) + shortage].index[-1]
        return original.loc[original.index <= trimmed_last]
    if method == "last":
        fill = [observed.iloc[-1]] * shortage
    elif method == "mean":
        fill = [observed.mean()] * shortage
    elif method == "ar1_t":
        fill = _ar1_t_impute(observed, shortage, rng)
    else:
        fill = [0.0] * shortage
    dates = pd.date_range(start=last, periods=shortage + 1, freq=offset)[1:]
    return pd.concat([original.loc[original.index <= last], pd.Series(fill, index=dates)])


def _regularise_series(values, frequency):
    """Materialise absent dates without changing month anchors or PeriodIndex."""
    series = values if values.index.is_monotonic_increasing else values.sort_index()
    first, last = series.first_valid_index(), series.last_valid_index()
    if first is None or last is None or frequency is None:
        return series
    periods = pd.period_range(
        start=pd.Period(first, freq=frequency),
        end=pd.Period(last, freq=frequency),
        freq=frequency,
    )
    if isinstance(series.index, pd.PeriodIndex):
        complete_index = periods
    else:
        anchor = (
            "start"
            if pd.DatetimeIndex(series.dropna().index).is_month_start.all()
            else "end"
        )
        complete_index = periods.to_timestamp(how=anchor).normalize()
    index = series.index.union(complete_index).sort_values()
    return series if index.equals(series.index) else series.reindex(index)


def _overlay_values(history, future):
    """Overlay non-null future cells, retaining history labels and date metadata."""
    if future is None or future.empty:
        return history
    combined = future.combine_first(history).sort_index()
    combined = (
        combined.reindex(columns=history.columns)
        if isinstance(history, pd.DataFrame)
        else combined.rename(history.name)
    )
    if isinstance(combined.index, pd.DatetimeIndex) and combined.index.freq is None:
        combined.index = combined.index.copy(deep=True)
        combined.index.freq = combined.index.inferred_freq
    return combined
