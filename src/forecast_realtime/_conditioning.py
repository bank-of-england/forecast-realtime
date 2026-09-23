"""Conditioning policy definitions and resolution helpers."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class _ConditioningEntry:
    variable: str
    source: str
    periods: int


@dataclass(frozen=True)
class _ConditioningPolicy:
    y: tuple[_ConditioningEntry, ...] = ()
    X: tuple[_ConditioningEntry, ...] = ()


def _parse_conditioning(value, label):
    """Copy and validate a model-owned conditioning policy."""
    if value is None:
        return None
    prefix = f"Model {label!r}: conditioning"
    if not isinstance(value, Mapping):
        raise TypeError(f"{prefix} must be a mapping or None.")
    if set(value) - {"y", "X"}:
        raise ValueError(f"{prefix} accepts only 'y' and 'X' roles.")
    roles = {}
    for role, entries in value.items():
        if not isinstance(entries, Mapping):
            raise TypeError(f"{prefix} {role} must be a variable mapping.")
        parsed = []
        for variable, entry in entries.items():
            name = f"{prefix} {role} variable {variable!r}"
            if not isinstance(variable, str) or not variable:
                raise TypeError(f"{name} must have a non-empty string name.")
            if not isinstance(entry, Mapping):
                raise TypeError(f"{name} must map source and periods.")
            if set(entry) != {"source", "periods"}:
                raise ValueError(f"{name} requires exactly source and periods.")
            source, periods = entry["source"], entry["periods"]
            if not isinstance(source, str) or not source:
                raise TypeError(f"{name} source must be a non-empty string.")
            if type(periods) is not int or periods <= 0:
                raise ValueError(f"{name} periods must be a positive integer.")
            parsed.append(_ConditioningEntry(variable, source, periods))
        roles[role] = tuple(parsed)
    return _ConditioningPolicy(**roles)


def _validate_conditioning(role, selected_variables, horizons, sources, steps):
    horizon_name = f"{role}_steps_ahead"
    source_name = f"{role}_sources"

    for name, value in ((horizon_name, horizons), (source_name, sources)):
        if value is not None and not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping or None.")
    if sources is not None:
        for variable, source in sources.items():
            if not isinstance(source, str) or not source:
                raise TypeError(
                    f"{source_name} variable {variable!r}: source must be a non-empty "
                    "string."
                )

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


def _resolve_conditioning(model, requirements, y_variables, fallback, sources, steps):
    """Resolve one replacement policy or project the validated run fallback."""
    variables = model._conditioning_variables(requirements, y_variables)
    policy = model.conditioning
    resolved = {}
    for role in ("y", "X"):
        if policy is None:
            for suffix in ("sources", "steps_ahead"):
                value = fallback[f"{role}_{suffix}"]
                resolved[f"{role}_{suffix}"] = (
                    None
                    if value is None
                    else {
                        key: item for key, item in value.items() if key in variables[role]
                    }
                )
        else:
            entries = getattr(policy, role)
            for entry in entries:
                if entry.variable not in variables[role]:
                    raise ValueError(
                        f"Model {model.label!r}: conditioning {role} variable "
                        f"{entry.variable!r} is not a selected raw input."
                    )
                if entry.periods > steps:
                    raise ValueError(
                        f"Model {model.label!r}: variable {entry.variable!r} "
                        f"conditioning periods must not exceed steps={steps}."
                    )
            resolved[f"{role}_sources"] = (
                {entry.variable: entry.source for entry in entries} if entries else None
            )
            resolved[f"{role}_steps_ahead"] = (
                {entry.variable: entry.periods - 1 for entry in entries}
                if entries
                else None
            )
        for variable, source in (resolved[f"{role}_sources"] or {}).items():
            if source not in sources:
                raise ValueError(
                    f"Model {model.label!r}: variable {variable!r} has unknown "
                    f"conditioning source {source!r}."
                )
    model._validate_target_conditioning(
        {
            variable
            for variable, horizon in (resolved["y_steps_ahead"] or {}).items()
            if horizon is not None
        }
    )
    return resolved
