"""Helpers for building lagged feature matrices used by ForecastModel."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ._model_data import ModelData
from ._model_data import _ar1_t_impute as _ar1_t_impute


def validate_forecast_horizons(horizons, steps: int, model_name: str) -> None:
    """Raise when a backend omits one of the requested forecast horizons."""
    returned = set(pd.to_numeric(horizons, errors="raise").astype(int))
    expected = set(range(steps))
    missing = sorted(expected - returned)
    if missing:
        raise ValueError(
            f"{model_name} did not produce forecasts for horizon(s) {missing}; "
            "the backend fit did not converge for every requested horizon."
        )


def resolve_X_lags(X_lags: int | dict, columns) -> dict:
    """Turn ``X_lags`` (int or dict) into a ``{col: int}`` map for ``columns``."""
    if isinstance(X_lags, dict):
        unknown_columns = sorted(set(X_lags) - set(columns), key=str)
        if unknown_columns:
            warnings.warn(
                f"X_lags contains columns absent from X: {unknown_columns}",
                UserWarning,
                stacklevel=2,
            )
        return {c: int(X_lags.get(c, 0)) for c in columns}
    return {c: int(X_lags) for c in columns}


def build_lagged_design(
    y: pd.DataFrame,
    X: pd.DataFrame | None,
    y_lags: int,
    X_lags: int | dict[str, int],
) -> pd.DataFrame:
    """Build augmented design matrix with lag columns for y and X.

    **Lag semantics — all lags from 1 to k are included.**
    ``y_lags=k`` appends columns ``y_lag1, y_lag2, ..., y_lagk``.
    ``X_lags=k`` (or ``X_lags={col: k}``) appends columns
    ``col_lag1, col_lag2, ..., col_lagk`` for each X column.
    Base (unlagged) X columns are always kept.

    Column order in output::

        [X_col1, X_col2, ...,            # base X (unlagged)
         y_lag1, y_lag2, ...,          # AR lags of y (if y_lags > 0)
         X_col1_lag1, X_col1_lag2, ...,  # X lags
         X_col2_lag1, X_col2_lag2, ...]

    Input series should already be regularised at their declared frequency.
    Rows with NaNs in lag features are retained for the model to handle.

    Parameters
    ----------
    y : pd.DataFrame
        Target variable. First column is used.
    X : pd.DataFrame | None
        Exogenous features. May extend beyond y's date range.
    y_lags : int
        Number of y autoregressive lags.
    X_lags : int | dict[str, int]
        Lags for each X column.

    Returns
    -------
    pd.DataFrame
        Augmented design matrix with lag features (no y, no intercept).
        Index preserves the input DatetimeIndex; rows with NaN lag values are
        retained.
    """
    common_index = y.index
    if X is not None:
        common_index = common_index.union(X.index)

    y_s = y.iloc[:, 0].astype(float).reindex(common_index)
    y_name = y.columns[0]
    if X is not None:
        X = X.astype(float).reindex(common_index)
        X_lags_map = resolve_X_lags(X_lags, X.columns)
    else:
        X_lags_map = {}

    parts = []
    if X is not None:
        parts.append(X)
    for k in range(1, y_lags + 1):
        parts.append(y_s.shift(k).rename(f"{y_name}_lag{k}"))
    if X is not None:
        for col, nlag in X_lags_map.items():
            for k in range(1, nlag + 1):
                parts.append(X[col].shift(k).rename(f"{col}_lag{k}"))

    if not parts:
        raise ValueError("Model requires regressors X or y_lags > 0")

    X_aug = pd.concat(parts, axis=1, sort=False)
    if X_aug.empty:
        raise ValueError("No rows left after dropping NaNs; reduce lags or add data")

    return X_aug


def init_recent_y(X_aug: pd.DataFrame, y_name: str, n_lags: int) -> list[float]:
    """Return recent target values for recursive lag columns.

    Parameters
    ----------
    X_aug : pd.DataFrame
        Forecast-row design matrix with the target lag columns.
    y_name : str
        Target column name.
    n_lags : int
        Number of target lag columns to read.

    Returns
    -------
    list[float]
        Lag values ordered from the most recent to the oldest.
    """
    first_row = X_aug.iloc[0]
    return [float(first_row[f"{y_name}_lag{lag}"]) for lag in range(1, n_lags + 1)]


def _period_label(ts: pd.Timestamp, target_frequency: str) -> str:
    """Format ``ts`` as a period-style dummy name for the target frequency.

    Returns ``D_2020Q1`` for quarterly data, ``D_2020M1`` for monthly,
    ``D_2020`` for annual, and falls back to ``D_2020-06-30`` (ISO date)
    when the frequency cannot be resolved to year/quarter/month/annual.
    """
    code = target_frequency.upper()

    if code in ("Q", "QE", "QS"):
        return f"D_{ts.year}Q{ts.quarter}"
    if code in ("M", "ME", "MS"):
        return f"D_{ts.year}M{ts.month}"
    if code in ("A", "Y", "YE", "YS"):
        return f"D_{ts.year}"
    return f"D_{ts.date()}"


def build_dummies(
    index: pd.DatetimeIndex,
    dummies: list | dict,
    target_frequency: str,
) -> pd.DataFrame:
    """Build 0/1 point-dummy columns from a ``DatetimeIndex``.

    ``dummies`` may be either:

    - a list of dates, e.g. ``["2020-06-30", "2020-09-30"]``. Each date
      becomes a column named after its period: ``D_2020Q2`` for quarterly
      data, ``D_2020M6`` for monthly, ``D_2020`` for annual (falling back
      to ``D_<YYYY-MM-DD>`` when the frequency cannot be inferred).
    - a dict mapping a column name to a date, e.g.
      ``{"covid": "2020-06-30"}``, when you want to name the columns
      yourself.

    Parameters
    ----------
    index : pd.DatetimeIndex
        The dates (history + forecast horizon) to generate dummies for.
    dummies : list | dict
        Dates at which to set the dummy columns to 1.
    target_frequency : str
        Resolved frequency of the target series used for list or tuple dummy
        names. Must be supplied even when names are explicit.

    Returns
    -------
    pd.DataFrame
        One 0/1 column per dummy date, indexed by ``index``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("build_dummies requires a DatetimeIndex")
    if isinstance(dummies, dict):
        items = list(dummies.items())
    elif isinstance(dummies, (list, tuple)):
        items = [(_period_label(pd.Timestamp(d), target_frequency), d) for d in dummies]
    else:
        raise TypeError("dummies must be a list of dates or a dict {name: date}")

    cols: dict[str, np.ndarray] = {}
    for name, date in items:
        ts = pd.Timestamp(date)
        cols[name] = (index == ts).astype(float)

    return pd.DataFrame(cols, index=index)


def impute_X(
    X: pd.DataFrame,
    last_date: pd.Timestamp,
    steps: int = 0,
    method: str = "zero",
    random_state: int | None = 0,
    *,
    frequencies: dict[str, str],
) -> pd.DataFrame:
    """Impute a regressor matrix so every column extends to a common last date.

    Used by ``ForecastModel.fit()``/``forecast()``, which apply it after
    semantic data transformation. Each series/column is imputed separately:
    shorter columns (ragged edges) are padded up to ``last_date`` plus
    ``steps`` future periods, while longer columns are trimmed to that same
    target date.

    - For the fitting design call with ``steps=0`` so columns are aligned to
      the last fitted date (fills ragged edges only, no future rows).
    - For the forecast design call with ``steps`` equal to the forecast
      horizon so the required future rows are padded as well.

    Parameters
    ----------
    X : pd.DataFrame
        The regressor matrix to impute (historical, or historical + future).
    last_date : pd.Timestamp
        The reference last date; rows after ``last_date + steps`` periods are
        treated as surplus and trimmed.
    steps : int
        Number of future periods (beyond ``last_date``) each column must
        reach. Default 0 (no future rows, used for the fitting design).
    method : str
        ``"zero"`` (default) — fill with 0.
        ``"last"`` — repeat the last observed value (random-walk).
        ``"mean"`` — fill with the in-sample column mean.
        ``"ar1_t"`` — simulate forward from a stationary AR(1) model fitted
        by maximum likelihood, with Student-t innovations. The model estimates
        the innovations' degrees of freedom from the data.
    random_state : int | None
        Seed for the random number generator used by the ``"ar1_t"``
        method. Default 0 (reproducible); pass None for non-deterministic
        draws.
    frequencies : dict[str, str]
        Resolved frequency for each X column. Each value controls that
        column's padding and trimming calendar.

    Returns
    -------
    pd.DataFrame
        ``X`` with every column extending to its own ``last_date + steps``
        periods, on its own supplied frequency.
    """
    return (
        ModelData.from_wide(X=X, frequencies=frequencies)
        .impute(last_date, steps, method, random_state)
        .to_wide("X")
    )


def regularise_missing_rows(
    data: pd.DataFrame | None,
    frequencies: dict[str, str],
) -> pd.DataFrame | None:
    """Materialise absent dates inside each series' observed span.

    Used by ``ForecastModel.fit()``, which applies it after semantic data
    transformation for models that do not handle missing values themselves.
    Every column must have a resolved frequency.
    """
    if data is None or data.empty:
        return data
    return ModelData.from_wide(y=data, frequencies=frequencies).regularise().to_wide()
