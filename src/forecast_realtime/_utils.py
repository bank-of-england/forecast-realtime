"""Helpers for building lagged feature matrices used by ForecastModel."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import t as student_t


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
    # RNG for stochastic imputation methods (e.g. "ar1_t").
    rng = np.random.default_rng(random_state)

    # Impute each column separately, each on its own frequency, so a
    # low-frequency column (e.g. quarterly) mixed with a higher-frequency
    # one (e.g. monthly) is not padded/trimmed using the wrong spacing.
    imputed_columns: dict[str, pd.Series] = {}
    all_missing_columns = X.columns[X.isna().all()].tolist()
    if all_missing_columns:
        raise ValueError(
            "Cannot impute regressors with no observations: "
            f"{all_missing_columns}. Provide at least one finite value for each "
            "regressor."
        )

    if X.empty:
        return X

    for col in X.columns:
        original_col = X[col].sort_index()
        col_values = original_col.dropna()

        try:
            freq = frequencies[col]
        except KeyError:
            raise ValueError(f"No frequency was supplied for X column '{col}'.") from None

        offset_frequency = {"M": "ME", "Q": "QE-DEC"}.get(freq, freq)
        offset = pd.tseries.frequencies.to_offset(offset_frequency)
        target_last_date = last_date + offset * steps

        last_valid_date_col = col_values.index[-1]

        last_period = pd.Period(target_last_date, freq=offset)
        last_col_period = pd.Period(last_valid_date_col, freq=offset)
        shortage = (last_period - last_col_period).n

        if shortage <= 0:
            # trim the surplus values not needed for this column's target,
            # keeping the original (NaN-preserving) values up to that point
            trimmed_last_date = col_values.iloc[: len(col_values) + shortage].index[-1]
            final_series = original_col.loc[original_col.index <= trimmed_last_date]
        else:
            # keep the original (NaN-preserving) values up to the last
            # observation, then append the generated fill values
            base_series = original_col.loc[original_col.index <= last_valid_date_col]
            if method == "last":
                fill_values = [col_values.iloc[-1]] * shortage
            elif method == "mean":
                fill_values = [col_values.mean()] * shortage
            elif method == "ar1_t":
                fill_values = _ar1_t_impute(col_values, shortage, rng)
            else:  # "zero"
                fill_values = [0.0] * shortage

            fill_index = pd.date_range(
                start=last_valid_date_col, periods=shortage + 1, freq=offset
            )[1:]
            final_series = pd.concat(
                [base_series, pd.Series(fill_values, index=fill_index)]
            )

        imputed_columns[col] = final_series

    # union index covering each column's own final (padded/trimmed) range;
    # genuine internal gaps survive since each column's final series is
    # sliced from its original (NaN-preserving) values, but surplus dates
    # trimmed off one column are not resurrected via another column's index
    union_index = None
    for series in imputed_columns.values():
        union_index = (
            series.index if union_index is None else union_index.union(series.index)
        )
    union_index = union_index.sort_values()

    X = pd.DataFrame(
        {col: series.reindex(union_index) for col, series in imputed_columns.items()},
        index=union_index,
    )

    return X


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

    columns = {}
    for column in data.columns:
        series = data[column].sort_index()
        first = series.first_valid_index()
        last = series.last_valid_index()
        frequency = frequencies[column]

        if first is None or last is None or frequency is None:
            columns[column] = series
            continue

        periods = pd.period_range(
            start=pd.Period(first, freq=frequency),
            end=pd.Period(last, freq=frequency),
            freq=frequency,
        )
        if isinstance(data.index, pd.PeriodIndex):
            complete_index = periods
        else:
            valid_dates = pd.DatetimeIndex(series.dropna().index)
            timestamp_anchor = "start" if valid_dates.is_month_start.all() else "end"
            complete_index = periods.to_timestamp(how=timestamp_anchor).normalize()
        columns[column] = series.reindex(series.index.union(complete_index)).sort_index()

    index = None
    for series in columns.values():
        index = series.index if index is None else index.union(series.index)

    result = pd.DataFrame(
        {column: series.reindex(index) for column, series in columns.items()},
        index=index.sort_values(),
    )
    result.index.name = data.index.name
    return result
