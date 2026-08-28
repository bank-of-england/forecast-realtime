"""Fable forecasting models backed by R."""

from pathlib import Path

from forecast_realtime.external_model import RModel

_R_SCRIPT = str(Path(__file__).parent / "r_scripts" / "fable.r")
_INDEX_TYPES = ("auto", "quarter", "month", "date")


def _r_string(value: str) -> str:
    """Return a quoted R string literal."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _r_literal(value: int | str) -> str:
    if isinstance(value, bool):
        raise TypeError("period must be an integer or string")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _r_string(value)
    raise TypeError("period must be an integer or string")


def _validate_component(value: str | None, name: str, choices: tuple[str, ...]):
    if value is not None and value not in choices:
        allowed = ", ".join(choices)
        raise ValueError(f"{name} must be one of {allowed}; got {value!r}")


def _order_term(
    name: str,
    orders: tuple[int | None, int | None, int | None],
) -> str:
    if all(order is None for order in orders):
        return f"{name}()"
    if any(order is None for order in orders):
        raise ValueError(f"{name} orders must be all specified or all None")
    if any(
        not isinstance(order, int) or isinstance(order, bool) or order < 0
        for order in orders
    ):
        raise ValueError(f"{name} orders must be non-negative integers")
    return f"{name}({orders[0]}, {orders[1]}, {orders[2]})"


class RFableModel(RModel):
    """Run a univariate fable model through the external R-model bridge.

    ``spec`` is an R expression such as ``ARIMA(value ~ 1 + pdq(1, 0, 0))``.
    It is evaluated as trusted R code in the external process; it is not
    treated as an inert data string or sandboxed expression.
    The response is named ``value`` inside R; regressor names retain their
    Python column names. The generic class is useful for fable models that do
    not need a dedicated Python convenience wrapper.
    """

    _handles_missing_values = False
    _R_SCRIPT = _R_SCRIPT

    def _normalise_forecasts(self, forecasts):
        if list(forecasts.columns) == ["value"]:
            return forecasts.rename(columns={"value": self.y.columns[0]})
        return forecasts

    def __init__(
        self,
        spec: str,
        *,
        index: str = "auto",
        allow_xreg: bool = True,
        label: str | None = None,
        formula: str | None = None,
        data_transformation=None,
        **params,
    ):
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("spec must be a non-empty R model expression")
        if index not in _INDEX_TYPES:
            allowed = ", ".join(_INDEX_TYPES)
            raise ValueError(f"index must be one of {allowed}; got {index!r}")

        super().__init__(
            script=self._R_SCRIPT,
            label=label,
            formula=formula,
            data_transformation=data_transformation,
            spec=spec,
            index=index,
            allow_xreg=allow_xreg,
            **params,
        )
        self.spec = spec
        self.index = index
        self.allow_xreg = allow_xreg


class RFableETS(RFableModel):
    """Fable exponential-smoothing model for a univariate series.

    Parameters
    ----------
    error : str | None
        Error type passed to the Fable model.
    trend : str | None
        Trend type passed to the Fable model.
    season : str | None
        Seasonal type passed to the Fable model. Default ``"N"``.
    period : int | str | None
        Seasonal period. Requires a seasonal term.
    index : str
        Date-index conversion used by the R model. Default ``"auto"``.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target and regressors.
    """

    def __init__(
        self,
        *,
        error: str | None = None,
        trend: str | None = None,
        season: str | None = "N",
        period: int | str | None = None,
        index: str = "auto",
        label: str | None = None,
        formula: str | None = None,
        **kwargs,
    ):
        _validate_component(error, "error", ("A", "M"))
        _validate_component(trend, "trend", ("N", "A", "Ad"))
        _validate_component(season, "season", ("N", "A", "M"))
        if period is not None and season in (None, "N"):
            raise ValueError("period cannot be supplied without a seasonal term")

        terms = []
        if error is not None:
            terms.append(f"error({_r_string(error)})")
        if trend is not None:
            terms.append(f"trend({_r_string(trend)})")
        if season is not None:
            season_term = f"season({_r_string(season)}"
            if period is not None:
                season_term += f", period={_r_literal(period)}"
            terms.append(f"{season_term})")
        rhs = " + ".join(terms) if terms else "1"
        spec = f"ETS(value ~ {rhs})"

        super().__init__(
            spec,
            index=index,
            allow_xreg=False,
            label=label,
            formula=formula,
            **kwargs,
        )
        self.error = error
        self.trend = trend
        self.season = season
        self.period = period


class RFableARIMA(RFableModel):
    """Fable ARIMA model for a univariate series.

    Parameters
    ----------
    p : int | None
        Non-seasonal ARIMA orders.
    d : int | None
        Non-seasonal differencing order.
    q : int | None
        Non-seasonal moving-average order.
    seasonal : bool
        Whether to include seasonal ARIMA terms. Default ``False``.
    P : int | None
        Seasonal autoregressive order. Requires ``seasonal=True``.
    D : int | None
        Seasonal differencing order. Requires ``seasonal=True``.
    Q : int | None
        Seasonal moving-average order. Requires ``seasonal=True``.
    period : int | str | None
        Seasonal period. Requires ``seasonal=True``.
    xreg : str | None
        R expression naming an exogenous regressor.
    index : str
        Date-index conversion used by the R model. Default ``"auto"``.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target and regressors.
    """

    _handles_missing_values = True

    def __init__(
        self,
        *,
        p: int | None = None,
        d: int | None = None,
        q: int | None = None,
        seasonal: bool = False,
        P: int | None = None,
        D: int | None = None,
        Q: int | None = None,
        period: int | str | None = None,
        xreg: str | None = None,
        index: str = "auto",
        label: str | None = None,
        formula: str | None = None,
        **kwargs,
    ):
        if not isinstance(seasonal, bool):
            raise TypeError("seasonal must be a boolean")
        if xreg is not None and (not isinstance(xreg, str) or not xreg.strip()):
            raise ValueError("xreg must be a non-empty R expression or None")
        if not seasonal and any(order is not None for order in (P, D, Q, period)):
            raise ValueError("P, D, Q, and period require seasonal=True")

        nonseasonal = _order_term("pdq", (p, d, q))
        if seasonal:
            seasonal_term = _order_term("PDQ", (P, D, Q))
            if period is not None:
                if seasonal_term == "PDQ()":
                    seasonal_term = f"PDQ(period={_r_literal(period)})"
                else:
                    seasonal_term = seasonal_term[:-1] + f", period={_r_literal(period)})"
        else:
            seasonal_term = "PDQ(0, 0, 0)"

        terms = ["1"]
        if xreg is not None:
            terms.append(xreg.strip())
        terms.extend((nonseasonal, seasonal_term))
        spec = f"ARIMA(value ~ {' + '.join(terms)})"

        super().__init__(
            spec,
            index=index,
            allow_xreg=xreg is not None,
            label=label,
            formula=formula,
            **kwargs,
        )
        self.p = p
        self.d = d
        self.q = q
        self.seasonal = seasonal
        self.P = P
        self.D = D
        self.Q = Q
        self.period = period
        self.xreg = xreg


__all__ = ["RFableModel", "RFableETS", "RFableARIMA"]
