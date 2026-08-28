"""Wrapper around R's ``lm()`` for AR-style forecasting.

Requires R with the ``arrow`` package installed::

    install.packages("arrow")
"""

from pathlib import Path

from forecast_realtime.external_model import RModel

# Path to the bundled R script (shipped alongside this module)
_R_SCRIPT = str(Path(__file__).parent / "r_scripts" / "forecast_lm.r")


class ForecastRlm(RModel):
    """AR regression model fitted in R via ``lm()``.

    Parameters
    ----------
    lags : int
        Number of autoregressive lags to include.

    """

    def __init__(self, lags: int = 1, **kwargs):
        if not isinstance(lags, int) or lags < 1:
            raise ValueError("lags must be a positive integer")

        super().__init__(
            script=_R_SCRIPT,
            lags=lags,
            **kwargs,
        )
