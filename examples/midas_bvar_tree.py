"""Example of a MIDAS nowcast feeding a conditional BVAR forecast tree."""

import numpy as np
import pandas as pd

from forecast_realtime import ForecastTree, TreeNode
from forecast_realtime.models import ForecastBVAR, ForecastMIDAS


# Children are matched to the BVAR's fitted variables by name, so each leaf
# must be labelled after the variable it nowcasts. Unmatched variables and
# uncovered horizons remain NaN and are left unconstrained.
class ConditionalBVAR(ForecastBVAR):
    """A ``ForecastBVAR`` conditioned by the MIDAS leaves' nowcasts.

    Parameters
    ----------
    conditioning_steps : int | None
        Condition on the first ``conditioning_steps`` horizons only, leaving
        later ones unconstrained. Default None (use every horizon supplied).
    """

    def __init__(self, conditioning_steps: int | None = None, **kwargs) -> None:
        if conditioning_steps is not None and (
            type(conditioning_steps) is not int or conditioning_steps < 1
        ):
            raise ValueError("conditioning_steps must be a positive integer or None")
        super().__init__(**kwargs)
        self.conditioning_steps = conditioning_steps

    def _prepare_estimation_inputs(self, y, X):
        return y, None

    def _forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ):
        """Turn the children's nowcasts in ``X`` into conditioning paths."""
        output_dates = None
        if X is not None:
            origin = kwargs.get("forecast_origin") or self.last_y_fit_date
            future_nowcasts = X.loc[X.index > origin].dropna(how="all")
            future_dates = future_nowcasts.index
            if len(future_dates) >= steps:
                output_dates = future_dates[:steps]
                frequency = self._forecast_frequency or self._infer_calendar_frequency(
                    self.y.index
                )
                steps = (
                    pd.Period(output_dates[-1], freq=frequency).ordinal
                    - pd.Period(origin, freq=frequency).ordinal
                )
            y = self._conditioning_from_nowcasts(X, steps, y)
        # Kept for inspection: which paths the BVAR was actually conditioned on.
        self.conditioning_ = y
        forecast = super()._forecast(steps=steps, X=None, y=y, **kwargs)
        return forecast if output_dates is None else forecast.reindex(output_dates)

    def _conditioning_from_nowcasts(
        self,
        midas_nowcasts: pd.DataFrame,
        steps: int,
        published_values: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Fill unpublished BVAR values with aligned MIDAS nowcasts."""
        matched = [col for col in self.y.columns if col in midas_nowcasts.columns]
        if not matched:
            raise ValueError(
                "none of the conditioning columns match the fitted BVAR variables"
            )

        dates = self._infer_forecast_dates(self.y.index, steps, frequency="Q")
        # Drop anything at or before the fit sample (and any NaT-dated horizon)
        # so the paths reindex cleanly onto the forecast dates.
        future = midas_nowcasts.loc[midas_nowcasts.index > self.last_y_fit_date, matched]

        conditioning = pd.DataFrame(np.nan, index=dates, columns=self.y.columns)
        if published_values is not None:
            conditioning = published_values.reindex(
                index=dates, columns=self.y.columns
            ).combine_first(conditioning)
        conditioning[matched] = conditioning[matched].combine_first(future.reindex(dates))

        if self.conditioning_steps is not None:
            conditioning.iloc[self.conditioning_steps :] = np.nan

        return conditioning


def BVARMIDASTree() -> ForecastTree:
    """Build a conditional BVAR fed by MIDAS nowcasts."""
    bvar_variables = ["quarterly_1", "quarterly_2"]
    leaves = [
        ForecastMIDAS(
            label="quarterly_1",
            formula="quarterly_1 ~ monthly_1",
            horizons=[0, 1],
            n_lags=6,
        ),
        ForecastMIDAS(
            label="quarterly_2",
            formula="quarterly_2 ~ monthly_2",
            horizons=[0, 1],
            n_lags=6,
        ),
    ]
    spec = TreeNode(
        name="conditional_bvar",
        children=leaves,
        transform=ConditionalBVAR(
            n_lags=2,
            mode_only=True,
            progressbar=False,
            formula=f"{' + '.join(bvar_variables)} ~ {' + '.join(bvar_variables)}",
            data_transformation={variable: "pop" for variable in bvar_variables},
        ),
    )
    return ForecastTree(spec=spec, label="MIDAS BVAR")


midas_bvar_model = BVARMIDASTree()
