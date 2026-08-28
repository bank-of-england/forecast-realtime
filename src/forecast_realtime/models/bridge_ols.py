"""Bridge-equation OLS model for mixed-frequency forecasting."""

import numpy as np
import pandas as pd

from forecast_realtime.forecast_model import ForecastModel
from forecast_realtime.models.ols import ForecastOLS

# ForecastBridgeOLS only supports a quarterly target with regressors that are
# themselves monthly or quarterly; these are the only two frequencies it
# knows how to classify a column by frequency. Ranges are (min, max) median
# day-spacing, so weekly/daily data (median spacing well below 20 days) is
# correctly rejected rather than misclassified as monthly.
_FREQ_DAY_RANGE = {"ME": (20, 45), "QE": (46, 150)}


class ForecastBridgeOLS(ForecastOLS):
    """Bridge-equation OLS: aggregate monthly X to quarterly y, then fit OLS.

    Only supports a quarterly target y. Each X column's frequency is
        inferred from the spacing of its non-NaN index (as in
        :class:`~forecast_realtime.models.midas_combo.ForecastMIDASCombo`) and
        must be monthly or quarterly. Quarterly columns are used as-is; monthly
        columns are aggregated onto `y`'s quarter using a complete-quarter mean. A
        quarter with fewer than three months is left as NaN rather than averaged
        from partial data.

    Parameters
    ----------
    aggregation : str
        Aggregation method for higher-frequency regressors. Only ``"mean"``
        (simple period-average) is currently supported. Default ``"mean"``.
    fit_intercept : bool
        Whether to include an intercept term. Default is True.
    forecast_strategy : str
        Forecasting strategy to use ("recursive" or "direct"). Default is
        "recursive".
    steps : int | None
        For direct forecasting, the horizon to fit. Required when
        ``forecast_strategy="direct"``.
    scale : bool
        Whether to scale X and y before fitting. Default is False.
    label : str | None
        Name used to identify the model's forecasts. Defaults to the class
        name.
    formula : str | None
        Optional patsy-style formula selecting the (aggregated) regressors.
        Default is None.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    drop_nans : bool
        Whether to remove rows containing missing values before fitting.
    align_start_dates : bool
        Whether to align the starts of the target and regressor series.
    """

    _handles_mixed_frequencies = True

    def __init__(
        self,
        aggregation: str = "mean",
        fit_intercept: bool = True,
        forecast_strategy: str = "recursive",
        steps: int | None = None,
        scale: bool = False,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        drop_nans: bool = False,
        align_start_dates: bool = True,
    ):
        if aggregation != "mean":
            raise ValueError(
                f"Unsupported aggregation '{aggregation}'; only 'mean' is currently "
                "supported."
            )
        super().__init__(
            fit_intercept=fit_intercept,
            forecast_strategy=forecast_strategy,
            steps=steps,
            scale=scale,
            label=label,
            formula=formula,
            data_transformation=data_transformation,
            drop_nans=drop_nans,
            align_start_dates=align_start_dates,
        )
        self.aggregation = aggregation

    # ------------------------------------------------------------------ #
    # Frequency inference / aggregation helpers                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _infer_freq_label(idx: pd.DatetimeIndex, col_name: str = "y") -> str:
        """Classify a DatetimeIndex's frequency as monthly ("ME") or quarterly ("QE").

        Raises ``ValueError`` if there are too few observations to infer a
        frequency, or if the inferred frequency is neither monthly nor quarterly.
        """
        if len(idx) < 2:
            raise ValueError(
                f"Cannot infer the frequency of '{col_name}': fewer than 2 "
                "non-NaN observations."
            )
        diffs = np.diff(idx.values).astype("timedelta64[D]").astype(int)
        median_days = float(np.median(diffs))
        for label, (min_days, max_days) in _FREQ_DAY_RANGE.items():
            if min_days <= median_days <= max_days:
                return label
        raise ValueError(
            f"Column '{col_name}' has an inferred frequency that is neither "
            "monthly nor quarterly. ForecastBridgeOLS only supports a "
            "quarterly target with monthly or quarterly regressors."
        )

    def _aggregate_column(self, series: pd.Series) -> pd.Series:
        """Aggregate one monthly X column onto y's quarter via a complete-quarter mean.

        A quarter is complete only when all 3 of its months are present;
        an incomplete trailing quarter is left as NaN rather than averaged
        from partial data.
        """
        s = series.dropna()
        counts = s.resample("QE").count()
        means = s.resample("QE").mean()
        return means.where(counts == 3)

    def _aggregate_X(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """Infer each X column's frequency and aggregate it onto y's quarter.

        Raises ``ValueError`` if y is not quarterly, or if a column is
        inferred to be neither monthly nor quarterly.
        """
        y_label = self._infer_freq_label(y.index, col_name="y")
        if y_label != "QE":
            raise ValueError(
                "ForecastBridgeOLS only supports a quarterly target y; "
                f"inferred frequency was '{y_label}'."
            )

        aggregated = {}
        for col in X.columns:
            x_idx = X[col].dropna().index
            x_label = self._infer_freq_label(x_idx, col_name=col)

            if x_label == "QE":
                aggregated[col] = X[col]
            else:
                aggregated[col] = self._aggregate_column(X[col])

        return pd.DataFrame(aggregated)

    # ------------------------------------------------------------------ #
    # Model-specific input preparation                                  #
    # ------------------------------------------------------------------ #

    def _validate_fit_inputs(self, y, X):
        """Validate inputs without aligning before mixed-frequency aggregation."""
        return ForecastModel._validate_fit_inputs(self, y, X)

    def _prepare_fit_inputs(self, y, X):
        if X is not None:
            X = self._aggregate_X(X, y)
        return y, X

    def _prepare_forecast_inputs(self, y, X):
        if X is not None:
            y_ref = y if y is not None else self.y
            X = self._aggregate_X(X, y_ref)
        return y, X
