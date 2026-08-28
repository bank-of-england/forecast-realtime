"""ForecastModel wrapper for ``nowcast_midas.MidasCombo``."""

import nowcast_midas as nm
import numpy as np
import pandas as pd
from nowcast_midas.specs import ComboSpec

from forecast_realtime._utils import validate_forecast_horizons
from forecast_realtime.forecast_model import ForecastModel


class ForecastMIDASCombo(ForecastModel):
    """MIDAS forecast-combination wrapper for forecast_realtime.

    Parameters
    ----------
    combo_specs : ComboSpec
        Root combination node of the SC-MIDAS tree. May contain nested
        ``ComboSpec`` / ``MidasSpec`` / ``OLSSpec`` / ``MultiMidasSpec``
        instances. All specs are auto-derived from the tree; no separate
        ``midas_specs`` or ``ols_specs`` parameters. The root node's
        ``name`` is used to extract forecasts from the output.
    horizons : int
        Number of forecast horizons to fit (default 3). Set dynamically
        if ``RealTimeModel`` passes ``steps`` kwarg at fit time.
    regressor_frequencies : dict[str, str] | None
        Explicit frequency mapping for each regressor column:
        ``{'var_name': 'ME'}`` for monthly, ``'QE'`` for quarterly.
        If omitted, frequencies are taken from the shared input-frequency map
        resolved by ``ForecastModel``.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target and regressors.
    aggregate_decomp : bool | None
        Whether to aggregate decomposition components. Default is ``False``.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    """

    _handles_mixed_frequencies = True

    # MIDASCombo infers its own info-date/horizon from ragged-edge regressor data.
    _needs_ragged_edge_imputation = False

    def __init__(
        self,
        combo_specs: ComboSpec,
        horizons: int = 3,
        regressor_frequencies: dict[str, str] | None = None,
        label: str | None = None,
        formula: str | None = None,
        aggregate_decomp: bool | None = False,
        data_transformation: dict[str, str] | None = None,
    ) -> None:

        super().__init__(
            label=label,
            formula=formula,
            data_transformation=data_transformation,
        )
        if not isinstance(combo_specs, ComboSpec):
            raise TypeError("combo_specs must be a ComboSpec instance")

        self.combo_specs = combo_specs
        self.horizons = horizons
        self.regressor_frequencies = regressor_frequencies
        self.aggregate_decomp = aggregate_decomp

        self.model: nm.MidasCombo | None = None
        self._target_long: pd.DataFrame | None = None
        self._regressors_long: pd.DataFrame | None = None
        self._inferred_frequencies: dict | None = None

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_nowcast_midas_frequency(frequency: str) -> str:
        aliases = {
            "M": "ME",
            "ME": "ME",
            "MS": "ME",
            "Q": "QE",
            "QE": "QE",
            "QS": "QE",
        }
        try:
            return aliases[frequency.upper()]
        except (AttributeError, KeyError) as error:
            raise ValueError(
                f"Unsupported MIDAS regressor frequency {frequency!r}; expected "
                "'M'/'ME' or 'Q'/'QE'."
            ) from error

    def _y_to_long(self, y: pd.DataFrame) -> pd.DataFrame:
        """Convert wide quarterly target DataFrame to long format."""
        if y.shape[1] != 1:
            raise ValueError("ForecastMIDASCombo expects y with a single column.")
        var = y.columns[0]
        return pd.DataFrame(
            {
                "date": y.index,
                "variable": var,
                "frequency": "QE",
                "value": y.iloc[:, 0].to_numpy(),
            }
        )

    def _X_to_long(self, X: pd.DataFrame) -> pd.DataFrame:
        """Convert wide regressor DataFrame to long format."""
        explicit_freqs = dict(self.regressor_frequencies or {})
        shared_freqs = getattr(self, "_input_frequencies", {})
        rows = []
        for col in X.columns:
            series = X[col].dropna()
            if col in explicit_freqs:
                frequency = explicit_freqs[col]
            elif col in shared_freqs:
                frequency = shared_freqs[col]
            else:
                raise ValueError(
                    f"No frequency resolved for MIDAS regressor column {col!r}; "
                    "provide regressor_frequencies or fit with input_frequencies."
                )
            freq = self._to_nowcast_midas_frequency(frequency)
            rows.append(
                pd.DataFrame(
                    {
                        "date": series.index,
                        "variable": col,
                        "frequency": freq,
                        "value": series.to_numpy(),
                    }
                )
            )
        return pd.concat(rows, ignore_index=True)

    # ------------------------------------------------------------------ #
    # ForecastModel interface                                            #
    # ------------------------------------------------------------------ #

    def _fit(
        self,
        y: pd.DataFrame,
        X: pd.DataFrame | None = None,
        **kwargs,
    ):
        if X is None:
            raise ValueError(
                "MidasCombo requires regressors (X). Pass mixed-frequency "
                "indicator data as X to fit()."
            )

        incoming_index = y.index
        target_name = y.columns[0]

        self._target_long = self._y_to_long(y)
        self._regressors_long = self._X_to_long(X)

        # ``steps`` from RealTimeModel determines horizons fitted
        steps = kwargs.pop("steps", None)
        horizons = int(steps) if steps is not None else int(self.horizons)
        self.horizons = horizons

        self.model = nm.MidasCombo(
            combo_specs=self.combo_specs,
            horizons=horizons,
        )
        self.model.fit(target=self._target_long, regressors=self._regressors_long)

        root_name = self.combo_specs.name
        root_fits = self.model.fitted_.get(root_name, {})
        if 0 in root_fits:
            self.fitted_values_ = root_fits[0].reindex(incoming_index)
            self.fitted_values_.name = target_name
        else:
            self.fitted_values_ = pd.Series(
                np.nan, index=incoming_index, name=target_name
            )

        return self

    def _forecast(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Refit with more horizons if more steps requested
        if steps > self.horizons:
            self.horizons = steps
            self.model = nm.MidasCombo(
                combo_specs=self.combo_specs,
                horizons=steps,
            )
            self.model.fit(target=self._target_long, regressors=self._regressors_long)

        # forecast() returns long format: [date, horizon, value, spec]
        df_long = self.model.forecast()

        root_name = self.combo_specs.name
        root_forecasts = df_long[df_long["spec"].astype(str) == root_name]
        validate_forecast_horizons(
            root_forecasts["horizon"], steps, self.__class__.__name__
        )
        forecasts = np.full((steps, 1), np.nan)
        dates: list = [pd.NaT] * steps

        for _, row in df_long.iterrows():
            horizon = int(row["horizon"])
            spec = str(row["spec"])
            if horizon >= steps:
                continue

            if spec == root_name:
                forecasts[horizon, 0] = row["value"]
                dates[horizon] = pd.Timestamp(row["date"])

        return pd.DataFrame(
            forecasts,
            index=pd.DatetimeIndex(dates, name="date"),
            columns=self.y.columns,
        )

    def _forecast_decomp(
        self,
        steps: int = 1,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> pd.DataFrame | None:
        """Return the MIDAS-combination decomposition in the model contract."""
        regressors_long = self._X_to_long(X) if X is not None else None
        raw = self.model.forecast_decomp(
            regressors=regressors_long, aggregate=self.aggregate_decomp
        )

        if raw is None or raw.empty:
            return None

        # Filter to horizons <= steps and extract columns matching the minimal contract
        raw = raw[raw["horizon"] < steps].copy()
        if raw.empty:
            return None

        raw = raw.rename(columns={"horizon": "forecast_horizon"})
        # Select only the required columns from the decomposition
        required_cols = ["forecast_horizon", "component", "contribution", "weight"]
        available_cols = [col for col in required_cols if col in raw.columns]
        return raw[available_cols]
