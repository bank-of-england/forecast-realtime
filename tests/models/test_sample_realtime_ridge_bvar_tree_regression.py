"""Snapshot of the IJF paper's UK-MD inflation backtest on generated data.

Mirrors ``opera-ijf-paper/application/backtest_ukmd_inflation.py`` without the
R-backed Fable model: keep the model classes in step with that script.
"""

import numpy as np
import pandas as pd
import pytest
from forecast_evaluation import ForecastData

pytest.importorskip("bvar")

import forecast_realtime as rt
from forecast_realtime.models import ForecastBVAR, ForecastOLS, ForecastRidge
from tests.realtime_fixtures import generate_synthetic_data

RANDOM_STATE = 1234
TARGET = "monthly_1"
OLS_X = ["monthly_2", "monthly_3", "monthly_4"]
BVAR_Y = [TARGET] + OLS_X
SOURCES = ["OLS-3", "Ridge", "Ridge-BVAR Tree"]
VINTAGES = ["2024-01-31", "2024-02-29"]
# The paper pins forecast-realtime 0.5.6, whose Ridge penalised AR lags and
# searched a three-point grid.
PAPER_RIDGE = dict(penalise_ar=True, alphas=(0.1, 1.0, 10.0))


class MonthlyConditionalBVAR(ForecastBVAR):
    """BVAR node conditioned on the Ridge nowcast."""

    def _prepare_estimation_inputs(self, y, X):
        self.target_origin_ = X[TARGET].last_valid_index()
        return y, None

    def _fit(self, y, X=None):
        super()._fit(y, X)
        self.last_y_fit_date = self.target_origin_
        return self

    def _forecast(self, steps=1, X=None, y=None, **kwargs):
        bvar_origin = self.y.index[-1]
        target_origin = kwargs["forecast_origin"]
        ridge_forecasts = X.loc[X.index > target_origin, TARGET].dropna()
        target_dates = ridge_forecasts.index[:steps]
        if len(target_dates) != steps:
            raise ValueError(
                f"The Ridge child returned {len(target_dates)} CPI forecasts; "
                f"expected {steps}."
            )

        internal_steps = (target_dates[-1].to_period("M") - bvar_origin.to_period("M")).n
        dates = self._infer_forecast_dates(self.y.index, internal_steps, frequency="M")
        condition = pd.DataFrame(np.nan, index=dates, columns=self.y.columns)
        ridge_path = (
            pd.concat([self._raw_X_history[TARGET], X[TARGET]]).dropna().sort_index()
        )
        ridge_path = ridge_path[~ridge_path.index.duplicated(keep="last")]
        bridge_dates = ridge_path.index.intersection(
            condition.loc[: target_dates[0]].index
        )
        condition.loc[bridge_dates, TARGET] = ridge_path.loc[bridge_dates]
        kwargs["forecast_origin"] = bvar_origin
        forecast = super()._forecast(steps=internal_steps, X=None, y=condition, **kwargs)
        return forecast.reindex(target_dates)


@pytest.fixture(scope="module")
def monthly_panel():
    panel = generate_synthetic_data(N=4, first_period="2015-01-31", endpoint="2024-12-31")
    return panel.loc[panel["frequency"].eq("M") & panel["metric"].eq("levels")]


def _make_models():
    ols_terms = [f"{TARGET}_lag1", *OLS_X, *(f"{v}_lag1" for v in OLS_X)]
    formula = f"{TARGET} ~ {' + '.join(ols_terms)}"
    ridge_bvar_tree = rt.ForecastTree(
        rt.TreeNode(
            name="conditional_bvar",
            children=[
                ForecastRidge(
                    label=TARGET,
                    formula=formula,
                    cv=5,
                    scale=True,
                    forecast_strategy="direct",
                    steps=6,
                    drop_nans=True,
                    **PAPER_RIDGE,
                )
            ],
            transform=MonthlyConditionalBVAR(
                stationary=True,
                n_lags=5,
                mode_only=True,
                nb_restart=0,
                progressbar=False,
                optim_random_state=RANDOM_STATE,
                sampling_random_state=RANDOM_STATE,
                forecast_random_state=RANDOM_STATE,
            ),
        ),
        label="Ridge-BVAR Tree",
    )
    # The paper anchors the tree's forecast origin on target availability.
    ridge_bvar_tree._formula = rt.Formula(f"{TARGET} ~ .")
    return [
        ForecastOLS(
            label="OLS-3",
            formula=formula,
            forecast_strategy="direct",
            steps=6,
            drop_nans=True,
        ),
        ForecastRidge(
            label="Ridge",
            formula=formula,
            cv=5,
            scale=True,
            forecast_strategy="direct",
            steps=6,
            drop_nans=True,
            **PAPER_RIDGE,
        ),
        ridge_bvar_tree,
    ]


@pytest.mark.filterwarnings("error::UserWarning")
def test_ukmd_inflation_backtest_matches_snapshot(
    monthly_panel, inline_executor, bvar_python_kernel, snapshot
):
    realtime = rt.RealTimeModel(
        ForecastData(outturns_data=monthly_panel, compute_levels=False, data_check=False),
        _make_models(),
    )
    realtime.forecast(
        y_variables=BVAR_Y,
        X_variables=OLS_X,
        data_transformation=dict.fromkeys(BVAR_Y, "levels"),
        step_frequency="M",
        steps=6,
        y_lags=1,
        X_lags=1,
        X_imputation="last",
        reconstruct_levels=False,
        parallel=True,
        batch_size=len(VINTAGES),
        max_workers=len(SOURCES),
        first_vintage=VINTAGES[0],
        last_vintage=VINTAGES[-1],
    )

    forecasts = realtime.data.forecasts
    forecasts = forecasts.loc[
        forecasts["variable"].eq(TARGET) & forecasts["metric"].eq("levels")
    ]
    assert set(forecasts["source"]) == set(SOURCES)
    assert (forecasts.groupby("source").size() == 6 * len(VINTAGES)).all()

    columns = ["source", "vintage_date", "date", "forecast_horizon", "value"]
    result = forecasts[columns].sort_values(columns[:3]).reset_index(drop=True)
    for column in ("vintage_date", "date"):
        result[column] = result[column].dt.strftime("%Y-%m-%d")
    result["value"] = result["value"].round(4)
    assert result.to_dict(orient="records") == snapshot
