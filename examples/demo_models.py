"""Demonstration of several forecast models on synthetic real-time data."""

import forecast_evaluation as fe
from sklearn.model_selection import TimeSeriesSplit

import forecast_realtime as rt
from examples.midas_bvar_tree import BVARMIDASTree


def run_demo(
    *,
    N_vintages: int = 30,
    parallel: bool = False,
    batch_size: int | None = None,
    max_workers: int | None = None,
    decomp: bool = False,
    reconstruct_levels: bool = False,
):
    """Run a compact demonstration of the built-in forecasting models.

    Set ``decomp=True`` to collect model-level additive decompositions in
    ``result.decompositions``. Decompositions require sequential execution.
    Forecasts retain their native transformed metrics by default; set
    ``reconstruct_levels=True`` to reconstruct levels where possible.
    """
    # Load the package's synthetic mixed-frequency real-time data.
    # It mimics a real-time dataset so there is ragged-edge in each vintage.
    sample_data = rt.generate_synthetic_data(
        N=3,
        first_period="2015-01-31",
    )

    # keep only 6 vintages for speed
    first_vintages = (
        sample_data["vintage_date"]
        .sort_values()
        .unique()[: min(N_vintages, sample_data["vintage_date"].nunique())]
    )
    sample_data = sample_data[sample_data["vintage_date"].isin(first_vintages)].copy()

    quarterly_variables = sample_data.query("frequency == 'Q'")["variable"].unique()
    monthly_variables = sample_data.query("frequency == 'M'")["variable"].unique()

    quarterly_target = "quarterly_1"
    y_variables = quarterly_variables.tolist()
    quarterly_indicators = quarterly_variables[
        quarterly_variables != quarterly_target
    ].tolist()
    X_variables = [*monthly_variables, *quarterly_variables]

    transformed_variables = [*y_variables, *X_variables]
    data_transformation = {
        variable: "pop" for variable in dict.fromkeys(transformed_variables)
    }

    # Common forecast settings.
    forecast_settings = {
        # Forecast the quarterly series; monthly series are MIDAS inputs.
        "y_variables": y_variables,
        # Make all candidate variables available as right-hand-side variables.
        # Each formula selects the subset supported by its model.
        "X_variables": X_variables,
        # Number of periods (here quarters) to forecast.
        "steps": 2,
        # Ragged-edge missing values are imputed with a naive forecast.
        "X_imputation": "last",
        # Use stationary data for the model inputs.
        "data_transformation": data_transformation,
    }

    big_regression = f"{quarterly_target} ~ {' + '.join(quarterly_indicators)}"
    models = [
        rt.models.ForecastOLS(label="Big OLS", formula=big_regression),
        rt.models.ForecastOLS(
            label="Small OLS",
            formula="quarterly_1 ~ quarterly_2",
        ),
        rt.models.ForecastRidge(
            label="Ridge",
            cv=2,
            scale=True,
            formula=big_regression,
        ),
        rt.models.ForecastLasso(
            label="LASSO",
            cv=2,
            scale=True,
            formula=big_regression,
        ),
        rt.models.ForecastElasticNet(
            label="Elastic Net",
            cv=TimeSeriesSplit(n_splits=2),
            l1_ratio=0.5,
            scale=True,
            formula=big_regression,
        ),
        rt.models.RFableARIMA(
            label="Fable ARIMA",
            p=1,
            d=0,
            q=0,
            xreg="quarterly_2",
            index="quarter",
            formula="quarterly_1 ~ quarterly_2",
        ),
        rt.models.ForecastBridgeOLS(
            label="Bridge OLS",
            formula="quarterly_1 ~ monthly_1 + quarterly_2",
        ),
        rt.models.ForecastMIDAS(
            label="MIDAS",
            method="almon",
            n_lags=6,
            estimator="ols",
            horizons=[0, 1],
            # Horizons must be known in advance because MIDAS uses direct
            # forecasting.
            formula="quarterly_1 ~ monthly_1",
        ),
        rt.models.ForecastBVAR(
            label="BVAR",
            n_lags=1,
            mode_only=True,
            progressbar=False,
            formula="quarterly_1 + quarterly_2 ~ quarterly_1 + quarterly_2",
        ),
        BVARMIDASTree(),
    ]

    # Create a NowcastData object and run every model.
    forecast_data = fe.NowcastData(outturns_data=sample_data)
    rt_model = rt.RealTimeModel(data=forecast_data, models=models)
    rt_model.forecast(
        **forecast_settings,
        batch_size=batch_size,
        max_workers=max_workers,
        decomp=decomp,
        reconstruct_levels=reconstruct_levels,
    )
    return rt_model


if __name__ == "__main__":
    from news_decomp import NewsData

    demo = run_demo(
        N_vintages=6,
        decomp=True,
        reconstruct_levels=False,
    )

    # launch forecast dashboard in a browser
    demo.data.run_dashboard()

    # news decomposition analysis
    news_data = NewsData(demo.decompositions)
    news_data.report(
        variable="quarterly_1",
        source="Ridge",
    )
