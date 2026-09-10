"""Demonstrate the R-backed Fable ARIMA model.

Install ``forecast-realtime[models]`` and the R packages ``arrow``, ``fable``,
``fabletools`` and ``tsibble`` before running this example.
"""

import forecast_evaluation as fe

import forecast_realtime as rt


def run_demo(*, N_vintages: int = 6):
    """Run a small Fable ARIMA forecast on synthetic real-time data."""
    sample_data = rt.generate_synthetic_data(
        N=3,
        first_period="2015-01-31",
    )
    first_vintages = (
        sample_data["vintage_date"]
        .sort_values()
        .unique()[: min(N_vintages, sample_data["vintage_date"].nunique())]
    )
    sample_data = sample_data[sample_data["vintage_date"].isin(first_vintages)].copy()

    forecast_data = fe.NowcastData(outturns_data=sample_data)
    model = rt.models.RFableARIMA(
        label="Fable ARIMA",
        p=1,
        d=0,
        q=0,
        xreg="quarterly_2",
        index="quarter",
        formula="quarterly_1 ~ quarterly_2",
    )
    rt_model = rt.RealTimeModel(data=forecast_data, models=[model])
    rt_model.forecast(
        y_variables=["quarterly_1"],
        X_variables=["quarterly_2"],
        steps=2,
        X_imputation="last",
        data_transformation={"quarterly_1": "pop", "quarterly_2": "pop"},
    )
    return rt_model


if __name__ == "__main__":
    demo = run_demo()
    print(demo.data.forecasts.head())
