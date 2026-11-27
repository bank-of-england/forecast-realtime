"""Backtest Timestral, t0, OLS and AR(p) on the package's synthetic data.

Run with ``python -m forecast_realtime.examples.foundation_models``.
Install Timestral and ``forecast-realtime[t0]`` first. The first run downloads
pretrained weights unless local checkpoint directories are supplied.

Monthly targets and regressors come from ``generate_synthetic_data``.
Future monthly_2 values are assumed known to models using regressors, while
future monthly_1 targets are withheld. Each origin adds one month of history.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

import forecast_realtime as rt


@dataclass
class BacktestResult:
    """Forecasts, horizon-specific accuracy and per-model execution timings."""

    forecasts: pd.DataFrame
    accuracy: pd.DataFrame
    timings: pd.DataFrame
    elapsed_seconds: float


def run_demo(
    *,
    n_train: int = 120,
    n_origins: int = 20,
    steps: int = 3,
    ar_order: int = 1,
    seed: int = 42,
    device: str = "cpu",
    timestral_model_id: str = "mistralai/Timestral-209M",
    t0_model_id: str = "theforecastingcompany/t0-gamma",
):
    """Compare point forecasts on identical expanding-window test samples.

    OLS uses monthly_2 and one target lag. AR(p) uses only target history.
    Neither baseline selects its order on the test data. Foundation models
    keep their pretrained weights fixed. Publication lags are disabled.
    Timings include refitting at each origin; the first fit also includes
    weight loading when the checkpoint is not already loaded in this process.
    """
    for name, value in (
        ("n_train", n_train),
        ("n_origins", n_origins),
        ("steps", steps),
        ("ar_order", ar_order),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if n_train <= max(ar_order, 1) + 3:
        raise ValueError("n_train must exceed max(ar_order, 1) + 3.")

    started = perf_counter()
    sample_data = rt.generate_synthetic_data(
        N=2,
        seed=seed,
        first_period="2010-01-31",
        endpoint="2025-12-31",
        publication_lags=False,
    )
    levels = sample_data.loc[
        sample_data["vintage_date"].eq(sample_data["vintage_date"].max())
        & sample_data["metric"].eq("levels")
        & sample_data["variable"].isin(["monthly_1", "monthly_2"])
    ]
    monthly = levels.pivot(index="date", columns="variable", values="value")
    monthly = monthly.sort_index().asfreq("ME")
    required = n_train + n_origins + steps - 1
    if required > len(monthly):
        raise ValueError(
            f"Backtest requires {required} monthly observations; "
            f"the simulated data provides {len(monthly)}."
        )
    monthly = monthly.iloc[-required:]
    targets = monthly[["monthly_1"]]
    regressors = monthly[["monthly_2"]]
    ar_label = f"AR({ar_order})"
    models = [
        (
            rt.models.ForecastTimestral(
                label="Timestral", model_id=timestral_model_id, device=device
            ),
            0,
            True,
        ),
        (
            rt.models.ForecastT0(label="t0", model_id=t0_model_id, device=device),
            0,
            True,
        ),
        (rt.models.ForecastOLS(label="OLS"), 1, True),
        (rt.models.ForecastOLS(label=ar_label), ar_order, False),
    ]
    forecasts = []
    timings = []
    for template, y_lags, with_regressors in models:
        fit_times = []
        forecast_times = []
        for cutoff in range(n_train, n_train + n_origins):
            model = deepcopy(template)
            history = targets.iloc[:cutoff]
            future = targets.iloc[cutoff : cutoff + steps]
            fit_started = perf_counter()
            model.fit(
                history,
                X=regressors.iloc[:cutoff] if with_regressors else None,
                y_lags=y_lags,
            )
            fit_times.append(perf_counter() - fit_started)
            forecast_started = perf_counter()
            prediction = model.forecast(
                steps=steps,
                X=(regressors.iloc[cutoff : cutoff + steps] if with_regressors else None),
            )
            forecast_times.append(perf_counter() - forecast_started)
            if not pd.DatetimeIndex(prediction["date"]).equals(future.index):
                raise ValueError(
                    f"{model.label} returned an unexpected forecast calendar."
                )
            values = prediction["value"].to_numpy()
            if not np.isfinite(values).all():
                raise ValueError(f"{model.label} returned non-finite forecasts.")
            forecasts.append(
                pd.DataFrame(
                    {
                        "source": model.label,
                        "origin": history.index[-1],
                        "date": future.index,
                        "horizon": np.arange(1, steps + 1),
                        "actual": future["monthly_1"].to_numpy(),
                        "value": values,
                    }
                )
            )
        timings.append(
            {
                "source": template.label,
                "n_origins": n_origins,
                "fit_seconds": sum(fit_times),
                "first_forecast_seconds": forecast_times[0],
                "warm_forecast_seconds": (
                    float(np.median(forecast_times[1:])) if n_origins > 1 else np.nan
                ),
                "total_seconds": sum(fit_times) + sum(forecast_times),
            }
        )

    forecasts = pd.concat(forecasts, ignore_index=True)
    forecasts["error"] = forecasts["actual"] - forecasts["value"]
    accuracy = forecasts.groupby(["source", "horizon"], as_index=False).agg(
        n_forecasts=("error", "size"),
        MAE=("error", lambda errors: errors.abs().mean()),
        MSE=("error", lambda errors: errors.pow(2).mean()),
    )
    accuracy["RMSE"] = np.sqrt(accuracy.pop("MSE"))
    baseline = accuracy.loc[accuracy["source"].eq(ar_label)].set_index("horizon")["RMSE"]
    accuracy["relative_RMSE"] = accuracy["RMSE"] / accuracy["horizon"].map(
        baseline.where(baseline.ne(0))
    )
    return BacktestResult(
        forecasts=forecasts,
        accuracy=accuracy,
        timings=pd.DataFrame(timings),
        elapsed_seconds=perf_counter() - started,
    )


def main():
    """Print backtest accuracy relative to AR(p), followed by runtime timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-train", type=int, default=120)
    parser.add_argument("--n-origins", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--ar-order", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timestral-model-id", default="mistralai/Timestral-209M")
    parser.add_argument("--t0-model-id", default="theforecastingcompany/t0-gamma")
    demo = run_demo(**vars(parser.parse_args()))
    print("Expanding-window backtest; future monthly_2 values are assumed known.")
    print("Accuracy by horizon (relative_RMSE < 1 beats the AR benchmark):")
    print(demo.accuracy.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nTimings in seconds (first fit includes uncached weight loading):")
    print(demo.timings.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nTotal elapsed: {demo.elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
