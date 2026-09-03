"""Runnable demo of :class:`MATLABModel` driving ``forecast_lm.m``.

Fits the toy OLS trend regression in ``forecast_lm.m`` over a handful of
real-time vintages and forecasts 12 steps ahead.

Requires ``matlab`` on ``PATH``. Every ``fit`` and ``forecast`` call launches a
fresh ``matlab -batch`` process, so runtime is dominated by MATLAB cold start
(roughly a minute per vintage) rather than by the estimation itself. That is
why ``n_vintages`` defaults to 3 — raise it only if you are happy to wait.

Run it with::

    python tests/models/matlab_scripts/demo_forecast_lm_matlab.py
"""

from pathlib import Path

import forecast_evaluation as fe

import forecast_realtime as rt
from forecast_realtime import MATLABModel

# Resolve the .m file relative to this file so the demo works regardless of the
# shell's working directory. MATLABModel resolves the script path against the
# process's current working directory at construction time, not against the
# location of the calling script.
_MATLAB_SCRIPT = str(Path(__file__).parent / "forecast_lm.m")


def run_demo(*, n_vintages: int = 3, steps: int = 12) -> rt.RealTimeModel:
    """Run the MATLAB OLS model over the most recent vintages.

    Parameters
    ----------
    n_vintages : int
        Number of vintages to back-test, counted back from the latest vintage
        in the data. Each vintage costs one MATLAB fit plus one MATLAB
        forecast, so keep this small. Default 3.
    steps : int
        Number of periods to forecast at each vintage. Default 12.

    Returns
    -------
    rt.RealTimeModel
        The fitted real-time model, with results on ``.data``.
    """
    forecast_data = fe.ForecastData(load_fer=True)

    # Take the last n_vintages vintage dates rather than hardcoding a window,
    # so the demo keeps working as the data moves forward.
    vintage_dates = forecast_data.outturns["vintage_date"].sort_values().unique()
    selected = vintage_dates[-min(n_vintages, len(vintage_dates)) :]

    model = MATLABModel(_MATLAB_SCRIPT, label="OLS MATLAB")
    rt_model = rt.RealTimeModel(data=forecast_data, models=model)
    rt_model.forecast(
        y_variables=["cpisa"],
        X_variables=["gdpkp"],
        data_transformation={"cpisa": "pop", "gdpkp": "pop"},
        steps=steps,
        first_vintage=str(selected[0]),
        last_vintage=str(selected[-1]),
    )
    return rt_model


if __name__ == "__main__":
    demo = run_demo()
    print(demo.data.forecasts)

    # demo.data.run_dashboard()
