import copy

import numpy as np
import pandas as pd
import pytest
from forecast_realtime import ForecastModel, RealTimeModel

from forecast_evaluation import SimulationData

from .test_conditioning_realtime import EchoAdditiveModel, _source_data

PATHS = [(1, "baseline"), (2, "baseline"), (1, "shock"), (2, "shock")]


class LastValueModel(ForecastModel):
    def _fit(self, y, X=None, **kwargs):
        self.last_value = y.iloc[-1].to_numpy()
        return self

    def _forecast(self, steps, X=None, y=None, **kwargs):
        return np.tile(self.last_value, (steps, 1))


def _simulation(*, unbalanced=False, conditioning=False):
    ordinary = _source_data(second_vintage=True)
    outturns = []
    forecasts = []
    for index, (draw, scenario) in enumerate(PATHS):
        offset = 1000 * index
        history = ordinary._raw_outturns.copy()
        if unbalanced and index == 3:
            history.loc[
                history["vintage_date"] == pd.Timestamp("2020-03-31"), "vintage_date"
            ] = pd.Timestamp("2020-04-30")
        outturns.append(
            history.assign(draw=draw, scenario=scenario, value=history["value"] + offset)
        )
        paths = ordinary._raw_forecasts.copy()
        forecasts.append(
            paths.assign(draw=draw, scenario=scenario, value=paths["value"] + offset)
        )
    return SimulationData(
        outturns_data=pd.concat(outturns, ignore_index=True),
        forecasts_data=pd.concat(forecasts, ignore_index=True) if conditioning else None,
        compute_levels=False,
        data_check=False,
    )


def _run(data, model, **options):
    defaults = {
        "y_variables": ["target_a"],
        "data_transformation": {"target_a": "levels"},
        "steps": 1,
        "first_forecast_horizon": 0,
        "first_vintage": "2020-03-01",
        "last_vintage": "2020-04-30",
    }
    return RealTimeModel(data, model).forecast(**(defaults | options))


def test_two_vintage_simulation_evaluates_on_original_object():
    simulation = _simulation()
    model = LastValueModel(label="last_value")
    original_panels = simulation.panels
    ordinary = {key: panel.copy() for key, panel in simulation.iter_panels()}

    runner = _run(simulation, model)

    assert runner.data is simulation
    assert not hasattr(model, "last_value")
    for key, panel in simulation.iter_panels():
        assert panel is original_panels[key]
        expected = _run(ordinary[key], copy.deepcopy(model))
        pd.testing.assert_frame_equal(panel.forecasts, expected.data.forecasts)
        assert len(panel._raw_forecasts) == 2
        assert not {"draw", "scenario"}.intersection(panel.forecasts)
    values = simulation.forecasts.query("metric == 'levels'")
    assert values.groupby(["draw", "scenario"])["value"].first().nunique() == 4
    accuracy = simulation.evaluate_accuracy(k=0)
    summary = simulation.aggregate_accuracy(accuracy)
    assert not summary.empty
    assert set(summary["n_draws"]) == {2}
    assert set(summary["scenario"]) == {"baseline", "shock"}


def test_simulation_uses_each_paths_release_calendar_and_active_filter():
    simulation = _simulation(unbalanced=True)
    simulation.filter(custom_filter=lambda frame: frame[frame["scenario"] == "shock"])

    _run(simulation, LastValueModel())

    assert simulation.panels[(1, "baseline")]._raw_forecasts.empty
    assert simulation.panels[(2, "baseline")]._raw_forecasts.empty
    assert len(simulation.panels[(1, "shock")]._raw_forecasts) == 2
    assert len(simulation.panels[(2, "shock")]._raw_forecasts) == 1
    assert set(simulation.forecasts["scenario"]) == {"shock"}


@pytest.mark.parametrize("parallel", [False, True])
def test_conditioning_and_future_regressors_remain_path_local(parallel):
    simulation = _simulation(conditioning=True)
    ordinary = {key: panel.copy() for key, panel in simulation.iter_panels()}
    model = EchoAdditiveModel(
        label="echo",
        formula="target_a ~ driver",
        conditioning={
            "y": {"target_a": {"source": "A", "periods": 1}},
            "X": {"driver": {"source": "B", "periods": 1}},
        },
    )
    options = {
        "last_vintage": "2020-03-31",
        "X_variables": ["driver"],
        "data_transformation": {"target_a": "levels", "driver": "levels"},
        "parallel": parallel,
        "max_workers": 2,
        "decomp": not parallel,
    }

    runner = _run(simulation, model, **options)

    for key, panel in simulation.iter_panels():
        expected = _run(ordinary[key], copy.deepcopy(model), **options)
        pd.testing.assert_frame_equal(panel.forecasts, expected.data.forecasts)
    actual = simulation.forecasts.query("source == 'echo' and metric == 'levels'")
    np.testing.assert_allclose(actual["value"], [102, 2102, 4102, 6102])
    if not parallel:
        assert runner.decompositions is not None
        assert set(
            runner.decompositions[["draw", "scenario"]].itertuples(index=False, name=None)
        ) == set(PATHS)


def test_native_forecasts_retain_simulation_ids():
    simulation = _simulation()

    runner = _run(
        simulation,
        LastValueModel(),
        data_transformation={"target_a": "diff"},
        reconstruct_levels=False,
    )

    assert runner.native_forecasts is not None
    assert set(
        runner.native_forecasts[["draw", "scenario"]].itertuples(index=False, name=None)
    ) == set(PATHS)
    assert set(runner.native_forecasts["source"]) == {"LastValueModel"}


def test_missing_conditioning_source_fails_with_path_key():
    simulation = _simulation(conditioning=True)
    panel = simulation.panels[(2, "shock")]
    panel._raw_forecasts = panel._raw_forecasts.query("source != 'B'")
    model = EchoAdditiveModel(
        formula="target_a ~ driver",
        conditioning={"X": {"driver": {"source": "B", "periods": 1}}},
    )

    with pytest.raises(RuntimeError, match="draw.*2.*scenario.*shock.*source"):
        _run(
            simulation,
            model,
            X_variables=["driver"],
            data_transformation={"target_a": "levels", "driver": "levels"},
            last_vintage="2020-03-31",
        )
