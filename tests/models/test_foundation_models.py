"""Foundation model contracts, without model downloads.

Set FORECAST_REALTIME_PRETRAINED=1 to test cached full checkpoints.
Run those tests with ``pytest -n auto --dist loadgroup -k pretrained``.
"""

import gc
import os
import sys
from functools import lru_cache
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import forecast_realtime as rt
from forecast_realtime.models.t0 import ForecastT0
from forecast_realtime.models.timestral import ForecastTimestral


@pytest.fixture(params=[ForecastTimestral, ForecastT0])
def local_model(request, monkeypatch):
    calls = []

    def predict(context, **kwargs):
        calls.append((context.copy(), kwargs))
        steps = kwargs.get("prediction_length", kwargs.get("horizon"))
        levels = np.array([0.1, 0.5, 0.9])
        point = np.tile(np.arange(context.shape[1])[None, :, None], (1, 1, steps))
        return SimpleNamespace(
            median=point,
            quantiles=point[..., None] + levels,
            quantile_levels=levels,
        )

    backend = SimpleNamespace(forecast=predict, predict=predict)
    module = request.param.__module__
    monkeypatch.setattr(f"{module}._load_forecaster", lambda *args: backend)
    return request.param(), calls


@pytest.fixture
def targets():
    dates = pd.date_range("2020-01-31", periods=8, freq="ME")
    return pd.DataFrame(
        {"first": np.arange(8.0), "second": np.arange(8.0) + 10}, index=dates
    )


def test_local_loads_backend_during_fit(local_model, targets, monkeypatch):
    model, calls = local_model
    module = sys.modules[type(model).__module__]
    loader = Mock(wraps=module._load_forecaster)
    monkeypatch.setattr(module, "_load_forecaster", lru_cache(maxsize=4)(loader))

    model = type(model)()
    loader.assert_not_called()
    model.fit(targets)
    loader.assert_called_once_with(model.model_id, model.device)
    assert not calls

    model.fit(targets)
    model.forecast(steps=2)
    loader.assert_called_once_with(model.model_id, model.device)
    assert len(calls) == 1


@pytest.mark.parametrize("model_class", [ForecastTimestral, ForecastT0])
@pytest.mark.parametrize("error_class", [ModuleNotFoundError, MemoryError])
def test_local_loading_errors_surface_during_fit(
    model_class, error_class, targets, monkeypatch
):
    loader = Mock(side_effect=error_class("Cannot load model weights"))
    monkeypatch.setattr(f"{model_class.__module__}._load_forecaster", loader)
    model = model_class()
    loader.assert_not_called()

    with pytest.raises(error_class, match="Cannot load model weights"):
        model.fit(targets)
    assert not model._is_fitted


@pytest.mark.parametrize("strategy", ["recursive", "direct"])
@pytest.mark.parametrize("entry_point", ["constructor", "fit", "forecast"])
def test_local_rejects_linear_forecast_strategy(
    local_model, targets, strategy, entry_point
):
    model, calls = local_model
    if entry_point == "forecast":
        model.fit(targets)

    with pytest.raises(TypeError, match="forecast_strategy"):
        if entry_point == "constructor":
            type(model)(forecast_strategy=strategy)
        elif entry_point == "fit":
            model.fit(targets, forecast_strategy=strategy)
        else:
            model.forecast(steps=2, forecast_strategy=strategy)
    assert not calls


@pytest.mark.parametrize("steps", [12, 1025])
def test_local_delegates_full_horizon_to_backend(local_model, targets, steps):
    model, calls = local_model
    model.fit(targets)
    result = model.forecast(steps=steps)

    assert len(calls) == 1
    _, options = calls[0]
    assert options.get("prediction_length", options.get("horizon")) == steps
    assert result["date"].nunique() == steps


def test_local_multivariate_point_forecast(local_model, targets):
    model, calls = local_model
    assert model.fit(targets) is model
    assert not calls
    result = model.forecast(steps=3)
    context, _ = calls[0]
    np.testing.assert_array_equal(context, targets.to_numpy().T[None, ...])
    assert result["variable"].tolist() == ["first", "second"] * 3
    assert result["value"].tolist() == [0.0, 1.0] * 3
    assert result["date"].drop_duplicates().tolist() == list(
        pd.date_range("2020-09-30", periods=3, freq="ME")
    )


def test_local_quantiles_keep_variable_and_date_order(local_model, targets):
    model, _ = local_model
    model.fit(targets)
    result = model.forecast(steps=3, quantiles=[0.1, 0.9])
    assert result["value"].tolist() == [0.1, 0.9, 1.1, 1.9] * 3
    assert result["variable"].tolist() == ["first", "first", "second", "second"] * 3


def test_local_covariates_cover_history_and_future(local_model, targets):
    model, calls = local_model
    regressors = pd.DataFrame({"regressor": np.arange(8.0)}, index=targets.index)
    future = pd.DataFrame(
        {"regressor": [8.0, 9.0]}, index=pd.date_range("2020-09-30", periods=2, freq="ME")
    )
    model.fit(targets, regressors)
    model.forecast(steps=2, X=future)
    _, options = calls[0]
    covariates = options.get("known_covariates", options.get("future_covariates"))
    np.testing.assert_array_equal(covariates, np.arange(10.0)[None, None, :])


def test_local_missing_dates_preserve_calendar_spacing(local_model, targets):
    model, calls = local_model
    model.fit(targets.drop(targets.index[3]))
    model.forecast(steps=2)
    assert calls[0][0].shape == (1, 2, 8)
    assert np.isnan(calls[0][0][0, :, 3]).all()


def test_local_rejects_unavailable_quantiles(local_model, targets):
    model, _ = local_model
    model.fit(targets)
    with pytest.raises(ValueError, match="does not provide quantile"):
        model.forecast(steps=2, quantiles=[0.2, 0.8])


def test_timestral_loader_passes_device_and_reuses_weights(monkeypatch):
    from forecast_realtime.models.timestral import _load_forecaster

    loader = Mock(return_value=object())
    monkeypatch.setitem(
        sys.modules,
        "timestral",
        SimpleNamespace(TimestralForecaster=SimpleNamespace(from_pretrained=loader)),
    )
    _load_forecaster.cache_clear()
    try:
        first = _load_forecaster("test-checkpoint", "cpu")
        assert _load_forecaster("test-checkpoint", "cpu") is first
        loader.assert_called_once_with("test-checkpoint", device="cpu")
    finally:
        _load_forecaster.cache_clear()


@pytest.mark.parametrize("model_class", [ForecastTimestral, ForecastT0])
def test_foundation_models_are_public_and_lightweight(model_class):
    assert model_class.__bases__ == (rt.ForecastModel,)
    assert getattr(rt.models, model_class.__name__) is model_class
    assert model_class.__name__ in rt.models.__all__
    model = model_class(formula="first ~ second", label="foundation")
    assert model.label == "foundation"


@pytest.mark.parametrize(
    "frequency", ["D", "h", "2h", "B", "W-MON", "QE", "QS", "ME", "MS", "YE", "YS", "BME"]
)
@pytest.mark.parametrize("quantiles", [False, [0.1, 0.9]])
def test_local_calendars(local_model, frequency, quantiles):
    model, calls = local_model
    dates = pd.date_range("2020-01-01", periods=8, freq=frequency)
    targets = pd.DataFrame({"target": np.arange(8.0)}, index=dates)
    model.fit(targets)
    result = model.forecast(steps=2, quantiles=quantiles)
    expected = pd.date_range(dates[-1], periods=3, freq=frequency)[1:]
    assert result["date"].drop_duplicates().tolist() == expected.tolist()
    np.testing.assert_array_equal(calls[0][0], targets.to_numpy().T[None, ...])


@pytest.mark.parametrize("with_regressors", [False, True])
def test_local_rejects_mixed_target_frequencies(local_model, with_regressors):
    model, calls = local_model
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    targets = pd.DataFrame({"monthly": np.arange(12.0)}, index=dates)
    targets["quarterly"] = targets["monthly"].where(dates.is_quarter_end)
    regressors = targets[["monthly"]].rename(columns={"monthly": "regressor"})
    with pytest.raises(ValueError, match="does not support mixed frequencies"):
        model.fit(
            targets,
            X=regressors if with_regressors else None,
            input_frequencies={"monthly": "M", "quarterly": "Q", "regressor": "M"},
        )
    assert not calls
    assert not model._is_fitted


def test_local_period_index(local_model):
    model, calls = local_model
    dates = pd.period_range("2020Q1", periods=8, freq="Q")
    targets = pd.DataFrame({"target": np.arange(8.0)}, index=dates)
    model.fit(targets)
    result = model.forecast(steps=2, quantiles=[0.1, 0.9])
    assert result["date"].drop_duplicates().tolist() == [
        pd.Timestamp("2022-03-31"),
        pd.Timestamp("2022-06-30"),
    ]
    assert calls[0][0].shape == (1, 1, 8)


def test_local_context_uses_published_targets(local_model, targets):
    model, calls = local_model
    model.fit(targets.iloc[:6])
    context = rt.ForecastContext(
        y_history=targets.iloc[:6],
        X_history=None,
        y_published=targets.iloc[6:],
        forecast_origin=targets.index[-1],
    )
    model.forecast(steps=2, context=context)
    np.testing.assert_array_equal(calls[0][0], targets.to_numpy().T[None, ...])


@pytest.fixture(params=[ForecastTimestral, ForecastT0])
def tiny_local_model(request, tmp_path):
    torch = pytest.importorskip("torch")
    loader = sys.modules[request.param.__module__]._load_forecaster
    loader.cache_clear()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            if request.param is ForecastTimestral:
                timestral = pytest.importorskip("timestral")
                from safetensors.torch import save_file
                from timestral.config import TimestralConfig
                from timestral.model import TimestralModel

                config = TimestralConfig(
                    patch_size=4,
                    d_model=16,
                    num_heads=2,
                    mlp_hidden_dim=32,
                    block_schedule=("time_variate", "time_variate"),
                    quantile_knots=(0.1, 0.5, 0.9),
                    block_size=2,
                    context_length=16,
                )
                backend = timestral.TimestralForecaster(TimestralModel(config).eval())
                config.save(tmp_path / "config.json")
                save_file(backend.model.state_dict(), tmp_path / "model.safetensors")
            else:
                t0 = pytest.importorskip("t0")
                from t0.config import T0Config

                config = T0Config(
                    embed_dim=16,
                    num_layers=2,
                    num_heads=2,
                    mlp_hidden_dim=32,
                    patch_size=4,
                    group_every_n=1,
                    dropout=0.0,
                    quantile_levels=(0.1, 0.5, 0.9),
                )
                backend = t0.T0Forecaster.from_config(config).to("cpu").eval()
                backend.max_horizon = 8
                backend.save_pretrained(tmp_path)
            loaded = loader(str(tmp_path), "cpu")
            if request.param is ForecastT0:
                loaded.max_horizon = 8
            assert loader(str(tmp_path), "cpu") is loaded
        yield request.param, backend, str(tmp_path)
    finally:
        loader.cache_clear()
        torch.set_num_threads(previous_threads)


@pytest.mark.filterwarnings("error:The given NumPy array is not writable:UserWarning")
def test_local_real_backend_inference(tiny_local_model, targets):
    model_class, backend, checkpoint = tiny_local_model
    steps = 9
    dates = pd.date_range(targets.index[-1], periods=steps + 1, freq="ME")[1:]
    history = targets.copy()
    history.iloc[2, 0] = np.nan
    history = history.drop(history.index[3])
    context = history.reindex(targets.index).to_numpy(copy=True).T[None, ...]
    regressors = pd.DataFrame(
        {"regressor": np.arange(len(targets) + steps, dtype=float)},
        index=targets.index.append(dates),
    )
    for with_regressors in (False, True):
        model = model_class(model_id=checkpoint)
        model.fit(history, regressors.loc[targets.index] if with_regressors else None)
        covariates = (
            regressors.to_numpy(copy=True).T[None, ...] if with_regressors else None
        )
        for quantiles in (False, [0.1, 0.5, 0.9]):
            result = model.forecast(
                steps=steps,
                X=regressors.loc[dates] if with_regressors else None,
                quantiles=quantiles,
            )
            if model_class is ForecastTimestral:
                prediction = backend.forecast(
                    context, prediction_length=steps, known_covariates=covariates
                )
            else:
                prediction = backend.predict(
                    context,
                    horizon=steps,
                    quantile_levels=(0.5,) if quantiles is False else tuple(quantiles),
                    future_covariates=covariates,
                )
            values = prediction.median if quantiles is False else prediction.quantiles
            expected = values.detach().cpu().numpy()[0].swapaxes(0, 1)
            np.testing.assert_allclose(result["value"], expected.reshape(-1))
            assert np.isfinite(result["value"]).all()
            assert result["date"].drop_duplicates().tolist() == dates.tolist()
            if quantiles is not False:
                ordered = result["value"].to_numpy().reshape(steps, 2, 3)
                assert (np.diff(ordered, axis=-1) >= 0).all()


@pytest.fixture(scope="module", params=[ForecastTimestral, ForecastT0])
def pretrained_local_model(request):
    if os.environ.get("FORECAST_REALTIME_PRETRAINED") != "1":
        pytest.skip("Set FORECAST_REALTIME_PRETRAINED=1 to test cached full checkpoints.")
    if hasattr(request.config, "workerinput"):
        arguments = request.config.workerinput.get("mainargv", [])
        if not any(
            argument in ("loadgroup", "--dist=loadgroup") for argument in arguments
        ):
            pytest.fail("Run pretrained tests with --dist loadgroup to limit memory use.")

    import torch
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    checkpoints = {
        ForecastTimestral: (
            "mistralai/Timestral-209M",
            "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
            209_312_256,
        ),
        ForecastT0: (
            "theforecastingcompany/t0-gamma",
            "e08aee083778f1f32da4974c0a34276776a35c52",
            255_614_272,
        ),
    }
    model_id, revision, parameter_count = checkpoints[request.param]
    try:
        checkpoint = snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=["config.json", "model.safetensors"],
            local_files_only=True,
        )
    except LocalEntryNotFoundError:
        pytest.fail(f"Cache the full checkpoint {model_id} at revision {revision} first.")

    loader = sys.modules[request.param.__module__]._load_forecaster
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 4))
    loader.cache_clear()
    try:
        with torch.random.fork_rng(devices=[]):
            backend = loader(checkpoint, "cpu")
        network = backend.model if request.param is ForecastTimestral else backend
        assert (
            sum(parameter.numel() for parameter in network.parameters())
            == parameter_count
        )
        yield request.param, backend, checkpoint
    finally:
        loader.cache_clear()
        torch.set_num_threads(previous_threads)
        gc.collect()


@pytest.fixture
def pretrained_data():
    dates = pd.date_range("2010-01-31", periods=64, freq="ME")
    positions = np.arange(len(dates), dtype=float)
    targets = pd.DataFrame(
        {
            "first": 2 + np.sin(positions / 3),
            "second": -1 + np.cos(positions / 5),
        },
        index=dates,
    )
    return targets


def _assert_pretrained_parity(
    model_class, backend, checkpoint, history, regressors, steps
):
    dates = pd.date_range(history.index[-1], periods=steps + 1, freq="ME")[1:]
    calendar = pd.date_range(history.index[0], history.index[-1], freq="ME")
    context = history.reindex(calendar).to_numpy(copy=True).T[None, ...]
    covariates = (
        None if regressors is None else regressors.to_numpy(copy=True).T[None, ...]
    )
    model = model_class(model_id=checkpoint)
    model.fit(history, X=None if regressors is None else regressors.loc[calendar])
    for probabilities in (False, [0.1, 0.5, 0.9]):
        result = model.forecast(
            steps=steps,
            X=None if regressors is None else regressors.loc[dates],
            quantiles=probabilities,
        )
        if model_class is ForecastTimestral:
            native = backend.forecast(
                context, prediction_length=steps, known_covariates=covariates
            )
        else:
            native = backend.predict(
                context,
                horizon=steps,
                quantile_levels=(0.5,) if probabilities is False else probabilities,
                future_covariates=covariates,
            )
        if probabilities is False:
            expected = native.median.detach().cpu().numpy()[0].T
            assert result["variable"].tolist() == list(history.columns) * steps
        else:
            levels = np.asarray(native.quantile_levels)
            selected = [
                np.flatnonzero(np.isclose(levels, probability))[0]
                for probability in probabilities
            ]
            expected = native.quantiles.detach().cpu().numpy()[0]
            expected = np.take(expected, selected, axis=-1).swapaxes(0, 1)
            assert result["variable"].tolist() == list(
                np.tile(np.repeat(history.columns, len(probabilities)), steps)
            )
            assert result["quantile"].tolist() == probabilities * (
                steps * len(history.columns)
            )
            assert (np.diff(expected, axis=-1) >= 0).all()
        np.testing.assert_allclose(
            result["value"], expected.reshape(-1), rtol=1e-5, atol=1e-6
        )
        assert np.isfinite(result["value"]).all()
        assert result["date"].drop_duplicates().tolist() == dates.tolist()
    return model


@pytest.mark.xdist_group(name="pretrained_foundation")
@pytest.mark.parametrize(
    "n_targets, missing, with_regressors",
    [
        (1, False, False),
        (1, False, True),
        (2, False, False),
        (2, False, True),
        (2, True, True),
    ],
)
def test_pretrained_backend_parity(
    pretrained_local_model, pretrained_data, n_targets, missing, with_regressors
):
    model_class, backend, checkpoint = pretrained_local_model
    history = pretrained_data.iloc[:, :n_targets].copy()
    steps = 3
    calendar = history.index
    dates = pd.date_range(calendar[-1], periods=steps + 1, freq="ME")[1:]
    regressors = None
    if with_regressors:
        regressors = pd.DataFrame(
            {"indicator": np.sin(np.arange(len(calendar) + steps) / 4)},
            index=calendar.append(dates),
        )
    if missing:
        history.iloc[2, 0] = np.nan
        history = history.drop(history.index[7])
        regressors.iloc[4, 0] = np.nan
    _assert_pretrained_parity(
        model_class, backend, checkpoint, history, regressors, steps
    )


@pytest.mark.xdist_group(name="pretrained_foundation")
def test_pretrained_changed_future_covariates(pretrained_local_model, pretrained_data):
    model_class, backend, checkpoint = pretrained_local_model
    steps = 3
    dates = pd.date_range(pretrained_data.index[-1], periods=steps + 1, freq="ME")[1:]
    regressors = pd.DataFrame(
        {"indicator": np.sin(np.arange(len(pretrained_data) + steps) / 4)},
        index=pretrained_data.index.append(dates),
    )
    _assert_pretrained_parity(
        model_class, backend, checkpoint, pretrained_data, regressors, steps
    )
    changed = regressors.copy()
    changed.loc[dates, "indicator"] += 50
    _assert_pretrained_parity(
        model_class, backend, checkpoint, pretrained_data, changed, steps
    )


@pytest.mark.xdist_group(name="pretrained_foundation")
def test_pretrained_autoregressive_rollout(pretrained_local_model, pretrained_data):
    model_class, backend, checkpoint = pretrained_local_model
    history = pretrained_data[["first"]]
    if model_class is ForecastTimestral:
        steps = backend.config.patch_size * backend.config.block_size + 1
    else:
        steps = backend.max_horizon + 1
    dates = pd.date_range(history.index[-1], periods=steps + 1, freq="ME")[1:]
    regressors = pd.DataFrame(
        {"indicator": np.sin(np.arange(len(history) + steps) / 4)},
        index=history.index.append(dates),
    )
    _assert_pretrained_parity(
        model_class, backend, checkpoint, history, regressors, steps
    )


@pytest.mark.xdist_group(name="pretrained_foundation")
def test_pretrained_realtime_parity_and_no_future_target_leakage(
    pretrained_local_model, pretrained_data
):
    from forecast_evaluation import ForecastData

    model_class, backend, checkpoint = pretrained_local_model
    history = pretrained_data
    origin = history.index[-1]
    steps = 3
    dates = pd.date_range(origin, periods=steps + 1, freq="ME")[1:]
    regressors = pd.DataFrame(
        {"indicator": np.sin(np.arange(len(history) + steps) / 4)},
        index=history.index.append(dates),
    )
    model = _assert_pretrained_parity(
        model_class, backend, checkpoint, history, regressors, steps
    )
    observed = pd.concat([history, regressors.loc[history.index]], axis=1)
    outturns = (
        observed.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="variable", value_name="value")
    )
    outturns = outturns.assign(
        vintage_date=outturns["date"], frequency="M", metric="levels"
    )
    withheld = pd.DataFrame(
        {"first": [10.0, 20.0, 30.0], "second": [40.0, 50.0, 60.0]}, index=dates
    )
    withheld = (
        withheld.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="variable", value_name="value")
    )
    withheld = withheld.assign(
        vintage_date=withheld["date"], frequency="M", metric="levels"
    )
    known = pd.DataFrame(
        {
            "date": dates,
            "variable": "indicator",
            "value": regressors.loc[dates, "indicator"].to_numpy(),
            "vintage_date": origin,
            "frequency": "M",
            "metric": "levels",
            "source": "known",
            "forecast_horizon": np.arange(steps),
        }
    )
    later_covariates = known.assign(vintage_date=dates[0], value=1_000_000.0)
    paths = pd.concat([known, later_covariates], ignore_index=True)
    mapping = dict.fromkeys([*history.columns, "indicator"], "levels")

    for probabilities in (False, [0.1, 0.5, 0.9]):
        expected = model.forecast(
            steps=steps, X=regressors.loc[dates], quantiles=probabilities
        )
        keys = ["date", "variable"]
        if probabilities is not False:
            keys.append("quantile")
        expected = expected[keys + ["value"]].sort_values(keys).reset_index(drop=True)
        for perturbation in (0.0, 1_000_000.0):
            unavailable = withheld.assign(value=withheld["value"] + perturbation)
            data = ForecastData(
                outturns_data=pd.concat([outturns, unavailable], ignore_index=True),
                forecasts_data=paths,
                metric="levels",
                compute_levels=False,
                data_check=False,
            )
            realtime = rt.RealTimeModel(
                data,
                [
                    model_class(
                        model_id=checkpoint,
                        label="pretrained",
                        conditioning={
                            "X": {"indicator": {"source": "known", "periods": steps}}
                        },
                    )
                ],
            )
            realtime.forecast(
                y_variables=list(history.columns),
                X_variables=["indicator"],
                data_transformation=mapping,
                steps=steps,
                first_forecast_horizon=1,
                first_vintage=origin,
                last_vintage=origin,
                parallel=False,
                reconstruct_levels=False,
                quantiles=probabilities,
            )
            actual = (
                realtime.data.forecasts if probabilities is False else realtime.quantiles
            )
            actual = actual.loc[
                actual["source"].eq("pretrained") & actual["metric"].eq("levels"),
                keys + ["value"],
            ]
            actual = actual.sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(
                actual,
                expected,
                check_dtype=False,
                check_exact=False,
                rtol=1e-5,
                atol=1e-6,
            )


@pytest.fixture
def example_backend(local_model, monkeypatch):
    from tests.realtime_fixtures import generate_synthetic_data

    model, calls = local_model
    loader = sys.modules[type(model).__module__]._load_forecaster
    for model_class in (ForecastTimestral, ForecastT0):
        monkeypatch.setattr(f"{model_class.__module__}._load_forecaster", loader)
    monkeypatch.setattr(rt, "generate_synthetic_data", generate_synthetic_data)
    return calls


def test_synthetic_foundation_example(example_backend):
    from forecast_realtime.examples.foundation_models import run_demo

    calls = example_backend

    demo = run_demo(n_train=12, n_origins=3, steps=2)

    assert len(calls) == 6
    assert set(demo.forecasts["source"]) == {"Timestral", "t0", "OLS", "AR(1)"}
    assert len(demo.forecasts) == 4 * 3 * 2
    assert demo.forecasts["origin"].nunique() == 3
    assert (demo.forecasts["date"] > demo.forecasts["origin"]).all()
    assert np.isfinite(demo.forecasts[["value", "actual"]]).all().all()
    assert not demo.forecasts.duplicated(["source", "origin", "horizon"]).any()
    assert demo.accuracy["n_forecasts"].eq(3).all()
    assert len(demo.accuracy) == 4 * 2
    assert demo.timings["n_origins"].eq(3).all()

    sample_data = rt.generate_synthetic_data(
        N=2,
        seed=42,
        first_period="2010-01-31",
        endpoint="2025-12-31",
        publication_lags=False,
    )
    levels = sample_data.loc[
        sample_data["vintage_date"].eq(sample_data["vintage_date"].max())
        & sample_data["metric"].eq("levels")
    ]
    monthly = levels.pivot(index="date", columns="variable", values="value")
    monthly = monthly.sort_index().asfreq("ME").iloc[-16:]
    targets = monthly[["monthly_1"]]
    regressors = monthly[["monthly_2"]]
    for position, (context, options) in enumerate(calls):
        cutoff = 12 + position % 3
        np.testing.assert_array_equal(
            context, targets.iloc[:cutoff].to_numpy().T[None, ...]
        )
        covariates = options.get("known_covariates", options.get("future_covariates"))
        np.testing.assert_array_equal(
            covariates, regressors.iloc[: cutoff + 2].to_numpy().T[None, ...]
        )

    for row in demo.accuracy.itertuples(index=False):
        predictions = demo.forecasts.loc[
            demo.forecasts["source"].eq(row.source)
            & demo.forecasts["horizon"].eq(row.horizon)
        ]
        errors = predictions["actual"] - predictions["value"]
        assert row.MAE == pytest.approx(errors.abs().mean())
        assert row.RMSE == pytest.approx(np.sqrt(errors.pow(2).mean()))
    baseline = demo.accuracy.loc[demo.accuracy["source"].eq("AR(1)")]
    np.testing.assert_allclose(baseline["relative_RMSE"], 1)


def test_synthetic_foundation_example_does_not_leak_future_targets(
    example_backend, monkeypatch
):
    from forecast_realtime.examples import foundation_models as example_module

    baseline = example_module.run_demo(n_train=30, n_origins=1, steps=3, seed=42)
    generator = rt.generate_synthetic_data
    origin = baseline.forecasts["origin"].max()

    def perturbed_data(**kwargs):
        sample_data = generator(**kwargs).copy()
        future_targets = (
            sample_data["variable"].eq("monthly_1")
            & sample_data["metric"].eq("levels")
            & sample_data["date"].gt(origin)
        )
        sample_data.loc[future_targets, "value"] += 1_000_000
        return sample_data

    monkeypatch.setattr(rt, "generate_synthetic_data", perturbed_data)
    perturbed = example_module.run_demo(n_train=30, n_origins=1, steps=3, seed=42)

    columns = ["source", "origin", "date", "horizon", "value"]
    pd.testing.assert_frame_equal(
        baseline.forecasts[columns], perturbed.forecasts[columns]
    )
    assert not np.array_equal(
        baseline.forecasts["actual"].to_numpy(), perturbed.forecasts["actual"].to_numpy()
    )
    assert baseline.timings["warm_forecast_seconds"].isna().all()
    assert perturbed.timings["warm_forecast_seconds"].isna().all()


@pytest.mark.parametrize("label", ["OLS", "AR(2)"])
def test_synthetic_foundation_baselines_match_statsmodels(example_backend, label):
    from statsmodels.tsa.ar_model import AutoReg

    from forecast_realtime.examples.foundation_models import run_demo

    n_train, n_origins, steps = 30, 3, 3
    sample_data = rt.generate_synthetic_data(
        N=2,
        seed=42,
        first_period="2010-01-31",
        endpoint="2025-12-31",
        publication_lags=False,
    )
    levels = sample_data.loc[
        sample_data["vintage_date"].eq(sample_data["vintage_date"].max())
        & sample_data["metric"].eq("levels")
    ]
    monthly = levels.pivot(index="date", columns="variable", values="value")
    monthly = monthly.sort_index().asfreq("ME").iloc[-(n_train + n_origins + steps - 1) :]
    targets = monthly["monthly_1"]
    regressors = monthly[["monthly_2"]]
    demo = run_demo(
        n_train=n_train, n_origins=n_origins, steps=steps, ar_order=2, seed=42
    )
    forecasts = demo.forecasts.loc[demo.forecasts["source"].eq(label)]

    for cutoff in range(n_train, n_train + n_origins):
        history = targets.iloc[:cutoff]
        future_regressors = regressors.iloc[cutoff : cutoff + steps]
        if label == "AR(2)":
            fitted = AutoReg(history, lags=2, trend="c").fit()
            expected = fitted.predict(start=len(history), end=len(history) + steps - 1)
        else:
            fitted = AutoReg(
                history,
                lags=1,
                exog=regressors.iloc[:cutoff],
                trend="c",
            ).fit()
            expected = fitted.predict(
                start=len(history),
                end=len(history) + steps - 1,
                exog_oos=future_regressors,
            )
        observed = forecasts.loc[
            forecasts["origin"].eq(history.index[-1]), "value"
        ].to_numpy()
        np.testing.assert_allclose(observed, np.asarray(expected), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "options",
    [
        {"n_train": 0},
        {"n_origins": 0},
        {"steps": 0},
        {"ar_order": 0},
        {"n_origins": True},
        {"n_train": 5, "ar_order": 2},
    ],
)
def test_synthetic_foundation_example_rejects_invalid_sizes(options):
    from forecast_realtime.examples.foundation_models import run_demo

    with pytest.raises(ValueError, match="positive integer|n_train must exceed"):
        run_demo(**options)
