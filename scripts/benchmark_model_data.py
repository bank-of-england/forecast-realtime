"""Deterministic baselines for the model-data refactor."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import io
import json
import multiprocessing
import os
import pickle
import platform
import sys
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from forecast_evaluation import ForecastData

import forecast_realtime as rt
from forecast_realtime._model_data import ModelData
from forecast_realtime.real_time_model import _run_forecast_task

DEFAULT_SEED = 20260908
DEFAULT_REPEATS = 3
HISTORY_MONTHS = 72
FORECAST_STEPS = 6
REALTIME_VINTAGES = 6
TREE_STEPS = 4
WORKER_COUNT = 2


def _peak_process_memory_bytes() -> int | None:
    """Return the process peak working-set or resident-set size."""
    if os.name == "nt":
        try:

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            process = kernel32.GetCurrentProcess()
            get_process_memory_info = psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ProcessMemoryCounters),
                ctypes.c_ulong,
            ]
            get_process_memory_info.restype = ctypes.c_int
            success = get_process_memory_info(
                process, ctypes.byref(counters), counters.cb
            )
            if success:
                return int(counters.PeakWorkingSetSize)
        except (AttributeError, OSError):
            return None
        return None

    try:
        import resource
    except ImportError:
        return None

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _make_monthly_data(seed: int, months: int = HISTORY_MONTHS):
    """Build positive monthly levels and a ragged future regressor edge."""
    rng = np.random.default_rng(seed)
    total = months + FORECAST_STEPS
    index = pd.date_range("2018-01-31", periods=total, freq="ME")
    innovations = rng.normal(0.0, 0.8, (total, 3))
    levels = 100.0 + np.cumsum(innovations, axis=0)
    levels[:, 1:] += np.array([30.0, 60.0])
    y = pd.DataFrame({"target": levels[:months, 0]}, index=index[:months])
    X_history = pd.DataFrame(
        {"fast": levels[:months, 1], "slow": levels[:months, 2]},
        index=index[:months],
    )
    X_future = pd.DataFrame(
        {"fast": levels[months:, 1], "slow": levels[months:, 2]},
        index=index[months:],
    )
    X_future.loc[X_future.index[3:], "slow"] = np.nan
    X_future.loc[X_future.index[4:], "fast"] = np.nan
    X_all = pd.concat([X_history, X_future])
    return y, X_history, X_all


def _make_realtime_panel(seed: int, months: int = HISTORY_MONTHS) -> pd.DataFrame:
    """Build six complete, revised monthly snapshots over one fixed history."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-31", periods=months, freq="ME")
    vintages = pd.date_range("2024-07-31", periods=REALTIME_VINTAGES, freq="ME")
    innovations = rng.normal(0.0, 0.7, (months, 3))
    levels = 100.0 + np.cumsum(innovations, axis=0)
    levels[:, 1:] += np.array([25.0, 55.0])
    rows = []
    for vintage_number, vintage in enumerate(vintages):
        revision = rng.normal(0.0, 0.03, levels.shape)
        revised = levels + revision * vintage_number
        for column_number, variable in enumerate(("target", "fast", "slow")):
            rows.extend(
                {
                    "date": date,
                    "vintage_date": vintage,
                    "variable": variable,
                    "frequency": "M",
                    "metric": "levels",
                    "value": float(value),
                }
                for date, value in zip(dates, revised[:, column_number], strict=True)
            )
    return pd.DataFrame(rows)


def _forecast_checksum(frame: pd.DataFrame, digits: int = 8) -> str:
    """Hash a rounded forecast with dates and columns made JSON-stable."""
    values = frame.reset_index()
    if "index" in values.columns:
        values = values.rename(columns={"index": "date"})
    return _frame_checksum(values, digits)


def _json_value(value, digits: int):
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return round(value, digits)
    return value


def _frame_checksum(frame: pd.DataFrame, digits: int = 10) -> str:
    frame = frame.copy()
    column_counts = {}
    column_names = []
    for column in frame.columns:
        count = column_counts.get(column, 0)
        column_counts[column] = count + 1
        column_names.append(column if count == 0 else f"{column}__{count}")
    frame.columns = column_names
    records = [
        {column: _json_value(value, digits) for column, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    encoded_records = sorted(
        json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
    )
    payload = "\n".join(encoded_records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decomposition_summary(frame: pd.DataFrame | None) -> dict:
    if frame is None:
        return {"rows": 0, "columns": [], "coverage": {}}

    coverage = {}
    grouping = (
        frame.groupby("decomposition", dropna=False, sort=True)
        if "decomposition" in frame
        else [("model", frame)]
    )
    for name, group in grouping:
        key = "missing" if pd.isna(name) else str(name)
        coverage[key] = {"rows": int(len(group))}
        if "vintage_date" in group:
            coverage[key]["vintages"] = int(group["vintage_date"].nunique())
        if "forecast_horizon" in group:
            coverage[key]["horizons"] = sorted(
                group["forecast_horizon"].unique().tolist()
            )
        if "variable" in group:
            coverage[key]["variables"] = sorted(
                group["variable"].astype(str).unique().tolist()
            )
    base_dates = frame.get("base_vintage_date", pd.Series(dtype="datetime64[ns]"))
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "checksum": _frame_checksum(frame),
        "coverage": coverage,
        "base_vintage_date_non_null": int(base_dates.notna().sum()),
    }


def _result_summary(forecast, decomposition=None) -> dict:
    forecast_frame = pd.DataFrame(forecast)
    summary = {
        "forecast_rows": int(len(forecast_frame)),
        "forecast_columns": list(forecast_frame.columns),
        "forecast_checksum": _forecast_checksum(forecast_frame),
        "forecast_rounding_digits": 8,
    }
    if decomposition is not None:
        summary["decomposition"] = _decomposition_summary(decomposition)
    return summary


def _run_direct_ols(seed: int) -> dict:
    y, X_history, X_all = _make_monthly_data(seed)
    model = rt.models.ForecastOLS(
        label="direct_ols",
        formula="target ~ fast + slow",
        data_transformation={
            "target": "levels",
            "fast": "levels",
            "slow": "levels",
        },
    )
    model.fit(y, X_history, frequency="M", X_imputation="last")
    result = model.forecast(
        steps=FORECAST_STEPS,
        X=X_all,
        frequency="M",
        X_imputation="last",
        decomp=True,
    )
    return _result_summary(result.forecast, result.decomposition)


def _run_realtime(seed: int, panel: pd.DataFrame | None = None) -> dict:
    panel = _make_realtime_panel(seed) if panel is None else panel
    data, model = _make_realtime_components(panel)
    realtime = rt.RealTimeModel(data, model)
    with contextlib.redirect_stderr(io.StringIO()):
        realtime.forecast(
            y_variables=["target"],
            X_variables=["fast", "slow"],
            steps=3,
            first_vintage="2024-07-31",
            last_vintage="2024-12-31",
            X_imputation="last",
            reconstruct_levels=False,
            decomp=True,
            parallel=False,
        )
    if realtime.decompositions is None:
        raise RuntimeError("realtime workload produced no decomposition")
    return _result_summary(realtime.data.forecasts, realtime.decompositions)


def _make_realtime_components(panel: pd.DataFrame):
    data = ForecastData(
        outturns_data=panel,
        compute_levels=False,
        data_check=False,
    )
    model = rt.models.ForecastOLS(
        label="realtime_ols",
        formula="target ~ fast + slow",
        data_transformation={
            "target": "levels",
            "fast": "levels",
            "slow": "levels",
        },
    )
    return data, model


def _mean_components(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    result = None
    for component in components.values():
        result = component if result is None else result.add(component, fill_value=0.0)
    return result / len(components)


def _make_tree(seed: int) -> tuple[rt.ForecastTree, pd.DataFrame, pd.DataFrame]:
    y, X_history, X_all = _make_monthly_data(seed, months=HISTORY_MONTHS - 6)
    shared = rt.models.ForecastOLS(
        label="shared",
        formula="target ~ fast",
        data_transformation={"target": "levels", "fast": "levels"},
    )
    log_leaf = rt.models.ForecastOLS(
        label="log_leaf",
        formula="target ~ slow",
        data_transformation={"target": "logs", "slow": "logs"},
    )
    pop_leaf = rt.models.ForecastOLS(
        label="pop_leaf",
        formula="target ~ fast + slow",
        data_transformation={
            "target": "pop",
            "fast": "pop",
            "slow": "pop",
        },
    )
    left = rt.TreeNode(
        transform=_mean_components,
        children=[shared, log_leaf],
        name="left",
        target="target",
    )
    right = rt.TreeNode(
        transform=_mean_components,
        children=[shared, pop_leaf],
        name="right",
        target="target",
    )
    root = rt.TreeNode(
        transform=_mean_components,
        children=[left, right],
        name="root",
        target="target",
    )
    return rt.ForecastTree(root, label="callable_tree"), y, X_all


def _run_tree(seed: int) -> dict:
    tree, y, X_all = _make_tree(seed)
    tree.fit(y, X=X_all.loc[y.index], frequency="M", X_imputation="last")
    result = tree.forecast(
        steps=TREE_STEPS,
        X=X_all,
        frequency="M",
        X_imputation="last",
    )
    return _result_summary(result.forecast)


def _timed(function, seed: int) -> dict:
    tracemalloc.start()
    process_memory_before = _peak_process_memory_bytes()
    started = time.perf_counter()
    summary = function(seed)
    runtime = time.perf_counter() - started
    _, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    process_memory_after = _peak_process_memory_bytes()
    return {
        "runtime_seconds": round(runtime, 6),
        "tracemalloc_peak_bytes": int(tracemalloc_peak),
        "process_peak_bytes": process_memory_after,
        "process_peak_delta_bytes": (
            process_memory_after - process_memory_before
            if process_memory_after is not None and process_memory_before is not None
            else None
        ),
        **summary,
    }


def _run_realtime_payload(payload: dict) -> dict:
    return _run_realtime(payload["seed"], payload["panel"])


def _spawn_worker(payload: dict) -> dict:
    with contextlib.redirect_stderr(io.StringIO()):
        return _timed(_run_realtime_payload, payload)


def _run_spawned(seed: int) -> dict:
    panel = _make_realtime_panel(seed)
    payload = {"seed": seed, "panel": panel}
    serial_started = time.perf_counter()
    serialised = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    serialisation_seconds = time.perf_counter() - serial_started
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=context, max_workers=WORKER_COUNT) as pool:
        worker_result = pool.submit(_spawn_worker, payload).result()
    worker_result["parent_runtime_seconds"] = round(time.perf_counter() - started, 6)
    worker_result["payload_bytes"] = len(serialised)
    worker_result["serialisation_seconds"] = round(serialisation_seconds, 6)
    return worker_result


def _make_forecast_task(panel: pd.DataFrame):
    _, model = _make_realtime_components(panel)
    archive = ModelData.from_archive(
        panel,
        frequencies={"target": "M", "fast": "M", "slow": "M"},
    )
    y_variables = ["target"]
    X_variables = ["fast", "slow"]
    mapping = model.resolve_input_data_transformation(
        y_variables=y_variables,
        X_variables=X_variables,
    ).data_transformation
    data = archive.select(model.input_requirements(y_variables, X_variables))
    vintages = np.sort(panel["vintage_date"].unique())
    common = {
        "y_steps_ahead": None,
        "X_steps_ahead": None,
        "steps": 3,
        "label": None,
        "first_forecast_horizon": None,
        "frequency": "M",
        "y_lags": 0,
        "X_lags": 0,
        "dummies": None,
        "decomp": True,
        "X_imputation": "last",
        "drop_transformation_nans": True,
    }
    tasks = rt.RealTimeModel._build_forecast_tasks(
        [(model, mapping, data)],
        vintages,
        common,
        batch_size=None,
        parallel=False,
        max_workers=WORKER_COUNT,
        model_kwargs={},
    )
    if len(tasks) != 1:
        raise RuntimeError(f"expected one complete forecast task, got {len(tasks)}")
    return tasks[0]


def _forecast_task_summary(task) -> dict:
    result = _run_forecast_task(task)
    return {
        "native_forecast_rows": int(len(result.forecasts)),
        "native_forecast_columns": list(result.forecasts.columns),
        "native_forecast_checksum": _frame_checksum(result.forecasts),
        "forecast_rounding_digits": 10,
        "decomposition": _decomposition_summary(result.decompositions),
    }


def _spawn_forecast_task_worker(task) -> dict:
    with contextlib.redirect_stderr(io.StringIO()):
        return _timed(_forecast_task_summary, task)


def _run_spawned_forecast_task(seed: int) -> dict:
    task = _make_forecast_task(_make_realtime_panel(seed))
    serial_started = time.perf_counter()
    serialised = pickle.dumps(task, protocol=pickle.HIGHEST_PROTOCOL)
    serialisation_seconds = time.perf_counter() - serial_started
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(mp_context=context, max_workers=WORKER_COUNT) as pool:
        worker_result = pool.submit(_spawn_forecast_task_worker, task).result()
    worker_result["parent_runtime_seconds"] = round(time.perf_counter() - started, 6)
    worker_result["payload_bytes"] = len(serialised)
    worker_result["serialisation_seconds"] = round(serialisation_seconds, 6)
    return worker_result


def _dependency_versions() -> dict[str, str | None]:
    names = {
        "forecast_realtime": "forecast_realtime",
        "forecast_evaluation": "forecast-evaluation",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "statsmodels": "statsmodels",
    }
    return {
        label: _distribution_version(distribution)
        for label, distribution in names.items()
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"number of repeats (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"base random seed (default: {DEFAULT_SEED})",
    )
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        parser.error("--repeats must be at least 1")
    return arguments


def main() -> None:
    arguments = _parse_args()
    seeds = [arguments.seed for _ in range(arguments.repeats)]
    benchmarks = {
        "direct_ols": [],
        "realtime_ols": [],
        "callable_tree": [],
        "spawned_realtime_ols": [],
        "spawned_forecast_task_ols": [],
    }
    for seed in seeds:
        benchmarks["direct_ols"].append(_timed(_run_direct_ols, seed))
        benchmarks["realtime_ols"].append(_timed(_run_realtime, seed))
        benchmarks["callable_tree"].append(_timed(_run_tree, seed))
        benchmarks["spawned_realtime_ols"].append(_run_spawned(seed))
        benchmarks["spawned_forecast_task_ols"].append(_run_spawned_forecast_task(seed))

    output = {
        "schema_version": 2,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": _dependency_versions(),
        },
        "configuration": {
            "repeats": arguments.repeats,
            "seeds": seeds,
            "sizes": {
                "history_months": HISTORY_MONTHS,
                "realtime_vintages": REALTIME_VINTAGES,
                "direct_steps": FORECAST_STEPS,
                "tree_steps": TREE_STEPS,
            },
            "workers": {
                "realtime_parallel": False,
                "realtime_batch_size": None,
                "spawn_context": "spawn",
                "spawn_max_workers": WORKER_COUNT,
                "spawn_tasks_per_repeat": 1,
                "spawn_forecast_task": "one complete task with sequential decomposition",
            },
        },
        "benchmarks": benchmarks,
    }
    json.dump(output, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
