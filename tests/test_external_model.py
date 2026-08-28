"""Tests for ``ExternalModel`` cache-directory isolation.

External models exchange data with the external process through fixed
filenames (``y.parquet``, ``forecasts.parquet``, ...) inside a temporary
cache directory.  ``RealTimeModel`` deep-copies a model per vintage and
pickles it to worker processes, so copies must not share a directory.

These tests use a fake subclass that emulates the external process
in-process, so no R/MATLAB/Julia installation is required.
"""

import copy
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forecast_realtime.external_model import (
    ExternalModel,
    ExternalProcessError,
    ExternalProcessTimeoutError,
    JuliaModel,
    MATLABModel,
    RModel,
    _julia_string_literal,
    _matlab_string_literal,
    _r_string_literal,
)


class FakeExternalModel(ExternalModel):
    """External model whose "external process" runs in-process.

    Communicates only via files in ``cache_dir``, exactly as a real
    external model does, so it exercises the directory isolation logic.
    """

    def _fit_command(self) -> list[str]:
        return ["fit"]

    def _forecast_command(self, steps: int) -> list[str]:
        return ["forecast", str(steps)]

    def _run(self, cmd: list[str]) -> None:
        if cmd[0] == "fit":
            y = pd.read_parquet(self.cache_dir / "y.parquet")
            (self.cache_dir / "model.txt").write_text(str(y["gdp"].mean()))
        else:
            value = float((self.cache_dir / "model.txt").read_text())
            steps = int(cmd[1])
            pd.DataFrame({"gdp": [value] * steps}).to_parquet(
                self.cache_dir / "forecasts.parquet"
            )


class NoOutputExternalModel(FakeExternalModel):
    """Fake process that succeeds without producing forecast output."""

    def _run(self, cmd: list[str]) -> None:
        if cmd[0] == "fit":
            super()._run(cmd)


class OutputVariantExternalModel(FakeExternalModel):
    """Fake process whose forecast output can exercise validation failures."""

    output = None

    def _run(self, cmd: list[str]) -> None:
        if cmd[0] == "fit":
            super()._run(cmd)
            return
        if self.output == "malformed":
            (self.cache_dir / "forecasts.parquet").write_text("not parquet")
        else:
            pd.DataFrame(self.output).to_parquet(self.cache_dir / "forecasts.parquet")


def _fit_and_report(payload):
    """Fit a pickled model on its own data; return its cache dir and forecast.

    Run in a worker process to mimic ``RealTimeModel._forecast_parallel``.
    """
    model, y = payload
    model.fit(y)
    fc = model.forecast(steps=2)
    return str(model.cache_dir), float(fc.values[0, 0])


@pytest.fixture
def quarterly_data():
    dates = pd.date_range("2000-03-31", periods=12, freq="QE")
    return pd.DataFrame({"gdp": np.arange(12, dtype=float)}, index=dates)


def _make_data(offset: float) -> pd.DataFrame:
    dates = pd.date_range("2000-03-31", periods=12, freq="QE")
    return pd.DataFrame({"gdp": np.arange(12, dtype=float) + offset}, index=dates)


class TestCacheDirIsolation:
    def test_params_path_tracks_cache_dir(self, quarterly_data):
        model = FakeExternalModel("dummy.R", window_size=4)
        before = model.params_path
        model.fit(quarterly_data)

        assert model.params_path != before
        assert Path(model.params_path).parent == model.cache_dir
        assert Path(model.params_path).exists()

    def test_refitting_uses_a_fresh_directory(self, quarterly_data):
        model = FakeExternalModel("dummy.R")
        model.fit(quarterly_data)
        first = model.cache_dir
        model.fit(quarterly_data)

        assert model.cache_dir != first

    def test_deepcopies_do_not_share_a_directory(self):
        """RealTimeModel deep-copies the model for every vintage."""
        base = FakeExternalModel("dummy.R")
        copies = [copy.deepcopy(base) for _ in range(3)]
        for i, model in enumerate(copies):
            model.fit(_make_data(offset=100 * i))

        dirs = {str(model.cache_dir) for model in copies}
        assert len(dirs) == 3

        # Each copy's directory holds its own data, not another copy's.
        for i, model in enumerate(copies):
            y = pd.read_parquet(model.cache_dir / "y.parquet")
            np.testing.assert_allclose(y["gdp"].values, np.arange(12) + 100 * i)

    def test_pickled_copy_does_not_disturb_the_original(self, quarterly_data):
        """Sending a model to a worker must not clobber the parent's files."""
        original = FakeExternalModel("dummy.R")
        original.fit(quarterly_data)
        original_dir = original.cache_dir

        worker_copy = pickle.loads(pickle.dumps(original))
        worker_copy.fit(_make_data(offset=500))

        assert worker_copy.cache_dir != original_dir
        assert (original_dir / "y.parquet").exists()
        y = pd.read_parquet(original_dir / "y.parquet")
        np.testing.assert_allclose(y["gdp"].values, np.arange(12))

    def test_serialises_input_dates_as_a_named_column(self, quarterly_data):
        model = FakeExternalModel("dummy.R")
        X = pd.DataFrame({"regressor": np.arange(12)}, index=quarterly_data.index)

        model.fit(quarterly_data, X)

        written_y = pd.read_parquet(model.cache_dir / "y.parquet")
        written_X = pd.read_parquet(model.cache_dir / "X.parquet")
        assert written_y.columns.tolist() == ["date", "gdp"]
        assert written_X.columns.tolist() == ["date", "regressor"]
        pd.testing.assert_series_equal(
            written_y["date"], pd.Series(quarterly_data.index, name="date")
        )
        pd.testing.assert_series_equal(written_X["date"], pd.Series(X.index, name="date"))

    def test_forecast_without_X_removes_fit_regressors(self, quarterly_data):
        model = FakeExternalModel("dummy.R")
        X = pd.DataFrame({"regressor": np.arange(12)}, index=quarterly_data.index)
        model.fit(quarterly_data, X)

        model.forecast(steps=2)

        assert not (model.cache_dir / "X.parquet").exists()

    def test_pipeline_forecast_without_X_removes_fit_regressors(self, quarterly_data):
        model = FakeExternalModel(
            "dummy.R",
            data_transformation={"gdp": "levels", "regressor": "levels"},
        )
        X = pd.DataFrame({"regressor": np.arange(12)}, index=quarterly_data.index)
        model.fit(quarterly_data, X)

        model.forecast(steps=2)

        assert not (model.cache_dir / "X.parquet").exists()


class TestParallelFits:
    def test_parallel_fits_are_independent(self):
        """Concurrent fits of copies of one model must not cross-contaminate."""
        base = FakeExternalModel("dummy.R")
        offsets = [0.0, 100.0, 200.0, 300.0]
        payloads = [(base, _make_data(offset)) for offset in offsets]

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(_fit_and_report, payloads))

        cache_dirs = [cache_dir for cache_dir, _ in results]
        assert len(set(cache_dirs)) == len(offsets)

        # Each forecast is the mean of that task's own data, not another's.
        expected = [np.arange(12).mean() + offset for offset in offsets]
        actual = [value for _, value in results]
        np.testing.assert_allclose(actual, expected)


class TestExternalContracts:
    def test_forecast_conditioning_y_is_written_to_cache(self, quarterly_data):
        model = FakeExternalModel("dummy.R")
        model.fit(quarterly_data)
        future_dates = pd.date_range(
            quarterly_data.index[-1] + pd.offsets.QuarterEnd(), periods=2, freq="QE"
        )
        conditioning = pd.DataFrame({"gdp": [100.0, 101.0]}, index=future_dates)

        model.forecast(steps=2, y=conditioning)

        written_y = pd.read_parquet(model.cache_dir / "y.parquet")
        assert written_y["gdp"].tail(2).tolist() == [100.0, 101.0]

    def test_stale_forecast_output_is_not_returned(self, quarterly_data):
        model = NoOutputExternalModel("dummy.R")
        model.fit(quarterly_data)
        pd.DataFrame({"gdp": [999.0, 999.0]}).to_parquet(
            model.cache_dir / "forecasts.parquet"
        )

        with pytest.raises(ValueError, match="did not produce forecast output"):
            model.forecast(steps=2)

    @pytest.mark.parametrize(
        ("output", "exception", "message"),
        [
            ({"gdp": [1.0]}, ValueError, "must have 2 rows"),
            ({"other": [1.0, 2.0]}, ValueError, "columns must match"),
            ({"gdp": ["one", "two"]}, TypeError, "must contain numeric"),
            ({"gdp": [1.0, np.nan]}, ValueError, "contains missing values"),
            ({"gdp": [1.0, np.inf]}, ValueError, "contains non-finite values"),
            ("malformed", ValueError, "not a readable Parquet file"),
        ],
    )
    def test_forecast_output_is_validated(
        self, quarterly_data, output, exception, message
    ):
        model = OutputVariantExternalModel("dummy.R")
        model.output = output
        model.fit(quarterly_data)

        with pytest.raises(exception, match=message):
            model.forecast(steps=2)

    def test_external_process_failure_includes_diagnostics(self):
        model = FakeExternalModel("dummy.R")
        process = subprocess.CompletedProcess(
            ["external"], 7, stdout="standard output", stderr="useful error"
        )
        with patch(
            "forecast_realtime.external_model.subprocess.run", return_value=process
        ):
            with pytest.raises(ExternalProcessError, match="useful error"):
                ExternalModel._run(model, ["external"])

    def test_external_process_timeout_is_reported(self):
        model = FakeExternalModel("dummy.R", subprocess_timeout=3)
        timeout = subprocess.TimeoutExpired(
            ["external"], 3, output="partial output", stderr="still running"
        )
        with patch(
            "forecast_realtime.external_model.subprocess.run", side_effect=timeout
        ) as run:
            with pytest.raises(ExternalProcessTimeoutError, match="3 seconds"):
                ExternalModel._run(model, ["external"])

        assert run.call_args.kwargs["timeout"] == 3

    @pytest.mark.parametrize("value", [[1], {"value": 1}, None, object()])
    def test_unsupported_external_parameter_type_fails_at_construction(self, value):
        with pytest.raises(TypeError, match="External parameter 'custom'"):
            FakeExternalModel("dummy.R", custom=value)

    def test_matlab_accepts_common_model_options(self):
        model = MATLABModel(
            "model.m",
            label="matlab",
            debug="fit",
            formula="gdp ~ indicator",
            data_transformation={"gdp": "levels"},
            custom_parameter=1,
        )

        assert model.label == "matlab"
        assert model.debug == "fit"
        assert model._formula is not None
        assert model.data_transformation == {"gdp": "levels"}
        assert model.params == {"custom_parameter": 1}


class TestExternalCommandBoundaries:
    @pytest.mark.parametrize(
        ("literal_builder", "value", "expected"),
        [
            (
                _r_string_literal,
                'C:/a "quoted"; path\\file',
                '"C:/a \\"quoted\\"; path\\\\file"',
            ),
            (_matlab_string_literal, "C:/a 'quoted'; path", "'C:/a ''quoted''; path'"),
            (
                _julia_string_literal,
                'C:/a "$value"; path\\file',
                '"C:/a \\"\\$value\\"; path\\\\file"',
            ),
        ],
    )
    def test_language_string_literals_escape_code_boundaries(
        self, literal_builder, value, expected
    ):
        assert literal_builder(value) == expected

    def test_r_debug_source_quotes_paths(self, quarterly_data, tmp_path):
        script = tmp_path / 'model "quoted"; path.R'
        model = RModel(str(script), debug="fit")
        with patch("subprocess.run"):
            model.fit(quarterly_data)

        source = (model.cache_dir / "_debug_init.R").read_text()

        r_cache_dir = str(model.cache_dir).replace("\\", "/")
        r_script = str(script.resolve()).replace("\\", "/")
        assert f"cache_dir   <- {_r_string_literal(r_cache_dir)}" in source
        assert f"source({_r_string_literal(r_script)})" in source
        assert 'cache_dir   <- "' in source
        assert "quoted" in source

    def test_julia_debug_source_quotes_paths(self, quarterly_data, tmp_path):
        script = tmp_path / 'model "$value"; path.jl'
        model = JuliaModel(str(script), debug="fit")
        with patch("subprocess.run"):
            model.fit(quarterly_data)

        command = model._debug_source("fit", 1)

        assert _julia_string_literal(str(model.cache_dir)) in command
        assert _julia_string_literal(str(script.resolve())) in command
        assert "$value" not in command.replace(r"\$value", "")

    def test_matlab_commands_quote_dynamic_values(self, tmp_path):
        script = tmp_path / "folder 'quoted'; path" / "model.m"
        model = MATLABModel(str(script))

        command = model._fit_command()[2]

        assert _matlab_string_literal(str(model.cache_dir)) in command
        assert _matlab_string_literal(str(script.resolve().parent)) in command
        assert _matlab_string_literal(model._function_name) in command
        assert "runner('model', 'fit'" in command

    def test_matlab_rejects_invalid_function_name(self, tmp_path):
        with pytest.raises(ValueError, match="valid MATLAB identifier"):
            MATLABModel(str(tmp_path / "model;delete.m"))

    def test_matlab_debug_source_uses_function_handle(self, tmp_path):
        script = tmp_path / "folder 'quoted'; path" / "model.m"
        model = MATLABModel(str(script))

        source = model._debug_source("forecast", 4)

        assert "user_function = str2func('model')" in source
        assert (
            "result = user_function('forecast', loaded.model, 4, X, y, params)" in source
        )
        assert "result = model('forecast'" not in source
        assert _matlab_string_literal(str(script.resolve().parent)) in source

    def test_normal_runner_paths_remain_individual_arguments(self, tmp_path):
        script = tmp_path / "model with spaces.R"
        model = RModel(str(script))

        command = model._forecast_command(3)

        assert command[0] == "Rscript"
        assert command[2] == str(script.resolve())
        assert command[4] == str(model.cache_dir)
        assert command[5] == "3"
        assert command[6] == model.params_path

    def test_julia_forecast_runner_paths_remain_individual_arguments(self, tmp_path):
        script = tmp_path / "model; with spaces.jl"
        model = JuliaModel(str(script))

        command = model._forecast_command(3)

        assert command == [
            "julia",
            model._RUNNER,
            str(script.resolve()),
            "forecast",
            str(model.cache_dir),
            "3",
            model.params_path,
        ]
