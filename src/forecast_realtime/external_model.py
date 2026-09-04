"""Base classes for forecasting models implemented in external languages.

Provides ``ExternalModel`` (abstract) and concrete subclasses for R, MATLAB
and Julia.  Extra model parameters passed as keyword arguments are written to
a ``params.parquet`` file (single-row DataFrame) in the shared cache
directory, alongside ``y.parquet`` and optionally ``X.parquet``.

Supported parameter types: ``str``, ``bool``, ``int``, ``float``.
"""

import re
import subprocess
import tempfile
from abc import abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast_model import ForecastModel


def _r_string_literal(value: str) -> str:
    """Return *value* as an R double-quoted string literal."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _matlab_string_literal(value: str) -> str:
    """Return *value* as a MATLAB single-quoted character-vector literal."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _julia_string_literal(value: str) -> str:
    """Return *value* as a Julia interpolating-string-safe literal."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _steps_literal(steps: int) -> str:
    """Return a validated integer suitable for generated external source."""
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise TypeError("steps must be an integer")
    return str(steps)


class ExternalProcessError(subprocess.CalledProcessError):
    """Raised when an external forecasting process exits unsuccessfully."""

    def __init__(self, returncode, cmd, stdout, stderr):
        super().__init__(returncode, cmd, output=stdout, stderr=stderr)

    def __str__(self) -> str:
        details = [f"External process failed with exit code {self.returncode}"]
        if self.stderr:
            details.append(f"stderr: {self.stderr.strip()}")
        if self.stdout:
            details.append(f"stdout: {self.stdout.strip()}")
        return "; ".join(details)


class ExternalProcessTimeoutError(subprocess.TimeoutExpired):
    """Raised when an external forecasting process exceeds its timeout."""

    def __str__(self) -> str:
        return f"External process timed out after {self.timeout} seconds: {self.cmd!r}"


class ExternalModel(ForecastModel):
    """Abstract base for models implemented in an external language.

    Handles temporary-directory lifecycle, Parquet I/O, parameter
    serialisation and subprocess execution.  Subclasses implement
    ``_fit_command`` and ``_forecast_command`` to specify the shell command
    for each stage.

    Parameters
    ----------
    script : str
        Path to the external script (e.g. ``my_model.R``). Resolved against the
        current working directory, so build it from ``__file__`` (see Examples)
        to keep it independent of where Python is launched from.
    debug : str | None
        If ``"fit"`` or ``"forecast"``, drop into an interactive debug REPL
        for that stage instead of running the script. Default None.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target and regressor variables.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    subprocess_timeout : float | None
        Maximum number of seconds allowed for an external process. ``None``
        disables the timeout.

    Examples
    --------
    >>> from pathlib import Path
    >>> script = str(Path(__file__).parent / "my_model.R")
    >>> model = RModel(script, p=4, horizon=8)
    >>> model.fit(y)
    >>> forecasts = model.forecast(steps=4)
    """

    def __init__(
        self,
        script: str,
        *,
        debug: str | None = None,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        subprocess_timeout: float | None = None,
        **params,
    ):
        if debug not in (None, "fit", "forecast"):
            raise ValueError("debug must be one of None, 'fit' or 'forecast'")
        if subprocess_timeout is not None and (
            isinstance(subprocess_timeout, bool)
            or not isinstance(subprocess_timeout, (int, float))
            or not np.isfinite(subprocess_timeout)
            or subprocess_timeout <= 0
        ):
            raise ValueError(
                "subprocess_timeout must be a positive finite number or None"
            )
        for name, value in params.items():
            if type(value) not in (str, bool, int, float):
                raise TypeError(
                    f"External parameter {name!r} must be a str, bool, int, or float; "
                    f"got {type(value).__name__}."
                )

        super().__init__(
            label=label,
            formula=formula,
            data_transformation=data_transformation,
        )
        self.script = str(script)
        self.debug = debug
        self.subprocess_timeout = subprocess_timeout
        self.params = params
        self._new_cache_dir()

    def _new_cache_dir(self) -> None:
        """Create the temporary cache directory for this model instance."""
        self._tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self._tmpdir.name)

    @property
    def params_path(self) -> str:
        """Path to ``params.parquet`` in the current cache directory."""
        return str(self.cache_dir / "params.parquet")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit_command(self) -> list[str]:
        """Return the shell command (as a list of tokens) to run for fitting."""
        ...

    @abstractmethod
    def _forecast_command(self, steps: int) -> list[str]:
        """Return the shell command (as a list of tokens) to run for forecasting."""
        ...

    # ------------------------------------------------------------------
    # ForecastModel implementation
    # ------------------------------------------------------------------

    def _write_params(self) -> None:
        """Write ``self.params`` to ``params.parquet`` as a single-row DataFrame."""
        if self.params:
            pd.DataFrame([self.params]).to_parquet(self.params_path)

    @staticmethod
    def _write_data(data: pd.DataFrame, path: Path) -> None:
        """Write *data* with its index in a named ``date`` column."""
        if "date" in data.columns:
            raise ValueError(
                "ExternalModel cannot serialise a data column named 'date'; "
                "that name is reserved for the input index."
            )
        data.rename_axis("date").reset_index().to_parquet(path, index=False)

    def _run(self, cmd: list[str]) -> None:
        """Run *cmd* via subprocess and preserve diagnostics on failure."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.subprocess_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalProcessTimeoutError(
                exc.cmd,
                exc.timeout,
                output=exc.output,
                stderr=exc.stderr,
            ) from exc
        if result.returncode != 0:
            raise ExternalProcessError(
                result.returncode,
                cmd,
                result.stdout,
                result.stderr,
            )

    def _fit(self, y: pd.DataFrame, X: pd.DataFrame | None = None):
        target_index = y.index
        target_name = y.columns[0]

        # Fit in a fresh directory so that concurrent fits by copies of this
        # model (see ``_new_cache_dir``) cannot overwrite each other's files.
        self._new_cache_dir()
        self._write_data(y, self.cache_dir / "y.parquet")
        if X is not None:
            self._write_data(X, self.cache_dir / "X.parquet")

        self._write_params()
        if self.debug == "fit":
            self.debug_repl("fit")
        else:
            self._run(self._fit_command())

        # Optional: a script may emit in-sample fitted values (e.g. the R
        # path writes fitted_values.parquet when the user script defines a
        # fitted_values() function). Absent for scripts/languages that don't
        # produce it, in which case fitted_values_ stays None.
        fitted_values_path = self.cache_dir / "fitted_values.parquet"
        if fitted_values_path.exists():
            values = pd.read_parquet(fitted_values_path).iloc[:, 0].to_numpy()
            # The external script receives y without its index (written with
            # index=False above), so its output is aligned by position to
            # the rows of y as fit received them.
            if len(values) != len(target_index):
                padded = np.full(len(target_index), np.nan)
                n = min(len(values), len(target_index))
                padded[:n] = values[:n]
                values = padded
            self.fitted_values_ = pd.Series(values, index=target_index, name=target_name)

        return self

    def _forecast(
        self,
        steps: int,
        X: pd.DataFrame | None = None,
        y: pd.DataFrame | None = None,
        **kwargs,
    ) -> np.ndarray:
        forecast_path = self.cache_dir / "forecasts.parquet"
        if forecast_path.exists():
            forecast_path.unlink()

        x_path = self.cache_dir / "X.parquet"
        if X is not None:
            self._write_data(X, x_path)
        elif x_path.exists():
            # Do not let fit regressors be mistaken for future regressors.
            x_path.unlink()

        if y is not None:
            self._write_data(y, self.cache_dir / "y.parquet")

        if self.debug == "forecast":
            self.debug_repl("forecast", steps=steps)
        else:
            self._run(self._forecast_command(steps))

        forecasts = self._read_forecasts(forecast_path, steps)
        return self._wrap_forecast(
            forecasts.to_numpy(), steps, forecast_origin=kwargs.get("forecast_origin")
        )

    def _read_forecasts(self, path: Path, steps: int) -> pd.DataFrame:
        """Read and validate the standard forecast file from an external model."""
        if not path.exists():
            raise ValueError(f"External model did not produce forecast output at {path}.")
        try:
            forecasts = pd.read_parquet(path)
        except Exception as exc:
            raise ValueError(
                f"External forecast output at {path} is not a readable Parquet file."
            ) from exc
        forecasts = self._normalise_forecasts(forecasts)

        configuration = getattr(self, "_fitted_model_configuration", None)
        expected_columns = (
            list(configuration.y_columns)
            if configuration is not None
            else list(self.y.columns)
        )
        if len(forecasts) != steps:
            raise ValueError(
                f"External forecast output must have {steps} rows; got {len(forecasts)}."
            )
        if list(forecasts.columns) != expected_columns:
            raise ValueError(
                "External forecast output columns must match the fitted target "
                f"columns in order; expected {expected_columns}, "
                f"got {list(forecasts.columns)}."
            )
        for column in expected_columns:
            values = forecasts[column]
            if not pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(
                values
            ):
                raise TypeError(
                    f"External forecast column {column!r} must contain numeric values."
                )
            if values.isna().any():
                raise ValueError(
                    f"External forecast column {column!r} contains missing values."
                )
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(
                    f"External forecast column {column!r} contains non-finite values."
                )
        return forecasts

    def _normalise_forecasts(self, forecasts: pd.DataFrame) -> pd.DataFrame:
        """Apply a model-specific output naming adapter before validation."""
        return forecasts


# ======================================================================
# Concrete language classes
# ======================================================================


class RModel(ExternalModel):
    """Wrapper for a model implemented in an R script.

    The script is sourced and executed by R, so it must be trusted code.

    The user script only needs to define two functions::

        fit(y, X, params)                     # returns a model object (saved by runner)
        forecast(model, steps, X, y, params)  # returns a data.frame (saved by runner)

    ``X`` is a data.frame of regressors (``NULL`` when there are none). At
    forecast time it may contain the fitting history followed by future
    regressor values; the script selects the forecast horizon.
    The argument order mirrors ``ForecastModel._fit`` / ``_forecast``.

    CLI dispatch, parameter reading, data loading, model serialisation,
    and forecast output are all handled by the bundled ``runner.r``.

    Parameters
    ----------
    script : str
        Path to the ``.R`` file containing ``fit()`` and
        ``forecast()``.
    **params
        Written to ``params.parquet``.
    """

    _RUNNER = str(Path(__file__).parent / "runners" / "runner.r")

    def _fit_command(self) -> list[str]:
        return [
            "Rscript",
            self._RUNNER,
            str(Path(self.script).resolve()),
            "fit",
            str(self.cache_dir),
            self.params_path,
        ]

    def _forecast_command(self, steps: int) -> list[str]:
        return [
            "Rscript",
            self._RUNNER,
            str(Path(self.script).resolve()),
            "forecast",
            str(self.cache_dir),
            str(steps),
            self.params_path,
        ]

    def _debug_source(self, action: str, steps: int = 1) -> str:
        cache_dir_r = str(self.cache_dir).replace("\\", "/")
        params_path_r = self.params_path.replace("\\", "/")
        script_r = str(Path(self.script).resolve()).replace("\\", "/")
        fit_call = (
            "model <- fit(y, X, params); "
            "saveRDS(model, file.path(cache_dir, 'model.rds'))"
        )
        forecast_call = (
            f"model <- readRDS(file.path(cache_dir, 'model.rds')); "
            f"result <- forecast(model, {_steps_literal(steps)}L, X, y, params); "
            f"write_parquet(as.data.frame(result), "
            f'file.path(cache_dir, "forecasts.parquet"))'
        )
        call = fit_call if action == "fit" else forecast_call
        return (
            "library(arrow)\n"
            "read_params <- function(pp) {\n"
            "  if (file.exists(pp)) {\n"
            "    df <- read_parquet(pp)\n"
            "    setNames(as.list(df[1, ]), names(df))\n"
            "  } else list()\n"
            "}\n"
            f"cache_dir   <- {_r_string_literal(cache_dir_r)}\n"
            f"params_path <- {_r_string_literal(params_path_r)}\n"
            "params      <- read_params(params_path)\n"
            'y           <- read_parquet(file.path(cache_dir, "y.parquet"))\n'
            'x_path      <- file.path(cache_dir, "X.parquet")\n'
            "X           <- if (file.exists(x_path)) read_parquet(x_path) else NULL\n"
            f"source({_r_string_literal(script_r)})\n"
            'cat("\n  cache_dir =", cache_dir, "\n\n")\n'
            f"{call}\n"
        )

    def debug_repl(self, action: str = "fit", steps: int = 1) -> None:
        """Launch an interactive R session with cache_dir and params pre-set.

        Sources the runner's ``read_params`` helper and the user script,
        then calls ``fit()`` or ``forecast()`` automatically.
        Add ``browser()`` calls to the user script to set breakpoints.
        """
        init_path = self.cache_dir / "_debug_init.R"
        init_path.write_text(self._debug_source(action, steps))
        print(f"Launching R REPL ({action}) — add browser() to set breakpoints")
        import os

        env = os.environ.copy()
        env["R_PROFILE_USER"] = str(init_path)
        subprocess.run(["R", "--no-save", "--no-restore"], env=env)


class MATLABModel(ExternalModel):
    """Wrapper for a model implemented in a MATLAB function.

    The function file is executed by MATLAB, so it must be trusted code.

    The user function only needs to handle two actions::

        function result = my_model(action, y, X, params)
            % action is 'fit' — return a model struct
            % y is a table loaded from y.parquet; X is a table of
            % regressors (empty [] when there are none); params is a
            % struct from keyword arguments

        function result = my_model(action, model, steps, X, y, params)
            % action is 'forecast' — return a table; X holds the future
            % regressor values (one row per step)

    CLI dispatch, parameter reading, data loading, model serialisation,
    and forecast output are all handled by the bundled ``runner.m``.

    Parameters
    ----------
    script : str
        Path to the ``.m`` file.
    debug : str | None
        Optional debugging stage, either ``"fit"`` or ``"forecast"``.
    label : str | None
        Name used to identify the model's forecasts.
    formula : str | None
        Optional formula selecting the target and regressor variables.
    data_transformation : dict[str, str] | None
        Optional model-owned raw-input transformation configuration.
    subprocess_timeout : float | None
        Maximum number of seconds allowed for the MATLAB process.
    """

    _RUNNER_DIR = str(Path(__file__).parent / "runners")

    def __init__(
        self,
        script: str,
        *,
        debug: str | None = None,
        label: str | None = None,
        formula: str | None = None,
        data_transformation: dict[str, str] | None = None,
        subprocess_timeout: float | None = None,
        **params,
    ):
        super().__init__(
            script,
            debug=debug,
            label=label,
            formula=formula,
            data_transformation=data_transformation,
            subprocess_timeout=subprocess_timeout,
            **params,
        )
        self._script_dir = str(Path(script).resolve().parent)
        self._function_name = Path(script).stem
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self._function_name):
            raise ValueError(
                f"MATLAB script stem {self._function_name!r} is not a valid "
                "MATLAB identifier"
            )

    def _fit_command(self) -> list[str]:
        return [
            "matlab",
            "-batch",
            f"addpath({_matlab_string_literal(self._RUNNER_DIR)}); "
            f"addpath({_matlab_string_literal(self._script_dir)}); "
            f"runner({_matlab_string_literal(self._function_name)}, 'fit', "
            f"{_matlab_string_literal(str(self.cache_dir))}, "
            f"{_matlab_string_literal(self.params_path)})",
        ]

    def _forecast_command(self, steps: int) -> list[str]:
        return [
            "matlab",
            "-batch",
            f"addpath({_matlab_string_literal(self._RUNNER_DIR)}); "
            f"addpath({_matlab_string_literal(self._script_dir)}); "
            f"runner({_matlab_string_literal(self._function_name)}, 'forecast', "
            f"{_matlab_string_literal(str(self.cache_dir))}, "
            f"{_steps_literal(steps)}, {_matlab_string_literal(self.params_path)})",
        ]

    def _debug_source(self, action: str, steps: int = 1) -> str:
        function_name = _matlab_string_literal(self._function_name)
        runner_dir = _matlab_string_literal(self._RUNNER_DIR)
        script_dir = _matlab_string_literal(self._script_dir)
        cache_dir = _matlab_string_literal(str(self.cache_dir))
        params_path = _matlab_string_literal(self.params_path)
        if action == "fit":
            call = (
                "model = user_function('fit', y, X, params); "
                "save(fullfile(cache_dir, 'model.mat'), 'model')"
            )
        else:
            call = (
                "loaded = load(fullfile(cache_dir, 'model.mat'), 'model'); "
                f"result = user_function('forecast', loaded.model, "
                f"{_steps_literal(steps)}, X, y, params); "
                "parquetwrite(fullfile(cache_dir, 'forecasts.parquet'), result)"
            )
        return (
            f"addpath({runner_dir}); "
            f"addpath({script_dir}); "
            f"user_function = str2func({function_name}); "
            f"cache_dir = {cache_dir}; "
            f"params_path = {params_path}; "
            "if isfile(params_path), params = table2struct(parquetread(params_path)); "
            "else params = struct(); end; "
            "y = parquetread(fullfile(cache_dir, 'y.parquet')); "
            "x_path = fullfile(cache_dir, 'X.parquet'); "
            "if isfile(x_path), X = parquetread(x_path); else X = []; end; "
            "disp('  cache_dir = ' + string(cache_dir)); "
            f"{call}"
        )

    def debug_repl(self, action: str = "fit", steps: int = 1) -> None:
        """Launch an interactive MATLAB session with variables pre-set.

        Opens MATLAB desktop with ``cache_dir``, ``params``, and the
        script directory already on the path.  The fit or forecast function
        is called automatically — add breakpoints in the MATLAB editor.
        """
        matlab_cmd = self._debug_source(action, steps)
        print(f"Launching MATLAB ({action}) — set breakpoints in the MATLAB editor")
        subprocess.run(["matlab", "-r", matlab_cmd])


class JuliaModel(ExternalModel):
    """Wrapper for a model implemented in a Julia script.

    The script is included and executed by Julia, so it must be trusted code.

    The user script only needs to define two functions::

        function fit(y, X, params)
        # returns a model object (serialised by runner)
        function forecast(model, steps, X, y, params)
        # returns a DataFrame (saved by runner)

    ``X`` is a DataFrame of regressors (``nothing`` when there are none); at
    forecast time it holds the future regressor values (one row per step).

    CLI dispatch, parameter reading, data loading, model serialisation,
    and forecast output are all handled by the bundled ``runner.jl``.

    Parameters
    ----------
    script : str
        Path to the ``.jl`` file containing ``fit()`` and
        ``forecast()``.
    **params
        Written to ``params.parquet``.
    """

    _RUNNER = str(Path(__file__).parent / "runners" / "runner.jl")

    def _fit_command(self) -> list[str]:
        return [
            "julia",
            self._RUNNER,
            str(Path(self.script).resolve()),
            "fit",
            str(self.cache_dir),
            self.params_path,
        ]

    def _forecast_command(self, steps: int) -> list[str]:
        return [
            "julia",
            self._RUNNER,
            str(Path(self.script).resolve()),
            "forecast",
            str(self.cache_dir),
            str(steps),
            self.params_path,
        ]

    def _debug_source(self, action: str, steps: int = 1) -> str:
        """Launch an interactive Julia session with variables pre-set.

        Opens a Julia REPL that includes the user script and sets
        ``cache_dir`` and ``params``.  The fit or forecast function
        is called automatically — add ``@bp`` (Debugger.jl) or
        ``@infiltrate`` (Infiltrator.jl) to set breakpoints.
        """
        if action == "fit":
            call = (
                "model = fit(y, X, params); "
                "using Serialization; "
                'serialize(joinpath(cache_dir, "model.jls"), model)'
            )
        else:
            call = (
                "using Serialization; "
                'model = deserialize(joinpath(cache_dir, "model.jls")); '
                f"result = forecast(model, {_steps_literal(steps)}, X, y, params); "
                'Parquet2.writefile(joinpath(cache_dir, "forecasts.parquet"), '
                "DataFrame(result))"
            )
        init_code = (
            "using Parquet2, DataFrames; "
            f"cache_dir = {_julia_string_literal(str(self.cache_dir))}; "
            f"params_path = {_julia_string_literal(self.params_path)}; "
            "params = isfile(params_path) ? "
            "Dict(pairs(DataFrame(Parquet2.Dataset(params_path))[1, :])) : "
            "Dict{String,Any}(); "
            'y = DataFrame(Parquet2.Dataset(joinpath(cache_dir, "y.parquet"))); '
            'x_path = joinpath(cache_dir, "X.parquet"); '
            "X = isfile(x_path) ? DataFrame(Parquet2.Dataset(x_path)) : nothing; "
            f"include({_julia_string_literal(str(Path(self.script).resolve()))}); "
            f'println("\\n  cache_dir = ", cache_dir, "\\n"); '
            f"{call}"
        )
        return init_code

    def debug_repl(self, action: str = "fit", steps: int = 1) -> None:
        """Launch an interactive Julia session with variables pre-set.

        Opens a Julia REPL that includes the user script and sets
        ``cache_dir`` and ``params``. The fit or forecast function is called
        automatically; add ``@bp`` or ``@infiltrate`` to set breakpoints.
        """
        init_code = self._debug_source(action, steps)
        print(
            f"Launching Julia REPL ({action}) — add @bp / @infiltrate to set breakpoints"
        )
        subprocess.run(["julia", "-i", "-e", init_code])
