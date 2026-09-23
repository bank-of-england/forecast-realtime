# ForecastModel design refactor – implementation plan

## Goal

`ForecastModel` should express one idea once: *fit derives a frozen
specification; fit and forecast both apply it to prepared inputs*. Today the
design pipeline is written twice (`_build_fit_design` and the body of
`_predict_data`), fitted state is stored three times, and the transformation
mapping has three representations.

## Expected outcome

The line counts below are estimates made by reading the code, not measurements.
`forecast_model.py` is about 1,290 lines today.

| Step | Net lines (est.) | Clarity gain |
|---|---|---|
| 1. Dead hooks and repeated validation | −50 to −60 | Moderate: `ForecastResult` becomes the only place that checks forecast output |
| 2. One calendar | −15 to −25 | Moderate: one function answers "what are the forecast dates?" |
| 3. One mapping and one resolution rule | −15 to −25 | Moderate: removes the `configuration.data_transformation.data_transformation` stutter |
| 4. `DesignSpec` | −20 to −30 | High: fitted state lives in one place, and the freeze/thaw helpers disappear |
| 5. Shared design builder | −90 to −110 | Highest: `_predict_data` becomes a short readable sequence of steps |
| 6. Stop re-sending fitted options | about −20 (mostly in `RealTimeModel`) | Moderate: fewer arguments that exist only to be checked |

Overall, `forecast_model.py` should lose roughly 220–280 lines (about 20%), with
a few dozen more removed from `real_time_model.py` and `forecast_tree.py`.
Step 5's regression tests add roughly 40 lines back. Step 7 is not included in
these figures; retiring `DataTransformationPipeline` would remove about 250 more
lines.

Where the gain is smaller than it looks:

- Step 4 adds a concept (`DesignSpec`). Subclasses and tests still read
  `y_lags`, `X_lags`, `y_name` and `_dummy_cols`. Keeping them as read-only
  properties saves little, but removing them spreads changes into
  `linear_regression.py` and several test files. Its real benefit is making the
  design specification explicit, so the revision-attribution forecasts in
  `RealTimeModel` cannot depend on data held over from fit.
- Step 3 moves a rule into one place rather than removing much code.
- Churn is low: only 8 references in 6 test files touch
  `_fitted_model_configuration`.

Priority: steps 1, 4 and 5 give most of the value; steps 2, 3 and 6 are
worthwhile but optional.

## Method

Every step below is behaviour-preserving and lands as its own commit. Run the
full suite after each step:

```bash
conda activate forecast-realtime
pytest -n auto
```

## Constraints from callers

- `RealTimeModel` applies fitted models to *other* vintages' data and origins
  (revision attribution: `D(β_old, X_new)`, `D(β_new, X_old)`). Forecasting must
  depend only on the frozen configuration and the supplied `ModelData`, never on
  fit-time frames.
- `RealTimeModel` reads `y.columns`, `last_y_fit_date`, `_raw_data`,
  `native_metric_mapping()`, `_forecast_dates_include_origin` and `label`.
- `ForecastTree._fit_data_impl` builds a `FittedModelConfiguration` by hand; it
  must be updated alongside any change to that class.
- Tests override public `fit()` (`test_model_data_boundary_hooks.py`), so the
  override detection in `_fit_from_data` is a contract and stays.
- `_prepare_estimation_inputs` may remove X entirely (`midas_bvar_tree.py`) and
  is not applied at forecast time; design columns must be captured *before* it.
- Column order of the forecast design must equal the fitted order.

## Step 1 – remove dead hooks and duplicated validation

- Delete `BridgeOLS._validate_fit_inputs` (it only calls the base) and the
  override-detection branch in `ForecastModel._fit_data`; keep
  `_select_fit_data` as the single path. Delete the base
  `_validate_fit_inputs` if no other caller remains.
- Make `ForecastResult` the sole owner of calendar, `steps` and
  quantiles-with-`decomp` checks:
  - drop the index checks (NaN, duplicate, monotonic) from
    `_normalise_point_forecast`;
  - drop the quantile/`decomp` check from `_finalise_forecast`;
  - keep one early `steps` check in `_predict_data` (needed before date
    inference) and remove the one in `_validate_explicit_target_path`, or vice
    versa.
- Remove the `configuration is None` fallbacks in `_wrap_forecast` and
  `_normalise_point_forecast`; both run only after fitting.
- Remove the `_prepared_y_history` / `_y_history` / `self.y` fallback chains in
  `_predict_data`; `y_input` is always present.

Commit: `refactor: remove redundant forecast validation and dead fit hooks`

## Step 2 – one forecast calendar

- Merge `_forecast_calendar` and `_conditioning_dates` into
  `_forecast_dates(origin, steps)`, driven by the fitted frequency and
  `_forecast_dates_include_origin`. Use it from `_wrap_forecast`,
  `_finalise_forecast`, `_validate_explicit_target_path` and the dummy index.
  `ForecastTree._conditioning_dates` becomes an override of the merged method.
- Add a small `_as_origin(date) -> pd.Timestamp` for the Period→Timestamp
  conversion used by `ForecastModel` and `ForecastTree`.
- Dummies at forecast time use the calendar from `forecast_origin`, not the
  last date of the supplied history.

Commit: `refactor: derive forecast dates from a single calendar`

## Step 3 – single transformation mapping and resolution rule

- Add `FittedDataTransformation.mapping -> dict[str, str] | None` and replace
  every `dict(x.data_transformation) if ... else None`.
- Add one resolver, e.g. `ForecastModel._resolve_mapping(fallback) ->
  (mapping, source)`, returning `"model" | "fallback" | "identity"`. Use it in
  `_fit_data_impl`, `resolve_input_data_transformation`,
  `_resolve_fit_transformation` and
  `ForecastTree._resolve_child_data_transformation`.
- Inside `ForecastModel`, stop constructing `DataTransformationPipeline` merely
  to read `.data_transformation`; validate the dict directly with
  `_validate_metric_mapping` / `_validate_mapping_coverage`. The pipeline class
  itself remains for its public/test users (see step 7).

Commit: `refactor: resolve data transformation mapping in one place`

## Step 4 – introduce `DesignSpec`

```python
@dataclass(frozen=True)
class DesignSpec:
    y_lags: int
    X_lags: tuple[tuple[str, int], ...]  # per column, normalised at fit
    dummies: tuple[tuple[str, pd.Timestamp], ...]  # every named definition
    columns: tuple[str, ...] | None  # design columns before the estimation hook
    frequency: str | None
```

- Fit normalises `X_lags` to per-column pairs and dummies to named pairs
  (`_dummy_spec` already does this), so freezing is a plain `tuple(...items())`
  and thawing is `dict(...)`. Delete `_freeze_option` and `_thaw_option`; this
  also removes the list-of-pairs ambiguity.
- Record `columns` after the formula and zero-dummy pruning, before
  `_prepare_estimation_inputs`.
- Restructure `FittedModelConfiguration` as:
  - `inputs: FittedDataTransformation` (rename to `FittedInputs` optional; move
    `drop_transformation_nans` into it);
  - `design: DesignSpec`;
  - `y_columns`, `X_columns`, `forecast_origin`.
- Delete `_FitDesignState`; make `_build_fit_design` return
  `(y_estimation, X_estimation, design_history, spec)` without writing to
  `self`.
- Collapse duplicated attributes: `y_estimation`/`X_estimation` (keep `y`/`X`),
  `_y_history`/`_prepared_y_history`. Keep `y_lags`, `X_lags`, `y_name`,
  `last_y_fit_date` and `_dummy_cols` as read-only properties over the
  configuration where subclasses/tests read them (`linear_regression.py`,
  `test_ar_penalisation.py`, `test_ols.py`), and drop `dummies`, `X_names`,
  `_dummy_definitions`, `_forecast_frequency`, `_input_frequencies` if unused
  (check `examples/midas_bvar_tree.py` and `tests/test_conditioning_realtime.py`
  for `_forecast_frequency`).
- Update `ForecastTree._fit_data_impl` to pass an empty `DesignSpec`.

Commit: `refactor: capture fitted design in a frozen DesignSpec`

## Step 5 – one design builder for fit and forecast

- Add `DesignSpec.build(y, X, index) -> pd.DataFrame | None` (or a module
  function) performing lags, dummies over `index`, formula selection and
  `design[list(columns)]`. The formula still runs over *all* dummy definitions,
  so a formula naming a dummy that fit dropped keeps working; the final column
  selection replaces the current keep/drop logic.
- Fit: derive the spec (including pruning to find `columns`), then call
  `spec.build`. Forecast: `configuration.design.build(...)` with the index
  extended by `_forecast_dates(forecast_origin, steps)` when there is no X.
- `_predict_data` shrinks to: validate → subset/transform/impute/regularise →
  `_prepare_forecast_inputs` → `build` → NaN handling → `_forecast` →
  `_finalise_forecast`.
- Add a regression test: formula naming an all-zero dummy, and a counterfactual
  forecast from a later origin (mirrors `RealTimeModel` revision attribution).

Commit: `refactor: share design construction between fit and forecast`

## Step 6 – stop re-sending fitted options to `_predict_data`

- Remove `data_transformation`, `frequency` and `X_imputation` from
  `_predict_data`; read them from the configuration.
- Update `_loop_through_vintages`, `_level_contributions`,
  `_compute_revision_decompositions` and `ForecastTree._predict_data`.
- Pass `configuration.forecast_origin` instead of `last_y_fit_date` from
  `RealTimeModel`.
- Public `forecast()` keeps the arguments and the conflict check for now;
  consider deprecating them separately (would be a `feat!`/breaking change).

Commit: `refactor: read fitted preprocessing options from configuration`

## Step 7 – optional: `DataTransformationPipeline`

Within `src`, the class is only a validated dict; its methods are used by tests
alone. Decide whether to keep it as a public façade over `ModelData`, or retire
it and migrate the tests to `ModelData`. Separate decision; not required for
steps 1–6.

## Out of scope

- The double deep copy (`RealTimeModel` per vintage, then `_fit_data`
  candidate): harmless, not worth restructuring.
- Edge cases listed in `/memories/repo/forecast_model_review.md` other than the
  dummy-origin behaviour addressed in step 2.
