# Forecast validation simplification

## Design

`ForecastResult.__init__` owns forecast validation. `ForecastModel` computes
forecasts and supplies the expectations needed to check them. There is no
factory, additional container or override-handling machinery.

```text
prepare inputs -> run model -> convert output -> ForecastResult(...)
                                                validates once
```

Validation guarantees correctness at construction, not after arbitrary user
mutation.

## Implementation plan

### 1. Simplify pandas behaviour

Remove `ForecastResult._constructor`. Slices and copies become ordinary pandas
DataFrames. Keep `.forecast`, `.forecast_origin` and `.decomposition` on the
original result.

### 2. Move validation into the constructor

Give the constructor explicit keyword arguments for expected targets, steps,
origin, requested quantiles and calendar rules. Move point, quantile and
decomposition checks into private methods on `ForecastResult`.

Validate and order ordinary DataFrames before initialising the result. Provide
no validation bypass and no `from_forecast()` factory.

### 3. Thin out ForecastModel

Keep model execution, array/wide-to-long conversion and forecast-date generation
in `ForecastModel`. Replace `_validate_result()` with direct
`ForecastResult(...)` construction.

Remove duplicate output checks, while preserving checks needed to prevent
malformed hook output being silently repaired during conversion. Input
validation and unsupported-request checks remain in the model.

### 4. Remove override dispatch

Delete `_validate_public_result()` and the override-aware `_predict_from_data()`
and `_forecast_from_data()` wrappers. Route direct forecasts, explicit contexts,
realtime runs and tree components through `_predict_data()`, ending in one
result construction.

Remove context-conversion helpers that become unused. Keep `_fit()`,
`_forecast()` and `_forecast_decomp()` as model extension points. Replacing
`forecast()` orchestration is not a supported model
extension point and has no useful role in this library. Remove it from the
contract rather than adding compatibility machinery for it.

### 5. Keep useful guarantees and replace obsolete tests

Test constructor validation directly:

- Complete, unique forecast keys and fitted-target coverage.
- Correct date counts, origin relationships and deterministic row ordering.
- Custom point calendars and permitted point `NaN` values.
- Exact requested calendar and probability coverage for quantile forecasts.
- Finite, non-crossing quantile values.
- Valid decomposition keys, values and coverage, with contributions reconciling
  to the forecast for every horizon and target.
- Rejection of decompositions for quantile forecasts.

Remove tests demanding arbitrary result payloads, metadata propagation through
slices or support for orchestration overrides. Retain integration coverage for
direct, explicit-context, realtime and tree forecasts, including rejection of
invalid output before publication.

### 6. Update documentation and verify

Document constructor-owned validation and ordinary DataFrame slicing. Remove
instructions promising orchestration override support.

Run focused tests first, then the full suite with `pytest -n auto` in the
`forecast-realtime` conda environment. Run the repository quality checks,
including lint, formatting, docstrings and documentation checks.

## Acceptance criteria

- Forecast output validation has one owner: `ForecastResult`.
- Each normal forecast execution constructs and validates its result once.
- No result factory, validation bypass or post-override validation path remains;
  public `forecast()` replacements are outside the
  supported contract.
- Forecast values, supported calendars and realtime storage contracts remain
  unchanged.
- Slices and copies return ordinary DataFrames without result metadata.
- Tests describe useful package behaviour rather than discarded implementation
  complexity.

The simplification comes from deleting alternative dispatch and validation
paths, not merely moving existing functions into a class.

## Suggested commit

```text
refactor!: centralise forecast validation in ForecastResult
```