# Input Metric Transformation Plan

## Problem

`RealTimeModel` receives long-form `ForecastData` containing a `metric` column,
but `ForecastModel.fit()` receives wide data containing only dates, variables,
and values. The metric describing those values is lost during the pivot.

The model transformation pipeline currently assumes that every wide input is in
levels. This is correct when `ForecastData` contains levels and the requested
model metric is derived, but incorrect when `ForecastData` has already been
filtered to that derived metric.

For example, with `data_transformation={"gdp": "pop"}`:

- levels input should be transformed with `pct_change() * 100`;
- pop input should pass through unchanged;
- another derived input metric should fail unless a supported conversion exists.

Without an explicit input metric, pop data can be transformed twice. Without
metric filtering before the long-to-wide pivot, multiple rows can also exist for
the same `(date, variable)` observation.

## Design Goals

1. Preserve each variable's input metric when converting long-form data to wide
   model inputs.
2. Keep observation selection in `RealTimeModel`.
3. Keep metric conversion in the transformation pipeline.
4. Support `ForecastData` filtered to either levels or the requested metric.
5. Reject ambiguous or unsupported conversions instead of silently selecting a
   metric.
6. Apply the same contract to target history, regressors, and conditioning
   forecasts.
7. Preserve model-owned transformation pipelines and parallel execution.

## Responsibility Split

### `RealTimeModel`

`RealTimeModel` should:

- select exactly one input metric for each requested variable before pivoting;
- prefer the requested model metric when it is available;
- otherwise select levels when the requested metric can be derived from levels;
- reject multiple remaining metrics or unavailable transformations;
- retain a `dict[str, str]` mapping from variable to selected input metric;
- pass the selected metric mapping with the wide values to model fitting and
  prediction.

It should not calculate growth rates, differences, logs, or reconstructed
levels.

### Transformation pipeline

The transformation pipeline should:

- receive both the selected input metric and requested model metric;
- use an identity operation when both metrics are equal;
- derive supported metrics from levels;
- raise a clear error for unsupported metric-to-metric conversions;
- use the same decision for fit history and forecast conditioning paths.

### `ForecastModel`

`ForecastModel` should:

- persist the resolved input metric mapping in its fitted transformation
  configuration;
- transform fit and forecast inputs consistently;
- continue exposing model inputs in the requested model metric;
- keep output metric resolution independent from input conversion.

## Metric Selection Rules

For each variable, let `requested_metric` come from the resolved model input
pipeline and let `available_metrics` come from the selected `ForecastData`.

1. If `requested_metric` is available, select it.
2. Otherwise, if levels are available and the requested metric is derivable from
   levels, select levels.
3. Otherwise, if no transformation pipeline is configured:
   - select the only available metric;
   - reject the input if more than one metric is available.
4. Otherwise, raise an error listing the variable, requested metric, and
   available metrics.

The selection must be deterministic and independent of row order.

## Transformation Rules

The initial conversion matrix should be deliberately small:

| Input metric | Requested metric | Operation |
| --- | --- | --- |
| Any metric | Same metric | Identity |
| Levels | Logs | Natural logarithm |
| Levels | Diff | First difference |
| Levels | Log diff | First difference of logs |
| Levels | Pop | One-period percentage growth |
| Levels | YoY | Frequency-specific annual percentage growth |

All other conversions should raise `ValueError` until explicitly supported.
Derived-to-derived conversion should not be inferred.

Monthly and quarterly transformations must use each variable's own frequency:

- monthly pop compares adjacent months;
- quarterly pop compares adjacent quarters;
- monthly YoY compares 12 months;
- quarterly YoY compares four quarters.

Calendar alignment must continue to treat a missing period as a missing base,
not as an adjacent observation.

## Proposed Data Flow

```text
ForecastData long-form rows
    -> resolve model input transformation
    -> select one input metric per variable
    -> retain input_metrics mapping
    -> select latest observation at each vintage
    -> pivot values to wide y/X frames
    -> ForecastModel.fit(values, input_metrics=...)
    -> compare input metric with requested metric
    -> identity or supported conversion
    -> estimate model in requested metric space
```

Conditioning forecasts should follow the same selection and conversion path so
that history and future values share one metric space.

## API Changes

Add optional input metric mappings to the internal model boundary:

```python
ForecastModel.fit(
    y,
    X=None,
    y_input_metrics=None,
    X_input_metrics=None,
    ...,
)
```

The prediction context should retain equivalent metadata for conditioning
inputs. Prefer storing the mappings in `FittedDataTransformation` so fit and
prediction cannot resolve different contracts accidentally.

These parameters should remain internal implementation details initially. A
public API should only be added if callers outside `RealTimeModel` need to pass
pre-transformed wide frames explicitly.

## Implementation Steps

1. Extract deterministic metric selection from `RealTimeModel.forecast()` into
   a focused helper.
2. Resolve each model's effective transformation pipeline before selecting
   metrics, including model-owned pipeline precedence.
3. Select input metrics separately for each model because models may request
   different transformations for the same variable.
4. Include selected metric mappings in `ForecastTask` so sequential and
   parallel execution use the same metadata.
5. Preserve metric identity while selecting latest vintage rows and pivoting
   target, regressor, and conditioning data.
6. Extend `FittedDataTransformation` with immutable y/X input metric mappings.
7. Update wide fit and forecast transformations to perform identity or
   levels-derived conversion according to the conversion matrix.
8. Remove the current ad hoc `filter_metrics()` fallback once all call paths use
   the explicit contract.
9. Update documentation describing `data_transformation` as the requested model
   metric, not necessarily an instruction to transform the supplied values.

## Test Plan

### Metric selection

- Select requested pop when both levels and pop exist.
- Select levels when pop is requested but only levels exist.
- Select levels when levels are requested and both metrics exist.
- Produce the same result regardless of row order.
- Reject ambiguous inputs when no requested metric is defined.
- Reject an unavailable requested metric with a useful error.

### Transformation behaviour

- Pop input requested as pop is unchanged.
- Levels input requested as pop receives one growth transformation.
- Levels input requested as levels is unchanged.
- Pop input requested as levels raises an unsupported-conversion error.
- Monthly and quarterly pop use one observation at their respective calendar
  frequencies.
- Monthly and quarterly YoY use 12 and four periods respectively.
- Missing calendar periods do not become transformation bases.

### Real-time integration

Run `RealTimeModel.forecast()` with otherwise identical data after:

```python
forecast_data.filter(metrics=["pop"])
```

and:

```python
forecast_data.filter(metrics=["levels"])
```

Both runs should estimate successfully with
`data_transformation={"variable": "pop"}` and expose equivalent model input
values for dates available in both data sets.

Also cover:

- mixed y and X input metrics;
- conditioning forecasts already in the requested metric;
- levels conditioning forecasts requiring transformation;
- model-owned pipelines overriding the call-level mapping;
- multiple models requesting different metrics in one realtime run;
- sequential and parallel equivalence;
- level reconstruction from native model forecasts.

## Migration and Compatibility

The change should preserve levels-only workflows. Existing direct calls to
`ForecastModel.fit()` should default their input metrics to levels unless the
caller explicitly supplies another mapping.

Pop-only realtime data will become a supported, explicit identity case rather
than an accidental fallback. Ambiguous multi-metric inputs that previously
selected a metric by row order should fail clearly.

## Acceptance Criteria

- No wide model input is transformed without a known input metric.
- A variable has exactly one selected metric before every long-to-wide pivot.
- Pop-filtered data is not transformed twice.
- Levels-filtered data can be transformed to pop.
- Monthly and quarterly growth calculations are calendar-correct.
- Model-owned and call-level transformation precedence remains unchanged.
- Sequential and parallel forecasts remain equivalent.
- Ruff and the complete `pytest -n auto` suite pass.
