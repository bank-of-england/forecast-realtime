# Model-owned conditioning: design proposal

**Status: implemented; awaiting release.** Individual model specifications own
conditioning settings while sharing one `RealTimeModel.forecast()` run. The
design brief below records the agreed contract; usage and migration guidance
are in [the forecasting guide](docs/forecasting_strategy.md#model-owned-conditioning).

Implement one immutable policy, one resolver and the existing forecasting
pipeline. Do not introduce a conditioning engine, a parallel task representation
or compatibility wrappers that preserve an obsolete path.

## Goal

Compare models that use different conditioning sources and durations against the
same source data, vintage range and forecast horizon. For example:

| Specification | Target conditioning | Forecast horizon |
| --- | --- | --- |
| BVAR baseline | No supplied target path | Six periods |
| BVAR nowcast | GDP from source A for the first period | Six periods |
| BVAR extended | GDP from source B for the first three periods | Six periods |

Each specification has a distinct model label. Results keep the existing source
labels and aggregation format, so these runs remain distinguishable.

## Current behaviour

`RealTimeModel.forecast()` accepts `y_steps_ahead`, `y_sources`,
`X_steps_ahead` and `X_sources`. These settings are shared across models in that
call. Source selection already happens separately for each model, using its
input requirements, but each selection receives the same source mappings.
Conditioning horizons are currently included in the shared task options.

The direct `model.forecast(y=..., X=...)` interface accepts raw, date-indexed
paths. The framework combines them with fitted history and transforms them
before invoking the model hook. It does not select sources.

Recursive linear and tree regressions currently accept future y constraints
without enforcing them: their recursive loops replace y lags with their own
predictions. The implementation rejects unsupported y conditioning rather than
adding constrained forecasting to those models. Historical y lags and supplied
future X paths remain supported.

## Ownership

| Component | Responsibility |
| --- | --- |
| `ForecastModel` | Own the optional conditioning policy and declare whether it supports target conditioning. |
| `RealTimeModel` | Resolve model policy against the run fallback, validate it and schedule tasks. |
| `ForecastTask` | Carry the selected data and resolved model-specific horizon options. |
| `ModelData` | Select sources and vintages, construct future paths and preserve their provenance and metrics. |
| `ForecastContext` | Carry authoritative raw frames across public overrides without merging explicit constraints with published observations. |
| Model forecasting hook | Consume prepared paths according to its declared capabilities. |

Keep policy separate from observations. Do not put source-selection preferences
on `ModelData`, store future observations on the model configuration, or build a
second DataFrame-based conditioning pipeline.

## Proposed configuration

Add an optional `conditioning` constructor argument to model specifications.
Normalise it into a small immutable, pickleable configuration. It has separate
y and X entries, keyed by variable. Each entry pairs a source label with a
duration, rather than keeping sources and durations in independent mappings.

Supported constructor configuration shape:

```text
conditioning = {
    "y": {"gdp": {"source": "nowcast_a", "periods": 1}},
    "X": {"oil": {"source": "futures_a", "periods": 3}},
}
```

Copy caller-supplied mappings when constructing the configuration. A frozen
record containing mutable dictionaries would not be sufficient. Resolve and
capture the policy before dispatch so later mutation cannot change scheduled
work, including serialised workers and revision counterfactuals.

Use two private frozen dataclasses in
[forecast_model.py](src/forecast_realtime/forecast_model.py):

- `_ConditioningEntry(variable: str, source: str, periods: int)`.
- `_ConditioningPolicy(y: tuple[_ConditioningEntry, ...],
  X: tuple[_ConditioningEntry, ...])`, with empty tuples as defaults.

The public constructor accepts ordinary mappings; users need not import these
records. Reject malformed mappings, unknown keys, invalid source types and
non-positive or boolean durations during construction. Validate selected
variables, sources and the duration limit against the run before
dispatch. Include the model and, where applicable, the variable in errors.

Do not add this policy to `FittedModelConfiguration`: source selection is not
fitted preprocessing. Direct forecasts remain independent of the conditioning
policy.

### Precedence

| Model setting | Effective policy |
| --- | --- |
| `None` or omitted | Inherit the run-level fallback. |
| Non-empty configuration | Replace the whole fallback. |
| Empty configuration | Disable externally supplied conditioning for both roles. |

Do not merge entries or roles implicitly. If a model supplies only X settings,
it does not inherit y settings from the fallback. This avoids accidentally
pairing one specification's source with another specification's duration.

Disabling external conditioning must not remove available published observations
or disable ordinary future X availability and the fitted X imputation policy.

### Model-owned settings versus inherited fallback

Validate model-owned entries strictly against that model's input requirements.
For trees, apply the root-target and raw-X rules below, not the union of every
consumer's inputs.

Validate legacy run-level mappings against the run-wide input selection, then
project them onto each inheriting model's requirements before checking target
conditioning capability. For example, a fallback containing GDP and inflation
settings must still serve separate GDP-only and inflation-only models. An entry
for another model is not an error on the inheriting model.

Retain run-level structural validation even when a model replaces the fallback.
Check source availability for effective entries, without selecting
streams from a replaced fallback.

### Horizon semantics

For the proposed configuration, `periods=N` means the first N forecast periods:
one means only the first period; three means the first three. Require a positive
integer no greater than the run's `steps`; reject booleans. Omit an entry to
leave that variable unconstrained.

Preserve the existing run-level API's inclusive, zero-based convention:
`y_steps_ahead={"gdp": 0}` selects one period, and a value of 2 selects three.
Convert model-owned `periods` to `periods - 1` once in the resolver; retain legacy
horizons unchanged. Do not force legacy arguments through the positive-duration
configuration schema. Do not silently reinterpret existing arguments or truncate
an overlong duration.

| Input | Resolved behaviour |
| --- | --- |
| Model `conditioning=None` | Inherit the projected legacy fallback. |
| Model `conditioning={}` or a role omitted from a replacement policy | Select no external source for that role; use `None` horizons to retain ordinary published/future observations. |
| Model entry with `periods=N` | Select its source and pass horizon `N - 1` to `ModelData.condition()`. |
| Legacy role mappings omitted (`None`) | Preserve the existing no-external-source behaviour, including ordinary future X availability. |
| Legacy empty horizon/source mappings (`{}`) | Preserve explicit empty mappings, including their existing X-path behaviour. |
| Legacy entry with horizon `None` | Preserve its source-selection and path semantics, but impose no explicit target constraint and do not trigger unsupported-y rejection for this entry. |
| Legacy entry with horizon `0` or greater | Retain the inclusive, zero-based duration. |

Preserve `None` versus an empty mapping after projecting the fallback onto a
model. Do not use truthiness to collapse these states. In particular, an explicit
empty X horizon mapping is not equivalent to an omitted one in the current
`ModelData.condition()` implementation. Test resulting paths and forecasts, not
just the resolver's return values.

Durations use the run's forecast-step calendar, not arbitrary row positions in
the source data. Preserve existing mixed-frequency X calendar expansion.
Source selection must respect each vintage's information set; a later vintage's
forecast must never fill an earlier vintage's path.

## Execution path

1. Build the shared source data once with `ModelData.from_archive()`.
2. In the existing per-model loop, resolve input requirements, transformation
  settings and effective conditioning using one resolver with separate
  model-owned and inherited-fallback branches.
3. Validate the effective policy against that model and the requested run,
  including target capability and tree routing.
4. Select paths using that model's resolved y and X source mappings.
5. Build its `ForecastTask` with the selected data and its own conditioning
   horizons. Keep vintage range, output horizon and execution controls shared.
6. Within the existing vintage loop, select the as-of data and fit the model.
7. Call `ModelData.condition()` with the model-specific horizons. Attach the
   resulting paths to fitted raw history through `with_conditioning()`.
8. Reuse the existing transformation, regularisation, imputation, lag
   construction, prediction and result aggregation path.

The main change is moving policy resolution into the existing per-model loop.
`ModelData.select()` already selects sources; `ModelData.condition()` already
limits paths by horizon. Reuse these operations rather than duplicating them.

The resolver returns the existing y/X source and horizon mappings. Use source
mappings for source selection and copy horizon mappings into
`ForecastTask.options`. Remove conditioning horizons from the shared options;
combine shared run controls with resolved horizons when constructing each task.
Do not persist a separate `ResolvedConditioning` record alongside those mappings.
The resolver handles policy precedence; workers process only resolved mappings.

Keep the vintage loop unchanged wherever possible. Revision counterfactuals
reuse the retained, date-specific paths in their vintage states, including their
source and metric metadata. Do not resolve policy again against a counterfactual
model's fitting origin or a different vintage. The resolved policy is constant
within one model's run; current and previous fitted states are not competing
model specifications.

## Validation and capability checks

- Validate variable names using the model-owned versus inherited-fallback rules
  above. Reject invalid source labels and duration types with errors naming the
  model and variable. Distinguish an invalid source label from a valid source
  with no path for a variable or vintage; preserve the latter's existing
  missing-path behaviour.
- Add `_supports_target_conditioning = False` to `ForecastModel`. Opt supporting
  models such as `ForecastBVAR` in; do not infer support from accepting a `y` argument.
  Custom and external model authors must declare support deliberately.
- Reject an effective y-conditioning request for an unsupported model before
  dispatch, even if the requested source has no observations at one vintage.
- Also validate direct `forecast()` and `predict()` requests at the shared
  prediction boundary. Unsupported models must reject non-missing explicit y
  constraints within the requested future horizon. Historical rows, empty
  frames and all-NaN future paths do not impose constraints. Use the model's
  forecast-date convention, including whether the origin is an output date,
  rather than an unconditional `index > origin` test.
- Inspect explicit conditioning provenance before combining paths with history.
  Do not mistake separately retained published observations for user-supplied
  constraints, or let transformation-induced NaNs hide an unsupported request.
- Preserve current source availability and missing-path behaviour initially;
  never substitute a different source silently. A strict completeness policy
  would be a separate feature.

Source settings apply to realtime orchestration. Direct model calls
continue to use the explicitly supplied frames; they do not look up sources or
silently clip those frames using a conditioning policy stored on the model.

Use one shared explicit-path validator at the relevant entry points. Internal
dispatch must validate before adapting to public `predict()` or `forecast()`
overrides; direct calls through base `predict()` must also validate before
preparation. Preserve supported public overrides and their delegation to base
methods. Do not duplicate validation logic or bypass it with an
"already validated" flag.

### Provenance across public overrides

Extend `ForecastContext` with a separate published-target frame and its metric
metadata where required. Append optional fields so existing positional arguments
keep their meaning. Keep `y_conditioning` exclusively for explicit constraints.
Context frames remain authoritative; do not add a hidden `ModelData` field,
duplicate observation store or mutable validation state.

Update `ForecastContext._from_data()` and `ModelData.from_context()` so the round
trip preserves explicit versus published paths and their units. Remove the
lossy context-boundary merge, including obsolete helpers if no callers remain.
The existing preparation pipeline combines paths after validation. Ordinary
forecast and decomposition hooks continue receiving prepared, combined inputs.

This deliberately changes the context contract: public overrides that previously
read published observations from `context.y_conditioning` must use the separate
published-target field. Document the migration and update affected overrides;
do not conceal it with a second compatibility path. Raw explicit constraints
must remain inspectable even when a later transformation produces NaNs.

## Forecast trees

Initially support a policy on the whole `ForecastTree`, with **root-only external
target conditioning**. Use the same routing rule for explicitly supplied y paths
in direct tree calls.

- The root transform must be a `ForecastModel` that supports target conditioning.
  A callable root or a supporting child alone does not confer support.
- Validate active y entries against the root model's target requirements.
- Leaves and intermediate nodes receive published observations but not the
  tree's explicit y constraints. Preserve all other required input paths.
- Attach the explicit y constraints only when forecasting the root, alongside
  its existing child-produced X inputs and published values.
- Reject every non-`None` conditioning policy on a contained model, including leaves,
  intermediate transforms and the root transform. An explicit empty policy is
  still a contained policy. Configure the policy on the `ForecastTree` only.
- Tree-level X policy selects raw source inputs requested by consumers; it does
  not select or replace component-generated regressors.

For example, a tree with regression leaves and a BVAR root may condition GDP at
the root without sending GDP constraints to the unsupported leaves. A regression
root with a BVAR leaf must reject the same request.

Arbitrary consumer routing, nested-tree target routing and independent contained
policies are deferred. Reject unsupported target-routing shapes explicitly. Do
not add a routing registry or choose consumers by scanning for the first
supporting child.

The existing `ConditionalBVAR` example's `conditioning_steps` controls paths
constructed from child nowcasts. It is not a source policy and should
not be silently reinterpreted as one. Its existing combination and clipping
behaviour must not silently shorten an explicit target constraint. Cover the
interaction in tests; reject a conflicting combination explicitly if supporting
it would require changing the example's numerical contract. Leave child-only
conditioning behaviour unchanged.

## Implementation locations

- [forecast_model.py](src/forecast_realtime/forecast_model.py): model policy,
  private frozen records, target-conditioning capability, shared path validation
  and provenance-preserving `ForecastContext` adapters.
- [real_time_model.py](src/forecast_realtime/real_time_model.py): fallback
  resolution, per-model selection and task-specific horizons.
- [_realtime_forecasting.py](src/forecast_realtime/_realtime_forecasting.py):
  reuse `ForecastTask.options`; no additional task or resolved-policy container.
- [_model_data.py](src/forecast_realtime/_model_data.py): reuse source,
  vintage and horizon selection; retain explicit conditioning provenance.
- Built-in model constructors and external wrappers: forward the new argument
  explicitly, without leaking it into estimator or external-script parameters.
- [forecast_tree.py](src/forecast_realtime/forecast_tree.py): define tree
  capability from the root, route explicit targets only to that root and reject
  policies on contained models.

## Implementation sequence

1. Add policy parsing and the resolver. Test legacy projection and task-specific
  source/horizon selection before changing prediction behaviour.
2. Preserve context provenance and add capability validation at dispatch and
  direct prediction boundaries. Update affected public overrides.
3. Forward constructor arguments explicitly and implement root-only tree routing.
4. Complete integration tests, migration documentation and generated API updates.

Keep architectural decisions in this brief; do not broaden scope to constrained
regression, a general tree-routing API or preprocessing refactors. Delete replaced
paths rather than leave two implementations. Use the `forecast-realtime` conda
environment and run tests with `-n auto`. Do not commit or push automatically.

## Acceptance tests

1. Two models conditioning the same variable from different sources in one run
   match equivalent separate runs.
2. Different durations produce the expected constrained and unconstrained
   horizons, including a baseline with explicit empty conditioning.
3. Test inherited, replaced and disabled policies, with no cross-model leakage.
  Include different formulas inheriting disjoint parts of one run-wide fallback,
  invalid model-owned variables, and strict rejection of malformed configuration.
  Cover omitted, empty and individual-`None` legacy mappings, including ordinary
  future X availability and fitted imputation under an empty model policy.
4. Test y and X independently and together on supporting models, including X
   lags, transformed/native metrics, sparse dates and mixed-frequency calendars.
5. Test vintage cutoffs and source provenance to prevent look-ahead leakage.
6. Unsupported regression models reject explicit y constraints regardless of
   lag settings or direct/recursive strategy; ordinary lagged and X-conditioned
   forecasts still work.
7. Direct calls distinguish explicit constraints from history, published values,
  empty paths and all-NaN paths. Include constraints outside the requested dates,
  origin-inclusive models and transformation-induced NaNs. Public `forecast()`
  and `predict()` overrides that delegate to base methods preserve this contract
  and path units across a context round trip.
8. Serial and process-parallel runs agree; caller mutation does not alter a
   captured task. Sequential decomposition and revision counterfactuals retain
   the resolved policy and reconcile as before.
9. Cover supporting roots with unsupported leaves, unsupported roots with
   supporting children, multilevel root-only routing, raw-X versus component-X
   inputs, and explicit rejection of contained policies (including empty ones).
   Cover direct tree paths and `ConditionalBVAR.conditioning_steps` interactions.
10. Existing calls without model-owned settings retain their numerical results,
  except for the deliberate unsupported-y rejection and context-contract
  migration. Update public overrides before assessing numerical parity.

## Scope and release notes

This adds model configuration and deliberately changes unsupported-y validation
and the public context's published-target representation. It does not implement
constrained regression, change existing numerical forecast algorithms, alter
legacy horizon meanings, or introduce per-model vintage ranges and output
horizons. Initial tree-level target support is restricted to a supporting root.

Document the unsupported-y rejection, constructor arguments, fallback projection,
root-only tree contract and context-field migration. Update generated API
documentation and coordinate the ecosystem's API/skill manifest with the release.
Identify any external manifest update that cannot be made in this repository in
the implementation handoff. Do not restore removed internal exports to satisfy
a snapshot of an older public API.

## Implementation handoff

The implementation uses the existing source selection, vintage loop and
task options. Context adapters now retain published targets separately from
explicit constraints. Built-in constructors forward the policy, BVAR opts
into target conditioning, and trees route explicit targets only to a supporting
root. BVAR paths are reindexed to the requested forecast calendar before they
reach its fixed-horizon constraint matrix; sparse or overlong direct paths no
longer cause a shape mismatch.

Acceptance coverage includes separate-versus-combined model runs, real BVAR
constraints, source/duration precedence, raw-path validation, mixed-frequency
calendars, source vintages, process execution, revision decomposition and
root-only tree routing. Existing conditioning test doubles and public context
overrides have been migrated to the explicit capability and provenance contract.

The generated API manifest still derives from unchanged public exports; the
documentation build renders the new constructor arguments and context fields
from source. No ecosystem API/skill manifest is present in this repository.
Its owner must update that external manifest with the release. No commit or
push is part of this implementation.
