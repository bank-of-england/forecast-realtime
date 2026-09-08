# Internal data refactor

Status: all seven stages implemented. The corrective follow-up restores the
public fit-annotation, known-calendar and tree context-hook contracts. Validation
passes: 875 tests, 11 skips, three snapshots, lint, formatting, docstrings and
documentation checks. The executed baseline, compatibility decisions and
historical performance comparison are recorded in
[INTERNAL_DATA_RESULTS.md](INTERNAL_DATA_RESULTS.md#corrective-follow-up).
The implementation accepts the recorded 20.6% / 34 ms tree overhead as a trade-off,
not a speed improvement; the follow-up does not rerun those benchmarks. Removed
helper APIs have an explicit
[migration note](docs/forecasting_strategy.md#transformation-helper-api-changes).
Parallel revision decomposition remains deferred; its existing guard is retained.

## Objective

Make data rules belong to one internal `ModelData` class. Models should describe
what they need, the data object should prepare it, and realtime execution should
schedule the work. Success means fewer representations, arguments and branches,
not merely shorter callers backed by another large wrapper.

Preserve the public DataFrame APIs and numerical behaviour. Separate any bug fixes
discovered during the migration from the structural changes. The target contracts
below do not authorise silent behaviour changes: resolve the stage 1 compatibility
decisions before migrating the affected path.

## Architectural decisions

1. **Use one data class, not a class hierarchy.** Do not introduce `TreeModelData`,
   `MonthlyModelData`, or separate mutable raw/prepared data classes.
2. **Use composition for forecast trees.** `ForecastTree` remains a
   `ForecastModel`; it traverses the graph and constructs ordinary `ModelData`
   inputs for its leaves and stacking models.
3. **Keep observations separate from policy.** Requested transformations,
   imputation choices and design settings belong to the model's fitted
   configuration. Source metrics, dates and frequencies belong to the data.
4. **Use long-form observations as the canonical storage.** Wide frames are
   projections for existing hooks, not a second transformation implementation.
5. **Preserve provenance until conversion is complete.** Never merge values in
   different metrics and then attach one metric label to the combined column.
6. **Return new data objects rather than mutate shared inputs.** Sharing raw data
   across leaves, vintages or counterfactuals must not change another run.
7. **Replace old paths as each new path is adopted.** Compatibility adapters may
   remain, but they must delegate; there must not be two implementations of a rule.

## Responsibility boundaries

| Owner | Owns | Does not own |
| --- | --- | --- |
| `ModelData` | Schema, source selection, as-of selection, calendar operations, conditioning, metric conversion, missing-data operations, long/wide projections and level reconstruction | Formula parsing, choosing model policies, estimation, graph traversal or scheduling |
| `ForecastModel` | Input requirements, fitted preprocessing policy, model-specific preparation hooks, lag/dummy design, estimation, effective fitting origin and result validation | Raw-table deduplication, metric arithmetic or repeated frequency inference |
| `ForecastTree` | Requirements across components, dependency order, component fitting, stacking inputs and graph-wide fit commit | Its own data schema or transformation implementation |
| `RealTimeModel` | Vintage/horizon policy, task scheduling, skip/failure handling, aggregation and external storage | Pivoting, per-role data cleaning or tree-specific source-selection algorithms |
| Fitted configuration | Immutable choices captured at fit time | Mutable observations or ownership of a second data pipeline |

Keep lags, dummies and model-specific estimation filtering outside `ModelData` in
the first refactor. They concern model design, and moving them at the same time
would obscure whether the data abstraction actually simplifies the code.

## The `ModelData` contract

### Logical schema

Use a private observation table and a small series catalogue inside the same
class. This avoids repeating every piece of metadata on every observation.

| Part | Required information |
| --- | --- |
| Observation | Series identity, observation date, value, and vintage date when the input is vintage-aware |
| Series catalogue | Variable/output-column identity, source, current metric, history/conditioning/output kind, resolved frequency and date-anchor information |
| Role bindings | Ordered y/X selections, binding each role and variable to its history and conditioning streams |
| Input layout | Whether each input was supplied, its boundary semantics, declared columns and date support, and original index convention |
| Compatibility metadata | Row-associated metadata needed to reproduce long-form outputs, including supplied horizon and extra columns; not a second observation table |

The series identity distinguishes source, metric and input kind. Role bindings
allow the same variable to be used as both y and X without confusing its source
streams. Unbound archive data can retain several candidate sources and metrics;
selection produces a bound view for a consumer.

Validate these invariants at construction and preserve them in every operation:

- One value per series/date/vintage key; reject unresolved conflicting duplicates
  rather than silently choosing whichever row happens to come first. Changing
  legacy duplicate acceptance requires a separate compatibility decision and fix.
- After as-of selection, one value per selected series/date.
- Values and their source metrics remain attached through every selection.
- Preserve missing observations, all-missing columns, declared column order and
  explicitly supplied empty inputs. Do not lose these through a default `stack()`
  or `dropna()` operation.
- `None`, an empty supplied frame and an all-missing supplied frame remain
  distinguishable. The public adapters preserve their current hook behaviour.
- Direct wide data has no release vintage; do not invent one. As-of selection
  requires vintage-aware data rather than treating missing release dates as
  observed vintages.
- Retain per-series frequency separately from forecast step frequency. Resolve
  frequency once at the boundary where it becomes necessary, then carry it.
- Calendar-dependent conversions retain the supported M/Q rules. Ordinary daily
  direct inputs remain valid when no calendar conversion is requested.
- Preserve `DatetimeIndex` anchors and supported `PeriodIndex` round-trips. Do not
  expand or remove PeriodIndex transformation support implicitly in this refactor.

This is a logical contract, not a requirement to build a generic schema engine.
Use pandas and ordinary private validation functions. No DataFrame subclass,
transformation graph, validation framework or new runtime dependency is needed.

### Boundary semantics and long-form compatibility

Constructors must distinguish revision archives from supplied vintage
trajectories. A vintage column alone does not establish which interpretation
applies; the adapter declares it.

- **Revision archives:** `as_of()` resolves the latest release at or before the
  requested vintage, then `transform()` converts that selected trajectory.
- **Supplied vintage trajectories:** existing long-form transformation callers
  operate within each supplied vintage group. Do not backfill earlier releases
  into that group or implicitly apply archive-style `as_of()` selection.
- **Direct wide inputs:** retain their date support and index convention without
  inventing release dates.

Long-form adapters must preserve their established output behaviour. In
particular, `DataTransformationPipeline.apply()` retains existing rows and adds
requested derived metrics rather than returning only a selected metric. Preserve
its existing skip and missing-derived-row rules unless separately changed.
Retain row-associated metadata such as `forecast_horizon`, `source`, `frequency`
and extra caller columns through projection and derived-row creation.

Stage 1 must record adapter-specific row/index ordering, column order, index
names, dtypes and empty-output behaviour. `to_long()` restores that contract from
the common representation; adapters choose the output layout, not a second
transformation algorithm. Long/wide parity compares equivalent trajectories,
not an archive against an unexpanded vintage group.

### Small operation surface

| Operation | Contract |
| --- | --- |
| Boundary constructors | Adapt revision archives, supplied vintage trajectories, direct wide inputs, public forecast contexts and named component outputs with explicit boundary semantics |
| `select()` | Bind role-specific variable/source/metric requirements without dropping provenance |
| `as_of()` | Select the latest available release for each selected archive stream/date at or before a vintage; do not implicitly apply to supplied trajectories |
| `condition()` | Select horizon-limited conditioning paths, retaining published and supplied streams until their metrics can be reconciled |
| `transform()` | Convert each bound trajectory to the requested metric; use the same arithmetic for long and wide callers |
| `regularise()` / `impute()` | Apply explicitly supplied missing-data policies on each series' own calendar |
| `to_wide()` | Materialise the requested role/path with its declared order and missingness contract |
| `to_long()` | Project labelled observations and retained row metadata into the long-form adapter's output layout, including retained and derived metrics |
| `reconstruct_levels()` | Add supported level reconstructions using historical anchors available at the relevant vintage |

Keep arithmetic in small private functions behind these operations. Do not turn
each helper into a new strategy class.

### Ordering rules

Select sources and metrics before vintage deduplication. Preserve the existing
scope of metric selection; do not silently introduce per-vintage source fallback.

For revision archives, select the available vintage before computing its
transformed trajectory. Supplied vintage trajectories stay within their declared
groups. When history and conditioning are both levels, overlay non-null
conditioning values before conversion so future differences use the correct
historical anchor. When conditioning is already in the requested metric,
transform the levels history first and then overlay compatible values. Reject
unsupported conversions.

Preserve calendar gaps, actual leading undefined prefixes and the current
fit-versus-forecast preparation order. Imputation remains a model-selected policy,
not an automatic consequence of constructing `ModelData`.

Stochastic imputation must preserve RNG lifetime and draw order, not just its
seed. The current `impute_X()` creates one generator per call and consumes it in
X-column order. Keep that order and the existing conditional consumption of draws
when moving arithmetic behind per-series operations. Do not reset the seed for
each series or use catalogue-sorted order. Characterise `ar1_t` with multiple
ragged columns before moving it.

### Ownership and performance

Copy caller-owned data on entry and return isolated projections at mutable model
hook boundaries. A frozen dataclass alone does not make pandas objects immutable.
Keep private internals private; use safe internal sharing only where it is tested.

Avoid full-table normalisation and repeated long/wide conversions inside every
leaf or vintage. Reuse validated selections and materialise wide frames at the
existing hook boundary. Keep raw and transformed objects separate; a tree must
never transform its shared raw object in place.

The class must survive deepcopy, pickle and Windows process spawning. Start
without persistent caches. Add caching only if profiling shows a need and it
cannot leak state across consumers.

Repeat the stage 1 runtime and peak-memory measurements after stages 3, 4 and 5,
as well as at completion. Record workload sizes, seeds, dependency versions,
worker/batch settings and repeated-run variability. Benchmark outside concurrent
pytest workers; include task serialisation and worker memory for spawned runs.
Investigate material conversion/copying regressions at each gate before proceeding.

## Model integration

Keep public `fit()`, `forecast()`, `predict()` and `ForecastContext` compatible.
Keep `_fit`, `_forecast`, `_forecast_decomp` and preparation hooks DataFrame-based.

Public adapters and internal callers should converge on one data-based fit path
and one data-based prediction path. Those paths retain atomic candidate fitting,
fitted-option validation and result validation. Do not create a shortcut that
bypasses the candidate/commit lifecycle or existing documented subclass hooks.
Characterise public-method overrides before changing internal dispatch; retain
an adapter where a supported override must still be called.

Give ordinary models and trees one input-requirements interface. An ordinary
model reports its formula-selected role requirements; a tree reports the
requirements of its consumers without flattening contradictory metrics. Rework
the existing `ModelInputRequirements` into this small immutable request type
instead of adding another competing wrapper. `ModelData` consumes requirements;
it must not inspect model classes, formulas or tree nodes.

Requirements enumerate inputs independently of explicit transformation settings.
An absent mapping must not remove a required variable. Preserve consumer and role
identity, declaration order, raw-versus-component input ownership, and the resolved
metric request/default separately from whether a transformation was explicit.
Retain ordinary-model levels preference and existing ambiguity/unsupported-source
errors; no mapping is not permission to accept arbitrary native units.

Use these examples to specify and test the record before implementing it:

| Consumer | Required interpretation |
| --- | --- |
| Ordinary model `y ~ x`, no transformation | Both roles remain required; implicit levels policy applies, without enabling explicit-pipeline hook behaviour |
| Tree with mapping-free leaves | Each leaf still reports its raw inputs; do not omit them because its metric mapping is empty; resolve the current selection defect separately |
| Leaves requesting `y: diff` and `y: pop` | Retain both requests and the common-source rule; levels can serve both, without adding per-vintage fallback |
| `z` used as both y and X with different conditioning sources | Keep separate role/source bindings; adopting this behaviour requires the source-selection fix recorded in stage 1 |
| Stacker consuming raw `y` and child outputs `child_a`, `child_b` | Only its raw target enters archive requirements; synthetic X requirements bind actual named child-output columns and their native metrics |

Retain `FittedModelConfiguration`. Reduce `FittedDataTransformation` to the
resolved fit-time policy and metadata needed to validate later calls. Data
operations move to `ModelData`; existing module-level adapters delegate there.
Do not introduce another fitted-plan alias alongside it.

## Forecast-tree integration: composition, not inheritance

Trees need different orchestration, not different data semantics:

1. Resolve consumer requirements, retaining nearest-tree and leaf-owned mapping
  precedence. Preserve the current common-source selection rules and errors,
  except for separately reviewed fixes recorded in stage 1.
2. Give each leaf a raw `ModelData` selection. Each leaf applies its own fitted
   policy exactly once; the tree applies no shared transformation first.
3. Evaluate nodes in the existing bottom-up order, deduplicating shared graph
   objects as today.
4. For a stacking model, build ordinary `ModelData` from named child outputs and
   the selected target history. Attach native metrics and frequencies to the
   actual synthetic columns, including prefixed multivariate output columns.
5. Apply the stacking model's own policy. Raw regressor frequencies, imputation,
   leaf lags and dummies must not leak onto synthetic columns.
6. Keep callable nodes on their current dictionary-of-DataFrames contract. Do not
   guess a callable's output units from arbitrary arithmetic; preserve the
   existing output-metric convention unless a separate API change defines one.
7. Preserve root-origin selection, native output metrics, component inspection
   attributes and atomic commit back into the original graph objects.

Do not add `is_tree` switches to `ModelData` or a second preparation pipeline for
stackers. A synthetic component output is another labelled input stream.

## Implementation stages

Each stage should be reviewable and end with passing relevant tests. Later stages
remove transitional adapters; the old and new implementations must not coexist
indefinitely.

### 1. Establish the behavioural baseline

- Run the current suite in the `forecast-realtime` conda environment with
  `pytest -n auto`; record existing failures and optional-model skips.
- Add missing characterisation tests for absent/empty conditioning, mixed source
  metrics, direct daily inputs, supported PeriodIndex paths, duplicate keys,
  mutation isolation and multivariate stacking provenance. Cover the requirements
  examples and long-form boundary/output contracts above.
- Record deterministic forecast/decomposition outputs and representative timing
  and peak-memory baselines for direct, realtime and tree workloads.
- Characterise `ar1_t` with multiple ragged columns, fixed seeds and declared
  column order; compare direct/realtime and sequential/spawned paths on equivalent
  inputs and preparation calls.
- Compare forecasts and complete level/revision decompositions across sequential
  execution and explicit Windows-compatible spawn, using `batch_size=1`, larger
  batches and one complete batch. Include skipped vintages between successful
  fits and check `base_vintage_date`, row coverage and reconciliation, not just
  forecast values.
- Complete the compatibility-decision register below with reproducing tests,
  approved behaviour and a separate fix reference or explicit deferral. A current
  defect is not a passing target contract; keep its reproducer distinct from the
  desired assertion until the fix is approved.

The following findings come from static inspection, not an executed baseline:

| Case | Current implementation / risk | Proposed target and decision required |
| --- | --- | --- |
| Conflicting duplicate keys | Realtime vintage slicing keeps the first row after sorting releases | Reject unresolved conflicts; decide exact-duplicate and metadata-conflict handling in a separate validation change before schema adoption |
| Same variable in y and X, different conditioning sources | Source dictionaries are merged; the X source overwrites the y source | Bind sources per role; test and fix selection separately before using independent bindings |
| Mapping-free tree consumers | Requirements collect only mapped variables, which can remove required inputs from selection | Retain all consumer inputs with explicit default semantics; fix the existing omission before requirements migration |
| Mixed-metric published/conditioning overlap | Values can be overlaid before one metric label is assigned to the column | Reconcile streams before overlay; isolate any changed forecasts as a semantic fix |
| Multivariate synthetic provenance | Metric lookup uses a child name and first output metric rather than each prefixed column | Attach each output's actual metric; isolate corrected stacker inputs and numerical results |
| Cross-batch revision decompositions | Each task starts with empty previous-vintage state | Compare consecutive successful vintages independently of batching; fix boundary continuity separately or explicitly defer it and retain the known limitation in regression expectations |

For each row, stage 1 must add the reproducer location, observed result, approved
decision and fix/deferral reference. A target invariant that depends on an
unapproved fix blocks that affected migration; do not hide legacy fallback
algorithms inside `ModelData` to make the structural patch pass.

Exit: an agreed regression baseline and completed decision register. Required
semantic fixes are reviewed separately before the stages that depend on them.

Suggested commit: `test: characterise internal data contracts`

### 2. Introduce the schema and replace vintage slicing

- Add one private implementation module for `ModelData` and its pure helpers.
- Implement boundary constructors, validation, selection, `as_of()` and wide
  projections. Exercise bound model inputs, unbound archives and supplied vintage
  trajectories; preserve row metadata for the later long-form adapter migration.
- Replace the repeated history/conditioning latest-vintage and pivot sequences
  in [real_time_model.py](src/forecast_realtime/real_time_model.py) with these
  operations. Keep current metric-selection policy while migrating it.
- Keep the external raw-table workaround at the ingestion boundary; do not change
  the dependency behaviour in the same patch.

Exit: four role-specific implementations become one tested selection path.

Suggested commit: `refactor: centralise vintage data selection`

### 3. Consolidate trajectory preparation

- Move metric selection, conditioning, calendar alignment and conversion behind
  `ModelData`. Resolve native-versus-levels provenance before overlay.
- Route long-form and wide-input transformation adapters through this path.
  Implement `to_long()` and test retained/derived rows, passthrough metadata and
  within-vintage semantics alongside numerical parity on equivalent trajectories.
- Move the numerical preparation operations from
  [_utils.py](src/forecast_realtime/_utils.py) behind the common data interface
  where appropriate; retain thin import-compatible wrappers when needed.
- Route supported level reconstruction through the same labelled-series model.
- Keep the private data implementation independent of model and compatibility
  modules, avoiding circular imports. Existing transformation entry points import
  and delegate to the private implementation, not the reverse.
- Verify stochastic-imputation draw order and repeat the stage 1 performance
  measurements before ordinary-model integration.

Exit: one implementation per data rule, with long/wide parity tests.

Suggested commit: `refactor: unify data transformation operations`

### 4. Simplify ordinary model preparation

- Adopt the requirements interface and data-based internal fit/predict paths in
  [forecast_model.py](src/forecast_realtime/forecast_model.py).
- Replace parallel raw-frame/metric/frequency arguments internally with
  `ModelData` plus the fitted policy and request controls.
- Remove duplicate input validation, frequency resolution and history merging
  from model orchestration. Retain model-specific capability decisions.
- Keep raw data, prepared design history and estimation data distinct where they
  serve different purposes; consolidate redundant copies, not necessary states.
- Preserve fitted-state immutability, atomic refits and DataFrame hook contracts.
- Verify mapping-free input requirements and repeat direct/realtime timing and
  peak-memory measurements against stage 1.

Exit: fit/predict read as preparation, design construction, model call and result
validation rather than a sequence of pandas repairs.

Suggested commit: `refactor: prepare model inputs through ModelData`

### 5. Integrate trees without a data subclass

- Adopt the shared requirements interface and data paths in
  [forecast_tree.py](src/forecast_realtime/forecast_tree.py).
- Replace leaf context reconstruction and synthetic metric dictionaries with
  labelled `ModelData` selections/component adapters.
- Remove raw-versus-synthetic keyword filtering where explicit input ownership
  now makes it unnecessary; preserve the underlying forwarding behaviour.
- Keep graph traversal and callable evaluation explicit. Do not rewrite the
  tree graph or public node API as part of this work.
- Verify raw-target versus synthetic-X requirements and repeat tree timing and
  peak-memory measurements against stage 1, including shared-leaf workloads.

Exit: trees use the same data operations as ordinary models, with no tree-specific
data class, double transformation or synthetic metadata loss.

Suggested commit: `refactor: compose shared data inputs in forecast trees`

### 6. Finish realtime task and counterfactual integration

- Replace redundant source-metric dictionaries and raw tables in `ForecastTask`
  with one data payload plus model/run controls. Keep model-specific kwargs
  separate from data metadata.
- Remove tree-specific source-selection dispatch from `RealTimeModel`; consumer
  requirements drive selection through the common interface.
- Reduce each vintage run to data selection, cutoff/conditioning requests,
  fitting, prediction and output collection.
- Reuse the fitted policy with replacement `ModelData` for revision
  counterfactuals. Keep model copies where hooks can mutate forecast caches.
- Preserve first-horizon behaviour, skipped-vintage handling, native forecasts,
  reconstruction and decomposition metadata.
- Test sequential and explicit spawned execution, including payload round-trips,
  stochastic imputation and the stage 1 batch-size/decomposition matrix. Preserve
  the separately agreed boundary-continuity behaviour and any documented deferral;
  payload consolidation alone does not establish decomposition parity.

Exit: realtime orchestration no longer implements data-preparation algorithms or
threads parallel provenance dictionaries through the call chain.

Suggested commit: `refactor: streamline realtime data orchestration`

### 7. Remove scaffolding and prove the simplification

- Remove unused `RawInputBundle`, `InputMetricMapping`, `PreparedModelInputs` and
  the unused `ResolvedTransformationPlan` alias after checking source consumers.
  Keep the now-used requirements record, not its obsolete shape.
- Reduce `DataTransformationPipeline` and existing helper entry points to thin
  adapters where compatibility is required. Do not remove de facto import paths
  silently; document deliberate removals separately.
- Delete superseded helpers, duplicate mappings and fallback state. Update tests
  of private task fields to assert provenance and behaviour through the new
  payload rather than preserving obsolete internals.
- Update the architecture and contributor-facing model guidance. Keep public
  exports unchanged unless an explicit compatibility decision requires otherwise.
- Run the full tests, lint/format/docstring checks and documentation checks in
  the named conda environment. Use `-n auto` for pytest runs. Compare performance
  with stage 1; investigate regressions before adding caches.

Exit: less production code across the affected modules including the new module,
fewer internal parameters, and one location for each data rule. Do not meet a
line-count target by compressing readable code.

Suggested commit: `refactor: remove superseded data plumbing`

## Verification matrix

| Area | Existing regression coverage and required additions |
| --- | --- |
| Source selection and vintage safety | [test_input_metric_selection.py](tests/test_input_metric_selection.py); add future-vintage isolation, approved duplicate policy, independent role/source bindings and mapping-free consumer requirements |
| Long/wide metric parity | [test_data_transformation.py](tests/test_data_transformation.py), [test_data_transformation_wide_inputs.py](tests/test_data_transformation_wide_inputs.py); cover all metrics, overlap, mixed-source conditioning, archive versus supplied-vintage semantics, retained/derived rows and metadata/layout round-trips |
| Frequency and missingness | [test_lagged_realtime_datasets.py](tests/test_lagged_realtime_datasets.py), [test_forecast_model_data_transformation.py](tests/test_forecast_model_data_transformation.py); cover M/Q gaps, actual leading undefined prefixes, date anchors and all-missing inputs |
| Stochastic imputation | [test_real_time_model.py](tests/test_real_time_model.py); add multi-column `ar1_t` baselines for RNG lifetime, column/draw order and equivalent direct/realtime and sequential/spawned preparation |
| Direct model compatibility | [test_forecast_model_edge_cases.py](tests/test_forecast_model_edge_cases.py), [test_model_contract_forwarding.py](tests/test_model_contract_forwarding.py), [test_fitted_values_contract.py](tests/test_fitted_values_contract.py), [test_ols.py](tests/models/test_ols.py); cover daily inputs, supported PeriodIndex paths and hook overrides |
| Tree behaviour | [test_forecast_tree.py](tests/test_forecast_tree.py); add mapping-free/conflicting consumer requirements, raw-target versus synthetic-X ownership, per-output synthetic provenance, shared-data mutation isolation and raw/stacking policy separation |
| End-to-end equivalence | [test_real_time_model.py](tests/test_real_time_model.py), [test_sample_realtime_model_regressions.py](tests/test_sample_realtime_model_regressions.py); preserve dates, horizons, native metrics, conditioning, reconstruction and decomposition reconciliation |
| Isolation and execution | Add constructor/projection mutation, deepcopy/pickle, atomic model/tree refit and explicit Windows-compatible spawn tests; retain counterfactual model isolation and compare revision coverage/metadata across batch sizes and skipped vintages against the agreed baseline |
| Performance | Repeat fixed direct, realtime and shared-leaf tree workloads after stages 3, 4, 5 and 7; compare runtime, peak memory and spawned payload costs with stage 1 outside concurrent test execution |

New `ModelData` unit tests should test domain operations directly. Integration
tests should compare outputs, source provenance and hook inputs rather than
private container layouts. Keep numerical tolerances explicit and justify any
snapshot change; a refactor alone does not justify changed forecasts.

## Definition of done

- `ModelData` is the only internal observation representation with data operations.
- Ordinary models, leaves and stackers use it; there is no tree-data subclass.
- No canonical observation table and canonical wide table are maintained together.
- Metric selection, vintage selection, overlay, calendar conversion and
  reconstruction each have one implementation.
- Data and provenance travel together, including through process tasks and
  counterfactual calls.
- Existing public APIs and supported model hooks still work.
- Long-form adapters preserve vintage interpretation, retained and derived rows,
  metadata and their characterised output layouts.
- The full suite matches the baseline, or separately reviewed fixes explain each
  difference. The decision register links each changed contract or deferred defect
  to its tests and review; batch-boundary revision gaps are not hidden by
  forecast-only comparisons. Stochastic imputation matches its ordered baseline.
- Representative runtime and memory show no unexplained regression at the
  intermediate gates or final comparison.
- The end-to-end path is easier to follow and contains fewer representations and
  compatibility branches than the starting point.

Implement and review the stages without committing or pushing automatically.
