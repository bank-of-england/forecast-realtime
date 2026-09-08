# Internal data refactor: verification

All seven implementation stages are complete. Ordinary models, trees, realtime
tasks and revision counterfactuals use one private `ModelData` implementation.
Public calls, `ForecastContext` fields and package exports remain compatible;
the tree signature details are recorded below. No runtime dependency or
persistent cache was added.

Final result: **762 passed, 11 skipped, 105 warnings and three passing
snapshots in 181.40 seconds**. All five semantic regressions pass. Ruff lint and
formatting, docstring checks, API documentation consistency, notebook freshness
and the strict documentation build pass.

## Baseline

The original suite passed in the `forecast-realtime` conda environment:
695 passed, 11 skipped, 103 warnings, three snapshots, 133.26 seconds with
`pytest -n auto` (eight workers). The added boundary characterisation passed
13 cases. Five isolated defect reproducers failed as expected before migration.

Environment: Windows 11, Python 3.14.7, pandas 3.0.5, NumPy 2.5.2,
SciPy 1.18.1, statsmodels 0.14.6 and forecast-evaluation 0.1.13. Installed
forecast-realtime distribution metadata reports 0.5.3; execution uses this
editable checkout, whose project version is 0.5.7.

## Compatibility decisions

The request to implement every phase includes the following explicit fixes.
They have separate regression cases rather than silently changed snapshots.
These are implementation decisions, not a claim of a separate human review.

| Case | Executed reproducer and observation | Decision / fix reference |
| --- | --- | --- |
| Conflicting archive keys | `test_realtime_rejects_conflicting_same_date_vintage_values`: no exception | Reject conflicting values or row metadata within a series/date/release key. Collapse identical archive duplicates. Preserve distinct sources and metrics until selection. `fix: reject conflicting observation keys` |
| Independent role sources | `test_realtime_keeps_y_and_x_conditioning_sources_separate`: y receives the X source | Select conditioning streams per role. `fix: retain role-specific conditioning sources` |
| Mapping-free tree consumers | `test_mapping_free_tree_retains_realtime_regressor_inputs`: every vintage skipped | Enumerate formula-selected inputs even without an explicit transformation; implicit raw inputs still request levels. `fix: retain unmapped tree input requirements` |
| Mixed published/conditioning units | `test_diff_forecast_conditioning_preserves_published_level_path`: April is 133.1 rather than 12.1 | Convert published levels before overlaying native differences; never relabel mixed units. `fix: reconcile conditioning metrics before overlay` |
| Multivariate synthetic provenance | `test_multivariate_child_dispatches_each_native_metric_once`: child metadata does not match prefixed columns | Label each actual output column with its native metric. `fix: preserve multivariate component provenance` |
| Revision batch boundaries | Public code rejects parallel decomposition and uses one sequential batch | Defer parallel revision decomposition. Retain the guard and the single ordered batch, including skipped-vintage continuity. Spawn the complete sequential request to test process parity; test forecast-only batches separately. |

Reproducers live in [tests/test_model_data_semantics.py](tests/test_model_data_semantics.py).
Boundary characterisation lives in
[tests/test_internal_data_baseline.py](tests/test_internal_data_baseline.py).
All five reproducers now use required assertions; no temporary xfail remains.
The original realtime code was also loaded into temporary in-memory modules to
verify the two source/provenance defects: it supplied 202 to both roles instead
of y=101 and X=202, and labelled April's published level 133.1 as a difference
instead of converting it to 12.1 before overlaying May's supplied difference 5.

### Adapter contracts

- Direct wide inputs preserve declared columns, index names, missing cells and
  date support. `None`, supplied empty and supplied all-missing conditioning
  remain distinguishable. An explicit pipeline supplies y history to forecast
  preparation even without conditioning; an implicit identity does not.
- Direct daily identity inputs work. Direct PeriodIndex identity inputs work;
  explicit wide transformation still requires a DatetimeIndex.
- Realtime archive projections retain the legacy pivot ordering (sorted dates
  and columns); formula selection subsequently determines design column order.
- Long-form transformation retains original rows and appends derived rows.
  Derivation resets the index; an unchanged projection retains its index.
  Original column order, datetime dtypes and caller metadata survive. Calendar
  transformations omit missing derived values. Vintage groups are supplied
  trajectories, not revision archives.
- Realtime calls public `fit()` and `predict()` overrides. Trees call public
  component `fit()` and `forecast()` overrides. Internal dispatch must retain
  these boundaries without bypassing candidate/commit fitting.
- Stochastic imputation uses one generator per call in declared X-column order.
  Columns which need no draws must not advance it.

The base `ForecastModel.fit()`, `forecast()` and `predict()` signatures,
`RealTimeModel.forecast()` and all `ForecastContext` fields match the original.
`ForecastTree.fit()` and `predict()` inherit the base methods: introspection now
exposes settings formerly accepted through `**kwargs`, while existing calls
remain valid. `ForecastTree.forecast()` retains its original positional
`context` argument through a thin delegating adapter. The final regression
checks both positional and keyword forms; the inherited base signature alone
would have misinterpreted the positional context as a transformation mapping.

### Final boundary and execution checks

- [tests/test_model_data.py](tests/test_model_data.py) covers caller/projection
  isolation, exact large-integer and nullable dtypes alongside floats, retained
  metadata, duplicate policy, pickle/deepcopy and vintage-safe views. Selecting
  a subset before `as_of()` does not expose future date support. Extending a
  published path retains its existing later observations.
- [tests/test_model_data_requirements.py](tests/test_model_data_requirements.py)
  covers immutable ordered requests, mapping-free inputs, conflicting consumers,
  common levels selection, nested policy precedence and synthetic ownership.
- [tests/test_model_data_boundary_hooks.py](tests/test_model_data_boundary_hooks.py)
  covers atomic validation hooks and public overrides which mutate or replace
  context frames. Public frames remain authoritative. Index frequency changes
  cannot leak into raw data, and unused formula inputs cannot change conditioning
  selection. Lightweight subclasses which omit the base constructor still work.
- [tests/test_model_data_execution.py](tests/test_model_data_execution.py) covers
  actual task round-trips, explicit Windows-compatible spawn, forecast-only
  batches of one, two and all four vintages, and complete sequential/spawned
  revision tables. An intermediate skipped vintage retains the previous
  successful base vintage. Full level, news, reestimation and interaction
  contributions reconcile. Unpatched multicolumn `ar1_t` agrees across direct,
  realtime and spawned preparation in both declared X orders at tolerance
  `1e-12`; repeated calls remain reproducible.

These checks repaired migration gaps rather than broadening the public API:
validation runs on the candidate, public context edits reach the model, formula
metadata is filtered at its established boundary, source metrics survive
replacement data and component ownership survives public adapters.

## Regression gates

All pytest runs used `pytest -n auto` in the named conda environment. The
intermediate xfails below identify semantic fixes not yet migrated at that gate;
none remains in the final result.

| Gate | Result | Elapsed seconds |
| --- | --- | ---: |
| Original full suite | 695 passed, 11 skipped; three snapshots | 133.26 |
| Boundary characterisation | 13 passed | — |
| Initial semantic reproducers | 5 xfailed against the original behaviour | — |
| Stage 2 realtime selection | 156 passed, 1 skipped, 4 xfailed | 39.92 |
| Initial schema | 12 passed | — |
| Stage 3 shared operations | 291 passed, 1 skipped | 80.15 |
| Stage 4 ordinary models | 360 passed, 1 skipped | 75.13 |
| Stage 5 tree composition | 115 passed, 2 xfailed | 42.95 |
| Shared requirements | 10 passed | 9.02 |
| Stage 6 realtime integration | 177 passed, 1 skipped | 70.64 |
| Payload, imputation and index isolation | 97 passed | 22.72 |
| Single-pass conversion regression gate | 377 passed, 1 skipped | 47.41 |
| Full suite before positional tree regression | 760 passed, 11 skipped; three snapshots | 140.11 |
| Full suite with positional tree adapter | 762 passed, 11 skipped; three snapshots | 187.20 |
| Final full suite | 762 passed, 11 skipped; three snapshots | 181.40 |

The final suite adds 67 required cases to the passing baseline and retains the
same 11 skips. No snapshot changed. During final integration, ten failures
exposed assumptions about lightweight subclasses' `_formula` attribute; the
shared boundary now tolerates an absent attribute. The affected follow-up suite
passed all 102 cases before the final full run.

| Quality gate | Result |
| --- | --- |
| Ruff lint | Passed across the project |
| Ruff formatting | All 102 Python files formatted |
| NumPy-style docstrings | `pydoclint` passed on source and tests |
| Public API documentation | Existing page equals the generated export manifest |
| Notebook documentation | Freshness check passed |
| Zensical documentation | Clean strict build passed |
| Diff whitespace and editor diagnostics | No errors in changed files |

The documentation build still emits Griffe parameter-parsing notices, then
reports no issues and completes successfully. The remaining EOF-only formatter
difference was removed with the user's authorisation; formatting is now clean.

## Performance

### Method

[scripts/benchmark_model_data.py](scripts/benchmark_model_data.py) runs outside
pytest. Seed: 20260908; three same-seed repeats; 72 historical months for direct
and realtime models; two X columns; six direct forecast steps. The realtime
panel has 1,296 observations over six revised snapshots with three forecast
steps. The shared-leaf tree uses 66 historical months and four forecast steps,
with levels, logs and period-on-period consumers. Realtime decomposition runs
sequentially in one complete batch. Spawn uses an explicit `spawn` context, two
configured workers and one complete request per repeat. Process peaks are
lifetime peaks; tracemalloc peaks are reset for each workload.

The final benchmark adds an actual `ForecastTask` workload. The original four
workloads remain unchanged for comparison. The raw-request spawn includes
ingestion and external storage; the task-only worker starts from selected data
and returns native forecasts plus complete decomposition. They measure different
boundaries and must not be compared as interchangeable requests.

### Baseline

| Workload | Runtime seconds, three repeats | Tracemalloc peak bytes, three repeats |
| --- | --- | --- |
| Direct OLS | 0.089180, 0.079586, 0.081154 | 403800, 217210, 215951 |
| Realtime OLS + complete decomposition | 1.895864, 1.846455, 1.782687 | 2428925, 1196862, 1192936 |
| Shared-leaf tree | 0.208876, 0.181543, 0.190636 | 353254, 327930, 328484 |
| Spawned realtime worker | 2.156869, 2.053525, 2.017474 | 2516603, 2516172, 2515136 |

Baseline spawned payload: 78,998 bytes; first serialisation: 0.000212 seconds;
first end-to-end spawned run: 5.496205 seconds; worker lifetime process peaks:
248238080, 248844288 and 249106432 bytes. The payload benchmark includes
ingestion and a complete request; task-payload tests separately exercise worker
task round-trips.

### Intermediate measurements

| Stage | Workload | Runtime seconds, three repeats | Tracemalloc peak bytes, three repeats |
| --- | --- | --- | --- |
| 3 | Direct | 0.244745, 0.158441, 0.160115 | 460583, 228672, 227483 |
| 3 | Realtime | 4.860188, 2.859090, 2.927292 | 2362478, 1236109, 1231191 |
| 3 | Tree | 0.487373, 0.367257, 0.415810 | 420701, 398429, 393425 |
| 3 | Spawned request | 3.900562, 3.584872, 3.654689 | 2481980, 2481078, 2479171 |
| 4 | Direct | 0.239096, 0.202550, 0.163631 | 456925, 251445, 252573 |
| 4 | Realtime | 3.171594, 3.924253, 2.379674 | 2385208, 1262249, 1260522 |
| 4 | Tree | 0.369346, 0.722303, 0.333650 | 438527, 414542, 411561 |
| 4 | Spawned request | 3.415067, 3.754925, 3.599579 | 2512011, 2509880, 2508893 |
| 5 | Direct | 0.144377, 0.227590, 0.254826 | 459297, 252220, 250838 |
| 5 | Realtime | 4.909786, 4.322519, 2.343001 | 2389672, 1264231, 1264975 |
| 5 | Tree | 0.664399, 0.660481, 0.338425 | 421077, 395314, 388082 |
| 5 | Spawned request | 4.644605, 5.468929, 3.407723 | 2509942, 2507581, 2509376 |
| 6 | Direct | 0.141333, 0.138608, 0.155914 | 434637, 246069, 245496 |
| 6 | Realtime | 2.548438, 2.280189, 2.319068 | 2252007, 1124233, 1120707 |
| 6 | Tree | 0.294099, 0.312433, 0.259480 | 408162, 382426, 381948 |
| 6 | Spawned request | 2.609811, 5.677142, 2.706287 | 2371789, 2368830, 2369099 |

Early measurements exposed repeated per-series DataFrame slicing, long/wide
round-trips and no-op rebuilding. Profiling led to one complete-trajectory
conversion, array-based canonical construction, filtered role/path projections
and reuse of unchanged native, regularised or imputed inputs. These replace work
rather than cache it. Frequency resolution now inspects carried metadata before
projecting observations, so known or unnecessary calendars do not materialise
series. The stage 6 spawned outlier took 5.677142 seconds in the worker and
11.831718 seconds end to end; the table retains it.

### Late-stage variability

These three-repeat runs preceded the final metadata-first frequency adjustment.
The first preceded the positional tree adapter; the second included it. The
adapter affects trees, but every workload slowed in the second run. The
measurements therefore prompted a contemporaneous original/final comparison
rather than a claim that the earlier speed improvements were reliable.

| Gate | Workload | Runtime seconds, three repeats |
| --- | --- | --- |
| Before tree adapter | Direct | 0.083060, 0.078623, 0.068186 |
| Before tree adapter | Realtime | 1.648550, 1.454891, 1.382100 |
| Before tree adapter | Tree | 0.171785, 0.206055, 0.167904 |
| Before tree adapter | Spawned request | 1.760342, 1.674678, 1.714657 |
| Before tree adapter | Spawned actual task | 1.457182, 1.387680, 1.283401 |
| With tree adapter | Direct | 0.108062, 0.146914, 0.206691 |
| With tree adapter | Realtime | 3.445770, 2.931337, 3.122378 |
| With tree adapter | Tree | 0.385440, 0.341482, 0.340396 |
| With tree adapter | Spawned request | 3.043548, 3.511449, 3.411693 |
| With tree adapter | Spawned actual task | 2.429150, 2.308445, 2.636549 |

### Final measurements

| Workload | Runtime seconds, three repeats | Tracemalloc peak bytes, three repeats |
| --- | --- | --- |
| Direct OLS | 0.169357, 0.269932, 0.145888 | 400773, 245573, 243056 |
| Realtime OLS + complete decomposition | 3.423710, 3.159191, 2.870707 | 2281426, 1156090, 1154800 |
| Shared-leaf tree | 0.434091, 0.424945, 0.330078 | 375570, 359292, 360120 |
| Spawned complete request | 3.236490, 3.515366, 3.422408 | 2388409, 2385705, 2387139 |
| Spawned actual task | 2.920414, 2.489610, 2.982644 | 909986, 913168, 914204 |

The final standalone timings are slower than the original session baseline;
later interleaved measurements below show that host variability accounts for
much, but not all, of that difference. No general runtime improvement is claimed.
Warm Python allocation peaks remain about 28 KB higher for direct models and
32 KB higher for trees, which now retain explicit catalogues and layout/provenance
metadata. Realtime and whole-request spawn medians use about 41 KB and 129 KB
less Python peak memory. There is no persistent cache or second canonical wide
table.

### Contemporaneous original/final comparison

A read-only importer loaded all original package modules from `HEAD` under an
isolated namespace, leaving the checkout unchanged. The same benchmark functions
ran against both packages with seed 20260908, alternating their order on each
repeat. Imports were outside the timer; tracemalloc remained enabled. This
comparison covers direct, realtime and tree execution, not spawned workers.

An initial three-repeat check before the final frequency adjustment found median
changes of +27.5% direct, +9.6% realtime and −10.6% tree. Profiling located
avoidable frequency-time projections, which were removed, and the remaining
per-series projection and calendar costs. The final five-repeat check is:

| Workload | Version | Wall seconds, five repeats | Tracemalloc peak bytes, five repeats |
| --- | --- | --- | --- |
| Direct | Original | 0.069377, 0.065958, 0.063189, 0.080001, 0.109673 | 338198, 216632, 214573, 215306, 215898 |
| Direct | Final | 0.073988, 0.075680, 0.072490, 0.071302, 0.081279 | 282727, 246942, 245118, 244841, 244403 |
| Realtime | Original | 1.670278, 1.461048, 1.340620, 1.655169, 1.572693 | 2402776, 1211575, 1208267, 1206997, 1244045 |
| Realtime | Final | 1.628088, 1.612660, 1.495845, 1.719872, 1.486161 | 1176030, 1167405, 1169701, 1169273, 1171872 |
| Tree | Original | 0.156361, 0.164713, 0.200947, 0.201461, 0.157752 | 361510, 326055, 327411, 323991, 332045 |
| Tree | Final | 0.198687, 0.216750, 0.221762, 0.187185, 0.194926 | 371284, 365106, 362060, 366176, 361743 |

| Workload | Original median seconds | Final median seconds | Wall-time change | Original/final median CPU seconds |
| --- | ---: | ---: | ---: | --- |
| Direct | 0.069377 | 0.073988 | +6.6% (+4.6 ms) | 0.062500 / 0.078125 |
| Realtime | 1.572693 | 1.612660 | +2.5% (+40.0 ms) | 1.546875 / 1.562500 |
| Tree | 0.164713 | 0.198687 | +20.6% (+34.0 ms) | 0.171875 / 0.187500 |

CPU time has coarse 15.625 ms increments on this host, so its direct-workload
percentage is not a precise estimate. All paired forecast and decomposition
checksums match. The remaining cost is materialising isolated per-series pandas
objects and calendars from canonical long data; trees repeat these operations
for consumers with different metrics. The workload pays a small absolute cost
for explicit provenance and boundary isolation. This is a measured trade-off,
not evidence that the refactor is uniformly faster. Further optimisation should
target those projections rather than introduce speculative caching.

### Spawn and task payloads

| Final spawned boundary | Payload bytes | Serialisation seconds, three repeats | Parent runtime seconds, three repeats | Worker lifetime process peak bytes, three repeats |
| --- | ---: | --- | --- | --- |
| Complete raw request | 78998 | 0.000574, 0.000371, 0.000192 | 6.426228, 7.171616, 6.744532 | 249053184, 248733696, 249147392 |
| Actual selected task | 67636 | 0.000707, 0.000555, 0.000606 | 6.593284, 5.723278, 7.097547 | 243351552, 244228096, 243982336 |

No actual-task worker baseline was recorded at stage 1. A separate read-only
comparison loaded the original task class and realtime task builder from `HEAD`
into temporary in-memory modules, using the same panel, unfitted OLS
configuration and complete sequential-decomposition request. The checkout was
not changed. This measures the old payload construction, not an original-code
worker runtime; module identifiers in the reconstructed pickle also differ.

| Task construction comparison | Pickle bytes | Serialisation seconds, three repeats |
| --- | ---: | --- |
| Original task shape | 92166 | 0.000468, 0.000196, 0.000184 |
| Consolidated task | 67636 | 0.000491, 0.000254, 0.000222 |

The selected task is about 26.6% smaller in this comparison. The sub-millisecond
serialisation timings do not establish a speed improvement. The unchanged
78,998-byte raw request is a separate benchmark, not the original task baseline.

### Numerical parity

All original checksums match at every measured stage and in all final repeats.
The complete tables, including dates, horizons and decomposition metadata, feed
the hashes; the integration tests also compare frames directly.

Rounded forecast SHA-256 checksums (eight decimal places):

- Direct: `6b846e95855049e42894e611342cb8eb6f50bfdfbe98fe700f3ae5aac4558b28`.
- Realtime: `59c897240911c5d4df4af464279bd09c25494898a2c19eb685e98eadd17d5b63`.
- Tree: `7733ff973b48aedd7ad7ee9dc681e1fc3f4d351b022e7eaad098e9897d77e9b3`.

Complete decomposition checksums (ten decimal places): direct
`895cf75e53036e584bbf027ca7bbef523104b041fb13ad70a22e4bf12b662fb2`;
realtime and spawn
`71e0a08909e61a5b8e3faa5caf3a8c5d3f6aef25da928eca9cdfa00295cb8650`.
Realtime has 54 level rows across six vintages and 135 revision rows across
five vintages. All revision rows have a base vintage.

The actual task returns 18 native forecast rows with `date`, `vintage_date`,
`forecast_horizon`, `variable`, `value`, `metric`, `source` and `frequency`.
Its ten-decimal native-table checksum is
`0edfe16944b78fc264fce8f4e9ee1ec3a7f5a103aef4a778fe9cd6ea6b12fcc2`.
Its 189 decomposition rows have the same checksum and coverage as the complete
sequential request. Native worker output and the public stored forecast table
have different schemas, so their forecast checksums are intentionally distinct.

## Structural result

Counts include docstrings, comments and blank lines, including the entire new
private module. Tests, documentation and the benchmark script are excluded from
this production comparison.

| Module | Original lines | Final lines |
| --- | ---: | ---: |
| [src/forecast_realtime/_model_data.py](src/forecast_realtime/_model_data.py) | 0 | 1751 |
| [src/forecast_realtime/data_transformation.py](src/forecast_realtime/data_transformation.py) | 1255 | 497 |
| [src/forecast_realtime/_utils.py](src/forecast_realtime/_utils.py) | 513 | 274 |
| [src/forecast_realtime/forecast_model.py](src/forecast_realtime/forecast_model.py) | 1643 | 1482 |
| [src/forecast_realtime/forecast_tree.py](src/forecast_realtime/forecast_tree.py) | 870 | 773 |
| [src/forecast_realtime/real_time_model.py](src/forecast_realtime/real_time_model.py) | 1937 | 1422 |
| [src/forecast_realtime/_realtime_forecasting.py](src/forecast_realtime/_realtime_forecasting.py) | 30 | 29 |
| **Total** | **6248** | **6228** |

The net reduction is 20 lines. The main simplification is ownership: archive
selection, metric arithmetic, overlays, calendars, imputation and reconstruction
each have one implementation. Raw data and necessary prepared/estimation frames
remain distinct; compatibility raw-history properties project the canonical data.
Trees compose labelled inputs, and realtime no longer threads duplicate raw
frames and provenance dictionaries through tasks or counterfactuals.

The deliberate private removals are `RawInputBundle`, `InputMetricMapping`,
`PreparedModelInputs`, `ResolvedTransformationPlan`, the old task fields and
operation wrappers on `FittedDataTransformation`. Existing public
`DataTransformationPipeline` and helper import paths delegate to the shared core.
No tree-data subclass or competing legacy preparation pipeline remains.

The existing rejection of parallel revision decomposition remains in force.
Sequential runs preserve the previous successful vintage in one complete batch;
complete-request spawn parity does not add parallel revision support. The
implementation run made no commits or pushes and left the unrelated review-file
deletion untouched.