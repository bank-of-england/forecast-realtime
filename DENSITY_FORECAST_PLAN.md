## Long Point Output: Implementation Plan

Date: 2026-09-22. Status: planned, not implemented.

The public point-output format may change. This section supersedes earlier
requirements to keep public point results wide, but does not authorise combined
point-and-quantile output or quantile reconstruction. Preserve existing
working-tree changes; do not create a branch, commit or push.

### Output Contract

`ForecastModel.forecast()` and `predict()` will return a DataFrame-compatible
`ForecastResult` with explicit columns and a RangeIndex in both modes:

- Point mode: `date`, `variable`, `value`.
- Quantile mode: `date`, `variable`, `quantile`, `value`.

Point results have no placeholder `quantile` column. Keep `forecast_origin`
and `decomposition` as result metadata; `.forecast` returns the same long
payload as an ordinary DataFrame. One call still returns one output mode.
Order rows by date, then fitted target order, then ascending probability when
present. `steps` counts distinct forecast dates, not rows.

Keep `_forecast()` point hooks accepting their existing arrays or wide
DataFrames. Keep fitting inputs, conditioning inputs, fitted values, external
Parquet files and R/MATLAB/Julia runners unchanged. The shared model finaliser
owns conversion to the public long format, not individual adapters or realtime.
Do not add a public format switch or retain parallel wide and long payloads.

### 1. Normalise and Validate Within ForecastModel

Owners: `ForecastResult`, `_finalise_forecast()`, `_predict_from_data()` and
`_forecast_from_data()` in `src/forecast_realtime/forecast_model.py`.

- Validate hook arrays and wide tables against the existing point contract
   before conversion. Preserve custom model-returned calendars, including
   origin-inclusive calendars; do not impose the density calendar on points.
- Convert once at the output boundary. Validate long point keys against the
   requested step count and fitted targets: unique date-variable pairs, complete
   target coverage at every date, valid dates and the declared origin relation.
- Preserve existing point-value semantics, including supported missing values.
   Retain those rows during reshaping; missing values are not missing keys.
   Do not silently impose the stricter density finite-value rule on point mode.
- Reuse the quantile validator for density results. Validation remains owned
   by `ForecastModel`; realtime consumes validated output.
- Preserve public override dispatch. Require public `forecast()` and `predict()`
   overrides to return the new long contract and validate their returned results
   at the existing model dispatch boundaries. Never refit or forecast twice.
   An arbitrary direct call to an override that bypasses `super()` cannot be
   intercepted by these boundaries; document the override's responsibility.
- Adapt decomposition reconciliation to labelled date-variable values rather
   than positions in the long table. Keep decomposition's existing horizon and
   component schema, metadata and reconciliation tolerance.
- Keep DataFrame construction, slicing, copying and pickling functional;
   pandas' internal `_constructor` calls must not require prediction arguments.

First regression: pivot a long result from the same hook payload and compare
it with the pre-change validated wide result. Cover multiple targets, custom
dates, origin inclusion and missing point values. Check metadata separately.

### 2. Adapt Trees and Decomposition Consumers

Owners: `src/forecast_realtime/forecast_tree.py` and the decomposition helpers
in `src/forecast_realtime/real_time_model.py`.

- At tree component consumption, pivot validated long point results to wide
   frames for existing callable transforms and stacking inputs. Preserve fitted
   target order explicitly; do not rely on pivot's default column sorting.
- Reuse one small private conversion helper where needed. It must reject
   quantile payloads rather than aggregate probabilities into point values.
- Keep tree callables, `leaf_forecasts_`, `node_forecasts_` and fitted values
   wide. Finalise the public tree result as long exactly once, including nested
   trees and model-valued roots. Forecast-tree densities remain unsupported.
- Replace uses of a public result's `.index` as a forecast calendar with its
   ordered distinct `date` values. Update current and counterfactual revision
   decomposition paths; never interpret a long row position as a horizon.

Acceptance: tree numerical results, callable inputs, decomposition totals and
public override call counts remain unchanged apart from public output shape.

### 3. Unify Realtime Publication

Owner: `_loop_through_vintages()` in `src/forecast_realtime/real_time_model.py`.

- Consume long tables in both modes. Remove point-only `reset_index()` and
   `melt()`, wide masking and target-column extraction after concatenation.
- Assign horizons using the full validated calendar before filtering rows.
   Origin-inclusive horizons use calendar positions; other horizons retain
   their existing period-offset rule. All targets and probabilities at one
   date receive the same horizon. Preserve vintage-relative cutoff semantics.
- Use one date-variable publication mask, followed by shared metadata,
   concatenation and cutoff handling. Check partial target publication and
   complete-date removal in both modes.
- Keep mode-specific storage: supported point metrics go to `data.forecasts`;
   with reconstruction disabled, logs/differences remain in `native_forecasts`.
   Quantiles go only to `runner.quantiles`. Existing point rows survive density
   calls, and rejected density batches leave stored results unchanged.
- Keep point reconstruction and decomposition behaviour unchanged. Quantile
   mode still disables reconstruction and rejects decomposition. Do not route
   marginal quantile curves through the point reconstruction code.

Acceptance: sorted realtime point and quantile tables match the pre-refactor
tables in values, dates, horizons, metrics and sources.
The shared publication path needs no output-shape branch.

### 4. Update Consumers and Documentation

Update direct-call assertions in existing model, tree, external-model,
forecast-contract and quantile tests. Adapt expected tables or use the explicit
private wide adapter only where a numerical oracle genuinely expects a matrix.
Do not weaken value, calendar, target-order or decomposition assertions.

Update README examples, usage and model-author documentation, public docstrings,
packaged examples and notebooks that index forecast results as wide tables.
Include an explicit `pivot(index="date", columns="variable", values="value")`
example for callers needing point matrices. Amend CONTRIBUTING's public-output
stability guidance for this approved change; leave model-hook guidance intact.
Regenerate API and notebook documentation. No external runner changes are needed.

### Verification and Delivery

Implement the model boundary first, then dependent consumers, then realtime
simplification. Run the narrow regressions after each slice in the
`forecast-realtime` conda environment with `pytest -n auto`.

Reuse existing test modules for result validation, quantiles, trees, boundary
hooks, realtime, reconstruction and external models. Cover multiple targets,
reserved-looking target names such as `date` and `value`, missing point values,
partial publication, custom calendars, decomposition, invalid public overrides,
state preservation, inline parallel errors and real spawned-process parity.
External-language tests may skip only when their runtimes are unavailable;
Python-side adapter tests must still run.

Before implementation, capture the current test baseline and sorted numerical
outputs for representative direct, realtime and tree forecasts. Compare direct
results after pivoting; compare stored realtime outputs without shape changes.
Keep numerical snapshots unchanged. The previously reported BVAR demo snapshot
failure remains a separate gate: reproduce and inspect it, and do not claim an
isolated baseline comparison succeeded without recorded output and exit status.

Finish with the full suite, Ruff lint and formatting (including changed Markdown
code blocks), pydoclint, API generation, notebook freshness and strict Zensical
build, following `.pre-commit-config.yaml` and `CONTRIBUTING.md`. Report final
test counts, skips, numerical parity and unresolved gates. Review the production
diff to confirm shared publication logic was removed rather than duplicated.

Suggested implementation commit message, without committing:

`feat!: standardise point forecast results as long tables`